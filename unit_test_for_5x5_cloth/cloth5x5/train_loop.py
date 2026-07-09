from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import torch

from .config import DatasetBundle, PhysicalConfig, RuntimeConfig
from .constants import (
    FREE_STATE_DIM,
    HIDDEN_DIM,
    LEARNING_RATE,
    MODEL_RANDOM_SEED,
    TORCH_DTYPE,
)
from .evaluate import evaluate_solver_on_dataset, validation_selection_key
from .io import save_json, state_dict_to_cpu
from .model import MLPOptimizer, apply_model_update, physical_energy_scale
from .physics import stationarity_residual_norm, variational_energy
from .plotting import plot_per_motion_boundary, plot_three_solver_rollout, plot_training_curves


def one_step_diagnostics(model: MLPOptimizer, dataset: DatasetBundle, physical: PhysicalConfig) -> dict[str, float]:
    with torch.no_grad():
        y0 = dataset.initial_y
        y1, delta = apply_model_update(model, y0, dataset.q, dataset.masses, physical)
        error0 = torch.linalg.vector_norm(y0 - dataset.exact_y, dim=-1)
        error1 = torch.linalg.vector_norm(y1 - dataset.exact_y, dim=-1)
        residual0 = stationarity_residual_norm(y0, dataset.q, dataset.masses, physical)
        residual1 = stationarity_residual_norm(y1, dataset.q, dataset.masses, physical)
        ideal = dataset.exact_y - y0
        cosine = torch.nn.functional.cosine_similarity(delta, ideal, dim=-1, eps=1e-30)
        return {
            "mean_error_before": float(error0.mean().item()),
            "mean_error_after": float(error1.mean().item()),
            "mean_residual_before": float(residual0.mean().item()),
            "mean_residual_after": float(residual1.mean().item()),
            "mean_update_norm": float(torch.linalg.vector_norm(delta, dim=-1).mean().item()),
            "update_ideal_cosine_mean": float(cosine.mean().item()),
            "error_improvement_fraction": float((error1 < error0).to(TORCH_DTYPE).mean().item()),
            "residual_improvement_fraction": float((residual1 < residual0).to(TORCH_DTYPE).mean().item()),
        }


def run_experiment(
    *,
    experiment_name: str,
    training_cpu: DatasetBundle,
    validation_cpu: DatasetBundle,
    evaluation_datasets: dict[str, DatasetBundle],
    output_dir: Path,
    config: RuntimeConfig,
    physical: PhysicalConfig,
    gd_step_size: float,
    shared_baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)
    model = MLPOptimizer(config.residual_length_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training = training_cpu.to(device)
    energy_scale = physical_energy_scale(training.masses, physical, config.residual_length_scale)
    initial_energy = variational_energy(training.initial_y, training.q, training.masses, physical).detach()
    exact_energy = variational_energy(training.exact_y, training.q, training.masses, physical).detach()

    print("\n" + "=" * 100)
    print(f"Training {experiment_name}")
    print(
        f"architecture={FREE_STATE_DIM}->{HIDDEN_DIM}->Identity->{FREE_STATE_DIM}, "
        f"points={len(training_cpu):,}, motions={training_cpu.metadata['num_motions']}, "
        f"problems={training_cpu.metadata['num_problems']}, device={device}, dtype=float64"
    )
    print("=" * 100)

    train_log: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []
    diagnostic_log: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, ...] | None = None
    best_epoch: int | None = None
    start_time = time.perf_counter()

    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        k = get_k_for_epoch(epoch_index, config)
        model.train()
        y = training.initial_y
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        energy_gap_sum = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        for _ in range(k):
            y, _ = apply_model_update(model, y, training.q, training.masses, physical)
            energy = variational_energy(y, training.q, training.masses, physical)
            objective = objective + ((energy - initial_energy) / energy_scale).mean()
            energy_gap_sum = energy_gap_sum + (energy - exact_energy).mean()
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"Non-finite training objective at epoch {epoch}")
        objective.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm).item())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}")
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError(f"Non-finite model parameter at epoch {epoch}")

        train_log.append({
            "epoch": epoch,
            "K": k,
            "dimensionless_objective": float(objective.item()),
            "training_energy_gap_sum": float(energy_gap_sum.item()),
            "gradient_norm_before_clip": grad_norm,
        })

        if epoch == 1 or epoch % config.diagnostic_interval == 0 or epoch == config.epochs:
            diagnostics = one_step_diagnostics(model, training, physical)
            diagnostics.update(epoch=epoch, K=k)
            diagnostic_log.append(diagnostics)

        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            metrics = evaluate_solver_on_dataset(
                solver="learned", model=model, dataset_cpu=validation_cpu,
                physical=physical, steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size, report_steps=config.report_steps,
                device=device,
            )
            key = validation_selection_key(metrics)
            validation_log.append({"epoch": epoch, "K": k, "selection_key": key, "metrics": metrics})
            if key is not None and (best_key is None or key < best_key):
                best_key = key
                best_epoch = epoch
                best_state = state_dict_to_cpu(model)
            print(
                f"epoch={epoch:4d} K={k} objective={float(objective.item()):.4e} "
                f"val_res_p95={metrics['final_residual_p95']:.4e} "
                f"val_res_max={metrics['final_residual_max']:.4e} "
                f"worst_motion_max={metrics['worst_motion_final_residual_max']:.4e} "
                f"best_epoch={best_epoch} elapsed={time.perf_counter()-start_time:.1f}s"
            )

    last_state = state_dict_to_cpu(model)
    if best_state is None:
        best_state = copy.deepcopy(last_state)
        best_epoch = config.epochs
        best_key = None
    torch.save(last_state, experiment_dir / "last_model_state_dict.pt")
    torch.save(best_state, experiment_dir / "best_validation_model_state_dict.pt")
    torch.save(best_state, experiment_dir / "mlp_optimizer_state_dict.pt")

    model.load_state_dict(best_state)
    model.to(device)
    learned_results: dict[str, Any] = {}
    for name, dataset in evaluation_datasets.items():
        learned_results[name] = evaluate_solver_on_dataset(
            solver="learned", model=model, dataset_cpu=dataset, physical=physical,
            steps=config.evaluation_steps, batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps, device=device,
        )

    comparison = {
        name: {
            "learned": learned_results[name],
            "gradient_descent": shared_baselines[name]["gradient_descent"],
            "full_newton": shared_baselines[name]["full_newton"],
        }
        for name in evaluation_datasets
    }
    report = {
        "experiment_name": experiment_name,
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": best_key,
        "training_dataset": training_cpu.metadata,
        "validation_dataset": validation_cpu.metadata,
        "model": {
            "architecture": f"{FREE_STATE_DIM}D residual -> {HIDDEN_DIM} -> Identity -> {FREE_STATE_DIM}D update",
            "bias_free": True,
            "first_layer_initialization": "orthogonal",
            "output_layer_initialization": "zero",
            "residual_length_scale": config.residual_length_scale,
            "dtype": str(TORCH_DTYPE),
        },
        "training": {
            "optimizer": "Adam", "learning_rate": LEARNING_RATE, "full_batch": True,
            "epochs": config.epochs, "gradient_clip_norm": config.gradient_clip_norm,
            "energy_scale": energy_scale,
        },
        "metric_policy": {
            "pooled": ["mean", "median", "p95", "max"],
            "boundary": ["pooled max", "worst-motion p95", "worst-motion max"],
        },
        "gradient_descent_step_size": gd_step_size,
        "train_log": train_log,
        "diagnostic_log": diagnostic_log,
        "validation_log": validation_log,
        "evaluation": comparison,
    }
    save_json(report, experiment_dir / "experiment_report.json")

    if not config.skip_plots:
        plot_training_curves(train_log, validation_log, best_epoch, experiment_dir / "training_and_validation.png")
        for split_name in ["seen_motion_temporal_interpolation", "seen_motion_temporal_extrapolation", "unseen_id_test", "ood_test"]:
            plot_three_solver_rollout(
                comparison[split_name],
                title=f"{experiment_name}: {split_name}",
                save_path=experiment_dir / f"{split_name}_three_solver_rollout.png",
            )
            plot_per_motion_boundary(
                comparison[split_name],
                title=f"{experiment_name}: {split_name}",
                save_path=experiment_dir / f"{split_name}_per_motion_boundary.png",
            )
    return report
