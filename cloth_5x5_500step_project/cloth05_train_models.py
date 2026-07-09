"""Script 5: train learned optimizer models and evaluate them on test datasets.

Training rule:
    one mini-batch = 16 train motions x 32 time-step problems per motion.
    Each selected time-step problem contributes all sampled initial states.

Inputs:
    data/datasets/train.pt
    data/datasets/validation.pt
    data/datasets/{seen_extrap,unseen_id,ood}.pt
    data/datasets/train_batch_plan.json

Outputs under models/<experiment_name>/:
    config.json
    latest_checkpoint.pt
    best_validation_model.pt
    train_log.csv
    validation_metrics.json
    test_metrics.json
    figures/*.png

Run:
    python cloth05_train_models.py --root cloth_5x5_500step_pipeline --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cloth02_dataset_catalog import load_dataset
from cloth03_solvers_and_models import (
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_EVALUATION_BATCH_SIZE,
    DEFAULT_EVALUATION_STEPS,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_K_VALUES,
    LEARNING_RATE,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    DEFAULT_VALIDATION_INTERVAL,
    HIDDEN_DEPTHS,
    HIDDEN_WIDTHS,
    ACTIVATION_NAMES,
    MLPOptimizer,
    ModelSpec,
    apply_model_update,
    physical_config_from_dict,
    physical_energy_scale,
    project_fixed_vertices,
    stationarity_residual_norm_full,
    variational_energy_full,
)

TEST_DATASETS = ("validation", "seen_extrap", "unseen_id", "ood")


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_physical_config(root: Path):
    runtime = load_json(root / "data" / "reference" / "runtime_config.json")
    return physical_config_from_dict(runtime["physical_config"])


def make_model_specs(activations: list[str], depths: list[int], widths: list[int], use_bias: bool) -> list[ModelSpec]:
    return [
        ModelSpec(activation=activation, depth=depth, width=width, use_bias=use_bias)
        for activation in activations
        for depth in depths
        for width in widths
    ]


def k_for_epoch(epoch: int, k_values: list[int], epochs_per_k: int) -> int:
    index = min((epoch - 1) // epochs_per_k, len(k_values) - 1)
    return int(k_values[index])


def dataset_rows_by_problem(dataset: dict[str, Any]) -> dict[int, torch.Tensor]:
    mapping: dict[int, list[int]] = {}
    for row, problem_index in enumerate(dataset["problem_index"].tolist()):
        mapping.setdefault(int(problem_index), []).append(row)
    return {key: torch.tensor(rows, dtype=torch.long) for key, rows in mapping.items()}


def rows_for_problem_batch(problem_indices: list[int], row_map: dict[int, torch.Tensor]) -> torch.Tensor:
    return torch.cat([row_map[int(problem_index)] for problem_index in problem_indices], dim=0)


def take_rows(dataset: dict[str, Any], rows: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "initial_y": dataset["initial_y"].index_select(0, rows).to(device),
        "q": dataset["q"].index_select(0, rows).to(device),
        "masses": dataset["masses"].index_select(0, rows).to(device),
        "exact_y": dataset["exact_y"].index_select(0, rows).to(device),
    }


def rollout_model_steps(
    model: MLPOptimizer,
    batch: dict[str, torch.Tensor],
    physical,
    steps: int,
) -> torch.Tensor:
    y = project_fixed_vertices(batch["initial_y"].clone(), physical)
    previous_residual = torch.zeros_like(y)
    previous_update = torch.zeros_like(y)
    for _ in range(int(steps)):
        y, delta, current_residual = apply_model_update(
            model,
            y,
            batch["q"],
            batch["masses"],
            physical,
            previous_residual=previous_residual,
            previous_update=previous_update,
        )
        previous_residual = current_residual.detach()
        previous_update = delta.detach()
    return y


def train_batch_loss(
    model: MLPOptimizer,
    batch: dict[str, torch.Tensor],
    physical,
    steps: int,
    residual_length_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    y = rollout_model_steps(model, batch, physical, steps)
    exact_energy = variational_energy_full(batch["exact_y"], batch["q"], batch["masses"], physical).detach()
    energy = variational_energy_full(y, batch["q"], batch["masses"], physical)
    scale = physical_energy_scale(batch["masses"].detach(), physical, residual_length_scale)
    energy_gap = torch.clamp(energy - exact_energy, min=0.0)
    loss = energy_gap.mean() / max(scale, 1e-30)
    with torch.no_grad():
        residual = stationarity_residual_norm_full(y, batch["q"], batch["masses"], physical)
    return loss, {
        "residual_mean": float(residual.mean().detach().cpu().item()),
        "residual_max": float(residual.max().detach().cpu().item()),
        "energy_gap_mean": float(energy_gap.mean().detach().cpu().item()),
    }


def summarize_curve(residual_curve: np.ndarray) -> dict[str, Any]:
    return {
        "residual_mean_by_iter": residual_curve.mean(axis=0).tolist(),
        "residual_max_by_iter": residual_curve.max(axis=0).tolist(),
        "residual_sum_by_iter": residual_curve.sum(axis=0).tolist(),
        "final_residual_mean": float(residual_curve[:, -1].mean()),
        "final_residual_max": float(residual_curve[:, -1].max()),
        "final_residual_sum": float(residual_curve[:, -1].sum()),
        "num_points": int(residual_curve.shape[0]),
        "num_iterations": int(residual_curve.shape[1] - 1),
    }


@torch.no_grad()
def evaluate_model(
    *,
    model: MLPOptimizer,
    dataset: dict[str, Any],
    physical,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    curves: list[torch.Tensor] = []
    start_time = time.perf_counter()
    n = len(dataset["initial_y"])
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = {
            "initial_y": dataset["initial_y"][start:end].to(device),
            "q": dataset["q"][start:end].to(device),
            "masses": dataset["masses"][start:end].to(device),
            "exact_y": dataset["exact_y"][start:end].to(device),
        }
        y = project_fixed_vertices(batch["initial_y"].clone(), physical)
        previous_residual = torch.zeros_like(y)
        previous_update = torch.zeros_like(y)
        batch_curve = []
        for step in range(steps + 1):
            residual = stationarity_residual_norm_full(y, batch["q"], batch["masses"], physical)
            batch_curve.append(residual.detach().cpu())
            if step == steps:
                break
            y, delta, current_residual = apply_model_update(
                model,
                y,
                batch["q"],
                batch["masses"],
                physical,
                previous_residual=previous_residual,
                previous_update=previous_update,
            )
            previous_residual = current_residual.detach()
            previous_update = delta.detach()
        curves.append(torch.stack(batch_curve, dim=1))
    residual_curve = torch.cat(curves, dim=0).numpy().astype(float)
    residual_curve[~np.isfinite(residual_curve)] = np.inf
    summary = summarize_curve(residual_curve)
    summary["elapsed_seconds"] = time.perf_counter() - start_time
    return {"summary": summary, "curve": residual_curve}


def plot_training_log(log_rows: list[dict[str, Any]], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in log_rows]
    losses = [max(row["loss_mean"], 1e-16) for row in log_rows]
    residuals = [max(row["train_residual_mean"], 1e-16) for row in log_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean training loss")
    ax.set_title("training loss")
    fig.tight_layout()
    fig.savefig(figure_dir / "training_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, residuals)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean training residual")
    ax.set_title("training residual")
    fig.tight_layout()
    fig.savefig(figure_dir / "training_residual.png", dpi=180)
    plt.close(fig)


def plot_validation_history(history: list[dict[str, Any]], figure_dir: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    values = [max(row["final_residual_mean"], 1e-16) for row in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, values, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation final residual mean")
    ax.set_title("validation residual")
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_residual.png", dpi=180)
    plt.close(fig)


def plot_test_curves(test_metrics: dict[str, Any], figure_dir: Path) -> None:
    for dataset_name, record in test_metrics.items():
        y = np.asarray(record["residual_mean_by_iter"], dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(len(y)), np.maximum(y, 1e-16))
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean residual")
        ax.set_title(f"{dataset_name}: learned model residual")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{dataset_name}_learned_iteration_vs_residual.png", dpi=180)
        plt.close(fig)


def write_train_log(log_rows: list[dict[str, Any]], path: Path) -> None:
    if not log_rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)


def save_checkpoint(
    *,
    path: Path,
    model: MLPOptimizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_spec: ModelSpec,
    best_validation: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_spec": asdict(model_spec),
            "best_validation": best_validation,
        },
        path,
    )


def train_one_model(
    *,
    root: Path,
    model_spec: ModelSpec,
    args: argparse.Namespace,
    physical,
    device: torch.device,
) -> None:
    train = load_dataset("train", root)
    validation = load_dataset("validation", root)
    batch_plan = load_json(root / "data" / "datasets" / "train_batch_plan.json")
    problem_batches = batch_plan["problem_indices_by_batch"]
    row_map = dataset_rows_by_problem(train)

    output_dir = root / "models" / model_spec.experiment_name
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = MLPOptimizer(args.residual_length_scale, model_spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    config = {
        "model_spec": asdict(model_spec),
        "architecture": model.architecture_description,
        "parameter_count": model.parameter_count,
        "epochs": args.epochs,
        "validation_interval": args.validation_interval,
        "k_values": args.k_values,
        "epochs_per_k": args.epochs_per_k,
        "learning_rate": args.learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "mini_batch_meaning": "16 train motions x 32 time-step problems per motion; each problem uses all sampled initial states",
        "num_batches_per_epoch": len(problem_batches),
        "device": str(device),
    }
    save_json(config, output_dir / "config.json")

    log_rows: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_validation = math.inf

    for epoch in range(1, args.epochs + 1):
        model.train()
        k = k_for_epoch(epoch, args.k_values, args.epochs_per_k)
        loss_values = []
        residual_values = []
        max_values = []
        epoch_start = time.perf_counter()

        for problem_indices in problem_batches:
            rows = rows_for_problem_batch(problem_indices, row_map)
            batch = take_rows(train, rows, device)
            optimizer.zero_grad(set_to_none=True)
            loss, batch_metrics = train_batch_loss(
                model,
                batch,
                physical,
                steps=k,
                residual_length_scale=args.residual_length_scale,
            )
            loss.backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            loss_values.append(float(loss.detach().cpu().item()))
            residual_values.append(batch_metrics["residual_mean"])
            max_values.append(batch_metrics["residual_max"])

        row = {
            "epoch": epoch,
            "k": k,
            "loss_mean": float(np.mean(loss_values)),
            "train_residual_mean": float(np.mean(residual_values)),
            "train_residual_max": float(np.max(max_values)),
            "elapsed_seconds": time.perf_counter() - epoch_start,
        }
        log_rows.append(row)
        print(
            f"{model_spec.experiment_name} epoch {epoch:04d}/{args.epochs} "
            f"K={k:02d} loss={row['loss_mean']:.3e} residual={row['train_residual_mean']:.3e}"
        )

        save_checkpoint(
            path=output_dir / "latest_checkpoint.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            model_spec=model_spec,
            best_validation=best_validation,
        )

        if epoch == 1 or epoch % args.validation_interval == 0 or epoch == args.epochs:
            val_result = evaluate_model(
                model=model,
                dataset=validation,
                physical=physical,
                steps=args.evaluation_steps,
                batch_size=args.evaluation_batch_size,
                device=device,
            )["summary"]
            val_record = {"epoch": epoch, **val_result}
            validation_history.append(val_record)
            save_json({"history": validation_history}, output_dir / "validation_metrics.json")
            if val_result["final_residual_mean"] < best_validation:
                best_validation = val_result["final_residual_mean"]
                save_checkpoint(
                    path=output_dir / "best_validation_model.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    model_spec=model_spec,
                    best_validation=best_validation,
                )
            print(
                f"  validation final mean={val_result['final_residual_mean']:.3e}, "
                f"max={val_result['final_residual_max']:.3e}, best={best_validation:.3e}"
            )

        write_train_log(log_rows, output_dir / "train_log.csv")
        plot_training_log(log_rows, figure_dir)
        plot_validation_history(validation_history, figure_dir)

    checkpoint = torch.load(output_dir / "best_validation_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics: dict[str, Any] = {}
    test_curves: dict[str, Any] = {}
    for dataset_name in args.test_datasets:
        dataset = load_dataset(dataset_name, root)
        result = evaluate_model(
            model=model,
            dataset=dataset,
            physical=physical,
            steps=args.evaluation_steps,
            batch_size=args.evaluation_batch_size,
            device=device,
        )
        test_metrics[dataset_name] = result["summary"]
        test_curves[dataset_name] = torch.from_numpy(result["curve"])
        print(
            f"  test {dataset_name}: mean={result['summary']['final_residual_mean']:.3e}, "
            f"max={result['summary']['final_residual_max']:.3e}"
        )
    save_json(test_metrics, output_dir / "test_metrics.json")
    torch.save(test_curves, output_dir / "test_curves.pt")
    plot_test_curves(test_metrics, figure_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learned optimizer models.")
    parser.add_argument("--root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--evaluation-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--epochs-per-k", type=int, default=100)
    parser.add_argument("--activations", nargs="+", default=list(ACTIVATION_NAMES))
    parser.add_argument("--depths", type=int, nargs="+", default=list(HIDDEN_DEPTHS))
    parser.add_argument("--widths", type=int, nargs="+", default=list(HIDDEN_WIDTHS))
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--test-datasets", nargs="+", default=list(TEST_DATASETS))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    physical = load_physical_config(args.root)
    specs = make_model_specs(args.activations, args.depths, args.widths, args.use_bias)
    if args.config_index is not None:
        specs = [specs[int(args.config_index)]]
    for spec in specs:
        train_one_model(root=args.root, model_spec=spec, args=args, physical=physical, device=device)


if __name__ == "__main__":
    main()
