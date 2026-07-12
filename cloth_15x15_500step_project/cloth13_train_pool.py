"""Metamizer-style live training pool for the 15x15 project.

Pool evolution follows the 5x5 implementation. Model selection and final reporting
use the same offline protocol as every architecture experiment: each validation or
test row is one physical time-step problem initialized at x_n and evaluated for
50 inner iterations by default.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from cloth02_dataset_catalog import DEFAULT_EVALUATION_ITERATIONS, load_dataset
from cloth03_solvers_and_models import (
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    LEARNING_RATE,
    MLPOptimizer,
    ModelSpec,
    apply_model_update,
    physical_energy_scale,
    stationarity_residual_norm_full,
    variational_energy_full,
)
from cloth05_train_models import (
    best_validation_from_history,
    load_training_log,
    load_validation_history,
    save_training_diagnostics,
)
from cloth_common import (
    evaluate_model_iterations,
    load_json,
    load_physical,
    save_json,
    write_csv,
)

_BASE_PATH = (
    Path(__file__).resolve().parent.parent
    / "cloth_5x5_500step_project"
    / "cloth13_train_metamizer_pool_models.py"
)
_spec = importlib.util.spec_from_file_location("_cloth5_pool_shared", _BASE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(_BASE_PATH)
_shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _shared
_spec.loader.exec_module(_shared)
ClothPool = _shared.ClothPool


def save_checkpoint(path, model, optimizer, epoch, updates, spec, best, config, best_epoch=None):
    torch.save(
        {
            "epoch": epoch,
            "update_count": updates,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_spec": asdict(spec),
            "best_validation": best,
            "best_validation_epoch": None if best_epoch is None else int(best_epoch),
            "config": config,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("cloth_15x15_500step_pipeline")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activation", required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--updates-per-epoch", type=int, default=1000)
    parser.add_argument(
        "--k-buckets", type=int, nargs="+", default=[1, 3, 5, 10, 30]
    )
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument(
        "--evaluation-steps", type=int, default=DEFAULT_EVALUATION_ITERATIONS
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE
    )
    parser.add_argument(
        "--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-energy", type=float, default=1e8)
    parser.add_argument("--max-residual", type=float, default=1e8)
    parser.add_argument("--max-abs-position", type=float, default=1e3)
    parser.add_argument("--min-spring-length", type=float, default=1e-8)
    parser.add_argument("--max-spring-length", type=float, default=1e3)
    parser.add_argument("--max-lifetime-physical-steps", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluation_steps <= 0:
        raise ValueError("evaluation-steps must be positive")
    device = torch.device(args.device)
    physical = load_physical(args.root)
    runtime = load_json(args.root / "data" / "reference" / "runtime_config.json")
    motions = list(runtime["motions"])
    manifest = load_json(args.root / "data" / "datasets" / "dataset_manifest.json")
    train_motions = [int(value) for value in manifest["splits"]["train"]]
    validation = load_dataset("validation_xn", args.root)

    spec = ModelSpec(args.activation, args.depth, args.width, args.use_bias)
    output_dir = (
        args.root
        / "experiments"
        / "training_pool"
        / "samples_0000"
        / spec.experiment_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = MLPOptimizer(args.residual_length_scale, spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch, best, update_count = 1, math.inf, 0
    logs = []
    validation_history = []
    best_epoch = None
    latest = output_dir / "latest_checkpoint.pt"
    config = {
        "sample_count": 0,
        "training_method": "Metamizer-style live pool",
        "model_spec": asdict(spec),
        "parameter_count": model.parameter_count,
        "train_motions": train_motions,
        "k_buckets": args.k_buckets,
        "updates_per_epoch": args.updates_per_epoch,
        "epochs": args.epochs,
        "loss": "mean physical energy after one learned update / energy scale",
        "validation": (
            "all validation x_n problems; "
            f"{args.evaluation_steps} inner iterations; no cross-frame propagation"
        ),
        "checkpoint_metric": "final residual p95",
        "evaluation_steps": args.evaluation_steps,
        "residual_length_scale": args.residual_length_scale,
    }
    save_json(config, output_dir / "config.json")

    if args.resume and not args.overwrite:
        logs = load_training_log(output_dir / "train_log.csv")
        validation_history = load_validation_history(output_dir / "validation_metrics.json")
        best_epoch, best = best_validation_from_history(validation_history)
    if args.resume and latest.exists() and not args.overwrite:
        saved = torch.load(latest, map_location=device)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best = float(saved.get("best_validation", math.inf))
        update_count = int(saved.get("update_count", 0))
        saved_best_epoch = saved.get("best_validation_epoch")
        if saved_best_epoch is not None:
            best_epoch = int(saved_best_epoch)

    pool = ClothPool(
        motions=motions,
        motion_indices=train_motions,
        k_buckets=args.k_buckets,
        physical=physical,
        device=device,
        args=args,
    )
    save_json(pool.manifest(), output_dir / "pool_manifest.json")
    scale = physical_energy_scale(
        pool.masses.detach(), physical, args.residual_length_scale
    )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        losses = []
        residuals = []
        reset_totals = {
            key: 0
            for key in [
                "resets_total",
                "resets_nonfinite",
                "resets_energy",
                "resets_residual",
                "resets_position",
                "resets_spring",
                "resets_lifetime",
            ]
        }
        for _ in range(args.updates_per_epoch):
            batch = pool.ask()
            optimizer.zero_grad(set_to_none=True)
            y_next, delta, current_residual = apply_model_update(
                model,
                batch["y"],
                batch["q"],
                batch["masses"],
                physical,
                previous_residual=batch["prev_residual"],
                previous_update=batch["prev_update"],
            )
            energy = variational_energy_full(
                y_next, batch["q"], batch["masses"], physical
            )
            loss = energy.mean() / max(float(scale), 1e-30)
            loss.backward()
            if args.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip_norm
                )
            optimizer.step()
            residual_norm = stationarity_residual_norm_full(
                y_next, batch["q"], batch["masses"], physical
            )
            stats = pool.tell(
                y_next=y_next,
                delta=delta,
                current_residual=current_residual,
                energy=energy,
                residual_norm=residual_norm,
            )
            for key in reset_totals:
                reset_totals[key] += int(stats.get(key, 0))
            losses.append(float(loss.detach().cpu()))
            residuals.append(float(residual_norm.mean().detach().cpu()))
            update_count += 1

        row = {
            "epoch": epoch,
            "update_count": update_count,
            "loss_mean": float(np.mean(losses)),
            "residual_mean": float(np.mean(residuals)),
            "elapsed_seconds": time.perf_counter() - epoch_start,
            **reset_totals,
        }
        logs.append(row)
        write_csv(logs, output_dir / "train_log.csv")
        save_checkpoint(
            latest,
            model,
            optimizer,
            epoch,
            update_count,
            spec,
            best,
            config,
            best_epoch,
        )

        if (
            epoch == 1
            or epoch % args.validation_interval == 0
            or epoch == args.epochs
        ):
            validation_result = evaluate_model_iterations(
                model=model,
                dataset=validation,
                physical=physical,
                steps=args.evaluation_steps,
                device=device,
                batch_size=args.evaluation_batch_size,
            )
            record = {
                "epoch": epoch,
                "update_count": update_count,
                **validation_result["summary"],
            }
            validation_history.append(record)
            save_json(
                {"history": validation_history},
                output_dir / "validation_metrics.json",
            )
            score = float(record["selection_metric"])
            best_path = output_dir / "best_validation_model.pt"
            if (not best_path.exists()) or score < best:
                best = score
                best_epoch = epoch
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    update_count,
                    spec,
                    best,
                    config,
                    best_epoch,
                )
                torch.save(
                    validation_result["curve"],
                    output_dir / "best_validation_curve.pt",
                )
            save_checkpoint(
                latest,
                model,
                optimizer,
                epoch,
                update_count,
                spec,
                best,
                config,
                best_epoch,
            )
            print(
                f"pool epoch={epoch}/{args.epochs} "
                f"loss={row['loss_mean']:.3e} "
                f"validation_final_p95={score:.3e} "
                f"resets={row['resets_total']}"
            )

    training_summary = save_training_diagnostics(
        out=output_dir,
        figure_dir=figure_dir,
        logs=logs,
        history=validation_history,
        best=best,
        best_epoch=best_epoch,
        completed_epoch=args.epochs,
    )

    best_checkpoint = torch.load(
        output_dir / "best_validation_model.pt", map_location=device
    )
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    evaluation_metrics = {}
    evaluation_curves = {}
    for name in ("validation_xn", "test_id_xn", "test_ood_xn", "test_all_xn"):
        result = evaluate_model_iterations(
            model=model,
            dataset=load_dataset(name, args.root),
            physical=physical,
            steps=args.evaluation_steps,
            device=device,
            batch_size=args.evaluation_batch_size,
        )
        evaluation_metrics[name] = result["summary"]
        evaluation_curves[name] = result["curve"]
    save_json(evaluation_metrics, output_dir / "evaluation_metrics.json")
    save_json(
        {k: v for k, v in evaluation_metrics.items() if k != "validation_xn"},
        output_dir / "test_metrics.json",
    )
    torch.save(evaluation_curves, output_dir / "evaluation_curves.pt")
    torch.save(
        {k: v for k, v in evaluation_curves.items() if k != "validation_xn"},
        output_dir / "test_curves.pt",
    )
    save_json(
        {
            "completed": True,
            "best_validation": best,
            "best_checkpoint_epoch": best_epoch,
            "total_training_elapsed_seconds": training_summary["total_training_elapsed_seconds"],
        },
        output_dir / "completed.json",
    )


if __name__ == "__main__":
    main()
