#!/usr/bin/env python3
"""Train MLP learned optimizers from an existing scaled dataset.

The script only reads a training dataset and validation_core. It does not build
reference trajectories and does not run Newton/GD/L-BFGS baselines.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import torch

from cloth_scale_common import (
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    MODEL_INPUT_DIM,
    MODEL_INPUT_SIGNATURE,
    LearnedOptimizerState,
    MLPOptimizer,
    ModelSpec,
    ProblemTable,
    SampleSplit,
    apply_model_update,
    default_physical_config,
    evaluate_learned_or_gd,
    load_dataset_package,
    load_json,
    physical_energy_scale,
    resolve_batch,
    save_checkpoint_atomic,
    save_json,
    stable_hash,
    validate_device,
    validation_selection_key,
    variational_energy,
)

MODEL_RANDOM_SEED = 42
LEARNING_RATE = 1e-3


def load_benchmark_validation(benchmark_root: Path, split_name: str) -> tuple[dict[str, Any], ProblemTable, SampleSplit]:
    split_dir = benchmark_root / split_name
    manifest = load_json(split_dir / "manifest.json")
    problems = ProblemTable.from_serializable(
        torch.load(split_dir / manifest.get("problem_file", "problems.pt"), map_location="cpu", weights_only=False)
    )
    samples = SampleSplit.from_serializable(
        torch.load(split_dir / manifest.get("sample_file", "samples.pt"), map_location="cpu", weights_only=False)
    )
    return manifest, problems, samples


def parse_int_tuple(values: Sequence[str | int]) -> tuple[int, ...]:
    return tuple(int(v) for v in values)


def get_k(epoch_index: int, k_values: tuple[int, ...], epochs_per_k: int) -> int:
    stage = min(epoch_index // epochs_per_k, len(k_values) - 1)
    return int(k_values[stage])


def build_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = [
        ModelSpec(activation=a, depth=d, width=w, use_bias=args.with_bias)
        for a in args.activations
        for d in args.depths
        for w in args.widths
    ]
    if args.config_index is not None:
        if args.config_index < 0 or args.config_index >= len(specs):
            raise ValueError(f"config-index must be in [0,{len(specs)-1}]")
        specs = [specs[args.config_index]]
    return specs


def write_train_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_one(
    *,
    model_spec: ModelSpec,
    dataset_manifest: dict[str, Any],
    train_problems: ProblemTable,
    train_split: SampleSplit,
    validation_manifest: dict[str, Any],
    validation_problems: ProblemTable,
    validation_split: SampleSplit,
    benchmark_id: str,
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    physical = default_physical_config()
    dataset_id = dataset_manifest["dataset_id"]
    run_core = {
        "dataset_id": dataset_id,
        "benchmark_id": benchmark_id,
        "model_spec": asdict(model_spec),
        "model_input_signature": MODEL_INPUT_SIGNATURE,
        "model_input_dim": MODEL_INPUT_DIM,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "k_values": args.k_values,
        "epochs_per_k": args.epochs_per_k,
        "lr": args.learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "residual_length_scale": args.residual_length_scale,
        "seed": args.seed,
    }
    run_id = f"{model_spec.experiment_name}_{stable_hash(run_core, 10)}"
    run_dir = output_root / dataset_manifest["dataset_spec"]["name"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    final_report_path = run_dir / "training_report.json"
    latest_path = run_dir / "latest_checkpoint.pt"
    best_path = run_dir / "best_validation_model.pt"

    if final_report_path.exists() and args.skip_completed:
        print(f"Skipping completed run: {run_dir}")
        return load_json(final_report_path)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = MLPOptimizer(args.residual_length_scale, model_spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    energy_scale = physical_energy_scale(train_problems.masses, physical, args.residual_length_scale)

    start_epoch = 0
    train_log: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch: int | None = None
    elapsed_before = 0.0

    if args.resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint["dataset_id"] != dataset_id:
            raise RuntimeError("Resume checkpoint dataset mismatch")
        if checkpoint["model_spec"] != asdict(model_spec):
            raise RuntimeError("Resume checkpoint model mismatch")
        if checkpoint.get("model_input_signature") != MODEL_INPUT_SIGNATURE:
            raise RuntimeError(
                "Resume checkpoint uses the legacy 250D fixed-one-hot input. "
                "The current model uses a 225D history-only input; start a new run."
            )
        if int(checkpoint.get("model_input_dim", -1)) != MODEL_INPUT_DIM:
            raise RuntimeError("Resume checkpoint input dimension mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        train_log = list(checkpoint.get("train_log", []))
        validation_log = list(checkpoint.get("validation_log", []))
        saved_key = checkpoint.get("best_key")
        best_key = tuple(saved_key) if saved_key is not None else None
        best_epoch = checkpoint.get("best_epoch")
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        print(f"Resumed {run_id} from epoch {start_epoch}")

    print("\n" + "=" * 100)
    print(f"Training {run_id}")
    print(f"dataset={dataset_id}, samples={len(train_split):,}, problems={len(train_problems):,}")
    print(f"architecture={model.architecture_description}")
    print(f"parameters={model.parameter_count:,}, device={device}, dtype=float64, batch_size={args.batch_size}")
    print("=" * 100)

    wall_start = time.perf_counter()

    for epoch_index in range(start_epoch, args.epochs):
        epoch = epoch_index + 1
        k = get_k(epoch_index, args.k_values, args.epochs_per_k)
        model.train()
        epoch_generator = torch.Generator(device="cpu")
        epoch_generator.manual_seed(args.seed + epoch_index)
        permutation = torch.randperm(len(train_split), generator=epoch_generator)
        objective_weighted = 0.0
        final_gap_weighted = 0.0
        grad_norm_max = 0.0
        processed = 0

        for begin in range(0, len(train_split), args.batch_size):
            sample_indices = permutation[begin: begin + args.batch_size]
            batch = resolve_batch(train_split, train_problems, sample_indices, device)
            y = batch.initial_y
            state = LearnedOptimizerState.zeros_like(y)
            initial_energy = variational_energy(batch.initial_y, batch.q, batch.masses, physical).detach()
            exact_energy = variational_energy(batch.exact_y, batch.q, batch.masses, physical).detach()
            optimizer.zero_grad(set_to_none=True)
            objective_sum = torch.zeros((), dtype=torch.float64, device=device)
            final_gap = torch.zeros((), dtype=torch.float64, device=device)

            for _ in range(k):
                y, _, state = apply_model_update(
                    model,
                    y,
                    batch.q,
                    batch.masses,
                    batch.fixed_mask,
                    batch.fixed_target,
                    physical,
                    state,
                )
                energy = variational_energy(y, batch.q, batch.masses, physical)
                objective_sum = objective_sum + ((energy - initial_energy) / energy_scale).mean()
                final_gap = (energy - exact_energy).mean()

            objective = objective_sum / float(k)
            if not bool(torch.isfinite(objective)):
                raise RuntimeError(f"Non-finite objective at epoch={epoch}, batch_begin={begin}")
            objective.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm).item())
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient at epoch={epoch}, batch_begin={begin}")
            optimizer.step()
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise RuntimeError(f"Non-finite model parameter at epoch={epoch}")

            n = int(sample_indices.numel())
            objective_weighted += float(objective.item()) * n
            final_gap_weighted += float(final_gap.item()) * n
            grad_norm_max = max(grad_norm_max, grad_norm)
            processed += n

        epoch_record = {
            "epoch": epoch,
            "K": k,
            "dimensionless_objective": objective_weighted / max(processed, 1),
            "final_energy_gap": final_gap_weighted / max(processed, 1),
            "max_gradient_norm_before_clip": grad_norm_max,
            "samples_processed": processed,
        }
        train_log.append(epoch_record)

        should_validate = epoch == 1 or epoch % args.validation_interval == 0 or epoch == args.epochs
        if should_validate:
            metrics = evaluate_learned_or_gd(
                solver="learned",
                model=model,
                problems=validation_problems,
                split=validation_split,
                physical=physical,
                steps=args.evaluation_steps,
                batch_size=args.evaluation_batch_size,
                device=device,
            )
            key = validation_selection_key(metrics)
            validation_record = {"epoch": epoch, "K": k, "selection_key": key, "metrics": metrics}
            validation_log.append(validation_record)
            print(
                f"epoch={epoch:5d}/{args.epochs}, K={k:2d}, "
                f"train_obj={epoch_record['dimensionless_objective']:.6e}, "
                f"val_res_p95={metrics['final_residual_p95']:.6e}, "
                f"val_res_max={metrics['final_residual_max']:.6e}, "
                f"worst_boundary={metrics['worst_boundary_final_residual_max']:.6e}"
            )
            if key is not None and (best_key is None or key < best_key):
                best_key = key
                best_epoch = epoch
                save_checkpoint_atomic({
                    "schema_version": 1,
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "benchmark_id": benchmark_id,
                    "validation_split_id": validation_manifest["split_id"],
                    "model_spec": asdict(model_spec),
                    "model_input_signature": MODEL_INPUT_SIGNATURE,
                    "model_input_dim": MODEL_INPUT_DIM,
                    "residual_length_scale": args.residual_length_scale,
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "best_epoch": best_epoch,
                    "best_key": best_key,
                    "validation_metrics": metrics,
                }, best_path)

            elapsed = elapsed_before + (time.perf_counter() - wall_start)
            save_checkpoint_atomic({
                "schema_version": 1,
                "run_id": run_id,
                "dataset_id": dataset_id,
                "benchmark_id": benchmark_id,
                "model_spec": asdict(model_spec),
                "model_input_signature": MODEL_INPUT_SIGNATURE,
                "model_input_dim": MODEL_INPUT_DIM,
                "residual_length_scale": args.residual_length_scale,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_key": best_key,
                "train_log": train_log,
                "validation_log": validation_log,
                "elapsed_seconds": elapsed,
                "run_config": run_core,
            }, latest_path)
            save_json({
                "run_id": run_id,
                "status": "running",
                "dataset_id": dataset_id,
                "model_spec": asdict(model_spec),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_key": best_key,
            }, run_dir / "status.json")
            write_train_csv(train_log, run_dir / "train_log.csv")

    elapsed = elapsed_before + (time.perf_counter() - wall_start)
    if not best_path.exists():
        raise RuntimeError("Training completed without a finite validation checkpoint")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_validation = evaluate_learned_or_gd(
        solver="learned",
        model=model,
        problems=validation_problems,
        split=validation_split,
        physical=physical,
        steps=args.evaluation_steps,
        batch_size=args.evaluation_batch_size,
        device=device,
    )
    report = {
        "status": "success",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dataset_id": dataset_id,
        "benchmark_id": benchmark_id,
        "validation_split_id": validation_manifest["split_id"],
        "model_spec": asdict(model_spec),
        "model_input_signature": MODEL_INPUT_SIGNATURE,
        "model_input_dim": MODEL_INPUT_DIM,
        "architecture": model.architecture_description,
        "parameter_count": model.parameter_count,
        "run_config": run_core,
        "best_epoch": best_epoch,
        "best_key": best_key,
        "best_validation_metrics": final_validation,
        "elapsed_seconds": elapsed,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    save_json(report, final_report_path)
    save_json({"run_id": run_id, "status": "success", "best_epoch": best_epoch}, run_dir / "status.json")
    write_train_csv(train_log, run_dir / "train_log.csv")
    save_json({"validation_log": validation_log}, run_dir / "validation_log.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--validation-split", default="validation_core")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "model_runs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activations", nargs="+", default=["identity", "relu", "tanh"])
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 5, 10])
    parser.add_argument("--widths", nargs="+", type=int, default=[75, 128, 256])
    parser.add_argument("--with-bias", action="store_true")
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10, 30])
    parser.add_argument("--epochs-per-k", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--evaluation-steps", type=int, default=50)
    parser.add_argument("--evaluation-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--seed", type=int, default=MODEL_RANDOM_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.k_values = tuple(args.k_values)
    if args.epochs_per_k <= 0 or args.epochs <= 0:
        raise ValueError("epochs and epochs-per-k must be positive")
    device = torch.device(args.device)
    validate_device(device)
    dataset_manifest, train_problems, train_split = load_dataset_package(args.dataset_dir.resolve())
    benchmark_manifest = load_json(args.benchmark_root.resolve() / "manifest.json")
    validation_manifest, validation_problems, validation_split = load_benchmark_validation(args.benchmark_root.resolve(), args.validation_split)
    specs = build_specs(args)
    if args.list_configs:
        for i, spec in enumerate(specs):
            print(i, spec.experiment_name)
        return

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for i, spec in enumerate(specs, start=1):
        print(f"\nConfiguration {i}/{len(specs)}: {spec.experiment_name}")
        try:
            report = train_one(
                model_spec=spec,
                dataset_manifest=dataset_manifest,
                train_problems=train_problems,
                train_split=train_split,
                validation_manifest=validation_manifest,
                validation_problems=validation_problems,
                validation_split=validation_split,
                benchmark_id=benchmark_manifest["benchmark_id"],
                args=args,
                device=device,
                output_root=output_root,
            )
        except Exception as exc:
            report = {
                "status": "failed",
                "model_spec": asdict(spec),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            failure_dir = output_root / dataset_manifest["dataset_spec"]["name"] / f"FAILED_{spec.experiment_name}"
            failure_dir.mkdir(parents=True, exist_ok=True)
            save_json(report, failure_dir / "failure_report.json")
            print(report["error"])
        reports.append(report)
        save_json({
            "dataset_id": dataset_manifest["dataset_id"],
            "benchmark_id": benchmark_manifest["benchmark_id"],
            "reports": reports,
        }, output_root / dataset_manifest["dataset_spec"]["name"] / "all_training_runs.json")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nTraining requests completed.")
    print(f"Output root: {output_root / dataset_manifest['dataset_spec']['name']}")


if __name__ == "__main__":
    main()
