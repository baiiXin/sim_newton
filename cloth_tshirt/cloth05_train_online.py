"""Train the dense learned optimizer with online-randomized T-shirt motions."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cloth02_batched_physics import load_frozen_motion_batch, load_physics
from cloth03_training_pool import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_K_BUCKETS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_POOL_SIZE,
    LearnedOptimizerMLP,
    ModelSpec,
    OnlineTrainingPool,
    training_step,
)
from cloth04_reference_free_validation import (
    FailureThresholds,
    run_reference_free_validation,
    save_validation_result,
)
from cloth11_plot_training_progress import plot_training_progress
from tshirt_config import DEFAULT_FIXED_DATA_DIR, DEFAULT_TRAIN_SEED, write_json
from validation_protocol import CHECKPOINT_VALIDATION, FAST_MONITOR, checkpoint_rank


DEFAULT_OUTPUT_ROOT = Path("cloth_tshirt_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--activation", default=ModelSpec().activation)
    parser.add_argument("--depth", type=int, default=ModelSpec().depth)
    parser.add_argument("--width", type=int, default=ModelSpec().width)
    parser.add_argument("--use-bias", action=argparse.BooleanOptionalAction, default=ModelSpec().use_bias)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--k-buckets", type=int, nargs="+", default=list(DEFAULT_K_BUCKETS))
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--step-regularization-weight", type=float, default=0.0)
    parser.add_argument("--max-updates", type=int, default=3_000_000)
    parser.add_argument("--max-wall-hours", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument("--fast-validation-interval", type=int, default=FAST_MONITOR.interval_updates)
    parser.add_argument(
        "--checkpoint-validation-interval",
        type=int,
        default=CHECKPOINT_VALIDATION.interval_updates,
    )
    parser.add_argument("--fast-rollout-frames", type=int, default=FAST_MONITOR.rollout_frames)
    parser.add_argument(
        "--checkpoint-rollout-frames", type=int, default=CHECKPOINT_VALIDATION.rollout_frames
    )
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--skip-initial-validation", action="store_true")
    parser.add_argument("--no-validation-plots", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def run_directory(root: Path, model_spec: ModelSpec, seed: int) -> Path:
    return Path(root) / model_spec.experiment_name / f"seed_{seed}"


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if "torch_cuda" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def model_checkpoint_payload(
    *,
    model: LearnedOptimizerMLP,
    optimizer: torch.optim.Optimizer,
    update: int,
    best_rank: tuple[float, ...] | None,
    best_update: int | None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "project": "cloth_tshirt_online_dynamics",
        "update_count": int(update),
        "model_spec": asdict(model.model_spec),
        "residual_length_scale": float(model.residual_length_scale.detach().cpu()),
        "mesh_sha256": model.physics.model.mesh_sha256,
        "dtype": str(model.physics.dtype).replace("torch.", ""),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_rank": None if best_rank is None else list(best_rank),
        "best_validation_update": best_update,
    }


def full_checkpoint_payload(
    *,
    model: LearnedOptimizerMLP,
    optimizer: torch.optim.Optimizer,
    pool: OnlineTrainingPool,
    update: int,
    elapsed_seconds: float,
    best_rank: tuple[float, ...] | None,
    best_update: int | None,
) -> dict[str, Any]:
    payload = model_checkpoint_payload(
        model=model, optimizer=optimizer, update=update,
        best_rank=best_rank, best_update=best_update,
    )
    payload.update(
        {
            "pool_state_dict": pool.state_dict(),
            "rng_state": _rng_state(),
            "elapsed_seconds": float(elapsed_seconds),
        }
    )
    return payload


def load_model_checkpoint(
    path: Path,
    *,
    physics,
    load_optimizer: bool = False,
) -> tuple[LearnedOptimizerMLP, torch.optim.Optimizer | None, dict[str, Any]]:
    payload = torch.load(path, map_location=physics.device, weights_only=False)
    if payload.get("mesh_sha256") != physics.model.mesh_sha256:
        raise ValueError("checkpoint mesh hash does not match the fixed T-shirt model")
    model = LearnedOptimizerMLP(
        physics=physics,
        residual_length_scale=float(payload.get("residual_length_scale", 5e-2)),
        model_spec=ModelSpec(**payload["model_spec"]),
        initialize=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    optimizer = None
    if load_optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return model, optimizer, payload


def _mean_metrics(rows: Sequence[dict[str, Any]], update: int, elapsed: float) -> dict[str, Any]:
    output: dict[str, Any] = {
        "update": int(update),
        "elapsed_seconds": float(elapsed),
        "updates_per_second": float(len(rows) / max(sum(float(row["step_seconds"]) for row in rows), 1e-12)),
    }
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "step_seconds"]
    count_keys = {key for key in keys if key.startswith("resets_") or key == "completed_physical_frames"}
    for key in keys:
        values = [float(row[key]) for row in rows]
        output[key] = int(sum(values)) if key in count_keys else float(np.mean(values))
    return output


def main() -> None:
    args = parse_args()
    if args.max_updates <= 0 or args.log_interval <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("update counts and intervals must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = resolve_dtype(args.dtype)
    physics = load_physics(fixed_data_dir=args.fixed_data_dir, device=args.device, dtype=dtype)
    model_spec = ModelSpec(args.activation, args.depth, args.width, args.use_bias)
    output = (
        Path(args.run_dir)
        if args.run_dir is not None
        else run_directory(args.output_root, model_spec, args.seed)
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "latest_checkpoint.pt"
    best_path = output / "best_validation_model.pt"
    model = LearnedOptimizerMLP(physics=physics, model_spec=model_spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pool = OnlineTrainingPool(
        physics=physics,
        seed=args.seed,
        pool_size=args.pool_size,
        batch_size=args.batch_size,
        k_buckets=args.k_buckets,
    )
    update = 0
    prior_elapsed = 0.0
    best_rank: tuple[float, ...] | None = None
    best_update: int | None = None
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"resume requested but checkpoint is missing: {latest_path}")
        saved = torch.load(latest_path, map_location=physics.device, weights_only=False)
        if saved.get("mesh_sha256") != physics.model.mesh_sha256:
            raise ValueError("resume checkpoint mesh hash mismatch")
        if saved.get("model_spec") != asdict(model_spec):
            raise ValueError("resume checkpoint model specification mismatch")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        pool.load_state_dict(saved["pool_state_dict"])
        _restore_rng_state(saved["rng_state"])
        update = int(saved["update_count"])
        prior_elapsed = float(saved.get("elapsed_seconds", 0.0))
        rank_value = saved.get("best_validation_rank")
        best_rank = None if rank_value is None else tuple(float(value) for value in rank_value)
        best_update = saved.get("best_validation_update")

    fast_protocol = replace(FAST_MONITOR, rollout_frames=args.fast_rollout_frames)
    full_protocol = replace(CHECKPOINT_VALIDATION, rollout_frames=args.checkpoint_rollout_frames)
    validation = load_frozen_motion_batch(
        Path(args.fixed_data_dir) / "validation_32.npz", device=args.device, dtype=dtype
    )
    manifest = {
        "project": "cloth_tshirt_online_dynamics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_model": asdict(physics.model),
        "model_spec": asdict(model_spec),
        "model_parameter_count": model.parameter_count,
        "optimizer": {"name": "Adam", "learning_rate": args.learning_rate},
        "pool": pool.manifest(),
        "training_samples_persisted": False,
        "validation_dataset": str((Path(args.fixed_data_dir) / "validation_32.npz").resolve()),
        "fast_validation": asdict(fast_protocol),
        "checkpoint_validation": asdict(full_protocol),
        "command_arguments": vars(args),
    }
    write_json(output / "run_manifest.json", manifest)
    start = time.monotonic()
    rows: list[dict[str, Any]] = []
    interrupted = False
    last_full_validation_update: int | None = None

    def elapsed() -> float:
        return prior_elapsed + time.monotonic() - start

    def save_latest() -> None:
        _atomic_torch_save(
            full_checkpoint_payload(
                model=model, optimizer=optimizer, pool=pool, update=update,
                elapsed_seconds=elapsed(), best_rank=best_rank, best_update=best_update,
            ),
            latest_path,
        )

    def validate(protocol) -> dict[str, Any]:
        nonlocal best_rank, best_update, last_full_validation_update
        result = run_reference_free_validation(
            model=model,
            physics=physics,
            motions=validation,
            protocol=protocol,
            batch_size=args.validation_batch_size,
            thresholds=FailureThresholds(),
        )
        save_validation_result(
            result=result, output_root=output, update=update,
            render_plots=not args.no_validation_plots,
        )
        if protocol.selects_checkpoint:
            last_full_validation_update = update
            rank = checkpoint_rank(result.summary)
            if best_rank is None or rank < best_rank:
                best_rank, best_update = rank, update
                _atomic_torch_save(
                    model_checkpoint_payload(
                        model=model, optimizer=optimizer, update=update,
                        best_rank=best_rank, best_update=best_update,
                    ),
                    best_path,
                )
        return result.summary

    try:
        if update == 0 and not args.skip_initial_validation:
            summary = validate(fast_protocol)
            print(
                f"initial validation: failed={summary['failed_motion_count']} "
                f"ratio_p95={summary['residual_ratio_p95']:.3e}"
            )
        while update < args.max_updates and elapsed() < args.max_wall_hours * 3600.0:
            step_start = time.monotonic()
            metrics = training_step(
                model=model,
                optimizer=optimizer,
                pool=pool,
                gradient_clip_norm=args.gradient_clip_norm,
                step_regularization_weight=args.step_regularization_weight,
            )
            metrics["step_seconds"] = time.monotonic() - step_start
            rows.append(metrics)
            update += 1
            if update % args.log_interval == 0:
                row = _mean_metrics(rows, update, elapsed())
                _append_csv(output / "training_log.csv", row)
                rows.clear()
                print(
                    f"update={update} loss={row['loss']:.4e} "
                    f"ratio_p95={row['residual_ratio_p95']:.3e} "
                    f"updates/s={row['updates_per_second']:.2f}"
                )
            if update % args.checkpoint_interval == 0:
                save_latest()
                _atomic_torch_save(
                    model_checkpoint_payload(
                        model=model, optimizer=optimizer, update=update,
                        best_rank=best_rank, best_update=best_update,
                    ),
                    output / "periodic" / f"checkpoint_update_{update:09d}.pt",
                )
            if args.fast_validation_interval > 0 and update % args.fast_validation_interval == 0:
                summary = validate(fast_protocol)
                print(
                    f"fast validation update={update}: failed={summary['failed_motion_count']} "
                    f"ratio_p95={summary['residual_ratio_p95']:.3e}"
                )
            if (
                args.checkpoint_validation_interval > 0
                and update % args.checkpoint_validation_interval == 0
            ):
                summary = validate(full_protocol)
                print(
                    f"full validation update={update}: failed={summary['failed_motion_count']} "
                    f"ratio_p95={summary['residual_ratio_p95']:.3e}"
                )
        if not interrupted and last_full_validation_update != update:
            summary = validate(full_protocol)
            print(
                f"final full validation update={update}: failed={summary['failed_motion_count']} "
                f"ratio_p95={summary['residual_ratio_p95']:.3e}"
            )
    except KeyboardInterrupt:
        interrupted = True
        print("training interrupted; saving a resumable checkpoint")
    finally:
        if rows:
            _append_csv(output / "training_log.csv", _mean_metrics(rows, update, elapsed()))
        save_latest()
        try:
            plot_training_progress(output)
        except Exception as error:  # plotting must not destroy a completed checkpoint
            write_json(output / "plotting_error.json", {"error": repr(error)})

    completed = update >= args.max_updates or elapsed() >= args.max_wall_hours * 3600.0
    write_json(
        output / "completed.json",
        {
            "completed": bool(completed and not interrupted),
            "interrupted": interrupted,
            "update_count": update,
            "elapsed_seconds": elapsed(),
            "best_validation_rank": best_rank,
            "best_validation_update": best_update,
            "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path) if best_path.exists() else None,
            "training_plots_generated_automatically": True,
        },
    )
    print(f"training state written to {output}")


if __name__ == "__main__":
    main()
