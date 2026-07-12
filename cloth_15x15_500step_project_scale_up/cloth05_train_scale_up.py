"""Train the heterogeneous 15x15 cloth live pool without reference trajectories."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np
import torch

from cloth03_training_pool import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENVIRONMENT_LIFETIME_FRAMES,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_K_BUCKETS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_POOL_SIZE,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    LearnedOptimizerMLP,
    LiveTrainingPool,
    ModelSpec,
    scenario_catalogue_fingerprint,
    training_step,
)
from cloth04_reference_free_validation import (
    FailureThresholds,
    checkpoint_rank,
    run_reference_free_validation,
    save_validation_result,
)
from scenario_catalogue import build_catalogues
from validation_protocol import CHECKPOINT_VALIDATION, FAST_MONITOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("cloth_15x15_scale_up_pipeline"),
    )
    parser.add_argument(
        "--catalogue",
        choices=("c1", "c2", "c3"),
        default="c2",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--activation", default="identity")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument(
        "--residual-length-scale",
        type=float,
        default=DEFAULT_RESIDUAL_LENGTH_SCALE,
    )
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--k-buckets",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_BUCKETS),
    )
    parser.add_argument(
        "--max-lifetime-physical-steps",
        type=int,
        default=DEFAULT_ENVIRONMENT_LIFETIME_FRAMES,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_NORM,
    )
    parser.add_argument("--step-regularization-weight", type=float, default=0.0)
    parser.add_argument("--max-wall-hours", type=float, default=6.0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--latest-checkpoint-interval", type=int, default=1000)
    parser.add_argument("--periodic-checkpoint-interval", type=int, default=10000)
    parser.add_argument(
        "--fast-validation-interval",
        type=int,
        default=FAST_MONITOR.interval_updates,
    )
    parser.add_argument(
        "--checkpoint-validation-interval",
        type=int,
        default=CHECKPOINT_VALIDATION.interval_updates,
    )
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument(
        "--validate-at-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--render-validation-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-energy", type=float, default=1e12)
    parser.add_argument("--max-residual", type=float, default=1e12)
    parser.add_argument("--max-abs-position", type=float, default=1e4)
    parser.add_argument("--min-spring-length", type=float, default=1e-8)
    parser.add_argument("--max-spring-length", type=float, default=1e4)
    parser.add_argument("--validation-min-edge-ratio", type=float, default=1e-5)
    parser.add_argument("--validation-max-edge-ratio", type=float, default=1e4)
    return parser.parse_args()


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _safe(value),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    combined = existing + rows
    fields: list[str] = []
    for row in combined:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)


def catalogue_key(name: str) -> str:
    return {
        "c1": "train_c1_1024",
        "c2": "train_c2_2048",
        "c3": "train_c3_3072",
    }[name]


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    *,
    model: LearnedOptimizerMLP,
    optimizer: torch.optim.Optimizer,
    pool: LiveTrainingPool,
    update_count: int,
    elapsed_training_seconds: float,
    config: dict[str, Any],
    best_rank: tuple[float, float, float, float] | None,
    best_update: int | None,
    include_pool: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "update_count": int(update_count),
        "elapsed_training_seconds": float(elapsed_training_seconds),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_spec": asdict(model.model_spec),
        "config": config,
        "catalogue_fingerprint": pool.catalogue_fingerprint,
        "best_validation_rank": best_rank,
        "best_validation_update": best_update,
        "rng_state": capture_rng_state(),
    }
    if include_pool:
        payload["pool_state_dict"] = pool.state_dict()
    torch.save(payload, path)


def aggregate_training_window(
    metrics: Sequence[dict[str, Any]],
    *,
    update_count: int,
    elapsed_seconds: float,
    interval_seconds: float,
    pool: LiveTrainingPool,
    device: torch.device,
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("metrics must not be empty")
    sum_fields = {
        "resets_total",
        "resets_nonfinite",
        "resets_energy",
        "resets_residual",
        "resets_position",
        "resets_spring",
        "resets_lifetime",
        "completed_physical_frames",
    }
    last_fields = {
        "unique_scenarios_seen",
        "total_scenario_assignments",
        *(f"physical_frames_k{k}" for k in pool.k_buckets),
    }
    row: dict[str, Any] = {
        "update_count": int(update_count),
        "wall_clock_seconds": float(elapsed_seconds),
        "window_seconds": float(interval_seconds),
        "optimizer_updates_per_second": len(metrics) / max(interval_seconds, 1e-30),
        "environment_updates_per_second": (
            len(metrics) * pool.batch_size / max(interval_seconds, 1e-30)
        ),
        "total_environment_updates": int(pool.total_environment_updates),
        "total_completed_physical_frames": int(pool.total_completed_physical_frames),
    }
    fields = sorted(set().union(*(item.keys() for item in metrics)))
    for field in fields:
        values = [
            float(item[field])
            for item in metrics
            if field in item and isinstance(item[field], (int, float))
        ]
        if not values:
            continue
        if field in sum_fields:
            row[field] = int(sum(values))
        elif field in last_fields:
            row[field] = values[-1]
        elif field == "batch_size":
            row[field] = int(values[-1])
        else:
            row[f"{field}_mean"] = float(np.mean(values))
    if device.type == "cuda":
        row["cuda_peak_allocated_gib"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        row["cuda_peak_reserved_gib"] = (
            torch.cuda.max_memory_reserved(device) / 1024**3
        )
    else:
        row["cuda_peak_allocated_gib"] = 0.0
        row["cuda_peak_reserved_gib"] = 0.0
    return row


def output_directory(args: argparse.Namespace, spec: ModelSpec) -> Path:
    return (
        args.root
        / "experiments"
        / f"train_{args.catalogue}"
        / spec.experiment_name
        / f"seed_{args.seed}"
    )


def main() -> None:
    args = parse_args()
    if args.max_wall_hours <= 0:
        raise ValueError("max-wall-hours must be positive")
    for name in (
        "log_interval",
        "latest_checkpoint_interval",
        "periodic_checkpoint_interval",
        "fast_validation_interval",
        "checkpoint_validation_interval",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    device = torch.device(args.device)
    dtype = torch_dtype(args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    catalogues = build_catalogues()
    train_scenarios = tuple(catalogues[catalogue_key(args.catalogue)])
    validation_scenarios = tuple(catalogues["validation_128"])
    fingerprint = scenario_catalogue_fingerprint(train_scenarios)

    spec = ModelSpec(
        activation=args.activation,
        depth=args.depth,
        width=args.width,
        use_bias=args.use_bias,
    )
    out = output_directory(args, spec)
    latest_path = out / "latest_checkpoint.pt"
    best_path = out / "best_validation_model.pt"
    periodic_dir = out / "periodic"

    if out.exists() and args.overwrite and not args.resume:
        import shutil
        shutil.rmtree(out)
    elif out.exists() and not args.resume and not args.overwrite and any(out.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {out}. "
            "Use --resume or --overwrite."
        )
    out.mkdir(parents=True, exist_ok=True)

    pool = LiveTrainingPool(
        scenarios=train_scenarios,
        device=device,
        dtype=dtype,
        pool_size=args.pool_size,
        batch_size=args.batch_size,
        k_buckets=args.k_buckets,
        max_lifetime_physical_steps=args.max_lifetime_physical_steps,
        scenario_offset=args.seed,
        max_energy=args.max_energy,
        max_residual=args.max_residual,
        max_abs_position=args.max_abs_position,
        min_spring_length=args.min_spring_length,
        max_spring_length=args.max_spring_length,
    )
    model = LearnedOptimizerMLP(
        full_state_dim=pool.parameter_bank.full_state_dim,
        residual_length_scale=args.residual_length_scale,
        model_spec=spec,
        dtype=dtype,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )
    fast_protocol = replace(
        FAST_MONITOR,
        interval_updates=args.fast_validation_interval,
    )
    checkpoint_protocol = replace(
        CHECKPOINT_VALIDATION,
        interval_updates=args.checkpoint_validation_interval,
    )
    thresholds = FailureThresholds(
        max_residual=args.max_residual,
        max_abs_position=args.max_abs_position,
        min_edge_ratio=args.validation_min_edge_ratio,
        max_edge_ratio=args.validation_max_edge_ratio,
    )
    config = {
        "training_catalogue": args.catalogue,
        "training_scenario_count": len(train_scenarios),
        "validation_scenario_count": len(validation_scenarios),
        "catalogue_fingerprint": fingerprint,
        "model_spec": asdict(spec),
        "parameter_count": model.parameter_count,
        "dtype": args.dtype,
        "device": str(device),
        "pool": pool.manifest(),
        "learning_rate": args.learning_rate,
        "gradient_clip_norm": args.gradient_clip_norm,
        "step_regularization_weight": args.step_regularization_weight,
        "max_wall_hours": args.max_wall_hours,
        "max_updates": args.max_updates,
        "loss": (
            "mean_i[(E_i(y_after)-stopgrad(E_i(y_before)))/S_i] "
            "+ optional normalized step regularizer"
        ),
        "validation_protocols": {
            "fast_monitor": asdict(fast_protocol),
            "checkpoint_validation": asdict(checkpoint_protocol),
        },
        "checkpoint_selection": [
            "failed_motion_count min",
            "survival_frame_p05 max",
            "residual_ratio_p95 min",
            "energy_increase_fraction min",
        ],
    }
    save_json(out / "config.json", config)
    save_json(out / "pool_manifest.json", pool.manifest())

    update_count = 0
    elapsed_before_resume = 0.0
    best_rank: tuple[float, float, float, float] | None = None
    best_update: int | None = None
    last_checkpoint_validation_update: int | None = None
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(latest_path)
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        if saved["catalogue_fingerprint"] != fingerprint:
            raise ValueError("Resume checkpoint uses a different scenario catalogue")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        pool.load_state_dict(saved["pool_state_dict"])
        update_count = int(saved["update_count"])
        elapsed_before_resume = float(saved.get("elapsed_training_seconds", 0.0))
        saved_rank = saved.get("best_validation_rank")
        if saved_rank is not None:
            best_rank = tuple(float(value) for value in saved_rank)
        saved_best_update = saved.get("best_validation_update")
        best_update = None if saved_best_update is None else int(saved_best_update)
        restore_rng_state(saved["rng_state"])

    run_start = time.perf_counter()
    window_start = run_start
    window_metrics: list[dict[str, Any]] = []
    train_log_path = out / "train_log.csv"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def elapsed() -> float:
        return elapsed_before_resume + (time.perf_counter() - run_start)

    def run_validation(protocol) -> dict[str, Any]:
        nonlocal best_rank, best_update, last_checkpoint_validation_update
        result = run_reference_free_validation(
            model=model,
            scenarios=validation_scenarios,
            protocol=protocol,
            device=device,
            dtype=dtype,
            batch_size=args.validation_batch_size,
            thresholds=thresholds,
        )
        save_validation_result(
            result=result,
            output_root=out,
            update_count=update_count,
            wall_clock_seconds=elapsed(),
            render_plots=args.render_validation_plots,
        )
        rank = checkpoint_rank(result.summary)
        if protocol.selects_checkpoint:
            last_checkpoint_validation_update = update_count
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_update = update_count
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    pool=pool,
                    update_count=update_count,
                    elapsed_training_seconds=elapsed(),
                    config=config,
                    best_rank=best_rank,
                    best_update=best_update,
                    include_pool=False,
                )
        return result.summary

    if args.validate_at_start and update_count == 0:
        summary = run_validation(fast_protocol)
        print(
            f"initial fast validation: failed={summary['failed_motion_count']} "
            f"residual_p95={summary['residual_ratio_p95']:.3e}"
        )

    stop_reason = "unknown"
    model.train()
    while True:
        if args.max_updates > 0 and update_count >= args.max_updates:
            stop_reason = "max_updates"
            break
        if elapsed() >= args.max_wall_hours * 3600.0:
            stop_reason = "max_wall_hours"
            break

        metrics = training_step(
            model=model,
            optimizer=optimizer,
            pool=pool,
            gradient_clip_norm=args.gradient_clip_norm,
            step_regularization_weight=args.step_regularization_weight,
        )
        update_count += 1
        window_metrics.append(metrics)

        if update_count % args.log_interval == 0:
            now = time.perf_counter()
            row = aggregate_training_window(
                window_metrics,
                update_count=update_count,
                elapsed_seconds=elapsed(),
                interval_seconds=now - window_start,
                pool=pool,
                device=device,
            )
            append_csv(train_log_path, [row])
            window_metrics.clear()
            window_start = now
            print(
                f"update={update_count} "
                f"loss={row.get('loss_mean', float('nan')):.3e} "
                f"residual_ratio_p95={row.get('residual_ratio_p95_mean', float('nan')):.3e} "
                f"seen={int(row.get('unique_scenarios_seen', 0))}/{len(train_scenarios)} "
                f"resets={int(row.get('resets_total', 0))}"
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        if update_count % fast_protocol.interval_updates == 0:
            summary = run_validation(fast_protocol)
            print(
                f"fast validation update={update_count}: "
                f"failed={summary['failed_motion_count']} "
                f"residual_p95={summary['residual_ratio_p95']:.3e}"
            )
            model.train()

        if update_count % checkpoint_protocol.interval_updates == 0:
            summary = run_validation(checkpoint_protocol)
            print(
                f"checkpoint validation update={update_count}: "
                f"failed={summary['failed_motion_count']} "
                f"survival_p05={summary['survival_frame_p05']:.1f} "
                f"residual_p95={summary['residual_ratio_p95']:.3e}"
            )
            model.train()

        if update_count % args.latest_checkpoint_interval == 0:
            save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                pool=pool,
                update_count=update_count,
                elapsed_training_seconds=elapsed(),
                config=config,
                best_rank=best_rank,
                best_update=best_update,
                include_pool=True,
            )

        if update_count % args.periodic_checkpoint_interval == 0:
            save_checkpoint(
                periodic_dir / f"checkpoint_update_{update_count:09d}.pt",
                model=model,
                optimizer=optimizer,
                pool=pool,
                update_count=update_count,
                elapsed_training_seconds=elapsed(),
                config=config,
                best_rank=best_rank,
                best_update=best_update,
                include_pool=True,
            )

    if window_metrics:
        now = time.perf_counter()
        row = aggregate_training_window(
            window_metrics,
            update_count=update_count,
            elapsed_seconds=elapsed(),
            interval_seconds=now - window_start,
            pool=pool,
            device=device,
        )
        append_csv(train_log_path, [row])

    if last_checkpoint_validation_update != update_count:
        summary = run_validation(checkpoint_protocol)
        print(
            f"final checkpoint validation update={update_count}: "
            f"failed={summary['failed_motion_count']} "
            f"residual_p95={summary['residual_ratio_p95']:.3e}"
        )

    save_checkpoint(
        latest_path,
        model=model,
        optimizer=optimizer,
        pool=pool,
        update_count=update_count,
        elapsed_training_seconds=elapsed(),
        config=config,
        best_rank=best_rank,
        best_update=best_update,
        include_pool=True,
    )
    save_json(
        out / "completed.json",
        {
            "completed": True,
            "stop_reason": stop_reason,
            "update_count": update_count,
            "elapsed_training_seconds": elapsed(),
            "best_validation_rank": best_rank,
            "best_validation_update": best_update,
            "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path) if best_path.exists() else None,
            "unique_scenarios_seen": int(pool.seen_scenarios.sum().item()),
            "total_environment_updates": int(pool.total_environment_updates),
            "total_completed_physical_frames": int(
                pool.total_completed_physical_frames
            ),
            "reset_counts": pool.reset_counts,
        },
    )


if __name__ == "__main__":
    main()
