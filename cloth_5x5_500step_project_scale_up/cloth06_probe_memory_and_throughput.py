"""Probe full training-step memory and throughput for the scale-up cloth pool.

The parent process launches one fresh worker process per batch size. This isolates
CUDA allocator state and allows an out-of-memory failure to be recorded without
poisoning later measurements.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import torch

from cloth03_training_pool import (
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_K_BUCKETS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_POOL_SIZE,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    LearnedOptimizerMLP,
    LiveTrainingPool,
    ModelSpec,
    training_step,
)
from scenario_catalogue import build_catalogues

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cloth_5x5_scale_up_pipeline/profiling/memory_probe"),
    )
    parser.add_argument("--catalogue", choices=("c1", "c2", "c3"), default="c2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--activation", default=ModelSpec().activation)
    parser.add_argument("--depth", type=int, default=ModelSpec().depth)
    parser.add_argument("--width", type=int, default=ModelSpec().width)
    parser.add_argument(
        "--use-bias",
        action=argparse.BooleanOptionalAction,
        default=ModelSpec().use_bias,
    )
    parser.add_argument(
        "--residual-length-scale",
        type=float,
        default=DEFAULT_RESIDUAL_LENGTH_SCALE,
    )
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument(
        "--k-buckets",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_BUCKETS),
    )
    parser.add_argument("--warmup-updates", type=int, default=20)
    parser.add_argument("--measured-updates", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_NORM,
    )
    parser.add_argument("--step-regularization-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--memory-headroom-fraction",
        type=float,
        default=0.85,
        help="Only configurations below this reserved-memory fraction are recommended.",
    )
    parser.add_argument(
        "--stop-after-oom",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keep-worker-files",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # Internal worker mode. Users normally do not set these directly.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def catalogue_key(name: str) -> str:
    return {
        "c1": "train_c1_1024",
        "c2": "train_c2_2048",
        "c3": "train_c3_3072",
    }[name]


def torch_dtype(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _safe(value) for key, value in row.items()} for row in rows])


def validate_common_args(args: argparse.Namespace) -> None:
    if args.pool_size <= 0:
        raise ValueError("pool-size must be positive")
    if not args.batch_sizes or any(value <= 0 for value in args.batch_sizes):
        raise ValueError("batch-sizes must contain positive integers")
    if args.warmup_updates < 0:
        raise ValueError("warmup-updates must be nonnegative")
    if args.measured_updates <= 0:
        raise ValueError("measured-updates must be positive")
    if not 0.0 < args.memory_headroom_fraction <= 1.0:
        raise ValueError("memory-headroom-fraction must be in (0, 1]")
    if not args.k_buckets or any(value <= 0 for value in args.k_buckets):
        raise ValueError("k-buckets must contain positive integers")
    if args.pool_size % len(args.k_buckets) != 0:
        raise ValueError("pool-size must be divisible by the number of K buckets")
    for batch_size in args.batch_sizes:
        if batch_size > args.pool_size:
            raise ValueError("every batch size must not exceed pool-size")
        if batch_size % len(args.k_buckets) != 0:
            raise ValueError("every batch size must be divisible by the number of K buckets")


def device_snapshot(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device_name": "cpu",
            "device_total_gib": 0.0,
            "device_free_gib": 0.0,
            "allocated_gib": 0.0,
            "reserved_gib": 0.0,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device_name": torch.cuda.get_device_name(device),
        "device_total_gib": total_bytes / GIB,
        "device_free_gib": free_bytes / GIB,
        "allocated_gib": torch.cuda.memory_allocated(device) / GIB,
        "reserved_gib": torch.cuda.memory_reserved(device) / GIB,
    }


def _is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    oom_types = tuple(
        candidate
        for candidate in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(torch.cuda, "OutOfMemoryError", None),
        )
        if isinstance(candidate, type)
    )
    return isinstance(error, oom_types) or "out of memory" in text


def _time_training_updates(
    *,
    model: LearnedOptimizerMLP,
    optimizer: torch.optim.Optimizer,
    pool: LiveTrainingPool,
    count: int,
    device: torch.device,
    gradient_clip_norm: float,
    step_regularization_weight: float,
) -> tuple[list[float], dict[str, Any]]:
    times: list[float] = []
    last_metrics: dict[str, Any] = {}
    if device.type == "cuda":
        events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        for _ in range(count):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            last_metrics = training_step(
                model=model,
                optimizer=optimizer,
                pool=pool,
                gradient_clip_norm=gradient_clip_norm,
                step_regularization_weight=step_regularization_weight,
            )
            end.record()
            events.append((start, end))
        torch.cuda.synchronize(device)
        times = [start.elapsed_time(end) / 1000.0 for start, end in events]
    else:
        for _ in range(count):
            start = time.perf_counter()
            last_metrics = training_step(
                model=model,
                optimizer=optimizer,
                pool=pool,
                gradient_clip_norm=gradient_clip_norm,
                step_regularization_weight=step_regularization_weight,
            )
            times.append(time.perf_counter() - start)
    return times, last_metrics


def run_worker(args: argparse.Namespace) -> int:
    if args.batch_size is None or args.result_file is None:
        raise ValueError("worker mode requires --batch-size and --result-file")
    batch_size = int(args.batch_size)
    result: dict[str, Any] = {
        "status": "error",
        "batch_size": batch_size,
        "pool_size": int(args.pool_size),
        "catalogue": args.catalogue,
        "dtype": args.dtype,
        "device": args.device,
        "warmup_updates": int(args.warmup_updates),
        "measured_updates": int(args.measured_updates),
    }
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        if batch_size <= 0 or batch_size > args.pool_size:
            raise ValueError("invalid worker batch size")
        if batch_size % len(args.k_buckets) != 0:
            raise ValueError("worker batch size must be divisible by K bucket count")

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        dtype = torch_dtype(args.dtype)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        result.update({f"before_{key}": value for key, value in device_snapshot(device).items()})

        scenarios = tuple(build_catalogues()[catalogue_key(args.catalogue)])
        spec = ModelSpec(
            activation=args.activation,
            depth=args.depth,
            width=args.width,
            use_bias=args.use_bias,
        )
        pool = LiveTrainingPool(
            scenarios=scenarios,
            device=device,
            dtype=dtype,
            pool_size=args.pool_size,
            batch_size=batch_size,
            k_buckets=args.k_buckets,
            scenario_offset=args.seed,
        )
        model = LearnedOptimizerMLP(
            full_state_dim=pool.parameter_bank.full_state_dim,
            residual_length_scale=args.residual_length_scale,
            model_spec=spec,
            dtype=dtype,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        result.update(
            {
                "scenario_count": len(scenarios),
                "model_spec": asdict(spec),
                "parameter_count": int(model.parameter_count),
                "full_state_dim": int(pool.parameter_bank.full_state_dim),
                "k_buckets": list(pool.k_buckets),
                "batch_per_k": int(pool.batch_per_k),
            }
        )
        result.update({f"initialized_{key}": value for key, value in device_snapshot(device).items()})

        if args.warmup_updates:
            _time_training_updates(
                model=model,
                optimizer=optimizer,
                pool=pool,
                count=args.warmup_updates,
                device=device,
                gradient_clip_norm=args.gradient_clip_norm,
                step_regularization_weight=args.step_regularization_weight,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["baseline_allocated_gib"] = torch.cuda.memory_allocated(device) / GIB
            result["baseline_reserved_gib"] = torch.cuda.memory_reserved(device) / GIB
            torch.cuda.reset_peak_memory_stats(device)
        else:
            result["baseline_allocated_gib"] = 0.0
            result["baseline_reserved_gib"] = 0.0

        times, last_metrics = _time_training_updates(
            model=model,
            optimizer=optimizer,
            pool=pool,
            count=args.measured_updates,
            device=device,
            gradient_clip_norm=args.gradient_clip_norm,
            step_regularization_weight=args.step_regularization_weight,
        )
        if not times or any(not math.isfinite(value) or value <= 0 for value in times):
            raise RuntimeError("invalid timing samples")
        times_array = np.asarray(times, dtype=np.float64)
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device) / GIB
            peak_reserved = torch.cuda.max_memory_reserved(device) / GIB
            total_gib = float(result["before_device_total_gib"])
        else:
            peak_allocated = 0.0
            peak_reserved = 0.0
            total_gib = 0.0
        mean_seconds = float(times_array.mean())
        result.update(
            {
                "status": "success",
                "peak_allocated_gib": peak_allocated,
                "peak_reserved_gib": peak_reserved,
                "peak_allocated_fraction_of_total": (
                    peak_allocated / total_gib if total_gib > 0 else 0.0
                ),
                "peak_reserved_fraction_of_total": (
                    peak_reserved / total_gib if total_gib > 0 else 0.0
                ),
                "update_seconds_mean": mean_seconds,
                "update_seconds_p50": float(np.quantile(times_array, 0.50)),
                "update_seconds_p95": float(np.quantile(times_array, 0.95)),
                "update_seconds_max": float(times_array.max()),
                "optimizer_updates_per_second": 1.0 / mean_seconds,
                "environment_updates_per_second": batch_size / mean_seconds,
                "estimated_optimizer_updates_6h": int(6.0 * 3600.0 / mean_seconds),
                "estimated_environment_updates_6h": int(
                    6.0 * 3600.0 * batch_size / mean_seconds
                ),
                "last_loss": float(last_metrics.get("loss", float("nan"))),
                "last_residual_ratio_p95": float(
                    last_metrics.get("residual_ratio_p95", float("nan"))
                ),
                "last_energy_increase_fraction": float(
                    last_metrics.get("energy_increase_fraction", float("nan"))
                ),
                "total_environment_updates": int(pool.total_environment_updates),
                "total_completed_physical_frames": int(
                    pool.total_completed_physical_frames
                ),
            }
        )
        write_json(args.result_file, result)
        print(json.dumps(_safe(result), ensure_ascii=False))
        return 0
    except BaseException as error:
        result.update(
            {
                "status": "oom" if _is_oom(error) else "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        if torch.cuda.is_available():
            try:
                device = torch.device(args.device)
                if device.type == "cuda":
                    result.update(
                        {
                            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
                            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / GIB,
                        }
                    )
            except BaseException:
                pass
        write_json(args.result_file, result)
        print(json.dumps(_safe(result), ensure_ascii=False), file=sys.stderr)
        return 2 if result["status"] == "oom" else 1


def worker_command(args: argparse.Namespace, batch_size: int, result_file: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--result-file",
        str(result_file),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(args.output_dir),
        "--catalogue",
        args.catalogue,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--activation",
        args.activation,
        "--depth",
        str(args.depth),
        "--width",
        str(args.width),
        "--residual-length-scale",
        str(args.residual_length_scale),
        "--pool-size",
        str(args.pool_size),
        "--batch-sizes",
        str(batch_size),
        "--k-buckets",
        *[str(value) for value in args.k_buckets],
        "--warmup-updates",
        str(args.warmup_updates),
        "--measured-updates",
        str(args.measured_updates),
        "--learning-rate",
        str(args.learning_rate),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--step-regularization-weight",
        str(args.step_regularization_weight),
        "--seed",
        str(args.seed),
        "--memory-headroom-fraction",
        str(args.memory_headroom_fraction),
    ]
    command.append("--use-bias" if args.use_bias else "--no-use-bias")
    return command


def recommendation(rows: Sequence[dict[str, Any]], headroom: float) -> dict[str, Any]:
    success = [row for row in rows if row.get("status") == "success"]
    eligible = [
        row
        for row in success
        if float(row.get("peak_reserved_fraction_of_total", 0.0)) <= headroom
        or float(row.get("before_device_total_gib", 0.0)) == 0.0
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(row.get("environment_updates_per_second", 0.0)),
            float(row.get("peak_reserved_fraction_of_total", 0.0)),
            int(row["batch_size"]),
        ),
    )
    selected = ranked[0] if ranked else None
    return {
        "memory_headroom_fraction": float(headroom),
        "recommended_batch_size": None if selected is None else int(selected["batch_size"]),
        "recommended_pool_size": None if selected is None else int(selected["pool_size"]),
        "selection_rule": (
            "Among successful configurations within the reserved-memory headroom, "
            "maximize environment updates per second; break ties by lower reserved "
            "memory fraction and then smaller batch size."
        ),
        "default_batch_32_result": next(
            (row for row in rows if int(row.get("batch_size", -1)) == 32),
            None,
        ),
        "selected_result": selected,
        "successful_batch_sizes": [int(row["batch_size"]) for row in success],
        "oom_batch_sizes": [
            int(row["batch_size"]) for row in rows if row.get("status") == "oom"
        ],
    }


def run_parent(args: argparse.Namespace) -> int:
    validate_common_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = args.output_dir / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        result_file = worker_dir / f"batch_{batch_size:04d}.json"
        if result_file.exists():
            result_file.unlink()
        command = worker_command(args, batch_size, result_file)
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
        wall_seconds = time.perf_counter() - started
        if result_file.exists():
            row = json.loads(result_file.read_text(encoding="utf-8"))
        else:
            row = {
                "status": "error",
                "batch_size": int(batch_size),
                "pool_size": int(args.pool_size),
                "error": "worker exited without writing a result file",
            }
        row["worker_returncode"] = int(completed.returncode)
        row["worker_wall_seconds"] = float(wall_seconds)
        row["worker_stdout_tail"] = completed.stdout[-4000:]
        row["worker_stderr_tail"] = completed.stderr[-4000:]
        rows.append(row)
        write_csv(args.output_dir / "memory_probe.csv", rows)
        write_json(
            args.output_dir / "memory_probe.json",
            {
                "config": {
                    "catalogue": args.catalogue,
                    "device": args.device,
                    "dtype": args.dtype,
                    "model_spec": {
                        "activation": args.activation,
                        "depth": args.depth,
                        "width": args.width,
                        "use_bias": args.use_bias,
                    },
                    "pool_size": args.pool_size,
                    "batch_sizes": args.batch_sizes,
                    "k_buckets": args.k_buckets,
                    "warmup_updates": args.warmup_updates,
                    "measured_updates": args.measured_updates,
                    "gradient_clip_norm": args.gradient_clip_norm,
                    "step_regularization_weight": args.step_regularization_weight,
                    "seed": args.seed,
                },
                "results": rows,
            },
        )
        print(
            f"batch={batch_size} status={row.get('status')} "
            f"peak_reserved={row.get('peak_reserved_gib', 'n/a')} GiB "
            f"update_mean={row.get('update_seconds_mean', 'n/a')} s"
        )
        if row.get("status") != "success":
            error = str(row.get("error") or "").strip()
            if not error:
                error = str(row.get("worker_stderr_tail") or "").strip().splitlines()[-1:]
                error = error[0] if error else "unknown worker error"
            error_type = row.get("error_type")
            prefix = f"{error_type}: " if error_type else ""
            print(f"  error={prefix}{error}", file=sys.stderr)
        if row.get("status") == "oom" and args.stop_after_oom:
            break

    selected = recommendation(rows, args.memory_headroom_fraction)
    write_json(args.output_dir / "recommended_training_config.json", selected)
    if not args.keep_worker_files:
        for path in worker_dir.glob("*.json"):
            path.unlink()
        try:
            worker_dir.rmdir()
        except OSError:
            pass
    success = [row for row in rows if row.get("status") == "success"]
    if not success:
        print("No memory-probe configuration completed successfully.", file=sys.stderr)
        return 1
    print(
        "recommended batch size:",
        selected["recommended_batch_size"],
        "within reserved-memory headroom",
        selected["memory_headroom_fraction"],
    )
    return 0


def main() -> None:
    args = parse_args()
    validate_common_args(args)
    code = run_worker(args) if args.worker else run_parent(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
