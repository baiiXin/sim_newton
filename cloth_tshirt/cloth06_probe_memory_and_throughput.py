"""Probe peak CUDA memory for a complete MLP or GNN T-shirt training step."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from tshirt_config import DEFAULT_FIXED_DATA_DIR, DEFAULT_TRAIN_SEED, write_json


DEFAULT_OUTPUT = Path("cloth_tshirt_pipeline/profiling/memory_probe")
GIB = 1024 ** 3
SUPPORTED_ACTIVATIONS = ("identity", "relu", "gelu", "silu", "tanh")
SUPPORTED_MODEL_TYPES = ("mlp", "gnn")
DEFAULT_BATCH_SIZES = (4, 8, 16, 32, 64, 128)
K_BUCKET_COUNT = 4
GNN_MESSAGE_PASSING_STEPS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    paths = parser.add_argument_group("inputs and outputs")
    paths.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for per-worker, aggregate, and recommendation files",
    )
    paths.add_argument(
        "--fixed-data-dir",
        type=Path,
        default=DEFAULT_FIXED_DATA_DIR,
        help="fixed model and topology directory",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="cuda:0", help="CUDA device used by every worker")
    runtime.add_argument(
        "--dtype",
        choices=("float64", "float32"),
        default="float64",
        help="model, physics, pool, and optimizer dtype",
    )
    runtime.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAIN_SEED,
        help="model initialization and online-pool seed",
    )

    network = parser.add_argument_group("network")
    network.add_argument(
        "--model-type",
        choices=SUPPORTED_MODEL_TYPES,
        default="mlp",
        help="learned-optimizer architecture to instantiate in every worker",
    )
    network.add_argument(
        "--activation",
        choices=SUPPORTED_ACTIVATIONS,
        default="relu",
        help="activation after every hidden linear layer",
    )
    network.add_argument("--depth", type=int, default=1, help="number of hidden linear layers")
    network.add_argument("--width", type=int, default=2048, help="units in every hidden layer")
    network.add_argument(
        "--use-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable or disable bias in all linear layers",
    )

    sweep = parser.add_argument_group("training pool and batch sweep")
    sweep.add_argument(
        "--pool-size",
        type=int,
        default=512,
        help=f"resident online-training pool rows; must be divisible by {K_BUCKET_COUNT}",
    )
    sweep.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        metavar="N",
        default=DEFAULT_BATCH_SIZES,
        help=(
            f"candidate training batch sizes; each must be divisible by {K_BUCKET_COUNT} "
            "and no larger than pool size"
        ),
    )

    measurement = parser.add_argument_group("measurement")
    measurement.add_argument(
        "--warmup-updates",
        type=int,
        default=1,
        help="complete training updates before peak reset and timing",
    )
    measurement.add_argument(
        "--measured-updates",
        type=int,
        default=3,
        help="complete training updates included in timing",
    )
    measurement.add_argument(
        "--memory-headroom-fraction",
        type=float,
        default=0.85,
        help="maximum reserved/total CUDA memory fraction for a recommendation",
    )
    measurement.add_argument(
        "--dry-run",
        action="store_true",
        help="print worker commands without launching them",
    )
    parser.add_argument("--worker-batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _row_configuration(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    """Flat configuration fields included in every CSV/JSON result row."""
    return {
        "batch_size": int(batch_size),
        "pool_size": int(args.pool_size),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "seed": int(args.seed),
        "model_type": str(args.model_type),
        "activation": str(args.activation),
        "depth": int(args.depth),
        "width": int(args.width),
        "use_bias": bool(args.use_bias),
        "message_passing_steps": (
            GNN_MESSAGE_PASSING_STEPS if args.model_type == "gnn" else None
        ),
        "warmup_updates": int(args.warmup_updates),
        "measured_updates": int(args.measured_updates),
    }


def _public_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Serializable public CLI configuration without private worker arguments."""
    return {
        "output_dir": Path(args.output_dir).resolve(),
        "fixed_data_dir": Path(args.fixed_data_dir).resolve(),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "seed": int(args.seed),
        "model_type": str(args.model_type),
        "activation": str(args.activation),
        "depth": int(args.depth),
        "width": int(args.width),
        "use_bias": bool(args.use_bias),
        "message_passing_steps": (
            GNN_MESSAGE_PASSING_STEPS if args.model_type == "gnn" else None
        ),
        "pool_size": int(args.pool_size),
        "batch_sizes": [int(value) for value in args.batch_sizes],
        "warmup_updates": int(args.warmup_updates),
        "measured_updates": int(args.measured_updates),
        "memory_headroom_fraction": float(args.memory_headroom_fraction),
        "dry_run": bool(args.dry_run),
    }


def _validate_controller_args(args: argparse.Namespace) -> None:
    if args.depth <= 0 or args.width <= 0:
        raise ValueError("network depth and width must be positive")
    if args.model_type == "gnn":
        if args.activation != "relu" or args.depth != 2 or args.use_bias:
            raise ValueError(
                "the GNN baseline requires --activation relu --depth 2 --no-use-bias"
            )
    if args.pool_size <= 0 or args.pool_size % K_BUCKET_COUNT:
        raise ValueError(f"pool size must be positive and divisible by {K_BUCKET_COUNT} K buckets")
    invalid_batches = [
        int(value)
        for value in args.batch_sizes
        if value <= 0 or value % K_BUCKET_COUNT or value > args.pool_size
    ]
    if invalid_batches:
        raise ValueError(
            "every batch size must be positive, divisible by "
            f"{K_BUCKET_COUNT}, and no larger than pool size; invalid: {invalid_batches}"
        )
    if not 0.0 < args.memory_headroom_fraction <= 1.0:
        raise ValueError("memory headroom fraction must be in (0, 1]")
    if args.warmup_updates < 0 or args.measured_updates <= 0:
        raise ValueError("warmup must be nonnegative and measured updates positive")


def _worker(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from cloth02_batched_physics import load_physics
    from cloth03_training_pool import OnlineTrainingPool, training_step

    if args.worker_output is None or args.worker_batch_size is None:
        raise ValueError("worker output and batch size are required")
    batch_size = int(args.worker_batch_size)
    result: dict[str, Any] = {
        **_row_configuration(args, batch_size),
        "status": "failed",
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if batch_size % K_BUCKET_COUNT:
            raise ValueError(f"batch size must be divisible by {K_BUCKET_COUNT} K buckets")
        if args.pool_size % K_BUCKET_COUNT or batch_size > args.pool_size:
            raise ValueError(
                f"pool size must be divisible by {K_BUCKET_COUNT} and at least batch size"
            )
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.set_device(torch.device(args.device))
        dtype = torch.float64 if args.dtype == "float64" else torch.float32
        physics = load_physics(
            fixed_data_dir=args.fixed_data_dir, device=args.device, dtype=dtype
        )
        if args.model_type == "gnn":
            from cloth16_gnn_model import GNNModelSpec, LearnedOptimizerGNN

            model_spec = GNNModelSpec(
                activation=args.activation,
                depth=args.depth,
                width=args.width,
                use_bias=args.use_bias,
                message_passing_steps=GNN_MESSAGE_PASSING_STEPS,
            )
            model = LearnedOptimizerGNN(physics=physics, model_spec=model_spec)
        else:
            from cloth03_training_pool import LearnedOptimizerMLP, ModelSpec

            model_spec = ModelSpec(args.activation, args.depth, args.width, args.use_bias)
            model = LearnedOptimizerMLP(physics=physics, model_spec=model_spec)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        pool = OnlineTrainingPool(
            physics=physics,
            seed=args.seed,
            pool_size=args.pool_size,
            batch_size=batch_size,
        )
        device = torch.device(args.device)
        props = torch.cuda.get_device_properties(device)
        result.update(
            {
                "device_name": props.name,
                "total_memory_gib": props.total_memory / GIB,
                "parameter_count": model.parameter_count,
                "baseline_allocated_gib": torch.cuda.memory_allocated(device) / GIB,
                "baseline_reserved_gib": torch.cuda.memory_reserved(device) / GIB,
            }
        )
        for _ in range(args.warmup_updates):
            training_step(model=model, optimizer=optimizer, pool=pool)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        last_metrics: dict[str, Any] = {}
        for _ in range(args.measured_updates):
            last_metrics = training_step(model=model, optimizer=optimizer, pool=pool)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        element_bytes = torch.empty((), dtype=dtype).element_size()
        pool_scalars = 7 * args.pool_size * physics.num_vertices * 3
        result.update(
            {
                "status": "success",
                "measured_seconds": elapsed,
                "updates_per_second": args.measured_updates / max(elapsed, 1e-12),
                "motions_per_second": args.measured_updates * batch_size / max(elapsed, 1e-12),
                "peak_allocated_gib": peak_allocated / GIB,
                "peak_reserved_gib": peak_reserved / GIB,
                "peak_allocated_fraction": peak_allocated / props.total_memory,
                "peak_reserved_fraction": peak_reserved / props.total_memory,
                "model_optimizer_gradient_estimate_gib": 4 * model.parameter_count * element_bytes / GIB,
                "pool_tensor_estimate_gib": pool_scalars * element_bytes / GIB,
                "last_loss": last_metrics.get("loss"),
                "last_residual_ratio_p95": last_metrics.get("residual_ratio_p95"),
            }
        )
    except Exception as error:
        text = f"{type(error).__name__}: {error}"
        result["status"] = "oom" if "out of memory" in text.lower() else "failed"
        result["error"] = text
        result["traceback"] = traceback.format_exc()
        try:
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
        except Exception:
            pass
    write_json(args.worker_output, result)
    return 0 if result["status"] == "success" else 2


def _worker_command(args: argparse.Namespace, batch_size: int, output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-batch-size", str(batch_size),
        "--worker-output", str(output),
        "--output-dir", str(args.output_dir),
        "--fixed-data-dir", str(Path(args.fixed_data_dir).resolve()),
        "--device", args.device,
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--model-type", args.model_type,
        "--activation", args.activation,
        "--depth", str(args.depth),
        "--width", str(args.width),
        "--pool-size", str(args.pool_size),
        "--warmup-updates", str(args.warmup_updates),
        "--measured-updates", str(args.measured_updates),
        "--memory-headroom-fraction", str(args.memory_headroom_fraction),
        "--use-bias" if args.use_bias else "--no-use-bias",
    ]
    return command


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _recommend(rows: list[dict[str, Any]], headroom: float) -> dict[str, Any] | None:
    feasible = [
        row for row in rows
        if row.get("status") == "success"
        and float(row.get("peak_reserved_fraction", 1.0)) <= headroom
    ]
    if not feasible:
        return None
    selected = max(feasible, key=lambda row: (float(row["motions_per_second"]), int(row["batch_size"])))
    recommendation = {
        "recommended_batch_size": int(selected["batch_size"]),
        "pool_size": int(selected["pool_size"]),
        "expected_peak_reserved_gib": float(selected["peak_reserved_gib"]),
        "expected_peak_reserved_fraction": float(selected["peak_reserved_fraction"]),
        "measured_motions_per_second": float(selected["motions_per_second"]),
        "memory_headroom_fraction": float(headroom),
        "selection_rule": "highest measured motions/s among runs below reserved-memory headroom",
    }
    for key in (
        "device", "dtype", "seed", "model_type", "activation", "depth", "width",
        "use_bias", "message_passing_steps",
    ):
        if key in selected:
            recommendation[key] = selected[key]
    return recommendation


def main() -> None:
    args = parse_args()
    if args.worker_batch_size is not None:
        raise SystemExit(_worker(args))
    _validate_controller_args(args)
    configuration = _public_configuration(args)
    print(
        "probe configuration: "
        f"model_type={args.model_type} activation={args.activation} "
        f"depth={args.depth} width={args.width} "
        f"use_bias={args.use_bias} pool_size={args.pool_size} "
        f"batch_sizes={list(args.batch_sizes)}"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        worker_output = output / f"worker_batch_{batch_size:04d}.json"
        command = _worker_command(args, batch_size, worker_output)
        print(" ".join(command))
        if args.dry_run:
            rows.append({**_row_configuration(args, batch_size), "status": "planned"})
            continue
        # A hard-crashed worker must not be mistaken for a successful result
        # left by an earlier sweep in the same output directory.
        worker_output.unlink(missing_ok=True)
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        if worker_output.exists():
            rows.append(json.loads(worker_output.read_text(encoding="utf-8")))
        else:
            rows.append(
                {
                    **_row_configuration(args, batch_size),
                    "status": "worker_crashed",
                    "return_code": completed.returncode,
                }
            )
    _write_csv(output / "memory_probe.csv", rows)
    write_json(output / "memory_probe.json", {"configuration": configuration, "results": rows})
    recommendation = _recommend(rows, args.memory_headroom_fraction)
    write_json(
        output / "recommended_training_config.json",
        {
            "available": recommendation is not None,
            "configuration": configuration,
            "recommendation": recommendation,
            "note": "Peak values include ask, forward, energy loss, backward, optimizer step, and pool tell/reset checks.",
        },
    )
    if recommendation:
        print(
            f"recommended batch={recommendation['recommended_batch_size']} "
            f"peak_reserved={recommendation['expected_peak_reserved_gib']:.2f} GiB"
        )
    else:
        print("no successful configuration stayed below the requested memory headroom")


if __name__ == "__main__":
    main()
