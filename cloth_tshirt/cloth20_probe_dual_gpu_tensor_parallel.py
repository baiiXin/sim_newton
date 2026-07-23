"""Probe a complete two-GPU tensor-parallel full-state MLP training step.

The single hidden layer is column-sharded and the output layer is row-sharded.
Both ranks execute the same physics/pool/loss work while PyTorch DTensor owns
only one half of each large Linear weight on each GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

from tshirt_config import (
    DEFAULT_FIXED_DATA_DIR,
    DEFAULT_TRAIN_SEED,
    load_model_spec,
    write_json,
)


DEFAULT_WIDTH = 39_936
DEFAULT_OUTPUT = Path("cloth_tshirt_pipeline/profiling/tp_width_39936_pool512_batch32")
TENSOR_PARALLEL_SIZE = 2
K_BUCKET_COUNT = 4
GIB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument(
        "--devices",
        type=int,
        nargs=TENSOR_PARALLEL_SIZE,
        default=(0, 1),
        metavar=("GPU0", "GPU1"),
        help="physical CUDA device indices exposed to the two torchrun ranks",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument(
        "--activation",
        choices=("identity", "relu", "gelu", "silu", "tanh"),
        default="relu",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--use-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="bias is currently rejected so every trainable tensor is sharded",
    )
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--measured-updates", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--memory-headroom-fraction", type=float, default=0.95)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the torchrun command without launching CUDA workers",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def network_dimensions(
    *, num_vertices: int, width: int, tensor_parallel_size: int = TENSOR_PARALLEL_SIZE
) -> dict[str, int | float]:
    full_state_dim = 3 * int(num_vertices)
    input_dim = 3 * full_state_dim
    global_parameters = input_dim * int(width) + int(width) * full_state_dim
    return {
        "num_vertices": int(num_vertices),
        "full_state_dim": full_state_dim,
        "input_dim": input_dim,
        "width": int(width),
        "width_to_input_ratio": float(width / input_dim),
        "global_parameter_count": global_parameters,
        "local_parameter_count": global_parameters // tensor_parallel_size,
    }


def _validate_args(args: argparse.Namespace, *, num_vertices: int) -> None:
    if (
        len(args.devices) != TENSOR_PARALLEL_SIZE
        or len(set(args.devices)) != len(args.devices)
    ):
        raise ValueError("--devices must name two distinct CUDA device indices")
    if args.width <= 0 or args.width % TENSOR_PARALLEL_SIZE:
        raise ValueError(f"--width must be positive and divisible by {TENSOR_PARALLEL_SIZE}")
    if args.use_bias:
        raise ValueError("this probe requires --no-use-bias so every parameter is sharded")
    if args.pool_size <= 0 or args.pool_size % K_BUCKET_COUNT:
        raise ValueError(f"--pool-size must be positive and divisible by {K_BUCKET_COUNT}")
    if (
        args.batch_size <= 0
        or args.batch_size % K_BUCKET_COUNT
        or args.batch_size > args.pool_size
    ):
        raise ValueError(
            f"--batch-size must be positive, divisible by {K_BUCKET_COUNT}, "
            "and no larger than --pool-size"
        )
    if args.warmup_updates < 0 or args.measured_updates <= 0:
        raise ValueError("warmup must be nonnegative and measured updates must be positive")
    if args.learning_rate <= 0.0 or args.gradient_clip_norm < 0.0:
        raise ValueError("learning rate must be positive and gradient clip norm nonnegative")
    if not 0.0 < args.memory_headroom_fraction <= 1.0:
        raise ValueError("memory headroom fraction must be in (0, 1]")
    dimensions = network_dimensions(num_vertices=num_vertices, width=args.width)
    if int(dimensions["global_parameter_count"]) % TENSOR_PARALLEL_SIZE:
        raise ValueError("global parameter count must split evenly across both ranks")


def _configuration(args: argparse.Namespace, *, num_vertices: int) -> dict[str, Any]:
    dimensions = network_dimensions(num_vertices=num_vertices, width=args.width)
    element_bytes = 4 if args.dtype == "float32" else 8
    local_parameters = int(dimensions["local_parameter_count"])
    return {
        "output_dir": args.output_dir.resolve(),
        "fixed_data_dir": args.fixed_data_dir.resolve(),
        "physical_devices": [int(value) for value in args.devices],
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "parallelism": "PyTorch DTensor ColwiseParallel hidden + RowwiseParallel output",
        "dtype": args.dtype,
        "seed": int(args.seed),
        "model_type": "tensor_parallel_full_state_mlp",
        "activation": args.activation,
        "depth": 1,
        "width": int(args.width),
        "use_bias": False,
        "initialization": "sharded_normal_with_global_fan_in_and_zero_output",
        "optimizer": "Adam(foreach=False)",
        "learning_rate": float(args.learning_rate),
        "gradient_clip_norm": float(args.gradient_clip_norm),
        "pool_size": int(args.pool_size),
        "batch_size": int(args.batch_size),
        "warmup_updates": int(args.warmup_updates),
        "measured_updates": int(args.measured_updates),
        "memory_headroom_fraction": float(args.memory_headroom_fraction),
        "local_model_gradient_adam_estimate_gib": (
            4 * local_parameters * element_bytes / GIB
        ),
        **dimensions,
    }


def _worker_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={TENSOR_PARALLEL_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        "--output-dir", str(args.output_dir.resolve()),
        "--fixed-data-dir", str(args.fixed_data_dir.resolve()),
        "--devices", *(str(value) for value in args.devices),
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--activation", args.activation,
        "--width", str(args.width),
        "--no-use-bias",
        "--pool-size", str(args.pool_size),
        "--batch-size", str(args.batch_size),
        "--warmup-updates", str(args.warmup_updates),
        "--measured-updates", str(args.measured_updates),
        "--learning-rate", str(args.learning_rate),
        "--gradient-clip-norm", str(args.gradient_clip_norm),
        "--memory-headroom-fraction", str(args.memory_headroom_fraction),
    ]


def _build_tensor_parallel_model(*, physics, args: argparse.Namespace, device_mesh, rank: int):
    from cloth_tensor_parallel import build_tensor_parallel_model

    return build_tensor_parallel_model(
        physics=physics,
        activation=args.activation,
        width=args.width,
        device_mesh=device_mesh,
        rank=rank,
        seed=args.seed,
    )


def _training_step(*, model, optimizer, pool, gradient_clip_norm: float) -> dict[str, float | int]:
    from cloth_tensor_parallel import tensor_parallel_training_step

    return tensor_parallel_training_step(
        model=model,
        optimizer=optimizer,
        pool=pool,
        gradient_clip_norm=gradient_clip_norm,
    )


def _worker(args: argparse.Namespace, *, num_vertices: int) -> int:
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from cloth02_batched_physics import load_physics
    from cloth03_training_pool import OnlineTrainingPool
    from cloth_tensor_parallel import SynchronizedOnlineTrainingPool, local_parameter_count

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    worker_output = output / f"worker_rank_{rank:02d}.json"
    configuration = _configuration(args, num_vertices=num_vertices)
    result: dict[str, Any] = {
        "status": "starting",
        "rank": rank,
        "local_rank": local_rank,
        **configuration,
    }
    write_json(worker_output, result)
    try:
        if world_size != TENSOR_PARALLEL_SIZE:
            raise RuntimeError(
                f"expected world size {TENSOR_PARALLEL_SIZE}, received {world_size}"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
            raise RuntimeError(
                f"expected {world_size} visible CUDA devices, found {torch.cuda.device_count()}"
            )
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        device_mesh = init_device_mesh("cuda", (world_size,))
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        dtype = torch.float32 if args.dtype == "float32" else torch.float64
        physics = load_physics(
            fixed_data_dir=args.fixed_data_dir,
            device=device,
            dtype=dtype,
        )
        model = _build_tensor_parallel_model(
            physics=physics,
            args=args,
            device_mesh=device_mesh,
            rank=rank,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            foreach=False,
        )
        pool = SynchronizedOnlineTrainingPool(
            OnlineTrainingPool(
                physics=physics,
                seed=args.seed,
                pool_size=args.pool_size,
                batch_size=args.batch_size,
            )
        )
        properties = torch.cuda.get_device_properties(device)
        actual_local_parameter_count = local_parameter_count(model)
        result.update(
            {
                "status": "initialized",
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device_name": properties.name,
                "total_memory_gib": properties.total_memory / GIB,
                "actual_local_parameter_count": actual_local_parameter_count,
                "baseline_allocated_gib": torch.cuda.memory_allocated(device) / GIB,
                "baseline_reserved_gib": torch.cuda.memory_reserved(device) / GIB,
            }
        )
        write_json(worker_output, result)

        last_metrics: dict[str, float | int] = {}
        for _ in range(args.warmup_updates):
            last_metrics = _training_step(
                model=model,
                optimizer=optimizer,
                pool=pool,
                gradient_clip_norm=args.gradient_clip_norm,
            )
        torch.cuda.synchronize(device)
        dist.barrier()
        result.update(
            {
                "post_warmup_allocated_gib": torch.cuda.memory_allocated(device) / GIB,
                "post_warmup_reserved_gib": torch.cuda.memory_reserved(device) / GIB,
            }
        )
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        started = time.perf_counter()
        for _ in range(args.measured_updates):
            last_metrics = _training_step(
                model=model,
                optimizer=optimizer,
                pool=pool,
                gradient_clip_norm=args.gradient_clip_norm,
            )
        torch.cuda.synchronize(device)
        dist.barrier()
        elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result.update(
            {
                "status": "success",
                "measured_seconds": elapsed,
                "peak_allocated_gib": peak_allocated / GIB,
                "peak_reserved_gib": peak_reserved / GIB,
                "peak_allocated_fraction": peak_allocated / properties.total_memory,
                "peak_reserved_fraction": peak_reserved / properties.total_memory,
                "last_metrics": last_metrics,
            }
        )
        write_json(worker_output, result)

        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, result)
        if rank == 0:
            rank_results = [item for item in gathered if item is not None]
            slowest_seconds = max(float(item["measured_seconds"]) for item in rank_results)
            max_reserved_fraction = max(
                float(item["peak_reserved_fraction"]) for item in rank_results
            )
            within_headroom = max_reserved_fraction <= args.memory_headroom_fraction
            summary = {
                "status": "success",
                "configuration": configuration,
                "within_memory_headroom": within_headroom,
                "max_peak_reserved_fraction": max_reserved_fraction,
                "max_peak_reserved_gib": max(
                    float(item["peak_reserved_gib"]) for item in rank_results
                ),
                "measured_seconds": slowest_seconds,
                "updates_per_second": args.measured_updates / max(slowest_seconds, 1e-12),
                "motions_per_second": (
                    args.measured_updates * args.batch_size / max(slowest_seconds, 1e-12)
                ),
                "rank_results": rank_results,
                "note": (
                    "Peak values include synchronized pool ask, tensor-parallel forward, "
                    "physics energy, backward, global gradient clipping, Adam, and pool tell."
                ),
            }
            write_json(output / "memory_probe.json", summary)
            write_json(
                output / "recommended_training_config.json",
                {
                    "available": within_headroom,
                    "configuration": configuration,
                    "recommendation": (
                        {
                            "width": args.width,
                            "batch_size": args.batch_size,
                            "pool_size": args.pool_size,
                            "physical_devices": list(args.devices),
                            "expected_max_peak_reserved_gib": summary[
                                "max_peak_reserved_gib"
                            ],
                        }
                        if within_headroom
                        else None
                    ),
                },
            )
            verdict = "fits" if within_headroom else "exceeds requested headroom"
            print(
                f"tensor-parallel width={args.width} {verdict}: "
                f"max peak_reserved={summary['max_peak_reserved_gib']:.2f} GiB, "
                f"motions/s={summary['motions_per_second']:.2f}",
                flush=True,
            )
        dist.destroy_process_group()
        return 0
    except Exception as error:
        text = f"{type(error).__name__}: {error}"
        result.update(
            {
                "status": "oom" if "out of memory" in text.lower() else "failed",
                "error": text,
                "traceback": traceback.format_exc(),
            }
        )
        try:
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
        except Exception:
            pass
        write_json(worker_output, result)
        # Let torchrun terminate the peer if one rank fails inside a collective.
        raise


def _write_failed_aggregate(
    *, output: Path, configuration: dict[str, Any], return_code: int
) -> None:
    rank_results: list[dict[str, Any]] = []
    for rank in range(TENSOR_PARALLEL_SIZE):
        path = output / f"worker_rank_{rank:02d}.json"
        if path.exists():
            rank_results.append(json.loads(path.read_text(encoding="utf-8")))
    write_json(
        output / "memory_probe.json",
        {
            "status": "worker_failed",
            "return_code": int(return_code),
            "configuration": configuration,
            "within_memory_headroom": False,
            "rank_results": rank_results,
        },
    )
    write_json(
        output / "recommended_training_config.json",
        {
            "available": False,
            "configuration": configuration,
            "recommendation": None,
        },
    )


def main() -> None:
    args = parse_args()
    spec = load_model_spec(Path(args.fixed_data_dir) / "model_spec.json")
    _validate_args(args, num_vertices=spec.num_vertices)
    if args.worker or "RANK" in os.environ:
        raise SystemExit(_worker(args, num_vertices=spec.num_vertices))

    configuration = _configuration(args, num_vertices=spec.num_vertices)
    command = _worker_command(args)
    print(
        "probe configuration: "
        f"input_dim={configuration['input_dim']} width={args.width} "
        f"ratio={configuration['width_to_input_ratio']:.3f} "
        f"global_parameters={configuration['global_parameter_count']:,} "
        f"local_parameters={configuration['local_parameter_count']:,} "
        f"estimated_local_model+gradient+Adam="
        f"{configuration['local_model_gradient_adam_estimate_gib']:.2f} GiB",
        flush=True,
    )
    print(" ".join(command), flush=True)
    if args.dry_run:
        return

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for rank in range(TENSOR_PARALLEL_SIZE):
        (output / f"worker_rank_{rank:02d}.json").unlink(missing_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in args.devices)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(command, env=environment)
    if completed.returncode:
        _write_failed_aggregate(
            output=output,
            configuration=configuration,
            return_code=completed.returncode,
        )
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
