"""Train the depth-one full-state T-shirt MLP on two tensor-parallel GPUs.

The controller launches two torchrun workers. Both workers own identical
physics and online-pool state, while PyTorch DTensor shards the hidden and
output weights, gradients, and Adam state across the two CUDA devices.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Mapping, Sequence

from tshirt_config import DEFAULT_FIXED_DATA_DIR, DEFAULT_TRAIN_SEED, load_model_spec


TENSOR_PARALLEL_SIZE = 2
RECOMMENDED_WIDTH = 39_936
DEFAULT_OUTPUT_ROOT = Path("cloth_tshirt_pipeline/tensor_parallel")
CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_DIRECTORY_PATTERN = re.compile(r"^step_(\d{9})_gen_(\d{6})$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument(
        "--devices",
        type=int,
        nargs=TENSOR_PARALLEL_SIZE,
        default=(0, 1),
        metavar=("GPU0", "GPU1"),
        help="physical CUDA devices exposed to the two torchrun ranks",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument(
        "--activation",
        choices=("identity", "relu", "gelu", "silu", "tanh"),
        default="relu",
    )
    parser.add_argument("--width", type=int, default=RECOMMENDED_WIDTH)
    parser.add_argument(
        "--use-bias",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="must remain disabled so every trainable tensor is sharded",
    )
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k-buckets", type=int, nargs="+", default=(1, 3, 10, 30))
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--step-regularization-weight", type=float, default=0.0)
    parser.add_argument("--max-updates", type=int, default=3_000_000)
    parser.add_argument("--max-wall-hours", type=float, default=10.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=5_000)
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=2,
        help="number of most recent complete full checkpoints retained",
    )
    parser.add_argument("--fast-validation-interval", type=int, default=10_000)
    parser.add_argument("--checkpoint-validation-interval", type=int, default=50_000)
    parser.add_argument("--fast-rollout-frames", type=int, default=32)
    parser.add_argument("--checkpoint-rollout-frames", type=int, default=100)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--skip-initial-validation", action="store_true")
    parser.add_argument("--skip-final-validation", action="store_true")
    parser.add_argument("--no-validation-plots", action="store_true")
    parser.add_argument(
        "--save-best-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the selected validation model as a model-only DCP checkpoint",
    )
    parser.add_argument("--resume", action="store_true", help="resume from run-dir/latest.json")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="resume from an explicit complete full-checkpoint directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def experiment_name(args: argparse.Namespace) -> str:
    return (
        f"activation_{args.activation}_depth_01_width_{args.width:05d}_"
        "no_bias"
    )


def run_directory(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return Path(args.run_dir).resolve()
    return (Path(args.output_root) / experiment_name(args) / f"seed_{args.seed}").resolve()


def validate_args(args: argparse.Namespace, *, num_vertices: int) -> None:
    if len(args.devices) != 2 or len(set(args.devices)) != 2:
        raise ValueError("--devices must name two distinct CUDA device indices")
    if args.width <= 0 or args.width % TENSOR_PARALLEL_SIZE:
        raise ValueError("--width must be positive and divisible by 2")
    if args.use_bias:
        raise ValueError("formal tensor-parallel training requires --no-use-bias")
    if len(args.k_buckets) == 0 or len(set(args.k_buckets)) != len(args.k_buckets):
        raise ValueError("--k-buckets must be a nonempty list of distinct values")
    if any(value <= 0 for value in args.k_buckets):
        raise ValueError("all --k-buckets values must be positive")
    bucket_count = len(args.k_buckets)
    if args.pool_size <= 0 or args.pool_size % bucket_count:
        raise ValueError("--pool-size must be positive and divisible by the bucket count")
    if (
        args.batch_size <= 0
        or args.batch_size > args.pool_size
        or args.batch_size % bucket_count
    ):
        raise ValueError(
            "--batch-size must be positive, no larger than the pool, and divisible "
            "by the bucket count"
        )
    if args.learning_rate <= 0.0 or args.gradient_clip_norm < 0.0:
        raise ValueError("learning rate must be positive and gradient clip norm nonnegative")
    if args.step_regularization_weight < 0.0:
        raise ValueError("step regularization weight must be nonnegative")
    if args.max_updates <= 0 or args.max_wall_hours <= 0.0:
        raise ValueError("max updates and max wall hours must be positive")
    if args.log_interval <= 0 or args.checkpoint_interval <= 0:
        raise ValueError("log and checkpoint intervals must be positive")
    if args.keep_checkpoints < 1:
        raise ValueError("--keep-checkpoints must be at least 1")
    if args.fast_validation_interval < 0 or args.checkpoint_validation_interval < 0:
        raise ValueError("validation intervals must be nonnegative (zero disables them)")
    if args.fast_rollout_frames <= 0 or args.checkpoint_rollout_frames <= 0:
        raise ValueError("validation rollout frame counts must be positive")
    if args.validation_batch_size <= 0:
        raise ValueError("validation batch size must be positive")
    if args.resume and args.resume_checkpoint is not None:
        raise ValueError("use either --resume or --resume-checkpoint, not both")
    full_state_dim = 3 * int(num_vertices)
    global_parameters = (3 * full_state_dim) * args.width + args.width * full_state_dim
    if global_parameters % TENSOR_PARALLEL_SIZE:
        raise ValueError("the global parameter count must split evenly across two ranks")


def _forwarded_arguments(args: argparse.Namespace) -> list[str]:
    output: list[str] = [
        "--output-root", str(Path(args.output_root).resolve()),
        "--fixed-data-dir", str(Path(args.fixed_data_dir).resolve()),
        "--devices", *(str(value) for value in args.devices),
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--activation", args.activation,
        "--width", str(args.width),
        "--no-use-bias",
        "--pool-size", str(args.pool_size),
        "--batch-size", str(args.batch_size),
        "--k-buckets", *(str(value) for value in args.k_buckets),
        "--learning-rate", str(args.learning_rate),
        "--gradient-clip-norm", str(args.gradient_clip_norm),
        "--step-regularization-weight", str(args.step_regularization_weight),
        "--max-updates", str(args.max_updates),
        "--max-wall-hours", str(args.max_wall_hours),
        "--log-interval", str(args.log_interval),
        "--checkpoint-interval", str(args.checkpoint_interval),
        "--keep-checkpoints", str(args.keep_checkpoints),
        "--fast-validation-interval", str(args.fast_validation_interval),
        "--checkpoint-validation-interval", str(args.checkpoint_validation_interval),
        "--fast-rollout-frames", str(args.fast_rollout_frames),
        "--checkpoint-rollout-frames", str(args.checkpoint_rollout_frames),
        "--validation-batch-size", str(args.validation_batch_size),
    ]
    if args.run_dir is not None:
        output.extend(("--run-dir", str(Path(args.run_dir).resolve())))
    if args.skip_initial_validation:
        output.append("--skip-initial-validation")
    if args.skip_final_validation:
        output.append("--skip-final-validation")
    if args.no_validation_plots:
        output.append("--no-validation-plots")
    output.append("--save-best-model" if args.save_best_model else "--no-save-best-model")
    if args.resume:
        output.append("--resume")
    if args.resume_checkpoint is not None:
        output.extend(("--resume-checkpoint", str(args.resume_checkpoint.resolve())))
    return output


def worker_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={TENSOR_PARALLEL_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        *_forwarded_arguments(args),
    ]


def _json_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: ([str(item) for item in value] if isinstance(value, tuple) else str(value))
        if isinstance(value, Path) or (
            isinstance(value, tuple) and any(isinstance(item, Path) for item in value)
        )
        else list(value) if isinstance(value, tuple)
        else value
        for key, value in vars(args).items()
        if key not in {"worker", "dry_run"}
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _truncate_training_log_for_resume(
    path: Path, *, update: int, checkpoint_generation: int
) -> None:
    """Drop uncheckpointed future rows while retaining an audit copy."""
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        return
    retained = [row for row in rows if int(row["update"]) <= int(update)]
    if len(retained) == len(rows):
        return
    audit = (
        path.parent
        / "resume_audit"
        / f"training_log_before_gen_{checkpoint_generation:06d}.csv"
    )
    audit.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, audit)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _mean_metrics(
    rows: Sequence[dict[str, Any]], *, update: int, elapsed_seconds: float
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "update": int(update),
        "elapsed_seconds": float(elapsed_seconds),
        "updates_per_second": float(
            len(rows) / max(sum(float(row["step_seconds"]) for row in rows), 1e-12)
        ),
    }
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key != "step_seconds"
    ]
    count_keys = {
        key
        for key in numeric_keys
        if key.startswith("resets_") or key == "completed_physical_frames"
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        output[key] = int(sum(values)) if key in count_keys else sum(values) / len(values)
    return output


def _checkpoint_name(update: int, generation: int) -> str:
    return f"step_{int(update):09d}_gen_{int(generation):06d}"


def _safe_remove_checkpoint_directory(path: Path, *, parent: Path) -> None:
    resolved_parent = parent.resolve()
    resolved = path.resolve()
    if resolved.parent != resolved_parent or not CHECKPOINT_DIRECTORY_PATTERN.fullmatch(path.name):
        raise RuntimeError(f"refusing to remove unexpected checkpoint path: {path}")
    shutil.rmtree(resolved)


def resolve_resume_checkpoint(run_dir: Path, explicit: Path | None = None) -> Path:
    """Resolve and validate a complete immutable full-checkpoint directory."""
    run_dir = Path(run_dir).resolve()
    if explicit is None:
        pointer_path = run_dir / "latest.json"
        if not pointer_path.exists():
            raise FileNotFoundError(f"resume requested but pointer is missing: {pointer_path}")
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        checkpoint = run_dir / str(pointer["checkpoint"])
    else:
        checkpoint = Path(explicit)
        if not checkpoint.is_absolute():
            checkpoint = run_dir / checkpoint
    checkpoint = checkpoint.resolve()
    if explicit is None and checkpoint.parent != (run_dir / "checkpoints").resolve():
        raise ValueError(f"latest.json points outside this run's checkpoints: {checkpoint}")
    if not checkpoint.is_dir() or not (checkpoint / "COMPLETE").is_file():
        raise FileNotFoundError(f"checkpoint is absent or incomplete: {checkpoint}")
    return checkpoint


def next_checkpoint_generation(run_dir: Path) -> int:
    """Return a generation newer than every immutable checkpoint in this run."""
    root = Path(run_dir) / "checkpoints"
    generations = []
    if root.exists():
        for path in root.iterdir():
            match = CHECKPOINT_DIRECTORY_PATTERN.fullmatch(path.name)
            if match and path.is_dir():
                generations.append(int(match.group(2)))
    return max(generations, default=-1) + 1


def _runtime_rng_state(*, device) -> dict[str, Any]:
    import numpy as np
    import random
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.device(device).type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_runtime_rng_state(value: Mapping[str, Any], *, device) -> None:
    import numpy as np
    import random
    import torch

    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if "torch_cuda" in value:
        torch.cuda.set_rng_state(value["torch_cuda"], device=device)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    import torch

    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _collective_local_action(action: Callable[[], None], *, context: str) -> None:
    """Run an action on every rank and propagate any local I/O error to all."""
    import torch.distributed as dist

    local_error: str | None = None
    try:
        action()
    except Exception as error:
        local_error = f"rank {dist.get_rank()} {type(error).__name__}: {error}"
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    failures = [error for error in errors if error is not None]
    if failures:
        raise RuntimeError(f"{context} failed: {'; '.join(failures)}")


def _collective_rank0_action(
    rank: int, action: Callable[[], None], *, context: str
) -> None:
    """Run rank-0 filesystem work while keeping every rank on the same path."""
    import torch.distributed as dist

    error_message: list[str | None] = [None]
    if rank == 0:
        try:
            action()
        except Exception as error:
            error_message[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(error_message, src=0)
    if error_message[0] is not None:
        raise RuntimeError(f"{context} failed on rank 0: {error_message[0]}")


def save_full_checkpoint(
    *,
    run_dir: Path,
    model,
    optimizer,
    pool,
    rank: int,
    device,
    update: int,
    generation: int,
    elapsed_seconds: float,
    best_rank: tuple[float, ...] | None,
    best_update: int | None,
    manifest_invariants: Mapping[str, Any],
) -> Path:
    """Collectively write one immutable sharded training checkpoint."""
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    root = run_dir / "checkpoints"
    name = _checkpoint_name(update, generation)
    final = root / name
    temporary = root / f".{name}.tmp"

    def prepare() -> None:
        root.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise FileExistsError(f"immutable checkpoint already exists: {final}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)

    _collective_rank0_action(rank, prepare, context="checkpoint preparation")

    model_state, optimizer_state = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optimizer_state},
        checkpoint_id=temporary / "distributed",
    )
    runtime = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "rank": int(rank),
        "update_count": int(update),
        "checkpoint_generation": int(generation),
        "elapsed_seconds": float(elapsed_seconds),
        "best_validation_rank": None if best_rank is None else list(best_rank),
        "best_validation_update": best_update,
        "pool_state_dict": pool.state_dict(),
        "rng_state": _runtime_rng_state(device=device),
    }
    _collective_local_action(
        lambda: _atomic_torch_save(
            runtime, temporary / f"runtime_rank_{rank:02d}.pt"
        ),
        context="rank-local runtime checkpoint",
    )

    def finalize() -> None:
        checkpoint_manifest = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "kind": "full_training_state",
            "update_count": int(update),
            "checkpoint_generation": int(generation),
            "elapsed_seconds": float(elapsed_seconds),
            "best_validation_rank": None if best_rank is None else list(best_rank),
            "best_validation_update": best_update,
            "world_size": dist.get_world_size(),
            "torch_version": str(torch.__version__),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            **dict(manifest_invariants),
        }
        _atomic_write_json(temporary / "manifest.json", checkpoint_manifest)
        (temporary / "COMPLETE").touch()
        os.replace(temporary, final)
        _atomic_write_json(
            run_dir / "latest.json",
            {
                "checkpoint": str(final.relative_to(run_dir)),
                "update_count": int(update),
                "checkpoint_generation": int(generation),
            },
        )

    _collective_rank0_action(rank, finalize, context="checkpoint publication")
    return final


def load_full_checkpoint(
    *,
    checkpoint: Path,
    model,
    optimizer,
    pool,
    rank: int,
    device,
    expected_invariants: Mapping[str, Any],
) -> dict[str, Any]:
    """Collectively restore DTensor model/Adam shards plus rank-local runtime."""
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if manifest.get("kind") != "full_training_state":
        raise ValueError("resume requires a full training-state checkpoint")
    if int(manifest.get("world_size", -1)) != dist.get_world_size():
        raise ValueError("checkpoint tensor-parallel world size mismatch")
    if manifest.get("torch_version") != str(torch.__version__):
        raise ValueError(
            "PyTorch version mismatch: distributed checkpoints are not guaranteed "
            f"compatible ({manifest.get('torch_version')} != {torch.__version__})"
        )
    for key, expected in expected_invariants.items():
        if manifest.get(key) != expected:
            raise ValueError(f"resume checkpoint {key} mismatch")

    model_state, optimizer_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=checkpoint / "distributed")
    incompatible = set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint state mismatch: {incompatible}")
    runtime = torch.load(
        checkpoint / f"runtime_rank_{rank:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if int(runtime["rank"]) != rank:
        raise ValueError("rank-local runtime checkpoint mismatch")
    if int(runtime["update_count"]) != int(manifest["update_count"]):
        raise ValueError("checkpoint manifest/runtime update mismatch")
    pool.load_state_dict(runtime["pool_state_dict"])
    _restore_runtime_rng_state(runtime["rng_state"], device=device)

    updates: list[int | None] = [None] * dist.get_world_size()
    dist.all_gather_object(updates, int(runtime["update_count"]))
    if len(set(updates)) != 1:
        raise RuntimeError(f"rank-local checkpoint updates differ: {updates}")
    return {**manifest, **runtime}


def save_best_model(
    *,
    run_dir: Path,
    model,
    rank: int,
    update: int,
    selection_rank: tuple[float, ...],
    manifest_invariants: Mapping[str, Any],
) -> Path:
    """Collectively write a model-only sharded checkpoint selected by validation."""
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    root = run_dir / "best_models"
    name = f"step_{update:09d}"
    final = root / name
    temporary = root / f".{name}.tmp"

    def prepare() -> None:
        root.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary)
        if final.exists():
            raise FileExistsError(f"immutable best model already exists: {final}")
        temporary.mkdir(parents=True)

    _collective_rank0_action(rank, prepare, context="best-model preparation")
    dcp.save(
        {"model": get_model_state_dict(model)},
        checkpoint_id=temporary / "distributed",
    )

    def finalize() -> None:
        _atomic_write_json(
            temporary / "manifest.json",
            {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "kind": "model_only",
                "update_count": int(update),
                "selection_rank": list(selection_rank),
                "world_size": dist.get_world_size(),
                "torch_version": str(torch.__version__),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                **dict(manifest_invariants),
            },
        )
        (temporary / "COMPLETE").touch()
        os.replace(temporary, final)
        prior: Path | None = None
        pointer_path = run_dir / "best.json"
        if pointer_path.exists():
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            prior = (run_dir / str(pointer["checkpoint"])).resolve()
        _atomic_write_json(
            pointer_path,
            {
                "checkpoint": str(final.relative_to(run_dir)),
                "update_count": int(update),
                "selection_rank": list(selection_rank),
            },
        )
        if prior is not None and prior != final.resolve() and prior.parent == root.resolve():
            if prior.name.startswith("step_") and prior.is_dir():
                shutil.rmtree(prior)

    _collective_rank0_action(rank, finalize, context="best-model publication")
    return final


def prune_full_checkpoints(run_dir: Path, *, keep: int) -> None:
    root = run_dir / "checkpoints"
    if not root.exists():
        return
    complete: list[tuple[int, int, Path]] = []
    for path in root.iterdir():
        match = CHECKPOINT_DIRECTORY_PATTERN.fullmatch(path.name)
        if match and path.is_dir() and (path / "COMPLETE").is_file():
            complete.append((int(match.group(2)), int(match.group(1)), path))
    complete.sort()
    for _, _, path in complete[:-keep]:
        _safe_remove_checkpoint_directory(path, parent=root)


def _worker(args: argparse.Namespace, *, num_vertices: int) -> int:
    import random
    import signal

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from cloth02_batched_physics import load_frozen_motion_batch, load_physics
    from cloth03_training_pool import ModelSpec, OnlineTrainingPool
    from cloth04_reference_free_validation import (
        FailureThresholds,
        run_reference_free_validation,
        save_validation_result,
    )
    from cloth11_plot_training_progress import plot_training_progress
    from cloth_tensor_parallel import (
        SynchronizedOnlineTrainingPool,
        assert_replicated_scalars,
        build_tensor_parallel_model,
        local_parameter_count,
        network_dimensions,
        tensor_parallel_training_step,
    )
    from validation_protocol import CHECKPOINT_VALIDATION, FAST_MONITOR, checkpoint_rank
    from dataclasses import replace

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    output = run_directory(args)
    output.mkdir(parents=True, exist_ok=True)
    failure_path = output / f"failure_rank_{rank:02d}.json"
    failure_path.unlink(missing_ok=True)
    device = torch.device("cuda", local_rank)
    initialized = False
    try:
        if world_size != TENSOR_PARALLEL_SIZE:
            raise RuntimeError(f"expected world size 2, received {world_size}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
            raise RuntimeError(
                f"expected {world_size} visible CUDA devices, found {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        initialized = True
        device_mesh = init_device_mesh("cuda", (world_size,))
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        dtype = torch.float32 if args.dtype == "float32" else torch.float64
        physics = load_physics(
            fixed_data_dir=args.fixed_data_dir,
            device=device,
            dtype=dtype,
        )
        model_spec = ModelSpec(args.activation, 1, args.width, False)
        model = build_tensor_parallel_model(
            physics=physics,
            activation=args.activation,
            width=args.width,
            device_mesh=device_mesh,
            rank=rank,
            seed=args.seed,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, foreach=False
        )
        pool = SynchronizedOnlineTrainingPool(
            OnlineTrainingPool(
                physics=physics,
                seed=args.seed,
                pool_size=args.pool_size,
                batch_size=args.batch_size,
                k_buckets=args.k_buckets,
            )
        )
        validation = load_frozen_motion_batch(
            Path(args.fixed_data_dir) / "validation_32.npz",
            device=device,
            dtype=dtype,
        )
        fast_protocol = replace(FAST_MONITOR, rollout_frames=args.fast_rollout_frames)
        full_protocol = replace(
            CHECKPOINT_VALIDATION, rollout_frames=args.checkpoint_rollout_frames
        )
        dimensions = network_dimensions(num_vertices=num_vertices, width=args.width)
        # Canonicalize tuples and NumPy scalar-like values exactly as JSON will;
        # resume comparisons then remain stable after reading manifest.json.
        invariants = json.loads(
            json.dumps(
                {
                    "mesh_sha256": physics.model.mesh_sha256,
                    "dtype": args.dtype,
                    "model_spec": asdict(model_spec),
                    "pool_manifest": pool.manifest(),
                    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                    "train_seed": args.seed,
                    "optimizer_spec": {
                        "name": "Adam",
                        "learning_rate": args.learning_rate,
                        "foreach": False,
                    },
                    "gradient_clip_norm": args.gradient_clip_norm,
                    "step_regularization_weight": args.step_regularization_weight,
                },
                default=str,
            )
        )
        run_manifest = {
            "format_version": 1,
            "project": "cloth_tshirt_tensor_parallel_online_dynamics",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "physical_devices": list(args.devices),
            "parallelism": "DTensor ColwiseParallel hidden + RowwiseParallel output",
            "initialization": "local-shard normal with global fan-in; zero output",
            "optimizer": {
                "name": "Adam",
                "learning_rate": args.learning_rate,
                "foreach": False,
            },
            "fixed_model": asdict(physics.model),
            "model_spec": asdict(model_spec),
            "dimensions": dimensions,
            "actual_local_parameter_count": local_parameter_count(model),
            "pool": pool.manifest(),
            "validation_dataset": str(
                (Path(args.fixed_data_dir) / "validation_32.npz").resolve()
            ),
            "fast_validation": asdict(fast_protocol),
            "checkpoint_validation": asdict(full_protocol),
            "checkpoint_format": {
                "library": "torch.distributed.checkpoint",
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "full_checkpoints_retained": args.keep_checkpoints,
                "best_model_only_retained": 1 if args.save_best_model else 0,
            },
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "command_arguments": _json_arguments(args),
        }
        manifest_path = output / "run_manifest.json"
        is_resume = args.resume or args.resume_checkpoint is not None
        resume_checkpoint = (
            resolve_resume_checkpoint(output, args.resume_checkpoint) if is_resume else None
        )
        preflight: list[str | None] = [None]
        if rank == 0 and not is_resume and manifest_path.exists():
            preflight[0] = (
                "run directory already contains a manifest; use --resume or a new "
                f"path: {output}"
            )
        dist.broadcast_object_list(preflight, src=0)
        if preflight[0] is not None:
            raise FileExistsError(preflight[0])
        def write_run_manifest_if_missing() -> None:
            if not manifest_path.exists():
                if resume_checkpoint is not None:
                    run_manifest["resumed_from"] = str(resume_checkpoint)
                _atomic_write_json(manifest_path, run_manifest)

        _collective_rank0_action(
            rank, write_run_manifest_if_missing, context="run manifest creation"
        )

        update = 0
        prior_elapsed = 0.0
        best_rank: tuple[float, ...] | None = None
        best_update: int | None = None
        generation = 0
        last_checkpoint: Path | None = None
        last_saved_signature: tuple[Any, ...] | None = None
        if is_resume:
            assert resume_checkpoint is not None
            checkpoint = resume_checkpoint
            restored = load_full_checkpoint(
                checkpoint=checkpoint,
                model=model,
                optimizer=optimizer,
                pool=pool,
                rank=rank,
                device=device,
                expected_invariants=invariants,
            )
            update = int(restored["update_count"])
            prior_elapsed = float(restored["elapsed_seconds"])
            saved_best = restored.get("best_validation_rank")
            best_rank = None if saved_best is None else tuple(float(v) for v in saved_best)
            best_update = restored.get("best_validation_update")
            best_pointer = output / "best.json"
            if best_pointer.exists():
                selected = json.loads(best_pointer.read_text(encoding="utf-8"))
                pointer_rank = tuple(float(v) for v in selected["selection_rank"])
                if best_rank is None or pointer_rank < best_rank:
                    best_rank = pointer_rank
                    best_update = int(selected["update_count"])
            generation = next_checkpoint_generation(output)
            last_checkpoint = checkpoint
            last_saved_signature = (update, best_rank, best_update)
            _collective_rank0_action(
                rank,
                lambda: _truncate_training_log_for_resume(
                    output / "training_log.csv",
                    update=update,
                    checkpoint_generation=generation,
                ),
                context="resume log reconciliation",
            )
            if rank == 0:
                print(f"resumed update={update} from {checkpoint}", flush=True)

        stop_requested = False

        def request_stop(_signum, _frame) -> None:
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        start = time.monotonic()
        rows: list[dict[str, Any]] = []
        last_full_validation_update: int | None = None

        def elapsed() -> float:
            return prior_elapsed + time.monotonic() - start

        def save_training_state() -> Path:
            nonlocal generation, last_checkpoint, last_saved_signature
            signature = (update, best_rank, best_update)
            if (
                signature == last_saved_signature
                and last_checkpoint is not None
                and (last_checkpoint / "COMPLETE").is_file()
            ):
                return last_checkpoint
            checkpoint = save_full_checkpoint(
                run_dir=output,
                model=model,
                optimizer=optimizer,
                pool=pool,
                rank=rank,
                device=device,
                update=update,
                generation=generation,
                elapsed_seconds=elapsed(),
                best_rank=best_rank,
                best_update=best_update,
                manifest_invariants=invariants,
            )
            generation += 1
            last_checkpoint = checkpoint
            last_saved_signature = signature
            _collective_rank0_action(
                rank,
                lambda: prune_full_checkpoints(output, keep=args.keep_checkpoints),
                context="checkpoint pruning",
            )
            return checkpoint

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
            numeric_summary = {
                key: value
                for key, value in result.summary.items()
                if isinstance(value, (int, float))
            }
            assert_replicated_scalars(numeric_summary)
            _collective_rank0_action(
                rank,
                lambda: save_validation_result(
                    result=result,
                    output_root=output,
                    update=update,
                    render_plots=not args.no_validation_plots,
                ),
                context=f"{protocol.id} validation output",
            )
            if protocol.selects_checkpoint:
                last_full_validation_update = update
                candidate = checkpoint_rank(result.summary)
                if best_rank is None or candidate < best_rank:
                    best_rank, best_update = candidate, update
                    if args.save_best_model:
                        save_best_model(
                            run_dir=output,
                            model=model,
                            rank=rank,
                            update=update,
                            selection_rank=candidate,
                            manifest_invariants=invariants,
                        )
            model.train()
            return result.summary

        if update == 0 and not is_resume and not args.skip_initial_validation:
            summary = validate(fast_protocol)
            if rank == 0:
                print(
                    f"initial validation: failed={summary['failed_motion_count']} "
                    f"ratio_p95={summary['residual_ratio_p95']:.3e}",
                    flush=True,
                )

        interrupted = False
        while update < args.max_updates and elapsed() < args.max_wall_hours * 3600.0:
            stop_flag = torch.tensor(int(stop_requested), device=device)
            dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            if int(stop_flag.item()):
                interrupted = True
                break
            step_started = time.monotonic()
            metrics = tensor_parallel_training_step(
                model=model,
                optimizer=optimizer,
                pool=pool,
                gradient_clip_norm=args.gradient_clip_norm,
                step_regularization_weight=args.step_regularization_weight,
            )
            metrics["step_seconds"] = time.monotonic() - step_started
            rows.append(metrics)
            update += 1
            if update % args.log_interval == 0:
                row = _mean_metrics(rows, update=update, elapsed_seconds=elapsed())
                assert_replicated_scalars(
                    {
                        "loss": row["loss"],
                        "residual_ratio_p95": row["residual_ratio_p95"],
                        "resets_total": row["resets_total"],
                        "completed_physical_frames": row["completed_physical_frames"],
                    }
                )
                _collective_rank0_action(
                    rank,
                    lambda: _append_csv(output / "training_log.csv", row),
                    context="training log append",
                )
                if rank == 0:
                    print(
                        f"update={update} loss={row['loss']:.4e} "
                        f"ratio_p95={row['residual_ratio_p95']:.3e} "
                        f"updates/s={row['updates_per_second']:.2f}",
                        flush=True,
                    )
                rows.clear()
            if update % args.checkpoint_interval == 0:
                checkpoint = save_training_state()
                if rank == 0:
                    print(f"checkpoint update={update}: {checkpoint}", flush=True)
            if args.fast_validation_interval and update % args.fast_validation_interval == 0:
                summary = validate(fast_protocol)
                if rank == 0:
                    print(
                        f"fast validation update={update}: "
                        f"failed={summary['failed_motion_count']} "
                        f"ratio_p95={summary['residual_ratio_p95']:.3e}",
                        flush=True,
                    )
            if (
                args.checkpoint_validation_interval
                and update % args.checkpoint_validation_interval == 0
            ):
                summary = validate(full_protocol)
                if rank == 0:
                    print(
                        f"full validation update={update}: "
                        f"failed={summary['failed_motion_count']} "
                        f"ratio_p95={summary['residual_ratio_p95']:.3e}",
                        flush=True,
                    )

        if stop_requested:
            interrupted = True
        stop_flag = torch.tensor(int(interrupted), device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        interrupted = bool(stop_flag.item())
        if (
            not interrupted
            and not args.skip_final_validation
            and last_full_validation_update != update
        ):
            summary = validate(full_protocol)
            if rank == 0:
                print(
                    f"final full validation update={update}: "
                    f"failed={summary['failed_motion_count']} "
                    f"ratio_p95={summary['residual_ratio_p95']:.3e}",
                    flush=True,
                )

        post_validation_stop = torch.tensor(int(stop_requested), device=device)
        dist.all_reduce(post_validation_stop, op=dist.ReduceOp.MAX)
        interrupted = interrupted or bool(post_validation_stop.item())

        if rows:
            row = _mean_metrics(rows, update=update, elapsed_seconds=elapsed())
            assert_replicated_scalars(
                {"loss": row["loss"], "residual_ratio_p95": row["residual_ratio_p95"]}
            )
            _collective_rank0_action(
                rank,
                lambda: _append_csv(output / "training_log.csv", row),
                context="final training log append",
            )
        latest = save_training_state()
        completed = update >= args.max_updates or elapsed() >= args.max_wall_hours * 3600.0
        if rank == 0:
            try:
                plot_training_progress(output)
                plotting_error = None
            except Exception as error:
                plotting_error = repr(error)
        else:
            plotting_error = None
        _collective_rank0_action(
            rank,
            lambda: _atomic_write_json(
                output / "completed.json",
                {
                    "completed": bool(completed and not interrupted),
                    "interrupted": bool(interrupted),
                    "update_count": int(update),
                    "elapsed_seconds": float(elapsed()),
                    "best_validation_rank": best_rank,
                    "best_validation_update": best_update,
                    "latest_checkpoint": str(latest),
                    "best_checkpoint_pointer": (
                        str(output / "best.json") if (output / "best.json").exists() else None
                    ),
                    "plotting_error": plotting_error,
                },
            ),
            context="completion metadata",
        )
        if rank == 0:
            print(f"training state written to {output}", flush=True)
        dist.destroy_process_group()
        return 0
    except Exception as error:
        try:
            _atomic_write_json(
                failure_path,
                {
                    "rank": rank,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            pass
        if initialized:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    fixed_spec = load_model_spec(Path(args.fixed_data_dir) / "model_spec.json")
    validate_args(args, num_vertices=fixed_spec.num_vertices)
    if args.worker or "RANK" in os.environ:
        raise SystemExit(_worker(args, num_vertices=fixed_spec.num_vertices))

    dimensions_full_state = 3 * fixed_spec.num_vertices
    input_dim = 3 * dimensions_full_state
    global_parameters = input_dim * args.width + args.width * dimensions_full_state
    local_parameters = global_parameters // TENSOR_PARALLEL_SIZE
    command = worker_command(args)
    print(
        "training configuration: "
        f"input_dim={input_dim} width={args.width} ratio={args.width / input_dim:.3f} "
        f"global_parameters={global_parameters:,} local_parameters={local_parameters:,} "
        f"run_dir={run_directory(args)}",
        flush=True,
    )
    print(" ".join(command), flush=True)
    if args.dry_run:
        return
    if not args.resume and args.resume_checkpoint is None:
        manifest = run_directory(args) / "run_manifest.json"
        if manifest.exists():
            raise FileExistsError(
                f"run directory already contains a manifest; use --resume or a new path: "
                f"{run_directory(args)}"
            )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in args.devices)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(command, env=environment)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
