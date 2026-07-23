"""Run one frozen T-shirt motion with a two-GPU tensor-parallel checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Sequence

from tshirt_config import DEFAULT_FIXED_DATA_DIR, DEFAULT_TRAIN_SEED, load_model_spec


TENSOR_PARALLEL_SIZE = 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument(
        "--devices",
        type=int,
        nargs=TENSOR_PARALLEL_SIZE,
        default=(0, 1),
        metavar=("GPU0", "GPU1"),
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--split", choices=("typical", "validation", "test"), default="typical")
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--rollout-frames", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--residual-ratio-tolerance", type=float, default=1e-3)
    parser.add_argument("--trajectory-stride", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _checkpoint_manifest(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir() or not (checkpoint / "COMPLETE").is_file():
        raise FileNotFoundError(f"checkpoint is not a complete DCP directory: {checkpoint}")
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "model_only":
        raise ValueError("single-motion rollout requires a model-only tensor-parallel checkpoint")
    if int(manifest.get("world_size", -1)) != TENSOR_PARALLEL_SIZE:
        raise ValueError("checkpoint tensor-parallel world size is not 2")
    model_spec = manifest.get("model_spec", {})
    if int(model_spec.get("depth", -1)) != 1 or bool(model_spec.get("use_bias", True)):
        raise ValueError("only the depth-one bias-free tensor-parallel MLP is supported")
    return manifest


def validate_args(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    if len(args.devices) != 2 or len(set(args.devices)) != 2:
        raise ValueError("--devices must name two distinct CUDA device indices")
    if args.motion_index < 0:
        raise ValueError("--motion-index must be nonnegative")
    if args.rollout_frames <= 0 or args.inner_steps <= 0 or args.trajectory_stride <= 0:
        raise ValueError("rollout frames, inner steps, and trajectory stride must be positive")
    if manifest.get("dtype") != args.dtype:
        raise ValueError(
            f"checkpoint dtype {manifest.get('dtype')!r} does not match --dtype {args.dtype!r}"
        )


def _forwarded_arguments(args: argparse.Namespace) -> list[str]:
    output = [
        "--checkpoint", str(args.checkpoint.resolve()),
        "--output-dir", str(args.output_dir.resolve()),
        "--fixed-data-dir", str(args.fixed_data_dir.resolve()),
        "--devices", *(str(value) for value in args.devices),
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--split", args.split,
        "--motion-index", str(args.motion_index),
        "--rollout-frames", str(args.rollout_frames),
        "--inner-steps", str(args.inner_steps),
        "--residual-ratio-tolerance", str(args.residual_ratio_tolerance),
        "--trajectory-stride", str(args.trajectory_stride),
    ]
    if args.overwrite:
        output.append("--overwrite")
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _worker(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh

    from cloth02_batched_physics import load_frozen_motion_batch, load_physics
    from cloth09_rollout_single_motion import (
        SingleMotionSettings,
        save_solver_rollout,
        select_motion,
        split_path,
        run_solver_rollout,
    )
    from cloth_tensor_parallel import build_tensor_parallel_model

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    output = args.output_dir.resolve()
    failure_path = output / f"failure_rank_{rank:02d}.json"
    initialized = False
    try:
        if world_size != TENSOR_PARALLEL_SIZE:
            raise RuntimeError(f"expected world size 2, received {world_size}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
            raise RuntimeError(
                f"expected {world_size} visible CUDA devices, found {torch.cuda.device_count()}"
            )
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        initialized = True
        device_mesh = init_device_mesh("cuda", (world_size,))
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        dtype = torch.float32 if args.dtype == "float32" else torch.float64
        physics = load_physics(
            fixed_data_dir=args.fixed_data_dir,
            device=device,
            dtype=dtype,
        )
        if manifest.get("mesh_sha256") != physics.model.mesh_sha256:
            raise ValueError("checkpoint mesh hash does not match the fixed T-shirt model")
        model_spec = manifest["model_spec"]
        model = build_tensor_parallel_model(
            physics=physics,
            activation=str(model_spec["activation"]),
            width=int(model_spec["width"]),
            device_mesh=device_mesh,
            rank=rank,
            seed=args.seed,
        )
        model_state = get_model_state_dict(model)
        state = {"model": model_state}
        dcp.load(state, checkpoint_id=args.checkpoint.resolve() / "distributed")
        incompatible = set_model_state_dict(model, state["model"])
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"checkpoint state mismatch: {incompatible}")
        model.eval()

        data_path = split_path(args.fixed_data_dir, args.split)
        dataset = load_frozen_motion_batch(data_path, device=device, dtype=dtype)
        motion = select_motion(dataset, args.motion_index)
        settings = SingleMotionSettings(
            rollout_frames=args.rollout_frames,
            inner_steps=args.inner_steps,
            residual_ratio_tolerance=args.residual_ratio_tolerance,
            trajectory_stride=args.trajectory_stride,
            early_stop=False,
        )
        result = run_solver_rollout(
            solver="network",
            physics=physics,
            motion=motion,
            settings=settings,
            model=model,
        )
        result.summary.update(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_update": int(manifest["update_count"]),
                "model_spec": model_spec,
                "tensor_parallel_size": world_size,
            }
        )
        summaries: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(summaries, result.summary)
        if summaries[0] != summaries[1]:
            raise RuntimeError("tensor-parallel ranks produced different rollout summaries")
        if rank == 0:
            if output.exists() and any(output.iterdir()) and not args.overwrite:
                raise FileExistsError(f"output directory is not empty; use --overwrite: {output}")
            save_solver_rollout(result, output / "network")
            _write_json(
                output / "manifest.json",
                {
                    "format_version": 1,
                    "completed": True,
                    "completed_utc": datetime.now(timezone.utc).isoformat(),
                    "checkpoint_manifest": manifest,
                    "motion_id": motion.motion_ids[0],
                    "split": args.split,
                    "motion_index": args.motion_index,
                    "settings": asdict(settings),
                    "result": result.summary,
                },
            )
            print(
                f"single-motion result: failed={result.summary['failed']} "
                f"survival={result.summary['survival_frames']}/{args.rollout_frames} "
                f"ratio_p95={result.summary['residual_ratio_p95']:.3e}",
                flush=True,
            )
            print(f"result written to {output}", flush=True)
        dist.barrier()
        dist.destroy_process_group()
        return 0
    except Exception as error:
        try:
            _write_json(
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
    manifest = _checkpoint_manifest(args.checkpoint)
    validate_args(args, manifest)
    fixed_spec = load_model_spec(Path(args.fixed_data_dir) / "model_spec.json")
    if manifest.get("mesh_sha256") != fixed_spec.mesh_sha256:
        raise ValueError("checkpoint mesh hash does not match fixed_data/model_spec.json")
    if args.worker or "RANK" in os.environ:
        raise SystemExit(_worker(args, manifest))

    command = worker_command(args)
    print(
        "tensor-parallel single-motion configuration: "
        f"checkpoint_update={manifest['update_count']} "
        f"width={manifest['model_spec']['width']} "
        f"motion={args.split}[{args.motion_index}] "
        f"frames={args.rollout_frames} inner_steps={args.inner_steps}",
        flush=True,
    )
    print(" ".join(command), flush=True)
    if args.dry_run:
        return
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in args.devices)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(command, env=environment)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
