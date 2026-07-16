"""Run the default 50-iteration full evaluation on frozen validation and test sets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from cloth02_batched_physics import load_frozen_motion_batch, load_physics
from cloth04_reference_free_validation import run_reference_free_validation, save_validation_result
from cloth05_train_online import load_model_checkpoint
from tshirt_config import DEFAULT_EVALUATION, DEFAULT_FIXED_DATA_DIR, write_json
from validation_protocol import ValidationProtocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--splits", choices=("validation", "test"), nargs="+", default=("validation", "test"))
    parser.add_argument("--rollout-frames", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=DEFAULT_EVALUATION.full_inner_steps)
    parser.add_argument(
        "--residual-ratio-tolerance",
        type=float,
        default=DEFAULT_EVALUATION.convergence_residual_ratio,
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _seed_dir(checkpoint: Path) -> Path:
    return checkpoint.parent.parent if checkpoint.parent.name == "periodic" else checkpoint.parent


def main() -> None:
    args = parse_args()
    if args.inner_steps <= 0 or args.rollout_frames <= 0:
        raise ValueError("inner steps and rollout frames must be positive")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    physics = load_physics(fixed_data_dir=args.fixed_data_dir, device=args.device, dtype=dtype)
    model, _, checkpoint = load_model_checkpoint(args.checkpoint, physics=physics)
    update = int(checkpoint.get("update_count", -1))
    output = (
        Path(args.output_dir)
        if args.output_dir is not None
        else _seed_dir(args.checkpoint.resolve()) / "final_evaluation" / f"update_{update:09d}"
    ).resolve()
    summaries = []
    for split in args.splits:
        count = (
            DEFAULT_EVALUATION.validation_count
            if split == "validation"
            else DEFAULT_EVALUATION.test_count
        )
        path = Path(args.fixed_data_dir) / (
            "validation_32.npz" if split == "validation" else "test_64.npz"
        )
        motions = load_frozen_motion_batch(path, device=args.device, dtype=dtype)
        protocol = ValidationProtocol(
            id=f"full_{split}_k{args.inner_steps}",
            motion_count=count,
            rollout_frames=args.rollout_frames,
            inner_steps=args.inner_steps,
            interval_updates=0,
            selects_checkpoint=False,
            residual_ratio_tolerance=args.residual_ratio_tolerance,
            early_stop=False,
        )
        result = run_reference_free_validation(
            model=model,
            physics=physics,
            motions=motions,
            protocol=protocol,
            batch_size=args.batch_size,
        )
        save_validation_result(
            result=result,
            output_root=output / split,
            update=update,
            render_plots=not args.no_plots,
        )
        summaries.append({"split": split, **result.summary})
        print(
            f"{split}: failed={result.summary['failed_motion_count']} "
            f"ratio_p95={result.summary['residual_ratio_p95']:.3e} "
            f"slow_first_step_frames={result.summary['single_step_le_two_orders_frame_count']}"
        )
    write_json(
        output / "evaluation_manifest.json",
        {
            "completed": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_update": update,
            "model_spec": checkpoint["model_spec"],
            "settings": {
                "rollout_frames": args.rollout_frames,
                "inner_steps": args.inner_steps,
                "early_stop": False,
                "fixed_inner_iteration_budget": True,
                "residual_ratio_tolerance": args.residual_ratio_tolerance,
                "single_step_ratio_threshold": DEFAULT_EVALUATION.two_order_single_step_ratio,
            },
            "summaries": summaries,
        },
    )
    print(f"full evaluation written to {output}")


if __name__ == "__main__":
    main()
