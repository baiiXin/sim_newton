"""Build fixed data, probe memory, train, evaluate, and run the horizontal-motion scan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from tshirt_config import DEFAULT_FIXED_DATA_DIR, DEFAULT_TRAIN_SEED, write_json


DEFAULT_ROOT = Path("cloth_tshirt_pipeline")


def experiment_name(activation: str, depth: int, width: int, use_bias: bool) -> str:
    return (
        f"activation_{activation}_depth_{depth:02d}_width_{width:04d}_"
        f"{'bias' if use_bias else 'no_bias'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--use-bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--fallback-batch-size", type=int, default=32)
    parser.add_argument("--max-updates", type=int, default=3_000_000)
    parser.add_argument("--max-wall-hours", type=float, default=10.0)
    parser.add_argument("--evaluation-rollout-frames", type=int, default=500)
    parser.add_argument("--skip-memory-probe", action="store_true")
    parser.add_argument("--skip-final-evaluation", action="store_true")
    parser.add_argument("--skip-single-motion-scan", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run(command: list[str], log: Path, dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    root = args.root.resolve()
    logs = root / "pipeline_logs"
    python = sys.executable
    state: dict[str, Any] = {"configuration": vars(args), "steps": {}}

    fixed_required = (
        Path(args.fixed_data_dir) / "model_spec.json",
        Path(args.fixed_data_dir) / "topology_cache.npz",
        Path(args.fixed_data_dir) / "validation_32.npz",
        Path(args.fixed_data_dir) / "test_64.npz",
        Path(args.fixed_data_dir) / "typical_single_motions_4.npz",
    )
    fixed_was_present = all(path.exists() for path in fixed_required)
    _run(
        [python, str(project / "cloth01_build_fixed_model_and_datasets.py"),
         "--output-dir", str(Path(args.fixed_data_dir).resolve())],
        logs / "01_build_fixed_data.log", args.dry_run,
    )
    state["steps"]["fixed_data"] = (
        "planned" if args.dry_run else "validated_and_reused" if fixed_was_present else "completed"
    )

    memory_dir = root / "profiling" / "memory_probe"
    recommendation_path = memory_dir / "recommended_training_config.json"
    if not args.skip_memory_probe:
        command = [
            python, str(project / "cloth06_probe_memory_and_throughput.py"),
            "--output-dir", str(memory_dir),
            "--fixed-data-dir", str(Path(args.fixed_data_dir).resolve()),
            "--device", args.device,
            "--dtype", args.dtype,
            "--seed", str(args.seed),
            "--activation", args.activation,
            "--depth", str(args.depth),
            "--width", str(args.width),
            "--pool-size", str(args.pool_size),
            "--use-bias" if args.use_bias else "--no-use-bias",
        ]
        _run(command, logs / "02_memory_probe.log", args.dry_run)
        state["steps"]["memory_probe"] = "planned" if args.dry_run else "completed"
    else:
        state["steps"]["memory_probe"] = "skipped"

    batch_size = args.fallback_batch_size
    if recommendation_path.exists() and not args.dry_run:
        recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
        value = recommendation.get("recommendation")
        if value:
            batch_size = int(value["recommended_batch_size"])
    run_dir = root / experiment_name(
        args.activation, args.depth, args.width, args.use_bias
    ) / f"seed_{args.seed}"
    train = [
        python, str(project / "cloth05_train_online.py"),
        "--output-root", str(root),
        "--fixed-data-dir", str(Path(args.fixed_data_dir).resolve()),
        "--device", args.device,
        "--dtype", args.dtype,
        "--seed", str(args.seed),
        "--activation", args.activation,
        "--depth", str(args.depth),
        "--width", str(args.width),
        "--pool-size", str(args.pool_size),
        "--batch-size", str(batch_size),
        "--max-updates", str(args.max_updates),
        "--max-wall-hours", str(args.max_wall_hours),
        "--use-bias" if args.use_bias else "--no-use-bias",
    ]
    if args.resume:
        train.append("--resume")
    _run(train, logs / "03_training.log", args.dry_run)
    state["steps"]["training"] = "planned" if args.dry_run else "completed"

    best = run_dir / "best_validation_model.pt"
    latest = run_dir / "latest_checkpoint.pt"
    checkpoint = best if best.exists() else latest
    if not args.skip_final_evaluation:
        _run(
            [
                python, str(project / "cloth07_evaluate_checkpoint.py"),
                "--checkpoint", str(checkpoint),
                "--fixed-data-dir", str(Path(args.fixed_data_dir).resolve()),
                "--device", args.device,
                "--dtype", args.dtype,
                "--rollout-frames", str(args.evaluation_rollout_frames),
                "--inner-steps", "50",
            ],
            logs / "04_final_evaluation.log", args.dry_run,
        )
        state["steps"]["final_evaluation"] = "planned" if args.dry_run else "completed"
    else:
        state["steps"]["final_evaluation"] = "skipped"

    if not args.skip_single_motion_scan:
        _run(
            [
                python, str(project / "cloth12_scan_single_motion_rollouts.py"),
                "--root", str(root),
                "--fixed-data-dir", str(Path(args.fixed_data_dir).resolve()),
                "--device", args.device,
                "--dtype", args.dtype,
                "--split", "typical",
                "--motion-index", "0",
                "--checkpoint-kind", "best",
                "--rollout-frames", str(args.evaluation_rollout_frames),
                "--inner-steps", "50",
                "--line-search-max-trials", "12",
            ],
            logs / "05_horizontal_single_motion_scan.log", args.dry_run,
        )
        state["steps"]["single_motion_scan"] = "planned" if args.dry_run else "completed"
    else:
        state["steps"]["single_motion_scan"] = "skipped"
    state["recommended_batch_size_used"] = batch_size
    state["run_directory"] = str(run_dir)
    write_json(root / "pipeline_state.json", state)
    print(f"pipeline state written to {root / 'pipeline_state.json'}")


if __name__ == "__main__":
    main()
