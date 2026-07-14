"""Run memory profiling, six-hour training, and final evaluation as one pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

from cloth03_training_pool import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENVIRONMENT_LIFETIME_FRAMES,
    DEFAULT_POOL_SIZE,
    ModelSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("cloth_5x5_scale_up_pipeline"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--catalogue", choices=("c1", "c2", "c3"), default="c2")
    parser.add_argument("--activation", default=ModelSpec().activation)
    parser.add_argument("--depth", type=int, default=ModelSpec().depth)
    parser.add_argument("--width", type=int, default=ModelSpec().width)
    parser.add_argument(
        "--use-bias",
        action=argparse.BooleanOptionalAction,
        default=ModelSpec().use_bias,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--training-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-lifetime-physical-steps",
        type=int,
        default=DEFAULT_ENVIRONMENT_LIFETIME_FRAMES,
    )
    parser.add_argument("--use-recommended-batch", action="store_true")
    parser.add_argument(
        "--memory-batch-sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512, 1024, 2048],
    )
    parser.add_argument("--memory-warmup-updates", type=int, default=20)
    parser.add_argument("--memory-measured-updates", type=int, default=100)
    parser.add_argument("--max-wall-hours", type=float, default=10.0)
    parser.add_argument("--max-updates", type=int, default=3000000)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--evaluation-frames", type=int, default=500)
    parser.add_argument("--evaluation-checkpoint-update", type=int, default=None)
    parser.add_argument("--inner-steps", type=int, nargs="+", default=[1, 3, 10, 30])
    parser.add_argument("--skip-memory-probe", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_command(
    command: Sequence[str],
    *,
    log_path: Path,
    dry_run: bool,
) -> None:
    text = " ".join(command)
    print(f"\n$ {text}\n")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(text + "\n", encoding="utf-8")
        return
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def run_directory(args: argparse.Namespace) -> Path:
    spec = ModelSpec(
        activation=args.activation,
        depth=args.depth,
        width=args.width,
        use_bias=args.use_bias,
    )
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
    if args.max_updates < 0:
        raise ValueError("max-updates must be nonnegative")
    if args.training_batch_size <= 0 or args.pool_size <= 0:
        raise ValueError("pool-size and training-batch-size must be positive")
    if args.max_lifetime_physical_steps <= 0:
        raise ValueError("max-lifetime-physical-steps must be positive")
    project_dir = Path(__file__).resolve().parent
    logs = args.root / "pipeline_logs"
    memory_dir = args.root / "profiling" / "memory_probe"
    state: dict[str, Any] = {
        "started_at_unix": time.time(),
        "device": args.device,
        "dtype": args.dtype,
        "catalogue": args.catalogue,
        "pool_size": args.pool_size,
        "requested_training_batch_size": args.training_batch_size,
        "max_wall_hours": args.max_wall_hours,
        "max_updates": args.max_updates,
        "max_lifetime_physical_steps": args.max_lifetime_physical_steps,
        "dry_run": args.dry_run,
        "steps": {},
    }

    if not args.skip_memory_probe:
        memory_command = [
            sys.executable,
            str(project_dir / "cloth06_probe_memory_and_throughput.py"),
            "--output-dir",
            str(memory_dir),
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
            "--pool-size",
            str(args.pool_size),
            "--batch-sizes",
            *[str(value) for value in args.memory_batch_sizes],
            "--warmup-updates",
            str(args.memory_warmup_updates),
            "--measured-updates",
            str(args.memory_measured_updates),
            "--seed",
            str(args.seed),
        ]
        memory_command.append("--use-bias" if args.use_bias else "--no-use-bias")
        run_command(
            memory_command,
            log_path=logs / "01_memory_probe.log",
            dry_run=args.dry_run,
        )
        state["steps"]["memory_probe"] = "planned" if args.dry_run else "completed"

    training_batch_size = int(args.training_batch_size)
    recommendation_path = memory_dir / "recommended_training_config.json"
    if args.use_recommended_batch and not args.dry_run:
        if not recommendation_path.exists():
            raise FileNotFoundError(recommendation_path)
        recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
        selected = recommendation.get("recommended_batch_size")
        if selected is None:
            raise RuntimeError("显存测试没有找到可用 batch size")
        training_batch_size = int(selected)
    state["training_batch_size"] = training_batch_size

    out = run_directory(args)
    if not args.skip_training:
        train_command = [
            sys.executable,
            str(project_dir / "cloth05_train_scale_up_robust.py"),
            "--root",
            str(args.root),
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
            "--pool-size",
            str(args.pool_size),
            "--batch-size",
            str(training_batch_size),
            "--max-wall-hours",
            str(args.max_wall_hours),
            "--max-updates",
            str(args.max_updates),
            "--max-lifetime-physical-steps",
            str(args.max_lifetime_physical_steps),
            "--validation-batch-size",
            str(args.validation_batch_size),
            "--seed",
            str(args.seed),
        ]
        train_command.append("--use-bias" if args.use_bias else "--no-use-bias")
        if args.resume:
            train_command.append("--resume")
        if args.overwrite:
            train_command.append("--overwrite")
        run_command(
            train_command,
            log_path=logs / "02_training.log",
            dry_run=args.dry_run,
        )
        state["steps"]["training"] = "planned" if args.dry_run else "completed"

    if not args.skip_evaluation:
        evaluate_command = [
            sys.executable,
            str(project_dir / "cloth07_evaluate_best_checkpoint.py"),
            "--run-dir",
            str(out),
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
            "--validation-frames",
            str(args.evaluation_frames),
            "--test-frames",
            str(args.evaluation_frames),
            "--inner-steps",
            *[str(value) for value in args.inner_steps],
            "--batch-size",
            str(args.evaluation_batch_size),
        ]
        evaluate_command.append("--use-bias" if args.use_bias else "--no-use-bias")
        if args.evaluation_checkpoint_update is not None:
            evaluate_command.extend(
                ["--checkpoint-update", str(args.evaluation_checkpoint_update)]
            )
        run_command(
            evaluate_command,
            log_path=logs / "03_final_evaluation.log",
            dry_run=args.dry_run,
        )
        state["steps"]["final_evaluation"] = (
            "planned" if args.dry_run else "completed"
        )

    state["run_directory"] = str(out)
    state["finished_at_unix"] = time.time()
    write_json(args.root / "pipeline_state.json", state)
    print(f"\n流水线完成。状态文件：{args.root / 'pipeline_state.json'}")


if __name__ == "__main__":
    main()
