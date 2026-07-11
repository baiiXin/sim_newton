"""Launch the nested initial-point-count ablation."""
from __future__ import annotations

import argparse
from pathlib import Path

from cloth05_train_models import main as train_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("cloth_15x15_500step_pipeline")
    )
    parser.add_argument("--sample-source-root", type=Path, default=None)
    parser.add_argument(
        "--sample-counts", type=int, nargs="+", default=[1, 8, 32, 128, 512, 1024]
    )
    parser.add_argument("--activation", required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument(
        "--bias-mode", choices=("no-bias", "with-bias"), required=True
    )
    parser.add_argument("--sample-chunk-size", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--evaluation-steps", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    source = args.sample_source_root or (
        args.root / "data" / "initial_point_ablation" / "max_1024"
    )
    for count in args.sample_counts:
        argv = [
            "--root", str(args.root),
            "--stage", "initial_points",
            "--sample-source-root", str(source),
            "--sample-count", str(count),
            "--sample-chunk-size", str(args.sample_chunk_size),
            "--activations", args.activation,
            "--depths", str(args.depth),
            "--widths", str(args.width),
            "--bias-mode", args.bias_mode,
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--validation-interval", str(args.validation_interval),
            "--evaluation-steps", str(args.evaluation_steps),
        ]
        if args.resume:
            argv.append("--resume")
        if args.skip_completed:
            argv.append("--skip-completed")
        train_main(argv)


if __name__ == "__main__":
    main()
