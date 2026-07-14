"""Plot training loss and validation residual histories for one scale-up run."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from cloth03_training_pool import ModelSpec


DEFAULT_ROOT = Path("cloth_5x5_scale_up_pipeline")
DEFAULT_OUTPUT_SUBDIR = Path("figures") / "training_progress"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
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
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--loss-column", default="loss_mean")
    parser.add_argument("--validation-metric", default="residual_ratio_p95")
    parser.add_argument(
        "--validation-linear-y",
        action="store_true",
        help="Use a linear y-axis for validation residual plots instead of log scale.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def run_directory(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_finite_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def extract_series(
    rows: Sequence[dict[str, str]],
    *,
    x_column: str,
    y_column: str,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x_value = parse_finite_float(row.get(x_column))
        y_value = parse_finite_float(row.get(y_column))
        if x_value is None or y_value is None:
            continue
        xs.append(x_value)
        ys.append(y_value)
    if not xs:
        available = sorted(rows[0].keys()) if rows else []
        raise ValueError(
            f"No finite data points for {x_column}/{y_column}; available columns: {available}"
        )
    return xs, ys


def load_best_update(run_dir: Path) -> int | None:
    completed_path = run_dir / "completed.json"
    if completed_path.exists():
        with completed_path.open("r", encoding="utf-8") as handle:
            completed = json.load(handle)
        value = completed.get("best_validation_update")
        if value is not None:
            return int(value)

    checkpoint_path = run_dir / "best_validation_model.pt"
    if checkpoint_path.exists():
        try:
            import torch

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception:
            return None
        for key in ("best_validation_update", "update_count"):
            value = checkpoint.get(key)
            if value is not None:
                return int(value)
    return None


def plot_series(
    *,
    xs: Sequence[float],
    ys: Sequence[float],
    title: str,
    y_label: str,
    output_path: Path,
    best_update: int | None,
    log_y: bool,
    y_limits: tuple[float, float] | None,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.3)
    if best_update is not None:
        ax.axvline(
            best_update,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=f"best checkpoint update = {best_update}",
        )
        ax.legend(fontsize=8)
    if log_y and all(value > 0.0 for value in ys):
        ax.set_yscale("log")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    ax.set_xlabel("optimizer update steps")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def centered_loss_limits(values: Sequence[float]) -> tuple[float, float]:
    radius = 1.2 * abs(min(values))
    if radius <= 0.0 or not math.isfinite(radius):
        radius = 1.2 * max(abs(value) for value in values)
    if radius <= 0.0 or not math.isfinite(radius):
        radius = 1.0
    return -radius, radius


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    run_dir = run_directory(args)
    output_dir = args.output_dir if args.output_dir is not None else run_dir / DEFAULT_OUTPUT_SUBDIR
    best_update = load_best_update(run_dir)

    train_log = run_dir / "train_log.csv"
    fast_history = run_dir / "validation" / "fast_monitor" / "history.csv"
    checkpoint_history = run_dir / "validation" / "checkpoint_validation" / "history.csv"

    outputs: dict[str, str] = {}

    loss_xs, loss_ys = extract_series(
        read_csv_rows(train_log),
        x_column="update_count",
        y_column=args.loss_column,
    )
    loss_path = output_dir / "loss_vs_update.png"
    plot_series(
        xs=loss_xs,
        ys=loss_ys,
        title="Training loss vs optimizer update steps",
        y_label=args.loss_column,
        output_path=loss_path,
        best_update=best_update,
        log_y=False,
        y_limits=centered_loss_limits(loss_ys),
        dpi=args.dpi,
    )
    outputs["loss"] = str(loss_path)

    validation_specs = (
        (
            "fast_monitor",
            fast_history,
            output_dir / "fast_monitor_residual_vs_update.png",
            "Fast validation residual vs optimizer update steps",
        ),
        (
            "checkpoint_validation",
            checkpoint_history,
            output_dir / "checkpoint_validation_residual_vs_update.png",
            "Checkpoint validation residual vs optimizer update steps",
        ),
    )
    for name, history_path, output_path, title in validation_specs:
        xs, ys = extract_series(
            read_csv_rows(history_path),
            x_column="update_count",
            y_column=args.validation_metric,
        )
        plot_series(
            xs=xs,
            ys=ys,
            title=title,
            y_label=args.validation_metric,
            output_path=output_path,
            best_update=best_update,
            log_y=not args.validation_linear_y,
            y_limits=None,
            dpi=args.dpi,
        )
        outputs[name] = str(output_path)

    manifest_path = output_dir / "training_progress_manifest.json"
    write_manifest(
        manifest_path,
        {
            "run_dir": str(run_dir),
            "best_validation_update": best_update,
            "loss_column": args.loss_column,
            "validation_metric": args.validation_metric,
            "sources": {
                "train_log": str(train_log),
                "fast_monitor_history": str(fast_history),
                "checkpoint_validation_history": str(checkpoint_history),
            },
            "outputs": outputs,
        },
    )
    outputs["manifest"] = str(manifest_path)

    print(f"run_dir: {run_dir}")
    print(f"best checkpoint update: {best_update if best_update is not None else 'n/a'}")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
