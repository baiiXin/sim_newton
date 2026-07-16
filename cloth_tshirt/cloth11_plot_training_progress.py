"""Create a unified set of training and validation plots for one run."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np

from tshirt_config import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _numeric(rows: list[dict[str, str]], column: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except (KeyError, TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def _plot_lines(
    *,
    rows: list[dict[str, str]],
    x_column: str,
    lines: Iterable[tuple[str, str]],
    output: Path,
    title: str,
    y_label: str,
    log_y: bool,
    dpi: int,
) -> bool:
    if not rows:
        return False
    import matplotlib.pyplot as plt

    x = _numeric(rows, x_column)
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    plotted = False
    for column, label in lines:
        y = _numeric(rows, column)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            continue
        if log_y:
            finite &= y > 0.0
        if not finite.any():
            continue
        axis.plot(x[finite], y[finite], label=label, linewidth=1.35)
        plotted = True
    if not plotted:
        plt.close(figure)
        return False
    if log_y:
        axis.set_yscale("log")
    axis.set(xlabel=x_column.replace("_", " "), ylabel=y_label, title=title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return True


def plot_training_progress(run_dir: Path, output_dir: Path | None = None, dpi: int = 150) -> list[Path]:
    run_dir = Path(run_dir)
    output_dir = run_dir / "figures" / "training_progress" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    training = _read_csv(run_dir / "training_log.csv")
    specs = (
        (
            training,
            "update",
            (("loss", "normalized loss"), ("normalized_energy_change_mean", "energy change")),
            "01_training_objective.png",
            "Online training objective",
            "normalized value",
            False,
        ),
        (
            training,
            "update",
            (("residual_ratio_p50", "ratio p50"), ("residual_ratio_p95", "ratio p95")),
            "02_training_residual.png",
            "Training-pool one-step residual ratio",
            "residual after / before",
            True,
        ),
        (
            training,
            "update",
            (("gradient_norm_before_clip", "before clip"), ("gradient_norm_after_clip", "after clip")),
            "03_gradient_norm.png",
            "Gradient norm and clipping",
            "L2 norm",
            True,
        ),
        (
            training,
            "update",
            (
                ("resets_nonfinite", "non-finite"), ("resets_energy", "energy"),
                ("resets_residual", "residual"), ("resets_area", "area"),
                ("resets_edge", "edge"), ("resets_lifetime", "lifetime"),
            ),
            "04_pool_resets.png",
            "Online pool resets per log interval",
            "row resets",
            False,
        ),
    )
    for rows, x, lines, filename, title, y_label, log_y in specs:
        path = output_dir / filename
        if _plot_lines(
            rows=rows, x_column=x, lines=lines, output=path, title=title,
            y_label=y_label, log_y=log_y, dpi=dpi,
        ):
            produced.append(path)

    validation_histories = sorted((run_dir / "validation").glob("*/history.csv"))
    if validation_histories:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        plotted = False
        for history in validation_histories:
            rows = _read_csv(history)
            x = _numeric(rows, "update")
            ratio = _numeric(rows, "residual_ratio_p95")
            slow = _numeric(rows, "single_step_le_two_orders_frame_fraction")
            finite_ratio = np.isfinite(x) & np.isfinite(ratio) & (ratio > 0)
            finite_slow = np.isfinite(x) & np.isfinite(slow)
            label = history.parent.name
            if finite_ratio.any():
                axes[0].semilogy(x[finite_ratio], ratio[finite_ratio], marker="o", ms=3, label=label)
                plotted = True
            if finite_slow.any():
                axes[1].plot(x[finite_slow], slow[finite_slow], marker="o", ms=3, label=label)
                plotted = True
        if plotted:
            axes[0].set(title="Frozen validation", xlabel="optimizer update", ylabel="residual ratio p95")
            axes[1].set(
                title="Frames with <=2-order first-step decrease",
                xlabel="optimizer update", ylabel="fraction",
            )
            for axis in axes:
                axis.grid(True, which="both", alpha=0.25)
                axis.legend()
            path = output_dir / "05_validation_progress.png"
            figure.tight_layout(); figure.savefig(path, dpi=dpi); plt.close(figure)
            produced.append(path)
        else:
            plt.close(figure)

    manifest = {
        "run_directory": str(run_dir.resolve()),
        "files": [str(path.resolve()) for path in produced],
        "training_rows": len(training),
        "validation_histories": [str(path.resolve()) for path in validation_histories],
    }
    write_json(output_dir / "plot_manifest.json", manifest)
    return produced


def main() -> None:
    args = parse_args()
    outputs = plot_training_progress(args.run_dir, args.output_dir, args.dpi)
    print(f"wrote {len(outputs)} plots to {(args.output_dir or args.run_dir / 'figures' / 'training_progress')}")


if __name__ == "__main__":
    main()
