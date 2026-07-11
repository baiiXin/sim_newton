"""Plot only the depth=1 width=256 identity/ReLU/Tanh models for both sources."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = (
    "seen_motion_temporal_interpolation",
    "seen_motion_temporal_extrapolation",
    "unseen_id_test",
    "ood_test",
)
ACTIVATIONS = ("identity", "relu", "tanh")
SOURCES = {
    "history": "history_input_default_init_activation_{activation}_depth_01_width_256_no_bias",
    "degenerate": "degenerate_no_initial_perturbation_history_input_default_init_activation_{activation}_depth_01_width_256_no_bias",
}
SOURCE_LABELS = {
    "history": "sampled init",
    "degenerate": "current-state init",
}
SOURCE_STYLES = {
    "history": "-",
    "degenerate": "--",
}
ACTIVATION_COLORS = {
    "identity": "#1f77b4",
    "relu": "#d62728",
    "tanh": "#2ca02c",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_path(root: Path, source: str, activation: str, dataset: str) -> Path:
    model_name = SOURCES[source].format(activation=activation)
    return root / source / model_name / dataset / "metrics.json"


def load_curve(root: Path, source: str, activation: str, dataset: str, metric_key: str) -> np.ndarray:
    path = metric_path(root, source, activation, dataset)
    if not path.exists():
        raise FileNotFoundError(path)
    metrics = load_json(path)
    if metric_key not in metrics:
        raise KeyError(f"{metric_key} not found in {path}")
    curve = np.asarray(metrics[metric_key], dtype=float)
    curve[~np.isfinite(curve)] = np.nan
    return curve


def display_dataset_name(name: str) -> str:
    return {
        "seen_motion_temporal_interpolation": "seen interpolation",
        "seen_motion_temporal_extrapolation": "seen extrapolation",
        "unseen_id_test": "unseen ID",
        "ood_test": "OOD",
    }.get(name, name)


def plot_1x256_six_models(
    *,
    root: Path = ROOT,
    output_dir: Path | None = None,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    metric_key: str = "residual_mean_by_step",
    y_floor: float = 1e-16,
) -> list[Path]:
    """Draw 6 curves: 2 sources x 3 activations, only depth=1 width=256."""
    output_dir = output_dir or (root / "figures_1x256")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # One compact 2x2 overview.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, dataset in zip(axes.flatten(), datasets):
        for activation in ACTIVATIONS:
            for source in SOURCES:
                curve = load_curve(root, source, activation, dataset, metric_key)
                x = np.arange(curve.size)
                y = np.maximum(curve, y_floor)
                label = f"{SOURCE_LABELS[source]} / {activation}"
                ax.plot(
                    x,
                    y,
                    linestyle=SOURCE_STYLES[source],
                    color=ACTIVATION_COLORS[activation],
                    linewidth=1.9,
                    label=label,
                )
        ax.set_yscale("log")
        ax.set_title(display_dataset_name(dataset))
        ax.set_xlabel("iteration")
        ax.set_ylabel(metric_key.replace("_by_step", "").replace("_", " "))
        ax.grid(True, which="major", alpha=0.25)
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Depth 1 / Width 256: six 3x69D models", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    overview_path = output_dir / f"overview_1x256_{metric_key}.png"
    fig.savefig(overview_path, dpi=220)
    plt.close(fig)
    saved.append(overview_path)

    # Individual dataset figures for reading details.
    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        for activation in ACTIVATIONS:
            for source in SOURCES:
                curve = load_curve(root, source, activation, dataset, metric_key)
                x = np.arange(curve.size)
                y = np.maximum(curve, y_floor)
                label = f"{SOURCE_LABELS[source]} / {activation}"
                ax.plot(
                    x,
                    y,
                    linestyle=SOURCE_STYLES[source],
                    color=ACTIVATION_COLORS[activation],
                    linewidth=2.1,
                    label=label,
                )
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel(metric_key.replace("_by_step", "").replace("_", " "))
        ax.set_title(f"{display_dataset_name(dataset)}: depth 1 / width 256")
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        save_path = output_dir / f"{dataset}_1x256_{metric_key}.png"
        fig.savefig(save_path, dpi=220)
        plt.close(fig)
        saved.append(save_path)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot only depth=1 width=256 six-model curves.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--metric-key", default="residual_mean_by_step")
    parser.add_argument("--y-floor", type=float, default=1e-16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = plot_1x256_six_models(
        root=args.root.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir is not None else None,
        datasets=tuple(args.datasets),
        metric_key=str(args.metric_key),
        y_floor=float(args.y_floor),
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
