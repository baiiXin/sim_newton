"""Plot iteration-vs-residual curves from saved scale-time-step evaluation.

Input:
    old_500step_vs_nonlinear_test_eval/evaluation_metrics.json

Outputs:
    old_500step_vs_nonlinear_test_eval/figures/<split>_iteration_vs_residual_*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "old_500step_vs_nonlinear_test_eval"
PLOT_FLOOR = 1e-16

FAMILY_LABELS = {
    "old_500step_full_state": "500-step old full-state",
    "nonlinear_history_input_default_init": "nonlinear default-init",
}
FAMILY_STYLES = {
    "old_500step_full_state": "-",
    "nonlinear_history_input_default_init": "--",
}
ACTIVATION_COLORS = {
    "identity": "tab:blue",
    "relu": "tab:orange",
    "tanh": "tab:green",
}
STAT_LABELS = {
    "mean": "mean residual",
    "p95": "p95 residual",
    "max": "max residual",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_curve(values: list[Any]) -> np.ndarray:
    curve = np.asarray(values, dtype=float)
    curve[~np.isfinite(curve)] = np.nan
    return np.maximum(curve, PLOT_FLOOR)


def split_title(split_name: str) -> str:
    return split_name.replace("_", " ")


def plot_split_stat(
    *,
    data: dict[str, Any],
    split_name: str,
    stat: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for family_name, activations in data["results"].items():
        for activation, split_metrics in activations.items():
            if split_name not in split_metrics:
                continue
            metrics = split_metrics[split_name]
            key = f"residual_{stat}_by_step"
            if key not in metrics:
                continue
            y = finite_curve(metrics[key])
            x = np.arange(len(y), dtype=int)
            label = f"{activation} | {FAMILY_LABELS.get(family_name, family_name)}"
            ax.plot(
                x,
                y,
                linestyle=FAMILY_STYLES.get(family_name, "-"),
                color=ACTIVATION_COLORS.get(activation),
                linewidth=1.9,
                alpha=0.95,
                label=label,
            )

    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(STAT_LABELS[stat])
    ax.set_title(f"{split_title(split_name)}: iteration vs {STAT_LABELS[stat]}")
    ax.grid(True, alpha=0.28, which="both")
    ax.legend(fontsize=8.5, ncol=1)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_split_combined(
    *,
    data: dict[str, Any],
    split_name: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharex=True)
    for ax, stat in zip(axes, ["mean", "p95", "max"], strict=True):
        for family_name, activations in data["results"].items():
            for activation, split_metrics in activations.items():
                if split_name not in split_metrics:
                    continue
                metrics = split_metrics[split_name]
                key = f"residual_{stat}_by_step"
                if key not in metrics:
                    continue
                y = finite_curve(metrics[key])
                x = np.arange(len(y), dtype=int)
                label = f"{activation} | {FAMILY_LABELS.get(family_name, family_name)}"
                ax.plot(
                    x,
                    y,
                    linestyle=FAMILY_STYLES.get(family_name, "-"),
                    color=ACTIVATION_COLORS.get(activation),
                    linewidth=1.7,
                    alpha=0.95,
                    label=label,
                )
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel(STAT_LABELS[stat])
        ax.set_title(STAT_LABELS[stat])
        ax.grid(True, alpha=0.28, which="both")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5)
    fig.suptitle(f"{split_title(split_name)}: iteration vs residual", y=0.98)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def per_motion_values(metrics: dict[str, Any], stat: str) -> tuple[np.ndarray, np.ndarray]:
    records = metrics.get("per_motion", {})
    motions = sorted(int(key) for key in records)
    values = [
        records[str(motion)]["final"]["residual"][stat]
        for motion in motions
    ]
    return np.asarray(motions, dtype=int), finite_curve(values)


def plot_per_motion_final_residual(
    *,
    data: dict[str, Any],
    split_name: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), sharex=True)
    for ax, stat in zip(axes, ["p95", "max"], strict=True):
        for family_name, activations in data["results"].items():
            for activation, split_metrics in activations.items():
                if split_name not in split_metrics:
                    continue
                motions, y = per_motion_values(split_metrics[split_name], stat)
                label = f"{activation} | {FAMILY_LABELS.get(family_name, family_name)}"
                ax.plot(
                    motions,
                    y,
                    marker="o",
                    markersize=4,
                    linestyle=FAMILY_STYLES.get(family_name, "-"),
                    color=ACTIVATION_COLORS.get(activation),
                    linewidth=1.7,
                    alpha=0.95,
                    label=label,
                )
        ax.set_yscale("log")
        ax.set_xlabel("motion index")
        ax.set_ylabel(f"final residual {stat}")
        ax.set_title(f"per-motion residual {stat}")
        ax.grid(True, alpha=0.28, which="both")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5)
    fig.suptitle(f"{split_title(split_name)}: per-motion final residual", y=0.98)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_index(generated: list[Path], output_path: Path) -> None:
    lines = ["# Iteration-vs-residual figures", ""]
    for path in generated:
        lines.append(f"- `{path.name}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot iteration-vs-residual curves.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--metrics-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stats", nargs="+", default=["mean", "p95", "max"], choices=["mean", "p95", "max"])
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--no-combined", action="store_true")
    parser.add_argument("--no-per-motion", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics_file or args.result_dir / "evaluation_metrics.json"
    output_dir = args.output_dir or args.result_dir / "figures"
    data = load_json(metrics_path)

    available_splits = data.get("settings", {}).get("splits") or []
    if not available_splits:
        for activations in data["results"].values():
            for split_metrics in activations.values():
                available_splits = list(split_metrics.keys())
                break
            if available_splits:
                break
    splits = args.splits or available_splits

    generated: list[Path] = []
    for split_name in splits:
        for stat in args.stats:
            path = output_dir / f"{split_name}_iteration_vs_residual_{stat}.png"
            plot_split_stat(data=data, split_name=split_name, stat=stat, output_path=path)
            generated.append(path)
        if not args.no_combined:
            path = output_dir / f"{split_name}_iteration_vs_residual_all_stats.png"
            plot_split_combined(data=data, split_name=split_name, output_path=path)
            generated.append(path)
        if not args.no_per_motion:
            path = output_dir / f"{split_name}_per_motion_final_residual.png"
            plot_per_motion_final_residual(data=data, split_name=split_name, output_path=path)
            generated.append(path)

    write_index(generated, output_dir / "README_iteration_residual_figures.md")
    print(f"saved {len(generated)} figures to {output_dir}")


if __name__ == "__main__":
    main()
