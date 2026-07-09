"""Script 6: plot summary comparisons between baselines and learned models.

Inputs:
    baselines/baseline_metrics.json
    models/*/test_metrics.json

Outputs:
    results/summary_figures/*.png
    results/summary_tables/*.csv

Run:
    python cloth06_plot_summary.py --root cloth_5x5_500step_pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEST_DATASETS = ("validation", "seen_extrap", "unseen_id", "ood")
METRICS = ("final_residual_mean", "final_residual_max", "final_residual_sum")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def collect_baseline_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "baselines" / "baseline_metrics.json"
    if not path.exists():
        return []
    metrics = load_json(path)
    rows: list[dict[str, Any]] = []
    for dataset_name, solver_records in metrics.items():
        for solver_name, record in solver_records.items():
            row = {
                "kind": "baseline",
                "method": solver_name,
                "dataset": dataset_name,
            }
            for metric in METRICS:
                row[metric] = float(record.get(metric, np.nan))
            rows.append(row)
    return rows


def collect_model_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_root = root / "models"
    if not model_root.exists():
        return rows
    for model_dir in sorted(p for p in model_root.iterdir() if p.is_dir()):
        metrics_path = model_dir / "test_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        for dataset_name, record in metrics.items():
            row = {
                "kind": "learned",
                "method": model_dir.name,
                "dataset": dataset_name,
            }
            for metric in METRICS:
                row[metric] = float(record.get(metric, np.nan))
            rows.append(row)
    return rows


def plot_metric_bar(rows: list[dict[str, Any]], dataset_name: str, metric: str, figure_dir: Path) -> None:
    selected = [row for row in rows if row["dataset"] == dataset_name]
    if not selected:
        return
    labels = [f"{row['kind']}:{row['method']}" for row in selected]
    values = np.asarray([float(row[metric]) for row in selected], dtype=float)
    values = np.maximum(values, 1e-16)

    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(labels)), 4.5))
    ax.bar(np.arange(len(labels)), values)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"{dataset_name}: {metric}")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{dataset_name}_{metric}_bar.png", dpi=180)
    plt.close(fig)


def plot_iteration_curves(root: Path, dataset_name: str, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plotted = False

    baseline_metrics_path = root / "baselines" / "baseline_metrics.json"
    if baseline_metrics_path.exists():
        baseline_metrics = load_json(baseline_metrics_path)
        if dataset_name in baseline_metrics:
            for solver_name, record in baseline_metrics[dataset_name].items():
                y = np.asarray(record.get("residual_mean_by_iter", []), dtype=float)
                if y.size:
                    ax.plot(np.arange(y.size), np.maximum(y, 1e-16), label=f"baseline:{solver_name}")
                    plotted = True

    model_root = root / "models"
    if model_root.exists():
        for model_dir in sorted(p for p in model_root.iterdir() if p.is_dir()):
            metrics_path = model_dir / "test_metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            if dataset_name not in metrics:
                continue
            y = np.asarray(metrics[dataset_name].get("residual_mean_by_iter", []), dtype=float)
            if y.size:
                ax.plot(np.arange(y.size), np.maximum(y, 1e-16), label=f"learned:{model_dir.name}")
                plotted = True

    if not plotted:
        plt.close(fig)
        return
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean residual")
    ax.set_title(f"{dataset_name}: iteration vs residual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{dataset_name}_iteration_vs_residual_all.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot baseline/model summary comparisons.")
    parser.add_argument("--root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--datasets", nargs="+", default=list(TEST_DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figure_dir = args.root / "results" / "summary_figures"
    table_dir = args.root / "results" / "summary_tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_baseline_rows(args.root) + collect_model_rows(args.root)
    write_csv(rows, table_dir / "summary_metrics.csv")
    if not rows:
        print("no baseline/model metrics found")
        return

    for dataset_name in args.datasets:
        for metric in METRICS:
            plot_metric_bar(rows, dataset_name, metric, figure_dir)
        plot_iteration_curves(args.root, dataset_name, figure_dir)

    print(f"saved summary figures to {figure_dir}")
    print(f"saved summary table to {table_dir / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
