#!/usr/bin/env python3
"""Plot baseline residual-vs-iteration curves.

This script reads cached baseline JSON results produced by evaluate_baselines.py
and writes residual-vs-iteration figures into:
    <baseline-root>/image/
by default.

Typical usage
-------------
python plot_baseline_residuals.py \
    --baseline-root /path/to/scaled_databaseline_results/baselines_64f719f0bd73

If you instead pass the parent directory (scaled_databaseline_results), the
script will automatically resolve the most recent run from latest.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SOLVER_ORDER = [
    ("full_newton", "Newton"),
    ("gradient_descent", "GD"),
    ("l_bfgs", "L-BFGS"),
]

SKIP_JSON_NAMES = {
    "summary.json",
    "latest.json",
    "gd_selection.json",
    "lbfgs_selection.json",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_run_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        raise ValueError(f"Expected a directory, got file: {path}")
    if (path / "summary.json").exists():
        return path
    latest_path = path / "latest.json"
    if latest_path.exists():
        latest = load_json(latest_path)
        run_dir = Path(latest["path"]).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"latest.json points to missing directory: {run_dir}")
        return run_dir
    candidates = sorted([p for p in path.iterdir() if p.is_dir() and (p / "summary.json").exists()])
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not resolve a baseline run directory from {path}. "
        f"Please pass a specific baselines_xxx directory."
    )


def finite_or_none(x: Any) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def collect_split_jsons(run_dir: Path) -> list[Path]:
    files = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in SKIP_JSON_NAMES:
            continue
        files.append(path)
    return files


def select_solvers(data: dict[str, Any], requested: Iterable[str] | None) -> list[tuple[str, str]]:
    available = [name for name, _label in DEFAULT_SOLVER_ORDER if name in data]
    if requested is None:
        return [(name, label) for name, label in DEFAULT_SOLVER_ORDER if name in available]
    requested = list(requested)
    unknown = [name for name in requested if name not in data]
    if unknown:
        raise ValueError(f"Requested solvers not found in split JSON: {unknown}")
    label_map = dict(DEFAULT_SOLVER_ORDER)
    return [(name, label_map.get(name, name)) for name in requested]


def plot_split(
    *,
    split_name: str,
    split_data: dict[str, Any],
    residual_stat: str,
    yscale: str,
    output_dir: Path,
    solvers: list[tuple[str, str]],
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    plotted = 0
    subtitle_parts: list[str] = []

    for solver_key, solver_label in solvers:
        solver_data = split_data[solver_key]
        key = f"residual_{residual_stat}_by_step"
        if key not in solver_data:
            continue
        ys_raw = solver_data[key]
        xs = list(range(len(ys_raw)))
        ys = []
        for y in ys_raw:
            v = finite_or_none(y)
            if v is None:
                ys.append(float("nan"))
            else:
                ys.append(max(v, 1e-16) if yscale == "log" else v)
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.8, label=solver_label)
        plotted += 1

        final_p95 = finite_or_none(solver_data.get("final_residual_p95"))
        final_max = finite_or_none(solver_data.get("final_residual_max"))
        if final_p95 is not None or final_max is not None:
            text = solver_label + ":"
            if final_p95 is not None:
                text += f" final p95={final_p95:.2e}"
            if final_max is not None:
                text += f", max={final_max:.2e}"
            subtitle_parts.append(text)

    if plotted == 0:
        plt.close(fig)
        raise RuntimeError(f"No solver curves were plotted for split {split_name}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Residual ({residual_stat})")
    ax.set_title(f"Baseline residual vs iteration: {split_name}")
    if yscale == "log":
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()

    if subtitle_parts:
        fig.text(
            0.01,
            0.01,
            " | ".join(subtitle_parts),
            ha="left",
            va="bottom",
            fontsize=9,
            wrap=True,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()

    output_path = output_dir / f"{split_name}_residual_{residual_stat}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_overview(
    *,
    summary: dict[str, Any],
    residual_stat: str,
    yscale: str,
    output_dir: Path,
    solvers: list[tuple[str, str]],
) -> Path:
    split_names = sorted(summary["splits"].keys())
    fig, ax = plt.subplots(figsize=(12, 7))
    x_positions = list(range(len(split_names)))

    for solver_key, solver_label in solvers:
        ys = []
        for split_name in split_names:
            solver_data = summary["splits"][split_name].get(solver_key)
            if solver_data is None:
                ys.append(float("nan"))
                continue
            metric_key = f"final_residual_{residual_stat}"
            value = finite_or_none(solver_data.get(metric_key))
            ys.append(max(value, 1e-16) if (value is not None and yscale == "log") else (value if value is not None else float("nan")))
        ax.plot(x_positions, ys, marker="o", linewidth=1.8, markersize=4.0, label=solver_label)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(split_names, rotation=25, ha="right")
    ax.set_xlabel("Benchmark split")
    ax.set_ylabel(f"Final residual ({residual_stat})")
    ax.set_title(f"Baseline final residual by split ({residual_stat})")
    if yscale == "log":
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / f"overview_final_residual_{residual_stat}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="Path to baselines_xxx or to its parent directory scaled_databaseline_results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for images. Default: <resolved-baseline-run>/../image",
    )
    parser.add_argument(
        "--residual-stat",
        choices=["mean", "median", "p95", "max"],
        default="p95",
        help="Which residual statistic to plot against iteration.",
    )
    parser.add_argument(
        "--yscale",
        choices=["log", "linear"],
        default="log",
        help="Y-axis scale for residual curves.",
    )
    parser.add_argument(
        "--solvers",
        nargs="*",
        default=None,
        help="Optional subset of solvers: full_newton gradient_descent l_bfgs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.baseline_root)
    summary = load_json(run_dir / "summary.json")

    output_dir = args.output_dir.resolve() if args.output_dir else (run_dir.parent / "image")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_jsons = collect_split_jsons(run_dir)
    if not split_jsons:
        raise FileNotFoundError(f"No split JSON files found in {run_dir}")

    # Determine solver set from the first split.
    first_split = load_json(split_jsons[0])
    solvers = select_solvers(first_split, args.solvers)

    generated: list[Path] = []
    for split_json in split_jsons:
        split_name = split_json.stem
        split_data = load_json(split_json)
        out_path = plot_split(
            split_name=split_name,
            split_data=split_data,
            residual_stat=args.residual_stat,
            yscale=args.yscale,
            output_dir=output_dir,
            solvers=solvers,
        )
        generated.append(out_path)

    overview_path = plot_overview(
        summary=summary,
        residual_stat=args.residual_stat,
        yscale=args.yscale,
        output_dir=output_dir,
        solvers=solvers,
    )
    generated.append(overview_path)

    print(f"Baseline run: {run_dir}")
    print(f"Image output: {output_dir}")
    print("Generated figures:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
