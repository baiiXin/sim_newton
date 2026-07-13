"""Plot lightweight initial-state thumbnails for scale-up scenario catalogues."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from cloth02_batched_physics import build_batched_parameters
from scenario_catalogue import build_catalogues
from scenario_templates import (
    BOUNDARY_BY_ID,
    DIRICHLET_BY_ID,
    MATERIAL_BY_ID,
    ORIENTATION_BY_ID,
    SHAPE_BY_ID,
    STRAIN_BY_ID,
    VELOCITY_BY_ID,
    ScenarioSpec,
)


DEFAULT_ROOT = Path("cloth_15x15_scale_up_pipeline")
DEFAULT_DATASETS = (
    "train_c1_1024",
    "train_c2_2048",
    "train_c3_3072",
    "validation_128",
    "test_256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render low-memory initial-state figures for scenario catalogues."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="catalogue keys to plot, or 'all'",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--figsize", type=float, nargs=2, default=(5.0, 4.0))
    parser.add_argument("--velocity-stride", type=int, default=4)
    parser.add_argument("--no-edges", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace, keys: Sequence[str]) -> list[str]:
    requested = list(args.datasets)
    if requested == ["all"] or "all" in requested:
        return list(DEFAULT_DATASETS)
    unknown = sorted(set(requested) - set(keys))
    if unknown:
        raise ValueError(f"unknown dataset keys: {unknown}; available={list(keys)}")
    return requested


def scenario_labels(scenario: ScenarioSpec) -> dict[str, str]:
    return {
        "shape": SHAPE_BY_ID[scenario.shape_id].name,
        "strain": STRAIN_BY_ID[scenario.strain_id].name,
        "velocity": VELOCITY_BY_ID[scenario.velocity_id].name,
        "boundary": BOUNDARY_BY_ID[scenario.boundary_id].name,
        "dirichlet": DIRICHLET_BY_ID[scenario.dirichlet_id].name,
        "material": MATERIAL_BY_ID[scenario.material_id].name,
        "orientation": ORIENTATION_BY_ID[scenario.orientation_id].name,
    }


def scenario_row(index: int, scenario: ScenarioSpec, output: Path) -> dict[str, Any]:
    labels = scenario_labels(scenario)
    return {
        "index": int(index),
        "scenario_id": int(scenario.scenario_id),
        "split": scenario.split,
        "group": scenario.group,
        "difficulty": scenario.difficulty,
        "shape_id": scenario.shape_id,
        "shape": labels["shape"],
        "strain_id": scenario.strain_id,
        "strain": labels["strain"],
        "velocity_id": scenario.velocity_id,
        "velocity": labels["velocity"],
        "boundary_id": scenario.boundary_id,
        "boundary": labels["boundary"],
        "dirichlet_id": scenario.dirichlet_id,
        "dirichlet": labels["dirichlet"],
        "material_id": scenario.material_id,
        "material": labels["material"],
        "orientation_id": scenario.orientation_id,
        "orientation": labels["orientation"],
        "figure": str(output),
    }


def axis_box(points: np.ndarray) -> tuple[np.ndarray, float]:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) * 0.58, 1e-3)
    return center, radius


def write_index(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_initial_state(
    *,
    motion_index: int,
    scenario: ScenarioSpec,
    output: Path,
    dtype: torch.dtype,
    dpi: int,
    figsize: tuple[float, float],
    velocity_stride: int,
    draw_edges: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    params = build_batched_parameters((scenario,), dtype=dtype, device="cpu")
    positions = params.initial_positions[0].detach().cpu().numpy()
    velocities = params.initial_velocities[0].detach().cpu().numpy()
    fixed = params.fixed_mask[0].detach().cpu().numpy().astype(bool)
    edges = np.asarray(params.topology.edges, dtype=np.int64)
    center, radius = axis_box(positions)

    fig = plt.figure(figsize=figsize)
    axis = fig.add_subplot(111, projection="3d")
    if draw_edges:
        segments = positions[edges]
        collection = Line3DCollection(
            segments,
            colors="0.72",
            linewidths=0.35,
            alpha=0.75,
        )
        axis.add_collection3d(collection)
    free = ~fixed
    axis.scatter(
        positions[free, 0],
        positions[free, 1],
        positions[free, 2],
        s=5,
        c="tab:blue",
        depthshade=False,
    )
    axis.scatter(
        positions[fixed, 0],
        positions[fixed, 1],
        positions[fixed, 2],
        s=28,
        c="tab:red",
        marker="s",
        depthshade=False,
    )

    speed = np.linalg.norm(velocities, axis=1)
    moving = np.flatnonzero(speed > 1e-12)
    if moving.size and velocity_stride > 0:
        selected = moving[::velocity_stride]
        axis.quiver(
            positions[selected, 0],
            positions[selected, 1],
            positions[selected, 2],
            velocities[selected, 0],
            velocities[selected, 1],
            velocities[selected, 2],
            color="tab:green",
            length=radius * 0.20,
            normalize=True,
            linewidth=0.7,
        )

    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("x", labelpad=-6)
    axis.set_ylabel("y", labelpad=-6)
    axis.set_zlabel("z", labelpad=-6)
    axis.tick_params(axis="both", which="major", labelsize=6, pad=-2)
    axis.view_init(elev=25, azim=-55)
    axis.set_title(
        f"idx={motion_index} sid={scenario.scenario_id} {scenario.group}",
        fontsize=8,
        pad=4,
    )
    info = "\n".join(
        (
            f"shape: {scenario.shape_id}",
            f"strain: {scenario.strain_id}",
            f"velocity: {scenario.velocity_id}",
            f"fixed: {scenario.boundary_id}",
            f"dirichlet: {scenario.dirichlet_id}",
            f"material: {scenario.material_id}",
            f"orientation: {scenario.orientation_id}",
            f"fixed vertices: {int(fixed.sum())}",
        )
    )
    axis.text2D(
        0.02,
        0.98,
        info,
        transform=axis.transAxes,
        va="top",
        fontsize=6,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.72, "linewidth": 0.2},
    )
    fig.tight_layout(pad=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def render_dataset(
    *,
    key: str,
    scenarios: Sequence[ScenarioSpec],
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[int, int]:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    start = max(0, int(args.start_index))
    stop = len(scenarios)
    if args.max_count is not None:
        stop = min(stop, start + max(0, int(args.max_count)))
    rows: list[dict[str, Any]] = []
    rendered = 0
    skipped = 0
    dataset_dir = output_root / key
    for index in range(start, stop):
        scenario = scenarios[index]
        output = dataset_dir / f"motion_{index:04d}_scenario_{scenario.scenario_id}.png"
        rows.append(scenario_row(index, scenario, output))
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        plot_initial_state(
            motion_index=index,
            scenario=scenario,
            output=output,
            dtype=dtype,
            dpi=args.dpi,
            figsize=(float(args.figsize[0]), float(args.figsize[1])),
            velocity_stride=int(args.velocity_stride),
            draw_edges=not args.no_edges,
        )
        rendered += 1
        if args.progress_interval > 0 and rendered % args.progress_interval == 0:
            print(f"{key}: rendered={rendered} skipped={skipped} latest={output}")
    write_index(dataset_dir / "index.csv", rows)
    return rendered, skipped


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.figsize[0] <= 0 or args.figsize[1] <= 0:
        raise ValueError("--figsize values must be positive")
    if args.velocity_stride < 0:
        raise ValueError("--velocity-stride must be non-negative")
    catalogues = build_catalogues()
    keys = selected_datasets(args, tuple(catalogues.keys()))
    output_root = args.output_dir or args.root / "data" / "figure"
    total_rendered = 0
    total_skipped = 0
    for key in keys:
        rendered, skipped = render_dataset(
            key=key,
            scenarios=tuple(catalogues[key]),
            output_root=output_root,
            args=args,
        )
        total_rendered += rendered
        total_skipped += skipped
        print(f"{key}: rendered={rendered} skipped={skipped} output={output_root / key}")
    print(f"initial-state figures done: rendered={total_rendered} skipped={total_skipped}")


if __name__ == "__main__":
    main()
