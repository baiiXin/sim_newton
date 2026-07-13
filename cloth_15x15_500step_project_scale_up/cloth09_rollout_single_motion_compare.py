"""Roll out one scale-up scenario and render learned optimizer vs. a baseline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from cloth02_batched_physics import (
    advance_state,
    build_batched_parameters,
    dirichlet_targets,
    flatten_positions,
    free_update_gate,
    make_q,
    project_positions,
    spring_lengths,
    stationarity_residual,
    stationarity_residual_norm,
    variational_energy,
)
from cloth03_training_pool import LearnedOptimizerMLP, ModelSpec, apply_model_update
from cloth04_reference_free_validation import FailureThresholds
from cloth07_evaluate_best_checkpoint import (
    checkpoint_path,
    resolve_dtype,
    run_directory,
    write_json,
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one reference-free scale-up rollout with a trained model and "
            "compare it to a mass-preconditioned gradient baseline."
        )
    )
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
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-update", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float64", "float32"), default="auto")
    parser.add_argument(
        "--split",
        choices=("validation", "test", "train"),
        default="test",
        help="scenario split used for the selected motion index",
    )
    parser.add_argument(
        "--list-motions",
        action="store_true",
        help="list available motion indices for --split and exit",
    )
    parser.add_argument("--list-offset", type=int, default=0)
    parser.add_argument("--list-limit", type=int, default=32)
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--rollout-frames", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--baseline-step-size", type=float, default=1.0)
    parser.add_argument("--baseline-line-search-reductions", type=int, default=12)
    parser.add_argument("--fixed-gd-step-size", type=float, default=5e-5)
    parser.add_argument("--render-format", choices=("mp4", "gif", "none"), default="mp4")
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-residual", type=float, default=1e12)
    parser.add_argument("--max-abs-position", type=float, default=1e4)
    parser.add_argument("--min-edge-ratio", type=float, default=1e-5)
    parser.add_argument("--max-edge-ratio", type=float, default=1e4)
    parser.add_argument("--max-constraint-error", type=float, default=1e-9)
    return parser.parse_args()


def split_key(split: str, catalogue: str) -> str:
    if split == "validation":
        return "validation_128"
    if split == "test":
        return "test_256"
    return {
        "c1": "train_c1_1024",
        "c2": "train_c2_2048",
        "c3": "train_c3_3072",
    }[catalogue]


def selected_scenario(args: argparse.Namespace) -> ScenarioSpec:
    scenarios, _ = selected_scenarios(args)
    if args.motion_index < 0 or args.motion_index >= len(scenarios):
        raise ValueError(
            f"--motion-index must be in [0, {len(scenarios) - 1}] for "
            f"{split_key(args.split, args.catalogue)}"
        )
    return scenarios[args.motion_index]


def selected_scenarios(args: argparse.Namespace) -> tuple[tuple[ScenarioSpec, ...], str]:
    catalogues = build_catalogues()
    key = split_key(args.split, args.catalogue)
    scenarios = tuple(catalogues[key])
    if not scenarios:
        raise ValueError(f"empty scenario split: {key}")
    return scenarios, key


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


def scenario_row(index: int, scenario: ScenarioSpec) -> dict[str, Any]:
    labels = scenario_labels(scenario)
    return {
        "index": index,
        "scenario_id": scenario.scenario_id,
        "group": scenario.group,
        "difficulty": scenario.difficulty,
        "shape": f"{scenario.shape_id}:{labels['shape']}",
        "strain": f"{scenario.strain_id}:{labels['strain']}",
        "velocity": f"{scenario.velocity_id}:{labels['velocity']}",
        "boundary": f"{scenario.boundary_id}:{labels['boundary']}",
        "dirichlet": f"{scenario.dirichlet_id}:{labels['dirichlet']}",
        "material": f"{scenario.material_id}:{labels['material']}",
        "orientation": f"{scenario.orientation_id}:{labels['orientation']}",
    }


def print_motion_table(args: argparse.Namespace) -> None:
    scenarios, key = selected_scenarios(args)
    if args.list_limit <= 0:
        raise ValueError("--list-limit must be positive")
    start = max(0, int(args.list_offset))
    end = min(len(scenarios), start + int(args.list_limit))
    print(f"catalogue={key} count={len(scenarios)} showing=[{start}, {end})")
    headers = [
        "index",
        "scenario_id",
        "group",
        "difficulty",
        "boundary",
        "dirichlet",
        "material",
        "velocity",
        "shape",
        "strain",
        "orientation",
    ]
    rows = [scenario_row(index, scenarios[index]) for index in range(start, end)]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    } if rows else {header: len(header) for header in headers}
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def default_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    bias = "bias" if args.use_bias else "no_bias"
    name = (
        f"{args.split}_motion_{args.motion_index:04d}_"
        f"k{args.inner_steps:03d}_{bias}"
    )
    return run_dir / "single_motion_rollouts" / name


def _finite_float(value: torch.Tensor, default: float = float("nan")) -> float:
    scalar = float(value.detach().cpu().item())
    return scalar if math.isfinite(scalar) else float(default)


def frame_diagnostics(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    params,
    targets: torch.Tensor,
    thresholds: FailureThresholds,
) -> dict[str, Any]:
    residual = stationarity_residual_norm(y, q, params, targets)
    energy = variational_energy(y, q, params, targets)
    lengths = spring_lengths(y, params, targets)
    ratios = lengths / params.rest_lengths.clamp_min(torch.finfo(params.dtype).eps)
    projected = project_positions(y, params, targets)
    constraint = torch.where(
        params.fixed_mask.unsqueeze(-1),
        (projected - targets).abs(),
        torch.zeros_like(projected),
    ).amax(dim=(-2, -1))
    finite = (
        torch.isfinite(y).flatten(start_dim=1).all(dim=1)
        & torch.isfinite(residual)
        & torch.isfinite(energy)
        & torch.isfinite(lengths).all(dim=-1)
    )
    edge_min = ratios.amin(dim=-1)
    edge_max = ratios.amax(dim=-1)
    failed = (
        ~finite
        | (residual > thresholds.max_residual)
        | (y.abs().amax(dim=(-2, -1)) > thresholds.max_abs_position)
        | (edge_min < thresholds.min_edge_ratio)
        | (edge_max > thresholds.max_edge_ratio)
        | (constraint > thresholds.max_constraint_error)
    )
    return {
        "residual": _finite_float(residual[0]),
        "energy": _finite_float(energy[0]),
        "edge_ratio_min": _finite_float(edge_min[0]),
        "edge_ratio_max": _finite_float(edge_max[0]),
        "constraint_error": _finite_float(constraint[0]),
        "failed": bool(failed[0].item()),
    }


@torch.no_grad()
def baseline_step(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    params,
    targets: torch.Tensor,
    step_size: float,
    max_reductions: int,
) -> tuple[torch.Tensor, bool, float]:
    energy = variational_energy(y, q, params, targets)
    residual = stationarity_residual(y, q, params, targets)
    residual_points = residual.reshape(params.batch_size, params.num_vertices, 3)
    preconditioned = (
        params.dt**2
        * residual_points
        / params.masses.clamp_min(torch.finfo(params.dtype).tiny).unsqueeze(-1)
    )
    direction = -flatten_positions(
        preconditioned * free_update_gate(params).to(params.dtype)
    )
    y_flat = y.reshape(params.batch_size, -1)
    scale = float(step_size)
    for _ in range(max(0, max_reductions) + 1):
        candidate = project_positions(y_flat + scale * direction, params, targets)
        candidate_energy = variational_energy(candidate, q, params, targets)
        accepted = (
            torch.isfinite(candidate).flatten(start_dim=1).all(dim=1)
            & torch.isfinite(candidate_energy)
            & (candidate_energy <= energy)
        )
        if bool(accepted[0].item()):
            return candidate.reshape_as(y), True, scale
        scale *= 0.5
    return y, False, 0.0


@torch.no_grad()
def fixed_gradient_descent_step(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    params,
    targets: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    residual = stationarity_residual(y, q, params, targets)
    direction = -residual.reshape(params.batch_size, -1)
    direction = direction * free_update_gate(params, flattened=True).to(params.dtype)
    y_flat = y.reshape(params.batch_size, -1)
    return project_positions(y_flat + float(step_size) * direction, params, targets)


@torch.no_grad()
def run_model_rollout(
    *,
    model: LearnedOptimizerMLP,
    scenario: ScenarioSpec,
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
) -> dict[str, Any]:
    params = build_batched_parameters((scenario,), device=device, dtype=dtype)
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    positions = [p[0].detach().cpu()]
    residual_curve = []
    frame_rows: list[dict[str, Any]] = []
    failure_frame: int | None = None

    for frame in range(rollout_frames):
        if failure_frame is not None:
            positions.append(p[0].detach().cpu())
            residual_curve.append([float("nan")] * (inner_steps + 1))
            frame_rows.append({"frame": frame, "failed": True})
            continue

        q = make_q(p, v, params)
        next_time = torch.full((1,), (frame + 1) * params.dt, dtype=dtype, device=device)
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(p, params, targets)
        previous_residual = torch.zeros(1, params.full_state_dim, dtype=dtype, device=device)
        previous_update = torch.zeros_like(previous_residual)
        curve = [frame_diagnostics(y=y, q=q, params=params, targets=targets, thresholds=thresholds)["residual"]]
        energy_before = variational_energy(y, q, params, targets)

        for _ in range(inner_steps):
            y_next, delta, current = apply_model_update(
                model,
                y,
                q,
                params,
                target_positions=targets,
                previous_residual=previous_residual,
                previous_update=previous_update,
            )
            y = y_next.reshape_as(y)
            previous_residual = current
            previous_update = delta
            curve.append(
                frame_diagnostics(
                    y=y,
                    q=q,
                    params=params,
                    targets=targets,
                    thresholds=thresholds,
                )["residual"]
            )

        diagnostics = frame_diagnostics(
            y=y,
            q=q,
            params=params,
            targets=targets,
            thresholds=thresholds,
        )
        energy_after = variational_energy(y, q, params, targets)
        diagnostics.update(
            {
                "frame": frame,
                "initial_residual": curve[0],
                "final_residual": curve[-1],
                "residual_ratio": curve[-1] / max(curve[0], torch.finfo(dtype).eps),
                "energy_before": _finite_float(energy_before[0]),
                "energy_after": _finite_float(energy_after[0]),
            }
        )
        frame_rows.append(diagnostics)
        residual_curve.append(curve)
        if diagnostics["failed"]:
            failure_frame = frame
            positions.append(p[0].detach().cpu())
            continue
        p, v = advance_state(p, y, params, next_time=next_time)
        positions.append(p[0].detach().cpu())

    return {
        "solver": "learned",
        "positions": torch.stack(positions, dim=0),
        "residual_by_frame_and_iteration": torch.tensor(residual_curve, dtype=torch.float64),
        "frames": frame_rows,
        "failure_frame": failure_frame,
        "line_search_failures": 0,
    }


@torch.no_grad()
def run_baseline_rollout(
    *,
    scenario: ScenarioSpec,
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
    step_size: float,
    max_reductions: int,
) -> dict[str, Any]:
    params = build_batched_parameters((scenario,), device=device, dtype=dtype)
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    positions = [p[0].detach().cpu()]
    residual_curve = []
    frame_rows: list[dict[str, Any]] = []
    failure_frame: int | None = None
    line_search_failures = 0

    for frame in range(rollout_frames):
        if failure_frame is not None:
            positions.append(p[0].detach().cpu())
            residual_curve.append([float("nan")] * (inner_steps + 1))
            frame_rows.append({"frame": frame, "failed": True})
            continue

        q = make_q(p, v, params)
        next_time = torch.full((1,), (frame + 1) * params.dt, dtype=dtype, device=device)
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(p, params, targets)
        curve = [frame_diagnostics(y=y, q=q, params=params, targets=targets, thresholds=thresholds)["residual"]]
        energy_before = variational_energy(y, q, params, targets)
        accepted_scales: list[float] = []

        for _ in range(inner_steps):
            y, accepted, accepted_scale = baseline_step(
                y=y,
                q=q,
                params=params,
                targets=targets,
                step_size=step_size,
                max_reductions=max_reductions,
            )
            if not accepted:
                line_search_failures += 1
            accepted_scales.append(float(accepted_scale))
            curve.append(
                frame_diagnostics(
                    y=y,
                    q=q,
                    params=params,
                    targets=targets,
                    thresholds=thresholds,
                )["residual"]
            )

        diagnostics = frame_diagnostics(
            y=y,
            q=q,
            params=params,
            targets=targets,
            thresholds=thresholds,
        )
        energy_after = variational_energy(y, q, params, targets)
        nonzero_scales = [value for value in accepted_scales if value > 0.0]
        diagnostics.update(
            {
                "frame": frame,
                "initial_residual": curve[0],
                "final_residual": curve[-1],
                "residual_ratio": curve[-1] / max(curve[0], torch.finfo(dtype).eps),
                "energy_before": _finite_float(energy_before[0]),
                "energy_after": _finite_float(energy_after[0]),
                "accepted_step_min": min(nonzero_scales) if nonzero_scales else 0.0,
                "accepted_step_median": (
                    sorted(nonzero_scales)[len(nonzero_scales) // 2]
                    if nonzero_scales
                    else 0.0
                ),
            }
        )
        frame_rows.append(diagnostics)
        residual_curve.append(curve)
        if diagnostics["failed"]:
            failure_frame = frame
            positions.append(p[0].detach().cpu())
            continue
        p, v = advance_state(p, y, params, next_time=next_time)
        positions.append(p[0].detach().cpu())

    return {
        "solver": "mass_preconditioned_gd",
        "positions": torch.stack(positions, dim=0),
        "residual_by_frame_and_iteration": torch.tensor(residual_curve, dtype=torch.float64),
        "frames": frame_rows,
        "failure_frame": failure_frame,
        "line_search_failures": line_search_failures,
    }


@torch.no_grad()
def run_fixed_gd_rollout(
    *,
    scenario: ScenarioSpec,
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
    step_size: float,
) -> dict[str, Any]:
    params = build_batched_parameters((scenario,), device=device, dtype=dtype)
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    positions = [p[0].detach().cpu()]
    residual_curve = []
    frame_rows: list[dict[str, Any]] = []
    failure_frame: int | None = None

    for frame in range(rollout_frames):
        if failure_frame is not None:
            positions.append(p[0].detach().cpu())
            residual_curve.append([float("nan")] * (inner_steps + 1))
            frame_rows.append({"frame": frame, "failed": True})
            continue

        q = make_q(p, v, params)
        next_time = torch.full((1,), (frame + 1) * params.dt, dtype=dtype, device=device)
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(p, params, targets)
        curve = [
            frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
        ]
        energy_before = variational_energy(y, q, params, targets)

        for _ in range(inner_steps):
            y = fixed_gradient_descent_step(
                y=y,
                q=q,
                params=params,
                targets=targets,
                step_size=step_size,
            ).reshape_as(y)
            curve.append(
                frame_diagnostics(
                    y=y,
                    q=q,
                    params=params,
                    targets=targets,
                    thresholds=thresholds,
                )["residual"]
            )

        diagnostics = frame_diagnostics(
            y=y,
            q=q,
            params=params,
            targets=targets,
            thresholds=thresholds,
        )
        energy_after = variational_energy(y, q, params, targets)
        diagnostics.update(
            {
                "frame": frame,
                "initial_residual": curve[0],
                "final_residual": curve[-1],
                "residual_ratio": curve[-1] / max(curve[0], torch.finfo(dtype).eps),
                "energy_before": _finite_float(energy_before[0]),
                "energy_after": _finite_float(energy_after[0]),
                "fixed_step_size": float(step_size),
            }
        )
        frame_rows.append(diagnostics)
        residual_curve.append(curve)
        if diagnostics["failed"]:
            failure_frame = frame
            positions.append(p[0].detach().cpu())
            continue
        p, v = advance_state(p, y, params, next_time=next_time)
        positions.append(p[0].detach().cpu())

    return {
        "solver": "gd_fixed_lr_5e-5" if step_size == 5e-5 else f"gd_fixed_lr_{step_size:g}",
        "positions": torch.stack(positions, dim=0),
        "residual_by_frame_and_iteration": torch.tensor(residual_curve, dtype=torch.float64),
        "frames": frame_rows,
        "failure_frame": failure_frame,
        "line_search_failures": 0,
        "fixed_step_size": float(step_size),
    }


def finite_mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def summarize_rollout(result: dict[str, Any], rollout_frames: int) -> dict[str, Any]:
    frame_rows = result["frames"]
    valid = [row for row in frame_rows if not bool(row.get("failed", False))]
    final_residual = [float(row.get("final_residual", float("nan"))) for row in valid]
    ratios = [float(row.get("residual_ratio", float("nan"))) for row in valid]
    energy_increases = [
        float(row["energy_after"]) > float(row["energy_before"])
        for row in valid
        if math.isfinite(float(row.get("energy_after", float("nan"))))
        and math.isfinite(float(row.get("energy_before", float("nan"))))
    ]
    failure_frame = result["failure_frame"]
    survived = rollout_frames if failure_frame is None else int(failure_frame)
    return {
        "solver": result["solver"],
        "rollout_frames": rollout_frames,
        "survival_frames": survived,
        "failed": failure_frame is not None,
        "failure_frame": failure_frame,
        "final_residual_mean": finite_mean(final_residual),
        "residual_ratio_mean": finite_mean(ratios),
        "energy_increase_fraction": (
            float(sum(energy_increases) / len(energy_increases))
            if energy_increases
            else None
        ),
        "line_search_failures": int(result.get("line_search_failures", 0)),
    }


def write_frame_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    import csv

    rows = []
    for result in results:
        for row in result["frames"]:
            rows.append({"solver": result["solver"], **row})
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


def plot_diagnostics(output: Path, results: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")
    for index, result in enumerate(results):
        frames = [int(row["frame"]) for row in result["frames"]]
        residual = [float(row.get("final_residual", float("nan"))) for row in result["frames"]]
        ratio = [float(row.get("residual_ratio", float("nan"))) for row in result["frames"]]
        color = colors[index % len(colors)]
        axes[0].plot(frames, residual, label=result["solver"], color=color)
        axes[1].plot(frames, ratio, label=result["solver"], color=color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("final residual")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("residual ratio")
    axes[1].set_xlabel("physical frame")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def axis_box(*arrays) -> tuple[Any, float]:
    import numpy as np

    valid = [array.reshape(-1, 3) for array in arrays if array.size]
    stacked = np.concatenate(valid, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) * 0.58, 1e-3)
    return center, radius


def render_comparison(
    *,
    output: Path,
    results: Sequence[dict[str, Any]],
    scenario: ScenarioSpec,
    fps: int,
    stride: int,
    format_name: Literal["mp4", "gif"],
) -> Path:
    if stride <= 0:
        raise ValueError("--render-stride must be positive")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    import numpy as np

    params = build_batched_parameters((scenario,), device="cpu", dtype=torch.float64)
    edges = [tuple(edge) for edge in params.topology.edges]
    fixed = torch.nonzero(params.fixed_mask[0], as_tuple=False).flatten().tolist()
    if not results:
        raise ValueError("at least one result is required for rendering")
    position_arrays = [result["positions"][::stride].numpy() for result in results]
    center, radius = axis_box(*position_arrays)

    fig = plt.figure(figsize=(6.0 * len(results), 6.0))
    axes = [
        fig.add_subplot(1, len(results), index + 1, projection="3d")
        for index in range(len(results))
    ]
    artists = []
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")
    for index, (axis, result) in enumerate(zip(axes, results)):
        color = colors[index % len(colors)]
        lines = [
            axis.plot([], [], [], color=color, linewidth=0.45)[0]
            for _ in edges
        ]
        points = axis.scatter([], [], [], s=5, color=color)
        pins = axis.scatter([], [], [], s=35, marker="s", color="tab:red")
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.view_init(elev=22, azim=-60)
        axis.set_title(str(result["solver"]))
        artists.append((axis, lines, points, pins))

    def residual_text(result: dict[str, Any], frame: int) -> str:
        if frame == 0:
            return "initial"
        rows = result["frames"]
        index = min(frame - 1, len(rows) - 1)
        value = float(rows[index].get("final_residual", float("nan")))
        return f"residual={value:.3e}" if math.isfinite(value) else "residual=nan"

    def update(frame: int):
        original = frame * stride
        drawn = []
        for (axis, lines, points, pins), positions, result in zip(
            artists,
            (array[frame] for array in position_arrays),
            results,
        ):
            for line, (left, right) in zip(lines, edges):
                line.set_data_3d(
                    [positions[left, 0], positions[right, 0]],
                    [positions[left, 1], positions[right, 1]],
                    [positions[left, 2], positions[right, 2]],
                )
            points._offsets3d = (
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
            )
            pins._offsets3d = (
                positions[fixed, 0],
                positions[fixed, 1],
                positions[fixed, 2],
            )
            axis.set_title(f"{result['solver']} frame {original:03d}\n{residual_text(result, original)}")
            drawn.extend([*lines, points, pins])
        return drawn

    animation = FuncAnimation(
        fig,
        update,
        frames=len(learned_positions),
        interval=1000 / fps,
        blit=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps) if format_name == "mp4" else PillowWriter(fps=fps)
    animation.save(output, writer=writer, dpi=140)
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    if args.list_motions:
        print_motion_table(args)
        return
    if args.rollout_frames <= 0 or args.inner_steps <= 0:
        raise ValueError("--rollout-frames and --inner-steps must be positive")
    if args.baseline_step_size <= 0:
        raise ValueError("--baseline-step-size must be positive")
    if args.fixed_gd_step_size <= 0:
        raise ValueError("--fixed-gd-step-size must be positive")
    if args.render_stride <= 0:
        raise ValueError("--render-stride must be positive")

    run_dir = run_directory(args)
    selected_checkpoint = checkpoint_path(args, run_dir)
    if not selected_checkpoint.exists():
        raise FileNotFoundError(selected_checkpoint)
    output_dir = args.output_dir or default_output_dir(args, run_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    dtype = resolve_dtype(args.dtype, checkpoint)
    spec = ModelSpec(**checkpoint["model_spec"])
    model = LearnedOptimizerMLP(
        full_state_dim=15 * 15 * 3,
        model_spec=spec,
        dtype=dtype,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scenario = selected_scenario(args)
    selected_row = scenario_row(args.motion_index, scenario)
    print(
        "selected motion: "
        f"index={selected_row['index']} "
        f"scenario_id={selected_row['scenario_id']} "
        f"group={selected_row['group']} "
        f"boundary={selected_row['boundary']} "
        f"dirichlet={selected_row['dirichlet']} "
        f"material={selected_row['material']}"
    )
    thresholds = FailureThresholds(
        max_residual=args.max_residual,
        max_abs_position=args.max_abs_position,
        min_edge_ratio=args.min_edge_ratio,
        max_edge_ratio=args.max_edge_ratio,
        max_constraint_error=args.max_constraint_error,
    )

    learned = run_model_rollout(
        model=model,
        scenario=scenario,
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
    )
    baseline = run_baseline_rollout(
        scenario=scenario,
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        step_size=args.baseline_step_size,
        max_reductions=args.baseline_line_search_reductions,
    )
    fixed_gd = run_fixed_gd_rollout(
        scenario=scenario,
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        step_size=args.fixed_gd_step_size,
    )

    learned_summary = summarize_rollout(learned, args.rollout_frames)
    baseline_summary = summarize_rollout(baseline, args.rollout_frames)
    fixed_gd_summary = summarize_rollout(fixed_gd, args.rollout_frames)
    results = [learned, baseline, fixed_gd]
    summaries = [learned_summary, baseline_summary, fixed_gd_summary]
    manifest = {
        "checkpoint": str(selected_checkpoint),
        "checkpoint_update": int(checkpoint.get("update_count", 0)),
        "run_directory": str(run_dir),
        "model_spec": asdict(spec),
        "requested_model_spec": asdict(
            ModelSpec(
                activation=args.activation,
                depth=args.depth,
                width=args.width,
                use_bias=args.use_bias,
            )
        ),
        "split": args.split,
        "catalogue_key": split_key(args.split, args.catalogue),
        "motion_index": int(args.motion_index),
        "scenario": asdict(scenario),
        "scenario_labels": scenario_labels(scenario),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "rollout_frames": int(args.rollout_frames),
        "inner_steps": int(args.inner_steps),
        "baselines": [
            {
                "solver": "mass_preconditioned_gd",
                "step_size": float(args.baseline_step_size),
                "line_search": "energy non-increase backtracking",
                "line_search_reductions": int(args.baseline_line_search_reductions),
            },
            {
                "solver": fixed_gd["solver"],
                "step_size": float(args.fixed_gd_step_size),
                "line_search": None,
            },
        ],
        "summaries": summaries,
    }
    torch.save(
        {
            "manifest": manifest,
            "learned": learned,
            "baseline": baseline,
            "fixed_gd": fixed_gd,
            "results": results,
        },
        output_dir / "rollout_compare.pt",
    )
    write_json(output_dir / "metrics.json", manifest)
    write_frame_csv(output_dir / "per_frame.csv", results)
    plot_diagnostics(output_dir / "diagnostics.png", results)

    render_output: Path | None = None
    if args.render_format != "none":
        render_output = output_dir / f"rollout_compare.{args.render_format}"
        render_comparison(
            output=render_output,
            results=results,
            scenario=scenario,
            fps=args.fps,
            stride=args.render_stride,
            format_name=args.render_format,
        )

    print(f"单 motion rollout 对比完成：{output_dir}")
    if render_output is not None:
        print(f"渲染输出：{render_output}")
    print(
        "summary: "
        f"learned failed={learned_summary['failed']} "
        f"baseline failed={baseline_summary['failed']} "
        f"fixed_gd failed={fixed_gd_summary['failed']} "
        f"learned_ratio_mean={learned_summary['residual_ratio_mean']} "
        f"baseline_ratio_mean={baseline_summary['residual_ratio_mean']} "
        f"fixed_gd_ratio_mean={fixed_gd_summary['residual_ratio_mean']}"
    )


if __name__ == "__main__":
    main()
