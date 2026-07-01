"""
500-frame continuous rollout for the fixed-left-edge 5x5 triangular cloth.

The script loads the validation-selected multi-problem MLP checkpoint produced
by ``fixed_left_edge_5x5_cloth_train_compare.py``. Starting from the
solver-independent hard extrapolation case selected by the training script, it
propagates four independent physical trajectories:

    1. high-accuracy damped-Newton reference,
    2. MLP learned optimizer,
    3. validation-selected fixed-step gradient descent,
    4. undamped full Newton.

MLP, gradient descent, and Newton each execute exactly 50 inner iterations per
physical frame. There is no convergence-based early stopping. Each method uses
its own predicted position and velocity to construct the next frame, so errors
are allowed to accumulate naturally.

The visualization uses cloth grid rendering. The MLP-reference and
GD-reference comparison panels render the predicted triangular cloth with a
per-triangle temperature map of vertex position error, while a translucent
reference wireframe is overlaid for context.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np
import torch

import fixed_left_edge_5x5_cloth_train_compare as core


TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

DEFAULT_DEVICE = "cuda:1"
DEFAULT_FRAMES = 500
FIXED_INNER_ITERATIONS = 50
DEFAULT_FPS = 25
ERROR_FLOOR = 1e-16


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    core.save_json(data, path)


def create_output_directory() -> Path:
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent / script_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed-50-iteration, 500-frame continuous solver comparison"
    )
    default_training_output = (
        Path(__file__).resolve().parent / "fixed_left_edge_5x5_cloth_train_compare"
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=default_training_output,
        help="Output directory created by fixed_left_edge_5x5_cloth_train_compare.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional MLP checkpoint. Default: "
            "<training-output-dir>/multi_problem/best_validation_model_state_dict.pt"
        ),
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if int(args.frames) <= 0:
        raise ValueError("frames must be positive")
    if int(args.fps) <= 0:
        raise ValueError("fps must be positive")


def tensor_state(
    values: Any,
    *,
    device: torch.device,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return torch.as_tensor(values, dtype=TORCH_DTYPE, device=device).reshape(shape)


def solve_fixed_iterations(
    *,
    solver: str,
    p_full: torch.Tensor,
    v_full: torch.Tensor,
    physical: core.PhysicalConfig,
    free_masses: torch.Tensor,
    inner_iterations: int,
    model: core.MLPOptimizer | None = None,
    gd_step_size: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Solve one frame using exactly ``inner_iterations`` updates."""
    q = core.make_q_free(p_full, v_full, physical).reshape(1, core.FREE_STATE_DIM)
    masses = free_masses.reshape(1, core.NUM_FREE_PARTICLES)
    y = core.free_state_from_full(p_full).reshape(1, core.FREE_STATE_DIM).clone()

    start_time = time.perf_counter()
    last_delta_norm = float("nan")
    for iteration in range(inner_iterations):
        if solver == "learned":
            if model is None:
                raise ValueError("model is required for learned rollout")
            y_next, delta = core.apply_model_update(model, y, q, masses, physical)
        elif solver == "gradient_descent":
            if gd_step_size is None:
                raise ValueError("gd_step_size is required")
            y_next, delta = core.apply_gradient_descent_update(
                y, q, masses, physical, gd_step_size
            )
        elif solver == "full_newton":
            y_next, delta = core.apply_newton_update(y, q, masses, physical)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        if not bool(torch.isfinite(y_next).all()):
            raise RuntimeError(
                f"{solver} produced non-finite state at inner iteration {iteration + 1}"
            )
        last_delta_norm = float(torch.linalg.vector_norm(delta).item())
        y = y_next

    elapsed = time.perf_counter() - start_time
    residual = float(core.stationarity_residual_norm(y, q, masses, physical).item())
    energy = float(core.variational_energy(y, q, masses, physical).item())
    next_p, next_v = core.advance_physical_state(
        p_full, y.squeeze(0), physical
    )
    return next_p, next_v, {
        "inner_iterations": inner_iterations,
        "residual": residual,
        "energy": energy,
        "last_update_norm": last_delta_norm,
        "solve_seconds": elapsed,
    }


def solve_reference_frame(
    *,
    p_full: torch.Tensor,
    v_full: torch.Tensor,
    physical: core.PhysicalConfig,
    free_masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    q = core.make_q_free(p_full, v_full, physical)
    initial_y = core.free_state_from_full(p_full).reshape(core.FREE_STATE_DIM)
    start_time = time.perf_counter()
    exact_y, info = core.solve_reference_solution(
        q=q,
        masses=free_masses,
        initial_y=initial_y,
        physical=physical,
        raise_on_nonconvergence=False,
    )
    elapsed = time.perf_counter() - start_time
    next_p, next_v = core.advance_physical_state(p_full, exact_y, physical)
    info = dict(info)
    info["solve_seconds"] = elapsed
    return next_p, next_v, info


def free_position_errors(
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    point_error = torch.linalg.vector_norm(
        prediction[list(core.FREE_VERTEX_INDICES), :] - reference[list(core.FREE_VERTEX_INDICES), :], dim=-1
    )
    rms = float(torch.sqrt(torch.mean(point_error**2)).item())
    maximum = float(torch.max(point_error).item())
    return rms, maximum


def free_velocity_rms_error(
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    point_error = torch.linalg.vector_norm(
        prediction[list(core.FREE_VERTEX_INDICES), :] - reference[list(core.FREE_VERTEX_INDICES), :], dim=-1
    )
    return float(torch.sqrt(torch.mean(point_error**2)).item())


def run_rollout(
    *,
    initial_p: torch.Tensor,
    initial_v: torch.Tensor,
    physical: core.PhysicalConfig,
    model: core.MLPOptimizer,
    gd_step_size: float,
    frames: int,
    device: torch.device,
) -> dict[str, Any]:
    free_masses_device = torch.tensor(
        [physical.masses[i] for i in core.FREE_VERTEX_INDICES], dtype=TORCH_DTYPE, device=device
    )
    free_masses_cpu = torch.tensor([physical.masses[i] for i in core.FREE_VERTEX_INDICES], dtype=TORCH_DTYPE)

    states = {
        "reference": {
            "p": initial_p.detach().cpu().clone(),
            "v": initial_v.detach().cpu().clone(),
        },
        "learned": {
            "p": initial_p.to(device).clone(),
            "v": initial_v.to(device).clone(),
        },
        "gradient_descent": {
            "p": initial_p.to(device).clone(),
            "v": initial_v.to(device).clone(),
        },
        "full_newton": {
            "p": initial_p.to(device).clone(),
            "v": initial_v.to(device).clone(),
        },
    }

    trajectory_positions: dict[str, list[np.ndarray]] = {
        name: [record["p"].detach().cpu().numpy().copy()]
        for name, record in states.items()
    }
    trajectory_velocities: dict[str, list[np.ndarray]] = {
        name: [record["v"].detach().cpu().numpy().copy()]
        for name, record in states.items()
    }
    frame_diagnostics: dict[str, list[dict[str, Any]]] = {
        name: [] for name in states
    }
    errors: dict[str, dict[str, list[float]]] = {
        name: {"position_rms": [], "position_max": [], "velocity_rms": []}
        for name in ["learned", "gradient_descent", "full_newton"]
    }

    start_total = time.perf_counter()
    for frame in range(1, frames + 1):
        ref_p, ref_v, ref_info = solve_reference_frame(
            p_full=states["reference"]["p"],
            v_full=states["reference"]["v"],
            physical=physical,
            free_masses=free_masses_cpu,
        )
        states["reference"] = {"p": ref_p, "v": ref_v}
        frame_diagnostics["reference"].append(ref_info)
        if ref_info.get("line_search_failed", False):
            print(
                f"Warning: reference solver line search failed at frame {frame}: "
                f"residual={ref_info['residual_norm']:.3e}, iterations={ref_info['iterations']}"
            )
        elif not ref_info.get("acceptable", True):
            print(
                f"Warning: reference solver missed strict residual target at frame {frame}: "
                f"residual={ref_info['residual_norm']:.3e}, iterations={ref_info['iterations']}"
            )

        for solver_name in ["learned", "gradient_descent", "full_newton"]:
            p_next, v_next, info = solve_fixed_iterations(
                solver=solver_name,
                p_full=states[solver_name]["p"],
                v_full=states[solver_name]["v"],
                physical=physical,
                free_masses=free_masses_device,
                inner_iterations=FIXED_INNER_ITERATIONS,
                model=model if solver_name == "learned" else None,
                gd_step_size=(
                    gd_step_size if solver_name == "gradient_descent" else None
                ),
            )
            states[solver_name] = {"p": p_next, "v": v_next}
            frame_diagnostics[solver_name].append(info)

        reference_p_device = states["reference"]["p"].to(device)
        reference_v_device = states["reference"]["v"].to(device)
        for solver_name in ["learned", "gradient_descent", "full_newton"]:
            rms, maximum = free_position_errors(
                states[solver_name]["p"], reference_p_device
            )
            velocity_rms = free_velocity_rms_error(
                states[solver_name]["v"], reference_v_device
            )
            errors[solver_name]["position_rms"].append(rms)
            errors[solver_name]["position_max"].append(maximum)
            errors[solver_name]["velocity_rms"].append(velocity_rms)

        for name, record in states.items():
            trajectory_positions[name].append(
                record["p"].detach().cpu().numpy().copy()
            )
            trajectory_velocities[name].append(
                record["v"].detach().cpu().numpy().copy()
            )

        if frame == 1 or frame % 25 == 0 or frame == frames:
            print(
                f"Frame {frame:3d}/{frames}: "
                f"MLP error={errors['learned']['position_rms'][-1]:.3e}, "
                f"GD error={errors['gradient_descent']['position_rms'][-1]:.3e}, "
                f"Newton error={errors['full_newton']['position_rms'][-1]:.3e}"
            )

    total_elapsed = time.perf_counter() - start_total
    positions = {
        name: np.stack(values, axis=0)
        for name, values in trajectory_positions.items()
    }
    velocities = {
        name: np.stack(values, axis=0)
        for name, values in trajectory_velocities.items()
    }
    return {
        "positions": positions,
        "velocities": velocities,
        "diagnostics": frame_diagnostics,
        "errors": errors,
        "total_elapsed_seconds": total_elapsed,
    }


def axis_limits(positions: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_positions = np.concatenate(list(positions.values()), axis=0)
    minimum = np.nanmin(all_positions.reshape(-1, 3), axis=0)
    maximum = np.nanmax(all_positions.reshape(-1, 3), axis=0)
    center = 0.5 * (minimum + maximum)
    span = float(np.max(maximum - minimum))
    span = max(span, 1.0) * 1.12
    return center - 0.5 * span, center + 0.5 * span


def set_3d_limits(ax: Any, minimum: np.ndarray, maximum: np.ndarray) -> None:
    ax.set_xlim(minimum[0], maximum[0])
    ax.set_ylim(minimum[1], maximum[1])
    ax.set_zlim(minimum[2], maximum[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=24, azim=-62)
    ax.grid(True, alpha=0.25)


def mesh_edge_segments(points: np.ndarray) -> list[np.ndarray]:
    return [points[[i, j], :] for i, j in core.SPRING_EDGES]


def mesh_triangle_polys(points: np.ndarray) -> list[np.ndarray]:
    return [points[list(face), :] for face in core.TRIANGLE_FACES]


def create_cloth_artists(ax: Any, label: str) -> dict[str, Any]:
    edges = Line3DCollection([np.zeros((2, 3))], linewidths=1.1, alpha=0.95, label=label)
    ax.add_collection3d(edges)
    vertices = ax.scatter([], [], [], s=16, depthshade=True)
    fixed = ax.scatter([], [], [], s=70, marker="s", depthshade=True, label="fixed vertices")
    return {"edges": edges, "vertices": vertices, "fixed": fixed}


def update_cloth_artists(artists: dict[str, Any], points: np.ndarray) -> None:
    artists["edges"].set_segments(mesh_edge_segments(points))
    artists["vertices"]._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
    fixed_points = points[list(core.FIXED_VERTEX_INDICES)]
    artists["fixed"]._offsets3d = (
        fixed_points[:, 0], fixed_points[:, 1], fixed_points[:, 2]
    )


def create_colored_difference_artists(
    ax: Any,
    *,
    norm: LogNorm,
    cmap: Any,
) -> dict[str, Any]:
    surface = Poly3DCollection(
        [np.zeros((3, 3))],
        linewidths=0.25,
        edgecolors=(0.2, 0.2, 0.2, 0.55),
        alpha=0.92,
    )
    ax.add_collection3d(surface)
    reference_edges = Line3DCollection([np.zeros((2, 3))], linewidths=0.75, alpha=0.42, linestyles="--")
    ax.add_collection3d(reference_edges)
    fixed = ax.scatter([], [], [], s=70, marker="s", depthshade=True)
    return {"surface": surface, "reference_edges": reference_edges, "fixed": fixed, "norm": norm, "cmap": cmap}


def update_colored_difference_artists(
    artists: dict[str, Any],
    prediction: np.ndarray,
    reference: np.ndarray,
    vertex_error: np.ndarray,
) -> None:
    triangles = np.asarray(core.TRIANGLE_FACES, dtype=int)
    face_errors = np.maximum(vertex_error[triangles].mean(axis=1), ERROR_FLOOR)
    facecolors = artists["cmap"](artists["norm"](face_errors))
    artists["surface"].set_verts(mesh_triangle_polys(prediction))
    artists["surface"].set_facecolors(facecolors)
    artists["reference_edges"].set_segments(mesh_edge_segments(reference))
    fixed_points = prediction[list(core.FIXED_VERTEX_INDICES)]
    artists["fixed"]._offsets3d = (
        fixed_points[:, 0], fixed_points[:, 1], fixed_points[:, 2]
    )


def per_vertex_position_error(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - reference, axis=-1)


def render_video(
    *,
    rollout: dict[str, Any],
    physical: core.PhysicalConfig,
    frames: int,
    fps: int,
    output_dir: Path,
) -> Path:
    positions = rollout["positions"]
    errors = rollout["errors"]
    minimum, maximum = axis_limits(positions)
    times = np.arange(1, frames + 1) * physical.dt

    learned_vertex_error = per_vertex_position_error(
        positions["learned"], positions["reference"]
    )
    gd_vertex_error = per_vertex_position_error(
        positions["gradient_descent"], positions["reference"]
    )
    all_diff_errors = np.concatenate(
        [learned_vertex_error[1:].reshape(-1), gd_vertex_error[1:].reshape(-1)]
    )
    finite_positive = all_diff_errors[np.isfinite(all_diff_errors) & (all_diff_errors > 0.0)]
    vmin = max(float(np.min(finite_positive)) if finite_positive.size else ERROR_FLOOR, ERROR_FLOOR)
    vmax = max(float(np.max(finite_positive)) if finite_positive.size else 1.0, vmin * 10.0)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(20, 12))
    axes_top = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(3)]
    ax_mlp_diff = fig.add_subplot(2, 3, 4, projection="3d")
    ax_gd_diff = fig.add_subplot(2, 3, 5, projection="3d")
    ax_error = fig.add_subplot(2, 3, 6)

    for ax in [*axes_top, ax_mlp_diff, ax_gd_diff]:
        set_3d_limits(ax, minimum, maximum)

    top_names = ["learned", "gradient_descent", "full_newton"]
    top_titles = ["MLP", "Gradient descent", "Full Newton"]
    top_artists = [
        create_cloth_artists(ax, title) for ax, title in zip(axes_top, top_titles)
    ]
    for ax in axes_top:
        ax.legend(fontsize=8)

    mlp_diff_artists = create_colored_difference_artists(ax_mlp_diff, norm=norm, cmap=cmap)
    gd_diff_artists = create_colored_difference_artists(ax_gd_diff, norm=norm, cmap=cmap)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = fig.colorbar(
        mappable,
        ax=[ax_mlp_diff, ax_gd_diff],
        shrink=0.62,
        pad=0.03,
        location="right",
    )
    colorbar.set_label("Per-triangle mean vertex position error")

    error_lines = {}
    for name, label in [
        ("learned", "MLP vs reference"),
        ("gradient_descent", "GD vs reference"),
        ("full_newton", "Newton vs reference"),
    ]:
        line, = ax_error.plot([], [], label=label)
        error_lines[name] = line
    ax_error.set_yscale("log")
    ax_error.set_xlim(0.0, max(frames * physical.dt, physical.dt))
    all_error_values = np.concatenate(
        [
            np.maximum(np.asarray(errors[name]["position_rms"]), ERROR_FLOOR)
            for name in error_lines
        ]
    )
    finite = all_error_values[np.isfinite(all_error_values)]
    ymin = max(float(np.min(finite)) * 0.5, ERROR_FLOOR) if finite.size else ERROR_FLOOR
    ymax = max(float(np.max(finite)) * 2.0, ymin * 10.0) if finite.size else 1.0
    ax_error.set_ylim(ymin, ymax)
    ax_error.set_xlabel("Physical time")
    ax_error.set_ylabel("Free-vertex position RMS error")
    ax_error.set_title("Accumulated trajectory error")
    ax_error.grid(True, alpha=0.3)
    ax_error.legend(fontsize=8)

    def update(video_index: int) -> list[Any]:
        # positions index 0 is the initial state; video index 0 displays frame 1.
        state_index = video_index + 1
        reference = positions["reference"][state_index]
        returned: list[Any] = []
        for solver_name, title, ax, artists in zip(
            top_names, top_titles, axes_top, top_artists
        ):
            current = positions[solver_name][state_index]
            update_cloth_artists(artists, current)
            diagnostic = rollout["diagnostics"][solver_name][video_index]
            position_error = errors[solver_name]["position_rms"][video_index]
            ax.set_title(
                f"{title} | frame {state_index}/{frames}\n"
                f"50 iterations, residual={diagnostic['residual']:.2e}, "
                f"error={position_error:.2e}"
            )
            returned.extend(artists.values())

        mlp_current = positions["learned"][state_index]
        gd_current = positions["gradient_descent"][state_index]
        update_colored_difference_artists(
            mlp_diff_artists,
            mlp_current,
            reference,
            learned_vertex_error[state_index],
        )
        update_colored_difference_artists(
            gd_diff_artists,
            gd_current,
            reference,
            gd_vertex_error[state_index],
        )
        ax_mlp_diff.set_title(
            "MLP vs high-accuracy reference\n"
            f"RMS={errors['learned']['position_rms'][video_index]:.2e}, "
            f"max={errors['learned']['position_max'][video_index]:.2e}"
        )
        ax_gd_diff.set_title(
            "GD vs high-accuracy reference\n"
            f"RMS={errors['gradient_descent']['position_rms'][video_index]:.2e}, "
            f"max={errors['gradient_descent']['position_max'][video_index]:.2e}"
        )
        returned.extend(
            [
                mlp_diff_artists["surface"],
                mlp_diff_artists["reference_edges"],
                mlp_diff_artists["fixed"],
                gd_diff_artists["surface"],
                gd_diff_artists["reference_edges"],
                gd_diff_artists["fixed"],
            ]
        )

        upto = video_index + 1
        x = times[:upto]
        for solver_name, line in error_lines.items():
            y = np.maximum(
                np.asarray(errors[solver_name]["position_rms"][:upto]),
                ERROR_FLOOR,
            )
            line.set_data(x, y)
            returned.append(line)
        fig.suptitle(
            "Fixed left edge 5x5 triangular cloth: exactly 50 inner iterations per frame",
            fontsize=15,
        )
        return returned

    plt.tight_layout(rect=(0, 0, 0.94, 0.96))
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    mp4_path = output_dir / "fixed_left_edge_5x5_cloth_500_frames.mp4"
    gif_path = output_dir / "fixed_left_edge_5x5_cloth_500_frames.gif"
    if animation.writers.is_available("ffmpeg"):
        writer = animation.FFMpegWriter(fps=fps, bitrate=4200)
        movie.save(mp4_path, writer=writer, dpi=120)
        output_path = mp4_path
    else:
        writer = animation.PillowWriter(fps=fps)
        movie.save(gif_path, writer=writer, dpi=90)
        output_path = gif_path
    plt.close(fig)
    return output_path


def plot_error_curves(
    rollout: dict[str, Any], physical: core.PhysicalConfig, frames: int, save_path: Path
) -> None:
    times = np.arange(1, frames + 1) * physical.dt
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    labels = {
        "learned": "MLP",
        "gradient_descent": "gradient descent",
        "full_newton": "full Newton",
    }
    metric_specs = [
        ("position_rms", "Position RMS error"),
        ("position_max", "Position maximum error"),
        ("velocity_rms", "Velocity RMS error"),
    ]
    for ax, (metric, title) in zip(axes, metric_specs):
        for solver_name, label in labels.items():
            values = np.maximum(
                np.asarray(rollout["errors"][solver_name][metric]), ERROR_FLOOR
            )
            ax.plot(times, values, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("Physical time")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Continuous-rollout errors against the high-accuracy reference", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def diagnostics_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    solve_times = np.asarray([float(r["solve_seconds"]) for r in records], dtype=float)
    result: dict[str, Any] = {
        "num_frames": len(records),
        "total_solve_seconds": float(np.sum(solve_times)),
        "mean_solve_seconds_per_frame": float(np.mean(solve_times)),
        "p95_solve_seconds_per_frame": float(np.percentile(solve_times, 95)),
    }
    if records and "residual" in records[0]:
        residuals = np.asarray([float(r["residual"]) for r in records], dtype=float)
        result.update(
            final_residual=float(residuals[-1]),
            median_residual=float(np.median(residuals)),
            p95_residual=float(np.percentile(residuals, 95)),
            fixed_inner_iterations=FIXED_INNER_ITERATIONS,
            convergence_early_stopping=False,
        )
    if records and "iterations" in records[0]:
        iterations = np.asarray([int(r["iterations"]) for r in records], dtype=int)
        result.update(
            mean_reference_iterations=float(np.mean(iterations)),
            max_reference_iterations=int(np.max(iterations)),
        )
    if records and "acceptable" in records[0]:
        acceptable = np.asarray([bool(r.get("acceptable", True)) for r in records], dtype=bool)
        result.update(
            strict_reference_success_rate=float(np.mean(acceptable)),
            nonacceptable_reference_frames=int(np.count_nonzero(~acceptable)),
        )
    if records and "line_search_failed" in records[0]:
        line_search_failed = np.asarray(
            [bool(r.get("line_search_failed", False)) for r in records], dtype=bool
        )
        result.update(
            reference_line_search_failure_frames=int(np.count_nonzero(line_search_failed)),
        )
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    training_output_dir = args.training_output_dir.resolve()
    output_dir = create_output_directory()
    device = torch.device(args.device)
    core.validate_device(device)

    runtime = load_json(training_output_dir / "runtime_config.json")
    physical = core.physical_config_from_dict(runtime["physical_config"])
    gd_selection = load_json(
        training_output_dir / "gradient_descent_step_selection.json"
    )
    gd_step_size = float(gd_selection["selected_step_size"])
    hard_case = load_json(training_output_dir / "hard_case_selection.json")
    selected_state = hard_case["selected_physical_state"]

    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else training_output_dir
        / "multi_problem"
        / "best_validation_model_state_dict.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"MLP checkpoint not found: {checkpoint}")

    residual_length_scale = float(
        runtime["runtime_config"]["residual_length_scale"]
    )
    model = core.MLPOptimizer(residual_length_scale).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    initial_p = tensor_state(
        selected_state["p_n_full"], device=torch.device("cpu"), shape=(core.NUM_PARTICLES, core.SPATIAL_DIM)
    )
    initial_v = tensor_state(
        selected_state["v_n_full"], device=torch.device("cpu"), shape=(core.NUM_PARTICLES, core.SPATIAL_DIM)
    )
    fixed = list(core.FIXED_VERTEX_INDICES)
    initial_p[fixed, :] = torch.tensor(physical.fixed_positions, dtype=TORCH_DTYPE)
    initial_v[fixed, :] = 0.0

    print(f"Training output: {training_output_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(
        "Selected hard case: problem "
        f"{selected_state['problem_index']} at physical time {selected_state['time']:.3f}s"
    )
    print(f"Gradient-descent step size: {gd_step_size:.3e}")
    print(
        f"Rollout: {args.frames} frames; MLP/GD/Newton use exactly "
        f"{FIXED_INNER_ITERATIONS} inner iterations per frame"
    )

    rollout = run_rollout(
        initial_p=initial_p,
        initial_v=initial_v,
        physical=physical,
        model=model,
        gd_step_size=gd_step_size,
        frames=int(args.frames),
        device=device,
    )

    np.savez_compressed(
        output_dir / "rollout_trajectories.npz",
        reference_positions=rollout["positions"]["reference"],
        mlp_positions=rollout["positions"]["learned"],
        gradient_descent_positions=rollout["positions"]["gradient_descent"],
        newton_positions=rollout["positions"]["full_newton"],
        reference_velocities=rollout["velocities"]["reference"],
        mlp_velocities=rollout["velocities"]["learned"],
        gradient_descent_velocities=rollout["velocities"]["gradient_descent"],
        newton_velocities=rollout["velocities"]["full_newton"],
    )

    plot_error_curves(
        rollout,
        physical,
        int(args.frames),
        output_dir / "rollout_error_curves.png",
    )

    video_path: Path | None = None
    if not args.skip_video:
        video_path = render_video(
            rollout=rollout,
            physical=physical,
            frames=int(args.frames),
            fps=int(args.fps),
            output_dir=output_dir,
        )

    error_summary: dict[str, Any] = {}
    for solver_name in ["learned", "gradient_descent", "full_newton"]:
        error_summary[solver_name] = {}
        for metric_name, values in rollout["errors"][solver_name].items():
            array = np.asarray(values, dtype=float)
            error_summary[solver_name][metric_name] = {
                "final": float(array[-1]),
                "median": float(np.median(array)),
                "p95": float(np.percentile(array, 95)),
                "max": float(np.max(array)),
            }

    report = {
        "training_output_dir": str(training_output_dir),
        "checkpoint": str(checkpoint),
        "physical_config": core.asdict(physical),
        "fixed_vertex_indices": list(core.FIXED_VERTEX_INDICES),
        "spring_edges": [list(edge) for edge in core.SPRING_EDGES],
        "triangle_faces": [list(face) for face in core.TRIANGLE_FACES],
        "hard_case_selection": hard_case,
        "gradient_descent_step_size": gd_step_size,
        "frames": int(args.frames),
        "physical_duration": int(args.frames) * physical.dt,
        "fixed_inner_iterations": FIXED_INNER_ITERATIONS,
        "same_iteration_budget_for_mlp_gd_newton": True,
        "convergence_early_stopping": False,
        "total_wall_seconds": rollout["total_elapsed_seconds"],
        "solver_timing": {
            name: diagnostics_summary(records)
            for name, records in rollout["diagnostics"].items()
        },
        "error_summary_against_reference": error_summary,
        "video_path": str(video_path) if video_path is not None else None,
        "trajectory_file": str(output_dir / "rollout_trajectories.npz"),
    }
    save_json(report, output_dir / "rollout_metrics.json")
    save_json(
        {
            "selected_problem_index": selected_state["problem_index"],
            "selected_physical_time": selected_state["time"],
            "p_n_full": selected_state["p_n_full"],
            "v_n_full": selected_state["v_n_full"],
            "selection_rule": hard_case["selection_rule"],
        },
        output_dir / "hard_case_used.json",
    )

    print("\nContinuous rollout completed.")
    print(f"Trajectories: {output_dir / 'rollout_trajectories.npz'}")
    print(f"Metrics: {output_dir / 'rollout_metrics.json'}")
    print(f"Error curves: {output_dir / 'rollout_error_curves.png'}")
    if video_path is not None:
        print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
