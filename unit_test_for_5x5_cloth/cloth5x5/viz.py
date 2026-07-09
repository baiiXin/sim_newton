from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from .config import PhysicalConfig
from .constants import FIXED_VERTEX_INDICES, SPRING_EDGES, TRIANGLE_FACES
from .rollout import FIXED_INNER_ITERATIONS

ERROR_FLOOR = 1e-16


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
    return [points[[i, j], :] for i, j in SPRING_EDGES]


def mesh_triangle_polys(points: np.ndarray) -> list[np.ndarray]:
    return [points[list(face), :] for face in TRIANGLE_FACES]


def create_cloth_artists(ax: Any, label: str) -> dict[str, Any]:
    edges = Line3DCollection([np.zeros((2, 3))], linewidths=1.1, alpha=0.95, label=label)
    ax.add_collection3d(edges)
    vertices = ax.scatter([], [], [], s=16, depthshade=True)
    fixed = ax.scatter([], [], [], s=70, marker="s", depthshade=True, label="fixed vertices")
    return {"edges": edges, "vertices": vertices, "fixed": fixed}


def update_cloth_artists(artists: dict[str, Any], points: np.ndarray) -> None:
    artists["edges"].set_segments(mesh_edge_segments(points))
    artists["vertices"]._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
    fixed_points = points[list(FIXED_VERTEX_INDICES)]
    artists["fixed"]._offsets3d = (
        fixed_points[:, 0],
        fixed_points[:, 1],
        fixed_points[:, 2],
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
    reference_edges = Line3DCollection(
        [np.zeros((2, 3))], linewidths=0.75, alpha=0.42, linestyles="--"
    )
    ax.add_collection3d(reference_edges)
    fixed = ax.scatter([], [], [], s=70, marker="s", depthshade=True)
    return {"surface": surface, "reference_edges": reference_edges, "fixed": fixed, "norm": norm, "cmap": cmap}


def update_colored_difference_artists(
    artists: dict[str, Any],
    prediction: np.ndarray,
    reference: np.ndarray,
    vertex_error: np.ndarray,
) -> None:
    triangles = np.asarray(TRIANGLE_FACES, dtype=int)
    face_errors = np.maximum(vertex_error[triangles].mean(axis=1), ERROR_FLOOR)
    facecolors = artists["cmap"](artists["norm"](face_errors))
    artists["surface"].set_verts(mesh_triangle_polys(prediction))
    artists["surface"].set_facecolors(facecolors)
    artists["reference_edges"].set_segments(mesh_edge_segments(reference))
    fixed_points = prediction[list(FIXED_VERTEX_INDICES)]
    artists["fixed"]._offsets3d = (
        fixed_points[:, 0],
        fixed_points[:, 1],
        fixed_points[:, 2],
    )


def per_vertex_position_error(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.linalg.norm(prediction - reference, axis=-1)


def render_video(
    *,
    rollout: dict[str, Any],
    physical: PhysicalConfig,
    frames: int,
    fps: int,
    output_dir: Path,
) -> Path:
    positions = rollout["positions"]
    errors = rollout["errors"]
    minimum, maximum = axis_limits(positions)
    times = np.arange(1, frames + 1) * physical.dt

    learned_vertex_error = per_vertex_position_error(positions["learned"], positions["reference"])
    gd_vertex_error = per_vertex_position_error(positions["gradient_descent"], positions["reference"])
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
    top_artists = [create_cloth_artists(ax, title) for ax, title in zip(axes_top, top_titles)]
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
        [np.maximum(np.asarray(errors[name]["position_rms"]), ERROR_FLOOR) for name in error_lines]
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
        state_index = video_index + 1
        reference = positions["reference"][state_index]
        returned: list[Any] = []
        for solver_name, title, ax, artists in zip(top_names, top_titles, axes_top, top_artists):
            current = positions[solver_name][state_index]
            update_cloth_artists(artists, current)
            diagnostic = rollout["diagnostics"][solver_name][video_index]
            position_error = errors[solver_name]["position_rms"][video_index]
            ax.set_title(
                f"{title} | frame {state_index}/{frames}\n"
                f"{FIXED_INNER_ITERATIONS} iterations, residual={diagnostic['residual']:.2e}, "
                f"error={position_error:.2e}"
            )
            returned.extend(artists.values())

        update_colored_difference_artists(
            mlp_diff_artists,
            positions["learned"][state_index],
            reference,
            learned_vertex_error[state_index],
        )
        update_colored_difference_artists(
            gd_diff_artists,
            positions["gradient_descent"][state_index],
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
            y = np.maximum(np.asarray(errors[solver_name]["position_rms"][:upto]), ERROR_FLOOR)
            line.set_data(x, y)
            returned.append(line)
        fig.suptitle(
            "5x5 triangular cloth multi-motion model: exactly 50 inner iterations per frame",
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

    mp4_path = output_dir / "fixed_left_edge_5x5_cloth_multi_motion_500_frames.mp4"
    gif_path = output_dir / "fixed_left_edge_5x5_cloth_multi_motion_500_frames.gif"
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
    rollout: dict[str, Any],
    physical: PhysicalConfig,
    frames: int,
    save_path: Path,
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
            values = np.maximum(np.asarray(rollout["errors"][solver_name][metric]), ERROR_FLOOR)
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
