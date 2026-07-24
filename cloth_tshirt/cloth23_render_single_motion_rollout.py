"""Plot diagnostics and render a saved single-motion rollout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np

from cloth13_inference import _resolve_mesh_path
from cloth15_render_vbd_reference import render_mp4
from tshirt_config import DEFAULT_FIXED_DATA_DIR, load_model_spec
from tshirt_mesh import load_tshirt_mesh


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cloth_tshirt_matplotlib")
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--frame-hold",
        type=int,
        default=1,
        help="repeat each saved state this many times in the MP4",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="auto-detect from DISPLAY; headless Linux uses Polyscope EGL",
    )
    parser.add_argument("--egl-device-index", type=int, default=-1)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.frame_stride <= 0 or args.frame_hold <= 0:
        parser.error("--fps, --frame-stride, and --frame-hold must be positive")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if not 0 <= args.video_crf <= 51:
        parser.error("--video-crf must be in [0, 51]")
    return args


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_curves(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _finite_positive(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > 0.0)


def plot_diagnostics(
    curves: dict[str, np.ndarray],
    output_dir: Path,
    *,
    residual_ratio_tolerance: float,
) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(len(curves["residual_ratio"]))
    outputs: list[Path] = []

    figure, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for name, label in (
        ("initial_residual", "initial residual"),
        ("final_residual", "after 50 inner steps"),
    ):
        valid = _finite_positive(curves[name])
        axes[0].semilogy(frames[valid], curves[name][valid], label=label)
    valid = _finite_positive(curves["residual_ratio"])
    axes[1].semilogy(
        frames[valid], curves["residual_ratio"][valid], color="#2ca02c", label="final / initial"
    )
    axes[1].axhline(
        residual_ratio_tolerance,
        color="black",
        ls="--",
        lw=1,
        label=f"target={residual_ratio_tolerance:g}",
    )
    axes[0].set(ylabel="residual L2 norm", title="Implicit-solve convergence")
    axes[1].set(xlabel="physical frame", ylabel="residual ratio")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    path = output_dir / "01_residual_convergence.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    valid = _finite_positive(curves["first_step_ratio"])
    axes[0].semilogy(
        frames[valid], curves["first_step_ratio"][valid], color="#9467bd", label="first step"
    )
    axes[0].axhline(1e-2, color="black", ls="--", lw=1, label="two-order target")
    finite_energy = np.isfinite(curves["energy_change"])
    axes[1].plot(
        frames[finite_energy],
        curves["energy_change"][finite_energy],
        color="#d62728",
        label="energy after - before",
    )
    axes[1].axhline(0.0, color="black", ls="--", lw=1)
    axes[0].set(ylabel="first-step residual ratio", title="First update and energy behavior")
    axes[1].set(xlabel="physical frame", ylabel="energy change")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    path = output_dir / "02_step_and_energy.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    for name, label in (("area_min", "minimum"), ("area_max", "maximum")):
        valid = _finite_positive(curves[name])
        axes[0].plot(frames[valid], curves[name][valid], label=label)
    for name, label in (("edge_min", "minimum"), ("edge_max", "maximum")):
        valid = _finite_positive(curves[name])
        axes[1].plot(frames[valid], curves[name][valid], label=label)
    axes[0].set(ylabel="area / rest area", title="Geometric quality")
    axes[1].set(xlabel="physical frame", ylabel="edge / rest edge")
    for axis in axes:
        axis.axhline(1.0, color="black", ls="--", lw=1, alpha=0.7)
        axis.grid(True, alpha=0.25)
        axis.legend()
    path = output_dir / "03_geometry_ratios.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    if (
        "line_search_alpha_mean" in curves
        and bool(np.isfinite(curves["line_search_alpha_mean"]).any())
    ):
        figure, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
        axes[0].plot(
            frames,
            curves["line_search_accepted_steps"],
            label="accepted",
            color="#2ca02c",
        )
        axes[0].plot(
            frames,
            curves["line_search_rejected_steps"],
            label="rejected",
            color="#d62728",
        )
        axes[1].plot(
            frames,
            curves["line_search_trials"],
            label="candidate energy evaluations",
            color="#9467bd",
        )
        for name, label in (
            ("line_search_alpha_mean", "mean accepted alpha"),
            ("line_search_alpha_min", "minimum accepted alpha"),
        ):
            valid = _finite_positive(curves[name])
            axes[2].semilogy(frames[valid], curves[name][valid], label=label)
        axes[0].set(ylabel="inner steps", title="Network Armijo line search")
        axes[1].set(ylabel="trials")
        axes[2].set(xlabel="physical frame", ylabel="accepted alpha")
        for axis in axes:
            axis.grid(True, which="both", alpha=0.25)
            axis.legend()
        path = output_dir / "05_line_search.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)
    return outputs


def _set_equal_limits(axis: Any, minimum: np.ndarray, maximum: np.ndarray) -> None:
    center = 0.5 * (minimum + maximum)
    radius = 0.55 * max(float(np.max(maximum - minimum)), 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plot_keyframes(
    *,
    positions: np.ndarray,
    physical_frames: np.ndarray,
    faces: np.ndarray,
    fixed_indices: np.ndarray,
    output: Path,
    title: str = "Tensor-parallel MLP: typical 0 rollout",
) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    indices = np.unique(np.asarray((0, len(positions) // 2, len(positions) - 1)))
    minimum = np.min(positions, axis=(0, 1))
    maximum = np.max(positions, axis=(0, 1))
    figure = plt.figure(figsize=(5.4 * len(indices), 5.2))
    for panel, index in enumerate(indices, start=1):
        axis = figure.add_subplot(1, len(indices), panel, projection="3d")
        surface = Poly3DCollection(
            positions[index][faces],
            facecolor="#2f7fc1",
            edgecolor="#172638",
            linewidth=0.08,
            alpha=0.96,
        )
        axis.add_collection3d(surface)
        axis.scatter(
            positions[index, fixed_indices, 0],
            positions[index, fixed_indices, 1],
            positions[index, fixed_indices, 2],
            color="#e31a1c",
            s=24,
            depthshade=False,
            label="fixed shoulders",
        )
        _set_equal_limits(axis, minimum, maximum)
        axis.view_init(elev=24, azim=-55)
        axis.set_title(f"physical frame {int(physical_frames[index])}")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.legend(loc="upper right")
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rollout_dir = args.rollout_dir.resolve()
    if (rollout_dir / "network").is_dir():
        result_dir = rollout_dir / "network"
    elif (rollout_dir / "newton").is_dir():
        result_dir = rollout_dir / "newton"
    else:
        result_dir = rollout_dir
    curves_path = result_dir / "curves.npz"
    trajectory_path = result_dir / "trajectory.npz"
    metrics_path = result_dir / "metrics.json"
    for path in (curves_path, trajectory_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"single-motion result is missing: {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    render_manifest_path = rollout_dir / "render_manifest.json"
    expected = (
        rollout_dir / "figures" / "01_residual_convergence.png",
        rollout_dir / "figures" / "02_step_and_energy.png",
        rollout_dir / "figures" / "03_geometry_ratios.png",
        rollout_dir / "figures" / "04_trajectory_keyframes.png",
    )
    if bool(metrics.get("network_line_search", False)):
        expected = (*expected, rollout_dir / "figures" / "05_line_search.png")
    if not args.skip_video:
        expected = (
            *expected,
            rollout_dir / "motion_final.png",
            rollout_dir / "motion.mp4",
        )
    if (
        not args.overwrite
        and render_manifest_path.is_file()
        and all(path.is_file() and path.stat().st_size > 0 for path in expected)
    ):
        print(f"visualization cache complete; skipped: {rollout_dir}")
        return

    model = load_model_spec(Path(args.fixed_data_dir) / "model_spec.json")
    mesh = load_tshirt_mesh(_resolve_mesh_path(args.fixed_data_dir, model.mesh_path))
    if mesh.sha256 != model.mesh_sha256:
        raise ValueError("rendering OBJ hash differs from the fixed model")
    curves = _load_curves(curves_path)
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        all_physical_frames = trajectory["frames"].astype(np.int64)
        all_positions = trajectory["positions"].astype(np.float32)
    physical_frames = all_physical_frames[:: args.frame_stride]
    positions = all_positions[:: args.frame_stride]
    fixed_indices = np.asarray(model.fixed_indices, dtype=np.int64)

    figures_dir = rollout_dir / "figures"
    plots = plot_diagnostics(
        curves,
        figures_dir,
        residual_ratio_tolerance=float(metrics["residual_ratio_tolerance"]),
    )
    keyframes = figures_dir / "04_trajectory_keyframes.png"
    plot_keyframes(
        positions=positions,
        physical_frames=physical_frames,
        faces=mesh.faces,
        fixed_indices=fixed_indices,
        output=keyframes,
        title=str(
            metrics.get(
                "visualization_title",
                "Tensor-parallel MLP: typical 0 rollout",
            )
        ),
    )
    plots.append(keyframes)

    video = rollout_dir / "motion.mp4"
    poster = rollout_dir / "motion_final.png"
    if args.skip_video:
        render: dict[str, Any] | None = None
    else:
        headless = (
            not bool(os.environ.get("DISPLAY")) if args.headless is None else args.headless
        )
        render = render_mp4(
            positions=np.repeat(positions, args.frame_hold, axis=0),
            faces=mesh.faces,
            fixed_indices=fixed_indices,
            output=video,
            poster=poster,
            fps=args.fps,
            width=args.width,
            height=args.height,
            headless=headless,
            egl_device_index=args.egl_device_index,
            crf=args.video_crf,
        )
    _write_json(
        render_manifest_path,
        {
            "completed": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "rollout_directory": str(rollout_dir),
                "curves": str(curves_path),
                "trajectory": str(trajectory_path),
                "metrics": str(metrics_path),
                "saved_trajectory_frame_count": int(len(all_positions)),
                "saved_physical_frame_range": [
                    int(all_physical_frames[0]),
                    int(all_physical_frames[-1]),
                ],
            },
            "visualization": {
                "figures": [str(path) for path in plots],
                "keyframe_physical_frames": [
                    int(physical_frames[0]),
                    int(physical_frames[len(physical_frames) // 2]),
                    int(physical_frames[-1]),
                ],
            },
            "render": (
                None
                if render is None
                else {
                    **render,
                    "frame_stride": args.frame_stride,
                    "frame_hold": args.frame_hold,
                    "video": str(video),
                    "final_frame": str(poster),
                }
            ),
        },
    )
    print(f"visualizations written to {figures_dir}", flush=True)
    if render is not None:
        print(f"MP4: {video}", flush=True)
        print(f"Final frame: {poster}", flush=True)


if __name__ == "__main__":
    main()
