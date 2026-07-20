"""Render the saved Newton VBD reference trajectory with Polyscope."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_DIR = PROJECT_DIR / "vbd_reference"
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cloth_tshirt_matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auto-detect from DISPLAY; headless Linux uses Polyscope EGL.",
    )
    parser.add_argument("--egl-device-index", type=int, default=-1)
    args = parser.parse_args()
    if args.fps <= 0 or args.frame_stride <= 0:
        parser.error("--fps and --frame-stride must be positive")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if not 0 <= args.video_crf <= 51:
        parser.error("--video-crf must be in [0, 51]")
    return args


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_diagnostics(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No diagnostic rows in {path}")
    numeric: dict[str, np.ndarray] = {}
    for name in rows[0]:
        if name == "finite":
            continue
        numeric[name] = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    return numeric


def residual_fields(data: dict[str, np.ndarray]) -> list[str]:
    return sorted(name for name in data if "residual" in name.lower())


def plot_diagnostics(data: dict[str, np.ndarray], output: Path) -> None:
    import matplotlib.pyplot as plt

    steps = data["step"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].semilogy(steps, np.maximum(data["mean_speed"], 1.0e-12), label="mean speed")
    axes[0].semilogy(steps, np.maximum(data["max_speed"], 1.0e-12), label="max speed")
    axes[0].set(ylabel="speed (m/s)", title="Newton VBD saved diagnostics")

    for field, label in (
        ("min_area_ratio", "min area ratio"),
        ("max_area_ratio", "max area ratio"),
        ("min_edge_length_ratio", "min edge ratio"),
        ("max_edge_length_ratio", "max edge ratio"),
    ):
        axes[1].plot(steps, data[field], label=label)
    axes[1].axhline(1.0, color="black", ls="--", lw=1, alpha=0.7)
    axes[1].set(xlabel="physical time step", ylabel="ratio to rest state")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_residual(data: dict[str, np.ndarray], fields: list[str], output: Path) -> None:
    import matplotlib.pyplot as plt

    steps = data["step"]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    for field in fields:
        values = data[field]
        valid = np.isfinite(values) & (values > 0.0)
        axis.semilogy(steps[valid], values[valid], label=field)
    axis.set(
        xlabel="physical time step",
        ylabel="residual",
        title="Newton VBD residual vs. physical time step",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def render_mp4(
    *,
    positions: np.ndarray,
    faces: np.ndarray,
    fixed_indices: np.ndarray,
    output: Path,
    poster: Path,
    fps: int,
    width: int,
    height: int,
    headless: bool,
    egl_device_index: int,
    crf: int,
) -> dict[str, Any]:
    try:
        import imageio_ffmpeg
        import polyscope as ps
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Rendering requires polyscope and imageio-ffmpeg in the active environment"
        ) from error

    ps.set_program_name("Newton VBD T-shirt reference")
    ps.set_use_prefs_file(False)
    ps.set_build_gui(False)
    ps.set_verbosity(0)
    if headless:
        ps.set_egl_device_index(egl_device_index)
        ps.init("openGL3_egl")
    else:
        ps.set_allow_headless_backends(True)
        ps.init()
    ps.set_window_size(width, height)
    ps.set_ground_plane_mode("none")
    mesh = ps.register_surface_mesh(
        "T-shirt",
        positions[0],
        faces,
        color=(0.18, 0.48, 0.82),
        edge_color=(0.055, 0.075, 0.11),
        edge_width=0.3,
        smooth_shade=True,
        material="candy",
    )
    ps.register_point_cloud(
        "fixed shoulders",
        positions[0, fixed_indices],
        color=(0.92, 0.10, 0.08),
        radius=0.012,
    )

    minimum = np.min(positions, axis=(0, 1))
    maximum = np.max(positions, axis=(0, 1))
    center = 0.5 * (minimum + maximum)
    extent = max(float(np.max(maximum - minimum)), 1.0e-3)
    camera = center + extent * np.asarray((1.35, 0.55, 1.55))
    ps.look_at(tuple(camera), tuple(center))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vbd_reference_polyscope_") as directory:
        frame_dir = Path(directory)
        last_frame = None
        for index, vertex_positions in enumerate(positions):
            mesh.update_vertex_positions(vertex_positions)
            frame_path = frame_dir / f"frame_{index:06d}.png"
            ps.screenshot(str(frame_path), transparent_bg=False, include_UI=False)
            last_frame = frame_path
            if index % 50 == 0 or index + 1 == len(positions):
                print(f"rendered frame {index + 1}/{len(positions)}", flush=True)
        if last_frame is None:
            raise RuntimeError("Trajectory contains no frames")
        shutil.copyfile(last_frame, poster)

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%06d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Polyscope/ffmpeg did not produce a non-empty MP4")
    return {
        "renderer": "Polyscope",
        "headless": headless,
        "egl_device_index": egl_device_index if headless else None,
        "resolution": [width, height],
        "fps": fps,
        "rendered_frame_count": int(len(positions)),
        "video_duration_seconds": float(len(positions) / fps),
        "ffmpeg": ffmpeg,
        "crf": crf,
    }


def main() -> None:
    args = parse_args()
    reference_dir = args.reference_dir.resolve()
    trajectory_path = reference_dir / "trajectory.npz"
    diagnostics_path = reference_dir / "diagnostics.csv"
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        all_steps = trajectory["steps"].astype(np.int64)
        all_times = trajectory["times"].astype(np.float64)
        positions = trajectory["positions"][:: args.frame_stride].astype(np.float32)
        steps = all_steps[:: args.frame_stride]
        times = all_times[:: args.frame_stride]
        faces = trajectory["faces"].astype(np.int32)
        fixed_indices = trajectory["fixed_indices"].astype(np.int64)

    diagnostics = load_diagnostics(diagnostics_path)
    residuals = residual_fields(diagnostics)
    diagnostics_plot = reference_dir / "diagnostics_vs_time_step.png"
    plot_diagnostics(diagnostics, diagnostics_plot)
    residual_plot = reference_dir / "residual_vs_time_step.png"
    if residuals:
        plot_residual(diagnostics, residuals, residual_plot)

    headless = (not bool(os.environ.get("DISPLAY"))) if args.headless is None else args.headless
    video = reference_dir / "motion.mp4"
    poster = reference_dir / "motion_final.png"
    render = render_mp4(
        positions=positions,
        faces=faces,
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
    manifest = {
        "completed": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "trajectory": str(trajectory_path),
            "diagnostics": str(diagnostics_path),
            "source_saved_frame_count": int(len(all_steps)),
            "source_step_range": [int(all_steps[0]), int(all_steps[-1])],
            "source_time_range_seconds": [float(all_times[0]), float(all_times[-1])],
        },
        "render": {
            **render,
            "frame_stride": args.frame_stride,
            "rendered_step_range": [int(steps[0]), int(steps[-1])],
            "rendered_time_range_seconds": [float(times[0]), float(times[-1])],
        },
        "outputs": {
            "video": str(video),
            "final_frame": str(poster),
            "diagnostics_plot": str(diagnostics_plot),
            "residual_plot": str(residual_plot) if residuals else None,
        },
        "residual": {
            "available": bool(residuals),
            "fields": residuals,
            "note": (
                "Plotted stored residual fields."
                if residuals
                else "No residual was stored by the VBD reference simulation; speed and geometry diagnostics are not relabeled as residual."
            ),
        },
    }
    write_json(reference_dir / "render_manifest.json", manifest)
    print(f"MP4: {video}", flush=True)
    print(f"Diagnostics plot: {diagnostics_plot}", flush=True)
    if not residuals:
        print("Residual plot skipped: no residual field was saved.", flush=True)


if __name__ == "__main__":
    main()
