"""Render a VBD segment aligned with a saved learned-optimizer rollout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import numpy as np

from cloth15_render_vbd_reference import render_mp4
from cloth23_render_single_motion_rollout import plot_keyframes
from tshirt_config import DEFAULT_FIXED_DATA_DIR, load_model_spec


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VBD_TRAJECTORY = PROJECT_DIR / "vbd_reference" / "data" / "trajectory.npz"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--network-rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vbd-trajectory", type=Path, default=DEFAULT_VBD_TRAJECTORY)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="auto-detect from DISPLAY; headless Linux uses Polyscope EGL",
    )
    parser.add_argument("--egl-device-index", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _aligned_vbd_positions(
    *,
    source_times: np.ndarray,
    source_positions: np.ndarray,
    target_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if target_times[0] < source_times[0] or target_times[-1] > source_times[-1]:
        raise ValueError(
            f"target time range [{target_times[0]}, {target_times[-1]}] exceeds "
            f"VBD range [{source_times[0]}, {source_times[-1]}]"
        )
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    exact = source_times[right] == target_times
    left = np.where(exact, right, np.maximum(right - 1, 0))
    denominator = source_times[right] - source_times[left]
    alpha = np.divide(
        target_times - source_times[left],
        denominator,
        out=np.zeros_like(target_times, dtype=np.float64),
        where=denominator > 0.0,
    )
    positions = (
        source_positions[left] * (1.0 - alpha[:, None, None])
        + source_positions[right] * alpha[:, None, None]
    ).astype(np.float32)
    return positions, left, right, alpha


def _side_by_side(
    *,
    ffmpeg: str,
    network_video: Path,
    vbd_video: Path,
    output: Path,
    crf: int,
) -> None:
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(network_video),
        "-i",
        str(vbd_video),
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v]",
        "-map",
        "[v]",
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
        raise RuntimeError(f"side-by-side ffmpeg failed: {completed.stderr.strip()}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not produce the side-by-side MP4")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    network_root = args.network_rollout_dir.resolve()
    network_result = (
        network_root / "network" if (network_root / "network").is_dir() else network_root
    )
    network_trajectory = network_result / "trajectory.npz"
    network_render_manifest = network_root / "render_manifest.json"
    network_video = network_root / "motion.mp4"
    for path in (network_trajectory, network_render_manifest, network_video):
        if not path.is_file():
            raise FileNotFoundError(f"network comparison input is missing: {path}")

    output = args.output_dir.resolve()
    expected = (
        output / "motion.mp4",
        output / "motion_final.png",
        output / "vbd_keyframes.png",
        output / "mlp_vs_vbd_side_by_side.mp4",
        output / "trajectory_aligned.npz",
        output / "render_manifest.json",
    )
    if not args.overwrite and all(path.is_file() and path.stat().st_size > 0 for path in expected):
        print(f"VBD comparison cache complete; skipped: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)

    fixed_model = load_model_spec(Path(args.fixed_data_dir) / "model_spec.json")
    with np.load(network_trajectory, allow_pickle=False) as network:
        physical_frames = network["frames"].astype(np.int64)
    target_times = physical_frames.astype(np.float64) * fixed_model.dt
    with np.load(args.vbd_trajectory.resolve(), allow_pickle=False) as vbd:
        source_steps = vbd["steps"].astype(np.int64)
        source_times = vbd["times"].astype(np.float64)
        source_positions = vbd["positions"].astype(np.float32)
        faces = vbd["faces"].astype(np.int32)
        fixed_indices = vbd["fixed_indices"].astype(np.int64)
    positions, left, right, alpha = _aligned_vbd_positions(
        source_times=source_times,
        source_positions=source_positions,
        target_times=target_times,
    )
    np.savez_compressed(
        output / "trajectory_aligned.npz",
        physical_frames=physical_frames,
        times=target_times,
        positions=positions,
        faces=faces,
        fixed_indices=fixed_indices,
        vbd_left_steps=source_steps[left],
        vbd_right_steps=source_steps[right],
        interpolation_alpha=alpha,
    )

    render_source = json.loads(network_render_manifest.read_text(encoding="utf-8"))["render"]
    fps = int(render_source["fps"])
    width, height = (int(value) for value in render_source["resolution"])
    crf = int(render_source["crf"])
    headless = not bool(os.environ.get("DISPLAY")) if args.headless is None else args.headless
    video = output / "motion.mp4"
    poster = output / "motion_final.png"
    render = render_mp4(
        positions=positions,
        faces=faces,
        fixed_indices=fixed_indices,
        output=video,
        poster=poster,
        fps=fps,
        width=width,
        height=height,
        headless=headless,
        egl_device_index=args.egl_device_index,
        crf=crf,
    )
    keyframes = output / "vbd_keyframes.png"
    plot_keyframes(
        positions=positions,
        physical_frames=physical_frames,
        faces=faces,
        fixed_indices=fixed_indices,
        output=keyframes,
        title="Newton VBD reference: typical 0 rollout",
    )
    comparison = output / "mlp_vs_vbd_side_by_side.mp4"
    _side_by_side(
        ffmpeg=str(render["ffmpeg"]),
        network_video=network_video,
        vbd_video=video,
        output=comparison,
        crf=crf,
    )
    _write_json(
        output / "render_manifest.json",
        {
            "completed": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "alignment": {
                "network_physical_frame_range": [
                    int(physical_frames[0]),
                    int(physical_frames[-1]),
                ],
                "physical_time_range_seconds": [
                    float(target_times[0]),
                    float(target_times[-1]),
                ],
                "output_frame_count": int(len(positions)),
                "output_fps": fps,
                "output_duration_seconds": float(len(positions) / fps),
                "VBD_source_sample_period_seconds": float(np.median(np.diff(source_times))),
                "target_sample_period_seconds": float(np.median(np.diff(target_times))),
                "position_resampling": (
                    "linear interpolation between saved VBD states for visualization only"
                ),
            },
            "source": {
                "network_rollout": str(network_root),
                "network_video": str(network_video),
                "VBD_trajectory": str(args.vbd_trajectory.resolve()),
                "VBD_source_step_range_used": [
                    int(source_steps[left].min()),
                    int(source_steps[right].max()),
                ],
            },
            "render": {
                **render,
                "video": str(video),
                "final_frame": str(poster),
                "keyframes": str(keyframes),
                "side_by_side_video": str(comparison),
                "side_by_side_layout": "left=MLP 5e-8, right=VBD reference",
            },
        },
    )
    print(f"VBD comparison segment written to {output}", flush=True)
    print(f"VBD MP4: {video}", flush=True)
    print(f"Side-by-side MP4: {comparison}", flush=True)


if __name__ == "__main__":
    main()
