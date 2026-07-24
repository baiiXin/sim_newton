"""Finalize, diagnose, and render a stopped cloth26 checkpoint without resuming it."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cloth13_inference import _resolve_mesh_path
from cloth15_render_vbd_reference import render_mp4
from cloth23_render_single_motion_rollout import plot_diagnostics, plot_keyframes
from tshirt_config import DEFAULT_FIXED_DATA_DIR, load_model_spec
from tshirt_mesh import load_tshirt_mesh


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--egl-device-index", type=int, default=0)
    parser.add_argument(
        "--safe-physical-frame",
        type=int,
        help=(
            "last saved physical frame used for the safe-prefix video and "
            "fixed camera; defaults to the frame before the first hard failure"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("--fps, --width, and --height must be positive")
    if not 0 <= args.video_crf <= 51:
        parser.error("--video-crf must be in [0, 51]")
    if args.safe_physical_frame is not None and args.safe_physical_frame < 0:
        parser.error("--safe-physical-frame must be nonnegative")
    return args


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _finite_quantile(values: np.ndarray, quantile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    return float(np.quantile(finite, quantile))


def _first_invalid_frame(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        if not bool(row.get("selected_valid", False)):
            return int(row["frame"])
    return None


def _safe_trajectory_mask(
    trajectory_frames: np.ndarray,
    first_invalid_solver_frame: int | None,
) -> np.ndarray:
    if first_invalid_solver_frame is None:
        return np.ones(len(trajectory_frames), dtype=np.bool_)
    # Solver frame j produces physical state j+1.  A saved physical frame <= j
    # therefore contains only solver frames strictly before the first invalid one.
    mask = trajectory_frames <= int(first_invalid_solver_frame)
    if not bool(mask.any()):
        mask[0] = True
    return mask


def run(args: argparse.Namespace) -> None:
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    progress_path = source / "newton" / "progress_state.pt"
    if not progress_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {progress_path}")
    if output == source:
        raise ValueError("--output-dir must differ from --source-dir")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is not empty; use --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_dir = output / "newton"
    figures_dir = output / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(progress_path, map_location="cpu", weights_only=False)
    completed_frames = int(state["next_frame"])
    requested_frames = int(state["rollout_frames"])
    if not 0 < completed_frames <= requested_frames:
        raise ValueError(
            f"invalid checkpoint frame count: {completed_frames}/{requested_frames}"
        )

    curves = {
        name: np.asarray(values)[:completed_frames].copy()
        for name, values in state["curves"].items()
    }
    inner_residual = np.asarray(state["inner_residual"])[:completed_frames].copy()
    inner_energy = np.asarray(state["inner_energy"])[:completed_frames].copy()
    line_search_alpha = np.asarray(state["line_search_alpha"])[:completed_frames].copy()
    line_search_trials = np.asarray(state["line_search_trials"])[:completed_frames].copy()
    minres_iterations = np.asarray(state["minres_iterations"])[:completed_frames].copy()
    minres_relative_residual = np.asarray(
        state["minres_relative_residual"]
    )[:completed_frames].copy()
    minimum_curvature = np.asarray(state["minimum_curvature"])[:completed_frames].copy()
    frame_rows = [
        dict(row) for row in state["frame_rows"] if int(row["frame"]) < completed_frames
    ]
    inner_rows = [
        dict(row) for row in state["inner_rows"] if int(row["frame"]) < completed_frames
    ]

    trajectory_frames = np.asarray(state["trajectory_frames"], dtype=np.int64)
    trajectory_positions = np.stack(state["trajectory_positions"]).astype(np.float32)
    if (
        trajectory_frames.ndim != 1
        or len(trajectory_frames) != len(trajectory_positions)
        or trajectory_frames[0] != 0
        or trajectory_frames[-1] > completed_frames
    ):
        raise ValueError("checkpoint trajectory is inconsistent with next_frame")

    first_invalid = _first_invalid_frame(frame_rows)
    if args.safe_physical_frame is None:
        safe_mask = _safe_trajectory_mask(trajectory_frames, first_invalid)
        safe_prefix_selection = "last saved state before first hard failure"
    else:
        if args.safe_physical_frame > int(trajectory_frames[-1]):
            raise ValueError(
                "--safe-physical-frame exceeds the last saved physical frame"
            )
        safe_mask = trajectory_frames <= int(args.safe_physical_frame)
        if not bool(safe_mask.any()):
            safe_mask[0] = True
        safe_prefix_selection = "explicit visually intact diagnostic prefix"
    safe_frames = trajectory_frames[safe_mask]
    safe_positions = trajectory_positions[safe_mask]
    invalid_count = sum(not bool(row.get("selected_valid", False)) for row in frame_rows)
    valid_count = len(frame_rows) - invalid_count

    metrics = {
        "solver": str(state["variant"]),
        "visualization_title": (
            f"Stopped partial {state['variant']}, initial={state['initial_guess']} "
            f"— typical 0"
        ),
        "motion_id": "typical_00_horizontal_gravity_release",
        "completed": True,
        "rollout_completed": False,
        "stopped_early": True,
        "stop_reason": "sustained solver and geometric divergence",
        "requested_rollout_frames": requested_frames,
        "completed_physical_frames": completed_frames,
        "saved_physical_frame_range": [
            int(trajectory_frames[0]),
            int(trajectory_frames[-1]),
        ],
        "safe_prefix_physical_frame_range": [
            int(safe_frames[0]),
            int(safe_frames[-1]),
        ],
        "safe_prefix_selection": safe_prefix_selection,
        "first_invalid_solver_frame": first_invalid,
        "valid_frame_count": valid_count,
        "invalid_frame_count": invalid_count,
        "initial_guess": str(state["initial_guess"]),
        "initial_guess_formula": (
            "x_n + dt*v_n" if state["initial_guess"] == "inertia" else "x_n"
        ),
        "residual_ratio_tolerance": 1.0e-3,
        "absolute_residual_tolerance": 1.0e-10,
        "residual_ratio_median": _finite_quantile(curves["residual_ratio"], 0.5),
        "residual_ratio_p95": _finite_quantile(curves["residual_ratio"], 0.95),
        "solver_issue_frame_count": int(np.count_nonzero(curves["solver_issue"])),
        "selected_initial_frame_count": int(
            np.count_nonzero(curves["selected_initial"])
        ),
        "min_area_ratio": _finite_quantile(curves["area_min"], 0.0),
        "max_area_ratio": _finite_quantile(curves["area_max"], 1.0),
        "min_edge_ratio": _finite_quantile(curves["edge_min"], 0.0),
        "max_edge_ratio": _finite_quantile(curves["edge_max"], 1.0),
        "source_directory": str(source),
        "source_checkpoint": str(progress_path),
    }

    np.savez_compressed(result_dir / "curves.npz", **curves)
    np.savez_compressed(
        result_dir / "inner_history.npz",
        residual_norm=inner_residual,
        energy=inner_energy,
        line_search_alpha=line_search_alpha,
        line_search_trials=line_search_trials,
        minres_iterations=minres_iterations,
        minres_relative_residual=minres_relative_residual,
        minimum_observed_curvature=minimum_curvature,
    )
    np.savez_compressed(
        result_dir / "trajectory.npz",
        frames=trajectory_frames,
        positions=trajectory_positions,
    )
    _write_json(result_dir / "metrics.json", metrics)
    _write_csv(result_dir / "per_frame.csv", frame_rows)
    _write_csv(result_dir / "inner_iterations.csv", inner_rows)

    model = load_model_spec(args.fixed_data_dir / "model_spec.json")
    mesh = load_tshirt_mesh(_resolve_mesh_path(args.fixed_data_dir, model.mesh_path))
    if mesh.sha256 != model.mesh_sha256:
        raise ValueError("rendering OBJ hash differs from the fixed model")
    fixed_indices = np.asarray(model.fixed_indices, dtype=np.int64)

    plots = plot_diagnostics(
        curves,
        figures_dir,
        residual_ratio_tolerance=float(metrics["residual_ratio_tolerance"]),
    )
    safe_keyframes = figures_dir / "04_safe_prefix_keyframes.png"
    plot_keyframes(
        positions=safe_positions,
        physical_frames=safe_frames,
        faces=mesh.faces,
        fixed_indices=fixed_indices,
        output=safe_keyframes,
        title=f"{metrics['visualization_title']} — safe prefix",
    )
    plots.append(safe_keyframes)

    full_video = output / "motion_stopped_partial.mp4"
    full_poster = output / "motion_stopped_partial_final.png"
    full_render = render_mp4(
        positions=trajectory_positions,
        camera_positions=safe_positions,
        faces=mesh.faces,
        fixed_indices=fixed_indices,
        output=full_video,
        poster=full_poster,
        fps=args.fps,
        width=args.width,
        height=args.height,
        headless=args.headless,
        egl_device_index=args.egl_device_index,
        crf=args.video_crf,
    )
    safe_video = output / "motion_safe_prefix.mp4"
    safe_poster = output / "motion_safe_prefix_final.png"
    safe_render = render_mp4(
        positions=safe_positions,
        camera_positions=safe_positions,
        faces=mesh.faces,
        fixed_indices=fixed_indices,
        output=safe_video,
        poster=safe_poster,
        fps=args.fps,
        width=args.width,
        height=args.height,
        headless=args.headless,
        egl_device_index=args.egl_device_index,
        crf=args.video_crf,
    )

    _write_json(
        output / "render_manifest.json",
        {
            "completed": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "partial_result": metrics,
            "source": {
                "directory": str(source),
                "checkpoint": str(progress_path),
            },
            "figures": [str(path) for path in plots],
            "full_stopped_partial": {
                **full_render,
                "video": str(full_video),
                "final_frame": str(full_poster),
                "physical_frame_range": [
                    int(trajectory_frames[0]),
                    int(trajectory_frames[-1]),
                ],
                "camera_reference_physical_frame_range": [
                    int(safe_frames[0]),
                    int(safe_frames[-1]),
                ],
            },
            "safe_prefix": {
                **safe_render,
                "video": str(safe_video),
                "final_frame": str(safe_poster),
                "physical_frame_range": [
                    int(safe_frames[0]),
                    int(safe_frames[-1]),
                ],
            },
        },
    )
    print(
        f"finalized {state['variant']}: frames={completed_frames}, "
        f"first_invalid={first_invalid}, safe_video_end={safe_frames[-1]}",
        flush=True,
    )
    print(f"full stopped video: {full_video}", flush=True)
    print(f"safe-prefix video: {safe_video}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
