from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import physical_config_from_dict
from .constants import (
    DEFAULT_DEVICE,
    FIXED_VERTEX_INDICES,
    NUM_PARTICLES,
    SPATIAL_DIM,
    SPRING_EDGES,
    TORCH_DTYPE,
    TRIANGLE_FACES,
)
from .io import create_output_directory, load_json, resolve_device, save_json, validate_device
from .model import MLPOptimizer
from .rollout import FIXED_INNER_ITERATIONS, diagnostics_summary, run_rollout
from .viz import plot_error_curves, render_video

DEFAULT_FRAMES = 500
DEFAULT_FPS = 25


def parse_args(*, script_file: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a fixed-50-iteration, 500-frame continuous solver comparison"
    )
    default_training_output = (
        script_file.resolve().parent / "fixed_left_edge_5x5_cloth_multi_motion_train_compare"
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=default_training_output,
        help="Output directory created by fixed_left_edge_5x5_cloth_multi_motion_train_compare.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional MLP checkpoint. Default: "
            "<training-output-dir>/multi_motion/best_validation_model_state_dict.pt"
        ),
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def validate_rollout_args(args: argparse.Namespace) -> None:
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


def main(*, script_file: Path) -> None:
    args = parse_args(script_file=script_file)
    validate_rollout_args(args)
    training_output_dir = args.training_output_dir.resolve()
    output_dir = create_output_directory(script_file=script_file)
    device = resolve_device(str(args.device))
    validate_device(device)

    runtime = load_json(training_output_dir / "runtime_config.json")
    physical = physical_config_from_dict(runtime["physical_config"])
    gd_selection = load_json(training_output_dir / "gradient_descent_step_selection.json")
    gd_step_size = float(gd_selection["selected_step_size"])
    hard_case = load_json(training_output_dir / "hard_case_selection.json")
    selected_state = hard_case["selected_physical_state"]

    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else training_output_dir / "multi_motion" / "best_validation_model_state_dict.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"MLP checkpoint not found: {checkpoint}")

    residual_length_scale = float(runtime["runtime_config"]["residual_length_scale"])
    model = MLPOptimizer(residual_length_scale).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    initial_p = tensor_state(
        selected_state["p_n_full"],
        device=torch.device("cpu"),
        shape=(NUM_PARTICLES, SPATIAL_DIM),
    )
    initial_v = tensor_state(
        selected_state["v_n_full"],
        device=torch.device("cpu"),
        shape=(NUM_PARTICLES, SPATIAL_DIM),
    )
    fixed = list(FIXED_VERTEX_INDICES)
    initial_p[fixed, :] = torch.tensor(physical.fixed_positions, dtype=TORCH_DTYPE)
    initial_v[fixed, :] = 0.0

    print(f"Training output: {training_output_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(
        "Selected hard OOD case: "
        f"motion={hard_case['motion_index']} ({hard_case['motion_name']}), "
        f"problem={hard_case['problem_index']}, time={hard_case['physical_time']:.3f}s, "
        f"initial residual max={hard_case['initial_residual_max']:.3e}"
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
        "physical_config": asdict(physical),
        "fixed_vertex_indices": list(FIXED_VERTEX_INDICES),
        "spring_edges": [list(edge) for edge in SPRING_EDGES],
        "triangle_faces": [list(face) for face in TRIANGLE_FACES],
        "hard_case_selection": hard_case,
        "gradient_descent_step_size": gd_step_size,
        "frames": int(args.frames),
        "physical_duration": int(args.frames) * physical.dt,
        "fixed_inner_iterations": FIXED_INNER_ITERATIONS,
        "same_iteration_budget_for_mlp_gd_newton": True,
        "convergence_early_stopping": False,
        "total_wall_seconds": rollout["total_elapsed_seconds"],
        "solver_timing": {
            name: diagnostics_summary(records) for name, records in rollout["diagnostics"].items()
        },
        "error_summary_against_reference": error_summary,
        "video_path": str(video_path) if video_path is not None else None,
        "trajectory_file": str(output_dir / "rollout_trajectories.npz"),
    }
    save_json(report, output_dir / "rollout_metrics.json")
    save_json(
        {
            "selected_problem_index": hard_case["problem_index"],
            "selected_motion_index": hard_case["motion_index"],
            "selected_motion_name": hard_case["motion_name"],
            "selected_physical_time": hard_case["physical_time"],
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
