"""Run safeguarded Newton variants and keep the best valid iterate per frame."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import torch

from cloth02_batched_physics import load_frozen_motion_batch, load_physics
from cloth04_reference_free_validation import FailureThresholds, detect_failures
from cloth09_rollout_single_motion import select_motion, split_path
from cloth25_rollout_newton_single_motion import (
    LinearSolveResult,
    _newton_step,
)
from tshirt_config import DEFAULT_FIXED_DATA_DIR, write_json


VARIANTS = (
    "raw_best",
    "newton_linesearch_best",
    "spd_block_linesearch_best",
)
DEFAULT_BASE = Path(
    "cloth_tshirt_pipeline/tensor_parallel/"
    "activation_relu_depth_01_width_39936_no_bias_lr5e-8/seed_42/"
    "single_motion_rollouts_newton"
)
DEFAULT_INERTIA_BASE = Path(
    "cloth_tshirt_pipeline/tensor_parallel/"
    "activation_relu_depth_01_width_39936_no_bias_lr5e-8/seed_42/"
    "single_motion_rollouts_newton_inertia"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variant", choices=VARIANTS, default="raw_best")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--split", choices=("typical", "validation", "test"), default="typical")
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--rollout-frames", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--trajectory-stride", type=int, default=5)
    parser.add_argument(
        "--initial-guess",
        choices=("current", "inertia"),
        default="current",
        help="'inertia' starts each solve at x_n + dt*v_n",
    )
    parser.add_argument("--residual-ratio-tolerance", type=float, default=1e-3)
    parser.add_argument("--absolute-residual-tolerance", type=float, default=1e-10)
    parser.add_argument("--minres-max-iterations", type=int, default=500)
    parser.add_argument("--minres-relative-tolerance", type=float, default=1e-2)
    parser.add_argument("--minres-absolute-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--minres-preconditioner",
        choices=("none", "mass", "block3x3"),
        default="block3x3",
    )
    parser.add_argument("--line-search-max-trials", type=int, default=12)
    parser.add_argument("--line-search-reduction", type=float, default=0.5)
    parser.add_argument("--armijo-c1", type=float, default=1e-4)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--render-width", type=int, default=1280)
    parser.add_argument("--render-height", type=int, default=720)
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--egl-device-index", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        base = (
            DEFAULT_INERTIA_BASE
            if args.initial_guess == "inertia"
            else DEFAULT_BASE
        )
        args.output_dir = base / f"typical_0000_{args.variant}"
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.rollout_frames,
        args.inner_steps,
        args.trajectory_stride,
        args.minres_max_iterations,
        args.line_search_max_trials,
        args.progress_interval,
        args.render_fps,
        args.render_width,
        args.render_height,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("frame, iteration, line-search, and render counts must be positive")
    if args.minres_relative_tolerance <= 0.0 or args.minres_absolute_tolerance <= 0.0:
        raise ValueError("MINRES tolerances must be positive")
    if not 0.0 < args.residual_ratio_tolerance < 1.0:
        raise ValueError("--residual-ratio-tolerance must be in (0, 1)")
    if args.absolute_residual_tolerance <= 0.0:
        raise ValueError("--absolute-residual-tolerance must be positive")
    if not 0.0 < args.line_search_reduction < 1.0:
        raise ValueError("--line-search-reduction must be in (0, 1)")
    if not 0.0 < args.armijo_c1 < 1.0:
        raise ValueError("--armijo-c1 must be in (0, 1)")
    if not 0 <= args.video_crf <= 51:
        raise ValueError("--video-crf must be in [0, 51]")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_progress(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def _finite_quantile(values: np.ndarray, quantile: float, default: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if finite.size else float(default)


def _initial_iterate(
    *,
    physics: Any,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    fixed_targets: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "current":
        guess = positions
    elif mode == "inertia":
        guess = positions + physics.dt * velocities
    else:
        raise ValueError(f"unsupported initial-guess mode: {mode}")
    return physics.project_positions(guess, fixed_targets)


def _state_diagnostics(
    *,
    physics: Any,
    positions: torch.Tensor,
    q: torch.Tensor,
    fixed_targets: torch.Tensor,
    thresholds: FailureThresholds,
) -> tuple[float, float, bool, list[str], dict[str, float]]:
    residual_tensor = physics.stationarity_residual_norm(
        positions, q, fixed_targets
    ).detach()
    energy_tensor = physics.variational_energy(
        positions, q, fixed_targets
    ).detach()
    bad, reasons, geometry = detect_failures(
        physics, positions, residual_tensor, fixed_targets, thresholds
    )
    diagnostics = {
        name: float(geometry[name][0].item())
        for name in ("area_min", "area_max", "edge_min", "edge_max")
    }
    return (
        float(residual_tensor.item()),
        float(energy_tensor.item()),
        not bool(bad[0]),
        list(reasons[0]),
        diagnostics,
    )


def _block_spd_step(
    *,
    physics: Any,
    y: torch.Tensor,
    q: torch.Tensor,
    fixed_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LinearSolveResult]:
    gradient = physics.stationarity_residual(y, q, fixed_targets).detach()
    energy = physics.variational_energy(y, q, fixed_targets).detach()
    try:
        direction = physics.block_hessian_preconditioned_residual(y, gradient)
    except torch.linalg.LinAlgError:
        result = LinearSolveResult(
            step=torch.zeros_like(y),
            iterations=0,
            relative_residual=math.inf,
            converged=False,
            breakdown=True,
            minimum_curvature=math.nan,
            preconditioner_fallback=False,
        )
        return y, gradient, energy, result
    step = -direction
    candidate = physics.project_positions(y + step, fixed_targets)
    result = LinearSolveResult(
        step=step,
        iterations=0,
        relative_residual=math.nan,
        converged=True,
        breakdown=False,
        minimum_curvature=math.nan,
    )
    return candidate, gradient, energy, result


def _armijo_line_search(
    *,
    physics: Any,
    y: torch.Tensor,
    step: torch.Tensor,
    gradient: torch.Tensor,
    energy: torch.Tensor,
    q: torch.Tensor,
    fixed_targets: torch.Tensor,
    max_trials: int,
    reduction: float,
    armijo_c1: float,
) -> tuple[torch.Tensor, bool, float, int, str]:
    slope = float(torch.sum(gradient * step).item())
    if not math.isfinite(slope) or slope >= 0.0:
        return y, False, 0.0, 0, "non_descent_direction"
    alpha = 1.0
    energy0 = float(energy.item())
    for trial in range(1, max_trials + 1):
        candidate = physics.project_positions(y + alpha * step, fixed_targets)
        if bool(torch.isfinite(candidate).all()):
            candidate_energy = float(
                physics.variational_energy(candidate, q, fixed_targets).item()
            )
            if math.isfinite(candidate_energy) and (
                candidate_energy <= energy0 + armijo_c1 * alpha * slope
            ):
                return candidate, True, alpha, trial, ""
        alpha *= reduction
    return y, False, 0.0, max_trials, "armijo_rejected"


def _plot_variant_diagnostics(
    *,
    output: Path,
    curves: dict[str, np.ndarray],
    inner_residual: np.ndarray,
    line_search_alpha: np.ndarray,
    line_search_trials: np.ndarray,
    minres_iterations: np.ndarray,
) -> list[Path]:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    frames = np.arange(len(curves["residual_ratio"]))
    paths: list[Path] = []

    figure, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
    valid = np.isfinite(curves["residual_ratio"]) & (curves["residual_ratio"] > 0)
    axes[0].semilogy(
        frames[valid],
        curves["residual_ratio"][valid],
        label="selected best / initial",
    )
    raw_valid = np.isfinite(curves["last_iterate_residual_ratio"]) & (
        curves["last_iterate_residual_ratio"] > 0
    )
    axes[0].semilogy(
        frames[raw_valid],
        curves["last_iterate_residual_ratio"][raw_valid],
        label="last attempted iterate / initial",
        alpha=0.8,
    )
    axes[1].plot(frames, curves["selected_iteration"], label="selected iteration")
    axes[1].plot(frames, curves["attempted_steps"], label="attempted steps")
    axes[2].plot(frames, curves["solver_issue"], label="solver issue")
    axes[2].plot(frames, curves["selected_initial"], label="selected initial state")
    axes[0].set(ylabel="residual ratio", title="Best-iterate safeguard")
    axes[1].set(ylabel="iteration")
    axes[2].set(xlabel="physical frame", ylabel="event")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    path = output / "05_best_iterate_selection.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    if bool(np.isfinite(line_search_alpha).any()):
        figure, axes = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True)
        accepted = np.sum(np.isfinite(line_search_alpha) & (line_search_alpha > 0), axis=1)
        rejected = np.sum(np.isfinite(line_search_alpha) & (line_search_alpha == 0), axis=1)
        axes[0].plot(frames, accepted, label="accepted")
        axes[0].plot(frames, rejected, label="rejected")
        positive = np.where(line_search_alpha > 0, line_search_alpha, np.nan)
        with np.errstate(invalid="ignore"):
            minimum = np.full(len(frames), np.nan)
            mean = np.full(len(frames), np.nan)
            for frame in frames:
                values = positive[frame]
                values = values[np.isfinite(values)]
                if values.size:
                    minimum[frame] = float(np.min(values))
                    mean[frame] = float(np.mean(values))
        axes[1].semilogy(frames, mean, label="mean accepted alpha")
        axes[1].semilogy(frames, minimum, label="minimum accepted alpha")
        axes[1].plot(
            frames,
            np.sum(line_search_trials, axis=1),
            label="total trials",
            alpha=0.7,
        )
        axes[0].set(ylabel="Newton steps", title="Armijo line search")
        axes[1].set(xlabel="physical frame", ylabel="alpha / trials")
        for axis in axes:
            axis.grid(True, which="both", alpha=0.25)
            axis.legend()
        path = output / "06_line_search.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)

    if bool((minres_iterations > 0).any()):
        figure, axes = plt.subplots(2, 1, figsize=(10, 7.0), sharex=True)
        axes[0].plot(
            frames,
            np.sum(minres_iterations, axis=1),
            label="total MINRES iterations",
        )
        ratios = np.full(len(frames), np.nan)
        for frame in frames:
            values = inner_residual[frame]
            values = values[np.isfinite(values) & (values > 0)]
            if values.size:
                ratios[frame] = float(np.min(values) / values[0])
        axes[1].semilogy(frames, ratios, label="minimum inner residual ratio")
        axes[0].set(ylabel="iterations", title="Linear solve and inner convergence")
        axes[1].set(xlabel="physical frame", ylabel="ratio")
        for axis in axes:
            axis.grid(True, which="both", alpha=0.25)
            axis.legend()
        path = output / "07_minres.png"
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths


def _visualization_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "cloth23_render_single_motion_rollout.py"),
        "--rollout-dir",
        str(args.output_dir.resolve()),
        "--fixed-data-dir",
        str(args.fixed_data_dir.resolve()),
        "--fps",
        str(args.render_fps),
        "--width",
        str(args.render_width),
        "--height",
        str(args.render_height),
        "--video-crf",
        str(args.video_crf),
        "--egl-device-index",
        str(args.egl_device_index),
    ]
    if args.headless is not None:
        command.append("--headless" if args.headless else "--no-headless")
    if args.overwrite:
        command.append("--overwrite")
    return command


def run(args: argparse.Namespace) -> None:
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    output = args.output_dir.resolve()
    result_dir = output / "newton"
    progress_path = result_dir / "progress_state.pt"
    resuming = args.resume and progress_path.is_file()
    if output.exists() and any(output.iterdir()) and not args.overwrite and not resuming:
        raise FileExistsError(f"output directory is not empty; use --overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    physics = load_physics(
        fixed_data_dir=args.fixed_data_dir,
        device=args.device,
        dtype=dtype,
    )
    dataset_path = split_path(args.fixed_data_dir, args.split)
    dataset = load_frozen_motion_batch(dataset_path, device=args.device, dtype=dtype)
    motion = select_motion(dataset, args.motion_index)
    thresholds = FailureThresholds()
    frames = args.rollout_frames
    inner = args.inner_steps

    curve_names = (
        "initial_residual",
        "final_residual",
        "residual_ratio",
        "first_step_ratio",
        "last_iterate_residual_ratio",
        "energy_change",
        "area_min",
        "area_max",
        "edge_min",
        "edge_max",
        "displacement_rms",
    )
    curves = {
        name: np.full(frames, np.nan, dtype=np.float64) for name in curve_names
    }
    curves["inner_steps"] = np.zeros(frames, dtype=np.int64)
    curves["objective_evaluations"] = np.zeros(frames, dtype=np.int64)
    curves["converged"] = np.zeros(frames, dtype=np.bool_)
    curves["selected_iteration"] = np.full(frames, -1, dtype=np.int64)
    curves["attempted_steps"] = np.zeros(frames, dtype=np.int64)
    curves["solver_issue"] = np.zeros(frames, dtype=np.bool_)
    curves["selected_initial"] = np.zeros(frames, dtype=np.bool_)

    inner_residual = np.full((frames, inner + 1), np.nan, dtype=np.float64)
    inner_energy = np.full((frames, inner + 1), np.nan, dtype=np.float64)
    line_search_alpha = np.full((frames, inner), np.nan, dtype=np.float64)
    line_search_trials = np.zeros((frames, inner), dtype=np.int64)
    minres_iterations = np.zeros((frames, inner), dtype=np.int64)
    minres_relative_residual = np.full((frames, inner), np.nan, dtype=np.float64)
    minimum_curvature = np.full((frames, inner), np.nan, dtype=np.float64)

    p = motion.positions.detach().clone()
    v = motion.velocities.detach().clone()
    fixed_targets = motion.positions.detach().clone()
    trajectory_frames = [0]
    trajectory_positions = [p[0].detach().cpu().numpy()]
    frame_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    start_frame = 0
    if resuming:
        progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        if progress["variant"] != args.variant:
            raise ValueError(
                f"progress variant is {progress['variant']}, requested {args.variant}"
            )
        if progress.get("initial_guess", "current") != args.initial_guess:
            raise ValueError(
                "resume initial guess differs from saved progress: "
                f"{progress.get('initial_guess', 'current')} != {args.initial_guess}"
            )
        if int(progress["rollout_frames"]) != frames or int(
            progress["inner_steps"]
        ) != inner:
            raise ValueError("resume rollout shape differs from saved progress")
        start_frame = int(progress["next_frame"])
        p = progress["p"].to(device=physics.device, dtype=dtype)
        v = progress["v"].to(device=physics.device, dtype=dtype)
        curves = progress["curves"]
        inner_residual = progress["inner_residual"]
        inner_energy = progress["inner_energy"]
        line_search_alpha = progress["line_search_alpha"]
        line_search_trials = progress["line_search_trials"]
        minres_iterations = progress["minres_iterations"]
        minres_relative_residual = progress["minres_relative_residual"]
        minimum_curvature = progress["minimum_curvature"]
        trajectory_frames = progress["trajectory_frames"]
        trajectory_positions = progress["trajectory_positions"]
        frame_rows = progress["frame_rows"]
        inner_rows = progress["inner_rows"]
        print(
            f"resumed {args.variant} at physical frame {start_frame}/{frames}",
            flush=True,
        )
    started = time.perf_counter()

    for frame in range(start_frame, frames):
        q = physics.make_q(p, v)
        y = _initial_iterate(
            physics=physics,
            positions=p,
            velocities=v,
            fixed_targets=fixed_targets,
            mode=args.initial_guess,
        )
        start_positions = p.clone()
        initial_residual, initial_energy, initial_valid, _, initial_geometry = (
            _state_diagnostics(
                physics=physics,
                positions=y,
                q=q,
                fixed_targets=fixed_targets,
                thresholds=thresholds,
            )
        )
        inner_residual[frame, 0] = initial_residual
        inner_energy[frame, 0] = initial_energy
        best_y = y.detach().clone()
        best_residual = initial_residual
        best_energy = initial_energy
        best_iteration = -1
        best_geometry = initial_geometry
        best_valid = initial_valid
        first_step_ratio = math.nan
        last_residual = initial_residual
        attempted = 0
        issue_reason = ""
        objective_evaluations = 2

        for iteration in range(inner):
            if args.variant == "spd_block_linesearch_best":
                candidate, gradient, energy, linear_solve = _block_spd_step(
                    physics=physics,
                    y=y,
                    q=q,
                    fixed_targets=fixed_targets,
                )
            else:
                candidate, gradient, energy, linear_solve = _newton_step(
                    physics=physics,
                    y=y,
                    q=q,
                    fixed_targets=fixed_targets,
                    minres_max_iterations=args.minres_max_iterations,
                    minres_relative_tolerance=args.minres_relative_tolerance,
                    minres_absolute_tolerance=args.minres_absolute_tolerance,
                    minres_preconditioner=args.minres_preconditioner,
                )
            attempted += 1
            minres_iterations[frame, iteration] = linear_solve.iterations
            minres_relative_residual[frame, iteration] = (
                linear_solve.relative_residual
            )
            minimum_curvature[frame, iteration] = linear_solve.minimum_curvature
            objective_evaluations += 1 + linear_solve.iterations
            accepted = False
            alpha = 1.0
            trials = 0
            iteration_issue = ""

            if not linear_solve.converged or linear_solve.breakdown:
                iteration_issue = (
                    "linear_breakdown"
                    if linear_solve.breakdown
                    else "linear_nonconvergence"
                )
            elif args.variant == "raw_best":
                accepted = bool(torch.isfinite(candidate).all())
                if not accepted:
                    iteration_issue = "nonfinite_full_step"
            else:
                candidate, accepted, alpha, trials, iteration_issue = (
                    _armijo_line_search(
                        physics=physics,
                        y=y,
                        step=linear_solve.step,
                        gradient=gradient,
                        energy=energy,
                        q=q,
                        fixed_targets=fixed_targets,
                        max_trials=args.line_search_max_trials,
                        reduction=args.line_search_reduction,
                        armijo_c1=args.armijo_c1,
                    )
                )
                line_search_alpha[frame, iteration] = alpha
                line_search_trials[frame, iteration] = trials
                objective_evaluations += trials

            candidate_residual = math.nan
            candidate_energy = math.nan
            candidate_valid = False
            candidate_reasons: list[str] = []
            candidate_geometry = {
                "area_min": math.nan,
                "area_max": math.nan,
                "edge_min": math.nan,
                "edge_max": math.nan,
            }
            if accepted:
                y = candidate.detach()
                (
                    candidate_residual,
                    candidate_energy,
                    candidate_valid,
                    candidate_reasons,
                    candidate_geometry,
                ) = _state_diagnostics(
                    physics=physics,
                    positions=y,
                    q=q,
                    fixed_targets=fixed_targets,
                    thresholds=thresholds,
                )
                objective_evaluations += 2
                last_residual = candidate_residual
                inner_residual[frame, iteration + 1] = candidate_residual
                inner_energy[frame, iteration + 1] = candidate_energy
                if iteration == 0:
                    first_step_ratio = candidate_residual / max(
                        initial_residual, 1e-30
                    )
                if (
                    candidate_valid
                    and math.isfinite(candidate_residual)
                    and (not best_valid or candidate_residual < best_residual)
                ):
                    best_y = y.detach().clone()
                    best_residual = candidate_residual
                    best_energy = candidate_energy
                    best_iteration = iteration
                    best_geometry = candidate_geometry
                    best_valid = True

            inner_rows.append(
                {
                    "frame": frame,
                    "iteration": iteration,
                    "residual_before": float(
                        torch.linalg.vector_norm(gradient).item()
                    ),
                    "energy_before": float(energy.item()),
                    "accepted": accepted,
                    "alpha": alpha if accepted else 0.0,
                    "line_search_trials": trials,
                    "candidate_residual": candidate_residual,
                    "candidate_energy": candidate_energy,
                    "candidate_valid": candidate_valid,
                    "candidate_failure_reasons": "+".join(candidate_reasons),
                    "best_residual_after": best_residual,
                    "best_iteration_after": best_iteration,
                    "minres_iterations": linear_solve.iterations,
                    "minres_relative_residual": linear_solve.relative_residual,
                    "minimum_observed_curvature": linear_solve.minimum_curvature,
                    "preconditioner_fallback": (
                        linear_solve.preconditioner_fallback
                    ),
                    "issue": iteration_issue,
                }
            )
            if iteration_issue:
                issue_reason = iteration_issue
                break

        selected_residual, selected_energy, selected_valid, selected_reasons, geometry = (
            _state_diagnostics(
                physics=physics,
                positions=best_y,
                q=q,
                fixed_targets=fixed_targets,
                thresholds=thresholds,
            )
        )
        objective_evaluations += 2
        ratio = selected_residual / max(initial_residual, 1e-30)
        curves["initial_residual"][frame] = initial_residual
        curves["final_residual"][frame] = selected_residual
        curves["residual_ratio"][frame] = ratio
        curves["first_step_ratio"][frame] = first_step_ratio
        curves["last_iterate_residual_ratio"][frame] = last_residual / max(
            initial_residual, 1e-30
        )
        curves["energy_change"][frame] = selected_energy - initial_energy
        curves["inner_steps"][frame] = attempted
        curves["objective_evaluations"][frame] = objective_evaluations
        curves["selected_iteration"][frame] = best_iteration
        curves["attempted_steps"][frame] = attempted
        curves["solver_issue"][frame] = bool(issue_reason)
        curves["selected_initial"][frame] = best_iteration < 0
        curves["converged"][frame] = selected_residual <= max(
            args.absolute_residual_tolerance,
            initial_residual * args.residual_ratio_tolerance,
        )
        curves["displacement_rms"][frame] = float(
            torch.sqrt(
                torch.mean(torch.sum((best_y - start_positions) ** 2, dim=-1))
            ).item()
        )
        for name in ("area_min", "area_max", "edge_min", "edge_max"):
            curves[name][frame] = geometry[name]

        frame_rows.append(
            {
                "frame": frame,
                "initial_residual": initial_residual,
                "selected_residual": selected_residual,
                "selected_residual_ratio": ratio,
                "last_iterate_residual_ratio": curves[
                    "last_iterate_residual_ratio"
                ][frame],
                "selected_iteration": best_iteration,
                "selected_initial": best_iteration < 0,
                "selected_valid": selected_valid,
                "selected_failure_reasons": "+".join(selected_reasons),
                "attempted_steps": attempted,
                "solver_issue": issue_reason,
                "energy_change": selected_energy - initial_energy,
                **geometry,
            }
        )
        print(
            f"frame={frame:04d} selected_ratio={ratio:.3e} "
            f"best_iteration={best_iteration:02d} attempted={attempted:02d} "
            f"issue={issue_reason or 'none'}",
            flush=True,
        )

        p, v = physics.advance_state(p, best_y, fixed_targets)
        physical_frame = frame + 1
        if physical_frame % args.trajectory_stride == 0 or physical_frame == frames:
            trajectory_frames.append(physical_frame)
            trajectory_positions.append(p[0].detach().cpu().numpy())
        if physical_frame % args.progress_interval == 0 or physical_frame == frames:
            _save_progress(
                progress_path,
                {
                    "format_version": 1,
                    "variant": args.variant,
                    "initial_guess": args.initial_guess,
                    "rollout_frames": frames,
                    "inner_steps": inner,
                    "next_frame": physical_frame,
                    "p": p.detach().cpu(),
                    "v": v.detach().cpu(),
                    "curves": curves,
                    "inner_residual": inner_residual,
                    "inner_energy": inner_energy,
                    "line_search_alpha": line_search_alpha,
                    "line_search_trials": line_search_trials,
                    "minres_iterations": minres_iterations,
                    "minres_relative_residual": minres_relative_residual,
                    "minimum_curvature": minimum_curvature,
                    "trajectory_frames": trajectory_frames,
                    "trajectory_positions": trajectory_positions,
                    "frame_rows": frame_rows,
                    "inner_rows": inner_rows,
                },
            )
            _write_csv(result_dir / "per_frame.partial.csv", frame_rows)
            _write_csv(result_dir / "inner_iterations.partial.csv", inner_rows)
            _atomic_write_json(
                result_dir / "progress.json",
                {
                    "completed_frames": physical_frame,
                    "rollout_frames": frames,
                    "variant": args.variant,
                    "initial_guess": args.initial_guess,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

    evaluated = np.isfinite(curves["residual_ratio"])
    valid_minres = np.isfinite(minres_relative_residual)
    accepted_line_search = np.isfinite(line_search_alpha) & (line_search_alpha > 0)
    rejected_line_search = np.isfinite(line_search_alpha) & (line_search_alpha == 0)
    metrics = {
        "solver": args.variant,
        "visualization_title": (
            f"Newton variant: {args.variant}, initial={args.initial_guess} — typical 0"
        ),
        "motion_id": motion.motion_ids[0],
        "completed": True,
        "rollout_completed": True,
        "rollout_frames": frames,
        "survival_frames": frames,
        "best_iterate_selection": True,
        "best_iterate_metric": (
            "minimum stationarity residual among finite iterates passing "
            "reference-free failure checks"
        ),
        "initial_iterate_is_candidate": True,
        "initial_guess": args.initial_guess,
        "initial_guess_formula": (
            "x_n + dt*v_n" if args.initial_guess == "inertia" else "x_n"
        ),
        "damping": False,
        "line_search": args.variant != "raw_best",
        "positive_definite_projection": args.variant
        == "spd_block_linesearch_best",
        "hessian": (
            "SPD-projected 3x3 block-diagonal approximation"
            if args.variant == "spd_block_linesearch_best"
            else "exact autograd Hessian-vector products"
        ),
        "residual_ratio_tolerance": args.residual_ratio_tolerance,
        "residual_ratio_definition": (
            "selected stationarity-residual L2 norm / "
            "initial stationarity-residual L2 norm, per physical frame"
        ),
        "residual_ratio_p95_definition": (
            "numpy.quantile(frame residual ratios, 0.95, method='linear')"
        ),
        "residual_ratio_median": _finite_quantile(
            curves["residual_ratio"], 0.5, math.inf
        ),
        "residual_ratio_p95": _finite_quantile(
            curves["residual_ratio"], 0.95, math.inf
        ),
        "converged_frame_count": int(curves["converged"].sum()),
        "converged_frame_fraction": float(curves["converged"].mean()),
        "solver_issue_frame_count": int(curves["solver_issue"].sum()),
        "selected_initial_frame_count": int(curves["selected_initial"].sum()),
        "selected_iteration_median": _finite_quantile(
            curves["selected_iteration"], 0.5, -1.0
        ),
        "energy_increase_fraction": float(
            np.mean(curves["energy_change"][evaluated] > 0.0)
        ),
        "min_area_ratio": _finite_quantile(curves["area_min"], 0.0, 0.0),
        "max_area_ratio": _finite_quantile(
            curves["area_max"], 1.0, math.inf
        ),
        "min_edge_ratio": _finite_quantile(curves["edge_min"], 0.0, 0.0),
        "max_edge_ratio": _finite_quantile(
            curves["edge_max"], 1.0, math.inf
        ),
        "line_search_accepted_step_count": int(accepted_line_search.sum()),
        "line_search_rejected_step_count": int(rejected_line_search.sum()),
        "line_search_alpha_min": _finite_quantile(
            line_search_alpha[accepted_line_search], 0.0, math.nan
        ),
        "minres_solve_count": int(valid_minres.sum()),
        "minres_iterations_total": int(minres_iterations.sum()),
        "minres_relative_residual_p95": _finite_quantile(
            minres_relative_residual, 0.95, math.nan
        ),
        "preconditioner_fallback_count": int(
            sum(bool(row.get("preconditioner_fallback", False)) for row in inner_rows)
        ),
        "minimum_observed_curvature": _finite_quantile(
            minimum_curvature, 0.0, math.nan
        ),
        "wall_seconds": time.perf_counter() - started,
        "dtype": args.dtype,
        "device": args.device,
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
        frames=np.asarray(trajectory_frames, dtype=np.int64),
        positions=np.stack(trajectory_positions).astype(np.float32),
    )
    write_json(result_dir / "metrics.json", metrics)
    _write_csv(result_dir / "per_frame.csv", frame_rows)
    _write_csv(result_dir / "inner_iterations.csv", inner_rows)
    plots = _plot_variant_diagnostics(
        output=output / "figures",
        curves=curves,
        inner_residual=inner_residual,
        line_search_alpha=line_search_alpha,
        line_search_trials=line_search_trials,
        minres_iterations=minres_iterations,
    )
    _atomic_write_json(
        output / "manifest.json",
        {
            "format_version": 1,
            "completed": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                **vars(args),
                "output_dir": str(output),
                "fixed_data_dir": str(Path(args.fixed_data_dir).resolve()),
                "dataset": str(dataset_path.resolve()),
                "thresholds": asdict(thresholds),
            },
            "result": metrics,
            "diagnostic_plots": [str(path) for path in plots],
        },
    )
    print(
        f"{args.variant}: completed {frames} frames; "
        f"ratio_p95={metrics['residual_ratio_p95']:.3e}; "
        f"issues={metrics['solver_issue_frame_count']}",
        flush=True,
    )
    print(f"result written to {output}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    try:
        run(args)
        if args.visualize:
            completed = subprocess.run(_visualization_command(args))
            raise SystemExit(completed.returncode)
    except Exception as error:
        try:
            _atomic_write_json(
                args.output_dir.resolve() / "failure.json",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
