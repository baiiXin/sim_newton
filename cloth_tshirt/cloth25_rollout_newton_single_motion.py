"""Run typical 0 with raw matrix-free Newton-MINRES and no globalization."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Sequence

import numpy as np
import torch

from cloth02_batched_physics import (
    FrozenMotionBatch,
    TShirtPhysics,
    load_frozen_motion_batch,
    load_physics,
)
from cloth04_reference_free_validation import FailureThresholds, detect_failures
from cloth09_rollout_single_motion import select_motion, split_path
from tshirt_config import DEFAULT_FIXED_DATA_DIR, write_json


DEFAULT_OUTPUT = Path(
    "cloth_tshirt_pipeline/tensor_parallel/"
    "activation_relu_depth_01_width_39936_no_bias_lr5e-8/seed_42/"
    "single_motion_rollouts_newton/"
    "typical_0000_newton_minres_no_damping_no_linesearch"
)


@dataclass(frozen=True)
class LinearSolveResult:
    step: torch.Tensor
    iterations: int
    relative_residual: float
    converged: bool
    breakdown: bool
    minimum_curvature: float
    preconditioner_fallback: bool = False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--split", choices=("typical", "validation", "test"), default="typical")
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--rollout-frames", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--residual-ratio-tolerance", type=float, default=1e-3)
    parser.add_argument("--absolute-residual-tolerance", type=float, default=1e-10)
    parser.add_argument("--trajectory-stride", type=int, default=5)
    parser.add_argument("--minres-max-iterations", type=int, default=200)
    parser.add_argument(
        "--minres-relative-tolerance",
        type=float,
        default=1e-2,
        help="inexact-Newton forcing tolerance for the linear solve",
    )
    parser.add_argument("--minres-absolute-tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--minres-preconditioner",
        choices=("none", "mass", "block3x3"),
        default="block3x3",
        help="linear-solver preconditioner; does not modify the Newton Hessian",
    )
    parser.add_argument(
        "--stop-on-minres-nonconvergence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="treat an inaccurate Newton linear solve as a solver failure",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--render-fps", type=int, default=30)
    parser.add_argument("--render-frame-hold", type=int, default=1)
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
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.rollout_frames <= 0 or args.inner_steps <= 0 or args.trajectory_stride <= 0:
        raise ValueError("rollout frames, inner steps, and trajectory stride must be positive")
    if args.minres_max_iterations <= 0:
        raise ValueError("--minres-max-iterations must be positive")
    if (
        args.minres_relative_tolerance <= 0.0
        or args.minres_absolute_tolerance <= 0.0
    ):
        raise ValueError("MINRES tolerances must be positive")
    if not 0.0 < args.residual_ratio_tolerance < 1.0:
        raise ValueError("--residual-ratio-tolerance must be in (0, 1)")
    if args.absolute_residual_tolerance <= 0.0:
        raise ValueError("--absolute-residual-tolerance must be positive")
    if (
        args.render_fps <= 0
        or args.render_frame_hold <= 0
        or args.render_width <= 0
        or args.render_height <= 0
    ):
        raise ValueError("render FPS, frame hold, and dimensions must be positive")
    if not 0 <= args.video_crf <= 51:
        raise ValueError("--video-crf must be in [0, 51]")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _minimum_residual(
    *,
    hvp: Callable[[torch.Tensor], torch.Tensor],
    preconditioner: Callable[[torch.Tensor], torch.Tensor],
    right_hand_side: torch.Tensor,
    max_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> LinearSolveResult:
    """Solve a symmetric, possibly indefinite system with preconditioned MINRES."""

    solution = torch.zeros_like(right_hand_side)
    r1 = right_hand_side.detach().clone()
    y = preconditioner(r1)
    beta1_squared = torch.sum(r1 * y)
    initial_norm = float(torch.linalg.vector_norm(right_hand_side).item())
    target = max(float(absolute_tolerance), float(relative_tolerance) * initial_norm)
    if not math.isfinite(initial_norm) or not bool(torch.isfinite(beta1_squared)):
        return LinearSolveResult(solution, 0, math.inf, False, True, math.nan)
    if initial_norm <= target:
        return LinearSolveResult(solution, 0, 0.0, True, False, math.inf)
    if float(beta1_squared.item()) <= 0.0:
        return LinearSolveResult(solution, 0, 1.0, False, True, math.nan)

    beta1 = torch.sqrt(beta1_squared)
    old_beta = torch.zeros((), dtype=right_hand_side.dtype, device=right_hand_side.device)
    beta = beta1
    diagonal_bar = torch.zeros_like(beta)
    epsilon_line = torch.zeros_like(beta)
    phi_bar = beta1
    cosine = -torch.ones_like(beta)
    sine = torch.zeros_like(beta)
    w = torch.zeros_like(right_hand_side)
    w2 = torch.zeros_like(right_hand_side)
    r2 = r1
    minimum_curvature = math.inf

    for iteration in range(1, int(max_iterations) + 1):
        if float(beta.item()) <= torch.finfo(beta.dtype).tiny:
            residual = right_hand_side - hvp(solution)
            norm = float(torch.linalg.vector_norm(residual).item())
            relative = norm / max(initial_norm, 1e-30)
            converged = norm <= target
            return LinearSolveResult(
                solution.detach(),
                iteration - 1,
                relative,
                converged,
                not converged,
                minimum_curvature,
            )

        lanczos_vector = y / beta
        y = hvp(lanczos_vector)
        if iteration >= 2:
            y = y - (beta / old_beta) * r1
        alpha = torch.sum(lanczos_vector * y)
        vector_squared = torch.sum(lanczos_vector * lanczos_vector).clamp_min(
            torch.finfo(lanczos_vector.dtype).tiny
        )
        curvature = float((alpha / vector_squared).item())
        minimum_curvature = min(minimum_curvature, curvature)
        if not math.isfinite(curvature):
            return LinearSolveResult(
                solution.detach(),
                iteration - 1,
                math.inf,
                False,
                True,
                minimum_curvature,
            )

        y = y - (alpha / beta) * r2
        r1 = r2
        r2 = y
        y = preconditioner(r2)
        old_beta = beta
        beta_squared = torch.sum(r2 * y)
        if not bool(torch.isfinite(beta_squared)) or float(beta_squared.item()) < 0.0:
            return LinearSolveResult(
                solution.detach(),
                iteration,
                math.inf,
                False,
                True,
                minimum_curvature,
            )
        beta = torch.sqrt(beta_squared.clamp_min(0.0))

        old_epsilon_line = epsilon_line
        delta = cosine * diagonal_bar + sine * alpha
        diagonal = sine * diagonal_bar - cosine * alpha
        epsilon_line = sine * beta
        diagonal_bar = -cosine * beta
        gamma = torch.sqrt(diagonal * diagonal + beta * beta).clamp_min(
            torch.finfo(beta.dtype).eps
        )
        cosine = diagonal / gamma
        sine = beta / gamma
        phi = cosine * phi_bar
        phi_bar = sine * phi_bar

        w1 = w2
        w2 = w
        w = (lanczos_vector - old_epsilon_line * w1 - delta * w2) / gamma
        solution = solution + phi * w

        estimated_relative = abs(float((phi_bar / beta1).item()))
        should_check = (
            estimated_relative <= relative_tolerance
            or iteration % 25 == 0
            or iteration == int(max_iterations)
            or float(beta.item()) <= torch.finfo(beta.dtype).tiny
        )
        if should_check:
            residual = right_hand_side - hvp(solution)
            norm = float(torch.linalg.vector_norm(residual).item())
            if not math.isfinite(norm):
                return LinearSolveResult(
                    solution.detach(),
                    iteration,
                    math.inf,
                    False,
                    True,
                    minimum_curvature,
                )
            if norm <= target:
                return LinearSolveResult(
                    solution.detach(),
                    iteration,
                    norm / max(initial_norm, 1e-30),
                    True,
                    False,
                    minimum_curvature,
                )

    residual = right_hand_side - hvp(solution)
    final_norm = float(torch.linalg.vector_norm(residual).item())
    return LinearSolveResult(
        solution.detach(),
        int(max_iterations),
        final_norm / max(initial_norm, 1e-30),
        final_norm <= target,
        not math.isfinite(final_norm),
        minimum_curvature,
    )


def _newton_step(
    *,
    physics: TShirtPhysics,
    y: torch.Tensor,
    q: torch.Tensor,
    fixed_targets: torch.Tensor,
    minres_max_iterations: int,
    minres_relative_tolerance: float,
    minres_absolute_tolerance: float,
    minres_preconditioner: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, LinearSolveResult]:
    with torch.enable_grad():
        variable = y.detach().requires_grad_(True)
        energy = physics.variational_energy(variable, q, fixed_targets)
        (gradient,) = torch.autograd.grad(
            energy.sum(),
            variable,
            create_graph=True,
            retain_graph=True,
        )
        gate = physics.free_update_gate(gradient.shape[0], dtype=gradient.dtype)
        gradient = gradient * gate

        def hvp(value: torch.Tensor) -> torch.Tensor:
            (product,) = torch.autograd.grad(
                gradient,
                variable,
                grad_outputs=value,
                retain_graph=True,
                create_graph=False,
            )
            return product * gate

        if minres_preconditioner == "mass":
            preconditioner = physics.mass_preconditioned_residual
        elif minres_preconditioner == "block3x3":
            preconditioner_fallback = False
            try:
                blocks = physics.block_diagonal_hessian(variable.detach()).detach()

                def preconditioner(value: torch.Tensor) -> torch.Tensor:
                    result = torch.linalg.solve(
                        blocks, value.unsqueeze(-1)
                    ).squeeze(-1)
                    return result * gate

            except torch.linalg.LinAlgError:
                # A preconditioner is not part of the Newton equation. Falling
                # back to identity preserves H*step=-g instead of aborting the
                # physical frame because an approximate block factorization
                # failed.
                preconditioner_fallback = True
                preconditioner = lambda value: value

        elif minres_preconditioner == "none":
            preconditioner = lambda value: value
        else:
            raise ValueError(
                f"unsupported MINRES preconditioner: {minres_preconditioner}"
            )
        linear_solve = _minimum_residual(
            hvp=hvp,
            preconditioner=preconditioner,
            right_hand_side=-gradient.detach(),
            max_iterations=minres_max_iterations,
            relative_tolerance=minres_relative_tolerance,
            absolute_tolerance=minres_absolute_tolerance,
        )
        if (
            minres_preconditioner == "block3x3"
            and preconditioner_fallback
        ):
            linear_solve = LinearSolveResult(
                step=linear_solve.step,
                iterations=linear_solve.iterations,
                relative_residual=linear_solve.relative_residual,
                converged=linear_solve.converged,
                breakdown=linear_solve.breakdown,
                minimum_curvature=linear_solve.minimum_curvature,
                preconditioner_fallback=True,
            )
    candidate = physics.project_positions(y + linear_solve.step, fixed_targets)
    return candidate.detach(), gradient.detach(), energy.detach(), linear_solve


def _finite_quantile(values: np.ndarray, quantile: float, default: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if finite.size else float(default)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_newton_diagnostics(
    *,
    output: Path,
    residual_history: np.ndarray,
    minres_iterations: np.ndarray,
    minres_relative_residual: np.ndarray,
    minres_converged: np.ndarray,
) -> list[Path]:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    valid_frames = np.flatnonzero(np.isfinite(residual_history[:, 0]))
    if valid_frames.size == 0:
        return paths
    selected = np.unique(
        valid_frames[np.linspace(0, len(valid_frames) - 1, min(6, len(valid_frames))).astype(int)]
    )

    figure, axis = plt.subplots(figsize=(10, 5.6))
    for frame in selected:
        values = residual_history[frame]
        valid = np.isfinite(values) & (values > 0.0)
        ratio = values / max(float(values[0]), 1e-30)
        axis.semilogy(np.flatnonzero(valid), ratio[valid], label=f"frame {frame}")
    axis.axhline(1e-3, color="black", ls="--", lw=1, label="target=1e-3")
    axis.set(
        xlabel="Newton iteration",
        ylabel="residual / initial residual",
        title="Raw Newton-MINRES inner convergence",
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(ncol=2)
    path = output / "05_newton_inner_residual.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    mean_iterations = np.full(residual_history.shape[0], np.nan, dtype=np.float64)
    max_iterations = np.full_like(mean_iterations, np.nan)
    linear_residual_p95 = np.full_like(mean_iterations, np.nan)
    for frame in valid_frames:
        iterations = minres_iterations[frame]
        iterations = iterations[iterations > 0]
        residuals = minres_relative_residual[frame]
        residuals = residuals[np.isfinite(residuals)]
        if iterations.size:
            mean_iterations[frame] = float(np.mean(iterations))
            max_iterations[frame] = float(np.max(iterations))
        if residuals.size:
            linear_residual_p95[frame] = float(np.quantile(residuals, 0.95))
    frames = np.arange(residual_history.shape[0])
    axes[0].plot(frames, mean_iterations, label="mean MINRES iterations")
    axes[0].plot(frames, max_iterations, label="max MINRES iterations")
    axes[1].semilogy(
        frames, linear_residual_p95, label="MINRES relative residual p95"
    )
    failed_linear_solves = np.sum(
        np.isfinite(minres_relative_residual) & ~minres_converged, axis=1
    )
    axes[1].plot(frames, failed_linear_solves, label="nonconverged solves")
    axes[0].set(ylabel="iterations", title="Newton linear solves")
    axes[1].set(xlabel="physical frame", ylabel="MINRES diagnostic")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    path = output / "06_newton_minres.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    paths.append(path)
    return paths


def _visualization_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "cloth23_render_single_motion_rollout.py"),
        "--rollout-dir", str(args.output_dir.resolve()),
        "--fixed-data-dir", str(args.fixed_data_dir.resolve()),
        "--fps", str(args.render_fps),
        "--frame-hold", str(args.render_frame_hold),
        "--width", str(args.render_width),
        "--height", str(args.render_height),
        "--video-crf", str(args.video_crf),
        "--egl-device-index", str(args.egl_device_index),
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
    if output.exists() and any(output.iterdir()) and not args.overwrite:
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
    frames = args.rollout_frames
    inner = args.inner_steps
    thresholds = FailureThresholds()

    curve_names = (
        "initial_residual",
        "final_residual",
        "residual_ratio",
        "first_step_ratio",
        "energy_change",
        "area_min",
        "area_max",
        "edge_min",
        "edge_max",
        "displacement_rms",
    )
    curves = {name: np.full(frames, np.nan, dtype=np.float64) for name in curve_names}
    curves["inner_steps"] = np.zeros(frames, dtype=np.int64)
    curves["objective_evaluations"] = np.zeros(frames, dtype=np.int64)
    curves["converged"] = np.zeros(frames, dtype=np.bool_)
    residual_history = np.full((frames, inner + 1), np.nan, dtype=np.float64)
    energy_history = np.full((frames, inner + 1), np.nan, dtype=np.float64)
    step_norm = np.full((frames, inner), np.nan, dtype=np.float64)
    minres_iterations = np.zeros((frames, inner), dtype=np.int64)
    minres_relative_residual = np.full((frames, inner), np.nan, dtype=np.float64)
    minres_converged = np.zeros((frames, inner), dtype=np.bool_)
    minres_breakdown = np.zeros((frames, inner), dtype=np.bool_)
    minres_minimum_curvature = np.full(
        (frames, inner), np.nan, dtype=np.float64
    )

    p = motion.positions.detach().clone()
    v = motion.velocities.detach().clone()
    fixed_targets = motion.positions.detach().clone()
    trajectory_frames = [0]
    trajectory_positions = [p[0].detach().cpu().numpy()]
    frame_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    failure_frame = frames
    failure_reason = ""
    epsilon = torch.finfo(dtype).eps
    started = time.perf_counter()

    for frame in range(frames):
        q = physics.make_q(p, v)
        y = physics.project_positions(p, fixed_targets)
        start_positions = p.clone()
        initial_energy = physics.variational_energy(y, q, fixed_targets).detach()
        failed_linear_solve = False
        completed_inner = 0
        for iteration in range(inner):
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
            residual = float(torch.linalg.vector_norm(gradient).item())
            residual_history[frame, iteration] = residual
            energy_history[frame, iteration] = float(energy.item())
            step_norm[frame, iteration] = float(
                torch.linalg.vector_norm(linear_solve.step).item()
            )
            minres_iterations[frame, iteration] = linear_solve.iterations
            minres_relative_residual[frame, iteration] = (
                linear_solve.relative_residual
            )
            minres_converged[frame, iteration] = linear_solve.converged
            minres_breakdown[frame, iteration] = linear_solve.breakdown
            minres_minimum_curvature[frame, iteration] = (
                linear_solve.minimum_curvature
            )
            inner_rows.append(
                {
                    "frame": frame,
                    "newton_iteration": iteration,
                    "residual_before": residual,
                    "energy_before": float(energy.item()),
                    "step_norm": step_norm[frame, iteration],
                    "minres_iterations": linear_solve.iterations,
                    "minres_relative_residual": linear_solve.relative_residual,
                    "minres_converged": linear_solve.converged,
                    "minres_breakdown": linear_solve.breakdown,
                    "minimum_observed_curvature": linear_solve.minimum_curvature,
                }
            )
            completed_inner += 1
            if linear_solve.breakdown or (
                args.stop_on_minres_nonconvergence and not linear_solve.converged
            ):
                failed_linear_solve = True
                failure_reason = (
                    "newton_minres_breakdown"
                    if linear_solve.breakdown
                    else "newton_minres_nonconvergence"
                )
                break
            if not bool(torch.isfinite(candidate).all()):
                failed_linear_solve = True
                failure_reason = "nonfinite_newton_update"
                break
            y = candidate

        final_residual_tensor = physics.stationarity_residual_norm(
            y, q, fixed_targets
        ).detach()
        final_energy = physics.variational_energy(y, q, fixed_targets).detach()
        residual_history[frame, completed_inner] = float(final_residual_tensor.item())
        energy_history[frame, completed_inner] = float(final_energy.item())
        initial_residual = residual_history[frame, 0]
        ratio = float(final_residual_tensor.item()) / max(initial_residual, float(epsilon))
        first_ratio = (
            residual_history[frame, 1] / max(initial_residual, float(epsilon))
            if completed_inner >= 1 and np.isfinite(residual_history[frame, 1])
            else math.nan
        )
        bad, reasons, diagnostics = detect_failures(
            physics, y, final_residual_tensor, fixed_targets, thresholds
        )
        if failed_linear_solve:
            bad[:] = True
            reasons[0].append(failure_reason)

        curves["initial_residual"][frame] = initial_residual
        curves["final_residual"][frame] = float(final_residual_tensor.item())
        curves["residual_ratio"][frame] = ratio
        curves["first_step_ratio"][frame] = first_ratio
        curves["energy_change"][frame] = float((final_energy - initial_energy).item())
        curves["inner_steps"][frame] = completed_inner
        curves["objective_evaluations"][frame] = (
            completed_inner
            + 1
            + int(minres_iterations[frame, :completed_inner].sum())
        )
        curves["converged"][frame] = (
            float(final_residual_tensor.item())
            <= max(args.absolute_residual_tolerance, initial_residual * args.residual_ratio_tolerance)
        )
        curves["displacement_rms"][frame] = float(
            torch.sqrt(torch.mean(torch.sum((y - start_positions) ** 2, dim=-1))).item()
        )
        for name in ("area_min", "area_max", "edge_min", "edge_max"):
            curves[name][frame] = float(diagnostics[name][0].item())

        frame_rows.append(
            {
                "frame": frame,
                "initial_residual": initial_residual,
                "final_residual": float(final_residual_tensor.item()),
                "residual_ratio": ratio,
                "first_step_ratio": first_ratio,
                "energy_change": curves["energy_change"][frame],
                "inner_steps": completed_inner,
                "minres_iterations_total": int(
                    minres_iterations[frame, :completed_inner].sum()
                ),
                "minres_nonconverged_count": int(
                    (~minres_converged[frame, :completed_inner]).sum()
                ),
                "minres_breakdown_count": int(
                    minres_breakdown[frame, :completed_inner].sum()
                ),
                "area_min": curves["area_min"][frame],
                "area_max": curves["area_max"][frame],
                "edge_min": curves["edge_min"][frame],
                "edge_max": curves["edge_max"][frame],
            }
        )
        print(
            f"frame={frame:04d} residual_ratio={ratio:.3e} "
            f"newton_steps={completed_inner} "
            f"minres_iterations="
            f"{int(minres_iterations[frame, :completed_inner].sum())}",
            flush=True,
        )
        if bool(bad[0]):
            failure_frame = frame
            failure_reason = "+".join(dict.fromkeys(reasons[0])) or "unknown"
            if trajectory_frames[-1] != frame:
                trajectory_frames.append(frame)
                trajectory_positions.append(start_positions[0].detach().cpu().numpy())
            failed_physical_frame = frame + 1
            if trajectory_frames[-1] != failed_physical_frame:
                trajectory_frames.append(failed_physical_frame)
                trajectory_positions.append(y[0].detach().cpu().numpy())
            break
        p, v = physics.advance_state(p, y, fixed_targets)
        physical_frame = frame + 1
        if physical_frame % args.trajectory_stride == 0 or physical_frame == frames:
            trajectory_frames.append(physical_frame)
            trajectory_positions.append(p[0].detach().cpu().numpy())

    evaluated = np.isfinite(curves["residual_ratio"])
    evaluated_count = int(evaluated.sum())
    converged = curves["converged"] & evaluated
    energy_increase = np.isfinite(curves["energy_change"]) & (curves["energy_change"] > 0.0)
    valid_minres = np.isfinite(minres_relative_residual)
    metrics = {
        "solver": "matrix_free_newton_minres",
        "visualization_title": (
            "Raw Newton-MINRES: no damping, no line search — typical 0"
        ),
        "motion_id": motion.motion_ids[0],
        "completed": True,
        "failed": failure_frame < frames,
        "failure_frame": failure_frame,
        "failure_reason": failure_reason,
        "survival_frames": min(failure_frame, frames),
        "rollout_frames": frames,
        "evaluated_frame_count": evaluated_count,
        "inner_steps_cap": inner,
        "early_stop": False,
        "fixed_inner_iteration_budget": True,
        "residual_ratio_tolerance": args.residual_ratio_tolerance,
        "residual_ratio_median": _finite_quantile(
            curves["residual_ratio"], 0.5, math.inf
        ),
        "residual_ratio_p95": _finite_quantile(
            curves["residual_ratio"], 0.95, math.inf
        ),
        "converged_frame_count": int(converged.sum()),
        "converged_frame_fraction": float(converged.sum() / max(evaluated_count, 1)),
        "inner_steps_mean": (
            float(np.mean(curves["inner_steps"][evaluated]))
            if evaluated_count
            else math.inf
        ),
        "objective_evaluations_total": int(curves["objective_evaluations"].sum()),
        "energy_increase_fraction": float(
            energy_increase.sum() / max(evaluated_count, 1)
        ),
        "min_area_ratio": _finite_quantile(curves["area_min"], 0.0, 0.0),
        "max_area_ratio": _finite_quantile(curves["area_max"], 1.0, math.inf),
        "min_edge_ratio": _finite_quantile(curves["edge_min"], 0.0, 0.0),
        "max_edge_ratio": _finite_quantile(curves["edge_max"], 1.0, math.inf),
        "damping": False,
        "line_search": False,
        "hessian": "exact autograd Hessian-vector products",
        "newton_variant": (
            "inexact Newton with a recorded MINRES forcing tolerance"
        ),
        "linear_solver": (
            {
                "block3x3": "3x3 block-Jacobi-preconditioned MINRES",
                "mass": "mass-preconditioned MINRES",
                "none": "unpreconditioned MINRES",
            }[args.minres_preconditioner]
        ),
        "minres_preconditioner": args.minres_preconditioner,
        "minres_max_iterations": args.minres_max_iterations,
        "minres_relative_tolerance": args.minres_relative_tolerance,
        "minres_absolute_tolerance": args.minres_absolute_tolerance,
        "minres_solve_count": int(valid_minres.sum()),
        "minres_converged_count": int(
            (minres_converged & valid_minres).sum()
        ),
        "minres_breakdown_count": int(minres_breakdown.sum()),
        "minres_iterations_total": int(minres_iterations.sum()),
        "minres_iterations_p95": _finite_quantile(
            minres_iterations[valid_minres], 0.95, math.inf
        ),
        "minres_relative_residual_p95": _finite_quantile(
            minres_relative_residual, 0.95, math.inf
        ),
        "wall_seconds": time.perf_counter() - started,
        "dtype": args.dtype,
        "device": str(args.device),
    }
    np.savez_compressed(result_dir / "curves.npz", **curves)
    np.savez_compressed(
        result_dir / "inner_history.npz",
        residual_norm=residual_history,
        energy=energy_history,
        step_norm=step_norm,
        minres_iterations=minres_iterations,
        minres_relative_residual=minres_relative_residual,
        minres_converged=minres_converged,
        minres_breakdown=minres_breakdown,
        minimum_observed_curvature=minres_minimum_curvature,
    )
    np.savez_compressed(
        result_dir / "trajectory.npz",
        frames=np.asarray(trajectory_frames, dtype=np.int64),
        positions=np.stack(trajectory_positions).astype(np.float32),
    )
    write_json(result_dir / "metrics.json", metrics)
    _write_csv(result_dir / "per_frame.csv", frame_rows)
    _write_csv(result_dir / "inner_iterations.csv", inner_rows)
    diagnostic_plots = _plot_newton_diagnostics(
        output=output / "figures",
        residual_history=residual_history,
        minres_iterations=minres_iterations,
        minres_relative_residual=minres_relative_residual,
        minres_converged=minres_converged,
    )
    _atomic_write_json(
        output / "manifest.json",
        {
            "format_version": 1,
            "completed": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": {
                "fixed_data_dir": str(Path(args.fixed_data_dir).resolve()),
                "dataset": str(dataset_path.resolve()),
                "split": args.split,
                "motion_index": args.motion_index,
                "rollout_frames": args.rollout_frames,
                "inner_steps": args.inner_steps,
                "residual_ratio_tolerance": args.residual_ratio_tolerance,
                "absolute_residual_tolerance": args.absolute_residual_tolerance,
                "trajectory_stride": args.trajectory_stride,
                "dtype": args.dtype,
                "device": args.device,
                "damping": False,
                "line_search": False,
                "minres_max_iterations": args.minres_max_iterations,
                "minres_relative_tolerance": args.minres_relative_tolerance,
                "minres_absolute_tolerance": args.minres_absolute_tolerance,
                "minres_preconditioner": args.minres_preconditioner,
                "stop_on_minres_nonconvergence": (
                    args.stop_on_minres_nonconvergence
                ),
            },
            "result": metrics,
            "diagnostic_plots": [str(path) for path in diagnostic_plots],
        },
    )
    print(
        f"Newton-MINRES result: failed={metrics['failed']} "
        f"survival={metrics['survival_frames']}/{frames} "
        f"ratio_p95={metrics['residual_ratio_p95']:.3e}",
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
