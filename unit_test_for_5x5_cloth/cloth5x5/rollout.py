from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from .config import PhysicalConfig
from .constants import (
    DISTANCE_EPS,
    FREE_STATE_DIM,
    FREE_VERTEX_INDICES,
    NUM_FREE_PARTICLES,
    REFERENCE_ACCEPTABLE_RESIDUAL,
    SPATIAL_DIM,
    TORCH_DTYPE,
)
from .model import MLPOptimizer
from .physics import (
    advance_physical_state,
    free_state_from_full,
    make_q_free,
    solve_reference_solution,
    spring_lengths_from_free,
    stationarity_residual,
    stationarity_residual_norm,
    variational_energy,
)
from .solvers import run_solver_steps

FIXED_INNER_ITERATIONS = 50


def solve_fixed_iterations(
    *,
    solver: str,
    p_full: torch.Tensor,
    v_full: torch.Tensor,
    physical: PhysicalConfig,
    free_masses: torch.Tensor,
    inner_iterations: int,
    model: MLPOptimizer | None = None,
    gd_step_size: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Solve one frame using exactly ``inner_iterations`` updates."""
    q = make_q_free(p_full, v_full, physical).reshape(1, FREE_STATE_DIM)
    masses = free_masses.reshape(1, NUM_FREE_PARTICLES)
    y = free_state_from_full(p_full).reshape(1, FREE_STATE_DIM).clone()

    start_time = time.perf_counter()
    y, last_delta = run_solver_steps(
        solver,
        y,
        q,
        masses,
        physical,
        inner_iterations,
        model=model,
        gd_step_size=gd_step_size,
        require_finite=True,
    )
    elapsed = time.perf_counter() - start_time
    residual = float(stationarity_residual_norm(y, q, masses, physical).item())
    energy = float(variational_energy(y, q, masses, physical).item())
    next_p, next_v = advance_physical_state(p_full, y.squeeze(0), physical)
    return next_p, next_v, {
        "inner_iterations": inner_iterations,
        "residual": residual,
        "energy": energy,
        "last_update_norm": float(torch.linalg.vector_norm(last_delta).item()),
        "solve_seconds": elapsed,
    }


def solve_reference_frame(
    *,
    p_full: torch.Tensor,
    v_full: torch.Tensor,
    physical: PhysicalConfig,
    free_masses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    q = make_q_free(p_full, v_full, physical)
    initial_y = free_state_from_full(p_full).reshape(FREE_STATE_DIM)
    start_time = time.perf_counter()
    emergency_message: str | None = None
    try:
        exact_y, info = solve_reference_solution(
            q=q,
            masses=free_masses,
            initial_y=initial_y,
            physical=physical,
            raise_on_nonconvergence=False,
        )
    except Exception as exc:
        emergency_message = f"{type(exc).__name__}: {exc}"
        y0 = initial_y.reshape(1, FREE_STATE_DIM)
        q_batch = q.reshape(1, FREE_STATE_DIM)
        masses_batch = free_masses.reshape(1, NUM_FREE_PARTICLES)
        gradient = stationarity_residual(y0, q_batch, masses_batch, physical)
        mass_per_coordinate = masses_batch.repeat_interleave(SPATIAL_DIM, dim=-1)
        direction = -physical.dt**2 * gradient / mass_per_coordinate
        energy0 = float(variational_energy(y0, q_batch, masses_batch, physical).item())
        exact_y = initial_y.clone()
        used_step = False
        alpha = 1.0
        while alpha >= 2.0**-30:
            candidate = y0 + alpha * direction
            if bool(torch.isfinite(candidate).all()) and bool(
                torch.all(spring_lengths_from_free(candidate, physical) > DISTANCE_EPS)
            ):
                candidate_energy = float(
                    variational_energy(candidate, q_batch, masses_batch, physical).item()
                )
                if math.isfinite(candidate_energy) and candidate_energy <= energy0:
                    exact_y = candidate.squeeze(0)
                    used_step = True
                    break
            alpha *= 0.5
        if not bool(torch.isfinite(exact_y).all()):
            raise RuntimeError("No finite emergency reference fallback was available") from exc
        fallback_residual = float(
            stationarity_residual_norm(
                exact_y.reshape(1, -1), q_batch, masses_batch, physical
            ).item()
        )
        info = {
            "iterations": 0,
            "residual_norm": fallback_residual,
            "line_search_reductions": 0,
            "converged": False,
            "acceptable": fallback_residual <= REFERENCE_ACCEPTABLE_RESIDUAL,
            "status": "emergency_mass_step" if used_step else "emergency_hold_state",
            "used_best_finite_iterate": True,
            "emergency_exception": emergency_message,
        }
    elapsed = time.perf_counter() - start_time
    next_p, next_v = advance_physical_state(p_full, exact_y, physical)
    info = dict(info)
    info["solve_seconds"] = elapsed
    info.setdefault("status", "converged" if info.get("converged") else "nonconverged")
    info.setdefault("used_best_finite_iterate", not info.get("converged", False))
    return next_p, next_v, info


def free_position_errors(
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    point_error = torch.linalg.vector_norm(
        prediction[list(FREE_VERTEX_INDICES), :] - reference[list(FREE_VERTEX_INDICES), :],
        dim=-1,
    )
    rms = float(torch.sqrt(torch.mean(point_error**2)).item())
    maximum = float(torch.max(point_error).item())
    return rms, maximum


def free_velocity_rms_error(
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    point_error = torch.linalg.vector_norm(
        prediction[list(FREE_VERTEX_INDICES), :] - reference[list(FREE_VERTEX_INDICES), :],
        dim=-1,
    )
    return float(torch.sqrt(torch.mean(point_error**2)).item())


def run_rollout(
    *,
    initial_p: torch.Tensor,
    initial_v: torch.Tensor,
    physical: PhysicalConfig,
    model: MLPOptimizer,
    gd_step_size: float,
    frames: int,
    device: torch.device,
    inner_iterations: int = FIXED_INNER_ITERATIONS,
) -> dict[str, Any]:
    free_masses_device = torch.tensor(
        [physical.masses[i] for i in FREE_VERTEX_INDICES],
        dtype=TORCH_DTYPE,
        device=device,
    )
    free_masses_cpu = torch.tensor(
        [physical.masses[i] for i in FREE_VERTEX_INDICES],
        dtype=TORCH_DTYPE,
    )

    states = {
        "reference": {"p": initial_p.detach().cpu().clone(), "v": initial_v.detach().cpu().clone()},
        "learned": {"p": initial_p.to(device).clone(), "v": initial_v.to(device).clone()},
        "gradient_descent": {"p": initial_p.to(device).clone(), "v": initial_v.to(device).clone()},
        "full_newton": {"p": initial_p.to(device).clone(), "v": initial_v.to(device).clone()},
    }

    trajectory_positions: dict[str, list[np.ndarray]] = {
        name: [record["p"].detach().cpu().numpy().copy()] for name, record in states.items()
    }
    trajectory_velocities: dict[str, list[np.ndarray]] = {
        name: [record["v"].detach().cpu().numpy().copy()] for name, record in states.items()
    }
    frame_diagnostics: dict[str, list[dict[str, Any]]] = {name: [] for name in states}
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
        if not ref_info.get("acceptable", True):
            print(
                f"Warning: reference residual target was not reached at frame {frame}; "
                f"status={ref_info.get('status')}, residual={ref_info['residual_norm']:.3e}. "
                "Continuing with the best finite/fallback iterate."
            )

        for solver_name in ["learned", "gradient_descent", "full_newton"]:
            p_next, v_next, info = solve_fixed_iterations(
                solver=solver_name,
                p_full=states[solver_name]["p"],
                v_full=states[solver_name]["v"],
                physical=physical,
                free_masses=free_masses_device,
                inner_iterations=inner_iterations,
                model=model if solver_name == "learned" else None,
                gd_step_size=gd_step_size if solver_name == "gradient_descent" else None,
            )
            states[solver_name] = {"p": p_next, "v": v_next}
            frame_diagnostics[solver_name].append(info)

        reference_p_device = states["reference"]["p"].to(device)
        reference_v_device = states["reference"]["v"].to(device)
        for solver_name in ["learned", "gradient_descent", "full_newton"]:
            rms, maximum = free_position_errors(states[solver_name]["p"], reference_p_device)
            velocity_rms = free_velocity_rms_error(states[solver_name]["v"], reference_v_device)
            errors[solver_name]["position_rms"].append(rms)
            errors[solver_name]["position_max"].append(maximum)
            errors[solver_name]["velocity_rms"].append(velocity_rms)

        for name, record in states.items():
            trajectory_positions[name].append(record["p"].detach().cpu().numpy().copy())
            trajectory_velocities[name].append(record["v"].detach().cpu().numpy().copy())

        if frame == 1 or frame % 25 == 0 or frame == frames:
            print(
                f"Frame {frame:3d}/{frames}: "
                f"MLP error={errors['learned']['position_rms'][-1]:.3e}, "
                f"GD error={errors['gradient_descent']['position_rms'][-1]:.3e}, "
                f"Newton error={errors['full_newton']['position_rms'][-1]:.3e}"
            )

    total_elapsed = time.perf_counter() - start_total
    return {
        "positions": {name: np.stack(values, axis=0) for name, values in trajectory_positions.items()},
        "velocities": {name: np.stack(values, axis=0) for name, values in trajectory_velocities.items()},
        "diagnostics": frame_diagnostics,
        "errors": errors,
        "total_elapsed_seconds": total_elapsed,
    }


def diagnostics_summary(records: list[dict[str, Any]], *, inner_iterations: int = FIXED_INNER_ITERATIONS) -> dict[str, Any]:
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
            max_residual=float(np.max(residuals)),
            fixed_inner_iterations=inner_iterations,
            convergence_early_stopping=False,
        )
    if records and "iterations" in records[0]:
        iterations = np.asarray([int(r.get("iterations", 0)) for r in records], dtype=int)
        acceptable = np.asarray([bool(r.get("acceptable", True)) for r in records], dtype=bool)
        statuses: dict[str, int] = {}
        for record in records:
            status = str(record.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
        used_best = np.asarray([bool(r.get("used_best_finite_iterate", False)) for r in records], dtype=bool)
        result.update(
            mean_reference_iterations=float(np.mean(iterations)),
            max_reference_iterations=int(np.max(iterations)),
            acceptable_reference_rate=float(np.mean(acceptable)),
            nonacceptable_reference_frames=int(np.count_nonzero(~acceptable)),
            used_best_finite_iterate_frames=int(np.count_nonzero(used_best)),
            reference_status_counts=statuses,
            rollout_continued_after_nonconvergence=True,
        )
    return result
