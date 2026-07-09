from __future__ import annotations

import math
from typing import Any

import torch

from .config import PhysicalConfig
from .constants import (
    DISTANCE_EPS,
    FIXED_VERTEX_INDICES,
    FREE_STATE_DIM,
    FREE_VERTEX_INDICES,
    GLOBAL_TO_FREE_INDEX,
    NEWTON_RESIDUAL_TOLERANCE,
    NUM_FREE_PARTICLES,
    NUM_PARTICLES,
    NUM_SPRINGS,
    REFERENCE_ACCEPTABLE_RESIDUAL,
    REFERENCE_LINE_SEARCH_MIN_ALPHA,
    REFERENCE_MAX_ITERATIONS,
    REFERENCE_RESIDUAL_TOLERANCE,
    SPATIAL_DIM,
    SPRING_EDGES,
    TRIANGLE_FACES,
)


def free_vertex_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(FREE_VERTEX_INDICES, dtype=torch.long, device=device)


def fixed_vertex_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(FIXED_VERTEX_INDICES, dtype=torch.long, device=device)


def spring_edge_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(SPRING_EDGES, dtype=torch.long, device=device)


def triangle_face_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(TRIANGLE_FACES, dtype=torch.long, device=device)


def reshape_free(y: torch.Tensor) -> torch.Tensor:
    if y.shape[-1] != FREE_STATE_DIM:
        raise ValueError(f"Expected final dimension {FREE_STATE_DIM}, got {tuple(y.shape)}")
    return y.reshape(*y.shape[:-1], NUM_FREE_PARTICLES, SPATIAL_DIM)


def full_positions_from_free(y: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    free = reshape_free(y)
    leading_shape = free.shape[:-2]
    base = torch.as_tensor(physical.p0, dtype=y.dtype, device=y.device)
    view_shape = (*([1] * len(leading_shape)), NUM_PARTICLES, SPATIAL_DIM)
    full = base.reshape(view_shape).expand(*leading_shape, NUM_PARTICLES, SPATIAL_DIM).clone()
    full[..., list(FREE_VERTEX_INDICES), :] = free
    return full


def free_state_from_full(full: torch.Tensor) -> torch.Tensor:
    if full.shape[-2:] != (NUM_PARTICLES, SPATIAL_DIM):
        raise ValueError(f"Expected (..., {NUM_PARTICLES}, {SPATIAL_DIM}), got {tuple(full.shape)}")
    return full[..., list(FREE_VERTEX_INDICES), :].reshape(*full.shape[:-2], FREE_STATE_DIM)


def spring_vectors_from_free(y: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    points = full_positions_from_free(y, physical)
    edges = spring_edge_tensor(y.device)
    return points[..., edges[:, 1], :] - points[..., edges[:, 0], :]


def spring_lengths_from_free(y: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    return torch.linalg.vector_norm(spring_vectors_from_free(y, physical), dim=-1)


def variational_energy(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """Implicit-Euler variational energy for the 23 free cloth vertices."""
    free = reshape_free(y)
    q_free = reshape_free(q)
    if masses.shape[-1] != NUM_FREE_PARTICLES:
        raise ValueError(f"masses must contain {NUM_FREE_PARTICLES} free-particle masses")

    inertial = (masses / (2.0 * physical.dt**2)) * torch.sum(
        (free - q_free) ** 2, dim=-1
    )
    lengths = spring_lengths_from_free(y, physical)
    stiffness = torch.as_tensor(
        physical.spring_stiffness, dtype=y.dtype, device=y.device
    )
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    spring = 0.5 * stiffness * (lengths - rest) ** 2
    return torch.sum(inertial, dim=-1) + torch.sum(spring, dim=-1)


def stationarity_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    free = reshape_free(y)
    q_free = reshape_free(q)
    full = full_positions_from_free(y, physical)

    grad_free = (masses[..., :, None] / physical.dt**2) * (free - q_free)
    full_grad = torch.zeros_like(full)
    full_grad[..., list(FREE_VERTEX_INDICES), :] = grad_free

    edges = spring_edge_tensor(y.device)
    edge_vectors = full[..., edges[:, 1], :] - full[..., edges[:, 0], :]
    lengths = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True).clamp_min(
        DISTANCE_EPS
    )
    stiffness = torch.as_tensor(
        physical.spring_stiffness, dtype=y.dtype, device=y.device
    )
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    parameter_shape = [1] * (edge_vectors.ndim - 2) + [NUM_SPRINGS, 1]
    edge_grad = (
        stiffness.reshape(parameter_shape)
        * (1.0 - rest.reshape(parameter_shape) / lengths)
        * edge_vectors
    )
    full_grad = full_grad.clone()
    full_grad.index_add_(-2, edges[:, 0], -edge_grad)
    full_grad.index_add_(-2, edges[:, 1], edge_grad)
    return full_grad[..., list(FREE_VERTEX_INDICES), :].reshape(*y.shape[:-1], FREE_STATE_DIM)


def stationarity_residual_norm(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    return torch.linalg.vector_norm(
        stationarity_residual(y, q, masses, physical), dim=-1
    )


def variational_hessian(
    y: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """Analytic Hessian after eliminating the two fixed corner vertices."""
    full = full_positions_from_free(y, physical)
    edges = spring_edge_tensor(y.device)
    edge_vectors = full[..., edges[:, 1], :] - full[..., edges[:, 0], :]
    lengths = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True).clamp_min(
        DISTANCE_EPS
    )
    identity = torch.eye(SPATIAL_DIM, dtype=y.dtype, device=y.device)
    outer = edge_vectors.unsqueeze(-1) * edge_vectors.unsqueeze(-2)
    stiffness = torch.as_tensor(
        physical.spring_stiffness, dtype=y.dtype, device=y.device
    )
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    parameter_shape = [1] * (edge_vectors.ndim - 2) + [NUM_SPRINGS, 1, 1]
    lengths_matrix = lengths.unsqueeze(-1)
    spring_blocks = stiffness.reshape(parameter_shape) * (
        (1.0 - rest.reshape(parameter_shape) / lengths_matrix) * identity
        + (rest.reshape(parameter_shape) / lengths_matrix.pow(3)) * outer
    )

    hessian = torch.zeros(
        (*y.shape[:-1], FREE_STATE_DIM, FREE_STATE_DIM),
        dtype=y.dtype,
        device=y.device,
    )
    for free_index in range(NUM_FREE_PARTICLES):
        block = slice(free_index * SPATIAL_DIM, (free_index + 1) * SPATIAL_DIM)
        hessian[..., block, block] += (
            masses[..., free_index] / physical.dt**2
        )[..., None, None] * identity

    for edge_index, (left_global, right_global) in enumerate(SPRING_EDGES):
        left_free = GLOBAL_TO_FREE_INDEX[left_global]
        right_free = GLOBAL_TO_FREE_INDEX[right_global]
        block = spring_blocks[..., edge_index, :, :]
        if left_free >= 0:
            left = slice(left_free * SPATIAL_DIM, (left_free + 1) * SPATIAL_DIM)
            hessian[..., left, left] += block
        if right_free >= 0:
            right = slice(right_free * SPATIAL_DIM, (right_free + 1) * SPATIAL_DIM)
            hessian[..., right, right] += block
        if left_free >= 0 and right_free >= 0:
            hessian[..., left, right] -= block
            hessian[..., right, left] -= block
    return hessian


def apply_newton_update(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_tolerance: float = NEWTON_RESIDUAL_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = stationarity_residual(y, q, masses, physical)
    residual = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True)
    rhs = torch.where(residual > residual_tolerance, -gradient, torch.zeros_like(gradient))
    hessian = variational_hessian(y, masses, physical)
    delta_col, info = torch.linalg.solve_ex(hessian, rhs.unsqueeze(-1))
    delta = delta_col.squeeze(-1)
    failed = info != 0
    if bool(torch.any(failed)):
        delta[failed] = torch.matmul(
            torch.linalg.pinv(hessian[failed]), rhs[failed].unsqueeze(-1)
        ).squeeze(-1)
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("Newton update produced non-finite values")
    return y + delta, delta


def apply_gradient_descent_update(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = -float(step_size) * stationarity_residual(y, q, masses, physical)
    return y + delta, delta


def solve_reference_solution(
    *,
    q: torch.Tensor,
    masses: torch.Tensor,
    initial_y: torch.Tensor,
    physical: PhysicalConfig,
    residual_tolerance: float = REFERENCE_RESIDUAL_TOLERANCE,
    max_iterations: int = REFERENCE_MAX_ITERATIONS,
    raise_on_nonconvergence: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Damped Newton reference solve with a best-finite-iterate fallback.

    With ``raise_on_nonconvergence=False`` this routine never rejects a finite
    best iterate merely because the strict residual target was missed or the
    Armijo line search stalled. This behavior is used by continuous rollout.
    """
    y = initial_y.detach().clone().reshape(1, FREE_STATE_DIM)
    q_batch = q.detach().clone().reshape(1, FREE_STATE_DIM)
    masses_batch = masses.detach().clone().reshape(1, NUM_FREE_PARTICLES)
    if not bool(torch.isfinite(y).all() and torch.isfinite(q_batch).all() and torch.isfinite(masses_batch).all()):
        raise RuntimeError("Reference solver received a non-finite input state")

    reductions = 0
    status = "max_iterations"
    best_y = y.clone()
    best_residual = float("inf")
    best_energy = float("inf")
    best_iteration = 0

    for iteration in range(max_iterations + 1):
        gradient = stationarity_residual(y, q_batch, masses_batch, physical)
        residual = float(torch.linalg.vector_norm(gradient).item())
        energy = float(variational_energy(y, q_batch, masses_batch, physical).item())
        if math.isfinite(residual) and math.isfinite(energy) and residual < best_residual:
            best_y = y.clone()
            best_residual = residual
            best_energy = energy
            best_iteration = iteration
        if residual <= residual_tolerance:
            return y.squeeze(0), {
                "iterations": iteration,
                "residual_norm": residual,
                "line_search_reductions": reductions,
                "converged": True,
                "acceptable": True,
                "status": "converged",
                "used_best_finite_iterate": False,
                "best_iteration": iteration,
            }
        if iteration == max_iterations:
            break

        hessian = variational_hessian(y, masses_batch, physical)
        direction_col, info = torch.linalg.solve_ex(hessian, -gradient.unsqueeze(-1))
        direction = direction_col.squeeze(-1)
        if bool(torch.any(info != 0)) or not bool(torch.isfinite(direction).all()):
            direction = torch.matmul(torch.linalg.pinv(hessian), -gradient.unsqueeze(-1)).squeeze(-1)

        directional_derivative = float(torch.sum(gradient * direction).item())
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            mass_per_coordinate = masses_batch.repeat_interleave(SPATIAL_DIM, dim=-1)
            direction = -physical.dt**2 * gradient / mass_per_coordinate
            directional_derivative = float(torch.sum(gradient * direction).item())
        if not bool(torch.isfinite(direction).all()):
            status = "nonfinite_direction"
            break

        if float(torch.linalg.vector_norm(direction).item()) <= 1e-12:
            status = "tiny_direction"
            break

        energy_before = energy
        alpha = 1.0
        accepted = False
        while alpha >= REFERENCE_LINE_SEARCH_MIN_ALPHA:
            candidate = y + alpha * direction
            if bool(torch.isfinite(candidate).all()) and bool(
                torch.all(spring_lengths_from_free(candidate, physical) > DISTANCE_EPS)
            ):
                candidate_energy = float(
                    variational_energy(candidate, q_batch, masses_batch, physical).item()
                )
                if math.isfinite(candidate_energy) and candidate_energy <= (
                    energy_before + 1e-4 * alpha * directional_derivative
                ):
                    y = candidate
                    accepted = True
                    break
            alpha *= 0.5
            reductions += 1
        if not accepted:
            status = "line_search_failed"
            break

    acceptable = best_residual <= REFERENCE_ACCEPTABLE_RESIDUAL
    if not math.isfinite(best_residual):
        raise RuntimeError("Reference solver did not produce any finite iterate")
    if not acceptable and raise_on_nonconvergence:
        raise RuntimeError(
            f"Reference solver failed with status={status}: best residual {best_residual:.6e}"
        )
    return best_y.squeeze(0), {
        "iterations": max_iterations if status == "max_iterations" else best_iteration,
        "residual_norm": best_residual,
        "best_energy": best_energy,
        "line_search_reductions": reductions,
        "converged": False,
        "acceptable": acceptable,
        "status": status,
        "used_best_finite_iterate": True,
        "best_iteration": best_iteration,
    }


def make_q_free(
    p_full: torch.Tensor,
    v_full: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    gravity = torch.tensor([0.0, 0.0, physical.g], dtype=p_full.dtype, device=p_full.device)
    free = list(FREE_VERTEX_INDICES)
    q_free_points = (
        p_full[free, :] + physical.dt * v_full[free, :] - physical.dt**2 * gravity
    )
    return q_free_points.reshape(FREE_STATE_DIM)


def advance_physical_state(
    p_full: torch.Tensor,
    y_next_free: torch.Tensor,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_p = full_positions_from_free(y_next_free, physical).reshape(NUM_PARTICLES, SPATIAL_DIM)
    next_v = torch.zeros_like(next_p)
    free = list(FREE_VERTEX_INDICES)
    fixed = list(FIXED_VERTEX_INDICES)
    next_v[free, :] = (next_p[free, :] - p_full[free, :]) / physical.dt
    fixed_positions = torch.as_tensor(physical.p0, dtype=next_p.dtype, device=next_p.device)[fixed]
    next_p[fixed, :] = fixed_positions
    next_v[fixed, :] = 0.0
    return next_p, next_v
