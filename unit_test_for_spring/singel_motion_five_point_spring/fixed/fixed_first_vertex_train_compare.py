"""
Fixed-first-vertex five-particle/four-spring learned optimizer experiment.

Main changes from the free-chain experiment
-------------------------------------------
1. Particle 1 is fixed at its initial world-space position. Only particles 2--5
   are optimization variables, so the learned optimizer input/output dimension
   is 12 rather than 15.
2. Chain-reversal augmentation is removed because reversing the chain changes
   which endpoint is fixed.
3. The same validation-selected checkpoint is evaluated against:
      - the learned MLP iteration,
      - fixed-step gradient descent,
      - undamped full Newton.
4. Gradient-descent step size is selected only on the validation set.
5. A hard extrapolation-test physical time step is selected independently of
   solver performance, using the p95 initial residual over its sampled starts.
   The selection is saved for the separate 500-frame rollout script.

The training objective uses only the physical variational energy. Numerical
reference solutions are used for synthetic trajectory generation, validation,
checkpoint selection, metrics, and hard-case selection, never as network input
or as a supervised training target.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# 0. Constants
# =============================================================================

NUM_PARTICLES = 5
NUM_FREE_PARTICLES = 4
NUM_SPRINGS = 4
SPATIAL_DIM = 3
FREE_STATE_DIM = NUM_FREE_PARTICLES * SPATIAL_DIM
HIDDEN_DIM = 64

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

DEFAULT_DEVICE = "cuda:1"
DEFAULT_TOTAL_TIME_STEPS = 100
DEFAULT_TRAIN_POINTS_PER_PROBLEM = 100
DEFAULT_EVAL_POINTS_PER_PROBLEM = 256
DEFAULT_EPOCHS = 50_000
DEFAULT_VALIDATION_INTERVAL = 500
DEFAULT_DIAGNOSTIC_INTERVAL = 500
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8192
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 10_000
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5
DEFAULT_REPORT_STEPS = (1, 5, 10, 50)
DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_GRADIENT_CLIP_NORM = 1.0

LEARNING_RATE = 1e-3
MODEL_RANDOM_SEED = 42
TRAIN_SOBOL_SEED = 20260620
VALIDATION_SOBOL_SEED = 20260621
INTERPOLATION_TEST_SOBOL_SEED = 20260622
EXTRAPOLATION_TEST_SOBOL_SEED = 20260623

GD_CANDIDATE_STEP_SIZES = (
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
)

PLOT_FLOOR = 1e-16
DISTANCE_EPS = 1e-12
MIN_SAMPLING_RADIUS = 1e-10
NEWTON_RESIDUAL_TOLERANCE = 1e-10
REFERENCE_RESIDUAL_TOLERANCE = 1e-11
REFERENCE_ACCEPTABLE_RESIDUAL = 1e-8
REFERENCE_MAX_ITERATIONS = 100
REFERENCE_LINE_SEARCH_MIN_ALPHA = 2.0**-30


# =============================================================================
# 1. Data classes and utilities
# =============================================================================


@dataclass(frozen=True)
class RuntimeConfig:
    total_time_steps: int
    train_points_per_problem: int
    eval_points_per_problem: int
    epochs: int
    validation_interval: int
    diagnostic_interval: int
    evaluation_steps: int
    evaluation_batch_size: int
    initial_k: int
    k_increase_interval: int
    k_increase_amount: int
    max_k: int
    report_steps: tuple[int, ...]
    residual_length_scale: float
    gradient_clip_norm: float
    device: str
    run_single_problem_baseline: bool
    skip_plots: bool
    save_datasets: bool


@dataclass(frozen=True)
class PhysicalConfig:
    masses: tuple[float, ...]
    g: float
    dt: float
    spring_stiffness: tuple[float, ...]
    rest_lengths: tuple[float, ...]
    p0: tuple[tuple[float, float, float], ...]
    v0: tuple[tuple[float, float, float], ...]

    @property
    def anchor(self) -> tuple[float, float, float]:
        return self.p0[0]


@dataclass(frozen=True)
class TimeStepProblem:
    index: int
    time: float
    p_n_full: torch.Tensor
    v_n_full: torch.Tensor
    q_free: torch.Tensor
    free_masses: torch.Tensor
    exact_y_free: torch.Tensor
    sampling_radius: float
    exact_energy: float
    exact_residual: float


@dataclass
class DatasetBundle:
    initial_y: torch.Tensor
    q: torch.Tensor
    masses: torch.Tensor
    exact_y: torch.Tensor
    problem_index: torch.Tensor
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return int(self.initial_y.shape[0])

    def to(self, device: torch.device) -> "DatasetBundle":
        return DatasetBundle(
            initial_y=self.initial_y.to(device=device, dtype=TORCH_DTYPE),
            q=self.q.to(device=device, dtype=TORCH_DTYPE),
            masses=self.masses.to(device=device, dtype=TORCH_DTYPE),
            exact_y=self.exact_y.to(device=device, dtype=TORCH_DTYPE),
            problem_index=self.problem_index.to(device=device, dtype=torch.long),
            metadata=copy.deepcopy(self.metadata),
        )


@dataclass(frozen=True)
class ProblemSplit:
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    interpolation_test_indices: tuple[int, ...]
    extrapolation_test_indices: tuple[int, ...]

    @property
    def all_test_indices(self) -> tuple[int, ...]:
        return self.interpolation_test_indices + self.extrapolation_test_indices


def default_physical_config() -> PhysicalConfig:
    # The first particle is the fixed anchor. Its velocity is explicitly zero.
    return PhysicalConfig(
        masses=(1.0, 1.0, 1.0, 1.0, 1.0),
        g=9.8,
        dt=0.01,
        spring_stiffness=(2500.0, 2500.0, 2500.0, 2500.0),
        rest_lengths=(1.0, 1.0, 1.0, 1.0),
        p0=(
            (-2.2, 0.00, 1.20),
            (-1.1, 0.15, 1.10),
            (0.0, -0.10, 1.25),
            (1.1, 0.20, 1.05),
            (2.2, 0.00, 1.15),
        ),
        v0=(
            (0.0, 0.0, 0.0),
            (0.05, 0.10, 0.00),
            (0.00, 0.00, 0.15),
            (-0.05, -0.10, 0.00),
            (-0.20, 0.00, -0.05),
        ),
    )


def physical_config_from_dict(data: dict[str, Any]) -> PhysicalConfig:
    return PhysicalConfig(
        masses=tuple(float(v) for v in data["masses"]),
        g=float(data["g"]),
        dt=float(data["dt"]),
        spring_stiffness=tuple(float(v) for v in data["spring_stiffness"]),
        rest_lengths=tuple(float(v) for v in data["rest_lengths"]),
        p0=tuple(tuple(float(x) for x in row) for row in data["p0"]),
        v0=tuple(tuple(float(x) for x in row) for row in data["v0"]),
    )


def create_output_directory() -> Path:
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent / script_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return make_json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return make_json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=2, ensure_ascii=False)


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested cuda:{index}, but only {torch.cuda.device_count()} CUDA devices are visible."
        )


def get_k_for_epoch(epoch_index: int, config: RuntimeConfig) -> int:
    return min(
        config.initial_k
        + (epoch_index // config.k_increase_interval) * config.k_increase_amount,
        config.max_k,
    )


def finite_plot_value(value: float | int | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return float("nan")
    return max(float(value), PLOT_FLOOR)


# =============================================================================
# 2. Fixed-anchor physics
# =============================================================================


def reshape_free(y: torch.Tensor) -> torch.Tensor:
    if y.shape[-1] != FREE_STATE_DIM:
        raise ValueError(f"Expected final dimension {FREE_STATE_DIM}, got {tuple(y.shape)}")
    return y.reshape(*y.shape[:-1], NUM_FREE_PARTICLES, SPATIAL_DIM)


def full_positions_from_free(y: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    free = reshape_free(y)
    anchor = torch.as_tensor(physical.anchor, dtype=y.dtype, device=y.device)
    anchor = anchor.reshape(*([1] * (free.ndim - 2)), 1, SPATIAL_DIM)
    anchor = anchor.expand(*free.shape[:-2], 1, SPATIAL_DIM)
    return torch.cat([anchor, free], dim=-2)


def free_state_from_full(full: torch.Tensor) -> torch.Tensor:
    if full.shape[-2:] != (NUM_PARTICLES, SPATIAL_DIM):
        raise ValueError(f"Expected (..., 5, 3), got {tuple(full.shape)}")
    return full[..., 1:, :].reshape(*full.shape[:-2], FREE_STATE_DIM)


def spring_lengths_from_free(y: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    points = full_positions_from_free(y, physical)
    edges = points[..., 1:, :] - points[..., :-1, :]
    return torch.linalg.vector_norm(edges, dim=-1)


def variational_energy(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """Implicit-Euler variational energy for the four free particles."""
    free = reshape_free(y)
    q_free = reshape_free(q)
    if masses.shape[-1] != NUM_FREE_PARTICLES:
        raise ValueError("masses must contain four free-particle masses")

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
    full_grad[..., 1:, :] = grad_free

    edges = full[..., 1:, :] - full[..., :-1, :]
    lengths = torch.linalg.vector_norm(edges, dim=-1, keepdim=True).clamp_min(
        DISTANCE_EPS
    )
    stiffness = torch.as_tensor(
        physical.spring_stiffness, dtype=y.dtype, device=y.device
    )
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    parameter_shape = [1] * (edges.ndim - 2) + [NUM_SPRINGS, 1]
    edge_grad = (
        stiffness.reshape(parameter_shape)
        * (1.0 - rest.reshape(parameter_shape) / lengths)
        * edges
    )
    full_grad = full_grad.clone()
    full_grad[..., :-1, :] -= edge_grad
    full_grad[..., 1:, :] += edge_grad
    return full_grad[..., 1:, :].reshape(*y.shape[:-1], FREE_STATE_DIM)


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
    """Analytic 12x12 Hessian after eliminating the fixed anchor."""
    full = full_positions_from_free(y, physical)
    edges = full[..., 1:, :] - full[..., :-1, :]
    lengths = torch.linalg.vector_norm(edges, dim=-1, keepdim=True).clamp_min(
        DISTANCE_EPS
    )
    identity = torch.eye(SPATIAL_DIM, dtype=y.dtype, device=y.device)
    outer = edges.unsqueeze(-1) * edges.unsqueeze(-2)
    stiffness = torch.as_tensor(
        physical.spring_stiffness, dtype=y.dtype, device=y.device
    )
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    parameter_shape = [1] * (edges.ndim - 2) + [NUM_SPRINGS, 1, 1]
    spring_blocks = stiffness.reshape(parameter_shape) * (
        (1.0 - rest.reshape(parameter_shape) / lengths.unsqueeze(-1)) * identity
        + (rest.reshape(parameter_shape) / lengths.pow(3).unsqueeze(-1)) * outer
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

    # Edge 0 connects the fixed anchor to free particle 0: only its free diagonal
    # block remains. Edges 1--3 connect two free particles.
    hessian[..., 0:3, 0:3] += spring_blocks[..., 0, :, :]
    for edge_index in range(1, NUM_SPRINGS):
        left_free = edge_index - 1
        right_free = edge_index
        left = slice(left_free * 3, (left_free + 1) * 3)
        right = slice(right_free * 3, (right_free + 1) * 3)
        block = spring_blocks[..., edge_index, :, :]
        hessian[..., left, left] += block
        hessian[..., right, right] += block
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
    """Damped Newton with Armijo backtracking, used only as reference."""
    y = initial_y.detach().clone().reshape(1, FREE_STATE_DIM)
    q_batch = q.detach().clone().reshape(1, FREE_STATE_DIM)
    masses_batch = masses.detach().clone().reshape(1, NUM_FREE_PARTICLES)
    reductions = 0

    for iteration in range(max_iterations):
        gradient = stationarity_residual(y, q_batch, masses_batch, physical)
        residual = float(torch.linalg.vector_norm(gradient).item())
        if residual <= residual_tolerance:
            return y.squeeze(0), {
                "iterations": iteration,
                "residual_norm": residual,
                "line_search_reductions": reductions,
                "converged": True,
                "acceptable": True,
            }

        hessian = variational_hessian(y, masses_batch, physical)
        direction_col, info = torch.linalg.solve_ex(hessian, -gradient.unsqueeze(-1))
        direction = direction_col.squeeze(-1)
        if bool(torch.any(info != 0)) or not bool(torch.isfinite(direction).all()):
            direction = torch.matmul(
                torch.linalg.pinv(hessian), -gradient.unsqueeze(-1)
            ).squeeze(-1)

        directional_derivative = float(torch.sum(gradient * direction).item())
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            mass_per_coordinate = masses_batch.repeat_interleave(3, dim=-1)
            direction = -physical.dt**2 * gradient / mass_per_coordinate
            directional_derivative = float(torch.sum(gradient * direction).item())

        if float(torch.linalg.vector_norm(direction).item()) <= 1e-9:
            y = y + direction
            continue

        energy_before = float(
            variational_energy(y, q_batch, masses_batch, physical).item()
        )
        alpha = 1.0
        accepted = False
        while alpha >= REFERENCE_LINE_SEARCH_MIN_ALPHA:
            candidate = y + alpha * direction
            if bool(torch.all(spring_lengths_from_free(candidate, physical) > DISTANCE_EPS)):
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
            raise RuntimeError("Reference Newton line search failed")

    final_residual = float(
        stationarity_residual_norm(y, q_batch, masses_batch, physical).item()
    )
    acceptable = final_residual <= REFERENCE_ACCEPTABLE_RESIDUAL
    if not acceptable and raise_on_nonconvergence:
        raise RuntimeError(
            f"Reference solver failed: final residual {final_residual:.6e}"
        )
    return y.squeeze(0), {
        "iterations": max_iterations,
        "residual_norm": final_residual,
        "line_search_reductions": reductions,
        "converged": False,
        "acceptable": acceptable,
    }


def make_q_free(
    p_full: torch.Tensor,
    v_full: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    gravity = torch.tensor([0.0, 0.0, physical.g], dtype=p_full.dtype, device=p_full.device)
    q_free_points = (
        p_full[1:, :] + physical.dt * v_full[1:, :] - physical.dt**2 * gravity
    )
    return q_free_points.reshape(FREE_STATE_DIM)


def advance_physical_state(
    p_full: torch.Tensor,
    y_next_free: torch.Tensor,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_p = full_positions_from_free(y_next_free, physical).reshape(NUM_PARTICLES, 3)
    next_v = torch.zeros_like(next_p)
    next_v[1:, :] = (next_p[1:, :] - p_full[1:, :]) / physical.dt
    next_p[0, :] = torch.as_tensor(physical.anchor, dtype=next_p.dtype, device=next_p.device)
    next_v[0, :] = 0.0
    return next_p, next_v


def generate_reference_sequence(
    physical: PhysicalConfig,
    total_steps: int,
) -> list[TimeStepProblem]:
    p_n = torch.tensor(physical.p0, dtype=TORCH_DTYPE)
    v_n = torch.tensor(physical.v0, dtype=TORCH_DTYPE)
    p_n[0] = torch.tensor(physical.anchor, dtype=TORCH_DTYPE)
    v_n[0] = 0.0
    free_masses = torch.tensor(physical.masses[1:], dtype=TORCH_DTYPE)
    problems: list[TimeStepProblem] = []

    for index in range(total_steps):
        q_free = make_q_free(p_n, v_n, physical)
        initial_y = p_n[1:, :].reshape(FREE_STATE_DIM)
        exact_y, info = solve_reference_solution(
            q=q_free,
            masses=free_masses,
            initial_y=initial_y,
            physical=physical,
        )
        radius = max(
            float(torch.max(torch.abs(initial_y - exact_y)).item()),
            MIN_SAMPLING_RADIUS,
        )
        exact_energy = float(
            variational_energy(
                exact_y.unsqueeze(0),
                q_free.unsqueeze(0),
                free_masses.unsqueeze(0),
                physical,
            ).item()
        )
        exact_residual = float(
            stationarity_residual_norm(
                exact_y.unsqueeze(0),
                q_free.unsqueeze(0),
                free_masses.unsqueeze(0),
                physical,
            ).item()
        )
        problems.append(
            TimeStepProblem(
                index=index,
                time=index * physical.dt,
                p_n_full=p_n.clone(),
                v_n_full=v_n.clone(),
                q_free=q_free.clone(),
                free_masses=free_masses.clone(),
                exact_y_free=exact_y.clone(),
                sampling_radius=radius,
                exact_energy=exact_energy,
                exact_residual=exact_residual,
            )
        )
        if index == 0 or (index + 1) % 20 == 0:
            print(
                f"Reference {index + 1:3d}/{total_steps}: "
                f"iterations={info['iterations']}, residual={exact_residual:.3e}, "
                f"radius={radius:.3e}"
            )
        p_n, v_n = advance_physical_state(p_n, exact_y, physical)
    return problems


def build_problem_split(total_steps: int) -> ProblemSplit:
    if total_steps != 100:
        raise ValueError("This experiment uses exactly 100 physical time-step problems")
    validation = tuple(range(3, 80, 8))
    interpolation = tuple(range(7, 80, 8))
    held_out = set(validation) | set(interpolation)
    train = tuple(i for i in range(80) if i not in held_out)
    extrapolation = tuple(range(80, 100))
    split = ProblemSplit(train, validation, interpolation, extrapolation)
    assert (len(train), len(validation), len(interpolation), len(extrapolation)) == (
        60,
        10,
        10,
        20,
    )
    return split


# =============================================================================
# 3. Dataset generation (no chain reversal)
# =============================================================================


def nondegenerate_mask(points: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    return torch.all(spring_lengths_from_free(points, physical) > DISTANCE_EPS, dim=-1)


def generate_sobol_points(
    *,
    count: int,
    center: torch.Tensor,
    radius: float,
    seed: int,
    physical: PhysicalConfig,
    explicit_points: Sequence[torch.Tensor] = (),
) -> tuple[torch.Tensor, dict[str, Any]]:
    if count <= 0 or radius <= 0.0:
        raise ValueError("count and radius must be positive")
    chunks: list[torch.Tensor] = []
    accepted = 0
    for point in explicit_points:
        point = point.detach().cpu().to(TORCH_DTYPE).reshape(1, FREE_STATE_DIM)
        if not bool(nondegenerate_mask(point, physical)[0]):
            raise ValueError("Explicit point is degenerate")
        chunks.append(point)
        accepted += 1
    if accepted > count:
        raise ValueError("Too many explicit points")

    engine = torch.quasirandom.SobolEngine(
        dimension=FREE_STATE_DIM, scramble=True, seed=seed
    )
    generated = rejected = 0
    while accepted < count:
        remaining = count - accepted
        draw_count = max(32, remaining * 2)
        unit = engine.draw(draw_count).to(dtype=TORCH_DTYPE)
        candidates = center.reshape(1, -1) + (2.0 * unit - 1.0) * radius
        keep = nondegenerate_mask(candidates, physical)
        selected = candidates[keep][:remaining]
        generated += draw_count
        rejected += int((~keep).sum().item())
        if selected.numel() > 0:
            chunks.append(selected)
            accepted += int(selected.shape[0])
    points = torch.cat(chunks, dim=0)[:count].contiguous()
    return points, {
        "mode": "scrambled_sobol_12d_linf_cube",
        "seed": seed,
        "count": count,
        "center": center.tolist(),
        "radius_linf": radius,
        "explicit_point_count": len(explicit_points),
        "generated_candidates": generated,
        "rejected_degenerate_candidates": rejected,
        "chain_reversal_augmented": False,
    }


def build_problem_dataset(
    *,
    problem: TimeStepProblem,
    size: int,
    seed: int,
    role: str,
    physical: PhysicalConfig,
    include_explicit_train_points: bool,
) -> DatasetBundle:
    explicit = (
        (
            problem.p_n_full[1:, :].reshape(FREE_STATE_DIM),
            problem.exact_y_free,
        )
        if include_explicit_train_points
        else ()
    )
    initial_y, sampling = generate_sobol_points(
        count=size,
        center=problem.exact_y_free,
        radius=problem.sampling_radius,
        seed=seed,
        physical=physical,
        explicit_points=explicit,
    )
    return DatasetBundle(
        initial_y=initial_y,
        q=problem.q_free.reshape(1, -1).expand(size, -1).clone(),
        masses=problem.free_masses.reshape(1, -1).expand(size, -1).clone(),
        exact_y=problem.exact_y_free.reshape(1, -1).expand(size, -1).clone(),
        problem_index=torch.full((size,), problem.index, dtype=torch.long),
        metadata={
            "role": role,
            "problem_index": problem.index,
            "physical_time": problem.time,
            "size": size,
            "sampling": sampling,
        },
    )


def concatenate_datasets(
    datasets: Sequence[DatasetBundle],
    *,
    role: str,
    problem_indices: Sequence[int],
    points_per_problem: int,
) -> DatasetBundle:
    return DatasetBundle(
        initial_y=torch.cat([d.initial_y for d in datasets], dim=0),
        q=torch.cat([d.q for d in datasets], dim=0),
        masses=torch.cat([d.masses for d in datasets], dim=0),
        exact_y=torch.cat([d.exact_y for d in datasets], dim=0),
        problem_index=torch.cat([d.problem_index for d in datasets], dim=0),
        metadata={
            "role": role,
            "problem_indices": [int(i) for i in problem_indices],
            "num_problems": len(problem_indices),
            "points_per_problem": points_per_problem,
            "size": sum(len(d) for d in datasets),
            "split_unit": "physical_time_step_problem",
            "chain_reversal_augmented": False,
        },
    )


def build_dataset_for_problem_indices(
    *,
    problems: Sequence[TimeStepProblem],
    indices: Sequence[int],
    points_per_problem: int,
    base_seed: int,
    role: str,
    physical: PhysicalConfig,
    include_explicit_train_points: bool,
) -> DatasetBundle:
    datasets = [
        build_problem_dataset(
            problem=problems[i],
            size=points_per_problem,
            seed=base_seed + 1009 * i,
            role=f"{role}_problem_{i}",
            physical=physical,
            include_explicit_train_points=include_explicit_train_points,
        )
        for i in indices
    ]
    return concatenate_datasets(
        datasets,
        role=role,
        problem_indices=indices,
        points_per_problem=points_per_problem,
    )


def build_special_state_dataset(
    *,
    problems: Sequence[TimeStepProblem],
    indices: Sequence[int],
    state: str,
    role: str,
) -> DatasetBundle:
    initial, q, masses, exact = [], [], [], []
    for i in indices:
        problem = problems[i]
        if state == "current":
            y0 = problem.p_n_full[1:, :].reshape(FREE_STATE_DIM)
        elif state == "exact":
            y0 = problem.exact_y_free
        else:
            raise ValueError(state)
        initial.append(y0.reshape(1, -1))
        q.append(problem.q_free.reshape(1, -1))
        masses.append(problem.free_masses.reshape(1, -1))
        exact.append(problem.exact_y_free.reshape(1, -1))
    return DatasetBundle(
        initial_y=torch.cat(initial),
        q=torch.cat(q),
        masses=torch.cat(masses),
        exact_y=torch.cat(exact),
        problem_index=torch.tensor(indices, dtype=torch.long),
        metadata={
            "role": role,
            "state": state,
            "problem_indices": [int(i) for i in indices],
            "num_problems": len(indices),
            "points_per_problem": 1,
            "size": len(indices),
        },
    )


def dataset_to_serializable_dict(dataset: DatasetBundle) -> dict[str, Any]:
    return {
        "initial_y": dataset.initial_y,
        "q": dataset.q,
        "masses": dataset.masses,
        "exact_y": dataset.exact_y,
        "problem_index": dataset.problem_index,
        "metadata": dataset.metadata,
    }


# =============================================================================
# 4. Learned optimizer
# =============================================================================


def mass_preconditioned_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    residual = stationarity_residual(y, q, masses, physical)
    mass_per_coordinate = masses.repeat_interleave(3, dim=-1)
    return physical.dt**2 * residual / mass_per_coordinate


class MLPOptimizer(nn.Module):
    def __init__(self, residual_length_scale: float) -> None:
        super().__init__()
        if residual_length_scale <= 0.0:
            raise ValueError("residual_length_scale must be positive")
        self.linear1 = nn.Linear(FREE_STATE_DIM, HIDDEN_DIM, bias=False)
        self.activation = nn.Identity()
        self.linear2 = nn.Linear(HIDDEN_DIM, FREE_STATE_DIM, bias=False)
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.zeros_(self.linear2.weight)
        self.register_buffer(
            "residual_length_scale",
            torch.tensor(float(residual_length_scale), dtype=TORCH_DTYPE),
        )

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        *,
        physical: PhysicalConfig,
    ) -> torch.Tensor:
        u = mass_preconditioned_residual(y, q, masses, physical)
        u = u / self.residual_length_scale
        return self.residual_length_scale * self.linear2(
            self.activation(self.linear1(u))
        )


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = model(y, q, masses, physical=physical)
    return y + delta, delta


def physical_energy_scale(
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_length_scale: float,
) -> float:
    return float(masses.mean().item()) * residual_length_scale**2 / physical.dt**2


# =============================================================================
# 5. Unified evaluation
# =============================================================================


def _statistics(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    result = {
        "num_nonfinite": int(values.size - finite.size),
        "mean": float("nan"),
        "median": float("nan"),
        "p95": float("nan"),
        "max": float("nan"),
    }
    if finite.size:
        result.update(
            mean=float(np.mean(finite)),
            median=float(np.median(finite)),
            p95=float(np.percentile(finite, 95)),
            max=float(np.max(finite)),
        )
    return result


def _state_metrics(
    y: torch.Tensor,
    dataset: DatasetBundle,
    exact_energy: torch.Tensor,
    physical: PhysicalConfig,
) -> dict[str, torch.Tensor]:
    point_errors = torch.linalg.vector_norm(
        reshape_free(y) - reshape_free(dataset.exact_y), dim=-1
    )
    energy = variational_energy(y, dataset.q, dataset.masses, physical)
    metrics: dict[str, torch.Tensor] = {
        "residual": stationarity_residual_norm(y, dataset.q, dataset.masses, physical),
        "energy_gap": energy - exact_energy,
        "exact_error": torch.linalg.vector_norm(y - dataset.exact_y, dim=-1),
        "particle_mean_error": point_errors.mean(dim=-1),
        "particle_max_error": point_errors.max(dim=-1).values,
        "spring_length_error": torch.mean(
            torch.abs(
                spring_lengths_from_free(y, physical)
                - spring_lengths_from_free(dataset.exact_y, physical)
            ),
            dim=-1,
        ),
        # The anchor is reconstructed from a constant, so this should be exactly zero.
        "anchor_error": torch.zeros_like(point_errors[..., 0]),
    }
    for free_index in range(NUM_FREE_PARTICLES):
        metrics[f"point{free_index + 2}_error"] = point_errors[..., free_index]
    return metrics


def _selected_steps(steps: int, report_steps: Sequence[int]) -> list[int]:
    return sorted(set([0, steps, *[s for s in report_steps if 0 <= s <= steps]]))


@torch.no_grad()
def evaluate_solver_on_dataset(
    *,
    solver: str,
    dataset_cpu: DatasetBundle,
    physical: PhysicalConfig,
    steps: int,
    batch_size: int,
    report_steps: Sequence[int],
    device: torch.device,
    model: MLPOptimizer | None = None,
    gd_step_size: float | None = None,
) -> dict[str, Any]:
    if solver not in {"learned", "gradient_descent", "full_newton"}:
        raise ValueError(solver)
    if solver == "learned" and model is None:
        raise ValueError("model is required")
    if solver == "gradient_descent" and gd_step_size is None:
        raise ValueError("gd_step_size is required")
    if model is not None:
        model.eval()

    metric_batches: dict[str, list[torch.Tensor]] = {}
    problem_batches: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for start in range(0, len(dataset_cpu), batch_size):
        end = min(start + batch_size, len(dataset_cpu))
        batch = DatasetBundle(
            initial_y=dataset_cpu.initial_y[start:end],
            q=dataset_cpu.q[start:end],
            masses=dataset_cpu.masses[start:end],
            exact_y=dataset_cpu.exact_y[start:end],
            problem_index=dataset_cpu.problem_index[start:end],
            metadata={},
        ).to(device)
        y = batch.initial_y.clone()
        exact_energy = variational_energy(batch.exact_y, batch.q, batch.masses, physical)
        step_values: dict[str, list[torch.Tensor]] = {}
        for step in range(steps + 1):
            for name, values in _state_metrics(y, batch, exact_energy, physical).items():
                step_values.setdefault(name, []).append(values.detach().cpu())
            if step == steps:
                break
            if solver == "learned":
                assert model is not None
                y, _ = apply_model_update(model, y, batch.q, batch.masses, physical)
            elif solver == "gradient_descent":
                assert gd_step_size is not None
                y, _ = apply_gradient_descent_update(
                    y, batch.q, batch.masses, physical, gd_step_size
                )
            else:
                y, _ = apply_newton_update(y, batch.q, batch.masses, physical)
        for name, values in step_values.items():
            metric_batches.setdefault(name, []).append(torch.stack(values, dim=1))
        problem_batches.append(batch.problem_index.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    arrays = {
        name: torch.cat(values, dim=0).numpy().astype(float)
        for name, values in metric_batches.items()
    }
    problem_indices = torch.cat(problem_batches).numpy().astype(int)
    for values in arrays.values():
        values[~np.isfinite(values)] = np.nan

    result: dict[str, Any] = {
        "solver": solver,
        "steps": steps,
        "num_points": len(dataset_cpu),
        "selected_report_steps": _selected_steps(steps, report_steps),
        "elapsed_seconds": elapsed,
        "seconds_per_point_per_iteration": elapsed / max(len(dataset_cpu) * steps, 1),
    }
    if gd_step_size is not None:
        result["gradient_descent_step_size"] = float(gd_step_size)

    for name, values in arrays.items():
        for stat_name in ["mean", "median", "p95", "max", "num_nonfinite"]:
            result[f"{name}_{stat_name}_by_step"] = []
        for step in range(values.shape[1]):
            stats = _statistics(values[:, step])
            for stat_name, value in stats.items():
                result[f"{name}_{stat_name}_by_step"].append(value)
        final_stats = _statistics(values[:, -1])
        for stat_name, value in final_stats.items():
            result[f"final_{name}_{stat_name}"] = value

    per_problem: dict[str, Any] = {}
    selected = result["selected_report_steps"]
    for problem_index in sorted(np.unique(problem_indices).tolist()):
        mask = problem_indices == problem_index
        record = {"problem_index": int(problem_index), "num_points": int(mask.sum()), "steps": {}}
        for step in selected:
            record["steps"][str(step)] = {
                name: _statistics(values[mask, step]) for name, values in arrays.items()
            }
        per_problem[str(problem_index)] = record
    result["per_problem"] = per_problem
    return result


def validation_selection_key(metrics: dict[str, Any]) -> tuple[float, ...] | None:
    values = (
        float(metrics["final_residual_num_nonfinite"]),
        float(metrics["final_residual_p95"]),
        float(metrics["final_exact_error_p95"]),
        float(metrics["final_energy_gap_p95"]),
    )
    return values if all(math.isfinite(v) for v in values) else None


def select_gradient_descent_step_size(
    *,
    validation: DatasetBundle,
    physical: PhysicalConfig,
    config: RuntimeConfig,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    best_alpha: float | None = None
    best_key: tuple[float, ...] | None = None
    best_metrics: dict[str, Any] | None = None
    for alpha in GD_CANDIDATE_STEP_SIZES:
        print(f"Evaluating validation GD step size alpha={alpha:.1e} ...")
        metrics = evaluate_solver_on_dataset(
            solver="gradient_descent",
            dataset_cpu=validation,
            physical=physical,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps,
            device=device,
            gd_step_size=alpha,
        )
        key = validation_selection_key(metrics)
        records.append({"step_size": alpha, "selection_key": key, "metrics": metrics})
        if key is not None and (best_key is None or key < best_key):
            best_key = key
            best_alpha = alpha
            best_metrics = metrics
    if best_alpha is None or best_metrics is None:
        raise RuntimeError("No finite gradient-descent candidate was found")
    return best_alpha, {
        "candidate_step_sizes": list(GD_CANDIDATE_STEP_SIZES),
        "selection_rule": (
            "lexicographic final residual nonfinite count, residual p95, "
            "exact-error p95, energy-gap p95"
        ),
        "selected_step_size": best_alpha,
        "selected_key": best_key,
        "records": records,
    }


# =============================================================================
# 6. Training
# =============================================================================


def one_step_diagnostics(
    model: MLPOptimizer,
    dataset: DatasetBundle,
    physical: PhysicalConfig,
) -> dict[str, float]:
    with torch.no_grad():
        y0 = dataset.initial_y
        y1, delta = apply_model_update(model, y0, dataset.q, dataset.masses, physical)
        error0 = torch.linalg.vector_norm(y0 - dataset.exact_y, dim=-1)
        error1 = torch.linalg.vector_norm(y1 - dataset.exact_y, dim=-1)
        residual0 = stationarity_residual_norm(y0, dataset.q, dataset.masses, physical)
        residual1 = stationarity_residual_norm(y1, dataset.q, dataset.masses, physical)
        ideal = dataset.exact_y - y0
        cosine = torch.nn.functional.cosine_similarity(delta, ideal, dim=-1, eps=1e-30)
        return {
            "mean_error_before": float(error0.mean().item()),
            "mean_error_after": float(error1.mean().item()),
            "mean_residual_before": float(residual0.mean().item()),
            "mean_residual_after": float(residual1.mean().item()),
            "mean_update_norm": float(torch.linalg.vector_norm(delta, dim=-1).mean().item()),
            "update_ideal_cosine_mean": float(cosine.mean().item()),
            "error_improvement_fraction": float((error1 < error0).to(TORCH_DTYPE).mean().item()),
            "residual_improvement_fraction": float((residual1 < residual0).to(TORCH_DTYPE).mean().item()),
        }


def run_experiment(
    *,
    experiment_name: str,
    training_cpu: DatasetBundle,
    validation_cpu: DatasetBundle,
    evaluation_datasets: dict[str, DatasetBundle],
    output_dir: Path,
    config: RuntimeConfig,
    physical: PhysicalConfig,
    gd_step_size: float,
    shared_baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)
    model = MLPOptimizer(config.residual_length_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training = training_cpu.to(device)
    energy_scale = physical_energy_scale(
        training.masses, physical, config.residual_length_scale
    )
    initial_energy = variational_energy(
        training.initial_y, training.q, training.masses, physical
    ).detach()
    exact_energy = variational_energy(
        training.exact_y, training.q, training.masses, physical
    ).detach()

    print("\n" + "=" * 96)
    print(f"Training {experiment_name}")
    print(
        f"architecture=12->64->Identity->12, points={len(training_cpu):,}, "
        f"problems={training_cpu.metadata['num_problems']}, device={device}, dtype=float64"
    )
    print("=" * 96)

    train_log: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []
    diagnostic_log: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, ...] | None = None
    best_epoch: int | None = None
    start_time = time.perf_counter()

    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        k = get_k_for_epoch(epoch_index, config)
        model.train()
        y = training.initial_y
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        energy_gap_sum = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        for _ in range(k):
            y, _ = apply_model_update(model, y, training.q, training.masses, physical)
            energy = variational_energy(y, training.q, training.masses, physical)
            objective = objective + ((energy - initial_energy) / energy_scale).mean()
            energy_gap_sum = energy_gap_sum + (energy - exact_energy).mean()
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"Non-finite training objective at epoch {epoch}")
        objective.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            ).item()
        )
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}")
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError(f"Non-finite model parameter at epoch {epoch}")

        train_log.append(
            {
                "epoch": epoch,
                "K": k,
                "dimensionless_objective": float(objective.item()),
                "training_energy_gap_sum": float(energy_gap_sum.item()),
                "gradient_norm_before_clip": grad_norm,
            }
        )

        if epoch == 1 or epoch % config.diagnostic_interval == 0 or epoch == config.epochs:
            diagnostics = one_step_diagnostics(model, training, physical)
            diagnostics.update(epoch=epoch, K=k)
            diagnostic_log.append(diagnostics)

        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            metrics = evaluate_solver_on_dataset(
                solver="learned",
                model=model,
                dataset_cpu=validation_cpu,
                physical=physical,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                report_steps=config.report_steps,
                device=device,
            )
            key = validation_selection_key(metrics)
            validation_log.append({"epoch": epoch, "K": k, "selection_key": key, "metrics": metrics})
            if key is not None and (best_key is None or key < best_key):
                best_key = key
                best_epoch = epoch
                best_state = state_dict_to_cpu(model)
            print(
                f"epoch={epoch:5d} K={k} objective={float(objective.item()):.4e} "
                f"train_gap={float(energy_gap_sum.item()):.4e} "
                f"val_res_p95={metrics['final_residual_p95']:.4e} "
                f"val_err_p95={metrics['final_exact_error_p95']:.4e} "
                f"best_epoch={best_epoch} elapsed={time.perf_counter()-start_time:.1f}s"
            )

    last_state = state_dict_to_cpu(model)
    if best_state is None:
        best_state = copy.deepcopy(last_state)
        best_epoch = config.epochs
    torch.save(last_state, experiment_dir / "last_model_state_dict.pt")
    torch.save(best_state, experiment_dir / "best_validation_model_state_dict.pt")
    torch.save(best_state, experiment_dir / "mlp_optimizer_state_dict.pt")

    model.load_state_dict(best_state)
    model.to(device)
    learned_results: dict[str, Any] = {}
    for name, dataset in evaluation_datasets.items():
        learned_results[name] = evaluate_solver_on_dataset(
            solver="learned",
            model=model,
            dataset_cpu=dataset,
            physical=physical,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps,
            device=device,
        )

    comparison = {
        name: {
            "learned": learned_results[name],
            "gradient_descent": shared_baselines[name]["gradient_descent"],
            "full_newton": shared_baselines[name]["full_newton"],
        }
        for name in evaluation_datasets
    }

    report = {
        "experiment_name": experiment_name,
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": best_key,
        "training_dataset": training_cpu.metadata,
        "validation_dataset": validation_cpu.metadata,
        "model": {
            "architecture": "12D dimensionless mass-preconditioned residual -> 64 -> Identity -> 12D update",
            "bias_free": True,
            "first_layer_initialization": "orthogonal",
            "output_layer_initialization": "zero",
            "residual_length_scale": config.residual_length_scale,
            "dtype": str(TORCH_DTYPE),
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "full_batch": True,
            "epochs": config.epochs,
            "gradient_clip_norm": config.gradient_clip_norm,
            "energy_scale": energy_scale,
        },
        "gradient_descent_step_size": gd_step_size,
        "train_log": train_log,
        "diagnostic_log": diagnostic_log,
        "validation_log": validation_log,
        "evaluation": comparison,
    }
    save_json(report, experiment_dir / "experiment_report.json")

    if not config.skip_plots:
        plot_training_curves(
            train_log, validation_log, best_epoch, experiment_dir / "training_and_validation.png"
        )
        for split_name in ["interpolation_test", "extrapolation_test"]:
            plot_three_solver_rollout(
                comparison[split_name],
                title=f"{experiment_name}: {split_name}",
                save_path=experiment_dir / f"{split_name}_three_solver_rollout.png",
            )
    return report


# =============================================================================
# 7. Plotting and hard-case selection
# =============================================================================


def plot_training_curves(
    train_log: Sequence[dict[str, Any]],
    validation_log: Sequence[dict[str, Any]],
    best_epoch: int | None,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(
        [r["epoch"] for r in train_log],
        [finite_plot_value(r["training_energy_gap_sum"]) for r in train_log],
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Training energy-gap sum")
    val_epochs = [r["epoch"] for r in validation_log]
    axes[1].plot(
        val_epochs,
        [finite_plot_value(r["metrics"]["final_residual_p95"]) for r in validation_log],
        marker="o",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Validation residual p95")
    axes[2].plot(
        val_epochs,
        [finite_plot_value(r["metrics"]["final_exact_error_p95"]) for r in validation_log],
        marker="o",
    )
    axes[2].set_yscale("log")
    axes[2].set_title("Validation exact-error p95")
    for ax in axes:
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_three_solver_rollout(
    comparison: dict[str, dict[str, Any]],
    *,
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("residual", "Residual p95"),
        ("energy_gap", "Energy gap p95"),
        ("exact_error", "Exact-solution error p95"),
    ]
    labels = {
        "learned": "MLP",
        "gradient_descent": "gradient descent",
        "full_newton": "full Newton",
    }
    for ax, (metric, metric_title) in zip(axes, specs):
        for solver_name, metrics in comparison.items():
            ax.plot(
                range(metrics["steps"] + 1),
                [finite_plot_value(v) for v in metrics[f"{metric}_p95_by_step"]],
                marker="o",
                markersize=3,
                label=labels[solver_name],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Solver iteration")
        ax.set_title(metric_title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reference_sequence(
    problems: Sequence[TimeStepProblem], split: ProblemSplit, save_path: Path
) -> None:
    times = np.array([p.time for p in problems])
    positions = np.array([p.p_n_full.numpy() for p in problems])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for particle in range(NUM_PARTICLES):
        axes[0, 0].plot(times, positions[:, particle, 0], label=f"point {particle+1}")
        axes[0, 1].plot(times, positions[:, particle, 2], label=f"point {particle+1}")
    axes[0, 0].set_title("x coordinate")
    axes[0, 1].set_title("z coordinate")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    radii = [p.sampling_radius for p in problems]
    axes[1, 0].plot(times, radii)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Sampling radius")
    rows = [
        (split.train_indices, 0, "train"),
        (split.validation_indices, 1, "validation"),
        (split.interpolation_test_indices, 2, "interpolation"),
        (split.extrapolation_test_indices, 3, "extrapolation"),
    ]
    for indices, level, label in rows:
        axes[1, 1].scatter(indices, [level] * len(indices), label=label)
    axes[1, 1].set_yticks(range(4), [r[2] for r in rows])
    axes[1, 1].set_title("Problem split")
    for ax in axes.reshape(-1):
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Physical time / problem index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def select_hard_extrapolation_case(
    dataset: DatasetBundle,
    problems: Sequence[TimeStepProblem],
    physical: PhysicalConfig,
) -> dict[str, Any]:
    residuals = stationarity_residual_norm(
        dataset.initial_y, dataset.q, dataset.masses, physical
    ).numpy()
    problem_indices = dataset.problem_index.numpy()
    records = []
    for index in sorted(np.unique(problem_indices).tolist()):
        values = residuals[problem_indices == index]
        records.append(
            {
                "problem_index": int(index),
                "physical_time": problems[index].time,
                "initial_residual_p95": float(np.percentile(values, 95)),
                "initial_residual_max": float(np.max(values)),
                "num_initial_states": int(values.size),
            }
        )
    selected = max(records, key=lambda r: r["initial_residual_p95"])
    problem = problems[selected["problem_index"]]
    return {
        "selection_rule": (
            "Among extrapolation-test physical time-step problems, choose the one "
            "with the largest p95 stationarity residual over its sampled initial states."
        ),
        "solver_independent": True,
        "selected": selected,
        "all_candidates": records,
        "selected_physical_state": {
            "p_n_full": problem.p_n_full.tolist(),
            "v_n_full": problem.v_n_full.tolist(),
            "time": problem.time,
            "problem_index": problem.index,
        },
    }


def run_physics_checks(physical: PhysicalConfig, problem: TimeStepProblem) -> dict[str, Any]:
    perturb = torch.linspace(-2e-3, 2e-3, FREE_STATE_DIM, dtype=TORCH_DTYPE)
    y = (problem.exact_y_free + perturb).detach().requires_grad_(True)
    q = problem.q_free
    masses = problem.free_masses
    energy = variational_energy(y, q, masses, physical)
    auto_grad = torch.autograd.grad(energy, y)[0]
    analytic_grad = stationarity_residual(y.detach(), q, masses, physical)

    def scalar_energy(state: torch.Tensor) -> torch.Tensor:
        return variational_energy(state, q, masses, physical)

    auto_hessian = torch.autograd.functional.hessian(scalar_energy, y.detach())
    analytic_hessian = variational_hessian(y.detach(), masses, physical)
    grad_error = float(torch.max(torch.abs(auto_grad - analytic_grad)).item())
    hess_error = float(torch.max(torch.abs(auto_hessian - analytic_hessian)).item())
    checks = {
        "gradient_max_abs_error": grad_error,
        "hessian_max_abs_error": hess_error,
        "gradient_check_passed": grad_error < 1e-8,
        "hessian_check_passed": hess_error < 1e-7,
    }
    if not checks["gradient_check_passed"] or not checks["hessian_check_passed"]:
        raise RuntimeError(f"Physics check failed: {checks}")
    return checks


# =============================================================================
# 8. CLI and orchestration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-first-vertex learned optimizer with GD/Newton comparison"
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--total-time-steps", type=int, default=DEFAULT_TOTAL_TIME_STEPS)
    parser.add_argument("--train-points-per-problem", type=int, default=DEFAULT_TRAIN_POINTS_PER_PROBLEM)
    parser.add_argument("--eval-points-per-problem", type=int, default=DEFAULT_EVAL_POINTS_PER_PROBLEM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--diagnostic-interval", type=int, default=DEFAULT_DIAGNOSTIC_INTERVAL)
    parser.add_argument("--evaluation-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument("--k-increase-interval", type=int, default=DEFAULT_K_INCREASE_INTERVAL)
    parser.add_argument("--k-increase-amount", type=int, default=DEFAULT_K_INCREASE_AMOUNT)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--report-steps", type=int, nargs="+", default=list(DEFAULT_REPORT_STEPS))
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--skip-single-problem-baseline", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--save-datasets", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    positive_names = [
        "total_time_steps",
        "train_points_per_problem",
        "eval_points_per_problem",
        "epochs",
        "validation_interval",
        "diagnostic_interval",
        "evaluation_steps",
        "evaluation_batch_size",
        "initial_k",
        "k_increase_interval",
        "k_increase_amount",
        "max_k",
    ]
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.total_time_steps) != 100:
        raise ValueError("total_time_steps must be 100")
    if int(args.initial_k) > int(args.max_k):
        raise ValueError("initial_k cannot exceed max_k")
    if float(args.residual_length_scale) <= 0 or float(args.gradient_clip_norm) <= 0:
        raise ValueError("scales must be positive")
    report_steps = tuple(
        sorted(
            set(
                [
                    int(s)
                    for s in args.report_steps
                    if 0 < int(s) <= int(args.evaluation_steps)
                ]
                + [int(args.evaluation_steps)]
            )
        )
    )
    return RuntimeConfig(
        total_time_steps=int(args.total_time_steps),
        train_points_per_problem=int(args.train_points_per_problem),
        eval_points_per_problem=int(args.eval_points_per_problem),
        epochs=int(args.epochs),
        validation_interval=int(args.validation_interval),
        diagnostic_interval=int(args.diagnostic_interval),
        evaluation_steps=int(args.evaluation_steps),
        evaluation_batch_size=int(args.evaluation_batch_size),
        initial_k=int(args.initial_k),
        k_increase_interval=int(args.k_increase_interval),
        k_increase_amount=int(args.k_increase_amount),
        max_k=int(args.max_k),
        report_steps=report_steps,
        residual_length_scale=float(args.residual_length_scale),
        gradient_clip_norm=float(args.gradient_clip_norm),
        device=str(args.device),
        run_single_problem_baseline=not bool(args.skip_single_problem_baseline),
        skip_plots=bool(args.skip_plots),
        save_datasets=bool(args.save_datasets),
    )


def problem_to_record(problem: TimeStepProblem) -> dict[str, Any]:
    return {
        "index": problem.index,
        "time": problem.time,
        "p_n_full": problem.p_n_full.tolist(),
        "v_n_full": problem.v_n_full.tolist(),
        "q_free": problem.q_free.tolist(),
        "free_masses": problem.free_masses.tolist(),
        "exact_y_free": problem.exact_y_free.tolist(),
        "sampling_radius": problem.sampling_radius,
        "exact_energy": problem.exact_energy,
        "exact_residual": problem.exact_residual,
    }


def main() -> None:
    config = validate_args(parse_args())
    physical = default_physical_config()
    output_dir = create_output_directory()
    device = torch.device(config.device)
    validate_device(device)

    problems = generate_reference_sequence(physical, config.total_time_steps)
    split = build_problem_split(config.total_time_steps)
    physics_checks = run_physics_checks(physical, problems[0])
    print(f"Output directory: {output_dir}")
    print(f"Physics checks: {physics_checks}")

    save_json(
        {
            "runtime_config": asdict(config),
            "physical_config": asdict(physical),
            "problem_split": asdict(split),
            "fixed_anchor": list(physical.anchor),
            "free_state_dimension": FREE_STATE_DIM,
            "chain_reversal_augmented": False,
            "physics_checks": physics_checks,
        },
        output_dir / "runtime_config.json",
    )
    save_json(
        {"problems": [problem_to_record(p) for p in problems]},
        output_dir / "reference_time_step_problems.json",
    )
    if not config.skip_plots:
        plot_reference_sequence(
            problems, split, output_dir / "reference_sequence_and_split.png"
        )

    multi_training = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.train_indices,
        points_per_problem=config.train_points_per_problem,
        base_seed=TRAIN_SOBOL_SEED,
        role="multi_problem_training",
        physical=physical,
        include_explicit_train_points=True,
    )
    single_training = build_dataset_for_problem_indices(
        problems=problems,
        indices=(0,),
        points_per_problem=config.train_points_per_problem,
        base_seed=TRAIN_SOBOL_SEED + 1_000_000,
        role="single_problem_training",
        physical=physical,
        include_explicit_train_points=True,
    )
    validation = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.validation_indices,
        points_per_problem=config.eval_points_per_problem,
        base_seed=VALIDATION_SOBOL_SEED,
        role="validation",
        physical=physical,
        include_explicit_train_points=False,
    )
    interpolation_test = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.interpolation_test_indices,
        points_per_problem=config.eval_points_per_problem,
        base_seed=INTERPOLATION_TEST_SOBOL_SEED,
        role="interpolation_test",
        physical=physical,
        include_explicit_train_points=False,
    )
    extrapolation_test = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.extrapolation_test_indices,
        points_per_problem=config.eval_points_per_problem,
        base_seed=EXTRAPOLATION_TEST_SOBOL_SEED,
        role="extrapolation_test",
        physical=physical,
        include_explicit_train_points=False,
    )
    current_state_all_test = build_special_state_dataset(
        problems=problems,
        indices=split.all_test_indices,
        state="current",
        role="current_state_all_test",
    )
    exact_state_all_test = build_special_state_dataset(
        problems=problems,
        indices=split.all_test_indices,
        state="exact",
        role="exact_state_all_test",
    )
    evaluation_datasets = {
        "interpolation_test": interpolation_test,
        "extrapolation_test": extrapolation_test,
        "current_state_all_test": current_state_all_test,
        "exact_state_all_test": exact_state_all_test,
    }

    hard_case = select_hard_extrapolation_case(
        extrapolation_test, problems, physical
    )
    save_json(hard_case, output_dir / "hard_case_selection.json")

    gd_step_size, gd_selection = select_gradient_descent_step_size(
        validation=validation,
        physical=physical,
        config=config,
        device=device,
    )
    save_json(gd_selection, output_dir / "gradient_descent_step_selection.json")
    print(f"Selected gradient-descent step size: {gd_step_size:.3e}")

    shared_baselines: dict[str, dict[str, Any]] = {}
    for name, dataset in evaluation_datasets.items():
        print(f"Evaluating GD and Newton on {name} ...")
        shared_baselines[name] = {
            "gradient_descent": evaluate_solver_on_dataset(
                solver="gradient_descent",
                dataset_cpu=dataset,
                physical=physical,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                report_steps=config.report_steps,
                device=device,
                gd_step_size=gd_step_size,
            ),
            "full_newton": evaluate_solver_on_dataset(
                solver="full_newton",
                dataset_cpu=dataset,
                physical=physical,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                report_steps=config.report_steps,
                device=device,
            ),
        }
    save_json(shared_baselines, output_dir / "shared_gd_newton_baselines.json")

    if config.save_datasets:
        torch.save(
            {
                "multi_problem_training": dataset_to_serializable_dict(multi_training),
                "single_problem_training": dataset_to_serializable_dict(single_training),
                "validation": dataset_to_serializable_dict(validation),
                "interpolation_test": dataset_to_serializable_dict(interpolation_test),
                "extrapolation_test": dataset_to_serializable_dict(extrapolation_test),
                "current_state_all_test": dataset_to_serializable_dict(current_state_all_test),
                "exact_state_all_test": dataset_to_serializable_dict(exact_state_all_test),
            },
            output_dir / "generated_datasets.pt",
        )

    reports = [
        run_experiment(
            experiment_name="multi_problem",
            training_cpu=multi_training,
            validation_cpu=validation,
            evaluation_datasets=evaluation_datasets,
            output_dir=output_dir,
            config=config,
            physical=physical,
            gd_step_size=gd_step_size,
            shared_baselines=shared_baselines,
        )
    ]
    if config.run_single_problem_baseline:
        reports.append(
            run_experiment(
                experiment_name="single_problem_baseline",
                training_cpu=single_training,
                validation_cpu=validation,
                evaluation_datasets=evaluation_datasets,
                output_dir=output_dir,
                config=config,
                physical=physical,
                gd_step_size=gd_step_size,
                shared_baselines=shared_baselines,
            )
        )

    summary = {
        "experiment_type": "fixed_first_vertex_independent_time_step_generalization",
        "runtime_config": asdict(config),
        "physical_config": asdict(physical),
        "problem_split": asdict(split),
        "physics_checks": physics_checks,
        "gradient_descent_selection": gd_selection,
        "hard_case_selection": hard_case,
        "shared_baselines": shared_baselines,
        "experiments": reports,
    }
    save_json(summary, output_dir / "all_experiments_summary.json")
    print("\nCompleted all experiments.")
    print(f"Summary: {output_dir / 'all_experiments_summary.json'}")
    print(
        "500-frame rollout input: "
        f"{output_dir / 'multi_problem' / 'best_validation_model_state_dict.pt'}"
    )


if __name__ == "__main__":
    main()
