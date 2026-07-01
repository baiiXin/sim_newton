"""
Five-particle four-spring open-chain learned optimizer:
independent multi-time-step problem generalization experiment.

Stable residual-input revision
------------------------------
1. Generate a 100-step high-accuracy numerical reference trajectory with dt=0.01.
2. Model five free particles connected by four consecutive springs:
       1 -- 2 -- 3 -- 4 -- 5
3. Treat every physical time step as an independent optimization problem.
   Network predictions are NOT propagated from one physical step to the next.
4. Use the mass-preconditioned variational residual as the network state:
       u = dt^2 M^{-1} grad E(y) / s
   where s is a characteristic length.
5. Predict a dimensionless update and map it back to position units:
       delta_y = s * MLP(u)
6. Use bias-free layers so zero residual maps exactly to zero update.
7. Orthogonally initialize the first layer and zero-initialize the output layer.
8. Train with the original physical variational energy, shifted by the initial
   energy and divided by a positive energy scale. This preserves exactly the
   original energy gradient direction and does not use reference solutions.
9. Apply global gradient-norm clipping and record one-step quality diagnostics.
10. Keep float64, Adam(lr=1e-3), full-batch training, K=1->5, and validation
    checkpoint selection from the original experiment.
11. Evaluate an undamped full-Newton baseline from exactly the same initial
    states, using the same datasets, rollout lengths, and reporting metrics.
12. Validate the analytic chain gradient and Hessian against PyTorch autograd.

Reference solutions are used only to generate the synthetic trajectory and
sampling domains, report errors, select checkpoints, and compute diagnostics.
They are not network inputs and do not appear in the backward training objective.
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


# ============================================================
# 0. Fixed experiment identity and defaults
# ============================================================

ACTIVATION_NAME = "identity"
OPTIMIZER_NAME = "adam"
LEARNING_RATE = 1e-3
DEFAULT_DEVICE = "cuda:1"

NUM_PARTICLES = 5
NUM_SPRINGS = NUM_PARTICLES - 1
SPATIAL_DIM = 3
STATE_DIM = NUM_PARTICLES * SPATIAL_DIM
HIDDEN_DIM = 64

# Characteristic length used to nondimensionalize the preconditioned residual.
# With rest_length=1, this is 5% of the spring rest length and matches the
# O(1e-2)-O(1e-1) initial-error range in the original datasets.
DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_DIAGNOSTIC_INTERVAL = 500

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

MODEL_RANDOM_SEED = 42
TRAIN_SOBOL_SEED = 20260620
VALIDATION_SOBOL_SEED = 20260621
INTERPOLATION_TEST_SOBOL_SEED = 20260622
EXTRAPOLATION_TEST_SOBOL_SEED = 20260623

PLOT_FLOOR = 1e-14
RESIDUAL_DISTANCE_EPS = 1e-12
SAMPLE_DISTANCE_EPS = 1e-8
MIN_SAMPLING_RADIUS = 1e-10
DEFAULT_SUMMARY_CURVE_POINTS = 1_000

DEFAULT_TOTAL_TIME_STEPS = 100
DEFAULT_TRAIN_POINTS_PER_PROBLEM = 100
DEFAULT_EVAL_POINTS_PER_PROBLEM = 256
DEFAULT_EPOCHS = 50_000
DEFAULT_VALIDATION_INTERVAL = 500
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8_192
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 10_000
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5
DEFAULT_REPORT_STEPS = (1, 5, 10, 50)

# Newton is evaluated for a fixed number of iterations, exactly like the
# learned solver. States whose residual is already below this tolerance are
# kept fixed to avoid introducing round-off motion at the analytic solution.
NEWTON_RESIDUAL_TOLERANCE = 1e-10
REFERENCE_RESIDUAL_TOLERANCE = 1e-11
REFERENCE_ACCEPTABLE_RESIDUAL = 1e-8
REFERENCE_MAX_ITERATIONS = 100
REFERENCE_LINE_SEARCH_MIN_ALPHA = 2.0 ** -30


# ============================================================
# 1. Data structures and utilities
# ============================================================


@dataclass(frozen=True)
class RuntimeConfig:
    total_time_steps: int
    train_points_per_problem: int
    eval_points_per_problem: int
    epochs: int
    validation_interval: int
    evaluation_steps: int
    evaluation_batch_size: int
    initial_k: int
    k_increase_interval: int
    k_increase_amount: int
    max_k: int
    report_steps: tuple[int, ...]
    residual_length_scale: float
    gradient_clip_norm: float
    diagnostic_interval: int
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
    def spring_k(self) -> tuple[float, ...]:
        """Compatibility alias used by the shared physics/evaluation calls."""
        return self.spring_stiffness

    @property
    def rest_length(self) -> tuple[float, ...]:
        """Compatibility alias used by the shared physics/evaluation calls."""
        return self.rest_lengths


@dataclass(frozen=True)
class TimeStepProblem:
    index: int
    time: float
    p_n: torch.Tensor
    v_n: torch.Tensor
    q: torch.Tensor
    masses: torch.Tensor
    exact_y: torch.Tensor
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

    def subset_by_problem_indices(self, indices: Iterable[int], role: str) -> "DatasetBundle":
        wanted = torch.tensor(sorted(set(int(i) for i in indices)), dtype=torch.long)
        mask = torch.isin(self.problem_index, wanted)
        return DatasetBundle(
            initial_y=self.initial_y[mask].clone(),
            q=self.q[mask].clone(),
            masses=self.masses[mask].clone(),
            exact_y=self.exact_y[mask].clone(),
            problem_index=self.problem_index[mask].clone(),
            metadata={
                **copy.deepcopy(self.metadata),
                "role": role,
                "selected_problem_indices": wanted.tolist(),
                "size": int(mask.sum().item()),
            },
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
    return PhysicalConfig(
        masses=(1.0, 1.0, 1.0, 1.0, 1.0),
        g=9.8,
        dt=0.01,
        spring_stiffness=(2500.0, 2500.0, 2500.0, 2500.0),
        rest_lengths=(1.0, 1.0, 1.0, 1.0),
        p0=(
            (-2.2, 0.00, 1.20),
            (-1.1, 0.15, 1.10),
            ( 0.0,-0.10, 1.25),
            ( 1.1, 0.20, 1.05),
            ( 2.2, 0.00, 1.15),
        ),
        v0=(
            ( 0.20, 0.00, 0.10),
            ( 0.05, 0.10, 0.00),
            ( 0.00, 0.00, 0.15),
            (-0.05,-0.10, 0.00),
            (-0.20, 0.00,-0.05),
        ),
    )


def create_output_directory() -> Path:
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent / script_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
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
    with path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(data), file, indent=2, ensure_ascii=False)


def tensor_to_list(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def is_model_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def finite_plot_value(value: float | int | None) -> float:
    if value is None:
        return float("nan")
    value_float = float(value)
    if not math.isfinite(value_float):
        return float("nan")
    return max(value_float, PLOT_FLOOR)


def downsample_log(
    records: Sequence[dict[str, Any]],
    max_points: int = DEFAULT_SUMMARY_CURVE_POINTS,
) -> list[dict[str, Any]]:
    if len(records) <= max_points:
        return copy.deepcopy(list(records))
    indices = np.linspace(0, len(records) - 1, num=max_points, dtype=int)
    indices = sorted(set(indices.tolist() + [len(records) - 1]))
    return [copy.deepcopy(records[index]) for index in indices]


def validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested cuda:{index}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
        )


def get_k_for_epoch(epoch_index: int, config: RuntimeConfig) -> int:
    return min(
        config.initial_k
        + (epoch_index // config.k_increase_interval) * config.k_increase_amount,
        config.max_k,
    )


def validate_even_size(name: str, value: int) -> None:
    if value < 4 or value % 2 != 0:
        raise ValueError(f"{name} must be an even integer >= 4, got {value}.")


# ============================================================
# 2. Physics, exact solution, energy, and residual
# ============================================================


def _check_state_dimension(values: torch.Tensor, name: str) -> None:
    if values.shape[-1] != STATE_DIM:
        raise ValueError(
            f"{name} must have final dimension {STATE_DIM}, got {tuple(values.shape)}."
        )


def _check_mass_dimension(masses: torch.Tensor) -> None:
    if masses.shape[-1] != NUM_PARTICLES:
        raise ValueError(
            f"masses must have final dimension {NUM_PARTICLES}, "
            f"got {tuple(masses.shape)}."
        )


def _edge_parameter_tensor(
    values: float | Sequence[float] | torch.Tensor,
    *,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(NUM_SPRINGS)
    tensor = tensor.reshape(-1)
    if tensor.numel() != NUM_SPRINGS:
        raise ValueError(
            f"{name} must contain {NUM_SPRINGS} values, got {tensor.numel()}."
        )
    return tensor


def reshape_points(values: torch.Tensor) -> torch.Tensor:
    _check_state_dimension(values, "state")
    return values.reshape(*values.shape[:-1], NUM_PARTICLES, SPATIAL_DIM)


def reverse_chain(values: torch.Tensor) -> torch.Tensor:
    """Reverse the particle numbering of an open chain."""
    points = reshape_points(values)
    return points.flip(dims=(-2,)).reshape(*values.shape[:-1], STATE_DIM)


def reverse_masses(masses: torch.Tensor) -> torch.Tensor:
    _check_mass_dimension(masses)
    return masses.flip(dims=(-1,))


def spring_lengths_from_state(y: torch.Tensor) -> torch.Tensor:
    points = reshape_points(y)
    edge_vectors = points[..., 1:, :] - points[..., :-1, :]
    return torch.linalg.vector_norm(edge_vectors, dim=-1)


def variational_energy(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    *,
    g: float,
    dt: float,
    spring_k: float | Sequence[float] | torch.Tensor,
    rest_length: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Implicit-Euler variational energy for a five-particle open spring chain."""
    _check_state_dimension(y, "y")
    _check_state_dimension(q, "q")
    _check_mass_dimension(masses)

    y_points = reshape_points(y)
    q_points = reshape_points(q)
    stiffness = _edge_parameter_tensor(
        spring_k, reference=y, name="spring stiffness"
    )
    rest = _edge_parameter_tensor(
        rest_length, reference=y, name="rest length"
    )

    inertial = (
        masses / (2.0 * dt**2)
    ) * torch.sum((y_points - q_points) ** 2, dim=-1)
    lengths = spring_lengths_from_state(y)
    spring = 0.5 * stiffness * (lengths - rest) ** 2

    # These y-independent constants are retained to preserve the convention
    # of the original script. They do not affect the optimizer.
    gravity_constant = (
        masses * g * q_points[..., 2]
        + 0.5 * masses * dt**2 * g**2
    )
    return (
        torch.sum(inertial, dim=-1)
        + torch.sum(spring, dim=-1)
        + torch.sum(gravity_constant, dim=-1)
    )


def stationarity_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    *,
    dt: float,
    spring_k: float | Sequence[float] | torch.Tensor,
    rest_length: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Analytic gradient of the chain variational energy."""
    _check_state_dimension(y, "y")
    _check_state_dimension(q, "q")
    _check_mass_dimension(masses)

    y_points = reshape_points(y)
    q_points = reshape_points(q)
    stiffness = _edge_parameter_tensor(
        spring_k, reference=y, name="spring stiffness"
    )
    rest = _edge_parameter_tensor(
        rest_length, reference=y, name="rest length"
    )

    gradient = (masses[..., :, None] / dt**2) * (y_points - q_points)
    edge_vectors = y_points[..., 1:, :] - y_points[..., :-1, :]
    lengths = torch.linalg.vector_norm(
        edge_vectors, dim=-1, keepdim=True
    ).clamp_min(RESIDUAL_DISTANCE_EPS)

    parameter_view_shape = [1] * (edge_vectors.ndim - 2) + [NUM_SPRINGS, 1]
    stiffness_view = stiffness.reshape(parameter_view_shape)
    rest_view = rest.reshape(parameter_view_shape)
    edge_gradient = (
        stiffness_view * (1.0 - rest_view / lengths) * edge_vectors
    )

    # Clone before indexed accumulation because gradient may be a broadcasted view.
    gradient = gradient.clone()
    gradient[..., :-1, :] -= edge_gradient
    gradient[..., 1:, :] += edge_gradient
    return gradient.reshape(*y.shape[:-1], STATE_DIM)


def stationarity_residual_norm(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    *,
    dt: float,
    spring_k: float | Sequence[float] | torch.Tensor,
    rest_length: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    return torch.linalg.vector_norm(
        stationarity_residual(
            y,
            q,
            masses,
            dt=dt,
            spring_k=spring_k,
            rest_length=rest_length,
        ),
        dim=-1,
    )


def variational_hessian(
    y: torch.Tensor,
    masses: torch.Tensor,
    *,
    dt: float,
    spring_k: float | Sequence[float] | torch.Tensor,
    rest_length: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Return the analytic 15x15 Hessian of the chain energy.

    For one edge d = y_j-y_i and r = ||d||, the 3x3 spring block is

        A = k[(1-l0/r) I + (l0/r^3) d d^T].

    Each edge contributes +A to its two diagonal blocks and -A to its two
    off-diagonal blocks. The resulting matrix is block tridiagonal.
    """
    _check_state_dimension(y, "y")
    _check_mass_dimension(masses)

    y_points = reshape_points(y)
    stiffness = _edge_parameter_tensor(
        spring_k, reference=y, name="spring stiffness"
    )
    rest = _edge_parameter_tensor(
        rest_length, reference=y, name="rest length"
    )
    edge_vectors = y_points[..., 1:, :] - y_points[..., :-1, :]
    lengths = torch.linalg.vector_norm(
        edge_vectors, dim=-1, keepdim=True
    ).clamp_min(RESIDUAL_DISTANCE_EPS)

    identity = torch.eye(SPATIAL_DIM, dtype=y.dtype, device=y.device)
    outer = edge_vectors.unsqueeze(-1) * edge_vectors.unsqueeze(-2)
    parameter_view_shape = [1] * (edge_vectors.ndim - 2) + [NUM_SPRINGS, 1, 1]
    stiffness_view = stiffness.reshape(parameter_view_shape)
    rest_view = rest.reshape(parameter_view_shape)
    spring_blocks = stiffness_view * (
        (1.0 - rest_view / lengths.unsqueeze(-1)) * identity
        + (rest_view / lengths.pow(3).unsqueeze(-1)) * outer
    )

    hessian = torch.zeros(
        (*y.shape[:-1], STATE_DIM, STATE_DIM),
        dtype=y.dtype,
        device=y.device,
    )

    for particle_index in range(NUM_PARTICLES):
        block = slice(
            particle_index * SPATIAL_DIM,
            (particle_index + 1) * SPATIAL_DIM,
        )
        mass_term = (masses[..., particle_index] / dt**2)[..., None, None]
        hessian[..., block, block] += mass_term * identity

    for edge_index in range(NUM_SPRINGS):
        left = slice(edge_index * SPATIAL_DIM, (edge_index + 1) * SPATIAL_DIM)
        right = slice(
            (edge_index + 1) * SPATIAL_DIM,
            (edge_index + 2) * SPATIAL_DIM,
        )
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
    *,
    residual_tolerance: float = NEWTON_RESIDUAL_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one undamped full-Newton step to the original physical energy."""
    gradient = stationarity_residual(
        y,
        q,
        masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    residual_norm = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True)
    active = residual_norm > residual_tolerance
    rhs = torch.where(active, -gradient, torch.zeros_like(gradient))
    hessian = variational_hessian(
        y,
        masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )

    delta_column, info = torch.linalg.solve_ex(hessian, rhs.unsqueeze(-1))
    delta = delta_column.squeeze(-1)
    failed = info != 0
    if bool(torch.any(failed)):
        delta[failed] = torch.matmul(
            torch.linalg.pinv(hessian[failed]), rhs[failed].unsqueeze(-1)
        ).squeeze(-1)
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("Newton update produced non-finite values.")
    return y + delta, delta


def solve_reference_solution(
    *,
    q: torch.Tensor,
    masses: torch.Tensor,
    initial_y: torch.Tensor,
    physical: PhysicalConfig,
    residual_tolerance: float = REFERENCE_RESIDUAL_TOLERANCE,
    max_iterations: int = REFERENCE_MAX_ITERATIONS,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve one chain problem with damped Newton and Armijo backtracking.

    This solver is used only to generate a high-accuracy numerical reference.
    It is deliberately more robust than the fixed-step undamped Newton baseline.
    """
    y = initial_y.detach().clone().reshape(1, STATE_DIM)
    q_batch = q.detach().clone().reshape(1, STATE_DIM)
    masses_batch = masses.detach().clone().reshape(1, NUM_PARTICLES)
    accepted_steps = 0
    line_search_reductions = 0

    for iteration in range(max_iterations):
        gradient = stationarity_residual(
            y,
            q_batch,
            masses_batch,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        )
        residual_norm = float(torch.linalg.vector_norm(gradient).item())
        if residual_norm <= residual_tolerance:
            return y.squeeze(0), {
                "iterations": iteration,
                "residual_norm": residual_norm,
                "accepted_steps": accepted_steps,
                "line_search_reductions": line_search_reductions,
                "converged": True,
            }

        hessian = variational_hessian(
            y,
            masses_batch,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        )
        direction_column, info = torch.linalg.solve_ex(
            hessian, -gradient.unsqueeze(-1)
        )
        direction = direction_column.squeeze(-1)
        if bool(torch.any(info != 0)) or not bool(torch.isfinite(direction).all()):
            direction = torch.matmul(
                torch.linalg.pinv(hessian), -gradient.unsqueeze(-1)
            ).squeeze(-1)

        directional_derivative = float(torch.sum(gradient * direction).item())
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            mass_per_coordinate = masses_batch.repeat_interleave(
                SPATIAL_DIM, dim=-1
            )
            direction = -physical.dt**2 * gradient / mass_per_coordinate
            directional_derivative = float(torch.sum(gradient * direction).item())

        # Very close to the minimizer, the predicted energy decrease can be
        # smaller than floating-point resolution even though the Newton
        # correction still removes a visible residual. Accept such tiny full
        # steps directly instead of letting Armijo reject them indefinitely.
        if float(torch.linalg.vector_norm(direction).item()) <= 1e-8:
            candidate = y + direction
            if bool(torch.all(spring_lengths_from_state(candidate) > SAMPLE_DISTANCE_EPS)):
                y = candidate
                accepted_steps += 1
                continue

        energy_before = float(
            variational_energy(
                y,
                q_batch,
                masses_batch,
                g=physical.g,
                dt=physical.dt,
                spring_k=physical.spring_k,
                rest_length=physical.rest_length,
            ).item()
        )
        alpha = 1.0
        accepted = False
        while alpha >= REFERENCE_LINE_SEARCH_MIN_ALPHA:
            candidate = y + alpha * direction
            candidate_lengths = spring_lengths_from_state(candidate)
            if bool(torch.all(candidate_lengths > SAMPLE_DISTANCE_EPS)):
                candidate_energy = float(
                    variational_energy(
                        candidate,
                        q_batch,
                        masses_batch,
                        g=physical.g,
                        dt=physical.dt,
                        spring_k=physical.spring_k,
                        rest_length=physical.rest_length,
                    ).item()
                )
                armijo_bound = energy_before + 1e-4 * alpha * directional_derivative
                if math.isfinite(candidate_energy) and candidate_energy <= armijo_bound:
                    y = candidate
                    accepted = True
                    accepted_steps += 1
                    break
            alpha *= 0.5
            line_search_reductions += 1

        if not accepted:
            raise RuntimeError(
                "Reference damped Newton line search failed to find an acceptable step."
            )

    final_residual = float(
        stationarity_residual_norm(
            y,
            q_batch,
            masses_batch,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        ).item()
    )
    if final_residual > REFERENCE_ACCEPTABLE_RESIDUAL:
        raise RuntimeError(
            "Reference solver did not reach the acceptable residual: "
            f"{final_residual:.6e}."
        )
    return y.squeeze(0), {
        "iterations": max_iterations,
        "residual_norm": final_residual,
        "accepted_steps": accepted_steps,
        "line_search_reductions": line_search_reductions,
        "converged": False,
    }


def generate_reference_sequence(
    physical: PhysicalConfig,
    total_steps: int,
) -> list[TimeStepProblem]:
    """Generate the numerical reference trajectory for independent problems."""
    if len(physical.masses) != NUM_PARTICLES:
        raise ValueError("PhysicalConfig.masses has the wrong length.")
    if len(physical.spring_stiffness) != NUM_SPRINGS:
        raise ValueError("PhysicalConfig.spring_stiffness has the wrong length.")
    if len(physical.rest_lengths) != NUM_SPRINGS:
        raise ValueError("PhysicalConfig.rest_lengths has the wrong length.")
    if len(physical.p0) != NUM_PARTICLES or len(physical.v0) != NUM_PARTICLES:
        raise ValueError("PhysicalConfig p0/v0 has the wrong number of particles.")

    p_n = torch.tensor(physical.p0, dtype=TORCH_DTYPE).reshape(STATE_DIM)
    v_n = torch.tensor(physical.v0, dtype=TORCH_DTYPE).reshape(STATE_DIM)
    masses = torch.tensor(physical.masses, dtype=TORCH_DTYPE)
    gravity = torch.tensor(
        [0.0, 0.0, physical.g], dtype=TORCH_DTYPE
    ).repeat(NUM_PARTICLES)
    problems: list[TimeStepProblem] = []

    for index in range(total_steps):
        q = p_n + physical.dt * v_n - physical.dt**2 * gravity
        exact_y, reference_info = solve_reference_solution(
            q=q,
            masses=masses,
            initial_y=q,
            physical=physical,
        )
        radius = max(
            float(torch.max(torch.abs(p_n - exact_y)).item()),
            MIN_SAMPLING_RADIUS,
        )
        exact_energy = float(
            variational_energy(
                exact_y.unsqueeze(0),
                q.unsqueeze(0),
                masses.unsqueeze(0),
                g=physical.g,
                dt=physical.dt,
                spring_k=physical.spring_k,
                rest_length=physical.rest_length,
            ).item()
        )
        exact_residual = float(
            stationarity_residual_norm(
                exact_y.unsqueeze(0),
                q.unsqueeze(0),
                masses.unsqueeze(0),
                dt=physical.dt,
                spring_k=physical.spring_k,
                rest_length=physical.rest_length,
            ).item()
        )
        problems.append(
            TimeStepProblem(
                index=index,
                time=index * physical.dt,
                p_n=p_n.clone(),
                v_n=v_n.clone(),
                q=q.clone(),
                masses=masses.clone(),
                exact_y=exact_y.clone(),
                sampling_radius=radius,
                exact_energy=exact_energy,
                exact_residual=exact_residual,
            )
        )

        if index == 0 or (index + 1) % 20 == 0:
            print(
                f"Reference step {index + 1:3d}/{total_steps}: "
                f"Newton iterations={reference_info['iterations']}, "
                f"residual={exact_residual:.3e}, radius={radius:.3e}"
            )

        next_v = (exact_y - p_n) / physical.dt
        p_n = exact_y
        v_n = next_v

    return problems

def build_problem_split(total_steps: int) -> ProblemSplit:
    if total_steps != 100:
        raise ValueError(
            "This confirmed experiment uses exactly 100 physical time-step problems. "
            f"Received total_steps={total_steps}."
        )
    validation = tuple(range(3, 80, 8))
    interpolation = tuple(range(7, 80, 8))
    held_out = set(validation) | set(interpolation)
    train = tuple(index for index in range(80) if index not in held_out)
    extrapolation = tuple(range(80, 100))
    split = ProblemSplit(
        train_indices=train,
        validation_indices=validation,
        interpolation_test_indices=interpolation,
        extrapolation_test_indices=extrapolation,
    )
    if not (
        len(split.train_indices) == 60
        and len(split.validation_indices) == 10
        and len(split.interpolation_test_indices) == 10
        and len(split.extrapolation_test_indices) == 20
    ):
        raise AssertionError("Unexpected fixed problem split sizes.")
    return split


# ============================================================
# 3. Per-problem Sobol sampling and dataset assembly
# ============================================================


def nondegenerate_mask(points: torch.Tensor) -> torch.Tensor:
    lengths = spring_lengths_from_state(points)
    return torch.all(lengths > SAMPLE_DISTANCE_EPS, dim=-1)


def generate_canonical_sobol_points(
    *,
    count: int,
    center: torch.Tensor,
    radius: float,
    seed: int,
    explicit_points: Sequence[torch.Tensor] = (),
) -> tuple[torch.Tensor, dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    chunks: list[torch.Tensor] = []
    accepted = 0
    for point in explicit_points:
        point_cpu = point.detach().cpu().to(TORCH_DTYPE).reshape(1, STATE_DIM)
        if not bool(nondegenerate_mask(point_cpu)[0]):
            raise ValueError("An explicit point is degenerate.")
        chunks.append(point_cpu)
        accepted += 1
    if accepted > count:
        raise ValueError("More explicit points were supplied than requested samples.")

    engine = torch.quasirandom.SobolEngine(dimension=STATE_DIM, scramble=True, seed=seed)
    generated = 0
    rejected = 0
    while accepted < count:
        remaining = count - accepted
        draw_count = max(32, remaining * 2)
        unit = engine.draw(draw_count).to(dtype=TORCH_DTYPE)
        candidates = center.reshape(1, STATE_DIM) + (2.0 * unit - 1.0) * radius
        keep = nondegenerate_mask(candidates)
        accepted_candidates = candidates[keep]
        generated += draw_count
        rejected += int((~keep).sum().item())
        if accepted_candidates.shape[0] > remaining:
            accepted_candidates = accepted_candidates[:remaining]
        if accepted_candidates.numel() > 0:
            chunks.append(accepted_candidates)
            accepted += int(accepted_candidates.shape[0])

    result = torch.cat(chunks, dim=0)[:count].contiguous()
    return result, {
        "mode": "scrambled_sobol_15d_linf_cube",
        "seed": int(seed),
        "canonical_count": int(count),
        "center": tensor_to_list(center),
        "radius_linf": float(radius),
        "explicit_point_count": len(explicit_points),
        "generated_candidates": generated,
        "rejected_degenerate_candidates": rejected,
    }


def build_problem_dataset(
    *,
    problem: TimeStepProblem,
    final_size: int,
    seed: int,
    role: str,
    include_explicit_train_points: bool,
) -> DatasetBundle:
    validate_even_size("final_size", final_size)
    canonical_count = final_size // 2
    explicit_points: tuple[torch.Tensor, ...]
    if include_explicit_train_points:
        explicit_points = (problem.p_n, problem.exact_y)
    else:
        explicit_points = ()

    canonical_points, sampling_metadata = generate_canonical_sobol_points(
        count=canonical_count,
        center=problem.exact_y,
        radius=problem.sampling_radius,
        seed=seed,
        explicit_points=explicit_points,
    )
    reversed_points = reverse_chain(canonical_points)
    initial_y = torch.cat([canonical_points, reversed_points], dim=0)

    q_canonical = problem.q.reshape(1, STATE_DIM)
    q_reversed = reverse_chain(q_canonical)
    masses_canonical = problem.masses.reshape(1, NUM_PARTICLES)
    masses_reversed = reverse_masses(masses_canonical)
    exact_canonical = problem.exact_y.reshape(1, STATE_DIM)
    exact_reversed = reverse_chain(exact_canonical)

    q = torch.cat(
        [
            q_canonical.expand(canonical_count, -1),
            q_reversed.expand(canonical_count, -1),
        ],
        dim=0,
    ).clone()
    masses = torch.cat(
        [
            masses_canonical.expand(canonical_count, -1),
            masses_reversed.expand(canonical_count, -1),
        ],
        dim=0,
    ).clone()
    exact_y = torch.cat(
        [
            exact_canonical.expand(canonical_count, -1),
            exact_reversed.expand(canonical_count, -1),
        ],
        dim=0,
    ).clone()
    problem_index = torch.full((final_size,), problem.index, dtype=torch.long)

    return DatasetBundle(
        initial_y=initial_y,
        q=q,
        masses=masses,
        exact_y=exact_y,
        problem_index=problem_index,
        metadata={
            "role": role,
            "problem_index": problem.index,
            "physical_time": problem.time,
            "final_size": final_size,
            "canonical_size": canonical_count,
            "chain_reversal_augmented": True,
            "sampling": sampling_metadata,
        },
    )


def concatenate_datasets(
    datasets: Sequence[DatasetBundle],
    *,
    role: str,
    problem_indices: Sequence[int],
    points_per_problem: int,
) -> DatasetBundle:
    if not datasets:
        raise ValueError("Cannot concatenate an empty dataset list.")
    return DatasetBundle(
        initial_y=torch.cat([dataset.initial_y for dataset in datasets], dim=0),
        q=torch.cat([dataset.q for dataset in datasets], dim=0),
        masses=torch.cat([dataset.masses for dataset in datasets], dim=0),
        exact_y=torch.cat([dataset.exact_y for dataset in datasets], dim=0),
        problem_index=torch.cat([dataset.problem_index for dataset in datasets], dim=0),
        metadata={
            "role": role,
            "problem_indices": [int(index) for index in problem_indices],
            "num_problems": len(problem_indices),
            "points_per_problem": int(points_per_problem),
            "size": int(sum(len(dataset) for dataset in datasets)),
            "split_unit": "physical_time_step_problem",
            "chain_reversal_augmented_per_problem": True,
        },
    )


def build_dataset_for_problem_indices(
    *,
    problems: Sequence[TimeStepProblem],
    indices: Sequence[int],
    points_per_problem: int,
    base_seed: int,
    role: str,
    include_explicit_train_points: bool,
) -> DatasetBundle:
    per_problem = []
    for index in indices:
        problem = problems[index]
        per_problem.append(
            build_problem_dataset(
                problem=problem,
                final_size=points_per_problem,
                seed=base_seed + 1009 * index,
                role=f"{role}_problem_{index}",
                include_explicit_train_points=include_explicit_train_points,
            )
        )
    return concatenate_datasets(
        per_problem,
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
    initial_points: list[torch.Tensor] = []
    q_values: list[torch.Tensor] = []
    masses_values: list[torch.Tensor] = []
    exact_values: list[torch.Tensor] = []
    problem_indices: list[int] = []
    for index in indices:
        problem = problems[index]
        if state == "current":
            initial = problem.p_n
        elif state == "exact":
            initial = problem.exact_y
        else:
            raise ValueError(f"Unsupported special state {state!r}.")
        initial_points.append(initial.reshape(1, STATE_DIM))
        q_values.append(problem.q.reshape(1, STATE_DIM))
        masses_values.append(problem.masses.reshape(1, NUM_PARTICLES))
        exact_values.append(problem.exact_y.reshape(1, STATE_DIM))
        problem_indices.append(index)
    return DatasetBundle(
        initial_y=torch.cat(initial_points, dim=0),
        q=torch.cat(q_values, dim=0),
        masses=torch.cat(masses_values, dim=0),
        exact_y=torch.cat(exact_values, dim=0),
        problem_index=torch.tensor(problem_indices, dtype=torch.long),
        metadata={
            "role": role,
            "state": state,
            "problem_indices": problem_indices,
            "size": len(problem_indices),
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


# ============================================================
# 4. Dimensionless residual input and learned update
# ============================================================


def mass_preconditioned_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """
    Convert the stationarity residual from force units back to length units:

        r_tilde = dt^2 M^{-1} grad E(y).

    M is the diagonal particle mass matrix. For this spring problem,
    r_tilde is not exactly y-y*, because the spring makes the energy nonlinear,
    but near a minimizer it has the same order of magnitude as the position
    error and vanishes exactly at every stationary point.
    """
    residual = stationarity_residual(
        y,
        q,
        masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    mass_per_coordinate = masses.repeat_interleave(3, dim=-1)
    return (physical.dt**2) * residual / mass_per_coordinate


def dimensionless_residual_input(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_length_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Return u = dt^2 M^{-1} grad E(y) / s."""
    return (
        mass_preconditioned_residual(y, q, masses, physical)
        / residual_length_scale
    )


def physical_energy_scale(
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_length_scale: float,
) -> float:
    """
    Characteristic physical energy m_ref * s^2 / dt^2.

    Dividing the physical energy by this positive scalar changes only gradient
    magnitude, never the minimizer or descent direction.
    """
    reference_mass = float(masses.detach().mean().item())
    return reference_mass * residual_length_scale**2 / physical.dt**2


class MLPOptimizer(nn.Module):
    """Bias-free residual optimizer for a fixed five-particle chain.

        u       = dt^2 M^{-1} grad E(y) / s
        delta_y = s * W2 * Identity(W1 * u)

    Both the input and output have STATE_DIM=15 components. Because both
    linear layers are bias-free, every stationary state is an exact fixed
    point of the learned iteration.
    """

    def __init__(self, residual_length_scale: float) -> None:
        super().__init__()
        if residual_length_scale <= 0.0:
            raise ValueError("residual_length_scale must be positive.")

        self.linear1 = nn.Linear(STATE_DIM, HIDDEN_DIM, bias=False)
        self.activation = nn.Identity()
        self.linear2 = nn.Linear(HIDDEN_DIM, STATE_DIM, bias=False)

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
        normalized_residual = dimensionless_residual_input(
            y,
            q,
            masses,
            physical,
            self.residual_length_scale,
        )
        hidden = self.activation(self.linear1(normalized_residual))
        dimensionless_update = self.linear2(hidden)
        return self.residual_length_scale * dimensionless_update


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    dataset: DatasetBundle,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    applied_delta = model(
        y,
        dataset.q,
        dataset.masses,
        physical=physical,
    )
    return y + applied_delta, applied_delta


# ============================================================
# 5. Evaluation and validation selection
# ============================================================


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
            {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95)),
                "max": float(np.max(finite)),
            }
        )
    return result


def _statistics_by_step(values: np.ndarray, prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stat_name in ["mean", "median", "p95", "max", "num_nonfinite"]:
        result[f"{prefix}_{stat_name}_by_step"] = []
    for step_index in range(values.shape[1]):
        stats = _statistics(values[:, step_index])
        for stat_name, value in stats.items():
            result[f"{prefix}_{stat_name}_by_step"].append(value)
    final_stats = _statistics(values[:, -1])
    for stat_name, value in final_stats.items():
        result[f"final_{prefix}_{stat_name}"] = value
    return result


def _selected_step_indices(steps: int, report_steps: Sequence[int]) -> list[int]:
    selected = sorted(set([0, steps, *[step for step in report_steps if 0 <= step <= steps]]))
    return selected


def _state_metric_tensors(
    *,
    y: torch.Tensor,
    batch: DatasetBundle,
    exact_energy: torch.Tensor,
    physical: PhysicalConfig,
) -> dict[str, torch.Tensor]:
    energy = variational_energy(
        y,
        batch.q,
        batch.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    point_errors = torch.linalg.vector_norm(
        reshape_points(y) - reshape_points(batch.exact_y), dim=-1
    )
    spring_length_error = torch.mean(
        torch.abs(
            spring_lengths_from_state(y)
            - spring_lengths_from_state(batch.exact_y)
        ),
        dim=-1,
    )
    metrics: dict[str, torch.Tensor] = {
        "residual": stationarity_residual_norm(
            y,
            batch.q,
            batch.masses,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        ),
        "energy_gap": energy - exact_energy,
        "exact_error": torch.linalg.vector_norm(y - batch.exact_y, dim=-1),
        "particle_mean_error": torch.mean(point_errors, dim=-1),
        "particle_max_error": torch.max(point_errors, dim=-1).values,
        "spring_length_error": spring_length_error,
    }
    for particle_index in range(NUM_PARTICLES):
        metrics[f"point{particle_index + 1}_error"] = point_errors[..., particle_index]
    return metrics


@torch.no_grad()
def _evaluate_solver_on_dataset(
    *,
    solver: str,
    model: MLPOptimizer | None,
    dataset_cpu: DatasetBundle,
    physical: PhysicalConfig,
    steps: int,
    batch_size: int,
    report_steps: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if solver not in {"learned", "full_newton"}:
        raise ValueError(f"Unsupported solver {solver!r}.")
    if solver == "learned":
        if model is None:
            raise ValueError("A model is required for learned evaluation.")
        model.eval()

    selected_steps = _selected_step_indices(steps, report_steps)
    metric_batches: dict[str, list[torch.Tensor]] = {}
    problem_index_batches: list[torch.Tensor] = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for start_index in range(0, len(dataset_cpu), batch_size):
        end_index = min(start_index + batch_size, len(dataset_cpu))
        batch = DatasetBundle(
            initial_y=dataset_cpu.initial_y[start_index:end_index],
            q=dataset_cpu.q[start_index:end_index],
            masses=dataset_cpu.masses[start_index:end_index],
            exact_y=dataset_cpu.exact_y[start_index:end_index],
            problem_index=dataset_cpu.problem_index[start_index:end_index],
            metadata={},
        ).to(device)
        y = batch.initial_y.clone()
        exact_energy = variational_energy(
            batch.exact_y,
            batch.q,
            batch.masses,
            g=physical.g,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        )
        metric_steps: dict[str, list[torch.Tensor]] = {}

        for step in range(steps + 1):
            current_metrics = _state_metric_tensors(
                y=y,
                batch=batch,
                exact_energy=exact_energy,
                physical=physical,
            )
            for metric_name, values in current_metrics.items():
                metric_steps.setdefault(metric_name, []).append(
                    values.detach().cpu()
                )
            if step < steps:
                if solver == "learned":
                    assert model is not None
                    y, _ = apply_model_update(model, y, batch, physical)
                else:
                    y, _ = apply_newton_update(
                        y, batch.q, batch.masses, physical
                    )

        for metric_name, values in metric_steps.items():
            metric_batches.setdefault(metric_name, []).append(
                torch.stack(values, dim=1)
            )
        problem_index_batches.append(batch.problem_index.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    arrays = {
        name: torch.cat(batches, dim=0).numpy().astype(float)
        for name, batches in metric_batches.items()
    }
    problem_indices = torch.cat(problem_index_batches, dim=0).numpy().astype(int)
    for values in arrays.values():
        values[~np.isfinite(values)] = np.nan

    result: dict[str, Any] = {
        "solver": solver,
        "steps": steps,
        "num_points": len(dataset_cpu),
        "selected_report_steps": selected_steps,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_point_per_iteration": (
            elapsed_seconds / (len(dataset_cpu) * steps)
        ),
    }
    for prefix, values in arrays.items():
        result.update(_statistics_by_step(values, prefix))

    per_problem: dict[str, Any] = {}
    for problem_index in sorted(np.unique(problem_indices).tolist()):
        mask = problem_indices == problem_index
        problem_record: dict[str, Any] = {
            "problem_index": int(problem_index),
            "num_points": int(mask.sum()),
            "steps": {},
        }
        for step in selected_steps:
            step_record: dict[str, Any] = {}
            for prefix, values in arrays.items():
                step_record[prefix] = _statistics(values[mask, step])
            problem_record["steps"][str(step)] = step_record
        per_problem[str(problem_index)] = problem_record
    result["per_problem"] = per_problem
    return result


@torch.no_grad()
def evaluate_model_on_dataset(
    *,
    model: MLPOptimizer,
    dataset_cpu: DatasetBundle,
    physical: PhysicalConfig,
    steps: int,
    batch_size: int,
    report_steps: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    return _evaluate_solver_on_dataset(
        solver="learned",
        model=model,
        dataset_cpu=dataset_cpu,
        physical=physical,
        steps=steps,
        batch_size=batch_size,
        report_steps=report_steps,
        device=device,
    )


@torch.no_grad()
def evaluate_newton_on_dataset(
    *,
    dataset_cpu: DatasetBundle,
    physical: PhysicalConfig,
    steps: int,
    batch_size: int,
    report_steps: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    return _evaluate_solver_on_dataset(
        solver="full_newton",
        model=None,
        dataset_cpu=dataset_cpu,
        physical=physical,
        steps=steps,
        batch_size=batch_size,
        report_steps=report_steps,
        device=device,
    )


def evaluate_newton_baseline(
    *,
    datasets: dict[str, DatasetBundle],
    physical: PhysicalConfig,
    config: RuntimeConfig,
    device: torch.device,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, dataset in datasets.items():
        print(f"Evaluating Newton baseline on {name} ({len(dataset):,} states)...")
        results[name] = evaluate_newton_on_dataset(
            dataset_cpu=dataset,
            physical=physical,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps,
            device=device,
        )
    return results


def validation_selection_key(metrics: dict[str, Any]) -> tuple[float, ...] | None:
    values = (
        float(metrics["final_residual_num_nonfinite"]),
        float(metrics["final_residual_p95"]),
        float(metrics["final_exact_error_p95"]),
        float(metrics["final_energy_gap_p95"]),
    )
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def worst_problem_final_residual_p95(metrics: dict[str, Any]) -> float:
    final_step = str(metrics["steps"])
    values = []
    for record in metrics["per_problem"].values():
        value = float(record["steps"][final_step]["residual"]["p95"])
        if math.isfinite(value):
            values.append(value)
    return max(values) if values else float("nan")


# ============================================================
# 6. Plotting
# ============================================================


def plot_reference_sequence_and_split(
    *,
    problems: Sequence[TimeStepProblem],
    split: ProblemSplit,
    save_path: Path,
) -> None:
    times = np.asarray([problem.time for problem in problems], dtype=float)
    positions = np.asarray(
        [reshape_points(problem.p_n).detach().cpu().numpy() for problem in problems],
        dtype=float,
    )
    radii = np.asarray([problem.sampling_radius for problem in problems], dtype=float)
    spring_lengths = np.asarray(
        [spring_lengths_from_state(problem.p_n).detach().cpu().numpy() for problem in problems],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    coordinate_names = ["x", "y", "z"]
    for coordinate, coordinate_name in enumerate(coordinate_names):
        ax = axes[0, coordinate]
        for particle_index in range(NUM_PARTICLES):
            ax.plot(
                times,
                positions[:, particle_index, coordinate],
                label=f"particle {particle_index + 1}",
            )
        ax.set_title(f"Reference current-state {coordinate_name}-coordinate")
        ax.set_xlabel("Physical time")
        ax.legend(fontsize=8)

    for edge_index in range(NUM_SPRINGS):
        axes[1, 0].plot(
            times,
            spring_lengths[:, edge_index],
            label=f"spring {edge_index + 1}",
        )
    axes[1, 0].set_title("Current-state spring lengths")
    axes[1, 0].set_xlabel("Physical time")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(times, radii)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Per-problem sampling radius")
    axes[1, 1].set_xlabel("Physical time")
    axes[1, 1].set_ylabel(r"$\|p^n-y_n^*\|_\infty$")

    split_rows = [
        (split.train_indices, 0, "train"),
        (split.validation_indices, 1, "validation"),
        (split.interpolation_test_indices, 2, "interpolation test"),
        (split.extrapolation_test_indices, 3, "extrapolation test"),
    ]
    for indices, level, label in split_rows:
        axes[1, 2].scatter(indices, [level] * len(indices), label=label)
    axes[1, 2].set_yticks([0, 1, 2, 3])
    axes[1, 2].set_yticklabels([row[2] for row in split_rows])
    axes[1, 2].set_xlabel("Physical problem index")
    axes[1, 2].set_title("Problem-level train/validation/test split")

    for ax in axes.reshape(-1):
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_training_and_validation_curves(
    *,
    train_log: Sequence[dict[str, Any]],
    validation_log: Sequence[dict[str, Any]],
    best_epoch: int | None,
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(23, 5))
    train_epochs = [record["epoch"] for record in train_log]
    axes[0].plot(
        train_epochs,
        [finite_plot_value(record["training_gap_for_readability"]) for record in train_log],
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Training trajectory energy-sum gap")
    axes[0].set_xlabel("Epoch")

    val_epochs = [record["epoch"] for record in validation_log]
    axes[1].plot(
        val_epochs,
        [finite_plot_value(record["metrics"]["final_residual_p95"]) for record in validation_log],
        marker="o",
        markersize=3,
        label="pooled p95",
    )
    axes[1].plot(
        val_epochs,
        [finite_plot_value(record["worst_problem_final_residual_p95"]) for record in validation_log],
        marker="s",
        markersize=3,
        label="worst problem p95",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Validation residual after fixed rollout")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(
        val_epochs,
        [finite_plot_value(record["metrics"]["final_exact_error_p95"]) for record in validation_log],
        marker="o",
        markersize=3,
    )
    axes[2].set_yscale("log")
    axes[2].set_title("Validation exact-solution error p95")
    axes[2].set_xlabel("Epoch")

    axes[3].plot(
        val_epochs,
        [finite_plot_value(record["metrics"]["final_energy_gap_p95"]) for record in validation_log],
        marker="o",
        markersize=3,
    )
    axes[3].set_yscale("log")
    axes[3].set_title("Validation energy gap p95")
    axes[3].set_xlabel("Epoch")

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, linestyle="--", alpha=0.7)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_metrics(
    *,
    metrics: dict[str, Any],
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("residual", "Residual"),
        ("energy_gap", "Energy gap"),
        ("exact_error", "Exact-solution error"),
    ]
    steps = list(range(metrics["steps"] + 1))
    for ax, (prefix, metric_title) in zip(axes, specs):
        for stat, marker in [("mean", "o"), ("median", "s"), ("p95", "^")]:
            values = [
                finite_plot_value(value)
                for value in metrics[f"{prefix}_{stat}_by_step"]
            ]
            ax.plot(steps, values, marker=marker, markersize=3, label=stat)
        ax.set_yscale("log")
        ax.set_title(metric_title)
        ax.set_xlabel("Learned-solver iteration")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def _extract_problem_stat(
    metrics: dict[str, Any],
    problem_index: int,
    step: int,
    metric: str,
    stat: str,
) -> float:
    return float(
        metrics["per_problem"][str(problem_index)]["steps"][str(step)][metric][stat]
    )


def plot_metric_vs_physical_time(
    *,
    metrics: dict[str, Any],
    problems: Sequence[TimeStepProblem],
    problem_indices: Sequence[int],
    report_steps: Sequence[int],
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    specs = [
        ("residual", "p95", "Residual p95"),
        ("energy_gap", "p95", "Energy gap p95"),
        ("exact_error", "p95", "Exact error p95"),
    ]
    times = [problems[index].time for index in problem_indices]
    valid_steps = [step for step in report_steps if str(step) in metrics["per_problem"][str(problem_indices[0])]["steps"]]
    for ax, (metric, stat, metric_title) in zip(axes, specs):
        for step in valid_steps:
            values = [
                finite_plot_value(
                    _extract_problem_stat(metrics, index, step, metric, stat)
                )
                for index in problem_indices
            ]
            ax.plot(times, values, marker="o", markersize=3, label=f"solver step {step}")
        ax.set_yscale("log")
        ax.set_xlabel("Physical time of independent problem")
        ax.set_title(metric_title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_special_state_vs_time(
    *,
    current_metrics: dict[str, Any],
    exact_metrics: dict[str, Any],
    problems: Sequence[TimeStepProblem],
    problem_indices: Sequence[int],
    report_steps: Sequence[int],
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    specs = [
        ("residual", "Residual"),
        ("energy_gap", "Energy gap"),
        ("exact_error", "Exact error"),
    ]
    times = [problems[index].time for index in problem_indices]
    valid_steps = [
        step
        for step in report_steps
        if str(step) in current_metrics["per_problem"][str(problem_indices[0])]["steps"]
    ]
    for row, (metrics, row_name) in enumerate(
        [(current_metrics, "start from physical current state"), (exact_metrics, "start from exact solution")]
    ):
        for col, (metric, metric_title) in enumerate(specs):
            ax = axes[row, col]
            for step in valid_steps:
                values = [
                    finite_plot_value(
                        _extract_problem_stat(metrics, index, step, metric, "mean")
                    )
                    for index in problem_indices
                ]
                ax.plot(times, values, marker="o", markersize=3, label=f"solver step {step}")
            ax.set_yscale("log")
            ax.set_xlabel("Physical time")
            ax.set_title(f"{row_name}: {metric_title}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(title, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)



def plot_learned_vs_newton_rollout(
    *,
    learned_metrics: dict[str, Any],
    newton_metrics: dict[str, Any],
    learned_name: str,
    title: str,
    save_path: Path,
) -> None:
    """Compare pooled p95 convergence at every solver iteration."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("residual", "Residual p95"),
        ("energy_gap", "Energy gap p95"),
        ("exact_error", "Exact-solution error p95"),
    ]
    steps = list(range(learned_metrics["steps"] + 1))
    for ax, (prefix, metric_title) in zip(axes, specs):
        ax.plot(
            steps,
            [
                finite_plot_value(value)
                for value in learned_metrics[f"{prefix}_p95_by_step"]
            ],
            marker="o",
            markersize=3,
            label=learned_name,
        )
        ax.plot(
            steps,
            [
                finite_plot_value(value)
                for value in newton_metrics[f"{prefix}_p95_by_step"]
            ],
            marker="s",
            markersize=3,
            label="full Newton",
        )
        ax.set_yscale("log")
        ax.set_title(metric_title)
        ax.set_xlabel("Solver iteration")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_all_solver_final_metrics(
    *,
    summaries: Sequence[dict[str, Any]],
    newton_results: dict[str, Any],
    save_path: Path,
) -> None:
    solver_names = [record["experiment_name"] for record in summaries] + [
        "full_newton"
    ]
    split_names = ["interpolation_test", "extrapolation_test"]
    metrics = [
        ("final_residual_p95", "Residual p95"),
        ("final_energy_gap_p95", "Energy gap p95"),
        ("final_exact_error_p95", "Exact error p95"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    x = np.arange(len(solver_names))
    for row, split_name in enumerate(split_names):
        for col, (metric_key, title) in enumerate(metrics):
            values = [
                finite_plot_value(
                    record["best_checkpoint_test"][split_name][metric_key]
                )
                for record in summaries
            ]
            values.append(finite_plot_value(newton_results[split_name][metric_key]))
            axes[row, col].bar(x, values)
            axes[row, col].set_xticks(x)
            axes[row, col].set_xticklabels(
                solver_names, rotation=15, ha="right"
            )
            axes[row, col].set_yscale("log")
            axes[row, col].set_title(f"{split_name}: {title}")
            axes[row, col].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Learned optimizers and full-Newton baseline", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_current_state_residual_all_solvers(
    *,
    summaries: Sequence[dict[str, Any]],
    newton_results: dict[str, Any],
    problems: Sequence[TimeStepProblem],
    problem_indices: Sequence[int],
    report_steps: Sequence[int],
    save_path: Path,
) -> None:
    valid_steps = [
        step
        for step in report_steps
        if str(step)
        in newton_results["current_state_all_test"]["per_problem"][
            str(problem_indices[0])
        ]["steps"]
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), squeeze=False)
    times = [problems[index].time for index in problem_indices]
    for ax, step in zip(axes.reshape(-1), valid_steps):
        for record in summaries:
            metrics = record["best_checkpoint_test"]["current_state_all_test"]
            values = [
                finite_plot_value(
                    _extract_problem_stat(
                        metrics, index, step, "residual", "mean"
                    )
                )
                for index in problem_indices
            ]
            ax.plot(
                times,
                values,
                marker="o",
                markersize=3,
                label=record["experiment_name"],
            )
        newton_metrics = newton_results["current_state_all_test"]
        newton_values = [
            finite_plot_value(
                _extract_problem_stat(
                    newton_metrics, index, step, "residual", "mean"
                )
            )
            for index in problem_indices
        ]
        ax.plot(
            times,
            newton_values,
            marker="s",
            markersize=3,
            label="full_newton",
        )
        ax.axvline(0.8, linestyle="--", alpha=0.6, label="extrapolation boundary")
        ax.set_yscale("log")
        ax.set_title(f"Current-state start: residual after {step} iterations")
        ax.set_xlabel("Physical time")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.reshape(-1)[len(valid_steps):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_final_metrics(
    *,
    summaries: Sequence[dict[str, Any]],
    save_path: Path,
) -> None:
    model_names = [record["experiment_name"] for record in summaries]
    split_names = ["interpolation_test", "extrapolation_test"]
    metrics = [
        ("final_residual_p95", "Residual p95"),
        ("final_energy_gap_p95", "Energy gap p95"),
        ("final_exact_error_p95", "Exact error p95"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    x = np.arange(len(model_names))
    for row, split_name in enumerate(split_names):
        for col, (metric_key, title) in enumerate(metrics):
            values = [
                finite_plot_value(
                    record["best_checkpoint_test"][split_name][metric_key]
                )
                for record in summaries
            ]
            axes[row, col].bar(x, values)
            axes[row, col].set_xticks(x)
            axes[row, col].set_xticklabels(model_names, rotation=15, ha="right")
            axes[row, col].set_yscale("log")
            axes[row, col].set_title(f"{split_name}: {title}")
            axes[row, col].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Validation-selected checkpoint comparison", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_current_state_residual_model_comparison(
    *,
    summaries: Sequence[dict[str, Any]],
    problems: Sequence[TimeStepProblem],
    problem_indices: Sequence[int],
    report_steps: Sequence[int],
    save_path: Path,
) -> None:
    valid_steps = [
        step
        for step in report_steps
        if str(step)
        in summaries[0]["best_checkpoint_test"]["current_state_all_test"]["per_problem"][str(problem_indices[0])]["steps"]
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), squeeze=False)
    times = [problems[index].time for index in problem_indices]
    for ax, step in zip(axes.reshape(-1), valid_steps):
        for record in summaries:
            metrics = record["best_checkpoint_test"]["current_state_all_test"]
            values = [
                finite_plot_value(
                    _extract_problem_stat(metrics, index, step, "residual", "mean")
                )
                for index in problem_indices
            ]
            ax.plot(times, values, marker="o", markersize=3, label=record["experiment_name"])
        ax.axvline(0.8, linestyle="--", alpha=0.6, label="extrapolation boundary")
        ax.set_yscale("log")
        ax.set_title(f"Current-state start: residual after {step} solver iterations")
        ax.set_xlabel("Physical time")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.reshape(-1)[len(valid_steps):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. Training one model and evaluating checkpoints
# ============================================================


def compact_test_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "final_residual_mean",
        "final_residual_median",
        "final_residual_p95",
        "final_residual_max",
        "final_residual_num_nonfinite",
        "final_energy_gap_mean",
        "final_energy_gap_median",
        "final_energy_gap_p95",
        "final_energy_gap_max",
        "final_energy_gap_num_nonfinite",
        "final_exact_error_mean",
        "final_exact_error_median",
        "final_exact_error_p95",
        "final_exact_error_max",
        "final_exact_error_num_nonfinite",
    ]
    return {key: metrics[key] for key in keys}


def evaluate_checkpoint(
    *,
    model: MLPOptimizer,
    state_dict: dict[str, torch.Tensor],
    datasets: dict[str, DatasetBundle],
    physical: PhysicalConfig,
    config: RuntimeConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    results: dict[str, Any] = {}
    for name, dataset in datasets.items():
        results[name] = evaluate_model_on_dataset(
            model=model,
            dataset_cpu=dataset,
            physical=physical,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps,
            device=device,
        )
    return results



@torch.no_grad()
def one_step_quality_diagnostics(
    *,
    model: MLPOptimizer,
    dataset: DatasetBundle,
    physical: PhysicalConfig,
) -> dict[str, Any]:
    """
    Diagnose whether one learned iteration improves the current training states.

    Exact solutions are used here only for reporting. Nothing in this function
    participates in backward propagation.
    """
    model.eval()
    y0 = dataset.initial_y
    y1, delta = apply_model_update(model, y0, dataset, physical)

    error_before = torch.linalg.vector_norm(y0 - dataset.exact_y, dim=-1)
    error_after = torch.linalg.vector_norm(y1 - dataset.exact_y, dim=-1)

    residual_before = stationarity_residual_norm(
        y0,
        dataset.q,
        dataset.masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    residual_after = stationarity_residual_norm(
        y1,
        dataset.q,
        dataset.masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )

    energy_before = variational_energy(
        y0,
        dataset.q,
        dataset.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    energy_after = variational_energy(
        y1,
        dataset.q,
        dataset.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )

    ideal_update = dataset.exact_y - y0
    flat_delta = delta.reshape(-1)
    flat_ideal = ideal_update.reshape(-1)
    denominator = (
        torch.linalg.vector_norm(flat_delta)
        * torch.linalg.vector_norm(flat_ideal)
    )
    if float(denominator.item()) > 0.0:
        cosine = float(
            (torch.dot(flat_delta, flat_ideal) / denominator).item()
        )
    else:
        cosine = float("nan")

    nonzero_mask = error_before > 1e-15
    contraction = error_after[nonzero_mask] / error_before[nonzero_mask]
    if contraction.numel() == 0:
        contraction_stats = {
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    else:
        contraction_stats = {
            "mean": float(contraction.mean().item()),
            "median": float(torch.quantile(contraction, 0.5).item()),
            "p95": float(torch.quantile(contraction, 0.95).item()),
            "max": float(contraction.max().item()),
        }

    fixed_point_delta = model(
        dataset.exact_y,
        dataset.q,
        dataset.masses,
        physical=physical,
    )
    fixed_point_update_norm = torch.linalg.vector_norm(
        fixed_point_delta, dim=-1
    )

    return {
        "mean_error_before": float(error_before.mean().item()),
        "mean_error_after": float(error_after.mean().item()),
        "mean_residual_before": float(residual_before.mean().item()),
        "mean_residual_after": float(residual_after.mean().item()),
        "mean_update_norm": float(
            torch.linalg.vector_norm(delta, dim=-1).mean().item()
        ),
        "update_ideal_cosine": cosine,
        "sample_error_improvement_fraction": float(
            (error_after < error_before).to(TORCH_DTYPE).mean().item()
        ),
        "sample_residual_improvement_fraction": float(
            (residual_after < residual_before).to(TORCH_DTYPE).mean().item()
        ),
        "sample_energy_improvement_fraction": float(
            (energy_after < energy_before).to(TORCH_DTYPE).mean().item()
        ),
        "contraction_ratio": contraction_stats,
        "fixed_point_update_mean": float(
            fixed_point_update_norm.mean().item()
        ),
        "fixed_point_update_max": float(
            fixed_point_update_norm.max().item()
        ),
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
    problems: Sequence[TimeStepProblem],
) -> dict[str, Any]:
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)

    model = MLPOptimizer(
        residual_length_scale=config.residual_length_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training = training_cpu.to(device)

    energy_scale = physical_energy_scale(
        training.masses,
        physical,
        config.residual_length_scale,
    )
    initial_energy = variational_energy(
        training.initial_y,
        training.q,
        training.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    ).detach()
    exact_energy = variational_energy(
        training.exact_y,
        training.q,
        training.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    ).detach()

    print("\n" + "=" * 100)
    print(f"Experiment: {experiment_name}")
    print(
        f"device={device}, dtype={TORCH_DTYPE}, "
        "architecture=15_dimless_residual->64->identity->15_dimless_update, "
        f"optimizer=Adam(lr={LEARNING_RATE:.0e}), "
        f"length_scale={config.residual_length_scale:.3e}, "
        f"energy_scale={energy_scale:.3e}"
    )
    print(
        f"training_points={len(training_cpu):,}, "
        f"training_problems={training_cpu.metadata['num_problems']}, "
        f"validation_points={len(validation_cpu):,}"
    )
    print(
        "bias_free=True; orthogonal_first_layer=True; "
        "zero_output_layer=True; gradient_clip="
        f"{config.gradient_clip_norm:g}"
    )
    print("no_early_stopping=True; validation_selects_best_checkpoint_only")
    print("=" * 100)

    train_log: list[dict[str, Any]] = []
    quality_diagnostic_log: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    best_epoch: int | None = None
    diverged = False
    divergence_epoch: int | None = None
    divergence_reason: str | None = None
    start_time = time.perf_counter()

    for epoch_index in range(config.epochs):
        epoch_number = epoch_index + 1
        rollout_k = get_k_for_epoch(epoch_index, config)
        model.train()
        y = training.initial_y
        optimizer.zero_grad(set_to_none=True)

        trajectory_objective = torch.zeros(
            (), dtype=TORCH_DTYPE, device=device
        )
        trajectory_energy_sum = torch.zeros(
            (), dtype=TORCH_DTYPE, device=device
        )
        trajectory_energy_gap = torch.zeros(
            (), dtype=TORCH_DTYPE, device=device
        )

        for _ in range(rollout_k):
            y, _ = apply_model_update(model, y, training, physical)
            current_energy = variational_energy(
                y,
                training.q,
                training.masses,
                g=physical.g,
                dt=physical.dt,
                spring_k=physical.spring_k,
                rest_length=physical.rest_length,
            )

            # This is the original physical energy, shifted by the fixed initial
            # energy and divided by a positive dimensional scale:
            #
            #   L_hat = (E(y) - stop_grad(E(y0))) / E_scale.
            #
            # The shift has zero derivative and the positive scale changes only
            # gradient magnitude. Therefore the minimizer and descent direction
            # are exactly those of the original physical energy. Unlike the
            # projectile quadratic case, we do NOT replace the nonlinear spring
            # energy with a residual-squared objective.
            trajectory_objective = trajectory_objective + (
                (current_energy - initial_energy) / energy_scale
            ).mean()

            trajectory_energy_sum = (
                trajectory_energy_sum + current_energy.mean()
            )
            trajectory_energy_gap = trajectory_energy_gap + (
                current_energy - exact_energy
            ).mean()

        if not bool(torch.isfinite(trajectory_objective)):
            diverged = True
            divergence_epoch = epoch_number
            divergence_reason = "non-finite dimensionless trajectory objective"
            gradient_norm = float("nan")
        else:
            try:
                trajectory_objective.backward()
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=config.gradient_clip_norm,
                    ).item()
                )
                if not math.isfinite(gradient_norm):
                    diverged = True
                    divergence_epoch = epoch_number
                    divergence_reason = "non-finite gradient norm"
                else:
                    optimizer.step()
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    diverged = True
                    divergence_epoch = epoch_number
                    divergence_reason = f"CUDA out of memory: {error}"
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

        if not diverged and not is_model_finite(model):
            diverged = True
            divergence_epoch = epoch_number
            divergence_reason = "non-finite model parameter after optimizer.step"

        if diverged:
            print(
                f"Training stopped at epoch={divergence_epoch}: "
                f"{divergence_reason}"
            )
            break

        objective_value = float(trajectory_objective.item())
        energy_value = float(trajectory_energy_sum.item())
        training_gap = float(trajectory_energy_gap.item())
        train_log.append(
            {
                "epoch": epoch_number,
                "K": rollout_k,
                "dimensionless_training_objective": objective_value,
                "trajectory_energy_sum": energy_value,
                "training_gap_for_readability": training_gap,
                "gradient_norm_before_clip": gradient_norm,
                "gradient_was_clipped": (
                    gradient_norm > config.gradient_clip_norm
                ),
            }
        )

        should_diagnose = (
            epoch_number == 1
            or epoch_number % config.diagnostic_interval == 0
            or epoch_number == config.epochs
        )
        if should_diagnose:
            diagnostics = one_step_quality_diagnostics(
                model=model,
                dataset=training,
                physical=physical,
            )
            diagnostics.update(
                {
                    "epoch": epoch_number,
                    "training_K": rollout_k,
                    "gradient_norm_before_clip": gradient_norm,
                    "gradient_was_clipped": (
                        gradient_norm > config.gradient_clip_norm
                    ),
                }
            )
            quality_diagnostic_log.append(diagnostics)

        should_validate = (
            epoch_number % config.validation_interval == 0
            or epoch_number == config.epochs
        )
        if should_validate:
            validation_metrics = evaluate_model_on_dataset(
                model=model,
                dataset_cpu=validation_cpu,
                physical=physical,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                report_steps=config.report_steps,
                device=device,
            )
            current_key = validation_selection_key(validation_metrics)
            worst_problem_p95 = worst_problem_final_residual_p95(
                validation_metrics
            )
            validation_log.append(
                {
                    "epoch": epoch_number,
                    "training_K": rollout_k,
                    "selection_key": (
                        list(current_key)
                        if current_key is not None
                        else None
                    ),
                    "worst_problem_final_residual_p95": worst_problem_p95,
                    "metrics": validation_metrics,
                }
            )
            if current_key is not None and (
                best_key is None or current_key < best_key
            ):
                best_key = current_key
                best_epoch = epoch_number
                best_validation_metrics = copy.deepcopy(validation_metrics)
                best_state_dict = state_dict_to_cpu(model)

            elapsed = time.perf_counter() - start_time
            message = (
                f"Epoch {epoch_number:5d} | K={rollout_k} | "
                f"objective={objective_value:.4e} | "
                f"train_gap={training_gap:.4e} | "
                f"grad_norm={gradient_norm:.4e} | "
                f"val_res_p95="
                f"{validation_metrics['final_residual_p95']:.4e} | "
                f"worst_problem_res_p95={worst_problem_p95:.4e} | "
                f"best_epoch={best_epoch} | elapsed={elapsed:.1f}s"
            )
            if (
                quality_diagnostic_log
                and quality_diagnostic_log[-1]["epoch"] == epoch_number
            ):
                diagnostic = quality_diagnostic_log[-1]
                message += (
                    f" | one_step_cos="
                    f"{diagnostic['update_ideal_cosine']:.4f}"
                    f" | improve_frac="
                    f"{diagnostic['sample_error_improvement_fraction']:.4f}"
                    f" | contraction_p95="
                    f"{diagnostic['contraction_ratio']['p95']:.4e}"
                )
            print(message)

    last_state_dict = state_dict_to_cpu(model)
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(last_state_dict)
        best_epoch = train_log[-1]["epoch"] if train_log else 0

    torch.save(last_state_dict, experiment_dir / "last_model_state_dict.pt")
    torch.save(
        best_state_dict,
        experiment_dir / "best_validation_model_state_dict.pt",
    )
    torch.save(
        best_state_dict,
        experiment_dir / "mlp_optimizer_state_dict.pt",
    )

    best_results = evaluate_checkpoint(
        model=model,
        state_dict=best_state_dict,
        datasets=evaluation_datasets,
        physical=physical,
        config=config,
        device=device,
    )
    last_results = evaluate_checkpoint(
        model=model,
        state_dict=last_state_dict,
        datasets=evaluation_datasets,
        physical=physical,
        config=config,
        device=device,
    )

    report = {
        "config": {
            "experiment_name": experiment_name,
            "torch_dtype": str(TORCH_DTYPE),
            "device": str(device),
            "architecture": (
                "6-dimensional mass-preconditioned residual -> "
                "64 -> identity -> 6-dimensional update"
            ),
            "activation": ACTIVATION_NAME,
            "optimizer": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
            "residual_length_scale": config.residual_length_scale,
            "energy_scale": energy_scale,
            "input_definition": (
                "dt^2 * M^{-1} * stationarity_residual / "
                "residual_length_scale"
            ),
            "output_definition": (
                "residual_length_scale * dimensionless network output"
            ),
            "bias_free": True,
            "first_layer_initialization": "orthogonal",
            "output_layer_initialization": "zero",
            "fixed_point_property": (
                "zero stationarity residual maps exactly to zero update"
            ),
            "epochs_requested": config.epochs,
            "completed_epochs": len(train_log),
            "validation_interval": config.validation_interval,
            "diagnostic_interval": config.diagnostic_interval,
            "gradient_clip_norm": config.gradient_clip_norm,
            "evaluation_steps": config.evaluation_steps,
            "report_steps": list(config.report_steps),
            "initial_K": config.initial_k,
            "K_increase_interval": config.k_increase_interval,
            "K_increase_amount": config.k_increase_amount,
            "max_K": config.max_k,
            "training_mode": "full_batch",
            "no_early_stopping": True,
            "checkpoint_selection": (
                "lexicographic: final residual nonfinite count, pooled "
                "residual p95, pooled exact-error p95, pooled energy-gap p95"
            ),
            "loss": (
                "sum over rollout steps of "
                "(physical_energy(y_k)-stop_grad(physical_energy(y_0)))"
                "/energy_scale"
            ),
            "loss_equivalence": (
                "same minimizer and parameter-gradient direction as the "
                "original physical variational energy"
            ),
            "backpropagation": (
                "full unroll without detach; one backward per epoch"
            ),
        },
        "training_dataset": training_cpu.metadata,
        "validation_dataset": validation_cpu.metadata,
        "training_status": {
            "diverged": diverged,
            "divergence_epoch": divergence_epoch,
            "divergence_reason": divergence_reason,
            "elapsed_seconds": time.perf_counter() - start_time,
        },
        "best_validation_checkpoint": {
            "epoch": best_epoch,
            "selection_key": (
                list(best_key) if best_key is not None else None
            ),
            "validation_metrics": best_validation_metrics,
        },
        "train_log": train_log,
        "quality_diagnostic_log": quality_diagnostic_log,
        "validation_log": validation_log,
        "final_test": {
            "best_validation_checkpoint": best_results,
            "last_epoch_checkpoint": last_results,
        },
    }
    save_json(report, experiment_dir / "optimization_report.json")

    if not config.skip_plots:
        plot_training_and_validation_curves(
            train_log=train_log,
            validation_log=validation_log,
            best_epoch=best_epoch,
            title=experiment_name,
            save_path=experiment_dir
            / "training_and_validation_curves.png",
        )
        for split_name in ["interpolation_test", "extrapolation_test"]:
            plot_rollout_metrics(
                metrics=best_results[split_name],
                title=f"{experiment_name}: {split_name}",
                save_path=experiment_dir
                / f"{split_name}_rollout_metrics.png",
            )
            problem_indices = evaluation_datasets[split_name].metadata[
                "problem_indices"
            ]
            plot_metric_vs_physical_time(
                metrics=best_results[split_name],
                problems=problems,
                problem_indices=problem_indices,
                report_steps=config.report_steps,
                title=f"{experiment_name}: {split_name} by physical time",
                save_path=experiment_dir
                / f"{split_name}_metrics_vs_physical_time.png",
            )

        all_test_indices = evaluation_datasets[
            "current_state_all_test"
        ].metadata["problem_indices"]
        plot_special_state_vs_time(
            current_metrics=best_results["current_state_all_test"],
            exact_metrics=best_results["exact_state_all_test"],
            problems=problems,
            problem_indices=all_test_indices,
            report_steps=config.report_steps,
            title=(
                f"{experiment_name}: physical-current and exact "
                "fixed-point tests"
            ),
            save_path=experiment_dir
            / "special_state_metrics_vs_physical_time.png",
        )

    summary = {
        "experiment_name": experiment_name,
        "training_num_problems": training_cpu.metadata["num_problems"],
        "training_num_points": len(training_cpu),
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": (
            list(best_key) if best_key is not None else None
        ),
        "diverged": diverged,
        "best_checkpoint_test": best_results,
        "last_checkpoint_test": last_results,
        "compact_best_checkpoint_test": {
            name: compact_test_metrics(metrics)
            for name, metrics in best_results.items()
        },
        "training_curve_for_summary": downsample_log(train_log),
        "quality_diagnostic_log": quality_diagnostic_log,
        "validation_curve_for_summary": downsample_log(validation_log),
    }
    save_json(summary, experiment_dir / "experiment_summary.json")
    return summary




def run_physics_consistency_checks(
    *,
    physical: PhysicalConfig,
    problems: Sequence[TimeStepProblem],
) -> dict[str, Any]:
    """Compare analytic derivatives with autograd and verify chain symmetry."""
    base = problems[0]
    perturbation = torch.linspace(
        -2.0e-3, 2.0e-3, STATE_DIM, dtype=TORCH_DTYPE
    )
    y = (base.exact_y + perturbation).detach().requires_grad_(True)
    q = base.q.detach()
    masses = base.masses.detach()

    energy = variational_energy(
        y,
        q,
        masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    autograd_gradient = torch.autograd.grad(energy, y, create_graph=False)[0]
    analytic_gradient = stationarity_residual(
        y.detach(),
        q,
        masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )

    def scalar_energy(state: torch.Tensor) -> torch.Tensor:
        return variational_energy(
            state,
            q,
            masses,
            g=physical.g,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        )

    autograd_hessian = torch.autograd.functional.hessian(scalar_energy, y.detach())
    analytic_hessian = variational_hessian(
        y.detach(),
        masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )

    original_energy = variational_energy(
        y.detach(),
        q,
        masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    reversed_energy = variational_energy(
        reverse_chain(y.detach()),
        reverse_chain(q),
        reverse_masses(masses),
        g=physical.g,
        dt=physical.dt,
        spring_k=tuple(reversed(physical.spring_stiffness)),
        rest_length=tuple(reversed(physical.rest_lengths)),
    )

    gradient_max_abs_error = float(
        torch.max(torch.abs(analytic_gradient - autograd_gradient)).item()
    )
    hessian_max_abs_error = float(
        torch.max(torch.abs(analytic_hessian - autograd_hessian)).item()
    )
    reversal_energy_abs_error = float(
        torch.abs(original_energy - reversed_energy).item()
    )
    maximum_reference_residual = max(
        problem.exact_residual for problem in problems
    )
    checks = {
        "gradient_max_abs_error": gradient_max_abs_error,
        "hessian_max_abs_error": hessian_max_abs_error,
        "chain_reversal_energy_abs_error": reversal_energy_abs_error,
        "maximum_reference_residual": maximum_reference_residual,
        "gradient_check_passed": gradient_max_abs_error < 1e-8,
        "hessian_check_passed": hessian_max_abs_error < 1e-7,
        "chain_reversal_check_passed": reversal_energy_abs_error < 1e-10,
        "reference_residual_check_passed": (
            maximum_reference_residual <= REFERENCE_ACCEPTABLE_RESIDUAL
        ),
    }
    if not all(
        checks[key]
        for key in [
            "gradient_check_passed",
            "hessian_check_passed",
            "chain_reversal_check_passed",
            "reference_residual_check_passed",
        ]
    ):
        raise RuntimeError(f"Physics consistency check failed: {checks}")
    return checks


# ============================================================
# 8. Command-line interface and main experiment orchestration
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare independent multi-time-step and single-problem "
            "learned optimizers for the five-particle four-spring open-chain problem."
        )
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--total-time-steps", type=int, default=DEFAULT_TOTAL_TIME_STEPS)
    parser.add_argument(
        "--train-points-per-problem",
        type=int,
        default=DEFAULT_TRAIN_POINTS_PER_PROBLEM,
    )
    parser.add_argument(
        "--eval-points-per-problem",
        type=int,
        default=DEFAULT_EVAL_POINTS_PER_PROBLEM,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL
    )
    parser.add_argument(
        "--evaluation-steps", type=int, default=DEFAULT_EVALUATION_STEPS
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=DEFAULT_EVALUATION_BATCH_SIZE,
    )
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument(
        "--k-increase-interval", type=int, default=DEFAULT_K_INCREASE_INTERVAL
    )
    parser.add_argument(
        "--k-increase-amount", type=int, default=DEFAULT_K_INCREASE_AMOUNT
    )
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument(
        "--report-steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_REPORT_STEPS),
    )
    parser.add_argument(
        "--residual-length-scale",
        type=float,
        default=DEFAULT_RESIDUAL_LENGTH_SCALE,
        help=(
            "Characteristic length s used in u=dt^2 M^{-1}gradE/s "
            "and delta_y=s*MLP(u). Default: 5e-2."
        ),
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_NORM,
        help="Global gradient-norm clipping threshold. Default: 1.0.",
    )
    parser.add_argument(
        "--diagnostic-interval",
        type=int,
        default=DEFAULT_DIAGNOSTIC_INTERVAL,
        help="Interval for one-step quality diagnostics. Default: 500.",
    )
    parser.add_argument(
        "--skip-single-problem-baseline",
        action="store_true",
        help="Train only the multi-problem model.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--save-datasets",
        action="store_true",
        help="Save all generated tensor datasets. Off by default to reduce output size.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    validate_even_size("train_points_per_problem", int(args.train_points_per_problem))
    validate_even_size("eval_points_per_problem", int(args.eval_points_per_problem))
    if int(args.total_time_steps) != 100:
        raise ValueError("The confirmed experiment requires --total-time-steps 100.")
    if int(args.epochs) <= 0:
        raise ValueError("epochs must be positive.")
    if int(args.validation_interval) <= 0:
        raise ValueError("validation_interval must be positive.")
    if int(args.evaluation_steps) <= 0:
        raise ValueError("evaluation_steps must be positive.")
    if int(args.evaluation_batch_size) <= 0:
        raise ValueError("evaluation_batch_size must be positive.")
    if int(args.initial_k) <= 0 or int(args.max_k) <= 0:
        raise ValueError("initial_k and max_k must be positive.")
    if int(args.initial_k) > int(args.max_k):
        raise ValueError("initial_k cannot exceed max_k.")
    if int(args.k_increase_interval) <= 0 or int(args.k_increase_amount) <= 0:
        raise ValueError("K schedule parameters must be positive.")
    if float(args.residual_length_scale) <= 0.0:
        raise ValueError("residual_length_scale must be positive.")
    if float(args.gradient_clip_norm) <= 0.0:
        raise ValueError("gradient_clip_norm must be positive.")
    if int(args.diagnostic_interval) <= 0:
        raise ValueError("diagnostic_interval must be positive.")

    report_steps = tuple(
        sorted(
            set(
                int(step)
                for step in args.report_steps
                if 0 < int(step) <= int(args.evaluation_steps)
            )
        )
    )
    if int(args.evaluation_steps) not in report_steps:
        report_steps = tuple(sorted(set([*report_steps, int(args.evaluation_steps)])))

    return RuntimeConfig(
        total_time_steps=int(args.total_time_steps),
        train_points_per_problem=int(args.train_points_per_problem),
        eval_points_per_problem=int(args.eval_points_per_problem),
        epochs=int(args.epochs),
        validation_interval=int(args.validation_interval),
        evaluation_steps=int(args.evaluation_steps),
        evaluation_batch_size=int(args.evaluation_batch_size),
        initial_k=int(args.initial_k),
        k_increase_interval=int(args.k_increase_interval),
        k_increase_amount=int(args.k_increase_amount),
        max_k=int(args.max_k),
        report_steps=report_steps,
        residual_length_scale=float(args.residual_length_scale),
        gradient_clip_norm=float(args.gradient_clip_norm),
        diagnostic_interval=int(args.diagnostic_interval),
        device=str(args.device),
        run_single_problem_baseline=not bool(args.skip_single_problem_baseline),
        skip_plots=bool(args.skip_plots),
        save_datasets=bool(args.save_datasets),
    )


def problem_to_record(problem: TimeStepProblem) -> dict[str, Any]:
    return {
        "index": problem.index,
        "time": problem.time,
        "p_n": tensor_to_list(problem.p_n),
        "v_n": tensor_to_list(problem.v_n),
        "q": tensor_to_list(problem.q),
        "masses": tensor_to_list(problem.masses),
        "exact_y": tensor_to_list(problem.exact_y),
        "sampling_radius_linf": problem.sampling_radius,
        "exact_energy": problem.exact_energy,
        "exact_residual": problem.exact_residual,
        "current_spring_lengths": tensor_to_list(spring_lengths_from_state(problem.p_n)),
        "exact_spring_lengths": tensor_to_list(spring_lengths_from_state(problem.exact_y)),
    }


def main() -> None:
    config = validate_args(parse_args())
    physical = default_physical_config()
    output_dir = create_output_directory()
    device = torch.device(config.device)
    validate_device(device)

    problems = generate_reference_sequence(physical, config.total_time_steps)
    split = build_problem_split(config.total_time_steps)
    physics_checks = run_physics_consistency_checks(
        physical=physical,
        problems=problems,
    )

    print(f"Output directory: {output_dir}")
    print(f"Runtime config: {asdict(config)}")
    print(f"Physical config: {asdict(physical)}")
    print(f"torch default dtype: {torch.get_default_dtype()}")
    print(f"Physics consistency checks: {physics_checks}")
    print(
        "Problem split sizes: "
        f"train={len(split.train_indices)}, validation={len(split.validation_indices)}, "
        f"interpolation_test={len(split.interpolation_test_indices)}, "
        f"extrapolation_test={len(split.extrapolation_test_indices)}"
    )

    save_json(
        {
            "runtime_config": asdict(config),
            "fixed_model_configuration": {
                "architecture": (
                    "15D dimensionless mass-preconditioned residual "
                    "-> 64 -> identity -> 15D dimensionless update"
                ),
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "residual_length_scale": config.residual_length_scale,
                "gradient_clip_norm": config.gradient_clip_norm,
                "bias_free": True,
                "first_layer_initialization": "orthogonal",
                "output_layer_initialization": "zero",
                "torch_dtype": str(TORCH_DTYPE),
                "default_device": DEFAULT_DEVICE,
            },
            "physical_config": asdict(physical),
            "topology": {"type": "open_chain", "num_particles": NUM_PARTICLES, "edges": [[0, 1], [1, 2], [2, 3], [3, 4]]},
            "problem_split": asdict(split),
            "physics_consistency_checks": physics_checks,
        },
        output_dir / "runtime_config.json",
    )
    save_json(
        {
            "description": (
                "High-accuracy numerical reference sequence used only to define independent "
                "time-step optimization problems. No learned prediction is propagated."
            ),
            "problems": [problem_to_record(problem) for problem in problems],
        },
        output_dir / "reference_time_step_problems.json",
    )

    if not config.skip_plots:
        plot_reference_sequence_and_split(
            problems=problems,
            split=split,
            save_path=output_dir / "reference_sequence_and_problem_split.png",
        )

    multi_training = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.train_indices,
        points_per_problem=config.train_points_per_problem,
        base_seed=TRAIN_SOBOL_SEED,
        role="multi_problem_training",
        include_explicit_train_points=True,
    )
    single_training = build_dataset_for_problem_indices(
        problems=problems,
        indices=(0,),
        points_per_problem=config.train_points_per_problem,
        base_seed=TRAIN_SOBOL_SEED,
        role="single_problem_training",
        include_explicit_train_points=True,
    )
    validation = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.validation_indices,
        points_per_problem=config.eval_points_per_problem,
        base_seed=VALIDATION_SOBOL_SEED,
        role="validation",
        include_explicit_train_points=False,
    )
    interpolation_test = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.interpolation_test_indices,
        points_per_problem=config.eval_points_per_problem,
        base_seed=INTERPOLATION_TEST_SOBOL_SEED,
        role="interpolation_test",
        include_explicit_train_points=False,
    )
    extrapolation_test = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.extrapolation_test_indices,
        points_per_problem=config.eval_points_per_problem,
        base_seed=EXTRAPOLATION_TEST_SOBOL_SEED,
        role="extrapolation_test",
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

    # Full Newton uses the same initial states, physical energy, datasets,
    # rollout length, and metrics as the learned optimizer. It does not use
    # exact_y in its update; exact_y is only used by the shared evaluator.
    newton_results = evaluate_newton_baseline(
        datasets=evaluation_datasets,
        physical=physical,
        config=config,
        device=device,
    )
    newton_dir = output_dir / "newton_baseline"
    newton_dir.mkdir(parents=True, exist_ok=True)
    newton_report = {
        "solver": "undamped full Newton",
        "gradient": "analytic gradient of the original five-particle variational energy",
        "hessian": "analytic 15x15 block-tridiagonal Hessian of the original variational energy",
        "uses_exact_solution_in_update": False,
        "residual_tolerance": NEWTON_RESIDUAL_TOLERANCE,
        "fixed_iterations": config.evaluation_steps,
        "report_steps": list(config.report_steps),
        "results": newton_results,
        "compact_results": {
            name: compact_test_metrics(metrics)
            for name, metrics in newton_results.items()
        },
    }
    save_json(newton_report, newton_dir / "newton_baseline_report.json")

    if not config.skip_plots:
        for split_name in ["interpolation_test", "extrapolation_test"]:
            plot_rollout_metrics(
                metrics=newton_results[split_name],
                title=f"full Newton: {split_name}",
                save_path=newton_dir / f"{split_name}_rollout_metrics.png",
            )
            plot_metric_vs_physical_time(
                metrics=newton_results[split_name],
                problems=problems,
                problem_indices=evaluation_datasets[split_name].metadata[
                    "problem_indices"
                ],
                report_steps=config.report_steps,
                title=f"full Newton: {split_name} by physical time",
                save_path=newton_dir / f"{split_name}_metrics_vs_physical_time.png",
            )
        plot_special_state_vs_time(
            current_metrics=newton_results["current_state_all_test"],
            exact_metrics=newton_results["exact_state_all_test"],
            problems=problems,
            problem_indices=split.all_test_indices,
            report_steps=config.report_steps,
            title="full Newton: physical-current and exact fixed-point tests",
            save_path=newton_dir / "special_state_metrics_vs_physical_time.png",
        )

    dataset_metadata = {
        "split_unit": "physical_time_step_problem",
        "multi_problem_training": multi_training.metadata,
        "single_problem_training": single_training.metadata,
        "validation": validation.metadata,
        "interpolation_test": interpolation_test.metadata,
        "extrapolation_test": extrapolation_test.metadata,
        "current_state_all_test": current_state_all_test.metadata,
        "exact_state_all_test": exact_state_all_test.metadata,
        "network_input": (
            "dimensionless mass-preconditioned stationarity residual; "
            "no absolute-state feature normalization"
        ),
        "residual_length_scale": config.residual_length_scale,
        "no_problem_leakage": True,
    }
    save_json(dataset_metadata, output_dir / "dataset_metadata.json")

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

    summaries: list[dict[str, Any]] = []
    summaries.append(
        run_experiment(
            experiment_name="multi_problem",
            training_cpu=multi_training,
            validation_cpu=validation,
            evaluation_datasets=evaluation_datasets,
            output_dir=output_dir,
            config=config,
            physical=physical,
            problems=problems,
        )
    )

    if config.run_single_problem_baseline:
        summaries.append(
            run_experiment(
                experiment_name="single_problem_baseline",
                training_cpu=single_training,
                validation_cpu=validation,
                evaluation_datasets=evaluation_datasets,
                output_dir=output_dir,
                config=config,
                physical=physical,
                problems=problems,
            )
        )

    overall_report = {
        "experiment_type": "independent_multi_time_step_problem_generalization",
        "purpose": (
            "Test whether one learned iterative solver can solve distinct physical "
            "time-step optimization problems without continuous learned rollout."
        ),
        "runtime_config": asdict(config),
        "physical_config": asdict(physical),
        "problem_split": asdict(split),
        "physics_consistency_checks": physics_checks,
        "dataset_metadata": dataset_metadata,
        "network": {
            "architecture": (
                "15D dimensionless mass-preconditioned residual "
                "-> 64 -> identity -> 15D dimensionless update"
            ),
            "input_features": (
                "dt^2 * M^{-1} * stationarity_residual / "
                "residual_length_scale"
            ),
            "output_mapping": (
                "residual_length_scale * dimensionless network output"
            ),
            "residual_length_scale": config.residual_length_scale,
            "bias_free": True,
            "first_layer_orthogonal_initialized": True,
            "final_layer_zero_initialized": True,
            "fixed_point_property": (
                "zero residual gives exactly zero learned update"
            ),
            "torch_dtype": str(TORCH_DTYPE),
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "gradient_clip_norm": config.gradient_clip_norm,
            "loss": (
                "sum of dimensionless shifted physical variational energies; "
                "same minimizer and gradient direction as original energy"
            ),
            "full_batch": True,
            "no_early_stopping": True,
        },
        "newton_baseline": newton_report,
        "experiments": summaries,
    }
    save_json(overall_report, output_dir / "all_experiments_summary.json")

    if not config.skip_plots and len(summaries) > 1:
        plot_model_comparison_final_metrics(
            summaries=summaries,
            save_path=output_dir / "model_comparison_final_metrics.png",
        )
        plot_current_state_residual_model_comparison(
            summaries=summaries,
            problems=problems,
            problem_indices=split.all_test_indices,
            report_steps=config.report_steps,
            save_path=output_dir / "current_state_residual_model_comparison.png",
        )

    if not config.skip_plots:
        plot_all_solver_final_metrics(
            summaries=summaries,
            newton_results=newton_results,
            save_path=output_dir / "learned_vs_newton_final_metrics.png",
        )
        plot_current_state_residual_all_solvers(
            summaries=summaries,
            newton_results=newton_results,
            problems=problems,
            problem_indices=split.all_test_indices,
            report_steps=config.report_steps,
            save_path=output_dir / "current_state_residual_learned_vs_newton.png",
        )
        for record in summaries:
            experiment_dir = output_dir / record["experiment_name"]
            for split_name in ["interpolation_test", "extrapolation_test"]:
                plot_learned_vs_newton_rollout(
                    learned_metrics=record["best_checkpoint_test"][split_name],
                    newton_metrics=newton_results[split_name],
                    learned_name=record["experiment_name"],
                    title=(
                        f"{record['experiment_name']} vs full Newton: "
                        f"{split_name}"
                    ),
                    save_path=experiment_dir
                    / f"{split_name}_learned_vs_newton.png",
                )

    print("\n" + "=" * 100)
    print("All requested experiments completed.")
    print(f"Summary JSON: {output_dir / 'all_experiments_summary.json'}")
    for record in summaries:
        interpolation = record["compact_best_checkpoint_test"]["interpolation_test"]
        extrapolation = record["compact_best_checkpoint_test"]["extrapolation_test"]
        print(
            f"- {record['experiment_name']}: best_epoch={record['best_validation_epoch']}, "
            f"interp_res_p95={interpolation['final_residual_p95']:.4e}, "
            f"extra_res_p95={extrapolation['final_residual_p95']:.4e}, "
            f"diverged={record['diverged']}"
        )
    newton_interpolation = newton_report["compact_results"]["interpolation_test"]
    newton_extrapolation = newton_report["compact_results"]["extrapolation_test"]
    print(
        "- full_newton: "
        f"interp_res_p95={newton_interpolation['final_residual_p95']:.4e}, "
        f"extra_res_p95={newton_extrapolation['final_residual_p95']:.4e}"
    )


if __name__ == "__main__":
    main()
