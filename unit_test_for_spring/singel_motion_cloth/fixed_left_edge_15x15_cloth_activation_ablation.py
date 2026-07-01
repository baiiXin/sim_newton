"""
Fixed-left-edge 15x15 triangular cloth activation-function ablation experiment.

Main changes from the fixed-first-vertex five-particle chain experiment
-----------------------------------------------------------------------
1. The physical system is a 15x15 cloth-like triangular spring mesh with two
   fixed vertices: the left-top and left-bottom corners.
2. Only the remaining 223 vertices are optimization variables, so the learned
   optimizer input/output dimension is 669.
3. The spring network is exactly the edge set of a triangulated 15x15 mesh:
      - 210 horizontal edges,
      - 210 vertical edges,
      - 196 alternating cell diagonals.
   The opposite diagonal is not added, and no bending springs are added.
4. The same validation-selected checkpoint is evaluated against:
      - the learned MLP iteration,
      - fixed-step gradient descent,
      - undamped full Newton.
5. Gradient-descent step size is selected only on the validation set, and the
   validation residual-vs-step-size curve is saved as a figure.
6. Extrapolation-test problems are ranked by the solver-independent p95 initial
   residual over sampled starts. After training, the hardest problem whose physical
   current state reaches a finite MLP residual below 1e-4 is selected for rollout.
7. In addition to p95 curves, worst-case maxima and non-finite update records are
   saved as separate figures and detailed JSON fields.

Activation ablation
-------------------
The script trains three otherwise identical learned optimizers using ReLU, Tanh,
and Identity activations. All datasets, seeds, initialization rules, optimizer
settings, training budgets, validation selection rules, and evaluation metrics are
shared, so activation type is the only experimental variable.

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

GRID_ROWS = 15
GRID_COLS = 15
SPATIAL_DIM = 3
NUM_PARTICLES = GRID_ROWS * GRID_COLS
FIXED_VERTEX_INDICES = (0, (GRID_ROWS - 1) * GRID_COLS)  # left-top and left-bottom
FREE_VERTEX_INDICES = tuple(
    index for index in range(NUM_PARTICLES) if index not in set(FIXED_VERTEX_INDICES)
)
NUM_FREE_PARTICLES = len(FREE_VERTEX_INDICES)
FREE_STATE_DIM = NUM_FREE_PARTICLES * SPATIAL_DIM
HIDDEN_DIM = FREE_STATE_DIM


def grid_index(row: int, col: int) -> int:
    return row * GRID_COLS + col


def build_triangular_cloth_topology() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Return unique spring edges and triangle faces for an alternating 15x15 mesh."""
    edge_set: set[tuple[int, int]] = set()
    faces: list[tuple[int, int, int]] = []

    def add_edge(a: int, b: int) -> None:
        if a == b:
            raise ValueError("Degenerate edge")
        edge_set.add((min(a, b), max(a, b)))

    # Horizontal and vertical structural mesh edges.
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS - 1):
            add_edge(grid_index(row, col), grid_index(row, col + 1))
    for row in range(GRID_ROWS - 1):
        for col in range(GRID_COLS):
            add_edge(grid_index(row, col), grid_index(row + 1, col))

    # One alternating diagonal per cell: this is the triangle-mesh spring network.
    for row in range(GRID_ROWS - 1):
        for col in range(GRID_COLS - 1):
            tl = grid_index(row, col)
            tr = grid_index(row, col + 1)
            bl = grid_index(row + 1, col)
            br = grid_index(row + 1, col + 1)
            if (row + col) % 2 == 0:
                add_edge(tl, br)
                faces.append((tl, tr, br))
                faces.append((tl, br, bl))
            else:
                add_edge(bl, tr)
                faces.append((tl, tr, bl))
                faces.append((tr, br, bl))

    edges = tuple(sorted(edge_set))
    faces_tuple = tuple(faces)
    expected_edges = GRID_ROWS * (GRID_COLS - 1) + (GRID_ROWS - 1) * GRID_COLS + (GRID_ROWS - 1) * (GRID_COLS - 1)
    expected_faces = 2 * (GRID_ROWS - 1) * (GRID_COLS - 1)
    if len(edges) != expected_edges:
        raise RuntimeError(f"Expected {expected_edges} spring edges, got {len(edges)}")
    if len(faces_tuple) != expected_faces:
        raise RuntimeError(f"Expected {expected_faces} triangle faces, got {len(faces_tuple)}")
    return edges, faces_tuple


SPRING_EDGES, TRIANGLE_FACES = build_triangular_cloth_topology()
NUM_SPRINGS = len(SPRING_EDGES)
NUM_TRIANGLES = len(TRIANGLE_FACES)
GLOBAL_TO_FREE_INDEX = tuple(
    FREE_VERTEX_INDICES.index(index) if index in FREE_VERTEX_INDICES else -1
    for index in range(NUM_PARTICLES)
)

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

DEFAULT_DEVICE = "cuda:1"
DEFAULT_TOTAL_TIME_STEPS = 100
DEFAULT_TRAIN_POINTS_PER_PROBLEM = 100
DEFAULT_EVAL_POINTS_PER_PROBLEM = 256
DEFAULT_EPOCHS = 1_000
DEFAULT_VALIDATION_INTERVAL = 500
DEFAULT_DIAGNOSTIC_INTERVAL = 500
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8192
NEWTON_MAX_BATCH_SIZE = 8
# Full Newton forms a dense 669x669 Hessian per active sample. Learned and GD
# evaluation retain the original large batch; only Newton is capped internally.
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 10_000
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5
DEFAULT_REPORT_STEPS = (1, 5, 10, 50)
DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_ACTIVATIONS = ("relu", "tanh", "identity")

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
HARD_CASE_FINAL_RESIDUAL_THRESHOLD = 1e-4


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
    activation_names: tuple[str, ...]
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
    def fixed_positions(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(self.p0[index] for index in FIXED_VERTEX_INDICES)


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
    # 15x15 cloth patch. Vertices are stored in row-major order.
    # Fixed vertices are the left-top and left-bottom corners.
    spacing = 0.5
    height = 1.20
    p0 = tuple(
        (col * spacing, -row * spacing, height)
        for row in range(GRID_ROWS)
        for col in range(GRID_COLS)
    )
    v0 = tuple((0.0, 0.0, 0.0) for _ in range(NUM_PARTICLES))
    rest_lengths = tuple(
        math.dist(p0[i], p0[j]) for i, j in SPRING_EDGES
    )
    return PhysicalConfig(
        masses=tuple(1.0 for _ in range(NUM_PARTICLES)),
        g=9.8,
        dt=0.01,
        spring_stiffness=tuple(2500.0 for _ in range(NUM_SPRINGS)),
        rest_lengths=rest_lengths,
        p0=p0,
        v0=v0,
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
# 2. Fixed-left-edge 15x15 triangular-cloth physics
# =============================================================================


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
    """Implicit-Euler variational energy for the 223 free cloth vertices."""
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
    active = residual > residual_tolerance
    if not bool(torch.any(active)):
        return y, torch.zeros_like(y)
    rhs = torch.where(active, -gradient, torch.zeros_like(gradient))
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
            mass_per_coordinate = masses_batch.repeat_interleave(SPATIAL_DIM, dim=-1)
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


def generate_reference_sequence(
    physical: PhysicalConfig,
    total_steps: int,
) -> list[TimeStepProblem]:
    p_n = torch.tensor(physical.p0, dtype=TORCH_DTYPE)
    v_n = torch.tensor(physical.v0, dtype=TORCH_DTYPE)
    fixed = list(FIXED_VERTEX_INDICES)
    p_n[fixed, :] = torch.tensor(physical.fixed_positions, dtype=TORCH_DTYPE)
    v_n[fixed, :] = 0.0
    free_masses = torch.tensor([physical.masses[i] for i in FREE_VERTEX_INDICES], dtype=TORCH_DTYPE)
    problems: list[TimeStepProblem] = []

    for index in range(total_steps):
        q_free = make_q_free(p_n, v_n, physical)
        initial_y = free_state_from_full(p_n)
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
        "mode": f"scrambled_sobol_{FREE_STATE_DIM}d_linf_cube",
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
            free_state_from_full(problem.p_n_full),
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
            y0 = free_state_from_full(problem.p_n_full)
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
    def __init__(self, residual_length_scale: float, activation_name: str) -> None:
        super().__init__()
        if residual_length_scale <= 0.0:
            raise ValueError("residual_length_scale must be positive")
        normalized_name = activation_name.strip().lower()
        activation_factories: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "identity": nn.Identity,
        }
        if normalized_name not in activation_factories:
            raise ValueError(
                f"Unsupported activation {activation_name!r}; "
                f"choose from {tuple(activation_factories)}"
            )

        self.activation_name = normalized_name
        self.linear1 = nn.Linear(FREE_STATE_DIM, HIDDEN_DIM, bias=False)
        self.activation = activation_factories[normalized_name]()
        self.linear2 = nn.Linear(HIDDEN_DIM, FREE_STATE_DIM, bias=False)

        # Keep the exact same initialization scale for all three variants so the
        # activation function is the only changed factor.
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
        # Fixed vertices are reconstructed from constants, so this should be exactly zero.
        "fixed_vertex_max_error": torch.zeros_like(point_errors[..., 0]),
    }
    for free_index, global_index in enumerate(FREE_VERTEX_INDICES):
        metrics[f"point{global_index + 1}_error"] = point_errors[..., free_index]
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

    requested_batch_size = int(batch_size)
    effective_batch_size = (
        min(requested_batch_size, NEWTON_MAX_BATCH_SIZE)
        if solver == "full_newton"
        else requested_batch_size
    )

    metric_batches: dict[str, list[torch.Tensor]] = {}
    nonfinite_update_batches: list[torch.Tensor] = []
    problem_batches: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for start in range(0, len(dataset_cpu), effective_batch_size):
        end = min(start + effective_batch_size, len(dataset_cpu))
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
        # Column 0 corresponds to the initial state and therefore has no update event.
        nonfinite_updates: list[torch.Tensor] = [
            torch.zeros(y.shape[0], dtype=torch.bool)
        ]
        for step in range(steps + 1):
            for name, values in _state_metrics(y, batch, exact_energy, physical).items():
                step_values.setdefault(name, []).append(values.detach().cpu())
            if step == steps:
                break

            if solver == "learned":
                assert model is not None
                y_next, delta = apply_model_update(model, y, batch.q, batch.masses, physical)
            elif solver == "gradient_descent":
                assert gd_step_size is not None
                y_next, delta = apply_gradient_descent_update(
                    y, batch.q, batch.masses, physical, gd_step_size
                )
            else:
                try:
                    y_next, delta = apply_newton_update(y, batch.q, batch.masses, physical)
                except RuntimeError:
                    # Retry sample-by-sample so one failed Newton update does not erase
                    # the finite trajectories of the other samples in this batch.
                    next_rows: list[torch.Tensor] = []
                    delta_rows: list[torch.Tensor] = []
                    for row in range(y.shape[0]):
                        try:
                            row_next, row_delta = apply_newton_update(
                                y[row : row + 1],
                                batch.q[row : row + 1],
                                batch.masses[row : row + 1],
                                physical,
                            )
                        except RuntimeError:
                            row_next = torch.full_like(y[row : row + 1], float("nan"))
                            row_delta = torch.full_like(y[row : row + 1], float("nan"))
                        next_rows.append(row_next)
                        delta_rows.append(row_delta)
                    y_next = torch.cat(next_rows, dim=0)
                    delta = torch.cat(delta_rows, dim=0)

            update_finite = torch.isfinite(y_next).all(dim=-1) & torch.isfinite(delta).all(dim=-1)
            nonfinite_updates.append((~update_finite).detach().cpu())
            # Preserve the last finite iterate for failed samples and continue all
            # remaining fixed iterations so failures are recorded instead of aborting.
            y = torch.where(update_finite[:, None], y_next, y)

        for name, values in step_values.items():
            metric_batches.setdefault(name, []).append(torch.stack(values, dim=1))
        nonfinite_update_batches.append(torch.stack(nonfinite_updates, dim=1))
        problem_batches.append(batch.problem_index.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    arrays = {
        name: torch.cat(values, dim=0).numpy().astype(float)
        for name, values in metric_batches.items()
    }
    nonfinite_update = torch.cat(nonfinite_update_batches, dim=0).numpy().astype(bool)
    problem_indices = torch.cat(problem_batches).numpy().astype(int)
    for values in arrays.values():
        values[~np.isfinite(values)] = np.nan

    nonfinite_update_count_by_step = np.count_nonzero(
        nonfinite_update, axis=0
    ).astype(int)
    ever_nonfinite_by_step = np.maximum.accumulate(nonfinite_update, axis=1)
    nonfinite_state_count_by_step = np.count_nonzero(
        ever_nonfinite_by_step, axis=0
    ).astype(int)
    nonfinite_events: list[dict[str, Any]] = []
    for sample_index in range(nonfinite_update.shape[0]):
        bad_steps = np.flatnonzero(nonfinite_update[sample_index])
        if bad_steps.size:
            nonfinite_events.append(
                {
                    "dataset_index": int(sample_index),
                    "problem_index": int(problem_indices[sample_index]),
                    "first_nonfinite_update_step": int(bad_steps[0]),
                    "nonfinite_update_steps": [int(v) for v in bad_steps.tolist()],
                }
            )

    result: dict[str, Any] = {
        "solver": solver,
        "steps": steps,
        "num_points": len(dataset_cpu),
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": effective_batch_size,
        "selected_report_steps": _selected_steps(steps, report_steps),
        "elapsed_seconds": elapsed,
        "seconds_per_point_per_iteration": elapsed / max(len(dataset_cpu) * steps, 1),
        "nonfinite_update_count_by_step": nonfinite_update_count_by_step.tolist(),
        "nonfinite_state_count_by_step": nonfinite_state_count_by_step.tolist(),
        "final_nonfinite_state_count": int(nonfinite_state_count_by_step[-1]),
        "num_samples_ever_nonfinite": len(nonfinite_events),
        "nonfinite_state_events": nonfinite_events,
        "nonfinite_handling": (
            "Keep the last finite iterate of each failed sample, record the failed "
            "update step, and continue the fixed evaluation budget."
        ),
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
        record = {
            "problem_index": int(problem_index),
            "num_points": int(mask.sum()),
            "steps": {},
            "num_samples_ever_nonfinite": int(
                np.count_nonzero(np.any(nonfinite_update[mask], axis=1))
            ),
        }
        for step in selected:
            record["steps"][str(step)] = {
                name: _statistics(values[mask, step]) for name, values in arrays.items()
            }
            record["steps"][str(step)]["nonfinite_update_count"] = int(
                np.count_nonzero(nonfinite_update[mask, step])
            )
            record["steps"][str(step)]["nonfinite_state_count"] = int(
                np.count_nonzero(ever_nonfinite_by_step[mask, step])
            )
        per_problem[str(problem_index)] = record
    result["per_problem"] = per_problem
    return result

def validation_selection_key(metrics: dict[str, Any]) -> tuple[float, ...] | None:
    values = (
        float(metrics["final_nonfinite_state_count"]),
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


def plot_gradient_descent_step_size_selection(
    gd_selection: dict[str, Any], save_path: Path
) -> None:
    records = gd_selection.get("records", [])
    if not records:
        return
    alphas = np.asarray([float(r["step_size"]) for r in records], dtype=float)
    residual_p95 = np.asarray(
        [float(r["metrics"].get("final_residual_p95", float("nan"))) for r in records],
        dtype=float,
    )
    residual_max = np.asarray(
        [float(r["metrics"].get("final_residual_max", float("nan"))) for r in records],
        dtype=float,
    )
    nonfinite = np.asarray(
        [int(r["metrics"].get("final_nonfinite_state_count", 0)) for r in records],
        dtype=int,
    )
    selected = float(gd_selection["selected_step_size"])
    selected_index = int(np.argmin(np.abs(alphas - selected)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, values, ylabel, title in [
        (axes[0], residual_p95, "Final residual p95", "Validation p95"),
        (axes[1], residual_max, "Final residual maximum", "Validation worst case"),
    ]:
        ax.plot(alphas, np.maximum(values, PLOT_FLOOR), marker="o")
        ax.scatter(
            [selected],
            [max(values[selected_index], PLOT_FLOOR)],
            s=90,
            zorder=3,
            label=f"selected {selected:.1e}",
        )
        ax.axvline(selected, linestyle="--", alpha=0.75)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Gradient-descent step size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()

    axes[2].plot(alphas, nonfinite, marker="o")
    axes[2].scatter([selected], [nonfinite[selected_index]], s=90, zorder=3)
    axes[2].axvline(selected, linestyle="--", alpha=0.75)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Gradient-descent step size")
    axes[2].set_ylabel("Final non-finite state count")
    axes[2].set_title("Validation non-finite records")
    axes[2].grid(True, alpha=0.3, which="both")

    fig.suptitle("Validation selection of gradient-descent step size", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

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
    activation_name: str,
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
    model = MLPOptimizer(config.residual_length_scale, activation_name).to(device)
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
        f"architecture={FREE_STATE_DIM}->{HIDDEN_DIM}->{model.activation.__class__.__name__}->{FREE_STATE_DIM}, "
        f"points={len(training_cpu):,}, "
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
                f"val_res_max={metrics['final_residual_max']:.4e} "
                f"val_nonfinite={metrics['final_nonfinite_state_count']} "
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
        "activation_name": activation_name,
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": best_key,
        "training_dataset": training_cpu.metadata,
        "validation_dataset": validation_cpu.metadata,
        "model": {
            "architecture": (
                f"{FREE_STATE_DIM}D dimensionless mass-preconditioned residual -> "
                f"{HIDDEN_DIM} -> {model.activation.__class__.__name__} -> "
                f"{FREE_STATE_DIM}D update"
            ),
            "activation_name": activation_name,
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
        plot_validation_worst_case_and_nonfinite(
            validation_log,
            best_epoch,
            experiment_dir / "validation_worst_case_and_nonfinite.png",
        )
        for split_name in ["interpolation_test", "extrapolation_test"]:
            plot_three_solver_rollout(
                comparison[split_name],
                title=f"{experiment_name}: {split_name}",
                save_path=experiment_dir / f"{split_name}_three_solver_rollout.png",
            )
            plot_three_solver_worst_case(
                comparison[split_name],
                title=f"{experiment_name}: {split_name}",
                save_path=experiment_dir / f"{split_name}_three_solver_worst_case.png",
            )
            plot_three_solver_nonfinite(
                comparison[split_name],
                title=f"{experiment_name}: {split_name}",
                save_path=experiment_dir / f"{split_name}_three_solver_nonfinite.png",
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


def plot_validation_worst_case_and_nonfinite(
    validation_log: Sequence[dict[str, Any]],
    best_epoch: int | None,
    save_path: Path,
) -> None:
    if not validation_log:
        return
    epochs = [r["epoch"] for r in validation_log]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("final_residual_max", "Validation residual maximum", True),
        ("final_exact_error_max", "Validation exact-error maximum", True),
        ("final_nonfinite_state_count", "Validation non-finite state count", False),
    ]
    for ax, (key, title, log_scale) in zip(axes, specs):
        values = [r["metrics"].get(key, 0) for r in validation_log]
        if log_scale:
            ax.plot(epochs, [finite_plot_value(v) for v in values], marker="o")
            ax.set_yscale("log")
        else:
            ax.plot(epochs, values, marker="o")
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
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


def plot_three_solver_worst_case(
    comparison: dict[str, dict[str, Any]],
    *,
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("residual", "Residual maximum"),
        ("energy_gap", "Energy-gap maximum"),
        ("exact_error", "Exact-solution error maximum"),
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
                [finite_plot_value(v) for v in metrics[f"{metric}_max_by_step"]],
                marker="o",
                markersize=3,
                label=labels[solver_name],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Solver iteration")
        ax.set_title(metric_title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"{title}: worst-case metrics", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_three_solver_nonfinite(
    comparison: dict[str, dict[str, Any]],
    *,
    title: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = {
        "learned": "MLP",
        "gradient_descent": "gradient descent",
        "full_newton": "full Newton",
    }
    specs = [
        ("nonfinite_state_count_by_step", "Non-finite states"),
        ("residual_num_nonfinite_by_step", "Non-finite residuals"),
        ("exact_error_num_nonfinite_by_step", "Non-finite exact errors"),
    ]
    for ax, (key, metric_title) in zip(axes, specs):
        for solver_name, metrics in comparison.items():
            values = metrics.get(key, [0] * (metrics["steps"] + 1))
            ax.plot(
                range(metrics["steps"] + 1),
                values,
                marker="o",
                markersize=3,
                label=labels[solver_name],
            )
        ax.set_xlabel("Solver iteration")
        ax.set_ylabel("Count")
        ax.set_title(metric_title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"{title}: non-finite-value records", y=1.02)
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
    learned_current_state_metrics: dict[str, Any],
    residual_threshold: float = HARD_CASE_FINAL_RESIDUAL_THRESHOLD,
) -> dict[str, Any]:
    """Select the hardest extrapolation problem whose rollout start converges.

    Difficulty is solver-independent: the p95 initial residual over all sampled
    starts of each extrapolation problem. Eligibility is model-dependent: the
    validation-selected multi-problem MLP must start from the physical current
    state p_n and finish the fixed evaluation budget with a finite residual
    below ``residual_threshold``.
    """
    residuals = stationarity_residual_norm(
        dataset.initial_y, dataset.q, dataset.masses, physical
    ).numpy()
    problem_indices = dataset.problem_index.numpy()
    final_step = str(int(learned_current_state_metrics["steps"]))
    per_problem = learned_current_state_metrics["per_problem"]

    records: list[dict[str, Any]] = []
    for index in sorted(np.unique(problem_indices).tolist()):
        values = residuals[problem_indices == index]
        current_record = per_problem[str(index)]
        final_step_record = current_record["steps"][final_step]
        residual_stats = final_step_record["residual"]
        nonfinite_state_count = int(
            final_step_record.get(
                "nonfinite_state_count", residual_stats.get("num_nonfinite", 0)
            )
        )
        final_residual = float(residual_stats.get("max", float("nan")))
        converged = (
            nonfinite_state_count == 0
            and math.isfinite(final_residual)
            and final_residual < residual_threshold
        )
        records.append(
            {
                "problem_index": int(index),
                "physical_time": problems[index].time,
                "initial_residual_p95": float(np.percentile(values, 95)),
                "initial_residual_max": float(np.max(values)),
                "num_initial_states": int(values.size),
                "mlp_current_state_evaluation_steps": int(
                    learned_current_state_metrics["steps"]
                ),
                "mlp_current_state_final_residual": final_residual,
                "mlp_current_state_final_residual_num_nonfinite": int(
                    residual_stats.get("num_nonfinite", 0)
                ),
                "mlp_current_state_final_nonfinite_state_count": nonfinite_state_count,
                "eligible_for_rollout": converged,
            }
        )

    ranked = sorted(records, key=lambda r: r["initial_residual_p95"], reverse=True)
    eligible = [r for r in ranked if r["eligible_for_rollout"]]
    if not eligible:
        finite_records = [
            r for r in ranked if math.isfinite(r["mlp_current_state_final_residual"])
        ]
        best_residual = (
            min(r["mlp_current_state_final_residual"] for r in finite_records)
            if finite_records
            else float("nan")
        )
        raise RuntimeError(
            "No extrapolation problem is eligible for rollout: the validation-selected "
            f"multi-problem MLP has no finite current-state final residual below "
            f"{residual_threshold:.1e}. Best finite residual={best_residual:.6e}."
        )

    selected = eligible[0]
    problem = problems[selected["problem_index"]]
    return {
        "selection_rule": (
            "Rank extrapolation-test physical time-step problems by descending p95 "
            "stationarity residual over sampled initial states; choose the first one "
            "whose physical current state, after the fixed MLP evaluation budget, has "
            f"a finite final residual below {residual_threshold:.1e}."
        ),
        "difficulty_ranking_solver_independent": True,
        "eligibility_depends_on_validation_selected_multi_problem_mlp": True,
        "final_residual_threshold": residual_threshold,
        "num_candidates": len(ranked),
        "num_eligible_candidates": len(eligible),
        "selected": selected,
        "all_candidates_ranked_by_initial_residual_p95": ranked,
        "selected_physical_state": {
            "p_n_full": problem.p_n_full.tolist(),
            "v_n_full": problem.v_n_full.tolist(),
            "time": problem.time,
            "problem_index": problem.index,
        },
    }


def plot_hard_case_selection(selection: dict[str, Any], save_path: Path) -> None:
    records = selection["all_candidates_ranked_by_initial_residual_p95"]
    problem_indices = np.asarray([r["problem_index"] for r in records], dtype=int)
    initial_p95 = np.asarray([r["initial_residual_p95"] for r in records], dtype=float)
    initial_max = np.asarray([r["initial_residual_max"] for r in records], dtype=float)
    final_residual = np.asarray(
        [r["mlp_current_state_final_residual"] for r in records], dtype=float
    )
    nonfinite = np.asarray(
        [r["mlp_current_state_final_nonfinite_state_count"] for r in records],
        dtype=int,
    )
    eligible = np.asarray([r["eligible_for_rollout"] for r in records], dtype=bool)
    selected_index = int(selection["selected"]["problem_index"])
    threshold = float(selection["final_residual_threshold"])

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].plot(problem_indices, np.maximum(initial_p95, PLOT_FLOOR), marker="o", label="initial residual p95")
    axes[0].plot(problem_indices, np.maximum(initial_max, PLOT_FLOOR), marker="s", label="initial residual max")
    axes[0].set_yscale("log")
    axes[0].set_title("Solver-independent difficulty")
    axes[0].set_ylabel("Initial residual")
    axes[0].legend()

    finite_final = np.where(np.isfinite(final_residual), final_residual, np.nan)
    axes[1].plot(problem_indices, np.maximum(finite_final, PLOT_FLOOR), marker="o")
    axes[1].axhline(threshold, linestyle="--", label=f"threshold {threshold:.1e}")
    if np.any(eligible):
        axes[1].scatter(
            problem_indices[eligible],
            np.maximum(finite_final[eligible], PLOT_FLOOR),
            s=70,
            label="eligible",
        )
    axes[1].set_yscale("log")
    axes[1].set_title("MLP current-state final residual")
    axes[1].set_ylabel("Residual after fixed iterations")
    axes[1].legend()

    axes[2].bar(problem_indices, nonfinite)
    axes[2].set_title("Current-state non-finite records")
    axes[2].set_ylabel("Final non-finite state count")

    for ax in axes:
        ax.axvline(selected_index, linestyle=":", alpha=0.8)
        ax.set_xlabel("Extrapolation problem index")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Hard rollout case selection: problem {selected_index}", y=1.02
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def run_physics_checks(physical: PhysicalConfig, problem: TimeStepProblem) -> dict[str, Any]:
    """Validate the analytic gradient and Hessian through Hessian-vector products.

    A full autograd Hessian check was inexpensive for the 69D 5x5 problem, but
    becomes unnecessarily slow for the 669D 15x15 state. Three deterministic
    Hessian-vector probes test the same assembly without constructing a second
    dense 669x669 autograd Hessian.
    """
    perturb = torch.linspace(-2e-3, 2e-3, FREE_STATE_DIM, dtype=TORCH_DTYPE)
    y = (problem.exact_y_free + perturb).detach().requires_grad_(True)
    q = problem.q_free
    masses = problem.free_masses
    energy = variational_energy(y, q, masses, physical)
    auto_grad = torch.autograd.grad(energy, y, create_graph=True)[0]
    analytic_grad = stationarity_residual(y.detach(), q, masses, physical)
    analytic_hessian = variational_hessian(y.detach(), masses, physical)

    directions = [
        torch.linspace(-1.0, 1.0, FREE_STATE_DIM, dtype=TORCH_DTYPE),
        torch.sin(torch.arange(FREE_STATE_DIM, dtype=TORCH_DTYPE) * 0.17),
        torch.cos(torch.arange(FREE_STATE_DIM, dtype=TORCH_DTYPE) * 0.11),
    ]
    hvp_errors: list[float] = []
    for direction in directions:
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-30)
        auto_hvp = torch.autograd.grad(
            torch.sum(auto_grad * direction), y, retain_graph=True
        )[0]
        analytic_hvp = analytic_hessian @ direction
        hvp_errors.append(float(torch.max(torch.abs(auto_hvp - analytic_hvp)).item()))

    grad_error = float(torch.max(torch.abs(auto_grad.detach() - analytic_grad)).item())
    hess_error = max(hvp_errors)
    checks = {
        "gradient_max_abs_error": grad_error,
        "hessian_vector_max_abs_error": hess_error,
        # Retain the old key so downstream readers of runtime_config.json remain compatible.
        "hessian_max_abs_error": hess_error,
        "hessian_vector_probe_count": len(directions),
        "gradient_check_passed": grad_error < 1e-8,
        "hessian_check_passed": hess_error < 1e-7,
    }
    if not checks["gradient_check_passed"] or not checks["hessian_check_passed"]:
        raise RuntimeError(f"Physics check failed: {checks}")
    return checks




def _report_activation_label(report: dict[str, Any]) -> str:
    return str(report["activation_name"]).upper() if report["activation_name"] == "relu" else str(report["activation_name"]).capitalize()


def activation_report_selection_key(report: dict[str, Any]) -> tuple[float, ...]:
    key = report.get("best_validation_selection_key")
    if key is None:
        return (float("inf"),) * 4
    values = tuple(float(v) for v in key)
    if not all(math.isfinite(v) for v in values):
        return (float("inf"),) * max(len(values), 4)
    return values


def plot_activation_training_comparison(
    reports: Sequence[dict[str, Any]], save_path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for report in reports:
        label = _report_activation_label(report)
        train_log = report["train_log"]
        validation_log = report["validation_log"]
        axes[0].plot(
            [r["epoch"] for r in train_log],
            [finite_plot_value(r["training_energy_gap_sum"]) for r in train_log],
            label=label,
        )
        axes[1].plot(
            [r["epoch"] for r in validation_log],
            [finite_plot_value(r["metrics"]["final_residual_p95"]) for r in validation_log],
            marker="o",
            label=label,
        )
        axes[2].plot(
            [r["epoch"] for r in validation_log],
            [finite_plot_value(r["metrics"]["final_exact_error_p95"]) for r in validation_log],
            marker="o",
            label=label,
        )
    titles = (
        "Training energy-gap sum",
        "Validation residual p95",
        "Validation exact-error p95",
    )
    for ax, title in zip(axes, titles):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Activation-function ablation: training and validation", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_activation_solver_iteration_comparison(
    reports: Sequence[dict[str, Any]],
    *,
    split_name: str,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = (
        ("residual_p95_by_step", "Residual p95"),
        ("exact_error_p95_by_step", "Exact-solution error p95"),
        ("residual_max_by_step", "Residual maximum"),
    )
    for report in reports:
        metrics = report["evaluation"][split_name]["learned"]
        label = _report_activation_label(report)
        steps = range(int(metrics["steps"]) + 1)
        for ax, (key, _) in zip(axes, specs):
            ax.plot(
                steps,
                [finite_plot_value(v) for v in metrics[key]],
                marker="o",
                markersize=3,
                label=label,
            )
    for ax, (_, title) in zip(axes, specs):
        ax.set_yscale("log")
        ax.set_xlabel("Solver iteration")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Activation-function ablation: {split_name}", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_activation_final_metrics(
    reports: Sequence[dict[str, Any]], save_path: Path
) -> None:
    labels = [_report_activation_label(report) for report in reports]
    x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = (
        ("final_residual_p95", "Final residual p95", True),
        ("final_residual_max", "Final residual maximum", True),
        ("final_nonfinite_state_count", "Final non-finite count", False),
    )
    width = 0.36
    for split_offset, split_name in [(-width / 2, "interpolation_test"), (width / 2, "extrapolation_test")]:
        for ax, (key, _, log_scale) in zip(axes, specs):
            values = [
                report["evaluation"][split_name]["learned"].get(key, float("nan"))
                for report in reports
            ]
            if log_scale:
                values = [finite_plot_value(v) for v in values]
            ax.bar(x + split_offset, values, width=width, label=split_name)
    for ax, (_, title, log_scale) in zip(axes, specs):
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()
    fig.suptitle("Activation-function ablation: final test metrics", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 8. CLI and orchestration
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-left-edge 15x15 cloth activation ablation: ReLU, Tanh, and Identity"
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
    parser.add_argument(
        "--activations",
        nargs="+",
        choices=list(DEFAULT_ACTIVATIONS),
        default=list(DEFAULT_ACTIVATIONS),
        help="Activation variants to run; defaults to relu tanh identity.",
    )
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
    activation_names = tuple(dict.fromkeys(str(name).lower() for name in args.activations))
    if not activation_names:
        raise ValueError("At least one activation must be selected")
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
        activation_names=activation_names,
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
    print(f"Activation variants: {', '.join(config.activation_names)}")

    save_json(
        {
            "experiment_type": "activation_function_ablation",
            "controlled_variable": "hidden-layer activation function",
            "activation_names": list(config.activation_names),
            "runtime_config": asdict(config),
            "physical_config": asdict(physical),
            "problem_split": asdict(split),
            "fixed_vertex_indices": list(FIXED_VERTEX_INDICES),
            "fixed_positions": [list(p) for p in physical.fixed_positions],
            "free_state_dimension": FREE_STATE_DIM,
            "chain_reversal_augmented": False,
            "grid_rows": GRID_ROWS,
            "grid_cols": GRID_COLS,
            "spring_edges": [list(edge) for edge in SPRING_EDGES],
            "triangle_faces": [list(face) for face in TRIANGLE_FACES],
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

    training = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.train_indices,
        points_per_problem=config.train_points_per_problem,
        base_seed=TRAIN_SOBOL_SEED,
        role="activation_ablation_training",
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

    # GD and Newton do not depend on the neural activation, so evaluate them once
    # and share their results across all activation variants.
    gd_step_size, gd_selection = select_gradient_descent_step_size(
        validation=validation,
        physical=physical,
        config=config,
        device=device,
    )
    save_json(gd_selection, output_dir / "gradient_descent_step_selection.json")
    if not config.skip_plots:
        plot_gradient_descent_step_size_selection(
            gd_selection,
            output_dir / "gradient_descent_step_size_selection.png",
        )
    print(f"Selected gradient-descent step size: {gd_step_size:.3e}")

    shared_baselines: dict[str, dict[str, Any]] = {}
    for name, dataset in evaluation_datasets.items():
        print(f"Evaluating shared GD and Newton baselines on {name} ...")
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
                "activation_ablation_training": dataset_to_serializable_dict(training),
                "validation": dataset_to_serializable_dict(validation),
                "interpolation_test": dataset_to_serializable_dict(interpolation_test),
                "extrapolation_test": dataset_to_serializable_dict(extrapolation_test),
                "current_state_all_test": dataset_to_serializable_dict(current_state_all_test),
                "exact_state_all_test": dataset_to_serializable_dict(exact_state_all_test),
            },
            output_dir / "generated_datasets.pt",
        )

    reports: list[dict[str, Any]] = []
    for activation_name in config.activation_names:
        reports.append(
            run_experiment(
                experiment_name=f"activation_{activation_name}",
                activation_name=activation_name,
                training_cpu=training,
                validation_cpu=validation,
                evaluation_datasets=evaluation_datasets,
                output_dir=output_dir,
                config=config,
                physical=physical,
                gd_step_size=gd_step_size,
                shared_baselines=shared_baselines,
            )
        )

    best_report = min(reports, key=activation_report_selection_key)
    best_activation = str(best_report["activation_name"])
    hard_case = select_hard_extrapolation_case(
        extrapolation_test,
        problems,
        physical,
        learned_current_state_metrics=best_report["evaluation"][
            "current_state_all_test"
        ]["learned"],
        residual_threshold=HARD_CASE_FINAL_RESIDUAL_THRESHOLD,
    )
    hard_case["selected_activation"] = best_activation
    save_json(hard_case, output_dir / "hard_case_selection.json")

    if not config.skip_plots:
        plot_activation_training_comparison(
            reports, output_dir / "activation_ablation_training_validation.png"
        )
        plot_activation_final_metrics(
            reports, output_dir / "activation_ablation_final_metrics.png"
        )
        for split_name in ("interpolation_test", "extrapolation_test"):
            plot_activation_solver_iteration_comparison(
                reports,
                split_name=split_name,
                save_path=output_dir / f"activation_ablation_{split_name}.png",
            )
        plot_hard_case_selection(
            hard_case, output_dir / "hard_case_selection_diagnostics.png"
        )

    activation_ranking = [
        {
            "rank": rank,
            "activation_name": report["activation_name"],
            "best_validation_epoch": report["best_validation_epoch"],
            "best_validation_selection_key": report["best_validation_selection_key"],
            "interpolation_final_residual_p95": report["evaluation"]["interpolation_test"]["learned"]["final_residual_p95"],
            "interpolation_final_residual_max": report["evaluation"]["interpolation_test"]["learned"]["final_residual_max"],
            "extrapolation_final_residual_p95": report["evaluation"]["extrapolation_test"]["learned"]["final_residual_p95"],
            "extrapolation_final_residual_max": report["evaluation"]["extrapolation_test"]["learned"]["final_residual_max"],
            "extrapolation_nonfinite_count": report["evaluation"]["extrapolation_test"]["learned"]["final_nonfinite_state_count"],
        }
        for rank, report in enumerate(
            sorted(reports, key=activation_report_selection_key), start=1
        )
    ]

    summary = {
        "experiment_type": "fixed_left_edge_15x15_cloth_activation_function_ablation",
        "controlled_variable": "hidden-layer activation function",
        "control_policy": (
            "All variants share identical data, seeds, architecture dimensions, "
            "bias-free layers, orthogonal first-layer initialization, zero output-layer "
            "initialization, optimizer, training budget, and evaluation procedure."
        ),
        "runtime_config": asdict(config),
        "physical_config": asdict(physical),
        "problem_split": asdict(split),
        "physics_checks": physics_checks,
        "gradient_descent_selection": gd_selection,
        "shared_baselines": shared_baselines,
        "activation_ranking": activation_ranking,
        "best_activation": best_activation,
        "hard_case_selection": hard_case,
        "experiments": reports,
    }
    save_json(summary, output_dir / "activation_ablation_summary.json")
    save_json(summary, output_dir / "all_experiments_summary.json")

    print("\nCompleted activation-function ablation.")
    print(f"Best activation by validation selection: {best_activation}")
    print(f"Summary: {output_dir / 'activation_ablation_summary.json'}")
    print(
        "Best checkpoint: "
        f"{output_dir / f'activation_{best_activation}' / 'best_validation_model_state_dict.pt'}"
    )
    print(
        "Selected rollout case: problem "
        f"{hard_case['selected']['problem_index']} with initial residual p95="
        f"{hard_case['selected']['initial_residual_p95']:.3e} and MLP final residual="
        f"{hard_case['selected']['mlp_current_state_final_residual']:.3e}"
    )


if __name__ == "__main__":
    main()
