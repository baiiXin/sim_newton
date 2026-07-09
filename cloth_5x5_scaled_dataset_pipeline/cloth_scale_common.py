"""Shared physics, data, model, and evaluation utilities for scaled 5x5 cloth experiments.

Design principles
-----------------
* The network state always contains all 25 vertices (75 coordinates).
* Fixed vertices are represented by a 25D boolean mask and a 75D target.
* Fixed coordinates are projected after every update; masked residuals ignore reaction forces.
* Dataset files store problem-level tensors once and sample-level initial states separately.
* Reference trajectories are cached per (boundary, motion), so multiple datasets reuse them.
"""

from __future__ import annotations

import hashlib
import concurrent.futures
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

TORCH_DTYPE = torch.float64

torch.set_default_dtype(TORCH_DTYPE)

GRID_ROWS = 5
GRID_COLS = 5
SPATIAL_DIM = 3
NUM_PARTICLES = GRID_ROWS * GRID_COLS
STATE_DIM = NUM_PARTICLES * SPATIAL_DIM
HISTORY_CHANNELS = 3
PER_VERTEX_INPUT_DIM = HISTORY_CHANNELS * SPATIAL_DIM
MODEL_INPUT_DIM = HISTORY_CHANNELS * STATE_DIM
MODEL_INPUT_SIGNATURE = "history_current_previous_update_no_fixed_onehot_v2"

DISTANCE_EPS = 1e-12
REFERENCE_RESIDUAL_TOL = 1e-11
REFERENCE_ACCEPTABLE_RESIDUAL = 1e-8
REFERENCE_MAX_ITERATIONS = 100
REFERENCE_LINE_SEARCH_MIN_ALPHA = 2.0 ** -30
PLOT_FLOOR = 1e-16

DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_SAMPLING_RADIUS_MIN = 1e-2
DEFAULT_SAMPLING_RADIUS_MAX = 1e-1

MODEL_RANDOM_SEED = 42
MOTION_SOBOL_SEED_TRAIN = 20260630
MOTION_SOBOL_SEED_VALIDATION = 20260701
MOTION_SOBOL_SEED_ID_TEST = 20260702
TRAIN_SOBOL_SEED = 20260620
VALIDATION_SOBOL_SEED = 20260621
TEST_SOBOL_SEED = 20260622

TRAIN_TIME_INDICES = (0, 5, 11, 16, 21, 26, 32, 37, 42, 47, 53, 58, 63, 68, 74, 79)
VALIDATION_TIME_INDICES = (4, 14, 24, 34, 44, 54, 64, 74, 84, 94)
SEEN_INTERPOLATION_TIME_INDICES = (2, 8, 13, 18, 24, 29, 34, 39, 45, 50, 55, 61, 66, 71, 76, 78)
UNSEEN_TEST_TIME_INDICES = tuple(range(0, 100, 5))


def grid_index(row: int, col: int) -> int:
    return row * GRID_COLS + col


def build_triangular_cloth_topology() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    edge_set: set[tuple[int, int]] = set()
    faces: list[tuple[int, int, int]] = []

    def add_edge(a: int, b: int) -> None:
        if a == b:
            raise ValueError("Degenerate edge")
        edge_set.add((min(a, b), max(a, b)))

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS - 1):
            add_edge(grid_index(row, col), grid_index(row, col + 1))
    for row in range(GRID_ROWS - 1):
        for col in range(GRID_COLS):
            add_edge(grid_index(row, col), grid_index(row + 1, col))
    for row in range(GRID_ROWS - 1):
        for col in range(GRID_COLS - 1):
            tl = grid_index(row, col)
            tr = grid_index(row, col + 1)
            bl = grid_index(row + 1, col)
            br = grid_index(row + 1, col + 1)
            if (row + col) % 2 == 0:
                add_edge(tl, br)
                faces.extend(((tl, tr, br), (tl, br, bl)))
            else:
                add_edge(bl, tr)
                faces.extend(((tl, tr, bl), (tr, br, bl)))

    edges = tuple(sorted(edge_set))
    faces_tuple = tuple(faces)
    expected_edges = GRID_ROWS * (GRID_COLS - 1) + (GRID_ROWS - 1) * GRID_COLS + (GRID_ROWS - 1) * (GRID_COLS - 1)
    expected_faces = 2 * (GRID_ROWS - 1) * (GRID_COLS - 1)
    if len(edges) != expected_edges or len(faces_tuple) != expected_faces:
        raise RuntimeError("Unexpected topology size")
    return edges, faces_tuple


SPRING_EDGES, TRIANGLE_FACES = build_triangular_cloth_topology()
NUM_SPRINGS = len(SPRING_EDGES)
NUM_TRIANGLES = len(TRIANGLE_FACES)


@dataclass(frozen=True)
class PhysicalConfig:
    masses: tuple[float, ...]
    gravity: float
    dt: float
    spring_stiffness: tuple[float, ...]
    rest_lengths: tuple[float, ...]
    rest_positions: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class BoundarySpec:
    index: int
    name: str
    fixed_indices: tuple[int, ...]
    category: str
    split: str
    target_offsets: tuple[tuple[float, float, float], ...] = ()

    @property
    def fixed_count(self) -> int:
        return len(self.fixed_indices)


@dataclass(frozen=True)
class MotionSpec:
    index: int
    name: str
    split: str
    category: str
    source: str
    positions: tuple[tuple[float, float, float], ...]
    velocities: tuple[tuple[float, float, float], ...]
    parameters: tuple[tuple[str, float], ...]
    ood_factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    boundary_indices: tuple[int, ...]
    motion_indices: tuple[int, ...]
    time_indices: tuple[int, ...]
    points_per_problem: int
    include_current_and_exact: bool = True


@dataclass
class ProblemTable:
    q: torch.Tensor
    masses: torch.Tensor
    exact_y: torch.Tensor
    current_y: torch.Tensor
    fixed_mask: torch.Tensor
    fixed_target: torch.Tensor
    sampling_radius: torch.Tensor
    boundary_index: torch.Tensor
    motion_index: torch.Tensor
    time_index: torch.Tensor
    exact_energy: torch.Tensor
    exact_residual: torch.Tensor

    def __len__(self) -> int:
        return int(self.q.shape[0])

    def to(self, device: torch.device | str) -> "ProblemTable":
        return ProblemTable(
            q=self.q.to(device=device, dtype=TORCH_DTYPE),
            masses=self.masses.to(device=device, dtype=TORCH_DTYPE),
            exact_y=self.exact_y.to(device=device, dtype=TORCH_DTYPE),
            current_y=self.current_y.to(device=device, dtype=TORCH_DTYPE),
            fixed_mask=self.fixed_mask.to(device=device, dtype=torch.bool),
            fixed_target=self.fixed_target.to(device=device, dtype=TORCH_DTYPE),
            sampling_radius=self.sampling_radius.to(device=device, dtype=TORCH_DTYPE),
            boundary_index=self.boundary_index.to(device=device, dtype=torch.long),
            motion_index=self.motion_index.to(device=device, dtype=torch.long),
            time_index=self.time_index.to(device=device, dtype=torch.long),
            exact_energy=self.exact_energy.to(device=device, dtype=TORCH_DTYPE),
            exact_residual=self.exact_residual.to(device=device, dtype=TORCH_DTYPE),
        )

    def serializable(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_serializable(cls, data: Mapping[str, torch.Tensor]) -> "ProblemTable":
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})


@dataclass
class SampleSplit:
    initial_y: torch.Tensor
    problem_index: torch.Tensor
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return int(self.initial_y.shape[0])

    def serializable(self) -> dict[str, Any]:
        return {
            "initial_y": self.initial_y,
            "problem_index": self.problem_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_serializable(cls, data: Mapping[str, Any]) -> "SampleSplit":
        return cls(
            initial_y=data["initial_y"],
            problem_index=data["problem_index"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ResolvedBatch:
    initial_y: torch.Tensor
    q: torch.Tensor
    masses: torch.Tensor
    exact_y: torch.Tensor
    fixed_mask: torch.Tensor
    fixed_target: torch.Tensor
    boundary_index: torch.Tensor
    motion_index: torch.Tensor
    time_index: torch.Tensor


@dataclass(frozen=True)
class ModelSpec:
    activation: str
    depth: int
    width: int
    use_bias: bool

    @property
    def experiment_name(self) -> str:
        bias = "bias" if self.use_bias else "no_bias"
        return f"activation_{self.activation}_depth_{self.depth:02d}_width_{self.width:04d}_{bias}"


@dataclass(frozen=True)
class LearnedOptimizerState:
    previous_residual: torch.Tensor
    previous_update: torch.Tensor

    @classmethod
    def zeros_like(cls, y: torch.Tensor) -> "LearnedOptimizerState":
        z = torch.zeros_like(y)
        return cls(z, z.clone())


def default_physical_config() -> PhysicalConfig:
    spacing = 0.5
    height = 1.2
    rest = tuple(
        (col * spacing, -row * spacing, height)
        for row in range(GRID_ROWS)
        for col in range(GRID_COLS)
    )
    lengths = tuple(math.dist(rest[i], rest[j]) for i, j in SPRING_EDGES)
    return PhysicalConfig(
        masses=tuple(1.0 for _ in range(NUM_PARTICLES)),
        gravity=9.8,
        dt=0.01,
        spring_stiffness=tuple(2500.0 for _ in range(NUM_SPRINGS)),
        rest_lengths=lengths,
        rest_positions=rest,
    )


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(data: Any, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()[:length]


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


def save_json(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(dict(data)), f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    index = 0 if device.index is None else int(device.index)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested cuda:{index}, but only {torch.cuda.device_count()} CUDA devices are visible")


def rest_positions_tensor(physical: PhysicalConfig, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(physical.rest_positions, dtype=TORCH_DTYPE, device=device)


def boundary_mask(boundary: BoundarySpec, device: torch.device | str = "cpu") -> torch.Tensor:
    mask = torch.zeros(NUM_PARTICLES, dtype=torch.bool, device=device)
    if boundary.fixed_indices:
        mask[list(boundary.fixed_indices)] = True
    return mask


def boundary_target(boundary: BoundarySpec, physical: PhysicalConfig, device: torch.device | str = "cpu") -> torch.Tensor:
    target = rest_positions_tensor(physical, device=device).clone()
    if boundary.target_offsets:
        if len(boundary.target_offsets) != len(boundary.fixed_indices):
            raise ValueError(f"Boundary {boundary.name}: target_offsets length mismatch")
        for idx, offset in zip(boundary.fixed_indices, boundary.target_offsets):
            target[idx] += torch.tensor(offset, dtype=TORCH_DTYPE, device=device)
    return target.reshape(STATE_DIM)


def coordinate_free_mask(fixed_mask: torch.Tensor) -> torch.Tensor:
    return (~fixed_mask).repeat_interleave(SPATIAL_DIM, dim=-1)


def reshape_state(y: torch.Tensor) -> torch.Tensor:
    if y.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected final dimension {STATE_DIM}, got {tuple(y.shape)}")
    return y.reshape(*y.shape[:-1], NUM_PARTICLES, SPATIAL_DIM)


def project_fixed(y: torch.Tensor, fixed_mask: torch.Tensor, fixed_target: torch.Tensor) -> torch.Tensor:
    coord_mask = fixed_mask.repeat_interleave(SPATIAL_DIM, dim=-1)
    return torch.where(coord_mask, fixed_target, y)


def spring_vectors(y: torch.Tensor) -> torch.Tensor:
    p = reshape_state(y)
    edges = torch.as_tensor(SPRING_EDGES, dtype=torch.long, device=y.device)
    return p[..., edges[:, 1], :] - p[..., edges[:, 0], :]


def spring_lengths(y: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(spring_vectors(y), dim=-1)


def variational_energy(y: torch.Tensor, q: torch.Tensor, masses: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    p = reshape_state(y)
    q_p = reshape_state(q)
    inertial = (masses / (2.0 * physical.dt**2)) * torch.sum((p - q_p) ** 2, dim=-1)
    lengths = spring_lengths(y)
    stiffness = torch.as_tensor(physical.spring_stiffness, dtype=y.dtype, device=y.device)
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    spring = 0.5 * stiffness * (lengths - rest) ** 2
    return inertial.sum(dim=-1) + spring.sum(dim=-1)


def stationarity_gradient(y: torch.Tensor, q: torch.Tensor, masses: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    p = reshape_state(y)
    q_p = reshape_state(q)
    grad = (masses[..., :, None] / physical.dt**2) * (p - q_p)
    edges = torch.as_tensor(SPRING_EDGES, dtype=torch.long, device=y.device)
    edge_vectors = p[..., edges[:, 1], :] - p[..., edges[:, 0], :]
    lengths = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True).clamp_min(DISTANCE_EPS)
    stiffness = torch.as_tensor(physical.spring_stiffness, dtype=y.dtype, device=y.device)
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    shape = [1] * (edge_vectors.ndim - 2) + [NUM_SPRINGS, 1]
    edge_grad = stiffness.reshape(shape) * (1.0 - rest.reshape(shape) / lengths) * edge_vectors
    grad = grad.clone()
    grad.index_add_(-2, edges[:, 0], -edge_grad)
    grad.index_add_(-2, edges[:, 1], edge_grad)
    return grad.reshape(*y.shape[:-1], STATE_DIM)


def masked_stationarity_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    fixed_mask: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    grad = stationarity_gradient(y, q, masses, physical)
    return grad * coordinate_free_mask(fixed_mask).to(dtype=grad.dtype)


def stationarity_residual_norm(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    fixed_mask: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    return torch.linalg.vector_norm(masked_stationarity_residual(y, q, masses, fixed_mask, physical), dim=-1)


def variational_hessian(y: torch.Tensor, masses: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    p = reshape_state(y)
    edges = torch.as_tensor(SPRING_EDGES, dtype=torch.long, device=y.device)
    edge_vectors = p[..., edges[:, 1], :] - p[..., edges[:, 0], :]
    lengths = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True).clamp_min(DISTANCE_EPS)
    identity = torch.eye(SPATIAL_DIM, dtype=y.dtype, device=y.device)
    outer = edge_vectors.unsqueeze(-1) * edge_vectors.unsqueeze(-2)
    stiffness = torch.as_tensor(physical.spring_stiffness, dtype=y.dtype, device=y.device)
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    shape = [1] * (edge_vectors.ndim - 2) + [NUM_SPRINGS, 1, 1]
    lengths_matrix = lengths.unsqueeze(-1)
    blocks = stiffness.reshape(shape) * (
        (1.0 - rest.reshape(shape) / lengths_matrix) * identity
        + rest.reshape(shape) / lengths_matrix.pow(3) * outer
    )
    h = torch.zeros((*y.shape[:-1], STATE_DIM, STATE_DIM), dtype=y.dtype, device=y.device)
    for vertex in range(NUM_PARTICLES):
        s = slice(vertex * 3, vertex * 3 + 3)
        h[..., s, s] += (masses[..., vertex] / physical.dt**2)[..., None, None] * identity
    for edge_idx, (a, b) in enumerate(SPRING_EDGES):
        sa = slice(a * 3, a * 3 + 3)
        sb = slice(b * 3, b * 3 + 3)
        block = blocks[..., edge_idx, :, :]
        h[..., sa, sa] += block
        h[..., sb, sb] += block
        h[..., sa, sb] -= block
        h[..., sb, sa] -= block
    return h


def apply_newton_update_same_mask(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    fixed_mask: torch.Tensor,
    fixed_target: torch.Tensor,
    physical: PhysicalConfig,
    residual_tolerance: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    if fixed_mask.ndim == 1:
        mask_1d = fixed_mask
    else:
        if not bool(torch.all(fixed_mask == fixed_mask[0])):
            raise ValueError("apply_newton_update_same_mask requires one shared mask per batch")
        mask_1d = fixed_mask[0]
    free = coordinate_free_mask(mask_1d)
    grad = stationarity_gradient(y, q, masses, physical)
    masked_grad = grad * free.to(dtype=grad.dtype)
    residual = torch.linalg.vector_norm(masked_grad, dim=-1)
    delta = torch.zeros_like(y)
    if bool(torch.any(residual > residual_tolerance)) and bool(torch.any(free)):
        h = variational_hessian(y, masses, physical)
        h_free = h[..., free, :][..., :, free]
        rhs = -grad[..., free]
        solution, info = torch.linalg.solve_ex(h_free, rhs.unsqueeze(-1))
        solution = solution.squeeze(-1)
        failed = info != 0
        if bool(torch.any(failed)):
            solution[failed] = torch.matmul(
                torch.linalg.pinv(h_free[failed]), rhs[failed].unsqueeze(-1)
            ).squeeze(-1)
        active = residual > residual_tolerance
        solution = torch.where(active[..., None], solution, torch.zeros_like(solution))
        delta[..., free] = solution
    next_y = project_fixed(y + delta, fixed_mask, fixed_target)
    return next_y, delta


def mass_preconditioned_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    fixed_mask: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    residual = masked_stationarity_residual(y, q, masses, fixed_mask, physical)
    mass_coord = masses.repeat_interleave(SPATIAL_DIM, dim=-1)
    return physical.dt**2 * residual / mass_coord


def solve_reference_solution(
    q: torch.Tensor,
    masses: torch.Tensor,
    initial_y: torch.Tensor,
    fixed_mask: torch.Tensor,
    fixed_target: torch.Tensor,
    physical: PhysicalConfig,
    residual_tolerance: float = REFERENCE_RESIDUAL_TOL,
    max_iterations: int = REFERENCE_MAX_ITERATIONS,
    raise_on_nonconvergence: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    y = project_fixed(initial_y.reshape(1, STATE_DIM).clone(), fixed_mask.reshape(1, NUM_PARTICLES), fixed_target.reshape(1, STATE_DIM))
    q_b = q.reshape(1, STATE_DIM)
    masses_b = masses.reshape(1, NUM_PARTICLES)
    mask_b = fixed_mask.reshape(1, NUM_PARTICLES)
    target_b = fixed_target.reshape(1, STATE_DIM)
    free = coordinate_free_mask(fixed_mask)

    best_y = y.clone()
    best_residual = float("inf")
    best_energy = float("inf")
    best_iteration = 0
    reductions = 0
    status = "max_iterations"

    for iteration in range(max_iterations + 1):
        gradient = stationarity_gradient(y, q_b, masses_b, physical)
        masked_gradient = gradient * free.to(dtype=gradient.dtype)
        residual = float(torch.linalg.vector_norm(masked_gradient).item())
        energy = float(variational_energy(y, q_b, masses_b, physical).item())
        if math.isfinite(residual) and math.isfinite(energy) and residual < best_residual:
            best_y = y.clone()
            best_residual = residual
            best_energy = energy
            best_iteration = iteration
        if residual <= residual_tolerance:
            return y.squeeze(0), {
                "iterations": iteration,
                "residual_norm": residual,
                "energy": energy,
                "line_search_reductions": reductions,
                "converged": True,
                "acceptable": True,
                "status": "converged",
            }
        if iteration == max_iterations:
            break
        if not bool(torch.any(free)):
            status = "no_free_coordinates"
            break

        hessian = variational_hessian(y, masses_b, physical)
        h_free = hessian[..., free, :][..., :, free]
        rhs = -gradient[..., free]
        direction_free, info = torch.linalg.solve_ex(h_free, rhs.unsqueeze(-1))
        direction_free = direction_free.squeeze(-1)
        if bool(torch.any(info != 0)) or not bool(torch.isfinite(direction_free).all()):
            direction_free = torch.matmul(torch.linalg.pinv(h_free), rhs.unsqueeze(-1)).squeeze(-1)
        direction = torch.zeros_like(y)
        direction[..., free] = direction_free
        directional_derivative = float(torch.sum(masked_gradient * direction).item())
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            mass_coord = masses_b.repeat_interleave(3, dim=-1)
            direction = -physical.dt**2 * masked_gradient / mass_coord
            directional_derivative = float(torch.sum(masked_gradient * direction).item())
        if not bool(torch.isfinite(direction).all()):
            status = "nonfinite_direction"
            break
        if float(torch.linalg.vector_norm(direction).item()) <= 1e-13:
            status = "tiny_direction"
            break

        alpha = 1.0
        accepted = False
        while alpha >= REFERENCE_LINE_SEARCH_MIN_ALPHA:
            candidate = project_fixed(y + alpha * direction, mask_b, target_b)
            if bool(torch.isfinite(candidate).all()) and bool(torch.all(spring_lengths(candidate) > DISTANCE_EPS)):
                candidate_energy = float(variational_energy(candidate, q_b, masses_b, physical).item())
                if math.isfinite(candidate_energy) and candidate_energy <= energy + 1e-4 * alpha * directional_derivative:
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
        raise RuntimeError("Reference solver did not produce a finite iterate")
    if raise_on_nonconvergence and not acceptable:
        raise RuntimeError(f"Reference solve failed: status={status}, residual={best_residual:.6e}")
    return best_y.squeeze(0), {
        "iterations": best_iteration,
        "residual_norm": best_residual,
        "energy": best_energy,
        "line_search_reductions": reductions,
        "converged": False,
        "acceptable": acceptable,
        "status": status,
    }


def make_q(p: torch.Tensor, v: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    gravity = torch.tensor([0.0, 0.0, physical.gravity], dtype=p.dtype, device=p.device)
    return (p + physical.dt * v - physical.dt**2 * gravity).reshape(STATE_DIM)


def advance_physical_state(
    p: torch.Tensor,
    exact_y: torch.Tensor,
    fixed_mask: torch.Tensor,
    fixed_target: torch.Tensor,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_p = reshape_state(project_fixed(exact_y, fixed_mask, fixed_target)).reshape(NUM_PARTICLES, 3)
    next_v = (next_p - p) / physical.dt
    next_p = next_p.clone()
    next_v = next_v.clone()
    next_p[fixed_mask] = reshape_state(fixed_target)[fixed_mask]
    next_v[fixed_mask] = 0.0
    return next_p, next_v


def make_motion_spec(
    *,
    index: int,
    name: str,
    split: str,
    category: str,
    source: str,
    physical: PhysicalConfig,
    stretch_x: float = 1.0,
    shear: float = 0.0,
    bend: float = 0.0,
    twist: float = 0.0,
    translation_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    angular_velocity: float = 0.0,
    velocity_gradient: float = 0.0,
    ood_factors: Sequence[str] = (),
) -> MotionSpec:
    if stretch_x <= 0:
        raise ValueError("stretch_x must be positive")
    base = np.asarray(physical.rest_positions, dtype=float)
    positions = base.copy()
    velocities = np.zeros_like(base)
    span_x = max((GRID_COLS - 1) * 0.5, 1e-12)
    height = float(base[0, 2])
    transl = np.asarray(translation_velocity, dtype=float)
    center = np.mean(base, axis=0)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            idx = grid_index(row, col)
            u = col / max(GRID_COLS - 1, 1)
            vv = row / max(GRID_ROWS - 1, 1)
            x0, y0, _ = base[idx]
            x = stretch_x * x0
            y = y0 + shear * x0 * (2.0 * vv - 1.0)
            z = height + bend * math.sin(math.pi * u) * math.sin(math.pi * vv) + twist * (x0 / span_x) * (2.0 * vv - 1.0)
            positions[idx] = (x, y, z)
            radial = positions[idx] - center
            rotational = float(angular_velocity) * np.cross(np.asarray([0.0, 1.0, 0.0]), radial)
            gradient = np.asarray([
                velocity_gradient * (2.0 * vv - 1.0) * u,
                0.35 * velocity_gradient * math.sin(math.pi * u),
                -0.5 * velocity_gradient * (2.0 * vv - 1.0) * u,
            ])
            velocities[idx] = transl + rotational + gradient

    params = {
        "stretch_x": stretch_x,
        "shear": shear,
        "bend": bend,
        "twist": twist,
        "vx": float(transl[0]),
        "vy": float(transl[1]),
        "vz": float(transl[2]),
        "angular_velocity": angular_velocity,
        "velocity_gradient": velocity_gradient,
    }
    return MotionSpec(
        index=index,
        name=name,
        split=split,
        category=category,
        source=source,
        positions=tuple(tuple(float(x) for x in row) for row in positions),
        velocities=tuple(tuple(float(x) for x in row) for row in velocities),
        parameters=tuple((k, float(v)) for k, v in params.items()),
        ood_factors=tuple(str(x) for x in ood_factors),
    )


def generate_sobol_motion_specs(
    *,
    count: int,
    start_index: int,
    seed: int,
    split: str,
    physical: PhysicalConfig,
    name_prefix: str,
) -> list[MotionSpec]:
    engine = torch.quasirandom.SobolEngine(dimension=9, scramble=True, seed=seed)
    unit = engine.draw(count).to(dtype=TORCH_DTYPE).cpu().numpy()
    result: list[MotionSpec] = []
    for offset, u in enumerate(unit):
        result.append(make_motion_spec(
            index=start_index + offset,
            name=f"{name_prefix}_{offset:03d}",
            split=split,
            category="in_domain_sobol",
            source=f"scrambled_sobol_seed_{seed}",
            physical=physical,
            stretch_x=0.80 + 0.50 * u[0],
            shear=-0.25 + 0.50 * u[1],
            bend=-0.25 + 0.50 * u[2],
            twist=-0.22 + 0.44 * u[3],
            translation_velocity=(-2.0 + 4.0 * u[4], -1.5 + 3.0 * u[5], -1.5 + 4.0 * u[6]),
            angular_velocity=-1.5 + 3.0 * u[7],
            velocity_gradient=-1.2 + 2.4 * u[8],
        ))
    return result


def build_motion_catalogue(physical: PhysicalConfig) -> dict[int, MotionSpec]:
    motions: list[MotionSpec] = [make_motion_spec(
        index=0,
        name="original_horizontal_static",
        split="train",
        category="original",
        source="base_physical_config",
        physical=physical,
    )]
    anchors = [
        ("high_horizontal_velocity", "velocity", 1.00, 0.00, 0.00, 0.00, (2.8, 0.0, 0.0), 0.0, 0.0),
        ("upward_throw", "velocity", 1.00, 0.00, 0.00, 0.00, (0.0, 0.0, 2.8), 0.0, 0.0),
        ("downward_side_flight", "velocity", 1.00, 0.00, 0.00, 0.00, (1.6, 0.8, -1.6), 0.0, 0.0),
        ("horizontal_stretch", "deformation", 1.35, 0.00, 0.00, 0.00, (0.0, 0.0, 0.0), 0.0, 0.0),
        ("horizontal_compression", "deformation", 0.75, 0.00, 0.00, 0.00, (0.0, 0.0, 0.0), 0.0, 0.0),
        ("out_of_plane_bend", "deformation", 1.00, 0.00, 0.38, 0.00, (0.3, 0.0, 0.2), 0.0, 0.0),
        ("twist_shear_rotation", "combined", 1.05, 0.24, 0.12, 0.26, (0.4, -0.2, 0.3), 1.6, 1.0),
    ]
    for index, item in enumerate(anchors, start=1):
        name, category, stretch, shear, bend, twist, velocity, angular, gradient = item
        motions.append(make_motion_spec(
            index=index,
            name=name,
            split="train",
            category=category,
            source="hand_designed_anchor",
            physical=physical,
            stretch_x=stretch,
            shear=shear,
            bend=bend,
            twist=twist,
            translation_velocity=velocity,
            angular_velocity=angular,
            velocity_gradient=gradient,
        ))

    # 8 anchors + 56 nested Sobol motions = 64 training motions.
    motions.extend(generate_sobol_motion_specs(
        count=56,
        start_index=8,
        seed=MOTION_SOBOL_SEED_TRAIN,
        split="train",
        physical=physical,
        name_prefix="train_sobol",
    ))
    motions.extend(generate_sobol_motion_specs(
        count=16,
        start_index=1000,
        seed=MOTION_SOBOL_SEED_VALIDATION,
        split="validation",
        physical=physical,
        name_prefix="validation_sobol",
    ))
    motions.extend(generate_sobol_motion_specs(
        count=32,
        start_index=2000,
        seed=MOTION_SOBOL_SEED_ID_TEST,
        split="id_test",
        physical=physical,
        name_prefix="id_test_sobol",
    ))
    ood_specs = [
        ("ood_fast_horizontal", "ood_velocity", 1.0, 0.0, 0.0, 0.0, (5.2, 0.0, 0.0), 0.0, 0.0, ("horizontal_speed",)),
        ("ood_fast_upward", "ood_velocity", 1.0, 0.0, 0.0, 0.0, (0.3, 0.0, 5.0), 0.0, 0.0, ("upward_speed",)),
        ("ood_strong_stretch", "ood_deformation", 1.65, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0, ("stretch",)),
        ("ood_strong_compression", "ood_deformation", 0.58, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0, ("compression",)),
        ("ood_strong_bend", "ood_deformation", 1.0, 0.0, 0.72, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0, ("bend",)),
        ("ood_strong_twist", "ood_deformation", 1.0, 0.35, 0.18, 0.58, (0.2, 0.0, 0.2), 2.8, 1.5, ("twist", "rotation")),
        ("ood_stretch_fast_side", "ood_combination", 1.55, 0.22, 0.25, 0.25, (4.2, 1.5, 0.8), 2.0, 1.8, ("stretch", "speed", "shear")),
        ("ood_compress_twist_up", "ood_combination", 0.62, -0.38, -0.35, -0.52, (1.0, -0.8, 4.2), -2.6, -2.0, ("compression", "twist", "upward_speed")),
    ]
    for offset, item in enumerate(ood_specs):
        name, category, stretch, shear, bend, twist, velocity, angular, gradient, factors = item
        motions.append(make_motion_spec(
            index=3000 + offset,
            name=name,
            split="ood_test",
            category=category,
            source="hand_designed_ood",
            physical=physical,
            stretch_x=stretch,
            shear=shear,
            bend=bend,
            twist=twist,
            translation_velocity=velocity,
            angular_velocity=angular,
            velocity_gradient=gradient,
            ood_factors=factors,
        ))
    return {m.index: m for m in motions}


def build_boundary_catalogue() -> dict[int, BoundarySpec]:
    specs: list[BoundarySpec] = []

    def add(index: int, name: str, fixed: Sequence[int], category: str, split: str) -> None:
        fixed_tuple = tuple(sorted(set(int(x) for x in fixed)))
        if any(x < 0 or x >= NUM_PARTICLES for x in fixed_tuple):
            raise ValueError(f"Invalid fixed vertex in {name}")
        specs.append(BoundarySpec(index, name, fixed_tuple, category, split))

    # D4 training catalogue: 6 singles + 8 pairs + 6 triples + 4 complete edges = 24.
    add(0, "legacy_left_two_corners", (0, 20), "pair_same_edge_far", "train")
    add(1, "top_two_corners", (0, 4), "pair_same_edge_far", "train")
    add(2, "left_adjacent_upper", (0, 5), "pair_same_edge_adjacent", "train")
    add(3, "top_adjacent_left", (0, 1), "pair_same_edge_adjacent", "train")
    add(4, "diagonal_corners", (0, 24), "pair_diagonal", "train")
    add(5, "opposite_edge_midpoints", (2, 22), "pair_opposite_edges", "train")
    add(6, "corner_right_midpoint", (0, 14), "pair_corner_edge_mid", "train")
    add(7, "asymmetric_noncorner_pair", (5, 19), "pair_asymmetric", "train")

    add(10, "single_top_left", (0,), "single", "train")
    add(11, "single_top_right", (4,), "single", "train")
    add(12, "single_bottom_left", (20,), "single", "train")
    add(13, "single_bottom_right", (24,), "single", "train")
    add(14, "single_top_mid", (2,), "single", "train")
    add(15, "single_left_mid", (10,), "single", "train")

    add(20, "triple_top_spread", (0, 2, 4), "triple_continuous_edge", "train")
    add(21, "triple_left_spread", (0, 10, 20), "triple_continuous_edge", "train")
    add(22, "triple_three_corners_a", (0, 4, 20), "triple_distributed", "train")
    add(23, "triple_three_corners_b", (0, 4, 24), "triple_distributed", "train")
    add(24, "triple_edge_midpoints", (2, 10, 22), "triple_distributed", "train")
    add(25, "triple_asymmetric", (0, 14, 22), "triple_asymmetric", "train")

    add(30, "fixed_top_edge", tuple(range(0, 5)), "full_edge", "train")
    add(31, "fixed_bottom_edge", tuple(range(20, 25)), "full_edge", "train")
    add(32, "fixed_left_edge", (0, 5, 10, 15, 20), "full_edge", "train")
    add(33, "fixed_right_edge", (4, 9, 14, 19, 24), "full_edge", "train")

    # Validation masks are never in a training set.
    add(100, "val_pair_top_left_to_bottom_mid", (1, 22), "pair_unseen", "validation")
    add(101, "val_pair_left_mid_to_top_right", (10, 4), "pair_unseen", "validation")
    add(102, "val_single_right_mid", (14,), "single_unseen", "validation")
    add(103, "val_triple_bottom_asymmetric", (20, 22, 24), "triple_unseen", "validation")

    # Boundary-generalization test masks.
    add(200, "test_pair_top_mid_right_bottom_left", (2, 20), "pair_unseen", "test")
    add(201, "test_pair_left_lower_right_upper", (15, 9), "pair_unseen", "test")
    add(202, "test_pair_bottom_adjacent", (23, 24), "pair_unseen", "test")
    add(203, "test_pair_cross_mid", (10, 14), "pair_unseen", "test")
    add(204, "test_single_bottom_mid", (22,), "single_unseen", "test")
    add(205, "test_triple_right_edge", (4, 14, 24), "triple_unseen", "test")
    add(206, "test_triple_mixed", (1, 15, 24), "triple_unseen", "test")
    add(207, "test_full_middle_row", (10, 11, 12, 13, 14), "five_nonedge_line", "test")

    # k=4 count-OOD: quantity 4 is deliberately absent from training.
    add(300, "ood_count4_corners", (0, 4, 20, 24), "count4", "count_ood")
    add(301, "ood_count4_top_segment", (0, 1, 2, 3), "count4", "count_ood")
    add(302, "ood_count4_distributed", (2, 10, 14, 22), "count4", "count_ood")
    add(303, "ood_count4_asymmetric", (0, 9, 20, 23), "count4", "count_ood")

    # Hard OOD: no fixed point or interior fixed points.
    add(400, "hard_no_fixed", (), "no_fixed", "hard_ood")
    add(401, "hard_single_center", (12,), "interior_fixed", "hard_ood")
    add(402, "hard_two_interior", (6, 18), "interior_fixed", "hard_ood")
    add(403, "hard_interior_plus_corner", (0, 12, 18), "mixed_interior", "hard_ood")

    if len({s.index for s in specs}) != len(specs):
        raise RuntimeError("Duplicate boundary index")
    return {s.index: s for s in specs}


def build_dataset_specs() -> dict[str, DatasetSpec]:
    d2_pairs = tuple(range(0, 8))
    d4_boundaries = tuple(range(10, 16)) + tuple(range(0, 8)) + tuple(range(20, 26)) + tuple(range(30, 34))
    return {
        "D0": DatasetSpec(
            name="D0",
            description="Legacy distribution in the new 75D full-state representation.",
            boundary_indices=(0,),
            motion_indices=tuple(range(16)),
            time_indices=TRAIN_TIME_INDICES,
            points_per_problem=32,
        ),
        "D1-B": DatasetSpec(
            name="D1-B",
            description="Motion-diversity scale-up with total sample count matched to D0.",
            boundary_indices=(0,),
            motion_indices=tuple(range(64)),
            time_indices=TRAIN_TIME_INDICES,
            points_per_problem=8,
        ),
        "D1-L": DatasetSpec(
            name="D1-L",
            description="Motion-diversity scale-up while retaining D0 sampling density.",
            boundary_indices=(0,),
            motion_indices=tuple(range(64)),
            time_indices=TRAIN_TIME_INDICES,
            points_per_problem=32,
        ),
        "D2-B": DatasetSpec(
            name="D2-B",
            description="Eight two-fixed-point boundary configurations with D0-matched sample count.",
            boundary_indices=d2_pairs,
            motion_indices=tuple(range(16)),
            time_indices=TRAIN_TIME_INDICES,
            points_per_problem=4,
        ),
        "D2-L": DatasetSpec(
            name="D2-L",
            description="Eight two-fixed-point boundary configurations at D0 sampling density.",
            boundary_indices=d2_pairs,
            motion_indices=tuple(range(16)),
            time_indices=TRAIN_TIME_INDICES,
            points_per_problem=32,
        ),
        "D4-M": DatasetSpec(
            name="D4-M",
            description="Joint scale-up: 24 masks across k={1,2,3,5}, 64 motions, 16 times, 8 states/problem.",
            boundary_indices=d4_boundaries,
            motion_indices=tuple(range(64)),
            time_indices=TRAIN_TIME_INDICES,
            points_per_problem=8,
        ),
    }


def trajectory_cache_path(
    cache_root: Path,
    physical: PhysicalConfig,
    boundary: BoundarySpec,
    motion: MotionSpec,
) -> Path:
    cache_identity = {
        "cache_schema_version": 2,
        "physical": asdict(physical),
        "reference_solver": {
            "residual_tolerance": REFERENCE_RESIDUAL_TOL,
            "acceptable_residual": REFERENCE_ACCEPTABLE_RESIDUAL,
            "max_iterations": REFERENCE_MAX_ITERATIONS,
            "line_search_min_alpha": REFERENCE_LINE_SEARCH_MIN_ALPHA,
        },
    }
    physical_id = stable_hash(cache_identity, 10)
    boundary_id = stable_hash(asdict(boundary), 8)
    motion_id = stable_hash(asdict(motion), 8)
    return (
        cache_root
        / physical_id
        / f"boundary_{boundary.index:04d}_{boundary_id}"
        / f"motion_{motion.index:05d}_{motion_id}.pt"
    )


def generate_reference_trajectory(
    *,
    physical: PhysicalConfig,
    boundary: BoundarySpec,
    motion: MotionSpec,
    total_steps: int,
    device: torch.device,
    sampling_radius_min: float = DEFAULT_SAMPLING_RADIUS_MIN,
    sampling_radius_max: float = DEFAULT_SAMPLING_RADIUS_MAX,
) -> dict[str, Any]:
    fixed_mask = boundary_mask(boundary, device=device)
    fixed_target = boundary_target(boundary, physical, device=device)
    p = torch.tensor(motion.positions, dtype=TORCH_DTYPE, device=device)
    v = torch.tensor(motion.velocities, dtype=TORCH_DTYPE, device=device)
    target_points = reshape_state(fixed_target)
    p = p.clone()
    v = v.clone()
    p[fixed_mask] = target_points[fixed_mask]
    v[fixed_mask] = 0.0
    masses = torch.tensor(physical.masses, dtype=TORCH_DTYPE, device=device)

    records: dict[str, list[torch.Tensor]] = {
        "q": [],
        "masses": [],
        "exact_y": [],
        "current_y": [],
        "fixed_mask": [],
        "fixed_target": [],
        "sampling_radius": [],
        "time_index": [],
        "exact_energy": [],
        "exact_residual": [],
        "reference_iterations": [],
        "reference_acceptable": [],
    }
    warnings: list[dict[str, Any]] = []

    for time_index in range(total_steps):
        current_y = p.reshape(STATE_DIM)
        q = make_q(p, v, physical)
        exact_y, info = solve_reference_solution(
            q=q,
            masses=masses,
            initial_y=current_y,
            fixed_mask=fixed_mask,
            fixed_target=fixed_target,
            physical=physical,
            raise_on_nonconvergence=False,
        )
        free = coordinate_free_mask(fixed_mask)
        raw_radius = float(torch.max(torch.abs(current_y[free] - exact_y[free])).item()) if bool(torch.any(free)) else 0.0
        radius = min(max(raw_radius, sampling_radius_min), sampling_radius_max)
        energy = float(variational_energy(exact_y.unsqueeze(0), q.unsqueeze(0), masses.unsqueeze(0), physical).item())
        residual = float(stationarity_residual_norm(
            exact_y.unsqueeze(0), q.unsqueeze(0), masses.unsqueeze(0), fixed_mask.unsqueeze(0), physical
        ).item())
        if not info.get("acceptable", False):
            warnings.append({"time_index": time_index, **info})

        records["q"].append(q.detach().cpu())
        records["masses"].append(masses.detach().cpu())
        records["exact_y"].append(exact_y.detach().cpu())
        records["current_y"].append(current_y.detach().cpu())
        records["fixed_mask"].append(fixed_mask.detach().cpu())
        records["fixed_target"].append(fixed_target.detach().cpu())
        records["sampling_radius"].append(torch.tensor(radius, dtype=TORCH_DTYPE))
        records["time_index"].append(torch.tensor(time_index, dtype=torch.long))
        records["exact_energy"].append(torch.tensor(energy, dtype=TORCH_DTYPE))
        records["exact_residual"].append(torch.tensor(residual, dtype=TORCH_DTYPE))
        records["reference_iterations"].append(torch.tensor(int(info.get("iterations", 0)), dtype=torch.long))
        records["reference_acceptable"].append(torch.tensor(bool(info.get("acceptable", False)), dtype=torch.bool))
        p, v = advance_physical_state(p, exact_y, fixed_mask, fixed_target, physical)

    return {
        "schema_version": 1,
        "physical_config": asdict(physical),
        "boundary": asdict(boundary),
        "motion": asdict(motion),
        "total_steps": total_steps,
        "records": {key: torch.stack(values, dim=0) for key, values in records.items()},
        "warnings": warnings,
    }


def load_or_generate_trajectory(
    *,
    cache_root: Path,
    physical: PhysicalConfig,
    boundary: BoundarySpec,
    motion: MotionSpec,
    total_steps: int,
    device: torch.device,
    overwrite: bool = False,
) -> dict[str, Any]:
    path = trajectory_cache_path(cache_root, physical, boundary, motion)
    if path.exists() and not overwrite:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if int(data.get("total_steps", 0)) >= total_steps:
            return data
    path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_reference_trajectory(
        physical=physical,
        boundary=boundary,
        motion=motion,
        total_steps=total_steps,
        device=device,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    os.replace(tmp, path)
    return data


def _cache_trajectory_worker(payload: tuple[Any, ...]) -> tuple[int, int, str]:
    (cache_root_str, physical, boundary, motion, total_steps, overwrite_cache) = payload
    torch.set_num_threads(1)
    cache_root = Path(cache_root_str)
    data = load_or_generate_trajectory(
        cache_root=cache_root,
        physical=physical,
        boundary=boundary,
        motion=motion,
        total_steps=total_steps,
        device=torch.device("cpu"),
        overwrite=bool(overwrite_cache),
    )
    path = trajectory_cache_path(cache_root, physical, boundary, motion)
    return boundary.index, motion.index, str(path)


def collect_problem_table(
    *,
    spec: DatasetSpec,
    cache_root: Path,
    physical: PhysicalConfig,
    boundaries: Mapping[int, BoundarySpec],
    motions: Mapping[int, MotionSpec],
    device: torch.device,
    overwrite_cache: bool = False,
    progress_prefix: str = "",
    workers: int = 1,
) -> ProblemTable:
    needed_steps = max(spec.time_indices) + 1
    entries: dict[str, list[torch.Tensor]] = {name: [] for name in ProblemTable.__dataclass_fields__}
    total_trajectories = len(spec.boundary_indices) * len(spec.motion_indices)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers > 1:
        if device.type != "cpu":
            raise ValueError("Parallel trajectory generation currently requires --device cpu")
        payloads = [
            (str(cache_root), physical, boundaries[b], motions[m], needed_steps, overwrite_cache)
            for b in spec.boundary_indices
            for m in spec.motion_indices
        ]
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_cache_trajectory_worker, payload) for payload in payloads]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += 1
                if completed == 1 or completed % 10 == 0 or completed == total_trajectories:
                    print(f"{progress_prefix}cached trajectories {completed}/{total_trajectories}")
    done = 0
    for boundary_index in spec.boundary_indices:
        boundary = boundaries[boundary_index]
        for motion_index in spec.motion_indices:
            motion = motions[motion_index]
            trajectory = load_or_generate_trajectory(
                cache_root=cache_root,
                physical=physical,
                boundary=boundary,
                motion=motion,
                total_steps=needed_steps,
                device=device,
                overwrite=overwrite_cache,
            )
            records = trajectory["records"]
            for time_index in spec.time_indices:
                entries["q"].append(records["q"][time_index])
                entries["masses"].append(records["masses"][time_index])
                entries["exact_y"].append(records["exact_y"][time_index])
                entries["current_y"].append(records["current_y"][time_index])
                entries["fixed_mask"].append(records["fixed_mask"][time_index])
                entries["fixed_target"].append(records["fixed_target"][time_index])
                entries["sampling_radius"].append(records["sampling_radius"][time_index])
                entries["boundary_index"].append(torch.tensor(boundary_index, dtype=torch.long))
                entries["motion_index"].append(torch.tensor(motion_index, dtype=torch.long))
                entries["time_index"].append(torch.tensor(time_index, dtype=torch.long))
                entries["exact_energy"].append(records["exact_energy"][time_index])
                entries["exact_residual"].append(records["exact_residual"][time_index])
            done += 1
            if done == 1 or done % 10 == 0 or done == total_trajectories:
                print(f"{progress_prefix}reference trajectories {done}/{total_trajectories}")
    return ProblemTable(**{name: torch.stack(values, dim=0) for name, values in entries.items()})


def nondegenerate_mask(points: torch.Tensor) -> torch.Tensor:
    return torch.all(spring_lengths(points) > DISTANCE_EPS, dim=-1)


def generate_problem_points(
    *,
    count: int,
    current_y: torch.Tensor,
    exact_y: torch.Tensor,
    fixed_mask: torch.Tensor,
    fixed_target: torch.Tensor,
    radius: float,
    seed: int,
    include_current_and_exact: bool,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    points: list[torch.Tensor] = []
    explicit: list[torch.Tensor] = []
    if include_current_and_exact:
        explicit = [current_y, exact_y]
    for point in explicit[:count]:
        projected = project_fixed(point.reshape(1, STATE_DIM), fixed_mask.reshape(1, NUM_PARTICLES), fixed_target.reshape(1, STATE_DIM)).squeeze(0)
        if not bool(nondegenerate_mask(projected.reshape(1, STATE_DIM))[0]):
            raise RuntimeError("Explicit point is degenerate")
        points.append(projected.cpu())

    engine = torch.quasirandom.SobolEngine(dimension=STATE_DIM, scramble=True, seed=seed)
    accepted = len(points)
    free_coord = coordinate_free_mask(fixed_mask)
    while accepted < count:
        remaining = count - accepted
        draw_count = max(32, remaining * 2)
        unit = engine.draw(draw_count).to(dtype=TORCH_DTYPE)
        perturbation = (2.0 * unit - 1.0) * radius
        perturbation[:, ~free_coord] = 0.0
        candidates = exact_y.reshape(1, STATE_DIM) + perturbation
        candidates = project_fixed(
            candidates,
            fixed_mask.reshape(1, NUM_PARTICLES).expand(draw_count, -1),
            fixed_target.reshape(1, STATE_DIM).expand(draw_count, -1),
        )
        keep = nondegenerate_mask(candidates)
        selected = candidates[keep][:remaining]
        if selected.numel() > 0:
            points.extend([row.cpu() for row in selected])
            accepted += int(selected.shape[0])
    return torch.stack(points[:count], dim=0)


def build_sample_split(
    *,
    problems: ProblemTable,
    points_per_problem: int,
    base_seed: int,
    role: str,
    include_current_and_exact: bool,
) -> SampleSplit:
    all_points: list[torch.Tensor] = []
    all_problem_indices: list[torch.Tensor] = []
    for problem_idx in range(len(problems)):
        seed = (
            base_seed
            + 1000003 * int(problems.boundary_index[problem_idx])
            + 10007 * int(problems.motion_index[problem_idx])
            + 101 * int(problems.time_index[problem_idx])
        ) % (2**31 - 1)
        points = generate_problem_points(
            count=points_per_problem,
            current_y=problems.current_y[problem_idx],
            exact_y=problems.exact_y[problem_idx],
            fixed_mask=problems.fixed_mask[problem_idx],
            fixed_target=problems.fixed_target[problem_idx],
            radius=float(problems.sampling_radius[problem_idx].item()),
            seed=seed,
            include_current_and_exact=include_current_and_exact,
        )
        all_points.append(points)
        all_problem_indices.append(torch.full((points_per_problem,), problem_idx, dtype=torch.long))
        if problem_idx == 0 or (problem_idx + 1) % 500 == 0 or problem_idx + 1 == len(problems):
            print(f"{role}: sampled problems {problem_idx + 1}/{len(problems)}")
    return SampleSplit(
        initial_y=torch.cat(all_points, dim=0),
        problem_index=torch.cat(all_problem_indices, dim=0),
        metadata={
            "role": role,
            "num_problems": len(problems),
            "points_per_problem": points_per_problem,
            "num_samples": len(problems) * points_per_problem,
            "include_current_and_exact": include_current_and_exact,
        },
    )


def save_dataset_package(
    *,
    output_dir: Path,
    spec: DatasetSpec,
    problems: ProblemTable,
    split: SampleSplit,
    physical: PhysicalConfig,
    boundaries: Mapping[int, BoundarySpec],
    motions: Mapping[int, MotionSpec],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(problems.serializable(), output_dir / "problems.pt")
    torch.save(split.serializable(), output_dir / "train.pt")
    manifest_core = {
        "schema_version": 1,
        "dataset_spec": asdict(spec),
        "physical_config": asdict(physical),
        "boundaries": [asdict(boundaries[i]) for i in spec.boundary_indices],
        "motions": [asdict(motions[i]) for i in spec.motion_indices],
        "problem_count": len(problems),
        "sample_count": len(split),
        "state_dim": STATE_DIM,
        "model_input_dim": MODEL_INPUT_DIM,
        "storage": {
            "problem_level": "problems.pt",
            "sample_level": "train.pt",
            "repeated_problem_tensors_in_samples": False,
        },
    }
    dataset_id = f"{spec.name.lower().replace('-', '_')}_v1_{stable_hash(manifest_core, 10)}"
    manifest = {**manifest_core, "dataset_id": dataset_id}
    save_json(manifest, output_dir / "manifest.json")
    return manifest


def load_dataset_package(dataset_dir: Path) -> tuple[dict[str, Any], ProblemTable, SampleSplit]:
    manifest = load_json(dataset_dir / "manifest.json")
    problems = ProblemTable.from_serializable(torch.load(dataset_dir / "problems.pt", map_location="cpu", weights_only=False))
    split = SampleSplit.from_serializable(torch.load(dataset_dir / "train.pt", map_location="cpu", weights_only=False))
    return manifest, problems, split


def resolve_batch(
    split: SampleSplit,
    problems: ProblemTable,
    sample_indices: torch.Tensor,
    device: torch.device,
) -> ResolvedBatch:
    sample_indices_cpu = sample_indices.detach().cpu()
    pidx = split.problem_index[sample_indices_cpu]
    return ResolvedBatch(
        initial_y=split.initial_y[sample_indices_cpu].to(device=device, dtype=TORCH_DTYPE),
        q=problems.q[pidx].to(device=device, dtype=TORCH_DTYPE),
        masses=problems.masses[pidx].to(device=device, dtype=TORCH_DTYPE),
        exact_y=problems.exact_y[pidx].to(device=device, dtype=TORCH_DTYPE),
        fixed_mask=problems.fixed_mask[pidx].to(device=device, dtype=torch.bool),
        fixed_target=problems.fixed_target[pidx].to(device=device, dtype=TORCH_DTYPE),
        boundary_index=problems.boundary_index[pidx].to(device=device, dtype=torch.long),
        motion_index=problems.motion_index[pidx].to(device=device, dtype=torch.long),
        time_index=problems.time_index[pidx].to(device=device, dtype=torch.long),
    )


def make_activation(name: str) -> nn.Module:
    if name == "identity":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLPOptimizer(nn.Module):
    def __init__(self, residual_length_scale: float, model_spec: ModelSpec) -> None:
        super().__init__()
        if residual_length_scale <= 0:
            raise ValueError("residual_length_scale must be positive")
        if model_spec.depth <= 0 or model_spec.width <= 0:
            raise ValueError("depth and width must be positive")
        self.model_spec = model_spec
        self.activation = make_activation(model_spec.activation)
        layers: list[nn.Linear] = []
        input_dim = MODEL_INPUT_DIM
        for _ in range(model_spec.depth):
            layers.append(nn.Linear(input_dim, model_spec.width, bias=model_spec.use_bias))
            input_dim = model_spec.width
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(model_spec.width, STATE_DIM, bias=model_spec.use_bias)
        self.register_buffer("residual_length_scale", torch.tensor(float(residual_length_scale), dtype=TORCH_DTYPE))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def architecture_description(self) -> str:
        return (
            f"per-vertex [current residual(3), previous residual(3), previous update(3)] "
            f"flattened to {MODEL_INPUT_DIM} -> [{self.model_spec.width}, {self.model_spec.activation}] x "
            f"{self.model_spec.depth} -> {STATE_DIM}; fixed updates are gated to zero"
        )

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        fixed_mask: torch.Tensor,
        optimizer_state: LearnedOptimizerState,
        physical: PhysicalConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_residual = mass_preconditioned_residual(y, q, masses, fixed_mask, physical)
        # Interleave features vertex by vertex:
        # [r_i(3), r_{i-1}(3), delta_{i-1}(3)] for each of 25 vertices.
        # fixed_mask is deliberately NOT an input feature. It is used only to
        # mask reaction forces and gate/project fixed-coordinate updates.
        current_vertex = reshape_state(current_residual / self.residual_length_scale)
        previous_vertex = reshape_state(optimizer_state.previous_residual / self.residual_length_scale)
        update_vertex = reshape_state(optimizer_state.previous_update / self.residual_length_scale)
        h = torch.cat([current_vertex, previous_vertex, update_vertex], dim=-1).reshape(
            y.shape[0], MODEL_INPUT_DIM
        )
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        delta = self.residual_length_scale * self.output_layer(h)
        delta = delta * coordinate_free_mask(fixed_mask).to(dtype=delta.dtype)
        return delta, current_residual


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    fixed_mask: torch.Tensor,
    fixed_target: torch.Tensor,
    physical: PhysicalConfig,
    optimizer_state: LearnedOptimizerState,
) -> tuple[torch.Tensor, torch.Tensor, LearnedOptimizerState]:
    delta, current_residual = model(y, q, masses, fixed_mask, optimizer_state, physical)
    next_y = project_fixed(y + delta, fixed_mask, fixed_target)
    next_state = LearnedOptimizerState(current_residual.detach(), delta.detach())
    return next_y, delta, next_state


def physical_energy_scale(masses: torch.Tensor, physical: PhysicalConfig, residual_length_scale: float) -> float:
    return float(masses.mean().item()) * residual_length_scale**2 / physical.dt**2


def statistics(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    result: dict[str, Any] = {
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


def state_metrics(y: torch.Tensor, batch: ResolvedBatch, physical: PhysicalConfig) -> dict[str, torch.Tensor]:
    free = coordinate_free_mask(batch.fixed_mask).to(dtype=y.dtype)
    diff = (y - batch.exact_y) * free
    exact_energy = variational_energy(batch.exact_y, batch.q, batch.masses, physical)
    return {
        "residual": stationarity_residual_norm(y, batch.q, batch.masses, batch.fixed_mask, physical),
        "energy_gap": variational_energy(y, batch.q, batch.masses, physical) - exact_energy,
        "exact_error": torch.linalg.vector_norm(diff, dim=-1),
    }


def validation_selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...] | None:
    values = (
        float(metrics["final_residual_num_nonfinite"]),
        float(metrics["worst_boundary_final_residual_max"]),
        float(metrics["final_residual_max"]),
        float(metrics["final_residual_p95"]),
        float(metrics["worst_motion_final_residual_max"]),
        float(metrics["final_exact_error_max"]),
        float(metrics["final_exact_error_p95"]),
    )
    return values if all(math.isfinite(v) for v in values) else None


def _aggregate_evaluation(
    *,
    arrays: Mapping[str, np.ndarray],
    boundary_indices: np.ndarray,
    motion_indices: np.ndarray,
    time_indices: np.ndarray,
    solver: str,
    steps: int,
    elapsed: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "solver": solver,
        "steps": steps,
        "num_points": int(boundary_indices.size),
        "elapsed_seconds": elapsed,
        "seconds_per_point_per_iteration": elapsed / max(int(boundary_indices.size) * steps, 1),
    }
    for name, values in arrays.items():
        for stat_name in ("mean", "median", "p95", "max", "num_nonfinite"):
            result[f"{name}_{stat_name}_by_step"] = []
        for step in range(values.shape[1]):
            stats = statistics(values[:, step])
            for stat_name, value in stats.items():
                result[f"{name}_{stat_name}_by_step"].append(value)
        for stat_name, value in statistics(values[:, -1]).items():
            result[f"final_{name}_{stat_name}"] = value

    for group_name, indices in (("boundary", boundary_indices), ("motion", motion_indices)):
        per_group: dict[str, Any] = {}
        for group_index in sorted(np.unique(indices).tolist()):
            mask = indices == group_index
            per_group[str(int(group_index))] = {
                "num_points": int(mask.sum()),
                "time_indices": sorted(np.unique(time_indices[mask]).astype(int).tolist()),
                "final": {name: statistics(values[mask, -1]) for name, values in arrays.items()},
            }
        result[f"per_{group_name}"] = per_group
        for metric_name in arrays:
            p95_records = [
                (int(k), float(v["final"][metric_name]["p95"]))
                for k, v in per_group.items()
                if math.isfinite(float(v["final"][metric_name]["p95"]))
            ]
            max_records = [
                (int(k), float(v["final"][metric_name]["max"]))
                for k, v in per_group.items()
                if math.isfinite(float(v["final"][metric_name]["max"]))
            ]
            p95_group, p95_value = max(p95_records, key=lambda x: x[1]) if p95_records else (-1, float("nan"))
            max_group, max_value = max(max_records, key=lambda x: x[1]) if max_records else (-1, float("nan"))
            result[f"worst_{group_name}_final_{metric_name}_p95"] = p95_value
            result[f"worst_{group_name}_final_{metric_name}_p95_{group_name}_index"] = p95_group
            result[f"worst_{group_name}_final_{metric_name}_max"] = max_value
            result[f"worst_{group_name}_final_{metric_name}_max_{group_name}_index"] = max_group
    return result


@torch.no_grad()
def evaluate_learned_or_gd(
    *,
    solver: str,
    problems: ProblemTable,
    split: SampleSplit,
    physical: PhysicalConfig,
    steps: int,
    batch_size: int,
    device: torch.device,
    model: MLPOptimizer | None = None,
    gd_step_size: float | None = None,
) -> dict[str, Any]:
    if solver not in {"learned", "gradient_descent"}:
        raise ValueError(solver)
    if solver == "learned" and model is None:
        raise ValueError("model is required")
    if solver == "gradient_descent" and gd_step_size is None:
        raise ValueError("gd_step_size is required")
    if model is not None:
        model.eval()
    metric_batches: dict[str, list[torch.Tensor]] = {}
    boundary_batches: list[torch.Tensor] = []
    motion_batches: list[torch.Tensor] = []
    time_batches: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for begin in range(0, len(split), batch_size):
        idx = torch.arange(begin, min(begin + batch_size, len(split)), dtype=torch.long)
        batch = resolve_batch(split, problems, idx, device)
        y = batch.initial_y.clone()
        state = LearnedOptimizerState.zeros_like(y)
        values: dict[str, list[torch.Tensor]] = {}
        for step in range(steps + 1):
            for name, metric in state_metrics(y, batch, physical).items():
                values.setdefault(name, []).append(metric.detach().cpu())
            if step == steps:
                break
            if solver == "learned":
                assert model is not None
                y, _, state = apply_model_update(
                    model, y, batch.q, batch.masses, batch.fixed_mask, batch.fixed_target, physical, state
                )
            else:
                grad = masked_stationarity_residual(y, batch.q, batch.masses, batch.fixed_mask, physical)
                delta = -float(gd_step_size) * grad
                y = project_fixed(y + delta, batch.fixed_mask, batch.fixed_target)
        for name, seq in values.items():
            metric_batches.setdefault(name, []).append(torch.stack(seq, dim=1))
        boundary_batches.append(batch.boundary_index.cpu())
        motion_batches.append(batch.motion_index.cpu())
        time_batches.append(batch.time_index.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    arrays = {name: torch.cat(chunks, dim=0).numpy().astype(float) for name, chunks in metric_batches.items()}
    for arr in arrays.values():
        arr[~np.isfinite(arr)] = np.nan
    result = _aggregate_evaluation(
        arrays=arrays,
        boundary_indices=torch.cat(boundary_batches).numpy().astype(int),
        motion_indices=torch.cat(motion_batches).numpy().astype(int),
        time_indices=torch.cat(time_batches).numpy().astype(int),
        solver=solver,
        steps=steps,
        elapsed=elapsed,
    )
    if gd_step_size is not None:
        result["gradient_descent_step_size"] = float(gd_step_size)
    return result


def _subset_split(split: SampleSplit, indices: torch.Tensor, role: str) -> SampleSplit:
    return SampleSplit(
        initial_y=split.initial_y[indices],
        problem_index=split.problem_index[indices],
        metadata={**split.metadata, "role": role, "num_samples": int(indices.numel())},
    )


@torch.no_grad()
def evaluate_newton(
    *,
    problems: ProblemTable,
    split: SampleSplit,
    physical: PhysicalConfig,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    metric_rows: dict[str, list[torch.Tensor]] = {}
    boundary_rows: list[torch.Tensor] = []
    motion_rows: list[torch.Tensor] = []
    time_rows: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    sample_problem = split.problem_index
    sample_boundary = problems.boundary_index[sample_problem]
    for boundary_index in sorted(torch.unique(sample_boundary).tolist()):
        group_indices = torch.nonzero(sample_boundary == int(boundary_index), as_tuple=False).squeeze(-1)
        for begin in range(0, int(group_indices.numel()), batch_size):
            sample_idx = group_indices[begin: begin + batch_size]
            batch = resolve_batch(split, problems, sample_idx, device)
            y = batch.initial_y.clone()
            values: dict[str, list[torch.Tensor]] = {}
            for step in range(steps + 1):
                for name, metric in state_metrics(y, batch, physical).items():
                    values.setdefault(name, []).append(metric.detach().cpu())
                if step == steps:
                    break
                y, _ = apply_newton_update_same_mask(
                    y, batch.q, batch.masses, batch.fixed_mask, batch.fixed_target, physical
                )
            for name, seq in values.items():
                metric_rows.setdefault(name, []).append(torch.stack(seq, dim=1))
            boundary_rows.append(batch.boundary_index.cpu())
            motion_rows.append(batch.motion_index.cpu())
            time_rows.append(batch.time_index.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    arrays = {name: torch.cat(chunks, dim=0).numpy().astype(float) for name, chunks in metric_rows.items()}
    for arr in arrays.values():
        arr[~np.isfinite(arr)] = np.nan
    return _aggregate_evaluation(
        arrays=arrays,
        boundary_indices=torch.cat(boundary_rows).numpy().astype(int),
        motion_indices=torch.cat(motion_rows).numpy().astype(int),
        time_indices=torch.cat(time_rows).numpy().astype(int),
        solver="full_newton",
        steps=steps,
        elapsed=elapsed,
    )


class BatchedLBFGSState:
    def __init__(self, memory: int) -> None:
        self.memory = int(memory)
        self.s_history: list[torch.Tensor] = []
        self.y_history: list[torch.Tensor] = []
        self.rho_history: list[torch.Tensor] = []
        self.previous_x: torch.Tensor | None = None
        self.previous_g: torch.Tensor | None = None

    def update_history(self, x: torch.Tensor, g: torch.Tensor) -> None:
        if self.previous_x is not None and self.previous_g is not None:
            s = x - self.previous_x
            yy = g - self.previous_g
            sy = torch.sum(s * yy, dim=-1)
            valid = sy > 1e-18
            rho = torch.where(valid, 1.0 / sy.clamp_min(1e-18), torch.zeros_like(sy))
            s = torch.where(valid[:, None], s, torch.zeros_like(s))
            yy = torch.where(valid[:, None], yy, torch.zeros_like(yy))
            self.s_history.append(s)
            self.y_history.append(yy)
            self.rho_history.append(rho)
            if len(self.s_history) > self.memory:
                self.s_history.pop(0)
                self.y_history.pop(0)
                self.rho_history.pop(0)
        self.previous_x = x.clone()
        self.previous_g = g.clone()

    def direction(self, g: torch.Tensor) -> torch.Tensor:
        q = g.clone()
        alphas: list[torch.Tensor] = []
        for s, yy, rho in zip(reversed(self.s_history), reversed(self.y_history), reversed(self.rho_history)):
            alpha = rho * torch.sum(s * q, dim=-1)
            q = q - alpha[:, None] * yy
            alphas.append(alpha)
        if self.s_history:
            s = self.s_history[-1]
            yy = self.y_history[-1]
            sy = torch.sum(s * yy, dim=-1)
            yy_norm = torch.sum(yy * yy, dim=-1).clamp_min(1e-18)
            gamma = (sy / yy_norm).clamp(1e-8, 1e8)
            r = gamma[:, None] * q
        else:
            r = q
        for s, yy, rho, alpha in zip(self.s_history, self.y_history, self.rho_history, reversed(alphas)):
            beta = rho * torch.sum(yy * r, dim=-1)
            r = r + s * (alpha - beta)[:, None]
        return -r


@torch.no_grad()
def evaluate_lbfgs(
    *,
    problems: ProblemTable,
    split: SampleSplit,
    physical: PhysicalConfig,
    steps: int,
    batch_size: int,
    device: torch.device,
    memory: int = 10,
    max_line_search: int = 20,
) -> dict[str, Any]:
    metric_batches: dict[str, list[torch.Tensor]] = {}
    boundary_batches: list[torch.Tensor] = []
    motion_batches: list[torch.Tensor] = []
    time_batches: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for begin in range(0, len(split), batch_size):
        idx = torch.arange(begin, min(begin + batch_size, len(split)), dtype=torch.long)
        batch = resolve_batch(split, problems, idx, device)
        x = batch.initial_y.clone()
        state = BatchedLBFGSState(memory)
        values: dict[str, list[torch.Tensor]] = {}
        for step in range(steps + 1):
            for name, metric in state_metrics(x, batch, physical).items():
                values.setdefault(name, []).append(metric.detach().cpu())
            if step == steps:
                break
            g = masked_stationarity_residual(x, batch.q, batch.masses, batch.fixed_mask, physical)
            state.update_history(x, g)
            direction = state.direction(g)
            descent = torch.sum(g * direction, dim=-1)
            fallback = -g
            direction = torch.where((descent < 0)[:, None], direction, fallback)
            descent = torch.sum(g * direction, dim=-1)
            energy_before = variational_energy(x, batch.q, batch.masses, physical)
            alpha = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
            accepted = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
            candidate = x.clone()
            for _ in range(max_line_search):
                trial = project_fixed(x + alpha[:, None] * direction, batch.fixed_mask, batch.fixed_target)
                trial_energy = variational_energy(trial, batch.q, batch.masses, physical)
                good = torch.isfinite(trial_energy) & (trial_energy <= energy_before + 1e-4 * alpha * descent)
                newly = good & (~accepted)
                candidate[newly] = trial[newly]
                accepted |= good
                if bool(torch.all(accepted)):
                    break
                alpha = torch.where(accepted, alpha, alpha * 0.5)
            # Safe fallback for samples whose line search failed.
            if bool(torch.any(~accepted)):
                small = 1e-6
                fallback_trial = project_fixed(x - small * g, batch.fixed_mask, batch.fixed_target)
                candidate[~accepted] = fallback_trial[~accepted]
            x = candidate
        for name, seq in values.items():
            metric_batches.setdefault(name, []).append(torch.stack(seq, dim=1))
        boundary_batches.append(batch.boundary_index.cpu())
        motion_batches.append(batch.motion_index.cpu())
        time_batches.append(batch.time_index.cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    arrays = {name: torch.cat(chunks, dim=0).numpy().astype(float) for name, chunks in metric_batches.items()}
    for arr in arrays.values():
        arr[~np.isfinite(arr)] = np.nan
    result = _aggregate_evaluation(
        arrays=arrays,
        boundary_indices=torch.cat(boundary_batches).numpy().astype(int),
        motion_indices=torch.cat(motion_batches).numpy().astype(int),
        time_indices=torch.cat(time_batches).numpy().astype(int),
        solver="l_bfgs",
        steps=steps,
        elapsed=elapsed,
    )
    result["memory"] = int(memory)
    result["max_line_search"] = int(max_line_search)
    return result


def build_model_from_checkpoint(checkpoint: Mapping[str, Any], device: torch.device) -> MLPOptimizer:
    signature = checkpoint.get("model_input_signature")
    if signature != MODEL_INPUT_SIGNATURE:
        found = "missing (legacy 250D one-hot input)" if signature is None else repr(signature)
        raise RuntimeError(
            "Checkpoint input definition is incompatible with the current 225D no-one-hot model: "
            f"expected {MODEL_INPUT_SIGNATURE!r}, found {found}. Retrain this model from scratch."
        )
    checkpoint_input_dim = int(checkpoint.get("model_input_dim", -1))
    if checkpoint_input_dim != MODEL_INPUT_DIM:
        raise RuntimeError(
            f"Checkpoint model_input_dim={checkpoint_input_dim} does not match current MODEL_INPUT_DIM={MODEL_INPUT_DIM}."
        )
    spec = ModelSpec(**checkpoint["model_spec"])
    model = MLPOptimizer(float(checkpoint["residual_length_scale"]), spec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def save_checkpoint_atomic(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(data), tmp)
    os.replace(tmp, path)
