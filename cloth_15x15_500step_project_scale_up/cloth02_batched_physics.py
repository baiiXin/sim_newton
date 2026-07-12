"""Batched full-state cloth physics for heterogeneous scale-up scenarios.

The legacy project eliminates a fixed global set of vertices and solves a reduced
state. That representation cannot batch scenarios with different fixed masks.
This module keeps a common ``N x 3`` full state for every environment, hard
projects prescribed vertices, gates their inertia/residual entries, and supports
per-environment masses and spring stiffnesses.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Sequence

import torch

from scenario_geometry import build_initial_state, build_triangular_edges
from scenario_templates import DIRICHLET_BY_ID, ScenarioSpec

DEFAULT_ROWS = 15
DEFAULT_COLS = 15
DEFAULT_SPACING = 0.5
DEFAULT_HEIGHT = 1.2
DEFAULT_DT = 0.01
DEFAULT_GRAVITY = 9.8
DEFAULT_BASE_AREAL_DENSITY = 4.0
DEFAULT_BASE_SPRING_STIFFNESS = 2500.0
DISTANCE_EPS = 1e-12

DIRICHLET_STATIC = 0
DIRICHLET_CIRCLE_HORIZONTAL = 1
DIRICHLET_CIRCLE_VERTICAL = 2
DIRICHLET_TWIST = 3


def _dirichlet_kind_code(kind: str) -> int:
    mapping = {
        "static": DIRICHLET_STATIC,
        "circle_horizontal": DIRICHLET_CIRCLE_HORIZONTAL,
        "circle_vertical": DIRICHLET_CIRCLE_VERTICAL,
        "twist": DIRICHLET_TWIST,
    }
    try:
        return mapping[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported Dirichlet kind: {kind}") from exc


@dataclass(frozen=True)
class ClothTopology:
    rows: int
    cols: int
    spacing: float
    edges: tuple[tuple[int, int], ...]
    edge_is_diagonal: tuple[bool, ...]
    rest_lengths: tuple[float, ...]
    vertex_areas: tuple[float, ...]

    @property
    def num_vertices(self) -> int:
        return self.rows * self.cols

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def edge_tensor(self, device: torch.device | str) -> torch.Tensor:
        return torch.as_tensor(self.edges, dtype=torch.long, device=device)


def build_cloth_topology(
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    spacing: float = DEFAULT_SPACING,
) -> ClothTopology:
    if rows < 2 or cols < 2:
        raise ValueError("rows and cols must both be >= 2")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    edges = build_triangular_edges(rows, cols)
    edge_is_diagonal: list[bool] = []
    rest_lengths: list[float] = []
    for left, right in edges:
        left_row, left_col = divmod(left, cols)
        right_row, right_col = divmod(right, cols)
        dr = abs(right_row - left_row)
        dc = abs(right_col - left_col)
        if max(dr, dc) != 1 or dr + dc == 0:
            raise ValueError(f"Unexpected cloth edge {(left, right)}")
        diagonal = dr == 1 and dc == 1
        edge_is_diagonal.append(diagonal)
        rest_lengths.append(spacing * sqrt(float(dr * dr + dc * dc)))

    # Lumped continuum area: each triangle contributes one third of its area to
    # each incident vertex. Every grid cell contains two triangles.
    triangle_area = 0.5 * spacing * spacing
    vertex_areas = [0.0] * (rows * cols)
    for row in range(rows - 1):
        for col in range(cols - 1):
            tl = row * cols + col
            tr = tl + 1
            bl = (row + 1) * cols + col
            br = bl + 1
            if (row + col) % 2 == 0:
                faces = ((tl, tr, br), (tl, br, bl))
            else:
                faces = ((tl, tr, bl), (tr, br, bl))
            for face in faces:
                for vertex in face:
                    vertex_areas[vertex] += triangle_area / 3.0

    return ClothTopology(
        rows=int(rows),
        cols=int(cols),
        spacing=float(spacing),
        edges=tuple(edges),
        edge_is_diagonal=tuple(edge_is_diagonal),
        rest_lengths=tuple(rest_lengths),
        vertex_areas=tuple(vertex_areas),
    )


@dataclass
class BatchedClothParameters:
    topology: ClothTopology
    dt: float
    gravity: torch.Tensor  # [3], positive magnitude in the downward z direction
    masses: torch.Tensor  # [B, N]
    spring_stiffness: torch.Tensor  # [B, E]
    rest_lengths: torch.Tensor  # [B, E]
    fixed_mask: torch.Tensor  # [B, N], bool
    initial_positions: torch.Tensor  # [B, N, 3]
    initial_velocities: torch.Tensor  # [B, N, 3]
    dirichlet_kind: torch.Tensor  # [B], long
    dirichlet_amplitude: torch.Tensor  # [B]
    dirichlet_omega: torch.Tensor  # [B]
    twist_signs: torch.Tensor  # [B, N]
    scenario_ids: torch.Tensor  # [B], long
    boundary_ids: tuple[str, ...]
    material_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size, num_vertices = self.masses.shape
        num_edges = self.topology.num_edges
        expected = {
            "spring_stiffness": (batch_size, num_edges),
            "rest_lengths": (batch_size, num_edges),
            "fixed_mask": (batch_size, num_vertices),
            "initial_positions": (batch_size, num_vertices, 3),
            "initial_velocities": (batch_size, num_vertices, 3),
            "twist_signs": (batch_size, num_vertices),
            "dirichlet_kind": (batch_size,),
            "dirichlet_amplitude": (batch_size,),
            "dirichlet_omega": (batch_size,),
            "scenario_ids": (batch_size,),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape)
            if actual != shape:
                raise ValueError(f"{name} must have shape {shape}, got {actual}")
        if num_vertices != self.topology.num_vertices:
            raise ValueError("masses do not match topology vertex count")
        if len(self.boundary_ids) != batch_size or len(self.material_ids) != batch_size:
            raise ValueError("metadata lengths must match batch size")
        if self.fixed_mask.dtype != torch.bool:
            raise TypeError("fixed_mask must be bool")
        if not bool(self.fixed_mask.any(dim=1).all()):
            raise ValueError("every scenario must contain at least one fixed vertex")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    @property
    def batch_size(self) -> int:
        return int(self.masses.shape[0])

    @property
    def num_vertices(self) -> int:
        return int(self.masses.shape[1])

    @property
    def full_state_dim(self) -> int:
        return 3 * self.num_vertices

    @property
    def device(self) -> torch.device:
        return self.masses.device

    @property
    def dtype(self) -> torch.dtype:
        return self.masses.dtype

    def index_select(
        self,
        indices: torch.Tensor | Sequence[int],
    ) -> "BatchedClothParameters":
        index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        host_indices = index.detach().cpu().tolist()
        return BatchedClothParameters(
            topology=self.topology,
            dt=self.dt,
            gravity=self.gravity,
            masses=self.masses.index_select(0, index),
            spring_stiffness=self.spring_stiffness.index_select(0, index),
            rest_lengths=self.rest_lengths.index_select(0, index),
            fixed_mask=self.fixed_mask.index_select(0, index),
            initial_positions=self.initial_positions.index_select(0, index),
            initial_velocities=self.initial_velocities.index_select(0, index),
            dirichlet_kind=self.dirichlet_kind.index_select(0, index),
            dirichlet_amplitude=self.dirichlet_amplitude.index_select(0, index),
            dirichlet_omega=self.dirichlet_omega.index_select(0, index),
            twist_signs=self.twist_signs.index_select(0, index),
            scenario_ids=self.scenario_ids.index_select(0, index),
            boundary_ids=tuple(self.boundary_ids[i] for i in host_indices),
            material_ids=tuple(self.material_ids[i] for i in host_indices),
        )


def _compute_twist_signs(
    positions: torch.Tensor,
    fixed_mask: torch.Tensor,
    kind_code: int,
) -> torch.Tensor:
    signs = torch.zeros(
        positions.shape[0],
        dtype=positions.dtype,
        device=positions.device,
    )
    if kind_code != DIRICHLET_TWIST:
        return signs
    indices = torch.nonzero(fixed_mask, as_tuple=False).flatten()
    fixed = positions.index_select(0, indices)
    centroid = fixed.mean(dim=0, keepdim=True)
    score = fixed[:, 0] - centroid[0, 0]
    if float(score.abs().max().item()) < 1e-12:
        score = fixed[:, 1] - centroid[0, 1]
    local_signs = torch.where(
        score >= 0,
        torch.ones_like(score),
        -torch.ones_like(score),
    )
    ties = score.abs() < 1e-12
    if bool(ties.any()):
        alternating = torch.where(
            torch.arange(indices.numel(), device=positions.device) % 2 == 0,
            torch.ones_like(score),
            -torch.ones_like(score),
        )
        local_signs = torch.where(ties, alternating, local_signs)
    signs.index_copy_(0, indices, local_signs)
    return signs


def build_batched_parameters(
    scenarios: Sequence[ScenarioSpec],
    *,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    spacing: float = DEFAULT_SPACING,
    height: float = DEFAULT_HEIGHT,
    dt: float = DEFAULT_DT,
    gravity: float = DEFAULT_GRAVITY,
    base_areal_density: float = DEFAULT_BASE_AREAL_DENSITY,
    base_spring_stiffness: float = DEFAULT_BASE_SPRING_STIFFNESS,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> BatchedClothParameters:
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    topology = build_cloth_topology(rows, cols, spacing)
    device = torch.device(device)
    area = torch.as_tensor(topology.vertex_areas, dtype=dtype, device=device)
    diagonal = torch.as_tensor(
        topology.edge_is_diagonal,
        dtype=torch.bool,
        device=device,
    )
    rest = torch.as_tensor(topology.rest_lengths, dtype=dtype, device=device)

    positions: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    masses: list[torch.Tensor] = []
    stiffnesses: list[torch.Tensor] = []
    kind_codes: list[int] = []
    amplitudes: list[float] = []
    omegas: list[float] = []
    scenario_ids: list[int] = []
    boundary_ids: list[str] = []
    material_ids: list[str] = []

    for scenario in scenarios:
        state = build_initial_state(
            scenario,
            rows=rows,
            cols=cols,
            spacing=spacing,
            height=height,
        )
        position = torch.as_tensor(
            state["positions"],
            dtype=dtype,
            device=device,
        )
        velocity = torch.as_tensor(
            state["velocities"],
            dtype=dtype,
            device=device,
        )
        fixed_mask = torch.as_tensor(
            state["fixed_mask"],
            dtype=torch.bool,
            device=device,
        )
        if not bool(fixed_mask.any()):
            raise ValueError(f"Scenario {scenario.scenario_id} has no fixed vertices")
        material = state["material"]
        mass = area * float(base_areal_density) * float(material["areal_density"])
        stretch = float(material["stretch_stiffness"])
        shear = float(material["shear_stiffness"])
        spring_scale = torch.where(
            diagonal,
            torch.full_like(rest, shear),
            torch.full_like(rest, stretch),
        )
        motion = DIRICHLET_BY_ID[scenario.dirichlet_id]
        positions.append(position)
        velocities.append(velocity)
        masks.append(fixed_mask)
        masses.append(mass)
        stiffnesses.append(float(base_spring_stiffness) * spring_scale)
        kind_codes.append(_dirichlet_kind_code(motion.kind))
        amplitudes.append(float(motion.amplitude))
        omegas.append(2.0 * pi * float(motion.frequency_hz))
        scenario_ids.append(int(scenario.scenario_id))
        boundary_ids.append(str(scenario.boundary_id))
        material_ids.append(str(scenario.material_id))

    initial_positions = torch.stack(positions, dim=0)
    fixed_mask = torch.stack(masks, dim=0)
    twist_signs = torch.stack(
        [
            _compute_twist_signs(
                initial_positions[index],
                fixed_mask[index],
                kind_codes[index],
            )
            for index in range(len(scenarios))
        ],
        dim=0,
    )
    batch_size = len(scenarios)
    return BatchedClothParameters(
        topology=topology,
        dt=float(dt),
        gravity=torch.tensor(
            (0.0, 0.0, float(gravity)),
            dtype=dtype,
            device=device,
        ),
        masses=torch.stack(masses, dim=0),
        spring_stiffness=torch.stack(stiffnesses, dim=0),
        rest_lengths=rest.unsqueeze(0).expand(batch_size, -1).clone(),
        fixed_mask=fixed_mask,
        initial_positions=initial_positions,
        initial_velocities=torch.stack(velocities, dim=0),
        dirichlet_kind=torch.as_tensor(kind_codes, dtype=torch.long, device=device),
        dirichlet_amplitude=torch.as_tensor(amplitudes, dtype=dtype, device=device),
        dirichlet_omega=torch.as_tensor(omegas, dtype=dtype, device=device),
        twist_signs=twist_signs,
        scenario_ids=torch.as_tensor(scenario_ids, dtype=torch.long, device=device),
        boundary_ids=tuple(boundary_ids),
        material_ids=tuple(material_ids),
    )


def _as_positions(
    value: torch.Tensor,
    params: BatchedClothParameters,
    name: str,
) -> tuple[torch.Tensor, bool]:
    flat_shape = (params.batch_size, params.num_vertices * 3)
    point_shape = (params.batch_size, params.num_vertices, 3)
    if value.ndim == 2 and tuple(value.shape) == flat_shape:
        return value.reshape(point_shape), True
    if value.ndim == 3 and tuple(value.shape) == point_shape:
        return value, False
    raise ValueError(
        f"{name} must have shape {point_shape} or {flat_shape}, got {tuple(value.shape)}"
    )


def flatten_positions(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("positions must have shape [B, N, 3]")
    return value.reshape(value.shape[0], -1)


def dirichlet_targets(
    params: BatchedClothParameters,
    t: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-size prescribed positions and velocities for every environment."""
    time = torch.as_tensor(t, dtype=params.dtype, device=params.device)
    if time.ndim == 0:
        time = time.expand(params.batch_size)
    if tuple(time.shape) != (params.batch_size,):
        raise ValueError(
            f"t must be scalar or shape {(params.batch_size,)}, got {tuple(time.shape)}"
        )
    theta = params.dirichlet_omega * time
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    amplitude = params.dirichlet_amplitude
    omega = params.dirichlet_omega

    target = params.initial_positions.clone()
    velocity = torch.zeros_like(target)
    fixed = params.fixed_mask.unsqueeze(-1)

    horizontal = params.dirichlet_kind == DIRICHLET_CIRCLE_HORIZONTAL
    vertical = params.dirichlet_kind == DIRICHLET_CIRCLE_VERTICAL
    twist = params.dirichlet_kind == DIRICHLET_TWIST

    global_displacement = torch.zeros(
        params.batch_size,
        3,
        dtype=params.dtype,
        device=params.device,
    )
    global_velocity = torch.zeros_like(global_displacement)
    global_displacement[horizontal, 0] = amplitude[horizontal] * sin_theta[horizontal]
    global_displacement[horizontal, 1] = amplitude[horizontal] * (
        1.0 - cos_theta[horizontal]
    )
    global_velocity[horizontal, 0] = (
        amplitude[horizontal] * omega[horizontal] * cos_theta[horizontal]
    )
    global_velocity[horizontal, 1] = (
        amplitude[horizontal] * omega[horizontal] * sin_theta[horizontal]
    )
    global_displacement[vertical, 0] = amplitude[vertical] * sin_theta[vertical]
    global_displacement[vertical, 2] = amplitude[vertical] * (
        1.0 - cos_theta[vertical]
    )
    global_velocity[vertical, 0] = (
        amplitude[vertical] * omega[vertical] * cos_theta[vertical]
    )
    global_velocity[vertical, 2] = (
        amplitude[vertical] * omega[vertical] * sin_theta[vertical]
    )
    target = target + fixed * global_displacement[:, None, :]
    velocity = velocity + fixed * global_velocity[:, None, :]

    twist_scale = params.twist_signs * (amplitude * sin_theta)[:, None]
    twist_velocity = params.twist_signs * (
        amplitude * omega * cos_theta
    )[:, None]
    target[twist, :, 2] += twist_scale[twist]
    target[twist, :, 1] += 0.25 * params.twist_signs[twist] * (
        amplitude[twist] * (1.0 - cos_theta[twist])
    )[:, None]
    velocity[twist, :, 2] += twist_velocity[twist]
    velocity[twist, :, 1] += 0.25 * params.twist_signs[twist] * (
        amplitude[twist] * omega[twist] * sin_theta[twist]
    )[:, None]
    velocity = torch.where(fixed, velocity, torch.zeros_like(velocity))
    return target, velocity


def free_update_gate(
    params: BatchedClothParameters,
    *,
    flattened: bool = False,
) -> torch.Tensor:
    gate = (~params.fixed_mask).unsqueeze(-1).expand(-1, -1, 3)
    return flatten_positions(gate) if flattened else gate


def project_positions(
    y: torch.Tensor,
    params: BatchedClothParameters,
    target_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    positions, was_flat = _as_positions(y, params, "y")
    if target_positions is None:
        target_positions = params.initial_positions
    target, _ = _as_positions(target_positions, params, "target_positions")
    projected = torch.where(
        params.fixed_mask.unsqueeze(-1),
        target,
        positions,
    )
    return flatten_positions(projected) if was_flat else projected


def make_q(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    params: BatchedClothParameters,
) -> torch.Tensor:
    points, points_flat = _as_positions(positions, params, "positions")
    speed, speed_flat = _as_positions(velocities, params, "velocities")
    if points_flat != speed_flat:
        raise ValueError("positions and velocities must use the same layout")
    q = points + params.dt * speed - params.dt**2 * params.gravity.reshape(1, 1, 3)
    return flatten_positions(q) if points_flat else q


def spring_lengths(
    y: torch.Tensor,
    params: BatchedClothParameters,
    target_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    projected = project_positions(y, params, target_positions)
    positions, _ = _as_positions(projected, params, "y")
    edges = params.topology.edge_tensor(params.device)
    vectors = positions[:, edges[:, 1], :] - positions[:, edges[:, 0], :]
    return torch.linalg.vector_norm(vectors, dim=-1)


def variational_energy(
    y: torch.Tensor,
    q: torch.Tensor,
    params: BatchedClothParameters,
    target_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    projected = project_positions(y, params, target_positions)
    positions, _ = _as_positions(projected, params, "y")
    q_positions, _ = _as_positions(q, params, "q")
    free = (~params.fixed_mask).to(params.dtype)
    inertial = params.masses / (2.0 * params.dt**2) * torch.sum(
        (positions - q_positions) ** 2,
        dim=-1,
    )
    lengths = spring_lengths(positions, params, target_positions)
    spring = 0.5 * params.spring_stiffness * (
        lengths - params.rest_lengths
    ) ** 2
    return torch.sum(free * inertial, dim=-1) + torch.sum(spring, dim=-1)


def stationarity_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    params: BatchedClothParameters,
    target_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    original_positions, was_flat = _as_positions(y, params, "y")
    projected = project_positions(
        original_positions,
        params,
        target_positions,
    )
    positions, _ = _as_positions(projected, params, "y")
    q_positions, _ = _as_positions(q, params, "q")
    free = (~params.fixed_mask).unsqueeze(-1).to(params.dtype)
    gradient = free * (
        params.masses.unsqueeze(-1) / params.dt**2
    ) * (positions - q_positions)

    edges = params.topology.edge_tensor(params.device)
    vectors = positions[:, edges[:, 1], :] - positions[:, edges[:, 0], :]
    lengths = torch.linalg.vector_norm(
        vectors,
        dim=-1,
        keepdim=True,
    ).clamp_min(DISTANCE_EPS)
    edge_gradient = params.spring_stiffness.unsqueeze(-1) * (
        1.0 - params.rest_lengths.unsqueeze(-1) / lengths
    ) * vectors
    gradient = gradient.clone()
    gradient.index_add_(1, edges[:, 0], -edge_gradient)
    gradient.index_add_(1, edges[:, 1], edge_gradient)
    gradient = torch.where(
        params.fixed_mask.unsqueeze(-1),
        torch.zeros_like(gradient),
        gradient,
    )
    return flatten_positions(gradient) if was_flat else gradient


def stationarity_residual_norm(
    y: torch.Tensor,
    q: torch.Tensor,
    params: BatchedClothParameters,
    target_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = stationarity_residual(y, q, params, target_positions)
    return torch.linalg.vector_norm(
        residual.reshape(params.batch_size, -1),
        dim=-1,
    )


def advance_state(
    current_positions: torch.Tensor,
    solved_positions: torch.Tensor,
    params: BatchedClothParameters,
    *,
    next_time: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    current, current_flat = _as_positions(
        current_positions,
        params,
        "current_positions",
    )
    solved, solved_flat = _as_positions(
        solved_positions,
        params,
        "solved_positions",
    )
    if current_flat != solved_flat:
        raise ValueError(
            "current_positions and solved_positions must use the same layout"
        )
    target, prescribed_velocity = dirichlet_targets(params, next_time)
    next_positions = project_positions(solved, params, target)
    finite_difference_velocity = (next_positions - current) / params.dt
    next_velocities = torch.where(
        params.fixed_mask.unsqueeze(-1),
        prescribed_velocity,
        finite_difference_velocity,
    )
    if current_flat:
        return flatten_positions(next_positions), flatten_positions(next_velocities)
    return next_positions, next_velocities
