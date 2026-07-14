"""Resolution-independent geometry realization and moving Dirichlet targets."""
from __future__ import annotations

from dataclasses import asdict
from math import cos, pi, sin
from typing import Any, Sequence

import numpy as np

from scenario_templates import (
    BOUNDARY_BY_ID,
    DIRICHLET_BY_ID,
    MATERIAL_BY_ID,
    ORIENTATION_BY_ID,
    SHAPE_BY_ID,
    STRAIN_BY_ID,
    VELOCITY_BY_ID,
    BoundaryTemplate,
    ScenarioSpec,
    ShapeTemplate,
)


def resolve_boundary_indices(
    boundary: BoundaryTemplate | str,
    rows: int,
    cols: int,
) -> tuple[int, ...]:
    if isinstance(boundary, str):
        boundary = BOUNDARY_BY_ID[boundary]
    if rows < 2 or cols < 2:
        raise ValueError("rows and cols must both be >= 2")
    if boundary.selector == "none":
        return ()
    if boundary.selector == "edge":
        if boundary.edge == "top":
            indices = [col for col in range(cols)]
        elif boundary.edge == "bottom":
            indices = [(rows - 1) * cols + col for col in range(cols)]
        elif boundary.edge == "left":
            indices = [row * cols for row in range(rows)]
        elif boundary.edge == "right":
            indices = [row * cols + cols - 1 for row in range(rows)]
        else:
            raise ValueError(f"Unknown edge selector: {boundary.edge}")
        return tuple(indices)
    if boundary.selector != "points":
        raise ValueError(f"Unknown selector: {boundary.selector}")
    indices = []
    for u, v in boundary.normalized_points:
        col = int(round(float(u) * (cols - 1)))
        row = int(round(float(v) * (rows - 1)))
        col = min(max(col, 0), cols - 1)
        row = min(max(row, 0), rows - 1)
        indices.append(row * cols + col)
    return tuple(dict.fromkeys(indices))


def is_compatible(boundary_id: str, dirichlet_id: str) -> bool:
    boundary = BOUNDARY_BY_ID[boundary_id]
    motion = DIRICHLET_BY_ID[dirichlet_id]
    if boundary.family == "none":
        return motion.kind == "static"
    if motion.kind == "twist":
        return boundary.family not in {"single", "none"}
    return True


def _rotation_matrix_xyz(degrees: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(degrees, dtype=np.float64))
    cx, sx = cos(rx), sin(rx)
    cy, sy = cos(ry), sin(ry)
    cz, sz = cos(rz), sin(rz)
    mx = np.asarray(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    my = np.asarray(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    mz = np.asarray(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return mz @ my @ mx


def _shape_height(shape: ShapeTemplate, u: np.ndarray, v: np.ndarray, span: float) -> np.ndarray:
    amplitude = shape.amplitude_fraction * span
    if shape.family == "plane":
        return np.zeros_like(u)
    if shape.family == "saddle_uv":
        return amplitude * u * v
    if shape.family == "saddle_quad":
        return amplitude * (u * u - v * v)
    if shape.family == "dome":
        u01 = 0.5 * (u + 1.0)
        v01 = 0.5 * (v + 1.0)
        return amplitude * np.sin(pi * u01) * np.sin(pi * v01)
    raise ValueError(f"Unknown shape family: {shape.family}")


def build_initial_state(
    scenario: ScenarioSpec,
    *,
    rows: int = 5,
    cols: int = 5,
    spacing: float = 0.5,
    height: float = 1.2,
) -> dict[str, Any]:
    """Realize one scenario without requiring a reference trajectory."""
    shape = SHAPE_BY_ID[scenario.shape_id]
    strain = STRAIN_BY_ID[scenario.strain_id]
    velocity = VELOCITY_BY_ID[scenario.velocity_id]
    orientation = ORIENTATION_BY_ID[scenario.orientation_id]
    boundary = BOUNDARY_BY_ID[scenario.boundary_id]
    material = MATERIAL_BY_ID[scenario.material_id]
    if not is_compatible(boundary.id, scenario.dirichlet_id):
        raise ValueError(f"Incompatible boundary/Dirichlet pair: {boundary.id}, {scenario.dirichlet_id}")

    col_grid, row_grid = np.meshgrid(
        np.arange(cols, dtype=np.float64),
        np.arange(rows, dtype=np.float64),
    )
    u01 = col_grid.reshape(-1) / (cols - 1)
    v01 = row_grid.reshape(-1) / (rows - 1)
    u = 2.0 * u01 - 1.0
    v = 2.0 * v01 - 1.0
    span_x = (cols - 1) * spacing
    span_y = (rows - 1) * spacing
    local_x = (u01 - 0.5) * span_x * strain.scale_x
    local_y = -(v01 - 0.5) * span_y * strain.scale_y
    local_z = _shape_height(shape, u, v, min(span_x, span_y))
    local = np.stack((local_x, local_y, local_z), axis=-1)

    rotation = _rotation_matrix_xyz(orientation.euler_xyz_degrees)
    rotated = local @ rotation.T
    center = np.asarray((0.5 * span_x, -0.5 * span_y, height), dtype=np.float64)
    positions = rotated + center

    if velocity.kind == "zero":
        velocities = np.zeros_like(positions)
    elif velocity.kind == "translation":
        velocities = np.broadcast_to(np.asarray(velocity.vector, dtype=np.float64), positions.shape).copy()
    elif velocity.kind == "rotation":
        axis = np.asarray(velocity.axis, dtype=np.float64)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-30)
        omega = velocity.angular_speed * axis
        velocities = np.cross(np.broadcast_to(omega, positions.shape), positions - center)
    else:
        raise ValueError(f"Unknown velocity kind: {velocity.kind}")

    fixed_indices = resolve_boundary_indices(boundary, rows, cols)
    fixed_mask = np.zeros(rows * cols, dtype=bool)
    if fixed_indices:
        fixed_mask[np.asarray(fixed_indices, dtype=np.int64)] = True
        target_positions, target_velocities = dirichlet_targets(
            scenario,
            positions,
            t=0.0,
            fixed_indices=fixed_indices,
        )
        positions[np.asarray(fixed_indices, dtype=np.int64)] = target_positions
        velocities[np.asarray(fixed_indices, dtype=np.int64)] = target_velocities

    return {
        "positions": np.ascontiguousarray(positions, dtype=np.float64),
        "velocities": np.ascontiguousarray(velocities, dtype=np.float64),
        "fixed_mask": fixed_mask,
        "fixed_indices": fixed_indices,
        "material": asdict(material),
        "grid": {"rows": rows, "cols": cols, "spacing": spacing, "height": height},
    }


def dirichlet_targets(
    scenario: ScenarioSpec,
    initial_positions: np.ndarray,
    *,
    t: float,
    fixed_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return prescribed fixed-point positions and velocities at time ``t``."""
    motion = DIRICHLET_BY_ID[scenario.dirichlet_id]
    indices = np.asarray(tuple(fixed_indices), dtype=np.int64)
    base = np.asarray(initial_positions, dtype=np.float64)[indices]
    if indices.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    if motion.kind == "static":
        return base.copy(), np.zeros_like(base)

    omega = 2.0 * pi * motion.frequency_hz
    theta = omega * float(t)
    amplitude = motion.amplitude
    if motion.kind == "circle_horizontal":
        displacement = np.asarray(
            (amplitude * sin(theta), amplitude * (1.0 - cos(theta)), 0.0),
            dtype=np.float64,
        )
        velocity = np.asarray(
            (amplitude * omega * cos(theta), amplitude * omega * sin(theta), 0.0),
            dtype=np.float64,
        )
        return base + displacement, np.broadcast_to(velocity, base.shape).copy()
    if motion.kind == "circle_vertical":
        displacement = np.asarray(
            (amplitude * sin(theta), 0.0, amplitude * (1.0 - cos(theta))),
            dtype=np.float64,
        )
        velocity = np.asarray(
            (amplitude * omega * cos(theta), 0.0, amplitude * omega * sin(theta)),
            dtype=np.float64,
        )
        return base + displacement, np.broadcast_to(velocity, base.shape).copy()
    if motion.kind == "twist":
        centroid = np.mean(base, axis=0)
        centered = base - centroid
        score = centered[:, 0]
        if float(np.max(np.abs(score))) < 1e-12:
            score = centered[:, 1]
        signs = np.where(score >= 0.0, 1.0, -1.0)
        ties = np.abs(score) < 1e-12
        signs[ties] = np.where(np.arange(indices.size)[ties] % 2 == 0, 1.0, -1.0)
        displacement = np.zeros_like(base)
        displacement[:, 2] = signs * amplitude * sin(theta)
        displacement[:, 1] = signs * 0.25 * amplitude * (1.0 - cos(theta))
        velocity = np.zeros_like(base)
        velocity[:, 2] = signs * amplitude * omega * cos(theta)
        velocity[:, 1] = signs * 0.25 * amplitude * omega * sin(theta)
        return base + displacement, velocity
    raise ValueError(f"Unknown Dirichlet kind: {motion.kind}")


def build_triangular_edges(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()

    def add(a: int, b: int) -> None:
        edges.add((min(a, b), max(a, b)))

    def index(row: int, col: int) -> int:
        return row * cols + col

    for row in range(rows):
        for col in range(cols - 1):
            add(index(row, col), index(row, col + 1))
    for row in range(rows - 1):
        for col in range(cols):
            add(index(row, col), index(row + 1, col))
    for row in range(rows - 1):
        for col in range(cols - 1):
            if (row + col) % 2 == 0:
                add(index(row, col), index(row + 1, col + 1))
            else:
                add(index(row + 1, col), index(row, col + 1))
    return tuple(sorted(edges))
