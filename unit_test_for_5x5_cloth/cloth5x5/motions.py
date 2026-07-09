from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import torch

from .config import MotionSpec, MotionSplit, PhysicalConfig
from .constants import (
    FIXED_VERTEX_INDICES,
    GRID_COLS,
    GRID_ROWS,
    MOTION_SOBOL_SEED_ID_TEST,
    MOTION_SOBOL_SEED_TRAIN,
    MOTION_SOBOL_SEED_VALIDATION,
    TORCH_DTYPE,
    grid_index,
)


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
    if stretch_x <= 0.0:
        raise ValueError("stretch_x must be positive")
    base = np.asarray(physical.p0, dtype=float)
    positions = base.copy()
    velocities = np.zeros_like(base)
    span_x = max((GRID_COLS - 1) * 0.5, 1e-12)
    span_y = max((GRID_ROWS - 1) * 0.5, 1e-12)
    height = float(base[0, 2])
    transl = np.asarray(translation_velocity, dtype=float)
    center = np.mean(base, axis=0)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            idx = grid_index(row, col)
            u = col / max(GRID_COLS - 1, 1)
            v = row / max(GRID_ROWS - 1, 1)
            x0, y0, _ = base[idx]
            x = stretch_x * x0
            y = y0 + shear * x0 * (2.0 * v - 1.0)
            z = (
                height
                + bend * math.sin(math.pi * u) * math.sin(math.pi * v)
                + twist * (x0 / span_x) * (2.0 * v - 1.0)
            )
            positions[idx] = (x, y, z)

            radial = positions[idx] - center
            rotational = float(angular_velocity) * np.cross(
                np.asarray([0.0, 1.0, 0.0]), radial
            )
            gradient = np.asarray(
                [
                    velocity_gradient * (2.0 * v - 1.0) * u,
                    0.35 * velocity_gradient * math.sin(math.pi * u),
                    -0.5 * velocity_gradient * (2.0 * v - 1.0) * u,
                ]
            )
            velocities[idx] = transl + rotational + gradient

    fixed = list(FIXED_VERTEX_INDICES)
    positions[fixed] = base[fixed]
    velocities[fixed] = 0.0
    return MotionSpec(
        index=index,
        name=name,
        split=split,
        category=category,
        source=source,
        p0=tuple(tuple(float(x) for x in row) for row in positions),
        v0=tuple(tuple(float(x) for x in row) for row in velocities),
        stretch_x=float(stretch_x),
        shear=float(shear),
        bend=float(bend),
        twist=float(twist),
        translation_velocity=tuple(float(x) for x in transl),
        angular_velocity=float(angular_velocity),
        velocity_gradient=float(velocity_gradient),
        ood_factors=tuple(str(x) for x in ood_factors),
    )


def generate_in_domain_sobol_motion_specs(
    *,
    count: int,
    start_index: int,
    seed: int,
    split: str,
    physical: PhysicalConfig,
    name_prefix: str,
) -> list[MotionSpec]:
    engine = torch.quasirandom.SobolEngine(dimension=10, scramble=True, seed=seed)
    unit = engine.draw(count).to(dtype=TORCH_DTYPE).cpu().numpy()
    motions: list[MotionSpec] = []
    for offset, u in enumerate(unit):
        motions.append(
            make_motion_spec(
                index=start_index + offset,
                name=f"{name_prefix}_{offset:02d}",
                split=split,
                category="in_domain_sobol",
                source=f"scrambled_sobol_seed_{seed}",
                physical=physical,
                stretch_x=0.80 + 0.50 * u[0],
                shear=-0.25 + 0.50 * u[1],
                bend=-0.25 + 0.50 * u[2],
                twist=-0.22 + 0.44 * u[3],
                translation_velocity=(
                    -2.0 + 4.0 * u[4],
                    -1.5 + 3.0 * u[5],
                    -1.5 + 4.0 * u[6],
                ),
                angular_velocity=-1.5 + 3.0 * u[7],
                velocity_gradient=-1.2 + 2.4 * u[8],
            )
        )
    return motions


def build_motion_catalogue(physical: PhysicalConfig) -> tuple[list[MotionSpec], MotionSplit]:
    motions: list[MotionSpec] = [
        make_motion_spec(
            index=0,
            name="original_horizontal_static",
            split="train",
            category="original",
            source="base_physical_config",
            physical=physical,
        )
    ]
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
        motions.append(
            make_motion_spec(
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
            )
        )
    motions.extend(generate_in_domain_sobol_motion_specs(
        count=8, start_index=8, seed=MOTION_SOBOL_SEED_TRAIN,
        split="train", physical=physical, name_prefix="train_sobol"
    ))
    motions.extend(generate_in_domain_sobol_motion_specs(
        count=4, start_index=16, seed=MOTION_SOBOL_SEED_VALIDATION,
        split="validation", physical=physical, name_prefix="validation_sobol"
    ))
    motions.extend(generate_in_domain_sobol_motion_specs(
        count=4, start_index=20, seed=MOTION_SOBOL_SEED_ID_TEST,
        split="id_test", physical=physical, name_prefix="id_test_sobol"
    ))
    ood = [
        ("ood_fast_horizontal", "ood_velocity", 1.0, 0.0, 0.0, 0.0, (5.2, 0.0, 0.0), 0.0, 0.0, ("horizontal_speed",)),
        ("ood_fast_upward", "ood_velocity", 1.0, 0.0, 0.0, 0.0, (0.3, 0.0, 5.0), 0.0, 0.0, ("upward_speed",)),
        ("ood_strong_stretch", "ood_deformation", 1.65, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0, ("stretch",)),
        ("ood_strong_compression", "ood_deformation", 0.58, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0, ("compression",)),
        ("ood_strong_bend", "ood_deformation", 1.0, 0.0, 0.72, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0, ("bend",)),
        ("ood_strong_twist", "ood_deformation", 1.0, 0.35, 0.18, 0.58, (0.2, 0.0, 0.2), 2.8, 1.5, ("twist", "rotation")),
        ("ood_stretch_fast_side", "ood_combination", 1.55, 0.22, 0.25, 0.25, (4.2, 1.5, 0.8), 2.0, 1.8, ("stretch", "speed", "shear")),
        ("ood_compress_twist_up", "ood_combination", 0.62, -0.38, -0.35, -0.52, (1.0, -0.8, 4.2), -2.6, -2.0, ("compression", "twist", "upward_speed")),
    ]
    for offset, item in enumerate(ood):
        name, category, stretch, shear, bend, twist, velocity, angular, gradient, factors = item
        motions.append(
            make_motion_spec(
                index=24 + offset,
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
            )
        )
    motions = sorted(motions, key=lambda item: item.index)
    if [motion.index for motion in motions] != list(range(32)):
        raise AssertionError("Motion indices must be exactly 0..31")
    split = MotionSplit(
        train_motion_indices=tuple(range(0, 16)),
        validation_motion_indices=tuple(range(16, 20)),
        id_test_motion_indices=tuple(range(20, 24)),
        ood_test_motion_indices=tuple(range(24, 32)),
    )
    return motions, split
