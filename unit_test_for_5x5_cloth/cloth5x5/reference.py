from __future__ import annotations

from typing import Any, Sequence

import torch

from .config import MotionSpec, PhysicalConfig, RuntimeConfig, TimeStepProblem
from .constants import (
    FIXED_VERTEX_INDICES,
    FREE_STATE_DIM,
    FREE_VERTEX_INDICES,
    TORCH_DTYPE,
)
from .physics import (
    advance_physical_state,
    free_state_from_full,
    make_q_free,
    solve_reference_solution,
    spring_lengths_from_free,
    stationarity_residual_norm,
    variational_energy,
)


def generate_reference_sequence_for_motion(
    physical: PhysicalConfig,
    motion: MotionSpec,
    total_steps: int,
    sampling_radius_min: float,
    sampling_radius_max: float,
) -> list[TimeStepProblem]:
    p_n = torch.tensor(motion.p0, dtype=TORCH_DTYPE)
    v_n = torch.tensor(motion.v0, dtype=TORCH_DTYPE)
    fixed = list(FIXED_VERTEX_INDICES)
    p_n[fixed, :] = torch.tensor(physical.fixed_positions, dtype=TORCH_DTYPE)
    v_n[fixed, :] = 0.0
    free_masses = torch.tensor(
        [physical.masses[i] for i in FREE_VERTEX_INDICES], dtype=TORCH_DTYPE
    )
    problems: list[TimeStepProblem] = []

    for local_index in range(total_steps):
        q_free = make_q_free(p_n, v_n, physical)
        initial_y = free_state_from_full(p_n)
        exact_y, info = solve_reference_solution(
            q=q_free,
            masses=free_masses,
            initial_y=initial_y,
            physical=physical,
            raise_on_nonconvergence=False,
        )
        if not info.get("acceptable", False):
            print(
                f"Warning: motion {motion.index:02d} time {local_index:03d} "
                f"reference residual={info['residual_norm']:.3e} status={info.get('status')}; "
                "continuing with the best finite iterate."
            )
        raw_radius = float(torch.max(torch.abs(initial_y - exact_y)).item())
        radius = min(max(raw_radius, sampling_radius_min), sampling_radius_max)
        exact_energy = float(
            variational_energy(
                exact_y.unsqueeze(0), q_free.unsqueeze(0), free_masses.unsqueeze(0), physical
            ).item()
        )
        exact_residual = float(
            stationarity_residual_norm(
                exact_y.unsqueeze(0), q_free.unsqueeze(0), free_masses.unsqueeze(0), physical
            ).item()
        )
        global_index = motion.index * total_steps + local_index
        problems.append(
            TimeStepProblem(
                index=global_index,
                motion_index=motion.index,
                motion_name=motion.name,
                motion_split=motion.split,
                motion_category=motion.category,
                local_time_index=local_index,
                time=local_index * physical.dt,
                p_n_full=p_n.clone(),
                v_n_full=v_n.clone(),
                q_free=q_free.clone(),
                free_masses=free_masses.clone(),
                exact_y_free=exact_y.clone(),
                raw_sampling_radius=raw_radius,
                sampling_radius=radius,
                exact_energy=exact_energy,
                exact_residual=exact_residual,
            )
        )
        if local_index == 0 or (local_index + 1) % 25 == 0:
            print(
                f"Motion {motion.index:02d} {motion.name}: "
                f"reference {local_index + 1:3d}/{total_steps}, "
                f"iterations={info['iterations']}, residual={exact_residual:.3e}, "
                f"radius={radius:.3e}"
            )
        p_n, v_n = advance_physical_state(p_n, exact_y, physical)
    return problems


def generate_all_reference_sequences(
    physical: PhysicalConfig,
    motions: Sequence[MotionSpec],
    config: RuntimeConfig,
) -> list[TimeStepProblem]:
    problems: list[TimeStepProblem] = []
    for motion in motions:
        problems.extend(
            generate_reference_sequence_for_motion(
                physical,
                motion,
                config.total_time_steps,
                config.sampling_radius_min,
                config.sampling_radius_max,
            )
        )
    expected = len(motions) * config.total_time_steps
    if len(problems) != expected:
        raise AssertionError(f"Expected {expected} problems, got {len(problems)}")
    return problems


def problem_lookup(problems: Sequence[TimeStepProblem]) -> dict[tuple[int, int], TimeStepProblem]:
    return {(p.motion_index, p.local_time_index): p for p in problems}
