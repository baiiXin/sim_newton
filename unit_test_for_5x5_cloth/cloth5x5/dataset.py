from __future__ import annotations

from typing import Any, Sequence

import torch

from .config import DatasetBundle, PhysicalConfig, TimeStepProblem
from .constants import DISTANCE_EPS, FREE_STATE_DIM, TORCH_DTYPE
from .physics import free_state_from_full, spring_lengths_from_free


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

    engine = torch.quasirandom.SobolEngine(dimension=FREE_STATE_DIM, scramble=True, seed=seed)
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
        (free_state_from_full(problem.p_n_full), problem.exact_y_free)
        if include_explicit_train_points else ()
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
        motion_index=torch.full((size,), problem.motion_index, dtype=torch.long),
        time_index=torch.full((size,), problem.local_time_index, dtype=torch.long),
        metadata={
            "role": role,
            "problem_index": problem.index,
            "motion_index": problem.motion_index,
            "motion_name": problem.motion_name,
            "motion_split": problem.motion_split,
            "local_time_index": problem.local_time_index,
            "physical_time": problem.time,
            "size": size,
            "sampling": sampling,
        },
    )


def concatenate_datasets(
    datasets: Sequence[DatasetBundle],
    *,
    role: str,
    points_per_problem: int,
) -> DatasetBundle:
    if not datasets:
        raise ValueError(f"No datasets supplied for role={role}")
    problem_indices = [int(d.metadata["problem_index"]) for d in datasets]
    motion_indices = sorted(set(int(d.metadata["motion_index"]) for d in datasets))
    return DatasetBundle(
        initial_y=torch.cat([d.initial_y for d in datasets], dim=0),
        q=torch.cat([d.q for d in datasets], dim=0),
        masses=torch.cat([d.masses for d in datasets], dim=0),
        exact_y=torch.cat([d.exact_y for d in datasets], dim=0),
        problem_index=torch.cat([d.problem_index for d in datasets], dim=0),
        motion_index=torch.cat([d.motion_index for d in datasets], dim=0),
        time_index=torch.cat([d.time_index for d in datasets], dim=0),
        metadata={
            "role": role,
            "problem_indices": problem_indices,
            "motion_indices": motion_indices,
            "num_motions": len(motion_indices),
            "num_problems": len(problem_indices),
            "points_per_problem": points_per_problem,
            "size": sum(len(d) for d in datasets),
            "split_unit": "complete_motion",
            "no_motion_leakage": True,
        },
    )


def build_dataset_for_motion_times(
    *,
    lookup: dict[tuple[int, int], TimeStepProblem],
    motion_indices: Sequence[int],
    time_indices: Sequence[int],
    points_per_problem: int,
    base_seed: int,
    role: str,
    physical: PhysicalConfig,
    include_explicit_train_points: bool,
) -> DatasetBundle:
    datasets: list[DatasetBundle] = []
    for motion_index in motion_indices:
        for time_index in time_indices:
            problem = lookup[(int(motion_index), int(time_index))]
            seed = base_seed + 100_003 * int(motion_index) + 1009 * int(time_index)
            datasets.append(
                build_problem_dataset(
                    problem=problem,
                    size=points_per_problem,
                    seed=seed,
                    role=f"{role}_m{motion_index:02d}_t{time_index:03d}",
                    physical=physical,
                    include_explicit_train_points=include_explicit_train_points,
                )
            )
    return concatenate_datasets(datasets, role=role, points_per_problem=points_per_problem)


def build_special_state_dataset(
    *,
    lookup: dict[tuple[int, int], TimeStepProblem],
    motion_indices: Sequence[int],
    time_indices: Sequence[int],
    state: str,
    role: str,
) -> DatasetBundle:
    records: list[DatasetBundle] = []
    for motion_index in motion_indices:
        for time_index in time_indices:
            problem = lookup[(int(motion_index), int(time_index))]
            if state == "current":
                y0 = free_state_from_full(problem.p_n_full)
            elif state == "exact":
                y0 = problem.exact_y_free
            else:
                raise ValueError(state)
            records.append(
                DatasetBundle(
                    initial_y=y0.reshape(1, -1),
                    q=problem.q_free.reshape(1, -1),
                    masses=problem.free_masses.reshape(1, -1),
                    exact_y=problem.exact_y_free.reshape(1, -1),
                    problem_index=torch.tensor([problem.index], dtype=torch.long),
                    motion_index=torch.tensor([problem.motion_index], dtype=torch.long),
                    time_index=torch.tensor([problem.local_time_index], dtype=torch.long),
                    metadata={
                        "problem_index": problem.index,
                        "motion_index": problem.motion_index,
                        "state": state,
                    },
                )
            )
    return concatenate_datasets(records, role=role, points_per_problem=1)


def dataset_to_serializable_dict(dataset: DatasetBundle) -> dict[str, Any]:
    return {
        "initial_y": dataset.initial_y,
        "q": dataset.q,
        "masses": dataset.masses,
        "exact_y": dataset.exact_y,
        "problem_index": dataset.problem_index,
        "motion_index": dataset.motion_index,
        "time_index": dataset.time_index,
        "metadata": dataset.metadata,
    }
