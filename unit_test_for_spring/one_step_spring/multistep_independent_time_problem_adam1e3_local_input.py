#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-particle single-spring learned optimizer:
independent multi-time-step problem generalization with local physical input.

Only the network input representation is changed relative to the original
multi-time-step spring experiment.  The network input is

    x_local = [e1, e2, d, s, beta1, beta2] in R^12,

where

    e1 = y1 - q1,
    e2 = y2 - q2,
    d  = y2 - y1,
    s  = ||d|| - l0,
    beta_i = k * dt^2 / m_i.

The remaining confirmed settings are retained:
- 100 independent physical time-step problems;
- 60 training, 10 validation, 10 interpolation-test, 20 extrapolation-test;
- 100 training initial points per problem and 256 evaluation points per problem;
- Adam(lr=1e-3), Identity activation, no output scaling;
- per-feature input normalization computed from the training dataset only;
- float64, default device cuda:1;
- full-batch training for 50,000 epochs;
- K increases from 1 to 5 every 10,000 epochs;
- physical variational energy is the only backward objective;
- exact solutions, residuals, energy gaps and exact errors are evaluation data.
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
OUTPUT_SCALE = 1.0
INPUT_REPRESENTATION = "local_physics"
INPUT_DIM = 12
HIDDEN_DIM = 64
DEFAULT_DEVICE = "cuda:1"

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
    device: str
    run_single_problem_baseline: bool
    skip_plots: bool
    save_datasets: bool


@dataclass(frozen=True)
class PhysicalConfig:
    m1: float
    m2: float
    g: float
    dt: float
    spring_k: float
    rest_length: float
    p1_0: tuple[float, float, float]
    p2_0: tuple[float, float, float]
    v1_0: tuple[float, float, float]
    v2_0: tuple[float, float, float]


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

    def subset_by_problem_indices(
        self,
        indices: Iterable[int],
        role: str,
    ) -> "DatasetBundle":
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
                "num_problems": len(wanted),
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
        m1=1.0,
        m2=1.0,
        g=9.8,
        dt=0.01,
        spring_k=2500.0,
        rest_length=1.0,
        p1_0=(-0.6, 0.0, 1.0),
        p2_0=(0.6, 0.2, 1.1),
        v1_0=(0.2, 0.0, 0.1),
        v2_0=(-0.1, 0.15, -0.05),
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def set_model_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Physics, exact solution, energy, and residual
# ============================================================


def swap_particles(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != 6:
        raise ValueError(f"Expected final dimension 6, got {tuple(values.shape)}")
    return torch.cat([values[..., 3:6], values[..., 0:3]], dim=-1)


def swap_masses(masses: torch.Tensor) -> torch.Tensor:
    if masses.shape[-1] != 2:
        raise ValueError(f"Expected final mass dimension 2, got {tuple(masses.shape)}")
    return torch.stack([masses[..., 1], masses[..., 0]], dim=-1)


def exact_solution_from_predictors(
    q: torch.Tensor,
    masses: torch.Tensor,
    dt: float,
    spring_k: float,
    rest_length: float,
) -> torch.Tensor:
    """Analytic minimizer for two free particles joined by one spring."""
    q1 = q[..., 0:3]
    q2 = q[..., 3:6]
    m1 = masses[..., 0:1]
    m2 = masses[..., 1:2]
    total_mass = m1 + m2
    reduced_mass = (m1 * m2) / total_mass
    center = (m1 * q1 + m2 * q2) / total_mass
    d_q = q2 - q1
    a = torch.linalg.vector_norm(d_q, dim=-1, keepdim=True)
    if bool(torch.any(a <= RESIDUAL_DISTANCE_EPS)):
        raise ValueError("The free-predictor relative vector is too close to zero.")
    lambda_value = spring_k * dt**2 / reduced_mass
    r_star = (a + lambda_value * rest_length) / (1.0 + lambda_value)
    d_star = r_star * d_q / a
    y1_star = center - (m2 / total_mass) * d_star
    y2_star = center + (m1 / total_mass) * d_star
    return torch.cat([y1_star, y2_star], dim=-1)


def variational_energy(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    *,
    g: float,
    dt: float,
    spring_k: float,
    rest_length: float,
) -> torch.Tensor:
    y1 = y[..., 0:3]
    y2 = y[..., 3:6]
    q1 = q[..., 0:3]
    q2 = q[..., 3:6]
    m1 = masses[..., 0]
    m2 = masses[..., 1]

    inertial_1 = (m1 / (2.0 * dt**2)) * torch.sum((y1 - q1) ** 2, dim=-1)
    inertial_2 = (m2 / (2.0 * dt**2)) * torch.sum((y2 - q2) ** 2, dim=-1)
    length = torch.linalg.vector_norm(y2 - y1, dim=-1)
    spring = 0.5 * spring_k * (length - rest_length) ** 2

    # Retain the y-independent gravity constants used by the original script.
    gravity_constant_1 = m1 * g * q1[..., 2] + 0.5 * m1 * dt**2 * g**2
    gravity_constant_2 = m2 * g * q2[..., 2] + 0.5 * m2 * dt**2 * g**2
    return inertial_1 + inertial_2 + spring + gravity_constant_1 + gravity_constant_2


def stationarity_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    *,
    dt: float,
    spring_k: float,
    rest_length: float,
) -> torch.Tensor:
    y1 = y[..., 0:3]
    y2 = y[..., 3:6]
    q1 = q[..., 0:3]
    q2 = q[..., 3:6]
    m1 = masses[..., 0:1]
    m2 = masses[..., 1:2]
    d = y2 - y1
    length = torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(
        RESIDUAL_DISTANCE_EPS
    )
    spring_gradient = spring_k * (1.0 - rest_length / length) * d
    grad_y1 = (m1 / dt**2) * (y1 - q1) - spring_gradient
    grad_y2 = (m2 / dt**2) * (y2 - q2) + spring_gradient
    return torch.cat([grad_y1, grad_y2], dim=-1)


def stationarity_residual_norm(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    *,
    dt: float,
    spring_k: float,
    rest_length: float,
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


def generate_reference_sequence(
    physical: PhysicalConfig,
    total_steps: int,
) -> list[TimeStepProblem]:
    """
    Generate the reference physical trajectory using the analytic minimizer.

    The trajectory defines independent optimization problems. Learned outputs
    are never propagated between physical time steps.
    """
    p_n = torch.tensor([*physical.p1_0, *physical.p2_0], dtype=TORCH_DTYPE)
    v_n = torch.tensor([*physical.v1_0, *physical.v2_0], dtype=TORCH_DTYPE)
    masses = torch.tensor([physical.m1, physical.m2], dtype=TORCH_DTYPE)
    gravity = torch.tensor([0.0, 0.0, physical.g], dtype=TORCH_DTYPE)
    problems: list[TimeStepProblem] = []

    for index in range(total_steps):
        p1 = p_n[0:3]
        p2 = p_n[3:6]
        v1 = v_n[0:3]
        v2 = v_n[3:6]
        q1 = p1 + physical.dt * v1 - physical.dt**2 * gravity
        q2 = p2 + physical.dt * v2 - physical.dt**2 * gravity
        q = torch.cat([q1, q2])
        exact_y = exact_solution_from_predictors(
            q.unsqueeze(0),
            masses.unsqueeze(0),
            physical.dt,
            physical.spring_k,
            physical.rest_length,
        ).squeeze(0)
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
    relative = points[:, 3:6] - points[:, 0:3]
    return torch.linalg.vector_norm(relative, dim=-1) > SAMPLE_DISTANCE_EPS


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
        point_cpu = point.detach().cpu().to(TORCH_DTYPE).reshape(1, 6)
        if not bool(nondegenerate_mask(point_cpu)[0]):
            raise ValueError("An explicit point is degenerate.")
        chunks.append(point_cpu)
        accepted += 1
    if accepted > count:
        raise ValueError("More explicit points were supplied than requested samples.")

    engine = torch.quasirandom.SobolEngine(dimension=6, scramble=True, seed=seed)
    generated = 0
    rejected = 0
    while accepted < count:
        remaining = count - accepted
        draw_count = max(32, remaining * 2)
        unit = engine.draw(draw_count).to(dtype=TORCH_DTYPE)
        candidates = center.reshape(1, 6) + (2.0 * unit - 1.0) * radius
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
        "mode": "scrambled_sobol_6d_linf_cube",
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
    swapped_points = swap_particles(canonical_points)
    initial_y = torch.cat([canonical_points, swapped_points], dim=0)

    q_canonical = problem.q.reshape(1, 6)
    q_swapped = swap_particles(q_canonical)
    masses_canonical = problem.masses.reshape(1, 2)
    masses_swapped = swap_masses(masses_canonical)
    exact_canonical = problem.exact_y.reshape(1, 6)
    exact_swapped = swap_particles(exact_canonical)

    q = torch.cat(
        [
            q_canonical.expand(canonical_count, -1),
            q_swapped.expand(canonical_count, -1),
        ],
        dim=0,
    ).clone()
    masses = torch.cat(
        [
            masses_canonical.expand(canonical_count, -1),
            masses_swapped.expand(canonical_count, -1),
        ],
        dim=0,
    ).clone()
    exact_y = torch.cat(
        [
            exact_canonical.expand(canonical_count, -1),
            exact_swapped.expand(canonical_count, -1),
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
            "swap_augmented": True,
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
            "swap_augmented_per_problem": True,
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
        initial_points.append(initial.reshape(1, 6))
        q_values.append(problem.q.reshape(1, 6))
        masses_values.append(problem.masses.reshape(1, 2))
        exact_values.append(problem.exact_y.reshape(1, 6))
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
            "num_problems": len(problem_indices),
            "points_per_problem": 1,
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
# 4. Local physical input, normalization, and learned update
# ============================================================


def build_local_input_features(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """
    Construct x_local = [e1, e2, d, s, beta1, beta2].

    Dimensions:
        e1: 3, e2: 3, d: 3, s: 1, beta: 2, total: 12.
    """
    if y.ndim != 2 or y.shape[-1] != 6:
        raise ValueError(f"Expected y shape [B,6], got {tuple(y.shape)}")
    if q.shape != y.shape:
        raise ValueError(f"q shape {tuple(q.shape)} must match y shape {tuple(y.shape)}")
    if masses.ndim != 2 or masses.shape != (y.shape[0], 2):
        raise ValueError(
            f"Expected masses shape {(y.shape[0], 2)}, got {tuple(masses.shape)}"
        )

    y1 = y[:, 0:3]
    y2 = y[:, 3:6]
    q1 = q[:, 0:3]
    q2 = q[:, 3:6]

    e1 = y1 - q1
    e2 = y2 - q2
    d = y2 - y1
    s = torch.linalg.vector_norm(d, dim=-1, keepdim=True) - physical.rest_length
    beta = physical.spring_k * physical.dt**2 / masses

    features = torch.cat([e1, e2, d, s, beta], dim=-1)
    if features.shape != (y.shape[0], INPUT_DIM):
        raise RuntimeError(
            f"Local input produced {tuple(features.shape)}, expected {(y.shape[0], INPUT_DIM)}."
        )
    return features


class MLPOptimizer(nn.Module):
    """12 -> 64 -> Identity -> 6 learned iterative update."""

    def __init__(self, input_mean: torch.Tensor, input_std: torch.Tensor) -> None:
        super().__init__()
        if input_mean.shape != (INPUT_DIM,) or input_std.shape != (INPUT_DIM,):
            raise ValueError(
                f"Expected normalizer shape {(INPUT_DIM,)}, got "
                f"mean={tuple(input_mean.shape)}, std={tuple(input_std.shape)}"
            )
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.Identity(),
            nn.Linear(HIDDEN_DIM, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("input_mean", input_mean.detach().clone().to(TORCH_DTYPE))
        self.register_buffer("input_std", input_std.detach().clone().to(TORCH_DTYPE))

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        physical: PhysicalConfig,
    ) -> torch.Tensor:
        network_input = build_local_input_features(y, q, masses, physical)
        normalized = (network_input - self.input_mean) / self.input_std
        return self.net(normalized)


def compute_input_normalizer(
    dataset: DatasetBundle,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = build_local_input_features(
        dataset.initial_y,
        dataset.q,
        dataset.masses,
        physical,
    )
    mean = inputs.mean(dim=0)
    std = inputs.std(dim=0, unbiased=False)
    std = torch.where(std > 0.0, std, torch.ones_like(std))
    return mean, std


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    dataset: DatasetBundle,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_output = model(y, dataset.q, dataset.masses, physical)
    # Confirmed best configuration: no output scaling.
    applied_delta = OUTPUT_SCALE * raw_output
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
    return sorted(set([0, steps, *[step for step in report_steps if 0 <= step <= steps]]))


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
    if steps <= 0:
        raise ValueError("steps must be positive.")
    model.eval()
    selected_steps = _selected_step_indices(steps, report_steps)

    residual_batches: list[torch.Tensor] = []
    gap_batches: list[torch.Tensor] = []
    exact_error_batches: list[torch.Tensor] = []
    point1_error_batches: list[torch.Tensor] = []
    point2_error_batches: list[torch.Tensor] = []
    problem_index_batches: list[torch.Tensor] = []

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
        exact_energy = variational_energy(
            batch.exact_y,
            batch.q,
            batch.masses,
            g=physical.g,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        )

        residual_steps: list[torch.Tensor] = []
        gap_steps: list[torch.Tensor] = []
        exact_error_steps: list[torch.Tensor] = []
        point1_error_steps: list[torch.Tensor] = []
        point2_error_steps: list[torch.Tensor] = []

        for step in range(steps + 1):
            residual_steps.append(
                stationarity_residual_norm(
                    y,
                    batch.q,
                    batch.masses,
                    dt=physical.dt,
                    spring_k=physical.spring_k,
                    rest_length=physical.rest_length,
                ).detach().cpu()
            )
            energy = variational_energy(
                y,
                batch.q,
                batch.masses,
                g=physical.g,
                dt=physical.dt,
                spring_k=physical.spring_k,
                rest_length=physical.rest_length,
            )
            gap_steps.append((energy - exact_energy).detach().cpu())
            exact_error_steps.append(
                torch.linalg.vector_norm(y - batch.exact_y, dim=-1).detach().cpu()
            )
            point1_error_steps.append(
                torch.linalg.vector_norm(
                    y[:, 0:3] - batch.exact_y[:, 0:3], dim=-1
                ).detach().cpu()
            )
            point2_error_steps.append(
                torch.linalg.vector_norm(
                    y[:, 3:6] - batch.exact_y[:, 3:6], dim=-1
                ).detach().cpu()
            )
            if step < steps:
                y, _ = apply_model_update(model, y, batch, physical)

        residual_batches.append(torch.stack(residual_steps, dim=1))
        gap_batches.append(torch.stack(gap_steps, dim=1))
        exact_error_batches.append(torch.stack(exact_error_steps, dim=1))
        point1_error_batches.append(torch.stack(point1_error_steps, dim=1))
        point2_error_batches.append(torch.stack(point2_error_steps, dim=1))
        problem_index_batches.append(batch.problem_index.detach().cpu())

    arrays = {
        "residual": torch.cat(residual_batches, dim=0).numpy().astype(float),
        "energy_gap": torch.cat(gap_batches, dim=0).numpy().astype(float),
        "exact_error": torch.cat(exact_error_batches, dim=0).numpy().astype(float),
        "point1_error": torch.cat(point1_error_batches, dim=0).numpy().astype(float),
        "point2_error": torch.cat(point2_error_batches, dim=0).numpy().astype(float),
    }
    problem_indices = torch.cat(problem_index_batches, dim=0).numpy().astype(int)
    for values in arrays.values():
        values[~np.isfinite(values)] = np.nan

    result: dict[str, Any] = {
        "steps": steps,
        "num_points": len(dataset_cpu),
        "selected_report_steps": selected_steps,
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
            problem_record["steps"][str(step)] = {
                name: _statistics(values[mask, step])
                for name, values in arrays.items()
            }
        per_problem[str(problem_index)] = problem_record
    result["per_problem"] = per_problem
    return result


def worst_problem_final_residual_p95(metrics: dict[str, Any]) -> float:
    values: list[float] = []
    final_step = str(metrics["steps"])
    for record in metrics.get("per_problem", {}).values():
        value = record["steps"][final_step]["residual"]["p95"]
        if value is None or not math.isfinite(float(value)):
            return float("inf")
        values.append(float(value))
    return max(values) if values else float("inf")


def validation_selection_key(metrics: dict[str, Any]) -> tuple[float, ...] | None:
    values = (
        float(metrics["final_residual_num_nonfinite"]),
        float(metrics["final_residual_p95"]),
        float(worst_problem_final_residual_p95(metrics)),
        float(metrics["final_residual_median"]),
        float(metrics["final_energy_gap_p95"]),
        float(metrics["final_exact_error_p95"]),
    )
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def compact_test_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": metrics["steps"],
        "num_points": metrics["num_points"],
        "final_residual_mean": metrics["final_residual_mean"],
        "final_residual_median": metrics["final_residual_median"],
        "final_residual_p95": metrics["final_residual_p95"],
        "final_residual_max": metrics["final_residual_max"],
        "final_energy_gap_mean": metrics["final_energy_gap_mean"],
        "final_energy_gap_median": metrics["final_energy_gap_median"],
        "final_energy_gap_p95": metrics["final_energy_gap_p95"],
        "final_energy_gap_max": metrics["final_energy_gap_max"],
        "final_exact_error_mean": metrics["final_exact_error_mean"],
        "final_exact_error_median": metrics["final_exact_error_median"],
        "final_exact_error_p95": metrics["final_exact_error_p95"],
        "final_exact_error_max": metrics["final_exact_error_max"],
        "worst_problem_final_residual_p95": worst_problem_final_residual_p95(metrics),
    }


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
    model.to(device=device, dtype=TORCH_DTYPE)
    return {
        name: evaluate_model_on_dataset(
            model=model,
            dataset_cpu=dataset,
            physical=physical,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps,
            device=device,
        )
        for name, dataset in datasets.items()
    }


# ============================================================
# 6. Plotting
# ============================================================


def safe_positive(values: Sequence[float], floor: float = PLOT_FLOOR) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.full_like(array, np.nan, dtype=float)
    mask = np.isfinite(array)
    result[mask] = np.maximum(array[mask], floor)
    return result


def plot_training_and_validation(
    train_log: Sequence[dict[str, Any]],
    validation_log: Sequence[dict[str, Any]],
    best_epoch: int | None,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(
        [item["epoch"] for item in train_log],
        safe_positive([item["training_gap_for_readability"] for item in train_log]),
    )
    axes[0].set_title("Training trajectory energy gap")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Energy-gap sum")
    axes[0].set_yscale("log")

    axes[1].plot(
        [item["epoch"] for item in validation_log],
        safe_positive([item["metrics"]["final_residual_p95"] for item in validation_log]),
        marker="o",
        markersize=3,
        label="sample residual p95",
    )
    axes[1].plot(
        [item["epoch"] for item in validation_log],
        safe_positive([item["worst_problem_final_residual_p95"] for item in validation_log]),
        marker="s",
        markersize=3,
        label="worst problem p95",
    )
    axes[1].set_title("Validation residual after fixed rollout")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Residual")
    axes[1].set_yscale("log")
    axes[1].legend()

    axes[2].plot(
        [item["epoch"] for item in validation_log],
        safe_positive([item["metrics"]["final_exact_error_p95"] for item in validation_log]),
        marker="o",
        markersize=3,
        label="exact error p95",
    )
    axes[2].plot(
        [item["epoch"] for item in validation_log],
        safe_positive([item["metrics"]["final_energy_gap_p95"] for item in validation_log]),
        marker="s",
        markersize=3,
        label="energy gap p95",
    )
    axes[2].set_title("Validation exact error and energy gap")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")
    axes[2].legend()

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, linestyle="--", alpha=0.7)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_metrics(
    metrics: dict[str, Any],
    title: str,
    save_path: Path,
) -> None:
    steps = np.arange(metrics["steps"] + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, prefix, axis_title in [
        (axes[0], "residual", "Residual"),
        (axes[1], "energy_gap", "Energy gap"),
        (axes[2], "exact_error", "Exact-solution error"),
    ]:
        ax.plot(
            steps,
            safe_positive(metrics[f"{prefix}_median_by_step"]),
            label="median",
        )
        ax.plot(
            steps,
            safe_positive(metrics[f"{prefix}_p95_by_step"]),
            label="p95",
        )
        ax.set_title(axis_title)
        ax.set_xlabel("Iteration")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_comparison(
    summaries: Sequence[dict[str, Any]],
    save_path: Path,
) -> None:
    if not summaries:
        return
    labels = [summary["experiment_name"] for summary in summaries]
    datasets = ["interpolation_test", "extrapolation_test"]
    metrics = [
        ("final_residual_p95", "Residual p95"),
        ("final_energy_gap_p95", "Energy-gap p95"),
        ("final_exact_error_p95", "Exact-error p95"),
    ]
    fig, axes = plt.subplots(len(datasets), len(metrics), figsize=(18, 9), squeeze=False)
    x = np.arange(len(labels))
    for row, dataset_name in enumerate(datasets):
        for col, (metric_key, title) in enumerate(metrics):
            values = [
                summary["compact_best_checkpoint_test"][dataset_name][metric_key]
                for summary in summaries
            ]
            axes[row, col].bar(x, safe_positive(values))
            axes[row, col].set_xticks(x, labels, rotation=15, ha="right")
            axes[row, col].set_yscale("log")
            axes[row, col].set_title(f"{dataset_name}: {title}")
            axes[row, col].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. Training
# ============================================================


def run_experiment(
    *,
    experiment_name: str,
    training_cpu: DatasetBundle,
    validation_cpu: DatasetBundle,
    evaluation_datasets: dict[str, DatasetBundle],
    physical: PhysicalConfig,
    config: RuntimeConfig,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    input_mean, input_std = compute_input_normalizer(training_cpu, physical)
    set_model_seed(MODEL_RANDOM_SEED, device)
    model = MLPOptimizer(
        input_mean=input_mean.to(device=device, dtype=TORCH_DTYPE),
        input_std=input_std.to(device=device, dtype=TORCH_DTYPE),
    ).to(device=device, dtype=TORCH_DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training = training_cpu.to(device)

    print("\n" + "=" * 96)
    print(f"Experiment: {experiment_name}")
    print(f"training problems={training_cpu.metadata.get('num_problems')}")
    print(f"training points={len(training_cpu):,}")
    print(f"architecture={INPUT_DIM}->{HIDDEN_DIM}->identity->6")
    print(f"input={INPUT_REPRESENTATION}, optimizer=Adam(lr={LEARNING_RATE:.0e})")
    print(f"device={device}, dtype={TORCH_DTYPE}, output_scale={OUTPUT_SCALE}")
    print("=" * 96)

    train_log: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    best_epoch: int | None = None
    diverged = False
    divergence_epoch: int | None = None
    divergence_reason: str | None = None
    start_time = time.perf_counter()

    exact_training_energy = variational_energy(
        training.exact_y,
        training.q,
        training.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    ).detach()

    for epoch_index in range(config.epochs):
        epoch_number = epoch_index + 1
        rollout_k = get_k_for_epoch(epoch_index, config)
        model.train()
        y = training.initial_y
        optimizer.zero_grad(set_to_none=True)
        trajectory_loss = torch.zeros((), device=device, dtype=TORCH_DTYPE)
        monitoring_gap = torch.zeros((), device=device, dtype=TORCH_DTYPE)

        for _ in range(rollout_k):
            y, _ = apply_model_update(model, y, training, physical)
            energy = variational_energy(
                y,
                training.q,
                training.masses,
                g=physical.g,
                dt=physical.dt,
                spring_k=physical.spring_k,
                rest_length=physical.rest_length,
            )
            # Backward objective remains the raw physical variational energy.
            trajectory_loss = trajectory_loss + energy.mean()
            # Monitoring only; this is not substituted for the backward loss.
            monitoring_gap = monitoring_gap + (energy - exact_training_energy).mean()

        if not bool(torch.isfinite(trajectory_loss)):
            diverged = True
            divergence_epoch = epoch_number
            divergence_reason = "non-finite full-batch trajectory loss"
        else:
            trajectory_loss.backward()
            optimizer.step()

        if not diverged and not is_model_finite(model):
            diverged = True
            divergence_epoch = epoch_number
            divergence_reason = "non-finite model parameter after optimizer.step"

        if diverged:
            print(
                f"Training stopped: epoch={divergence_epoch}, reason={divergence_reason}",
                flush=True,
            )
            break

        train_log.append(
            {
                "epoch": epoch_number,
                "K": rollout_k,
                "trajectory_energy_sum": float(trajectory_loss.detach().item()),
                "training_gap_for_readability": float(monitoring_gap.detach().item()),
            }
        )

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
            worst_problem_p95 = worst_problem_final_residual_p95(validation_metrics)
            validation_log.append(
                {
                    "epoch": epoch_number,
                    "training_K": rollout_k,
                    "selection_key": list(current_key) if current_key is not None else None,
                    "worst_problem_final_residual_p95": worst_problem_p95,
                    "metrics": validation_metrics,
                }
            )
            if current_key is not None and (best_key is None or current_key < best_key):
                best_key = current_key
                best_epoch = epoch_number
                best_validation_metrics = copy.deepcopy(validation_metrics)
                best_state_dict = state_dict_to_cpu(model)

            elapsed = time.perf_counter() - start_time
            print(
                f"Epoch {epoch_number:5d} | K={rollout_k} | "
                f"train_gap={train_log[-1]['training_gap_for_readability']:.4e} | "
                f"val_res_p95={validation_metrics['final_residual_p95']:.4e} | "
                f"worst_problem_res_p95={worst_problem_p95:.4e} | "
                f"best_epoch={best_epoch} | elapsed={elapsed:.1f}s",
                flush=True,
            )

    last_state_dict = state_dict_to_cpu(model)
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(last_state_dict)
        best_epoch = train_log[-1]["epoch"] if train_log else 0

    # Preserve the original checkpoint format: save the raw state_dict.
    # input_mean and input_std are registered buffers and are therefore already
    # included in each state_dict.
    torch.save(last_state_dict, experiment_dir / "last_model_state_dict.pt")
    torch.save(
        best_state_dict,
        experiment_dir / "best_validation_model_state_dict.pt",
    )
    torch.save(best_state_dict, experiment_dir / "mlp_optimizer_state_dict.pt")

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
            "architecture": f"{INPUT_DIM} -> {HIDDEN_DIM} -> identity -> 6",
            "activation": ACTIVATION_NAME,
            "optimizer": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
            "output_scale": OUTPUT_SCALE,
            "input_representation": INPUT_REPRESENTATION,
            "input_definition": {
                "e1": "y1 - q1",
                "e2": "y2 - q2",
                "d": "y2 - y1",
                "s": "||d|| - rest_length",
                "beta_i": "spring_k * dt^2 / m_i",
            },
            "input_dimension": INPUT_DIM,
            "input_normalization": "per-feature, computed from this training dataset only",
            "input_mean": tensor_to_list(input_mean),
            "input_std": tensor_to_list(input_std),
            "epochs_requested": config.epochs,
            "completed_epochs": len(train_log),
            "validation_interval": config.validation_interval,
            "evaluation_steps": config.evaluation_steps,
            "report_steps": list(config.report_steps),
            "initial_K": config.initial_k,
            "K_increase_interval": config.k_increase_interval,
            "K_increase_amount": config.k_increase_amount,
            "max_K": config.max_k,
            "training_mode": "full_batch",
            "training_objective": "sum of mean raw physical variational energies over the unrolled trajectory",
            "backpropagation": "full unroll, no detach, one backward and one Adam step per epoch",
            "training_num_problems": training_cpu.metadata.get("num_problems"),
            "training_num_points": len(training_cpu),
        },
        "training_status": {
            "diverged": diverged,
            "divergence_epoch": divergence_epoch,
            "divergence_reason": divergence_reason,
            "elapsed_seconds": time.perf_counter() - start_time,
        },
        "best_validation_checkpoint": {
            "epoch": best_epoch,
            "selection_key": list(best_key) if best_key is not None else None,
            "validation_metrics": best_validation_metrics,
        },
        "train_log": train_log,
        "validation_log": validation_log,
        "best_checkpoint_test": best_results,
        "last_checkpoint_test": last_results,
    }
    save_json(report, experiment_dir / "optimization_report.json")

    if not config.skip_plots:
        plot_training_and_validation(
            train_log,
            validation_log,
            best_epoch,
            experiment_dir / "training_and_validation_curves.png",
        )
        for dataset_name in [
            "validation",
            "interpolation_test",
            "extrapolation_test",
            "current_state_all_test",
            "exact_state_all_test",
        ]:
            plot_rollout_metrics(
                best_results[dataset_name],
                f"{experiment_name}: {dataset_name}",
                experiment_dir / f"{dataset_name}_best_checkpoint_rollout.png",
            )

    summary = {
        "experiment_name": experiment_name,
        "training_num_problems": training_cpu.metadata.get("num_problems"),
        "training_num_points": len(training_cpu),
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": list(best_key) if best_key is not None else None,
        "diverged": diverged,
        "best_checkpoint_test": best_results,
        "last_checkpoint_test": last_results,
        "compact_best_checkpoint_test": {
            name: compact_test_metrics(metrics)
            for name, metrics in best_results.items()
        },
        "training_curve_for_summary": downsample_log(train_log),
        "validation_curve_for_summary": downsample_log(validation_log),
    }
    save_json(summary, experiment_dir / "experiment_summary.json")
    return summary


# ============================================================
# 8. Command-line interface and orchestration
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train independent multi-time-step two-particle spring learned "
            "optimizers using x_local=[e1,e2,d,s,beta1,beta2]."
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
        "--skip-single-problem-baseline",
        action="store_true",
        help="Train only the 60-problem model.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG output during quick tests.",
    )
    parser.add_argument(
        "--save-datasets",
        action="store_true",
        help="Save complete dataset tensors in addition to metadata.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    validate_even_size("train_points_per_problem", int(args.train_points_per_problem))
    validate_even_size("eval_points_per_problem", int(args.eval_points_per_problem))
    if int(args.total_time_steps) != 100:
        raise ValueError("The confirmed experiment requires --total-time-steps 100.")
    positive_fields = {
        "epochs": args.epochs,
        "validation_interval": args.validation_interval,
        "evaluation_steps": args.evaluation_steps,
        "evaluation_batch_size": args.evaluation_batch_size,
        "initial_k": args.initial_k,
        "k_increase_interval": args.k_increase_interval,
        "k_increase_amount": args.k_increase_amount,
        "max_k": args.max_k,
    }
    for name, value in positive_fields.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")
    if int(args.initial_k) > int(args.max_k):
        raise ValueError("initial_k cannot exceed max_k.")

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
    }


def main() -> None:
    config = validate_args(parse_args())
    physical = default_physical_config()
    output_dir = create_output_directory()
    device = torch.device(config.device)
    validate_device(device)

    problems = generate_reference_sequence(physical, config.total_time_steps)
    split = build_problem_split(config.total_time_steps)

    training_multi = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.train_indices,
        points_per_problem=config.train_points_per_problem,
        base_seed=TRAIN_SOBOL_SEED,
        role="training_multi_problem",
        include_explicit_train_points=True,
    )
    training_single = training_multi.subset_by_problem_indices(
        [0], role="training_single_problem_baseline"
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
        "validation": validation,
        "interpolation_test": interpolation_test,
        "extrapolation_test": extrapolation_test,
        "current_state_all_test": current_state_all_test,
        "exact_state_all_test": exact_state_all_test,
    }

    metadata = {
        "physical": asdict(physical),
        "runtime": asdict(config),
        "problem_split": asdict(split),
        "network": {
            "input_representation": INPUT_REPRESENTATION,
            "input_definition": "[e1,e2,d,s,beta1,beta2]",
            "input_dimension": INPUT_DIM,
            "architecture": f"{INPUT_DIM}->{HIDDEN_DIM}->identity->6",
            "optimizer": f"Adam(lr={LEARNING_RATE})",
            "output_scale": OUTPUT_SCALE,
            "dtype": str(TORCH_DTYPE),
        },
        "problems": [problem_to_record(problem) for problem in problems],
        "datasets": {
            "training_multi": training_multi.metadata,
            "training_single": training_single.metadata,
            **{name: dataset.metadata for name, dataset in evaluation_datasets.items()},
        },
    }
    save_json(metadata, output_dir / "shared_experiment_metadata.json")

    if config.save_datasets:
        torch.save(
            {
                "training_multi": dataset_to_serializable_dict(training_multi),
                "training_single": dataset_to_serializable_dict(training_single),
                **{
                    name: dataset_to_serializable_dict(dataset)
                    for name, dataset in evaluation_datasets.items()
                },
                "metadata": metadata,
            },
            output_dir / "fixed_datasets.pt",
        )

    print("=" * 96)
    print("Two-particle spring independent multi-time-step experiment")
    print(f"Input           : x_local=[e1,e2,d,s,beta1,beta2], dim={INPUT_DIM}")
    print(f"Architecture    : {INPUT_DIM}->{HIDDEN_DIM}->Identity->6")
    print(f"Optimizer       : Adam(lr={LEARNING_RATE:.0e})")
    print(f"Output scaling  : none")
    print(f"Device          : {device}")
    print(f"Dtype           : {TORCH_DTYPE}")
    print(f"Output directory: {output_dir}")
    print("=" * 96)

    summaries: list[dict[str, Any]] = []
    summaries.append(
        run_experiment(
            experiment_name="multi_problem_local_input",
            training_cpu=training_multi,
            validation_cpu=validation,
            evaluation_datasets=evaluation_datasets,
            physical=physical,
            config=config,
            device=device,
            output_dir=output_dir,
        )
    )

    if config.run_single_problem_baseline:
        summaries.append(
            run_experiment(
                experiment_name="single_problem_baseline_local_input",
                training_cpu=training_single,
                validation_cpu=validation,
                evaluation_datasets=evaluation_datasets,
                physical=physical,
                config=config,
                device=device,
                output_dir=output_dir,
            )
        )

    summary_report = {
        "physical": asdict(physical),
        "runtime": asdict(config),
        "input_representation": INPUT_REPRESENTATION,
        "input_definition": "[e1,e2,d,s,beta1,beta2]",
        "dtype": str(TORCH_DTYPE),
        "device": str(device),
        "experiments": summaries,
    }
    save_json(summary_report, output_dir / "multi_time_step_summary.json")
    if not config.skip_plots:
        plot_experiment_comparison(
            summaries,
            output_dir / "multi_vs_single_problem_test_comparison.png",
        )

    print("\nAll experiments completed.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
