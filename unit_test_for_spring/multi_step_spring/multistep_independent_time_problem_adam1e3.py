"""
Two-particle single-spring learned optimizer:
independent multi-time-step problem generalization experiment.

Confirmed experiment design
---------------------------
1. Generate a 100-step analytic reference trajectory with dt=0.01.
2. Treat every physical time step as an independent optimization problem.
   Network predictions are NOT propagated from one physical step to the next.
3. Split by physical problem index, never by individual sampled initial point:
   - training: 60 problems in steps 0..79
   - validation: steps 3, 11, ..., 75 (10 problems)
   - interpolation test: steps 7, 15, ..., 79 (10 problems)
   - extrapolation test: steps 80..99 (20 problems)
4. Each training problem has 100 optimization initial points.
   Each validation/test problem has 256 optimization initial points.
5. Fixed best single-step configuration:
   - Adam(lr=1e-3)
   - identity activation
   - per-feature input normalization from training data only
   - raw network output (no output scaling)
   - float64
   - cuda:1
6. Full-batch training for 50,000 epochs. K grows 1 -> 5 every 10,000 epochs.
7. No early stopping. Validation selects the best checkpoint.
8. Train both:
   - multi_problem: all 60 training problems
   - single_problem_baseline: only time step 0 with 100 points
   Both use the same validation and test sets.

The training objective contains only physical variational energies along the
unrolled learned-optimizer trajectory. Exact solutions, residuals, and energy
gaps are used only for reporting and checkpoint selection.
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

    # Exact y-independent constants retained to match the original script.
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

    This trajectory is used only to define independent optimization problems.
    Learned predictions are never propagated between physical time steps.
    """
    p_n = torch.tensor(
        [*physical.p1_0, *physical.p2_0], dtype=TORCH_DTYPE
    )
    v_n = torch.tensor(
        [*physical.v1_0, *physical.v2_0], dtype=TORCH_DTYPE
    )
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
# 4. Network, normalization, and learned update
# ============================================================


class MLPOptimizer(nn.Module):
    """17 -> 64 -> identity -> 6 learned iterative update."""

    def __init__(self, input_mean: torch.Tensor, input_std: torch.Tensor) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(17, 64),
            nn.Identity(),
            nn.Linear(64, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("input_mean", input_mean.detach().clone().to(TORCH_DTYPE))
        self.register_buffer("input_std", input_std.detach().clone().to(TORCH_DTYPE))

    def build_input(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        *,
        dt: float,
        spring_k: float,
        rest_length: float,
    ) -> torch.Tensor:
        batch_size = y.shape[0]
        scalar_features = torch.tensor(
            [dt, spring_k, rest_length],
            dtype=y.dtype,
            device=y.device,
        ).reshape(1, 3).expand(batch_size, -1)
        return torch.cat([y, q, masses, scalar_features], dim=-1)

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        *,
        dt: float,
        spring_k: float,
        rest_length: float,
    ) -> torch.Tensor:
        network_input = self.build_input(
            y,
            q,
            masses,
            dt=dt,
            spring_k=spring_k,
            rest_length=rest_length,
        )
        normalized = (network_input - self.input_mean) / self.input_std
        return self.net(normalized)


def compute_input_normalizer(
    dataset: DatasetBundle,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(dataset)
    scalars = torch.tensor(
        [physical.dt, physical.spring_k, physical.rest_length],
        dtype=TORCH_DTYPE,
    ).reshape(1, 3).expand(batch_size, -1)
    inputs = torch.cat(
        [dataset.initial_y, dataset.q, dataset.masses, scalars],
        dim=-1,
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
    raw_output = model(
        y,
        dataset.q,
        dataset.masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
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
    selected = sorted(set([0, steps, *[step for step in report_steps if 0 <= step <= steps]]))
    return selected


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
    selected_set = set(selected_steps)

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
            step_record: dict[str, Any] = {}
            for prefix, values in arrays.items():
                step_record[prefix] = _statistics(values[mask, step])
            problem_record["steps"][str(step)] = step_record
        per_problem[str(problem_index)] = problem_record
    result["per_problem"] = per_problem
    return result


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
    p1 = np.asarray([tensor_to_list(problem.p_n[0:3]) for problem in problems], dtype=float)
    p2 = np.asarray([tensor_to_list(problem.p_n[3:6]) for problem in problems], dtype=float)
    radii = np.asarray([problem.sampling_radius for problem in problems], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for coordinate, label in enumerate(["x", "y", "z"]):
        axes[0, 0].plot(times, p1[:, coordinate], label=f"particle 1 {label}")
        axes[0, 1].plot(times, p2[:, coordinate], label=f"particle 2 {label}")
    axes[0, 0].set_title("Reference current state: particle 1")
    axes[0, 1].set_title("Reference current state: particle 2")
    axes[0, 0].set_xlabel("Physical time")
    axes[0, 1].set_xlabel("Physical time")
    axes[0, 0].legend()
    axes[0, 1].legend()

    axes[1, 0].plot(times, radii)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Per-problem sampling radius")
    axes[1, 0].set_xlabel("Physical time")
    axes[1, 0].set_ylabel(r"$\|p^n-y_n^*\|_\infty$")

    split_rows = [
        (split.train_indices, 0, "train"),
        (split.validation_indices, 1, "validation"),
        (split.interpolation_test_indices, 2, "interpolation test"),
        (split.extrapolation_test_indices, 3, "extrapolation test"),
    ]
    for indices, level, label in split_rows:
        axes[1, 1].scatter(indices, [level] * len(indices), label=label)
    axes[1, 1].set_yticks([0, 1, 2, 3])
    axes[1, 1].set_yticklabels([row[2] for row in split_rows])
    axes[1, 1].set_xlabel("Physical problem index")
    axes[1, 1].set_title("Problem-level train/validation/test split")
    axes[1, 1].grid(True, axis="x", alpha=0.25)

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

    input_mean, input_std = compute_input_normalizer(training_cpu, physical)
    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)
    model = MLPOptimizer(input_mean=input_mean, input_std=input_std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training = training_cpu.to(device)

    exact_energy_mean = float(
        variational_energy(
            training.exact_y,
            training.q,
            training.masses,
            g=physical.g,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        ).mean().item()
    )

    print("\n" + "=" * 100)
    print(f"Experiment: {experiment_name}")
    print(
        f"device={device}, dtype={TORCH_DTYPE}, architecture=17->64->identity->6, "
        f"optimizer=Adam(lr={LEARNING_RATE:.0e}), output_scale={OUTPUT_SCALE:g}"
    )
    print(
        f"training_points={len(training_cpu):,}, "
        f"training_problems={training_cpu.metadata['num_problems']}, "
        f"validation_points={len(validation_cpu):,}"
    )
    print("no_early_stopping=True; validation_selects_best_checkpoint_only")
    print("=" * 100)

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

    for epoch_index in range(config.epochs):
        epoch_number = epoch_index + 1
        rollout_k = get_k_for_epoch(epoch_index, config)
        model.train()
        y = training.initial_y
        optimizer.zero_grad(set_to_none=True)
        trajectory_loss = torch.zeros((), dtype=TORCH_DTYPE, device=device)

        for _ in range(rollout_k):
            y, _ = apply_model_update(model, y, training, physical)
            trajectory_loss = trajectory_loss + variational_energy(
                y,
                training.q,
                training.masses,
                g=physical.g,
                dt=physical.dt,
                spring_k=physical.spring_k,
                rest_length=physical.rest_length,
            ).mean()

        if not bool(torch.isfinite(trajectory_loss)):
            diverged = True
            divergence_epoch = epoch_number
            divergence_reason = "non-finite full-batch trajectory loss"
        else:
            try:
                trajectory_loss.backward()
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
                f"Training stopped at epoch={divergence_epoch}: {divergence_reason}"
            )
            break

        loss_value = float(trajectory_loss.item())
        training_gap = loss_value - rollout_k * exact_energy_mean
        train_log.append(
            {
                "epoch": epoch_number,
                "K": rollout_k,
                "trajectory_energy_sum": loss_value,
                "training_gap_for_readability": training_gap,
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
                f"train_gap={training_gap:.4e} | "
                f"val_res_p95={validation_metrics['final_residual_p95']:.4e} | "
                f"worst_problem_res_p95={worst_problem_p95:.4e} | "
                f"best_epoch={best_epoch} | elapsed={elapsed:.1f}s"
            )

    last_state_dict = state_dict_to_cpu(model)
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(last_state_dict)
        best_epoch = train_log[-1]["epoch"] if train_log else 0

    torch.save(last_state_dict, experiment_dir / "last_model_state_dict.pt")
    torch.save(best_state_dict, experiment_dir / "best_validation_model_state_dict.pt")
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
            "architecture": "17 -> 64 -> identity -> 6",
            "activation": ACTIVATION_NAME,
            "optimizer": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
            "output_scale": OUTPUT_SCALE,
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
            "no_early_stopping": True,
            "checkpoint_selection": (
                "lexicographic: final residual nonfinite count, pooled residual p95, "
                "pooled exact-error p95, pooled energy-gap p95"
            ),
            "loss": "sum of stepwise mean physical variational energy",
            "backpropagation": "full unroll without detach; one backward per epoch",
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
            "selection_key": list(best_key) if best_key is not None else None,
            "validation_metrics": best_validation_metrics,
        },
        "train_log": train_log,
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
            save_path=experiment_dir / "training_and_validation_curves.png",
        )
        for split_name in ["interpolation_test", "extrapolation_test"]:
            plot_rollout_metrics(
                metrics=best_results[split_name],
                title=f"{experiment_name}: {split_name}",
                save_path=experiment_dir / f"{split_name}_rollout_metrics.png",
            )
            problem_indices = evaluation_datasets[split_name].metadata["problem_indices"]
            plot_metric_vs_physical_time(
                metrics=best_results[split_name],
                problems=problems,
                problem_indices=problem_indices,
                report_steps=config.report_steps,
                title=f"{experiment_name}: {split_name} by physical time",
                save_path=experiment_dir / f"{split_name}_metrics_vs_physical_time.png",
            )

        all_test_indices = evaluation_datasets["current_state_all_test"].metadata[
            "problem_indices"
        ]
        plot_special_state_vs_time(
            current_metrics=best_results["current_state_all_test"],
            exact_metrics=best_results["exact_state_all_test"],
            problems=problems,
            problem_indices=all_test_indices,
            report_steps=config.report_steps,
            title=f"{experiment_name}: physical-current and exact fixed-point tests",
            save_path=experiment_dir / "special_state_metrics_vs_physical_time.png",
        )

    summary = {
        "experiment_name": experiment_name,
        "training_num_problems": training_cpu.metadata["num_problems"],
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
# 8. Command-line interface and main experiment orchestration
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare independent multi-time-step and single-problem "
            "learned optimizers for the two-particle spring problem."
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

    print(f"Output directory: {output_dir}")
    print(f"Runtime config: {asdict(config)}")
    print(f"Physical config: {asdict(physical)}")
    print(f"torch default dtype: {torch.get_default_dtype()}")
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
                "architecture": "17 -> 64 -> identity -> 6",
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "output_scale": OUTPUT_SCALE,
                "torch_dtype": str(TORCH_DTYPE),
                "default_device": DEFAULT_DEVICE,
            },
            "physical_config": asdict(physical),
            "problem_split": asdict(split),
        },
        output_dir / "runtime_config.json",
    )
    save_json(
        {
            "description": (
                "Analytic reference sequence used only to define independent "
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

    dataset_metadata = {
        "split_unit": "physical_time_step_problem",
        "multi_problem_training": multi_training.metadata,
        "single_problem_training": single_training.metadata,
        "validation": validation.metadata,
        "interpolation_test": interpolation_test.metadata,
        "extrapolation_test": extrapolation_test.metadata,
        "current_state_all_test": current_state_all_test.metadata,
        "exact_state_all_test": exact_state_all_test.metadata,
        "normalization": "computed separately from each model's training dataset only",
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
        "dataset_metadata": dataset_metadata,
        "network": {
            "architecture": "17 -> 64 -> identity -> 6",
            "input_features": "y(6), q(6), masses(2), dt, spring_k, rest_length",
            "input_normalization": "per-feature, training data only",
            "final_layer_zero_initialized": True,
            "torch_dtype": str(TORCH_DTYPE),
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "output_scaling": "none",
            "loss": "sum of stepwise mean physical variational energies",
            "full_batch": True,
            "no_early_stopping": True,
        },
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


if __name__ == "__main__":
    main()
