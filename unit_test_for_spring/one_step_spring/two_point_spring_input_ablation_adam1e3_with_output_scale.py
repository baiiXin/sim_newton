"""
Two-particle single-spring input-representation ablation.

Fixed training configuration
----------------------------
- Optimizer: Adam(lr=1e-3)
- Activation: Identity (the two Linear layers form an affine map)
- Hidden width: 64
- Input normalization: enabled, computed independently for each input mode
- Output scaling: enabled in this file; applied update is dt * network_output
- Training-set size: 100 after exact particle-swap augmentation
- Numerical precision: torch.float64
- Default device: cuda:0
- Epochs: 5,000
- Unroll schedule: K=1..5, increased every 1,000 epochs
- Full-batch training, one backward pass per epoch, no detach inside rollout
- Validation chooses the best checkpoint; no early stopping

Input modes
-----------
1. current_only:            [y1, y2]                                      (6)
2. current_and_predictor:   [y1, y2, q1, q2]                              (12)
3. predictor_full:          [y1, y2, q1, q2, m1, m2, dt, k, l0]          (17)
4. raw_full:                [y1, y2, p1n, p2n, v1n, v2n, m1, m2,
                              g, dt, k, l0]                                (24)
5. local_physics:           [y1-q1, y2-q2, y2-y1, ||y2-y1||-l0,
                              k*dt^2/m1, k*dt^2/m2]                        (12)

The training objective contains only the sum of mean variational energies over
all unrolled steps. Residual, energy gap, and exact-solution error are used only
for validation, testing, and visualization.
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
# 0. Fixed experiment configuration
# ============================================================

SCRIPT_VARIANT = "with_output_scale"
OUTPUT_SCALE_MODE = "dt"
DEFAULT_DEVICE = "cuda:0"

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

PLOT_FLOOR = 1e-14
MODEL_RANDOM_SEED = 42
TRAIN_SOBOL_SEED = 20260620
VALIDATION_SOBOL_SEED = 20260621
TEST_SOBOL_SEED = 20260622
SAMPLE_DISTANCE_EPS = 1e-8
RESIDUAL_DISTANCE_EPS = 1e-12
DEFAULT_SUMMARY_CURVE_POINTS = 1_000
MAX_SCATTER_POINTS = 4_000

DEFAULT_TRAIN_SIZE = 100
DEFAULT_EPOCHS = 5_000
DEFAULT_VALIDATION_INTERVAL = 100
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8_192
DEFAULT_VALIDATION_SIZE = 2_048
DEFAULT_TEST_SIZE = 8_192
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 1_000
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5

OPTIMIZER_NAME = "adam"
LEARNING_RATE = 1e-3
ACTIVATION_NAME = "identity"
HIDDEN_WIDTH = 64
USE_INPUT_NORMALIZATION = True

INPUT_MODES = [
    "current_only",
    "current_and_predictor",
    "predictor_full",
    "raw_full",
    "local_physics",
]
INPUT_DIMS = {
    "current_only": 6,
    "current_and_predictor": 12,
    "predictor_full": 17,
    "raw_full": 24,
    "local_physics": 12,
}


# ============================================================
# 1. Data structures and utilities
# ============================================================


@dataclass(frozen=True)
class RuntimeConfig:
    input_modes: list[str]
    train_size: int
    epochs: int
    validation_interval: int
    evaluation_steps: int
    evaluation_batch_size: int
    validation_size: int
    test_size: int
    initial_k: int
    k_increase_interval: int
    k_increase_amount: int
    max_k: int
    device: str
    skip_individual_plots: bool
    skip_dataset_plot: bool


@dataclass(frozen=True)
class PhysicalProblem:
    m1: float
    m2: float
    g: float
    dt: float
    spring_k: float
    rest_length: float
    p1_n: tuple[float, float, float]
    p2_n: tuple[float, float, float]
    v1_n: tuple[float, float, float]
    v2_n: tuple[float, float, float]


@dataclass
class DatasetBundle:
    initial_y: torch.Tensor
    q: torch.Tensor
    p_n: torch.Tensor
    v_n: torch.Tensor
    masses: torch.Tensor
    exact_y: torch.Tensor
    metadata: dict[str, Any]

    def to(self, device: torch.device) -> "DatasetBundle":
        return DatasetBundle(
            initial_y=self.initial_y.to(device=device, dtype=TORCH_DTYPE),
            q=self.q.to(device=device, dtype=TORCH_DTYPE),
            p_n=self.p_n.to(device=device, dtype=TORCH_DTYPE),
            v_n=self.v_n.to(device=device, dtype=TORCH_DTYPE),
            masses=self.masses.to(device=device, dtype=TORCH_DTYPE),
            exact_y=self.exact_y.to(device=device, dtype=TORCH_DTYPE),
            metadata=copy.deepcopy(self.metadata),
        )

    def slice(self, start: int, end: int) -> "DatasetBundle":
        return DatasetBundle(
            initial_y=self.initial_y[start:end],
            q=self.q[start:end],
            p_n=self.p_n[start:end],
            v_n=self.v_n[start:end],
            masses=self.masses[start:end],
            exact_y=self.exact_y[start:end],
            metadata={},
        )


@dataclass(frozen=True)
class ProblemTensors:
    p1_n: torch.Tensor
    p2_n: torch.Tensor
    v1_n: torch.Tensor
    v2_n: torch.Tensor
    q1: torch.Tensor
    q2: torch.Tensor
    y_n: torch.Tensor
    y_star: torch.Tensor
    masses: torch.Tensor
    sampling_radius: float
    lambda_value: float
    exact_energy: float
    exact_residual: float


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


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
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
    result = float(value)
    if not math.isfinite(result):
        return float("nan")
    return max(result, PLOT_FLOOR)


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
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) exist."
            )


def get_k_for_epoch(epoch_index: int, config: RuntimeConfig) -> int:
    return min(
        config.initial_k
        + (epoch_index // config.k_increase_interval) * config.k_increase_amount,
        config.max_k,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


# ============================================================
# 2. Physical problem and analytic solution
# ============================================================


def default_physical_problem() -> PhysicalProblem:
    return PhysicalProblem(
        m1=1.0,
        m2=1.0,
        g=9.8,
        dt=0.01,
        spring_k=2_500.0,
        rest_length=1.0,
        p1_n=(-0.6, 0.0, 1.0),
        p2_n=(0.6, 0.2, 1.1),
        v1_n=(0.2, 0.0, 0.1),
        v2_n=(-0.1, 0.15, -0.05),
    )


def swap_particles(values: torch.Tensor) -> torch.Tensor:
    return torch.cat([values[..., 3:6], values[..., 0:3]], dim=-1)


def swap_masses(values: torch.Tensor) -> torch.Tensor:
    return torch.cat([values[..., 1:2], values[..., 0:1]], dim=-1)


def exact_solution_from_predictors(
    q: torch.Tensor,
    masses: torch.Tensor,
    dt: float,
    spring_k: float,
    rest_length: float,
) -> torch.Tensor:
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

    # The constant makes this algebraically equal to the original p/v/gravity form.
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


def build_problem_tensors(problem: PhysicalProblem) -> ProblemTensors:
    p1_n = torch.tensor(problem.p1_n, dtype=TORCH_DTYPE)
    p2_n = torch.tensor(problem.p2_n, dtype=TORCH_DTYPE)
    v1_n = torch.tensor(problem.v1_n, dtype=TORCH_DTYPE)
    v2_n = torch.tensor(problem.v2_n, dtype=TORCH_DTYPE)
    gravity_vector = torch.tensor([0.0, 0.0, problem.g], dtype=TORCH_DTYPE)
    q1 = p1_n + problem.dt * v1_n - problem.dt**2 * gravity_vector
    q2 = p2_n + problem.dt * v2_n - problem.dt**2 * gravity_vector
    q = torch.cat([q1, q2])
    masses = torch.tensor([problem.m1, problem.m2], dtype=TORCH_DTYPE)
    y_n = torch.cat([p1_n, p2_n])
    y_star = exact_solution_from_predictors(
        q.unsqueeze(0),
        masses.unsqueeze(0),
        problem.dt,
        problem.spring_k,
        problem.rest_length,
    ).squeeze(0)
    sampling_radius = float(torch.max(torch.abs(y_n - y_star)).item())
    reduced_mass = problem.m1 * problem.m2 / (problem.m1 + problem.m2)
    lambda_value = problem.spring_k * problem.dt**2 / reduced_mass
    exact_energy = float(
        variational_energy(
            y_star.unsqueeze(0),
            q.unsqueeze(0),
            masses.unsqueeze(0),
            g=problem.g,
            dt=problem.dt,
            spring_k=problem.spring_k,
            rest_length=problem.rest_length,
        ).item()
    )
    exact_residual = float(
        stationarity_residual_norm(
            y_star.unsqueeze(0),
            q.unsqueeze(0),
            masses.unsqueeze(0),
            dt=problem.dt,
            spring_k=problem.spring_k,
            rest_length=problem.rest_length,
        ).item()
    )
    return ProblemTensors(
        p1_n=p1_n,
        p2_n=p2_n,
        v1_n=v1_n,
        v2_n=v2_n,
        q1=q1,
        q2=q2,
        y_n=y_n,
        y_star=y_star,
        masses=masses,
        sampling_radius=sampling_radius,
        lambda_value=lambda_value,
        exact_energy=exact_energy,
        exact_residual=exact_residual,
    )


# ============================================================
# 3. Sobol datasets and particle-swap augmentation
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
    for point in explicit_points:
        point_cpu = point.detach().cpu().to(TORCH_DTYPE).reshape(1, 6)
        if not bool(nondegenerate_mask(point_cpu)[0]):
            raise ValueError("An explicit point is degenerate.")
        chunks.append(point_cpu)
    if len(chunks) > count:
        raise ValueError("More explicit points than requested canonical samples.")

    engine = torch.quasirandom.SobolEngine(dimension=6, scramble=True, seed=seed)
    accepted = len(chunks)
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

    points = torch.cat(chunks, dim=0)[:count].contiguous()
    return points, {
        "mode": "scrambled_sobol_6d_cube",
        "seed": seed,
        "canonical_count": count,
        "center": tensor_to_list(center),
        "radius_linf": radius,
        "explicit_point_count": len(explicit_points),
        "generated_candidates": generated,
        "rejected_degenerate_candidates": rejected,
    }


def build_augmented_dataset(
    *,
    canonical_points: torch.Tensor,
    problem_tensors: ProblemTensors,
    role: str,
    source_metadata: dict[str, Any],
) -> DatasetBundle:
    canonical_points = canonical_points.detach().cpu().to(TORCH_DTYPE)
    swapped_points = swap_particles(canonical_points)
    initial_y = torch.cat([canonical_points, swapped_points], dim=0)

    q_canonical = torch.cat([problem_tensors.q1, problem_tensors.q2]).reshape(1, 6)
    p_canonical = torch.cat([problem_tensors.p1_n, problem_tensors.p2_n]).reshape(1, 6)
    v_canonical = torch.cat([problem_tensors.v1_n, problem_tensors.v2_n]).reshape(1, 6)
    masses_canonical = problem_tensors.masses.reshape(1, 2)
    exact_canonical = problem_tensors.y_star.reshape(1, 6)

    base_count = canonical_points.shape[0]

    def expanded_pair(base: torch.Tensor, swap_fn) -> torch.Tensor:
        swapped = swap_fn(base)
        return torch.cat(
            [base.expand(base_count, -1), swapped.expand(base_count, -1)], dim=0
        ).clone()

    return DatasetBundle(
        initial_y=initial_y,
        q=expanded_pair(q_canonical, swap_particles),
        p_n=expanded_pair(p_canonical, swap_particles),
        v_n=expanded_pair(v_canonical, swap_particles),
        masses=expanded_pair(masses_canonical, swap_masses),
        exact_y=expanded_pair(exact_canonical, swap_particles),
        metadata={
            "role": role,
            "final_size": int(initial_y.shape[0]),
            "canonical_size": int(base_count),
            "swapped_size": int(base_count),
            "swap_augmented": True,
            "source": copy.deepcopy(source_metadata),
        },
    )


def build_special_dataset(
    *,
    initial_y: torch.Tensor,
    problem_tensors: ProblemTensors,
    role: str,
) -> DatasetBundle:
    return DatasetBundle(
        initial_y=initial_y.reshape(1, 6).detach().cpu().to(TORCH_DTYPE),
        q=torch.cat([problem_tensors.q1, problem_tensors.q2]).reshape(1, 6),
        p_n=torch.cat([problem_tensors.p1_n, problem_tensors.p2_n]).reshape(1, 6),
        v_n=torch.cat([problem_tensors.v1_n, problem_tensors.v2_n]).reshape(1, 6),
        masses=problem_tensors.masses.reshape(1, 2),
        exact_y=problem_tensors.y_star.reshape(1, 6),
        metadata={"role": role, "final_size": 1, "swap_augmented": False},
    )


# ============================================================
# 4. Input representations and network
# ============================================================


def build_input_features(
    *,
    input_mode: str,
    y: torch.Tensor,
    dataset: DatasetBundle,
    problem: PhysicalProblem,
) -> torch.Tensor:
    if y.ndim != 2 or y.shape[1] != 6:
        raise ValueError(f"Expected y shape [B,6], got {tuple(y.shape)}")
    batch_size = y.shape[0]

    if input_mode == "current_only":
        features = y
    elif input_mode == "current_and_predictor":
        features = torch.cat([y, dataset.q], dim=-1)
    elif input_mode == "predictor_full":
        scalars = torch.tensor(
            [problem.dt, problem.spring_k, problem.rest_length],
            dtype=y.dtype,
            device=y.device,
        ).reshape(1, 3).expand(batch_size, -1)
        features = torch.cat([y, dataset.q, dataset.masses, scalars], dim=-1)
    elif input_mode == "raw_full":
        scalars = torch.tensor(
            [problem.g, problem.dt, problem.spring_k, problem.rest_length],
            dtype=y.dtype,
            device=y.device,
        ).reshape(1, 4).expand(batch_size, -1)
        features = torch.cat(
            [y, dataset.p_n, dataset.v_n, dataset.masses, scalars], dim=-1
        )
    elif input_mode == "local_physics":
        y1 = y[:, 0:3]
        y2 = y[:, 3:6]
        q1 = dataset.q[:, 0:3]
        q2 = dataset.q[:, 3:6]
        e1 = y1 - q1
        e2 = y2 - q2
        d = y2 - y1
        length_error = (
            torch.linalg.vector_norm(d, dim=-1, keepdim=True)
            - problem.rest_length
        )
        beta = problem.spring_k * problem.dt**2 / dataset.masses
        features = torch.cat([e1, e2, d, length_error, beta], dim=-1)
    else:
        raise ValueError(f"Unsupported input mode {input_mode!r}")

    expected_dim = INPUT_DIMS[input_mode]
    if features.shape != (batch_size, expected_dim):
        raise RuntimeError(
            f"Input mode {input_mode} produced {tuple(features.shape)}, "
            f"expected {(batch_size, expected_dim)}."
        )
    return features


class MLPOptimizer(nn.Module):
    """Input-dependent affine learned update: D -> 64 -> Identity -> 6."""

    def __init__(
        self,
        *,
        input_mode: str,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
    ) -> None:
        super().__init__()
        if input_mode not in INPUT_DIMS:
            raise ValueError(f"Unsupported input mode {input_mode!r}")
        self.input_mode = input_mode
        input_dim = INPUT_DIMS[input_mode]
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_WIDTH),
            nn.Identity(),
            nn.Linear(HIDDEN_WIDTH, 6),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("input_mean", input_mean.detach().clone().to(TORCH_DTYPE))
        self.register_buffer("input_std", input_std.detach().clone().to(TORCH_DTYPE))

    def forward(
        self,
        y: torch.Tensor,
        dataset: DatasetBundle,
        problem: PhysicalProblem,
    ) -> torch.Tensor:
        features = build_input_features(
            input_mode=self.input_mode,
            y=y,
            dataset=dataset,
            problem=problem,
        )
        normalized = (features - self.input_mean) / self.input_std
        return self.net(normalized)


def compute_input_normalizer(
    *,
    input_mode: str,
    dataset: DatasetBundle,
    problem: PhysicalProblem,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = build_input_features(
        input_mode=input_mode,
        y=dataset.initial_y,
        dataset=dataset,
        problem=problem,
    )
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False)
    std = torch.where(std > 0.0, std, torch.ones_like(std))
    return mean, std


def apply_model_update(
    *,
    model: MLPOptimizer,
    y: torch.Tensor,
    dataset: DatasetBundle,
    problem: PhysicalProblem,
    output_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_output = model(y, dataset, problem)
    applied_delta = output_scale * raw_output
    return y + applied_delta, raw_output, applied_delta


# ============================================================
# 5. Evaluation
# ============================================================


def _statistics_by_step(values: np.ndarray, prefix: str) -> dict[str, Any]:
    mean_by_step: list[float] = []
    median_by_step: list[float] = []
    p95_by_step: list[float] = []
    max_by_step: list[float] = []
    nonfinite_by_step: list[int] = []

    for step_index in range(values.shape[1]):
        column = values[:, step_index]
        finite = column[np.isfinite(column)]
        nonfinite_by_step.append(int(column.size - finite.size))
        if finite.size == 0:
            mean_by_step.append(float("nan"))
            median_by_step.append(float("nan"))
            p95_by_step.append(float("nan"))
            max_by_step.append(float("nan"))
        else:
            mean_by_step.append(float(np.mean(finite)))
            median_by_step.append(float(np.median(finite)))
            p95_by_step.append(float(np.percentile(finite, 95)))
            max_by_step.append(float(np.max(finite)))

    final_values = values[:, -1]
    final_finite = final_values[np.isfinite(final_values)]
    result: dict[str, Any] = {
        f"{prefix}_mean_by_step": mean_by_step,
        f"{prefix}_median_by_step": median_by_step,
        f"{prefix}_p95_by_step": p95_by_step,
        f"{prefix}_max_by_step": max_by_step,
        f"{prefix}_num_nonfinite_by_step": nonfinite_by_step,
        f"final_{prefix}_num_nonfinite": int(final_values.size - final_finite.size),
    }
    for stat_name, function in [
        ("mean", np.mean),
        ("median", np.median),
        ("p95", lambda x: np.percentile(x, 95)),
        ("max", np.max),
    ]:
        result[f"final_{prefix}_{stat_name}"] = (
            float(function(final_finite)) if final_finite.size else float("nan")
        )
    return result


@torch.no_grad()
def evaluate_model_on_dataset(
    *,
    model: MLPOptimizer,
    dataset_cpu: DatasetBundle,
    problem: PhysicalProblem,
    output_scale: float,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    residual_batches: list[torch.Tensor] = []
    gap_batches: list[torch.Tensor] = []
    exact_error_batches: list[torch.Tensor] = []
    num_points = int(dataset_cpu.initial_y.shape[0])

    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        batch = dataset_cpu.slice(start, end).to(device)
        y = batch.initial_y.clone()
        exact_energy = variational_energy(
            batch.exact_y,
            batch.q,
            batch.masses,
            g=problem.g,
            dt=problem.dt,
            spring_k=problem.spring_k,
            rest_length=problem.rest_length,
        )
        residual_steps: list[torch.Tensor] = []
        gap_steps: list[torch.Tensor] = []
        exact_error_steps: list[torch.Tensor] = []

        for step in range(steps + 1):
            residual_steps.append(
                stationarity_residual_norm(
                    y,
                    batch.q,
                    batch.masses,
                    dt=problem.dt,
                    spring_k=problem.spring_k,
                    rest_length=problem.rest_length,
                ).detach().cpu()
            )
            energy = variational_energy(
                y,
                batch.q,
                batch.masses,
                g=problem.g,
                dt=problem.dt,
                spring_k=problem.spring_k,
                rest_length=problem.rest_length,
            )
            gap_steps.append((energy - exact_energy).detach().cpu())
            exact_error_steps.append(
                torch.linalg.vector_norm(y - batch.exact_y, dim=-1).detach().cpu()
            )
            if step < steps:
                y, _, _ = apply_model_update(
                    model=model,
                    y=y,
                    dataset=batch,
                    problem=problem,
                    output_scale=output_scale,
                )

        residual_batches.append(torch.stack(residual_steps, dim=1))
        gap_batches.append(torch.stack(gap_steps, dim=1))
        exact_error_batches.append(torch.stack(exact_error_steps, dim=1))

    arrays = {
        "residual": torch.cat(residual_batches, dim=0).numpy().astype(float),
        "energy_gap": torch.cat(gap_batches, dim=0).numpy().astype(float),
        "exact_error": torch.cat(exact_error_batches, dim=0).numpy().astype(float),
    }
    result: dict[str, Any] = {"steps": steps, "num_points": num_points}
    for prefix, values in arrays.items():
        values[~np.isfinite(values)] = np.nan
        result.update(_statistics_by_step(values, prefix))
        if num_points == 1:
            result[f"single_point_{prefix}_by_step"] = [
                float(value) if math.isfinite(float(value)) else None
                for value in values[0].tolist()
            ]
            result[f"single_point_final_{prefix}"] = result[
                f"single_point_{prefix}_by_step"
            ][-1]
    return result


@torch.no_grad()
def evaluate_single_trajectory(
    *,
    model: MLPOptimizer,
    dataset_cpu: DatasetBundle,
    problem: PhysicalProblem,
    output_scale: float,
    steps: int,
    device: torch.device,
) -> dict[str, Any]:
    if dataset_cpu.initial_y.shape[0] != 1:
        raise ValueError("evaluate_single_trajectory requires exactly one initial point.")
    model.eval()
    batch = dataset_cpu.to(device)
    y = batch.initial_y.clone()
    exact_energy = variational_energy(
        batch.exact_y,
        batch.q,
        batch.masses,
        g=problem.g,
        dt=problem.dt,
        spring_k=problem.spring_k,
        rest_length=problem.rest_length,
    )[0]

    iterations: list[dict[str, Any]] = []
    for step in range(steps + 1):
        energy = variational_energy(
            y,
            batch.q,
            batch.masses,
            g=problem.g,
            dt=problem.dt,
            spring_k=problem.spring_k,
            rest_length=problem.rest_length,
        )[0]
        residual = stationarity_residual_norm(
            y,
            batch.q,
            batch.masses,
            dt=problem.dt,
            spring_k=problem.spring_k,
            rest_length=problem.rest_length,
        )[0]
        exact_error = torch.linalg.vector_norm(y[0] - batch.exact_y[0])
        record: dict[str, Any] = {
            "step": step,
            "y": tensor_to_list(y[0]),
            "energy": float(energy.item()),
            "energy_gap": float((energy - exact_energy).item()),
            "residual_norm": float(residual.item()),
            "exact_error": float(exact_error.item()),
        }
        if step < steps:
            next_y, raw_output, applied_delta = apply_model_update(
                model=model,
                y=y,
                dataset=batch,
                problem=problem,
                output_scale=output_scale,
            )
            record["raw_output_norm"] = float(
                torch.linalg.vector_norm(raw_output[0]).item()
            )
            record["applied_delta_norm"] = float(
                torch.linalg.vector_norm(applied_delta[0]).item()
            )
            y = next_y
        iterations.append(record)

    return {
        "role": dataset_cpu.metadata.get("role"),
        "initial_y": tensor_to_list(dataset_cpu.initial_y[0]),
        "iterations": iterations,
    }


def validation_selection_key(metrics: dict[str, Any]) -> tuple[float, ...] | None:
    values = (
        float(metrics["final_residual_num_nonfinite"]),
        float(metrics["final_residual_p95"]),
        float(metrics["final_residual_median"]),
        float(metrics["final_energy_gap_p95"]),
        float(metrics["final_exact_error_p95"]),
    )
    if not all(math.isfinite(value) for value in values):
        return None
    return values


# ============================================================
# 6. Plotting
# ============================================================


def plot_dataset_distribution(
    *,
    training: DatasetBundle,
    validation: DatasetBundle,
    test: DatasetBundle,
    problem_tensors: ProblemTensors,
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(15, 5))
    datasets = [
        (training, "Training"),
        (validation, "Validation"),
        (test, "Test"),
    ]
    for index, (dataset, title) in enumerate(datasets, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        points = dataset.initial_y.detach().cpu().numpy()
        if points.shape[0] > MAX_SCATTER_POINTS:
            selected = np.linspace(0, points.shape[0] - 1, MAX_SCATTER_POINTS).astype(int)
            points = points[selected]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, alpha=0.25, label="particle 1")
        ax.scatter(points[:, 3], points[:, 4], points[:, 5], s=5, alpha=0.25, label="particle 2")
        y_n = problem_tensors.y_n.numpy()
        y_star = problem_tensors.y_star.numpy()
        ax.scatter(y_n[[0, 3]], y_n[[1, 4]], y_n[[2, 5]], marker="x", s=80, label=r"$y^n$")
        ax.scatter(y_star[[0, 3]], y_star[[1, 4]], y_star[[2, 5]], marker="*", s=130, label=r"$y^*$")
        ax.set_title(f"{title} N={dataset.initial_y.shape[0]:,}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=7)
    fig.suptitle("Shared initial-point distributions for every input mode")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_training_and_validation_curves(
    *,
    train_log: Sequence[dict[str, Any]],
    validation_log: Sequence[dict[str, Any]],
    best_epoch: int | None,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    train_epochs = [record["epoch"] for record in train_log]
    train_gap = [finite_plot_value(record["training_gap_for_readability"]) for record in train_log]
    axes[0].plot(train_epochs, train_gap)
    axes[0].set_yscale("log")
    axes[0].set_title("Training trajectory energy-sum gap")

    val_epochs = [record["epoch"] for record in validation_log]
    metrics = [
        ("final_residual_p95", "Validation residual p95"),
        ("final_energy_gap_p95", "Validation energy-gap p95"),
        ("final_exact_error_p95", "Validation exact-error p95"),
    ]
    for ax, (key, title) in zip(axes[1:], metrics):
        ax.plot(
            val_epochs,
            [finite_plot_value(record["metrics"][key]) for record in validation_log],
            marker="o",
            markersize=3,
        )
        ax.set_yscale("log")
        ax.set_title(title)

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_single_rollout(
    *,
    trajectory: dict[str, Any],
    save_path: Path,
) -> None:
    iterations = trajectory["iterations"]
    steps = [item["step"] for item in iterations]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, key, title in [
        (axes[0], "residual_norm", "Residual norm"),
        (axes[1], "energy_gap", "Energy gap"),
        (axes[2], "exact_error", "Exact-solution error"),
    ]:
        ax.plot(steps, [finite_plot_value(item[key]) for item in iterations], marker="o", markersize=3)
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_particle_trajectory_3d(
    *,
    trajectory: dict[str, Any],
    exact_y: torch.Tensor,
    save_path: Path,
) -> None:
    points = np.asarray([item["y"] for item in trajectory["iterations"]], dtype=float)
    exact = exact_y.detach().cpu().numpy()
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(points[:, 0], points[:, 1], points[:, 2], "-o", markersize=3, label="particle 1")
    ax.plot(points[:, 3], points[:, 4], points[:, 5], "-o", markersize=3, label="particle 2")
    ax.scatter(exact[[0, 3]], exact[[1, 4]], exact[[2, 5]], marker="*", s=180, label=r"exact $y^*$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Best-checkpoint rollout from the n-state")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_cross_mode_summary(records: Sequence[dict[str, Any]], save_path: Path) -> None:
    labels = [record["input_mode"] for record in records]
    metrics = [
        ("final_residual_p95", "Test residual p95"),
        ("final_energy_gap_p95", "Test energy-gap p95"),
        ("final_exact_error_p95", "Test exact-error p95"),
        ("num_trainable_parameters", "Trainable parameters"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    for ax, (key, title) in zip(axes, metrics):
        if key == "num_trainable_parameters":
            values = [record[key] for record in records]
            ax.bar(labels, values)
        else:
            values = [finite_plot_value(record["best_checkpoint_test"][key]) for record in records]
            ax.bar(labels, values)
            ax.set_yscale("log")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        f"Input representation ablation | Adam 1e-3 | output scale={OUTPUT_SCALE_MODE}"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_cross_mode_training(records: Sequence[dict[str, Any]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for record in records:
        curve = record["training_curve_for_summary"]
        axes[0].plot(
            [point["epoch"] for point in curve],
            [finite_plot_value(point["training_gap_for_readability"]) for point in curve],
            label=record["input_mode"],
        )
        val_curve = record["validation_curve_for_summary"]
        axes[1].plot(
            [point["epoch"] for point in val_curve],
            [finite_plot_value(point["metrics"]["final_residual_p95"]) for point in val_curve],
            marker="o",
            markersize=2,
            label=record["input_mode"],
        )
    axes[0].set_title("Training energy-sum gap")
    axes[1].set_title("Validation residual p95")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. One input-mode experiment
# ============================================================


def run_experiment(
    *,
    base_output_dir: Path,
    input_mode: str,
    training_cpu: DatasetBundle,
    validation_cpu: DatasetBundle,
    test_cpu: DatasetBundle,
    n_state_cpu: DatasetBundle,
    exact_state_cpu: DatasetBundle,
    problem: PhysicalProblem,
    config: RuntimeConfig,
    device: torch.device,
) -> dict[str, Any]:
    output_dir = base_output_dir / input_mode
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)

    input_mean, input_std = compute_input_normalizer(
        input_mode=input_mode,
        dataset=training_cpu,
        problem=problem,
    )
    model = MLPOptimizer(
        input_mode=input_mode,
        input_mean=input_mean,
        input_std=input_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    output_scale = problem.dt if OUTPUT_SCALE_MODE == "dt" else 1.0
    training = training_cpu.to(device)

    exact_energy_mean = float(
        variational_energy(
            training.exact_y,
            training.q,
            training.masses,
            g=problem.g,
            dt=problem.dt,
            spring_k=problem.spring_k,
            rest_length=problem.rest_length,
        ).mean().item()
    )

    print("\n" + "=" * 88)
    print(f"Input mode: {input_mode} ({INPUT_DIMS[input_mode]} dimensions)")
    print(f"Device={device}, dtype={TORCH_DTYPE}, optimizer=Adam lr={LEARNING_RATE:.0e}")
    print(f"Output scale mode={OUTPUT_SCALE_MODE}, applied scale={output_scale:g}")
    print(f"Training N={training.initial_y.shape[0]}, epochs={config.epochs}")
    print("=" * 88)

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
            y, _, _ = apply_model_update(
                model=model,
                y=y,
                dataset=training,
                problem=problem,
                output_scale=output_scale,
            )
            trajectory_loss = trajectory_loss + variational_energy(
                y,
                training.q,
                training.masses,
                g=problem.g,
                dt=problem.dt,
                spring_k=problem.spring_k,
                rest_length=problem.rest_length,
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
            print(f"Training stopped at epoch {divergence_epoch}: {divergence_reason}")
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
                problem=problem,
                output_scale=output_scale,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                device=device,
            )
            current_key = validation_selection_key(validation_metrics)
            validation_log.append(
                {
                    "epoch": epoch_number,
                    "training_K": rollout_k,
                    "selection_key": list(current_key) if current_key is not None else None,
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
                f"val_gap_p95={validation_metrics['final_energy_gap_p95']:.4e} | "
                f"val_exact_p95={validation_metrics['final_exact_error_p95']:.4e} | "
                f"best_epoch={best_epoch} | elapsed={elapsed:.1f}s"
            )

    last_state_dict = state_dict_to_cpu(model)
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(last_state_dict)
        best_epoch = train_log[-1]["epoch"] if train_log else 0

    torch.save(best_state_dict, output_dir / "best_validation_model_state_dict.pt")
    torch.save(last_state_dict, output_dir / "last_model_state_dict.pt")
    torch.save(best_state_dict, output_dir / "mlp_optimizer_state_dict.pt")

    def evaluate_checkpoint(
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        test_eval = evaluate_model_on_dataset(
            model=model,
            dataset_cpu=test_cpu,
            problem=problem,
            output_scale=output_scale,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            device=device,
        )
        n_state_eval = evaluate_model_on_dataset(
            model=model,
            dataset_cpu=n_state_cpu,
            problem=problem,
            output_scale=output_scale,
            steps=config.evaluation_steps,
            batch_size=1,
            device=device,
        )
        exact_state_eval = evaluate_model_on_dataset(
            model=model,
            dataset_cpu=exact_state_cpu,
            problem=problem,
            output_scale=output_scale,
            steps=config.evaluation_steps,
            batch_size=1,
            device=device,
        )
        n_state_trajectory = evaluate_single_trajectory(
            model=model,
            dataset_cpu=n_state_cpu,
            problem=problem,
            output_scale=output_scale,
            steps=config.evaluation_steps,
            device=device,
        )
        return test_eval, n_state_eval, exact_state_eval, n_state_trajectory

    best_test, best_n_state, best_exact_state, best_trajectory = evaluate_checkpoint(
        best_state_dict
    )
    last_test, last_n_state, last_exact_state, last_trajectory = evaluate_checkpoint(
        last_state_dict
    )

    report = {
        "config": {
            "script_variant": SCRIPT_VARIANT,
            "input_mode": input_mode,
            "input_dim": INPUT_DIMS[input_mode],
            "architecture": f"{INPUT_DIMS[input_mode]} -> {HIDDEN_WIDTH} -> Identity -> 6",
            "num_trainable_parameters": count_trainable_parameters(model),
            "optimizer_name": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
            "output_scale_mode": OUTPUT_SCALE_MODE,
            "output_scale_value": output_scale,
            "use_input_normalization": USE_INPUT_NORMALIZATION,
            "input_mean": tensor_to_list(model.input_mean),
            "input_std": tensor_to_list(model.input_std),
            "torch_dtype": str(TORCH_DTYPE),
            "device": str(device),
            "training_size": int(training.initial_y.shape[0]),
            "epochs_requested": config.epochs,
            "completed_epochs": len(train_log),
            "validation_interval": config.validation_interval,
            "evaluation_steps": config.evaluation_steps,
            "initial_K": config.initial_k,
            "K_increase_interval": config.k_increase_interval,
            "K_increase_amount": config.k_increase_amount,
            "max_K": config.max_k,
            "loss": "sum of stepwise mean variational energy",
            "backpropagation": "full unroll without detach; one backward per epoch",
            "checkpoint_selection": (
                "final validation nonfinite count, residual p95, residual median, "
                "energy-gap p95, exact-error p95"
            ),
            "model_random_seed": MODEL_RANDOM_SEED,
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
        "final_test": {
            "best_validation_checkpoint": {
                "heldout_test": best_test,
                "n_state": best_n_state,
                "exact_state_fixed_point": best_exact_state,
            },
            "last_epoch_checkpoint": {
                "heldout_test": last_test,
                "n_state": last_n_state,
                "exact_state_fixed_point": last_exact_state,
            },
        },
        "n_state_trajectories": {
            "best_validation_checkpoint": best_trajectory,
            "last_epoch_checkpoint": last_trajectory,
        },
    }
    save_json(report, output_dir / "optimization_report.json")

    if not config.skip_individual_plots:
        plot_training_and_validation_curves(
            train_log=train_log,
            validation_log=validation_log,
            best_epoch=best_epoch,
            save_path=output_dir / "training_and_validation_curves.png",
        )
        plot_single_rollout(
            trajectory=best_trajectory,
            save_path=output_dir / "n_state_best_checkpoint_rollout.png",
        )
        plot_particle_trajectory_3d(
            trajectory=best_trajectory,
            exact_y=n_state_cpu.exact_y[0],
            save_path=output_dir / "n_state_particle_trajectory_3d.png",
        )

    summary = {
        "input_mode": input_mode,
        "input_dim": INPUT_DIMS[input_mode],
        "num_trainable_parameters": count_trainable_parameters(model),
        "diverged": diverged,
        "divergence_epoch": divergence_epoch,
        "divergence_reason": divergence_reason,
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": list(best_key) if best_key is not None else None,
        "best_checkpoint_test": {
            "final_residual_mean": best_test["final_residual_mean"],
            "final_residual_median": best_test["final_residual_median"],
            "final_residual_p95": best_test["final_residual_p95"],
            "final_residual_max": best_test["final_residual_max"],
            "final_energy_gap_mean": best_test["final_energy_gap_mean"],
            "final_energy_gap_median": best_test["final_energy_gap_median"],
            "final_energy_gap_p95": best_test["final_energy_gap_p95"],
            "final_energy_gap_max": best_test["final_energy_gap_max"],
            "final_exact_error_mean": best_test["final_exact_error_mean"],
            "final_exact_error_median": best_test["final_exact_error_median"],
            "final_exact_error_p95": best_test["final_exact_error_p95"],
            "final_exact_error_max": best_test["final_exact_error_max"],
            "n_state_final_residual": best_n_state["single_point_final_residual"],
            "n_state_final_energy_gap": best_n_state["single_point_final_energy_gap"],
            "n_state_final_exact_error": best_n_state["single_point_final_exact_error"],
            "exact_state_final_residual": best_exact_state["single_point_final_residual"],
            "exact_state_final_exact_error": best_exact_state["single_point_final_exact_error"],
        },
        "last_checkpoint_test": {
            "final_residual_p95": last_test["final_residual_p95"],
            "final_energy_gap_p95": last_test["final_energy_gap_p95"],
            "final_exact_error_p95": last_test["final_exact_error_p95"],
        },
        "training_curve_for_summary": downsample_log(train_log),
        "validation_curve_for_summary": downsample_log(validation_log),
        "output_directory": str(output_dir),
    }

    print(
        f"Finished {input_mode}: best_epoch={best_epoch}, "
        f"test_residual_p95={best_test['final_residual_p95']:.4e}, "
        f"test_exact_error_p95={best_test['final_exact_error_p95']:.4e}"
    )
    return summary


# ============================================================
# 8. CLI and main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-particle spring input ablation: Adam 1e-3, Identity, "
            f"output scale mode={OUTPUT_SCALE_MODE}."
        )
    )
    parser.add_argument(
        "--input-modes",
        type=str,
        nargs="+",
        default=INPUT_MODES,
        choices=INPUT_MODES,
    )
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--evaluation-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    parser.add_argument("--validation-size", type=int, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument("--k-increase-interval", type=int, default=DEFAULT_K_INCREASE_INTERVAL)
    parser.add_argument("--k-increase-amount", type=int, default=DEFAULT_K_INCREASE_AMOUNT)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--skip-individual-plots", action="store_true")
    parser.add_argument("--skip-dataset-plot", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    input_modes = list(dict.fromkeys(args.input_modes))
    positive_fields = {
        "train_size": args.train_size,
        "epochs": args.epochs,
        "validation_interval": args.validation_interval,
        "evaluation_steps": args.evaluation_steps,
        "evaluation_batch_size": args.evaluation_batch_size,
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "initial_k": args.initial_k,
        "k_increase_interval": args.k_increase_interval,
        "k_increase_amount": args.k_increase_amount,
        "max_k": args.max_k,
    }
    for name, value in positive_fields.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")
    for name in ["train_size", "validation_size", "test_size"]:
        if int(getattr(args, name)) % 2 != 0:
            raise ValueError(f"{name} must be even for exact swap augmentation.")
    if args.train_size < 4:
        raise ValueError("train_size must be at least 4 to include y_n and y_star before swapping.")
    if args.max_k < args.initial_k:
        raise ValueError("max_k must be >= initial_k.")
    return RuntimeConfig(
        input_modes=input_modes,
        train_size=int(args.train_size),
        epochs=int(args.epochs),
        validation_interval=int(args.validation_interval),
        evaluation_steps=int(args.evaluation_steps),
        evaluation_batch_size=int(args.evaluation_batch_size),
        validation_size=int(args.validation_size),
        test_size=int(args.test_size),
        initial_k=int(args.initial_k),
        k_increase_interval=int(args.k_increase_interval),
        k_increase_amount=int(args.k_increase_amount),
        max_k=int(args.max_k),
        device=str(args.device),
        skip_individual_plots=bool(args.skip_individual_plots),
        skip_dataset_plot=bool(args.skip_dataset_plot),
    )


def main() -> None:
    config = validate_args(parse_args())
    output_dir = create_output_directory()
    device = torch.device(config.device)
    validate_device(device)
    problem = default_physical_problem()
    problem_tensors = build_problem_tensors(problem)
    output_scale = problem.dt if OUTPUT_SCALE_MODE == "dt" else 1.0

    print(f"Output directory: {output_dir}")
    print(f"Script variant: {SCRIPT_VARIANT}")
    print(f"Runtime config: {asdict(config)}")
    print(f"torch default dtype: {torch.get_default_dtype()}")
    print(f"Output scale mode: {OUTPUT_SCALE_MODE}, value={output_scale:g}")
    print(f"Exact solution: {tensor_to_list(problem_tensors.y_star)}")
    print(f"Sampling radius (L_inf): {problem_tensors.sampling_radius:.12e}")
    print(f"Exact residual: {problem_tensors.exact_residual:.12e}")

    save_json(
        {
            "script_variant": SCRIPT_VARIANT,
            "output_scale_mode": OUTPUT_SCALE_MODE,
            "default_device": DEFAULT_DEVICE,
            "runtime_config": asdict(config),
            "torch_dtype": str(TORCH_DTYPE),
            "optimizer": {"name": OPTIMIZER_NAME, "learning_rate": LEARNING_RATE},
            "activation": ACTIVATION_NAME,
            "hidden_width": HIDDEN_WIDTH,
            "input_dims": INPUT_DIMS,
        },
        output_dir / "runtime_config.json",
    )
    save_json(
        {
            **asdict(problem),
            "q1": tensor_to_list(problem_tensors.q1),
            "q2": tensor_to_list(problem_tensors.q2),
            "y_n": tensor_to_list(problem_tensors.y_n),
            "y_star": tensor_to_list(problem_tensors.y_star),
            "lambda": problem_tensors.lambda_value,
            "sampling_radius_linf": problem_tensors.sampling_radius,
            "exact_energy": problem_tensors.exact_energy,
            "exact_residual": problem_tensors.exact_residual,
        },
        output_dir / "physical_problem.json",
    )

    canonical_training, train_metadata = generate_canonical_sobol_points(
        count=config.train_size // 2,
        center=problem_tensors.y_star,
        radius=problem_tensors.sampling_radius,
        seed=TRAIN_SOBOL_SEED,
        explicit_points=(problem_tensors.y_n, problem_tensors.y_star),
    )
    canonical_validation, validation_metadata = generate_canonical_sobol_points(
        count=config.validation_size // 2,
        center=problem_tensors.y_star,
        radius=problem_tensors.sampling_radius,
        seed=VALIDATION_SOBOL_SEED,
    )
    canonical_test, test_metadata = generate_canonical_sobol_points(
        count=config.test_size // 2,
        center=problem_tensors.y_star,
        radius=problem_tensors.sampling_radius,
        seed=TEST_SOBOL_SEED,
    )

    training_cpu = build_augmented_dataset(
        canonical_points=canonical_training,
        problem_tensors=problem_tensors,
        role="training",
        source_metadata=train_metadata,
    )
    validation_cpu = build_augmented_dataset(
        canonical_points=canonical_validation,
        problem_tensors=problem_tensors,
        role="validation",
        source_metadata=validation_metadata,
    )
    test_cpu = build_augmented_dataset(
        canonical_points=canonical_test,
        problem_tensors=problem_tensors,
        role="test",
        source_metadata=test_metadata,
    )
    n_state_cpu = build_special_dataset(
        initial_y=problem_tensors.y_n,
        problem_tensors=problem_tensors,
        role="n_state",
    )
    exact_state_cpu = build_special_dataset(
        initial_y=problem_tensors.y_star,
        problem_tensors=problem_tensors,
        role="exact_state",
    )

    save_json(
        {
            "training": training_cpu.metadata,
            "validation": validation_cpu.metadata,
            "test": test_cpu.metadata,
            "all_input_modes_share_exactly_the_same_initial_points": True,
        },
        output_dir / "dataset_metadata.json",
    )

    if not config.skip_dataset_plot:
        plot_dataset_distribution(
            training=training_cpu,
            validation=validation_cpu,
            test=test_cpu,
            problem_tensors=problem_tensors,
            save_path=output_dir / "dataset_distribution_overview.png",
        )

    records: list[dict[str, Any]] = []
    for input_mode in config.input_modes:
        record = run_experiment(
            base_output_dir=output_dir,
            input_mode=input_mode,
            training_cpu=training_cpu,
            validation_cpu=validation_cpu,
            test_cpu=test_cpu,
            n_state_cpu=n_state_cpu,
            exact_state_cpu=exact_state_cpu,
            problem=problem,
            config=config,
            device=device,
        )
        records.append(record)
        save_json(
            {"records": records},
            output_dir / "input_ablation_summary_in_progress.json",
        )

    save_json(
        {
            "script_variant": SCRIPT_VARIANT,
            "output_scale_mode": OUTPUT_SCALE_MODE,
            "records": records,
        },
        output_dir / "input_ablation_summary.json",
    )
    plot_cross_mode_summary(records, output_dir / "final_test_summary.png")
    plot_cross_mode_training(records, output_dir / "training_and_validation_summary.png")

    ranking = sorted(
        records,
        key=lambda record: (
            finite_plot_value(record["best_checkpoint_test"]["final_residual_p95"]),
            finite_plot_value(record["best_checkpoint_test"]["final_exact_error_p95"]),
            finite_plot_value(record["best_checkpoint_test"]["final_energy_gap_p95"]),
        ),
    )
    save_json(
        {
            "ranking_metric": (
                "best-checkpoint held-out test residual p95; tie-break by exact-error "
                "p95 and energy-gap p95"
            ),
            "ranking": [
                {
                    "rank": index,
                    "input_mode": record["input_mode"],
                    "input_dim": record["input_dim"],
                    "num_trainable_parameters": record["num_trainable_parameters"],
                    "best_validation_epoch": record["best_validation_epoch"],
                    "test_residual_p95": record["best_checkpoint_test"]["final_residual_p95"],
                    "test_energy_gap_p95": record["best_checkpoint_test"]["final_energy_gap_p95"],
                    "test_exact_error_p95": record["best_checkpoint_test"]["final_exact_error_p95"],
                }
                for index, record in enumerate(ranking, start=1)
            ],
        },
        output_dir / "input_mode_ranking.json",
    )
    print(f"\nAll input modes completed. Results: {output_dir}")


if __name__ == "__main__":
    main()
