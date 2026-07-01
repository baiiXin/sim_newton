"""
Two-particle single-spring, single-step learned optimizer activation ablation.

Activations in this single script: identity, relu, silu
Default device: cuda:0
Numerical precision: torch.float64

Experiment matrix:
- Activations: Identity, ReLU, SiLU
- Training sizes: 10, 100, 1_000, 10_000, 100_000
- Optimizer: SGD(lr=1e-2) only
- Learned update: delta_y = dt * network_output
- 50_000 epochs, K=1..5, increased every 10_000 epochs
- Full-batch training
- Validation-selected checkpoint; no early stopping

The training loss contains only the sum of mean physical variational energies
along the unrolled trajectory. Residual, energy gap, and exact-solution error
are evaluation/model-selection metrics only.
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
# 0. Script identity and default experiment parameters
# ============================================================

DEFAULT_ACTIVATION_NAMES = ["identity", "relu", "silu"]
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
MAX_SCATTER_POINTS = 4_000
DEFAULT_SUMMARY_CURVE_POINTS = 1_000

DEFAULT_TARGET_DATASET_SIZES = [10, 100, 1_000, 10_000, 100_000]
DEFAULT_EPOCHS = 50_000
DEFAULT_VALIDATION_INTERVAL = 500
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8_192
DEFAULT_VALIDATION_SIZE = 2_048
DEFAULT_TEST_SIZE = 8_192
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 10_000
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5

ALL_OPTIMIZER_CONFIGS = [
    {
        "optimizer_name": "sgd",
        "learning_rate": 1e-2,
        "output_scale_mode": "dt",
    },
]


# ============================================================
# 1. Runtime configuration and utility functions
# ============================================================


@dataclass(frozen=True)
class RuntimeConfig:
    target_dataset_sizes: list[int]
    activation_names: list[str]
    optimizer_configs: list[dict[str, Any]]
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
    skip_contour: bool
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
    masses: torch.Tensor
    exact_y: torch.Tensor
    metadata: dict[str, Any]

    def to(self, device: torch.device) -> "DatasetBundle":
        return DatasetBundle(
            initial_y=self.initial_y.to(device=device, dtype=TORCH_DTYPE),
            q=self.q.to(device=device, dtype=TORCH_DTYPE),
            masses=self.masses.to(device=device, dtype=TORCH_DTYPE),
            exact_y=self.exact_y.to(device=device, dtype=TORCH_DTYPE),
            metadata=copy.deepcopy(self.metadata),
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


def ensure_even_positive_sizes(values: Iterable[int]) -> list[int]:
    cleaned = sorted({int(value) for value in values})
    if not cleaned:
        raise ValueError("target_dataset_sizes must not be empty.")
    if cleaned[0] <= 0:
        raise ValueError("Every target dataset size must be positive.")
    if any(value % 2 != 0 for value in cleaned):
        raise ValueError("Every dataset size must be even for exact swap augmentation.")
    if cleaned[0] < 4:
        raise ValueError("Every dataset size must be at least 4.")
    return cleaned


def get_k_for_epoch(epoch_index: int, config: RuntimeConfig) -> int:
    return min(
        config.initial_k
        + (epoch_index // config.k_increase_interval) * config.k_increase_amount,
        config.max_k,
    )


def downsample_log(
    records: Sequence[dict[str, Any]],
    max_points: int = DEFAULT_SUMMARY_CURVE_POINTS,
) -> list[dict[str, Any]]:
    if not records:
        return []
    if len(records) <= max_points:
        return copy.deepcopy(list(records))
    indices = np.linspace(0, len(records) - 1, num=max_points, dtype=int)
    indices = sorted(set(indices.tolist() + [len(records) - 1]))
    return [copy.deepcopy(records[index]) for index in indices]


def parse_activation_names(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(DEFAULT_ACTIVATION_NAMES)
    parsed: list[str] = []
    allowed = set(DEFAULT_ACTIVATION_NAMES)
    for raw in values:
        name = raw.strip().lower()
        if name not in allowed:
            raise ValueError(
                f"Unsupported activation {raw!r}. "
                f"Available: {', '.join(DEFAULT_ACTIVATION_NAMES)}."
            )
        if name not in parsed:
            parsed.append(name)
    if not parsed:
        raise ValueError("activation_names must not be empty.")
    return parsed


def parse_optimizer_configs(values: Sequence[str] | None) -> list[dict[str, Any]]:
    if not values:
        return copy.deepcopy(ALL_OPTIMIZER_CONFIGS)
    for raw in values:
        try:
            name_raw, lr_raw = raw.split(":", maxsplit=1)
            name = name_raw.strip().lower()
            learning_rate = float(lr_raw)
        except Exception as error:
            raise ValueError(
                f"Invalid optimizer config {raw!r}. Use sgd:1e-2."
            ) from error
        if name != "sgd" or not math.isclose(learning_rate, 1e-2, rel_tol=1e-12):
            raise ValueError(
                "This script supports only SGD(lr=1e-2) with dt output scaling."
            )
    return copy.deepcopy(ALL_OPTIMIZER_CONFIGS)


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


# ============================================================
# 2. Physical problem, exact solution, energy, and residual
# ============================================================


def default_physical_problem() -> PhysicalProblem:
    return PhysicalProblem(
        m1=1.0,
        m2=1.0,
        g=9.8,
        dt=0.01,
        spring_k=2500.0,
        rest_length=1.0,
        p1_n=(-0.6, 0.0, 1.0),
        p2_n=(0.6, 0.2, 1.1),
        v1_n=(0.2, 0.0, 0.1),
        v2_n=(-0.1, 0.15, -0.05),
    )


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
    """Analytic minimizer for one spring with two free particles."""
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
    """
    Original single-step variational energy, expressed using q plus its exact
    y-independent constant. This is algebraically equal to the p/v/gravity form.
    """
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
# 3. Sobol datasets and swap augmentation
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

    points: list[torch.Tensor] = []
    for point in explicit_points:
        point_cpu = point.detach().cpu().to(TORCH_DTYPE).reshape(1, 6)
        if not bool(nondegenerate_mask(point_cpu)[0]):
            raise ValueError("An explicit point is degenerate.")
        points.append(point_cpu)
    if len(points) > count:
        raise ValueError("More explicit points than requested canonical samples.")

    engine = torch.quasirandom.SobolEngine(dimension=6, scramble=True, seed=seed)
    accepted = len(points)
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
            points.append(accepted_candidates)
            accepted += int(accepted_candidates.shape[0])

    result = torch.cat(points, dim=0) if points else torch.empty((0, 6), dtype=TORCH_DTYPE)
    result = result[:count].contiguous()
    return result, {
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
    q_swapped = swap_particles(q_canonical)
    masses_canonical = problem_tensors.masses.reshape(1, 2)
    masses_swapped = swap_masses(masses_canonical)
    exact_canonical = problem_tensors.y_star.reshape(1, 6)
    exact_swapped = swap_particles(exact_canonical)

    base_count = canonical_points.shape[0]
    q = torch.cat(
        [
            q_canonical.expand(base_count, -1),
            q_swapped.expand(base_count, -1),
        ],
        dim=0,
    ).clone()
    masses = torch.cat(
        [
            masses_canonical.expand(base_count, -1),
            masses_swapped.expand(base_count, -1),
        ],
        dim=0,
    ).clone()
    exact_y = torch.cat(
        [
            exact_canonical.expand(base_count, -1),
            exact_swapped.expand(base_count, -1),
        ],
        dim=0,
    ).clone()

    return DatasetBundle(
        initial_y=initial_y,
        q=q,
        masses=masses,
        exact_y=exact_y,
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
        masses=problem_tensors.masses.reshape(1, 2),
        exact_y=problem_tensors.y_star.reshape(1, 6),
        metadata={"role": role, "final_size": 1, "swap_augmented": False},
    )


# ============================================================
# 4. Network and training optimizer
# ============================================================


class MLPOptimizer(nn.Module):
    """17 -> 64 -> selected activation -> 6 learned iterative update."""

    def __init__(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        activation_name: str,
    ) -> None:
        super().__init__()
        self.activation_name = activation_name.lower()
        if self.activation_name == "identity":
            activation: nn.Module = nn.Identity()
        elif self.activation_name == "relu":
            activation = nn.ReLU()
        elif self.activation_name == "silu":
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation {activation_name!r}")

        self.net = nn.Sequential(
            nn.Linear(17, 64),
            activation,
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
        if y.ndim != 2 or y.shape[1] != 6:
            raise ValueError(f"Expected y shape [B,6], got {tuple(y.shape)}")
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
        inp = self.build_input(
            y,
            q,
            masses,
            dt=dt,
            spring_k=spring_k,
            rest_length=rest_length,
        )
        normalized = (inp - self.input_mean) / self.input_std
        return self.net(normalized)


def compute_input_normalizer(
    dataset: DatasetBundle,
    problem: PhysicalProblem,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = dataset.initial_y.shape[0]
    scalars = torch.tensor(
        [problem.dt, problem.spring_k, problem.rest_length],
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


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unsupported optimizer {optimizer_name!r}")


def output_scale_value(mode: str, dt: float) -> float:
    if mode == "dt":
        return dt
    if mode == "raw":
        return 1.0
    raise ValueError(f"Unsupported output scale mode {mode!r}")


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    dataset: DatasetBundle,
    problem: PhysicalProblem,
    output_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_output = model(
        y,
        dataset.q,
        dataset.masses,
        dt=problem.dt,
        spring_k=problem.spring_k,
        rest_length=problem.rest_length,
    )
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
        key = f"final_{prefix}_{stat_name}"
        result[key] = float(function(final_finite)) if final_finite.size else float("nan")
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
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model.eval()
    residual_batches: list[torch.Tensor] = []
    gap_batches: list[torch.Tensor] = []
    exact_error_batches: list[torch.Tensor] = []
    point1_error_batches: list[torch.Tensor] = []
    point2_error_batches: list[torch.Tensor] = []
    num_points = int(dataset_cpu.initial_y.shape[0])

    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        batch = DatasetBundle(
            initial_y=dataset_cpu.initial_y[start:end],
            q=dataset_cpu.q[start:end],
            masses=dataset_cpu.masses[start:end],
            exact_y=dataset_cpu.exact_y[start:end],
            metadata={},
        ).to(device)
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
        point1_error_steps: list[torch.Tensor] = []
        point2_error_steps: list[torch.Tensor] = []

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
                y, _, _ = apply_model_update(
                    model, y, batch, problem, output_scale
                )

        residual_batches.append(torch.stack(residual_steps, dim=1))
        gap_batches.append(torch.stack(gap_steps, dim=1))
        exact_error_batches.append(torch.stack(exact_error_steps, dim=1))
        point1_error_batches.append(torch.stack(point1_error_steps, dim=1))
        point2_error_batches.append(torch.stack(point2_error_steps, dim=1))

    metric_arrays = {
        "residual": torch.cat(residual_batches, dim=0).numpy().astype(float),
        "energy_gap": torch.cat(gap_batches, dim=0).numpy().astype(float),
        "exact_error": torch.cat(exact_error_batches, dim=0).numpy().astype(float),
        "point1_error": torch.cat(point1_error_batches, dim=0).numpy().astype(float),
        "point2_error": torch.cat(point2_error_batches, dim=0).numpy().astype(float),
    }
    result: dict[str, Any] = {"steps": steps, "num_points": num_points}
    for prefix, values in metric_arrays.items():
        values[~np.isfinite(values)] = np.nan
        result.update(_statistics_by_step(values, prefix))

    if num_points == 1:
        for prefix, values in metric_arrays.items():
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
        raise ValueError("evaluate_single_trajectory requires one point.")
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
        record = {
            "step": step,
            "y": tensor_to_list(y[0]),
            "energy": float(energy.item()),
            "energy_gap": float((energy - exact_energy).item()),
            "residual_norm": float(residual.item()),
            "exact_error": float(exact_error.item()),
        }
        if step < steps:
            next_y, raw_output, applied_delta = apply_model_update(
                model, y, batch, problem, output_scale
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
        float(metrics["final_exact_error_p95"]),
        float(metrics["final_energy_gap_p95"]),
    )
    if not all(math.isfinite(value) for value in values):
        return None
    return values


# ============================================================
# 6. Plotting
# ============================================================


def finite_rows(points: np.ndarray, width: int) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, width)
    return points[np.isfinite(points).all(axis=1)]


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    points = finite_rows(points, 3)
    if points.shape[0] == 0:
        points = np.zeros((1, 3), dtype=float)
    center = points.mean(axis=0)
    radius = max(float(np.ptp(points, axis=0).max()) / 2.0, 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def sample_rows(tensor: torch.Tensor, max_points: int = MAX_SCATTER_POINTS) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.shape[0] <= max_points:
        return array
    indices = np.linspace(0, array.shape[0] - 1, max_points, dtype=int)
    return array[indices]


def representation(points: np.ndarray, mode: str) -> np.ndarray:
    y1 = points[:, 0:3]
    y2 = points[:, 3:6]
    if mode == "y1":
        return y1
    if mode == "y2":
        return y2
    if mode == "center":
        return 0.5 * (y1 + y2)
    if mode == "relative":
        return y2 - y1
    raise ValueError(mode)


def plot_dataset_distribution_overview(
    *,
    training: DatasetBundle,
    validation: DatasetBundle,
    test: DatasetBundle,
    y_n: torch.Tensor,
    y_star: torch.Tensor,
    save_path: Path,
) -> None:
    train = sample_rows(training.initial_y)
    val = sample_rows(validation.initial_y)
    test_np = sample_rows(test.initial_y)
    y_n_np = y_n.detach().cpu().numpy().reshape(1, 6)
    y_star_np = y_star.detach().cpu().numpy().reshape(1, 6)
    modes = [
        ("y1", "Particle 1 position"),
        ("y2", "Particle 2 position"),
        ("center", "Center position"),
        ("relative", "Relative vector y2-y1"),
    ]
    fig = plt.figure(figsize=(20, 5))
    for index, (mode, title) in enumerate(modes, start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        train_rep = representation(train, mode)
        val_rep = representation(val, mode)
        test_rep = representation(test_np, mode)
        n_rep = representation(y_n_np, mode)[0]
        star_rep = representation(y_star_np, mode)[0]
        ax.scatter(train_rep[:, 0], train_rep[:, 1], train_rep[:, 2], s=4, alpha=0.22, label="train")
        ax.scatter(val_rep[:, 0], val_rep[:, 1], val_rep[:, 2], s=5, alpha=0.25, label="validation")
        ax.scatter(test_rep[:, 0], test_rep[:, 1], test_rep[:, 2], s=5, alpha=0.18, label="test")
        ax.scatter(*n_rep, marker="x", s=100, linewidths=2, label=r"$y^n$")
        ax.scatter(*star_rep, marker="*", s=180, label=r"$y^*$")
        set_equal_3d_axes(
            ax,
            np.vstack([train_rep, val_rep, test_rep, n_rep[None, :], star_rep[None, :]]),
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=7)
    fig.suptitle("Training / validation / test distributions", y=1.02)
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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    train_epochs = [record["epoch"] for record in train_log]
    axes[0].plot(
        train_epochs,
        [finite_plot_value(record["training_gap_for_readability"]) for record in train_log],
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Training trajectory energy-sum gap")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training gap")

    val_epochs = [record["epoch"] for record in validation_log]
    axes[1].plot(
        val_epochs,
        [finite_plot_value(record["metrics"]["final_residual_p95"]) for record in validation_log],
        marker="o",
        markersize=3,
        label="residual p95",
    )
    axes[1].plot(
        val_epochs,
        [finite_plot_value(record["metrics"]["final_residual_median"]) for record in validation_log],
        marker="s",
        markersize=3,
        label="residual median",
    )
    axes[1].set_yscale("log")
    axes[1].set_title("Validation residual after fixed rollout")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Residual")
    axes[1].legend()

    axes[2].plot(
        val_epochs,
        [finite_plot_value(record["metrics"]["final_energy_gap_p95"]) for record in validation_log],
        marker="o",
        markersize=3,
        label="energy gap p95",
    )
    axes[2].plot(
        val_epochs,
        [finite_plot_value(record["metrics"]["final_exact_error_p95"]) for record in validation_log],
        marker="s",
        markersize=3,
        label="exact error p95",
    )
    axes[2].set_yscale("log")
    axes[2].set_title("Validation energy / exact error")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, linestyle="--", alpha=0.7)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_special_metric_comparison(
    *,
    best_trajectory: dict[str, Any],
    last_trajectory: dict[str, Any],
    save_path: Path,
    title_prefix: str,
) -> None:
    metrics = [
        ("residual_norm", "Residual"),
        ("energy_gap", "Energy gap"),
        ("exact_error", "Exact-solution error"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for trajectory, label, linestyle in [
        (best_trajectory, "best validation checkpoint", "-"),
        (last_trajectory, "last epoch checkpoint", "--"),
    ]:
        steps = [item["step"] for item in trajectory["iterations"]]
        for ax, (metric, title) in zip(axes, metrics):
            values = [finite_plot_value(item[metric]) for item in trajectory["iterations"]]
            ax.plot(steps, values, linestyle=linestyle, marker="o", markersize=3, label=label)
            ax.set_yscale("log")
            ax.set_title(f"{title_prefix}: {title}")
            ax.set_xlabel("Iteration")
            ax.grid(True, alpha=0.3)
            ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_two_particle_trajectory_3d(
    *,
    trajectory: dict[str, Any],
    y_star: torch.Tensor,
    save_path: Path,
) -> None:
    points = finite_rows(
        np.asarray([item["y"] for item in trajectory["iterations"]], dtype=float), 6
    )
    star = y_star.detach().cpu().numpy()
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    if points.shape[0] == 0:
        ax.text2D(
            0.5,
            0.5,
            "No finite trajectory points were available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title("Best-checkpoint trajectory from n-state")
        plt.tight_layout()
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        return
    y1 = points[:, 0:3]
    y2 = points[:, 3:6]
    ax.plot(y1[:, 0], y1[:, 1], y1[:, 2], "-o", markersize=3, label="particle 1")
    ax.plot(y2[:, 0], y2[:, 1], y2[:, 2], "-s", markersize=3, label="particle 2")
    ax.scatter(*y1[0], marker="x", s=100, linewidths=2, label="initial particle 1")
    ax.scatter(*y2[0], marker="x", s=100, linewidths=2, label="initial particle 2")
    ax.scatter(*star[0:3], marker="*", s=180, label="exact particle 1")
    ax.scatter(*star[3:6], marker="*", s=180, label="exact particle 2")
    spring_steps = sorted(set([0, len(points) - 1] + list(range(0, len(points), 10))))
    for index in spring_steps:
        ax.plot(
            [y1[index, 0], y2[index, 0]],
            [y1[index, 1], y2[index, 1]],
            [y1[index, 2], y2[index, 2]],
            linestyle=":",
            alpha=0.45,
        )
    set_equal_3d_axes(ax, np.vstack([y1, y2, star.reshape(2, 3)]))
    ax.set_title("Best-checkpoint trajectory from n-state")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_fixed_point_check(
    *,
    best_trajectory: dict[str, Any],
    last_trajectory: dict[str, Any],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    metric_specs = [
        ("applied_delta_norm", "Applied update norm"),
        ("residual_norm", "Residual"),
        ("energy_gap", "Energy gap"),
        ("exact_error", "Exact-solution error"),
    ]
    for trajectory, label, linestyle in [
        (best_trajectory, "best validation checkpoint", "-"),
        (last_trajectory, "last epoch checkpoint", "--"),
    ]:
        for ax, (metric, title) in zip(axes, metric_specs):
            if metric == "applied_delta_norm":
                records = trajectory["iterations"][:-1]
            else:
                records = trajectory["iterations"]
            steps = [item["step"] for item in records]
            values = [finite_plot_value(item[metric]) for item in records]
            ax.plot(steps, values, linestyle=linestyle, marker="o", markersize=3, label=label)
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle("Exact-solution fixed-point check", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_test_rollout_metrics(metrics: dict[str, Any], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("residual", "Test residual"),
        ("energy_gap", "Test energy gap"),
        ("exact_error", "Test exact-solution error"),
    ]
    steps = list(range(metrics["steps"] + 1))
    for ax, (prefix, title) in zip(axes, specs):
        for stat, marker in [("mean", "o"), ("median", "s"), ("p95", "^")]:
            values = [
                finite_plot_value(value)
                for value in metrics[f"{prefix}_{stat}_by_step"]
            ]
            ax.plot(steps, values, marker=marker, markersize=3, label=stat)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_slice_directions(y_n: torch.Tensor, y_star: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    direction1 = (y_n - y_star).detach().cpu().numpy().astype(float)
    direction1 /= np.linalg.norm(direction1)
    basis_candidates = np.eye(6)
    dot_abs = np.abs(basis_candidates @ direction1)
    candidate = basis_candidates[int(np.argmin(dot_abs))]
    direction2 = candidate - np.dot(candidate, direction1) * direction1
    direction2 /= np.linalg.norm(direction2)
    return direction1, direction2


def plot_energy_contour_2d_slice(
    *,
    trajectory: dict[str, Any],
    problem: PhysicalProblem,
    problem_tensors: ProblemTensors,
    save_path: Path,
) -> None:
    y_star = problem_tensors.y_star.detach().cpu().numpy().astype(float)
    y_n = problem_tensors.y_n.detach().cpu().numpy().astype(float)
    direction1, direction2 = make_slice_directions(problem_tensors.y_n, problem_tensors.y_star)
    first_coordinate_n = float(np.dot(y_n - y_star, direction1))
    extent = max(abs(first_coordinate_n) * 1.35, problem_tensors.sampling_radius)
    alpha = np.linspace(-extent, extent, 180)
    beta = np.linspace(-extent, extent, 180)
    aa, bb = np.meshgrid(alpha, beta)
    points = (
        y_star.reshape(1, 1, 6)
        + aa[..., None] * direction1.reshape(1, 1, 6)
        + bb[..., None] * direction2.reshape(1, 1, 6)
    )
    flat = torch.tensor(points.reshape(-1, 6), dtype=TORCH_DTYPE)
    q = torch.cat([problem_tensors.q1, problem_tensors.q2]).reshape(1, 6).expand(flat.shape[0], -1)
    masses = problem_tensors.masses.reshape(1, 2).expand(flat.shape[0], -1)
    energies = variational_energy(
        flat,
        q,
        masses,
        g=problem.g,
        dt=problem.dt,
        spring_k=problem.spring_k,
        rest_length=problem.rest_length,
    ).reshape(aa.shape).numpy()
    gap = np.maximum(energies - problem_tensors.exact_energy, PLOT_FLOOR)
    max_gap = float(np.max(gap))
    min_level = max(max_gap * 1e-8, PLOT_FLOOR)
    max_level = max(max_gap, min_level * 10.0)
    levels = np.geomspace(min_level, max_level, 30)

    trajectory_points = np.asarray([item["y"] for item in trajectory["iterations"]], dtype=float)
    centered = trajectory_points - y_star.reshape(1, 6)
    trajectory_alpha = centered @ direction1
    trajectory_beta = centered @ direction2

    fig, ax = plt.subplots(figsize=(9, 7))
    contour = ax.contourf(
        aa,
        bb,
        gap,
        levels=levels,
        norm=matplotlib.colors.LogNorm(vmin=min_level, vmax=max_level),
        alpha=0.82,
        extend="both",
    )
    ax.plot(trajectory_alpha, trajectory_beta, "-o", markersize=3, label="best checkpoint")
    ax.scatter(first_coordinate_n, 0.0, marker="x", s=100, linewidths=2, label=r"$y^n$")
    ax.scatter(0.0, 0.0, marker="*", s=180, label=r"$y^*$")
    ax.set_xlabel("Coordinate along y^n - y*")
    ax.set_ylabel("Orthogonal slice coordinate")
    ax.set_title("Energy-gap contour on a 2D slice of 6D space")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.colorbar(contour, ax=ax, label="E(y)-E(y*)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def optimizer_key(record: dict[str, Any]) -> tuple[str, float, str]:
    return (
        str(record["optimizer_name"]),
        float(record["learning_rate"]),
        str(record["output_scale_mode"]),
    )


def optimizer_label(key: tuple[str, float, str]) -> str:
    suffix = "output×dt" if key[2] == "dt" else "raw output"
    return f"{key[0].upper()} lr={key[1]:.0e}, {suffix}"


def activation_label(name: str) -> str:
    return {"identity": "Identity", "relu": "ReLU", "silu": "SiLU"}.get(name, name)


def ordered_activation_names(records: Sequence[dict[str, Any]]) -> list[str]:
    present = {str(record["activation_name"]) for record in records}
    return [name for name in DEFAULT_ACTIVATION_NAMES if name in present]


def plot_final_test_summary(records: Sequence[dict[str, Any]], save_path: Path) -> None:
    metric_rows = [
        [
            ("final_residual_mean", "Test residual mean"),
            ("final_residual_median", "Test residual median"),
            ("final_residual_p95", "Test residual p95"),
            ("n_state_final_residual", "n-state residual"),
        ],
        [
            ("final_energy_gap_mean", "Test energy gap mean"),
            ("final_energy_gap_median", "Test energy gap median"),
            ("final_energy_gap_p95", "Test energy gap p95"),
            ("n_state_final_energy_gap", "n-state energy gap"),
        ],
        [
            ("final_exact_error_mean", "Test exact error mean"),
            ("final_exact_error_median", "Test exact error median"),
            ("final_exact_error_p95", "Test exact error p95"),
            ("n_state_final_exact_error", "n-state exact error"),
        ],
    ]
    fig, axes = plt.subplots(3, 4, figsize=(23, 15), squeeze=False)
    for activation_name in ordered_activation_names(records):
        selected = sorted(
            [record for record in records if record["activation_name"] == activation_name],
            key=lambda item: int(item["dataset_size"]),
        )
        sizes = [int(record["dataset_size"]) for record in selected]
        for row_index, row in enumerate(metric_rows):
            for col_index, (metric, title) in enumerate(row):
                axes[row_index, col_index].plot(
                    sizes,
                    [finite_plot_value(record["best_checkpoint_test"][metric]) for record in selected],
                    marker="o",
                    label=activation_label(activation_name),
                )
                axes[row_index, col_index].set_title(title)
    for ax in axes.reshape(-1):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training dataset size")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("SGD lr=1e-2 with dt-scaled output: activation comparison", y=1.002)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss_summary(records: Sequence[dict[str, Any]], save_path: Path) -> None:
    sizes = sorted({int(record["dataset_size"]) for record in records})
    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 4.8), squeeze=False)
    for col, size in enumerate(sizes):
        ax = axes[0, col]
        selected = [record for record in records if int(record["dataset_size"]) == size]
        for activation_name in ordered_activation_names(selected):
            record = next(item for item in selected if item["activation_name"] == activation_name)
            curve = record["training_curve_for_summary"]
            ax.plot(
                [point["epoch"] for point in curve],
                [finite_plot_value(point["training_gap_for_readability"]) for point in curve],
                label=activation_label(activation_name),
            )
        ax.set_yscale("log")
        ax.set_title(f"SGD lr=1e-2, output×dt\nN={size:,}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training gap")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Training loss by activation function", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_validation_summary(records: Sequence[dict[str, Any]], save_path: Path) -> None:
    sizes = sorted({int(record["dataset_size"]) for record in records})
    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 4.8), squeeze=False)
    for col, size in enumerate(sizes):
        ax = axes[0, col]
        selected = [record for record in records if int(record["dataset_size"]) == size]
        for activation_name in ordered_activation_names(selected):
            record = next(item for item in selected if item["activation_name"] == activation_name)
            curve = record["validation_curve_for_summary"]
            ax.plot(
                [point["epoch"] for point in curve],
                [finite_plot_value(point["metrics"]["final_residual_p95"]) for point in curve],
                marker="o",
                markersize=3,
                label=activation_label(activation_name),
            )
            if record["best_validation_epoch"] is not None:
                ax.axvline(record["best_validation_epoch"], linestyle="--", alpha=0.4)
        ax.set_yscale("log")
        ax.set_title(f"SGD lr=1e-2, output×dt\nN={size:,}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation residual p95")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Validation curves used for checkpoint selection", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. One experiment
# ============================================================


def run_experiment(
    *,
    base_output_dir: Path,
    training_cpu: DatasetBundle,
    validation_cpu: DatasetBundle,
    test_cpu: DatasetBundle,
    n_state_cpu: DatasetBundle,
    exact_state_cpu: DatasetBundle,
    optimizer_config: dict[str, Any],
    activation_name: str,
    config: RuntimeConfig,
    problem: PhysicalProblem,
    problem_tensors: ProblemTensors,
) -> dict[str, Any]:
    dataset_size = int(training_cpu.initial_y.shape[0])
    optimizer_name = str(optimizer_config["optimizer_name"])
    learning_rate = float(optimizer_config["learning_rate"])
    output_scale_mode = str(optimizer_config["output_scale_mode"])
    output_tag = "output_dt" if output_scale_mode == "dt" else "output_raw"
    experiment_name = (
        f"{optimizer_name}_lr_{learning_rate:.0e}_{output_tag}_"
        f"{activation_name}_num_samples_{dataset_size}"
    )
    output_dir = base_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)
    training = training_cpu.to(device)
    input_mean, input_std = compute_input_normalizer(training_cpu, problem)

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)
    model = MLPOptimizer(
        input_mean=input_mean,
        input_std=input_std,
        activation_name=activation_name,
    ).to(device)
    optimizer = create_optimizer(model, optimizer_name, learning_rate)
    output_scale = output_scale_value(output_scale_mode, problem.dt)

    exact_energy_per_sample = variational_energy(
        training.exact_y,
        training.q,
        training.masses,
        g=problem.g,
        dt=problem.dt,
        spring_k=problem.spring_k,
        rest_length=problem.rest_length,
    )
    exact_energy_mean = float(exact_energy_per_sample.mean().item())

    print("\n" + "=" * 96)
    print(f"Experiment: {experiment_name}")
    print(
        f"device={device}, dtype={TORCH_DTYPE}, architecture=17->64->"
        f"{activation_name}->6"
    )
    print(
        f"training_N={dataset_size:,}, validation_N={config.validation_size:,}, "
        f"test_N={config.test_size:,}"
    )
    print(
        f"optimizer={optimizer_name}, lr={learning_rate:.0e}, "
        f"output_scale_mode={output_scale_mode}, output_scale={output_scale:g}"
    )
    print("no_early_stopping=True; validation_selects_best_checkpoint_only")
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

    for epoch_index in range(config.epochs):
        epoch_number = epoch_index + 1
        rollout_k = get_k_for_epoch(epoch_index, config)
        model.train()
        y = training.initial_y
        optimizer.zero_grad(set_to_none=True)
        trajectory_loss = torch.zeros((), dtype=TORCH_DTYPE, device=device)

        for _ in range(rollout_k):
            y, _, _ = apply_model_update(
                model, y, training, problem, output_scale
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
            print(f"Training stopped: epoch={divergence_epoch}, reason={divergence_reason}")
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
                f"val_exact_p95={validation_metrics['final_exact_error_p95']:.4e} | "
                f"best_epoch={best_epoch} | elapsed={elapsed:.1f}s"
            )

    last_state_dict = state_dict_to_cpu(model)
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(last_state_dict)
        best_epoch = train_log[-1]["epoch"] if train_log else 0
        best_validation_metrics = None

    torch.save(last_state_dict, output_dir / "last_model_state_dict.pt")
    torch.save(best_state_dict, output_dir / "best_validation_model_state_dict.pt")
    torch.save(best_state_dict, output_dir / "mlp_optimizer_state_dict.pt")

    def evaluate_checkpoint(
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
        exact_state_trajectory = evaluate_single_trajectory(
            model=model,
            dataset_cpu=exact_state_cpu,
            problem=problem,
            output_scale=output_scale,
            steps=config.evaluation_steps,
            device=device,
        )
        return (
            test_eval,
            n_state_eval,
            exact_state_eval,
            n_state_trajectory,
            exact_state_trajectory,
        )

    (
        best_test_eval,
        best_n_state_eval,
        best_exact_state_eval,
        best_n_state_trajectory,
        best_exact_state_trajectory,
    ) = evaluate_checkpoint(best_state_dict)
    (
        last_test_eval,
        last_n_state_eval,
        last_exact_state_eval,
        last_n_state_trajectory,
        last_exact_state_trajectory,
    ) = evaluate_checkpoint(last_state_dict)

    physical_record = {
        **asdict(problem),
        "q1": tensor_to_list(problem_tensors.q1),
        "q2": tensor_to_list(problem_tensors.q2),
        "y_n": tensor_to_list(problem_tensors.y_n),
        "y_star": tensor_to_list(problem_tensors.y_star),
        "lambda": problem_tensors.lambda_value,
        "sampling_radius_linf": problem_tensors.sampling_radius,
        "exact_energy": problem_tensors.exact_energy,
        "exact_residual": problem_tensors.exact_residual,
    }

    report = {
        "config": {
            "experiment_name": experiment_name,
            "torch_dtype": str(TORCH_DTYPE),
            "device": str(device),
            "architecture": f"17 -> 64 -> {activation_name} -> 6",
            "activation_name": activation_name,
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "output_scale_mode": output_scale_mode,
            "output_scale_value": output_scale,
            "dataset_size": dataset_size,
            "training_mode": "full_batch",
            "epochs_requested": config.epochs,
            "completed_epochs": len(train_log),
            "no_early_stopping": True,
            "validation_interval": config.validation_interval,
            "evaluation_steps": config.evaluation_steps,
            "validation_size": config.validation_size,
            "test_size": config.test_size,
            "checkpoint_selection": (
                "lexicographic: final residual nonfinite count, residual p95, "
                "exact-error p95, energy-gap p95"
            ),
            "initial_K": config.initial_k,
            "K_increase_interval": config.k_increase_interval,
            "K_increase_amount": config.k_increase_amount,
            "max_K": config.max_k,
            "input_mean": tensor_to_list(model.input_mean),
            "input_std": tensor_to_list(model.input_std),
            "loss": "sum of stepwise mean original variational energy over full batch",
            "backpropagation": "full unroll without detach; one backward and one optimizer step per epoch",
            "model_random_seed": MODEL_RANDOM_SEED,
        },
        "physical_problem": physical_record,
        "dataset": training_cpu.metadata,
        "training_status": {
            "diverged": diverged,
            "divergence_epoch": divergence_epoch,
            "divergence_reason": divergence_reason,
            "completed_epochs": len(train_log),
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
                "heldout_test": best_test_eval,
                "n_state": best_n_state_eval,
                "exact_state": best_exact_state_eval,
            },
            "last_epoch_checkpoint": {
                "heldout_test": last_test_eval,
                "n_state": last_n_state_eval,
                "exact_state": last_exact_state_eval,
            },
        },
        "special_trajectories": {
            "best_validation_checkpoint": {
                "n_state": best_n_state_trajectory,
                "exact_state": best_exact_state_trajectory,
            },
            "last_epoch_checkpoint": {
                "n_state": last_n_state_trajectory,
                "exact_state": last_exact_state_trajectory,
            },
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
        plot_special_metric_comparison(
            best_trajectory=best_n_state_trajectory,
            last_trajectory=last_n_state_trajectory,
            save_path=output_dir / "n_state_best_vs_last.png",
            title_prefix="n-state",
        )
        plot_two_particle_trajectory_3d(
            trajectory=best_n_state_trajectory,
            y_star=problem_tensors.y_star,
            save_path=output_dir / "n_state_two_particle_trajectory_3d.png",
        )
        plot_fixed_point_check(
            best_trajectory=best_exact_state_trajectory,
            last_trajectory=last_exact_state_trajectory,
            save_path=output_dir / "exact_solution_fixed_point_check.png",
        )
        plot_test_rollout_metrics(
            best_test_eval,
            output_dir / "test_rollout_metrics.png",
        )
        if not config.skip_contour:
            plot_energy_contour_2d_slice(
                trajectory=best_n_state_trajectory,
                problem=problem,
                problem_tensors=problem_tensors,
                save_path=output_dir / "energy_contour_2d_slice.png",
            )

    def compact_test_summary(
        test_eval: dict[str, Any],
        n_state_eval: dict[str, Any],
        exact_state_eval: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "final_residual_mean": test_eval["final_residual_mean"],
            "final_residual_median": test_eval["final_residual_median"],
            "final_residual_p95": test_eval["final_residual_p95"],
            "final_residual_max": test_eval["final_residual_max"],
            "final_energy_gap_mean": test_eval["final_energy_gap_mean"],
            "final_energy_gap_median": test_eval["final_energy_gap_median"],
            "final_energy_gap_p95": test_eval["final_energy_gap_p95"],
            "final_energy_gap_max": test_eval["final_energy_gap_max"],
            "final_exact_error_mean": test_eval["final_exact_error_mean"],
            "final_exact_error_median": test_eval["final_exact_error_median"],
            "final_exact_error_p95": test_eval["final_exact_error_p95"],
            "final_exact_error_max": test_eval["final_exact_error_max"],
            "n_state_final_residual": n_state_eval["single_point_final_residual"],
            "n_state_final_energy_gap": n_state_eval["single_point_final_energy_gap"],
            "n_state_final_exact_error": n_state_eval["single_point_final_exact_error"],
            "exact_state_final_residual": exact_state_eval["single_point_final_residual"],
            "exact_state_final_energy_gap": exact_state_eval["single_point_final_energy_gap"],
            "exact_state_final_exact_error": exact_state_eval["single_point_final_exact_error"],
        }

    best_summary = compact_test_summary(
        best_test_eval, best_n_state_eval, best_exact_state_eval
    )
    last_summary = compact_test_summary(
        last_test_eval, last_n_state_eval, last_exact_state_eval
    )

    print(
        f"Completed {experiment_name}: best_epoch={best_epoch}, "
        f"test_residual_p95={best_test_eval['final_residual_p95']:.4e}, "
        f"test_exact_p95={best_test_eval['final_exact_error_p95']:.4e}, "
        f"n_state_residual={best_n_state_eval['single_point_final_residual']:.4e}"
    )

    del training
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment_name": experiment_name,
        "activation_name": activation_name,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "output_scale_mode": output_scale_mode,
        "output_scale_value": output_scale,
        "dataset_size": dataset_size,
        "diverged": diverged,
        "divergence_epoch": divergence_epoch,
        "divergence_reason": divergence_reason,
        "completed_epochs": len(train_log),
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": list(best_key) if best_key is not None else None,
        "best_validation_metrics": best_validation_metrics,
        "best_checkpoint_test": best_summary,
        "last_checkpoint_test": last_summary,
        "training_curve_for_summary": downsample_log(train_log),
        "validation_curve_for_summary": downsample_log(validation_log),
        "output_directory": str(output_dir),
    }


# ============================================================
# 8. CLI and main program
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-particle spring activation ablation with SGD(lr=1e-2), "
            "dt-scaled output, and Identity/ReLU/SiLU."
        )
    )
    parser.add_argument(
        "--target-dataset-sizes",
        "--dataset-sizes",
        dest="target_dataset_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_DATASET_SIZES,
    )
    parser.add_argument(
        "--activation-names",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of: identity relu silu. Default: all three.",
    )
    parser.add_argument(
        "--optimizer-configs",
        type=str,
        nargs="+",
        default=None,
        help="Only sgd:1e-2 is supported; normally omit this argument.",
    )
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
    parser.add_argument("--skip-contour", action="store_true")
    parser.add_argument("--skip-individual-plots", action="store_true")
    parser.add_argument("--skip-dataset-plot", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    target_sizes = ensure_even_positive_sizes(args.target_dataset_sizes)
    activation_names = parse_activation_names(args.activation_names)
    optimizer_configs = parse_optimizer_configs(args.optimizer_configs)
    positive_fields = {
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
    if args.validation_size % 2 != 0 or args.test_size % 2 != 0:
        raise ValueError("validation_size and test_size must be even for swap augmentation.")
    if args.max_k < args.initial_k:
        raise ValueError("max_k must be >= initial_k.")
    return RuntimeConfig(
        target_dataset_sizes=target_sizes,
        activation_names=activation_names,
        optimizer_configs=optimizer_configs,
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
        skip_contour=bool(args.skip_contour),
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

    print(f"Output directory: {output_dir}")
    print(f"Activations: {config.activation_names}")
    print(f"Runtime config: {asdict(config)}")
    print(f"torch default dtype: {torch.get_default_dtype()}")
    print(f"Exact solution: {tensor_to_list(problem_tensors.y_star)}")
    print(f"Sampling radius (L_inf): {problem_tensors.sampling_radius:.12e}")
    print(f"Exact residual: {problem_tensors.exact_residual:.12e}")

    save_json(
        {
            "activation_names": config.activation_names,
            "default_device": DEFAULT_DEVICE,
            "runtime_config": asdict(config),
            "torch_dtype": str(TORCH_DTYPE),
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

    max_canonical_train = max(config.target_dataset_sizes) // 2
    canonical_train_pool, train_pool_metadata = generate_canonical_sobol_points(
        count=max_canonical_train,
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

    torch.save(
        {
            "validation": {
                "initial_y": validation_cpu.initial_y,
                "q": validation_cpu.q,
                "masses": validation_cpu.masses,
                "exact_y": validation_cpu.exact_y,
                "metadata": validation_cpu.metadata,
            },
            "test": {
                "initial_y": test_cpu.initial_y,
                "q": test_cpu.q,
                "masses": test_cpu.masses,
                "exact_y": test_cpu.exact_y,
                "metadata": test_cpu.metadata,
            },
        },
        output_dir / "fixed_validation_test_split.pt",
    )

    dataset_metadata = {
        "sampling_center": tensor_to_list(problem_tensors.y_star),
        "sampling_radius_linf": problem_tensors.sampling_radius,
        "train_sizes_after_swap_augmentation": config.target_dataset_sizes,
        "train_pool": train_pool_metadata,
        "validation": validation_cpu.metadata,
        "test": test_cpu.metadata,
        "special_points": {
            "n_state": tensor_to_list(problem_tensors.y_n),
            "exact_state": tensor_to_list(problem_tensors.y_star),
        },
        "nested_training_sets": True,
        "normalization": "computed independently from each final augmented training set only",
    }
    save_json(dataset_metadata, output_dir / "dataset_metadata.json")

    largest_training_cpu = build_augmented_dataset(
        canonical_points=canonical_train_pool,
        problem_tensors=problem_tensors,
        role="training",
        source_metadata={
            **train_pool_metadata,
            "final_requested_size": max(config.target_dataset_sizes),
        },
    )
    if not config.skip_dataset_plot:
        plot_dataset_distribution_overview(
            training=largest_training_cpu,
            validation=validation_cpu,
            test=test_cpu,
            y_n=problem_tensors.y_n,
            y_star=problem_tensors.y_star,
            save_path=output_dir / "dataset_distribution_overview.png",
        )

    summaries: list[dict[str, Any]] = []
    for dataset_size in config.target_dataset_sizes:
        canonical_count = dataset_size // 2
        training_cpu = build_augmented_dataset(
            canonical_points=canonical_train_pool[:canonical_count],
            problem_tensors=problem_tensors,
            role="training",
            source_metadata={
                **train_pool_metadata,
                "canonical_prefix_size": canonical_count,
                "final_requested_size": dataset_size,
                "explicitly_contains_y_n": True,
                "explicitly_contains_y_star": True,
            },
        )
        print(
            f"\nPrepared nested training set: final_N={dataset_size:,}, "
            f"canonical_N={canonical_count:,}"
        )
        for activation_name in config.activation_names:
            for optimizer_config in config.optimizer_configs:
                summaries.append(
                    run_experiment(
                        base_output_dir=output_dir,
                        training_cpu=training_cpu,
                        validation_cpu=validation_cpu,
                        test_cpu=test_cpu,
                        n_state_cpu=n_state_cpu,
                        exact_state_cpu=exact_state_cpu,
                        optimizer_config=optimizer_config,
                        activation_name=activation_name,
                        config=config,
                        problem=problem,
                        problem_tensors=problem_tensors,
                    )
                )

    overall_report = {
        "experiment_type": "two_particle_single_spring_sgd_activation_ablation",
        "activation_names": config.activation_names,
        "purpose": (
            "Compare Identity, ReLU, and SiLU for one fixed nonlinear two-particle "
            "spring variational problem using only SGD(lr=1e-2) with dt-scaled output."
        ),
        "runtime_config": asdict(config),
        "network": {
            "architectures": [
                f"17 -> 64 -> {name} -> 6" for name in config.activation_names
            ],
            "final_layer_zero_initialized": True,
            "input_normalization": "per-feature, training set only",
            "torch_dtype": str(TORCH_DTYPE),
        },
        "optimizer_output_scaling": {
            "sgd_1e-2": "network output multiplied by dt",
        },
        "physical_problem": {
            **asdict(problem),
            "y_star": tensor_to_list(problem_tensors.y_star),
            "sampling_radius_linf": problem_tensors.sampling_radius,
            "exact_energy": problem_tensors.exact_energy,
            "exact_residual": problem_tensors.exact_residual,
        },
        "dataset_metadata": dataset_metadata,
        "data_roles": {
            "training": "full-batch gradient updates",
            "validation": "checkpoint selection only; never backpropagated",
            "test": "final report only; never evaluated during training",
            "n_state": "separate physical initial state evaluation",
            "exact_state": "separate fixed-point evaluation",
        },
        "no_early_stopping": True,
        "num_experiments": len(summaries),
        "experiments": summaries,
    }
    save_json(overall_report, output_dir / "all_experiments_summary.json")

    plot_final_test_summary(summaries, output_dir / "final_test_summary.png")
    plot_training_loss_summary(summaries, output_dir / "training_loss_summary.png")
    plot_validation_summary(summaries, output_dir / "validation_summary.png")

    print("\n" + "=" * 96)
    print("All experiments completed.")
    print(f"Summary JSON: {output_dir / 'all_experiments_summary.json'}")
    for record in summaries:
        print(
            f"- {record['experiment_name']}: best_epoch={record['best_validation_epoch']}, "
            f"test_residual_p95={record['best_checkpoint_test']['final_residual_p95']:.4e}, "
            f"test_exact_p95={record['best_checkpoint_test']['final_exact_error_p95']:.4e}, "
            f"diverged={record['diverged']}"
        )


if __name__ == "__main__":
    main()
