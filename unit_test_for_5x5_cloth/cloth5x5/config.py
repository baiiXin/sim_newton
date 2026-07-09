from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch

from .constants import (
    FIXED_VERTEX_INDICES,
    GRID_COLS,
    GRID_ROWS,
    NUM_PARTICLES,
    NUM_SPRINGS,
    PLOT_FLOOR,
    SPRING_EDGES,
    TORCH_DTYPE,
)


@dataclass(frozen=True)
class RuntimeConfig:
    total_time_steps: int
    train_points_per_problem: int
    eval_points_per_problem: int
    epochs: int
    validation_interval: int
    diagnostic_interval: int
    evaluation_steps: int
    evaluation_batch_size: int
    initial_k: int
    k_increase_interval: int
    k_increase_amount: int
    max_k: int
    report_steps: tuple[int, ...]
    residual_length_scale: float
    gradient_clip_norm: float
    sampling_radius_min: float
    sampling_radius_max: float
    device: str
    run_single_motion_baseline: bool
    skip_plots: bool
    save_datasets: bool


@dataclass(frozen=True)
class PhysicalConfig:
    masses: tuple[float, ...]
    g: float
    dt: float
    spring_stiffness: tuple[float, ...]
    rest_lengths: tuple[float, ...]
    p0: tuple[tuple[float, float, float], ...]
    v0: tuple[tuple[float, float, float], ...]

    @property
    def fixed_positions(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(self.p0[index] for index in FIXED_VERTEX_INDICES)


@dataclass(frozen=True)
class MotionSpec:
    index: int
    name: str
    split: str
    category: str
    source: str
    p0: tuple[tuple[float, float, float], ...]
    v0: tuple[tuple[float, float, float], ...]
    stretch_x: float
    shear: float
    bend: float
    twist: float
    translation_velocity: tuple[float, float, float]
    angular_velocity: float
    velocity_gradient: float
    ood_factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class MotionSplit:
    train_motion_indices: tuple[int, ...]
    validation_motion_indices: tuple[int, ...]
    id_test_motion_indices: tuple[int, ...]
    ood_test_motion_indices: tuple[int, ...]


@dataclass(frozen=True)
class TimeStepProblem:
    index: int
    motion_index: int
    motion_name: str
    motion_split: str
    motion_category: str
    local_time_index: int
    time: float
    p_n_full: torch.Tensor
    v_n_full: torch.Tensor
    q_free: torch.Tensor
    free_masses: torch.Tensor
    exact_y_free: torch.Tensor
    raw_sampling_radius: float
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
    motion_index: torch.Tensor
    time_index: torch.Tensor
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
            motion_index=self.motion_index.to(device=device, dtype=torch.long),
            time_index=self.time_index.to(device=device, dtype=torch.long),
            metadata=copy.deepcopy(self.metadata),
        )


def default_physical_config() -> PhysicalConfig:
    spacing = 0.5
    height = 1.20
    p0 = tuple(
        (col * spacing, -row * spacing, height)
        for row in range(GRID_ROWS)
        for col in range(GRID_COLS)
    )
    v0 = tuple((0.0, 0.0, 0.0) for _ in range(NUM_PARTICLES))
    rest_lengths = tuple(math.dist(p0[i], p0[j]) for i, j in SPRING_EDGES)
    return PhysicalConfig(
        masses=tuple(1.0 for _ in range(NUM_PARTICLES)),
        g=9.8,
        dt=0.01,
        spring_stiffness=tuple(2500.0 for _ in range(NUM_SPRINGS)),
        rest_lengths=rest_lengths,
        p0=p0,
        v0=v0,
    )


def physical_config_from_dict(data: dict[str, Any]) -> PhysicalConfig:
    return PhysicalConfig(
        masses=tuple(float(v) for v in data["masses"]),
        g=float(data["g"]),
        dt=float(data["dt"]),
        spring_stiffness=tuple(float(v) for v in data["spring_stiffness"]),
        rest_lengths=tuple(float(v) for v in data["rest_lengths"]),
        p0=tuple(tuple(float(x) for x in row) for row in data["p0"]),
        v0=tuple(tuple(float(x) for x in row) for row in data["v0"]),
    )


def get_k_for_epoch(epoch_index: int, config: RuntimeConfig) -> int:
    return min(
        config.initial_k + (epoch_index // config.k_increase_interval) * config.k_increase_amount,
        config.max_k,
    )


def finite_plot_value(value: float | int | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return float("nan")
    return max(float(value), PLOT_FLOOR)
