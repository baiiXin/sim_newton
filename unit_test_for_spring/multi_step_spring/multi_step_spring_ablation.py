"""
Two-particle single-spring learned-optimizer ablation experiment.

This script compares eight controlled variants that reconstruct the transition
from the failed absolute-state solver to the stable residual solver. Every
variant uses exactly the same physical problems, problem-level data split,
Sobol samples, optimizer, learning rate, full-batch schedule, validation
checkpoint rule, test datasets, and evaluation metrics. Only the explicitly
listed ablation factors are changed.

Default training schedule:
    5,000 epochs, K=1/2/3/4/5 for successive 1,000-epoch stages.

Experiments:
    A0_original_raw_state
    A1_raw_state_stabilized
    A2_residual_core_only
    A3_stable_full
    A4_stable_with_bias
    A5_stable_default_init
    A6_stable_no_energy_scale
    A7_stable_no_gradient_clip

Exact solutions are used only to generate the synthetic datasets, report
errors, select checkpoints, and compute diagnostics. They are not network
inputs and do not appear in the backward training objective.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
DEFAULT_DEVICE = "cuda:1"

# Characteristic length used to nondimensionalize the preconditioned residual.
# With rest_length=1, this is 5% of the spring rest length and matches the
# O(1e-2)-O(1e-1) initial-error range in the original datasets.
DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_DIAGNOSTIC_INTERVAL = 500

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
DEFAULT_EPOCHS = 5_000
DEFAULT_VALIDATION_INTERVAL = 500
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8_192
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 1_000
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5
DEFAULT_REPORT_STEPS = (1, 5, 10, 50)

# Newton is evaluated for a fixed number of iterations, exactly like the
# learned solver. States whose residual is already below this tolerance are
# kept fixed to avoid introducing round-off motion at the analytic solution.
NEWTON_RESIDUAL_TOLERANCE = 1e-12


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
    residual_length_scale: float
    gradient_clip_norm: float
    diagnostic_interval: int
    device: str
    experiment_names: tuple[str, ...]
    run_newton_baseline: bool
    skip_plots: bool
    save_datasets: bool


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    short_label: str
    input_mode: str
    use_bias: bool
    first_layer_initialization: str
    output_scale_mode: str
    shift_energy: bool
    use_energy_scale: bool
    use_gradient_clip: bool
    comparison_role: str
    changed_factor: str
    interpretation: str

    @property
    def input_dimension(self) -> int:
        if self.input_mode == "raw_state_standardized":
            return 17
        if self.input_mode in {
            "mass_preconditioned_residual",
            "dimensionless_mass_preconditioned_residual",
        }:
            return 6
        raise ValueError(f"Unsupported input_mode={self.input_mode!r}.")


EXPERIMENT_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        name="A0_original_raw_state",
        short_label="A0",
        input_mode="raw_state_standardized",
        use_bias=True,
        first_layer_initialization="pytorch_default",
        output_scale_mode="direct_position_update",
        shift_energy=False,
        use_energy_scale=False,
        use_gradient_clip=False,
        comparison_role="failed baseline",
        changed_factor="Original absolute-state formulation",
        interpretation=(
            "Reproduces the failed reference: standardized [y,q,m,dt,k,l0] "
            "input, affine network, raw physical-energy objective, and no clipping."
        ),
    ),
    ExperimentSpec(
        name="A1_raw_state_stabilized",
        short_label="A1",
        input_mode="raw_state_standardized",
        use_bias=False,
        first_layer_initialization="orthogonal",
        output_scale_mode="multiply_by_length_scale",
        shift_energy=True,
        use_energy_scale=True,
        use_gradient_clip=True,
        comparison_role="forward reconstruction",
        changed_factor="Add all stabilization except residual state",
        interpretation=(
            "Tests whether initialization, bias removal, output scaling, energy "
            "normalization, and clipping can rescue the absolute-state input."
        ),
    ),
    ExperimentSpec(
        name="A2_residual_core_only",
        short_label="A2",
        input_mode="mass_preconditioned_residual",
        use_bias=True,
        first_layer_initialization="pytorch_default",
        output_scale_mode="direct_position_update",
        shift_energy=False,
        use_energy_scale=False,
        use_gradient_clip=False,
        comparison_role="forward reconstruction",
        changed_factor="Replace absolute state by preconditioned residual only",
        interpretation=(
            "Tests whether the local residual representation alone produces the "
            "main improvement while retaining the original-style training setup."
        ),
    ),
    ExperimentSpec(
        name="A3_stable_full",
        short_label="A3",
        input_mode="dimensionless_mass_preconditioned_residual",
        use_bias=False,
        first_layer_initialization="orthogonal",
        output_scale_mode="multiply_by_length_scale",
        shift_energy=True,
        use_energy_scale=True,
        use_gradient_clip=True,
        comparison_role="successful reference",
        changed_factor="All stable residual-solver changes enabled",
        interpretation=(
            "Complete successful formulation and the reference denominator for "
            "all reported order-of-magnitude degradations."
        ),
    ),
    ExperimentSpec(
        name="A4_stable_with_bias",
        short_label="A4",
        input_mode="dimensionless_mass_preconditioned_residual",
        use_bias=True,
        first_layer_initialization="orthogonal",
        output_scale_mode="multiply_by_length_scale",
        shift_energy=True,
        use_energy_scale=True,
        use_gradient_clip=True,
        comparison_role="reverse deletion from A3",
        changed_factor="Restore trainable biases",
        interpretation=(
            "Removes the structural guarantee that zero residual maps exactly to "
            "zero update; isolates fixed-point drift under repeated iteration."
        ),
    ),
    ExperimentSpec(
        name="A5_stable_default_init",
        short_label="A5",
        input_mode="dimensionless_mass_preconditioned_residual",
        use_bias=False,
        first_layer_initialization="pytorch_default",
        output_scale_mode="multiply_by_length_scale",
        shift_energy=True,
        use_energy_scale=True,
        use_gradient_clip=True,
        comparison_role="reverse deletion from A3",
        changed_factor="Replace orthogonal first-layer initialization",
        interpretation=(
            "Tests whether orthogonal initialization is decisive or mainly affects "
            "early optimization speed and conditioning."
        ),
    ),
    ExperimentSpec(
        name="A6_stable_no_energy_scale",
        short_label="A6",
        input_mode="dimensionless_mass_preconditioned_residual",
        use_bias=False,
        first_layer_initialization="orthogonal",
        output_scale_mode="multiply_by_length_scale",
        shift_energy=True,
        use_energy_scale=False,
        use_gradient_clip=True,
        comparison_role="reverse deletion from A3",
        changed_factor="Remove division by the characteristic energy scale",
        interpretation=(
            "Keeps the same physical objective and minimizer but restores its raw "
            "gradient magnitude, exposing interaction with clipping and Adam."
        ),
    ),
    ExperimentSpec(
        name="A7_stable_no_gradient_clip",
        short_label="A7",
        input_mode="dimensionless_mass_preconditioned_residual",
        use_bias=False,
        first_layer_initialization="orthogonal",
        output_scale_mode="multiply_by_length_scale",
        shift_energy=True,
        use_energy_scale=True,
        use_gradient_clip=False,
        comparison_role="reverse deletion from A3",
        changed_factor="Disable global gradient-norm clipping",
        interpretation=(
            "Tests whether clipping is required for stability, especially at the "
            "K=1 to 5 curriculum transitions."
        ),
    ),
)

EXPERIMENT_SPEC_BY_NAME = {spec.name: spec for spec in EXPERIMENT_SPECS}
DEFAULT_EXPERIMENT_NAMES = tuple(spec.name for spec in EXPERIMENT_SPECS)


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



def variational_hessian(
    y: torch.Tensor,
    masses: torch.Tensor,
    *,
    dt: float,
    spring_k: float,
    rest_length: float,
) -> torch.Tensor:
    """Return the analytic 6x6 Hessian of the variational energy.

    For d = y2-y1 and r = ||d||, the spring Hessian with respect to d is

        A = k[(1-l0/r) I + (l0/r^3) d d^T].

    The complete two-particle Hessian is

        [m1/dt^2 I + A,       -A]
        [      -A,      m2/dt^2 I + A].

    Gravity contributes no Hessian because the gravity terms retained in this
    script are independent of y.
    """
    y1 = y[..., 0:3]
    y2 = y[..., 3:6]
    d = y2 - y1
    length = torch.linalg.vector_norm(d, dim=-1, keepdim=True).clamp_min(
        RESIDUAL_DISTANCE_EPS
    )

    identity = torch.eye(3, dtype=y.dtype, device=y.device)
    identity = identity.expand(*y.shape[:-1], 3, 3)
    outer = d.unsqueeze(-1) * d.unsqueeze(-2)
    spring_block = spring_k * (
        (1.0 - rest_length / length).unsqueeze(-1) * identity
        + (rest_length / length.pow(3)).unsqueeze(-1) * outer
    )

    m1_over_dt2 = (masses[..., 0] / dt**2)[..., None, None]
    m2_over_dt2 = (masses[..., 1] / dt**2)[..., None, None]
    hessian = torch.zeros(
        (*y.shape[:-1], 6, 6), dtype=y.dtype, device=y.device
    )
    hessian[..., 0:3, 0:3] = m1_over_dt2 * identity + spring_block
    hessian[..., 3:6, 3:6] = m2_over_dt2 * identity + spring_block
    hessian[..., 0:3, 3:6] = -spring_block
    hessian[..., 3:6, 0:3] = -spring_block
    return hessian


def apply_newton_update(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    *,
    residual_tolerance: float = NEWTON_RESIDUAL_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one undamped full-Newton step to the original physical energy.

    The exact solution is not used. The update is obtained only from the
    analytic gradient and Hessian of the same variational energy used for
    training:

        H(y) delta = -grad E(y),    y_new = y + delta.
    """
    gradient = stationarity_residual(
        y,
        q,
        masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    residual_norm = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True)
    active = residual_norm > residual_tolerance
    rhs = torch.where(active, -gradient, torch.zeros_like(gradient))
    hessian = variational_hessian(
        y,
        masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )

    # The inertial blocks make the Hessian nonsingular for this experiment.
    # solve_ex also exposes a per-sample status so any unexpected singular
    # batch can be handled explicitly rather than silently producing NaNs.
    delta_column, info = torch.linalg.solve_ex(hessian, rhs.unsqueeze(-1))
    delta = delta_column.squeeze(-1)
    failed = info != 0
    if bool(torch.any(failed)):
        failed_hessian = hessian[failed]
        failed_rhs = rhs[failed]
        delta[failed] = torch.matmul(
            torch.linalg.pinv(failed_hessian), failed_rhs.unsqueeze(-1)
        ).squeeze(-1)

    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("Newton update produced non-finite values.")
    return y + delta, delta


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
# 4. Ablation-controlled network inputs and learned updates
# ============================================================


def mass_preconditioned_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """Return dt^2 M^{-1} grad E(y), which has position units."""
    residual = stationarity_residual(
        y,
        q,
        masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    mass_per_coordinate = masses.repeat_interleave(3, dim=-1)
    return (physical.dt**2) * residual / mass_per_coordinate


def dimensionless_residual_input(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_length_scale: torch.Tensor | float,
) -> torch.Tensor:
    return mass_preconditioned_residual(y, q, masses, physical) / residual_length_scale


def physical_energy_scale(
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_length_scale: float,
) -> float:
    reference_mass = float(masses.detach().mean().item())
    return reference_mass * residual_length_scale**2 / physical.dt**2


def build_raw_state_features(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """Build the original 17D feature vector [y,q,m1,m2,dt,k,l0]."""
    batch_shape = y.shape[:-1]
    physical_constants = torch.tensor(
        [physical.dt, physical.spring_k, physical.rest_length],
        dtype=y.dtype,
        device=y.device,
    ).expand(*batch_shape, 3)
    return torch.cat([y, q, masses, physical_constants], dim=-1)


def compute_raw_input_normalizer(
    dataset: DatasetBundle,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = build_raw_state_features(
        dataset.initial_y,
        dataset.q,
        dataset.masses,
        physical,
    )
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False)
    # Constant features (masses and fixed physical parameters) become exactly
    # zero after centering. Setting their divisor to one avoids division by zero.
    std = torch.where(std > 1e-12, std, torch.ones_like(std))
    return mean, std


class MLPOptimizer(nn.Module):
    """One network implementation controlled by an ExperimentSpec."""

    def __init__(
        self,
        *,
        spec: ExperimentSpec,
        residual_length_scale: float,
        raw_input_mean: torch.Tensor | None = None,
        raw_input_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if residual_length_scale <= 0.0:
            raise ValueError("residual_length_scale must be positive.")
        self.spec = spec
        self.linear1 = nn.Linear(
            spec.input_dimension,
            64,
            bias=spec.use_bias,
        )
        self.activation = nn.Identity()
        self.linear2 = nn.Linear(64, 6, bias=spec.use_bias)

        if spec.first_layer_initialization == "orthogonal":
            nn.init.orthogonal_(self.linear1.weight)
            if self.linear1.bias is not None:
                nn.init.zeros_(self.linear1.bias)
        elif spec.first_layer_initialization != "pytorch_default":
            raise ValueError(
                "Unsupported first-layer initialization: "
                f"{spec.first_layer_initialization!r}."
            )

        # All groups retain the original zero-output initialization so every
        # experiment starts from the same zero-update behavior.
        nn.init.zeros_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

        self.register_buffer(
            "residual_length_scale",
            torch.tensor(float(residual_length_scale), dtype=TORCH_DTYPE),
        )

        if spec.input_mode == "raw_state_standardized":
            if raw_input_mean is None or raw_input_std is None:
                raise ValueError("Raw-state experiments require training-set normalizer statistics.")
            if tuple(raw_input_mean.shape) != (17,) or tuple(raw_input_std.shape) != (17,):
                raise ValueError("Raw-state normalizer tensors must have shape (17,).")
            self.register_buffer("raw_input_mean", raw_input_mean.detach().clone())
            self.register_buffer("raw_input_std", raw_input_std.detach().clone())
        else:
            self.register_buffer("raw_input_mean", torch.empty(0, dtype=TORCH_DTYPE))
            self.register_buffer("raw_input_std", torch.empty(0, dtype=TORCH_DTYPE))

    def build_input(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        physical: PhysicalConfig,
    ) -> torch.Tensor:
        if self.spec.input_mode == "raw_state_standardized":
            features = build_raw_state_features(y, q, masses, physical)
            return (features - self.raw_input_mean) / self.raw_input_std
        if self.spec.input_mode == "mass_preconditioned_residual":
            return mass_preconditioned_residual(y, q, masses, physical)
        if self.spec.input_mode == "dimensionless_mass_preconditioned_residual":
            return dimensionless_residual_input(
                y,
                q,
                masses,
                physical,
                self.residual_length_scale,
            )
        raise ValueError(f"Unsupported input mode {self.spec.input_mode!r}.")

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        *,
        physical: PhysicalConfig,
    ) -> torch.Tensor:
        network_input = self.build_input(y, q, masses, physical)
        hidden = self.activation(self.linear1(network_input))
        raw_update = self.linear2(hidden)
        if self.spec.output_scale_mode == "multiply_by_length_scale":
            return self.residual_length_scale * raw_update
        if self.spec.output_scale_mode == "direct_position_update":
            return raw_update
        raise ValueError(
            f"Unsupported output_scale_mode={self.spec.output_scale_mode!r}."
        )


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    dataset: DatasetBundle,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    applied_delta = model(y, dataset.q, dataset.masses, physical=physical)
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



@torch.no_grad()
def evaluate_newton_on_dataset(
    *,
    dataset_cpu: DatasetBundle,
    physical: PhysicalConfig,
    steps: int,
    batch_size: int,
    report_steps: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate full Newton on the same states and metrics as the learned solver."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    selected_steps = _selected_step_indices(steps, report_steps)

    residual_batches: list[torch.Tensor] = []
    gap_batches: list[torch.Tensor] = []
    exact_error_batches: list[torch.Tensor] = []
    point1_error_batches: list[torch.Tensor] = []
    point2_error_batches: list[torch.Tensor] = []
    problem_index_batches: list[torch.Tensor] = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

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
                y, _ = apply_newton_update(y, batch.q, batch.masses, physical)

        residual_batches.append(torch.stack(residual_steps, dim=1))
        gap_batches.append(torch.stack(gap_steps, dim=1))
        exact_error_batches.append(torch.stack(exact_error_steps, dim=1))
        point1_error_batches.append(torch.stack(point1_error_steps, dim=1))
        point2_error_batches.append(torch.stack(point2_error_steps, dim=1))
        problem_index_batches.append(batch.problem_index.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

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
        "solver": "full_newton",
        "steps": steps,
        "num_points": len(dataset_cpu),
        "selected_report_steps": selected_steps,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_point_per_iteration": (
            elapsed_seconds / (len(dataset_cpu) * steps)
        ),
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


def evaluate_newton_baseline(
    *,
    datasets: dict[str, DatasetBundle],
    physical: PhysicalConfig,
    config: RuntimeConfig,
    device: torch.device,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, dataset in datasets.items():
        print(f"Evaluating Newton baseline on {name} ({len(dataset):,} states)...")
        results[name] = evaluate_newton_on_dataset(
            dataset_cpu=dataset,
            physical=physical,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps,
            device=device,
        )
    return results


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



def plot_learned_vs_newton_rollout(
    *,
    learned_metrics: dict[str, Any],
    newton_metrics: dict[str, Any],
    learned_name: str,
    title: str,
    save_path: Path,
) -> None:
    """Compare pooled p95 convergence at every solver iteration."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("residual", "Residual p95"),
        ("energy_gap", "Energy gap p95"),
        ("exact_error", "Exact-solution error p95"),
    ]
    steps = list(range(learned_metrics["steps"] + 1))
    for ax, (prefix, metric_title) in zip(axes, specs):
        ax.plot(
            steps,
            [
                finite_plot_value(value)
                for value in learned_metrics[f"{prefix}_p95_by_step"]
            ],
            marker="o",
            markersize=3,
            label=learned_name,
        )
        ax.plot(
            steps,
            [
                finite_plot_value(value)
                for value in newton_metrics[f"{prefix}_p95_by_step"]
            ],
            marker="s",
            markersize=3,
            label="full Newton",
        )
        ax.set_yscale("log")
        ax.set_title(metric_title)
        ax.set_xlabel("Solver iteration")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_all_solver_final_metrics(
    *,
    summaries: Sequence[dict[str, Any]],
    newton_results: dict[str, Any],
    save_path: Path,
) -> None:
    solver_names = [record["experiment_name"] for record in summaries] + [
        "full_newton"
    ]
    split_names = ["interpolation_test", "extrapolation_test"]
    metrics = [
        ("final_residual_p95", "Residual p95"),
        ("final_energy_gap_p95", "Energy gap p95"),
        ("final_exact_error_p95", "Exact error p95"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    x = np.arange(len(solver_names))
    for row, split_name in enumerate(split_names):
        for col, (metric_key, title) in enumerate(metrics):
            values = [
                finite_plot_value(
                    record["best_checkpoint_test"][split_name][metric_key]
                )
                for record in summaries
            ]
            values.append(finite_plot_value(newton_results[split_name][metric_key]))
            axes[row, col].bar(x, values)
            axes[row, col].set_xticks(x)
            axes[row, col].set_xticklabels(
                solver_names, rotation=15, ha="right"
            )
            axes[row, col].set_yscale("log")
            axes[row, col].set_title(f"{split_name}: {title}")
            axes[row, col].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Learned optimizers and full-Newton baseline", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_current_state_residual_all_solvers(
    *,
    summaries: Sequence[dict[str, Any]],
    newton_results: dict[str, Any],
    problems: Sequence[TimeStepProblem],
    problem_indices: Sequence[int],
    report_steps: Sequence[int],
    save_path: Path,
) -> None:
    valid_steps = [
        step
        for step in report_steps
        if str(step)
        in newton_results["current_state_all_test"]["per_problem"][
            str(problem_indices[0])
        ]["steps"]
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), squeeze=False)
    times = [problems[index].time for index in problem_indices]
    for ax, step in zip(axes.reshape(-1), valid_steps):
        for record in summaries:
            metrics = record["best_checkpoint_test"]["current_state_all_test"]
            values = [
                finite_plot_value(
                    _extract_problem_stat(
                        metrics, index, step, "residual", "mean"
                    )
                )
                for index in problem_indices
            ]
            ax.plot(
                times,
                values,
                marker="o",
                markersize=3,
                label=record["experiment_name"],
            )
        newton_metrics = newton_results["current_state_all_test"]
        newton_values = [
            finite_plot_value(
                _extract_problem_stat(
                    newton_metrics, index, step, "residual", "mean"
                )
            )
            for index in problem_indices
        ]
        ax.plot(
            times,
            newton_values,
            marker="s",
            markersize=3,
            label="full_newton",
        )
        ax.axvline(0.8, linestyle="--", alpha=0.6, label="extrapolation boundary")
        ax.set_yscale("log")
        ax.set_title(f"Current-state start: residual after {step} iterations")
        ax.set_xlabel("Physical time")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.reshape(-1)[len(valid_steps):]:
        ax.axis("off")
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



@torch.no_grad()
def one_step_quality_diagnostics(
    *,
    model: MLPOptimizer,
    dataset: DatasetBundle,
    physical: PhysicalConfig,
) -> dict[str, Any]:
    """
    Diagnose whether one learned iteration improves the current training states.

    Exact solutions are used here only for reporting. Nothing in this function
    participates in backward propagation.
    """
    model.eval()
    y0 = dataset.initial_y
    y1, delta = apply_model_update(model, y0, dataset, physical)

    error_before = torch.linalg.vector_norm(y0 - dataset.exact_y, dim=-1)
    error_after = torch.linalg.vector_norm(y1 - dataset.exact_y, dim=-1)

    residual_before = stationarity_residual_norm(
        y0,
        dataset.q,
        dataset.masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    residual_after = stationarity_residual_norm(
        y1,
        dataset.q,
        dataset.masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )

    energy_before = variational_energy(
        y0,
        dataset.q,
        dataset.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    energy_after = variational_energy(
        y1,
        dataset.q,
        dataset.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )

    ideal_update = dataset.exact_y - y0
    flat_delta = delta.reshape(-1)
    flat_ideal = ideal_update.reshape(-1)
    denominator = (
        torch.linalg.vector_norm(flat_delta)
        * torch.linalg.vector_norm(flat_ideal)
    )
    if float(denominator.item()) > 0.0:
        cosine = float(
            (torch.dot(flat_delta, flat_ideal) / denominator).item()
        )
    else:
        cosine = float("nan")

    nonzero_mask = error_before > 1e-15
    contraction = error_after[nonzero_mask] / error_before[nonzero_mask]
    if contraction.numel() == 0:
        contraction_stats = {
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    else:
        contraction_stats = {
            "mean": float(contraction.mean().item()),
            "median": float(torch.quantile(contraction, 0.5).item()),
            "p95": float(torch.quantile(contraction, 0.95).item()),
            "max": float(contraction.max().item()),
        }

    fixed_point_delta = model(
        dataset.exact_y,
        dataset.q,
        dataset.masses,
        physical=physical,
    )
    fixed_point_update_norm = torch.linalg.vector_norm(
        fixed_point_delta, dim=-1
    )

    return {
        "mean_error_before": float(error_before.mean().item()),
        "mean_error_after": float(error_after.mean().item()),
        "mean_residual_before": float(residual_before.mean().item()),
        "mean_residual_after": float(residual_after.mean().item()),
        "mean_update_norm": float(
            torch.linalg.vector_norm(delta, dim=-1).mean().item()
        ),
        "update_ideal_cosine": cosine,
        "sample_error_improvement_fraction": float(
            (error_after < error_before).to(TORCH_DTYPE).mean().item()
        ),
        "sample_residual_improvement_fraction": float(
            (residual_after < residual_before).to(TORCH_DTYPE).mean().item()
        ),
        "sample_energy_improvement_fraction": float(
            (energy_after < energy_before).to(TORCH_DTYPE).mean().item()
        ),
        "contraction_ratio": contraction_stats,
        "fixed_point_update_mean": float(
            fixed_point_update_norm.mean().item()
        ),
        "fixed_point_update_max": float(
            fixed_point_update_norm.max().item()
        ),
    }


def global_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared_norm = torch.zeros((), dtype=TORCH_DTYPE)
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        grad_norm = torch.linalg.vector_norm(parameter.grad.detach())
        squared_norm = squared_norm.to(grad_norm.device) + grad_norm.square()
    if not found:
        return 0.0
    return float(torch.sqrt(squared_norm).item())


def run_experiment(
    *,
    spec: ExperimentSpec,
    training_cpu: DatasetBundle,
    validation_cpu: DatasetBundle,
    evaluation_datasets: dict[str, DatasetBundle],
    output_dir: Path,
    config: RuntimeConfig,
    physical: PhysicalConfig,
    problems: Sequence[TimeStepProblem],
) -> dict[str, Any]:
    experiment_name = spec.name
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)

    raw_mean: torch.Tensor | None = None
    raw_std: torch.Tensor | None = None
    if spec.input_mode == "raw_state_standardized":
        raw_mean, raw_std = compute_raw_input_normalizer(training_cpu, physical)

    model = MLPOptimizer(
        spec=spec,
        residual_length_scale=config.residual_length_scale,
        raw_input_mean=raw_mean,
        raw_input_std=raw_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training = training_cpu.to(device)

    characteristic_energy_scale = physical_energy_scale(
        training.masses,
        physical,
        config.residual_length_scale,
    )
    objective_divisor = characteristic_energy_scale if spec.use_energy_scale else 1.0
    initial_energy = variational_energy(
        training.initial_y,
        training.q,
        training.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    ).detach()
    exact_energy = variational_energy(
        training.exact_y,
        training.q,
        training.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    ).detach()

    print("\n" + "=" * 100)
    print(f"Experiment: {experiment_name} ({spec.short_label})")
    print(f"Changed factor: {spec.changed_factor}")
    print(f"Interpretation: {spec.interpretation}")
    print(
        f"device={device}, dtype={TORCH_DTYPE}, input={spec.input_mode}, "
        f"architecture={spec.input_dimension}->64->identity->6, "
        f"bias={spec.use_bias}, first_init={spec.first_layer_initialization}, "
        f"output_scale={spec.output_scale_mode}"
    )
    print(
        f"optimizer=Adam(lr={LEARNING_RATE:.0e}), "
        f"energy_shift={spec.shift_energy}, energy_scale={spec.use_energy_scale}, "
        f"objective_divisor={objective_divisor:.3e}, "
        f"gradient_clip={spec.use_gradient_clip}"
    )
    print(
        f"training_points={len(training_cpu):,}, "
        f"training_problems={training_cpu.metadata['num_problems']}, "
        f"validation_points={len(validation_cpu):,}"
    )
    print("no_early_stopping=True; validation_selects_best_checkpoint_only")
    print("=" * 100)

    train_log: list[dict[str, Any]] = []
    quality_diagnostic_log: list[dict[str, Any]] = []
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

        trajectory_objective = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        trajectory_energy_sum = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        trajectory_energy_gap = torch.zeros((), dtype=TORCH_DTYPE, device=device)

        for _ in range(rollout_k):
            y, _ = apply_model_update(model, y, training, physical)
            current_energy = variational_energy(
                y,
                training.q,
                training.masses,
                g=physical.g,
                dt=physical.dt,
                spring_k=physical.spring_k,
                rest_length=physical.rest_length,
            )
            objective_energy = current_energy - initial_energy if spec.shift_energy else current_energy
            trajectory_objective = trajectory_objective + (
                objective_energy / objective_divisor
            ).mean()
            trajectory_energy_sum = trajectory_energy_sum + current_energy.mean()
            trajectory_energy_gap = trajectory_energy_gap + (
                current_energy - exact_energy
            ).mean()

        gradient_norm = float("nan")
        gradient_was_clipped = False
        if not bool(torch.isfinite(trajectory_objective)):
            diverged = True
            divergence_epoch = epoch_number
            divergence_reason = "non-finite trajectory objective"
        else:
            try:
                trajectory_objective.backward()
                if spec.use_gradient_clip:
                    gradient_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.gradient_clip_norm,
                        ).item()
                    )
                    gradient_was_clipped = gradient_norm > config.gradient_clip_norm
                else:
                    gradient_norm = global_gradient_norm(model.parameters())
                if not math.isfinite(gradient_norm):
                    diverged = True
                    divergence_epoch = epoch_number
                    divergence_reason = "non-finite gradient norm"
                else:
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
            print(f"Training stopped at epoch={divergence_epoch}: {divergence_reason}")
            break

        objective_value = float(trajectory_objective.item())
        energy_value = float(trajectory_energy_sum.item())
        training_gap = float(trajectory_energy_gap.item())
        train_log.append(
            {
                "epoch": epoch_number,
                "K": rollout_k,
                "training_objective": objective_value,
                "dimensionless_training_objective": objective_value,
                "trajectory_energy_sum": energy_value,
                "training_gap_for_readability": training_gap,
                "gradient_norm_before_clip": gradient_norm,
                "gradient_clip_enabled": spec.use_gradient_clip,
                "gradient_was_clipped": gradient_was_clipped,
            }
        )

        should_diagnose = (
            epoch_number == 1
            or epoch_number % config.diagnostic_interval == 0
            or epoch_number == config.epochs
        )
        if should_diagnose:
            diagnostics = one_step_quality_diagnostics(
                model=model,
                dataset=training,
                physical=physical,
            )
            diagnostics.update(
                {
                    "epoch": epoch_number,
                    "training_K": rollout_k,
                    "gradient_norm_before_clip": gradient_norm,
                    "gradient_clip_enabled": spec.use_gradient_clip,
                    "gradient_was_clipped": gradient_was_clipped,
                }
            )
            quality_diagnostic_log.append(diagnostics)

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
            message = (
                f"Epoch {epoch_number:5d} | K={rollout_k} | "
                f"objective={objective_value:.4e} | train_gap={training_gap:.4e} | "
                f"grad_norm={gradient_norm:.4e} | "
                f"val_res_p95={validation_metrics['final_residual_p95']:.4e} | "
                f"worst_problem_res_p95={worst_problem_p95:.4e} | "
                f"best_epoch={best_epoch} | elapsed={elapsed:.1f}s"
            )
            if quality_diagnostic_log and quality_diagnostic_log[-1]["epoch"] == epoch_number:
                diagnostic = quality_diagnostic_log[-1]
                message += (
                    f" | one_step_cos={diagnostic['update_ideal_cosine']:.4f}"
                    f" | improve_frac={diagnostic['sample_error_improvement_fraction']:.4f}"
                    f" | contraction_p95={diagnostic['contraction_ratio']['p95']:.4e}"
                )
            print(message)

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

    fixed_point_guarantee = (
        spec.input_mode in {
            "mass_preconditioned_residual",
            "dimensionless_mass_preconditioned_residual",
        }
        and not spec.use_bias
    )
    report = {
        "ablation_spec": asdict(spec),
        "config": {
            "experiment_name": experiment_name,
            "torch_dtype": str(TORCH_DTYPE),
            "device": str(device),
            "architecture": f"{spec.input_dimension}->64->identity->6",
            "activation": ACTIVATION_NAME,
            "optimizer": OPTIMIZER_NAME,
            "learning_rate": LEARNING_RATE,
            "input_mode": spec.input_mode,
            "raw_input_standardization": spec.input_mode == "raw_state_standardized",
            "raw_input_mean": tensor_to_list(raw_mean) if raw_mean is not None else None,
            "raw_input_std": tensor_to_list(raw_std) if raw_std is not None else None,
            "residual_length_scale": config.residual_length_scale,
            "output_scale_mode": spec.output_scale_mode,
            "use_bias": spec.use_bias,
            "first_layer_initialization": spec.first_layer_initialization,
            "output_layer_initialization": "zero weights and zero bias when present",
            "strict_zero_residual_fixed_point": fixed_point_guarantee,
            "shift_energy": spec.shift_energy,
            "use_energy_scale": spec.use_energy_scale,
            "characteristic_energy_scale": characteristic_energy_scale,
            "objective_divisor": objective_divisor,
            "use_gradient_clip": spec.use_gradient_clip,
            "gradient_clip_norm": config.gradient_clip_norm if spec.use_gradient_clip else None,
            "epochs_requested": config.epochs,
            "completed_epochs": len(train_log),
            "validation_interval": config.validation_interval,
            "diagnostic_interval": config.diagnostic_interval,
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
        "quality_diagnostic_log": quality_diagnostic_log,
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
            title=f"{experiment_name}: current-state and exact fixed-point tests",
            save_path=experiment_dir / "special_state_metrics_vs_physical_time.png",
        )

    clipped_count = sum(bool(record["gradient_was_clipped"]) for record in train_log)
    finite_gradients = [
        float(record["gradient_norm_before_clip"])
        for record in train_log
        if math.isfinite(float(record["gradient_norm_before_clip"]))
    ]
    summary = {
        "experiment_name": experiment_name,
        "short_label": spec.short_label,
        "ablation_spec": asdict(spec),
        "training_num_problems": training_cpu.metadata["num_problems"],
        "training_num_points": len(training_cpu),
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": list(best_key) if best_key is not None else None,
        "diverged": diverged,
        "divergence_epoch": divergence_epoch,
        "gradient_clip_trigger_fraction": (
            clipped_count / len(train_log) if train_log and spec.use_gradient_clip else 0.0
        ),
        "maximum_gradient_norm": max(finite_gradients) if finite_gradients else float("nan"),
        "best_checkpoint_test": best_results,
        "last_checkpoint_test": last_results,
        "compact_best_checkpoint_test": {
            name: compact_test_metrics(metrics) for name, metrics in best_results.items()
        },
        "training_curve_for_summary": downsample_log(train_log),
        "quality_diagnostic_log": quality_diagnostic_log,
        "validation_curve_for_summary": downsample_log(validation_log),
    }
    save_json(summary, experiment_dir / "experiment_summary.json")
    return summary


def safe_log10_degradation(value: float, reference: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference):
        return float("nan")
    return math.log10(max(value, PLOT_FLOOR) / max(reference, PLOT_FLOOR))


def build_ablation_summary_rows(
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_name = {record["experiment_name"]: record for record in summaries}
    reference = by_name.get("A3_stable_full")
    rows: list[dict[str, Any]] = []
    for record in summaries:
        spec = record["ablation_spec"]
        interp = record["best_checkpoint_test"]["interpolation_test"]
        extra = record["best_checkpoint_test"]["extrapolation_test"]
        exact_state = record["best_checkpoint_test"]["exact_state_all_test"]
        current_state = record["best_checkpoint_test"]["current_state_all_test"]
        row: dict[str, Any] = {
            "experiment": record["experiment_name"],
            "label": record["short_label"],
            "comparison_role": spec["comparison_role"],
            "changed_factor": spec["changed_factor"],
            "input_mode": spec["input_mode"],
            "use_bias": spec["use_bias"],
            "first_layer_initialization": spec["first_layer_initialization"],
            "output_scale_mode": spec["output_scale_mode"],
            "shift_energy": spec["shift_energy"],
            "use_energy_scale": spec["use_energy_scale"],
            "use_gradient_clip": spec["use_gradient_clip"],
            "best_epoch": record["best_validation_epoch"],
            "diverged": record["diverged"],
            "divergence_epoch": record.get("divergence_epoch"),
            "interpolation_residual_p95": interp["final_residual_p95"],
            "extrapolation_residual_p95": extra["final_residual_p95"],
            "interpolation_exact_error_p95": interp["final_exact_error_p95"],
            "extrapolation_exact_error_p95": extra["final_exact_error_p95"],
            "interpolation_energy_gap_p95": interp["final_energy_gap_p95"],
            "extrapolation_energy_gap_p95": extra["final_energy_gap_p95"],
            "exact_state_final_residual_p95": exact_state["final_residual_p95"],
            "exact_state_final_drift_p95": exact_state["final_exact_error_p95"],
            "current_state_final_residual_p95": current_state["final_residual_p95"],
            "gradient_clip_trigger_fraction": record["gradient_clip_trigger_fraction"],
            "maximum_gradient_norm": record["maximum_gradient_norm"],
        }
        if reference is not None:
            ref_interp = reference["best_checkpoint_test"]["interpolation_test"]
            ref_extra = reference["best_checkpoint_test"]["extrapolation_test"]
            ref_exact = reference["best_checkpoint_test"]["exact_state_all_test"]
            row["interp_residual_log10_degradation_vs_A3"] = safe_log10_degradation(
                interp["final_residual_p95"], ref_interp["final_residual_p95"]
            )
            row["extra_residual_log10_degradation_vs_A3"] = safe_log10_degradation(
                extra["final_residual_p95"], ref_extra["final_residual_p95"]
            )
            row["fixed_point_drift_log10_degradation_vs_A3"] = safe_log10_degradation(
                exact_state["final_exact_error_p95"], ref_exact["final_exact_error_p95"]
            )
        rows.append(row)
    return rows


def save_ablation_summary_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 8. Command-line interface and main experiment orchestration
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the eight controlled learned-optimizer ablations for the "
            "independent multi-time-step two-particle spring problem."
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
        "--residual-length-scale",
        type=float,
        default=DEFAULT_RESIDUAL_LENGTH_SCALE,
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_NORM,
    )
    parser.add_argument(
        "--diagnostic-interval",
        type=int,
        default=DEFAULT_DIAGNOSTIC_INTERVAL,
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(DEFAULT_EXPERIMENT_NAMES),
        choices=list(DEFAULT_EXPERIMENT_NAMES),
        help="Run all eight by default, or provide a selected subset in order.",
    )
    parser.add_argument(
        "--skip-newton-baseline",
        action="store_true",
        help="Skip the shared full-Newton evaluation.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--save-datasets",
        action="store_true",
        help="Save generated tensor datasets. Off by default.",
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
    if float(args.residual_length_scale) <= 0.0:
        raise ValueError("residual_length_scale must be positive.")
    if float(args.gradient_clip_norm) <= 0.0:
        raise ValueError("gradient_clip_norm must be positive.")
    if int(args.diagnostic_interval) <= 0:
        raise ValueError("diagnostic_interval must be positive.")

    experiment_names = tuple(dict.fromkeys(str(name) for name in args.experiments))
    unknown = [name for name in experiment_names if name not in EXPERIMENT_SPEC_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown experiments: {unknown}")

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
        residual_length_scale=float(args.residual_length_scale),
        gradient_clip_norm=float(args.gradient_clip_norm),
        diagnostic_interval=int(args.diagnostic_interval),
        device=str(args.device),
        experiment_names=experiment_names,
        run_newton_baseline=not bool(args.skip_newton_baseline),
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
    selected_specs = [EXPERIMENT_SPEC_BY_NAME[name] for name in config.experiment_names]

    print(f"Output directory: {output_dir}")
    print(f"Runtime config: {asdict(config)}")
    print(f"Physical config: {asdict(physical)}")
    print(f"torch default dtype: {torch.get_default_dtype()}")
    print("Selected experiments: " + ", ".join(config.experiment_names))
    print(
        "Problem split sizes: "
        f"train={len(split.train_indices)}, validation={len(split.validation_indices)}, "
        f"interpolation_test={len(split.interpolation_test_indices)}, "
        f"extrapolation_test={len(split.extrapolation_test_indices)}"
    )

    save_json(
        {
            "runtime_config": asdict(config),
            "ablation_specs": [asdict(spec) for spec in selected_specs],
            "shared_configuration": {
                "optimizer": "Adam",
                "learning_rate": LEARNING_RATE,
                "hidden_width": 64,
                "activation": ACTIVATION_NAME,
                "output_layer_initialization": "zero",
                "torch_dtype": str(TORCH_DTYPE),
                "model_random_seed": MODEL_RANDOM_SEED,
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

    training = build_dataset_for_problem_indices(
        problems=problems,
        indices=split.train_indices,
        points_per_problem=config.train_points_per_problem,
        base_seed=TRAIN_SOBOL_SEED,
        role="multi_problem_training",
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
        "training": training.metadata,
        "validation": validation.metadata,
        "interpolation_test": interpolation_test.metadata,
        "extrapolation_test": extrapolation_test.metadata,
        "current_state_all_test": current_state_all_test.metadata,
        "exact_state_all_test": exact_state_all_test.metadata,
        "same_datasets_for_all_ablation_groups": True,
        "no_problem_leakage": True,
    }
    save_json(dataset_metadata, output_dir / "dataset_metadata.json")

    if config.save_datasets:
        torch.save(
            {
                "training": dataset_to_serializable_dict(training),
                "validation": dataset_to_serializable_dict(validation),
                "interpolation_test": dataset_to_serializable_dict(interpolation_test),
                "extrapolation_test": dataset_to_serializable_dict(extrapolation_test),
                "current_state_all_test": dataset_to_serializable_dict(current_state_all_test),
                "exact_state_all_test": dataset_to_serializable_dict(exact_state_all_test),
            },
            output_dir / "generated_datasets.pt",
        )

    newton_results: dict[str, Any] | None = None
    newton_report: dict[str, Any] | None = None
    if config.run_newton_baseline:
        newton_results = evaluate_newton_baseline(
            datasets=evaluation_datasets,
            physical=physical,
            config=config,
            device=device,
        )
        newton_dir = output_dir / "newton_baseline"
        newton_dir.mkdir(parents=True, exist_ok=True)
        newton_report = {
            "solver": "undamped full Newton",
            "gradient": "analytic gradient of the original variational energy",
            "hessian": "analytic 6x6 Hessian of the original variational energy",
            "uses_exact_solution_in_update": False,
            "residual_tolerance": NEWTON_RESIDUAL_TOLERANCE,
            "fixed_iterations": config.evaluation_steps,
            "report_steps": list(config.report_steps),
            "results": newton_results,
            "compact_results": {
                name: compact_test_metrics(metrics)
                for name, metrics in newton_results.items()
            },
        }
        save_json(newton_report, newton_dir / "newton_baseline_report.json")
        if not config.skip_plots:
            for split_name in ["interpolation_test", "extrapolation_test"]:
                plot_rollout_metrics(
                    metrics=newton_results[split_name],
                    title=f"full Newton: {split_name}",
                    save_path=newton_dir / f"{split_name}_rollout_metrics.png",
                )

    summaries: list[dict[str, Any]] = []
    for spec in selected_specs:
        summaries.append(
            run_experiment(
                spec=spec,
                training_cpu=training,
                validation_cpu=validation,
                evaluation_datasets=evaluation_datasets,
                output_dir=output_dir,
                config=config,
                physical=physical,
                problems=problems,
            )
        )

    ablation_rows = build_ablation_summary_rows(summaries)
    save_ablation_summary_csv(ablation_rows, output_dir / "ablation_summary.csv")
    save_json({"rows": ablation_rows}, output_dir / "ablation_summary.json")

    overall_report = {
        "experiment_type": "controlled_eight_group_ablation",
        "purpose": (
            "Identify which changes from the failed absolute-state solver to the "
            "stable residual solver produce the dominant improvement."
        ),
        "runtime_config": asdict(config),
        "physical_config": asdict(physical),
        "problem_split": asdict(split),
        "dataset_metadata": dataset_metadata,
        "shared_training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "full_batch": True,
            "model_seed": MODEL_RANDOM_SEED,
            "epochs": config.epochs,
            "K_schedule": {
                "initial": config.initial_k,
                "increase_interval": config.k_increase_interval,
                "increase_amount": config.k_increase_amount,
                "maximum": config.max_k,
            },
            "validation_checkpoint_selection": True,
            "no_early_stopping": True,
        },
        "ablation_specs": [asdict(spec) for spec in selected_specs],
        "newton_baseline": newton_report,
        "experiments": summaries,
        "ablation_summary_rows": ablation_rows,
    }
    save_json(overall_report, output_dir / "all_experiments_summary.json")

    if not config.skip_plots and summaries:
        plot_model_comparison_final_metrics(
            summaries=summaries,
            save_path=output_dir / "ablation_final_metrics.png",
        )
        plot_current_state_residual_model_comparison(
            summaries=summaries,
            problems=problems,
            problem_indices=split.all_test_indices,
            report_steps=config.report_steps,
            save_path=output_dir / "ablation_current_state_residual.png",
        )
        if newton_results is not None:
            plot_all_solver_final_metrics(
                summaries=summaries,
                newton_results=newton_results,
                save_path=output_dir / "ablation_and_newton_final_metrics.png",
            )
            plot_current_state_residual_all_solvers(
                summaries=summaries,
                newton_results=newton_results,
                problems=problems,
                problem_indices=split.all_test_indices,
                report_steps=config.report_steps,
                save_path=output_dir / "ablation_current_state_with_newton.png",
            )
            for record in summaries:
                experiment_dir = output_dir / record["experiment_name"]
                for split_name in ["interpolation_test", "extrapolation_test"]:
                    plot_learned_vs_newton_rollout(
                        learned_metrics=record["best_checkpoint_test"][split_name],
                        newton_metrics=newton_results[split_name],
                        learned_name=record["experiment_name"],
                        title=f"{record['experiment_name']} vs full Newton: {split_name}",
                        save_path=experiment_dir / f"{split_name}_learned_vs_newton.png",
                    )

    print("\n" + "=" * 100)
    print("All requested ablation experiments completed.")
    print(f"Summary CSV: {output_dir / 'ablation_summary.csv'}")
    print(f"Summary JSON: {output_dir / 'all_experiments_summary.json'}")
    for row in ablation_rows:
        print(
            f"- {row['experiment']}: best_epoch={row['best_epoch']}, "
            f"interp_res_p95={row['interpolation_residual_p95']:.4e}, "
            f"extra_res_p95={row['extrapolation_residual_p95']:.4e}, "
            f"diverged={row['diverged']}"
        )



if __name__ == "__main__":
    main()
