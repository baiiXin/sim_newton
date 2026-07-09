"""
Fixed-left-edge 5x5 triangular-cloth learned-optimizer ablation.

This script evaluates activation in {identity, ReLU, Tanh}, hidden depth 1,
and hidden width 256.  In this variant, all linear layers are bias-free.  The
default device is cuda:0.

The learned optimizer uses a Metamizer-inspired one-step history input.  At
iteration k, the MLP receives three full-cloth 75D channels concatenated into
225D:
    1. current full mass-preconditioned residual u_k / s,
    2. previous full mass-preconditioned residual u_{k-1} / s,
    3. previous applied full optimizer displacement delta_y_{k-1} / s.
The two fixed vertices are included in the MLP input and output tensors.  After
the MLP output is produced, the fixed-vertex update components are discarded;
only the 69D free-vertex update is applied to the physical state.  History is
zero-initialized for every independent optimization problem and is carried only
within that problem's iterative solve.  Stored history tensors are detached
between iterations, while the state y remains differentiable through the
complete K-step rollout.  No residual-threshold update guard is used.

Training uses 8192 full-batch states, float64, Adam(lr=1e-3), 5000 epochs,
K={1, 3, 5, 10, 30} for 1000 epochs each, validation every 200 epochs,
and gradient clipping with maximum norm 10.  The loss is averaged over K.
Reference solutions are used only for sampling, validation, and evaluation.
Every nn.Linear weight matrix, including the output layer, uses the standard
PyTorch default initialization.  After all requested model configurations
finish, the script writes ranked CSV/Markdown summaries and aggregate plots.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# 0. Constants and triangular cloth topology
# =============================================================================

GRID_ROWS = 5
GRID_COLS = 5
SPATIAL_DIM = 3
NUM_PARTICLES = GRID_ROWS * GRID_COLS
FIXED_VERTEX_INDICES = (0, (GRID_ROWS - 1) * GRID_COLS)  # left-top and left-bottom
FREE_VERTEX_INDICES = tuple(
    index for index in range(NUM_PARTICLES) if index not in set(FIXED_VERTEX_INDICES)
)
NUM_FREE_PARTICLES = len(FREE_VERTEX_INDICES)
FREE_STATE_DIM = NUM_FREE_PARTICLES * SPATIAL_DIM
FULL_STATE_DIM = NUM_PARTICLES * SPATIAL_DIM
HISTORY_INPUT_CHANNELS = 3
MODEL_INPUT_CHANNEL_DIM = FULL_STATE_DIM
MODEL_INPUT_DIM = HISTORY_INPUT_CHANNELS * MODEL_INPUT_CHANNEL_DIM
MODEL_OUTPUT_DIM = FULL_STATE_DIM


def grid_index(row: int, col: int) -> int:
    return row * GRID_COLS + col


def build_triangular_cloth_topology() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Return unique spring edges and triangle faces for an alternating 5x5 mesh."""
    edge_set: set[tuple[int, int]] = set()
    faces: list[tuple[int, int, int]] = []

    def add_edge(a: int, b: int) -> None:
        if a == b:
            raise ValueError("Degenerate edge")
        edge_set.add((min(a, b), max(a, b)))

    # Horizontal and vertical structural mesh edges.
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS - 1):
            add_edge(grid_index(row, col), grid_index(row, col + 1))
    for row in range(GRID_ROWS - 1):
        for col in range(GRID_COLS):
            add_edge(grid_index(row, col), grid_index(row + 1, col))

    # One alternating diagonal per cell: this is the triangle-mesh spring network.
    for row in range(GRID_ROWS - 1):
        for col in range(GRID_COLS - 1):
            tl = grid_index(row, col)
            tr = grid_index(row, col + 1)
            bl = grid_index(row + 1, col)
            br = grid_index(row + 1, col + 1)
            if (row + col) % 2 == 0:
                add_edge(tl, br)
                faces.append((tl, tr, br))
                faces.append((tl, br, bl))
            else:
                add_edge(bl, tr)
                faces.append((tl, tr, bl))
                faces.append((tr, br, bl))

    edges = tuple(sorted(edge_set))
    faces_tuple = tuple(faces)
    expected_edges = GRID_ROWS * (GRID_COLS - 1) + (GRID_ROWS - 1) * GRID_COLS + (GRID_ROWS - 1) * (GRID_COLS - 1)
    expected_faces = 2 * (GRID_ROWS - 1) * (GRID_COLS - 1)
    if len(edges) != expected_edges:
        raise RuntimeError(f"Expected {expected_edges} spring edges, got {len(edges)}")
    if len(faces_tuple) != expected_faces:
        raise RuntimeError(f"Expected {expected_faces} triangle faces, got {len(faces_tuple)}")
    return edges, faces_tuple


SPRING_EDGES, TRIANGLE_FACES = build_triangular_cloth_topology()
NUM_SPRINGS = len(SPRING_EDGES)
NUM_TRIANGLES = len(TRIANGLE_FACES)
GLOBAL_TO_FREE_INDEX = tuple(
    FREE_VERTEX_INDICES.index(index) if index in FREE_VERTEX_INDICES else -1
    for index in range(NUM_PARTICLES)
)


TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

ACTIVATION_NAMES = ("identity", "relu", "tanh")
HIDDEN_DEPTHS = (1,)
HIDDEN_WIDTHS = (256,)
USE_BIAS = False
OPTIMIZER_NAME = "adam"
LEARNING_RATE = 1e-3
DEFAULT_DEVICE = "cuda:0"

DEFAULT_TOTAL_TIME_STEPS = 100
DEFAULT_TRAIN_POINTS_PER_PROBLEM = 32
DEFAULT_EVAL_POINTS_PER_PROBLEM = 128
DEFAULT_EPOCHS = 5000
DEFAULT_VALIDATION_INTERVAL = 200
DEFAULT_DIAGNOSTIC_INTERVAL = 200
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8192
DEFAULT_K_VALUES = (1, 3, 5, 10, 30)
DEFAULT_EPOCHS_PER_K = 1000
DEFAULT_REPORT_STEPS = (1, 3, 5, 10, 30, 50)
DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_GRADIENT_CLIP_NORM = 10.0
DEFAULT_SAMPLING_RADIUS_MIN = 1e-2
DEFAULT_SAMPLING_RADIUS_MAX = 1e-1

TRAIN_TIME_INDICES = (0, 5, 11, 16, 21, 26, 32, 37, 42, 47, 53, 58, 63, 68, 74, 79)
SEEN_INTERPOLATION_TIME_INDICES = (2, 8, 13, 18, 24, 29, 34, 39, 45, 50, 55, 61, 66, 71, 76, 78)
SEEN_EXTRAPOLATION_TIME_INDICES = tuple(range(80, 100))
VALIDATION_TIME_INDICES = (4, 14, 24, 34, 44, 54, 64, 74, 84, 94)
UNSEEN_TEST_TIME_INDICES = tuple(range(0, 100, 5))

MODEL_RANDOM_SEED = 42
MOTION_SOBOL_SEED_TRAIN = 20260630
MOTION_SOBOL_SEED_VALIDATION = 20260701
MOTION_SOBOL_SEED_ID_TEST = 20260702
TRAIN_SOBOL_SEED = 20260620
VALIDATION_SOBOL_SEED = 20260621
SEEN_INTERPOLATION_TEST_SOBOL_SEED = 20260622
SEEN_EXTRAPOLATION_TEST_SOBOL_SEED = 20260623
UNSEEN_ID_TEST_SOBOL_SEED = 20260624
OOD_TEST_SOBOL_SEED = 20260625

GD_CANDIDATE_STEP_SIZES = (
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
)

PLOT_FLOOR = 1e-16
DISTANCE_EPS = 1e-12
NEWTON_RESIDUAL_TOLERANCE = 1e-10
REFERENCE_RESIDUAL_TOLERANCE = 1e-11
REFERENCE_ACCEPTABLE_RESIDUAL = 1e-8
REFERENCE_MAX_ITERATIONS = 100
REFERENCE_LINE_SEARCH_MIN_ALPHA = 2.0**-30


# =============================================================================
# 1. Data structures, physical configuration, and motion catalogue
# =============================================================================


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
    k_values: tuple[int, ...]
    epochs_per_k: int
    report_steps: tuple[int, ...]
    residual_length_scale: float
    gradient_clip_norm: float
    sampling_radius_min: float
    sampling_radius_max: float
    device: str
    activations: tuple[str, ...]
    depths: tuple[int, ...]
    widths: tuple[int, ...]
    config_index: int | None
    list_configs: bool
    skip_completed: bool
    resume: bool
    skip_plots: bool
    save_datasets: bool


@dataclass(frozen=True)
class ModelSpec:
    activation: str
    depth: int
    width: int
    use_bias: bool

    @property
    def experiment_name(self) -> str:
        bias_name = "with_bias" if self.use_bias else "no_bias"
        return (
            f"history_input_default_init_activation_{self.activation}_depth_{self.depth:02d}_"
            f"width_{self.width:03d}_{bias_name}"
        )


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


def create_output_directory() -> Path:
    output_dir = Path(__file__).resolve().parent / Path(__file__).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
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
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(data), f, indent=2, ensure_ascii=False)


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested cuda:{index}, but only {torch.cuda.device_count()} devices are visible")


def get_k_for_epoch(epoch_index: int, config: RuntimeConfig) -> int:
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    stage_index = min(epoch_index // config.epochs_per_k, len(config.k_values) - 1)
    return int(config.k_values[stage_index])


def build_model_specs(config: RuntimeConfig) -> list[ModelSpec]:
    specs = [
        ModelSpec(activation=activation, depth=depth, width=width, use_bias=USE_BIAS)
        for activation in config.activations
        for depth in config.depths
        for width in config.widths
    ]
    if config.config_index is not None:
        if config.config_index < 0 or config.config_index >= len(specs):
            raise ValueError(
                f"config-index must be in [0, {len(specs) - 1}], got {config.config_index}"
            )
        specs = [specs[config.config_index]]
    return specs



def finite_plot_value(value: float | int | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return float("nan")
    return max(float(value), PLOT_FLOOR)


# =============================================================================
# 2. Fixed-left-edge 5x5 triangular-cloth physics
# =============================================================================


def free_vertex_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(FREE_VERTEX_INDICES, dtype=torch.long, device=device)


def fixed_vertex_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(FIXED_VERTEX_INDICES, dtype=torch.long, device=device)


def spring_edge_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(SPRING_EDGES, dtype=torch.long, device=device)


def triangle_face_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(TRIANGLE_FACES, dtype=torch.long, device=device)


def reshape_free(y: torch.Tensor) -> torch.Tensor:
    if y.shape[-1] != FREE_STATE_DIM:
        raise ValueError(f"Expected final dimension {FREE_STATE_DIM}, got {tuple(y.shape)}")
    return y.reshape(*y.shape[:-1], NUM_FREE_PARTICLES, SPATIAL_DIM)


def full_positions_from_free(y: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    free = reshape_free(y)
    leading_shape = free.shape[:-2]
    base = torch.as_tensor(physical.p0, dtype=y.dtype, device=y.device)
    view_shape = (*([1] * len(leading_shape)), NUM_PARTICLES, SPATIAL_DIM)
    full = base.reshape(view_shape).expand(*leading_shape, NUM_PARTICLES, SPATIAL_DIM).clone()
    full[..., list(FREE_VERTEX_INDICES), :] = free
    return full


def free_state_from_full(full: torch.Tensor) -> torch.Tensor:
    if full.shape[-2:] != (NUM_PARTICLES, SPATIAL_DIM):
        raise ValueError(f"Expected (..., {NUM_PARTICLES}, {SPATIAL_DIM}), got {tuple(full.shape)}")
    return full[..., list(FREE_VERTEX_INDICES), :].reshape(*full.shape[:-2], FREE_STATE_DIM)


def flatten_full_state(full: torch.Tensor) -> torch.Tensor:
    if full.shape[-2:] != (NUM_PARTICLES, SPATIAL_DIM):
        raise ValueError(f"Expected (..., {NUM_PARTICLES}, {SPATIAL_DIM}), got {tuple(full.shape)}")
    return full.reshape(*full.shape[:-2], FULL_STATE_DIM)


def free_state_from_flat_full(full_flat: torch.Tensor) -> torch.Tensor:
    if full_flat.shape[-1] != FULL_STATE_DIM:
        raise ValueError(f"Expected final dimension {FULL_STATE_DIM}, got {tuple(full_flat.shape)}")
    return full_flat.reshape(*full_flat.shape[:-1], NUM_PARTICLES, SPATIAL_DIM)[
        ..., list(FREE_VERTEX_INDICES), :
    ].reshape(*full_flat.shape[:-1], FREE_STATE_DIM)


def flat_full_update_from_free_update(free_update: torch.Tensor) -> torch.Tensor:
    if free_update.shape[-1] != FREE_STATE_DIM:
        raise ValueError(f"Expected final dimension {FREE_STATE_DIM}, got {tuple(free_update.shape)}")
    full = torch.zeros(*free_update.shape[:-1], NUM_PARTICLES, SPATIAL_DIM, dtype=free_update.dtype, device=free_update.device)
    full[..., list(FREE_VERTEX_INDICES), :] = free_update.reshape(*free_update.shape[:-1], NUM_FREE_PARTICLES, SPATIAL_DIM)
    return flatten_full_state(full)


def spring_vectors_from_free(y: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    points = full_positions_from_free(y, physical)
    edges = spring_edge_tensor(y.device)
    return points[..., edges[:, 1], :] - points[..., edges[:, 0], :]


def spring_lengths_from_free(y: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    return torch.linalg.vector_norm(spring_vectors_from_free(y, physical), dim=-1)


def variational_energy(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """Implicit-Euler variational energy for the 23 free cloth vertices."""
    free = reshape_free(y)
    q_free = reshape_free(q)
    if masses.shape[-1] != NUM_FREE_PARTICLES:
        raise ValueError(f"masses must contain {NUM_FREE_PARTICLES} free-particle masses")

    inertial = (masses / (2.0 * physical.dt**2)) * torch.sum(
        (free - q_free) ** 2, dim=-1
    )
    lengths = spring_lengths_from_free(y, physical)
    stiffness = torch.as_tensor(
        physical.spring_stiffness, dtype=y.dtype, device=y.device
    )
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    spring = 0.5 * stiffness * (lengths - rest) ** 2
    return torch.sum(inertial, dim=-1) + torch.sum(spring, dim=-1)


def full_stationarity_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """Return a 75D residual-like force vector before eliminating fixed vertices.

    The inertial gradient is defined only for the 23 free vertices because the
    two fixed vertices are constrained variables.  Spring forces are accumulated
    on all 25 vertices, so fixed vertices can still provide boundary-force
    information to the learned optimizer input.  The free slice of this full
    vector is exactly the eliminated 69D stationarity residual used by the
    physical optimizer.
    """
    free = reshape_free(y)
    q_free = reshape_free(q)
    full = full_positions_from_free(y, physical)

    grad_free = (masses[..., :, None] / physical.dt**2) * (free - q_free)
    full_grad = torch.zeros_like(full)
    full_grad[..., list(FREE_VERTEX_INDICES), :] = grad_free

    edges = spring_edge_tensor(y.device)
    edge_vectors = full[..., edges[:, 1], :] - full[..., edges[:, 0], :]
    lengths = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True).clamp_min(
        DISTANCE_EPS
    )
    stiffness = torch.as_tensor(
        physical.spring_stiffness, dtype=y.dtype, device=y.device
    )
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    parameter_shape = [1] * (edge_vectors.ndim - 2) + [NUM_SPRINGS, 1]
    edge_grad = (
        stiffness.reshape(parameter_shape)
        * (1.0 - rest.reshape(parameter_shape) / lengths)
        * edge_vectors
    )
    full_grad = full_grad.clone()
    full_grad.index_add_(-2, edges[:, 0], -edge_grad)
    full_grad.index_add_(-2, edges[:, 1], edge_grad)
    return flatten_full_state(full_grad)


def stationarity_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    return free_state_from_flat_full(full_stationarity_residual(y, q, masses, physical))


def stationarity_residual_norm(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    return torch.linalg.vector_norm(
        stationarity_residual(y, q, masses, physical), dim=-1
    )


def variational_hessian(
    y: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """Analytic Hessian after eliminating the two fixed corner vertices."""
    full = full_positions_from_free(y, physical)
    edges = spring_edge_tensor(y.device)
    edge_vectors = full[..., edges[:, 1], :] - full[..., edges[:, 0], :]
    lengths = torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True).clamp_min(
        DISTANCE_EPS
    )
    identity = torch.eye(SPATIAL_DIM, dtype=y.dtype, device=y.device)
    outer = edge_vectors.unsqueeze(-1) * edge_vectors.unsqueeze(-2)
    stiffness = torch.as_tensor(
        physical.spring_stiffness, dtype=y.dtype, device=y.device
    )
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
    parameter_shape = [1] * (edge_vectors.ndim - 2) + [NUM_SPRINGS, 1, 1]
    lengths_matrix = lengths.unsqueeze(-1)
    spring_blocks = stiffness.reshape(parameter_shape) * (
        (1.0 - rest.reshape(parameter_shape) / lengths_matrix) * identity
        + (rest.reshape(parameter_shape) / lengths_matrix.pow(3)) * outer
    )

    hessian = torch.zeros(
        (*y.shape[:-1], FREE_STATE_DIM, FREE_STATE_DIM),
        dtype=y.dtype,
        device=y.device,
    )
    for free_index in range(NUM_FREE_PARTICLES):
        block = slice(free_index * SPATIAL_DIM, (free_index + 1) * SPATIAL_DIM)
        hessian[..., block, block] += (
            masses[..., free_index] / physical.dt**2
        )[..., None, None] * identity

    for edge_index, (left_global, right_global) in enumerate(SPRING_EDGES):
        left_free = GLOBAL_TO_FREE_INDEX[left_global]
        right_free = GLOBAL_TO_FREE_INDEX[right_global]
        block = spring_blocks[..., edge_index, :, :]
        if left_free >= 0:
            left = slice(left_free * SPATIAL_DIM, (left_free + 1) * SPATIAL_DIM)
            hessian[..., left, left] += block
        if right_free >= 0:
            right = slice(right_free * SPATIAL_DIM, (right_free + 1) * SPATIAL_DIM)
            hessian[..., right, right] += block
        if left_free >= 0 and right_free >= 0:
            hessian[..., left, right] -= block
            hessian[..., right, left] -= block
    return hessian


def apply_newton_update(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_tolerance: float = NEWTON_RESIDUAL_TOLERANCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = stationarity_residual(y, q, masses, physical)
    residual = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True)
    rhs = torch.where(residual > residual_tolerance, -gradient, torch.zeros_like(gradient))
    hessian = variational_hessian(y, masses, physical)
    delta_col, info = torch.linalg.solve_ex(hessian, rhs.unsqueeze(-1))
    delta = delta_col.squeeze(-1)
    failed = info != 0
    if bool(torch.any(failed)):
        delta[failed] = torch.matmul(
            torch.linalg.pinv(hessian[failed]), rhs[failed].unsqueeze(-1)
        ).squeeze(-1)
    if not bool(torch.isfinite(delta).all()):
        raise RuntimeError("Newton update produced non-finite values")
    return y + delta, delta


def apply_gradient_descent_update(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = -float(step_size) * stationarity_residual(y, q, masses, physical)
    return y + delta, delta


def solve_reference_solution(
    *,
    q: torch.Tensor,
    masses: torch.Tensor,
    initial_y: torch.Tensor,
    physical: PhysicalConfig,
    residual_tolerance: float = REFERENCE_RESIDUAL_TOLERANCE,
    max_iterations: int = REFERENCE_MAX_ITERATIONS,
    raise_on_nonconvergence: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Damped Newton reference solve with a best-finite-iterate fallback.

    With ``raise_on_nonconvergence=False`` this routine never rejects a finite
    best iterate merely because the strict residual target was missed or the
    Armijo line search stalled. This behavior is used by continuous rollout.
    """
    y = initial_y.detach().clone().reshape(1, FREE_STATE_DIM)
    q_batch = q.detach().clone().reshape(1, FREE_STATE_DIM)
    masses_batch = masses.detach().clone().reshape(1, NUM_FREE_PARTICLES)
    if not bool(torch.isfinite(y).all() and torch.isfinite(q_batch).all() and torch.isfinite(masses_batch).all()):
        raise RuntimeError("Reference solver received a non-finite input state")

    reductions = 0
    status = "max_iterations"
    best_y = y.clone()
    best_residual = float("inf")
    best_energy = float("inf")
    best_iteration = 0

    for iteration in range(max_iterations + 1):
        gradient = stationarity_residual(y, q_batch, masses_batch, physical)
        residual = float(torch.linalg.vector_norm(gradient).item())
        energy = float(variational_energy(y, q_batch, masses_batch, physical).item())
        if math.isfinite(residual) and math.isfinite(energy) and residual < best_residual:
            best_y = y.clone()
            best_residual = residual
            best_energy = energy
            best_iteration = iteration
        if residual <= residual_tolerance:
            return y.squeeze(0), {
                "iterations": iteration,
                "residual_norm": residual,
                "line_search_reductions": reductions,
                "converged": True,
                "acceptable": True,
                "status": "converged",
                "used_best_finite_iterate": False,
                "best_iteration": iteration,
            }
        if iteration == max_iterations:
            break

        hessian = variational_hessian(y, masses_batch, physical)
        direction_col, info = torch.linalg.solve_ex(hessian, -gradient.unsqueeze(-1))
        direction = direction_col.squeeze(-1)
        if bool(torch.any(info != 0)) or not bool(torch.isfinite(direction).all()):
            direction = torch.matmul(torch.linalg.pinv(hessian), -gradient.unsqueeze(-1)).squeeze(-1)

        directional_derivative = float(torch.sum(gradient * direction).item())
        if not math.isfinite(directional_derivative) or directional_derivative >= 0.0:
            mass_per_coordinate = masses_batch.repeat_interleave(SPATIAL_DIM, dim=-1)
            direction = -physical.dt**2 * gradient / mass_per_coordinate
            directional_derivative = float(torch.sum(gradient * direction).item())
        if not bool(torch.isfinite(direction).all()):
            status = "nonfinite_direction"
            break

        if float(torch.linalg.vector_norm(direction).item()) <= 1e-12:
            status = "tiny_direction"
            break

        energy_before = energy
        alpha = 1.0
        accepted = False
        while alpha >= REFERENCE_LINE_SEARCH_MIN_ALPHA:
            candidate = y + alpha * direction
            if bool(torch.isfinite(candidate).all()) and bool(
                torch.all(spring_lengths_from_free(candidate, physical) > DISTANCE_EPS)
            ):
                candidate_energy = float(
                    variational_energy(candidate, q_batch, masses_batch, physical).item()
                )
                if math.isfinite(candidate_energy) and candidate_energy <= (
                    energy_before + 1e-4 * alpha * directional_derivative
                ):
                    y = candidate
                    accepted = True
                    break
            alpha *= 0.5
            reductions += 1
        if not accepted:
            status = "line_search_failed"
            break

    acceptable = best_residual <= REFERENCE_ACCEPTABLE_RESIDUAL
    if not math.isfinite(best_residual):
        raise RuntimeError("Reference solver did not produce any finite iterate")
    if not acceptable and raise_on_nonconvergence:
        raise RuntimeError(
            f"Reference solver failed with status={status}: best residual {best_residual:.6e}"
        )
    return best_y.squeeze(0), {
        "iterations": max_iterations if status == "max_iterations" else best_iteration,
        "residual_norm": best_residual,
        "best_energy": best_energy,
        "line_search_reductions": reductions,
        "converged": False,
        "acceptable": acceptable,
        "status": status,
        "used_best_finite_iterate": True,
        "best_iteration": best_iteration,
    }


def make_q_free(
    p_full: torch.Tensor,
    v_full: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    gravity = torch.tensor([0.0, 0.0, physical.g], dtype=p_full.dtype, device=p_full.device)
    free = list(FREE_VERTEX_INDICES)
    q_free_points = (
        p_full[free, :] + physical.dt * v_full[free, :] - physical.dt**2 * gravity
    )
    return q_free_points.reshape(FREE_STATE_DIM)


def advance_physical_state(
    p_full: torch.Tensor,
    y_next_free: torch.Tensor,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    next_p = full_positions_from_free(y_next_free, physical).reshape(NUM_PARTICLES, SPATIAL_DIM)
    next_v = torch.zeros_like(next_p)
    free = list(FREE_VERTEX_INDICES)
    fixed = list(FIXED_VERTEX_INDICES)
    next_v[free, :] = (next_p[free, :] - p_full[free, :]) / physical.dt
    fixed_positions = torch.as_tensor(physical.p0, dtype=next_p.dtype, device=next_p.device)[fixed]
    next_p[fixed, :] = fixed_positions
    next_v[fixed, :] = 0.0
    return next_p, next_v


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


# =============================================================================
# 3. Motion-isolated dataset generation
# =============================================================================


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



# =============================================================================
# 4. Learned optimizer
# =============================================================================


def mass_preconditioned_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    """Return the full 75D mass-preconditioned residual used as MLP input."""
    residual = full_stationarity_residual(y, q, masses, physical)
    full_masses = torch.as_tensor(
        physical.masses, dtype=y.dtype, device=y.device
    ).reshape(*([1] * (residual.ndim - 1)), NUM_PARTICLES)
    full_masses = full_masses.expand(*residual.shape[:-1], NUM_PARTICLES)
    mass_per_coordinate = full_masses.repeat_interleave(SPATIAL_DIM, dim=-1)
    return physical.dt**2 * residual / mass_per_coordinate


def make_activation(name: str) -> nn.Module:
    if name == "identity":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


@dataclass(frozen=True)
class LearnedOptimizerState:
    """One-step history carried within one independent iterative solve."""

    previous_residual: torch.Tensor
    previous_update: torch.Tensor

    @classmethod
    def zeros_like(cls, y: torch.Tensor) -> "LearnedOptimizerState":
        zeros = torch.zeros(*y.shape[:-1], FULL_STATE_DIM, dtype=y.dtype, device=y.device)
        return cls(previous_residual=zeros, previous_update=zeros.clone())


class MLPOptimizer(nn.Module):
    def __init__(self, residual_length_scale: float, model_spec: ModelSpec) -> None:
        super().__init__()
        if residual_length_scale <= 0.0:
            raise ValueError("residual_length_scale must be positive")
        if model_spec.depth <= 0 or model_spec.width <= 0:
            raise ValueError("depth and width must be positive")
        if model_spec.activation not in ACTIVATION_NAMES:
            raise ValueError(f"Unsupported activation: {model_spec.activation}")

        self.model_spec = model_spec
        self.activation = make_activation(model_spec.activation)
        hidden_layers: list[nn.Linear] = []
        input_dim = MODEL_INPUT_DIM
        for _ in range(model_spec.depth):
            hidden_layers.append(
                nn.Linear(input_dim, model_spec.width, bias=model_spec.use_bias)
            )
            input_dim = model_spec.width
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(
            model_spec.width, MODEL_OUTPUT_DIM, bias=model_spec.use_bias
        )

        # Keep the standard initialization performed by nn.Linear.reset_parameters:
        # Kaiming-uniform weights with a=sqrt(5), and the corresponding uniform
        # bias initialization when bias is enabled.  No layer is manually reset.

        self.register_buffer(
            "residual_length_scale",
            torch.tensor(float(residual_length_scale), dtype=TORCH_DTYPE),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def architecture_description(self) -> str:
        bias_text = "bias" if self.model_spec.use_bias else "no bias"
        return (
            f"3 x {MODEL_INPUT_CHANNEL_DIM}D full-cloth history channels "
            f"(current residual, previous residual, previous applied update) "
            f"concatenated to {MODEL_INPUT_DIM}D -> "
            f"[{self.model_spec.width}, {self.model_spec.activation}] x "
            f"{self.model_spec.depth} -> {MODEL_OUTPUT_DIM}D raw update; "
            f"fixed-vertex components discarded before applying ({bias_text})"
        )

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        optimizer_state: LearnedOptimizerState,
        *,
        physical: PhysicalConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_residual = mass_preconditioned_residual(y, q, masses, physical)
        expected_shape = current_residual.shape
        if optimizer_state.previous_residual.shape != expected_shape:
            raise ValueError(
                "previous_residual shape mismatch: "
                f"expected {tuple(expected_shape)}, got "
                f"{tuple(optimizer_state.previous_residual.shape)}"
            )
        if optimizer_state.previous_update.shape != expected_shape:
            raise ValueError(
                "previous_update shape mismatch: "
                f"expected {tuple(expected_shape)}, got "
                f"{tuple(optimizer_state.previous_update.shape)}"
            )
        h = torch.cat(
            [
                current_residual / self.residual_length_scale,
                optimizer_state.previous_residual / self.residual_length_scale,
                optimizer_state.previous_update / self.residual_length_scale,
            ],
            dim=-1,
        )
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        delta = self.residual_length_scale * self.output_layer(h)
        return delta, current_residual


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    optimizer_state: LearnedOptimizerState,
) -> tuple[torch.Tensor, torch.Tensor, LearnedOptimizerState]:
    raw_delta_full, current_residual = model(
        y, q, masses, optimizer_state, physical=physical
    )
    delta_free = free_state_from_flat_full(raw_delta_full)
    applied_delta_full = flat_full_update_from_free_update(delta_free)
    next_state = LearnedOptimizerState(
        previous_residual=current_residual.detach(),
        previous_update=applied_delta_full.detach(),
    )
    return y + delta_free, delta_free, next_state


def physical_energy_scale(
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_length_scale: float,
) -> float:
    return float(masses.mean().item()) * residual_length_scale**2 / physical.dt**2




# =============================================================================
# 5. Unified pooled, maximum, and worst-motion evaluation
# =============================================================================


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
            mean=float(np.mean(finite)),
            median=float(np.median(finite)),
            p95=float(np.percentile(finite, 95)),
            max=float(np.max(finite)),
        )
    return result


def _state_metrics(
    y: torch.Tensor,
    dataset: DatasetBundle,
    exact_energy: torch.Tensor,
    physical: PhysicalConfig,
) -> dict[str, torch.Tensor]:
    point_errors = torch.linalg.vector_norm(
        reshape_free(y) - reshape_free(dataset.exact_y), dim=-1
    )
    energy = variational_energy(y, dataset.q, dataset.masses, physical)
    return {
        "residual": stationarity_residual_norm(y, dataset.q, dataset.masses, physical),
        "energy_gap": energy - exact_energy,
        "exact_error": torch.linalg.vector_norm(y - dataset.exact_y, dim=-1),
        "particle_mean_error": point_errors.mean(dim=-1),
        "particle_max_error": point_errors.max(dim=-1).values,
        "spring_length_error": torch.mean(
            torch.abs(
                spring_lengths_from_free(y, physical)
                - spring_lengths_from_free(dataset.exact_y, physical)
            ),
            dim=-1,
        ),
        "fixed_vertex_max_error": torch.zeros_like(point_errors[..., 0]),
    }


def _selected_steps(steps: int, report_steps: Sequence[int]) -> list[int]:
    return sorted(set([0, steps, *[s for s in report_steps if 0 <= s <= steps]]))


@torch.no_grad()
def evaluate_solver_on_dataset(
    *,
    solver: str,
    dataset_cpu: DatasetBundle,
    physical: PhysicalConfig,
    steps: int,
    batch_size: int,
    report_steps: Sequence[int],
    device: torch.device,
    model: MLPOptimizer | None = None,
    gd_step_size: float | None = None,
) -> dict[str, Any]:
    if solver not in {"learned", "gradient_descent", "full_newton"}:
        raise ValueError(solver)
    if solver == "learned" and model is None:
        raise ValueError("model is required")
    if solver == "gradient_descent" and gd_step_size is None:
        raise ValueError("gd_step_size is required")
    if model is not None:
        model.eval()

    metric_batches: dict[str, list[torch.Tensor]] = {}
    problem_batches: list[torch.Tensor] = []
    motion_batches: list[torch.Tensor] = []
    time_batches: list[torch.Tensor] = []
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
            motion_index=dataset_cpu.motion_index[start:end],
            time_index=dataset_cpu.time_index[start:end],
            metadata={},
        ).to(device)
        y = batch.initial_y.clone()
        learned_optimizer_state = (
            LearnedOptimizerState.zeros_like(y) if solver == "learned" else None
        )
        exact_energy = variational_energy(batch.exact_y, batch.q, batch.masses, physical)
        step_values: dict[str, list[torch.Tensor]] = {}
        for step in range(steps + 1):
            for name, values in _state_metrics(y, batch, exact_energy, physical).items():
                step_values.setdefault(name, []).append(values.detach().cpu())
            if step == steps:
                break
            if solver == "learned":
                assert model is not None
                assert learned_optimizer_state is not None
                y, _, learned_optimizer_state = apply_model_update(
                    model,
                    y,
                    batch.q,
                    batch.masses,
                    physical,
                    learned_optimizer_state,
                )
            elif solver == "gradient_descent":
                assert gd_step_size is not None
                y, _ = apply_gradient_descent_update(y, batch.q, batch.masses, physical, gd_step_size)
            else:
                y, _ = apply_newton_update(y, batch.q, batch.masses, physical)
        for name, values in step_values.items():
            metric_batches.setdefault(name, []).append(torch.stack(values, dim=1))
        problem_batches.append(batch.problem_index.detach().cpu())
        motion_batches.append(batch.motion_index.detach().cpu())
        time_batches.append(batch.time_index.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    arrays = {name: torch.cat(values, dim=0).numpy().astype(float) for name, values in metric_batches.items()}
    problem_indices = torch.cat(problem_batches).numpy().astype(int)
    motion_indices = torch.cat(motion_batches).numpy().astype(int)
    time_indices = torch.cat(time_batches).numpy().astype(int)
    for values in arrays.values():
        values[~np.isfinite(values)] = np.nan

    result: dict[str, Any] = {
        "solver": solver,
        "steps": steps,
        "num_points": len(dataset_cpu),
        "num_motions": int(np.unique(motion_indices).size),
        "selected_report_steps": _selected_steps(steps, report_steps),
        "elapsed_seconds": elapsed,
        "seconds_per_point_per_iteration": elapsed / max(len(dataset_cpu) * steps, 1),
    }
    if gd_step_size is not None:
        result["gradient_descent_step_size"] = float(gd_step_size)

    for name, values in arrays.items():
        for stat_name in ["mean", "median", "p95", "max", "num_nonfinite"]:
            result[f"{name}_{stat_name}_by_step"] = []
        for step in range(values.shape[1]):
            stats = _statistics(values[:, step])
            for stat_name, value in stats.items():
                result[f"{name}_{stat_name}_by_step"].append(value)
        for stat_name, value in _statistics(values[:, -1]).items():
            result[f"final_{name}_{stat_name}"] = value

    selected = result["selected_report_steps"]
    per_motion: dict[str, Any] = {}
    for motion_index in sorted(np.unique(motion_indices).tolist()):
        mask = motion_indices == motion_index
        record: dict[str, Any] = {
            "motion_index": int(motion_index),
            "num_points": int(mask.sum()),
            "time_indices": sorted(np.unique(time_indices[mask]).astype(int).tolist()),
            "steps": {},
            "final": {},
        }
        for step in selected:
            record["steps"][str(step)] = {
                name: _statistics(values[mask, step]) for name, values in arrays.items()
            }
        record["final"] = {
            name: _statistics(values[mask, -1]) for name, values in arrays.items()
        }
        per_motion[str(motion_index)] = record
    result["per_motion"] = per_motion

    per_problem: dict[str, Any] = {}
    for problem_index in sorted(np.unique(problem_indices).tolist()):
        mask = problem_indices == problem_index
        per_problem[str(problem_index)] = {
            "problem_index": int(problem_index),
            "motion_index": int(motion_indices[mask][0]),
            "time_index": int(time_indices[mask][0]),
            "num_points": int(mask.sum()),
            "final": {name: _statistics(values[mask, -1]) for name, values in arrays.items()},
        }
    result["per_problem"] = per_problem

    worst_motion: dict[str, Any] = {}
    for metric_name in arrays:
        metric_records = []
        for motion_key, record in per_motion.items():
            stats = record["final"][metric_name]
            metric_records.append((int(motion_key), stats))
        finite_p95 = [(m, float(s["p95"])) for m, s in metric_records if math.isfinite(float(s["p95"]))]
        finite_max = [(m, float(s["max"])) for m, s in metric_records if math.isfinite(float(s["max"]))]
        p95_motion, p95_value = max(finite_p95, key=lambda item: item[1]) if finite_p95 else (-1, float("nan"))
        max_motion, max_value = max(finite_max, key=lambda item: item[1]) if finite_max else (-1, float("nan"))
        worst_motion[metric_name] = {
            "p95_motion_index": p95_motion,
            "p95": p95_value,
            "max_motion_index": max_motion,
            "max": max_value,
        }
        result[f"worst_motion_final_{metric_name}_p95"] = p95_value
        result[f"worst_motion_final_{metric_name}_p95_motion_index"] = p95_motion
        result[f"worst_motion_final_{metric_name}_max"] = max_value
        result[f"worst_motion_final_{metric_name}_max_motion_index"] = max_motion
    result["worst_motion"] = worst_motion
    return result


def validation_selection_key(metrics: dict[str, Any]) -> tuple[float, ...] | None:
    values = (
        float(metrics["final_residual_num_nonfinite"]),
        float(metrics["worst_motion_final_residual_max"]),
        float(metrics["worst_motion_final_residual_p95"]),
        float(metrics["final_residual_max"]),
        float(metrics["final_residual_p95"]),
        float(metrics["worst_motion_final_exact_error_max"]),
        float(metrics["final_exact_error_max"]),
        float(metrics["final_exact_error_p95"]),
        float(metrics["final_energy_gap_max"]),
        float(metrics["final_energy_gap_p95"]),
    )
    return values if all(math.isfinite(v) for v in values) else None


def select_gradient_descent_step_size(
    *,
    validation: DatasetBundle,
    physical: PhysicalConfig,
    config: RuntimeConfig,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    best_alpha: float | None = None
    best_key: tuple[float, ...] | None = None
    for alpha in GD_CANDIDATE_STEP_SIZES:
        print(f"Evaluating validation GD step size alpha={alpha:.1e} ...")
        metrics = evaluate_solver_on_dataset(
            solver="gradient_descent",
            dataset_cpu=validation,
            physical=physical,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps,
            device=device,
            gd_step_size=alpha,
        )
        key = validation_selection_key(metrics)
        records.append({"step_size": alpha, "selection_key": key, "metrics": metrics})
        if key is not None and (best_key is None or key < best_key):
            best_key = key
            best_alpha = alpha
    if best_alpha is None:
        raise RuntimeError("No finite gradient-descent candidate was found")
    return best_alpha, {
        "candidate_step_sizes": list(GD_CANDIDATE_STEP_SIZES),
        "selection_rule": (
            "lexicographic: nonfinite count, worst-motion residual max, "
            "worst-motion residual p95, pooled residual max, pooled residual p95, "
            "then exact-error and energy-gap boundary metrics"
        ),
        "selected_step_size": best_alpha,
        "selected_key": best_key,
        "records": records,
    }


def plot_gradient_descent_step_size_selection(gd_selection: dict[str, Any], save_path: Path) -> None:
    records = gd_selection.get("records", [])
    if not records:
        return
    alphas = np.asarray([float(r["step_size"]) for r in records], dtype=float)
    residual_p95 = np.asarray([float(r["metrics"]["final_residual_p95"]) for r in records])
    residual_max = np.asarray([float(r["metrics"]["final_residual_max"]) for r in records])
    worst_motion_max = np.asarray([float(r["metrics"]["worst_motion_final_residual_max"]) for r in records])
    selected = float(gd_selection["selected_step_size"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(alphas, np.maximum(residual_p95, PLOT_FLOOR), marker="o", label="pooled residual p95")
    ax.plot(alphas, np.maximum(residual_max, PLOT_FLOOR), marker="s", label="pooled residual max")
    ax.plot(alphas, np.maximum(worst_motion_max, PLOT_FLOOR), marker="^", label="worst-motion residual max")
    ax.axvline(selected, linestyle="--", alpha=0.8, label=f"selected {selected:.1e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gradient-descent step size")
    ax.set_ylabel("Validation final residual after fixed iterations")
    ax.set_title("Validation selection of gradient-descent step size")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)



# =============================================================================
# 6. Full-batch training
# =============================================================================


def one_step_diagnostics(model: MLPOptimizer, dataset: DatasetBundle, physical: PhysicalConfig) -> dict[str, float]:
    with torch.no_grad():
        y0 = dataset.initial_y
        optimizer_state = LearnedOptimizerState.zeros_like(y0)
        y1, delta, _ = apply_model_update(
            model, y0, dataset.q, dataset.masses, physical, optimizer_state
        )
        error0 = torch.linalg.vector_norm(y0 - dataset.exact_y, dim=-1)
        error1 = torch.linalg.vector_norm(y1 - dataset.exact_y, dim=-1)
        residual0 = stationarity_residual_norm(y0, dataset.q, dataset.masses, physical)
        residual1 = stationarity_residual_norm(y1, dataset.q, dataset.masses, physical)
        ideal = dataset.exact_y - y0
        cosine = torch.nn.functional.cosine_similarity(delta, ideal, dim=-1, eps=1e-30)
        return {
            "mean_error_before": float(error0.mean().item()),
            "mean_error_after": float(error1.mean().item()),
            "mean_residual_before": float(residual0.mean().item()),
            "mean_residual_after": float(residual1.mean().item()),
            "mean_update_norm": float(torch.linalg.vector_norm(delta, dim=-1).mean().item()),
            "update_ideal_cosine_mean": float(cosine.mean().item()),
            "error_improvement_fraction": float((error1 < error0).to(TORCH_DTYPE).mean().item()),
            "residual_improvement_fraction": float((residual1 < residual0).to(TORCH_DTYPE).mean().item()),
        }


def run_experiment(
    *,
    model_spec: ModelSpec,
    training_cpu: DatasetBundle,
    validation_cpu: DatasetBundle,
    evaluation_datasets: dict[str, DatasetBundle],
    output_dir: Path,
    config: RuntimeConfig,
    physical: PhysicalConfig,
    gd_step_size: float,
    shared_baselines: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    experiment_name = model_spec.experiment_name
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    report_path = experiment_dir / "experiment_report.json"
    latest_checkpoint_path = experiment_dir / "latest_training_checkpoint.pt"
    device = torch.device(config.device)

    if config.skip_completed and report_path.exists():
        with report_path.open("r", encoding="utf-8") as file:
            print(f"Skipping completed experiment: {experiment_name}")
            return json.load(file)

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)
    model = MLPOptimizer(config.residual_length_scale, model_spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training = training_cpu.to(device)
    energy_scale = physical_energy_scale(
        training.masses, physical, config.residual_length_scale
    )
    initial_energy = variational_energy(
        training.initial_y, training.q, training.masses, physical
    ).detach()
    exact_energy = variational_energy(
        training.exact_y, training.q, training.masses, physical
    ).detach()

    print("\n" + "=" * 100)
    print(f"Training {experiment_name}")
    print(
        f"architecture={model.architecture_description}, "
        f"parameters={model.parameter_count:,}, points={len(training_cpu):,}, "
        f"motions={training_cpu.metadata['num_motions']}, "
        f"problems={training_cpu.metadata['num_problems']}, "
        f"device={device}, dtype=float64, full_batch=True"
    )
    print("=" * 100)

    train_log: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []
    diagnostic_log: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, ...] | None = None
    best_epoch: int | None = None
    best_k: int | None = None
    stage_best_states: dict[int, dict[str, torch.Tensor]] = {}
    stage_best_keys: dict[int, tuple[float, ...]] = {}
    stage_best_epochs: dict[int, int] = {}
    start_epoch = 0
    elapsed_before_resume = 0.0

    if config.resume and latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        saved_spec = checkpoint.get("model_spec", {})
        if saved_spec and saved_spec != asdict(model_spec):
            raise RuntimeError(
                f"Resume checkpoint model spec mismatch: {saved_spec} != {asdict(model_spec)}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        train_log = checkpoint.get("train_log", [])
        validation_log = checkpoint.get("validation_log", [])
        diagnostic_log = checkpoint.get("diagnostic_log", [])
        best_state = checkpoint.get("best_state")
        loaded_best_key = checkpoint.get("best_key")
        best_key = tuple(loaded_best_key) if loaded_best_key is not None else None
        best_epoch = checkpoint.get("best_epoch")
        best_k = checkpoint.get("best_k")
        stage_best_states = {
            int(key): value
            for key, value in checkpoint.get("stage_best_states", {}).items()
        }
        stage_best_keys = {
            int(key): tuple(value)
            for key, value in checkpoint.get("stage_best_keys", {}).items()
        }
        stage_best_epochs = {
            int(key): int(value)
            for key, value in checkpoint.get("stage_best_epochs", {}).items()
        }
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        print(f"Resumed {experiment_name} from epoch {start_epoch}")

    start_time = time.perf_counter()

    for epoch_index in range(start_epoch, config.epochs):
        epoch = epoch_index + 1
        k = get_k_for_epoch(epoch_index, config)
        model.train()
        y = training.initial_y
        learned_optimizer_state = LearnedOptimizerState.zeros_like(y)
        optimizer.zero_grad(set_to_none=True)
        objective_sum = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        energy_gap_sum = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        final_step_objective = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        final_step_energy_gap = torch.zeros((), dtype=TORCH_DTYPE, device=device)

        for _ in range(k):
            y, _, learned_optimizer_state = apply_model_update(
                model,
                y,
                training.q,
                training.masses,
                physical,
                learned_optimizer_state,
            )
            energy = variational_energy(y, training.q, training.masses, physical)
            step_objective = ((energy - initial_energy) / energy_scale).mean()
            step_energy_gap = (energy - exact_energy).mean()
            objective_sum = objective_sum + step_objective
            energy_gap_sum = energy_gap_sum + step_energy_gap
            final_step_objective = step_objective
            final_step_energy_gap = step_energy_gap

        objective_mean = objective_sum / float(k)
        energy_gap_mean = energy_gap_sum / float(k)
        if not bool(torch.isfinite(objective_mean)):
            raise RuntimeError(f"Non-finite training objective at epoch {epoch}")
        objective_mean.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            ).item()
        )
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}")
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise RuntimeError(f"Non-finite model parameter at epoch {epoch}")

        train_log.append({
            "epoch": epoch,
            "K": k,
            "dimensionless_objective": float(objective_mean.item()),
            "dimensionless_objective_mean_over_k": float(objective_mean.item()),
            "dimensionless_objective_sum_over_k": float(objective_sum.item()),
            "final_step_dimensionless_objective": float(final_step_objective.item()),
            "training_energy_gap_mean": float(energy_gap_mean.item()),
            "training_energy_gap_sum": float(energy_gap_sum.item()),
            "final_step_energy_gap": float(final_step_energy_gap.item()),
            "gradient_norm_before_clip": grad_norm,
        })

        if epoch == 1 or epoch % config.diagnostic_interval == 0 or epoch == config.epochs:
            diagnostics = one_step_diagnostics(model, training, physical)
            diagnostics.update(epoch=epoch, K=k)
            diagnostic_log.append(diagnostics)

        should_validate = (
            epoch % config.validation_interval == 0 or epoch == config.epochs
        )
        if should_validate:
            metrics = evaluate_solver_on_dataset(
                solver="learned",
                model=model,
                dataset_cpu=validation_cpu,
                physical=physical,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                report_steps=config.report_steps,
                device=device,
            )
            key = validation_selection_key(metrics)
            validation_log.append({
                "epoch": epoch,
                "K": k,
                "selection_key": key,
                "metrics": metrics,
            })
            current_state = state_dict_to_cpu(model)
            if key is not None and (best_key is None or key < best_key):
                best_key = key
                best_epoch = epoch
                best_k = k
                best_state = copy.deepcopy(current_state)
                torch.save(
                    best_state,
                    experiment_dir / "best_validation_model_state_dict.pt",
                )
            if key is not None and (
                k not in stage_best_keys or key < stage_best_keys[k]
            ):
                stage_best_keys[k] = key
                stage_best_epochs[k] = epoch
                stage_best_states[k] = copy.deepcopy(current_state)
                torch.save(
                    stage_best_states[k],
                    experiment_dir / f"best_validation_K{k:02d}.pt",
                )

            elapsed = elapsed_before_resume + time.perf_counter() - start_time
            checkpoint = {
                "epoch": epoch,
                "model_spec": asdict(model_spec),
                "model_state_dict": current_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "train_log": train_log,
                "validation_log": validation_log,
                "diagnostic_log": diagnostic_log,
                "best_state": best_state,
                "best_key": best_key,
                "best_epoch": best_epoch,
                "best_k": best_k,
                "stage_best_states": stage_best_states,
                "stage_best_keys": stage_best_keys,
                "stage_best_epochs": stage_best_epochs,
                "elapsed_seconds": elapsed,
            }
            torch.save(checkpoint, latest_checkpoint_path)

            if epoch % config.epochs_per_k == 0:
                torch.save(
                    current_state,
                    experiment_dir / f"stage_end_K{k:02d}.pt",
                )

            print(
                f"epoch={epoch:4d} K={k:2d} "
                f"objective_mean={float(objective_mean.item()):.4e} "
                f"objective_sum={float(objective_sum.item()):.4e} "
                f"grad_norm={grad_norm:.4e} "
                f"val_res_p95={metrics['final_residual_p95']:.4e} "
                f"val_res_max={metrics['final_residual_max']:.4e} "
                f"worst_motion_max={metrics['worst_motion_final_residual_max']:.4e} "
                f"best_epoch={best_epoch} best_K={best_k} elapsed={elapsed:.1f}s"
            )

    elapsed_seconds = elapsed_before_resume + time.perf_counter() - start_time
    last_state = state_dict_to_cpu(model)
    if best_state is None:
        best_state = copy.deepcopy(last_state)
        best_epoch = config.epochs
        best_k = get_k_for_epoch(max(config.epochs - 1, 0), config)
        best_key = None
    torch.save(last_state, experiment_dir / "last_model_state_dict.pt")
    torch.save(best_state, experiment_dir / "best_validation_model_state_dict.pt")
    torch.save(best_state, experiment_dir / "mlp_optimizer_state_dict.pt")

    model.load_state_dict(best_state)
    model.to(device)
    learned_results: dict[str, Any] = {}
    for name, dataset in evaluation_datasets.items():
        print(f"Evaluating {experiment_name} on {name} ...")
        learned_results[name] = evaluate_solver_on_dataset(
            solver="learned",
            model=model,
            dataset_cpu=dataset,
            physical=physical,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps,
            device=device,
        )

    comparison = {
        name: {
            "learned": learned_results[name],
            "gradient_descent": shared_baselines[name]["gradient_descent"],
            "full_newton": shared_baselines[name]["full_newton"],
        }
        for name in evaluation_datasets
    }
    report = {
        "experiment_name": experiment_name,
        "model_spec": asdict(model_spec),
        "parameter_count": model.parameter_count,
        "best_validation_epoch": best_epoch,
        "best_validation_K": best_k,
        "best_validation_selection_key": best_key,
        "stage_best_epochs": stage_best_epochs,
        "stage_best_selection_keys": stage_best_keys,
        "training_dataset": training_cpu.metadata,
        "validation_dataset": validation_cpu.metadata,
        "model": {
            "architecture": model.architecture_description,
            "activation": model_spec.activation,
            "hidden_depth": model_spec.depth,
            "hidden_width": model_spec.width,
            "use_bias": model_spec.use_bias,
            "input_channels": [
                "current_mass_preconditioned_residual",
                "previous_mass_preconditioned_residual",
                "previous_optimizer_displacement",
            ],
            "input_channel_dimension": MODEL_INPUT_CHANNEL_DIM,
            "flattened_input_dimension": MODEL_INPUT_DIM,
            "raw_output_dimension": MODEL_OUTPUT_DIM,
            "applied_update_dimension": FREE_STATE_DIM,
            "fixed_output_policy": "fixed-vertex raw output components are discarded before state update and history stores zero applied fixed updates",
            "history_initialization": "zeros for every independent optimization problem",
            "history_scope": "within one iterative solve only; not across physical time steps",
            "history_gradient_policy": "previous residual and previous update are detached between iterations",
            "residual_threshold_update_guard": False,
            "linear_weight_initialization": (
                "PyTorch nn.Linear default reset_parameters for every layer, including output: "
                "Kaiming uniform with a=sqrt(5)"
            ),
            "linear_bias_initialization": (
                "PyTorch nn.Linear default uniform initialization when bias is enabled; "
                "this experiment is bias-free"
            ),
            "residual_length_scale": config.residual_length_scale,
            "dtype": str(TORCH_DTYPE),
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "full_batch": True,
            "batch_size": len(training_cpu),
            "epochs": config.epochs,
            "k_values": list(config.k_values),
            "epochs_per_k": config.epochs_per_k,
            "loss_reduction_over_k": "mean",
            "gradient_clip_norm": config.gradient_clip_norm,
            "energy_scale": energy_scale,
            "elapsed_seconds": elapsed_seconds,
        },
        "metric_policy": {
            "pooled": ["mean", "median", "p95", "max"],
            "boundary": ["pooled max", "worst-motion p95", "worst-motion max"],
        },
        "gradient_descent_step_size": gd_step_size,
        "train_log": train_log,
        "diagnostic_log": diagnostic_log,
        "validation_log": validation_log,
        "evaluation": comparison,
    }
    save_json(report, report_path)

    if not config.skip_plots:
        plot_training_curves(
            train_log,
            validation_log,
            best_epoch,
            experiment_dir / "training_and_validation.png",
        )
        for split_name in [
            "seen_motion_temporal_interpolation",
            "seen_motion_temporal_extrapolation",
            "unseen_id_test",
            "ood_test",
        ]:
            plot_three_solver_rollout(
                comparison[split_name],
                title=f"{experiment_name}: {split_name}",
                save_path=experiment_dir / f"{split_name}_three_solver_rollout.png",
            )
            plot_per_motion_boundary(
                comparison[split_name],
                title=f"{experiment_name}: {split_name}",
                save_path=experiment_dir / f"{split_name}_per_motion_boundary.png",
            )
    return report



# =============================================================================
# 7. Plotting, boundary-case selection, checks, and orchestration
# =============================================================================


def plot_training_curves(
    train_log: Sequence[dict[str, Any]],
    validation_log: Sequence[dict[str, Any]],
    best_epoch: int | None,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    axes[0].plot(
        [record["epoch"] for record in train_log],
        [finite_plot_value(record["training_energy_gap_mean"]) for record in train_log],
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Training energy-gap mean over K")
    val_epochs = [record["epoch"] for record in validation_log]
    specs = [
        ("final_residual_p95", "Validation residual p95"),
        ("final_residual_max", "Validation residual maximum"),
        ("worst_motion_final_residual_max", "Worst-motion residual maximum"),
    ]
    for axis, (key, title) in zip(axes[1:], specs):
        axis.plot(
            val_epochs,
            [finite_plot_value(record["metrics"][key]) for record in validation_log],
            marker="o",
        )
        axis.set_yscale("log")
        axis.set_title(title)
    for axis in axes:
        if best_epoch is not None:
            axis.axvline(best_epoch, linestyle="--", alpha=0.6)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_three_solver_rollout(comparison: dict[str, dict[str, Any]], *, title: str, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [("residual", "Residual"), ("energy_gap", "Energy gap"), ("exact_error", "Exact error")]
    labels = {"learned": "MLP", "gradient_descent": "gradient descent", "full_newton": "full Newton"}
    for col, (metric, metric_title) in enumerate(metrics):
        for row, stat in enumerate(["p95", "max"]):
            ax = axes[row, col]
            for solver_name, values in comparison.items():
                ax.plot(
                    range(values["steps"] + 1),
                    [finite_plot_value(v) for v in values[f"{metric}_{stat}_by_step"]],
                    marker="o", markersize=3, label=labels[solver_name],
                )
            ax.set_yscale("log")
            ax.set_xlabel("Solver iteration")
            ax.set_title(f"{metric_title} {stat}")
            ax.grid(True, alpha=0.3)
            if col == 0 and row == 0:
                ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_motion_boundary(comparison: dict[str, dict[str, Any]], *, title: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    labels = {"learned": "MLP", "gradient_descent": "GD", "full_newton": "Newton"}
    for ax, stat in zip(axes, ["p95", "max"]):
        for solver_name, values in comparison.items():
            motions = sorted(int(k) for k in values["per_motion"])
            y = [values["per_motion"][str(m)]["final"]["residual"][stat] for m in motions]
            ax.plot(motions, np.maximum(np.asarray(y, dtype=float), PLOT_FLOOR), marker="o", label=labels[solver_name])
        ax.set_yscale("log")
        ax.set_xlabel("Motion index")
        ax.set_ylabel(f"Final residual {stat}")
        ax.set_title(f"Per-motion residual {stat}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reference_motion_overview(motions: Sequence[MotionSpec], save_path: Path) -> None:
    fig = plt.figure(figsize=(16, 12))
    for panel, motion in enumerate(motions[:12]):
        ax = fig.add_subplot(3, 4, panel + 1, projection="3d")
        points = np.asarray(motion.p0)
        for i, j in SPRING_EDGES:
            ax.plot(points[[i, j], 0], points[[i, j], 1], points[[i, j], 2], linewidth=0.7)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8)
        fixed = points[list(FIXED_VERTEX_INDICES)]
        ax.scatter(fixed[:, 0], fixed[:, 1], fixed[:, 2], marker="s", s=35)
        ax.set_title(f"{motion.index}: {motion.name}", fontsize=8)
        ax.view_init(elev=22, azim=-62)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_hard_ood_case(
    ood_dataset: DatasetBundle,
    problems_by_index: dict[int, TimeStepProblem],
    physical: PhysicalConfig,
) -> dict[str, Any]:
    residual = stationarity_residual_norm(
        ood_dataset.initial_y, ood_dataset.q, ood_dataset.masses, physical
    ).numpy()
    best_record: dict[str, Any] | None = None
    for problem_index in sorted(torch.unique(ood_dataset.problem_index).tolist()):
        mask = ood_dataset.problem_index.numpy() == int(problem_index)
        values = residual[mask]
        local_indices = np.flatnonzero(mask)
        local_argmax = int(np.nanargmax(values))
        sample_index = int(local_indices[local_argmax])
        record = {
            "problem_index": int(problem_index),
            "initial_residual_max": float(np.nanmax(values)),
            "initial_residual_p95": float(np.nanpercentile(values, 95)),
            "sample_index_in_dataset": sample_index,
            "sample_initial_y": ood_dataset.initial_y[sample_index].tolist(),
        }
        if best_record is None or (
            record["initial_residual_max"], record["initial_residual_p95"]
        ) > (
            best_record["initial_residual_max"], best_record["initial_residual_p95"]
        ):
            best_record = record
    if best_record is None:
        raise RuntimeError("Could not select a hard OOD case")
    problem = problems_by_index[best_record["problem_index"]]
    best_record.update({
        "selection_rule": "largest sampled initial residual maximum; p95 is the tie breaker",
        "motion_index": problem.motion_index,
        "motion_name": problem.motion_name,
        "motion_category": problem.motion_category,
        "local_time_index": problem.local_time_index,
        "physical_time": problem.time,
        "selected_physical_state": {
            "p_n_full": problem.p_n_full.tolist(),
            "v_n_full": problem.v_n_full.tolist(),
        },
    })
    return best_record


def run_physics_checks(physical: PhysicalConfig, motion: MotionSpec) -> dict[str, Any]:
    p = torch.tensor(motion.p0, dtype=TORCH_DTYPE)
    y = free_state_from_full(p).reshape(1, -1)
    reconstructed = full_positions_from_free(y, physical).reshape(NUM_PARTICLES, SPATIAL_DIM)
    fixed_error = float(torch.max(torch.abs(reconstructed[list(FIXED_VERTEX_INDICES)] - p[list(FIXED_VERTEX_INDICES)])).item())
    lengths = spring_lengths_from_free(y, physical).squeeze(0)
    return {
        "num_particles": NUM_PARTICLES,
        "num_free_particles": NUM_FREE_PARTICLES,
        "free_state_dimension": FREE_STATE_DIM,
        "full_state_dimension": FULL_STATE_DIM,
        "model_input_channel_dimension": MODEL_INPUT_CHANNEL_DIM,
        "model_flattened_input_dimension": MODEL_INPUT_DIM,
        "model_raw_output_dimension": MODEL_OUTPUT_DIM,
        "num_springs": NUM_SPRINGS,
        "num_triangles": NUM_TRIANGLES,
        "fixed_reconstruction_error": fixed_error,
        "minimum_initial_spring_length": float(lengths.min().item()),
    }


def problem_to_record(problem: TimeStepProblem) -> dict[str, Any]:
    return {
        "index": problem.index,
        "motion_index": problem.motion_index,
        "motion_name": problem.motion_name,
        "motion_split": problem.motion_split,
        "motion_category": problem.motion_category,
        "local_time_index": problem.local_time_index,
        "time": problem.time,
        "p_n_full": problem.p_n_full.tolist(),
        "v_n_full": problem.v_n_full.tolist(),
        "q_free": problem.q_free.tolist(),
        "free_masses": problem.free_masses.tolist(),
        "exact_y_free": problem.exact_y_free.tolist(),
        "raw_sampling_radius": problem.raw_sampling_radius,
        "sampling_radius": problem.sampling_radius,
        "exact_energy": problem.exact_energy,
        "exact_residual": problem.exact_residual,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "5x5 triangular-cloth activation/depth/width ablation; "
            f"all models use_bias={USE_BIAS}"
        )
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--total-time-steps", type=int, default=DEFAULT_TOTAL_TIME_STEPS)
    parser.add_argument("--train-points-per-problem", type=int, default=DEFAULT_TRAIN_POINTS_PER_PROBLEM)
    parser.add_argument("--eval-points-per-problem", type=int, default=DEFAULT_EVAL_POINTS_PER_PROBLEM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--diagnostic-interval", type=int, default=DEFAULT_DIAGNOSTIC_INTERVAL)
    parser.add_argument("--evaluation-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--epochs-per-k", type=int, default=DEFAULT_EPOCHS_PER_K)
    parser.add_argument("--report-steps", type=int, nargs="+", default=list(DEFAULT_REPORT_STEPS))
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--sampling-radius-min", type=float, default=DEFAULT_SAMPLING_RADIUS_MIN)
    parser.add_argument("--sampling-radius-max", type=float, default=DEFAULT_SAMPLING_RADIUS_MAX)
    parser.add_argument(
        "--activations",
        nargs="+",
        choices=list(ACTIVATION_NAMES),
        default=list(ACTIVATION_NAMES),
    )
    parser.add_argument("--depths", type=int, nargs="+", default=list(HIDDEN_DEPTHS))
    parser.add_argument("--widths", type=int, nargs="+", default=list(HIDDEN_WIDTHS))
    parser.add_argument(
        "--config-index",
        type=int,
        default=None,
        help="Run one zero-based index from the filtered activation/depth/width grid.",
    )
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--save-datasets", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    positive_ints = {
        "total_time_steps": args.total_time_steps,
        "train_points_per_problem": args.train_points_per_problem,
        "eval_points_per_problem": args.eval_points_per_problem,
        "epochs": args.epochs,
        "validation_interval": args.validation_interval,
        "diagnostic_interval": args.diagnostic_interval,
        "evaluation_steps": args.evaluation_steps,
        "evaluation_batch_size": args.evaluation_batch_size,
        "epochs_per_k": args.epochs_per_k,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.total_time_steps) != 100:
        raise ValueError("The confirmed experiment requires exactly 100 time steps per motion")
    k_values = tuple(int(value) for value in args.k_values)
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("All K values must be positive")
    if float(args.gradient_clip_norm) <= 0.0:
        raise ValueError("gradient_clip_norm must be positive")
    if (
        float(args.sampling_radius_min) <= 0
        or float(args.sampling_radius_max) < float(args.sampling_radius_min)
    ):
        raise ValueError("Invalid sampling-radius clamp")
    depths = tuple(int(value) for value in args.depths)
    widths = tuple(int(value) for value in args.widths)
    if not depths or any(value <= 0 for value in depths):
        raise ValueError("All depths must be positive")
    if not widths or any(value <= 0 for value in widths):
        raise ValueError("All widths must be positive")
    report_steps = tuple(sorted(set(
        [
            int(step)
            for step in args.report_steps
            if 0 < int(step) <= int(args.evaluation_steps)
        ]
        + [int(args.evaluation_steps)]
    )))
    return RuntimeConfig(
        total_time_steps=int(args.total_time_steps),
        train_points_per_problem=int(args.train_points_per_problem),
        eval_points_per_problem=int(args.eval_points_per_problem),
        epochs=int(args.epochs),
        validation_interval=int(args.validation_interval),
        diagnostic_interval=int(args.diagnostic_interval),
        evaluation_steps=int(args.evaluation_steps),
        evaluation_batch_size=int(args.evaluation_batch_size),
        k_values=k_values,
        epochs_per_k=int(args.epochs_per_k),
        report_steps=report_steps,
        residual_length_scale=float(args.residual_length_scale),
        gradient_clip_norm=float(args.gradient_clip_norm),
        sampling_radius_min=float(args.sampling_radius_min),
        sampling_radius_max=float(args.sampling_radius_max),
        device=str(args.device),
        activations=tuple(str(value) for value in args.activations),
        depths=depths,
        widths=widths,
        config_index=args.config_index,
        list_configs=bool(args.list_configs),
        skip_completed=bool(args.skip_completed),
        resume=bool(args.resume),
        skip_plots=bool(args.skip_plots),
        save_datasets=bool(args.save_datasets),
    )


def compact_experiment_summary(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") == "failed":
        return {
            "status": "failed",
            "experiment_name": report.get("experiment_name"),
            "model_spec": report.get("model_spec"),
            "error": report.get("error"),
        }
    best_epoch = report.get("best_validation_epoch")
    best_validation_metrics: dict[str, Any] = {}
    for record in report.get("validation_log", []):
        if record.get("epoch") == best_epoch:
            metrics = record.get("metrics", {})
            best_validation_metrics = {
                "final_residual_p95": metrics.get("final_residual_p95"),
                "final_residual_max": metrics.get("final_residual_max"),
                "worst_motion_final_residual_p95": metrics.get(
                    "worst_motion_final_residual_p95"
                ),
                "worst_motion_final_residual_max": metrics.get(
                    "worst_motion_final_residual_max"
                ),
            }
            break
    ood = (
        report.get("evaluation", {})
        .get("ood_test", {})
        .get("learned", {})
    )
    return {
        "status": "success",
        "experiment_name": report.get("experiment_name"),
        "model_spec": report.get("model_spec"),
        "parameter_count": report.get("parameter_count"),
        "best_validation_epoch": best_epoch,
        "best_validation_K": report.get("best_validation_K"),
        "best_validation_selection_key": report.get("best_validation_selection_key"),
        **best_validation_metrics,
        "ood_final_residual_p95": ood.get("final_residual_p95"),
        "ood_final_residual_max": ood.get("final_residual_max"),
        "ood_worst_motion_residual_p95": ood.get(
            "worst_motion_final_residual_p95"
        ),
        "ood_worst_motion_residual_max": ood.get(
            "worst_motion_final_residual_max"
        ),
        "elapsed_seconds": report.get("training", {}).get("elapsed_seconds"),
    }


def save_experiment_summary_csv(
    summaries: Sequence[dict[str, Any]], save_path: Path
) -> None:
    fieldnames = [
        "status",
        "experiment_name",
        "activation",
        "depth",
        "width",
        "use_bias",
        "parameter_count",
        "best_validation_epoch",
        "best_validation_K",
        "final_residual_p95",
        "final_residual_max",
        "worst_motion_final_residual_p95",
        "worst_motion_final_residual_max",
        "ood_final_residual_p95",
        "ood_final_residual_max",
        "ood_worst_motion_residual_p95",
        "ood_worst_motion_residual_max",
        "elapsed_seconds",
        "error",
    ]
    with save_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            spec = summary.get("model_spec") or {}
            writer.writerow({
                "status": summary.get("status"),
                "experiment_name": summary.get("experiment_name"),
                "activation": spec.get("activation"),
                "depth": spec.get("depth"),
                "width": spec.get("width"),
                "use_bias": spec.get("use_bias"),
                "parameter_count": summary.get("parameter_count"),
                "best_validation_epoch": summary.get("best_validation_epoch"),
                "best_validation_K": summary.get("best_validation_K"),
                "final_residual_p95": summary.get("final_residual_p95"),
                "final_residual_max": summary.get("final_residual_max"),
                "worst_motion_final_residual_p95": summary.get(
                    "worst_motion_final_residual_p95"
                ),
                "worst_motion_final_residual_max": summary.get(
                    "worst_motion_final_residual_max"
                ),
                "ood_final_residual_p95": summary.get("ood_final_residual_p95"),
                "ood_final_residual_max": summary.get("ood_final_residual_max"),
                "ood_worst_motion_residual_p95": summary.get(
                    "ood_worst_motion_residual_p95"
                ),
                "ood_worst_motion_residual_max": summary.get(
                    "ood_worst_motion_residual_max"
                ),
                "elapsed_seconds": summary.get("elapsed_seconds"),
                "error": summary.get("error"),
            })



def _successful_experiment_summaries(
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        summary
        for summary in summaries
        if summary.get("status") == "success"
        and summary.get("best_validation_selection_key") is not None
    ]


def _summary_sort_key(summary: dict[str, Any]) -> tuple[float, ...]:
    raw = summary.get("best_validation_selection_key")
    if raw is None:
        return (float("inf"),)
    return tuple(float(value) for value in raw)


def save_ranked_experiment_summary_csv(
    summaries: Sequence[dict[str, Any]], save_path: Path
) -> None:
    ranked = sorted(_successful_experiment_summaries(summaries), key=_summary_sort_key)
    fieldnames = [
        "rank",
        "experiment_name",
        "activation",
        "depth",
        "width",
        "use_bias",
        "parameter_count",
        "best_validation_epoch",
        "best_validation_K",
        "final_residual_p95",
        "final_residual_max",
        "worst_motion_final_residual_p95",
        "worst_motion_final_residual_max",
        "ood_final_residual_p95",
        "ood_final_residual_max",
        "ood_worst_motion_residual_p95",
        "ood_worst_motion_residual_max",
        "elapsed_seconds",
    ]
    with save_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, summary in enumerate(ranked, start=1):
            spec = summary.get("model_spec") or {}
            writer.writerow({
                "rank": rank,
                "experiment_name": summary.get("experiment_name"),
                "activation": spec.get("activation"),
                "depth": spec.get("depth"),
                "width": spec.get("width"),
                "use_bias": spec.get("use_bias"),
                "parameter_count": summary.get("parameter_count"),
                "best_validation_epoch": summary.get("best_validation_epoch"),
                "best_validation_K": summary.get("best_validation_K"),
                "final_residual_p95": summary.get("final_residual_p95"),
                "final_residual_max": summary.get("final_residual_max"),
                "worst_motion_final_residual_p95": summary.get(
                    "worst_motion_final_residual_p95"
                ),
                "worst_motion_final_residual_max": summary.get(
                    "worst_motion_final_residual_max"
                ),
                "ood_final_residual_p95": summary.get("ood_final_residual_p95"),
                "ood_final_residual_max": summary.get("ood_final_residual_max"),
                "ood_worst_motion_residual_p95": summary.get(
                    "ood_worst_motion_residual_p95"
                ),
                "ood_worst_motion_residual_max": summary.get(
                    "ood_worst_motion_residual_max"
                ),
                "elapsed_seconds": summary.get("elapsed_seconds"),
            })


def _finite_metric(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if value is None:
        return float("nan")
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def plot_ranked_experiment_metrics(
    summaries: Sequence[dict[str, Any]],
    *,
    metric_specs: Sequence[tuple[str, str]],
    title: str,
    save_path: Path,
) -> None:
    ranked = sorted(_successful_experiment_summaries(summaries), key=_summary_sort_key)
    if not ranked:
        return
    ranks = np.arange(1, len(ranked) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for axis, (metric_key, metric_title) in zip(axes.ravel(), metric_specs):
        values = np.asarray(
            [_finite_metric(summary, metric_key) for summary in ranked], dtype=float
        )
        axis.plot(ranks, np.maximum(values, PLOT_FLOOR), marker="o", markersize=4)
        axis.set_yscale("log")
        axis.set_title(metric_title)
        axis.set_ylabel("Residual")
        axis.grid(True, alpha=0.3, which="both")
    for axis in axes[-1, :]:
        axis.set_xlabel("Configuration rank by validation selection rule")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_heatmaps(
    summaries: Sequence[dict[str, Any]],
    *,
    activations: Sequence[str],
    depths: Sequence[int],
    widths: Sequence[int],
    save_path: Path,
) -> None:
    successful = _successful_experiment_summaries(summaries)
    if not successful:
        return
    by_spec: dict[tuple[str, int, int], dict[str, Any]] = {}
    for summary in successful:
        spec = summary.get("model_spec") or {}
        key = (str(spec.get("activation")), int(spec.get("depth")), int(spec.get("width")))
        by_spec[key] = summary

    metrics = [
        ("final_residual_p95", "Validation residual p95"),
        ("ood_final_residual_p95", "OOD residual p95"),
    ]
    matrices: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_log_values: list[float] = []
    for row, (metric_key, _) in enumerate(metrics):
        for col, activation in enumerate(activations):
            log_matrix = np.full((len(depths), len(widths)), np.nan, dtype=float)
            raw_matrix = np.full_like(log_matrix, np.nan)
            for i, depth in enumerate(depths):
                for j, width in enumerate(widths):
                    summary = by_spec.get((str(activation), int(depth), int(width)))
                    if summary is None:
                        continue
                    value = _finite_metric(summary, metric_key)
                    if math.isfinite(value) and value > 0.0:
                        raw_matrix[i, j] = value
                        log_value = math.log10(max(value, PLOT_FLOOR))
                        log_matrix[i, j] = log_value
                        all_log_values.append(log_value)
            matrices[(row, col)] = (log_matrix, raw_matrix)
    if not all_log_values:
        return
    vmin = min(all_log_values)
    vmax = max(all_log_values)
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0

    fig, axes = plt.subplots(
        len(metrics), len(activations),
        figsize=(5.2 * len(activations), 4.8 * len(metrics)),
        squeeze=False,
    )
    image = None
    for row, (_, metric_title) in enumerate(metrics):
        for col, activation in enumerate(activations):
            log_matrix, raw_matrix = matrices[(row, col)]
            image = axes[row, col].imshow(
                log_matrix, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
            )
            axes[row, col].set_xticks(range(len(widths)), [str(width) for width in widths])
            axes[row, col].set_yticks(range(len(depths)), [str(depth) for depth in depths])
            axes[row, col].set_xlabel("Hidden width")
            axes[row, col].set_ylabel("Hidden depth")
            axes[row, col].set_title(f"{activation}: {metric_title}")
            for i in range(len(depths)):
                for j in range(len(widths)):
                    value = raw_matrix[i, j]
                    label = "N/A" if not math.isfinite(value) else f"{value:.1e}"
                    axes[row, col].text(j, i, label, ha="center", va="center", fontsize=8)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
        colorbar.set_label("log10 residual")
    fig.suptitle("Activation-depth-width ablation summary")
    fig.subplots_adjust(top=0.92, wspace=0.28, hspace=0.32)
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_efficiency(
    summaries: Sequence[dict[str, Any]], save_path: Path
) -> None:
    successful = _successful_experiment_summaries(summaries)
    if not successful:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    metric_specs = [
        ("final_residual_p95", "Validation residual p95"),
        ("ood_final_residual_p95", "OOD residual p95"),
    ]
    for axis, (metric_key, title) in zip(axes, metric_specs):
        for activation in sorted({
            str((summary.get("model_spec") or {}).get("activation"))
            for summary in successful
        }):
            selected = [
                summary for summary in successful
                if str((summary.get("model_spec") or {}).get("activation")) == activation
            ]
            x = np.asarray([float(summary["parameter_count"]) for summary in selected])
            y = np.asarray([_finite_metric(summary, metric_key) for summary in selected])
            axis.scatter(x, np.maximum(y, PLOT_FLOOR), label=activation)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Trainable parameter count")
        axis.set_ylabel("Residual")
        axis.set_title(title)
        axis.grid(True, alpha=0.3, which="both")
        axis.legend()
    fig.suptitle("Parameter efficiency across all successful configurations")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_all_experiments_markdown_report(
    summaries: Sequence[dict[str, Any]],
    *,
    overall_best: dict[str, Any] | None,
    save_path: Path,
) -> None:
    ranked = sorted(_successful_experiment_summaries(summaries), key=_summary_sort_key)
    failed = [summary for summary in summaries if summary.get("status") != "success"]
    lines = [
        "# All Experiments Summary",
        "",
        f"- Requested configurations: {len(summaries)}",
        f"- Successful configurations: {len(ranked)}",
        f"- Failed configurations: {len(failed)}",
        "- Weight initialization: PyTorch `nn.Linear` default initialization for every weight matrix, including the output layer.",
        "- Learned-optimizer input: current full residual, previous full residual, and previous applied full displacement (`3 x 75`, flattened to `225D`).",
        "- Residual-threshold update guard: disabled.",
        "",
    ]
    if overall_best is not None:
        spec = overall_best.get("model_spec") or {}
        lines.extend([
            "## Overall Best Configuration",
            "",
            f"- Experiment: `{overall_best.get('experiment_name')}`",
            f"- Activation: `{spec.get('activation')}`",
            f"- Depth: `{spec.get('depth')}`",
            f"- Width: `{spec.get('width')}`",
            f"- Validation residual p95: `{overall_best.get('final_residual_p95')}`",
            f"- Validation residual max: `{overall_best.get('final_residual_max')}`",
            f"- OOD residual p95: `{overall_best.get('ood_final_residual_p95')}`",
            f"- OOD residual max: `{overall_best.get('ood_final_residual_max')}`",
            "",
        ])
    lines.extend([
        "## Top Configurations",
        "",
        "| Rank | Activation | Depth | Width | Parameters | Validation p95 | Validation max | OOD p95 | OOD max |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, summary in enumerate(ranked[:10], start=1):
        spec = summary.get("model_spec") or {}
        lines.append(
            "| "
            f"{rank} | {spec.get('activation')} | {spec.get('depth')} | {spec.get('width')} | "
            f"{summary.get('parameter_count')} | {summary.get('final_residual_p95')} | "
            f"{summary.get('final_residual_max')} | {summary.get('ood_final_residual_p95')} | "
            f"{summary.get('ood_final_residual_max')} |"
        )
    if failed:
        lines.extend(["", "## Failed Configurations", ""])
        for summary in failed:
            lines.append(
                f"- `{summary.get('experiment_name')}`: {summary.get('error', 'unknown error')}"
            )
    lines.extend([
        "",
        "## Aggregate Figures",
        "",
        "- `all_experiments_ranked_validation.png`",
        "- `all_experiments_ranked_ood.png`",
        "- `all_experiments_ablation_heatmaps.png`",
        "- `all_experiments_parameter_efficiency.png`",
        "",
    ])
    save_path.write_text("\n".join(lines), encoding="utf-8")


def save_output_artifact_manifest(
    *,
    summaries: Sequence[dict[str, Any]],
    skip_plots: bool,
    save_path: Path,
) -> None:
    successful_plot_reports = [
        summary for summary in summaries if summary.get("status") == "success"
    ]
    per_experiment_images = [
        "training_and_validation.png",
        "seen_motion_temporal_interpolation_three_solver_rollout.png",
        "seen_motion_temporal_interpolation_per_motion_boundary.png",
        "seen_motion_temporal_extrapolation_three_solver_rollout.png",
        "seen_motion_temporal_extrapolation_per_motion_boundary.png",
        "unseen_id_test_three_solver_rollout.png",
        "unseen_id_test_per_motion_boundary.png",
        "ood_test_three_solver_rollout.png",
        "ood_test_per_motion_boundary.png",
    ]
    global_original_images = [
        "motion_catalogue_overview.png",
        "gradient_descent_step_size_selection.png",
    ]
    aggregate_images = [
        "all_experiments_ranked_validation.png",
        "all_experiments_ranked_ood.png",
        "all_experiments_ablation_heatmaps.png",
        "all_experiments_parameter_efficiency.png",
    ]
    expected_count = 0 if skip_plots else (
        len(global_original_images)
        + len(successful_plot_reports) * len(per_experiment_images)
        + len(aggregate_images)
    )
    save_json({
        "plots_enabled": not skip_plots,
        "original_plot_set_preserved": True,
        "global_original_images": global_original_images,
        "per_successful_experiment_original_images": per_experiment_images,
        "new_aggregate_images": aggregate_images,
        "successful_experiment_count": len(successful_plot_reports),
        "expected_png_count_when_output_directory_is_clean": expected_count,
        "note": (
            "The history-input version preserves every original plot filename pattern "
            "and adds four aggregate figures after all requested configurations finish."
        ),
    }, save_path)


def main() -> None:
    config = validate_args(parse_args())
    model_specs = build_model_specs(config)
    if config.list_configs:
        for index, spec in enumerate(model_specs):
            print(f"{index:02d}: {spec.experiment_name}")
        return

    physical = default_physical_config()
    motions, motion_split = build_motion_catalogue(physical)
    output_dir = create_output_directory()
    device = torch.device(config.device)
    validate_device(device)

    physics_checks = run_physics_checks(physical, motions[0])
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}; use_bias={USE_BIAS}")
    print(f"Model configurations: {len(model_specs)}")
    print(f"Physics checks: {physics_checks}")
    if not config.skip_plots:
        plot_reference_motion_overview(
            motions, output_dir / "motion_catalogue_overview.png"
        )

    problems = generate_all_reference_sequences(physical, motions, config)
    lookup = problem_lookup(problems)
    problems_by_index = {problem.index: problem for problem in problems}

    save_json({
        "runtime_config": asdict(config),
        "use_bias": USE_BIAS,
        "model_specs": [asdict(spec) for spec in model_specs],
        "physical_config": asdict(physical),
        "motion_split": asdict(motion_split),
        "motions": [asdict(motion) for motion in motions],
        "fixed_vertex_indices": list(FIXED_VERTEX_INDICES),
        "fixed_positions": [list(position) for position in physical.fixed_positions],
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "spring_edges": [list(edge) for edge in SPRING_EDGES],
        "triangle_faces": [list(face) for face in TRIANGLE_FACES],
        "physics_checks": physics_checks,
    }, output_dir / "runtime_config.json")
    save_json(
        {"problems": [problem_to_record(problem) for problem in problems]},
        output_dir / "reference_time_step_problems.json",
    )
    save_json(
        {
            "motions": [asdict(motion) for motion in motions],
            "motion_split": asdict(motion_split),
        },
        output_dir / "motion_catalogue.json",
    )

    multi_training = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.train_motion_indices,
        time_indices=TRAIN_TIME_INDICES,
        points_per_problem=config.train_points_per_problem,
        base_seed=TRAIN_SOBOL_SEED,
        role="multi_motion_training",
        physical=physical,
        include_explicit_train_points=True,
    )
    if len(multi_training) != 8192 and config.train_points_per_problem == 32:
        raise AssertionError(f"Expected 8192 training states, got {len(multi_training)}")

    validation = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.validation_motion_indices,
        time_indices=VALIDATION_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=VALIDATION_SOBOL_SEED,
        role="unseen_motion_validation",
        physical=physical,
        include_explicit_train_points=False,
    )
    seen_interp = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.train_motion_indices,
        time_indices=SEEN_INTERPOLATION_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=SEEN_INTERPOLATION_TEST_SOBOL_SEED,
        role="seen_motion_temporal_interpolation",
        physical=physical,
        include_explicit_train_points=False,
    )
    seen_extrap = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.train_motion_indices,
        time_indices=SEEN_EXTRAPOLATION_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=SEEN_EXTRAPOLATION_TEST_SOBOL_SEED,
        role="seen_motion_temporal_extrapolation",
        physical=physical,
        include_explicit_train_points=False,
    )
    unseen_id = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.id_test_motion_indices,
        time_indices=UNSEEN_TEST_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=UNSEEN_ID_TEST_SOBOL_SEED,
        role="unseen_id_test",
        physical=physical,
        include_explicit_train_points=False,
    )
    ood_test = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.ood_test_motion_indices,
        time_indices=UNSEEN_TEST_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=OOD_TEST_SOBOL_SEED,
        role="ood_test",
        physical=physical,
        include_explicit_train_points=False,
    )
    current_seen = build_special_state_dataset(
        lookup=lookup,
        motion_indices=motion_split.train_motion_indices,
        time_indices=SEEN_INTERPOLATION_TIME_INDICES,
        state="current",
        role="current_state_seen_motion",
    )
    current_id = build_special_state_dataset(
        lookup=lookup,
        motion_indices=motion_split.id_test_motion_indices,
        time_indices=UNSEEN_TEST_TIME_INDICES,
        state="current",
        role="current_state_unseen_id",
    )
    current_ood = build_special_state_dataset(
        lookup=lookup,
        motion_indices=motion_split.ood_test_motion_indices,
        time_indices=UNSEEN_TEST_TIME_INDICES,
        state="current",
        role="current_state_ood",
    )
    evaluation_datasets = {
        "seen_motion_temporal_interpolation": seen_interp,
        "seen_motion_temporal_extrapolation": seen_extrap,
        "unseen_id_test": unseen_id,
        "ood_test": ood_test,
        "current_state_seen_motion": current_seen,
        "current_state_unseen_id": current_id,
        "current_state_ood": current_ood,
    }

    hard_case = select_hard_ood_case(ood_test, problems_by_index, physical)
    save_json(hard_case, output_dir / "hard_case_selection.json")

    gd_step_size, gd_selection = select_gradient_descent_step_size(
        validation=validation,
        physical=physical,
        config=config,
        device=device,
    )
    save_json(gd_selection, output_dir / "gradient_descent_step_selection.json")
    if not config.skip_plots:
        plot_gradient_descent_step_size_selection(
            gd_selection,
            output_dir / "gradient_descent_step_size_selection.png",
        )
    print(f"Selected gradient-descent step size: {gd_step_size:.3e}")

    shared_baselines: dict[str, dict[str, Any]] = {}
    for name, dataset in evaluation_datasets.items():
        print(f"Evaluating GD and Newton on {name} ...")
        shared_baselines[name] = {
            "gradient_descent": evaluate_solver_on_dataset(
                solver="gradient_descent",
                dataset_cpu=dataset,
                physical=physical,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                report_steps=config.report_steps,
                device=device,
                gd_step_size=gd_step_size,
            ),
            "full_newton": evaluate_solver_on_dataset(
                solver="full_newton",
                dataset_cpu=dataset,
                physical=physical,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                report_steps=config.report_steps,
                device=device,
            ),
        }
    save_json(
        shared_baselines, output_dir / "shared_gd_newton_baselines.json"
    )

    if config.save_datasets:
        torch.save({
            "multi_motion_training": dataset_to_serializable_dict(multi_training),
            "validation": dataset_to_serializable_dict(validation),
            **{
                name: dataset_to_serializable_dict(dataset)
                for name, dataset in evaluation_datasets.items()
            },
        }, output_dir / "generated_datasets.pt")

    experiment_summaries: list[dict[str, Any]] = []
    for config_number, model_spec in enumerate(model_specs):
        print(
            f"\nStarting configuration {config_number + 1}/{len(model_specs)}: "
            f"{model_spec.experiment_name}"
        )
        try:
            report = run_experiment(
                model_spec=model_spec,
                training_cpu=multi_training,
                validation_cpu=validation,
                evaluation_datasets=evaluation_datasets,
                output_dir=output_dir,
                config=config,
                physical=physical,
                gd_step_size=gd_step_size,
                shared_baselines=shared_baselines,
            )
        except Exception as error:
            failure = {
                "status": "failed",
                "experiment_name": model_spec.experiment_name,
                "model_spec": asdict(model_spec),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            failure_dir = output_dir / model_spec.experiment_name
            failure_dir.mkdir(parents=True, exist_ok=True)
            save_json(failure, failure_dir / "failure_report.json")
            print(f"Configuration failed: {failure['error']}")
            report = failure
        experiment_summaries.append(compact_experiment_summary(report))
        save_json({
            "use_bias": USE_BIAS,
            "device": config.device,
            "experiment_summaries": experiment_summaries,
        }, output_dir / "all_experiments_summary.json")
        save_experiment_summary_csv(
            experiment_summaries, output_dir / "all_experiments_summary.csv"
        )
        del report
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    successful = [
        summary
        for summary in experiment_summaries
        if summary.get("status") == "success"
        and summary.get("best_validation_selection_key") is not None
    ]
    overall_best = None
    if successful:
        overall_best = min(
            successful,
            key=lambda summary: tuple(summary["best_validation_selection_key"]),
        )
        save_json(overall_best, output_dir / "overall_best_experiment.json")

    summary = {
        "experiment_type": "fixed_left_edge_5x5_cloth_history_input_default_init_activation_depth_width_ablation",
        "use_bias": USE_BIAS,
        "runtime_config": asdict(config),
        "physical_config": asdict(physical),
        "motion_split": asdict(motion_split),
        "physics_checks": physics_checks,
        "gradient_descent_selection": gd_selection,
        "hard_case_selection": hard_case,
        "experiment_summaries": experiment_summaries,
        "overall_best_experiment": overall_best,
        "metric_policy": {
            "pooled_statistics": ["mean", "median", "p95", "max"],
            "motion_boundary_statistics": [
                "worst-motion p95",
                "worst-motion max",
            ],
        },
    }
    save_json(summary, output_dir / "all_experiments_summary.json")
    save_experiment_summary_csv(
        experiment_summaries, output_dir / "all_experiments_summary.csv"
    )
    save_ranked_experiment_summary_csv(
        experiment_summaries, output_dir / "all_experiments_ranked_summary.csv"
    )
    save_all_experiments_markdown_report(
        experiment_summaries,
        overall_best=overall_best,
        save_path=output_dir / "all_experiments_summary.md",
    )
    if not config.skip_plots:
        plot_ranked_experiment_metrics(
            experiment_summaries,
            metric_specs=[
                ("final_residual_p95", "Validation pooled residual p95"),
                ("final_residual_max", "Validation pooled residual maximum"),
                ("worst_motion_final_residual_p95", "Validation worst-motion residual p95"),
                ("worst_motion_final_residual_max", "Validation worst-motion residual maximum"),
            ],
            title="All configurations ranked by validation selection rule",
            save_path=output_dir / "all_experiments_ranked_validation.png",
        )
        plot_ranked_experiment_metrics(
            experiment_summaries,
            metric_specs=[
                ("ood_final_residual_p95", "OOD pooled residual p95"),
                ("ood_final_residual_max", "OOD pooled residual maximum"),
                ("ood_worst_motion_residual_p95", "OOD worst-motion residual p95"),
                ("ood_worst_motion_residual_max", "OOD worst-motion residual maximum"),
            ],
            title="OOD performance of all configurations in validation rank order",
            save_path=output_dir / "all_experiments_ranked_ood.png",
        )
        plot_ablation_heatmaps(
            experiment_summaries,
            activations=config.activations,
            depths=config.depths,
            widths=config.widths,
            save_path=output_dir / "all_experiments_ablation_heatmaps.png",
        )
        plot_parameter_efficiency(
            experiment_summaries,
            output_dir / "all_experiments_parameter_efficiency.png",
        )
    save_output_artifact_manifest(
        summaries=experiment_summaries,
        skip_plots=config.skip_plots,
        save_path=output_dir / "output_artifact_manifest.json",
    )
    print("\nCompleted requested experiments.")
    print(f"Summary JSON: {output_dir / 'all_experiments_summary.json'}")
    print(f"Ranked CSV: {output_dir / 'all_experiments_ranked_summary.csv'}")
    print(f"Markdown report: {output_dir / 'all_experiments_summary.md'}")
    if not config.skip_plots:
        print(f"Aggregate figures: {output_dir / 'all_experiments_ranked_validation.png'}")
        print(f"                   {output_dir / 'all_experiments_ranked_ood.png'}")
        print(f"                   {output_dir / 'all_experiments_ablation_heatmaps.png'}")
        print(f"                   {output_dir / 'all_experiments_parameter_efficiency.png'}")
    if overall_best is not None:
        best_name = overall_best["experiment_name"]
        print(
            "Overall best checkpoint: "
            + str(output_dir / best_name / "best_validation_model_state_dict.pt")
        )


if __name__ == "__main__":
    main()
