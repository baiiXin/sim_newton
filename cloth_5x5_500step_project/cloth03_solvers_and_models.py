"""
Fixed-left-edge 5x5 triangular-cloth learned-optimizer ablation.

Full-state fixed-point variant inspired by Metamizer boundary handling:
- Dataset and learned rollout states are full 25 x 3 = 75D vectors.
- The reduced 23-free-vertex 69D physics, reference Newton solve, and analytic Hessian are retained internally.
- Fixed vertices are hard-projected to their prescribed positions after every learned/GD/Newton update.
- The network does not receive a fixed-point one-hot channel and does not use velocities or accelerations.
- Network input: 3 x 75D = 225D [current residual, previous residual, previous update].
- Network output: 75D update, with fixed-vertex entries gated to zero before applying it.

Default experiment grid: activation in {identity, ReLU, Tanh}, depth=1, width=256,
all linear layers are bias-free, default device cuda:0.
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

USE_HISTORY_INPUT = True
MODEL_INPUT_DIM = (3 if USE_HISTORY_INPUT else 1) * FULL_STATE_DIM

ACTIVATION_NAMES = ("identity", "relu", "tanh")
HIDDEN_DEPTHS = (1,)
HIDDEN_WIDTHS = (256,)
USE_BIAS = False
OPTIMIZER_NAME = "adam"
LEARNING_RATE = 1e-3
DEFAULT_DEVICE = "cuda:0"

DEFAULT_TOTAL_TIME_STEPS = 500
DEFAULT_TRAIN_POINTS_PER_PROBLEM = 32
DEFAULT_EVAL_POINTS_PER_PROBLEM = 128
DEFAULT_EPOCHS = 500
DEFAULT_VALIDATION_INTERVAL = 50
DEFAULT_DIAGNOSTIC_INTERVAL = 200
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8192
DEFAULT_K_VALUES = (1, 3, 5, 10, 30)
DEFAULT_EPOCHS_PER_K = 100
DEFAULT_REPORT_STEPS = (1, 3, 5, 10, 30, 50)
DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_GRADIENT_CLIP_NORM = 10.0
DEFAULT_SAMPLING_RADIUS_MIN = 1e-2
DEFAULT_SAMPLING_RADIUS_MAX = 1e-1

LEGACY_TRAIN_TIME_INDICES = (0, 5, 11, 16, 21, 26, 32, 37, 42, 47, 53, 58, 63, 68, 74, 79)
# Dataset time splits for the 500-step pipeline are built in cloth02_dataset_catalog.py.
TRAIN_TIME_INDICES = tuple(range(0, 400))
SEEN_EXTRAPOLATION_TIME_INDICES = tuple(range(400, 500))
VALIDATION_TIME_INDICES = tuple(range(0, 500, 10))
UNSEEN_TEST_TIME_INDICES = tuple(range(0, 500, 10))

MODEL_RANDOM_SEED = 42
MOTION_SOBOL_SEED_TRAIN = 20260630
MOTION_SOBOL_SEED_VALIDATION = 20260701
MOTION_SOBOL_SEED_ID_TEST = 20260702
TRAIN_SOBOL_SEED = 20260620
VALIDATION_SOBOL_SEED = 20260621
SEEN_EXTRAPOLATION_TEST_SOBOL_SEED = 20260623
UNSEEN_ID_TEST_SOBOL_SEED = 20260624
OOD_TEST_SOBOL_SEED = 20260625

GD_CANDIDATE_STEP_SIZES = (
    1e-8,
    2e-8,
    5e-8,
    1e-7,
    2e-7,
    5e-7,
    1e-6,
    2e-6,
    5e-6,
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
)

ADAM_CANDIDATE_LRS = (
    1e-5,
    2e-5,
    5e-5,
    1e-4,
    2e-4,
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
)

LBFGS_CANDIDATES = (
    {"learning_rate": 0.05, "history_size": 10},
    {"learning_rate": 0.10, "history_size": 10},
    {"learning_rate": 0.25, "history_size": 10},
    {"learning_rate": 0.50, "history_size": 10},
    {"learning_rate": 1.00, "history_size": 10},
    {"learning_rate": 2.00, "history_size": 10},
    {"learning_rate": 0.50, "history_size": 5},
    {"learning_rate": 0.50, "history_size": 20},
    {"learning_rate": 1.00, "history_size": 5},
    {"learning_rate": 1.00, "history_size": 20},
    {"learning_rate": 1.00, "history_size": 50},
    {"learning_rate": 2.00, "history_size": 20},
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
            f"activation_{self.activation}_depth_{self.depth:02d}_"
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


def reshape_full(y: torch.Tensor) -> torch.Tensor:
    if y.shape[-1] != FULL_STATE_DIM:
        raise ValueError(f"Expected final dimension {FULL_STATE_DIM}, got {tuple(y.shape)}")
    return y.reshape(*y.shape[:-1], NUM_PARTICLES, SPATIAL_DIM)


def full_state_from_positions(full: torch.Tensor) -> torch.Tensor:
    if full.shape[-2:] != (NUM_PARTICLES, SPATIAL_DIM):
        raise ValueError(f"Expected (..., {NUM_PARTICLES}, {SPATIAL_DIM}), got {tuple(full.shape)}")
    return full.reshape(*full.shape[:-2], FULL_STATE_DIM)


def full_state_from_free_state(y_free: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    return full_positions_from_free(y_free, physical).reshape(*y_free.shape[:-1], FULL_STATE_DIM)


def free_state_from_full_state(y_full: torch.Tensor) -> torch.Tensor:
    full = reshape_full(y_full)
    return full[..., list(FREE_VERTEX_INDICES), :].reshape(*y_full.shape[:-1], FREE_STATE_DIM)


def full_vector_from_free_vector(v_free: torch.Tensor) -> torch.Tensor:
    free = reshape_free(v_free)
    full = torch.zeros(
        (*free.shape[:-2], NUM_PARTICLES, SPATIAL_DIM),
        dtype=v_free.dtype,
        device=v_free.device,
    )
    full[..., list(FREE_VERTEX_INDICES), :] = free
    return full.reshape(*v_free.shape[:-1], FULL_STATE_DIM)


def free_update_gate_like(y_full: torch.Tensor) -> torch.Tensor:
    gate = torch.ones_like(reshape_full(y_full))
    gate[..., list(FIXED_VERTEX_INDICES), :] = 0.0
    return gate.reshape(*y_full.shape[:-1], FULL_STATE_DIM)


def project_fixed_vertices(y_full: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    """Metamizer-style hard projection: fixed vertices are overwritten after every update."""
    full = reshape_full(y_full).clone()
    fixed_positions = torch.as_tensor(
        physical.fixed_positions,
        dtype=y_full.dtype,
        device=y_full.device,
    )
    full[..., list(FIXED_VERTEX_INDICES), :] = fixed_positions
    return full.reshape(*y_full.shape[:-1], FULL_STATE_DIM)


def variational_energy_full(
    y_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    y_free = free_state_from_full_state(project_fixed_vertices(y_full, physical))
    q_free = free_state_from_full_state(q_full)
    return variational_energy(y_free, q_free, masses, physical)


def stationarity_residual_full(
    y_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    y_free = free_state_from_full_state(project_fixed_vertices(y_full, physical))
    q_free = free_state_from_full_state(q_full)
    residual_free = stationarity_residual(y_free, q_free, masses, physical)
    return full_vector_from_free_vector(residual_free)


def stationarity_residual_norm_full(
    y_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    residual_free = free_state_from_full_state(
        stationarity_residual_full(y_full, q_full, masses, physical)
    )
    return torch.linalg.vector_norm(residual_free, dim=-1)


def spring_lengths_from_full(y_full: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    return spring_lengths_from_free(
        free_state_from_full_state(project_fixed_vertices(y_full, physical)),
        physical,
    )


def apply_newton_update_full(
    y_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_next_free, delta_free = apply_newton_update(
        free_state_from_full_state(project_fixed_vertices(y_full, physical)),
        free_state_from_full_state(q_full),
        masses,
        physical,
    )
    y_next_full = project_fixed_vertices(full_state_from_free_state(y_next_free, physical), physical)
    delta_full = full_vector_from_free_vector(delta_free)
    return y_next_full, delta_full


def apply_gradient_descent_update_full(
    y_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    step_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_next_free, delta_free = apply_gradient_descent_update(
        free_state_from_full_state(project_fixed_vertices(y_full, physical)),
        free_state_from_full_state(q_full),
        masses,
        physical,
        step_size,
    )
    y_next_full = project_fixed_vertices(full_state_from_free_state(y_next_free, physical), physical)
    delta_full = full_vector_from_free_vector(delta_free)
    return y_next_full, delta_full


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


def stationarity_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
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
    return full_grad[..., list(FREE_VERTEX_INDICES), :].reshape(*y.shape[:-1], FREE_STATE_DIM)


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
    initial_y_free, sampling = generate_sobol_points(
        count=size,
        center=problem.exact_y_free,
        radius=problem.sampling_radius,
        seed=seed,
        physical=physical,
        explicit_points=explicit,
    )
    initial_y = project_fixed_vertices(full_state_from_free_state(initial_y_free, physical), physical)
    q = project_fixed_vertices(
        full_state_from_free_state(problem.q_free.reshape(1, -1).expand(size, -1).clone(), physical),
        physical,
    )
    exact_y = project_fixed_vertices(
        full_state_from_free_state(problem.exact_y_free.reshape(1, -1).expand(size, -1).clone(), physical),
        physical,
    )
    return DatasetBundle(
        initial_y=initial_y,
        q=q,
        masses=problem.free_masses.reshape(1, -1).expand(size, -1).clone(),
        exact_y=exact_y,
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
            "state_representation": "full_75d_positions_with_fixed_vertices_projected",
            "q_representation": "full_75d; fixed entries are placeholders and ignored by reduced physics",
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
    physical: PhysicalConfig,
) -> DatasetBundle:
    records: list[DatasetBundle] = []
    for motion_index in motion_indices:
        for time_index in time_indices:
            problem = lookup[(int(motion_index), int(time_index))]
            if state == "current":
                y0 = project_fixed_vertices(full_state_from_positions(problem.p_n_full), physical)
            elif state == "exact":
                y0 = project_fixed_vertices(full_state_from_free_state(problem.exact_y_free.reshape(1, -1), physical), physical).squeeze(0)
            else:
                raise ValueError(state)
            q = project_fixed_vertices(full_state_from_free_state(problem.q_free.reshape(1, -1), physical), physical)
            exact_y = project_fixed_vertices(full_state_from_free_state(problem.exact_y_free.reshape(1, -1), physical), physical)
            records.append(
                DatasetBundle(
                    initial_y=y0.reshape(1, -1),
                    q=q,
                    masses=problem.free_masses.reshape(1, -1),
                    exact_y=exact_y,
                    problem_index=torch.tensor([problem.index], dtype=torch.long),
                    motion_index=torch.tensor([problem.motion_index], dtype=torch.long),
                    time_index=torch.tensor([problem.local_time_index], dtype=torch.long),
                    metadata={
                        "problem_index": problem.index,
                        "motion_index": problem.motion_index,
                        "state": state,
                        "state_representation": "full_75d_positions_with_fixed_vertices_projected",
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




def mass_preconditioned_residual_full(
    y_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    residual_free = stationarity_residual(
        free_state_from_full_state(project_fixed_vertices(y_full, physical)),
        free_state_from_full_state(q_full),
        masses,
        physical,
    )
    mass_per_coordinate = masses.repeat_interleave(3, dim=-1)
    preconditioned_free = physical.dt**2 * residual_free / mass_per_coordinate
    return full_vector_from_free_vector(preconditioned_free)


def make_activation(name: str) -> nn.Module:
    if name == "identity":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def activation_gain(name: str) -> float:
    if name == "identity":
        return 1.0
    if name == "relu":
        return math.sqrt(2.0)
    if name == "tanh":
        return 5.0 / 3.0
    raise ValueError(f"Unsupported activation: {name}")


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
            model_spec.width, FULL_STATE_DIM, bias=model_spec.use_bias
        )

        gain = activation_gain(model_spec.activation)
        for layer in self.hidden_layers:
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

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
        history_text = "current/full residual + previous/full residual + previous/full update" if USE_HISTORY_INPUT else "current/full residual only"
        return (
            f"{MODEL_INPUT_DIM}D input ({history_text}) -> "
            f"[{self.model_spec.width}, {self.model_spec.activation}] x "
            f"{self.model_spec.depth} -> {FULL_STATE_DIM}D update; "
            f"fixed vertices are hard-projected after output ({bias_text})"
        )

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        *,
        physical: PhysicalConfig,
        previous_residual: torch.Tensor | None = None,
        previous_update: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_residual = mass_preconditioned_residual_full(y, q, masses, physical)
        if USE_HISTORY_INPUT:
            if previous_residual is None:
                previous_residual = torch.zeros_like(current_residual)
            if previous_update is None:
                previous_update = torch.zeros_like(current_residual)
            h = torch.cat([current_residual, previous_residual, previous_update], dim=-1)
        else:
            h = current_residual
        h = h / self.residual_length_scale
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        raw_delta = self.residual_length_scale * self.output_layer(h)
        gated_delta = raw_delta * free_update_gate_like(raw_delta)
        return gated_delta, current_residual


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    previous_residual: torch.Tensor | None = None,
    previous_update: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta, current_residual = model(
        y,
        q,
        masses,
        physical=physical,
        previous_residual=previous_residual,
        previous_update=previous_update,
    )
    y_next = project_fixed_vertices(y + delta, physical)
    return y_next, delta, current_residual


def physical_energy_scale(
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_length_scale: float,
) -> float:
    return float(masses.mean().item()) * residual_length_scale**2 / physical.dt**2






# =============================================================================
# 5. Baseline optimizer helpers used by script 04 and rollout script 07
# =============================================================================


@dataclass
class AdamState:
    step: int
    first_moment: torch.Tensor
    second_moment: torch.Tensor


def initial_adam_state_like(y_full: torch.Tensor) -> AdamState:
    zeros = torch.zeros_like(y_full)
    return AdamState(step=0, first_moment=zeros.clone(), second_moment=zeros.clone())


def apply_adam_update_full(
    y_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    state: AdamState | None,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, AdamState]:
    """One Adam step on the implicit-Euler energy, treating y as the optimized variable."""
    if state is None:
        state = initial_adam_state_like(y_full)
    grad = stationarity_residual_full(y_full, q_full, masses, physical) * free_update_gate_like(y_full)
    next_step = state.step + 1
    m = beta1 * state.first_moment + (1.0 - beta1) * grad
    v = beta2 * state.second_moment + (1.0 - beta2) * grad.square()
    m_hat = m / (1.0 - beta1**next_step)
    v_hat = v / (1.0 - beta2**next_step)
    delta = -float(learning_rate) * m_hat / (torch.sqrt(v_hat) + eps)
    delta = delta * free_update_gate_like(y_full)
    y_next = project_fixed_vertices(y_full + delta, physical)
    return y_next, delta, AdamState(next_step, m.detach(), v.detach())


def run_lbfgs_iterations_full(
    y0_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    *,
    steps: int,
    learning_rate: float,
    history_size: int = 10,
    line_search_fn: str | None = None,
) -> list[torch.Tensor]:
    """Return states [y_0, ..., y_steps] from PyTorch L-BFGS on y.

    This intentionally keeps the implementation short. For batched inputs, PyTorch
    optimizes the summed objective over the batch.
    """
    y_param = nn.Parameter(project_fixed_vertices(y0_full.detach().clone(), physical))
    optimizer = torch.optim.LBFGS(
        [y_param],
        lr=float(learning_rate),
        max_iter=1,
        history_size=int(history_size),
        line_search_fn=line_search_fn,
    )
    states = [project_fixed_vertices(y_param.detach().clone(), physical)]
    for _ in range(int(steps)):
        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            y_projected = project_fixed_vertices(y_param, physical)
            energy = variational_energy_full(y_projected, q_full, masses, physical).sum()
            energy.backward()
            if y_param.grad is not None:
                y_param.grad.mul_(free_update_gate_like(y_param.grad))
            return energy

        optimizer.step(closure)
        with torch.no_grad():
            y_param.copy_(project_fixed_vertices(y_param, physical))
        states.append(project_fixed_vertices(y_param.detach().clone(), physical))
    return states
