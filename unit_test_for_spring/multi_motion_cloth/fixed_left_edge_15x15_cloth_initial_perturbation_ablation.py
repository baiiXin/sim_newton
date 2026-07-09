"""
Fixed-left-edge 15x15 triangular-cloth learned optimizer:
initial-perturbation coverage ablation.

Experiment implemented in this script
-------------------------------------
1. Keep the confirmed 32-motion / 100-step physical problem, complete-motion
   train/validation/ID/OOD split, network, optimizer, K schedule, evaluation
   metrics, numerical baselines, and all original per-model plots.
2. Remove the two explicit training starts (the current state p_n and the exact
   solution y*). Every reported N is therefore exactly N perturbed starts.
3. Compare static L-infinity-cube Sobol sampling and static radially stratified
   Sobol sampling at N in {32, 64, 128, 256} starts per motion-time problem.
4. Add online radially stratified sampling. Its perturbations are regenerated
   every epoch while q, masses, exact states, motion split, and time steps stay
   fixed. The default online budget is N=32 per problem.
5. Before reference generation or training, run a real CUDA full-batch forward,
   K-step unroll, backward, gradient clipping, and Adam update at the largest
   requested N and K. On CUDA OOM the report is saved and the program stops;
   it never silently switches to micro-batches.
6. Validate at a denser regular interval and force validation at the beginning
   and end of every K stage, so every unroll-length stage contributes checkpoint
   candidates.
7. Preserve all original plots and add cross-experiment plots for sample count,
   normalized perturbation radius, success rate, training cost, and peak memory.

Reference solutions are used only for dataset construction and diagnostics.
They are never network inputs or supervised labels. Training still uses the
physical variational-energy objective only.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# 0. Constants and triangular cloth topology
# =============================================================================

GRID_ROWS = 15
GRID_COLS = 15
SPATIAL_DIM = 3
NUM_PARTICLES = GRID_ROWS * GRID_COLS
FIXED_VERTEX_INDICES = (0, (GRID_ROWS - 1) * GRID_COLS)  # left-top and left-bottom
FREE_VERTEX_INDICES = tuple(
    index for index in range(NUM_PARTICLES) if index not in set(FIXED_VERTEX_INDICES)
)
NUM_FREE_PARTICLES = len(FREE_VERTEX_INDICES)
FREE_STATE_DIM = NUM_FREE_PARTICLES * SPATIAL_DIM
HIDDEN_DIM = FREE_STATE_DIM


def grid_index(row: int, col: int) -> int:
    return row * GRID_COLS + col


def build_triangular_cloth_topology() -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Return unique spring edges and triangle faces for an alternating 15x15 mesh."""
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

ACTIVATION_NAME = "identity"
OPTIMIZER_NAME = "adam"
LEARNING_RATE = 1e-3
DEFAULT_DEVICE = "cuda:0"

DEFAULT_TOTAL_TIME_STEPS = 100
DEFAULT_TRAIN_POINTS_PER_PROBLEM = 32
DEFAULT_EVAL_POINTS_PER_PROBLEM = 128
DEFAULT_EPOCHS = 500
DEFAULT_VALIDATION_INTERVAL = 20
DEFAULT_DIAGNOSTIC_INTERVAL = 50
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8192
# Full Newton materializes a dense 669x669 Hessian per sample. Keep the
# user-facing evaluation batch unchanged for MLP/GD, but cap only Newton.
NEWTON_MAX_BATCH_SIZE = 8
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 100
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5
DEFAULT_REPORT_STEPS = (1, 5, 10, 50)
DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_GRADIENT_CLIP_NORM = 1.0
DEFAULT_SAMPLING_RADIUS_MIN = 1e-2
DEFAULT_SAMPLING_RADIUS_MAX = 1e-1
DEFAULT_ABLATION_SAMPLE_COUNTS = (32, 64, 128, 256)
DEFAULT_ONLINE_SAMPLE_COUNTS = (32,)
DEFAULT_RADIUS_EVAL_POINTS_PER_PROBLEM = 32
SUCCESS_RESIDUAL_THRESHOLDS = (1e-2, 1e-4, 1e-6)
STRATIFIED_RADIUS_BINS = (
    ("near", 0.05, 0.25),
    ("medium", 0.25, 0.50),
    ("far", 0.50, 0.75),
    ("boundary", 0.75, 1.00),
)
RADIUS_EVALUATION_BINS = (
    ("rho_005_025", 0.05, 0.25),
    ("rho_025_050", 0.25, 0.50),
    ("rho_050_075", 0.50, 0.75),
    ("rho_075_100", 0.75, 1.00),
    ("rho_100_200", 1.00, 2.00),
)
ONLINE_EPOCH_SEED_STRIDE = 10_000_019

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
    ablation_sample_counts: tuple[int, ...]
    online_sample_counts: tuple[int, ...]
    eval_points_per_problem: int
    radius_eval_points_per_problem: int
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
    skip_plots: bool
    save_datasets: bool
    run_memory_test: bool
    memory_test_only: bool
    continue_after_memory_test_oom: bool


@dataclass(frozen=True)
class TrainingExperimentSpec:
    name: str
    sampling_mode: str
    points_per_problem: int
    online: bool


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


def initialize_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    validate_device(device)
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.empty((), dtype=TORCH_DTYPE, device=device)
    torch.cuda.synchronize(device)


def get_k_for_epoch(epoch_index: int, config: RuntimeConfig) -> int:
    return min(
        config.initial_k + (epoch_index // config.k_increase_interval) * config.k_increase_amount,
        config.max_k,
    )


def finite_plot_value(value: float | int | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return float("nan")
    return max(float(value), PLOT_FLOOR)


# =============================================================================
# 2. Fixed-left-edge 15x15 triangular-cloth physics
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
    """Implicit-Euler variational energy for the 223 free cloth vertices."""
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
    active = residual > residual_tolerance
    if not bool(torch.any(active)):
        return y, torch.zeros_like(y)
    rhs = torch.where(active, -gradient, torch.zeros_like(gradient))
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


def bounded_sobol_seed(seed: int) -> int:
    # Keep epoch-dependent seeds inside the range accepted by all supported
    # PyTorch SobolEngine implementations.
    return int(seed) % 2_147_483_647


def normalized_linf_radius(points: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    return torch.amax(torch.abs(points - center.reshape(1, -1)), dim=-1) / float(radius)


def _draw_radially_stratified_candidates(
    *,
    draw_count: int,
    center: torch.Tensor,
    radius: float,
    direction_engine: torch.quasirandom.SobolEngine,
    radial_engine: torch.quasirandom.SobolEngine,
    generated_before: int,
    normalized_radius_range: tuple[float, float] | None,
) -> torch.Tensor:
    unit = direction_engine.draw(draw_count).to(dtype=TORCH_DTYPE)
    direction = 2.0 * unit - 1.0
    direction = direction / torch.amax(torch.abs(direction), dim=-1, keepdim=True).clamp_min(1e-15)
    within = radial_engine.draw(draw_count).to(dtype=TORCH_DTYPE).reshape(-1)

    if normalized_radius_range is not None:
        low, high = normalized_radius_range
        rho = low + (high - low) * within
    else:
        bin_indices = (torch.arange(draw_count, dtype=torch.long) + generated_before) % len(STRATIFIED_RADIUS_BINS)
        lows = torch.tensor([item[1] for item in STRATIFIED_RADIUS_BINS], dtype=TORCH_DTYPE)
        highs = torch.tensor([item[2] for item in STRATIFIED_RADIUS_BINS], dtype=TORCH_DTYPE)
        rho = lows[bin_indices] + (highs[bin_indices] - lows[bin_indices]) * within
    return center.reshape(1, -1) + radius * rho[:, None] * direction


def generate_sobol_points(
    *,
    count: int,
    center: torch.Tensor,
    radius: float,
    seed: int,
    physical: PhysicalConfig,
    sampling_mode: str = "cube",
    normalized_radius_range: tuple[float, float] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generate perturbed starts only; no explicit current/exact states are inserted."""
    if count <= 0 or radius <= 0.0:
        raise ValueError("count and radius must be positive")
    if sampling_mode not in {"cube", "stratified", "radial_range"}:
        raise ValueError(f"Unknown sampling_mode={sampling_mode}")
    if sampling_mode == "radial_range":
        if normalized_radius_range is None:
            raise ValueError("radial_range sampling requires normalized_radius_range")
        low, high = normalized_radius_range
        if low < 0.0 or high <= low:
            raise ValueError("Invalid normalized_radius_range")
    elif normalized_radius_range is not None:
        raise ValueError("normalized_radius_range is only valid for radial_range")

    chunks: list[torch.Tensor] = []
    accepted = generated = rejected = 0
    direction_engine = torch.quasirandom.SobolEngine(
        dimension=FREE_STATE_DIM, scramble=True, seed=bounded_sobol_seed(seed)
    )
    radial_engine = torch.quasirandom.SobolEngine(
        dimension=1, scramble=True, seed=bounded_sobol_seed(seed + 7_919)
    )

    while accepted < count:
        remaining = count - accepted
        draw_count = max(32, remaining * 2)
        if sampling_mode == "cube":
            unit = direction_engine.draw(draw_count).to(dtype=TORCH_DTYPE)
            candidates = center.reshape(1, -1) + (2.0 * unit - 1.0) * radius
            # Keep the radial engine aligned only conceptually; it is unused for cube sampling.
        else:
            candidates = _draw_radially_stratified_candidates(
                draw_count=draw_count,
                center=center,
                radius=radius,
                direction_engine=direction_engine,
                radial_engine=radial_engine,
                generated_before=generated,
                normalized_radius_range=normalized_radius_range,
            )
        keep = nondegenerate_mask(candidates, physical)
        selected = candidates[keep][:remaining]
        generated += draw_count
        rejected += int((~keep).sum().item())
        if selected.numel() > 0:
            chunks.append(selected)
            accepted += int(selected.shape[0])

    points = torch.cat(chunks, dim=0)[:count].contiguous()
    rho = normalized_linf_radius(points, center, radius)
    return points, {
        "mode": sampling_mode,
        "seed": seed,
        "count": count,
        "center": center.tolist(),
        "radius_linf": radius,
        "normalized_radius_range": normalized_radius_range,
        "explicit_point_count": 0,
        "generated_candidates": generated,
        "rejected_degenerate_candidates": rejected,
        "normalized_radius_min": float(rho.min().item()),
        "normalized_radius_mean": float(rho.mean().item()),
        "normalized_radius_max": float(rho.max().item()),
    }


def build_problem_dataset(
    *,
    problem: TimeStepProblem,
    size: int,
    seed: int,
    role: str,
    physical: PhysicalConfig,
    sampling_mode: str = "cube",
    normalized_radius_range: tuple[float, float] | None = None,
) -> DatasetBundle:
    initial_y, sampling = generate_sobol_points(
        count=size,
        center=problem.exact_y_free,
        radius=problem.sampling_radius,
        seed=seed,
        physical=physical,
        sampling_mode=sampling_mode,
        normalized_radius_range=normalized_radius_range,
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
    first_sampling = datasets[0].metadata.get("sampling", {})
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
            "explicit_training_starts": 0,
            "sampling_mode": first_sampling.get("mode"),
            "normalized_radius_range": first_sampling.get("normalized_radius_range"),
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
    sampling_mode: str = "cube",
    normalized_radius_range: tuple[float, float] | None = None,
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
                    sampling_mode=sampling_mode,
                    normalized_radius_range=normalized_radius_range,
                )
            )
    return concatenate_datasets(datasets, role=role, points_per_problem=points_per_problem)


class OnlineTrainingSampler:
    """Regenerate radially stratified training starts every epoch."""

    def __init__(
        self,
        *,
        lookup: dict[tuple[int, int], TimeStepProblem],
        motion_indices: Sequence[int],
        time_indices: Sequence[int],
        points_per_problem: int,
        base_seed: int,
        role: str,
        physical: PhysicalConfig,
    ) -> None:
        self.problems = [
            lookup[(int(motion_index), int(time_index))]
            for motion_index in motion_indices
            for time_index in time_indices
        ]
        self.points_per_problem = int(points_per_problem)
        self.base_seed = int(base_seed)
        self.role = role
        self.physical = physical
        self.centers = torch.stack([p.exact_y_free for p in self.problems], dim=0)
        self.radii = torch.tensor([p.sampling_radius for p in self.problems], dtype=TORCH_DTYPE)
        self.template = self._make_template(self.sample_initial_y(epoch_index=0))
        self.template.metadata.update({
            "online_sampling": True,
            "sampling_mode": "online_stratified",
            "online_epoch_seed_stride": ONLINE_EPOCH_SEED_STRIDE,
        })

    def _make_template(self, initial_y: torch.Tensor) -> DatasetBundle:
        size = len(self.problems) * self.points_per_problem
        q = torch.cat([
            p.q_free.reshape(1, -1).expand(self.points_per_problem, -1)
            for p in self.problems
        ], dim=0).clone()
        masses = torch.cat([
            p.free_masses.reshape(1, -1).expand(self.points_per_problem, -1)
            for p in self.problems
        ], dim=0).clone()
        exact_y = torch.cat([
            p.exact_y_free.reshape(1, -1).expand(self.points_per_problem, -1)
            for p in self.problems
        ], dim=0).clone()
        return DatasetBundle(
            initial_y=initial_y,
            q=q,
            masses=masses,
            exact_y=exact_y,
            problem_index=torch.cat([
                torch.full((self.points_per_problem,), p.index, dtype=torch.long)
                for p in self.problems
            ]),
            motion_index=torch.cat([
                torch.full((self.points_per_problem,), p.motion_index, dtype=torch.long)
                for p in self.problems
            ]),
            time_index=torch.cat([
                torch.full((self.points_per_problem,), p.local_time_index, dtype=torch.long)
                for p in self.problems
            ]),
            metadata={
                "role": self.role,
                "problem_indices": [p.index for p in self.problems],
                "motion_indices": sorted(set(p.motion_index for p in self.problems)),
                "num_motions": len(set(p.motion_index for p in self.problems)),
                "num_problems": len(self.problems),
                "points_per_problem": self.points_per_problem,
                "size": size,
                "split_unit": "complete_motion",
                "no_motion_leakage": True,
                "explicit_training_starts": 0,
            },
        )

    def sample_initial_y(self, epoch_index: int) -> torch.Tensor:
        num_problems = len(self.problems)
        n = self.points_per_problem
        total = num_problems * n
        seed = bounded_sobol_seed(self.base_seed + ONLINE_EPOCH_SEED_STRIDE * int(epoch_index))
        direction_engine = torch.quasirandom.SobolEngine(
            dimension=FREE_STATE_DIM, scramble=True, seed=bounded_sobol_seed(seed)
        )
        radial_engine = torch.quasirandom.SobolEngine(
            dimension=1, scramble=True, seed=bounded_sobol_seed(seed + 7_919)
        )
        unit = direction_engine.draw(total).to(dtype=TORCH_DTYPE).reshape(num_problems, n, FREE_STATE_DIM)
        direction = 2.0 * unit - 1.0
        direction = direction / torch.amax(torch.abs(direction), dim=-1, keepdim=True).clamp_min(1e-15)
        within = radial_engine.draw(total).to(dtype=TORCH_DTYPE).reshape(num_problems, n)
        bin_indices = torch.arange(n, dtype=torch.long) % len(STRATIFIED_RADIUS_BINS)
        lows = torch.tensor([item[1] for item in STRATIFIED_RADIUS_BINS], dtype=TORCH_DTYPE)[bin_indices]
        highs = torch.tensor([item[2] for item in STRATIFIED_RADIUS_BINS], dtype=TORCH_DTYPE)[bin_indices]
        rho = lows.reshape(1, n) + (highs - lows).reshape(1, n) * within
        candidates = self.centers[:, None, :] + self.radii[:, None, None] * rho[:, :, None] * direction
        flat = candidates.reshape(total, FREE_STATE_DIM).contiguous()

        keep = nondegenerate_mask(flat, self.physical)
        if bool(torch.all(keep)):
            return flat

        # Degeneracy is very unlikely at the configured radii. Rebuild only the
        # affected problems with the robust rejection sampler if it occurs.
        chunks: list[torch.Tensor] = []
        keep_by_problem = keep.reshape(num_problems, n)
        for problem_idx, problem in enumerate(self.problems):
            if bool(torch.all(keep_by_problem[problem_idx])):
                chunks.append(candidates[problem_idx])
            else:
                replacement, _ = generate_sobol_points(
                    count=n,
                    center=problem.exact_y_free,
                    radius=problem.sampling_radius,
                    seed=seed + 100_003 * problem.motion_index + 1009 * problem.local_time_index,
                    physical=self.physical,
                    sampling_mode="stratified",
                )
                chunks.append(replacement)
        return torch.cat(chunks, dim=0).contiguous()


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
    residual = stationarity_residual(y, q, masses, physical)
    mass_per_coordinate = masses.repeat_interleave(3, dim=-1)
    return physical.dt**2 * residual / mass_per_coordinate


class MLPOptimizer(nn.Module):
    def __init__(self, residual_length_scale: float) -> None:
        super().__init__()
        if residual_length_scale <= 0.0:
            raise ValueError("residual_length_scale must be positive")
        self.linear1 = nn.Linear(FREE_STATE_DIM, HIDDEN_DIM, bias=False)
        self.activation = nn.Identity()
        self.linear2 = nn.Linear(HIDDEN_DIM, FREE_STATE_DIM, bias=False)
        nn.init.orthogonal_(self.linear1.weight)
        nn.init.zeros_(self.linear2.weight)
        self.register_buffer(
            "residual_length_scale",
            torch.tensor(float(residual_length_scale), dtype=TORCH_DTYPE),
        )

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        *,
        physical: PhysicalConfig,
    ) -> torch.Tensor:
        u = mass_preconditioned_residual(y, q, masses, physical)
        u = u / self.residual_length_scale
        return self.residual_length_scale * self.linear2(
            self.activation(self.linear1(u))
        )


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = model(y, q, masses, physical=physical)
    return y + delta, delta


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

    requested_batch_size = int(batch_size)
    effective_batch_size = (
        min(requested_batch_size, NEWTON_MAX_BATCH_SIZE)
        if solver == "full_newton"
        else requested_batch_size
    )

    metric_batches: dict[str, list[torch.Tensor]] = {}
    problem_batches: list[torch.Tensor] = []
    motion_batches: list[torch.Tensor] = []
    time_batches: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for start in range(0, len(dataset_cpu), effective_batch_size):
        end = min(start + effective_batch_size, len(dataset_cpu))
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
        exact_energy = variational_energy(batch.exact_y, batch.q, batch.masses, physical)
        step_values: dict[str, list[torch.Tensor]] = {}
        for step in range(steps + 1):
            for name, values in _state_metrics(y, batch, exact_energy, physical).items():
                step_values.setdefault(name, []).append(values.detach().cpu())
            if step == steps:
                break
            if solver == "learned":
                assert model is not None
                y, _ = apply_model_update(model, y, batch.q, batch.masses, physical)
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
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": effective_batch_size,
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

    residual_values = arrays["residual"]
    for threshold in SUCCESS_RESIDUAL_THRESHOLDS:
        suffix = f"lt_{threshold:.0e}".replace("-", "m")
        rates = []
        for step in range(residual_values.shape[1]):
            current = residual_values[:, step]
            rates.append(float(np.mean(np.isfinite(current) & (current < threshold))))
        result[f"residual_success_rate_{suffix}_by_step"] = rates
        result[f"final_residual_success_rate_{suffix}"] = rates[-1]
    initial_residual = residual_values[:, 0]
    final_residual = residual_values[:, -1]
    diverged = (
        ~np.isfinite(final_residual)
        | (final_residual > 10.0 * np.maximum(initial_residual, PLOT_FLOOR))
    )
    result["final_divergence_rate"] = float(np.mean(diverged))
    result["final_nonimprovement_rate"] = float(np.mean(
        ~np.isfinite(final_residual) | (final_residual >= initial_residual)
    ))
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
        y1, delta = apply_model_update(model, y0, dataset.q, dataset.masses, physical)
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


def should_validate_epoch(epoch: int, config: RuntimeConfig) -> bool:
    if epoch <= 0 or epoch > config.epochs:
        raise ValueError("epoch out of range")
    k_now = get_k_for_epoch(epoch - 1, config)
    k_before = get_k_for_epoch(epoch - 2, config) if epoch > 1 else None
    k_after = get_k_for_epoch(epoch, config) if epoch < config.epochs else None
    is_stage_start = epoch == 1 or k_before != k_now
    is_stage_end = epoch == config.epochs or k_after != k_now
    return (
        epoch % config.validation_interval == 0
        or is_stage_start
        or is_stage_end
        or epoch == config.epochs
    )


def _cuda_memory_snapshot(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run_maximum_memory_test(
    *,
    config: RuntimeConfig,
    physical: PhysicalConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Probe the largest requested full-batch configuration before the real run."""
    device = torch.device(config.device)
    max_points = max((*config.ablation_sample_counts, *config.online_sample_counts))
    num_problems = len(range(16)) * len(TRAIN_TIME_INDICES)
    batch_size = num_problems * max_points
    test_k = config.max_k
    report: dict[str, Any] = {
        "device": str(device),
        "dtype": str(TORCH_DTYPE),
        "num_training_problems": num_problems,
        "points_per_problem": max_points,
        "full_batch_size": batch_size,
        "unroll_steps_k": test_k,
        "micro_batching_used": False,
        "success": False,
    }
    if device.type != "cuda":
        report.update({
            "status": "skipped_non_cuda_device",
            "message": "The maximum-memory test is meaningful only on CUDA.",
        })
        save_json(report, output_dir / "maximum_memory_test.json")
        return report

    initialize_cuda_device(device)
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    report["before"] = _cuda_memory_snapshot(device)
    objects: dict[str, Any] = {}
    base_full = base = initial_y = q = exact_y = masses = None
    model = optimizer = initial_energy = exact_energy = None
    y = objective = energy_gap_sum = energy = grad_norm = None
    try:
        base_full = torch.tensor(physical.p0, dtype=TORCH_DTYPE, device=device)
        base = free_state_from_full(base_full).reshape(1, -1)
        initial_y = base.expand(batch_size, -1).clone()
        initial_y.add_(0.05 * (2.0 * torch.rand_like(initial_y) - 1.0))
        q = base.expand(batch_size, -1).clone()
        exact_y = base.expand(batch_size, -1).clone()
        masses = torch.ones((batch_size, NUM_FREE_PARTICLES), dtype=TORCH_DTYPE, device=device)
        model = MLPOptimizer(config.residual_length_scale).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        objects.update(
            initial_y=initial_y, q=q, exact_y=exact_y, masses=masses,
            model=model, optimizer=optimizer,
        )
        energy_scale = physical_energy_scale(masses, physical, config.residual_length_scale)
        initial_energy = variational_energy(initial_y, q, masses, physical).detach()
        exact_energy = variational_energy(exact_y, q, masses, physical).detach()
        optimizer.zero_grad(set_to_none=True)
        y = initial_y
        objective = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        energy_gap_sum = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        for _ in range(test_k):
            y, _ = apply_model_update(model, y, q, masses, physical)
            energy = variational_energy(y, q, masses, physical)
            objective = objective + ((energy - initial_energy) / energy_scale).mean()
            energy_gap_sum = energy_gap_sum + (energy - exact_energy).mean()
        objective.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
        torch.cuda.synchronize(device)
        report.update({
            "success": True,
            "status": "fit",
            "objective": float(objective.item()),
            "energy_gap_sum": float(energy_gap_sum.item()),
            "gradient_norm_before_clip": float(grad_norm.item()),
            "after": _cuda_memory_snapshot(device),
        })
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        report.update({
            "success": False,
            "status": "cuda_oom" if is_oom else "runtime_error",
            "error": str(exc),
            "at_failure": _cuda_memory_snapshot(device),
        })
        if not is_oom:
            save_json(report, output_dir / "maximum_memory_test.json")
            raise
    finally:
        objects.clear()
        base_full = base = initial_y = q = exact_y = masses = None
        model = optimizer = initial_energy = exact_energy = None
        y = objective = energy_gap_sum = energy = grad_norm = None
        gc.collect()
        torch.cuda.empty_cache()
        report["after_cleanup"] = _cuda_memory_snapshot(device)
        save_json(report, output_dir / "maximum_memory_test.json")
    return report


def run_experiment(
    *,
    experiment_spec: TrainingExperimentSpec,
    training_cpu: DatasetBundle,
    validation_cpu: DatasetBundle,
    evaluation_datasets: dict[str, DatasetBundle],
    perturbation_evaluation_datasets: dict[str, DatasetBundle],
    output_dir: Path,
    config: RuntimeConfig,
    physical: PhysicalConfig,
    gd_step_size: float,
    shared_baselines: dict[str, dict[str, Any]],
    online_sampler: OnlineTrainingSampler | None = None,
) -> dict[str, Any]:
    experiment_name = experiment_spec.name
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        initialize_cuda_device(device)
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = MLPOptimizer(config.residual_length_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    training = training_cpu.to(device)
    energy_scale = physical_energy_scale(training.masses, physical, config.residual_length_scale)
    exact_energy = variational_energy(training.exact_y, training.q, training.masses, physical).detach()
    static_initial_energy = None
    if online_sampler is None:
        static_initial_energy = variational_energy(
            training.initial_y, training.q, training.masses, physical
        ).detach()

    print("\n" + "=" * 100)
    print(f"Training {experiment_name}")
    print(
        f"architecture={FREE_STATE_DIM}->{HIDDEN_DIM}->Identity->{FREE_STATE_DIM}, "
        f"points={len(training_cpu):,}, motions={training_cpu.metadata['num_motions']}, "
        f"problems={training_cpu.metadata['num_problems']}, sampling={experiment_spec.sampling_mode}, "
        f"online={experiment_spec.online}, device={device}, dtype=float64"
    )
    print("=" * 100)

    train_log: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []
    diagnostic_log: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, ...] | None = None
    best_epoch: int | None = None
    start_time = time.perf_counter()

    for epoch_index in range(config.epochs):
        epoch = epoch_index + 1
        k = get_k_for_epoch(epoch_index, config)
        if online_sampler is not None:
            refreshed = online_sampler.sample_initial_y(epoch_index=epoch_index)
            training.initial_y = refreshed.to(device=device, dtype=TORCH_DTYPE)
            initial_energy = variational_energy(
                training.initial_y, training.q, training.masses, physical
            ).detach()
        else:
            assert static_initial_energy is not None
            initial_energy = static_initial_energy

        model.train()
        y = training.initial_y
        optimizer.zero_grad(set_to_none=True)
        objective = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        energy_gap_sum = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        for _ in range(k):
            y, _ = apply_model_update(model, y, training.q, training.masses, physical)
            energy = variational_energy(y, training.q, training.masses, physical)
            objective = objective + ((energy - initial_energy) / energy_scale).mean()
            energy_gap_sum = energy_gap_sum + (energy - exact_energy).mean()
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(f"Non-finite training objective at epoch {epoch}")
        objective.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm).item())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient norm at epoch {epoch}")
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError(f"Non-finite model parameter at epoch {epoch}")

        train_log.append({
            "epoch": epoch,
            "K": k,
            "dimensionless_objective": float(objective.item()),
            "training_energy_gap_sum": float(energy_gap_sum.item()),
            "gradient_norm_before_clip": grad_norm,
            "online_resampled": online_sampler is not None,
        })

        if epoch == 1 or epoch % config.diagnostic_interval == 0 or epoch == config.epochs:
            diagnostics = one_step_diagnostics(model, training, physical)
            diagnostics.update(epoch=epoch, K=k)
            diagnostic_log.append(diagnostics)

        if should_validate_epoch(epoch, config):
            metrics = evaluate_solver_on_dataset(
                solver="learned", model=model, dataset_cpu=validation_cpu,
                physical=physical, steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size, report_steps=config.report_steps,
                device=device,
            )
            key = validation_selection_key(metrics)
            validation_log.append({"epoch": epoch, "K": k, "selection_key": key, "metrics": metrics})
            if key is not None and (best_key is None or key < best_key):
                best_key = key
                best_epoch = epoch
                best_state = state_dict_to_cpu(model)
            print(
                f"epoch={epoch:4d} K={k} objective={float(objective.item()):.4e} "
                f"val_res_p95={metrics['final_residual_p95']:.4e} "
                f"val_res_max={metrics['final_residual_max']:.4e} "
                f"worst_motion_max={metrics['worst_motion_final_residual_max']:.4e} "
                f"best_epoch={best_epoch} elapsed={time.perf_counter()-start_time:.1f}s"
            )

    training_elapsed = time.perf_counter() - start_time
    peak_memory = _cuda_memory_snapshot(device) if device.type == "cuda" else {}
    last_state = state_dict_to_cpu(model)
    if best_state is None:
        best_state = copy.deepcopy(last_state)
        best_epoch = config.epochs
        best_key = None
    torch.save(last_state, experiment_dir / "last_model_state_dict.pt")
    torch.save(best_state, experiment_dir / "best_validation_model_state_dict.pt")
    torch.save(best_state, experiment_dir / "mlp_optimizer_state_dict.pt")

    model.load_state_dict(best_state)
    model.to(device)
    learned_results: dict[str, Any] = {}
    for name, dataset in evaluation_datasets.items():
        learned_results[name] = evaluate_solver_on_dataset(
            solver="learned", model=model, dataset_cpu=dataset, physical=physical,
            steps=config.evaluation_steps, batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps, device=device,
        )
    perturbation_results: dict[str, Any] = {}
    for name, dataset in perturbation_evaluation_datasets.items():
        perturbation_results[name] = evaluate_solver_on_dataset(
            solver="learned", model=model, dataset_cpu=dataset, physical=physical,
            steps=config.evaluation_steps, batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps, device=device,
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
        "experiment_spec": asdict(experiment_spec),
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": best_key,
        "training_dataset": training_cpu.metadata,
        "validation_dataset": validation_cpu.metadata,
        "model": {
            "architecture": f"{FREE_STATE_DIM}D residual -> {HIDDEN_DIM} -> Identity -> {FREE_STATE_DIM}D update",
            "bias_free": True,
            "first_layer_initialization": "orthogonal",
            "output_layer_initialization": "zero",
            "residual_length_scale": config.residual_length_scale,
            "dtype": str(TORCH_DTYPE),
        },
        "training": {
            "optimizer": "Adam", "learning_rate": LEARNING_RATE, "full_batch": True,
            "micro_batching": False,
            "epochs": config.epochs, "gradient_clip_norm": config.gradient_clip_norm,
            "energy_scale": energy_scale,
            "elapsed_seconds": training_elapsed,
            "cuda_memory": peak_memory,
            "explicit_training_starts": 0,
            "online_sampling": experiment_spec.online,
        },
        "checkpoint_schedule": {
            "regular_validation_interval": config.validation_interval,
            "forced_k_stage_boundaries": True,
            "validation_epochs": [record["epoch"] for record in validation_log],
        },
        "metric_policy": {
            "pooled": ["mean", "median", "p95", "max"],
            "boundary": ["pooled max", "worst-motion p95", "worst-motion max"],
            "success_thresholds": list(SUCCESS_RESIDUAL_THRESHOLDS),
        },
        "gradient_descent_step_size": gd_step_size,
        "train_log": train_log,
        "diagnostic_log": diagnostic_log,
        "validation_log": validation_log,
        "evaluation": comparison,
        "perturbation_evaluation": perturbation_results,
    }
    save_json(report, experiment_dir / "experiment_report.json")

    if not config.skip_plots:
        # Preserve every original per-model plot.
        plot_training_curves(train_log, validation_log, best_epoch, experiment_dir / "training_and_validation.png")
        for split_name in ["seen_motion_temporal_interpolation", "seen_motion_temporal_extrapolation", "unseen_id_test", "ood_test"]:
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
    axes[0].plot([r["epoch"] for r in train_log], [finite_plot_value(r["training_energy_gap_sum"]) for r in train_log])
    axes[0].set_yscale("log")
    axes[0].set_title("Training energy-gap sum")
    val_epochs = [r["epoch"] for r in validation_log]
    specs = [
        ("final_residual_p95", "Validation residual p95"),
        ("final_residual_max", "Validation residual maximum"),
        ("worst_motion_final_residual_max", "Worst-motion residual maximum"),
    ]
    for ax, (key, title) in zip(axes[1:], specs):
        ax.plot(val_epochs, [finite_plot_value(r["metrics"][key]) for r in validation_log], marker="o")
        ax.set_yscale("log")
        ax.set_title(title)
    for ax in axes:
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
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


def _success_metric_key(threshold: float = 1e-4) -> str:
    return f"final_residual_success_rate_lt_{threshold:.0e}".replace("-", "m")


def compact_ablation_record(report: dict[str, Any]) -> dict[str, Any]:
    spec = report["experiment_spec"]
    record: dict[str, Any] = {
        **spec,
        "best_validation_epoch": report["best_validation_epoch"],
        "training_elapsed_seconds": report["training"]["elapsed_seconds"],
        "peak_memory_allocated_bytes": report["training"].get("cuda_memory", {}).get("max_allocated_bytes"),
        "peak_memory_reserved_bytes": report["training"].get("cuda_memory", {}).get("max_reserved_bytes"),
        "standard_evaluation": {},
        "radius_evaluation": {},
    }
    success_key = _success_metric_key(1e-4)
    for split_name in ["unseen_id_test", "ood_test"]:
        metrics = report["evaluation"][split_name]["learned"]
        record["standard_evaluation"][split_name] = {
            "residual_p95": metrics["final_residual_p95"],
            "residual_max": metrics["final_residual_max"],
            "success_rate_1e-4": metrics[success_key],
            "divergence_rate": metrics["final_divergence_rate"],
        }
    for dataset_name, metrics in report["perturbation_evaluation"].items():
        record["radius_evaluation"][dataset_name] = {
            "residual_p95": metrics["final_residual_p95"],
            "residual_max": metrics["final_residual_max"],
            "success_rate_1e-4": metrics[success_key],
            "divergence_rate": metrics["final_divergence_rate"],
        }
    return record


def _series_style(record: dict[str, Any]) -> tuple[str, str]:
    if record["online"]:
        return "online stratified", "X"
    if record["sampling_mode"] == "cube":
        return "static cube", "o"
    return "static stratified", "s"


def plot_initial_radius_distribution(
    *,
    lookup: dict[tuple[int, int], TimeStepProblem],
    physical: PhysicalConfig,
    max_points: int,
    save_path: Path,
) -> None:
    problem = lookup[(0, TRAIN_TIME_INDICES[0])]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode, label in [("cube", "static cube"), ("stratified", "radially stratified")]:
        points, _ = generate_sobol_points(
            count=max_points,
            center=problem.exact_y_free,
            radius=problem.sampling_radius,
            seed=TRAIN_SOBOL_SEED,
            physical=physical,
            sampling_mode=mode,
        )
        rho = normalized_linf_radius(points, problem.exact_y_free, problem.sampling_radius).numpy()
        ax.hist(rho, bins=np.linspace(0.0, 1.0, 21), alpha=0.55, label=label)
    ax.set_xlabel(r"Normalized initial radius $\rho=\|y^{(0)}-y^*\|_\infty/r$")
    ax.set_ylabel("Count")
    ax.set_title(f"Initial-radius coverage for one training problem (N={max_points})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_summary(records: Sequence[dict[str, Any]], save_dir: Path) -> None:
    if not records:
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    # Sample count versus standard OOD residual.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for category in ["static cube", "static stratified", "online stratified"]:
        selected = [r for r in records if _series_style(r)[0] == category]
        if not selected:
            continue
        selected = sorted(selected, key=lambda r: r["points_per_problem"])
        marker = _series_style(selected[0])[1]
        x = [r["points_per_problem"] for r in selected]
        axes[0].plot(x, [finite_plot_value(r["standard_evaluation"]["ood_test"]["residual_p95"]) for r in selected], marker=marker, label=category)
        axes[1].plot(x, [finite_plot_value(r["standard_evaluation"]["ood_test"]["residual_max"]) for r in selected], marker=marker, label=category)
    for ax, title in zip(axes, ["OOD residual p95", "OOD residual maximum"]):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Training starts per motion-time problem")
        ax.set_ylabel("Final residual")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "sample_count_vs_ood_residual.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Sample count versus standard OOD success rate.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for category in ["static cube", "static stratified", "online stratified"]:
        selected = [r for r in records if _series_style(r)[0] == category]
        if not selected:
            continue
        selected = sorted(selected, key=lambda r: r["points_per_problem"])
        marker = _series_style(selected[0])[1]
        ax.plot(
            [r["points_per_problem"] for r in selected],
            [r["standard_evaluation"]["ood_test"]["success_rate_1e-4"] for r in selected],
            marker=marker,
            label=category,
        )
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Training starts per motion-time problem")
    ax.set_ylabel(r"Success rate: final residual $<10^{-4}$")
    ax.set_title("OOD convergence success versus training-start count")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "sample_count_vs_ood_success_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    bin_labels = [item[0] for item in RADIUS_EVALUATION_BINS]
    display_labels = ["0.05–0.25", "0.25–0.50", "0.50–0.75", "0.75–1.00", "1.00–2.00"]
    for split_prefix, split_title in [("id", "Unseen ID motions"), ("ood", "OOD motions")]:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        for record in records:
            values_p95 = []
            values_success = []
            for label in bin_labels:
                key = f"initial_radius_{split_prefix}_{label}"
                values_p95.append(finite_plot_value(record["radius_evaluation"][key]["residual_p95"]))
                values_success.append(record["radius_evaluation"][key]["success_rate_1e-4"])
            category, marker = _series_style(record)
            line_label = f"{category}, N={record['points_per_problem']}"
            axes[0].plot(display_labels, values_p95, marker=marker, label=line_label)
            axes[1].plot(display_labels, values_success, marker=marker, label=line_label)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Final residual p95")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_ylabel(r"Success rate: residual $<10^{-4}$")
        for ax in axes:
            ax.set_xlabel(r"Normalized initial-radius interval $\rho$")
            ax.grid(True, alpha=0.3, which="both")
        axes[0].set_title(f"{split_title}: residual by perturbation radius")
        axes[1].set_title(f"{split_title}: success by perturbation radius")
        axes[1].legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(save_dir / f"radius_bin_performance_{split_prefix}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # Cost and peak-memory diagnostics.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for record in records:
        label, marker = _series_style(record)
        name = f"{label} N={record['points_per_problem']}"
        ood_p95 = finite_plot_value(record["standard_evaluation"]["ood_test"]["residual_p95"])
        axes[0].scatter(record["training_elapsed_seconds"] / 3600.0, ood_p95, marker=marker, s=60)
        axes[0].annotate(name, (record["training_elapsed_seconds"] / 3600.0, ood_p95), fontsize=7)
        peak = record.get("peak_memory_allocated_bytes")
        if peak is not None:
            axes[1].scatter(float(peak) / (1024.0**3), ood_p95, marker=marker, s=60)
            axes[1].annotate(name, (float(peak) / (1024.0**3), ood_p95), fontsize=7)
    axes[0].set_xlabel("Training time (hours)")
    axes[1].set_xlabel("Peak allocated CUDA memory (GiB)")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylabel("OOD final residual p95")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_title("Training cost versus OOD performance")
    axes[1].set_title("Peak memory versus OOD performance")
    plt.tight_layout()
    plt.savefig(save_dir / "cost_and_memory_vs_ood_performance.png", dpi=220, bbox_inches="tight")
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
        description="15x15 cloth initial-perturbation coverage ablation"
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--total-time-steps", type=int, default=DEFAULT_TOTAL_TIME_STEPS)
    parser.add_argument(
        "--ablation-sample-counts", type=int, nargs="+",
        default=list(DEFAULT_ABLATION_SAMPLE_COUNTS),
        help="Static cube and static stratified starts per problem.",
    )
    parser.add_argument(
        "--online-sample-counts", type=int, nargs="+",
        default=list(DEFAULT_ONLINE_SAMPLE_COUNTS),
        help="Online stratified starts per problem. Default: 32.",
    )
    parser.add_argument("--eval-points-per-problem", type=int, default=DEFAULT_EVAL_POINTS_PER_PROBLEM)
    parser.add_argument(
        "--radius-eval-points-per-problem", type=int,
        default=DEFAULT_RADIUS_EVAL_POINTS_PER_PROBLEM,
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--diagnostic-interval", type=int, default=DEFAULT_DIAGNOSTIC_INTERVAL)
    parser.add_argument("--evaluation-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument("--k-increase-interval", type=int, default=DEFAULT_K_INCREASE_INTERVAL)
    parser.add_argument("--k-increase-amount", type=int, default=DEFAULT_K_INCREASE_AMOUNT)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--report-steps", type=int, nargs="+", default=list(DEFAULT_REPORT_STEPS))
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--sampling-radius-min", type=float, default=DEFAULT_SAMPLING_RADIUS_MIN)
    parser.add_argument("--sampling-radius-max", type=float, default=DEFAULT_SAMPLING_RADIUS_MAX)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--save-datasets", action="store_true")
    parser.add_argument(
        "--skip-memory-test", action="store_true",
        help="Skip the default maximum full-batch CUDA memory test.",
    )
    parser.add_argument(
        "--memory-test-only", action="store_true",
        help="Run the maximum CUDA memory test, save JSON, and exit.",
    )
    parser.add_argument(
        "--continue-after-memory-test-oom", action="store_true",
        help="Continue despite a failed maximum-memory test. No micro-batching is enabled.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    sample_counts = tuple(sorted(set(int(v) for v in args.ablation_sample_counts)))
    online_counts = tuple(sorted(set(int(v) for v in args.online_sample_counts)))
    positive_ints = {
        "total_time_steps": args.total_time_steps,
        "eval_points_per_problem": args.eval_points_per_problem,
        "radius_eval_points_per_problem": args.radius_eval_points_per_problem,
        "epochs": args.epochs,
        "validation_interval": args.validation_interval,
        "diagnostic_interval": args.diagnostic_interval,
        "evaluation_steps": args.evaluation_steps,
        "evaluation_batch_size": args.evaluation_batch_size,
        "initial_k": args.initial_k,
        "k_increase_interval": args.k_increase_interval,
        "k_increase_amount": args.k_increase_amount,
        "max_k": args.max_k,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if not sample_counts or any(v <= 0 for v in sample_counts):
        raise ValueError("ablation_sample_counts must contain positive integers")
    if not online_counts or any(v <= 0 for v in online_counts):
        raise ValueError("online_sample_counts must contain positive integers")
    if int(args.total_time_steps) != 100:
        raise ValueError("The confirmed experiment requires exactly 100 time steps per motion")
    if int(args.initial_k) > int(args.max_k):
        raise ValueError("initial_k cannot exceed max_k")
    if float(args.sampling_radius_min) <= 0 or float(args.sampling_radius_max) < float(args.sampling_radius_min):
        raise ValueError("Invalid sampling-radius clamp")
    if int(args.validation_interval) > int(args.k_increase_interval):
        raise ValueError(
            "validation_interval must not exceed k_increase_interval; "
            "each K stage needs regular checkpoint candidates"
        )
    report_steps = tuple(sorted(set(
        [int(s) for s in args.report_steps if 0 < int(s) <= int(args.evaluation_steps)]
        + [int(args.evaluation_steps)]
    )))
    return RuntimeConfig(
        total_time_steps=int(args.total_time_steps),
        ablation_sample_counts=sample_counts,
        online_sample_counts=online_counts,
        eval_points_per_problem=int(args.eval_points_per_problem),
        radius_eval_points_per_problem=int(args.radius_eval_points_per_problem),
        epochs=int(args.epochs),
        validation_interval=int(args.validation_interval),
        diagnostic_interval=int(args.diagnostic_interval),
        evaluation_steps=int(args.evaluation_steps),
        evaluation_batch_size=int(args.evaluation_batch_size),
        initial_k=int(args.initial_k),
        k_increase_interval=int(args.k_increase_interval),
        k_increase_amount=int(args.k_increase_amount),
        max_k=int(args.max_k),
        report_steps=report_steps,
        residual_length_scale=float(args.residual_length_scale),
        gradient_clip_norm=float(args.gradient_clip_norm),
        sampling_radius_min=float(args.sampling_radius_min),
        sampling_radius_max=float(args.sampling_radius_max),
        device=str(args.device),
        skip_plots=bool(args.skip_plots),
        save_datasets=bool(args.save_datasets),
        run_memory_test=not bool(args.skip_memory_test),
        memory_test_only=bool(args.memory_test_only),
        continue_after_memory_test_oom=bool(args.continue_after_memory_test_oom),
    )


def _build_standard_evaluation_datasets(
    *,
    lookup: dict[tuple[int, int], TimeStepProblem],
    motion_split: MotionSplit,
    config: RuntimeConfig,
    physical: PhysicalConfig,
) -> tuple[DatasetBundle, dict[str, DatasetBundle]]:
    validation = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.validation_motion_indices,
        time_indices=VALIDATION_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=VALIDATION_SOBOL_SEED,
        role="unseen_motion_validation",
        physical=physical,
        sampling_mode="cube",
    )
    seen_interp = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.train_motion_indices,
        time_indices=SEEN_INTERPOLATION_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=SEEN_INTERPOLATION_TEST_SOBOL_SEED,
        role="seen_motion_temporal_interpolation",
        physical=physical,
        sampling_mode="cube",
    )
    seen_extrap = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.train_motion_indices,
        time_indices=SEEN_EXTRAPOLATION_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=SEEN_EXTRAPOLATION_TEST_SOBOL_SEED,
        role="seen_motion_temporal_extrapolation",
        physical=physical,
        sampling_mode="cube",
    )
    unseen_id = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.id_test_motion_indices,
        time_indices=UNSEEN_TEST_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=UNSEEN_ID_TEST_SOBOL_SEED,
        role="unseen_id_test",
        physical=physical,
        sampling_mode="cube",
    )
    ood_test = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.ood_test_motion_indices,
        time_indices=UNSEEN_TEST_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=OOD_TEST_SOBOL_SEED,
        role="ood_test",
        physical=physical,
        sampling_mode="cube",
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
    return validation, {
        "seen_motion_temporal_interpolation": seen_interp,
        "seen_motion_temporal_extrapolation": seen_extrap,
        "unseen_id_test": unseen_id,
        "ood_test": ood_test,
        "current_state_seen_motion": current_seen,
        "current_state_unseen_id": current_id,
        "current_state_ood": current_ood,
    }


def _build_radius_evaluation_datasets(
    *,
    lookup: dict[tuple[int, int], TimeStepProblem],
    motion_split: MotionSplit,
    config: RuntimeConfig,
    physical: PhysicalConfig,
) -> dict[str, DatasetBundle]:
    result: dict[str, DatasetBundle] = {}
    split_specs = [
        ("id", motion_split.id_test_motion_indices, UNSEEN_ID_TEST_SOBOL_SEED + 2_000_000),
        ("ood", motion_split.ood_test_motion_indices, OOD_TEST_SOBOL_SEED + 2_000_000),
    ]
    for split_name, motion_indices, base_seed in split_specs:
        for bin_offset, (label, low, high) in enumerate(RADIUS_EVALUATION_BINS):
            role = f"initial_radius_{split_name}_{label}"
            result[role] = build_dataset_for_motion_times(
                lookup=lookup,
                motion_indices=motion_indices,
                time_indices=UNSEEN_TEST_TIME_INDICES,
                points_per_problem=config.radius_eval_points_per_problem,
                base_seed=base_seed + 100_000 * bin_offset,
                role=role,
                physical=physical,
                sampling_mode="radial_range",
                normalized_radius_range=(low, high),
            )
    return result


def main() -> None:
    config = validate_args(parse_args())
    physical = default_physical_config()
    motions, motion_split = build_motion_catalogue(physical)
    output_dir = create_output_directory()
    device = torch.device(config.device)
    validate_device(device)

    print(f"Output directory: {output_dir}")
    memory_test: dict[str, Any] | None = None
    if config.run_memory_test:
        print("Running maximum full-batch CUDA memory test before reference generation ...")
        memory_test = run_maximum_memory_test(
            config=config, physical=physical, output_dir=output_dir
        )
        print(f"Maximum-memory test status: {memory_test['status']}")
        if config.memory_test_only:
            print(f"Memory-test report: {output_dir / 'maximum_memory_test.json'}")
            return
        if not memory_test.get("success", False) and device.type == "cuda" and not config.continue_after_memory_test_oom:
            raise RuntimeError(
                "The maximum requested full-batch configuration did not fit in CUDA memory. "
                "The program stopped before expensive reference generation. Inspect "
                f"{output_dir / 'maximum_memory_test.json'}. Reduce --ablation-sample-counts, "
                "or explicitly pass --continue-after-memory-test-oom to test smaller runs. "
                "No micro-batching was enabled."
            )

    physics_checks = run_physics_checks(physical, motions[0])
    print(f"Physics checks: {physics_checks}")
    if not config.skip_plots:
        plot_reference_motion_overview(motions, output_dir / "motion_catalogue_overview.png")

    problems = generate_all_reference_sequences(physical, motions, config)
    lookup = problem_lookup(problems)
    problems_by_index = {p.index: p for p in problems}

    save_json({
        "runtime_config": asdict(config),
        "physical_config": asdict(physical),
        "motion_split": asdict(motion_split),
        "motions": [asdict(m) for m in motions],
        "fixed_vertex_indices": list(FIXED_VERTEX_INDICES),
        "fixed_positions": [list(p) for p in physical.fixed_positions],
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "spring_edges": [list(e) for e in SPRING_EDGES],
        "triangle_faces": [list(f) for f in TRIANGLE_FACES],
        "physics_checks": physics_checks,
        "maximum_memory_test": memory_test,
    }, output_dir / "runtime_config.json")
    save_json({"problems": [problem_to_record(p) for p in problems]}, output_dir / "reference_time_step_problems.json")
    save_json({"motions": [asdict(m) for m in motions], "motion_split": asdict(motion_split)}, output_dir / "motion_catalogue.json")

    validation, evaluation_datasets = _build_standard_evaluation_datasets(
        lookup=lookup, motion_split=motion_split, config=config, physical=physical
    )
    perturbation_evaluation_datasets = _build_radius_evaluation_datasets(
        lookup=lookup, motion_split=motion_split, config=config, physical=physical
    )
    if not config.skip_plots:
        plot_initial_radius_distribution(
            lookup=lookup,
            physical=physical,
            max_points=max(config.ablation_sample_counts),
            save_path=output_dir / "initial_radius_distribution.png",
        )

    hard_case = select_hard_ood_case(
        evaluation_datasets["ood_test"], problems_by_index, physical
    )
    save_json(hard_case, output_dir / "hard_case_selection.json")

    gd_step_size, gd_selection = select_gradient_descent_step_size(
        validation=validation, physical=physical, config=config, device=device
    )
    save_json(gd_selection, output_dir / "gradient_descent_step_selection.json")
    if not config.skip_plots:
        plot_gradient_descent_step_size_selection(
            gd_selection, output_dir / "gradient_descent_step_size_selection.png"
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
    save_json(shared_baselines, output_dir / "shared_gd_newton_baselines.json")

    experiment_specs: list[TrainingExperimentSpec] = []
    for count in config.ablation_sample_counts:
        experiment_specs.append(TrainingExperimentSpec(
            name=f"static_cube_n{count}",
            sampling_mode="cube",
            points_per_problem=count,
            online=False,
        ))
        experiment_specs.append(TrainingExperimentSpec(
            name=f"static_stratified_n{count}",
            sampling_mode="stratified",
            points_per_problem=count,
            online=False,
        ))
    for count in config.online_sample_counts:
        experiment_specs.append(TrainingExperimentSpec(
            name=f"online_stratified_n{count}",
            sampling_mode="online_stratified",
            points_per_problem=count,
            online=True,
        ))

    compact_records: list[dict[str, Any]] = []
    report_paths: list[str] = []
    for spec in experiment_specs:
        print("\n" + "#" * 100)
        print(f"Preparing experiment {spec.name}")
        print("#" * 100)
        online_sampler: OnlineTrainingSampler | None = None
        if spec.online:
            online_sampler = OnlineTrainingSampler(
                lookup=lookup,
                motion_indices=motion_split.train_motion_indices,
                time_indices=TRAIN_TIME_INDICES,
                points_per_problem=spec.points_per_problem,
                base_seed=TRAIN_SOBOL_SEED + 5_000_000,
                role=spec.name,
                physical=physical,
            )
            training = online_sampler.template
        else:
            training = build_dataset_for_motion_times(
                lookup=lookup,
                motion_indices=motion_split.train_motion_indices,
                time_indices=TRAIN_TIME_INDICES,
                points_per_problem=spec.points_per_problem,
                base_seed=TRAIN_SOBOL_SEED,
                role=spec.name,
                physical=physical,
                sampling_mode=spec.sampling_mode,
            )
        if config.save_datasets:
            experiment_dir = output_dir / spec.name
            experiment_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                dataset_to_serializable_dict(training),
                experiment_dir / "training_dataset_or_online_epoch0.pt",
            )

        report = run_experiment(
            experiment_spec=spec,
            training_cpu=training,
            validation_cpu=validation,
            evaluation_datasets=evaluation_datasets,
            perturbation_evaluation_datasets=perturbation_evaluation_datasets,
            output_dir=output_dir,
            config=config,
            physical=physical,
            gd_step_size=gd_step_size,
            shared_baselines=shared_baselines,
            online_sampler=online_sampler,
        )
        compact_records.append(compact_ablation_record(report))
        report_paths.append(str(output_dir / spec.name / "experiment_report.json"))
        save_json({"experiments": compact_records}, output_dir / "ablation_compact_results.json")
        if not config.skip_plots:
            plot_ablation_summary(compact_records, output_dir / "initial_perturbation_ablation_plots")

        del report, training, online_sampler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "experiment_type": "fixed_left_edge_15x15_cloth_initial_perturbation_ablation",
        "runtime_config": asdict(config),
        "physical_config": asdict(physical),
        "motion_split": asdict(motion_split),
        "motions": [asdict(m) for m in motions],
        "physics_checks": physics_checks,
        "maximum_memory_test": memory_test,
        "gradient_descent_selection": gd_selection,
        "hard_case_selection": hard_case,
        "shared_baselines_path": str(output_dir / "shared_gd_newton_baselines.json"),
        "experiment_report_paths": report_paths,
        "compact_experiments": compact_records,
        "metric_policy": {
            "pooled_statistics": ["mean", "median", "p95", "max"],
            "motion_boundary_statistics": ["worst-motion p95", "worst-motion max"],
            "success_thresholds": list(SUCCESS_RESIDUAL_THRESHOLDS),
        },
        "training_policy": {
            "explicit_current_state_starts": 0,
            "explicit_exact_solution_starts": 0,
            "micro_batching": False,
            "online_sampling_regenerates_each_epoch": True,
        },
    }
    save_json(summary, output_dir / "all_experiments_summary.json")
    if not config.skip_plots:
        plot_ablation_summary(compact_records, output_dir / "initial_perturbation_ablation_plots")

    print("\nCompleted all initial-perturbation experiments.")
    print(f"Summary: {output_dir / 'all_experiments_summary.json'}")
    print(f"Compact results: {output_dir / 'ablation_compact_results.json'}")


if __name__ == "__main__":
    main()
