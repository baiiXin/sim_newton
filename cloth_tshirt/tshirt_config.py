"""Shared configuration for the fixed T-shirt model and online dynamics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OBJ_PATH = PROJECT_DIR / "t-shirt" / "tshirt_from_garment_meshes.obj"
DEFAULT_FIXED_DATA_DIR = PROJECT_DIR / "fixed_data"


@dataclass(frozen=True)
class MaterialSpec:
    """One material draw, fixed for every train/validation/test motion."""

    areal_density: float
    lame_mu: float
    lame_lambda: float
    bending_stiffness: float
    thickness: float = 4.7e-4


@dataclass(frozen=True)
class FixedModelSpec:
    version: int
    model_seed: int
    mesh_path: str
    mesh_sha256: str
    num_vertices: int
    num_faces: int
    num_edges: int
    num_hinges: int
    fixed_indices: tuple[int, ...]
    dt: float
    gravity: tuple[float, float, float]
    material: MaterialSpec


@dataclass(frozen=True)
class DynamicsDistribution:
    """Distribution used only when a motion/environment is reset."""

    translation_speed_max: float = 5.0
    angular_speed_max: float = 3.0
    smooth_velocity_rms_max: float = 2.0
    high_frequency_velocity_rms_max: float = 2.5
    velocity_clip: float = 12.0
    position_perturb_rms_edge_fraction_max: float = 0.20
    position_high_frequency_fraction: float = 0.15
    min_area_ratio: float = 0.40
    max_area_ratio: float = 2.50
    min_singular_value: float = 0.45
    max_condition_number: float = 4.0
    max_position_sampling_attempts: int = 64


@dataclass(frozen=True)
class EvaluationSetSpec:
    validation_count: int = 32
    test_count: int = 64
    typical_count: int = 4
    validation_seed: int = 20260721
    test_seed: int = 20260722
    typical_seed: int = 20260723
    quick_inner_steps: int = 15
    full_inner_steps: int = 50
    convergence_residual_ratio: float = 1e-3
    two_order_single_step_ratio: float = 1e-2


DEFAULT_MODEL_SEED = 42
DEFAULT_TRAIN_SEED = 42
DEFAULT_DYNAMICS = DynamicsDistribution()
DEFAULT_EVALUATION = EvaluationSetSpec()


# Exact ranges used by the public HOOD post-CVPR configuration.  HOOD samples
# mu and bending log-uniformly, lambda and density uniformly.
HOOD_MATERIAL_RANGES = {
    "areal_density": (4.34e-2, 7.0e-1, "uniform"),
    "lame_mu": (15909.0, 63636.0, "log_uniform"),
    "lame_lambda": (3535.414406069427, 93333.73508005822, "uniform"),
    "bending_stiffness": (
        6.370782056371576e-8,
        1.3139737991266374e-3,
        "log_uniform",
    ),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def save_model_spec(path: Path, spec: FixedModelSpec) -> None:
    write_json(path, asdict(spec))


def load_model_spec(path: Path) -> FixedModelSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["fixed_indices"] = tuple(int(i) for i in raw["fixed_indices"])
    raw["gravity"] = tuple(float(x) for x in raw["gravity"])
    raw["material"] = MaterialSpec(**raw["material"])
    return FixedModelSpec(**raw)
