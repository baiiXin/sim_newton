"""Fixed-model creation and online/frozen motion sampling."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from tshirt_config import (
    DynamicsDistribution,
    FixedModelSpec,
    HOOD_MATERIAL_RANGES,
    MaterialSpec,
    PROJECT_DIR,
)
from tshirt_mesh import TShirtMesh, select_four_shoulder_vertices


@dataclass(frozen=True)
class MotionState:
    motion_id: str
    positions: np.ndarray
    velocities: np.ndarray
    metadata: dict[str, Any]


def _sample_range(
    rng: np.random.Generator,
    lower: float,
    upper: float,
    distribution: str,
) -> float:
    if distribution == "uniform":
        return float(rng.uniform(lower, upper))
    if distribution == "log_uniform":
        return float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
    raise ValueError(f"Unknown distribution: {distribution}")


def sample_fixed_material(seed: int) -> MaterialSpec:
    rng = np.random.default_rng(seed)
    values = {
        name: _sample_range(rng, lower, upper, distribution)
        for name, (lower, upper, distribution) in HOOD_MATERIAL_RANGES.items()
    }
    return MaterialSpec(**values, thickness=4.7e-4)


def build_fixed_model_spec(
    mesh: TShirtMesh,
    *,
    model_seed: int,
    dt: float = 0.01,
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0),
) -> FixedModelSpec:
    try:
        mesh_path = str(mesh.path.relative_to(PROJECT_DIR))
    except ValueError:
        mesh_path = str(mesh.path)
    return FixedModelSpec(
        version=1,
        model_seed=int(model_seed),
        mesh_path=mesh_path,
        mesh_sha256=mesh.sha256,
        num_vertices=mesh.num_vertices,
        num_faces=mesh.num_faces,
        num_edges=int(mesh.edges.shape[0]),
        num_hinges=int(mesh.hinge_indices.shape[0]),
        fixed_indices=select_four_shoulder_vertices(mesh),
        dt=float(dt),
        gravity=tuple(float(x) for x in gravity),
        material=sample_fixed_material(model_seed),
    )


def uniform_rotation(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform SO(3) rotation via a unit quaternion."""

    u1, u2, u3 = rng.random(3)
    qx = math.sqrt(1.0 - u1) * math.sin(2.0 * math.pi * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(2.0 * math.pi * u2)
    qz = math.sqrt(u1) * math.sin(2.0 * math.pi * u3)
    qw = math.sqrt(u1) * math.cos(2.0 * math.pi * u3)
    return np.asarray(
        (
            (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
            (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
            (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
        ),
        dtype=np.float64,
    )


def align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(source, target))
    if sine < 1e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.asarray((1.0, 0.0, 0.0))
        if abs(float(source[0])) > 0.9:
            axis = np.asarray((0.0, 1.0, 0.0))
        axis -= source * float(np.dot(axis, source))
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.asarray(
        ((0.0, -cross[2], cross[1]), (cross[2], 0.0, -cross[0]), (-cross[1], cross[0], 0.0))
    )
    return np.eye(3) + skew + (skew @ skew) * ((1.0 - cosine) / (sine * sine))


def graph_smooth(field: np.ndarray, mesh: TShirtMesh, steps: int = 1) -> np.ndarray:
    result = np.asarray(field, dtype=np.float64).copy()
    degree = np.zeros(mesh.num_vertices, dtype=np.float64)
    np.add.at(degree, mesh.edges[:, 0], 1.0)
    np.add.at(degree, mesh.edges[:, 1], 1.0)
    for _ in range(steps):
        neighbor_sum = np.zeros_like(result)
        np.add.at(neighbor_sum, mesh.edges[:, 0], result[mesh.edges[:, 1]])
        np.add.at(neighbor_sum, mesh.edges[:, 1], result[mesh.edges[:, 0]])
        neighbor_mean = neighbor_sum / np.maximum(degree[:, None], 1.0)
        result = 0.5 * result + 0.5 * neighbor_mean
    return result


def _zero_mean_unit_rms(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64) - np.mean(field, axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(field * field, axis=-1))))
    if rms <= 1e-15:
        raise ValueError("Random field has zero RMS")
    return field / rms


def deformation_quality(mesh: TShirtMesh, positions: np.ndarray) -> dict[str, float]:
    triangles = positions[mesh.faces]
    ds = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
    deformation = ds @ mesh.inv_dm
    cauchy_green = np.swapaxes(deformation, -1, -2) @ deformation
    eigenvalues = np.linalg.eigvalsh(cauchy_green)
    singular = np.sqrt(np.maximum(eigenvalues, 0.0))
    area_ratio = singular[:, 0] * singular[:, 1]
    condition = singular[:, 1] / np.maximum(singular[:, 0], 1e-15)
    return {
        "min_area_ratio": float(area_ratio.min()),
        "max_area_ratio": float(area_ratio.max()),
        "min_singular_value": float(singular.min()),
        "max_condition_number": float(condition.max()),
    }


def quality_is_accepted(quality: dict[str, float], config: DynamicsDistribution) -> bool:
    return (
        quality["min_area_ratio"] >= config.min_area_ratio
        and quality["max_area_ratio"] <= config.max_area_ratio
        and quality["min_singular_value"] >= config.min_singular_value
        and quality["max_condition_number"] <= config.max_condition_number
    )


def _sample_position_perturbation(
    mesh: TShirtMesh,
    rng: np.random.Generator,
    config: DynamicsDistribution,
    *,
    requested_edge_fraction: float | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    max_rms = config.position_perturb_rms_edge_fraction_max * mesh.median_edge_length
    if requested_edge_fraction is not None and requested_edge_fraction < 0.0:
        raise ValueError("requested position perturbation must be nonnegative")
    for attempt in range(1, config.max_position_sampling_attempts + 1):
        raw = _zero_mean_unit_rms(rng.normal(size=mesh.vertices.shape))
        smooth = _zero_mean_unit_rms(graph_smooth(raw, mesh, steps=2))
        high_fraction = config.position_high_frequency_fraction
        mixed = _zero_mean_unit_rms((1.0 - high_fraction) * smooth + high_fraction * raw)
        requested_rms = (
            float(rng.uniform(0.0, max_rms))
            if requested_edge_fraction is None
            else float(requested_edge_fraction * mesh.median_edge_length)
        )
        perturbation = requested_rms * mixed
        candidate = mesh.vertices + perturbation
        quality = deformation_quality(mesh, candidate)
        if quality_is_accepted(quality, config):
            actual_rms = float(np.sqrt(np.mean(np.sum(perturbation * perturbation, axis=-1))))
            return perturbation, {
                **quality,
                "position_sampling_attempts": attempt,
                "position_perturbation_rms": actual_rms,
            }
    raise RuntimeError(
        "Could not sample a non-compressed position perturbation after "
        f"{config.max_position_sampling_attempts} attempts"
    )


def build_inference_motion(
    mesh: TShirtMesh,
    model: FixedModelSpec,
    config: DynamicsDistribution,
    *,
    seed: int,
    pose: str = "horizontal",
    translation_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    angular_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    smooth_velocity_rms: float = 0.0,
    high_frequency_velocity_rms: float = 0.0,
    position_perturb_rms_edge_fraction: float = 0.0,
    velocity_clip: float = 12.0,
) -> MotionState:
    """Construct one user-controlled inference state.

    Unlike the training sampler, RMS values here are exact requested values,
    not upper bounds of random distributions.
    """

    if pose not in {"horizontal", "vertical", "random"}:
        raise ValueError("pose must be horizontal, vertical, or random")
    if smooth_velocity_rms < 0.0 or high_frequency_velocity_rms < 0.0:
        raise ValueError("velocity RMS values must be nonnegative")
    if velocity_clip <= 0.0:
        raise ValueError("velocity_clip must be positive")
    translation = np.asarray(translation_velocity, dtype=np.float64)
    angular = np.asarray(angular_velocity, dtype=np.float64)
    if translation.shape != (3,) or angular.shape != (3,):
        raise ValueError("translation and angular velocity must be three-vectors")

    rng = np.random.default_rng(seed)
    if position_perturb_rms_edge_fraction > 0.0:
        perturbation, quality_metadata = _sample_position_perturbation(
            mesh,
            rng,
            config,
            requested_edge_fraction=position_perturb_rms_edge_fraction,
        )
    else:
        perturbation = np.zeros_like(mesh.vertices)
        quality_metadata = {
            **deformation_quality(mesh, mesh.vertices),
            "position_sampling_attempts": 0,
            "position_perturbation_rms": 0.0,
        }

    center = np.mean(mesh.vertices, axis=0)
    centered = mesh.vertices - center
    if pose == "horizontal":
        _, eigenvectors = np.linalg.eigh(centered.T @ centered)
        garment_normal = eigenvectors[:, 0]
        rotation = align_vectors(garment_normal, np.asarray(model.gravity, dtype=np.float64))
    elif pose == "vertical":
        rotation = np.eye(3, dtype=np.float64)
    else:
        rotation = uniform_rotation(rng)
    positions = (mesh.vertices + perturbation - center) @ rotation.T + center

    rigid_velocity = translation + np.cross(
        np.broadcast_to(angular, positions.shape), positions - center
    )
    smooth = np.zeros_like(positions)
    high = np.zeros_like(positions)
    if smooth_velocity_rms > 0.0:
        smooth = smooth_velocity_rms * _zero_mean_unit_rms(
            graph_smooth(rng.normal(size=positions.shape), mesh, steps=2)
        )
    if high_frequency_velocity_rms > 0.0:
        high = high_frequency_velocity_rms * _zero_mean_unit_rms(
            rng.normal(size=positions.shape)
        )
    velocities = rigid_velocity + smooth + high
    speed = np.linalg.norm(velocities, axis=-1)
    velocities *= np.minimum(1.0, velocity_clip / np.maximum(speed, 1e-15))[:, None]
    velocities[np.asarray(model.fixed_indices, dtype=np.int64)] = 0.0
    final_speed = np.linalg.norm(velocities, axis=-1)
    metadata: dict[str, Any] = {
        "motion_id": f"inference_{pose}_seed_{seed}",
        "split": "inference",
        "seed": int(seed),
        "pose": pose,
        "rotation_matrix": rotation.tolist(),
        "translation_velocity": translation.tolist(),
        "angular_velocity": angular.tolist(),
        "smooth_velocity_rms_requested": float(smooth_velocity_rms),
        "high_frequency_velocity_rms_requested": float(high_frequency_velocity_rms),
        "position_perturb_rms_edge_fraction_requested": float(
            position_perturb_rms_edge_fraction
        ),
        "velocity_clip": float(velocity_clip),
        "velocity_rms": float(np.sqrt(np.mean(final_speed * final_speed))),
        "velocity_max": float(final_speed.max()),
        **quality_metadata,
    }
    return MotionState(metadata["motion_id"], positions, velocities, metadata)


def sample_random_motion(
    mesh: TShirtMesh,
    model: FixedModelSpec,
    config: DynamicsDistribution,
    *,
    seed: int,
    motion_id: str,
    split: str,
) -> MotionState:
    rng = np.random.default_rng(seed)
    perturbation, quality_metadata = _sample_position_perturbation(mesh, rng, config)
    rotation = uniform_rotation(rng)
    center = np.mean(mesh.vertices, axis=0)
    local = mesh.vertices + perturbation - center
    positions = local @ rotation.T + center

    translation = rng.uniform(-config.translation_speed_max, config.translation_speed_max, size=3)
    angular_velocity = rng.uniform(-config.angular_speed_max, config.angular_speed_max, size=3)
    rigid_velocity = translation + np.cross(
        np.broadcast_to(angular_velocity, positions.shape), positions - center
    )
    smooth_unit = _zero_mean_unit_rms(graph_smooth(rng.normal(size=positions.shape), mesh, steps=2))
    high_unit = _zero_mean_unit_rms(rng.normal(size=positions.shape))
    smooth_rms = float(rng.uniform(0.0, config.smooth_velocity_rms_max))
    high_rms = float(rng.uniform(0.0, config.high_frequency_velocity_rms_max))
    velocities = rigid_velocity + smooth_rms * smooth_unit + high_rms * high_unit
    speed = np.linalg.norm(velocities, axis=-1)
    scale = np.minimum(1.0, config.velocity_clip / np.maximum(speed, 1e-15))
    velocities *= scale[:, None]
    velocities[np.asarray(model.fixed_indices, dtype=np.int64)] = 0.0
    final_speed = np.linalg.norm(velocities, axis=-1)
    metadata: dict[str, Any] = {
        "motion_id": motion_id,
        "split": split,
        "seed": int(seed),
        "rotation_matrix": rotation.tolist(),
        "translation_velocity": translation.tolist(),
        "angular_velocity": angular_velocity.tolist(),
        "smooth_velocity_rms_requested": smooth_rms,
        "high_frequency_velocity_rms_requested": high_rms,
        "velocity_rms": float(np.sqrt(np.mean(final_speed * final_speed))),
        "velocity_max": float(final_speed.max()),
        **quality_metadata,
    }
    return MotionState(motion_id, positions, velocities, metadata)


def sample_frozen_motions(
    mesh: TShirtMesh,
    model: FixedModelSpec,
    config: DynamicsDistribution,
    *,
    count: int,
    base_seed: int,
    split: str,
) -> list[MotionState]:
    if count <= 0:
        raise ValueError("count must be positive")
    parent = np.random.default_rng(base_seed)
    seeds = parent.integers(0, np.iinfo(np.int64).max, size=count, dtype=np.int64)
    return [
        sample_random_motion(
            mesh,
            model,
            config,
            seed=int(seed),
            motion_id=f"{split}_{index:04d}",
            split=split,
        )
        for index, seed in enumerate(seeds)
    ]


def _motion_from_pose(
    mesh: TShirtMesh,
    model: FixedModelSpec,
    *,
    motion_id: str,
    rotation: np.ndarray,
    velocities: np.ndarray | None = None,
    description: str,
) -> MotionState:
    center = np.mean(mesh.vertices, axis=0)
    positions = (mesh.vertices - center) @ rotation.T + center
    if velocities is None:
        velocities = np.zeros_like(positions)
    else:
        velocities = np.asarray(velocities, dtype=np.float64).copy()
    velocities[np.asarray(model.fixed_indices, dtype=np.int64)] = 0.0
    quality = deformation_quality(mesh, positions)
    return MotionState(
        motion_id,
        positions,
        velocities,
        {
            "motion_id": motion_id,
            "split": "typical",
            "description": description,
            "rotation_matrix": rotation.tolist(),
            "velocity_rms": float(np.sqrt(np.mean(np.sum(velocities * velocities, axis=-1)))),
            "velocity_max": float(np.linalg.norm(velocities, axis=-1).max()),
            **quality,
        },
    )


def build_typical_motions(
    mesh: TShirtMesh,
    model: FixedModelSpec,
    config: DynamicsDistribution,
    *,
    seed: int,
) -> list[MotionState]:
    centered = mesh.vertices - np.mean(mesh.vertices, axis=0)
    _, eigenvectors = np.linalg.eigh(centered.T @ centered)
    garment_normal = eigenvectors[:, 0]
    gravity = np.asarray(model.gravity, dtype=np.float64)
    horizontal_rotation = align_vectors(garment_normal, gravity)
    horizontal = _motion_from_pose(
        mesh,
        model,
        motion_id="typical_00_horizontal_gravity_release",
        rotation=horizontal_rotation,
        description="Garment plane horizontal, zero initial velocity, gravity-only swing.",
    )
    vertical = _motion_from_pose(
        mesh,
        model,
        motion_id="typical_01_vertical_rest_release",
        rotation=np.eye(3),
        description="Original HOOD pose, zero initial velocity, gravity-only release.",
    )
    random_states = sample_frozen_motions(
        mesh,
        model,
        config,
        count=2,
        base_seed=seed,
        split="typical_random",
    )
    high = random_states[0]
    high.metadata["description"] = "Mixed rigid, smooth, and per-vertex high-frequency velocity field."
    high.metadata["motion_id"] = "typical_02_high_frequency_velocity"
    high = MotionState(
        "typical_02_high_frequency_velocity", high.positions, high.velocities, high.metadata
    )
    pose = random_states[1]
    pose.metadata["description"] = "Random SO(3) pose with position perturbation and mixed velocity."
    pose.metadata["motion_id"] = "typical_03_random_pose_and_velocity"
    pose = MotionState(
        "typical_03_random_pose_and_velocity", pose.positions, pose.velocities, pose.metadata
    )
    return [horizontal, vertical, high, pose]


def stack_motion_states(states: Sequence[MotionState]) -> dict[str, np.ndarray]:
    return {
        "motion_ids": np.asarray([state.motion_id for state in states]),
        "positions": np.stack([state.positions for state in states], axis=0),
        "velocities": np.stack([state.velocities for state in states], axis=0),
        "seeds": np.asarray([int(state.metadata.get("seed", -1)) for state in states], dtype=np.int64),
    }
