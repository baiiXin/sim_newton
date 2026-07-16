"""Create the immutable T-shirt model and reproducible validation/test motions."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np

from tshirt_config import (
    DEFAULT_DYNAMICS,
    DEFAULT_EVALUATION,
    DEFAULT_FIXED_DATA_DIR,
    DEFAULT_MODEL_SEED,
    DEFAULT_OBJ_PATH,
    load_model_spec,
    save_model_spec,
    write_json,
)
from tshirt_mesh import connected_component_sizes, load_tshirt_mesh
from tshirt_sampling import (
    MotionState,
    build_fixed_model_spec,
    build_typical_motions,
    sample_frozen_motions,
    stack_motion_states,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=DEFAULT_OBJ_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plot-max-count", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_states(path: Path, states: Sequence[MotionState]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **stack_motion_states(states))
    os.replace(temporary, path)
    with np.load(path) as archive:
        for key in ("motion_ids", "positions", "velocities", "seeds"):
            if key not in archive:
                raise RuntimeError(f"incomplete motion archive {path}: missing {key}")
    write_json(
        path.with_suffix(".json"),
        {
            "count": len(states),
            "motions": [state.metadata for state in states],
        },
    )


def save_topology(path: Path, mesh, model) -> None:
    masses = mesh.vertex_areas * model.material.areal_density
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            rest_positions=mesh.vertices,
            faces=mesh.faces,
            edges=mesh.edges,
            face_areas=mesh.face_areas,
            inv_dm=mesh.inv_dm,
            vertex_areas=mesh.vertex_areas,
            vertex_masses=masses,
            vertex_normals=mesh.vertex_normals,
            hinge_indices=mesh.hinge_indices,
            hinge_rest_angles=mesh.hinge_rest_angles,
            hinge_rest_lengths=mesh.hinge_rest_lengths,
            boundary_edges=mesh.boundary_edges,
            fixed_indices=np.asarray(model.fixed_indices, dtype=np.int64),
        )
    os.replace(temporary, path)
    with np.load(path) as archive:
        for key in ("rest_positions", "faces", "inv_dm", "vertex_masses", "hinge_indices"):
            if key not in archive:
                raise RuntimeError(f"incomplete topology archive {path}: missing {key}")


def fixed_output_is_valid(output: Path) -> bool:
    checks = {
        "topology_cache.npz": ("rest_positions", "faces", "inv_dm", "hinge_indices"),
        "validation_32.npz": ("motion_ids", "positions", "velocities", "seeds"),
        "test_64.npz": ("motion_ids", "positions", "velocities", "seeds"),
        "typical_single_motions_4.npz": ("motion_ids", "positions", "velocities", "seeds"),
    }
    if not (output / "model_spec.json").exists():
        return False
    try:
        for filename, required in checks.items():
            with np.load(output / filename) as archive:
                if not all(key in archive for key in required):
                    return False
                # Force decompression now so a truncated zip cannot pass the check.
                for key in required:
                    _ = archive[key].shape
    except (OSError, ValueError, KeyError):
        return False
    return True


def load_saved_states(path: Path) -> list[MotionState]:
    import json

    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["motions"]
    with np.load(path) as archive:
        ids = tuple(str(value) for value in archive["motion_ids"].tolist())
        positions = np.asarray(archive["positions"])
        velocities = np.asarray(archive["velocities"])
    if not (len(ids) == positions.shape[0] == velocities.shape[0] == len(metadata)):
        raise ValueError(f"metadata and array count mismatch in {path}")
    return [
        MotionState(ids[index], positions[index], velocities[index], metadata[index])
        for index in range(len(ids))
    ]


def plot_state(state: MotionState, mesh, model, output: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    positions = state.positions
    fixed = np.asarray(model.fixed_indices, dtype=np.int64)
    segments = positions[mesh.edges]
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.add_collection3d(Line3DCollection(segments, colors="#4c78a8", linewidths=0.22, alpha=0.22))
    axis.scatter(
        positions[fixed, 0],
        positions[fixed, 1],
        positions[fixed, 2],
        c="#d62728",
        marker="s",
        s=34,
        depthshade=False,
        label="fixed",
    )
    free = np.setdiff1d(np.arange(mesh.num_vertices), fixed)
    if free.size:
        stride = max(1, free.size // 72)
        sample = free[::stride][:72]
        velocity = state.velocities[sample]
        speed = np.linalg.norm(velocity, axis=-1)
        active = speed > 1e-10
        if np.any(active):
            sample = sample[active]
            velocity = velocity[active]
            scale = max(float(np.quantile(np.linalg.norm(velocity, axis=-1), 0.9)), 1e-12)
            velocity = velocity / scale * (0.12 * float(np.ptp(positions, axis=0).max()))
            axis.quiver(
                positions[sample, 0], positions[sample, 1], positions[sample, 2],
                velocity[:, 0], velocity[:, 1], velocity[:, 2],
                color="#2ca02c", linewidth=0.55, arrow_length_ratio=0.22,
            )
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.55 * max(float((maximum - minimum).max()), 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(
        f"{state.motion_id}\nvelocity RMS={state.metadata['velocity_rms']:.3g} m/s, "
        f"min area ratio={state.metadata['min_area_ratio']:.3g}"
    )
    axis.legend(loc="upper right")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def render_states(
    states: Sequence[MotionState],
    mesh,
    model,
    output_dir: Path,
    *,
    dpi: int,
    max_count: int,
) -> None:
    selected = states if max_count <= 0 else states[:max_count]
    rows: list[dict[str, Any]] = []
    for index, state in enumerate(selected):
        filename = f"motion_{index:04d}_{state.motion_id}.png"
        plot_state(state, mesh, model, output_dir / filename, dpi)
        rows.append({"index": index, "file": filename, **state.metadata})
    write_json(output_dir / "index.json", rows)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if not args.overwrite and fixed_output_is_valid(output):
        rendered = False
        if not args.skip_plots:
            figure_sets = (
                ("validation_32", "validation_32.npz"),
                ("test_64", "test_64.npz"),
                ("typical_4", "typical_single_motions_4.npz"),
            )
            missing = [
                (name, filename)
                for name, filename in figure_sets
                if not (output / "figures" / name / "index.json").exists()
            ]
            if missing:
                mesh = load_tshirt_mesh(args.obj)
                model = load_model_spec(output / "model_spec.json")
                if mesh.sha256 != model.mesh_sha256:
                    raise ValueError("OBJ hash differs from the existing fixed model")
                for name, filename in missing:
                    render_states(
                        load_saved_states(output / filename),
                        mesh,
                        model,
                        output / "figures" / name,
                        dpi=args.dpi,
                        max_count=args.plot_max_count,
                    )
                rendered = True
        print(
            f"Fixed model and datasets already exist and passed archive checks at {output}; "
            + ("missing figures were rendered." if rendered else
               "use --overwrite to rebuild or cloth10_plot_initial_states.py to redraw figures.")
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    mesh = load_tshirt_mesh(args.obj)
    model = build_fixed_model_spec(mesh, model_seed=args.model_seed)
    validation = sample_frozen_motions(
        mesh,
        model,
        DEFAULT_DYNAMICS,
        count=DEFAULT_EVALUATION.validation_count,
        base_seed=DEFAULT_EVALUATION.validation_seed,
        split="validation",
    )
    test = sample_frozen_motions(
        mesh,
        model,
        DEFAULT_DYNAMICS,
        count=DEFAULT_EVALUATION.test_count,
        base_seed=DEFAULT_EVALUATION.test_seed,
        split="test",
    )
    typical = build_typical_motions(
        mesh,
        model,
        DEFAULT_DYNAMICS,
        seed=DEFAULT_EVALUATION.typical_seed,
    )
    save_model_spec(output / "model_spec.json", model)
    save_topology(output / "topology_cache.npz", mesh, model)
    save_states(output / "validation_32.npz", validation)
    save_states(output / "test_64.npz", test)
    save_states(output / "typical_single_motions_4.npz", typical)
    write_json(
        output / "dataset_manifest.json",
        {
            "model": asdict(model),
            "dynamics_distribution": asdict(DEFAULT_DYNAMICS),
            "evaluation_protocol": asdict(DEFAULT_EVALUATION),
            "mesh": {
                "connected_component_sizes": connected_component_sizes(mesh.num_vertices, mesh.edges),
                "boundary_edges": int(mesh.boundary_edges.shape[0]),
                "median_edge_length": mesh.median_edge_length,
                "rest_area": float(mesh.face_areas.sum()),
            },
            "training_policy": {
                "persist_individual_training_motions": False,
                "sample_when_environment_resets": True,
                "checkpoint_rng_state": True,
            },
        },
    )
    if not args.skip_plots:
        for name, states in (("validation_32", validation), ("test_64", test), ("typical_4", typical)):
            render_states(
                states,
                mesh,
                model,
                output / "figures" / name,
                dpi=args.dpi,
                max_count=args.plot_max_count,
            )
    print(f"Wrote fixed model and datasets to {output}")
    print(f"fixed_indices={model.fixed_indices}")
    print(f"material={asdict(model.material)}")


if __name__ == "__main__":
    main()
