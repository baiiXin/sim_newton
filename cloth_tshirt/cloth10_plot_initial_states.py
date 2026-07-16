"""Redraw frozen validation, test, or typical initial states without resampling them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cloth01_build_fixed_model_and_datasets import render_states
from tshirt_config import DEFAULT_FIXED_DATA_DIR, DEFAULT_OBJ_PATH, load_model_spec
from tshirt_mesh import load_tshirt_mesh
from tshirt_sampling import MotionState


FILES = {
    "validation": "validation_32.npz",
    "test": "test_64.npz",
    "typical": "typical_single_motions_4.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--obj", type=Path, default=DEFAULT_OBJ_PATH)
    parser.add_argument("--splits", choices=tuple(FILES), nargs="+", default=tuple(FILES))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-count", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=130)
    return parser.parse_args()


def load_states(path: Path) -> list[MotionState]:
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["motions"]
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


def main() -> None:
    args = parse_args()
    fixed = args.fixed_data_dir.resolve()
    output = fixed / "figures" if args.output_dir is None else args.output_dir.resolve()
    mesh = load_tshirt_mesh(args.obj)
    model = load_model_spec(fixed / "model_spec.json")
    if mesh.sha256 != model.mesh_sha256:
        raise ValueError("OBJ hash differs from the fixed model")
    for split in args.splits:
        states = load_states(fixed / FILES[split])
        render_states(
            states,
            mesh,
            model,
            output / f"{split}_{len(states)}",
            dpi=args.dpi,
            max_count=args.max_count,
        )
        print(f"rendered {split}: {len(states) if args.max_count <= 0 else min(len(states), args.max_count)}")


if __name__ == "__main__":
    main()

