"""Prepare nested initial-state samples for the points-per-problem ablation.

The existing 32-motion/500-step reference data are reused.  For every training
problem (motions 0-15, time steps 0-399), this script creates one physical
initial state followed by scrambled Sobol states around the stored reference
solution.  A single 1024-point sequence is generated per problem, so the
experiments with {1, 8, 32, 64, 128, 1024} points use nested prefixes.

Storage is split by training motion to avoid constructing one very large tensor
in memory.  No reference trajectory is regenerated.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from cloth03_solvers_and_models import (
    FULL_STATE_DIM,
    TORCH_DTYPE,
    TRAIN_SOBOL_SEED,
    free_state_from_full_state,
    full_state_from_free_state,
    generate_sobol_points,
    physical_config_from_dict,
    project_fixed_vertices,
)

TRAIN_MOTIONS = tuple(range(16))
DEFAULT_SAMPLE_COUNTS = (1, 8, 32, 64, 128, 1024)
DEFAULT_TRAIN_TIME_STOP = 400


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def validate_counts(sample_counts: tuple[int, ...], max_points: int) -> None:
    if not sample_counts:
        raise ValueError("sample-counts cannot be empty")
    if any(value <= 0 for value in sample_counts):
        raise ValueError("all sample counts must be positive")
    if tuple(sorted(set(sample_counts))) != sample_counts:
        raise ValueError("sample-counts must be sorted and unique")
    if sample_counts[-1] > max_points:
        raise ValueError("largest sample count exceeds max-points")


def rows_for_motion(reference: dict[str, Any], motion_index: int, train_time_stop: int) -> torch.Tensor:
    mask = (
        (reference["motion_index"] == int(motion_index))
        & (reference["time_index"] >= 0)
        & (reference["time_index"] < int(train_time_stop))
    )
    rows = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if rows.numel() != train_time_stop:
        raise RuntimeError(
            f"motion {motion_index} has {rows.numel()} training problems; expected {train_time_stop}"
        )
    times = reference["time_index"].index_select(0, rows)
    order = torch.argsort(times)
    rows = rows.index_select(0, order)
    expected = torch.arange(train_time_stop, dtype=torch.long)
    actual = reference["time_index"].index_select(0, rows).to(torch.long)
    if not torch.equal(actual, expected):
        raise RuntimeError(f"motion {motion_index} training time indices are not 0..{train_time_stop - 1}")
    return rows


def generate_motion_file(
    *,
    reference: dict[str, Any],
    motion_index: int,
    rows: torch.Tensor,
    output_path: Path,
    max_points: int,
    base_seed: int,
    physical,
) -> dict[str, Any]:
    num_problems = int(rows.numel())
    initial_y = torch.empty(
        (num_problems, max_points, FULL_STATE_DIM),
        dtype=TORCH_DTYPE,
        device="cpu",
    )

    for local_row, source_row in enumerate(rows.tolist()):
        p_n_full = reference["p_n_full"][source_row].to(dtype=TORCH_DTYPE).reshape(1, -1)
        physical_initial_free = free_state_from_full_state(p_n_full).squeeze(0)
        center = reference["exact_y_free"][source_row].to(dtype=TORCH_DTYPE)
        radius = float(reference["sampling_radius"][source_row].item())
        time_index = int(reference["time_index"][source_row].item())
        seed = int(base_seed + 100_003 * motion_index + 1009 * time_index)

        sampled_free, _ = generate_sobol_points(
            count=max_points,
            center=center,
            radius=radius,
            seed=seed,
            physical=physical,
            explicit_points=(physical_initial_free,),
        )
        sampled_full = project_fixed_vertices(
            full_state_from_free_state(sampled_free, physical),
            physical,
        )
        initial_y[local_row].copy_(sampled_full.cpu())

        if local_row == 0 or (local_row + 1) % 50 == 0 or local_row + 1 == num_problems:
            print(
                f"motion {motion_index:02d}: sampled {local_row + 1:03d}/{num_problems}, "
                f"points={max_points}"
            )

    record = {
        "initial_y": initial_y.contiguous(),
        "q": reference["q"].index_select(0, rows).to(dtype=TORCH_DTYPE).contiguous(),
        "masses": reference["masses"].index_select(0, rows).to(dtype=TORCH_DTYPE).contiguous(),
        "exact_y": reference["exact_y"].index_select(0, rows).to(dtype=TORCH_DTYPE).contiguous(),
        "problem_index": reference["problem_index"].index_select(0, rows).to(torch.long).contiguous(),
        "motion_index": reference["motion_index"].index_select(0, rows).to(torch.long).contiguous(),
        "time_index": reference["time_index"].index_select(0, rows).to(torch.long).contiguous(),
        "sampling_radius": reference["sampling_radius"].index_select(0, rows).to(dtype=TORCH_DTYPE).contiguous(),
        "metadata": {
            "motion_index": motion_index,
            "num_problems": num_problems,
            "max_points_per_problem": max_points,
            "sample_axis_semantics": "slot 0 is physical initial state; slots 1.. are scrambled Sobol samples",
            "nested_prefixes": True,
            "physical_initial_slot": 0,
            "sobol_center": "stored exact_y_free",
            "sobol_region": "Linf cube with stored per-problem sampling_radius",
            "base_seed": base_seed,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(record, output_path)
    return record["metadata"]


def create_experiment_manifests(
    *,
    ablation_root: Path,
    source_root: Path,
    shared_sample_dir: Path,
    sample_counts: tuple[int, ...],
    max_points: int,
    train_time_stop: int,
) -> None:
    for count in sample_counts:
        point_root = ablation_root / f"points_{count:04d}"
        (point_root / "models").mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "sample_count": count,
                "sample_prefix": [0, count - 1],
                "physical_initial_included": True,
                "physical_initial_slot": 0,
                "nested_from_max_points": max_points,
                "source_pipeline_root": str(source_root.resolve()),
                "shared_sample_dir": str(shared_sample_dir.resolve()),
                "train_motion_indices": list(TRAIN_MOTIONS),
                "train_time_range": [0, train_time_stop - 1],
                "epoch_semantics": "visit every selected sample of every training problem once",
                "optimizer_update_semantics": (
                    "for each original time-problem minibatch, accumulate mean gradients over sample slots, "
                    "then perform one optimizer update"
                ),
            },
            point_root / "experiment.json",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare nested samples for initial-point-count ablation.")
    parser.add_argument("--source-root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--ablation-root", type=Path, default=Path("cloth_5x5_initial_sample_ablation"))
    parser.add_argument("--sample-counts", type=int, nargs="+", default=list(DEFAULT_SAMPLE_COUNTS))
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--train-time-stop", type=int, default=DEFAULT_TRAIN_TIME_STOP)
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SOBOL_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_counts = tuple(int(value) for value in args.sample_counts)
    validate_counts(sample_counts, int(args.max_points))
    if args.train_time_stop <= 0:
        raise ValueError("train-time-stop must be positive")

    source_root = args.source_root.resolve()
    ablation_root = args.ablation_root.resolve()
    reference_path = source_root / "data" / "reference" / "reference_problems.pt"
    runtime_path = source_root / "data" / "reference" / "runtime_config.json"
    if not reference_path.exists() or not runtime_path.exists():
        raise FileNotFoundError("source reference_problems.pt or runtime_config.json is missing")

    runtime = load_json(runtime_path)
    physical = physical_config_from_dict(runtime["physical_config"])
    reference = torch.load(reference_path, map_location="cpu")
    shared_sample_dir = ablation_root / "shared_samples_1024"
    shared_reference_dir = ablation_root / "shared_reference"
    shared_reference_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "source_pipeline_root": str(source_root),
            "reference_problems": str(reference_path),
            "reference_motion_states": str(source_root / "data" / "reference" / "reference_motion_states.pt"),
            "runtime_config": str(runtime_path),
            "reference_regenerated": False,
        },
        shared_reference_dir / "REFERENCE_SOURCE.json",
    )
    for name in ("runtime_config.json", "motion_catalogue.json"):
        source = source_root / "data" / "reference" / name
        if source.exists():
            shutil.copy2(source, shared_reference_dir / name)

    motion_metadata: dict[str, Any] = {}
    for motion_index in TRAIN_MOTIONS:
        output_path = shared_sample_dir / f"motion_{motion_index:03d}.pt"
        if output_path.exists() and not args.overwrite:
            existing = torch.load(output_path, map_location="cpu")
            metadata = existing.get("metadata", {})
            if int(metadata.get("max_points_per_problem", -1)) != int(args.max_points):
                raise RuntimeError(f"existing {output_path} does not match max-points={args.max_points}")
            print(f"skip existing {output_path}")
            motion_metadata[str(motion_index)] = metadata
            continue
        rows = rows_for_motion(reference, motion_index, int(args.train_time_stop))
        motion_metadata[str(motion_index)] = generate_motion_file(
            reference=reference,
            motion_index=motion_index,
            rows=rows,
            output_path=output_path,
            max_points=int(args.max_points),
            base_seed=int(args.sample_seed),
            physical=physical,
        )

    manifest = {
        "source_pipeline_root": str(source_root),
        "sample_counts": list(sample_counts),
        "max_points": int(args.max_points),
        "train_motion_indices": list(TRAIN_MOTIONS),
        "train_time_range": [0, int(args.train_time_stop) - 1],
        "num_training_problems": len(TRAIN_MOTIONS) * int(args.train_time_stop),
        "physical_initial_included_in_every_prefix": True,
        "nested_prefixes": True,
        "motion_files": motion_metadata,
    }
    save_json(manifest, shared_sample_dir / "manifest.json")
    create_experiment_manifests(
        ablation_root=ablation_root,
        source_root=source_root,
        shared_sample_dir=shared_sample_dir,
        sample_counts=sample_counts,
        max_points=int(args.max_points),
        train_time_stop=int(args.train_time_stop),
    )
    print(f"prepared ablation at {ablation_root}")


if __name__ == "__main__":
    main()
