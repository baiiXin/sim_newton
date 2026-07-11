"""Build compact train/evaluation manifests for the 15x15 pipeline.

Validation/test semantics are deliberately different from the old 5x5 pipeline:
- use every physical time step of every original split motion;
- use exactly one initial state per problem, y^(0)=x_n;
- learned evaluation performs exactly one update (handled by cloth05).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import torch

from cloth_common import load_json, resolve_exclusions, save_json

TRAIN = tuple(range(0, 16))
VALIDATION = tuple(range(16, 20))
TEST_ID = tuple(range(20, 24))
TEST_OOD = tuple(range(24, 32))


def select_reference_dataset(
    reference: dict[str, Any],
    motion_indices: Sequence[int],
    total_steps: int,
    name: str,
) -> dict[str, Any]:
    motion_set = torch.tensor(list(motion_indices), dtype=torch.long)
    time_set = torch.arange(total_steps, dtype=torch.long)
    mask = torch.isin(reference["motion_index"], motion_set) & torch.isin(reference["time_index"], time_set)
    return {
        "initial_y": reference["p_n_full"][mask].contiguous(),
        "q": reference["q"][mask].contiguous(),
        "masses": reference["masses"][mask].contiguous(),
        "exact_y": reference["exact_y"][mask].contiguous(),
        "problem_index": reference["problem_index"][mask].contiguous(),
        "motion_index": reference["motion_index"][mask].contiguous(),
        "time_index": reference["time_index"][mask].contiguous(),
        "metadata": {
            "name": name,
            "motion_indices": list(motion_indices),
            "time_range": [0, total_steps - 1],
            "num_points": int(mask.sum().item()),
            "points_per_problem": 1,
            "initial_state": "x_n",
            "evaluation_updates": 1,
        },
    }


def build_windows(train_time_stop: int, width: int) -> list[dict[str, int]]:
    return [
        {"time_start": start, "time_stop": min(start + width, train_time_stop)}
        for start in range(0, train_time_stop, width)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact 15x15 train/validation/test catalogues.")
    parser.add_argument("--root", type=Path, default=Path("cloth_15x15_500step_pipeline"))
    parser.add_argument("--train-time-stop", type=int, default=400)
    parser.add_argument("--time-steps-per-motion-batch", type=int, default=32)
    parser.add_argument("--exclude-motion-indices", type=int, nargs="*", default=[])
    parser.add_argument("--exclusion-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_json(args.root / "data" / "reference" / "runtime_config.json")
    total_steps = int(runtime["total_time_steps"])
    reference = torch.load(args.root / "data" / "reference" / "reference_problems.pt", map_location="cpu")
    exclusion_file = args.exclusion_file or (args.root / "data" / "motion_exclusions.json")
    exclusions = resolve_exclusions(args.exclude_motion_indices, exclusion_file)

    train = tuple(i for i in TRAIN if i not in exclusions)
    validation = tuple(i for i in VALIDATION if i not in exclusions)
    test_id = tuple(i for i in TEST_ID if i not in exclusions)
    test_ood = tuple(i for i in TEST_OOD if i not in exclusions)
    if not train or not validation or not (test_id or test_ood):
        raise ValueError("exclusions emptied train, validation, or test split")

    dataset_dir = args.root / "data" / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    named = {
        "validation_xn": select_reference_dataset(reference, validation, total_steps, "validation_xn"),
        "test_id_xn": select_reference_dataset(reference, test_id, total_steps, "test_id_xn"),
        "test_ood_xn": select_reference_dataset(reference, test_ood, total_steps, "test_ood_xn"),
        "test_all_xn": select_reference_dataset(reference, test_id + test_ood, total_steps, "test_all_xn"),
    }
    for name, data in named.items():
        torch.save(data, dataset_dir / f"{name}.pt")
        print(f"saved {name}: {data['metadata']['num_points']} one-step problems")

    sample_manifest = load_json(args.root / "data" / "samples" / "manifest.json")
    missing = [i for i in train if i not in set(sample_manifest["motion_indices"])]
    if missing:
        raise RuntimeError(f"missing sample shards for included train motions: {missing}")
    windows = build_windows(args.train_time_stop, args.time_steps_per_motion_batch)
    train_manifest = {
        "format": "motion_sharded_training_v1",
        "train_motion_indices": list(train),
        "train_time_range": [0, args.train_time_stop - 1],
        "time_steps_per_motion_batch": args.time_steps_per_motion_batch,
        "num_optimizer_updates_per_epoch": len(windows),
        "windows": windows,
        "sample_root": str((args.root / "data" / "samples").resolve()),
        "points_per_problem_available": int(sample_manifest["points_per_problem"]),
        "sample_prefix_semantics": "first N Sobol samples; all N sets are nested",
        "excluded_motion_indices": list(exclusions),
    }
    save_json(train_manifest, dataset_dir / "train_manifest.json")
    save_json({
        "total_time_steps": total_steps,
        "excluded_motion_indices": list(exclusions),
        "splits": {
            "train": list(train),
            "validation": list(validation),
            "test_id": list(test_id),
            "test_ood": list(test_ood),
        },
        "validation_semantics": "all time steps, y0=x_n, exactly one learned update",
        "test_semantics": "all time steps, y0=x_n, exactly one learned update",
        "rollout_semantics": "one hardest converged test motion, 500 physical frames, separate script",
        "train_manifest": train_manifest,
        "datasets": {k: v["metadata"] for k, v in named.items()},
    }, dataset_dir / "dataset_manifest.json")


def load_dataset(name: str, root: str | Path = "cloth_15x15_500step_pipeline") -> dict[str, Any]:
    return torch.load(Path(root) / "data" / "datasets" / f"{name}.pt", map_location="cpu")


if __name__ == "__main__":
    main()
