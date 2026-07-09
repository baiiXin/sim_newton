"""Script 2: build named datasets from Script 1 outputs.

This script does not resample. It only filters and groups all_sampled_problems.pt.

Default split:
    train       : motions 0-15,  time 0-399
    validation  : motions 16-19, time 0-499 sampled uniformly
    seen_extrap : motions 0-15,  time 400-499
    unseen_id   : motions 20-23, time 0-499 sampled uniformly
    ood         : motions 24-31, time 0-499 sampled uniformly

Training mini-batch plan:
    one mini-batch = 16 train motions x 32 time-step problems per motion.
    With 400 train time steps this gives 13 batches per epoch:
    12 full batches of 16x32 problems and 1 final batch of 16x16 problems.

Run:
    python cloth02_dataset_catalog.py --root cloth_5x5_500step_pipeline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cloth03_solvers_and_models import DEFAULT_TOTAL_TIME_STEPS

TRAIN_MOTIONS = tuple(range(0, 16))
VALIDATION_MOTIONS = tuple(range(16, 20))
UNSEEN_ID_MOTIONS = tuple(range(20, 24))
OOD_MOTIONS = tuple(range(24, 32))


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_all_samples(root: Path) -> dict[str, Any]:
    path = root / "data" / "samples" / "all_sampled_problems.pt"
    return torch.load(path, map_location="cpu")


def load_reference_problems(root: Path) -> dict[str, Any]:
    path = root / "data" / "reference" / "reference_problems.pt"
    return torch.load(path, map_location="cpu")


def uniform_time_indices(total_time_steps: int, count: int) -> tuple[int, ...]:
    if count >= total_time_steps:
        return tuple(range(total_time_steps))
    values = np.linspace(0, total_time_steps - 1, count).round().astype(int).tolist()
    return tuple(sorted(set(values)))


def filter_sampled_dataset(
    all_samples: dict[str, Any],
    *,
    name: str,
    motion_indices: tuple[int, ...],
    time_indices: tuple[int, ...],
) -> dict[str, Any]:
    motion = all_samples["motion_index"]
    time = all_samples["time_index"]
    motion_set = torch.tensor(motion_indices, dtype=torch.long)
    time_set = torch.tensor(time_indices, dtype=torch.long)
    mask = torch.isin(motion, motion_set) & torch.isin(time, time_set)

    keys = ["initial_y", "q", "masses", "exact_y", "problem_index", "motion_index", "time_index"]
    dataset = {key: all_samples[key][mask].contiguous() for key in keys}
    dataset["metadata"] = {
        "name": name,
        "motion_indices": list(motion_indices),
        "time_indices": list(time_indices),
        "num_motions": len(motion_indices),
        "num_time_indices": len(time_indices),
        "num_samples": int(mask.sum().item()),
        "source": "data/samples/all_sampled_problems.pt",
        "points_per_problem": int(all_samples["metadata"]["points_per_problem"]),
    }
    return dataset


def filter_reference_state_dataset(
    reference: dict[str, Any],
    *,
    name: str,
    motion_indices: tuple[int, ...],
    time_indices: tuple[int, ...],
    state: str = "current",
) -> dict[str, Any]:
    motion = reference["motion_index"]
    time = reference["time_index"]
    mask = torch.isin(motion, torch.tensor(motion_indices)) & torch.isin(time, torch.tensor(time_indices))
    initial = reference["p_n_full"] if state == "current" else reference["exact_y"]
    return {
        "initial_y": initial[mask].contiguous(),
        "q": reference["q"][mask].contiguous(),
        "masses": reference["masses"][mask].contiguous(),
        "exact_y": reference["exact_y"][mask].contiguous(),
        "problem_index": reference["problem_index"][mask].contiguous(),
        "motion_index": reference["motion_index"][mask].contiguous(),
        "time_index": reference["time_index"][mask].contiguous(),
        "metadata": {
            "name": name,
            "state": state,
            "motion_indices": list(motion_indices),
            "time_indices": list(time_indices),
            "num_samples": int(mask.sum().item()),
            "source": "data/reference/reference_problems.pt",
        },
    }


def build_train_problem_batches(
    *,
    total_time_steps: int,
    train_motion_indices: tuple[int, ...] = TRAIN_MOTIONS,
    train_time_start: int = 0,
    train_time_stop: int = 400,
    time_steps_per_motion: int = 32,
) -> list[list[int]]:
    batches: list[list[int]] = []
    for start in range(train_time_start, train_time_stop, time_steps_per_motion):
        stop = min(start + time_steps_per_motion, train_time_stop)
        batch_problem_indices = []
        for motion_index in train_motion_indices:
            for time_index in range(start, stop):
                batch_problem_indices.append(int(motion_index) * int(total_time_steps) + int(time_index))
        batches.append(batch_problem_indices)
    return batches


def save_dataset(dataset: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_dir / f"{dataset['metadata']['name']}.pt")


def load_dataset(name: str, root: str | Path = "cloth_5x5_500step_pipeline") -> dict[str, Any]:
    root = Path(root)
    return torch.load(root / "data" / "datasets" / f"{name}.pt", map_location="cpu")


def build_all_datasets(
    *,
    root: Path,
    total_time_steps: int = DEFAULT_TOTAL_TIME_STEPS,
    eval_time_count: int = 50,
    train_time_stop: int = 400,
    time_steps_per_motion_batch: int = 32,
) -> dict[str, Any]:
    all_samples = load_all_samples(root)
    reference = load_reference_problems(root)
    dataset_dir = root / "data" / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_times = tuple(range(0, train_time_stop))
    validation_times = uniform_time_indices(total_time_steps, eval_time_count)
    extrap_times = tuple(range(train_time_stop, total_time_steps))
    eval_times = uniform_time_indices(total_time_steps, eval_time_count)

    datasets = {
        "train": filter_sampled_dataset(
            all_samples,
            name="train",
            motion_indices=TRAIN_MOTIONS,
            time_indices=train_times,
        ),
        "validation": filter_sampled_dataset(
            all_samples,
            name="validation",
            motion_indices=VALIDATION_MOTIONS,
            time_indices=validation_times,
        ),
        "seen_extrap": filter_sampled_dataset(
            all_samples,
            name="seen_extrap",
            motion_indices=TRAIN_MOTIONS,
            time_indices=extrap_times,
        ),
        "unseen_id": filter_sampled_dataset(
            all_samples,
            name="unseen_id",
            motion_indices=UNSEEN_ID_MOTIONS,
            time_indices=eval_times,
        ),
        "ood": filter_sampled_dataset(
            all_samples,
            name="ood",
            motion_indices=OOD_MOTIONS,
            time_indices=eval_times,
        ),
        "current_state_seen_extrap": filter_reference_state_dataset(
            reference,
            name="current_state_seen_extrap",
            motion_indices=TRAIN_MOTIONS,
            time_indices=extrap_times,
            state="current",
        ),
        "current_state_unseen_id": filter_reference_state_dataset(
            reference,
            name="current_state_unseen_id",
            motion_indices=UNSEEN_ID_MOTIONS,
            time_indices=eval_times,
            state="current",
        ),
        "current_state_ood": filter_reference_state_dataset(
            reference,
            name="current_state_ood",
            motion_indices=OOD_MOTIONS,
            time_indices=eval_times,
            state="current",
        ),
    }

    for dataset in datasets.values():
        save_dataset(dataset, dataset_dir)
        print(f"saved {dataset['metadata']['name']}: {dataset['metadata']['num_samples']} samples")

    train_batches = build_train_problem_batches(
        total_time_steps=total_time_steps,
        train_time_stop=train_time_stop,
        time_steps_per_motion=time_steps_per_motion_batch,
    )
    batch_plan = {
        "meaning": (
            "one mini-batch = all 16 train motions x time_steps_per_motion_batch "
            "time-step problems per motion; each problem uses all sampled initial states"
        ),
        "train_motion_indices": list(TRAIN_MOTIONS),
        "train_time_range": [0, train_time_stop - 1],
        "time_steps_per_motion_batch": time_steps_per_motion_batch,
        "num_batches_per_epoch": len(train_batches),
        "problem_indices_by_batch": train_batches,
    }
    save_json(batch_plan, dataset_dir / "train_batch_plan.json")

    manifest = {
        "total_time_steps": total_time_steps,
        "train_time_indices": [0, train_time_stop - 1],
        "validation_time_indices": list(validation_times),
        "seen_extrap_time_indices": [train_time_stop, total_time_steps - 1],
        "eval_time_indices": list(eval_times),
        "datasets": {name: data["metadata"] for name, data in datasets.items()},
        "train_batch_plan": {
            "num_batches_per_epoch": len(train_batches),
            "time_steps_per_motion_batch": time_steps_per_motion_batch,
        },
    }
    save_json(manifest, dataset_dir / "dataset_manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build named train/validation/test datasets.")
    parser.add_argument("--root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--total-time-steps", type=int, default=DEFAULT_TOTAL_TIME_STEPS)
    parser.add_argument("--eval-time-count", type=int, default=50)
    parser.add_argument("--train-time-stop", type=int, default=400)
    parser.add_argument("--time-steps-per-motion-batch", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_all_datasets(
        root=args.root,
        total_time_steps=args.total_time_steps,
        eval_time_count=args.eval_time_count,
        train_time_stop=args.train_time_stop,
        time_steps_per_motion_batch=args.time_steps_per_motion_batch,
    )
    print(json.dumps(manifest["train_batch_plan"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
