#!/usr/bin/env python3
"""Build reusable scaled training datasets and a shared evaluation benchmark.

Examples
--------
python build_scaled_datasets.py --datasets D0 D1-B D2-B
python build_scaled_datasets.py --datasets all --build-benchmark --device cpu
python build_scaled_datasets.py --datasets D0 --build-benchmark --smoke-test

Reference trajectories are cached per (boundary, motion), so D0/D1/D2/D4 reuse
previously solved trajectories. Dataset generation never trains a network and
never evaluates Newton/GD/L-BFGS baselines.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from cloth_scale_common import (
    DatasetSpec,
    ProblemTable,
    SampleSplit,
    TEST_SOBOL_SEED,
    TRAIN_SOBOL_SEED,
    VALIDATION_SOBOL_SEED,
    VALIDATION_TIME_INDICES,
    SEEN_INTERPOLATION_TIME_INDICES,
    UNSEEN_TEST_TIME_INDICES,
    build_boundary_catalogue,
    build_dataset_specs,
    build_motion_catalogue,
    build_sample_split,
    collect_problem_table,
    default_physical_config,
    load_json,
    save_dataset_package,
    save_json,
    stable_hash,
    validate_device,
)


def benchmark_specs() -> dict[str, DatasetSpec]:
    ten_times = tuple(range(0, 100, 10))
    return {
        "validation_core": DatasetSpec(
            name="validation_core",
            description="Shared checkpoint-selection set with unseen motions and mixed legacy/unseen boundaries.",
            boundary_indices=(0, 100, 102, 103),
            motion_indices=tuple(range(1000, 1016)),
            time_indices=VALIDATION_TIME_INDICES,
            points_per_problem=8,
            include_current_and_exact=False,
        ),
        "state_id_legacy": DatasetSpec(
            name="state_id_legacy",
            description="Seen legacy boundary and anchor motions; unseen interpolation times and initial states.",
            boundary_indices=(0,),
            motion_indices=tuple(range(0, 8)),
            time_indices=SEEN_INTERPOLATION_TIME_INDICES,
            points_per_problem=32,
            include_current_and_exact=False,
        ),
        "motion_generalization_legacy": DatasetSpec(
            name="motion_generalization_legacy",
            description="Legacy boundary with completely unseen in-domain motions.",
            boundary_indices=(0,),
            motion_indices=tuple(range(2000, 2016)),
            time_indices=UNSEEN_TEST_TIME_INDICES,
            points_per_problem=16,
            include_current_and_exact=False,
        ),
        "boundary_generalization_seen_motion": DatasetSpec(
            name="boundary_generalization_seen_motion",
            description="Unseen boundary masks combined with anchor motions present in every training dataset.",
            boundary_indices=tuple(range(200, 208)),
            motion_indices=tuple(range(0, 8)),
            time_indices=VALIDATION_TIME_INDICES,
            points_per_problem=8,
            include_current_and_exact=False,
        ),
        "joint_generalization": DatasetSpec(
            name="joint_generalization",
            description="Both boundary masks and in-domain motions are unseen.",
            boundary_indices=tuple(range(200, 208)),
            motion_indices=tuple(range(2000, 2016)),
            time_indices=ten_times,
            points_per_problem=4,
            include_current_and_exact=False,
        ),
        "count_ood": DatasetSpec(
            name="count_ood",
            description="Fixed-point count k=4, deliberately absent from D4-M training.",
            boundary_indices=tuple(range(300, 304)),
            motion_indices=tuple(range(0, 8)),
            time_indices=ten_times,
            points_per_problem=8,
            include_current_and_exact=False,
        ),
        "hard_ood": DatasetSpec(
            name="hard_ood",
            description="No fixed point or interior fixed points, combined with OOD motions.",
            boundary_indices=tuple(range(400, 404)),
            motion_indices=tuple(range(3000, 3008)),
            time_indices=ten_times,
            points_per_problem=8,
            include_current_and_exact=False,
        ),
    }


def smoke_spec(spec: DatasetSpec) -> DatasetSpec:
    return replace(
        spec,
        name=f"{spec.name}_SMOKE",
        description=f"SMOKE TEST: {spec.description}",
        boundary_indices=spec.boundary_indices[:1],
        motion_indices=spec.motion_indices[:1],
        time_indices=(0, 1),
        points_per_problem=min(spec.points_per_problem, 3),
    )


def save_benchmark_split(
    *,
    root: Path,
    spec: DatasetSpec,
    problems: ProblemTable,
    samples: SampleSplit,
    physical: Any,
    boundaries: dict[int, Any],
    motions: dict[int, Any],
) -> dict[str, Any]:
    split_dir = root / spec.name
    split_dir.mkdir(parents=True, exist_ok=True)
    torch.save(problems.serializable(), split_dir / "problems.pt")
    torch.save(samples.serializable(), split_dir / "samples.pt")
    core = {
        "schema_version": 1,
        "benchmark_split": spec.name,
        "dataset_spec": asdict(spec),
        "physical_config": asdict(physical),
        "boundaries": [asdict(boundaries[i]) for i in spec.boundary_indices],
        "motions": [asdict(motions[i]) for i in spec.motion_indices],
        "problem_count": len(problems),
        "sample_count": len(samples),
        "problem_file": "problems.pt",
        "sample_file": "samples.pt",
    }
    manifest = {**core, "split_id": f"{spec.name}_v1_{stable_hash(core, 10)}"}
    save_json(manifest, split_dir / "manifest.json")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["D0"],
        help="Any of D0 D1-B D1-L D2-B D2-L D4-M, or 'all'.",
    )
    parser.add_argument("--build-benchmark", action="store_true")
    parser.add_argument("--benchmark-splits", nargs="*", default=None, help="Optional subset of benchmark split names.")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "scaled_data")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu", help="Reference-solve device. CPU is the conservative default.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel CPU trajectory workers; use 1 with CUDA.")
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    validate_device(device)
    output_root: Path = args.output_root.resolve()
    cache_root = (args.cache_root or (output_root / "_trajectory_cache")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    physical = default_physical_config()
    boundaries = build_boundary_catalogue()
    motions = build_motion_catalogue(physical)
    specs = build_dataset_specs()

    requested = list(args.datasets)
    if "all" in requested:
        requested = list(specs)
    unknown = sorted(set(requested) - set(specs))
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}; available={list(specs)}")

    root_manifest: dict[str, Any] = {
        "schema_version": 1,
        "physical_config": asdict(physical),
        "training_datasets": {},
        "benchmark": {},
        "trajectory_cache": str(cache_root),
    }

    for name in requested:
        spec = smoke_spec(specs[name]) if args.smoke_test else specs[name]
        dataset_dir = output_root / "training" / spec.name
        if (dataset_dir / "manifest.json").exists() and not args.overwrite_dataset:
            manifest = load_json(dataset_dir / "manifest.json")
            print(f"Skipping existing dataset {spec.name}: {manifest['dataset_id']}")
            root_manifest["training_datasets"][spec.name] = manifest
            continue
        if dataset_dir.exists() and args.overwrite_dataset:
            shutil.rmtree(dataset_dir)
        print("\n" + "=" * 100)
        print(f"Building {spec.name}: {spec.description}")
        print(
            f"boundaries={len(spec.boundary_indices)}, motions={len(spec.motion_indices)}, "
            f"times={len(spec.time_indices)}, points/problem={spec.points_per_problem}, "
            f"problems={len(spec.boundary_indices) * len(spec.motion_indices) * len(spec.time_indices):,}, "
            f"samples={len(spec.boundary_indices) * len(spec.motion_indices) * len(spec.time_indices) * spec.points_per_problem:,}"
        )
        problems = collect_problem_table(
            spec=spec,
            cache_root=cache_root,
            physical=physical,
            boundaries=boundaries,
            motions=motions,
            device=device,
            overwrite_cache=args.overwrite_cache,
            progress_prefix=f"{spec.name}: ",
            workers=args.workers,
        )
        samples = build_sample_split(
            problems=problems,
            points_per_problem=spec.points_per_problem,
            base_seed=TRAIN_SOBOL_SEED,
            role=f"train_{spec.name}",
            include_current_and_exact=spec.include_current_and_exact,
        )
        manifest = save_dataset_package(
            output_dir=dataset_dir,
            spec=spec,
            problems=problems,
            split=samples,
            physical=physical,
            boundaries=boundaries,
            motions=motions,
        )
        root_manifest["training_datasets"][spec.name] = manifest
        print(f"Saved {spec.name}: {dataset_dir}")

    if args.build_benchmark:
        benchmark_root = output_root / ("benchmark_SMOKE" if args.smoke_test else "benchmark_v1")
        benchmark_root.mkdir(parents=True, exist_ok=True)
        all_benchmark_specs = benchmark_specs()
        selected_benchmark_names = args.benchmark_splits or list(all_benchmark_specs)
        unknown_benchmark = sorted(set(selected_benchmark_names) - set(all_benchmark_specs))
        if unknown_benchmark:
            raise ValueError(f"Unknown benchmark splits: {unknown_benchmark}")
        for split_name in selected_benchmark_names:
            original_spec = all_benchmark_specs[split_name]
            spec = smoke_spec(original_spec) if args.smoke_test else original_spec
            split_dir = benchmark_root / spec.name
            if (split_dir / "manifest.json").exists() and not args.overwrite_dataset:
                manifest = load_json(split_dir / "manifest.json")
                print(f"Skipping existing benchmark split {spec.name}: {manifest['split_id']}")
                root_manifest["benchmark"][spec.name] = manifest
                continue
            if split_dir.exists() and args.overwrite_dataset:
                shutil.rmtree(split_dir)
            print("\n" + "-" * 100)
            print(f"Building benchmark split {spec.name}: {spec.description}")
            problems = collect_problem_table(
                spec=spec,
                cache_root=cache_root,
                physical=physical,
                boundaries=boundaries,
                motions=motions,
                device=device,
                overwrite_cache=args.overwrite_cache,
                progress_prefix=f"benchmark/{spec.name}: ",
                workers=args.workers,
            )
            samples = build_sample_split(
                problems=problems,
                points_per_problem=spec.points_per_problem,
                base_seed=VALIDATION_SOBOL_SEED if "validation" in split_name else TEST_SOBOL_SEED,
                role=spec.name,
                include_current_and_exact=spec.include_current_and_exact,
            )
            manifest = save_benchmark_split(
                root=benchmark_root,
                spec=spec,
                problems=problems,
                samples=samples,
                physical=physical,
                boundaries=boundaries,
                motions=motions,
            )
            root_manifest["benchmark"][spec.name] = manifest
        all_benchmark_manifests: dict[str, Any] = {}
        for candidate in sorted(benchmark_root.iterdir()):
            manifest_path = candidate / "manifest.json"
            if candidate.is_dir() and manifest_path.exists():
                manifest = load_json(manifest_path)
                all_benchmark_manifests[manifest["benchmark_split"]] = manifest
        root_manifest["benchmark"] = all_benchmark_manifests
        benchmark_core = {
            "schema_version": 1,
            "physical_config": asdict(physical),
            "splits": all_benchmark_manifests,
        }
        benchmark_manifest = {
            **benchmark_core,
            "benchmark_id": f"benchmark_v1_{stable_hash(benchmark_core, 10)}",
        }
        save_json(benchmark_manifest, benchmark_root / "manifest.json")
        root_manifest["benchmark_manifest"] = benchmark_manifest
        print(f"Saved shared benchmark: {benchmark_root}")

    training_root = output_root / "training"
    if training_root.exists():
        all_training_manifests: dict[str, Any] = {}
        for candidate in sorted(training_root.iterdir()):
            manifest_path = candidate / "manifest.json"
            if candidate.is_dir() and manifest_path.exists():
                manifest = load_json(manifest_path)
                all_training_manifests[manifest["dataset_spec"]["name"]] = manifest
        root_manifest["training_datasets"] = all_training_manifests
    save_json(root_manifest, output_root / ("build_manifest_smoke.json" if args.smoke_test else "build_manifest.json"))
    print("\nDataset construction completed.")
    print(f"Output root: {output_root}")
    print(f"Trajectory cache: {cache_root}")


if __name__ == "__main__":
    main()
