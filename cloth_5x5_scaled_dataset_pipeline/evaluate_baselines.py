#!/usr/bin/env python3
"""Evaluate and cache Newton, GD, and L-BFGS on the shared benchmark.

This script never builds datasets and never trains a neural network. GD step
size and L-BFGS memory are selected once on validation_core, then reused for
all benchmark splits.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from cloth_scale_common import (
    ProblemTable,
    SampleSplit,
    default_physical_config,
    evaluate_lbfgs,
    evaluate_learned_or_gd,
    evaluate_newton,
    load_json,
    save_json,
    stable_hash,
    validate_device,
    validation_selection_key,
)

GD_CANDIDATES = (1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4)
LBFGS_MEMORY_CANDIDATES = (5, 10, 20)


def load_split(split_dir: Path) -> tuple[dict[str, Any], ProblemTable, SampleSplit]:
    manifest = load_json(split_dir / "manifest.json")
    problems = ProblemTable.from_serializable(
        torch.load(split_dir / manifest.get("problem_file", "problems.pt"), map_location="cpu", weights_only=False)
    )
    samples = SampleSplit.from_serializable(
        torch.load(split_dir / manifest.get("sample_file", "samples.pt"), map_location="cpu", weights_only=False)
    )
    return manifest, problems, samples


def select_gd(
    *,
    problems: ProblemTable,
    samples: SampleSplit,
    physical: Any,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    best: tuple[tuple[float, ...], float] | None = None
    for alpha in GD_CANDIDATES:
        print(f"Validation GD alpha={alpha:.1e}")
        metrics = evaluate_learned_or_gd(
            solver="gradient_descent",
            problems=problems,
            split=samples,
            physical=physical,
            steps=steps,
            batch_size=batch_size,
            device=device,
            gd_step_size=alpha,
        )
        key = validation_selection_key(metrics)
        records.append({"step_size": alpha, "selection_key": key, "metrics": metrics})
        if key is not None and (best is None or key < best[0]):
            best = (key, alpha)
    if best is None:
        raise RuntimeError("No finite GD candidate")
    return {
        "candidate_step_sizes": list(GD_CANDIDATES),
        "selected_step_size": best[1],
        "selected_key": best[0],
        "records": records,
    }


def select_lbfgs(
    *,
    problems: ProblemTable,
    samples: SampleSplit,
    physical: Any,
    steps: int,
    batch_size: int,
    device: torch.device,
    max_line_search: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    best: tuple[tuple[float, ...], int] | None = None
    for memory in LBFGS_MEMORY_CANDIDATES:
        print(f"Validation L-BFGS memory={memory}")
        metrics = evaluate_lbfgs(
            problems=problems,
            split=samples,
            physical=physical,
            steps=steps,
            batch_size=batch_size,
            device=device,
            memory=memory,
            max_line_search=max_line_search,
        )
        key = validation_selection_key(metrics)
        records.append({"memory": memory, "selection_key": key, "metrics": metrics})
        if key is not None and (best is None or key < best[0]):
            best = (key, memory)
    if best is None:
        raise RuntimeError("No finite L-BFGS candidate")
    return {
        "candidate_memories": list(LBFGS_MEMORY_CANDIDATES),
        "selected_memory": best[1],
        "selected_key": best[0],
        "max_line_search": max_line_search,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--validation-split", default="validation_core")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--gd-batch-size", type=int, default=8192)
    parser.add_argument("--newton-batch-size", type=int, default=512)
    parser.add_argument("--lbfgs-batch-size", type=int, default=4096)
    parser.add_argument("--lbfgs-max-line-search", type=int, default=20)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-newton", action="store_true")
    parser.add_argument("--skip-gd", action="store_true")
    parser.add_argument("--skip-lbfgs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    benchmark_manifest = load_json(benchmark_root / "manifest.json")
    benchmark_id = benchmark_manifest["benchmark_id"]
    output_root = (args.output_root or (benchmark_root.parent / "baseline_results")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    validate_device(device)
    physical = default_physical_config()

    config_core = {
        "benchmark_id": benchmark_id,
        "steps": args.steps,
        "gd_candidates": GD_CANDIDATES,
        "lbfgs_memory_candidates": LBFGS_MEMORY_CANDIDATES,
        "lbfgs_max_line_search": args.lbfgs_max_line_search,
        "dtype": "float64",
    }
    baseline_id = f"baselines_{stable_hash(config_core, 12)}"
    run_dir = output_root / baseline_id
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"Baseline cache already exists: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    validation_dir = benchmark_root / args.validation_split
    _, validation_problems, validation_samples = load_split(validation_dir)

    gd_selection: dict[str, Any] | None = None
    if not args.skip_gd:
        gd_selection = select_gd(
            problems=validation_problems,
            samples=validation_samples,
            physical=physical,
            steps=args.steps,
            batch_size=args.gd_batch_size,
            device=device,
        )
        save_json(gd_selection, run_dir / "gd_selection.json")

    lbfgs_selection: dict[str, Any] | None = None
    if not args.skip_lbfgs:
        lbfgs_selection = select_lbfgs(
            problems=validation_problems,
            samples=validation_samples,
            physical=physical,
            steps=args.steps,
            batch_size=args.lbfgs_batch_size,
            device=device,
            max_line_search=args.lbfgs_max_line_search,
        )
        save_json(lbfgs_selection, run_dir / "lbfgs_selection.json")

    available_splits = sorted(
        p.name for p in benchmark_root.iterdir() if p.is_dir() and (p / "manifest.json").exists()
    )
    requested_splits = args.splits or available_splits
    unknown = sorted(set(requested_splits) - set(available_splits))
    if unknown:
        raise ValueError(f"Unknown benchmark splits: {unknown}")

    results: dict[str, Any] = {}
    for split_name in requested_splits:
        print("\n" + "=" * 100)
        print(f"Evaluating baselines on {split_name}")
        split_manifest, problems, samples = load_split(benchmark_root / split_name)
        split_results: dict[str, Any] = {"split_id": split_manifest["split_id"]}
        if not args.skip_gd:
            assert gd_selection is not None
            split_results["gradient_descent"] = evaluate_learned_or_gd(
                solver="gradient_descent",
                problems=problems,
                split=samples,
                physical=physical,
                steps=args.steps,
                batch_size=args.gd_batch_size,
                device=device,
                gd_step_size=float(gd_selection["selected_step_size"]),
            )
        if not args.skip_newton:
            split_results["full_newton"] = evaluate_newton(
                problems=problems,
                split=samples,
                physical=physical,
                steps=args.steps,
                batch_size=args.newton_batch_size,
                device=device,
            )
        if not args.skip_lbfgs:
            assert lbfgs_selection is not None
            split_results["l_bfgs"] = evaluate_lbfgs(
                problems=problems,
                split=samples,
                physical=physical,
                steps=args.steps,
                batch_size=args.lbfgs_batch_size,
                device=device,
                memory=int(lbfgs_selection["selected_memory"]),
                max_line_search=args.lbfgs_max_line_search,
            )
        results[split_name] = split_results
        save_json(split_results, run_dir / f"{split_name}.json")

    summary = {
        "baseline_id": baseline_id,
        "benchmark_id": benchmark_id,
        "config": config_core,
        "device": str(device),
        "gd_selection": gd_selection,
        "lbfgs_selection": lbfgs_selection,
        "splits": results,
    }
    save_json(summary, summary_path)
    save_json({"baseline_id": baseline_id, "path": str(run_dir)}, output_root / "latest.json")
    print("\nBaseline evaluation completed.")
    print(f"Cached results: {run_dir}")


if __name__ == "__main__":
    main()
