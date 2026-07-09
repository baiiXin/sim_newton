#!/usr/bin/env python3
"""Evaluate trained MLP checkpoints on an existing benchmark.

The script never rebuilds datasets and never reruns baselines. When a cached
baseline summary is supplied, it only merges the existing statistics into the
comparison report.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from cloth_scale_common import (
    ProblemTable,
    SampleSplit,
    build_model_from_checkpoint,
    default_physical_config,
    evaluate_learned_or_gd,
    load_json,
    save_json,
    stable_hash,
    validate_device,
)


def load_split(split_dir: Path) -> tuple[dict[str, Any], ProblemTable, SampleSplit]:
    manifest = load_json(split_dir / "manifest.json")
    problems = ProblemTable.from_serializable(
        torch.load(split_dir / manifest.get("problem_file", "problems.pt"), map_location="cpu", weights_only=False)
    )
    samples = SampleSplit.from_serializable(
        torch.load(split_dir / manifest.get("sample_file", "samples.pt"), map_location="cpu", weights_only=False)
    )
    return manifest, problems, samples


def compact_metrics(name: str, split_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "solver": name,
        "split": split_name,
        "final_residual_mean": metrics.get("final_residual_mean"),
        "final_residual_p95": metrics.get("final_residual_p95"),
        "final_residual_max": metrics.get("final_residual_max"),
        "worst_boundary_residual_p95": metrics.get("worst_boundary_final_residual_p95"),
        "worst_boundary_residual_max": metrics.get("worst_boundary_final_residual_max"),
        "worst_motion_residual_p95": metrics.get("worst_motion_final_residual_p95"),
        "worst_motion_residual_max": metrics.get("worst_motion_final_residual_max"),
        "final_exact_error_p95": metrics.get("final_exact_error_p95"),
        "final_exact_error_max": metrics.get("final_exact_error_max"),
        "num_nonfinite": metrics.get("final_residual_num_nonfinite"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "model_evaluations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.resolve()
    benchmark_manifest = load_json(benchmark_root / "manifest.json")
    benchmark_id = benchmark_manifest["benchmark_id"]
    device = torch.device(args.device)
    validate_device(device)
    physical = default_physical_config()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    available_splits = sorted(
        p.name for p in benchmark_root.iterdir() if p.is_dir() and (p / "manifest.json").exists()
    )
    requested_splits = args.splits or [name for name in available_splits if name != "validation_core"]
    unknown = sorted(set(requested_splits) - set(available_splits))
    if unknown:
        raise ValueError(f"Unknown splits: {unknown}")

    loaded_splits: dict[str, tuple[dict[str, Any], ProblemTable, SampleSplit]] = {
        name: load_split(benchmark_root / name) for name in requested_splits
    }

    baseline_summary: dict[str, Any] | None = None
    if args.baseline_summary is not None:
        baseline_summary = load_json(args.baseline_summary.resolve())
        if baseline_summary["benchmark_id"] != benchmark_id:
            raise RuntimeError("Baseline summary benchmark_id does not match requested benchmark")

    all_rows: list[dict[str, Any]] = []
    evaluation_reports: list[dict[str, Any]] = []

    for checkpoint_path in args.checkpoints:
        checkpoint_path = checkpoint_path.resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("benchmark_id") != benchmark_id:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} uses benchmark_id={checkpoint.get('benchmark_id')}, expected {benchmark_id}"
            )
        model = build_model_from_checkpoint(checkpoint, device)
        model_name = checkpoint.get("run_id", checkpoint_path.parent.name)
        eval_core = {
            "checkpoint": str(checkpoint_path),
            "benchmark_id": benchmark_id,
            "steps": args.steps,
        }
        evaluation_id = f"{model_name}_{stable_hash(eval_core, 10)}"
        report_path = output_root / f"{evaluation_id}.json"
        if report_path.exists() and not args.overwrite:
            report = load_json(report_path)
            evaluation_reports.append(report)
            for row in report.get("compact_rows", []):
                all_rows.append(row)
            print(f"Skipping cached model evaluation: {report_path}")
            continue

        print("\n" + "=" * 100)
        print(f"Evaluating {model_name}")
        print(f"checkpoint={checkpoint_path}")
        split_results: dict[str, Any] = {}
        compact_rows: list[dict[str, Any]] = []
        for split_name in requested_splits:
            print(f"  split={split_name}")
            split_manifest, problems, samples = loaded_splits[split_name]
            metrics = evaluate_learned_or_gd(
                solver="learned",
                model=model,
                problems=problems,
                split=samples,
                physical=physical,
                steps=args.steps,
                batch_size=args.batch_size,
                device=device,
            )
            split_results[split_name] = {
                "split_id": split_manifest["split_id"],
                "metrics": metrics,
            }
            row = compact_metrics(model_name, split_name, metrics)
            row.update({
                "kind": "learned",
                "checkpoint": str(checkpoint_path),
                "dataset_id": checkpoint.get("dataset_id"),
                "model_activation": checkpoint["model_spec"]["activation"],
                "model_depth": checkpoint["model_spec"]["depth"],
                "model_width": checkpoint["model_spec"]["width"],
                "model_bias": checkpoint["model_spec"]["use_bias"],
            })
            compact_rows.append(row)
            all_rows.append(row)

        report = {
            "evaluation_id": evaluation_id,
            "model_name": model_name,
            "checkpoint": str(checkpoint_path),
            "dataset_id": checkpoint.get("dataset_id"),
            "benchmark_id": benchmark_id,
            "model_spec": checkpoint["model_spec"],
            "best_epoch": checkpoint.get("best_epoch"),
            "steps": args.steps,
            "split_results": split_results,
            "compact_rows": compact_rows,
        }
        save_json(report, report_path)
        evaluation_reports.append(report)

    if baseline_summary is not None:
        for split_name in requested_splits:
            split_baselines = baseline_summary.get("splits", {}).get(split_name, {})
            for solver_key, label in (
                ("gradient_descent", "gradient_descent"),
                ("full_newton", "full_newton"),
                ("l_bfgs", "l_bfgs"),
            ):
                if solver_key not in split_baselines:
                    continue
                row = compact_metrics(label, split_name, split_baselines[solver_key])
                row.update({"kind": "baseline", "baseline_id": baseline_summary["baseline_id"]})
                all_rows.append(row)

    combined = {
        "benchmark_id": benchmark_id,
        "baseline_id": baseline_summary.get("baseline_id") if baseline_summary else None,
        "model_evaluations": evaluation_reports,
        "compact_rows": all_rows,
    }
    save_json(combined, output_root / "combined_comparison.json")
    write_csv(all_rows, output_root / "combined_comparison.csv")
    print("\nModel evaluation completed.")
    print(f"Comparison JSON: {output_root / 'combined_comparison.json'}")
    print(f"Comparison CSV:  {output_root / 'combined_comparison.csv'}")


if __name__ == "__main__":
    main()
