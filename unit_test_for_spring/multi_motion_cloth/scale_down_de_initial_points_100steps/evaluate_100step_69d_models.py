"""Evaluate 100-step 3x69D history-input cloth models with dataset-style tests."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
NONLINEAR_DIR = REPO_ROOT / "unit_test_for_spring" / "multi_motion_cloth" / "nonlinear"
HISTORY_SCRIPT = NONLINEAR_DIR / "fixed_left_edge_5x5_cloth_history_input_default_init_ablation.py"
HISTORY_ROOT = NONLINEAR_DIR / "fixed_left_edge_5x5_cloth_history_input_default_init_ablation"
DEGENERATE_SCRIPT = NONLINEAR_DIR / "fixed_left_edge_5x5_cloth_degenerate_no_initial_perturbation_no_repetition.py"
DEGENERATE_ROOT = NONLINEAR_DIR / "fixed_left_edge_5x5_cloth_degenerate_no_initial_perturbation_no_repetition"

DEFAULT_DATASETS = (
    "seen_motion_temporal_interpolation",
    "seen_motion_temporal_extrapolation",
    "unseen_id_test",
    "ood_test",
)
DEFAULT_REPORT_STEPS = (1, 3, 5, 10, 30, 50)
MODEL_DIR_RE = re.compile(
    r"^(?P<prefix>.*)activation_(?P<activation>identity|relu|tanh)_depth_(?P<depth>\d+)_width_(?P<width>\d+)_(?P<bias>no_bias|with_bias)$"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_json(data: Any, path: Path) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, torch.Tensor):
            return safe(value.detach().cpu().tolist())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe(data), handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def default_config(module, args: argparse.Namespace):
    return module.RuntimeConfig(
        total_time_steps=module.DEFAULT_TOTAL_TIME_STEPS,
        train_points_per_problem=module.DEFAULT_TRAIN_POINTS_PER_PROBLEM,
        eval_points_per_problem=module.DEFAULT_EVAL_POINTS_PER_PROBLEM,
        epochs=module.DEFAULT_EPOCHS,
        validation_interval=module.DEFAULT_VALIDATION_INTERVAL,
        diagnostic_interval=module.DEFAULT_DIAGNOSTIC_INTERVAL,
        evaluation_steps=int(args.steps),
        evaluation_batch_size=int(args.batch_size),
        k_values=tuple(module.DEFAULT_K_VALUES),
        epochs_per_k=module.DEFAULT_EPOCHS_PER_K,
        report_steps=tuple(sorted(set(int(s) for s in args.report_steps if 0 < int(s) <= int(args.steps)) | {int(args.steps)})),
        residual_length_scale=module.DEFAULT_RESIDUAL_LENGTH_SCALE,
        gradient_clip_norm=module.DEFAULT_GRADIENT_CLIP_NORM,
        sampling_radius_min=module.DEFAULT_SAMPLING_RADIUS_MIN,
        sampling_radius_max=module.DEFAULT_SAMPLING_RADIUS_MAX,
        device=str(args.device),
        activations=tuple(module.ACTIVATION_NAMES),
        depths=tuple(module.HIDDEN_DEPTHS),
        widths=tuple(module.HIDDEN_WIDTHS),
        config_index=None,
        list_configs=False,
        skip_completed=False,
        resume=False,
        skip_plots=True,
        save_datasets=False,
    )


def load_reference_problems(module, reference_path: Path) -> list[Any]:
    data = load_json(reference_path)
    problems: list[Any] = []
    for record in data["problems"]:
        problems.append(
            module.TimeStepProblem(
                index=int(record["index"]),
                motion_index=int(record["motion_index"]),
                motion_name=str(record["motion_name"]),
                motion_split=str(record["motion_split"]),
                motion_category=str(record["motion_category"]),
                local_time_index=int(record["local_time_index"]),
                time=float(record["time"]),
                p_n_full=torch.tensor(record["p_n_full"], dtype=module.TORCH_DTYPE),
                v_n_full=torch.tensor(record["v_n_full"], dtype=module.TORCH_DTYPE),
                q_free=torch.tensor(record["q_free"], dtype=module.TORCH_DTYPE),
                free_masses=torch.tensor(record["free_masses"], dtype=module.TORCH_DTYPE),
                exact_y_free=torch.tensor(record["exact_y_free"], dtype=module.TORCH_DTYPE),
                raw_sampling_radius=float(record["raw_sampling_radius"]),
                sampling_radius=float(record["sampling_radius"]),
                exact_energy=float(record["exact_energy"]),
                exact_residual=float(record["exact_residual"]),
            )
        )
    return problems


def build_history_eval_datasets(module, config, reference_root: Path) -> tuple[Any, dict[str, Any]]:
    physical = module.default_physical_config()
    motions, motion_split = module.build_motion_catalogue(physical)
    reference_path = reference_root / "reference_time_step_problems.json"
    if reference_path.exists():
        print(f"loading reference problems from {reference_path}")
        problems = load_reference_problems(module, reference_path)
    else:
        print("reference_time_step_problems.json not found; regenerating references")
        problems = module.generate_all_reference_sequences(physical, motions, config)
    lookup = module.problem_lookup(problems)
    datasets = {
        "validation": module.build_dataset_for_motion_times(
            lookup=lookup,
            motion_indices=motion_split.validation_motion_indices,
            time_indices=module.VALIDATION_TIME_INDICES,
            points_per_problem=config.eval_points_per_problem,
            base_seed=module.VALIDATION_SOBOL_SEED,
            role="unseen_motion_validation",
            physical=physical,
            include_explicit_train_points=False,
        ),
        "seen_motion_temporal_interpolation": module.build_dataset_for_motion_times(
            lookup=lookup,
            motion_indices=motion_split.train_motion_indices,
            time_indices=module.SEEN_INTERPOLATION_TIME_INDICES,
            points_per_problem=config.eval_points_per_problem,
            base_seed=module.SEEN_INTERPOLATION_TEST_SOBOL_SEED,
            role="seen_motion_temporal_interpolation",
            physical=physical,
            include_explicit_train_points=False,
        ),
        "seen_motion_temporal_extrapolation": module.build_dataset_for_motion_times(
            lookup=lookup,
            motion_indices=motion_split.train_motion_indices,
            time_indices=module.SEEN_EXTRAPOLATION_TIME_INDICES,
            points_per_problem=config.eval_points_per_problem,
            base_seed=module.SEEN_EXTRAPOLATION_TEST_SOBOL_SEED,
            role="seen_motion_temporal_extrapolation",
            physical=physical,
            include_explicit_train_points=False,
        ),
        "unseen_id_test": module.build_dataset_for_motion_times(
            lookup=lookup,
            motion_indices=motion_split.id_test_motion_indices,
            time_indices=module.UNSEEN_TEST_TIME_INDICES,
            points_per_problem=config.eval_points_per_problem,
            base_seed=module.UNSEEN_ID_TEST_SOBOL_SEED,
            role="unseen_id_test",
            physical=physical,
            include_explicit_train_points=False,
        ),
        "ood_test": module.build_dataset_for_motion_times(
            lookup=lookup,
            motion_indices=motion_split.ood_test_motion_indices,
            time_indices=module.UNSEEN_TEST_TIME_INDICES,
            points_per_problem=config.eval_points_per_problem,
            base_seed=module.OOD_TEST_SOBOL_SEED,
            role="ood_test",
            physical=physical,
            include_explicit_train_points=False,
        ),
        "current_state_seen_motion": module.build_special_state_dataset(
            lookup=lookup,
            motion_indices=motion_split.train_motion_indices,
            time_indices=module.SEEN_INTERPOLATION_TIME_INDICES,
            state="current",
            role="current_state_seen_motion",
        ),
        "current_state_unseen_id": module.build_special_state_dataset(
            lookup=lookup,
            motion_indices=motion_split.id_test_motion_indices,
            time_indices=module.UNSEEN_TEST_TIME_INDICES,
            state="current",
            role="current_state_unseen_id",
        ),
        "current_state_ood": module.build_special_state_dataset(
            lookup=lookup,
            motion_indices=motion_split.ood_test_motion_indices,
            time_indices=module.UNSEEN_TEST_TIME_INDICES,
            state="current",
            role="current_state_ood",
        ),
    }
    metadata = {
        "dataset_source": "fixed_left_edge_5x5_cloth_history_input_default_init_ablation",
        "reference_problem_source": str(reference_path) if reference_path.exists() else "regenerated",
        "total_time_steps": config.total_time_steps,
        "eval_points_per_problem": config.eval_points_per_problem,
        "motion_split": asdict(motion_split),
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
    }
    return physical, datasets, metadata


def parse_model_dir(path: Path) -> dict[str, Any] | None:
    match = MODEL_DIR_RE.match(path.name)
    if match is None:
        return None
    return {
        "activation": match.group("activation"),
        "depth": int(match.group("depth")),
        "width": int(match.group("width")),
        "use_bias": match.group("bias") == "with_bias",
    }


def discover_models(root: Path, source_key: str, filters: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for checkpoint in sorted(root.glob("*/best_validation_model_state_dict.pt")):
        model_dir = checkpoint.parent
        parsed = parse_model_dir(model_dir)
        if parsed is None:
            continue
        if filters.activations and parsed["activation"] not in set(filters.activations):
            continue
        if filters.depths and parsed["depth"] not in set(int(v) for v in filters.depths):
            continue
        if filters.widths and parsed["width"] not in set(int(v) for v in filters.widths):
            continue
        records.append({
            "source_key": source_key,
            "model_dir": model_dir,
            "checkpoint": checkpoint,
            **parsed,
        })
    return records


def load_model(module, record: dict[str, Any], device: torch.device):
    model_spec = module.ModelSpec(
        activation=record["activation"],
        depth=int(record["depth"]),
        width=int(record["width"]),
        use_bias=bool(record["use_bias"]),
    )
    model = module.MLPOptimizer(module.DEFAULT_RESIDUAL_LENGTH_SCALE, model_spec).to(device)
    state_dict = torch.load(record["checkpoint"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, model_spec


def summary_row(source_key: str, model_spec, dataset_name: str, metrics: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    return {
        "source": source_key,
        "experiment_name": model_spec.experiment_name,
        "activation": model_spec.activation,
        "depth": model_spec.depth,
        "width": model_spec.width,
        "use_bias": model_spec.use_bias,
        "dataset": dataset_name,
        "num_points": metrics.get("num_points"),
        "num_motions": metrics.get("num_motions"),
        "steps": metrics.get("steps"),
        "final_residual_mean": metrics.get("final_residual_mean"),
        "final_residual_p95": metrics.get("final_residual_p95"),
        "final_residual_max": metrics.get("final_residual_max"),
        "final_residual_num_nonfinite": metrics.get("final_residual_num_nonfinite"),
        "worst_motion_final_residual_p95": metrics.get("worst_motion_final_residual_p95"),
        "worst_motion_final_residual_p95_motion_index": metrics.get("worst_motion_final_residual_p95_motion_index"),
        "worst_motion_final_residual_max": metrics.get("worst_motion_final_residual_max"),
        "worst_motion_final_residual_max_motion_index": metrics.get("worst_motion_final_residual_max_motion_index"),
        "final_exact_error_p95": metrics.get("final_exact_error_p95"),
        "final_exact_error_max": metrics.get("final_exact_error_max"),
        "final_energy_gap_p95": metrics.get("final_energy_gap_p95"),
        "final_energy_gap_max": metrics.get("final_energy_gap_max"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "checkpoint": str(checkpoint),
    }


def plot_residual_curves(records: list[dict[str, Any]], output_root: Path) -> None:
    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(str(record["dataset"]), []).append(record)
    for dataset_name, selected in by_dataset.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for record in selected:
            curve = np.asarray(record["residual_mean_by_step"], dtype=float)
            label = f"{record['source']} {record['activation']} d{record['depth']} w{record['width']}"
            ax.plot(np.arange(curve.size), np.maximum(curve, 1e-16), label=label, alpha=0.75)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean residual")
        ax.set_title(f"{dataset_name}: 100-step 3x69D dataset-style solve")
        if len(selected) <= 18:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{dataset_name}_residual_mean_by_iteration.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 100-step 3x69D history-input models on shared dataset-style tests.")
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--degenerate-root", type=Path, default=DEGENERATE_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "unit_test_for_spring" / "multi_motion_cloth" / "scale_down_de_initial_points_100steps")
    parser.add_argument("--sources", nargs="+", default=["history", "degenerate"], choices=["history", "degenerate"])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--activations", nargs="*", default=None, choices=["identity", "relu", "tanh"])
    parser.add_argument("--depths", type=int, nargs="*", default=None)
    parser.add_argument("--widths", type=int, nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--report-steps", type=int, nargs="+", default=list(DEFAULT_REPORT_STEPS))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    module = load_module(HISTORY_SCRIPT, "cloth100_history_eval_source")
    config = default_config(module, args)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    physical, all_datasets, dataset_metadata = build_history_eval_datasets(module, config, args.history_root.resolve())
    datasets = {name: all_datasets[name] for name in args.datasets}

    source_roots = {
        "history": args.history_root.resolve(),
        "degenerate": args.degenerate_root.resolve(),
    }
    model_records: list[dict[str, Any]] = []
    for source_key in args.sources:
        model_records.extend(discover_models(source_roots[source_key], source_key, args))
    if not model_records:
        raise RuntimeError("No matching model checkpoints were found")

    save_json(
        {
            "history_script": str(HISTORY_SCRIPT),
            "degenerate_script": str(DEGENERATE_SCRIPT),
            "history_root": str(args.history_root.resolve()),
            "degenerate_root": str(args.degenerate_root.resolve()),
            "sources": list(args.sources),
            "datasets": list(args.datasets),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "report_steps": list(config.report_steps),
            "device": str(device),
            "dataset_metadata": dataset_metadata,
            "model_count": len(model_records),
            "model_definitions_note": (
                "The two source scripts use the same 3x69D MLPOptimizer definition; "
                "only ModelSpec.experiment_name prefixes differ."
            ),
        },
        output_root / "run_config.json",
    )

    rows: list[dict[str, Any]] = []
    plot_records: list[dict[str, Any]] = []
    all_metrics: dict[str, Any] = {}
    for record in model_records:
        model, model_spec = load_model(module, record, device)
        for dataset_name, dataset in datasets.items():
            result_dir = output_root / record["source_key"] / record["model_dir"].name / dataset_name
            metrics_path = result_dir / "metrics.json"
            if metrics_path.exists() and (not args.overwrite or args.plot_only):
                metrics = load_json(metrics_path)
                print(f"reuse {record['source_key']} {record['model_dir'].name} on {dataset_name}")
            elif args.plot_only:
                print(f"skip missing metrics for {record['source_key']} {record['model_dir'].name} on {dataset_name}")
                continue
            else:
                print(f"evaluating {record['source_key']} {record['model_dir'].name} on {dataset_name} ({len(dataset)} points)")
                metrics = module.evaluate_solver_on_dataset(
                    solver="learned",
                    model=model,
                    dataset_cpu=dataset,
                    physical=physical,
                    steps=int(args.steps),
                    batch_size=int(args.batch_size),
                    report_steps=config.report_steps,
                    device=device,
                )
                metrics["solver_name"] = f"{record['source_key']}/{record['model_dir'].name}"
                metrics["checkpoint"] = str(record["checkpoint"])
                result_dir.mkdir(parents=True, exist_ok=True)
                save_json(metrics, metrics_path)
            rows.append(summary_row(record["source_key"], model_spec, dataset_name, metrics, record["checkpoint"]))
            plot_records.append(
                {
                    "source": record["source_key"],
                    "activation": model_spec.activation,
                    "depth": model_spec.depth,
                    "width": model_spec.width,
                    "dataset": dataset_name,
                    "residual_mean_by_step": metrics["residual_mean_by_step"],
                }
            )
            all_metrics[f"{record['source_key']}/{record['model_dir'].name}/{dataset_name}"] = metrics

    write_csv(rows, output_root / "summary_metrics.csv")
    save_json({"records": all_metrics}, output_root / "all_metrics.json")
    plot_residual_curves(plot_records, output_root)
    print(f"wrote summary to {output_root / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
