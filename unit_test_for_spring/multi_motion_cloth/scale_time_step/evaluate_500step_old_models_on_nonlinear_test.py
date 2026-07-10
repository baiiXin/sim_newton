"""Evaluate 500-step old cloth models with the nonlinear ablation test protocol.

This script reuses the test-set construction and metric aggregation from:

    unit_test_for_spring/multi_motion_cloth/nonlinear/
    fixed_left_edge_5x5_cloth_history_input_default_init_ablation.py

It compares three old full-state 500-step models against the matching
history-input default-init nonlinear models:

    identity / relu / tanh, depth 1, width 256, no bias

The nonlinear ablation model is a 69D free-state optimizer.  The 500-step model
is a 75D full-state optimizer, so this script wraps it with a small adapter that
converts the nonlinear test batch's free states to full states before each model
update, then converts the update back to free coordinates.  Metrics are still
computed by the original nonlinear evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]

NONLINEAR_SCRIPT = (
    REPO_ROOT
    / "unit_test_for_spring"
    / "multi_motion_cloth"
    / "nonlinear"
    / "fixed_left_edge_5x5_cloth_history_input_default_init_ablation.py"
)
CLOTH03_SCRIPT = REPO_ROOT / "cloth_5x5_500step_project" / "cloth03_solvers_and_models.py"

OLD_MODEL_ROOT = (
    REPO_ROOT
    / "cloth_5x5_500step_project"
    / "cloth_5x5_500step_pipeline"
    / "models"
    / "old"
)
NONLINEAR_MODEL_ROOT = (
    REPO_ROOT
    / "unit_test_for_spring"
    / "multi_motion_cloth"
    / "nonlinear"
    / "fixed_left_edge_5x5_cloth_history_input_default_init_ablation"
)
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "old_500step_vs_nonlinear_test_eval"

ACTIVATIONS = ("identity", "relu", "tanh")
DEFAULT_SPLITS = (
    "seen_motion_temporal_interpolation",
    "seen_motion_temporal_extrapolation",
    "unseen_id_test",
    "ood_test",
)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def make_eval_config(base: ModuleType, args: argparse.Namespace) -> Any:
    return base.RuntimeConfig(
        total_time_steps=args.total_time_steps,
        train_points_per_problem=base.DEFAULT_TRAIN_POINTS_PER_PROBLEM,
        eval_points_per_problem=args.eval_points_per_problem,
        epochs=base.DEFAULT_EPOCHS,
        validation_interval=base.DEFAULT_VALIDATION_INTERVAL,
        diagnostic_interval=base.DEFAULT_DIAGNOSTIC_INTERVAL,
        evaluation_steps=args.steps,
        evaluation_batch_size=args.batch_size,
        k_values=tuple(base.DEFAULT_K_VALUES),
        epochs_per_k=base.DEFAULT_EPOCHS_PER_K,
        report_steps=tuple(base.DEFAULT_REPORT_STEPS),
        residual_length_scale=args.residual_length_scale,
        gradient_clip_norm=base.DEFAULT_GRADIENT_CLIP_NORM,
        sampling_radius_min=base.DEFAULT_SAMPLING_RADIUS_MIN,
        sampling_radius_max=base.DEFAULT_SAMPLING_RADIUS_MAX,
        device=args.device,
        activations=ACTIVATIONS,
        depths=(1,),
        widths=(256,),
        config_index=None,
        list_configs=False,
        skip_completed=False,
        resume=False,
        skip_plots=True,
        save_datasets=False,
    )


def build_test_datasets(base: ModuleType, config: Any, physical: Any) -> dict[str, Any]:
    motions, motion_split = base.build_motion_catalogue(physical)
    problems = base.generate_all_reference_sequences(physical, motions, config)
    lookup = base.problem_lookup(problems)
    return {
        "seen_motion_temporal_interpolation": base.build_dataset_for_motion_times(
            lookup=lookup,
            motion_indices=motion_split.train_motion_indices,
            time_indices=base.SEEN_INTERPOLATION_TIME_INDICES,
            points_per_problem=config.eval_points_per_problem,
            base_seed=base.SEEN_INTERPOLATION_TEST_SOBOL_SEED,
            role="seen_motion_temporal_interpolation",
            physical=physical,
            include_explicit_train_points=False,
        ),
        "seen_motion_temporal_extrapolation": base.build_dataset_for_motion_times(
            lookup=lookup,
            motion_indices=motion_split.train_motion_indices,
            time_indices=base.SEEN_EXTRAPOLATION_TIME_INDICES,
            points_per_problem=config.eval_points_per_problem,
            base_seed=base.SEEN_EXTRAPOLATION_TEST_SOBOL_SEED,
            role="seen_motion_temporal_extrapolation",
            physical=physical,
            include_explicit_train_points=False,
        ),
        "unseen_id_test": base.build_dataset_for_motion_times(
            lookup=lookup,
            motion_indices=motion_split.id_test_motion_indices,
            time_indices=base.UNSEEN_TEST_TIME_INDICES,
            points_per_problem=config.eval_points_per_problem,
            base_seed=base.UNSEEN_ID_TEST_SOBOL_SEED,
            role="unseen_id_test",
            physical=physical,
            include_explicit_train_points=False,
        ),
        "ood_test": base.build_dataset_for_motion_times(
            lookup=lookup,
            motion_indices=motion_split.ood_test_motion_indices,
            time_indices=base.UNSEEN_TEST_TIME_INDICES,
            points_per_problem=config.eval_points_per_problem,
            base_seed=base.OOD_TEST_SOBOL_SEED,
            role="ood_test",
            physical=physical,
            include_explicit_train_points=False,
        ),
    }


def old_model_dir_name(activation: str) -> str:
    return f"activation_{activation}_depth_01_width_256_no_bias"


def nonlinear_model_dir_name(activation: str) -> str:
    return f"history_input_default_init_activation_{activation}_depth_01_width_256_no_bias"


def load_nonlinear_model(
    base: ModuleType,
    *,
    activation: str,
    model_root: Path,
    device: torch.device,
    residual_length_scale: float,
) -> nn.Module:
    spec = base.ModelSpec(activation=activation, depth=1, width=256, use_bias=False)
    model = base.MLPOptimizer(residual_length_scale, spec).to(device)
    state_path = model_root / nonlinear_model_dir_name(activation) / "best_validation_model_state_dict.pt"
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


class FullStateModelAdapter(nn.Module):
    """Expose a 75D full-state optimizer through the original 69D eval API."""

    def __init__(self, full_model: nn.Module, full: ModuleType) -> None:
        super().__init__()
        self.full_model = full_model
        self.full = full

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        optimizer_state: Any,
        *,
        physical: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_full = self.full.project_fixed_vertices(
            self.full.full_state_from_free_state(y, physical),
            physical,
        )
        q_full = self.full.project_fixed_vertices(
            self.full.full_state_from_free_state(q, physical),
            physical,
        )
        previous_residual_full = self.full.full_vector_from_free_vector(
            optimizer_state.previous_residual
        )
        previous_update_full = self.full.full_vector_from_free_vector(
            optimizer_state.previous_update
        )
        delta_full, current_residual_full = self.full_model(
            y_full,
            q_full,
            masses,
            physical=physical,
            previous_residual=previous_residual_full,
            previous_update=previous_update_full,
        )
        return (
            self.full.free_state_from_full_state(delta_full),
            self.full.free_state_from_full_state(current_residual_full),
        )


def load_old_full_state_model(
    full: ModuleType,
    *,
    activation: str,
    model_root: Path,
    device: torch.device,
    residual_length_scale: float,
) -> nn.Module:
    spec = full.ModelSpec(activation=activation, depth=1, width=256, use_bias=False)
    model = full.MLPOptimizer(residual_length_scale, spec).to(device)
    checkpoint_path = model_root / old_model_dir_name(activation) / "best_validation_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return FullStateModelAdapter(model, full).to(device)


def evaluate_model(
    base: ModuleType,
    *,
    model: nn.Module,
    dataset: Any,
    physical: Any,
    config: Any,
    device: torch.device,
) -> dict[str, Any]:
    return base.evaluate_solver_on_dataset(
        solver="learned",
        model=model,
        dataset_cpu=dataset,
        physical=physical,
        steps=config.evaluation_steps,
        batch_size=config.evaluation_batch_size,
        report_steps=config.report_steps,
        device=device,
    )


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "final_residual_mean",
        "final_residual_median",
        "final_residual_p95",
        "final_residual_max",
        "final_residual_num_nonfinite",
        "worst_motion_final_residual_p95",
        "worst_motion_final_residual_p95_motion_index",
        "worst_motion_final_residual_max",
        "worst_motion_final_residual_max_motion_index",
        "final_exact_error_mean",
        "final_exact_error_p95",
        "final_exact_error_max",
        "elapsed_seconds",
        "seconds_per_point_per_iteration",
        "num_points",
        "num_motions",
    ]
    return {key: metrics.get(key) for key in keys}


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "activation",
        "family",
        "model_name",
        "checkpoint",
        "num_points",
        "num_motions",
        "final_residual_mean",
        "final_residual_p95",
        "final_residual_max",
        "final_residual_num_nonfinite",
        "worst_motion_final_residual_p95",
        "worst_motion_final_residual_p95_motion_index",
        "worst_motion_final_residual_max",
        "worst_motion_final_residual_max_motion_index",
        "final_exact_error_p95",
        "final_exact_error_max",
        "elapsed_seconds",
        "seconds_per_point_per_iteration",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare old 500-step full-state models on the nonlinear ablation test sets."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--old-model-root", type=Path, default=OLD_MODEL_ROOT)
    parser.add_argument("--nonlinear-model-root", type=Path, default=NONLINEAR_MODEL_ROOT)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-points-per-problem", type=int, default=128)
    parser.add_argument("--total-time-steps", type=int, default=100)
    parser.add_argument("--residual-length-scale", type=float, default=5e-2)
    parser.add_argument("--activations", nargs="+", default=list(ACTIVATIONS), choices=list(ACTIVATIONS))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=[
            "seen_motion_temporal_interpolation",
            "seen_motion_temporal_extrapolation",
            "unseen_id_test",
            "ood_test",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_module("nonlinear_history_eval", NONLINEAR_SCRIPT)
    full = load_module("cloth500_full_state_model", CLOTH03_SCRIPT)

    device = torch.device(args.device)
    base.validate_device(device)
    physical = base.default_physical_config()
    config = make_eval_config(base, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building nonlinear test datasets for splits: {', '.join(args.splits)}")
    all_datasets = build_test_datasets(base, config, physical)
    datasets = {name: all_datasets[name] for name in args.splits}

    raw_results: dict[str, Any] = {
        "settings": {
            "device": str(device),
            "steps": config.evaluation_steps,
            "batch_size": config.evaluation_batch_size,
            "eval_points_per_problem": config.eval_points_per_problem,
            "splits": list(args.splits),
            "activations": list(args.activations),
            "old_model_root": str(args.old_model_root),
            "nonlinear_model_root": str(args.nonlinear_model_root),
            "nonlinear_eval_script": str(NONLINEAR_SCRIPT),
            "full_state_model_script": str(CLOTH03_SCRIPT),
        },
        "results": {},
    }
    summary_rows: list[dict[str, Any]] = []

    for activation in args.activations:
        print(f"Loading activation={activation} models")
        models = {
            "old_500step_full_state": {
                "model_name": old_model_dir_name(activation),
                "checkpoint": str(
                    args.old_model_root
                    / old_model_dir_name(activation)
                    / "best_validation_model.pt"
                ),
                "model": load_old_full_state_model(
                    full,
                    activation=activation,
                    model_root=args.old_model_root,
                    device=device,
                    residual_length_scale=args.residual_length_scale,
                ),
            },
            "nonlinear_history_input_default_init": {
                "model_name": nonlinear_model_dir_name(activation),
                "checkpoint": str(
                    args.nonlinear_model_root
                    / nonlinear_model_dir_name(activation)
                    / "best_validation_model_state_dict.pt"
                ),
                "model": load_nonlinear_model(
                    base,
                    activation=activation,
                    model_root=args.nonlinear_model_root,
                    device=device,
                    residual_length_scale=args.residual_length_scale,
                ),
            },
        }

        for family, record in models.items():
            raw_results["results"].setdefault(family, {}).setdefault(activation, {})
            for split_name, dataset in datasets.items():
                print(f"Evaluating {family}/{activation} on {split_name}")
                metrics = evaluate_model(
                    base,
                    model=record["model"],
                    dataset=dataset,
                    physical=physical,
                    config=config,
                    device=device,
                )
                raw_results["results"][family][activation][split_name] = metrics
                row = {
                    "split": split_name,
                    "activation": activation,
                    "family": family,
                    "model_name": record["model_name"],
                    "checkpoint": record["checkpoint"],
                    **compact_metrics(metrics),
                }
                summary_rows.append(row)

    save_json(raw_results, args.output_dir / "evaluation_metrics.json")
    write_summary_csv(summary_rows, args.output_dir / "comparison_summary.csv")
    save_json({"rows": summary_rows}, args.output_dir / "comparison_summary.json")
    print(f"Saved metrics to {args.output_dir / 'evaluation_metrics.json'}")
    print(f"Saved summary to {args.output_dir / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
