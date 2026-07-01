#!/usr/bin/env python3
"""
Compare the trained single-motion and multi-motion learned optimizers with
fixed-step gradient descent and full Newton on the spring test sets.

Expected directory layout
-------------------------
multi_motion_spring/
├── multi_motion_spring_optimizer.py
├── compare_single_multi_gd_newton.py
└── multi_motion_spring_optimizer/
    ├── runtime_config.json
    ├── generated_datasets.pt                         # optional
    ├── multi_motion/
    │   └── best_validation_model_state_dict.pt
    └── single_motion_equal_budget_baseline/
        └── best_validation_model_state_dict.pt

The script:
1. Loads the two validation-selected learned-optimizer checkpoints.
2. Loads generated_datasets.pt when available, otherwise deterministically
   rebuilds the validation and four main test datasets with the original
   script's functions and Sobol seeds.
3. Chooses one global raw-gradient-descent step size on the validation set.
4. Runs Single, Multi, Gradient Descent, and full Newton for a fixed number
   of iterations from exactly the same initial states.
5. Plots residual p95 versus iteration on all four main test sets.
6. Defines the hardest test set by the largest iteration-0 residual p95,
   selects a deterministic random problem/sample from it, and plots all four
   solver trajectories on a common two-dimensional PCA slice of the true
   six-dimensional variational energy.

The exact solution is used only for metrics, validation selection, and plot
centering. It is not used in any solver update.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch


MAIN_TEST_DATASETS = (
    "seen_motion_temporal_interpolation",
    "seen_motion_temporal_extrapolation",
    "unseen_id_test",
    "ood_test",
)

SOLVER_ORDER = ("single", "multi", "gradient_descent", "newton")
SOLVER_DISPLAY_NAMES = {
    "single": "Single MLP",
    "multi": "Multi MLP",
    "gradient_descent": "Gradient Descent",
    "newton": "Full Newton",
}

PLOT_FLOOR = 1e-16
DEFAULT_RANDOM_SEED = 20260627


# ============================================================
# 1. Command line and original-module loading
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Single MLP, Multi MLP, validation-selected gradient "
            "descent, and full Newton on the four spring test sets."
        )
    )
    parser.add_argument(
        "--original-script",
        type=Path,
        default=None,
        help=(
            "Path to multi_motion_spring_optimizer.py. By default, use the "
            "file with that name next to this comparison script."
        ),
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing runtime_config.json and checkpoint folders. "
            "Default: <original-script parent>/multi_motion_spring_optimizer."
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--alpha-log10-min", type=float, default=-8.0)
    parser.add_argument("--alpha-log10-max", type=float, default=-3.5)
    parser.add_argument("--alpha-count", type=int, default=37)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--contour-grid-size", type=int, default=220)
    parser.add_argument(
        "--force-rebuild-datasets",
        action="store_true",
        help="Ignore generated_datasets.pt and rebuild from runtime_config.json.",
    )
    parser.add_argument(
        "--skip-contour",
        action="store_true",
        help="Skip the hardest-problem energy contour plot.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.alpha_count < 2:
        raise ValueError("--alpha-count must be at least 2.")
    if args.alpha_log10_max <= args.alpha_log10_min:
        raise ValueError("--alpha-log10-max must exceed --alpha-log10-min.")
    if args.contour_grid_size < 50:
        raise ValueError("--contour-grid-size must be at least 50.")


def load_original_module(path: Path) -> ModuleType:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Original script not found: {path}")
    module_name = "_multi_motion_spring_optimizer_for_comparison"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    comparison_script_dir = Path(__file__).resolve().parent
    original_script = (
        args.original_script.resolve()
        if args.original_script is not None
        else comparison_script_dir / "multi_motion_spring_optimizer.py"
    )
    experiment_dir = (
        args.experiment_dir.resolve()
        if args.experiment_dir is not None
        else original_script.parent / original_script.stem
    )
    output_dir = experiment_dir / "solver_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    return original_script, experiment_dir, output_dir


# ============================================================
# 2. JSON, checkpoint, and dataset loading
# ============================================================


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(data), file, indent=2, ensure_ascii=False)


def torch_load_compat(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool | None = None,
) -> Any:
    kwargs: dict[str, Any] = {"map_location": map_location}
    if weights_only is not None:
        kwargs["weights_only"] = weights_only
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, **kwargs)


def load_configs(
    original: ModuleType,
    experiment_dir: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    config_path = experiment_dir / "runtime_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing {config_path}. Run the training script first."
        )
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    runtime_raw = dict(payload["runtime_config"])
    runtime_raw["report_steps"] = tuple(int(x) for x in runtime_raw["report_steps"])
    runtime = original.RuntimeConfig(**runtime_raw)

    physical_raw = dict(payload["physical_config"])
    for name in ("p1_0", "p2_0", "v1_0", "v2_0"):
        physical_raw[name] = tuple(float(x) for x in physical_raw[name])
    physical = original.PhysicalConfig(**physical_raw)
    return runtime, physical, payload


def dataset_from_serialized(original: ModuleType, record: dict[str, Any]) -> Any:
    return original.DatasetBundle(
        initial_y=record["initial_y"].detach().cpu().to(original.TORCH_DTYPE),
        q=record["q"].detach().cpu().to(original.TORCH_DTYPE),
        masses=record["masses"].detach().cpu().to(original.TORCH_DTYPE),
        exact_y=record["exact_y"].detach().cpu().to(original.TORCH_DTYPE),
        problem_index=record["problem_index"].detach().cpu().to(torch.long),
        motion_index=record["motion_index"].detach().cpu().to(torch.long),
        time_index=record["time_index"].detach().cpu().to(torch.long),
        metadata=dict(record.get("metadata", {})),
    )


def rebuild_required_datasets(
    original: ModuleType,
    runtime: Any,
    physical: Any,
) -> tuple[Any, dict[str, Any], list[Any], list[Any]]:
    motions, motion_split = original.build_motion_catalogue(physical)
    problems = original.generate_all_reference_sequences(
        physical=physical,
        motions=motions,
        total_steps=runtime.total_time_steps,
        sampling_radius_min=runtime.sampling_radius_min,
        sampling_radius_max=runtime.sampling_radius_max,
    )

    validation_indices = original.problem_indices_for(
        motion_split.validation_motion_indices,
        original.VALIDATION_TIME_INDICES,
        runtime.total_time_steps,
    )
    seen_interpolation_indices = original.problem_indices_for(
        motion_split.train_motion_indices,
        original.SEEN_INTERPOLATION_TIME_INDICES,
        runtime.total_time_steps,
    )
    seen_extrapolation_indices = original.problem_indices_for(
        motion_split.train_motion_indices,
        original.SEEN_EXTRAPOLATION_TIME_INDICES,
        runtime.total_time_steps,
    )
    unseen_id_indices = original.problem_indices_for(
        motion_split.id_test_motion_indices,
        original.UNSEEN_TEST_TIME_INDICES,
        runtime.total_time_steps,
    )
    ood_indices = original.problem_indices_for(
        motion_split.ood_test_motion_indices,
        original.UNSEEN_TEST_TIME_INDICES,
        runtime.total_time_steps,
    )

    validation = original.build_dataset_for_problem_indices(
        problems=problems,
        indices=validation_indices,
        points_per_problem=runtime.eval_points_per_problem,
        base_seed=original.VALIDATION_SOBOL_SEED,
        role="unseen_motion_validation",
        include_explicit_train_points=False,
    )
    tests = {
        "seen_motion_temporal_interpolation":
            original.build_dataset_for_problem_indices(
                problems=problems,
                indices=seen_interpolation_indices,
                points_per_problem=runtime.eval_points_per_problem,
                base_seed=original.SEEN_INTERPOLATION_TEST_SOBOL_SEED,
                role="seen_motion_temporal_interpolation",
                include_explicit_train_points=False,
            ),
        "seen_motion_temporal_extrapolation":
            original.build_dataset_for_problem_indices(
                problems=problems,
                indices=seen_extrapolation_indices,
                points_per_problem=runtime.eval_points_per_problem,
                base_seed=original.SEEN_EXTRAPOLATION_TEST_SOBOL_SEED,
                role="seen_motion_temporal_extrapolation",
                include_explicit_train_points=False,
            ),
        "unseen_id_test":
            original.build_dataset_for_problem_indices(
                problems=problems,
                indices=unseen_id_indices,
                points_per_problem=runtime.eval_points_per_problem,
                base_seed=original.UNSEEN_ID_TEST_SOBOL_SEED,
                role="unseen_id_test",
                include_explicit_train_points=False,
            ),
        "ood_test":
            original.build_dataset_for_problem_indices(
                problems=problems,
                indices=ood_indices,
                points_per_problem=runtime.eval_points_per_problem,
                base_seed=original.OOD_TEST_SOBOL_SEED,
                role="ood_test",
                include_explicit_train_points=False,
            ),
    }
    return validation, tests, motions, problems


def load_or_rebuild_datasets(
    original: ModuleType,
    experiment_dir: Path,
    runtime: Any,
    physical: Any,
    *,
    force_rebuild: bool,
) -> tuple[Any, dict[str, Any], list[Any] | None, list[Any] | None, str]:
    datasets_path = experiment_dir / "generated_datasets.pt"
    if datasets_path.is_file() and not force_rebuild:
        print(f"Loading saved datasets: {datasets_path}")
        payload = torch_load_compat(
            datasets_path, map_location="cpu", weights_only=False
        )
        validation = dataset_from_serialized(original, payload["validation"])
        tests = {
            name: dataset_from_serialized(original, payload[name])
            for name in MAIN_TEST_DATASETS
        }
        return validation, tests, None, None, "generated_datasets.pt"

    print("Saved datasets unavailable or ignored; rebuilding deterministically.")
    validation, tests, motions, problems = rebuild_required_datasets(
        original, runtime, physical
    )
    return validation, tests, motions, problems, "deterministic_rebuild"


def load_model(
    original: ModuleType,
    checkpoint_path: Path,
    residual_length_scale: float,
    device: torch.device,
) -> Any:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    model = original.MLPOptimizer(residual_length_scale)
    state_dict = torch_load_compat(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=original.TORCH_DTYPE)
    model.eval()
    return model


# ============================================================
# 3. Dataset slicing and safe solver updates
# ============================================================


def slice_dataset(
    original: ModuleType,
    dataset: Any,
    start: int,
    end: int,
    device: torch.device,
) -> Any:
    return original.DatasetBundle(
        initial_y=dataset.initial_y[start:end].to(
            device=device, dtype=original.TORCH_DTYPE
        ),
        q=dataset.q[start:end].to(device=device, dtype=original.TORCH_DTYPE),
        masses=dataset.masses[start:end].to(
            device=device, dtype=original.TORCH_DTYPE
        ),
        exact_y=dataset.exact_y[start:end].to(
            device=device, dtype=original.TORCH_DTYPE
        ),
        problem_index=dataset.problem_index[start:end].to(
            device=device, dtype=torch.long
        ),
        motion_index=dataset.motion_index[start:end].to(
            device=device, dtype=torch.long
        ),
        time_index=dataset.time_index[start:end].to(
            device=device, dtype=torch.long
        ),
        metadata={},
    )


def one_sample_dataset(
    original: ModuleType,
    dataset: Any,
    index: int,
    device: torch.device,
) -> Any:
    return slice_dataset(original, dataset, index, index + 1, device)


def safe_update(
    original: ModuleType,
    solver_name: str,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: Any,
    *,
    model: Any | None = None,
    alpha: float | None = None,
) -> torch.Tensor:
    finite_input = torch.isfinite(y).all(dim=-1)
    result = torch.full_like(y, float("nan"))
    if not bool(torch.any(finite_input)):
        return result

    indices = torch.nonzero(finite_input, as_tuple=False).squeeze(-1)
    y_active = y[indices]
    q_active = q[indices]
    masses_active = masses[indices]

    if solver_name in ("single", "multi"):
        if model is None:
            raise ValueError(f"{solver_name} requires a model.")
        with torch.no_grad():
            delta = model(
                y_active,
                q_active,
                masses_active,
                physical=physical,
            )
            candidate = y_active + delta
    elif solver_name == "gradient_descent":
        if alpha is None or alpha <= 0.0:
            raise ValueError("Gradient descent requires a positive alpha.")
        gradient = original.stationarity_residual(
            y_active,
            q_active,
            masses_active,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        )
        candidate = y_active - float(alpha) * gradient
    elif solver_name == "newton":
        try:
            candidate, _ = original.apply_newton_update(
                y_active, q_active, masses_active, physical
            )
        except RuntimeError:
            candidate = torch.full_like(y_active, float("nan"))
            for local_index in range(y_active.shape[0]):
                try:
                    updated, _ = original.apply_newton_update(
                        y_active[local_index:local_index + 1],
                        q_active[local_index:local_index + 1],
                        masses_active[local_index:local_index + 1],
                        physical,
                    )
                    candidate[local_index] = updated[0]
                except RuntimeError:
                    pass
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    finite_candidate = torch.isfinite(candidate).all(dim=-1)
    if bool(torch.any(finite_candidate)):
        result[indices[finite_candidate]] = candidate[finite_candidate]
    return result


# ============================================================
# 4. Statistics and complete rollout evaluation
# ============================================================


def finite_statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result: dict[str, float | int] = {
        "num_nonfinite": int(values.size - finite.size),
        "mean": float("nan"),
        "median": float("nan"),
        "p95": float("nan"),
        "max": float("nan"),
    }
    if finite.size:
        result.update(
            {
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "p95": float(np.percentile(finite, 95)),
                "max": float(np.max(finite)),
            }
        )
    return result


def metric_curve_statistics(values: np.ndarray) -> dict[str, list[Any]]:
    if values.ndim != 2:
        raise ValueError(f"Expected [num_points, steps+1], got {values.shape}")
    output = {
        "mean_by_step": [],
        "median_by_step": [],
        "p95_by_step": [],
        "max_by_step": [],
        "num_nonfinite_by_step": [],
    }
    for step in range(values.shape[1]):
        stats = finite_statistics(values[:, step])
        for stat_name in ("mean", "median", "p95", "max", "num_nonfinite"):
            output[f"{stat_name}_by_step"].append(stats[stat_name])
    return output


def compute_metrics_at_state(
    original: ModuleType,
    y: torch.Tensor,
    batch: Any,
    exact_energy: torch.Tensor,
    physical: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = original.stationarity_residual_norm(
        y,
        batch.q,
        batch.masses,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    energy = original.variational_energy(
        y,
        batch.q,
        batch.masses,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )
    energy_gap = torch.clamp(energy - exact_energy, min=0.0)
    exact_error = torch.linalg.vector_norm(y - batch.exact_y, dim=-1)
    return residual, energy_gap, exact_error


@torch.no_grad()
def evaluate_solver_curve(
    original: ModuleType,
    *,
    solver_name: str,
    dataset_name: str,
    dataset: Any,
    physical: Any,
    steps: int,
    batch_size: int,
    device: torch.device,
    model: Any | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    residual_chunks: list[torch.Tensor] = []
    gap_chunks: list[torch.Tensor] = []
    error_chunks: list[torch.Tensor] = []
    motion_chunks: list[torch.Tensor] = []
    problem_chunks: list[torch.Tensor] = []
    time_chunks: list[torch.Tensor] = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = slice_dataset(original, dataset, start, end, device)
        y = batch.initial_y.clone()
        exact_energy = original.variational_energy(
            batch.exact_y,
            batch.q,
            batch.masses,
            g=physical.g,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        )

        residual_steps: list[torch.Tensor] = []
        gap_steps: list[torch.Tensor] = []
        error_steps: list[torch.Tensor] = []
        for iteration in range(steps + 1):
            residual, gap, error = compute_metrics_at_state(
                original, y, batch, exact_energy, physical
            )
            residual_steps.append(residual.detach().cpu())
            gap_steps.append(gap.detach().cpu())
            error_steps.append(error.detach().cpu())
            if iteration < steps:
                y = safe_update(
                    original,
                    solver_name,
                    y,
                    batch.q,
                    batch.masses,
                    physical,
                    model=model,
                    alpha=alpha,
                )

        residual_chunks.append(torch.stack(residual_steps, dim=1))
        gap_chunks.append(torch.stack(gap_steps, dim=1))
        error_chunks.append(torch.stack(error_steps, dim=1))
        motion_chunks.append(batch.motion_index.detach().cpu())
        problem_chunks.append(batch.problem_index.detach().cpu())
        time_chunks.append(batch.time_index.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    residual_values = torch.cat(residual_chunks, dim=0).numpy().astype(float)
    gap_values = torch.cat(gap_chunks, dim=0).numpy().astype(float)
    error_values = torch.cat(error_chunks, dim=0).numpy().astype(float)
    motion_indices = torch.cat(motion_chunks).numpy().astype(int)
    problem_indices = torch.cat(problem_chunks).numpy().astype(int)
    time_indices = torch.cat(time_chunks).numpy().astype(int)

    metrics = {
        "residual": metric_curve_statistics(residual_values),
        "energy_gap": metric_curve_statistics(gap_values),
        "exact_error": metric_curve_statistics(error_values),
    }

    per_motion_final: dict[str, Any] = {}
    for motion_index in sorted(np.unique(motion_indices).tolist()):
        mask = motion_indices == motion_index
        per_motion_final[str(motion_index)] = {
            "motion_index": int(motion_index),
            "num_points": int(mask.sum()),
            "residual": finite_statistics(residual_values[mask, -1]),
            "energy_gap": finite_statistics(gap_values[mask, -1]),
            "exact_error": finite_statistics(error_values[mask, -1]),
        }

    return {
        "solver": solver_name,
        "solver_display_name": SOLVER_DISPLAY_NAMES[solver_name],
        "dataset": dataset_name,
        "steps": steps,
        "num_points": len(dataset),
        "elapsed_seconds": elapsed,
        "seconds_per_point_per_iteration": elapsed / max(len(dataset) * steps, 1),
        "gradient_descent_alpha": alpha if solver_name == "gradient_descent" else None,
        "metrics": metrics,
        "per_motion_final": per_motion_final,
        "problem_index_min": int(problem_indices.min()),
        "problem_index_max": int(problem_indices.max()),
        "time_index_min": int(time_indices.min()),
        "time_index_max": int(time_indices.max()),
    }


# ============================================================
# 5. Validation-only gradient-descent step-size selection
# ============================================================


@torch.no_grad()
def evaluate_gradient_descent_final(
    original: ModuleType,
    *,
    dataset: Any,
    physical: Any,
    alpha: float,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    residual_chunks: list[torch.Tensor] = []
    gap_chunks: list[torch.Tensor] = []
    error_chunks: list[torch.Tensor] = []
    motion_chunks: list[torch.Tensor] = []

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = slice_dataset(original, dataset, start, end, device)
        y = batch.initial_y.clone()
        for _ in range(steps):
            y = safe_update(
                original,
                "gradient_descent",
                y,
                batch.q,
                batch.masses,
                physical,
                alpha=alpha,
            )

        exact_energy = original.variational_energy(
            batch.exact_y,
            batch.q,
            batch.masses,
            g=physical.g,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        )
        residual, gap, error = compute_metrics_at_state(
            original, y, batch, exact_energy, physical
        )
        residual_chunks.append(residual.detach().cpu())
        gap_chunks.append(gap.detach().cpu())
        error_chunks.append(error.detach().cpu())
        motion_chunks.append(batch.motion_index.detach().cpu())

    residual_values = torch.cat(residual_chunks).numpy().astype(float)
    gap_values = torch.cat(gap_chunks).numpy().astype(float)
    error_values = torch.cat(error_chunks).numpy().astype(float)
    motion_indices = torch.cat(motion_chunks).numpy().astype(int)

    pooled_residual = finite_statistics(residual_values)
    pooled_gap = finite_statistics(gap_values)
    pooled_error = finite_statistics(error_values)

    per_motion: dict[str, Any] = {}
    worst_motion_residual_p95 = float("-inf")
    worst_motion_error_p95 = float("-inf")
    for motion_index in sorted(np.unique(motion_indices).tolist()):
        mask = motion_indices == motion_index
        residual_stats = finite_statistics(residual_values[mask])
        gap_stats = finite_statistics(gap_values[mask])
        error_stats = finite_statistics(error_values[mask])
        per_motion[str(motion_index)] = {
            "residual": residual_stats,
            "energy_gap": gap_stats,
            "exact_error": error_stats,
        }
        residual_p95 = float(residual_stats["p95"])
        error_p95 = float(error_stats["p95"])
        if math.isfinite(residual_p95):
            worst_motion_residual_p95 = max(
                worst_motion_residual_p95, residual_p95
            )
        if math.isfinite(error_p95):
            worst_motion_error_p95 = max(worst_motion_error_p95, error_p95)

    if not math.isfinite(worst_motion_residual_p95):
        worst_motion_residual_p95 = float("inf")
    if not math.isfinite(worst_motion_error_p95):
        worst_motion_error_p95 = float("inf")

    def selection_value(value: float | int) -> float:
        value_float = float(value)
        return value_float if math.isfinite(value_float) else float("inf")

    selection_key = (
        selection_value(pooled_residual["num_nonfinite"]),
        selection_value(worst_motion_residual_p95),
        selection_value(pooled_residual["p95"]),
        selection_value(worst_motion_error_p95),
        selection_value(pooled_error["p95"]),
        selection_value(pooled_gap["p95"]),
    )
    return {
        "alpha": float(alpha),
        "selection_key": list(selection_key),
        "pooled": {
            "residual": pooled_residual,
            "energy_gap": pooled_gap,
            "exact_error": pooled_error,
        },
        "worst_motion_residual_p95": worst_motion_residual_p95,
        "worst_motion_exact_error_p95": worst_motion_error_p95,
        "per_motion": per_motion,
    }


def select_gradient_descent_alpha(
    original: ModuleType,
    *,
    validation: Any,
    physical: Any,
    candidates: Sequence[float],
    steps: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[float, list[dict[str, Any]]]:
    print(
        f"Selecting gradient-descent alpha on validation only "
        f"({len(candidates)} candidates, {steps} iterations each)..."
    )
    records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None

    for candidate_index, alpha in enumerate(candidates, start=1):
        record = evaluate_gradient_descent_final(
            original,
            dataset=validation,
            physical=physical,
            alpha=float(alpha),
            steps=steps,
            batch_size=batch_size,
            device=device,
        )
        records.append(record)
        key = tuple(float(x) for x in record["selection_key"])
        if best_key is None or key < best_key:
            best_key = key
            best_record = record
        print(
            f"  [{candidate_index:02d}/{len(candidates):02d}] "
            f"alpha={alpha:.6e} | "
            f"nonfinite={record['pooled']['residual']['num_nonfinite']} | "
            f"worst-motion residual p95="
            f"{record['worst_motion_residual_p95']:.6e} | "
            f"pooled residual p95="
            f"{record['pooled']['residual']['p95']:.6e}"
        )

    if best_record is None:
        raise RuntimeError("Gradient-descent alpha selection produced no result.")
    selected_alpha = float(best_record["alpha"])
    print(f"Selected gradient-descent alpha: {selected_alpha:.12e}")

    save_json(
        {
            "selection_dataset": "unseen_motion_validation",
            "uses_test_data": False,
            "iterations": steps,
            "selection_priority": [
                "pooled final residual nonfinite count",
                "worst validation-motion final residual p95",
                "pooled final residual p95",
                "worst validation-motion final exact-error p95",
                "pooled final exact-error p95",
                "pooled final energy-gap p95",
            ],
            "selected_alpha": selected_alpha,
            "selected_record": best_record,
            "all_candidates": records,
        },
        output_dir / "gradient_descent_validation_search.json",
    )
    save_gradient_search_csv(records, selected_alpha, output_dir)
    plot_gradient_search(records, selected_alpha, output_dir)
    return selected_alpha, records


def save_gradient_search_csv(
    records: Sequence[dict[str, Any]],
    selected_alpha: float,
    output_dir: Path,
) -> None:
    path = output_dir / "gradient_descent_validation_search.csv"
    fieldnames = [
        "alpha",
        "selected",
        "final_residual_num_nonfinite",
        "worst_motion_final_residual_p95",
        "pooled_final_residual_p95",
        "worst_motion_final_exact_error_p95",
        "pooled_final_exact_error_p95",
        "pooled_final_energy_gap_p95",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "alpha": record["alpha"],
                    "selected": math.isclose(
                        float(record["alpha"]),
                        float(selected_alpha),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    ),
                    "final_residual_num_nonfinite":
                        record["pooled"]["residual"]["num_nonfinite"],
                    "worst_motion_final_residual_p95":
                        record["worst_motion_residual_p95"],
                    "pooled_final_residual_p95":
                        record["pooled"]["residual"]["p95"],
                    "worst_motion_final_exact_error_p95":
                        record["worst_motion_exact_error_p95"],
                    "pooled_final_exact_error_p95":
                        record["pooled"]["exact_error"]["p95"],
                    "pooled_final_energy_gap_p95":
                        record["pooled"]["energy_gap"]["p95"],
                }
            )


def plot_gradient_search(
    records: Sequence[dict[str, Any]],
    selected_alpha: float,
    output_dir: Path,
) -> None:
    alphas = np.asarray([float(record["alpha"]) for record in records])
    pooled = np.asarray(
        [float(record["pooled"]["residual"]["p95"]) for record in records]
    )
    worst = np.asarray(
        [float(record["worst_motion_residual_p95"]) for record in records]
    )
    pooled[~np.isfinite(pooled)] = np.nan
    worst[~np.isfinite(worst)] = np.nan

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(alphas, pooled, marker="o", label="Pooled validation residual p95")
    ax.plot(alphas, worst, marker="s", label="Worst-motion residual p95")
    ax.axvline(
        selected_alpha,
        linestyle="--",
        linewidth=1.5,
        label=f"Selected alpha = {selected_alpha:.3e}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gradient-descent step size")
    ax.set_ylabel(f"Residual after validation rollout")
    ax.set_title("Validation-only gradient-descent step-size selection")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        output_dir / "gradient_descent_validation_search.png", dpi=220
    )
    plt.close(fig)


# ============================================================
# 6. Residual comparison plots and CSV export
# ============================================================


def positive_plot_curve(values: Sequence[float | int | None]) -> np.ndarray:
    array = np.asarray(
        [float("nan") if value is None else float(value) for value in values],
        dtype=float,
    )
    array[~np.isfinite(array)] = np.nan
    finite = np.isfinite(array)
    array[finite] = np.maximum(array[finite], PLOT_FLOOR)
    return array


def plot_residual_comparisons(
    all_results: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    for ax, dataset_name in zip(axes.flat, MAIN_TEST_DATASETS):
        for solver_name in SOLVER_ORDER:
            result = all_results[dataset_name][solver_name]
            curve = positive_plot_curve(
                result["metrics"]["residual"]["p95_by_step"]
            )
            ax.plot(
                np.arange(curve.size),
                curve,
                linewidth=2.0,
                label=SOLVER_DISPLAY_NAMES[solver_name],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Solver iteration")
        ax.set_ylabel("Residual p95")
        ax.set_title(dataset_name)
        ax.grid(True, which="both", alpha=0.3)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Residual decrease comparison on spring test sets", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    fig.savefig(output_dir / "residual_p95_vs_iteration.png", dpi=240)
    plt.close(fig)

    for dataset_name in MAIN_TEST_DATASETS:
        fig, ax = plt.subplots(figsize=(8.5, 6.0))
        for solver_name in SOLVER_ORDER:
            result = all_results[dataset_name][solver_name]
            curve = positive_plot_curve(
                result["metrics"]["residual"]["p95_by_step"]
            )
            ax.plot(
                np.arange(curve.size),
                curve,
                linewidth=2.2,
                label=SOLVER_DISPLAY_NAMES[solver_name],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Solver iteration")
        ax.set_ylabel("Residual p95")
        ax.set_title(dataset_name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{dataset_name}_residual_p95.png", dpi=220
        )
        plt.close(fig)


def save_comparison_csv(
    all_results: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> None:
    path = output_dir / "solver_comparison_metrics.csv"
    fields = [
        "dataset",
        "solver",
        "iteration",
        "residual_mean",
        "residual_median",
        "residual_p95",
        "residual_max",
        "residual_num_nonfinite",
        "energy_gap_mean",
        "energy_gap_median",
        "energy_gap_p95",
        "energy_gap_max",
        "energy_gap_num_nonfinite",
        "exact_error_mean",
        "exact_error_median",
        "exact_error_p95",
        "exact_error_max",
        "exact_error_num_nonfinite",
        "elapsed_seconds",
        "gradient_descent_alpha",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for dataset_name in MAIN_TEST_DATASETS:
            for solver_name in SOLVER_ORDER:
                result = all_results[dataset_name][solver_name]
                steps = int(result["steps"])
                for iteration in range(steps + 1):
                    row: dict[str, Any] = {
                        "dataset": dataset_name,
                        "solver": solver_name,
                        "iteration": iteration,
                        "elapsed_seconds": result["elapsed_seconds"],
                        "gradient_descent_alpha":
                            result["gradient_descent_alpha"],
                    }
                    for metric_name in (
                        "residual", "energy_gap", "exact_error"
                    ):
                        metric = result["metrics"][metric_name]
                        for stat_name in (
                            "mean", "median", "p95", "max", "num_nonfinite"
                        ):
                            row[f"{metric_name}_{stat_name}"] = metric[
                                f"{stat_name}_by_step"
                            ][iteration]
                    writer.writerow(row)


def print_selected_iterations(
    all_results: dict[str, dict[str, dict[str, Any]]],
    steps: int,
) -> None:
    report_iterations = sorted(
        set(iteration for iteration in (1, 5, 10, 50, steps)
            if 0 <= iteration <= steps)
    )
    print("\nResidual p95 summary")
    print("=" * 88)
    for dataset_name in MAIN_TEST_DATASETS:
        print(f"\n[{dataset_name}]")
        for solver_name in SOLVER_ORDER:
            curve = all_results[dataset_name][solver_name][
                "metrics"
            ]["residual"]["p95_by_step"]
            values = " | ".join(
                f"k={iteration}: {float(curve[iteration]):.6e}"
                for iteration in report_iterations
            )
            print(f"  {SOLVER_DISPLAY_NAMES[solver_name]:18s} | {values}")


# ============================================================
# 7. Hardest test set and energy-contour trajectory plot
# ============================================================


def select_hardest_dataset(
    all_results: dict[str, dict[str, dict[str, Any]]],
) -> tuple[str, dict[str, float]]:
    initial_p95 = {
        dataset_name: float(
            all_results[dataset_name]["multi"]["metrics"]["residual"][
                "p95_by_step"
            ][0]
        )
        for dataset_name in MAIN_TEST_DATASETS
    }
    finite_items = [
        (name, value)
        for name, value in initial_p95.items()
        if math.isfinite(value)
    ]
    if not finite_items:
        raise RuntimeError("No finite initial residual p95 is available.")
    hardest_name, _ = max(finite_items, key=lambda item: item[1])
    return hardest_name, initial_p95


def select_random_problem_sample(
    dataset: Any,
    seed: int,
) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    problem_values = dataset.problem_index.detach().cpu().numpy().astype(int)
    unique_problems = np.unique(problem_values)
    selected_problem = int(rng.choice(unique_problems))
    candidate_samples = np.flatnonzero(problem_values == selected_problem)
    selected_sample = int(rng.choice(candidate_samples))
    return selected_problem, selected_sample


@torch.no_grad()
def rollout_one_sample(
    original: ModuleType,
    *,
    dataset: Any,
    sample_index: int,
    physical: Any,
    steps: int,
    device: torch.device,
    single_model: Any,
    multi_model: Any,
    alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    batch = one_sample_dataset(original, dataset, sample_index, device)
    trajectories: dict[str, np.ndarray] = {}
    trajectory_metrics: dict[str, Any] = {}

    for solver_name, model in (
        ("single", single_model),
        ("multi", multi_model),
        ("gradient_descent", None),
        ("newton", None),
    ):
        y = batch.initial_y.clone()
        states = [y[0].detach().cpu().numpy().astype(float)]
        residuals: list[float] = []
        energy_gaps: list[float] = []
        exact_errors: list[float] = []
        exact_energy = original.variational_energy(
            batch.exact_y,
            batch.q,
            batch.masses,
            g=physical.g,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        )
        for iteration in range(steps + 1):
            residual, gap, error = compute_metrics_at_state(
                original, y, batch, exact_energy, physical
            )
            residuals.append(float(residual[0].item()))
            energy_gaps.append(float(gap[0].item()))
            exact_errors.append(float(error[0].item()))
            if iteration < steps:
                y = safe_update(
                    original,
                    solver_name,
                    y,
                    batch.q,
                    batch.masses,
                    physical,
                    model=model,
                    alpha=alpha,
                )
                states.append(y[0].detach().cpu().numpy().astype(float))
        trajectories[solver_name] = np.stack(states, axis=0)
        trajectory_metrics[solver_name] = {
            "residual_by_step": residuals,
            "energy_gap_by_step": energy_gaps,
            "exact_error_by_step": exact_errors,
        }

    sample_record = {
        "sample_index_in_dataset": int(sample_index),
        "problem_index": int(dataset.problem_index[sample_index].item()),
        "motion_index": int(dataset.motion_index[sample_index].item()),
        "time_index": int(dataset.time_index[sample_index].item()),
        "initial_y": dataset.initial_y[sample_index].tolist(),
        "q": dataset.q[sample_index].tolist(),
        "masses": dataset.masses[sample_index].tolist(),
        "exact_y": dataset.exact_y[sample_index].tolist(),
        "trajectory_metrics": trajectory_metrics,
    }
    motion_records = dataset.metadata.get("motion_records", {})
    motion_record = motion_records.get(
        str(sample_record["motion_index"]),
        motion_records.get(sample_record["motion_index"], None),
    )
    if motion_record is not None:
        sample_record["motion_metadata"] = motion_record
    return trajectories, sample_record


def make_two_dimensional_basis(
    trajectories: dict[str, np.ndarray],
    exact_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_displacements: list[np.ndarray] = []
    for trajectory in trajectories.values():
        displacement = trajectory - exact_y.reshape(1, 6)
        finite_rows = np.isfinite(displacement).all(axis=1)
        finite_displacements.append(displacement[finite_rows])
    stacked = np.concatenate(finite_displacements, axis=0)
    if stacked.shape[0] == 0:
        raise RuntimeError("All selected trajectories are non-finite.")

    _, singular_values, vh = np.linalg.svd(stacked, full_matrices=False)
    basis_vectors: list[np.ndarray] = []
    for row in vh:
        candidate = row.copy()
        for existing in basis_vectors:
            candidate -= np.dot(candidate, existing) * existing
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-12:
            basis_vectors.append(candidate / norm)
        if len(basis_vectors) == 2:
            break

    for axis_index in range(6):
        if len(basis_vectors) == 2:
            break
        candidate = np.zeros(6, dtype=float)
        candidate[axis_index] = 1.0
        for existing in basis_vectors:
            candidate -= np.dot(candidate, existing) * existing
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-12:
            basis_vectors.append(candidate / norm)

    if len(basis_vectors) < 2:
        raise RuntimeError("Could not construct a two-dimensional basis.")
    basis = np.stack(basis_vectors[:2], axis=0)
    return basis, singular_values


def evaluate_energy_grid(
    original: ModuleType,
    *,
    exact_y: np.ndarray,
    q: np.ndarray,
    masses: np.ndarray,
    basis: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    physical: Any,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    coordinates = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    states = (
        exact_y.reshape(1, 6)
        + coordinates[:, 0:1] * basis[0].reshape(1, 6)
        + coordinates[:, 1:2] * basis[1].reshape(1, 6)
    )
    exact_tensor = torch.tensor(
        exact_y.reshape(1, 6),
        dtype=original.TORCH_DTYPE,
        device=device,
    )
    q_tensor = torch.tensor(
        q.reshape(1, 6), dtype=original.TORCH_DTYPE, device=device
    )
    masses_tensor = torch.tensor(
        masses.reshape(1, 2), dtype=original.TORCH_DTYPE, device=device
    )
    exact_energy = original.variational_energy(
        exact_tensor,
        q_tensor,
        masses_tensor,
        g=physical.g,
        dt=physical.dt,
        spring_k=physical.spring_k,
        rest_length=physical.rest_length,
    )[0]

    gap_chunks: list[torch.Tensor] = []
    for start in range(0, states.shape[0], batch_size):
        end = min(start + batch_size, states.shape[0])
        state_batch = torch.tensor(
            states[start:end],
            dtype=original.TORCH_DTYPE,
            device=device,
        )
        q_batch = q_tensor.expand(end - start, -1)
        mass_batch = masses_tensor.expand(end - start, -1)
        energy = original.variational_energy(
            state_batch,
            q_batch,
            mass_batch,
            g=physical.g,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        )
        gap_chunks.append(torch.clamp(energy - exact_energy, min=0.0).cpu())
    gaps = torch.cat(gap_chunks).numpy().astype(float)
    return gaps.reshape(grid_x.shape)


def plot_hard_problem_energy_contour(
    original: ModuleType,
    *,
    trajectories: dict[str, np.ndarray],
    sample_record: dict[str, Any],
    hardest_dataset_name: str,
    alpha: float,
    physical: Any,
    device: torch.device,
    grid_size: int,
    batch_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    exact_y = np.asarray(sample_record["exact_y"], dtype=float)
    q = np.asarray(sample_record["q"], dtype=float)
    masses = np.asarray(sample_record["masses"], dtype=float)
    basis, singular_values = make_two_dimensional_basis(
        trajectories, exact_y
    )

    projected: dict[str, np.ndarray] = {}
    all_coordinates: list[np.ndarray] = [np.zeros((1, 2), dtype=float)]
    for solver_name, trajectory in trajectories.items():
        displacement = trajectory - exact_y.reshape(1, 6)
        coordinates = displacement @ basis.T
        projected[solver_name] = coordinates
        finite = coordinates[np.isfinite(coordinates).all(axis=1)]
        if finite.size:
            all_coordinates.append(finite)
    combined = np.concatenate(all_coordinates, axis=0)

    x_min, y_min = np.min(combined, axis=0)
    x_max, y_max = np.max(combined, axis=0)
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_min -= 0.15 * x_span
    x_max += 0.15 * x_span
    y_min -= 0.15 * y_span
    y_max += 0.15 * y_span

    x_values = np.linspace(x_min, x_max, grid_size)
    y_values = np.linspace(y_min, y_max, grid_size)
    energy_gap = evaluate_energy_grid(
        original,
        exact_y=exact_y,
        q=q,
        masses=masses,
        basis=basis,
        x_values=x_values,
        y_values=y_values,
        physical=physical,
        device=device,
        batch_size=batch_size,
    )

    finite_positive = energy_gap[
        np.isfinite(energy_gap) & (energy_gap > 0.0)
    ]
    if finite_positive.size:
        max_gap = float(np.max(finite_positive))
        min_gap = max(
            float(np.min(finite_positive)),
            max_gap * 1e-12,
            PLOT_FLOOR,
        )
    else:
        min_gap, max_gap = PLOT_FLOOR, PLOT_FLOOR * 10.0
    if max_gap <= min_gap:
        max_gap = min_gap * 10.0
    levels = np.geomspace(min_gap, max_gap, 32)
    plotted_gap = np.maximum(energy_gap, min_gap)

    fig, ax = plt.subplots(figsize=(10.0, 8.0))
    contour = ax.contourf(
        x_values,
        y_values,
        plotted_gap,
        levels=levels,
        norm=LogNorm(vmin=min_gap, vmax=max_gap),
        cmap="viridis",
    )
    ax.contour(
        x_values,
        y_values,
        plotted_gap,
        levels=levels[::3],
        norm=LogNorm(vmin=min_gap, vmax=max_gap),
        colors="black",
        linewidths=0.35,
        alpha=0.45,
    )
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label(r"$E(y)-E(y^*)$")

    styles = {
        "single": dict(marker="o", linewidth=2.0),
        "multi": dict(marker="s", linewidth=2.0),
        "gradient_descent": dict(marker="^", linewidth=2.0),
        "newton": dict(marker="D", linewidth=2.0),
    }
    annotate_steps = sorted(
        set(step for step in (0, 1, 5, 10, 50)
            if step < next(iter(projected.values())).shape[0])
    )
    for solver_name in SOLVER_ORDER:
        coordinates = projected[solver_name]
        finite_mask = np.isfinite(coordinates).all(axis=1)
        finite_indices = np.flatnonzero(finite_mask)
        if finite_indices.size == 0:
            continue
        # Keep only the finite prefix so a divergence does not connect across NaNs.
        prefix_end = int(finite_indices[-1]) + 1
        prefix = coordinates[:prefix_end]
        prefix_finite = np.isfinite(prefix).all(axis=1)
        prefix = prefix[prefix_finite]
        marker_every = max(1, prefix.shape[0] // 10)
        ax.plot(
            prefix[:, 0],
            prefix[:, 1],
            label=SOLVER_DISPLAY_NAMES[solver_name],
            markevery=marker_every,
            markersize=4.0,
            **styles[solver_name],
        )
        for step in annotate_steps:
            if step < coordinates.shape[0] and np.isfinite(coordinates[step]).all():
                ax.annotate(
                    str(step),
                    (coordinates[step, 0], coordinates[step, 1]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

    initial_coordinate = projected["multi"][0]
    ax.scatter(
        [initial_coordinate[0]],
        [initial_coordinate[1]],
        marker="X",
        s=110,
        edgecolors="black",
        linewidths=0.8,
        label="Common initial state",
        zorder=6,
    )
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=190,
        edgecolors="black",
        linewidths=0.8,
        label="Exact minimizer",
        zorder=7,
    )
    ax.set_xlabel("PCA slice coordinate 1")
    ax.set_ylabel("PCA slice coordinate 2")
    ax.set_title(
        "Solver trajectories on a common 2D slice of the 6D energy\n"
        f"dataset={hardest_dataset_name}, "
        f"problem={sample_record['problem_index']}, "
        f"motion={sample_record['motion_index']}, "
        f"time index={sample_record['time_index']}, "
        f"GD alpha={alpha:.3e}"
    )
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(
        output_dir / "hardest_test_energy_contour_trajectories.png",
        dpi=240,
    )
    plt.close(fig)

    projection_record = {
        "interpretation": (
            "All six-dimensional solver states are projected onto a common "
            "two-dimensional PCA basis centered at the exact minimizer. The "
            "contours evaluate the true six-dimensional variational energy "
            "restricted to that affine plane."
        ),
        "basis_vectors": basis.tolist(),
        "singular_values": singular_values.tolist(),
        "x_bounds": [float(x_min), float(x_max)],
        "y_bounds": [float(y_min), float(y_max)],
        "energy_gap_bounds": [float(min_gap), float(max_gap)],
        "projected_trajectories": {
            name: coordinates.tolist()
            for name, coordinates in projected.items()
        },
    }
    return projection_record


# ============================================================
# 8. Main
# ============================================================


def main() -> None:
    args = parse_args()
    validate_args(args)
    original_script, experiment_dir, output_dir = resolve_paths(args)
    original = load_original_module(original_script)

    torch.set_default_dtype(original.TORCH_DTYPE)
    device = torch.device(args.device)
    original.validate_device(device)

    runtime, physical, runtime_payload = load_configs(
        original, experiment_dir
    )
    validation, tests, motions, problems, dataset_source = (
        load_or_rebuild_datasets(
            original,
            experiment_dir,
            runtime,
            physical,
            force_rebuild=args.force_rebuild_datasets,
        )
    )

    single_checkpoint = (
        experiment_dir
        / "single_motion_equal_budget_baseline"
        / "best_validation_model_state_dict.pt"
    )
    multi_checkpoint = (
        experiment_dir
        / "multi_motion"
        / "best_validation_model_state_dict.pt"
    )
    single_model = load_model(
        original,
        single_checkpoint,
        runtime.residual_length_scale,
        device,
    )
    multi_model = load_model(
        original,
        multi_checkpoint,
        runtime.residual_length_scale,
        device,
    )

    print(f"Original script: {original_script}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Comparison output: {output_dir}")
    print(f"Device: {device}")
    print(f"Dtype: {original.TORCH_DTYPE}")
    print(f"Dataset source: {dataset_source}")
    print(f"Single checkpoint: {single_checkpoint}")
    print(f"Multi checkpoint: {multi_checkpoint}")
    print(
        "Test sizes: "
        + ", ".join(f"{name}={len(dataset):,}"
                    for name, dataset in tests.items())
    )

    alpha_candidates = np.logspace(
        args.alpha_log10_min,
        args.alpha_log10_max,
        num=args.alpha_count,
        dtype=float,
    )
    selected_alpha, alpha_search_records = select_gradient_descent_alpha(
        original,
        validation=validation,
        physical=physical,
        candidates=alpha_candidates,
        steps=args.steps,
        batch_size=args.batch_size,
        device=device,
        output_dir=output_dir,
    )

    all_results: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_name in MAIN_TEST_DATASETS:
        dataset = tests[dataset_name]
        all_results[dataset_name] = {}
        print(f"\nEvaluating all solvers on {dataset_name} ({len(dataset):,} states)")
        for solver_name in SOLVER_ORDER:
            print(f"  {SOLVER_DISPLAY_NAMES[solver_name]}...")
            model = (
                single_model if solver_name == "single"
                else multi_model if solver_name == "multi"
                else None
            )
            result = evaluate_solver_curve(
                original,
                solver_name=solver_name,
                dataset_name=dataset_name,
                dataset=dataset,
                physical=physical,
                steps=args.steps,
                batch_size=args.batch_size,
                device=device,
                model=model,
                alpha=(
                    selected_alpha
                    if solver_name == "gradient_descent"
                    else None
                ),
            )
            all_results[dataset_name][solver_name] = result
            final_p95 = result["metrics"]["residual"]["p95_by_step"][-1]
            print(
                f"    final residual p95={float(final_p95):.6e}, "
                f"elapsed={result['elapsed_seconds']:.3f}s"
            )

    plot_residual_comparisons(all_results, output_dir)
    save_comparison_csv(all_results, output_dir)
    print_selected_iterations(all_results, args.steps)

    hardest_dataset_name, initial_residual_p95 = select_hardest_dataset(
        all_results
    )
    hardest_dataset = tests[hardest_dataset_name]
    selected_problem, selected_sample = select_random_problem_sample(
        hardest_dataset, args.seed
    )
    print(
        f"\nHardest test set by iteration-0 residual p95: "
        f"{hardest_dataset_name}"
    )
    print(
        f"Selected deterministic random problem/sample: "
        f"problem={selected_problem}, sample_index={selected_sample}"
    )

    trajectories, sample_record = rollout_one_sample(
        original,
        dataset=hardest_dataset,
        sample_index=selected_sample,
        physical=physical,
        steps=args.steps,
        device=device,
        single_model=single_model,
        multi_model=multi_model,
        alpha=selected_alpha,
    )
    sample_record.update(
        {
            "selection_seed": args.seed,
            "selection_rule": (
                "Choose the test dataset with largest iteration-0 pooled "
                "residual p95, then choose one problem index and one sample "
                "within that problem using a fixed NumPy random seed."
            ),
            "hardest_dataset": hardest_dataset_name,
            "initial_residual_p95_by_dataset": initial_residual_p95,
            "gradient_descent_alpha": selected_alpha,
            "trajectories_6d": {
                name: trajectory.tolist()
                for name, trajectory in trajectories.items()
            },
        }
    )

    projection_record: dict[str, Any] | None = None
    if not args.skip_contour:
        projection_record = plot_hard_problem_energy_contour(
            original,
            trajectories=trajectories,
            sample_record=sample_record,
            hardest_dataset_name=hardest_dataset_name,
            alpha=selected_alpha,
            physical=physical,
            device=device,
            grid_size=args.contour_grid_size,
            batch_size=args.batch_size,
            output_dir=output_dir,
        )
        sample_record["energy_contour_projection"] = projection_record
    save_json(sample_record, output_dir / "selected_hard_problem.json")

    report = {
        "comparison": (
            "Single-motion equal-budget MLP versus multi-motion MLP versus "
            "validation-selected fixed-step raw gradient descent versus "
            "undamped full Newton."
        ),
        "original_script": str(original_script),
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "torch_dtype": str(original.TORCH_DTYPE),
        "fixed_iterations": args.steps,
        "batch_size": args.batch_size,
        "dataset_source": dataset_source,
        "runtime_config": asdict(runtime),
        "physical_config": asdict(physical),
        "checkpoints": {
            "single": str(single_checkpoint),
            "multi": str(multi_checkpoint),
        },
        "gradient_descent": {
            "update": "y_{k+1} = y_k - alpha * grad E(y_k)",
            "uses_mass_preconditioning": False,
            "selected_alpha": selected_alpha,
            "candidate_log10_range": [
                args.alpha_log10_min,
                args.alpha_log10_max,
            ],
            "candidate_count": args.alpha_count,
            "selection_dataset": "unseen_motion_validation",
            "test_data_used_for_selection": False,
        },
        "hardest_test_selection": {
            "definition": (
                "Largest pooled residual p95 at solver iteration 0."
            ),
            "initial_residual_p95_by_dataset": initial_residual_p95,
            "selected_dataset": hardest_dataset_name,
            "selected_problem": selected_problem,
            "selected_sample_index": selected_sample,
            "random_seed": args.seed,
        },
        "results": all_results,
    }
    save_json(report, output_dir / "solver_comparison_metrics.json")

    print("\nSaved comparison outputs:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}")


if __name__ == "__main__":
    main()
