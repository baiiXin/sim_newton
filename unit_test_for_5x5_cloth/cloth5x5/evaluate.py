from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DatasetBundle, PhysicalConfig, RuntimeConfig
from .constants import GD_CANDIDATE_STEP_SIZES, PLOT_FLOOR
from .model import MLPOptimizer
from .physics import (
    reshape_free,
    spring_lengths_from_free,
    stationarity_residual_norm,
    variational_energy,
)
from .solvers import run_solver_steps


def _statistics(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    result = {
        "num_nonfinite": int(values.size - finite.size),
        "mean": float("nan"),
        "median": float("nan"),
        "p95": float("nan"),
        "max": float("nan"),
    }
    if finite.size:
        result.update(
            mean=float(np.mean(finite)),
            median=float(np.median(finite)),
            p95=float(np.percentile(finite, 95)),
            max=float(np.max(finite)),
        )
    return result


def _state_metrics(
    y: torch.Tensor,
    dataset: DatasetBundle,
    exact_energy: torch.Tensor,
    physical: PhysicalConfig,
) -> dict[str, torch.Tensor]:
    point_errors = torch.linalg.vector_norm(
        reshape_free(y) - reshape_free(dataset.exact_y), dim=-1
    )
    energy = variational_energy(y, dataset.q, dataset.masses, physical)
    return {
        "residual": stationarity_residual_norm(y, dataset.q, dataset.masses, physical),
        "energy_gap": energy - exact_energy,
        "exact_error": torch.linalg.vector_norm(y - dataset.exact_y, dim=-1),
        "particle_mean_error": point_errors.mean(dim=-1),
        "particle_max_error": point_errors.max(dim=-1).values,
        "spring_length_error": torch.mean(
            torch.abs(
                spring_lengths_from_free(y, physical)
                - spring_lengths_from_free(dataset.exact_y, physical)
            ),
            dim=-1,
        ),
        "fixed_vertex_max_error": torch.zeros_like(point_errors[..., 0]),
    }


def _selected_steps(steps: int, report_steps: Sequence[int]) -> list[int]:
    return sorted(set([0, steps, *[s for s in report_steps if 0 <= s <= steps]]))


@torch.no_grad()
def evaluate_solver_on_dataset(
    *,
    solver: str,
    dataset_cpu: DatasetBundle,
    physical: PhysicalConfig,
    steps: int,
    batch_size: int,
    report_steps: Sequence[int],
    device: torch.device,
    model: MLPOptimizer | None = None,
    gd_step_size: float | None = None,
) -> dict[str, Any]:
    if solver not in {"learned", "gradient_descent", "full_newton"}:
        raise ValueError(solver)
    if solver == "learned" and model is None:
        raise ValueError("model is required")
    if solver == "gradient_descent" and gd_step_size is None:
        raise ValueError("gd_step_size is required")
    if model is not None:
        model.eval()

    metric_batches: dict[str, list[torch.Tensor]] = {}
    problem_batches: list[torch.Tensor] = []
    motion_batches: list[torch.Tensor] = []
    time_batches: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for start in range(0, len(dataset_cpu), batch_size):
        end = min(start + batch_size, len(dataset_cpu))
        batch = DatasetBundle(
            initial_y=dataset_cpu.initial_y[start:end],
            q=dataset_cpu.q[start:end],
            masses=dataset_cpu.masses[start:end],
            exact_y=dataset_cpu.exact_y[start:end],
            problem_index=dataset_cpu.problem_index[start:end],
            motion_index=dataset_cpu.motion_index[start:end],
            time_index=dataset_cpu.time_index[start:end],
            metadata={},
        ).to(device)
        y = batch.initial_y.clone()
        exact_energy = variational_energy(batch.exact_y, batch.q, batch.masses, physical)
        step_values: dict[str, list[torch.Tensor]] = {}
        for step in range(steps + 1):
            for name, values in _state_metrics(y, batch, exact_energy, physical).items():
                step_values.setdefault(name, []).append(values.detach().cpu())
            if step == steps:
                break
            y, _ = run_solver_steps(
                solver,
                y,
                batch.q,
                batch.masses,
                physical,
                1,
                model=model,
                gd_step_size=gd_step_size,
                require_finite=True,
            )
        for name, values in step_values.items():
            metric_batches.setdefault(name, []).append(torch.stack(values, dim=1))
        problem_batches.append(batch.problem_index.detach().cpu())
        motion_batches.append(batch.motion_index.detach().cpu())
        time_batches.append(batch.time_index.detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    arrays = {name: torch.cat(values, dim=0).numpy().astype(float) for name, values in metric_batches.items()}
    problem_indices = torch.cat(problem_batches).numpy().astype(int)
    motion_indices = torch.cat(motion_batches).numpy().astype(int)
    time_indices = torch.cat(time_batches).numpy().astype(int)
    for values in arrays.values():
        values[~np.isfinite(values)] = np.nan

    result: dict[str, Any] = {
        "solver": solver,
        "steps": steps,
        "num_points": len(dataset_cpu),
        "num_motions": int(np.unique(motion_indices).size),
        "selected_report_steps": _selected_steps(steps, report_steps),
        "elapsed_seconds": elapsed,
        "seconds_per_point_per_iteration": elapsed / max(len(dataset_cpu) * steps, 1),
    }
    if gd_step_size is not None:
        result["gradient_descent_step_size"] = float(gd_step_size)

    for name, values in arrays.items():
        for stat_name in ["mean", "median", "p95", "max", "num_nonfinite"]:
            result[f"{name}_{stat_name}_by_step"] = []
        for step in range(values.shape[1]):
            stats = _statistics(values[:, step])
            for stat_name, value in stats.items():
                result[f"{name}_{stat_name}_by_step"].append(value)
        for stat_name, value in _statistics(values[:, -1]).items():
            result[f"final_{name}_{stat_name}"] = value

    selected = result["selected_report_steps"]
    per_motion: dict[str, Any] = {}
    for motion_index in sorted(np.unique(motion_indices).tolist()):
        mask = motion_indices == motion_index
        record: dict[str, Any] = {
            "motion_index": int(motion_index),
            "num_points": int(mask.sum()),
            "time_indices": sorted(np.unique(time_indices[mask]).astype(int).tolist()),
            "steps": {},
            "final": {},
        }
        for step in selected:
            record["steps"][str(step)] = {
                name: _statistics(values[mask, step]) for name, values in arrays.items()
            }
        record["final"] = {
            name: _statistics(values[mask, -1]) for name, values in arrays.items()
        }
        per_motion[str(motion_index)] = record
    result["per_motion"] = per_motion

    per_problem: dict[str, Any] = {}
    for problem_index in sorted(np.unique(problem_indices).tolist()):
        mask = problem_indices == problem_index
        per_problem[str(problem_index)] = {
            "problem_index": int(problem_index),
            "motion_index": int(motion_indices[mask][0]),
            "time_index": int(time_indices[mask][0]),
            "num_points": int(mask.sum()),
            "final": {name: _statistics(values[mask, -1]) for name, values in arrays.items()},
        }
    result["per_problem"] = per_problem

    worst_motion: dict[str, Any] = {}
    for metric_name in arrays:
        metric_records = []
        for motion_key, record in per_motion.items():
            stats = record["final"][metric_name]
            metric_records.append((int(motion_key), stats))
        finite_p95 = [(m, float(s["p95"])) for m, s in metric_records if math.isfinite(float(s["p95"]))]
        finite_max = [(m, float(s["max"])) for m, s in metric_records if math.isfinite(float(s["max"]))]
        p95_motion, p95_value = max(finite_p95, key=lambda item: item[1]) if finite_p95 else (-1, float("nan"))
        max_motion, max_value = max(finite_max, key=lambda item: item[1]) if finite_max else (-1, float("nan"))
        worst_motion[metric_name] = {
            "p95_motion_index": p95_motion,
            "p95": p95_value,
            "max_motion_index": max_motion,
            "max": max_value,
        }
        result[f"worst_motion_final_{metric_name}_p95"] = p95_value
        result[f"worst_motion_final_{metric_name}_p95_motion_index"] = p95_motion
        result[f"worst_motion_final_{metric_name}_max"] = max_value
        result[f"worst_motion_final_{metric_name}_max_motion_index"] = max_motion
    result["worst_motion"] = worst_motion
    return result


def validation_selection_key(metrics: dict[str, Any]) -> tuple[float, ...] | None:
    values = (
        float(metrics["final_residual_num_nonfinite"]),
        float(metrics["worst_motion_final_residual_max"]),
        float(metrics["worst_motion_final_residual_p95"]),
        float(metrics["final_residual_max"]),
        float(metrics["final_residual_p95"]),
        float(metrics["worst_motion_final_exact_error_max"]),
        float(metrics["final_exact_error_max"]),
        float(metrics["final_exact_error_p95"]),
        float(metrics["final_energy_gap_max"]),
        float(metrics["final_energy_gap_p95"]),
    )
    return values if all(math.isfinite(v) for v in values) else None


def select_gradient_descent_step_size(
    *,
    validation: DatasetBundle,
    physical: PhysicalConfig,
    config: RuntimeConfig,
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    best_alpha: float | None = None
    best_key: tuple[float, ...] | None = None
    for alpha in GD_CANDIDATE_STEP_SIZES:
        print(f"Evaluating validation GD step size alpha={alpha:.1e} ...")
        metrics = evaluate_solver_on_dataset(
            solver="gradient_descent",
            dataset_cpu=validation,
            physical=physical,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            report_steps=config.report_steps,
            device=device,
            gd_step_size=alpha,
        )
        key = validation_selection_key(metrics)
        records.append({"step_size": alpha, "selection_key": key, "metrics": metrics})
        if key is not None and (best_key is None or key < best_key):
            best_key = key
            best_alpha = alpha
    if best_alpha is None:
        raise RuntimeError("No finite gradient-descent candidate was found")
    return best_alpha, {
        "candidate_step_sizes": list(GD_CANDIDATE_STEP_SIZES),
        "selection_rule": (
            "lexicographic: nonfinite count, worst-motion residual max, "
            "worst-motion residual p95, pooled residual max, pooled residual p95, "
            "then exact-error and energy-gap boundary metrics"
        ),
        "selected_step_size": best_alpha,
        "selected_key": best_key,
        "records": records,
    }


def plot_gradient_descent_step_size_selection(gd_selection: dict[str, Any], save_path: Path) -> None:
    records = gd_selection.get("records", [])
    if not records:
        return
    alphas = np.asarray([float(r["step_size"]) for r in records], dtype=float)
    residual_p95 = np.asarray([float(r["metrics"]["final_residual_p95"]) for r in records])
    residual_max = np.asarray([float(r["metrics"]["final_residual_max"]) for r in records])
    worst_motion_max = np.asarray([float(r["metrics"]["worst_motion_final_residual_max"]) for r in records])
    selected = float(gd_selection["selected_step_size"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(alphas, np.maximum(residual_p95, PLOT_FLOOR), marker="o", label="pooled residual p95")
    ax.plot(alphas, np.maximum(residual_max, PLOT_FLOOR), marker="s", label="pooled residual max")
    ax.plot(alphas, np.maximum(worst_motion_max, PLOT_FLOOR), marker="^", label="worst-motion residual max")
    ax.axvline(selected, linestyle="--", alpha=0.8, label=f"selected {selected:.1e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gradient-descent step size")
    ax.set_ylabel("Validation final residual after fixed iterations")
    ax.set_title("Validation selection of gradient-descent step size")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
