from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DatasetBundle, MotionSpec, PhysicalConfig, TimeStepProblem, finite_plot_value
from .constants import (
    FIXED_VERTEX_INDICES,
    FREE_STATE_DIM,
    NUM_FREE_PARTICLES,
    NUM_PARTICLES,
    NUM_SPRINGS,
    NUM_TRIANGLES,
    PLOT_FLOOR,
    SPATIAL_DIM,
    SPRING_EDGES,
    TORCH_DTYPE,
)
from .physics import free_state_from_full, full_positions_from_free, spring_lengths_from_free, stationarity_residual_norm


def plot_training_curves(
    train_log: Sequence[dict[str, Any]],
    validation_log: Sequence[dict[str, Any]],
    best_epoch: int | None,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    axes[0].plot([r["epoch"] for r in train_log], [finite_plot_value(r["training_energy_gap_sum"]) for r in train_log])
    axes[0].set_yscale("log")
    axes[0].set_title("Training energy-gap sum")
    val_epochs = [r["epoch"] for r in validation_log]
    specs = [
        ("final_residual_p95", "Validation residual p95"),
        ("final_residual_max", "Validation residual maximum"),
        ("worst_motion_final_residual_max", "Worst-motion residual maximum"),
    ]
    for ax, (key, title) in zip(axes[1:], specs):
        ax.plot(val_epochs, [finite_plot_value(r["metrics"][key]) for r in validation_log], marker="o")
        ax.set_yscale("log")
        ax.set_title(title)
    for ax in axes:
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_three_solver_rollout(comparison: dict[str, dict[str, Any]], *, title: str, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics = [("residual", "Residual"), ("energy_gap", "Energy gap"), ("exact_error", "Exact error")]
    labels = {"learned": "MLP", "gradient_descent": "gradient descent", "full_newton": "full Newton"}
    for col, (metric, metric_title) in enumerate(metrics):
        for row, stat in enumerate(["p95", "max"]):
            ax = axes[row, col]
            for solver_name, values in comparison.items():
                ax.plot(
                    range(values["steps"] + 1),
                    [finite_plot_value(v) for v in values[f"{metric}_{stat}_by_step"]],
                    marker="o", markersize=3, label=labels[solver_name],
                )
            ax.set_yscale("log")
            ax.set_xlabel("Solver iteration")
            ax.set_title(f"{metric_title} {stat}")
            ax.grid(True, alpha=0.3)
            if col == 0 and row == 0:
                ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_motion_boundary(comparison: dict[str, dict[str, Any]], *, title: str, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    labels = {"learned": "MLP", "gradient_descent": "GD", "full_newton": "Newton"}
    for ax, stat in zip(axes, ["p95", "max"]):
        for solver_name, values in comparison.items():
            motions = sorted(int(k) for k in values["per_motion"])
            y = [values["per_motion"][str(m)]["final"]["residual"][stat] for m in motions]
            ax.plot(motions, np.maximum(np.asarray(y, dtype=float), PLOT_FLOOR), marker="o", label=labels[solver_name])
        ax.set_yscale("log")
        ax.set_xlabel("Motion index")
        ax.set_ylabel(f"Final residual {stat}")
        ax.set_title(f"Per-motion residual {stat}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reference_motion_overview(motions: Sequence[MotionSpec], save_path: Path) -> None:
    fig = plt.figure(figsize=(16, 12))
    for panel, motion in enumerate(motions[:12]):
        ax = fig.add_subplot(3, 4, panel + 1, projection="3d")
        points = np.asarray(motion.p0)
        for i, j in SPRING_EDGES:
            ax.plot(points[[i, j], 0], points[[i, j], 1], points[[i, j], 2], linewidth=0.7)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=8)
        fixed = points[list(FIXED_VERTEX_INDICES)]
        ax.scatter(fixed[:, 0], fixed[:, 1], fixed[:, 2], marker="s", s=35)
        ax.set_title(f"{motion.index}: {motion.name}", fontsize=8)
        ax.view_init(elev=22, azim=-62)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_hard_ood_case(
    ood_dataset: DatasetBundle,
    problems_by_index: dict[int, TimeStepProblem],
    physical: PhysicalConfig,
) -> dict[str, Any]:
    residual = stationarity_residual_norm(
        ood_dataset.initial_y, ood_dataset.q, ood_dataset.masses, physical
    ).numpy()
    best_record: dict[str, Any] | None = None
    for problem_index in sorted(torch.unique(ood_dataset.problem_index).tolist()):
        mask = ood_dataset.problem_index.numpy() == int(problem_index)
        values = residual[mask]
        local_indices = np.flatnonzero(mask)
        local_argmax = int(np.nanargmax(values))
        sample_index = int(local_indices[local_argmax])
        record = {
            "problem_index": int(problem_index),
            "initial_residual_max": float(np.nanmax(values)),
            "initial_residual_p95": float(np.nanpercentile(values, 95)),
            "sample_index_in_dataset": sample_index,
            "sample_initial_y": ood_dataset.initial_y[sample_index].tolist(),
        }
        if best_record is None or (
            record["initial_residual_max"], record["initial_residual_p95"]
        ) > (
            best_record["initial_residual_max"], best_record["initial_residual_p95"]
        ):
            best_record = record
    if best_record is None:
        raise RuntimeError("Could not select a hard OOD case")
    problem = problems_by_index[best_record["problem_index"]]
    best_record.update({
        "selection_rule": "largest sampled initial residual maximum; p95 is the tie breaker",
        "motion_index": problem.motion_index,
        "motion_name": problem.motion_name,
        "motion_category": problem.motion_category,
        "local_time_index": problem.local_time_index,
        "physical_time": problem.time,
        "selected_physical_state": {
            "p_n_full": problem.p_n_full.tolist(),
            "v_n_full": problem.v_n_full.tolist(),
        },
    })
    return best_record


def run_physics_checks(physical: PhysicalConfig, motion: MotionSpec) -> dict[str, Any]:
    p = torch.tensor(motion.p0, dtype=TORCH_DTYPE)
    y = free_state_from_full(p).reshape(1, -1)
    reconstructed = full_positions_from_free(y, physical).reshape(NUM_PARTICLES, SPATIAL_DIM)
    fixed_error = float(torch.max(torch.abs(reconstructed[list(FIXED_VERTEX_INDICES)] - p[list(FIXED_VERTEX_INDICES)])).item())
    lengths = spring_lengths_from_free(y, physical).squeeze(0)
    return {
        "num_particles": NUM_PARTICLES,
        "num_free_particles": NUM_FREE_PARTICLES,
        "free_state_dimension": FREE_STATE_DIM,
        "num_springs": NUM_SPRINGS,
        "num_triangles": NUM_TRIANGLES,
        "fixed_reconstruction_error": fixed_error,
        "minimum_initial_spring_length": float(lengths.min().item()),
    }


def problem_to_record(problem: TimeStepProblem) -> dict[str, Any]:
    return {
        "index": problem.index,
        "motion_index": problem.motion_index,
        "motion_name": problem.motion_name,
        "motion_split": problem.motion_split,
        "motion_category": problem.motion_category,
        "local_time_index": problem.local_time_index,
        "time": problem.time,
        "p_n_full": problem.p_n_full.tolist(),
        "v_n_full": problem.v_n_full.tolist(),
        "q_free": problem.q_free.tolist(),
        "free_masses": problem.free_masses.tolist(),
        "exact_y_free": problem.exact_y_free.tolist(),
        "raw_sampling_radius": problem.raw_sampling_radius,
        "sampling_radius": problem.sampling_radius,
        "exact_energy": problem.exact_energy,
        "exact_residual": problem.exact_residual,
    }
