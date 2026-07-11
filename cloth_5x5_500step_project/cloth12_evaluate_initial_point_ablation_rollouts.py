"""Evaluate initial-point-count ablation models by 500 x 50 continuous rollout.

Default test motions are 20-31, i.e. all existing motions outside training and
validation. Every solver starts each physical frame from its own propagated
physical state, runs 50 inner iterations, and stores:
- frame initial value y^(0)
- frame solution y^(50)
- all 51 residuals, including iteration 0
- propagated positions and velocities
- reference trajectory error

Each solver/model curve is saved independently. Plotting reads only saved curves,
so figures can be regenerated without rerunning rollout. Existing reference
trajectories are reused; their solved states are plotted only at each frame's
iteration-50 endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cloth03_solvers_and_models import (
    DEFAULT_DEVICE,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    FIXED_VERTEX_INDICES,
    MLPOptimizer,
    ModelSpec,
    NUM_FREE_PARTICLES,
    NUM_PARTICLES,
    SPATIAL_DIM,
    TORCH_DTYPE,
    AdamState,
    apply_adam_update_full,
    apply_gradient_descent_update_full,
    apply_model_update,
    apply_newton_update_full,
    full_state_from_free_state,
    full_state_from_positions,
    make_q_free,
    physical_config_from_dict,
    project_fixed_vertices,
    run_lbfgs_iterations_full,
    stationarity_residual_norm_full,
)

DEFAULT_SAMPLE_COUNTS = (1, 8, 32, 64, 128, 1024)
DEFAULT_TEST_MOTIONS = tuple(range(20, 32))
DEFAULT_BASELINES = ("gd", "adam", "lbfgs", "newton")
DEFAULT_ROLLOUT_LENGTH = 500
DEFAULT_INNER_STEPS = 50
PLOT_FLOOR = 1e-16


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_float(value: float | torch.Tensor) -> float:
    number = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    return number if math.isfinite(number) else float("inf")


def load_physical(source_root: Path):
    runtime = load_json(source_root / "data" / "reference" / "runtime_config.json")
    return physical_config_from_dict(runtime["physical_config"])


def load_reference_states(source_root: Path) -> dict[str, Any]:
    path = source_root / "data" / "reference" / "reference_motion_states.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def reference_for_motion(
    reference_states: dict[str, Any], motion_index: int, rollout_length: int
) -> dict[str, torch.Tensor]:
    motion_ids = [int(value) for value in reference_states["motion_index"].tolist()]
    row = motion_ids.index(int(motion_index))
    positions = reference_states["positions"][row, : rollout_length + 1].contiguous()
    velocities = reference_states["velocities"][row, : rollout_length + 1].contiguous()
    if positions.shape[0] != rollout_length + 1:
        raise RuntimeError(f"motion {motion_index} reference has only {positions.shape[0] - 1} steps")
    return {"positions": positions, "velocities": velocities}


def free_masses_tensor(physical, device: torch.device) -> torch.Tensor:
    fixed = set(FIXED_VERTEX_INDICES)
    values = [physical.masses[i] for i in range(NUM_PARTICLES) if i not in fixed]
    return torch.tensor(values, dtype=TORCH_DTYPE, device=device).reshape(1, NUM_FREE_PARTICLES)


def velocity_from_positions(p_prev: torch.Tensor, p_next: torch.Tensor, physical) -> torch.Tensor:
    velocity = (p_next - p_prev) / physical.dt
    velocity[list(FIXED_VERTEX_INDICES), :] = 0.0
    return velocity


def load_baseline_params(source_root: Path, name: str) -> dict[str, Any]:
    selection_path = source_root / "baselines" / "parameter_selection.json"
    if selection_path.exists():
        selection = load_json(selection_path)
        if name in selection:
            return dict(selection[name].get("selected", {}))
    defaults = {
        "gd": {"step_size": 1e-5},
        "adam": {"learning_rate": 1e-3},
        "lbfgs": {"learning_rate": 1.0, "history_size": 10},
        "newton": {},
    }
    return defaults[name]


def model_dir_for_count(
    ablation_root: Path,
    sample_count: int,
    model_spec: ModelSpec,
) -> Path:
    return (
        ablation_root
        / f"points_{sample_count:04d}"
        / "models"
        / model_spec.experiment_name
    )


def load_model(
    model_dir: Path,
    device: torch.device,
    residual_length_scale: float,
) -> tuple[MLPOptimizer, dict[str, Any]]:
    checkpoint_path = model_dir / "best_validation_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    spec_data = checkpoint["model_spec"]
    model_spec = ModelSpec(
        activation=str(spec_data["activation"]),
        depth=int(spec_data["depth"]),
        width=int(spec_data["width"]),
        use_bias=bool(spec_data["use_bias"]),
    )
    config = checkpoint.get("config", {})
    scale = float(config.get("residual_length_scale", residual_length_scale))
    model = MLPOptimizer(scale, model_spec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "sample_count": int(checkpoint.get("sample_count", -1)),
        "best_validation_max": float(checkpoint.get("best_validation_max", float("inf"))),
        "model_spec": asdict(model_spec),
        "residual_length_scale": scale,
    }


def residual_value(y: torch.Tensor, q: torch.Tensor, masses: torch.Tensor, physical) -> float:
    return safe_float(stationarity_residual_norm_full(y, q, masses, physical))


@torch.no_grad()
def solve_model_frame(
    *,
    model: MLPOptimizer,
    y0: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical,
    inner_steps: int,
) -> tuple[torch.Tensor, list[float]]:
    y = project_fixed_vertices(y0.clone(), physical)
    residuals = [residual_value(y, q, masses, physical)]
    previous_residual = torch.zeros_like(y)
    previous_update = torch.zeros_like(y)
    for _ in range(inner_steps):
        y, delta, current_residual = apply_model_update(
            model,
            y,
            q,
            masses,
            physical,
            previous_residual=previous_residual,
            previous_update=previous_update,
        )
        previous_residual = current_residual.detach()
        previous_update = delta.detach()
        residuals.append(residual_value(y, q, masses, physical))
    return y, residuals


def solve_baseline_frame(
    *,
    method: str,
    params: dict[str, Any],
    y0: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical,
    inner_steps: int,
) -> tuple[torch.Tensor, list[float]]:
    y = project_fixed_vertices(y0.clone(), physical)
    residuals = [residual_value(y, q, masses, physical)]

    if method == "gd":
        for _ in range(inner_steps):
            y, _ = apply_gradient_descent_update_full(
                y, q, masses, physical, float(params["step_size"])
            )
            residuals.append(residual_value(y, q, masses, physical))
        return y, residuals

    if method == "adam":
        state: AdamState | None = None
        for _ in range(inner_steps):
            y, _, state = apply_adam_update_full(
                y,
                q,
                masses,
                physical,
                state,
                learning_rate=float(params["learning_rate"]),
            )
            residuals.append(residual_value(y, q, masses, physical))
        return y, residuals

    if method == "newton":
        for _ in range(inner_steps):
            y, _ = apply_newton_update_full(y, q, masses, physical)
            residuals.append(residual_value(y, q, masses, physical))
        return y, residuals

    if method == "lbfgs":
        states = run_lbfgs_iterations_full(
            y,
            q,
            masses,
            physical,
            steps=inner_steps,
            learning_rate=float(params["learning_rate"]),
            history_size=int(params.get("history_size", 10)),
        )
        residuals = [residual_value(state, q, masses, physical) for state in states]
        return states[-1], residuals

    raise ValueError(method)


def compute_reference_endpoint_record(
    *,
    reference: dict[str, torch.Tensor],
    physical,
    device: torch.device,
    rollout_length: int,
    inner_steps: int,
) -> dict[str, Any]:
    masses = free_masses_tensor(physical, device)
    solutions = []
    residuals = []
    x = []
    for frame_index in range(rollout_length):
        p_n = reference["positions"][frame_index].to(device=device, dtype=TORCH_DTYPE)
        v_n = reference["velocities"][frame_index].to(device=device, dtype=TORCH_DTYPE)
        q_free = make_q_free(p_n, v_n, physical).reshape(1, -1)
        q_full = project_fixed_vertices(full_state_from_free_state(q_free, physical), physical)
        exact = full_state_from_positions(
            reference["positions"][frame_index + 1].to(device=device, dtype=TORCH_DTYPE)
        ).reshape(1, -1)
        exact = project_fixed_vertices(exact, physical)
        solutions.append(exact.squeeze(0).cpu())
        residuals.append(residual_value(exact, q_full, masses, physical))
        x.append((frame_index + 1) * inner_steps - 1)
    return {
        "solver_name": "reference",
        "solution_y_by_frame": torch.stack(solutions, dim=0),
        "endpoint_residual_by_frame": torch.tensor(residuals, dtype=TORCH_DTYPE),
        "global_iteration": torch.tensor(x, dtype=torch.long),
        "global_residual": torch.tensor(residuals, dtype=TORCH_DTYPE),
        "metadata": {
            "rollout_length": rollout_length,
            "inner_steps": inner_steps,
            "plot_semantics": "one stored reference solution point at each frame's final inner-iteration endpoint",
            "reference_regenerated": False,
        },
    }


def save_partial_rollout(
    *,
    path: Path,
    solver_name: str,
    motion_index: int,
    positions: list[torch.Tensor],
    velocities: list[torch.Tensor],
    initial_y: list[torch.Tensor],
    solution_y: list[torch.Tensor],
    residual_curves: list[torch.Tensor],
    reference_errors: list[float],
    elapsed_by_frame: list[float],
    rollout_length: int,
    inner_steps: int,
    solver_info: dict[str, Any],
    failed: bool,
    failure_frame: int | None,
) -> None:
    completed = len(solution_y)
    if residual_curves:
        residual_tensor = torch.stack(residual_curves, dim=0).contiguous()
        update_curve = residual_tensor[:, 1:].reshape(-1).contiguous()
        global_iteration = torch.arange(update_curve.numel(), dtype=torch.long)
    else:
        residual_tensor = torch.empty((0, inner_steps + 1), dtype=TORCH_DTYPE)
        update_curve = torch.empty((0,), dtype=TORCH_DTYPE)
        global_iteration = torch.empty((0,), dtype=torch.long)
    payload = {
        "solver_name": solver_name,
        "motion_index": motion_index,
        "positions": torch.stack(positions, dim=0).contiguous(),
        "velocities": torch.stack(velocities, dim=0).contiguous(),
        "initial_y_by_frame": (
            torch.stack(initial_y, dim=0).contiguous()
            if initial_y
            else torch.empty((0, NUM_PARTICLES * SPATIAL_DIM), dtype=TORCH_DTYPE)
        ),
        "solution_y_by_frame": (
            torch.stack(solution_y, dim=0).contiguous()
            if solution_y
            else torch.empty((0, NUM_PARTICLES * SPATIAL_DIM), dtype=TORCH_DTYPE)
        ),
        "residual_by_frame_and_iteration": residual_tensor,
        "final_residual_by_frame": residual_tensor[:, -1].clone(),
        "global_iteration": global_iteration,
        "global_residual": update_curve,
        "reference_error_by_frame": torch.tensor(reference_errors, dtype=TORCH_DTYPE),
        "elapsed_seconds_by_frame": torch.tensor(elapsed_by_frame, dtype=TORCH_DTYPE),
        "metadata": {
            "requested_rollout_length": rollout_length,
            "completed_steps": completed,
            "inner_steps": inner_steps,
            "residual_shape_semantics": "[frame, iteration 0..inner_steps]",
            "global_curve_semantics": "flattened iterations 1..inner_steps without separators",
            "initial_and_solution_saved": True,
            "solver_info": solver_info,
            "failed": failed,
            "failure_frame": failure_frame,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def validate_resume_rollout(
    *,
    saved: dict[str, Any],
    output_path: Path,
    rollout_length: int,
    inner_steps: int,
) -> None:
    metadata = saved.get("metadata", {})
    try:
        saved_inner_steps = int(metadata["inner_steps"])
        saved_rollout_length = int(metadata["requested_rollout_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot resume {output_path}: saved metadata is missing rollout settings; "
            "rerun with --overwrite to regenerate it"
        ) from exc

    if saved_inner_steps != int(inner_steps):
        raise RuntimeError(
            f"cannot resume {output_path}: saved inner_steps={saved_inner_steps}, "
            f"requested inner_steps={inner_steps}; rerun with --overwrite to regenerate it"
        )

    if saved_rollout_length != int(rollout_length):
        raise RuntimeError(
            f"cannot resume {output_path}: saved rollout_length={saved_rollout_length}, "
            f"requested rollout_length={rollout_length}; rerun with --overwrite to regenerate it"
        )

    residuals = saved.get("residual_by_frame_and_iteration")
    if not torch.is_tensor(residuals) or residuals.ndim != 2:
        raise RuntimeError(f"cannot resume {output_path}: missing 2D residual tensor")
    expected_width = int(inner_steps) + 1
    if int(residuals.shape[1]) != expected_width:
        raise RuntimeError(
            f"cannot resume {output_path}: saved residual width={int(residuals.shape[1])}, "
            f"expected {expected_width}; rerun with --overwrite to regenerate it"
        )


def run_solver_rollout(
    *,
    solver_name: str,
    solver_info: dict[str, Any],
    solve_frame: Callable[..., tuple[torch.Tensor, list[float]]],
    reference: dict[str, torch.Tensor],
    physical,
    device: torch.device,
    motion_index: int,
    rollout_length: int,
    inner_steps: int,
    output_path: Path,
    overwrite: bool,
    checkpoint_every: int,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        saved = torch.load(output_path, map_location="cpu")
        validate_resume_rollout(
            saved=saved,
            output_path=output_path,
            rollout_length=rollout_length,
            inner_steps=inner_steps,
        )
        if int(saved["metadata"]["completed_steps"]) >= rollout_length:
            print(f"skip {solver_name} motion {motion_index}: complete")
            return saved
        positions = [value.clone() for value in saved["positions"]]
        velocities = [value.clone() for value in saved["velocities"]]
        initial_y = [value.clone() for value in saved["initial_y_by_frame"]]
        solution_y = [value.clone() for value in saved["solution_y_by_frame"]]
        residual_curves = [value.clone() for value in saved["residual_by_frame_and_iteration"]]
        reference_errors = [float(value) for value in saved["reference_error_by_frame"].tolist()]
        elapsed_by_frame = [float(value) for value in saved["elapsed_seconds_by_frame"].tolist()]
        start_frame = int(saved["metadata"]["completed_steps"])
        print(f"resume {solver_name} motion {motion_index}: {start_frame}/{rollout_length}")
    else:
        positions = [reference["positions"][0].clone()]
        velocities = [reference["velocities"][0].clone()]
        initial_y = []
        solution_y = []
        residual_curves = []
        reference_errors = []
        elapsed_by_frame = []
        start_frame = 0

    masses = free_masses_tensor(physical, device)
    failed = False
    failure_frame: int | None = None

    for frame_index in range(start_frame, rollout_length):
        frame_start = time.perf_counter()
        p_n = positions[-1].to(device=device, dtype=TORCH_DTYPE)
        v_n = velocities[-1].to(device=device, dtype=TORCH_DTYPE)
        q_free = make_q_free(p_n, v_n, physical).reshape(1, -1)
        q_full = project_fixed_vertices(full_state_from_free_state(q_free, physical), physical)
        y0 = project_fixed_vertices(full_state_from_positions(p_n).reshape(1, -1), physical)

        try:
            y_next, residuals = solve_frame(
                y0=y0,
                q=q_full,
                masses=masses,
                physical=physical,
                inner_steps=inner_steps,
            )
        except Exception as exc:
            print(f"{solver_name} motion {motion_index} frame {frame_index} failed: {exc}")
            failed = True
            failure_frame = frame_index
            break

        if len(residuals) != inner_steps + 1:
            raise RuntimeError(f"{solver_name} returned {len(residuals)} residuals")
        if not bool(torch.isfinite(y_next).all()) or not all(math.isfinite(v) for v in residuals):
            failed = True
            failure_frame = frame_index
            print(f"{solver_name} motion {motion_index} non-finite at frame {frame_index}")
            break

        p_next = y_next.reshape(NUM_PARTICLES, SPATIAL_DIM)
        v_next = velocity_from_positions(p_n, p_next, physical)
        ref_next = reference["positions"][frame_index + 1].to(device=device, dtype=TORCH_DTYPE)

        initial_y.append(y0.squeeze(0).detach().cpu())
        solution_y.append(y_next.squeeze(0).detach().cpu())
        residual_curves.append(torch.tensor(residuals, dtype=TORCH_DTYPE))
        reference_errors.append(safe_float(torch.linalg.vector_norm(p_next - ref_next)))
        elapsed_by_frame.append(time.perf_counter() - frame_start)
        positions.append(p_next.detach().cpu())
        velocities.append(v_next.detach().cpu())

        if (
            frame_index == start_frame
            or (frame_index + 1) % 25 == 0
            or frame_index + 1 == rollout_length
        ):
            print(
                f"{solver_name} motion={motion_index:03d} frame={frame_index + 1:04d}/{rollout_length} "
                f"residual={residuals[-1]:.3e} error={reference_errors[-1]:.3e}"
            )

        if (frame_index + 1) % checkpoint_every == 0 or frame_index + 1 == rollout_length:
            save_partial_rollout(
                path=output_path,
                solver_name=solver_name,
                motion_index=motion_index,
                positions=positions,
                velocities=velocities,
                initial_y=initial_y,
                solution_y=solution_y,
                residual_curves=residual_curves,
                reference_errors=reference_errors,
                elapsed_by_frame=elapsed_by_frame,
                rollout_length=rollout_length,
                inner_steps=inner_steps,
                solver_info=solver_info,
                failed=False,
                failure_frame=None,
            )

    save_partial_rollout(
        path=output_path,
        solver_name=solver_name,
        motion_index=motion_index,
        positions=positions,
        velocities=velocities,
        initial_y=initial_y,
        solution_y=solution_y,
        residual_curves=residual_curves,
        reference_errors=reference_errors,
        elapsed_by_frame=elapsed_by_frame,
        rollout_length=rollout_length,
        inner_steps=inner_steps,
        solver_info=solver_info,
        failed=failed,
        failure_frame=failure_frame,
    )
    return torch.load(output_path, map_location="cpu")


def padded_global_curve(record: dict[str, Any], rollout_length: int, inner_steps: int) -> np.ndarray:
    target = rollout_length * inner_steps
    curve = record["global_residual"].detach().cpu().numpy().astype(float)
    if curve.size >= target:
        return curve[:target]
    padded = np.full(target, np.inf, dtype=float)
    padded[: curve.size] = curve
    return padded


def summarize_record(record: dict[str, Any], rollout_length: int, inner_steps: int) -> dict[str, Any]:
    final = record["final_residual_by_frame"].detach().cpu().numpy().astype(float)
    errors = record["reference_error_by_frame"].detach().cpu().numpy().astype(float)
    finite_final = final[np.isfinite(final)]
    finite_errors = errors[np.isfinite(errors)]
    completed = int(record["metadata"]["completed_steps"])
    return {
        "motion_index": int(record["motion_index"]),
        "solver_name": str(record["solver_name"]),
        "completed_steps": completed,
        "completed_full_rollout": completed >= rollout_length,
        "failed": bool(record["metadata"].get("failed", False)),
        "failure_frame": record["metadata"].get("failure_frame"),
        "final_residual": float(final[-1]) if final.size else float("inf"),
        "max_final_residual": float(np.max(final)) if final.size else float("inf"),
        "p95_final_residual": (
            float(np.percentile(finite_final, 95)) if finite_final.size else float("inf")
        ),
        "max_reference_error": float(np.max(errors)) if errors.size else float("inf"),
        "p95_reference_error": (
            float(np.percentile(finite_errors, 95)) if finite_errors.size else float("inf")
        ),
        "mean_frame_seconds": (
            float(record["elapsed_seconds_by_frame"].mean().item())
            if record["elapsed_seconds_by_frame"].numel()
            else float("inf")
        ),
        "rollout_length": rollout_length,
        "inner_steps": inner_steps,
    }


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_motion(
    *,
    motion_dir: Path,
    records: list[dict[str, Any]],
    reference_record: dict[str, Any],
    rollout_length: int,
    inner_steps: int,
) -> None:
    figure_dir = motion_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    all_curves: dict[str, Any] = {"reference": reference_record}

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(rollout_length * inner_steps)
    for record in records:
        name = str(record["solver_name"])
        curve = padded_global_curve(record, rollout_length, inner_steps)
        all_curves[name] = {
            "global_iteration": torch.from_numpy(x.copy()),
            "global_residual": torch.from_numpy(curve.copy()),
            "source": str(motion_dir / name / "curve.pt"),
        }
        linestyle = "--" if name.startswith("baseline_") else "-"
        linewidth = 1.5 if "points_0032" in name else 0.9
        ax.plot(
            x,
            np.maximum(curve, PLOT_FLOOR),
            label=name,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    ref_x = reference_record["global_iteration"].numpy()
    ref_y = reference_record["global_residual"].numpy()
    ax.plot(
        ref_x,
        np.maximum(ref_y, PLOT_FLOOR),
        label="reference endpoints",
        linestyle="None",
        marker=".",
        markersize=2.5,
    )
    ax.set_yscale("log")
    ax.set_xlabel("rollout frame x inner iteration")
    ax.set_ylabel("stationarity residual")
    ax.set_title(
        f"motion {int(records[0]['motion_index']) if records else '?'}: "
        "rollout x iteration residual"
    )
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "rollout_x_iteration_vs_residual.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    frame_x = np.arange(rollout_length)
    for record in records:
        name = str(record["solver_name"])
        final = record["final_residual_by_frame"].numpy().astype(float)
        padded = np.full(rollout_length, np.inf, dtype=float)
        padded[: final.size] = final
        linestyle = "--" if name.startswith("baseline_") else "-"
        ax.plot(
            frame_x,
            np.maximum(padded, PLOT_FLOOR),
            label=name,
            linestyle=linestyle,
            linewidth=0.9,
        )
    ax.plot(
        frame_x,
        np.maximum(reference_record["endpoint_residual_by_frame"].numpy(), PLOT_FLOOR),
        label="reference",
        linestyle=":",
        linewidth=1.0,
    )
    ax.set_yscale("log")
    ax.set_xlabel("rollout frame")
    ax.set_ylabel(f"residual after iteration {inner_steps}")
    ax.set_title("per-frame final residual")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "rollout_frame_vs_final_residual.png", dpi=200)
    plt.close(fig)

    torch.save(all_curves, motion_dir / "all_curves.pt")


def evaluate_motion(
    *,
    motion_index: int,
    args: argparse.Namespace,
    source_root: Path,
    ablation_root: Path,
    physical,
    reference_states: dict[str, Any],
    device: torch.device,
    model_spec: ModelSpec,
) -> None:
    reference = reference_for_motion(reference_states, motion_index, args.rollout_length)
    motion_dir = ablation_root / "rollout_evaluation" / f"motion_{motion_index:03d}"
    motion_dir.mkdir(parents=True, exist_ok=True)
    torch.save(reference, motion_dir / f"reference_len_{args.rollout_length}.pt")

    reference_record = compute_reference_endpoint_record(
        reference=reference,
        physical=physical,
        device=device,
        rollout_length=args.rollout_length,
        inner_steps=args.inner_steps,
    )
    torch.save(reference_record, motion_dir / "reference_endpoints.pt")

    records: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    if not args.plot_only:
        for sample_count in args.sample_counts:
            model_dir = model_dir_for_count(ablation_root, int(sample_count), model_spec)
            model, model_info = load_model(model_dir, device, args.residual_length_scale)
            solver_name = f"model_points_{int(sample_count):04d}"
            output_path = motion_dir / solver_name / "curve.pt"
            record = run_solver_rollout(
                solver_name=solver_name,
                solver_info=model_info,
                solve_frame=lambda **kwargs: solve_model_frame(model=model, **kwargs),
                reference=reference,
                physical=physical,
                device=device,
                motion_index=motion_index,
                rollout_length=args.rollout_length,
                inner_steps=args.inner_steps,
                output_path=output_path,
                overwrite=args.overwrite,
                checkpoint_every=args.checkpoint_every,
            )
            records.append(record)
            summary_rows.append(summarize_record(record, args.rollout_length, args.inner_steps))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for method in args.baselines:
            params = load_baseline_params(source_root, method)
            solver_name = f"baseline_{method}"
            output_path = motion_dir / solver_name / "curve.pt"
            record = run_solver_rollout(
                solver_name=solver_name,
                solver_info={"method": method, "params": params},
                solve_frame=lambda method=method, params=params, **kwargs: solve_baseline_frame(
                    method=method, params=params, **kwargs
                ),
                reference=reference,
                physical=physical,
                device=device,
                motion_index=motion_index,
                rollout_length=args.rollout_length,
                inner_steps=args.inner_steps,
                output_path=output_path,
                overwrite=args.overwrite,
                checkpoint_every=args.checkpoint_every,
            )
            records.append(record)
            summary_rows.append(summarize_record(record, args.rollout_length, args.inner_steps))
    else:
        for path in sorted(motion_dir.glob("*/curve.pt")):
            record = torch.load(path, map_location="cpu")
            records.append(record)
            summary_rows.append(summarize_record(record, args.rollout_length, args.inner_steps))

    write_summary_csv(summary_rows, motion_dir / "summary_metrics.csv")
    plot_motion(
        motion_dir=motion_dir,
        records=records,
        reference_record=reference_record,
        rollout_length=args.rollout_length,
        inner_steps=args.inner_steps,
    )


def aggregate_all_motions(ablation_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted((ablation_root / "rollout_evaluation").glob("motion_*/summary_metrics.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if rows:
        write_summary_csv(rows, ablation_root / "rollout_evaluation" / "all_motion_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate initial-point ablation rollout curves.")
    parser.add_argument("--source-root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--ablation-root", type=Path, default=Path("cloth_5x5_initial_sample_ablation"))
    parser.add_argument("--sample-counts", type=int, nargs="+", default=list(DEFAULT_SAMPLE_COUNTS))
    parser.add_argument("--motion-indices", type=int, nargs="+", default=list(DEFAULT_TEST_MOTIONS))
    parser.add_argument(
        "--baselines",
        nargs="*",
        choices=DEFAULT_BASELINES,
        default=list(DEFAULT_BASELINES),
    )
    parser.add_argument("--rollout-length", type=int, default=DEFAULT_ROLLOUT_LENGTH)
    parser.add_argument("--inner-steps", type=int, default=DEFAULT_INNER_STEPS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--activation", choices=("identity", "relu", "tanh"), default="identity")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollout_length <= 0 or args.inner_steps <= 0 or args.checkpoint_every <= 0:
        raise ValueError("rollout settings must be positive")
    if any(index in range(0, 20) for index in args.motion_indices):
        raise ValueError("ablation test must use only motions outside train and validation (20-31)")

    source_root = args.source_root.resolve()
    ablation_root = args.ablation_root.resolve()
    device = torch.device(args.device)
    physical = load_physical(source_root)
    reference_states = load_reference_states(source_root)
    model_spec = ModelSpec(
        activation=args.activation,
        depth=int(args.depth),
        width=int(args.width),
        use_bias=bool(args.use_bias),
    )

    for motion_index in args.motion_indices:
        evaluate_motion(
            motion_index=int(motion_index),
            args=args,
            source_root=source_root,
            ablation_root=ablation_root,
            physical=physical,
            reference_states=reference_states,
            device=device,
            model_spec=model_spec,
        )
    aggregate_all_motions(ablation_root)


if __name__ == "__main__":
    main()
