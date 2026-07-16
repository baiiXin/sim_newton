"""Reference-free free-rollout validation for a learned T-shirt optimizer.

The fixed validation/test arrays contain initial states only.  No reference
trajectory or converged solution is required: quality is measured from the
implicit objective's stationarity residual and geometric failure checks.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch

from cloth02_batched_physics import FrozenMotionBatch, TShirtPhysics
from cloth03_training_pool import LearnedOptimizerMLP, apply_model_update
from tshirt_config import write_json
from validation_protocol import ValidationProtocol


@dataclass(frozen=True)
class FailureThresholds:
    max_residual: float = 1e12
    max_abs_position: float = 1e4
    min_area_ratio: float = 1e-3
    max_area_ratio: float = 1e3
    min_edge_ratio: float = 1e-3
    max_edge_ratio: float = 1e3
    max_constraint_error: float = 1e-9


@dataclass
class ValidationResult:
    protocol: ValidationProtocol
    summary: dict[str, Any]
    per_motion: list[dict[str, Any]]
    curves: dict[str, np.ndarray]


def _finite_quantile(values: np.ndarray, quantile: float, default: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if finite.size else float(default)


def _finite_mean(values: np.ndarray, default: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float(default)


def detect_failures(
    physics: TShirtPhysics,
    positions: torch.Tensor,
    residual: torch.Tensor,
    fixed_targets: torch.Tensor,
    thresholds: FailureThresholds,
) -> tuple[torch.Tensor, list[list[str]], dict[str, torch.Tensor]]:
    finite = torch.isfinite(positions).flatten(start_dim=1).all(dim=1) & torch.isfinite(residual)
    safe = torch.where(torch.isfinite(positions), positions, fixed_targets)
    area = physics.triangle_area_ratios(safe)
    edge = physics.edge_length_ratios(safe)
    fixed_error = torch.linalg.vector_norm(
        (safe[:, physics.fixed_mask] - fixed_targets[:, physics.fixed_mask]).reshape(safe.shape[0], -1),
        dim=-1,
    )
    checks = {
        "nonfinite": ~finite,
        "residual": finite & (residual > thresholds.max_residual),
        "position": finite & (safe.abs().amax(dim=(-2, -1)) > thresholds.max_abs_position),
        "area": finite & (
            (area.amin(dim=-1) < thresholds.min_area_ratio)
            | (area.amax(dim=-1) > thresholds.max_area_ratio)
        ),
        "edge": finite & (
            (edge.amin(dim=-1) < thresholds.min_edge_ratio)
            | (edge.amax(dim=-1) > thresholds.max_edge_ratio)
        ),
        "constraint": finite & (fixed_error > thresholds.max_constraint_error),
    }
    failed = torch.zeros_like(finite)
    reasons: list[list[str]] = [[] for _ in range(positions.shape[0])]
    for name, mask in checks.items():
        failed |= mask
        for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            reasons[index].append(name)
    diagnostics = {
        "area_min": area.amin(dim=-1),
        "area_max": area.amax(dim=-1),
        "edge_min": edge.amin(dim=-1),
        "edge_max": edge.amax(dim=-1),
        "constraint_error": fixed_error,
    }
    return failed, reasons, diagnostics


def _evaluate_chunk(
    *,
    model: LearnedOptimizerMLP,
    physics: TShirtPhysics,
    motions: FrozenMotionBatch,
    protocol: ValidationProtocol,
    thresholds: FailureThresholds,
) -> dict[str, Any]:
    batch_size = motions.batch_size
    frames = protocol.rollout_frames
    shape = (batch_size, frames)
    initial_residual = np.full(shape, np.nan, dtype=np.float64)
    final_residual = np.full(shape, np.nan, dtype=np.float64)
    residual_ratio = np.full(shape, np.nan, dtype=np.float64)
    first_step_ratio = np.full(shape, np.nan, dtype=np.float64)
    inner_steps_used = np.zeros(shape, dtype=np.int64)
    energy_change = np.full(shape, np.nan, dtype=np.float64)
    area_min = np.full(shape, np.nan, dtype=np.float64)
    area_max = np.full(shape, np.nan, dtype=np.float64)
    edge_min = np.full(shape, np.nan, dtype=np.float64)
    edge_max = np.full(shape, np.nan, dtype=np.float64)
    failed = np.zeros(shape, dtype=np.bool_)
    failure_frame = np.full(batch_size, frames, dtype=np.int64)
    failure_reason: list[str] = [""] * batch_size
    invalid_candidate = torch.zeros(batch_size, dtype=torch.bool, device=physics.device)
    solver_stopped = torch.zeros(batch_size, dtype=torch.bool, device=physics.device)

    p = motions.positions.detach().clone()
    v = motions.velocities.detach().clone()
    fixed_targets = motions.positions.detach().clone()
    alive = torch.ones(batch_size, dtype=torch.bool, device=physics.device)
    epsilon = torch.finfo(physics.dtype).eps

    model.eval()
    for frame in range(frames):
        if not bool(alive.any()):
            break
        q = physics.make_q(p, v)
        y = physics.project_positions(p, fixed_targets)
        previous_residual = torch.zeros(
            (batch_size, model.full_state_dim), dtype=physics.dtype, device=physics.device
        )
        previous_update = torch.zeros_like(previous_residual)
        residual0 = physics.stationarity_residual_norm(y, q, fixed_targets).detach()
        energy0 = physics.variational_energy(y, q, fixed_targets).detach()
        target = torch.maximum(
            residual0 * protocol.residual_ratio_tolerance,
            torch.full_like(residual0, protocol.absolute_residual_tolerance),
        )
        converged = residual0 <= target
        first_ratio = torch.zeros_like(residual0)
        steps = torch.zeros(batch_size, dtype=torch.long, device=physics.device)

        for inner in range(protocol.inner_steps):
            active = alive & ~solver_stopped
            if protocol.early_stop:
                active &= ~converged
            if not bool(active.any()):
                break
            with torch.no_grad():
                candidate, delta, current = apply_model_update(
                    model,
                    y,
                    q,
                    fixed_targets,
                    previous_residual=previous_residual,
                    previous_update=previous_update,
                )
            candidate_finite = torch.isfinite(candidate).flatten(start_dim=1).all(dim=1)
            invalid_candidate |= active & ~candidate_finite
            solver_stopped |= active & ~candidate_finite
            accepted = active & candidate_finite
            y = torch.where(accepted.view(-1, 1, 1), candidate, y)
            residual_now = physics.stationarity_residual_norm(y, q, fixed_targets).detach()
            if inner == 0:
                first_ratio = torch.where(
                    active,
                    residual_now / residual0.clamp_min(epsilon),
                    first_ratio,
                )
            steps += active.to(torch.long)
            previous_residual = torch.where(active.view(-1, 1), current, previous_residual)
            previous_update = torch.where(active.view(-1, 1), delta, previous_update)
            converged |= residual_now <= target
            # Non-finite candidates are recorded by the common failure checks
            # below while y itself stays finite to avoid poisoning other rows.
            if bool((active & ~candidate_finite).any()):
                residual_now = torch.where(
                    active & ~candidate_finite,
                    torch.full_like(residual_now, math.inf),
                    residual_now,
                )
                converged |= active & ~candidate_finite

        residual1 = physics.stationarity_residual_norm(y, q, fixed_targets).detach()
        energy1 = physics.variational_energy(y, q, fixed_targets).detach()
        bad, reasons, diagnostics = detect_failures(
            physics, y, residual1, fixed_targets, thresholds
        )
        bad |= invalid_candidate
        for index in torch.nonzero(invalid_candidate, as_tuple=False).flatten().tolist():
            reasons[index].append("nonfinite_solver_update")
        newly_failed = alive & bad
        alive_after = alive & ~bad

        alive_np = alive.detach().cpu().numpy()
        initial_residual[alive_np, frame] = residual0[alive].cpu().numpy()
        final_residual[alive_np, frame] = residual1[alive].cpu().numpy()
        residual_ratio[alive_np, frame] = (
            residual1[alive] / residual0[alive].clamp_min(epsilon)
        ).cpu().numpy()
        first_step_ratio[alive_np, frame] = first_ratio[alive].cpu().numpy()
        inner_steps_used[alive_np, frame] = steps[alive].cpu().numpy()
        energy_change[alive_np, frame] = (energy1[alive] - energy0[alive]).cpu().numpy()
        for name, destination in (
            ("area_min", area_min), ("area_max", area_max),
            ("edge_min", edge_min), ("edge_max", edge_max),
        ):
            destination[alive_np, frame] = diagnostics[name][alive].detach().cpu().numpy()
        failed[:, frame] = ~alive_after.detach().cpu().numpy()

        for index in torch.nonzero(newly_failed, as_tuple=False).flatten().tolist():
            failure_frame[index] = frame
            failure_reason[index] = "+".join(reasons[index]) or "unknown"

        if bool(alive_after.any()):
            p_next, v_next = physics.advance_state(p, y, fixed_targets)
            p = torch.where(alive_after.view(-1, 1, 1), p_next, p)
            v = torch.where(alive_after.view(-1, 1, 1), v_next, torch.zeros_like(v))
        alive = alive_after

    for index, frame in enumerate(failure_frame.tolist()):
        if frame < frames:
            failed[index, frame:] = True
    return {
        "motion_ids": motions.motion_ids,
        "initial_residual": initial_residual,
        "final_residual": final_residual,
        "residual_ratio": residual_ratio,
        "first_step_ratio": first_step_ratio,
        "inner_steps_used": inner_steps_used,
        "energy_change": energy_change,
        "area_min": area_min,
        "area_max": area_max,
        "edge_min": edge_min,
        "edge_max": edge_max,
        "failed": failed,
        "failure_frame": failure_frame,
        "failure_reason": failure_reason,
    }


def _slice_motions(motions: FrozenMotionBatch, start: int, stop: int) -> FrozenMotionBatch:
    return FrozenMotionBatch(
        motion_ids=motions.motion_ids[start:stop],
        positions=motions.positions[start:stop],
        velocities=motions.velocities[start:stop],
        seeds=motions.seeds[start:stop],
    )


def run_reference_free_validation(
    *,
    model: LearnedOptimizerMLP,
    physics: TShirtPhysics,
    motions: FrozenMotionBatch,
    protocol: ValidationProtocol,
    batch_size: int = 4,
    thresholds: FailureThresholds = FailureThresholds(),
) -> ValidationResult:
    if protocol.motion_count > motions.batch_size:
        raise ValueError(
            f"protocol asks for {protocol.motion_count} motions, dataset has {motions.batch_size}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    count = protocol.motion_count
    chunks = [
        _evaluate_chunk(
            model=model,
            physics=physics,
            motions=_slice_motions(motions, start, min(start + batch_size, count)),
            protocol=protocol,
            thresholds=thresholds,
        )
        for start in range(0, count, batch_size)
    ]
    keys = (
        "initial_residual", "final_residual", "residual_ratio", "first_step_ratio",
        "inner_steps_used", "energy_change", "area_min", "area_max", "edge_min",
        "edge_max", "failed", "failure_frame",
    )
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}
    motion_ids = tuple(item for chunk in chunks for item in chunk["motion_ids"])
    failure_reason = [item for chunk in chunks for item in chunk["failure_reason"]]
    valid_frames = np.isfinite(arrays["residual_ratio"])
    converged = valid_frames & (
        arrays["residual_ratio"] <= protocol.residual_ratio_tolerance
    )
    slow_single_step = valid_frames & (
        arrays["first_step_ratio"] >= protocol.single_step_ratio_threshold
    )
    energy_increase = np.isfinite(arrays["energy_change"]) & (arrays["energy_change"] > 0.0)
    failed_motion = arrays["failure_frame"] < protocol.rollout_frames

    per_motion: list[dict[str, Any]] = []
    for index, motion_id in enumerate(motion_ids):
        mask = valid_frames[index]
        slow_count = int(slow_single_step[index].sum())
        per_motion.append(
            {
                "motion_index": index,
                "motion_id": motion_id,
                "failed": bool(failed_motion[index]),
                "failure_frame": int(arrays["failure_frame"][index]),
                "failure_reason": failure_reason[index],
                "survival_frames": int(min(arrays["failure_frame"][index], protocol.rollout_frames)),
                "evaluated_frame_count": int(mask.sum()),
                "residual_ratio_median": _finite_quantile(arrays["residual_ratio"][index], 0.5, math.inf),
                "residual_ratio_p95": _finite_quantile(arrays["residual_ratio"][index], 0.95, math.inf),
                "converged_frame_fraction": float(converged[index].sum() / max(int(mask.sum()), 1)),
                "inner_steps_mean": _finite_mean(arrays["inner_steps_used"][index][mask], math.inf),
                "single_step_le_two_orders_frame_count": slow_count,
                "single_step_le_two_orders_frame_fraction": float(slow_count / max(int(mask.sum()), 1)),
                "min_area_ratio": _finite_quantile(arrays["area_min"][index], 0.0, 0.0),
                "max_area_ratio": _finite_quantile(arrays["area_max"][index], 1.0, math.inf),
                "min_edge_ratio": _finite_quantile(arrays["edge_min"][index], 0.0, 0.0),
                "max_edge_ratio": _finite_quantile(arrays["edge_max"][index], 1.0, math.inf),
            }
        )

    total_valid = max(int(valid_frames.sum()), 1)
    survival = np.minimum(arrays["failure_frame"], protocol.rollout_frames)
    summary: dict[str, Any] = {
        "protocol_id": protocol.id,
        "motion_count": count,
        "rollout_frames": protocol.rollout_frames,
        "inner_steps_cap": protocol.inner_steps,
        "residual_ratio_tolerance": protocol.residual_ratio_tolerance,
        "single_step_ratio_threshold": protocol.single_step_ratio_threshold,
        "early_stop": protocol.early_stop,
        "fixed_inner_iteration_budget": not protocol.early_stop,
        "failed_motion_count": int(failed_motion.sum()),
        "failed_motion_fraction": float(failed_motion.mean()),
        "survival_frame_p05": _finite_quantile(survival, 0.05, 0.0),
        "residual_ratio_median": _finite_quantile(arrays["residual_ratio"], 0.50, math.inf),
        "residual_ratio_p95": _finite_quantile(arrays["residual_ratio"], 0.95, math.inf),
        "residual_ratio_max": _finite_quantile(arrays["residual_ratio"], 1.0, math.inf),
        "final_residual_p95": _finite_quantile(arrays["final_residual"], 0.95, math.inf),
        "converged_frame_count": int(converged.sum()),
        "converged_frame_fraction": float(converged.sum() / total_valid),
        "inner_steps_mean": _finite_mean(arrays["inner_steps_used"][valid_frames], math.inf),
        "inner_steps_p95": _finite_quantile(arrays["inner_steps_used"][valid_frames], 0.95, math.inf),
        "single_step_le_two_orders_frame_count": int(slow_single_step.sum()),
        "single_step_le_two_orders_frame_fraction": float(slow_single_step.sum() / total_valid),
        "energy_increase_fraction": float(energy_increase.sum() / total_valid),
        "min_area_ratio": _finite_quantile(arrays["area_min"], 0.0, 0.0),
        "max_area_ratio": _finite_quantile(arrays["area_max"], 1.0, math.inf),
        "min_edge_ratio": _finite_quantile(arrays["edge_min"], 0.0, 0.0),
        "max_edge_ratio": _finite_quantile(arrays["edge_max"], 1.0, math.inf),
    }

    curves: dict[str, np.ndarray] = {**arrays}
    curves["residual_ratio_median_by_frame"] = np.asarray([
        _finite_quantile(arrays["residual_ratio"][:, frame], 0.5, np.nan)
        for frame in range(protocol.rollout_frames)
    ])
    curves["residual_ratio_p95_by_frame"] = np.asarray([
        _finite_quantile(arrays["residual_ratio"][:, frame], 0.95, np.nan)
        for frame in range(protocol.rollout_frames)
    ])
    curves["single_step_slow_count_by_frame"] = slow_single_step.sum(axis=0)
    curves["alive_count_by_frame"] = (~arrays["failed"]).sum(axis=0)
    return ValidationResult(protocol, summary, per_motion, curves)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def _plot_result(result: ValidationResult, output_dir: Path, update: int) -> list[str]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = np.arange(result.protocol.rollout_frames)
    paths: list[str] = []

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.semilogy(frames, result.curves["residual_ratio_median_by_frame"], label="median")
    axis.semilogy(frames, result.curves["residual_ratio_p95_by_frame"], label="p95")
    axis.axhline(result.protocol.residual_ratio_tolerance, color="black", ls="--", lw=1)
    axis.set(xlabel="physical frame", ylabel="final / initial residual", title=result.protocol.id)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    path = output_dir / f"update_{update:09d}_residual_ratio.png"
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)
    paths.append(str(path))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(frames, result.curves["alive_count_by_frame"])
    axes[0].set(xlabel="physical frame", ylabel="alive motions", title="rollout survival")
    axes[1].plot(frames, result.curves["single_step_slow_count_by_frame"])
    axes[1].set(
        xlabel="physical frame",
        ylabel="frame count",
        title=f"first-step ratio >= {result.protocol.single_step_ratio_threshold:g}",
    )
    for axis in axes:
        axis.grid(True, alpha=0.25)
    path = output_dir / f"update_{update:09d}_survival_and_single_step.png"
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)
    paths.append(str(path))
    return paths


def save_validation_result(
    *,
    result: ValidationResult,
    output_root: Path,
    update: int,
    render_plots: bool = True,
) -> Path:
    root = Path(output_root) / "validation" / result.protocol.id
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    history_row = {"update": int(update), "timestamp_utc": timestamp, **result.summary}
    _write_csv(root / "history.csv", [history_row], append=True)
    per_motion = [
        {"update": int(update), "timestamp_utc": timestamp, **row}
        for row in result.per_motion
    ]
    _write_csv(root / f"per_motion_update_{update:09d}.csv", per_motion)
    _write_csv(root / "per_motion_latest.csv", per_motion)
    np.savez_compressed(root / f"curves_update_{update:09d}.npz", **result.curves)
    payload = {
        "update": int(update),
        "timestamp_utc": timestamp,
        "protocol": asdict(result.protocol),
        "summary": result.summary,
        "failure_metric_definition": (
            "single_step_le_two_orders counts evaluated frames whose residual after "
            "one learned update divided by its initial residual is >= threshold"
        ),
    }
    if render_plots:
        payload["figures"] = _plot_result(result, root / "figures", update)
    write_json(root / f"summary_update_{update:09d}.json", payload)
    write_json(root / "summary_latest.json", payload)
    return root
