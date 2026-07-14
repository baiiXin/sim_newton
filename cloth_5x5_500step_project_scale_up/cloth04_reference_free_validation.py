"""Reference-free continuous-rollout validation for scale-up cloth training."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cloth02_batched_physics import (
    advance_state,
    build_batched_parameters,
    dirichlet_targets,
    make_q,
    project_positions,
    spring_lengths,
    stationarity_residual_norm,
    variational_energy,
)
from cloth03_training_pool import LearnedOptimizerMLP, apply_model_update
from scenario_templates import ScenarioSpec
from validation_protocol import ValidationProtocol


@dataclass(frozen=True)
class FailureThresholds:
    max_residual: float = 1e12
    max_abs_position: float = 1e4
    min_edge_ratio: float = 1e-5
    max_edge_ratio: float = 1e4
    max_constraint_error: float = 1e-9


@dataclass
class ValidationResult:
    protocol: dict[str, Any]
    summary: dict[str, Any]
    per_motion: list[dict[str, Any]]
    curves: dict[str, torch.Tensor]
    raw: dict[str, torch.Tensor]


def _finite_quantile(values: torch.Tensor, q: float, default: float) -> float:
    values = values.detach().double().flatten()
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float(default)
    return float(torch.quantile(finite, q).item())


def checkpoint_rank(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    """Stability-first lexicographic ordering; smaller tuple is better."""
    return (
        float(summary["failed_motion_count"]),
        -float(summary["survival_frame_p05"]),
        float(summary["residual_ratio_p95"]),
        float(summary["energy_increase_fraction"]),
    )


@torch.no_grad()
def _run_rollout_chunk(
    *,
    model: LearnedOptimizerMLP,
    scenarios: Sequence[ScenarioSpec],
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
) -> ValidationResult:
    params = build_batched_parameters(
        scenarios,
        device=device,
        dtype=dtype,
    )
    batch_size = params.batch_size
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    alive = torch.ones(batch_size, dtype=torch.bool, device=device)
    failure_frame = torch.full(
        (batch_size,),
        rollout_frames,
        dtype=torch.long,
        device=device,
    )
    failure_code = torch.zeros(batch_size, dtype=torch.long, device=device)
    # 0 none, 1 nonfinite, 2 residual, 3 position, 4 edge, 5 constraint

    residual_initial = torch.full(
        (batch_size, rollout_frames),
        float("nan"),
        dtype=dtype,
        device=device,
    )
    residual_final = torch.full_like(residual_initial, float("nan"))
    residual_ratio = torch.full_like(residual_initial, float("nan"))
    energy_before = torch.full_like(residual_initial, float("nan"))
    energy_after = torch.full_like(residual_initial, float("nan"))
    normalized_energy_change = torch.full_like(residual_initial, float("nan"))
    edge_ratio_min = torch.full_like(residual_initial, float("nan"))
    edge_ratio_max = torch.full_like(residual_initial, float("nan"))
    constraint_error = torch.full_like(residual_initial, float("nan"))
    alive_by_frame = torch.zeros(
        batch_size,
        rollout_frames + 1,
        dtype=torch.bool,
        device=device,
    )
    alive_by_frame[:, 0] = True
    inner_residual = torch.full(
        (batch_size, rollout_frames, inner_steps + 1),
        float("nan"),
        dtype=dtype,
        device=device,
    )

    failure_names = {
        0: "",
        1: "nonfinite",
        2: "residual",
        3: "position",
        4: "edge_ratio",
        5: "constraint",
    }
    eps = torch.finfo(dtype).eps

    for frame in range(rollout_frames):
        q = make_q(p, v, params)
        next_time = torch.full(
            (batch_size,),
            (frame + 1) * params.dt,
            dtype=dtype,
            device=device,
        )
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(p, params, targets)
        previous_residual = torch.zeros(
            batch_size,
            params.full_state_dim,
            dtype=dtype,
            device=device,
        )
        previous_update = torch.zeros_like(previous_residual)

        r0 = stationarity_residual_norm(y, q, params, targets)
        e0 = variational_energy(y, q, params, targets)
        residual_initial[:, frame] = torch.where(
            alive, r0, torch.full_like(r0, float("nan"))
        )
        energy_before[:, frame] = torch.where(
            alive, e0, torch.full_like(e0, float("nan"))
        )
        inner_residual[:, frame, 0] = torch.where(
            alive, r0, torch.full_like(r0, float("nan"))
        )

        for inner in range(inner_steps):
            y_candidate, delta, current = apply_model_update(
                model,
                y,
                q,
                params,
                target_positions=targets,
                previous_residual=previous_residual,
                previous_update=previous_update,
            )
            alive_points = alive[:, None, None]
            y = torch.where(
                alive_points,
                y_candidate.reshape(batch_size, params.num_vertices, 3),
                y,
            )
            previous_residual = torch.where(
                alive[:, None],
                current,
                previous_residual,
            )
            previous_update = torch.where(
                alive[:, None],
                delta,
                previous_update,
            )
            r_inner = stationarity_residual_norm(y, q, params, targets)
            inner_residual[:, frame, inner + 1] = torch.where(
                alive,
                r_inner,
                torch.full_like(r_inner, float("nan")),
            )

        r1 = stationarity_residual_norm(y, q, params, targets)
        e1 = variational_energy(y, q, params, targets)
        lengths = spring_lengths(y, params, targets)
        ratios = lengths / params.rest_lengths.clamp_min(eps)
        projected = project_positions(y, params, targets)
        constraint = torch.where(
            params.fixed_mask.unsqueeze(-1),
            (projected - targets).abs(),
            torch.zeros_like(projected),
        ).amax(dim=(-2, -1))

        finite = (
            torch.isfinite(y).flatten(start_dim=1).all(dim=1)
            & torch.isfinite(r1)
            & torch.isfinite(e1)
            & torch.isfinite(lengths).all(dim=-1)
        )
        residual_bad = r1 > thresholds.max_residual
        position_bad = y.abs().amax(dim=(-2, -1)) > thresholds.max_abs_position
        edge_bad = (
            ratios.amin(dim=-1) < thresholds.min_edge_ratio
        ) | (
            ratios.amax(dim=-1) > thresholds.max_edge_ratio
        )
        constraint_bad = constraint > thresholds.max_constraint_error
        failed_now = alive & (
            ~finite | residual_bad | position_bad | edge_bad | constraint_bad
        )

        code = torch.zeros_like(failure_code)
        code = torch.where(~finite, torch.ones_like(code), code)
        code = torch.where(
            finite & residual_bad,
            torch.full_like(code, 2),
            code,
        )
        code = torch.where(
            finite & ~residual_bad & position_bad,
            torch.full_like(code, 3),
            code,
        )
        code = torch.where(
            finite & ~residual_bad & ~position_bad & edge_bad,
            torch.full_like(code, 4),
            code,
        )
        code = torch.where(
            finite
            & ~residual_bad
            & ~position_bad
            & ~edge_bad
            & constraint_bad,
            torch.full_like(code, 5),
            code,
        )
        failure_frame = torch.where(
            failed_now,
            torch.full_like(failure_frame, frame),
            failure_frame,
        )
        failure_code = torch.where(failed_now, code, failure_code)

        valid_before = alive
        residual_final[:, frame] = torch.where(
            valid_before, r1, torch.full_like(r1, float("nan"))
        )
        ratio_value = r1 / (r0 + eps)
        residual_ratio[:, frame] = torch.where(
            valid_before,
            ratio_value,
            torch.full_like(ratio_value, float("nan")),
        )
        energy_after[:, frame] = torch.where(
            valid_before, e1, torch.full_like(e1, float("nan"))
        )
        energy_scale = (
            params.masses.mean(dim=-1)
            * params.rest_lengths.mean(dim=-1).square()
            / params.dt**2
            + (
                params.spring_stiffness
                * params.rest_lengths.square()
            ).mean(dim=-1)
        ).clamp_min(eps)
        normalized_change = (e1 - e0) / energy_scale
        normalized_energy_change[:, frame] = torch.where(
            valid_before,
            normalized_change,
            torch.full_like(normalized_change, float("nan")),
        )
        edge_ratio_min[:, frame] = torch.where(
            valid_before,
            ratios.amin(dim=-1),
            torch.full_like(r1, float("nan")),
        )
        edge_ratio_max[:, frame] = torch.where(
            valid_before,
            ratios.amax(dim=-1),
            torch.full_like(r1, float("nan")),
        )
        constraint_error[:, frame] = torch.where(
            valid_before,
            constraint,
            torch.full_like(r1, float("nan")),
        )

        alive = alive & ~failed_now
        alive_by_frame[:, frame + 1] = alive
        safe_y = torch.where(alive[:, None, None], y, p)
        p_candidate, v_candidate = advance_state(
            p,
            safe_y,
            params,
            next_time=next_time,
        )
        p = torch.where(alive[:, None, None], p_candidate, p)
        v = torch.where(alive[:, None, None], v_candidate, v)

    per_motion: list[dict[str, Any]] = []
    trajectory_selection_ratio = []
    trajectory_energy_fraction = []
    for row, scenario in enumerate(scenarios):
        survived = int(failure_frame[row].item())
        failed = survived < rollout_frames
        valid_frames = torch.arange(rollout_frames, device=device) < max(
            survived if failed else rollout_frames,
            0,
        )
        ratios = residual_ratio[row, valid_frames]
        changes = normalized_energy_change[row, valid_frames]
        diagnostic_p95 = _finite_quantile(ratios, 0.95, float("inf"))
        selection_ratio = float("inf") if failed else diagnostic_p95
        energy_fraction = (
            float((changes > 0).double().mean().item())
            if changes.numel() and bool(torch.isfinite(changes).any())
            else 1.0
        )
        if failed:
            energy_fraction = max(energy_fraction, 1.0)
        trajectory_selection_ratio.append(selection_ratio)
        trajectory_energy_fraction.append(energy_fraction)
        valid_final = residual_final[row, valid_frames]
        final_residual = (
            float(valid_final[-1].item())
            if valid_final.numel() and torch.isfinite(valid_final[-1])
            else float("inf")
        )
        per_motion.append(
            {
                "scenario_id": int(scenario.scenario_id),
                "scenario_group": str(scenario.group),
                "boundary_id": str(scenario.boundary_id),
                "material_id": str(scenario.material_id),
                "failed": bool(failed),
                "failure_frame": None if not failed else survived,
                "survival_frames": survived,
                "failure_reason": failure_names[int(failure_code[row].item())],
                "residual_ratio_p95_diagnostic": diagnostic_p95,
                "residual_ratio_selection": selection_ratio,
                "final_residual": final_residual,
                "energy_increase_fraction": energy_fraction,
                "minimum_edge_ratio": _finite_quantile(
                    edge_ratio_min[row, valid_frames], 0.0, 0.0
                ),
                "maximum_edge_ratio": _finite_quantile(
                    edge_ratio_max[row, valid_frames], 1.0, float("inf")
                ),
                "maximum_constraint_error": _finite_quantile(
                    constraint_error[row, valid_frames], 1.0, float("inf")
                ),
            }
        )

    survival = failure_frame.double()
    selection_tensor = torch.as_tensor(
        trajectory_selection_ratio,
        dtype=torch.float64,
    )
    energy_fraction_tensor = torch.as_tensor(
        trajectory_energy_fraction,
        dtype=torch.float64,
    )
    failed_count = int((failure_frame < rollout_frames).sum().item())
    summary = {
        "motion_count": batch_size,
        "rollout_frames": rollout_frames,
        "inner_steps": inner_steps,
        "failed_motion_count": failed_count,
        "survival_rate": float((failure_frame >= rollout_frames).double().mean().item()),
        "survival_frame_p05": float(torch.quantile(survival, 0.05).item()),
        "survival_frame_median": float(torch.quantile(survival, 0.50).item()),
        "residual_ratio_p95": float(torch.quantile(selection_tensor, 0.95).item()),
        "energy_increase_fraction": float(energy_fraction_tensor.mean().item()),
    }

    frame_residual_p50 = torch.full((rollout_frames,), float("nan"), dtype=torch.float64)
    frame_residual_p95 = torch.full_like(frame_residual_p50, float("nan"))
    frame_energy_increase = torch.full_like(frame_residual_p50, float("nan"))
    for frame in range(rollout_frames):
        frame_residual_p50[frame] = _finite_quantile(
            residual_ratio[:, frame], 0.50, float("nan")
        )
        frame_residual_p95[frame] = _finite_quantile(
            residual_ratio[:, frame], 0.95, float("nan")
        )
        change = normalized_energy_change[:, frame]
        finite = torch.isfinite(change)
        if bool(finite.any()):
            frame_energy_increase[frame] = (
                (change[finite] > 0).double().mean().cpu()
            )
    curves = {
        "frame": torch.arange(rollout_frames, dtype=torch.long),
        "alive_count": alive_by_frame[:, 1:].sum(dim=0).detach().cpu(),
        "failed_cumulative": (
            batch_size - alive_by_frame[:, 1:].sum(dim=0)
        ).detach().cpu(),
        "residual_ratio_p50": frame_residual_p50,
        "residual_ratio_p95": frame_residual_p95,
        "energy_increase_fraction": frame_energy_increase,
        "inner_residual": inner_residual.detach().cpu(),
    }
    raw = {
        "scenario_ids": params.scenario_ids.detach().cpu(),
        "failure_frame": failure_frame.detach().cpu(),
        "failure_code": failure_code.detach().cpu(),
        "residual_initial": residual_initial.detach().cpu(),
        "residual_final": residual_final.detach().cpu(),
        "residual_ratio": residual_ratio.detach().cpu(),
        "energy_before": energy_before.detach().cpu(),
        "energy_after": energy_after.detach().cpu(),
        "normalized_energy_change": normalized_energy_change.detach().cpu(),
        "edge_ratio_min": edge_ratio_min.detach().cpu(),
        "edge_ratio_max": edge_ratio_max.detach().cpu(),
        "constraint_error": constraint_error.detach().cpu(),
        "alive_by_frame": alive_by_frame.detach().cpu(),
    }
    return ValidationResult(
        protocol={},
        summary=summary,
        per_motion=per_motion,
        curves=curves,
        raw=raw,
    )


@torch.no_grad()
def run_reference_free_validation(
    *,
    model: LearnedOptimizerMLP,
    scenarios: Sequence[ScenarioSpec],
    protocol: ValidationProtocol,
    device: torch.device | str,
    dtype: torch.dtype = torch.float64,
    batch_size: int = 32,
    thresholds: FailureThresholds = FailureThresholds(),
) -> ValidationResult:
    selected = tuple(scenarios[: protocol.motion_count])
    if len(selected) != protocol.motion_count:
        raise ValueError(
            f"Protocol {protocol.id} needs {protocol.motion_count} scenarios, "
            f"but only {len(scenarios)} were provided"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_was_training = model.training
    model.eval()
    chunks: list[ValidationResult] = []
    for start in range(0, len(selected), batch_size):
        chunks.append(
            _run_rollout_chunk(
                model=model,
                scenarios=selected[start : start + batch_size],
                rollout_frames=protocol.rollout_frames,
                inner_steps=protocol.inner_steps,
                device=torch.device(device),
                dtype=dtype,
                thresholds=thresholds,
            )
        )
    if model_was_training:
        model.train()

    per_motion = [
        row
        for chunk in chunks
        for row in chunk.per_motion
    ]
    raw_keys = chunks[0].raw.keys()
    raw = {
        key: torch.cat([chunk.raw[key] for chunk in chunks], dim=0)
        for key in raw_keys
    }
    residual_ratio = raw["residual_ratio"].double()
    normalized_change = raw["normalized_energy_change"].double()
    alive_by_frame = raw["alive_by_frame"].bool()
    rollout_frames = protocol.rollout_frames
    frame_p50 = torch.full((rollout_frames,), float("nan"), dtype=torch.float64)
    frame_p95 = torch.full_like(frame_p50, float("nan"))
    frame_energy = torch.full_like(frame_p50, float("nan"))
    for frame in range(rollout_frames):
        frame_p50[frame] = _finite_quantile(
            residual_ratio[:, frame], 0.50, float("nan")
        )
        frame_p95[frame] = _finite_quantile(
            residual_ratio[:, frame], 0.95, float("nan")
        )
        values = normalized_change[:, frame]
        finite = torch.isfinite(values)
        if bool(finite.any()):
            frame_energy[frame] = (values[finite] > 0).double().mean()
    curves = {
        "frame": torch.arange(rollout_frames, dtype=torch.long),
        "alive_count": alive_by_frame[:, 1:].sum(dim=0),
        "failed_cumulative": len(selected) - alive_by_frame[:, 1:].sum(dim=0),
        "residual_ratio_p50": frame_p50,
        "residual_ratio_p95": frame_p95,
        "energy_increase_fraction": frame_energy,
        "inner_residual": torch.cat(
            [chunk.curves["inner_residual"] for chunk in chunks],
            dim=0,
        ),
    }
    failure_frame = raw["failure_frame"].double()
    selection_ratio = torch.as_tensor(
        [row["residual_ratio_selection"] for row in per_motion],
        dtype=torch.float64,
    )
    summary = {
        "motion_count": len(selected),
        "rollout_frames": protocol.rollout_frames,
        "inner_steps": protocol.inner_steps,
        "failed_motion_count": int(
            sum(bool(row["failed"]) for row in per_motion)
        ),
        "survival_rate": float(
            sum(not bool(row["failed"]) for row in per_motion) / len(per_motion)
        ),
        "survival_frame_p05": float(torch.quantile(failure_frame, 0.05).item()),
        "survival_frame_median": float(torch.quantile(failure_frame, 0.50).item()),
        "residual_ratio_p95": float(torch.quantile(selection_ratio, 0.95).item()),
        "energy_increase_fraction": float(
            np.mean([row["energy_increase_fraction"] for row in per_motion])
        ),
    }
    return ValidationResult(
        protocol=asdict(protocol),
        summary=summary,
        per_motion=per_motion,
        curves=curves,
        raw=raw,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(value),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    combined = existing + rows
    fields: list[str] = []
    for row in combined:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)


def _plot_validation(
    *,
    result: ValidationResult,
    figure_dir: Path,
    update_count: int,
) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = result.curves["frame"].numpy()

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(frame, result.curves["residual_ratio_p50"].numpy(), label="p50")
    plt.plot(frame, result.curves["residual_ratio_p95"].numpy(), label="p95")
    plt.yscale("log")
    plt.xlabel("Physical frame")
    plt.ylabel("Residual ratio")
    plt.title(f"{result.protocol['id']} residual ratio — update {update_count}")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / f"residual_ratio_update_{update_count:09d}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(frame, result.curves["alive_count"].numpy())
    plt.xlabel("Physical frame")
    plt.ylabel("Alive motions")
    plt.title(f"{result.protocol['id']} survival — update {update_count}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / f"survival_update_{update_count:09d}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(frame, result.curves["energy_increase_fraction"].numpy())
    plt.xlabel("Physical frame")
    plt.ylabel("Energy increase fraction")
    plt.ylim(-0.02, 1.02)
    plt.title(f"{result.protocol['id']} energy — update {update_count}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / f"energy_update_{update_count:09d}.png", dpi=180)
    plt.close()


def save_validation_result(
    *,
    result: ValidationResult,
    output_root: Path,
    update_count: int,
    wall_clock_seconds: float,
    render_plots: bool = True,
) -> None:
    protocol_id = str(result.protocol["id"])
    root = output_root / "validation" / protocol_id
    run_dir = root / "runs" / f"update_{update_count:09d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history_row = {
        "update_count": int(update_count),
        "wall_clock_seconds": float(wall_clock_seconds),
        **result.summary,
    }
    _append_csv(root / "history.csv", [history_row])
    per_motion_rows = [
        {
            "update_count": int(update_count),
            "wall_clock_seconds": float(wall_clock_seconds),
            **row,
        }
        for row in result.per_motion
    ]
    _append_csv(root / "per_motion.csv", per_motion_rows)
    _write_json(run_dir / "summary.json", history_row)
    _write_json(run_dir / "per_motion.json", per_motion_rows)
    torch.save(
        {
            "protocol": result.protocol,
            "summary": result.summary,
            "curves": result.curves,
            "raw": result.raw,
        },
        run_dir / "curves.pt",
    )
    curve_store_path = root / "curves.pt"
    curve_store = (
        torch.load(curve_store_path, map_location="cpu", weights_only=False)
        if curve_store_path.exists()
        else {}
    )
    curve_store[str(update_count)] = {
        "summary": result.summary,
        "curves": result.curves,
    }
    torch.save(curve_store, curve_store_path)
    if render_plots:
        _plot_validation(
            result=result,
            figure_dir=root / "figures",
            update_count=update_count,
        )
