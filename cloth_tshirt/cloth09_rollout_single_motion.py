"""Run and cache one frozen T-shirt motion for a network or three GD baselines."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cloth02_batched_physics import FrozenMotionBatch, TShirtPhysics, load_frozen_motion_batch, load_physics
from cloth03_training_pool import LearnedOptimizerMLP, apply_model_update
from cloth04_reference_free_validation import FailureThresholds, detect_failures
from cloth05_train_online import load_model_checkpoint
from tshirt_config import DEFAULT_EVALUATION, DEFAULT_FIXED_DATA_DIR, write_json


DEFAULT_ROOT = Path("cloth_tshirt_pipeline")
BASELINE_SOLVERS = ("gd_fixed", "gd_mass_ls", "gd_block3x3_ls")


@dataclass(frozen=True)
class SingleMotionSettings:
    rollout_frames: int = 500
    inner_steps: int = DEFAULT_EVALUATION.full_inner_steps
    residual_ratio_tolerance: float = DEFAULT_EVALUATION.convergence_residual_ratio
    absolute_residual_tolerance: float = 1e-10
    single_step_ratio_threshold: float = DEFAULT_EVALUATION.two_order_single_step_ratio
    fixed_gd_step_size: float = 5e-5
    mass_ls_step_size: float = 1.0
    block_ls_step_size: float = 1.0
    line_search_max_trials: int = 12
    line_search_reduction: float = 0.5
    armijo_c1: float = 1e-4
    network_line_search: bool = False
    trajectory_stride: int = 5
    early_stop: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.line_search_max_trials <= 12:
            raise ValueError("line_search_max_trials must be in [1, 12]")
        if not 0.0 < self.line_search_reduction < 1.0:
            raise ValueError("line_search_reduction must be in (0, 1)")
        if not 0.0 < self.armijo_c1 < 1.0:
            raise ValueError("armijo_c1 must be in (0, 1)")


@dataclass
class SolverRollout:
    solver: str
    summary: dict[str, Any]
    curves: dict[str, np.ndarray]
    trajectory_frames: np.ndarray
    trajectory_positions: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "network"), required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--split", choices=("typical", "validation", "test"), default="typical")
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--list-motions", action="store_true")
    parser.add_argument("--rollout-frames", type=int, default=SingleMotionSettings.rollout_frames)
    parser.add_argument("--inner-steps", type=int, default=SingleMotionSettings.inner_steps)
    parser.add_argument(
        "--residual-ratio-tolerance",
        type=float,
        default=SingleMotionSettings.residual_ratio_tolerance,
    )
    parser.add_argument("--absolute-residual-tolerance", type=float, default=1e-10)
    parser.add_argument("--fixed-gd-step-size", type=float, default=5e-5)
    parser.add_argument("--mass-ls-step-size", type=float, default=1.0)
    parser.add_argument("--block-ls-step-size", type=float, default=1.0)
    parser.add_argument("--line-search-max-trials", type=int, default=12)
    parser.add_argument("--line-search-reduction", type=float, default=0.5)
    parser.add_argument("--armijo-c1", type=float, default=1e-4)
    parser.add_argument(
        "--network-line-search",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply Armijo backtracking to learned network updates",
    )
    parser.add_argument("--trajectory-stride", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_path(fixed_data_dir: Path, split: str) -> Path:
    names = {
        "typical": "typical_single_motions_4.npz",
        "validation": "validation_32.npz",
        "test": "test_64.npz",
    }
    return Path(fixed_data_dir) / names[split]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def select_motion(dataset: FrozenMotionBatch, index: int) -> FrozenMotionBatch:
    if index < 0 or index >= dataset.batch_size:
        raise ValueError(f"motion index must be in [0, {dataset.batch_size - 1}]")
    return FrozenMotionBatch(
        motion_ids=(dataset.motion_ids[index],),
        positions=dataset.positions[index:index + 1],
        velocities=dataset.velocities[index:index + 1],
        seeds=dataset.seeds[index:index + 1],
    )


def _line_search_step(
    *,
    physics: TShirtPhysics,
    y: torch.Tensor,
    q: torch.Tensor,
    fixed_targets: torch.Tensor,
    gradient: torch.Tensor,
    direction: torch.Tensor,
    initial_step: float,
    settings: SingleMotionSettings,
) -> tuple[torch.Tensor, bool, float, int]:
    energy0 = physics.variational_energy(y, q, fixed_targets).detach()
    slope = torch.sum(gradient * direction, dim=(-2, -1)).detach()
    if not bool(torch.isfinite(slope).all()) or bool((slope <= 0.0).any()):
        return y, False, 0.0, 0
    alpha = float(initial_step)
    for trial in range(1, settings.line_search_max_trials + 1):
        candidate = physics.project_positions(y - alpha * direction, fixed_targets)
        finite = bool(torch.isfinite(candidate).all())
        if finite:
            energy = physics.variational_energy(candidate, q, fixed_targets).detach()
            armijo = energy0 - settings.armijo_c1 * alpha * slope
            if bool(torch.isfinite(energy).all() and (energy <= armijo).all()):
                return candidate, True, alpha, trial
        alpha *= settings.line_search_reduction
    return y, False, 0.0, settings.line_search_max_trials


def _baseline_update(
    *,
    solver: str,
    physics: TShirtPhysics,
    y: torch.Tensor,
    q: torch.Tensor,
    fixed_targets: torch.Tensor,
    settings: SingleMotionSettings,
) -> tuple[torch.Tensor, int, bool]:
    gradient = physics.stationarity_residual(y, q, fixed_targets).detach()
    if solver == "gd_fixed":
        candidate = physics.project_positions(y - settings.fixed_gd_step_size * gradient, fixed_targets)
        return candidate, 0, bool(torch.isfinite(candidate).all())
    if solver == "gd_mass_ls":
        direction = physics.mass_preconditioned_residual(gradient)
        initial_step = settings.mass_ls_step_size
    elif solver == "gd_block3x3_ls":
        direction = physics.block_hessian_preconditioned_residual(y, gradient)
        initial_step = settings.block_ls_step_size
    else:
        raise ValueError(f"unsupported baseline: {solver}")
    candidate, accepted, _, trials = _line_search_step(
        physics=physics,
        y=y,
        q=q,
        fixed_targets=fixed_targets,
        gradient=gradient,
        direction=direction,
        initial_step=initial_step,
        settings=settings,
    )
    return candidate, trials, accepted


def run_solver_rollout(
    *,
    solver: str,
    physics: TShirtPhysics,
    motion: FrozenMotionBatch,
    settings: SingleMotionSettings,
    model: LearnedOptimizerMLP | None = None,
    thresholds: FailureThresholds = FailureThresholds(),
) -> SolverRollout:
    if solver == "network" and model is None:
        raise ValueError("network rollout requires a model")
    if solver != "network" and solver not in BASELINE_SOLVERS:
        raise ValueError(f"unknown solver: {solver}")
    frames = settings.rollout_frames
    names = (
        "initial_residual", "final_residual", "residual_ratio", "first_step_ratio",
        "energy_change", "area_min", "area_max", "edge_min", "edge_max",
        "displacement_rms",
    )
    curves = {name: np.full(frames, np.nan, dtype=np.float64) for name in names}
    curves["inner_steps"] = np.zeros(frames, dtype=np.int64)
    curves["objective_evaluations"] = np.zeros(frames, dtype=np.int64)
    curves["converged"] = np.zeros(frames, dtype=np.bool_)
    curves["line_search_accepted_steps"] = np.zeros(frames, dtype=np.int64)
    curves["line_search_rejected_steps"] = np.zeros(frames, dtype=np.int64)
    curves["line_search_trials"] = np.zeros(frames, dtype=np.int64)
    curves["line_search_alpha_mean"] = np.full(frames, np.nan, dtype=np.float64)
    curves["line_search_alpha_min"] = np.full(frames, np.nan, dtype=np.float64)

    p = motion.positions.detach().clone()
    v = motion.velocities.detach().clone()
    fixed_targets = motion.positions.detach().clone()
    trajectory_frames = [0]
    trajectory_positions = [p[0].detach().cpu().numpy()]
    failure_frame = frames
    failure_reason = ""
    epsilon = torch.finfo(physics.dtype).eps
    model_name = solver
    if model is not None:
        model.eval()

    for frame in range(frames):
        q = physics.make_q(p, v)
        y = physics.project_positions(p, fixed_targets)
        start_positions = p.clone()
        initial = physics.stationarity_residual_norm(y, q, fixed_targets).detach()
        energy0 = physics.variational_energy(y, q, fixed_targets).detach()
        target = max(
            settings.absolute_residual_tolerance,
            float(initial.item()) * settings.residual_ratio_tolerance,
        )
        previous_residual = None
        previous_update = None
        first_ratio = 0.0
        steps = 0
        evaluations = 0
        accepted_steps = 0
        rejected_steps = 0
        line_search_trials = 0
        accepted_alphas: list[float] = []
        invalid_update = False
        current_norm = initial
        for inner in range(settings.inner_steps):
            if settings.early_stop and float(current_norm.item()) <= target:
                break
            if solver == "network":
                assert model is not None
                with torch.no_grad():
                    raw_candidate, raw_delta, current = apply_model_update(
                        model,
                        y,
                        q,
                        fixed_targets,
                        previous_residual=previous_residual,
                        previous_update=previous_update,
                    )
                if settings.network_line_search:
                    # `current` is the mass-preconditioned stationarity
                    # residual. Undo that diagonal preconditioner to recover
                    # the exact energy gradient without evaluating it twice.
                    gradient = current.reshape_as(y) * (
                        physics.vertex_masses.view(1, -1, 1)
                        / (physics.dt * physics.dt)
                    )
                    candidate, accepted, alpha, trials = _line_search_step(
                        physics=physics,
                        y=y,
                        q=q,
                        fixed_targets=fixed_targets,
                        gradient=gradient,
                        direction=-raw_delta.reshape_as(y),
                        initial_step=1.0,
                        settings=settings,
                    )
                    delta = raw_delta * alpha if accepted else torch.zeros_like(raw_delta)
                    valid = bool(torch.isfinite(candidate).all())
                    line_search_trials += trials
                    if accepted:
                        accepted_alphas.append(alpha)
                    else:
                        rejected_steps += 1
                    # Model residual + current energy + candidate energies +
                    # post-step residual.
                    evaluations += 3 + trials
                else:
                    candidate, delta = raw_candidate, raw_delta
                    accepted = bool(torch.isfinite(candidate).all())
                    valid = accepted
                    # One residual inside the model input and one after the update.
                    evaluations += 2
                if valid:
                    y = candidate
                    previous_residual, previous_update = current, delta
                    accepted_steps += int(accepted)
                else:
                    invalid_update = True
            else:
                candidate, trials, accepted = _baseline_update(
                    solver=solver,
                    physics=physics,
                    y=y,
                    q=q,
                    fixed_targets=fixed_targets,
                    settings=settings,
                )
                # Gradient + post-step residual; line search additionally uses
                # the current energy and `trials` candidate energies.
                evaluations += 2 + trials + int(solver != "gd_fixed")
                if bool(torch.isfinite(candidate).all()):
                    y = candidate
                else:
                    invalid_update = True
                accepted_steps += int(accepted)
            steps += 1
            if invalid_update:
                break
            current_norm = physics.stationarity_residual_norm(y, q, fixed_targets).detach()
            if inner == 0:
                first_ratio = float((current_norm / initial.clamp_min(epsilon)).item())

        final = physics.stationarity_residual_norm(y, q, fixed_targets).detach()
        energy1 = physics.variational_energy(y, q, fixed_targets).detach()
        bad, reasons, diagnostics = detect_failures(
            physics, y, final, fixed_targets, thresholds
        )
        if invalid_update:
            bad[:] = True
            reasons[0].append("nonfinite_solver_update")

        curves["initial_residual"][frame] = float(initial.item())
        curves["final_residual"][frame] = float(final.item())
        curves["residual_ratio"][frame] = float((final / initial.clamp_min(epsilon)).item())
        curves["first_step_ratio"][frame] = first_ratio
        curves["energy_change"][frame] = float((energy1 - energy0).item())
        curves["inner_steps"][frame] = steps
        curves["objective_evaluations"][frame] = evaluations
        curves["converged"][frame] = float(final.item()) <= target
        curves["line_search_accepted_steps"][frame] = accepted_steps
        curves["line_search_rejected_steps"][frame] = rejected_steps
        curves["line_search_trials"][frame] = line_search_trials
        if accepted_alphas:
            curves["line_search_alpha_mean"][frame] = float(np.mean(accepted_alphas))
            curves["line_search_alpha_min"][frame] = float(np.min(accepted_alphas))
        curves["displacement_rms"][frame] = float(
            torch.sqrt(torch.mean(torch.sum((y - start_positions) ** 2, dim=-1))).item()
        )
        for key in ("area_min", "area_max", "edge_min", "edge_max"):
            curves[key][frame] = float(diagnostics[key][0].item())

        if bool(bad[0]):
            failure_frame = frame
            failure_reason = "+".join(reasons[0]) or "unknown"
            break
        p, v = physics.advance_state(p, y, fixed_targets)
        physical_frame = frame + 1
        if physical_frame % settings.trajectory_stride == 0 or physical_frame == frames:
            trajectory_frames.append(physical_frame)
            trajectory_positions.append(p[0].detach().cpu().numpy())

    valid = np.isfinite(curves["residual_ratio"])
    slow = valid & (curves["first_step_ratio"] >= settings.single_step_ratio_threshold)
    converged = curves["converged"] & valid
    energy_increase = np.isfinite(curves["energy_change"]) & (curves["energy_change"] > 0.0)
    evaluated = int(valid.sum())
    summary = {
        "solver": model_name,
        "motion_id": motion.motion_ids[0],
        "completed": True,
        "failed": failure_frame < frames,
        "failure_frame": failure_frame,
        "failure_reason": failure_reason,
        "survival_frames": min(failure_frame, frames),
        "rollout_frames": frames,
        "evaluated_frame_count": evaluated,
        "inner_steps_cap": settings.inner_steps,
        "early_stop": settings.early_stop,
        "fixed_inner_iteration_budget": not settings.early_stop,
        "network_line_search": settings.network_line_search,
        "line_search_max_trials": settings.line_search_max_trials,
        "line_search_reduction": settings.line_search_reduction,
        "armijo_c1": settings.armijo_c1,
        "residual_ratio_tolerance": settings.residual_ratio_tolerance,
        "residual_ratio_median": float(np.nanmedian(curves["residual_ratio"])) if evaluated else math.inf,
        "residual_ratio_p95": float(np.nanquantile(curves["residual_ratio"], 0.95)) if evaluated else math.inf,
        "converged_frame_count": int(converged.sum()),
        "converged_frame_fraction": float(converged.sum() / max(evaluated, 1)),
        "inner_steps_mean": float(np.mean(curves["inner_steps"][valid])) if evaluated else math.inf,
        "objective_evaluations_total": int(curves["objective_evaluations"].sum()),
        "single_step_le_two_orders_frame_count": int(slow.sum()),
        "single_step_le_two_orders_frame_fraction": float(slow.sum() / max(evaluated, 1)),
        "energy_increase_fraction": float(energy_increase.sum() / max(evaluated, 1)),
        "line_search_accepted_step_count": int(
            curves["line_search_accepted_steps"].sum()
        ),
        "line_search_rejected_step_count": int(
            curves["line_search_rejected_steps"].sum()
        ),
        "min_area_ratio": float(np.nanmin(curves["area_min"])) if evaluated else 0.0,
        "max_area_ratio": float(np.nanmax(curves["area_max"])) if evaluated else math.inf,
        "min_edge_ratio": float(np.nanmin(curves["edge_min"])) if evaluated else 0.0,
        "max_edge_ratio": float(np.nanmax(curves["edge_max"])) if evaluated else math.inf,
    }
    return SolverRollout(
        solver=model_name,
        summary=summary,
        curves=curves,
        trajectory_frames=np.asarray(trajectory_frames, dtype=np.int64),
        trajectory_positions=np.stack(trajectory_positions),
    )


def save_solver_rollout(result: SolverRollout, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", result.summary)
    np.savez_compressed(output_dir / "curves.npz", **result.curves)
    np.savez_compressed(
        output_dir / "trajectory.npz",
        frames=result.trajectory_frames,
        positions=result.trajectory_positions,
    )


def _baseline_output(args: argparse.Namespace, motion_id: str) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return Path(args.root) / "single_motion_baselines" / f"{args.split}_{args.motion_index:04d}_{motion_id}"


def _network_output(args: argparse.Namespace, motion_id: str, update: int) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    assert args.checkpoint is not None
    checkpoint = args.checkpoint.resolve()
    seed_dir = checkpoint.parent.parent if checkpoint.parent.name == "periodic" else checkpoint.parent
    label = f"update_{update:09d}_{checkpoint.stem}"
    return seed_dir / "single_motion_rollouts" / f"{args.split}_{args.motion_index:04d}_{motion_id}" / label


def _cache_complete(output: Path, config_hash: str, solvers: tuple[str, ...]) -> bool:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("completed") is True
        and manifest.get("config_hash") == config_hash
        and all((output / solver / "metrics.json").exists() for solver in solvers)
    )


def main() -> None:
    args = parse_args()
    if args.line_search_max_trials < 1 or args.line_search_max_trials > 12:
        raise ValueError("--line-search-max-trials must be in [1, 12]")
    if not 0.0 < args.line_search_reduction < 1.0:
        raise ValueError("--line-search-reduction must be in (0, 1)")
    if not 0.0 < args.armijo_c1 < 1.0:
        raise ValueError("--armijo-c1 must be in (0, 1)")
    if args.inner_steps <= 0 or args.rollout_frames <= 0 or args.trajectory_stride <= 0:
        raise ValueError("rollout frames, inner steps, and trajectory stride must be positive")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    data_path = split_path(args.fixed_data_dir, args.split)
    dataset = load_frozen_motion_batch(data_path, device=args.device, dtype=dtype)
    if args.list_motions:
        for index, motion_id in enumerate(dataset.motion_ids):
            print(f"{index:4d} {motion_id}")
        return
    motion = select_motion(dataset, args.motion_index)
    physics = load_physics(fixed_data_dir=args.fixed_data_dir, device=args.device, dtype=dtype)
    settings = SingleMotionSettings(
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        residual_ratio_tolerance=args.residual_ratio_tolerance,
        absolute_residual_tolerance=args.absolute_residual_tolerance,
        fixed_gd_step_size=args.fixed_gd_step_size,
        mass_ls_step_size=args.mass_ls_step_size,
        block_ls_step_size=args.block_ls_step_size,
        line_search_max_trials=args.line_search_max_trials,
        line_search_reduction=args.line_search_reduction,
        armijo_c1=args.armijo_c1,
        network_line_search=args.network_line_search,
        trajectory_stride=args.trajectory_stride,
        # Single-motion evaluation is a fixed-budget protocol.  Threshold-based
        # early stopping is available only in cloth13_inference.py.
        early_stop=False,
    )
    common = {
        "format_version": 1,
        "mesh_sha256": physics.model.mesh_sha256,
        "dataset_sha256": _sha256(data_path),
        "split": args.split,
        "motion_index": args.motion_index,
        "motion_id": motion.motion_ids[0],
        "settings": asdict(settings),
        "dtype": args.dtype,
    }

    if args.mode == "baseline":
        output = _baseline_output(args, motion.motion_ids[0]).resolve()
        config = {**common, "mode": "baseline", "solvers": BASELINE_SOLVERS}
        config_hash = _config_hash(config)
        if not args.overwrite and _cache_complete(output, config_hash, BASELINE_SOLVERS):
            print(f"baseline cache complete; skipped: {output}")
            return
        summaries = []
        for solver in BASELINE_SOLVERS:
            print(f"running {solver} on {motion.motion_ids[0]}")
            result = run_solver_rollout(
                solver=solver, physics=physics, motion=motion, settings=settings
            )
            save_solver_rollout(result, output / solver)
            summaries.append(result.summary)
        write_json(
            output / "manifest.json",
            {
                **config,
                "config_hash": config_hash,
                "completed": True,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "results": summaries,
            },
        )
        print(f"baseline results written to {output}")
        return

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in network mode")
    model, _, checkpoint = load_model_checkpoint(args.checkpoint, physics=physics)
    update = int(checkpoint.get("update_count", -1))
    checkpoint_stat = args.checkpoint.stat()
    config = {
        **common,
        "mode": "network",
        "solver": "network",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_update": update,
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "model_spec": checkpoint["model_spec"],
    }
    config_hash = _config_hash(config)
    output = _network_output(args, motion.motion_ids[0], update).resolve()
    if not args.overwrite and _cache_complete(output, config_hash, ("network",)):
        print(f"network cache complete; skipped: {output}")
        return
    result = run_solver_rollout(
        solver="network", physics=physics, motion=motion, settings=settings, model=model
    )
    result.summary["checkpoint_update"] = update
    result.summary["model_spec"] = checkpoint["model_spec"]
    result.summary["checkpoint"] = str(args.checkpoint.resolve())
    save_solver_rollout(result, output / "network")
    write_json(
        output / "manifest.json",
        {
            **config,
            "config_hash": config_hash,
            "completed": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "results": [result.summary],
        },
    )
    print(f"network result written to {output}")


if __name__ == "__main__":
    main()
