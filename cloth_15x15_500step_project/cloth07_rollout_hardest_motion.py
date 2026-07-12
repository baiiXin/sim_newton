"""Run a continuous rollout for an MLP optimizer or a baseline solver."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cloth03_solvers_and_models import (
    FIXED_VERTEX_INDICES,
    MLPOptimizer,
    ModelSpec,
    NUM_PARTICLES,
    SPATIAL_DIM,
    TORCH_DTYPE,
    AdamState,
    apply_adam_update_full,
    apply_gradient_descent_update_full,
    apply_model_update,
    apply_newton_update_full,
    free_state_from_full_state,
    full_state_from_free_state,
    full_state_from_positions,
    make_q_free,
    physical_config_from_dict,
    project_fixed_vertices,
    stationarity_residual,
    stationarity_residual_norm_full,
    variational_energy,
)
from cloth08_evaluate_baselines import (
    _armijo_step,
    _lbfgs_direction,
    _mass_inverse_diagonal,
)
from cloth_common import load_json, save_json

BASELINE_SOLVERS = {"gd", "adam", "lbfgs", "bfgs", "newton"}


def choose_motion(root: Path, excluded: set[int], candidates: list[int]) -> int:
    audit = load_json(root / "data" / "reference" / "residual_audit" / "reference_audit.json")
    rows = {int(row["motion_index"]): row for row in audit["ranking_rows"]}
    valid = [
        motion
        for motion in candidates
        if motion not in excluded and rows[motion]["num_nonfinite"] == 0
    ]
    if not valid:
        raise RuntimeError("no finite candidate test motions")
    return max(valid, key=lambda motion: float(rows[motion]["residual_p95"]))


def load_model(path: Path, device: torch.device) -> tuple[MLPOptimizer, ModelSpec]:
    checkpoint = torch.load(path, map_location=device)
    saved_spec = checkpoint["model_spec"]
    spec = ModelSpec(
        str(saved_spec["activation"]),
        int(saved_spec["depth"]),
        int(saved_spec["width"]),
        bool(saved_spec["use_bias"]),
    )
    scale = float(checkpoint.get("config", {}).get("residual_length_scale", 5e-2))
    model = MLPOptimizer(scale, spec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, spec


def selected_baseline_params(args: argparse.Namespace) -> dict[str, Any]:
    if args.solver == "gd" and args.gd_step_size is not None:
        return {"step_size": float(args.gd_step_size)}
    if args.solver == "adam" and args.adam_learning_rate is not None:
        return {"learning_rate": float(args.adam_learning_rate)}
    if args.solver in {"lbfgs", "bfgs"} and args.initial_step is not None:
        params = {"initial_step": float(args.initial_step)}
        if args.solver == "lbfgs":
            params["history_size"] = int(args.lbfgs_history_size)
        return params
    if args.solver == "newton":
        return {}

    selection_path = args.baseline_selection or (args.root / "baselines" / "parameter_selection.json")
    if not selection_path.exists():
        raise FileNotFoundError(
            f"missing baseline parameter selection: {selection_path}; run cloth08_evaluate_baselines.py first "
            "or pass explicit baseline parameters"
        )
    selection = load_json(selection_path)
    params = dict(selection[args.solver]["selected"])
    if args.solver == "lbfgs":
        params.setdefault("history_size", int(args.lbfgs_history_size))
    return params


@torch.no_grad()
def run_mlp_inner(
    *,
    model: MLPOptimizer,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical,
    steps: int,
) -> tuple[torch.Tensor, list[float]]:
    previous_residual = torch.zeros_like(y)
    previous_update = torch.zeros_like(y)
    residual_curve = [float(stationarity_residual_norm_full(y, q, masses, physical).item())]
    for _ in range(steps):
        y, delta, residual = apply_model_update(
            model,
            y,
            q,
            masses,
            physical,
            previous_residual=previous_residual,
            previous_update=previous_update,
        )
        previous_residual = residual.detach()
        previous_update = delta.detach()
        residual_curve.append(float(stationarity_residual_norm_full(y, q, masses, physical).item()))
    return y, residual_curve


@torch.no_grad()
def run_standard_baseline_inner(
    *,
    solver: str,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical,
    steps: int,
    params: dict[str, Any],
) -> tuple[torch.Tensor, list[float]]:
    adam_state: AdamState | None = None
    residual_curve: list[float] = []
    for iteration in range(steps + 1):
        residual_curve.append(float(stationarity_residual_norm_full(y, q, masses, physical).item()))
        if iteration == steps:
            break
        if solver == "gd":
            y, _ = apply_gradient_descent_update_full(
                y, q, masses, physical, float(params["step_size"])
            )
        elif solver == "adam":
            y, _, adam_state = apply_adam_update_full(
                y,
                q,
                masses,
                physical,
                adam_state,
                learning_rate=float(params["learning_rate"]),
            )
        elif solver == "newton":
            y, _ = apply_newton_update_full(y, q, masses, physical)
        else:
            raise ValueError(solver)
    return y, residual_curve


def run_quasi_newton_inner(
    *,
    solver: str,
    y_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical,
    steps: int,
    params: dict[str, Any],
) -> tuple[torch.Tensor, list[float], dict[str, int]]:
    y = free_state_from_full_state(y_full)
    q = free_state_from_full_state(q_full)
    gradient = stationarity_residual(y, q, masses, physical)
    energy = variational_energy(y, q, masses, physical)
    residual_curve = [float(torch.linalg.vector_norm(gradient, dim=-1).item())]
    line_search_failures = 0
    curvature_skips = 0
    s_history: list[torch.Tensor] = []
    y_history: list[torch.Tensor] = []
    rho_history: list[torch.Tensor] = []
    valid_history: list[torch.Tensor] = []
    inverse_hessian: torch.Tensor | None = None
    if solver == "bfgs":
        inverse_hessian = torch.diag_embed(_mass_inverse_diagonal(masses, physical))

    for _ in range(steps):
        if solver == "lbfgs":
            direction = _lbfgs_direction(
                gradient,
                masses,
                physical,
                s_history,
                y_history,
                rho_history,
                valid_history,
            )
        elif solver == "bfgs":
            assert inverse_hessian is not None
            direction = -torch.matmul(inverse_hessian, gradient.unsqueeze(-1)).squeeze(-1)
        else:
            raise ValueError(solver)

        y_next, energy_next, _, accepted, _ = _armijo_step(
            y=y,
            q=q,
            masses=masses,
            gradient=gradient,
            direction=direction,
            energy=energy,
            physical=physical,
            initial_step=float(params.get("initial_step", 1.0)),
            c1=1e-4,
            shrink=0.5,
            max_reductions=30,
        )
        line_search_failures += int((~accepted).sum().item())
        gradient_next = stationarity_residual(y_next, q, masses, physical)
        s_vec = y_next - y
        y_vec = gradient_next - gradient
        ys = torch.sum(y_vec * s_vec, dim=-1)
        threshold = (
            1e-10
            * torch.linalg.vector_norm(s_vec, dim=-1)
            * torch.linalg.vector_norm(y_vec, dim=-1)
        )
        valid = accepted & torch.isfinite(ys) & (
            ys > torch.maximum(threshold, torch.full_like(threshold, 1e-30))
        )
        curvature_skips += int((accepted & ~valid).sum().item())

        if solver == "lbfgs":
            safe_ys = torch.where(valid, ys, torch.ones_like(ys))
            s_history.append(s_vec.detach())
            y_history.append(y_vec.detach())
            rho_history.append((1.0 / safe_ys).detach())
            valid_history.append(valid.detach())
            if len(s_history) > int(params.get("history_size", 10)):
                s_history.pop(0)
                y_history.pop(0)
                rho_history.pop(0)
                valid_history.pop(0)
        else:
            assert inverse_hessian is not None
            hy = torch.matmul(inverse_hessian, y_vec.unsqueeze(-1)).squeeze(-1)
            yhy = torch.sum(y_vec * hy, dim=-1)
            safe_ys = torch.where(valid, ys, torch.ones_like(ys))
            coefficient = (1.0 + yhy / safe_ys) / safe_ys
            term_ss = coefficient[:, None, None] * (
                s_vec.unsqueeze(-1) * s_vec.unsqueeze(-2)
            )
            term_cross = (
                hy.unsqueeze(-1) * s_vec.unsqueeze(-2)
                + s_vec.unsqueeze(-1) * hy.unsqueeze(-2)
            ) / safe_ys[:, None, None]
            candidate_h = inverse_hessian + term_ss - term_cross
            candidate_h = 0.5 * (candidate_h + candidate_h.transpose(-1, -2))
            finite_h = torch.isfinite(candidate_h).flatten(1).all(dim=1)
            update = valid & finite_h
            inverse_hessian[update] = candidate_h[update]

        y = y_next
        energy = energy_next
        gradient = gradient_next
        residual_curve.append(float(torch.linalg.vector_norm(gradient, dim=-1).item()))

    y_full_next = project_fixed_vertices(full_state_from_free_state(y, physical), physical)
    stats = {
        "line_search_failures": int(line_search_failures),
        "curvature_update_skips": int(curvature_skips),
    }
    return y_full_next, residual_curve, stats


def final_residuals(curves: torch.Tensor) -> np.ndarray:
    if curves.numel() == 0:
        return np.empty((0,), dtype=float)
    values = curves[:, -1].detach().cpu().numpy().astype(float)
    values[~np.isfinite(values)] = np.inf
    return values


def plot_rollout_diagnostics(
    *,
    curves: torch.Tensor,
    output_dir: Path,
    label: str,
    motion: int,
) -> dict[str, Any]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    final = final_residuals(curves)
    if final.size == 0:
        return {
            "worst_frame_index": None,
            "worst_physical_frame": None,
            "worst_final_residual": None,
            "figures": {},
        }

    frame_indices = np.arange(1, final.size + 1)
    worst_index = int(np.argmax(final))
    worst_physical_frame = int(worst_index + 1)
    worst_final_residual = float(final[worst_index])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame_indices, np.maximum(final, 1e-30), linewidth=1.2)
    ax.axvline(worst_physical_frame, color="black", linestyle="--", linewidth=1.1)
    ax.scatter([worst_physical_frame], [max(worst_final_residual, 1e-30)], color="tab:red", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("physical frame")
    ax.set_ylabel("final stationarity residual after inner iterations")
    ax.set_title(f"{label}: motion {motion:03d} rollout residual vs. timestep")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    residual_timestep_path = figure_dir / "residual_vs_timestep.png"
    fig.savefig(residual_timestep_path, dpi=180)
    plt.close(fig)

    curve = curves[worst_index].detach().cpu().numpy().astype(float)
    curve[~np.isfinite(curve)] = np.inf
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(curve.size), np.maximum(curve, 1e-30), marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("inner iteration")
    ax.set_ylabel("stationarity residual")
    ax.set_title(
        f"{label}: motion {motion:03d} worst frame {worst_physical_frame:03d}"
    )
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    worst_iteration_path = figure_dir / "worst_frame_residual_vs_iteration.png"
    fig.savefig(worst_iteration_path, dpi=180)
    plt.close(fig)

    return {
        "worst_frame_index": worst_index,
        "worst_physical_frame": worst_physical_frame,
        "worst_final_residual": worst_final_residual,
        "figures": {
            "residual_vs_timestep": str(residual_timestep_path),
            "worst_frame_residual_vs_iteration": str(worst_iteration_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("cloth_15x15_500step_pipeline"))
    parser.add_argument("--solver", choices=("mlp", "gd", "adam", "lbfgs", "bfgs", "newton"), default="mlp")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--motion-index", type=int, default=None)
    parser.add_argument("--candidate-motions", type=int, nargs="*", default=list(range(20, 32)))
    parser.add_argument("--exclude-motion-indices", type=int, nargs="*", default=[])
    parser.add_argument("--rollout-length", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline-selection", type=Path, default=None)
    parser.add_argument("--gd-step-size", type=float, default=None)
    parser.add_argument("--adam-learning-rate", type=float, default=None)
    parser.add_argument("--initial-step", type=float, default=None)
    parser.add_argument("--lbfgs-history-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollout_length <= 0 or args.inner_steps <= 0:
        raise ValueError("--rollout-length and --inner-steps must be positive")
    if args.solver == "mlp" and args.checkpoint is None:
        raise ValueError("--checkpoint is required when --solver mlp")

    excluded = set(args.exclude_motion_indices)
    exclusion_file = args.root / "data" / "motion_exclusions.json"
    if exclusion_file.exists():
        excluded.update(load_json(exclusion_file).get("excluded_motion_indices", []))
    motion = (
        int(args.motion_index)
        if args.motion_index is not None
        else choose_motion(args.root, excluded, args.candidate_motions)
    )

    device = torch.device(args.device)
    runtime = load_json(args.root / "data" / "reference" / "runtime_config.json")
    physical = physical_config_from_dict(runtime["physical_config"])
    states = torch.load(
        args.root / "data" / "reference" / "reference_motion_states.pt",
        map_location="cpu",
    )
    motion_ids = [int(value) for value in states["motion_index"].tolist()]
    row = motion_ids.index(motion)
    reference = states["positions"][row, : args.rollout_length + 1]
    reference_velocity = states["velocities"][row, : args.rollout_length + 1]
    actual_rollout_length = min(args.rollout_length, int(reference.shape[0]) - 1)

    model: MLPOptimizer | None = None
    params: dict[str, Any] = {}
    extra_stats: dict[str, int] = {}
    if args.solver == "mlp":
        assert args.checkpoint is not None
        model, spec = load_model(args.checkpoint, device)
        label = spec.experiment_name
        solver_info = {"solver": "mlp", "checkpoint": str(args.checkpoint), "model_spec": spec.__dict__}
    else:
        params = selected_baseline_params(args)
        label = f"baseline_{args.solver}"
        solver_info = {"solver": args.solver, "params": params}

    fixed = set(FIXED_VERTEX_INDICES)
    masses = torch.tensor(
        [physical.masses[i] for i in range(NUM_PARTICLES) if i not in fixed],
        dtype=TORCH_DTYPE,
        device=device,
    ).reshape(1, -1)

    positions = [reference[0].clone()]
    velocities = [reference_velocity[0].clone()]
    curves: list[torch.Tensor] = []
    errors: list[float] = []
    elapsed: list[float] = []

    for frame in range(actual_rollout_length):
        started = time.perf_counter()
        p_n = positions[-1].to(device)
        v_n = velocities[-1].to(device)
        q_free = make_q_free(p_n, v_n, physical).reshape(1, -1)
        q = project_fixed_vertices(full_state_from_free_state(q_free, physical), physical)
        y = project_fixed_vertices(full_state_from_positions(p_n).reshape(1, -1), physical)

        if args.solver == "mlp":
            assert model is not None
            y, residual_curve = run_mlp_inner(
                model=model,
                y=y,
                q=q,
                masses=masses,
                physical=physical,
                steps=args.inner_steps,
            )
        elif args.solver in {"gd", "adam", "newton"}:
            y, residual_curve = run_standard_baseline_inner(
                solver=args.solver,
                y=y,
                q=q,
                masses=masses,
                physical=physical,
                steps=args.inner_steps,
                params=params,
            )
        else:
            y, residual_curve, stats = run_quasi_newton_inner(
                solver=args.solver,
                y_full=y,
                q_full=q,
                masses=masses,
                physical=physical,
                steps=args.inner_steps,
                params=params,
            )
            for key, value in stats.items():
                extra_stats[key] = extra_stats.get(key, 0) + int(value)

        if not torch.isfinite(y).all() or not all(math.isfinite(value) for value in residual_curve):
            print(f"failed at frame {frame}")
            break

        p_next = y.reshape(NUM_PARTICLES, SPATIAL_DIM)
        v_next = (p_next - p_n) / physical.dt
        v_next[list(FIXED_VERTEX_INDICES)] = 0.0
        positions.append(p_next.detach().cpu())
        velocities.append(v_next.detach().cpu())
        curves.append(torch.tensor(residual_curve, dtype=TORCH_DTYPE))
        errors.append(
            float(torch.linalg.vector_norm(p_next - reference[frame + 1].to(device)).item())
        )
        elapsed.append(time.perf_counter() - started)
        if frame == 0 or (frame + 1) % 25 == 0:
            print(
                f"solver={args.solver} motion={motion} frame={frame + 1}/{actual_rollout_length} "
                f"residual={residual_curve[-1]:.3e} error={errors[-1]:.3e}"
            )

    output = args.output or args.root / "rollouts" / f"motion_{motion:03d}" / label / "curve.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    curve_tensor = torch.stack(curves) if curves else torch.empty(0, args.inner_steps + 1)
    diagnostics = plot_rollout_diagnostics(
        curves=curve_tensor,
        output_dir=output.parent,
        label=label,
        motion=motion,
    )
    payload = {
        "motion_index": motion,
        "solver": args.solver,
        "solver_info": solver_info,
        "positions": torch.stack(positions),
        "velocities": torch.stack(velocities),
        "residual_by_frame_and_iteration": curve_tensor,
        "reference_error_by_frame": torch.tensor(errors, dtype=TORCH_DTYPE),
        "elapsed_seconds_by_frame": torch.tensor(elapsed, dtype=TORCH_DTYPE),
        "metadata": {
            "requested_rollout_length": args.rollout_length,
            "completed_frames": len(curves),
            "inner_steps": args.inner_steps,
            "selection": (
                "explicit motion-index"
                if args.motion_index is not None
                else "highest reference residual_p95 among finite candidate motions"
            ),
            **extra_stats,
            **diagnostics,
        },
    }
    if args.solver == "mlp":
        payload["model_spec"] = solver_info["model_spec"]
    torch.save(payload, output)

    summary = {
        "motion_index": motion,
        "solver": args.solver,
        "label": label,
        "output": str(output),
        "completed_frames": len(curves),
        "inner_steps": args.inner_steps,
        **solver_info,
        **diagnostics,
    }
    save_json(summary, output.with_suffix(".json"))
    print(output)


if __name__ == "__main__":
    main()
