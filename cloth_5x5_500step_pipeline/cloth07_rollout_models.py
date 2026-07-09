"""Script 7: continuous rollout evaluation for baselines and learned models.

A rollout directory is organized by motion id:
    rollouts/motion_003/
        reference_len_500.pt
        baseline_gd/rollout.pt
        baseline_gd/metrics.json
        baseline_gd/status.json
        model_<model_dir_name>/rollout.pt
        model_<model_dir_name>/metrics.json
        model_<model_dir_name>/status.json

If the same motion/solver already has enough frames, it is skipped.
If it has fewer frames, rollout resumes from the last saved frame.

Run examples:
    python cloth07_rollout_models.py --root cloth_5x5_500step_pipeline --motion-index 3 --rollout-length 500 --baselines gd adam newton
    python cloth07_rollout_models.py --root cloth_5x5_500step_pipeline --motion-index 3 --model-dirs models/activation_identity_depth_01_width_0256_no_bias
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from cloth03_solvers_and_models import (
    DEFAULT_DEVICE,
    DEFAULT_EVALUATION_STEPS,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    FIXED_VERTEX_INDICES,
    NUM_FREE_PARTICLES,
    NUM_PARTICLES,
    SPATIAL_DIM,
    AdamState,
    MLPOptimizer,
    ModelSpec,
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


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_physical_config(root: Path):
    runtime = load_json(root / "data" / "reference" / "runtime_config.json")
    return physical_config_from_dict(runtime["physical_config"])


def sanitize_name(name: str) -> str:
    name = str(name).replace("\\", "/").rstrip("/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_reference_for_motion(root: Path, motion_index: int, rollout_length: int) -> dict[str, torch.Tensor]:
    reference = torch.load(root / "data" / "reference" / "reference_motion_states.pt", map_location="cpu")
    motion_ids = reference["motion_index"].tolist()
    row = motion_ids.index(int(motion_index))
    positions = reference["positions"][row, : rollout_length + 1].contiguous()
    velocities = reference["velocities"][row, : rollout_length + 1].contiguous()
    return {"positions": positions, "velocities": velocities}


def save_reference_copy(root: Path, motion_dir: Path, motion_index: int, rollout_length: int) -> dict[str, torch.Tensor]:
    reference = load_reference_for_motion(root, motion_index, rollout_length)
    torch.save(reference, motion_dir / f"reference_len_{rollout_length}.pt")
    return reference


def load_baseline_params(root: Path, name: str) -> dict[str, Any]:
    path = root / "baselines" / "parameter_selection.json"
    if path.exists():
        data = load_json(path)
        if name in data:
            return dict(data[name].get("selected", {}))
    defaults = {
        "gd": {"step_size": 1e-5},
        "adam": {"learning_rate": 1e-3},
        "lbfgs": {"learning_rate": 1.0, "history_size": 10},
        "newton": {},
    }
    return defaults[name]


def load_model(model_dir: Path, device: torch.device, residual_length_scale: float) -> tuple[MLPOptimizer, dict[str, Any]]:
    config = load_json(model_dir / "config.json") if (model_dir / "config.json").exists() else {}
    checkpoint_path = model_dir / "best_validation_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    spec_data = checkpoint.get("model_spec", config.get("model_spec"))
    model_spec = ModelSpec(
        activation=spec_data["activation"],
        depth=int(spec_data["depth"]),
        width=int(spec_data["width"]),
        use_bias=bool(spec_data["use_bias"]),
    )
    scale = float(config.get("residual_length_scale", residual_length_scale))
    model = MLPOptimizer(scale, model_spec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {"model_spec": asdict(model_spec), "checkpoint": str(checkpoint_path), "residual_length_scale": scale}


def reference_error(y_full: torch.Tensor, reference_position: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(y_full.reshape(NUM_PARTICLES, SPATIAL_DIM) - reference_position).cpu().item())


def velocity_from_positions(p_prev: torch.Tensor, p_next: torch.Tensor, physical) -> torch.Tensor:
    v = (p_next - p_prev) / physical.dt
    v[list(FIXED_VERTEX_INDICES), :] = 0.0
    return v


def solve_one_frame_baseline(
    *,
    solver_name: str,
    y0_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical,
    inner_steps: int,
    params: dict[str, Any],
) -> torch.Tensor:
    y = project_fixed_vertices(y0_full.clone(), physical)
    if solver_name == "gd":
        for _ in range(inner_steps):
            y, _ = apply_gradient_descent_update_full(y, q_full, masses, physical, float(params["step_size"]))
        return y
    if solver_name == "adam":
        state: AdamState | None = None
        for _ in range(inner_steps):
            y, _, state = apply_adam_update_full(y, q_full, masses, physical, state, learning_rate=float(params["learning_rate"]))
        return y
    if solver_name == "newton":
        for _ in range(inner_steps):
            y, _ = apply_newton_update_full(y, q_full, masses, physical)
        return y
    if solver_name == "lbfgs":
        states = run_lbfgs_iterations_full(
            y,
            q_full,
            masses,
            physical,
            steps=inner_steps,
            learning_rate=float(params["learning_rate"]),
            history_size=int(params.get("history_size", 10)),
        )
        return states[-1]
    raise ValueError(solver_name)


def solve_one_frame_model(
    *,
    model: MLPOptimizer,
    y0_full: torch.Tensor,
    q_full: torch.Tensor,
    masses: torch.Tensor,
    physical,
    inner_steps: int,
) -> torch.Tensor:
    y = project_fixed_vertices(y0_full.clone(), physical)
    previous_residual = torch.zeros_like(y)
    previous_update = torch.zeros_like(y)
    for _ in range(inner_steps):
        y, delta, current_residual = apply_model_update(
            model,
            y,
            q_full,
            masses,
            physical,
            previous_residual=previous_residual,
            previous_update=previous_update,
        )
        previous_residual = current_residual.detach()
        previous_update = delta.detach()
    return y


def run_rollout(
    *,
    root: Path,
    motion_index: int,
    rollout_length: int,
    solver_name: str,
    output_dir: Path,
    physical,
    device: torch.device,
    inner_steps: int,
    reference: dict[str, torch.Tensor],
    baseline_params: dict[str, Any] | None = None,
    model: MLPOptimizer | None = None,
    model_info: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = output_dir / "rollout.pt"
    status_path = output_dir / "status.json"

    if rollout_path.exists() and not overwrite:
        saved = torch.load(rollout_path, map_location="cpu")
        completed_steps = int(saved["metadata"]["completed_steps"])
        if completed_steps >= rollout_length:
            print(f"skip {solver_name}: already completed {completed_steps} steps")
            return
        positions = [frame.clone() for frame in saved["positions"]]
        velocities = [frame.clone() for frame in saved["velocities"]]
        residuals = list(saved["residual_by_step"])
        errors = list(saved["reference_error_by_step"])
        start_step = completed_steps
        print(f"resume {solver_name}: {completed_steps}/{rollout_length} steps")
    else:
        positions = [reference["positions"][0].clone()]
        velocities = [reference["velocities"][0].clone()]
        residuals = []
        errors = []
        start_step = 0
        if overwrite and rollout_path.exists():
            print(f"overwrite {solver_name}")

    free_masses = torch.tensor([physical.masses[i] for i in range(NUM_PARTICLES) if i not in set(FIXED_VERTEX_INDICES)], dtype=torch.float64, device=device).reshape(1, NUM_FREE_PARTICLES)

    for frame_index in range(start_step, rollout_length):
        p_n = positions[-1].to(device)
        v_n = velocities[-1].to(device)
        q_free = make_q_free(p_n, v_n, physical).reshape(1, -1)
        q_full = project_fixed_vertices(full_state_from_free_state(q_free, physical), physical)
        y0_full = project_fixed_vertices(full_state_from_positions(p_n).reshape(1, -1), physical)

        if model is not None:
            y_next = solve_one_frame_model(
                model=model,
                y0_full=y0_full,
                q_full=q_full,
                masses=free_masses,
                physical=physical,
                inner_steps=inner_steps,
            )
        else:
            assert baseline_params is not None
            baseline_method = solver_name.removeprefix("baseline_")
            y_next = solve_one_frame_baseline(
                solver_name=baseline_method,
                y0_full=y0_full,
                q_full=q_full,
                masses=free_masses,
                physical=physical,
                inner_steps=inner_steps,
                params=baseline_params,
            )

        residual = stationarity_residual_norm_full(y_next, q_full, free_masses, physical)
        p_next = y_next.reshape(NUM_PARTICLES, SPATIAL_DIM).detach().cpu()
        p_prev = positions[-1]
        v_next = velocity_from_positions(p_prev, p_next, physical).detach().cpu()

        reference_next = reference["positions"][frame_index + 1]
        residuals.append(float(residual.detach().cpu().item()))
        errors.append(reference_error(p_next, reference_next))
        positions.append(p_next)
        velocities.append(v_next)

        if frame_index == start_step or (frame_index + 1) % 25 == 0 or frame_index + 1 == rollout_length:
            print(
                f"{solver_name}: frame {frame_index + 1:04d}/{rollout_length}, "
                f"residual={residuals[-1]:.3e}, error={errors[-1]:.3e}"
            )

        torch.save(
            {
                "positions": torch.stack(positions, dim=0).contiguous(),
                "velocities": torch.stack(velocities, dim=0).contiguous(),
                "residual_by_step": residuals,
                "reference_error_by_step": errors,
                "metadata": {
                    "motion_index": motion_index,
                    "solver_name": solver_name,
                    "requested_rollout_length": rollout_length,
                    "completed_steps": len(positions) - 1,
                    "inner_steps": inner_steps,
                    "baseline_params": baseline_params,
                    "model_info": model_info,
                },
            },
            rollout_path,
        )
        save_json(
            {
                "motion_index": motion_index,
                "solver_name": solver_name,
                "requested_rollout_length": rollout_length,
                "completed_steps": len(positions) - 1,
                "can_resume": True,
                "rollout_path": str(rollout_path),
            },
            status_path,
        )

    metrics = {
        "motion_index": motion_index,
        "solver_name": solver_name,
        "rollout_length": rollout_length,
        "inner_steps": inner_steps,
        "final_residual": residuals[-1] if residuals else None,
        "mean_residual": float(torch.tensor(residuals).mean().item()) if residuals else None,
        "max_residual": float(torch.tensor(residuals).max().item()) if residuals else None,
        "final_reference_error": errors[-1] if errors else None,
        "mean_reference_error": float(torch.tensor(errors).mean().item()) if errors else None,
        "max_reference_error": float(torch.tensor(errors).max().item()) if errors else None,
        "baseline_params": baseline_params,
        "model_info": model_info,
    }
    save_json(metrics, output_dir / "metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous rollout for baselines and learned models.")
    parser.add_argument("--root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--motion-index", type=int, required=True)
    parser.add_argument("--rollout-length", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--baselines", nargs="*", default=[] , choices=["gd", "adam", "lbfgs", "newton"])
    parser.add_argument("--model-dirs", nargs="*", default=[])
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    device = torch.device(args.device)
    physical = load_physical_config(root)

    motion_dir = root / "rollouts" / f"motion_{args.motion_index:03d}"
    motion_dir.mkdir(parents=True, exist_ok=True)
    reference = save_reference_copy(root, motion_dir, args.motion_index, args.rollout_length)

    for baseline_name in args.baselines:
        params = load_baseline_params(root, baseline_name)
        run_rollout(
            root=root,
            motion_index=args.motion_index,
            rollout_length=args.rollout_length,
            solver_name=f"baseline_{baseline_name}",
            output_dir=motion_dir / f"baseline_{baseline_name}",
            physical=physical,
            device=device,
            inner_steps=args.inner_steps,
            reference=reference,
            baseline_params=params,
            model=None,
            model_info=None,
            overwrite=args.overwrite,
        )

    for model_dir_text in args.model_dirs:
        model_dir = Path(model_dir_text)
        if not model_dir.is_absolute():
            candidate = root / model_dir
            if candidate.exists():
                model_dir = candidate
        model, info = load_model(model_dir, device, args.residual_length_scale)
        solver_name = f"model_{sanitize_name(model_dir.name)}"
        run_rollout(
            root=root,
            motion_index=args.motion_index,
            rollout_length=args.rollout_length,
            solver_name=solver_name,
            output_dir=motion_dir / solver_name,
            physical=physical,
            device=device,
            inner_steps=args.inner_steps,
            reference=reference,
            baseline_params=None,
            model=model,
            model_info=info,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
