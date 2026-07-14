"""Roll out one scale-up scenario and render learned optimizer vs. a baseline."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from cloth02_batched_physics import (
    advance_state,
    build_batched_parameters,
    dirichlet_targets,
    flatten_positions,
    free_update_gate,
    make_q,
    project_positions,
    spring_lengths,
    stationarity_residual,
    stationarity_residual_norm,
    variational_energy,
)
from cloth03_training_pool import LearnedOptimizerMLP, ModelSpec, apply_model_update
from cloth04_reference_free_validation import FailureThresholds
from cloth07_evaluate_best_checkpoint import (
    checkpoint_path,
    resolve_dtype,
    run_directory,
    write_json,
)
from scenario_catalogue import build_catalogues
from scenario_templates import (
    BOUNDARY_BY_ID,
    DIRICHLET_BY_ID,
    MATERIAL_BY_ID,
    ORIENTATION_BY_ID,
    SHAPE_BY_ID,
    STRAIN_BY_ID,
    VELOCITY_BY_ID,
    ScenarioSpec,
)


DEFAULT_ROOT = Path("cloth_15x15_scale_up_pipeline")
BASELINE_SUBDIR = "single_motion_rollout_baseline"
REQUIRED_BASELINE_SOLVERS = (
    "gd_fixed_lr_5e-5",
    "line_search_gd",
    "mass_preconditioned_gd_fixed",
    "mass_preconditioned_line_search_gd",
    "lbfgs_line_search_h5",
    "newton",
)


@dataclass(frozen=True)
class InnerConvergence:
    enabled: bool
    residual_ratio_tol: float
    absolute_residual_tol: float
    step_rms_tol: float
    step_residual_ratio_guard: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one reference-free scale-up rollout with a trained model and "
            "compare it to Newton and gradient-descent baselines."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--mode", choices=("mlp", "baseline"), default="mlp")
    parser.add_argument("--catalogue", choices=("c1", "c2", "c3"), default="c2")
    parser.add_argument("--activation", default=ModelSpec().activation)
    parser.add_argument("--depth", type=int, default=ModelSpec().depth)
    parser.add_argument("--width", type=int, default=ModelSpec().width)
    parser.add_argument(
        "--use-bias",
        action=argparse.BooleanOptionalAction,
        default=ModelSpec().use_bias,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-update", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float64", "float32"), default="auto")
    parser.add_argument(
        "--split",
        choices=("validation", "test", "train"),
        default="test",
        help="scenario split used for the selected motion index",
    )
    parser.add_argument(
        "--list-motions",
        action="store_true",
        help="list available motion indices for --split and exit",
    )
    parser.add_argument("--list-offset", type=int, default=0)
    parser.add_argument("--list-limit", type=int, default=32)
    parser.add_argument("--motion-index", type=int, default=0)
    parser.add_argument("--rollout-frames", type=int, default=500)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--baseline-step-size", type=float, default=1.0)
    parser.add_argument("--baseline-line-search-reductions", type=int, default=12)
    parser.add_argument("--mass-preconditioned-gd-step-size", type=float, default=1.0)
    parser.add_argument("--fixed-gd-step-size", type=float, default=5e-5)
    parser.add_argument("--line-search-gd-step-size", type=float, default=5e-5)
    parser.add_argument("--line-search-gd-reductions", type=int, default=30)
    parser.add_argument("--line-search-gd-growths", type=int, default=8)
    parser.add_argument("--lbfgs-history-size", type=int, default=5)
    parser.add_argument("--lbfgs-step-size", type=float, default=1.0)
    parser.add_argument("--lbfgs-line-search-reductions", type=int, default=30)
    parser.add_argument("--refresh-baseline", action="store_true")
    parser.add_argument("--disable-inner-early-stop", action="store_true")
    parser.add_argument("--convergence-residual-ratio-tol", type=float, default=1e-10)
    parser.add_argument("--convergence-absolute-residual-tol", type=float, default=1e-10)
    parser.add_argument("--convergence-step-rms-tol", type=float, default=1e-12)
    parser.add_argument("--convergence-step-residual-ratio-guard", type=float, default=1e-8)
    parser.add_argument("--render-format", choices=("mp4", "gif", "none"), default="mp4")
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-residual", type=float, default=1e12)
    parser.add_argument("--max-abs-position", type=float, default=1e4)
    parser.add_argument("--min-edge-ratio", type=float, default=1e-5)
    parser.add_argument("--max-edge-ratio", type=float, default=1e4)
    parser.add_argument("--max-constraint-error", type=float, default=1e-9)
    return parser.parse_args()


def split_key(split: str, catalogue: str) -> str:
    if split == "validation":
        return "validation_128"
    if split == "test":
        return "test_256"
    return {
        "c1": "train_c1_1024",
        "c2": "train_c2_2048",
        "c3": "train_c3_3072",
    }[catalogue]


def selected_scenario(args: argparse.Namespace) -> ScenarioSpec:
    scenarios, _ = selected_scenarios(args)
    if args.motion_index < 0 or args.motion_index >= len(scenarios):
        raise ValueError(
            f"--motion-index must be in [0, {len(scenarios) - 1}] for "
            f"{split_key(args.split, args.catalogue)}"
        )
    return scenarios[args.motion_index]


def selected_scenarios(args: argparse.Namespace) -> tuple[tuple[ScenarioSpec, ...], str]:
    catalogues = build_catalogues()
    key = split_key(args.split, args.catalogue)
    scenarios = tuple(catalogues[key])
    if not scenarios:
        raise ValueError(f"empty scenario split: {key}")
    return scenarios, key


def scenario_labels(scenario: ScenarioSpec) -> dict[str, str]:
    return {
        "shape": SHAPE_BY_ID[scenario.shape_id].name,
        "strain": STRAIN_BY_ID[scenario.strain_id].name,
        "velocity": VELOCITY_BY_ID[scenario.velocity_id].name,
        "boundary": BOUNDARY_BY_ID[scenario.boundary_id].name,
        "dirichlet": DIRICHLET_BY_ID[scenario.dirichlet_id].name,
        "material": MATERIAL_BY_ID[scenario.material_id].name,
        "orientation": ORIENTATION_BY_ID[scenario.orientation_id].name,
    }


def scenario_row(index: int, scenario: ScenarioSpec) -> dict[str, Any]:
    labels = scenario_labels(scenario)
    return {
        "index": index,
        "scenario_id": scenario.scenario_id,
        "group": scenario.group,
        "difficulty": scenario.difficulty,
        "shape": f"{scenario.shape_id}:{labels['shape']}",
        "strain": f"{scenario.strain_id}:{labels['strain']}",
        "velocity": f"{scenario.velocity_id}:{labels['velocity']}",
        "boundary": f"{scenario.boundary_id}:{labels['boundary']}",
        "dirichlet": f"{scenario.dirichlet_id}:{labels['dirichlet']}",
        "material": f"{scenario.material_id}:{labels['material']}",
        "orientation": f"{scenario.orientation_id}:{labels['orientation']}",
    }


def print_motion_table(args: argparse.Namespace) -> None:
    scenarios, key = selected_scenarios(args)
    if args.list_limit <= 0:
        raise ValueError("--list-limit must be positive")
    start = max(0, int(args.list_offset))
    end = min(len(scenarios), start + int(args.list_limit))
    print(f"catalogue={key} count={len(scenarios)} showing=[{start}, {end})")
    headers = [
        "index",
        "scenario_id",
        "group",
        "difficulty",
        "boundary",
        "dirichlet",
        "material",
        "velocity",
        "shape",
        "strain",
        "orientation",
    ]
    rows = [scenario_row(index, scenarios[index]) for index in range(start, end)]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    } if rows else {header: len(header) for header in headers}
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def default_output_dir(args: argparse.Namespace, run_dir: Path) -> Path:
    bias = "bias" if args.use_bias else "no_bias"
    name = (
        f"{args.split}_motion_{args.motion_index:04d}_"
        f"k{args.inner_steps:03d}_{bias}"
    )
    return run_dir / "single_motion_rollouts" / name


def checkpoint_run_dir(checkpoint: Path, fallback: Path) -> Path:
    resolved = checkpoint
    if resolved.name.startswith("checkpoint_update_") and resolved.parent.name == "periodic":
        return resolved.parent.parent
    if resolved.name in {
        "best_validation_model.pt",
        "latest_checkpoint.pt",
    }:
        return resolved.parent
    return fallback


def baseline_output_dir(args: argparse.Namespace) -> Path:
    name = (
        f"{split_key(args.split, args.catalogue)}_"
        f"motion_{args.motion_index:04d}_"
        f"f{args.rollout_frames:03d}_"
        f"k{args.inner_steps:03d}"
    )
    return args.root / BASELINE_SUBDIR / name


def baseline_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.dtype == "float32":
        return torch.float32
    return torch.float64


def _finite_float(value: torch.Tensor, default: float = float("nan")) -> float:
    scalar = float(value.detach().cpu().item())
    return scalar if math.isfinite(scalar) else float(default)


def convergence_from_args(args: argparse.Namespace) -> InnerConvergence:
    return InnerConvergence(
        enabled=not bool(args.disable_inner_early_stop),
        residual_ratio_tol=float(args.convergence_residual_ratio_tol),
        absolute_residual_tol=float(args.convergence_absolute_residual_tol),
        step_rms_tol=float(args.convergence_step_rms_tol),
        step_residual_ratio_guard=float(args.convergence_step_residual_ratio_guard),
    )


def validate_convergence(config: InnerConvergence) -> None:
    if config.residual_ratio_tol <= 0:
        raise ValueError("--convergence-residual-ratio-tol must be positive")
    if config.absolute_residual_tol <= 0:
        raise ValueError("--convergence-absolute-residual-tol must be positive")
    if config.step_rms_tol <= 0:
        raise ValueError("--convergence-step-rms-tol must be positive")
    if config.step_residual_ratio_guard <= 0:
        raise ValueError("--convergence-step-residual-ratio-guard must be positive")


def normalized_free_step(
    before: torch.Tensor,
    after: torch.Tensor,
    params,
) -> float:
    delta = (after.reshape_as(before) - before).reshape(params.batch_size, params.num_vertices, 3)
    free = free_update_gate(params).to(delta.dtype)
    free_count = (~params.fixed_mask).sum(dim=-1).clamp_min(1).to(delta.dtype)
    rms = torch.sqrt(torch.sum((delta * free).square(), dim=(-2, -1)) / free_count)
    scale = params.rest_lengths.mean(dim=-1).clamp_min(torch.finfo(params.dtype).tiny)
    return _finite_float((rms / scale)[0])


def convergence_reason(
    *,
    initial_residual: float,
    current_residual: float,
    normalized_step: float | None,
    config: InnerConvergence,
) -> str | None:
    if not config.enabled:
        return None
    if not math.isfinite(current_residual):
        return None
    denominator = max(float(initial_residual), 1e-300)
    ratio = float(current_residual) / denominator
    if current_residual <= config.absolute_residual_tol:
        return "absolute_residual"
    if ratio <= config.residual_ratio_tol:
        return "relative_residual"
    if (
        normalized_step is not None
        and math.isfinite(normalized_step)
        and normalized_step <= config.step_rms_tol
        and ratio <= config.step_residual_ratio_guard
    ):
        return "step_rms_guarded"
    return None


def padded_curve(curve: Sequence[float], length: int) -> list[float]:
    values = [float(value) for value in curve]
    if len(values) >= length:
        return values[:length]
    fill = values[-1] if values else float("nan")
    return values + [fill] * (length - len(values))


def frame_diagnostics(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    params,
    targets: torch.Tensor,
    thresholds: FailureThresholds,
) -> dict[str, Any]:
    residual = stationarity_residual_norm(y, q, params, targets)
    energy = variational_energy(y, q, params, targets)
    lengths = spring_lengths(y, params, targets)
    ratios = lengths / params.rest_lengths.clamp_min(torch.finfo(params.dtype).eps)
    projected = project_positions(y, params, targets)
    constraint = torch.where(
        params.fixed_mask.unsqueeze(-1),
        (projected - targets).abs(),
        torch.zeros_like(projected),
    ).amax(dim=(-2, -1))
    finite = (
        torch.isfinite(y).flatten(start_dim=1).all(dim=1)
        & torch.isfinite(residual)
        & torch.isfinite(energy)
        & torch.isfinite(lengths).all(dim=-1)
    )
    edge_min = ratios.amin(dim=-1)
    edge_max = ratios.amax(dim=-1)
    failed = (
        ~finite
        | (residual > thresholds.max_residual)
        | (y.abs().amax(dim=(-2, -1)) > thresholds.max_abs_position)
        | (edge_min < thresholds.min_edge_ratio)
        | (edge_max > thresholds.max_edge_ratio)
        | (constraint > thresholds.max_constraint_error)
    )
    return {
        "residual": _finite_float(residual[0]),
        "energy": _finite_float(energy[0]),
        "edge_ratio_min": _finite_float(edge_min[0]),
        "edge_ratio_max": _finite_float(edge_max[0]),
        "constraint_error": _finite_float(constraint[0]),
        "failed": bool(failed[0].item()),
    }


@torch.no_grad()
def baseline_step(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    params,
    targets: torch.Tensor,
    step_size: float,
    max_reductions: int,
) -> tuple[torch.Tensor, bool, float, int]:
    energy = variational_energy(y, q, params, targets)
    residual = stationarity_residual(y, q, params, targets)
    residual_points = residual.reshape(params.batch_size, params.num_vertices, 3)
    preconditioned = (
        params.dt**2
        * residual_points
        / params.masses.clamp_min(torch.finfo(params.dtype).tiny).unsqueeze(-1)
    )
    direction = -flatten_positions(
        preconditioned * free_update_gate(params).to(params.dtype)
    )
    y_flat = y.reshape(params.batch_size, -1)
    scale = float(step_size)
    trials = 0
    for _ in range(max(0, max_reductions) + 1):
        trials += 1
        candidate = project_positions(y_flat + scale * direction, params, targets)
        candidate_energy = variational_energy(candidate, q, params, targets)
        accepted = (
            torch.isfinite(candidate).flatten(start_dim=1).all(dim=1)
            & torch.isfinite(candidate_energy)
            & (candidate_energy <= energy)
        )
        if bool(accepted[0].item()):
            return candidate.reshape_as(y), True, scale, trials
        scale *= 0.5
    return y, False, 0.0, trials


@torch.no_grad()
def mass_preconditioned_gradient_descent_step(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    params,
    targets: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    residual = stationarity_residual(y, q, params, targets)
    residual_points = residual.reshape(params.batch_size, params.num_vertices, 3)
    preconditioned = (
        params.dt**2
        * residual_points
        / params.masses.clamp_min(torch.finfo(params.dtype).tiny).unsqueeze(-1)
    )
    direction = -flatten_positions(
        preconditioned * free_update_gate(params).to(params.dtype)
    )
    y_flat = y.reshape(params.batch_size, -1)
    return project_positions(y_flat + float(step_size) * direction, params, targets)


@torch.no_grad()
def fixed_gradient_descent_step(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    params,
    targets: torch.Tensor,
    step_size: float,
) -> torch.Tensor:
    residual = stationarity_residual(y, q, params, targets)
    direction = -residual.reshape(params.batch_size, -1)
    direction = direction * free_update_gate(params, flattened=True).to(params.dtype)
    y_flat = y.reshape(params.batch_size, -1)
    return project_positions(y_flat + float(step_size) * direction, params, targets)


@torch.no_grad()
def armijo_direction_step(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    params,
    targets: torch.Tensor,
    direction: torch.Tensor,
    step_size: float,
    max_reductions: int,
    c1: float = 1e-4,
) -> tuple[torch.Tensor, bool, float, int]:
    energy = variational_energy(y, q, params, targets)
    residual = stationarity_residual(y, q, params, targets).reshape(params.batch_size, -1)
    directional = torch.sum(residual * direction, dim=-1)
    y_flat = y.reshape(params.batch_size, -1)
    scale = float(step_size)
    trials = 0
    for _ in range(max(0, max_reductions) + 1):
        trials += 1
        candidate = project_positions(y_flat + scale * direction, params, targets)
        candidate_energy = variational_energy(candidate, q, params, targets)
        armijo_rhs = energy + c1 * scale * directional
        accepted = (
            torch.isfinite(candidate).flatten(start_dim=1).all(dim=1)
            & torch.isfinite(candidate_energy)
            & (candidate_energy <= armijo_rhs)
        )
        if bool(accepted[0].item()):
            return candidate.reshape_as(y), True, scale, trials
        scale *= 0.5
    return y, False, 0.0, trials


def lbfgs_direction(
    gradient: torch.Tensor,
    s_history: Sequence[torch.Tensor],
    y_history: Sequence[torch.Tensor],
) -> torch.Tensor:
    if not s_history:
        return -gradient
    q_vec = gradient.clone()
    alphas: list[torch.Tensor] = []
    rhos: list[torch.Tensor] = []
    for s_vec, y_vec in zip(reversed(s_history), reversed(y_history)):
        sy = torch.sum(s_vec * y_vec, dim=-1).clamp_min(torch.finfo(gradient.dtype).tiny)
        rho = 1.0 / sy
        alpha = rho * torch.sum(s_vec * q_vec, dim=-1)
        q_vec = q_vec - alpha.unsqueeze(-1) * y_vec
        alphas.append(alpha)
        rhos.append(rho)
    last_s = s_history[-1]
    last_y = y_history[-1]
    yy = torch.sum(last_y * last_y, dim=-1).clamp_min(torch.finfo(gradient.dtype).tiny)
    gamma = torch.sum(last_s * last_y, dim=-1).clamp_min(torch.finfo(gradient.dtype).tiny) / yy
    r_vec = gamma.unsqueeze(-1) * q_vec
    for s_vec, y_vec, alpha, rho in zip(
        s_history,
        y_history,
        reversed(alphas),
        reversed(rhos),
    ):
        beta = rho * torch.sum(y_vec * r_vec, dim=-1)
        r_vec = r_vec + s_vec * (alpha - beta).unsqueeze(-1)
    return -r_vec


@torch.no_grad()
def line_search_gradient_descent_step(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    params,
    targets: torch.Tensor,
    step_size: float,
    max_reductions: int,
    max_growths: int,
    c1: float = 1e-4,
) -> tuple[torch.Tensor, bool, float, int]:
    energy = variational_energy(y, q, params, targets)
    residual = stationarity_residual(y, q, params, targets).reshape(params.batch_size, -1)
    direction = -residual * free_update_gate(params, flattened=True).to(params.dtype)
    directional = torch.sum(residual * direction, dim=-1)
    y_flat = y.reshape(params.batch_size, -1)
    scale = float(step_size)
    best_candidate: torch.Tensor | None = None
    best_scale = 0.0
    trials = 0

    for _ in range(max(0, max_growths) + 1):
        trials += 1
        candidate = project_positions(y_flat + scale * direction, params, targets)
        candidate_energy = variational_energy(candidate, q, params, targets)
        armijo_rhs = energy + c1 * scale * directional
        accepted = (
            torch.isfinite(candidate).flatten(start_dim=1).all(dim=1)
            & torch.isfinite(candidate_energy)
            & (candidate_energy <= armijo_rhs)
        )
        if not bool(accepted[0].item()):
            break
        best_candidate = candidate
        best_scale = scale
        scale *= 2.0

    if best_candidate is not None:
        return best_candidate.reshape_as(y), True, best_scale, trials

    scale = float(step_size)
    for _ in range(max(0, max_reductions) + 1):
        trials += 1
        candidate = project_positions(y_flat + scale * direction, params, targets)
        candidate_energy = variational_energy(candidate, q, params, targets)
        armijo_rhs = energy + c1 * scale * directional
        accepted = (
            torch.isfinite(candidate).flatten(start_dim=1).all(dim=1)
            & torch.isfinite(candidate_energy)
            & (candidate_energy <= armijo_rhs)
        )
        if bool(accepted[0].item()):
            return candidate.reshape_as(y), True, scale, trials
        scale *= 0.5
    return y, False, 0.0, trials


def newton_step(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    params,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, bool, dict[str, int]]:
    if params.batch_size != 1:
        raise ValueError("newton_step currently supports one scenario at a time")
    free_mask = free_update_gate(params, flattened=True)[0].bool()
    if not bool(free_mask.any()):
        return y.detach(), True, {
            "linear_solve_failures": 0,
        }

    base = project_positions(y, params, targets).reshape(1, -1)[0].detach()
    x0 = base[free_mask].clone().detach()

    def assemble(free_values: torch.Tensor) -> torch.Tensor:
        full = base.clone()
        full[free_mask] = free_values
        return full.reshape(1, params.num_vertices, 3)

    def energy_of_free(free_values: torch.Tensor) -> torch.Tensor:
        return variational_energy(assemble(free_values), q, params, targets)[0]

    residual = stationarity_residual(
        base.reshape(1, params.num_vertices, 3),
        q,
        params,
        targets,
    ).reshape(1, -1)[0, free_mask]
    hessian = torch.autograd.functional.hessian(
        energy_of_free,
        x0,
        vectorize=False,
    )
    hessian = 0.5 * (hessian + hessian.transpose(-1, -2))
    direction, info = torch.linalg.solve_ex(hessian, -residual.unsqueeze(-1))
    direction = direction.squeeze(-1)
    linear_solve_failures = int(bool(torch.any(info != 0)) or not bool(torch.isfinite(direction).all()))
    if linear_solve_failures:
        return y.detach(), False, {
            "linear_solve_failures": linear_solve_failures,
        }
    candidate = project_positions(assemble(x0 + direction), params, targets)
    accepted = bool(torch.isfinite(candidate).all())
    return candidate.detach(), accepted, {
        "linear_solve_failures": 0 if accepted else 1,
    }


@torch.no_grad()
def run_model_rollout(
    *,
    model: LearnedOptimizerMLP,
    scenario: ScenarioSpec,
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
    convergence: InnerConvergence,
) -> dict[str, Any]:
    params = build_batched_parameters((scenario,), device=device, dtype=dtype)
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    positions = [p[0].detach().cpu()]
    residual_curve = []
    frame_rows: list[dict[str, Any]] = []
    failure_frame: int | None = None

    for frame in range(rollout_frames):
        if failure_frame is not None:
            positions.append(p[0].detach().cpu())
            residual_curve.append([float("nan")] * (inner_steps + 1))
            frame_rows.append({"frame": frame, "failed": True})
            continue

        q = make_q(p, v, params)
        next_time = torch.full((1,), (frame + 1) * params.dt, dtype=dtype, device=device)
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(p, params, targets)
        previous_residual = torch.zeros(1, params.full_state_dim, dtype=dtype, device=device)
        previous_update = torch.zeros_like(previous_residual)
        curve = [frame_diagnostics(y=y, q=q, params=params, targets=targets, thresholds=thresholds)["residual"]]
        energy_before = variational_energy(y, q, params, targets)
        convergence_hit = False
        convergence_hit_reason = ""

        for _ in range(inner_steps):
            y_before = y.clone()
            y_next, delta, current = apply_model_update(
                model,
                y,
                q,
                params,
                target_positions=targets,
                previous_residual=previous_residual,
                previous_update=previous_update,
            )
            y = y_next.reshape_as(y)
            previous_residual = current
            previous_update = delta
            current_residual = frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
            curve.append(current_residual)
            reason = convergence_reason(
                initial_residual=curve[0],
                current_residual=current_residual,
                normalized_step=normalized_free_step(y_before, y, params),
                config=convergence,
            )
            if reason is not None:
                convergence_hit = True
                convergence_hit_reason = reason
                break

        diagnostics = frame_diagnostics(
            y=y,
            q=q,
            params=params,
            targets=targets,
            thresholds=thresholds,
        )
        energy_after = variational_energy(y, q, params, targets)
        diagnostics.update(
            {
                "frame": frame,
                "initial_residual": curve[0],
                "final_residual": curve[-1],
                "residual_ratio": curve[-1] / max(curve[0], torch.finfo(dtype).eps),
                "energy_before": _finite_float(energy_before[0]),
                "energy_after": _finite_float(energy_after[0]),
                "inner_steps_used": len(curve) - 1,
                "inner_converged": convergence_hit,
                "convergence_reason": convergence_hit_reason,
            }
        )
        frame_rows.append(diagnostics)
        residual_curve.append(padded_curve(curve, inner_steps + 1))
        if diagnostics["failed"]:
            failure_frame = frame
            positions.append(p[0].detach().cpu())
            continue
        p, v = advance_state(p, y, params, next_time=next_time)
        positions.append(p[0].detach().cpu())

    return {
        "solver": "mlp",
        "positions": torch.stack(positions, dim=0),
        "residual_by_frame_and_iteration": torch.tensor(residual_curve, dtype=torch.float64),
        "frames": frame_rows,
        "failure_frame": failure_frame,
        "line_search_failures": 0,
    }


@torch.no_grad()
def run_baseline_rollout(
    *,
    scenario: ScenarioSpec,
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
    step_size: float,
    max_reductions: int,
    convergence: InnerConvergence,
) -> dict[str, Any]:
    params = build_batched_parameters((scenario,), device=device, dtype=dtype)
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    positions = [p[0].detach().cpu()]
    residual_curve = []
    frame_rows: list[dict[str, Any]] = []
    failure_frame: int | None = None
    line_search_failures = 0

    for frame in range(rollout_frames):
        if failure_frame is not None:
            positions.append(p[0].detach().cpu())
            residual_curve.append([float("nan")] * (inner_steps + 1))
            frame_rows.append({"frame": frame, "failed": True})
            continue

        q = make_q(p, v, params)
        next_time = torch.full((1,), (frame + 1) * params.dt, dtype=dtype, device=device)
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(p, params, targets)
        curve = [frame_diagnostics(y=y, q=q, params=params, targets=targets, thresholds=thresholds)["residual"]]
        energy_before = variational_energy(y, q, params, targets)
        accepted_scales: list[float] = []
        line_search_trials = 0
        frame_line_search_failures = 0
        convergence_hit = False
        convergence_hit_reason = ""

        for _ in range(inner_steps):
            y_before = y.clone()
            y, accepted, accepted_scale, trials = baseline_step(
                y=y,
                q=q,
                params=params,
                targets=targets,
                step_size=step_size,
                max_reductions=max_reductions,
            )
            line_search_trials += int(trials)
            if not accepted:
                line_search_failures += 1
                frame_line_search_failures += 1
            accepted_scales.append(float(accepted_scale))
            current_residual = frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
            curve.append(current_residual)
            reason = convergence_reason(
                initial_residual=curve[0],
                current_residual=current_residual,
                normalized_step=normalized_free_step(y_before, y, params),
                config=convergence,
            )
            if reason is not None:
                convergence_hit = True
                convergence_hit_reason = reason
                break

        diagnostics = frame_diagnostics(
            y=y,
            q=q,
            params=params,
            targets=targets,
            thresholds=thresholds,
        )
        energy_after = variational_energy(y, q, params, targets)
        nonzero_scales = [value for value in accepted_scales if value > 0.0]
        diagnostics.update(
            {
                "frame": frame,
                "initial_residual": curve[0],
                "final_residual": curve[-1],
                "residual_ratio": curve[-1] / max(curve[0], torch.finfo(dtype).eps),
                "energy_before": _finite_float(energy_before[0]),
                "energy_after": _finite_float(energy_after[0]),
                "accepted_step_min": min(nonzero_scales) if nonzero_scales else 0.0,
                "accepted_step_median": (
                    sorted(nonzero_scales)[len(nonzero_scales) // 2]
                    if nonzero_scales
                    else 0.0
                ),
                "line_search_trials": line_search_trials,
                "line_search_failures_frame": frame_line_search_failures,
                "inner_steps_used": len(curve) - 1,
                "inner_converged": convergence_hit,
                "convergence_reason": convergence_hit_reason,
            }
        )
        frame_rows.append(diagnostics)
        residual_curve.append(padded_curve(curve, inner_steps + 1))
        if diagnostics["failed"]:
            failure_frame = frame
            positions.append(p[0].detach().cpu())
            continue
        p, v = advance_state(p, y, params, next_time=next_time)
        positions.append(p[0].detach().cpu())

    return {
        "solver": "mass_preconditioned_line_search_gd",
        "positions": torch.stack(positions, dim=0),
        "residual_by_frame_and_iteration": torch.tensor(residual_curve, dtype=torch.float64),
        "frames": frame_rows,
        "failure_frame": failure_frame,
        "line_search_failures": line_search_failures,
        "initial_step_size": float(step_size),
    }


@torch.no_grad()
def run_fixed_gd_rollout(
    *,
    scenario: ScenarioSpec,
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
    step_size: float,
    convergence: InnerConvergence,
) -> dict[str, Any]:
    params = build_batched_parameters((scenario,), device=device, dtype=dtype)
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    positions = [p[0].detach().cpu()]
    residual_curve = []
    frame_rows: list[dict[str, Any]] = []
    failure_frame: int | None = None

    for frame in range(rollout_frames):
        if failure_frame is not None:
            positions.append(p[0].detach().cpu())
            residual_curve.append([float("nan")] * (inner_steps + 1))
            frame_rows.append({"frame": frame, "failed": True})
            continue

        q = make_q(p, v, params)
        next_time = torch.full((1,), (frame + 1) * params.dt, dtype=dtype, device=device)
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(p, params, targets)
        curve = [
            frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
        ]
        energy_before = variational_energy(y, q, params, targets)
        convergence_hit = False
        convergence_hit_reason = ""

        for _ in range(inner_steps):
            y_before = y.clone()
            y = fixed_gradient_descent_step(
                y=y,
                q=q,
                params=params,
                targets=targets,
                step_size=step_size,
            ).reshape_as(y)
            current_residual = frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
            curve.append(current_residual)
            reason = convergence_reason(
                initial_residual=curve[0],
                current_residual=current_residual,
                normalized_step=normalized_free_step(y_before, y, params),
                config=convergence,
            )
            if reason is not None:
                convergence_hit = True
                convergence_hit_reason = reason
                break

        diagnostics = frame_diagnostics(
            y=y,
            q=q,
            params=params,
            targets=targets,
            thresholds=thresholds,
        )
        energy_after = variational_energy(y, q, params, targets)
        diagnostics.update(
            {
                "frame": frame,
                "initial_residual": curve[0],
                "final_residual": curve[-1],
                "residual_ratio": curve[-1] / max(curve[0], torch.finfo(dtype).eps),
                "energy_before": _finite_float(energy_before[0]),
                "energy_after": _finite_float(energy_after[0]),
                "fixed_step_size": float(step_size),
                "inner_steps_used": len(curve) - 1,
                "inner_converged": convergence_hit,
                "convergence_reason": convergence_hit_reason,
            }
        )
        frame_rows.append(diagnostics)
        residual_curve.append(padded_curve(curve, inner_steps + 1))
        if diagnostics["failed"]:
            failure_frame = frame
            positions.append(p[0].detach().cpu())
            continue
        p, v = advance_state(p, y, params, next_time=next_time)
        positions.append(p[0].detach().cpu())

    return {
        "solver": "gd_fixed_lr_5e-5" if step_size == 5e-5 else f"gd_fixed_lr_{step_size:g}",
        "positions": torch.stack(positions, dim=0),
        "residual_by_frame_and_iteration": torch.tensor(residual_curve, dtype=torch.float64),
        "frames": frame_rows,
        "failure_frame": failure_frame,
        "line_search_failures": 0,
        "fixed_step_size": float(step_size),
    }


@torch.no_grad()
def run_mass_preconditioned_fixed_gd_rollout(
    *,
    scenario: ScenarioSpec,
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
    step_size: float,
    convergence: InnerConvergence,
) -> dict[str, Any]:
    params = build_batched_parameters((scenario,), device=device, dtype=dtype)
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    positions = [p[0].detach().cpu()]
    residual_curve = []
    frame_rows: list[dict[str, Any]] = []
    failure_frame: int | None = None

    for frame in range(rollout_frames):
        if failure_frame is not None:
            positions.append(p[0].detach().cpu())
            residual_curve.append([float("nan")] * (inner_steps + 1))
            frame_rows.append({"frame": frame, "failed": True})
            continue

        q = make_q(p, v, params)
        next_time = torch.full((1,), (frame + 1) * params.dt, dtype=dtype, device=device)
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(p, params, targets)
        curve = [
            frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
        ]
        energy_before = variational_energy(y, q, params, targets)
        convergence_hit = False
        convergence_hit_reason = ""

        for _ in range(inner_steps):
            y_before = y.clone()
            y = mass_preconditioned_gradient_descent_step(
                y=y,
                q=q,
                params=params,
                targets=targets,
                step_size=step_size,
            ).reshape_as(y)
            current_residual = frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
            curve.append(current_residual)
            reason = convergence_reason(
                initial_residual=curve[0],
                current_residual=current_residual,
                normalized_step=normalized_free_step(y_before, y, params),
                config=convergence,
            )
            if reason is not None:
                convergence_hit = True
                convergence_hit_reason = reason
                break

        diagnostics = frame_diagnostics(
            y=y,
            q=q,
            params=params,
            targets=targets,
            thresholds=thresholds,
        )
        energy_after = variational_energy(y, q, params, targets)
        diagnostics.update(
            {
                "frame": frame,
                "initial_residual": curve[0],
                "final_residual": curve[-1],
                "residual_ratio": curve[-1] / max(curve[0], torch.finfo(dtype).eps),
                "energy_before": _finite_float(energy_before[0]),
                "energy_after": _finite_float(energy_after[0]),
                "fixed_step_size": float(step_size),
                "inner_steps_used": len(curve) - 1,
                "inner_converged": convergence_hit,
                "convergence_reason": convergence_hit_reason,
            }
        )
        frame_rows.append(diagnostics)
        residual_curve.append(padded_curve(curve, inner_steps + 1))
        if diagnostics["failed"]:
            failure_frame = frame
            positions.append(p[0].detach().cpu())
            continue
        p, v = advance_state(p, y, params, next_time=next_time)
        positions.append(p[0].detach().cpu())

    return {
        "solver": "mass_preconditioned_gd_fixed",
        "positions": torch.stack(positions, dim=0),
        "residual_by_frame_and_iteration": torch.tensor(residual_curve, dtype=torch.float64),
        "frames": frame_rows,
        "failure_frame": failure_frame,
        "line_search_failures": 0,
        "fixed_step_size": float(step_size),
    }


@torch.no_grad()
def run_line_search_gd_rollout(
    *,
    scenario: ScenarioSpec,
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
    step_size: float,
    max_reductions: int,
    max_growths: int,
    convergence: InnerConvergence,
) -> dict[str, Any]:
    params = build_batched_parameters((scenario,), device=device, dtype=dtype)
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    positions = [p[0].detach().cpu()]
    residual_curve = []
    frame_rows: list[dict[str, Any]] = []
    failure_frame: int | None = None
    line_search_failures = 0

    for frame in range(rollout_frames):
        if failure_frame is not None:
            positions.append(p[0].detach().cpu())
            residual_curve.append([float("nan")] * (inner_steps + 1))
            frame_rows.append({"frame": frame, "failed": True})
            continue

        q = make_q(p, v, params)
        next_time = torch.full((1,), (frame + 1) * params.dt, dtype=dtype, device=device)
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(p, params, targets)
        curve = [
            frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
        ]
        energy_before = variational_energy(y, q, params, targets)
        accepted_scales: list[float] = []
        line_search_trials = 0
        frame_line_search_failures = 0
        convergence_hit = False
        convergence_hit_reason = ""

        for _ in range(inner_steps):
            y_before = y.clone()
            y, accepted, accepted_scale, trials = line_search_gradient_descent_step(
                y=y,
                q=q,
                params=params,
                targets=targets,
                step_size=step_size,
                max_reductions=max_reductions,
                max_growths=max_growths,
            )
            line_search_trials += int(trials)
            if not accepted:
                line_search_failures += 1
                frame_line_search_failures += 1
            accepted_scales.append(float(accepted_scale))
            current_residual = frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
            curve.append(current_residual)
            reason = convergence_reason(
                initial_residual=curve[0],
                current_residual=current_residual,
                normalized_step=normalized_free_step(y_before, y, params),
                config=convergence,
            )
            if reason is not None:
                convergence_hit = True
                convergence_hit_reason = reason
                break

        diagnostics = frame_diagnostics(
            y=y,
            q=q,
            params=params,
            targets=targets,
            thresholds=thresholds,
        )
        energy_after = variational_energy(y, q, params, targets)
        nonzero_scales = [value for value in accepted_scales if value > 0.0]
        diagnostics.update(
            {
                "frame": frame,
                "initial_residual": curve[0],
                "final_residual": curve[-1],
                "residual_ratio": curve[-1] / max(curve[0], torch.finfo(dtype).eps),
                "energy_before": _finite_float(energy_before[0]),
                "energy_after": _finite_float(energy_after[0]),
                "accepted_step_min": min(nonzero_scales) if nonzero_scales else 0.0,
                "accepted_step_median": (
                    sorted(nonzero_scales)[len(nonzero_scales) // 2]
                    if nonzero_scales
                    else 0.0
                ),
                "inner_steps_used": len(curve) - 1,
                "inner_converged": convergence_hit,
                "convergence_reason": convergence_hit_reason,
                "line_search_trials": line_search_trials,
                "line_search_failures_frame": frame_line_search_failures,
            }
        )
        frame_rows.append(diagnostics)
        residual_curve.append(padded_curve(curve, inner_steps + 1))
        if diagnostics["failed"]:
            failure_frame = frame
            positions.append(p[0].detach().cpu())
            continue
        p, v = advance_state(p, y, params, next_time=next_time)
        positions.append(p[0].detach().cpu())

    return {
        "solver": "line_search_gd",
        "positions": torch.stack(positions, dim=0),
        "residual_by_frame_and_iteration": torch.tensor(residual_curve, dtype=torch.float64),
        "frames": frame_rows,
        "failure_frame": failure_frame,
        "line_search_failures": line_search_failures,
        "initial_step_size": float(step_size),
        "line_search_growths": int(max_growths),
    }


@torch.no_grad()
def run_lbfgs_rollout(
    *,
    scenario: ScenarioSpec,
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
    step_size: float,
    history_size: int,
    max_reductions: int,
    convergence: InnerConvergence,
) -> dict[str, Any]:
    params = build_batched_parameters((scenario,), device=device, dtype=dtype)
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    positions = [p[0].detach().cpu()]
    residual_curve = []
    frame_rows: list[dict[str, Any]] = []
    failure_frame: int | None = None
    line_search_failures = 0
    descent_fallbacks = 0

    for frame in range(rollout_frames):
        if failure_frame is not None:
            positions.append(p[0].detach().cpu())
            residual_curve.append([float("nan")] * (inner_steps + 1))
            frame_rows.append({"frame": frame, "failed": True})
            continue

        q = make_q(p, v, params)
        next_time = torch.full((1,), (frame + 1) * params.dt, dtype=dtype, device=device)
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(p, params, targets)
        curve = [
            frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
        ]
        energy_before = variational_energy(y, q, params, targets)
        accepted_scales: list[float] = []
        s_history: list[torch.Tensor] = []
        y_history: list[torch.Tensor] = []
        line_search_trials = 0
        frame_line_search_failures = 0
        frame_descent_fallbacks = 0
        convergence_hit = False
        convergence_hit_reason = ""

        for _ in range(inner_steps):
            y_before = y.clone()
            y_flat_before = y.reshape(params.batch_size, -1)
            gradient_before = stationarity_residual(
                y,
                q,
                params,
                targets,
            ).reshape(params.batch_size, -1)
            gradient_before = gradient_before * free_update_gate(
                params,
                flattened=True,
            ).to(params.dtype)
            direction = lbfgs_direction(gradient_before, s_history, y_history)
            direction = direction * free_update_gate(params, flattened=True).to(params.dtype)
            directional = torch.sum(gradient_before * direction, dim=-1)
            if (
                not bool(torch.isfinite(direction).all())
                or not bool(torch.isfinite(directional).all())
                or float(directional[0].item()) >= 0.0
            ):
                direction = -gradient_before
                descent_fallbacks += 1
                frame_descent_fallbacks += 1

            y, accepted, accepted_scale, trials = armijo_direction_step(
                y=y,
                q=q,
                params=params,
                targets=targets,
                direction=direction,
                step_size=step_size,
                max_reductions=max_reductions,
            )
            line_search_trials += int(trials)
            if not accepted:
                line_search_failures += 1
                frame_line_search_failures += 1
            accepted_scales.append(float(accepted_scale))

            gradient_after = stationarity_residual(
                y,
                q,
                params,
                targets,
            ).reshape(params.batch_size, -1)
            gradient_after = gradient_after * free_update_gate(
                params,
                flattened=True,
            ).to(params.dtype)
            s_vec = y.reshape(params.batch_size, -1) - y_flat_before
            y_vec = gradient_after - gradient_before
            sy = torch.sum(s_vec * y_vec, dim=-1)
            if bool(torch.isfinite(s_vec).all()) and bool(torch.isfinite(y_vec).all()) and float(sy[0].item()) > torch.finfo(dtype).eps:
                s_history.append(s_vec.detach())
                y_history.append(y_vec.detach())
                if len(s_history) > history_size:
                    s_history.pop(0)
                    y_history.pop(0)

            current_residual = frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )["residual"]
            curve.append(current_residual)
            reason = convergence_reason(
                initial_residual=curve[0],
                current_residual=current_residual,
                normalized_step=normalized_free_step(y_before, y, params),
                config=convergence,
            )
            if reason is not None:
                convergence_hit = True
                convergence_hit_reason = reason
                break

        diagnostics = frame_diagnostics(
            y=y,
            q=q,
            params=params,
            targets=targets,
            thresholds=thresholds,
        )
        energy_after = variational_energy(y, q, params, targets)
        nonzero_scales = [value for value in accepted_scales if value > 0.0]
        diagnostics.update(
            {
                "frame": frame,
                "initial_residual": curve[0],
                "final_residual": curve[-1],
                "residual_ratio": curve[-1] / max(curve[0], torch.finfo(dtype).eps),
                "energy_before": _finite_float(energy_before[0]),
                "energy_after": _finite_float(energy_after[0]),
                "accepted_step_min": min(nonzero_scales) if nonzero_scales else 0.0,
                "accepted_step_median": (
                    sorted(nonzero_scales)[len(nonzero_scales) // 2]
                    if nonzero_scales
                    else 0.0
                ),
                "line_search_trials": line_search_trials,
                "line_search_failures_frame": frame_line_search_failures,
                "lbfgs_descent_fallbacks_frame": frame_descent_fallbacks,
                "lbfgs_history_size": len(s_history),
                "inner_steps_used": len(curve) - 1,
                "inner_converged": convergence_hit,
                "convergence_reason": convergence_hit_reason,
            }
        )
        frame_rows.append(diagnostics)
        residual_curve.append(padded_curve(curve, inner_steps + 1))
        if diagnostics["failed"]:
            failure_frame = frame
            positions.append(p[0].detach().cpu())
            continue
        p, v = advance_state(p, y, params, next_time=next_time)
        positions.append(p[0].detach().cpu())

    return {
        "solver": f"lbfgs_line_search_h{history_size}",
        "positions": torch.stack(positions, dim=0),
        "residual_by_frame_and_iteration": torch.tensor(residual_curve, dtype=torch.float64),
        "frames": frame_rows,
        "failure_frame": failure_frame,
        "line_search_failures": line_search_failures,
        "descent_fallbacks": descent_fallbacks,
        "history_size": int(history_size),
        "initial_step_size": float(step_size),
    }


def run_newton_rollout(
    *,
    scenario: ScenarioSpec,
    rollout_frames: int,
    inner_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
    convergence: InnerConvergence,
) -> dict[str, Any]:
    params = build_batched_parameters((scenario,), device=device, dtype=dtype)
    p = params.initial_positions.clone()
    v = params.initial_velocities.clone()
    positions = [p[0].detach().cpu()]
    residual_curve = []
    frame_rows: list[dict[str, Any]] = []
    failure_frame: int | None = None
    linear_solve_failures = 0

    for frame in range(rollout_frames):
        if failure_frame is not None:
            positions.append(p[0].detach().cpu())
            residual_curve.append([float("nan")] * (inner_steps + 1))
            frame_rows.append({"frame": frame, "failed": True})
            continue

        with torch.no_grad():
            q = make_q(p, v, params)
            next_time = torch.full((1,), (frame + 1) * params.dt, dtype=dtype, device=device)
            targets, _ = dirichlet_targets(params, next_time)
            y = project_positions(p, params, targets)
            curve = [
                frame_diagnostics(
                    y=y,
                    q=q,
                    params=params,
                    targets=targets,
                    thresholds=thresholds,
                )["residual"]
            ]
            energy_before = variational_energy(y, q, params, targets)
        convergence_hit = False
        convergence_hit_reason = ""

        for _ in range(inner_steps):
            y_before = y.clone()
            y, accepted, stats = newton_step(
                y=y,
                q=q,
                params=params,
                targets=targets,
            )
            linear_solve_failures += int(stats["linear_solve_failures"])
            with torch.no_grad():
                current_residual = frame_diagnostics(
                    y=y,
                    q=q,
                    params=params,
                    targets=targets,
                    thresholds=thresholds,
                )["residual"]
                curve.append(current_residual)
                reason = convergence_reason(
                    initial_residual=curve[0],
                    current_residual=current_residual,
                    normalized_step=normalized_free_step(y_before, y, params),
                    config=convergence,
                )
                if reason is not None:
                    convergence_hit = True
                    convergence_hit_reason = reason
                    break

        with torch.no_grad():
            diagnostics = frame_diagnostics(
                y=y,
                q=q,
                params=params,
                targets=targets,
                thresholds=thresholds,
            )
            energy_after = variational_energy(y, q, params, targets)
            diagnostics.update(
                {
                    "frame": frame,
                    "initial_residual": curve[0],
                    "final_residual": curve[-1],
                    "residual_ratio": curve[-1] / max(curve[0], torch.finfo(dtype).eps),
                    "energy_before": _finite_float(energy_before[0]),
                    "energy_after": _finite_float(energy_after[0]),
                    "newton_step_accepted": bool(accepted),
                    "inner_steps_used": len(curve) - 1,
                    "inner_converged": convergence_hit,
                    "convergence_reason": convergence_hit_reason,
                }
            )
            frame_rows.append(diagnostics)
            residual_curve.append(padded_curve(curve, inner_steps + 1))
            if diagnostics["failed"]:
                failure_frame = frame
                positions.append(p[0].detach().cpu())
                continue
            p, v = advance_state(p, y, params, next_time=next_time)
            positions.append(p[0].detach().cpu())

    return {
        "solver": "newton",
        "positions": torch.stack(positions, dim=0),
        "residual_by_frame_and_iteration": torch.tensor(residual_curve, dtype=torch.float64),
        "frames": frame_rows,
        "failure_frame": failure_frame,
        "line_search_failures": 0,
        "linear_solve_failures": linear_solve_failures,
    }


def finite_mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def summarize_rollout(result: dict[str, Any], rollout_frames: int) -> dict[str, Any]:
    frame_rows = result["frames"]
    valid = [row for row in frame_rows if not bool(row.get("failed", False))]
    final_residual = [float(row.get("final_residual", float("nan"))) for row in valid]
    ratios = [float(row.get("residual_ratio", float("nan"))) for row in valid]
    inner_steps_used = [
        float(row["inner_steps_used"])
        for row in valid
        if "inner_steps_used" in row and math.isfinite(float(row["inner_steps_used"]))
    ]
    converged = [bool(row.get("inner_converged", False)) for row in valid]
    energy_increases = [
        float(row["energy_after"]) > float(row["energy_before"])
        for row in valid
        if math.isfinite(float(row.get("energy_after", float("nan"))))
        and math.isfinite(float(row.get("energy_before", float("nan"))))
    ]
    failure_frame = result["failure_frame"]
    survived = rollout_frames if failure_frame is None else int(failure_frame)
    return {
        "solver": result["solver"],
        "rollout_frames": rollout_frames,
        "survival_frames": survived,
        "failed": failure_frame is not None,
        "failure_frame": failure_frame,
        "final_residual_mean": finite_mean(final_residual),
        "residual_ratio_mean": finite_mean(ratios),
        "inner_steps_used_mean": finite_mean(inner_steps_used),
        "inner_convergence_fraction": (
            float(sum(converged) / len(converged))
            if converged
            else None
        ),
        "energy_increase_fraction": (
            float(sum(energy_increases) / len(energy_increases))
            if energy_increases
            else None
        ),
        "line_search_failures": int(result.get("line_search_failures", 0)),
    }


def write_frame_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    import csv

    rows = []
    for result in results:
        for row in result["frames"]:
            rows.append({"solver": result["solver"], **row})
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def worst_residual_frame(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best = {
        "frame": 0,
        "solver": "",
        "final_residual": float("-inf"),
    }
    for result in results:
        for row in result["frames"]:
            value = float(row.get("final_residual", float("nan")))
            if math.isfinite(value) and value > best["final_residual"]:
                best = {
                    "frame": int(row["frame"]),
                    "solver": str(result["solver"]),
                    "final_residual": value,
                }
    if not math.isfinite(float(best["final_residual"])):
        return {"frame": 0, "solver": "", "final_residual": float("nan")}
    return best


def median_residual_frame(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for row in result["frames"]:
            value = float(row.get("final_residual", float("nan")))
            if math.isfinite(value):
                rows.append(
                    {
                        "frame": int(row["frame"]),
                        "solver": str(result["solver"]),
                        "final_residual": value,
                    }
                )
    if not rows:
        return {"frame": 0, "solver": "", "final_residual": float("nan")}
    rows.sort(key=lambda row: float(row["final_residual"]))
    return rows[len(rows) // 2]


def plot_diagnostics(output: Path, results: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    worst = worst_residual_frame(results)
    median = median_residual_frame(results)
    worst_frame = int(worst["frame"])
    median_frame = int(median["frame"])
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 12.5), sharex=False)
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")
    for index, result in enumerate(results):
        frames = [int(row["frame"]) for row in result["frames"]]
        residual = [float(row.get("final_residual", float("nan"))) for row in result["frames"]]
        ratio = [float(row.get("residual_ratio", float("nan"))) for row in result["frames"]]
        color = colors[index % len(colors)]
        axes[0].plot(frames, residual, label=result["solver"], color=color)
        axes[1].plot(frames, ratio, label=result["solver"], color=color)
        inner = result.get("residual_by_frame_and_iteration")
        if torch.is_tensor(inner) and inner.ndim == 2:
            for axis, frame in ((axes[2], worst_frame), (axes[3], median_frame)):
                if frame >= inner.shape[0]:
                    continue
                curve = inner[frame].detach().cpu().double()
                finite = torch.isfinite(curve)
                if bool(finite.any()):
                    x = list(range(int(curve.shape[0])))
                    y = [float(value) for value in curve.tolist()]
                    axis.plot(x, y, label=result["solver"], color=color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("final residual")
    axes[0].axvline(worst_frame, color="0.25", linestyle="--", linewidth=1.0, label="worst frame")
    axes[0].axvline(median_frame, color="0.55", linestyle="--", linewidth=1.0, label="median frame")
    axes[0].set_title(
        f"worst final residual frame={worst_frame} "
        f"solver={worst['solver']} value={float(worst['final_residual']):.3e}; "
        f"median frame={median_frame} solver={median['solver']} "
        f"value={float(median['final_residual']):.3e}"
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("residual ratio")
    axes[1].set_xlabel("physical frame")
    axes[1].axvline(worst_frame, color="0.25", linestyle="--", linewidth=1.0)
    axes[1].axvline(median_frame, color="0.55", linestyle="--", linewidth=1.0)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("inner iteration")
    axes[2].set_ylabel("residual")
    axes[2].set_title(f"inner residual at worst frame {worst_frame}")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("inner iteration")
    axes[3].set_ylabel("residual")
    axes[3].set_title(f"inner residual at median frame {median_frame}")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_line_search_times(output: Path, results: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    colors = ("tab:orange", "tab:red", "tab:purple", "tab:brown")
    plotted = False
    for index, result in enumerate(results):
        frames: list[int] = []
        trials: list[float] = []
        for row in result["frames"]:
            value = row.get("line_search_trials")
            if value is None:
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                frames.append(int(row["frame"]))
                trials.append(numeric)
        if not frames:
            continue
        ax.plot(
            frames,
            trials,
            label=result["solver"],
            color=colors[index % len(colors)],
        )
        plotted = True
    ax.set_xlabel("physical frame")
    ax.set_ylabel("line search candidate evaluations")
    ax.set_title("Line search times vs frame")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no line-search solver", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def result_payload(results: Sequence[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "manifest": manifest,
        "results": list(results),
    }
    for result in results:
        key = str(result["solver"]).replace("-", "_")
        payload[key] = result
    return payload


def safe_solver_filename(name: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"_", "-", "."} else "_"
        for character in name
    )


def save_rollout_outputs(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    results: Sequence[dict[str, Any]],
    render_results: Sequence[dict[str, Any]] | None,
    scenario: ScenarioSpec,
    args: argparse.Namespace,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result_payload(results, manifest), output_dir / "rollout_compare.pt")
    write_json(output_dir / "metrics.json", manifest)
    write_frame_csv(output_dir / "per_frame.csv", results)
    plot_diagnostics(output_dir / "diagnostics.png", results)
    plot_line_search_times(output_dir / "line_search_times_vs_frame.png", results)

    if args.render_format == "none":
        return []
    render_outputs: list[Path] = []
    selected_render_results = results if render_results is None else render_results
    for result in selected_render_results:
        solver = safe_solver_filename(str(result["solver"]))
        render_output = output_dir / f"rollout_{solver}.{args.render_format}"
        render_comparison(
            output=render_output,
            results=[result],
            scenario=scenario,
            fps=args.fps,
            stride=args.render_stride,
            format_name=args.render_format,
        )
        render_outputs.append(render_output)
    return render_outputs


def axis_box(*arrays) -> tuple[Any, float]:
    import numpy as np

    valid = [array.reshape(-1, 3) for array in arrays if array.size]
    stacked = np.concatenate(valid, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) * 0.58, 1e-3)
    return center, radius


def render_comparison(
    *,
    output: Path,
    results: Sequence[dict[str, Any]],
    scenario: ScenarioSpec,
    fps: int,
    stride: int,
    format_name: Literal["mp4", "gif"],
) -> Path:
    if stride <= 0:
        raise ValueError("--render-stride must be positive")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    import numpy as np

    params = build_batched_parameters((scenario,), device="cpu", dtype=torch.float64)
    edges = [tuple(edge) for edge in params.topology.edges]
    fixed = torch.nonzero(params.fixed_mask[0], as_tuple=False).flatten().tolist()
    if not results:
        raise ValueError("at least one result is required for rendering")
    position_arrays = [result["positions"][::stride].numpy() for result in results]
    center, radius = axis_box(*position_arrays)

    cols = 2 if len(results) > 1 else 1
    rows = int(math.ceil(len(results) / cols))
    fig = plt.figure(figsize=(6.0 * cols, 5.5 * rows))
    axes = [
        fig.add_subplot(rows, cols, index + 1, projection="3d")
        for index in range(len(results))
    ]
    artists = []
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple")
    for index, (axis, result) in enumerate(zip(axes, results)):
        color = colors[index % len(colors)]
        lines = [
            axis.plot([], [], [], color=color, linewidth=0.45)[0]
            for _ in edges
        ]
        points = axis.scatter([], [], [], s=5, color=color)
        pins = axis.scatter([], [], [], s=35, marker="s", color="tab:red")
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.view_init(elev=22, azim=-60)
        axis.set_title(str(result["solver"]))
        artists.append((axis, lines, points, pins))

    def residual_text(result: dict[str, Any], frame: int) -> str:
        if frame == 0:
            return "initial"
        rows = result["frames"]
        index = min(frame - 1, len(rows) - 1)
        value = float(rows[index].get("final_residual", float("nan")))
        return f"residual={value:.3e}" if math.isfinite(value) else "residual=nan"

    def update(frame: int):
        original = frame * stride
        drawn = []
        for (axis, lines, points, pins), positions, result in zip(
            artists,
            (array[frame] for array in position_arrays),
            results,
        ):
            for line, (left, right) in zip(lines, edges):
                line.set_data_3d(
                    [positions[left, 0], positions[right, 0]],
                    [positions[left, 1], positions[right, 1]],
                    [positions[left, 2], positions[right, 2]],
                )
            points._offsets3d = (
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
            )
            pins._offsets3d = (
                positions[fixed, 0],
                positions[fixed, 1],
                positions[fixed, 2],
            )
            axis.set_title(f"{result['solver']} frame {original:03d}\n{residual_text(result, original)}")
            drawn.extend([*lines, points, pins])
        return drawn

    animation = FuncAnimation(
        fig,
        update,
        frames=min(len(array) for array in position_arrays),
        interval=1000 / fps,
        blit=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps) if format_name == "mp4" else PillowWriter(fps=fps)
    animation.save(output, writer=writer, dpi=140)
    plt.close(fig)
    return output


def run_baseline_suite(
    *,
    scenario: ScenarioSpec,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    thresholds: FailureThresholds,
    convergence: InnerConvergence,
) -> list[dict[str, Any]]:
    fixed_gd = run_fixed_gd_rollout(
        scenario=scenario,
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        step_size=args.fixed_gd_step_size,
        convergence=convergence,
    )
    line_search_gd = run_line_search_gd_rollout(
        scenario=scenario,
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        step_size=args.line_search_gd_step_size,
        max_reductions=args.line_search_gd_reductions,
        max_growths=args.line_search_gd_growths,
        convergence=convergence,
    )
    mass_preconditioned_fixed = run_mass_preconditioned_fixed_gd_rollout(
        scenario=scenario,
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        step_size=args.mass_preconditioned_gd_step_size,
        convergence=convergence,
    )
    mass_preconditioned = run_baseline_rollout(
        scenario=scenario,
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        step_size=args.baseline_step_size,
        max_reductions=args.baseline_line_search_reductions,
        convergence=convergence,
    )
    lbfgs = run_lbfgs_rollout(
        scenario=scenario,
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        step_size=args.lbfgs_step_size,
        history_size=args.lbfgs_history_size,
        max_reductions=args.lbfgs_line_search_reductions,
        convergence=convergence,
    )
    newton = run_newton_rollout(
        scenario=scenario,
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        convergence=convergence,
    )
    return [
        fixed_gd,
        line_search_gd,
        mass_preconditioned_fixed,
        mass_preconditioned,
        lbfgs,
        newton,
    ]


def baseline_manifest(
    *,
    args: argparse.Namespace,
    scenario: ScenarioSpec,
    selected_row: dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
    convergence: InnerConvergence,
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "completed": True,
        "mode": "baseline",
        "output_kind": "baseline_cache",
        "split": args.split,
        "catalogue_key": split_key(args.split, args.catalogue),
        "motion_index": int(args.motion_index),
        "selected_motion": selected_row,
        "scenario": asdict(scenario),
        "scenario_labels": scenario_labels(scenario),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "rollout_frames": int(args.rollout_frames),
        "inner_steps": int(args.inner_steps),
        "inner_early_stop": asdict(convergence),
        "baselines": [
            {
                "solver": "gd_fixed_lr_5e-5",
                "step_size": float(args.fixed_gd_step_size),
                "line_search": None,
            },
            {
                "solver": "line_search_gd",
                "step_size": float(args.line_search_gd_step_size),
                "line_search": "Armijo growth and backtracking",
                "line_search_growths": int(args.line_search_gd_growths),
                "line_search_reductions": int(args.line_search_gd_reductions),
            },
            {
                "solver": "mass_preconditioned_gd_fixed",
                "step_size": float(args.mass_preconditioned_gd_step_size),
                "line_search": None,
            },
            {
                "solver": "mass_preconditioned_line_search_gd",
                "step_size": float(args.baseline_step_size),
                "line_search": "energy non-increase backtracking",
                "line_search_reductions": int(args.baseline_line_search_reductions),
            },
            {
                "solver": f"lbfgs_line_search_h{args.lbfgs_history_size}",
                "step_size": float(args.lbfgs_step_size),
                "history_size": int(args.lbfgs_history_size),
                "line_search": "Armijo backtracking",
                "line_search_reductions": int(args.lbfgs_line_search_reductions),
            },
            {
                "solver": "newton",
                "line_search": None,
                "damping": None,
                "state": "free degrees of freedom only",
            },
        ],
        "summaries": [summarize_rollout(result, args.rollout_frames) for result in results],
    }


def baseline_is_complete(output_dir: Path) -> bool:
    metrics_path = output_dir / "metrics.json"
    payload_path = output_dir / "rollout_compare.pt"
    if not metrics_path.exists() or not payload_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if metrics.get("mode") != "baseline" or not bool(metrics.get("completed", False)):
        return False
    summaries = metrics.get("summaries", [])
    solvers = {str(row.get("solver", "")) for row in summaries if isinstance(row, dict)}
    return set(REQUIRED_BASELINE_SOLVERS).issubset(solvers)


def load_baseline_results(output_dir: Path) -> list[dict[str, Any]]:
    payload = torch.load(
        output_dir / "rollout_compare.pt",
        map_location="cpu",
        weights_only=False,
    )
    return list(payload["results"])


def run_and_save_baseline_suite(
    *,
    args: argparse.Namespace,
    scenario: ScenarioSpec,
    selected_row: dict[str, Any],
    thresholds: FailureThresholds,
    convergence: InnerConvergence,
    strict_existing: bool,
) -> tuple[list[dict[str, Any]], Path, list[Path]]:
    output_dir = args.output_dir if args.output_dir is not None and args.mode == "baseline" else baseline_output_dir(args)
    if output_dir.exists() and strict_existing and not args.overwrite and baseline_is_complete(output_dir):
        print(f"baseline cache 已完成，跳过：{output_dir}")
        return load_baseline_results(output_dir), output_dir, []
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    dtype = baseline_dtype(args)
    results = run_baseline_suite(
        scenario=scenario,
        args=args,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        convergence=convergence,
    )
    manifest = baseline_manifest(
        args=args,
        scenario=scenario,
        selected_row=selected_row,
        dtype=dtype,
        device=device,
        convergence=convergence,
        results=results,
    )
    render_outputs = save_rollout_outputs(
        output_dir=output_dir,
        manifest=manifest,
        results=results,
        render_results=None,
        scenario=scenario,
        args=args,
    )
    return results, output_dir, render_outputs


def load_or_create_baselines(
    *,
    args: argparse.Namespace,
    scenario: ScenarioSpec,
    selected_row: dict[str, Any],
    thresholds: FailureThresholds,
    convergence: InnerConvergence,
) -> tuple[list[dict[str, Any]], Path]:
    output_dir = baseline_output_dir(args)
    if baseline_is_complete(output_dir) and not args.refresh_baseline:
        print(f"baseline cache 已完成，读取：{output_dir}")
        return load_baseline_results(output_dir), output_dir
    results, output_dir, _ = run_and_save_baseline_suite(
        args=args,
        scenario=scenario,
        selected_row=selected_row,
        thresholds=thresholds,
        convergence=convergence,
        strict_existing=False,
    )
    return results, output_dir


def main() -> None:
    args = parse_args()
    if args.list_motions:
        print_motion_table(args)
        return
    if args.rollout_frames <= 0 or args.inner_steps <= 0:
        raise ValueError("--rollout-frames and --inner-steps must be positive")
    if args.baseline_step_size <= 0:
        raise ValueError("--baseline-step-size must be positive")
    if args.mass_preconditioned_gd_step_size <= 0:
        raise ValueError("--mass-preconditioned-gd-step-size must be positive")
    if args.fixed_gd_step_size <= 0:
        raise ValueError("--fixed-gd-step-size must be positive")
    if args.line_search_gd_step_size <= 0:
        raise ValueError("--line-search-gd-step-size must be positive")
    if args.line_search_gd_reductions < 0:
        raise ValueError("--line-search-gd-reductions must be non-negative")
    if args.line_search_gd_growths < 0:
        raise ValueError("--line-search-gd-growths must be non-negative")
    if args.lbfgs_history_size <= 0:
        raise ValueError("--lbfgs-history-size must be positive")
    if args.lbfgs_step_size <= 0:
        raise ValueError("--lbfgs-step-size must be positive")
    if args.lbfgs_line_search_reductions < 0:
        raise ValueError("--lbfgs-line-search-reductions must be non-negative")
    if args.render_stride <= 0:
        raise ValueError("--render-stride must be positive")
    convergence = convergence_from_args(args)
    validate_convergence(convergence)

    scenario = selected_scenario(args)
    selected_row = scenario_row(args.motion_index, scenario)
    print(
        "selected motion: "
        f"index={selected_row['index']} "
        f"scenario_id={selected_row['scenario_id']} "
        f"group={selected_row['group']} "
        f"boundary={selected_row['boundary']} "
        f"dirichlet={selected_row['dirichlet']} "
        f"material={selected_row['material']}"
    )
    thresholds = FailureThresholds(
        max_residual=args.max_residual,
        max_abs_position=args.max_abs_position,
        min_edge_ratio=args.min_edge_ratio,
        max_edge_ratio=args.max_edge_ratio,
        max_constraint_error=args.max_constraint_error,
    )

    if args.mode == "baseline":
        results, output_dir, render_outputs = run_and_save_baseline_suite(
            args=args,
            scenario=scenario,
            selected_row=selected_row,
            thresholds=thresholds,
            convergence=convergence,
            strict_existing=True,
        )
        summaries = [summarize_rollout(result, args.rollout_frames) for result in results]
        print(f"baseline rollout 完成：{output_dir}")
        for render_output in render_outputs:
            print(f"渲染输出：{render_output}")
        print(
            "summary: "
            + " ".join(
                f"{summary['solver']}_ratio_mean={summary['residual_ratio_mean']}"
                for summary in summaries
            )
        )
        return

    run_dir = run_directory(args)
    selected_checkpoint = checkpoint_path(args, run_dir)
    if not selected_checkpoint.exists():
        raise FileNotFoundError(selected_checkpoint)

    checkpoint = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    actual_run_dir = checkpoint_run_dir(selected_checkpoint, run_dir)
    output_dir = args.output_dir or default_output_dir(args, actual_run_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    dtype = resolve_dtype(args.dtype, checkpoint)
    spec = ModelSpec(**checkpoint["model_spec"])
    model = LearnedOptimizerMLP(
        full_state_dim=15 * 15 * 3,
        model_spec=spec,
        dtype=dtype,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    learned = run_model_rollout(
        model=model,
        scenario=scenario,
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        device=device,
        dtype=dtype,
        thresholds=thresholds,
        convergence=convergence,
    )
    baseline_results, baseline_dir = load_or_create_baselines(
        args=args,
        scenario=scenario,
        thresholds=thresholds,
        convergence=convergence,
        selected_row=selected_row,
    )

    learned_summary = summarize_rollout(learned, args.rollout_frames)
    results = [learned, *baseline_results]
    summaries = [learned_summary, *[
        summarize_rollout(result, args.rollout_frames) for result in baseline_results
    ]]
    worst_frame = worst_residual_frame(results)
    manifest = {
        "mode": "mlp",
        "checkpoint": str(selected_checkpoint),
        "checkpoint_update": int(checkpoint.get("update_count", 0)),
        "run_directory": str(actual_run_dir),
        "baseline_directory": str(baseline_dir),
        "model_spec": asdict(spec),
        "requested_model_spec": asdict(
            ModelSpec(
                activation=args.activation,
                depth=args.depth,
                width=args.width,
                use_bias=args.use_bias,
            )
        ),
        "split": args.split,
        "catalogue_key": split_key(args.split, args.catalogue),
        "motion_index": int(args.motion_index),
        "scenario": asdict(scenario),
        "scenario_labels": scenario_labels(scenario),
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "rollout_frames": int(args.rollout_frames),
        "inner_steps": int(args.inner_steps),
        "inner_early_stop": asdict(convergence),
        "worst_final_residual_frame": worst_frame,
        "baselines": [
            {
                "solver": "gd_fixed_lr_5e-5",
                "step_size": float(args.fixed_gd_step_size),
                "line_search": None,
            },
            {
                "solver": "line_search_gd",
                "step_size": float(args.line_search_gd_step_size),
                "line_search": "Armijo growth and backtracking",
                "line_search_growths": int(args.line_search_gd_growths),
                "line_search_reductions": int(args.line_search_gd_reductions),
            },
            {
                "solver": "mass_preconditioned_gd_fixed",
                "step_size": float(args.mass_preconditioned_gd_step_size),
                "line_search": None,
            },
            {
                "solver": "mass_preconditioned_line_search_gd",
                "step_size": float(args.baseline_step_size),
                "line_search": "energy non-increase backtracking",
                "line_search_reductions": int(args.baseline_line_search_reductions),
            },
            {
                "solver": f"lbfgs_line_search_h{args.lbfgs_history_size}",
                "step_size": float(args.lbfgs_step_size),
                "history_size": int(args.lbfgs_history_size),
                "line_search": "Armijo backtracking",
                "line_search_reductions": int(args.lbfgs_line_search_reductions),
            },
            {
                "solver": "newton",
                "line_search": None,
                "damping": None,
                "state": "free degrees of freedom only",
            },
        ],
        "summaries": summaries,
    }
    render_outputs = save_rollout_outputs(
        output_dir=output_dir,
        manifest=manifest,
        results=results,
        render_results=[learned],
        scenario=scenario,
        args=args,
    )

    print(f"单 motion rollout 对比完成：{output_dir}")
    print(f"baseline 来源：{baseline_dir}")
    for render_output in render_outputs:
        print(f"渲染输出：{render_output}")
    print(
        "summary: "
        + " ".join(
            f"{summary['solver']}_ratio_mean={summary['residual_ratio_mean']}"
            for summary in summaries
        )
    )


if __name__ == "__main__":
    main()
