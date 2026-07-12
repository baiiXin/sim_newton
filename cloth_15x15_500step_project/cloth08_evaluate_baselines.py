"""Evaluate GD, Adam, independent L-BFGS, full BFGS, and Newton baselines.

Every dataset row is an independent physical time-step problem initialized at x_n.
Each baseline runs exactly `--steps` inner iterations (default 50). L-BFGS and BFGS
use independent curvature state and independent Armijo backtracking for every row;
no curvature history is shared across different time-step problems.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cloth02_dataset_catalog import DEFAULT_EVALUATION_ITERATIONS, load_dataset
from cloth03_solvers_and_models import (
    DEFAULT_DEVICE,
    SPATIAL_DIM,
    AdamState,
    apply_adam_update_full,
    apply_gradient_descent_update_full,
    apply_newton_update_full,
    free_state_from_full_state,
    project_fixed_vertices,
    stationarity_residual,
    stationarity_residual_norm_full,
    variational_energy,
)
from cloth_common import load_json, load_physical, save_json, summarize_residual_curve

DATASETS = ("validation_xn", "test_id_xn", "test_ood_xn", "test_all_xn")

GD_CANDIDATE_STEP_SIZES = (
    1e-9, 2e-9, 5e-9,
    1e-8, 2e-8, 5e-8,
    1e-7, 2e-7, 5e-7,
    1e-6, 2e-6, 5e-6,
    1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4,
    1e-3,
)

ADAM_CANDIDATE_LRS = (
    1e-8, 2e-8, 5e-8,
    1e-7, 2e-7, 5e-7,
    1e-6, 2e-6, 5e-6,
    1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 5e-2,
    1e-1, 2e-1, 5e-1,
    1.0,
)

LBFGS_CANDIDATES = tuple(
    {"history_size": history_size, "initial_step": initial_step}
    for history_size in (5, 10, 20)
    for initial_step in (0.25, 0.5, 1.0, 2.0)
)


def stratified_subset(dataset: dict[str, Any], max_points: int) -> dict[str, Any]:
    n = int(dataset["initial_y"].shape[0])
    if max_points <= 0 or n <= max_points:
        return dataset
    indices = torch.from_numpy(
        np.linspace(0, n - 1, max_points).round().astype(np.int64)
    ).unique(sorted=True)
    keys = [
        "initial_y", "q", "masses", "exact_y",
        "problem_index", "motion_index", "time_index",
    ]
    result = {key: dataset[key].index_select(0, indices).contiguous() for key in keys}
    result["metadata"] = dict(dataset.get("metadata", {}))
    result["metadata"]["parameter_selection_subset"] = int(indices.numel())
    result["metadata"]["subset_method"] = "evenly spaced rows across all validation motions/times"
    return result


def _mass_inverse_diagonal(masses: torch.Tensor, physical) -> torch.Tensor:
    return physical.dt**2 / masses.repeat_interleave(SPATIAL_DIM, dim=-1)


def _armijo_step(
    *,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    gradient: torch.Tensor,
    direction: torch.Tensor,
    energy: torch.Tensor,
    physical,
    initial_step: float,
    c1: float,
    shrink: float,
    max_reductions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    used_direction = direction.clone()
    directional = torch.sum(gradient * used_direction, dim=-1)
    fallback = -_mass_inverse_diagonal(masses, physical) * gradient
    bad_direction = (~torch.isfinite(directional)) | (directional >= 0.0)
    used_direction[bad_direction] = fallback[bad_direction]
    directional = torch.sum(gradient * used_direction, dim=-1)

    batch = y.shape[0]
    alpha = torch.full(
        (batch,), float(initial_step), dtype=y.dtype, device=y.device
    )
    accepted = torch.zeros(batch, dtype=torch.bool, device=y.device)
    y_next = y.clone()
    energy_next = energy.clone()
    accepted_alpha = torch.zeros_like(alpha)

    for _ in range(max_reductions + 1):
        active = ~accepted
        if not bool(active.any()):
            break
        candidate = y + alpha[:, None] * used_direction
        candidate_energy = variational_energy(candidate, q, masses, physical)
        armijo_rhs = energy + c1 * alpha * directional
        good = (
            active
            & torch.isfinite(candidate_energy)
            & torch.isfinite(candidate).all(dim=-1)
            & (candidate_energy <= armijo_rhs)
        )
        if bool(good.any()):
            y_next[good] = candidate[good]
            energy_next[good] = candidate_energy[good]
            accepted_alpha[good] = alpha[good]
            accepted[good] = True
        alpha[~accepted] *= shrink

    return y_next, energy_next, accepted_alpha, accepted, used_direction


def _lbfgs_direction(
    gradient: torch.Tensor,
    masses: torch.Tensor,
    physical,
    s_history: list[torch.Tensor],
    y_history: list[torch.Tensor],
    rho_history: list[torch.Tensor],
    valid_history: list[torch.Tensor],
) -> torch.Tensor:
    q_vec = gradient.clone()
    alphas: list[torch.Tensor] = []
    for s_vec, y_vec, rho, valid in zip(
        reversed(s_history),
        reversed(y_history),
        reversed(rho_history),
        reversed(valid_history),
    ):
        alpha = rho * torch.sum(s_vec * q_vec, dim=-1)
        alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
        q_vec = q_vec - alpha[:, None] * y_vec
        alphas.append(alpha)

    base_diagonal = _mass_inverse_diagonal(masses, physical)
    gamma = torch.ones(
        gradient.shape[0], dtype=gradient.dtype, device=gradient.device
    )
    gamma_set = torch.zeros_like(gamma, dtype=torch.bool)
    for s_vec, y_vec, valid in zip(
        reversed(s_history), reversed(y_history), reversed(valid_history)
    ):
        denominator = torch.sum(y_vec * (base_diagonal * y_vec), dim=-1)
        numerator = torch.sum(s_vec * y_vec, dim=-1)
        usable = valid & (~gamma_set) & (denominator > 1e-30) & (numerator > 0.0)
        gamma[usable] = numerator[usable] / denominator[usable]
        gamma_set |= usable
    r_vec = gamma[:, None] * base_diagonal * q_vec

    for index, (s_vec, y_vec, rho, valid) in enumerate(
        zip(s_history, y_history, rho_history, valid_history)
    ):
        beta = rho * torch.sum(y_vec * r_vec, dim=-1)
        beta = torch.where(valid, beta, torch.zeros_like(beta))
        alpha = alphas[len(alphas) - 1 - index]
        r_vec = r_vec + s_vec * (alpha - beta)[:, None]
    return -r_vec


def evaluate_quasi_newton(
    *,
    dataset: dict[str, Any],
    physical,
    method: str,
    steps: int,
    batch_size: int,
    device: torch.device,
    history_size: int = 10,
    initial_step: float = 1.0,
    armijo_c1: float = 1e-4,
    line_search_shrink: float = 0.5,
    max_line_search_reductions: int = 30,
) -> dict[str, Any]:
    if method not in {"lbfgs", "bfgs"}:
        raise ValueError(method)
    curves: list[torch.Tensor] = []
    line_search_failures = 0
    curvature_skips = 0
    started = time.perf_counter()

    for start in range(0, len(dataset["initial_y"]), batch_size):
        stop = min(start + batch_size, len(dataset["initial_y"]))
        y_full = project_fixed_vertices(
            dataset["initial_y"][start:stop].to(device), physical
        )
        q_full = dataset["q"][start:stop].to(device)
        masses = dataset["masses"][start:stop].to(device)
        y = free_state_from_full_state(y_full)
        q = free_state_from_full_state(q_full)
        gradient = stationarity_residual(y, q, masses, physical)
        energy = variational_energy(y, q, masses, physical)
        batch_curve = [torch.linalg.vector_norm(gradient, dim=-1).cpu()]

        s_history: list[torch.Tensor] = []
        y_history: list[torch.Tensor] = []
        rho_history: list[torch.Tensor] = []
        valid_history: list[torch.Tensor] = []
        inverse_hessian: torch.Tensor | None = None
        if method == "bfgs":
            diagonal = _mass_inverse_diagonal(masses, physical)
            inverse_hessian = torch.diag_embed(diagonal)

        for _ in range(steps):
            if method == "lbfgs":
                direction = _lbfgs_direction(
                    gradient,
                    masses,
                    physical,
                    s_history,
                    y_history,
                    rho_history,
                    valid_history,
                )
            else:
                assert inverse_hessian is not None
                direction = -torch.matmul(
                    inverse_hessian, gradient.unsqueeze(-1)
                ).squeeze(-1)

            y_next, energy_next, _, accepted, _ = _armijo_step(
                y=y,
                q=q,
                masses=masses,
                gradient=gradient,
                direction=direction,
                energy=energy,
                physical=physical,
                initial_step=initial_step,
                c1=armijo_c1,
                shrink=line_search_shrink,
                max_reductions=max_line_search_reductions,
            )
            line_search_failures += int((~accepted).sum().item())
            gradient_next = stationarity_residual(y_next, q, masses, physical)
            s_vec = y_next - y
            y_vec = gradient_next - gradient
            ys = torch.sum(y_vec * s_vec, dim=-1)
            threshold = 1e-10 * torch.linalg.vector_norm(s_vec, dim=-1) * torch.linalg.vector_norm(
                y_vec, dim=-1
            )
            valid = accepted & torch.isfinite(ys) & (ys > torch.maximum(threshold, torch.full_like(threshold, 1e-30)))
            curvature_skips += int((accepted & ~valid).sum().item())

            if method == "lbfgs":
                safe_ys = torch.where(valid, ys, torch.ones_like(ys))
                s_history.append(s_vec.detach())
                y_history.append(y_vec.detach())
                rho_history.append((1.0 / safe_ys).detach())
                valid_history.append(valid.detach())
                if len(s_history) > history_size:
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
                candidate_h = 0.5 * (
                    candidate_h + candidate_h.transpose(-1, -2)
                )
                finite_h = torch.isfinite(candidate_h).flatten(1).all(dim=1)
                update = valid & finite_h
                inverse_hessian[update] = candidate_h[update]

            y = y_next
            energy = energy_next
            gradient = gradient_next
            batch_curve.append(torch.linalg.vector_norm(gradient, dim=-1).cpu())

        curves.append(torch.stack(batch_curve, dim=1))

    curve = torch.cat(curves, dim=0).numpy().astype(float)
    curve[~np.isfinite(curve)] = np.inf
    summary = summarize_residual_curve(curve)
    summary.update(
        {
            "solver": method,
            "history_size": int(history_size) if method == "lbfgs" else None,
            "initial_step": float(initial_step),
            "line_search": "independent Armijo backtracking per problem",
            "line_search_failures": int(line_search_failures),
            "curvature_update_skips": int(curvature_skips),
            "elapsed_seconds": time.perf_counter() - started,
            "independent_problem_state": True,
        }
    )
    return {"summary": summary, "curve": torch.from_numpy(curve)}


def evaluate_standard_baseline(
    *,
    dataset: dict[str, Any],
    physical,
    solver: str,
    steps: int,
    batch_size: int,
    device: torch.device,
    params: dict[str, Any],
) -> dict[str, Any]:
    curves: list[torch.Tensor] = []
    started = time.perf_counter()
    for start in range(0, len(dataset["initial_y"]), batch_size):
        stop = min(start + batch_size, len(dataset["initial_y"]))
        y = project_fixed_vertices(dataset["initial_y"][start:stop].to(device), physical)
        q = dataset["q"][start:stop].to(device)
        masses = dataset["masses"][start:stop].to(device)
        adam_state: AdamState | None = None
        batch_curve: list[torch.Tensor] = []
        with torch.no_grad():
            for iteration in range(steps + 1):
                batch_curve.append(
                    stationarity_residual_norm_full(y, q, masses, physical).cpu()
                )
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
        curves.append(torch.stack(batch_curve, dim=1))

    curve = torch.cat(curves, dim=0).numpy().astype(float)
    curve[~np.isfinite(curve)] = np.inf
    summary = summarize_residual_curve(curve)
    summary.update(
        {
            "solver": solver,
            "params": params,
            "elapsed_seconds": time.perf_counter() - started,
            "independent_problem_state": True,
        }
    )
    return {"summary": summary, "curve": torch.from_numpy(curve)}


def evaluate_solver(
    *,
    dataset: dict[str, Any],
    physical,
    solver: str,
    steps: int,
    device: torch.device,
    params: dict[str, Any],
    batch_sizes: dict[str, int],
) -> dict[str, Any]:
    if solver in {"lbfgs", "bfgs"}:
        return evaluate_quasi_newton(
            dataset=dataset,
            physical=physical,
            method=solver,
            steps=steps,
            batch_size=batch_sizes[solver],
            device=device,
            history_size=int(params.get("history_size", 10)),
            initial_step=float(params.get("initial_step", 1.0)),
        )
    return evaluate_standard_baseline(
        dataset=dataset,
        physical=physical,
        solver=solver,
        steps=steps,
        batch_size=batch_sizes[solver],
        device=device,
        params=params,
    )


def select_parameters(
    *,
    validation: dict[str, Any],
    physical,
    steps: int,
    device: torch.device,
    selection_max_points: int,
    batch_sizes: dict[str, int],
) -> dict[str, Any]:
    subset = stratified_subset(validation, selection_max_points)
    selection: dict[str, Any] = {}

    gd_trials = []
    for step_size in GD_CANDIDATE_STEP_SIZES:
        result = evaluate_solver(
            dataset=subset,
            physical=physical,
            solver="gd",
            steps=steps,
            device=device,
            params={"step_size": step_size},
            batch_sizes=batch_sizes,
        )["summary"]
        gd_trials.append(result)
    selection["gd"] = {
        "selected": min(gd_trials, key=lambda item: item["selection_metric"])["params"],
        "trials": gd_trials,
    }

    adam_trials = []
    for learning_rate in ADAM_CANDIDATE_LRS:
        result = evaluate_solver(
            dataset=subset,
            physical=physical,
            solver="adam",
            steps=steps,
            device=device,
            params={"learning_rate": learning_rate},
            batch_sizes=batch_sizes,
        )["summary"]
        adam_trials.append(result)
    selection["adam"] = {
        "selected": min(adam_trials, key=lambda item: item["selection_metric"])["params"],
        "trials": adam_trials,
    }

    lbfgs_trials = []
    for params in LBFGS_CANDIDATES:
        result = evaluate_solver(
            dataset=subset,
            physical=physical,
            solver="lbfgs",
            steps=steps,
            device=device,
            params=params,
            batch_sizes=batch_sizes,
        )["summary"]
        result["params"] = dict(params)
        lbfgs_trials.append(result)
    selection["lbfgs"] = {
        "selected": min(lbfgs_trials, key=lambda item: item["selection_metric"])["params"],
        "trials": lbfgs_trials,
    }

    selection["bfgs"] = {"selected": {"initial_step": 1.0}, "trials": []}
    selection["newton"] = {"selected": {}, "trials": []}
    selection["metadata"] = {
        "selection_dataset": "validation_xn stratified subset",
        "selection_points": int(subset["initial_y"].shape[0]),
        "selection_steps": steps,
        "selection_metric": "final_residual_p95",
    }
    return selection


def plot_parameter_selection(selection: dict[str, Any], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for solver in ("gd", "adam", "lbfgs"):
        trials = selection[solver]["trials"]
        labels: list[str] = []
        values: list[float] = []
        for trial in trials:
            params = trial["params"]
            if solver == "gd":
                labels.append(f"{params['step_size']:.0e}")
            elif solver == "adam":
                labels.append(f"{params['learning_rate']:.0e}")
            else:
                labels.append(
                    f"h={params['history_size']},a0={params['initial_step']}"
                )
            values.append(float(trial["selection_metric"]))
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(labels)), 5))
        ax.plot(range(len(values)), np.maximum(values, 1e-30), marker="o")
        ax.set_yscale("log")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("candidate")
        ax.set_ylabel("validation final residual p95")
        ax.set_title(f"{solver} parameter selection")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{solver}_parameter_selection.png", dpi=180)
        plt.close(fig)


def plot_dataset_curves(metrics: dict[str, Any], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fields = {
        "mean": "residual_mean_by_iter",
        "p95": "residual_p95_by_iter",
        "max": "residual_max_by_iter",
    }
    for dataset_name, solver_metrics in metrics.items():
        for statistic, field in fields.items():
            fig, ax = plt.subplots(figsize=(8, 5))
            for solver, record in solver_metrics.items():
                values = np.asarray(record[field], dtype=float)
                ax.plot(
                    np.arange(len(values)),
                    np.maximum(values, 1e-30),
                    label=solver,
                )
            ax.set_yscale("log")
            ax.set_xlabel("inner iteration")
            ax.set_ylabel(f"{statistic} stationarity residual")
            ax.set_title(f"{dataset_name}: residual vs. iteration")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                figure_dir / f"{dataset_name}_residual_{statistic}_vs_iteration.png",
                dpi=180,
            )
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GD/Adam/independent L-BFGS/full BFGS/Newton baselines."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("cloth_15x15_500step_pipeline")
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--steps", type=int, default=DEFAULT_EVALUATION_ITERATIONS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--newton-batch-size", type=int, default=4)
    parser.add_argument("--lbfgs-batch-size", type=int, default=64)
    parser.add_argument("--bfgs-batch-size", type=int, default=2)
    parser.add_argument("--selection-max-points", type=int, default=256)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["gd", "adam", "lbfgs", "bfgs", "newton"],
        choices=["gd", "adam", "lbfgs", "bfgs", "newton"],
    )
    parser.add_argument("--skip-selection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    device = torch.device(args.device)
    physical = load_physical(args.root)
    baseline_dir = args.root / "baselines"
    figure_dir = baseline_dir / "figures"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    batch_sizes = {
        "gd": args.batch_size,
        "adam": args.batch_size,
        "lbfgs": args.lbfgs_batch_size,
        "bfgs": args.bfgs_batch_size,
        "newton": args.newton_batch_size,
    }

    validation = load_dataset("validation_xn", args.root)
    selection_path = baseline_dir / "parameter_selection.json"
    if args.skip_selection and selection_path.exists():
        selection = load_json(selection_path)
    else:
        selection = select_parameters(
            validation=validation,
            physical=physical,
            steps=args.steps,
            device=device,
            selection_max_points=args.selection_max_points,
            batch_sizes=batch_sizes,
        )
        save_json(selection, selection_path)
        plot_parameter_selection(selection, figure_dir)

    all_metrics: dict[str, Any] = {}
    all_curves: dict[str, Any] = {}
    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name, args.root)
        all_metrics[dataset_name] = {}
        all_curves[dataset_name] = {}
        for solver in args.solvers:
            params = selection[solver]["selected"]
            print(f"evaluating {solver} on {dataset_name} with {params}")
            result = evaluate_solver(
                dataset=dataset,
                physical=physical,
                solver=solver,
                steps=args.steps,
                device=device,
                params=params,
                batch_sizes=batch_sizes,
            )
            summary = result["summary"]
            summary["params"] = params
            all_metrics[dataset_name][solver] = summary
            all_curves[dataset_name][solver] = result["curve"]
            print(
                f"  final p95={summary['final_residual_p95']:.3e}, "
                f"mean={summary['final_residual_mean']:.3e}, "
                f"max={summary['final_residual_max']:.3e}"
            )

    save_json(all_metrics, baseline_dir / "baseline_metrics.json")
    torch.save(all_curves, baseline_dir / "baseline_curves.pt")
    plot_dataset_curves(all_metrics, figure_dir)
    save_json(
        {
            "steps": args.steps,
            "datasets": args.datasets,
            "solvers": args.solvers,
            "batch_sizes": batch_sizes,
            "semantics": (
                "each row is one independent physical time-step problem initialized at x_n; "
                "all solvers run the same number of inner iterations"
            ),
            "lbfgs_semantics": (
                "independent two-loop history per row with independent Armijo line search; "
                "no shared PyTorch LBFGS state across a batch"
            ),
            "bfgs_semantics": (
                "full inverse-Hessian BFGS matrix per row with independent Armijo line search"
            ),
        },
        baseline_dir / "baseline_manifest.json",
    )
    print(f"saved baseline results to {baseline_dir}")


if __name__ == "__main__":
    main()
