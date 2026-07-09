"""Script 4: evaluate baseline solvers on named datasets.

Baselines:
    GD      : fixed step size selected on validation set
    Adam    : learning rate selected on validation set
    L-BFGS  : lr/history selected on validation set
    Newton  : analytic Hessian update

Inputs:
    data/datasets/*.pt from cloth02_dataset_catalog.py
    data/reference/runtime_config.json from cloth01_generate_reference_and_samples.py

Outputs:
    baselines/parameter_selection.json
    baselines/baseline_metrics.json
    baselines/baseline_curves.pt
    baselines/figures/*.png

Run:
    python cloth04_evaluate_baselines.py --root cloth_5x5_500step_pipeline --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cloth02_dataset_catalog import load_dataset
from cloth03_solvers_and_models import (
    DEFAULT_DEVICE,
    DEFAULT_EVALUATION_BATCH_SIZE,
    DEFAULT_EVALUATION_STEPS,
    DEFAULT_REPORT_STEPS,
    GD_CANDIDATE_STEP_SIZES,
    AdamState,
    apply_adam_update_full,
    apply_gradient_descent_update_full,
    apply_newton_update_full,
    physical_config_from_dict,
    project_fixed_vertices,
    run_lbfgs_iterations_full,
    stationarity_residual_norm_full,
)

ADAM_CANDIDATE_LRS = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)
LBFGS_CANDIDATES = (
    {"learning_rate": 0.25, "history_size": 10},
    {"learning_rate": 0.50, "history_size": 10},
    {"learning_rate": 1.00, "history_size": 10},
    {"learning_rate": 1.00, "history_size": 20},
)
TEST_DATASETS = ("validation", "seen_extrap", "unseen_id", "ood")


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_physical_config(root: Path):
    with (root / "data" / "reference" / "runtime_config.json").open("r", encoding="utf-8") as f:
        runtime = json.load(f)
    return physical_config_from_dict(runtime["physical_config"])


def dataset_slice(dataset: dict[str, Any], start: int, end: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "initial_y": dataset["initial_y"][start:end].to(device),
        "q": dataset["q"][start:end].to(device),
        "masses": dataset["masses"][start:end].to(device),
        "exact_y": dataset["exact_y"][start:end].to(device),
    }


def subset_dataset(dataset: dict[str, Any], max_points: int | None) -> dict[str, Any]:
    if max_points is None or max_points <= 0 or len(dataset["initial_y"]) <= max_points:
        return dataset
    keys = ["initial_y", "q", "masses", "exact_y", "problem_index", "motion_index", "time_index"]
    subset = {key: dataset[key][:max_points].contiguous() for key in keys}
    subset["metadata"] = dict(dataset.get("metadata", {}))
    subset["metadata"]["subsampled_num_points"] = int(max_points)
    return subset


def summarize_curve(residual_curve: np.ndarray) -> dict[str, Any]:
    return {
        "residual_mean_by_iter": residual_curve.mean(axis=0).tolist(),
        "residual_max_by_iter": residual_curve.max(axis=0).tolist(),
        "residual_sum_by_iter": residual_curve.sum(axis=0).tolist(),
        "final_residual_mean": float(residual_curve[:, -1].mean()),
        "final_residual_max": float(residual_curve[:, -1].max()),
        "final_residual_sum": float(residual_curve[:, -1].sum()),
        "num_points": int(residual_curve.shape[0]),
        "num_iterations": int(residual_curve.shape[1] - 1),
    }


def evaluate_baseline(
    *,
    dataset: dict[str, Any],
    physical,
    solver_name: str,
    steps: int,
    batch_size: int,
    device: torch.device,
    params: dict[str, Any],
) -> dict[str, Any]:
    curves: list[torch.Tensor] = []
    start_time = time.perf_counter()

    for start in range(0, len(dataset["initial_y"]), batch_size):
        end = min(start + batch_size, len(dataset["initial_y"]))
        batch = dataset_slice(dataset, start, end, device)
        y = project_fixed_vertices(batch["initial_y"].clone(), physical)
        batch_curve = []

        if solver_name == "lbfgs":
            states = run_lbfgs_iterations_full(
                y,
                batch["q"],
                batch["masses"],
                physical,
                steps=steps,
                learning_rate=float(params["learning_rate"]),
                history_size=int(params.get("history_size", 10)),
                line_search_fn=params.get("line_search_fn"),
            )
            for state in states:
                residual = stationarity_residual_norm_full(state, batch["q"], batch["masses"], physical)
                batch_curve.append(residual.detach().cpu())
        else:
            adam_state: AdamState | None = None
            with torch.no_grad():
                for step in range(steps + 1):
                    residual = stationarity_residual_norm_full(y, batch["q"], batch["masses"], physical)
                    batch_curve.append(residual.detach().cpu())
                    if step == steps:
                        break
                    if solver_name == "gd":
                        y, _ = apply_gradient_descent_update_full(
                            y, batch["q"], batch["masses"], physical, float(params["step_size"])
                        )
                    elif solver_name == "adam":
                        y, _, adam_state = apply_adam_update_full(
                            y,
                            batch["q"],
                            batch["masses"],
                            physical,
                            adam_state,
                            learning_rate=float(params["learning_rate"]),
                        )
                    elif solver_name == "newton":
                        y, _ = apply_newton_update_full(y, batch["q"], batch["masses"], physical)
                    else:
                        raise ValueError(f"unknown solver: {solver_name}")
        curves.append(torch.stack(batch_curve, dim=1))

    residual_curve = torch.cat(curves, dim=0).numpy().astype(float)
    residual_curve[~np.isfinite(residual_curve)] = np.inf
    summary = summarize_curve(residual_curve)
    summary.update(
        {
            "solver": solver_name,
            "params": params,
            "elapsed_seconds": time.perf_counter() - start_time,
            "seconds_per_point_per_iteration": (time.perf_counter() - start_time)
            / max(1, len(dataset["initial_y"]) * steps),
        }
    )
    return {"summary": summary, "curve": residual_curve}


def select_parameters(
    *,
    validation: dict[str, Any],
    physical,
    steps: int,
    batch_size: int,
    device: torch.device,
    selection_max_points: int,
) -> dict[str, Any]:
    validation = subset_dataset(validation, selection_max_points)
    results: dict[str, Any] = {}

    gd_trials = []
    for step_size in GD_CANDIDATE_STEP_SIZES:
        result = evaluate_baseline(
            dataset=validation,
            physical=physical,
            solver_name="gd",
            steps=steps,
            batch_size=batch_size,
            device=device,
            params={"step_size": float(step_size)},
        )["summary"]
        gd_trials.append(result)
    results["gd"] = {
        "selected": min(gd_trials, key=lambda x: x["final_residual_mean"])["params"],
        "trials": gd_trials,
    }

    adam_trials = []
    for lr in ADAM_CANDIDATE_LRS:
        result = evaluate_baseline(
            dataset=validation,
            physical=physical,
            solver_name="adam",
            steps=steps,
            batch_size=batch_size,
            device=device,
            params={"learning_rate": float(lr)},
        )["summary"]
        adam_trials.append(result)
    results["adam"] = {
        "selected": min(adam_trials, key=lambda x: x["final_residual_mean"])["params"],
        "trials": adam_trials,
    }

    lbfgs_trials = []
    for params in LBFGS_CANDIDATES:
        result = evaluate_baseline(
            dataset=validation,
            physical=physical,
            solver_name="lbfgs",
            steps=steps,
            batch_size=batch_size,
            device=device,
            params=params,
        )["summary"]
        lbfgs_trials.append(result)
    results["lbfgs"] = {
        "selected": min(lbfgs_trials, key=lambda x: x["final_residual_mean"])["params"],
        "trials": lbfgs_trials,
    }

    results["newton"] = {"selected": {}, "trials": []}
    return results


def plot_parameter_selection(selection: dict[str, Any], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for solver_name in ["gd", "adam", "lbfgs"]:
        trials = selection[solver_name]["trials"]
        labels = []
        values = []
        for trial in trials:
            params = trial["params"]
            if solver_name == "gd":
                labels.append(f"{params['step_size']:.0e}")
            elif solver_name == "adam":
                labels.append(f"{params['learning_rate']:.0e}")
            else:
                labels.append(f"lr={params['learning_rate']},h={params['history_size']}")
            values.append(trial["final_residual_mean"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(len(values)), values, marker="o")
        ax.set_yscale("log")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("candidate")
        ax.set_ylabel("validation final residual mean")
        ax.set_title(f"{solver_name} parameter selection")
        fig.tight_layout()
        fig.savefig(figure_dir / f"{solver_name}_parameter_selection.png", dpi=180)
        plt.close(fig)


def plot_dataset_curves(metrics: dict[str, Any], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name, dataset_metrics in metrics.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for solver_name, record in dataset_metrics.items():
            y = np.asarray(record["residual_mean_by_iter"], dtype=float)
            ax.plot(np.arange(len(y)), np.maximum(y, 1e-16), label=solver_name)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean residual")
        ax.set_title(f"{dataset_name}: iteration vs residual")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{dataset_name}_iteration_vs_residual_mean.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for solver_name, record in dataset_metrics.items():
            y = np.asarray(record["residual_max_by_iter"], dtype=float)
            ax.plot(np.arange(len(y)), np.maximum(y, 1e-16), label=solver_name)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("max residual")
        ax.set_title(f"{dataset_name}: iteration vs max residual")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"{dataset_name}_iteration_vs_residual_max.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GD/Adam/L-BFGS/Newton baselines.")
    parser.add_argument("--root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    parser.add_argument("--lbfgs-batch-size", type=int, default=512)
    parser.add_argument("--selection-max-points", type=int, default=8192)
    parser.add_argument("--datasets", nargs="+", default=list(TEST_DATASETS))
    parser.add_argument("--skip-selection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    physical = load_physical_config(args.root)
    baseline_dir = args.root / "baselines"
    figure_dir = baseline_dir / "figures"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    validation = load_dataset("validation", args.root)
    if args.skip_selection and (baseline_dir / "parameter_selection.json").exists():
        with (baseline_dir / "parameter_selection.json").open("r", encoding="utf-8") as f:
            selection = json.load(f)
    else:
        selection = select_parameters(
            validation=validation,
            physical=physical,
            steps=args.steps,
            batch_size=args.batch_size,
            device=device,
            selection_max_points=args.selection_max_points,
        )
        save_json(selection, baseline_dir / "parameter_selection.json")
        plot_parameter_selection(selection, figure_dir)

    all_metrics: dict[str, Any] = {}
    all_curves: dict[str, Any] = {}
    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name, args.root)
        all_metrics[dataset_name] = {}
        all_curves[dataset_name] = {}
        for solver_name in ["gd", "adam", "lbfgs", "newton"]:
            params = selection[solver_name]["selected"]
            batch_size = args.lbfgs_batch_size if solver_name == "lbfgs" else args.batch_size
            print(f"evaluating {solver_name} on {dataset_name} with {params}")
            result = evaluate_baseline(
                dataset=dataset,
                physical=physical,
                solver_name=solver_name,
                steps=args.steps,
                batch_size=batch_size,
                device=device,
                params=params,
            )
            all_metrics[dataset_name][solver_name] = result["summary"]
            all_curves[dataset_name][solver_name] = torch.from_numpy(result["curve"])
            print(
                f"  final mean={result['summary']['final_residual_mean']:.3e}, "
                f"max={result['summary']['final_residual_max']:.3e}"
            )

    save_json(all_metrics, baseline_dir / "baseline_metrics.json")
    torch.save(all_curves, baseline_dir / "baseline_curves.pt")
    plot_dataset_curves(all_metrics, figure_dir)
    print(f"saved baseline results to {baseline_dir}")


if __name__ == "__main__":
    main()
