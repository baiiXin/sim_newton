"""Evaluate 500-step full-dataset and pool-trained 5x5 cloth models on test datasets.

This script uses the original dataset-style evaluation from the nonlinear
history-input ablation: run a fixed number of independent optimizer iterations
from sampled initial states, then report pooled, per-motion, per-problem, and
worst-motion metrics.  The implementation is adapted to the newer 75D full-state
models used by cloth_5x5_500step_project.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "cloth_5x5_500step_project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from cloth03_solvers_and_models import (  # noqa: E402
    ACTIVATION_NAMES,
    DEFAULT_DEVICE,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    FIXED_VERTEX_INDICES,
    MLPOptimizer,
    ModelSpec,
    TORCH_DTYPE,
    apply_model_update,
    physical_config_from_dict,
    project_fixed_vertices,
    reshape_full,
    spring_lengths_from_full,
    stationarity_residual_norm_full,
    variational_energy_full,
)


DEFAULT_DATASETS = (
    "validation",
    "seen_extrap",
    "unseen_id",
    "ood",
)
DEFAULT_REPORT_STEPS = (0, 1, 3, 5, 10, 30, 50)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict[str, Any], path: Path) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(v) for v in value]
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, torch.Tensor):
            return safe(value.detach().cpu().tolist())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe(data), handle, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_physical(source_root: Path):
    runtime = load_json(source_root / "data" / "reference" / "runtime_config.json")
    return physical_config_from_dict(runtime["physical_config"])


def load_dataset(source_root: Path, name: str) -> dict[str, Any]:
    path = source_root / "data" / "datasets" / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def resolve_full_model_root(source_root: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    direct = source_root / "models"
    old = direct / "old"
    has_direct_models = direct.exists() and any((direct / name).is_dir() for name in direct.iterdir())
    if not has_direct_models and old.exists():
        return old
    expected = direct / "activation_identity_depth_01_width_256_no_bias"
    if not expected.exists() and old.exists():
        return old
    return direct


def make_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs = [
        ModelSpec(activation=str(a), depth=int(d), width=int(w), use_bias=bool(args.use_bias))
        for a in args.activations
        for d in args.depths
        for w in args.widths
    ]
    return [specs[int(args.config_index)]] if args.config_index is not None else specs


def load_model_from_dir(
    model_dir: Path,
    device: torch.device,
    residual_length_scale: float,
) -> tuple[MLPOptimizer, dict[str, Any]]:
    checkpoint_path = model_dir / "best_validation_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    spec_data = checkpoint.get("model_spec")
    if spec_data is None:
        raise KeyError(f"{checkpoint_path} has no model_spec")
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
        "update_count": int(checkpoint.get("update_count", -1)),
        "model_spec": asdict(model_spec),
        "residual_length_scale": scale,
    }


def statistics(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    result = {
        "num_nonfinite": int(values.size - finite.size),
        "mean": float("nan"),
        "median": float("nan"),
        "p95": float("nan"),
        "max": float("nan"),
    }
    if finite.size:
        result.update(
            mean=float(np.mean(finite)),
            median=float(np.median(finite)),
            p95=float(np.percentile(finite, 95)),
            max=float(np.max(finite)),
        )
    return result


def selected_steps(steps: int, report_steps: Sequence[int]) -> list[int]:
    return sorted(set([0, steps, *[int(s) for s in report_steps if 0 <= int(s) <= steps]]))


def state_metrics(
    y: torch.Tensor,
    batch: dict[str, torch.Tensor],
    exact_energy: torch.Tensor,
    physical,
) -> dict[str, torch.Tensor]:
    y = project_fixed_vertices(y, physical)
    exact_y = project_fixed_vertices(batch["exact_y"], physical)
    point_errors = torch.linalg.vector_norm(
        reshape_full(y) - reshape_full(exact_y), dim=-1
    )
    fixed_errors = point_errors[:, list(FIXED_VERTEX_INDICES)]
    energy = variational_energy_full(y, batch["q"], batch["masses"], physical)
    return {
        "residual": stationarity_residual_norm_full(y, batch["q"], batch["masses"], physical),
        "energy_gap": energy - exact_energy,
        "exact_error": torch.linalg.vector_norm(y - exact_y, dim=-1),
        "particle_mean_error": point_errors.mean(dim=-1),
        "particle_max_error": point_errors.max(dim=-1).values,
        "spring_length_error": torch.mean(
            torch.abs(
                spring_lengths_from_full(y, physical)
                - spring_lengths_from_full(exact_y, physical)
            ),
            dim=-1,
        ),
        "fixed_vertex_max_error": fixed_errors.max(dim=-1).values,
    }


@torch.no_grad()
def evaluate_model_on_dataset(
    *,
    model: MLPOptimizer,
    model_info: dict[str, Any],
    solver_name: str,
    dataset_name: str,
    dataset_cpu: dict[str, Any],
    physical,
    steps: int,
    batch_size: int,
    report_steps: Sequence[int],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model.eval()
    metric_batches: dict[str, list[torch.Tensor]] = {}
    problem_batches: list[torch.Tensor] = []
    motion_batches: list[torch.Tensor] = []
    time_batches: list[torch.Tensor] = []

    n = int(dataset_cpu["initial_y"].shape[0])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = {
            "initial_y": dataset_cpu["initial_y"][start:end].to(device=device, dtype=TORCH_DTYPE),
            "q": dataset_cpu["q"][start:end].to(device=device, dtype=TORCH_DTYPE),
            "masses": dataset_cpu["masses"][start:end].to(device=device, dtype=TORCH_DTYPE),
            "exact_y": dataset_cpu["exact_y"][start:end].to(device=device, dtype=TORCH_DTYPE),
            "problem_index": dataset_cpu["problem_index"][start:end],
            "motion_index": dataset_cpu["motion_index"][start:end],
            "time_index": dataset_cpu["time_index"][start:end],
        }
        y = project_fixed_vertices(batch["initial_y"].clone(), physical)
        previous_residual = torch.zeros_like(y)
        previous_update = torch.zeros_like(y)
        exact_energy = variational_energy_full(batch["exact_y"], batch["q"], batch["masses"], physical)
        step_values: dict[str, list[torch.Tensor]] = {}

        for step in range(steps + 1):
            for name, values in state_metrics(y, batch, exact_energy, physical).items():
                step_values.setdefault(name, []).append(values.detach().cpu())
            if step == steps:
                break
            y, delta, current_residual = apply_model_update(
                model,
                y,
                batch["q"],
                batch["masses"],
                physical,
                previous_residual=previous_residual,
                previous_update=previous_update,
            )
            previous_residual = current_residual.detach()
            previous_update = delta.detach()

        for name, values in step_values.items():
            metric_batches.setdefault(name, []).append(torch.stack(values, dim=1))
        problem_batches.append(batch["problem_index"].detach().cpu())
        motion_batches.append(batch["motion_index"].detach().cpu())
        time_batches.append(batch["time_index"].detach().cpu())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    arrays = {
        name: torch.cat(values, dim=0).numpy().astype(float)
        for name, values in metric_batches.items()
    }
    for values in arrays.values():
        values[~np.isfinite(values)] = np.nan
    problem_indices = torch.cat(problem_batches).numpy().astype(int)
    motion_indices = torch.cat(motion_batches).numpy().astype(int)
    time_indices = torch.cat(time_batches).numpy().astype(int)

    result: dict[str, Any] = {
        "solver": solver_name,
        "dataset": dataset_name,
        "model_info": model_info,
        "steps": int(steps),
        "num_points": int(n),
        "num_motions": int(np.unique(motion_indices).size),
        "selected_report_steps": selected_steps(steps, report_steps),
        "elapsed_seconds": elapsed,
        "seconds_per_point_per_iteration": elapsed / max(n * steps, 1),
    }

    for name, values in arrays.items():
        for stat_name in ["mean", "median", "p95", "max", "num_nonfinite"]:
            result[f"{name}_{stat_name}_by_step"] = []
        for step in range(values.shape[1]):
            stats = statistics(values[:, step])
            for stat_name, value in stats.items():
                result[f"{name}_{stat_name}_by_step"].append(value)
        for stat_name, value in statistics(values[:, -1]).items():
            result[f"final_{name}_{stat_name}"] = value

    selected = result["selected_report_steps"]
    per_motion: dict[str, Any] = {}
    for motion_index in sorted(np.unique(motion_indices).tolist()):
        mask = motion_indices == motion_index
        record: dict[str, Any] = {
            "motion_index": int(motion_index),
            "num_points": int(mask.sum()),
            "time_indices": sorted(np.unique(time_indices[mask]).astype(int).tolist()),
            "steps": {},
            "final": {},
        }
        for step in selected:
            record["steps"][str(step)] = {
                name: statistics(values[mask, step]) for name, values in arrays.items()
            }
        record["final"] = {
            name: statistics(values[mask, -1]) for name, values in arrays.items()
        }
        per_motion[str(motion_index)] = record
    result["per_motion"] = per_motion

    per_problem: dict[str, Any] = {}
    for problem_index in sorted(np.unique(problem_indices).tolist()):
        mask = problem_indices == problem_index
        per_problem[str(problem_index)] = {
            "problem_index": int(problem_index),
            "motion_index": int(motion_indices[mask][0]),
            "time_index": int(time_indices[mask][0]),
            "num_points": int(mask.sum()),
            "final": {name: statistics(values[mask, -1]) for name, values in arrays.items()},
        }
    result["per_problem"] = per_problem

    worst_motion: dict[str, Any] = {}
    for metric_name in arrays:
        records = []
        for motion_key, record in per_motion.items():
            records.append((int(motion_key), record["final"][metric_name]))
        finite_p95 = [(m, float(s["p95"])) for m, s in records if math.isfinite(float(s["p95"]))]
        finite_max = [(m, float(s["max"])) for m, s in records if math.isfinite(float(s["max"]))]
        p95_motion, p95_value = max(finite_p95, key=lambda item: item[1]) if finite_p95 else (-1, float("nan"))
        max_motion, max_value = max(finite_max, key=lambda item: item[1]) if finite_max else (-1, float("nan"))
        worst_motion[metric_name] = {
            "p95_motion_index": p95_motion,
            "p95": p95_value,
            "max_motion_index": max_motion,
            "max": max_value,
        }
        result[f"worst_motion_final_{metric_name}_p95"] = p95_value
        result[f"worst_motion_final_{metric_name}_p95_motion_index"] = p95_motion
        result[f"worst_motion_final_{metric_name}_max"] = max_value
        result[f"worst_motion_final_{metric_name}_max_motion_index"] = max_motion
    result["worst_motion"] = worst_motion
    return result, arrays


def summary_row(group: str, spec: ModelSpec, dataset_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "group": group,
        "activation": spec.activation,
        "depth": spec.depth,
        "width": spec.width,
        "use_bias": spec.use_bias,
        "dataset": dataset_name,
        "num_points": metrics.get("num_points"),
        "num_motions": metrics.get("num_motions"),
        "steps": metrics.get("steps"),
        "final_residual_mean": metrics.get("final_residual_mean"),
        "final_residual_p95": metrics.get("final_residual_p95"),
        "final_residual_max": metrics.get("final_residual_max"),
        "final_residual_num_nonfinite": metrics.get("final_residual_num_nonfinite"),
        "worst_motion_final_residual_p95": metrics.get("worst_motion_final_residual_p95"),
        "worst_motion_final_residual_p95_motion_index": metrics.get("worst_motion_final_residual_p95_motion_index"),
        "worst_motion_final_residual_max": metrics.get("worst_motion_final_residual_max"),
        "worst_motion_final_residual_max_motion_index": metrics.get("worst_motion_final_residual_max_motion_index"),
        "final_exact_error_p95": metrics.get("final_exact_error_p95"),
        "final_exact_error_max": metrics.get("final_exact_error_max"),
        "final_energy_gap_p95": metrics.get("final_energy_gap_p95"),
        "final_energy_gap_max": metrics.get("final_energy_gap_max"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "checkpoint": (metrics.get("model_info") or {}).get("checkpoint"),
    }


def plot_residual_curves(records: list[dict[str, Any]], output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(str(record["dataset"]), []).append(record)
    for dataset_name, selected in by_dataset.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        for record in selected:
            curve = np.asarray(record["residual_mean_by_step"], dtype=float)
            label = f"{record['group']}_{record['activation']}"
            ax.plot(np.arange(curve.size), np.maximum(curve, 1e-16), label=label)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("mean residual")
        ax.set_title(f"{dataset_name}: independent test-set solve")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{dataset_name}_residual_mean_by_iteration.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate full-dataset and Metamizer-pool 75D models on original dataset-style tests."
    )
    parser.add_argument("--source-root", type=Path, default=PROJECT_DIR / "cloth_5x5_500step_pipeline")
    parser.add_argument("--pool-root", type=Path, default=PROJECT_DIR / "cloth_5x5_metamizer_pool_training")
    parser.add_argument("--full-model-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "unit_test_for_spring" / "multi_motion_cloth" / "training_pool")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--groups", nargs="+", default=["full_500step", "pool"], choices=["full_500step", "pool"])
    parser.add_argument("--activations", nargs="+", default=list(ACTIVATION_NAMES))
    parser.add_argument("--depths", type=int, nargs="+", default=[1])
    parser.add_argument("--widths", type=int, nargs="+", default=[256])
    parser.add_argument("--use-bias", action="store_true")
    parser.add_argument("--config-index", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--report-steps", type=int, nargs="+", default=list(DEFAULT_REPORT_STEPS))
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-curves", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    source_root = args.source_root.resolve()
    pool_root = args.pool_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    full_model_root = resolve_full_model_root(source_root, args.full_model_root)
    physical = load_physical(source_root)
    specs = make_model_specs(args)
    datasets = {name: load_dataset(source_root, name) for name in args.datasets}

    save_json(
        {
            "source_root": str(source_root),
            "pool_root": str(pool_root),
            "full_model_root": str(full_model_root),
            "output_root": str(output_root),
            "datasets": list(args.datasets),
            "groups": list(args.groups),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "report_steps": list(args.report_steps),
            "model_specs": [asdict(spec) for spec in specs],
            "dataset_source_confirmation": (
                "Reuses cloth_5x5_500step_pipeline/data/datasets, matching README split: "
                "default test datasets are validation, seen_extrap, unseen_id, and ood."
            ),
        },
        output_root / "run_config.json",
    )

    rows: list[dict[str, Any]] = []
    plot_records: list[dict[str, Any]] = []
    all_metrics: dict[str, Any] = {}

    group_roots = {
        "full_500step": full_model_root,
        "pool": pool_root / "models",
    }
    for group in args.groups:
        for spec in specs:
            model_dir = group_roots[group] / spec.experiment_name
            try:
                model, model_info = load_model_from_dir(model_dir, device, args.residual_length_scale)
            except FileNotFoundError as exc:
                print(f"skip {group} {spec.experiment_name}: {exc}")
                continue

            solver_name = f"{group}_{spec.activation}_d{spec.depth:02d}_w{spec.width:03d}"
            for dataset_name, dataset in datasets.items():
                output_dir = output_root / group / spec.experiment_name / dataset_name
                metrics_path = output_dir / "metrics.json"
                curves_path = output_dir / "curves.pt"
                if metrics_path.exists() and (not args.overwrite or args.plot_only):
                    metrics = load_json(metrics_path)
                    print(f"reuse {solver_name} on {dataset_name}")
                    rows.append(summary_row(group, spec, dataset_name, metrics))
                    plot_records.append(
                        {
                            "group": group,
                            "activation": spec.activation,
                            "dataset": dataset_name,
                            "residual_mean_by_step": metrics["residual_mean_by_step"],
                        }
                    )
                    all_metrics[f"{group}/{spec.experiment_name}/{dataset_name}"] = metrics
                    continue
                if args.plot_only:
                    print(f"skip missing metrics for {solver_name} on {dataset_name}: {metrics_path}")
                    continue

                print(f"evaluating {solver_name} on {dataset_name} ({len(dataset['initial_y'])} points)")
                metrics, curves = evaluate_model_on_dataset(
                    model=model,
                    model_info=model_info,
                    solver_name=solver_name,
                    dataset_name=dataset_name,
                    dataset_cpu=dataset,
                    physical=physical,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    report_steps=args.report_steps,
                    device=device,
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                save_json(metrics, metrics_path)
                if not args.skip_curves:
                    torch.save(
                        {
                            "solver": solver_name,
                            "dataset": dataset_name,
                            "metrics": {name: torch.from_numpy(values) for name, values in curves.items()},
                        },
                        curves_path,
                    )
                rows.append(summary_row(group, spec, dataset_name, metrics))
                plot_records.append(
                    {
                        "group": group,
                        "activation": spec.activation,
                        "dataset": dataset_name,
                        "residual_mean_by_step": metrics["residual_mean_by_step"],
                    }
                )
                all_metrics[f"{group}/{spec.experiment_name}/{dataset_name}"] = metrics

    write_csv(rows, output_root / "summary_metrics.csv")
    save_json({"records": all_metrics}, output_root / "all_metrics.json")
    plot_residual_curves(plot_records, output_root)
    print(f"wrote summary to {output_root / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
