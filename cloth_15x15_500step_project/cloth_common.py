"""Shared I/O, residual summaries, and learned-model evaluation helpers."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from cloth03_solvers_and_models import (
    MLPOptimizer,
    TORCH_DTYPE,
    apply_model_update,
    physical_config_from_dict,
    project_fixed_vertices,
    stationarity_residual_norm_full,
)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(data), handle, indent=2, ensure_ascii=False, allow_nan=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def load_physical(root: Path):
    runtime = load_json(root / "data" / "reference" / "runtime_config.json")
    return physical_config_from_dict(runtime["physical_config"])


def finite_float(value: float | torch.Tensor) -> float:
    number = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    return number if math.isfinite(number) else float("inf")


def _higher_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("inf")
    try:
        return float(np.percentile(values, q, method="higher"))
    except TypeError:
        return float(np.percentile(values, q, interpolation="higher"))


def summarize_residual_curve(curve: np.ndarray) -> dict[str, Any]:
    residual = np.asarray(curve, dtype=float).copy()
    residual[~np.isfinite(residual)] = np.inf
    if residual.ndim != 2 or residual.shape[1] < 1:
        raise ValueError(f"expected [N,T] residual curve, got {residual.shape}")

    mean_by_iter: list[float] = []
    p50_by_iter: list[float] = []
    p95_by_iter: list[float] = []
    p99_by_iter: list[float] = []
    max_by_iter: list[float] = []
    nonfinite_by_iter: list[int] = []
    for iteration in range(residual.shape[1]):
        values = residual[:, iteration]
        nonfinite_by_iter.append(int((~np.isfinite(values)).sum()))
        mean_by_iter.append(float(np.mean(values)))
        p50_by_iter.append(_higher_percentile(values, 50.0))
        p95_by_iter.append(_higher_percentile(values, 95.0))
        p99_by_iter.append(_higher_percentile(values, 99.0))
        max_by_iter.append(float(np.max(values)))

    initial = residual[:, 0]
    final = residual[:, -1]
    finite_pair = np.isfinite(initial) & np.isfinite(final)
    ratio = np.full_like(final, np.inf)
    ratio[finite_pair] = final[finite_pair] / np.maximum(initial[finite_pair], 1e-30)

    return {
        "num_points": int(residual.shape[0]),
        "num_iterations": int(residual.shape[1] - 1),
        "residual_mean_by_iter": mean_by_iter,
        "residual_p50_by_iter": p50_by_iter,
        "residual_p95_by_iter": p95_by_iter,
        "residual_p99_by_iter": p99_by_iter,
        "residual_max_by_iter": max_by_iter,
        "nonfinite_count_by_iter": nonfinite_by_iter,
        "final_residual_mean": mean_by_iter[-1],
        "final_residual_p50": p50_by_iter[-1],
        "final_residual_p95": p95_by_iter[-1],
        "final_residual_p99": p99_by_iter[-1],
        "final_residual_max": max_by_iter[-1],
        "final_nonfinite_count": nonfinite_by_iter[-1],
        "final_improvement_fraction": (
            float(np.mean(final[finite_pair] < initial[finite_pair])) if finite_pair.any() else 0.0
        ),
        "final_ratio_mean": float(np.mean(ratio)),
        "final_ratio_p95": _higher_percentile(ratio, 95.0),
        "selection_metric_name": "final_residual_p95",
        "selection_metric": p95_by_iter[-1],
    }


@torch.no_grad()
def evaluate_model_iterations(
    *,
    model: MLPOptimizer,
    dataset: dict[str, Any],
    physical,
    steps: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    model.eval()
    curves: list[torch.Tensor] = []
    n = int(dataset["initial_y"].shape[0])
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        y = dataset["initial_y"][start:stop].to(device=device, dtype=TORCH_DTYPE)
        q = dataset["q"][start:stop].to(device=device, dtype=TORCH_DTYPE)
        masses = dataset["masses"][start:stop].to(device=device, dtype=TORCH_DTYPE)
        y = project_fixed_vertices(y, physical)
        previous_residual = torch.zeros_like(y)
        previous_update = torch.zeros_like(y)
        batch_curve: list[torch.Tensor] = []
        for iteration in range(steps + 1):
            batch_curve.append(
                stationarity_residual_norm_full(y, q, masses, physical).detach().cpu()
            )
            if iteration == steps:
                break
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
        curves.append(torch.stack(batch_curve, dim=1))

    curve_tensor = torch.cat(curves, dim=0).contiguous()
    curve_np = curve_tensor.numpy().astype(float)
    curve_np[~np.isfinite(curve_np)] = np.inf
    return {
        "summary": summarize_residual_curve(curve_np),
        "curve": torch.from_numpy(curve_np),
    }


@torch.no_grad()
def evaluate_one_step(
    *,
    model: MLPOptimizer,
    dataset: dict[str, Any],
    physical,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    result = evaluate_model_iterations(
        model=model,
        dataset=dataset,
        physical=physical,
        steps=1,
        device=device,
        batch_size=batch_size,
    )
    curve = result["curve"]
    return {
        "summary": result["summary"],
        "residual_before": curve[:, 0],
        "residual_after": curve[:, 1],
        "curve": curve,
    }


def resolve_exclusions(
    cli_values: Iterable[int],
    exclusion_file: Path | None,
) -> tuple[int, ...]:
    values = {int(v) for v in cli_values}
    if exclusion_file is not None and exclusion_file.exists():
        data = load_json(exclusion_file)
        for key in ("excluded_motion_indices", "motion_indices", "exclude_motion_indices"):
            if key in data:
                values.update(int(v) for v in data[key])
    bad = sorted(v for v in values if v < 0 or v >= 32)
    if bad:
        raise ValueError(f"motion indices must be in 0..31, got {bad}")
    return tuple(sorted(values))
