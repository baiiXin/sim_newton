"""Small shared I/O and evaluation helpers for the 15x15 project."""
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


def summarize_one_step(r0: np.ndarray, r1: np.ndarray) -> dict[str, Any]:
    r0 = np.asarray(r0, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    finite = np.isfinite(r0) & np.isfinite(r1)
    safe0 = np.maximum(r0[finite], 1e-30)
    safe1 = np.maximum(r1[finite], 1e-30)
    ratio = safe1 / safe0
    log_ratio = np.log10(ratio)

    def stat(values: np.ndarray, kind: str) -> float:
        if values.size == 0:
            return float("inf")
        if kind == "mean":
            return float(np.mean(values))
        if kind == "p50":
            return float(np.percentile(values, 50))
        if kind == "p95":
            return float(np.percentile(values, 95))
        if kind == "p99":
            return float(np.percentile(values, 99))
        if kind == "max":
            return float(np.max(values))
        raise ValueError(kind)

    result: dict[str, Any] = {
        "num_points": int(r0.size),
        "num_finite": int(finite.sum()),
        "num_nonfinite": int((~finite).sum()),
        "improvement_fraction": float(np.mean(r1[finite] < r0[finite])) if finite.any() else 0.0,
    }
    for label, values in (("r0", r0[finite]), ("r1", r1[finite]), ("ratio", ratio), ("log10_ratio", log_ratio)):
        for kind in ("mean", "p50", "p95", "p99", "max"):
            result[f"{label}_{kind}"] = stat(values, kind)
    # Primary checkpoint score: robust tail behavior of relative one-step progress.
    result["selection_metric_name"] = "log10_ratio_p95"
    result["selection_metric"] = result["log10_ratio_p95"]
    return result


@torch.no_grad()
def evaluate_one_step(
    *,
    model: MLPOptimizer,
    dataset: dict[str, Any],
    physical,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    residual0: list[torch.Tensor] = []
    residual1: list[torch.Tensor] = []
    n = int(dataset["initial_y"].shape[0])
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        y = dataset["initial_y"][start:stop].to(device=device, dtype=TORCH_DTYPE)
        q = dataset["q"][start:stop].to(device=device, dtype=TORCH_DTYPE)
        masses = dataset["masses"][start:stop].to(device=device, dtype=TORCH_DTYPE)
        y = project_fixed_vertices(y, physical)
        r0 = stationarity_residual_norm_full(y, q, masses, physical)
        zeros = torch.zeros_like(y)
        y1, _, _ = apply_model_update(
            model,
            y,
            q,
            masses,
            physical,
            previous_residual=zeros,
            previous_update=zeros,
        )
        r1 = stationarity_residual_norm_full(y1, q, masses, physical)
        residual0.append(r0.cpu())
        residual1.append(r1.cpu())
    r0_np = torch.cat(residual0).numpy().astype(float)
    r1_np = torch.cat(residual1).numpy().astype(float)
    return {
        "summary": summarize_one_step(r0_np, r1_np),
        "residual_before": torch.from_numpy(r0_np),
        "residual_after": torch.from_numpy(r1_np),
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
