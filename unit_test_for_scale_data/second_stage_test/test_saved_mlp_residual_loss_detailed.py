r"""
用已保存的 MLP optimizer 参数批量测试 held-out 测试集 residual / energy loss，
并额外绘制训练集与测试集的 set distribution，以及 Adam lr=1e-4 的详细收敛轨迹。

适配训练脚本输出结构：
    <results_dir>/dataset_scale_ablation_summary.json
    <results_dir>/<experiment_name>/optimization_report.json
    <results_dir>/<experiment_name>/mlp_optimizer_state_dict.pt

核心设计：
1. 训练集只用于分布可视化，不评估训练集 residual/loss。
2. 测试集是 y_star 附近的固定随机点，并显式剔除所有训练规则网格点。
3. 对测试集评估：
       residual: r_k(y0) = ||grad E(y^(k)(y0))||_2
       loss gap: l_k(y0) = E(y^(k)(y0)) - E(y*)
   统计 mean / median / p95 / max。
4. 对原始物理初值 p_n 单独评估 residual 和 loss gap。
5. 从测试集中选 3 个点，对 Adam lr=1e-4 的 N=8 与 N≈10000(实际通常为 10648)
   两个模型单独画：
       final_reference_residual_comparison.png
       final_reference_energy_contour_2d.png
       final_reference_trajectory_3d.png

典型用法：
    python test_saved_mlp_residual_loss_detailed.py --results-dir "Pasted code(15)" --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PLOT_FLOOR = 1e-14
GRID_MATCH_TOL = 1e-10


# ============================================================
# 1. 网络、物理能量、残差
# ============================================================


class MLPOptimizer(nn.Module):
    """训练脚本中的 12 -> 32 -> 32 -> 3 学习型迭代优化器。"""

    def __init__(
        self,
        *,
        dtype: torch.dtype,
        use_input_normalization: bool = True,
        use_output_dt_scaling: bool = True,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.use_input_normalization = bool(use_input_normalization)
        self.use_output_dt_scaling = bool(use_output_dt_scaling)

        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        if input_mean is None:
            input_mean = torch.zeros(12, dtype=dtype)
        if input_std is None:
            input_std = torch.ones(12, dtype=dtype)

        self.register_buffer("input_mean", input_mean.clone().detach().to(dtype=dtype))
        self.register_buffer("input_std", input_std.clone().detach().to(dtype=dtype))
        self.to(dtype=dtype)

    @staticmethod
    def _expand_feature_for_batch(feature: torch.Tensor, batch_size: int) -> torch.Tensor:
        if feature.ndim == 1:
            return feature.unsqueeze(0).expand(batch_size, -1)
        if feature.ndim == 2 and feature.shape[0] == batch_size:
            return feature
        raise ValueError(
            f"Feature shape is incompatible: feature={tuple(feature.shape)}, batch={batch_size}"
        )

    def forward(
        self,
        y: torch.Tensor,
        history: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        if y.ndim == 1:
            inp = torch.cat([y, history, params], dim=-1)
            if self.use_input_normalization:
                inp = (inp - self.input_mean) / self.input_std
            delta = self.net(inp)
            if self.use_output_dt_scaling:
                delta = params[2] * delta
            return delta

        if y.ndim != 2 or y.shape[-1] != 3:
            raise ValueError(f"Expected y shape [3] or [B, 3], got {tuple(y.shape)}")

        batch_size = y.shape[0]
        history_batch = self._expand_feature_for_batch(history, batch_size)
        params_batch = self._expand_feature_for_batch(params, batch_size)
        inp = torch.cat([y, history_batch, params_batch], dim=-1)

        if self.use_input_normalization:
            inp = (inp - self.input_mean) / self.input_std

        delta = self.net(inp)
        if self.use_output_dt_scaling:
            delta = params_batch[:, 2:3] * delta
        return delta


def variational_energy(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> torch.Tensor:
    residual = y - p_n - dt * v_n
    kinetic_term = (m / (2.0 * dt**2)) * torch.sum(residual**2, dim=-1)
    potential_term = m * g * y[..., 2]
    return kinetic_term + potential_term


def stationarity_residual(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> torch.Tensor:
    residual = (m / dt**2) * (y - p_n - dt * v_n)
    gravity = torch.zeros_like(residual)
    gravity[..., 2] = m * g
    return residual + gravity


def stationarity_residual_norm(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> torch.Tensor:
    return torch.linalg.vector_norm(
        stationarity_residual(y, p_n, v_n, m, g, dt),
        dim=-1,
    )


def newton_direction(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> torch.Tensor:
    grad = stationarity_residual(y, p_n, v_n, m, g, dt)
    return -(dt**2 / m) * grad


# ============================================================
# 2. 文件与元数据
# ============================================================


@dataclass(frozen=True)
class ExperimentFile:
    experiment_dir: Path
    report_path: Path
    model_path: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class RegularGridSpec:
    target_dataset_size: int
    actual_dataset_size: int
    points_per_axis: int
    sampling_radius: float
    axis_spacing: float


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_plot_value(value: float | int | None) -> float:
    if value is None:
        return float("nan")
    value = float(value)
    if not math.isfinite(value):
        return float("nan")
    return max(value, PLOT_FLOOR)


def safe_torch_load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"{path} does not contain a state_dict-like object.")
    return state


def infer_state_dtype(state_dict: dict[str, torch.Tensor]) -> torch.dtype:
    for value in state_dict.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.dtype
    return torch.float64


def discover_results_dir(results_dir_arg: str | None) -> Path:
    if results_dir_arg is not None:
        path = Path(results_dir_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"results_dir does not exist: {path}")
        return path

    candidates: list[Path] = []
    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    seen_roots: set[Path] = set()

    for root in search_roots:
        root = root.resolve()
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        candidates.extend(root.glob("dataset_scale_ablation_summary.json"))
        candidates.extend(root.glob("*/dataset_scale_ablation_summary.json"))
        candidates.extend(root.glob("*/*/dataset_scale_ablation_summary.json"))

    if not candidates:
        raise FileNotFoundError(
            "Cannot find dataset_scale_ablation_summary.json automatically. "
            "Please pass --results-dir <training_output_dir>."
        )

    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].parent.resolve()


def find_experiment_files(results_dir: Path) -> list[ExperimentFile]:
    experiment_files: list[ExperimentFile] = []
    for report_path in sorted(results_dir.glob("*/optimization_report.json")):
        experiment_dir = report_path.parent
        model_path = experiment_dir / "mlp_optimizer_state_dict.pt"
        if not model_path.exists():
            print(f"[skip] missing model state_dict: {model_path}")
            continue
        try:
            report = load_json(report_path)
        except Exception as exc:
            print(f"[skip] cannot read report {report_path}: {exc}")
            continue
        experiment_files.append(
            ExperimentFile(
                experiment_dir=experiment_dir,
                report_path=report_path,
                model_path=model_path,
                report=report,
            )
        )

    if not experiment_files:
        raise FileNotFoundError(
            f"No experiment subdirectories found under {results_dir}. "
            "Expected */optimization_report.json and */mlp_optimizer_state_dict.pt."
        )
    return experiment_files


def extract_unique_grid_specs(experiment_files: Sequence[ExperimentFile]) -> list[RegularGridSpec]:
    specs: dict[tuple[int, int], RegularGridSpec] = {}
    for exp in experiment_files:
        cfg = exp.report.get("config", {})
        points_per_axis = int(cfg.get("points_per_axis", -1))
        actual_dataset_size = int(cfg.get("actual_dataset_size", -1))
        target_dataset_size = int(cfg.get("target_dataset_size", actual_dataset_size))
        radius = float(cfg.get("sampling_radius_per_axis", cfg.get("sampling_radius", 0.01)))
        axis_spacing = float(cfg.get("axis_spacing", (2.0 * radius) / max(points_per_axis - 1, 1)))
        if points_per_axis <= 0 or actual_dataset_size <= 0:
            continue
        specs[(points_per_axis, actual_dataset_size)] = RegularGridSpec(
            target_dataset_size=target_dataset_size,
            actual_dataset_size=actual_dataset_size,
            points_per_axis=points_per_axis,
            sampling_radius=radius,
            axis_spacing=axis_spacing,
        )
    return sorted(specs.values(), key=lambda s: s.actual_dataset_size)


def parse_dataset_size_from_name(name: str) -> int | None:
    match = re.search(r"num_samples_(\d+)", name)
    return int(match.group(1)) if match else None


def instantiate_model_from_report_and_state(
    *,
    experiment: ExperimentFile,
    device: torch.device,
) -> tuple[MLPOptimizer, torch.dtype, dict[str, torch.Tensor]]:
    state_dict = safe_torch_load_state_dict(experiment.model_path)
    dtype = infer_state_dtype(state_dict)
    cfg = experiment.report.get("config", {})

    input_mean = torch.tensor(cfg.get("input_mean", [0.0] * 12), dtype=dtype)
    input_std = torch.tensor(cfg.get("input_std", [1.0] * 12), dtype=dtype)

    model = MLPOptimizer(
        dtype=dtype,
        use_input_normalization=cfg.get("use_input_normalization", True),
        use_output_dt_scaling=cfg.get("use_output_dt_scaling", True),
        input_mean=input_mean,
        input_std=input_std,
    ).to(device=device, dtype=dtype)

    model.load_state_dict(state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()

    p_n = torch.tensor(cfg.get("p_n", [3.0, 4.0, 5.0]), dtype=dtype)
    v_n = torch.tensor(cfg.get("v_n", [0.5, -0.5, 0.0]), dtype=dtype)
    return model, dtype, {"p_n": p_n, "v_n": v_n}


# ============================================================
# 3. 训练/测试集分布与 held-out 测试集构造
# ============================================================


def sample_regular_grid_points_for_plot(
    *,
    grid_spec: RegularGridSpec,
    y_star: np.ndarray,
    max_points: int,
) -> np.ndarray:
    n = int(grid_spec.points_per_axis)
    total = int(grid_spec.actual_dataset_size)
    radius = float(grid_spec.sampling_radius)
    spacing = float(grid_spec.axis_spacing)
    lower = y_star - radius

    if total <= max_points:
        flat_indices = np.arange(total, dtype=np.int64)
    else:
        flat_indices = np.linspace(0, total - 1, max_points).round().astype(np.int64)

    n2 = n * n
    ix = flat_indices // n2
    rem = flat_indices % n2
    iy = rem // n
    iz = rem % n

    points = np.empty((flat_indices.shape[0], 3), dtype=float)
    points[:, 0] = lower[0] + ix * spacing
    points[:, 1] = lower[1] + iy * spacing
    points[:, 2] = lower[2] + iz * spacing
    return points


def points_in_grid_mask(
    points: torch.Tensor,
    *,
    grid_spec: RegularGridSpec,
    y_star: torch.Tensor,
    tol: float = GRID_MATCH_TOL,
) -> torch.Tensor:
    n = int(grid_spec.points_per_axis)
    radius = float(grid_spec.sampling_radius)
    spacing = float(grid_spec.axis_spacing)

    lower = y_star.to(dtype=points.dtype).unsqueeze(0) - radius
    coords = (points - lower) / spacing
    rounded = torch.round(coords)
    close_to_integer = torch.abs(coords - rounded) <= tol
    in_range = (rounded >= 0) & (rounded <= (n - 1))
    return torch.all(close_to_integer & in_range, dim=1)


def build_fixed_heldout_test_set_excluding_training_grids(
    *,
    y_star: torch.Tensor,
    radius: float,
    num_random: int,
    seed: int,
    radius_scale: float,
    dtype: torch.dtype,
    grid_specs: Sequence[RegularGridSpec],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """构造 held-out 测试集，并显式剔除所有训练规则网格点。"""

    if num_random <= 0:
        raise ValueError("num_random must be positive.")
    if radius <= 0.0:
        raise ValueError("sampling radius must be positive.")
    if radius_scale <= 0.0:
        raise ValueError("radius_scale must be positive.")

    effective_radius = float(radius) * float(radius_scale)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    y_star_cpu = y_star.detach().cpu().to(dtype=dtype)

    collected_chunks: list[torch.Tensor] = []
    num_collected = 0
    num_candidates_generated = 0
    num_rejected_overlap = 0
    batch_candidates = max(2048, min(65536, num_random * 2))
    max_rounds = 10000

    for _ in range(max_rounds):
        if num_collected >= num_random:
            break

        remaining = num_random - num_collected
        current_batch = max(batch_candidates, remaining * 2)
        offsets = (
            2.0 * torch.rand((current_batch, 3), generator=generator, dtype=dtype) - 1.0
        ) * effective_radius
        candidates = y_star_cpu.unsqueeze(0) + offsets
        keep_mask = torch.ones(current_batch, dtype=torch.bool)

        for grid_spec in grid_specs:
            keep_mask &= ~points_in_grid_mask(
                candidates,
                grid_spec=grid_spec,
                y_star=y_star_cpu,
            )

        kept = candidates[keep_mask]
        num_candidates_generated += int(current_batch)
        num_rejected_overlap += int((~keep_mask).sum().item())

        if kept.shape[0] > remaining:
            kept = kept[:remaining]
        if kept.numel() > 0:
            collected_chunks.append(kept)
            num_collected += int(kept.shape[0])

    if num_collected < num_random:
        raise RuntimeError(
            "Failed to construct enough non-overlapping held-out points. "
            f"Requested {num_random}, collected {num_collected}."
        )

    test_points = torch.cat(collected_chunks, dim=0)

    metadata = {
        "mode": "uniform_random_cube_near_y_star_excluding_all_training_grids",
        "num_random_points": int(num_random),
        "num_total_points": int(test_points.shape[0]),
        "seed": int(seed),
        "sampling_center": "y_star",
        "base_sampling_radius": float(radius),
        "radius_scale": float(radius_scale),
        "effective_sampling_radius": effective_radius,
        "strictly_excludes_all_training_points": True,
        "num_candidate_points_generated": int(num_candidates_generated),
        "num_candidate_points_rejected_due_to_training_overlap": int(num_rejected_overlap),
        "num_unique_training_grids_checked": int(len(grid_specs)),
        "training_grid_specs_checked": [
            {
                "target_dataset_size": spec.target_dataset_size,
                "actual_dataset_size": spec.actual_dataset_size,
                "points_per_axis": spec.points_per_axis,
                "sampling_radius": spec.sampling_radius,
                "axis_spacing": spec.axis_spacing,
            }
            for spec in grid_specs
        ],
        "reason": (
            "The test set is generated by continuous random sampling near y_star and then "
            "explicitly filtered to remove any point lying on any saved regular training grid."
        ),
    }
    return test_points, metadata


# ============================================================
# 4. 批量评估：测试集和 p_n residual / loss gap
# ============================================================


@torch.no_grad()
def evaluate_model_on_initial_set(
    *,
    model: MLPOptimizer,
    initial_points_cpu: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    y_star: torch.Tensor,
    m: float,
    g: float,
    dt: float,
    steps: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model.eval()

    p_n_device = p_n.to(device=device, dtype=dtype)
    v_n_device = v_n.to(device=device, dtype=dtype)
    y_star_device = y_star.to(device=device, dtype=dtype)
    history = torch.cat([p_n_device, v_n_device], dim=0)
    params = torch.tensor([m, g, dt], device=device, dtype=dtype)
    e_star = variational_energy(y_star_device, p_n_device, v_n_device, m, g, dt)

    all_residuals: list[torch.Tensor] = []
    all_gaps: list[torch.Tensor] = []
    num_points = int(initial_points_cpu.shape[0])

    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        y = initial_points_cpu[start:end].to(device=device, dtype=dtype)
        batch_residuals = []
        batch_gaps = []

        for step in range(steps + 1):
            residual = stationarity_residual_norm(y, p_n_device, v_n_device, m, g, dt)
            energy = variational_energy(y, p_n_device, v_n_device, m, g, dt)
            gap = energy - e_star

            batch_residuals.append(residual.detach().cpu())
            batch_gaps.append(gap.detach().cpu())

            if step == steps:
                break
            delta = model(y, history, params)
            y = y + delta

        all_residuals.append(torch.stack(batch_residuals, dim=1))
        all_gaps.append(torch.stack(batch_gaps, dim=1))

    residuals = torch.cat(all_residuals, dim=0).numpy().astype(float)
    gaps = torch.cat(all_gaps, dim=0).numpy().astype(float)

    residuals[~np.isfinite(residuals)] = np.nan
    gaps[~np.isfinite(gaps)] = np.nan

    def stats(prefix: str, values: np.ndarray) -> dict[str, Any]:
        final_values = values[:, -1]
        return {
            f"{prefix}_mean_by_step": [float(x) for x in np.nanmean(values, axis=0).tolist()],
            f"{prefix}_median_by_step": [float(x) for x in np.nanmedian(values, axis=0).tolist()],
            f"{prefix}_p95_by_step": [float(x) for x in np.nanpercentile(values, 95, axis=0).tolist()],
            f"{prefix}_max_by_step": [float(x) for x in np.nanmax(values, axis=0).tolist()],
            f"final_{prefix}_mean": float(np.nanmean(final_values)),
            f"final_{prefix}_median": float(np.nanmedian(final_values)),
            f"final_{prefix}_p95": float(np.nanpercentile(final_values, 95)),
            f"final_{prefix}_max": float(np.nanmax(final_values)),
            f"final_{prefix}_num_nonfinite": int(np.count_nonzero(~np.isfinite(final_values))),
        }

    result = {
        "steps": int(steps),
        "num_points": int(num_points),
    }
    result.update(stats("residual", residuals))
    result.update(stats("loss_gap", gaps))

    if num_points == 1:
        result["single_point_residual_by_step"] = [
            float(x) if math.isfinite(float(x)) else None for x in residuals[0].tolist()
        ]
        result["single_point_loss_gap_by_step"] = [
            float(x) if math.isfinite(float(x)) else None for x in gaps[0].tolist()
        ]
        result["single_point_final_residual"] = result["single_point_residual_by_step"][-1]
        result["single_point_final_loss_gap"] = result["single_point_loss_gap_by_step"][-1]

    return result


@torch.no_grad()
def evaluate_single_trajectory(
    *,
    model: MLPOptimizer,
    initial_y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    y_star: torch.Tensor,
    m: float,
    g: float,
    dt: float,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    p_n_device = p_n.to(device=device, dtype=dtype)
    v_n_device = v_n.to(device=device, dtype=dtype)
    y_star_device = y_star.to(device=device, dtype=dtype)
    y = initial_y.to(device=device, dtype=dtype).clone()
    history = torch.cat([p_n_device, v_n_device], dim=0)
    params = torch.tensor([m, g, dt], device=device, dtype=dtype)
    e_star = variational_energy(y_star_device, p_n_device, v_n_device, m, g, dt)

    iterations: list[dict[str, Any]] = []
    for step in range(steps + 1):
        energy = variational_energy(y, p_n_device, v_n_device, m, g, dt)
        residual_norm = stationarity_residual_norm(y, p_n_device, v_n_device, m, g, dt)
        iterations.append(
            {
                "step": int(step),
                "y": [float(x) for x in y.detach().cpu().tolist()],
                "energy": float(energy.detach().cpu().item()),
                "gap": float((energy - e_star).detach().cpu().item()),
                "residual_norm": float(residual_norm.detach().cpu().item()),
            }
        )
        if step == steps:
            break

        delta = model(y, history, params)
        y = y + delta
        iterations[-1]["next_delta_norm"] = float(torch.linalg.vector_norm(delta).detach().cpu().item())

    return {
        "initial_y": [float(x) for x in initial_y.detach().cpu().tolist()],
        "iterations": iterations,
    }


@torch.no_grad()
def evaluate_newton_trajectory(
    *,
    initial_y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    y_star: torch.Tensor,
    m: float,
    g: float,
    dt: float,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    p_n_device = p_n.to(device=device, dtype=dtype)
    v_n_device = v_n.to(device=device, dtype=dtype)
    y_star_device = y_star.to(device=device, dtype=dtype)
    y = initial_y.to(device=device, dtype=dtype).clone()
    e_star = variational_energy(y_star_device, p_n_device, v_n_device, m, g, dt)

    iterations: list[dict[str, Any]] = []
    for step in range(steps + 1):
        energy = variational_energy(y, p_n_device, v_n_device, m, g, dt)
        residual_norm = stationarity_residual_norm(y, p_n_device, v_n_device, m, g, dt)
        iterations.append(
            {
                "step": int(step),
                "y": [float(x) for x in y.detach().cpu().tolist()],
                "energy": float(energy.detach().cpu().item()),
                "gap": float((energy - e_star).detach().cpu().item()),
                "residual_norm": float(residual_norm.detach().cpu().item()),
            }
        )
        if step == steps:
            break

        delta = newton_direction(y, p_n_device, v_n_device, m, g, dt)
        y = y + delta
        iterations[-1]["next_delta_norm"] = float(torch.linalg.vector_norm(delta).detach().cpu().item())

    return {
        "initial_y": [float(x) for x in initial_y.detach().cpu().tolist()],
        "iterations": iterations,
    }


# ============================================================
# 5. 绘图工具
# ============================================================


def optimizer_key(record: dict[str, Any]) -> tuple[str, float]:
    return str(record["optimizer_name"]).lower(), float(record["learning_rate"])


def optimizer_label(record_or_key: dict[str, Any] | tuple[str, float]) -> str:
    if isinstance(record_or_key, tuple):
        name, lr = record_or_key
    else:
        name, lr = optimizer_key(record_or_key)
    return f"{name.upper()} lr={lr:.0e}"


def unique_optimizer_keys(records: Sequence[dict[str, Any]]) -> list[tuple[str, float]]:
    keys = sorted({optimizer_key(r) for r in records}, key=lambda item: (item[0], item[1]))
    preferred_order = {"sgd": 0, "adam": 1}
    return sorted(keys, key=lambda item: (preferred_order.get(item[0], 99), -item[1]))


def finite_rows(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    points = finite_rows(points)
    if points.shape[0] == 0:
        points = np.zeros((1, 3), dtype=float)
    center = points.mean(axis=0)
    radius = max(float(np.ptp(points, axis=0).max()) / 2.0, 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_training_and_test_distribution_overview(
    *,
    grid_specs: Sequence[RegularGridSpec],
    test_points_cpu: torch.Tensor,
    y_star: Sequence[float],
    p_n: Sequence[float],
    save_path: Path,
    max_training_points_per_subplot: int = 6000,
    max_test_points: int = 6000,
) -> None:
    y_star_np = np.asarray(y_star, dtype=float)
    p_n_np = np.asarray(p_n, dtype=float)
    test_points = test_points_cpu.detach().cpu().numpy()
    if test_points.shape[0] > max_test_points:
        idx = np.linspace(0, test_points.shape[0] - 1, max_test_points).round().astype(int)
        test_points = test_points[idx]

    num_plots = len(grid_specs) + 1
    num_cols = min(4, num_plots)
    num_rows = math.ceil(num_plots / num_cols)
    fig = plt.figure(figsize=(5.2 * num_cols, 4.9 * num_rows))

    for i, spec in enumerate(grid_specs):
        ax = fig.add_subplot(num_rows, num_cols, i + 1, projection="3d")
        train_points = sample_regular_grid_points_for_plot(
            grid_spec=spec,
            y_star=y_star_np,
            max_points=max_training_points_per_subplot,
        )
        ax.scatter(
            train_points[:, 0],
            train_points[:, 1],
            train_points[:, 2],
            s=5,
            alpha=0.35,
            color="C0",
            label=f"training ({train_points.shape[0]}/{spec.actual_dataset_size})",
        )
        ax.scatter(p_n_np[0], p_n_np[1], p_n_np[2], marker="x", s=120, linewidths=2.0, color="C3", label=r"$p_n$")
        ax.scatter(y_star_np[0], y_star_np[1], y_star_np[2], marker="*", s=220, color="C2", label=r"$y^*$")
        set_equal_3d_axes(ax, np.vstack([train_points, p_n_np[None, :], y_star_np[None, :]]))
        ax.set_title(f"Training set distribution\nN={spec.actual_dataset_size:,}, axis={spec.points_per_axis}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=7)

    ax = fig.add_subplot(num_rows, num_cols, len(grid_specs) + 1, projection="3d")
    ax.scatter(
        test_points[:, 0],
        test_points[:, 1],
        test_points[:, 2],
        s=5,
        alpha=0.35,
        color="C1",
        label=f"held-out test ({test_points.shape[0]}/{test_points_cpu.shape[0]})",
    )
    ax.scatter(p_n_np[0], p_n_np[1], p_n_np[2], marker="x", s=120, linewidths=2.0, color="C3", label=r"$p_n$")
    ax.scatter(y_star_np[0], y_star_np[1], y_star_np[2], marker="*", s=220, color="C2", label=r"$y^*$")
    set_equal_3d_axes(ax, np.vstack([test_points, p_n_np[None, :], y_star_np[None, :]]))
    ax.set_title("Held-out test set distribution\n(excludes all training-grid points)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=7)

    fig.suptitle(
        "Set distribution overview: training grids and held-out test set\n"
        "Color code: training=C0, test=C1, p_n=C3, y*=C2",
        y=1.02,
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_residual_loss_vs_dataset_size(records: Sequence[dict[str, Any]], save_path: Path) -> None:
    """测试集和 p_n 的 residual / energy loss gap 汇总图。"""

    fig, axes = plt.subplots(2, 4, figsize=(22, 10.8))

    residual_metrics = [
        ("final_residual_mean", "Test mean residual", r"$\frac{1}{|T|}\sum_{y_0\in T} r_K(y_0)$"),
        ("final_residual_median", "Test median residual", r"$\mathrm{median}_{y_0\in T}\, r_K(y_0)$"),
        ("final_residual_p95", "Test p95 residual", r"$\mathrm{p95}_{y_0\in T}\, r_K(y_0)$"),
        ("pn_final_residual", r"$p_n$ residual", r"$r_K(p_n)$"),
    ]
    loss_metrics = [
        ("final_loss_gap_mean", "Test mean energy loss gap", r"$\frac{1}{|T|}\sum_{y_0\in T} \ell_K(y_0)$"),
        ("final_loss_gap_median", "Test median energy loss gap", r"$\mathrm{median}_{y_0\in T}\, \ell_K(y_0)$"),
        ("final_loss_gap_p95", "Test p95 energy loss gap", r"$\mathrm{p95}_{y_0\in T}\, \ell_K(y_0)$"),
        ("pn_final_loss_gap", r"$p_n$ energy loss gap", r"$\ell_K(p_n)$"),
    ]

    for key in unique_optimizer_keys(records):
        selected = sorted(
            [r for r in records if optimizer_key(r) == key],
            key=lambda r: int(r["dataset_size"]),
        )
        sizes = [int(r["dataset_size"]) for r in selected]

        for ax, (metric_name, title, ylabel) in zip(axes[0], residual_metrics):
            values = [finite_plot_value(r.get(metric_name)) for r in selected]
            ax.plot(sizes, values, marker="o", label=optimizer_label(key))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Number of regular-grid training initial states")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        for ax, (metric_name, title, ylabel) in zip(axes[1], loss_metrics):
            values = [finite_plot_value(r.get(metric_name)) for r in selected]
            ax.plot(sizes, values, marker="o", label=optimizer_label(key))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Number of regular-grid training initial states")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

    for row in axes:
        for ax in row:
            ax.legend(fontsize=7)

    fig.suptitle(
        "Held-out test set and original p_n: residual / energy loss gap vs. training dataset size",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        (
            r"$r_k(y_0)=\|\nabla E(y^{(k)}(y_0))\|_2$, "
            r"$\ell_k(y_0)=E(y^{(k)}(y_0))-E(y^*)$. "
            r"The first three columns aggregate over held-out test set $T$; the last column is the single-point $p_n$ metric."
        ),
        ha="center",
        va="top",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_metric_trajectory_6_subplots(
    *,
    records: Sequence[dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    keys = unique_optimizer_keys(records)
    num_cols = 3
    num_rows = math.ceil(len(keys) / num_cols)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(6.2 * num_cols, 4.8 * num_rows),
        sharex=True,
    )
    axes_array = np.asarray(axes).reshape(-1)

    for ax_index, key in enumerate(keys):
        ax = axes_array[ax_index]
        selected = sorted(
            [r for r in records if optimizer_key(r) == key],
            key=lambda r: int(r["dataset_size"]),
        )
        for r in selected:
            curve = [finite_plot_value(v) for v in r[metric_key]]
            ax.plot(
                range(len(curve)),
                curve,
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                label=f"N={int(r['dataset_size']):,}",
            )
        ax.set_yscale("log")
        ax.set_title(optimizer_label(key))
        ax.set_xlabel("Iteration k")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes_array[len(keys):]:
        ax.axis("off")

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reference_residual_comparison(
    *,
    mlp_trajectory: dict,
    newton_trajectory: dict,
    save_path: Path,
    title: str,
) -> None:
    mlp_iterations = mlp_trajectory["iterations"]
    newton_iterations = newton_trajectory["iterations"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        [item["step"] for item in mlp_iterations],
        [finite_plot_value(item["residual_norm"]) for item in mlp_iterations],
        marker="o",
        label="MLP optimizer",
    )
    ax.plot(
        [item["step"] for item in newton_iterations],
        [finite_plot_value(item["residual_norm"]) for item in newton_iterations],
        marker="s",
        linestyle="--",
        label="Newton method",
    )
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Stationarity residual $\|\nabla E(y)\|_2$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reference_trajectory_3d(
    *,
    mlp_trajectory: dict,
    newton_trajectory: dict,
    y_star: Sequence[float],
    save_path: Path,
    title: str,
) -> None:
    mlp_points = np.asarray([item["y"] for item in mlp_trajectory["iterations"]], dtype=float)
    newton_points = np.asarray([item["y"] for item in newton_trajectory["iterations"]], dtype=float)
    initial_point = np.asarray(mlp_trajectory["initial_y"], dtype=float)
    y_star_np = np.asarray(y_star, dtype=float)

    mlp_points_finite = finite_rows(mlp_points)
    newton_points_finite = finite_rows(newton_points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        mlp_points_finite[:, 0],
        mlp_points_finite[:, 1],
        mlp_points_finite[:, 2],
        "-o",
        linewidth=1.5,
        markersize=4,
        label="MLP trajectory",
    )
    ax.plot(
        newton_points_finite[:, 0],
        newton_points_finite[:, 1],
        newton_points_finite[:, 2],
        "--s",
        linewidth=1.2,
        markersize=4,
        label="Newton trajectory",
    )
    for step, point in enumerate(mlp_points_finite):
        ax.text(point[0], point[1], point[2], f"  {step}", fontsize=7)

    ax.scatter(initial_point[0], initial_point[1], initial_point[2], marker="x", s=140, linewidths=2.0, label="test initial point")
    ax.scatter(y_star_np[0], y_star_np[1], y_star_np[2], marker="*", s=320, label=r"$y^*$ / Newton solution")

    set_equal_3d_axes(ax, np.vstack([mlp_points, newton_points, initial_point.reshape(1, 3), y_star_np.reshape(1, 3)]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reference_energy_contour_2d(
    *,
    mlp_trajectory: dict,
    newton_trajectory: dict,
    y_star: Sequence[float],
    p_n: Sequence[float],
    v_n: Sequence[float],
    m: float,
    g: float,
    dt: float,
    save_path: Path,
    title: str,
) -> None:
    mlp_points = np.asarray([item["y"] for item in mlp_trajectory["iterations"]], dtype=float)
    newton_points = np.asarray([item["y"] for item in newton_trajectory["iterations"]], dtype=float)
    y_star_np = np.asarray(y_star, dtype=float)
    p_n_np = np.asarray(p_n, dtype=float)
    v_n_np = np.asarray(v_n, dtype=float)
    initial_np = np.asarray(mlp_trajectory["initial_y"], dtype=float)

    mlp_points_finite = finite_rows(mlp_points)
    newton_points_finite = finite_rows(newton_points)

    projected = np.vstack(
        [
            mlp_points_finite[:, [0, 2]],
            newton_points_finite[:, [0, 2]],
            y_star_np[[0, 2]].reshape(1, 2),
            initial_np[[0, 2]].reshape(1, 2),
        ]
    )
    lower = projected.min(axis=0)
    upper = projected.max(axis=0)
    span = np.maximum(upper - lower, 2e-4)
    lower = lower - 0.2 * span
    upper = upper + 0.2 * span

    x_values = np.linspace(lower[0], upper[0], 240)
    z_values = np.linspace(lower[1], upper[1], 240)
    x_grid, z_grid = np.meshgrid(x_values, z_values)

    points = np.broadcast_to(y_star_np.reshape(1, 1, 3), (240, 240, 3)).copy()
    points[..., 0] = x_grid
    points[..., 2] = z_grid

    residual = points - p_n_np - dt * v_n_np
    energy = (m / (2.0 * dt**2)) * np.sum(residual**2, axis=-1) + m * g * points[..., 2]

    y_star_residual = y_star_np - p_n_np - dt * v_n_np
    e_star = (m / (2.0 * dt**2)) * np.sum(y_star_residual**2) + m * g * y_star_np[2]
    gap = np.maximum(energy - e_star, PLOT_FLOOR)

    max_gap = float(np.max(gap))
    min_level = max(max_gap * 1e-8, PLOT_FLOOR)
    max_level = max(max_gap, min_level * 10.0)
    levels = np.geomspace(min_level, max_level, 28)

    fig, ax = plt.subplots(figsize=(9, 7))
    contour = ax.contourf(
        x_grid,
        z_grid,
        gap,
        levels=levels,
        norm=matplotlib.colors.LogNorm(vmin=min_level, vmax=max_level),
        alpha=0.82,
        extend="both",
    )
    ax.contour(
        x_grid,
        z_grid,
        gap,
        levels=levels,
        norm=matplotlib.colors.LogNorm(vmin=min_level, vmax=max_level),
        linewidths=0.35,
        alpha=0.45,
    )
    ax.plot(
        mlp_points_finite[:, 0],
        mlp_points_finite[:, 2],
        "-o",
        linewidth=1.8,
        markersize=4,
        label="MLP projected trajectory",
    )
    ax.plot(
        newton_points_finite[:, 0],
        newton_points_finite[:, 2],
        "--s",
        linewidth=1.5,
        markersize=4,
        label="Newton projected trajectory",
    )
    for step, point in enumerate(mlp_points_finite):
        ax.text(point[0], point[2], f"  {step}", fontsize=7)

    ax.scatter(initial_np[0], initial_np[2], marker="x", s=120, linewidths=2.0, label="test initial point")
    ax.scatter(y_star_np[0], y_star_np[2], marker="*", s=260, label=r"$y^*$")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label(r"Energy loss gap $E(y)-E(y^*)$")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. 详细测试点选择与绘图
# ============================================================


def find_best_matching_experiment(
    *,
    experiment_files: Sequence[ExperimentFile],
    optimizer_name: str,
    learning_rate: float,
    target_dataset_size: int,
) -> ExperimentFile:
    candidates = []
    for exp in experiment_files:
        cfg = exp.report.get("config", {})
        if str(cfg.get("optimizer_name", "")).lower() != optimizer_name.lower():
            continue
        if not math.isclose(float(cfg.get("learning_rate", float("nan"))), learning_rate, rel_tol=0.0, abs_tol=learning_rate * 1e-8):
            continue
        actual = int(cfg.get("actual_dataset_size", parse_dataset_size_from_name(exp.experiment_dir.name) or -1))
        if actual <= 0:
            continue
        candidates.append((abs(actual - target_dataset_size), actual, exp))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find experiment optimizer={optimizer_name}, lr={learning_rate:.0e}, "
            f"target_dataset_size≈{target_dataset_size}."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def select_detailed_test_indices(num_test_points: int, num_points: int = 3) -> list[int]:
    if num_test_points <= 0:
        raise ValueError("num_test_points must be positive.")
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    if num_test_points < num_points:
        return list(range(num_test_points))
    return sorted(set(np.linspace(0, num_test_points - 1, num_points).round().astype(int).tolist()))


def run_detailed_test_point_plots(
    *,
    experiment_files: Sequence[ExperimentFile],
    test_points_cpu: torch.Tensor,
    p_n_cpu: torch.Tensor,
    v_n_cpu: torch.Tensor,
    y_star_cpu: torch.Tensor,
    m: float,
    g: float,
    dt: float,
    device: torch.device,
    steps: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    detailed_root = output_dir / "detailed_test_points"
    detailed_root.mkdir(parents=True, exist_ok=True)

    selected_experiments = [
        ("adam", 1e-4, 8),
        ("adam", 1e-4, 10_000),
    ]
    test_indices = select_detailed_test_indices(int(test_points_cpu.shape[0]), num_points=3)

    results = []
    for optimizer_name, learning_rate, target_size in selected_experiments:
        exp = find_best_matching_experiment(
            experiment_files=experiment_files,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            target_dataset_size=target_size,
        )
        cfg = exp.report.get("config", {})
        actual_size = int(cfg.get("actual_dataset_size", parse_dataset_size_from_name(exp.experiment_dir.name) or -1))
        model, dtype, tensors = instantiate_model_from_report_and_state(experiment=exp, device=device)

        model_dir = detailed_root / f"{optimizer_name}_lr_{learning_rate:.0e}_N_{actual_size}"
        model_dir.mkdir(parents=True, exist_ok=True)

        for local_id, test_index in enumerate(test_indices):
            initial_y = test_points_cpu[test_index].to(dtype=dtype)
            point_dir = model_dir / f"test_point_{local_id:03d}_index_{test_index}"
            point_dir.mkdir(parents=True, exist_ok=True)

            mlp_trajectory = evaluate_single_trajectory(
                model=model,
                initial_y=initial_y,
                p_n=tensors["p_n"],
                v_n=tensors["v_n"],
                y_star=y_star_cpu.to(dtype=dtype),
                m=m,
                g=g,
                dt=dt,
                steps=steps,
                device=device,
                dtype=dtype,
            )
            newton_trajectory = evaluate_newton_trajectory(
                initial_y=initial_y,
                p_n=tensors["p_n"],
                v_n=tensors["v_n"],
                y_star=y_star_cpu.to(dtype=dtype),
                m=m,
                g=g,
                dt=dt,
                steps=steps,
                device=device,
                dtype=dtype,
            )

            title_prefix = f"Adam lr=1e-4, N={actual_size:,}, test point {local_id} (index {test_index})"
            plot_reference_residual_comparison(
                mlp_trajectory=mlp_trajectory,
                newton_trajectory=newton_trajectory,
                save_path=point_dir / "final_reference_residual_comparison.png",
                title=title_prefix + "\nResidual comparison",
            )
            plot_reference_energy_contour_2d(
                mlp_trajectory=mlp_trajectory,
                newton_trajectory=newton_trajectory,
                y_star=y_star_cpu.tolist(),
                p_n=p_n_cpu.tolist(),
                v_n=v_n_cpu.tolist(),
                m=m,
                g=g,
                dt=dt,
                save_path=point_dir / "final_reference_energy_contour_2d.png",
                title=title_prefix + "\nProjected trajectory on energy loss contours",
            )
            plot_reference_trajectory_3d(
                mlp_trajectory=mlp_trajectory,
                newton_trajectory=newton_trajectory,
                y_star=y_star_cpu.tolist(),
                save_path=point_dir / "final_reference_trajectory_3d.png",
                title=title_prefix + "\n3D trajectory",
            )

            report = {
                "experiment_name": cfg.get("experiment_name", exp.experiment_dir.name),
                "experiment_dir": str(exp.experiment_dir),
                "optimizer_name": optimizer_name,
                "learning_rate": learning_rate,
                "target_dataset_size_requested": target_size,
                "actual_dataset_size": actual_size,
                "test_point_local_id": int(local_id),
                "test_point_global_index": int(test_index),
                "initial_y": [float(x) for x in initial_y.detach().cpu().tolist()],
                "mlp_trajectory": mlp_trajectory,
                "newton_trajectory": newton_trajectory,
                "plots": {
                    "final_reference_residual_comparison": str(point_dir / "final_reference_residual_comparison.png"),
                    "final_reference_energy_contour_2d": str(point_dir / "final_reference_energy_contour_2d.png"),
                    "final_reference_trajectory_3d": str(point_dir / "final_reference_trajectory_3d.png"),
                },
            }

            report_path = point_dir / "detailed_trajectory_report.json"
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(make_json_safe(report), f, indent=2, ensure_ascii=False)

            results.append(report)

    detailed_summary_path = detailed_root / "detailed_test_points_summary.json"
    with detailed_summary_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe({"selected_test_indices": test_indices, "results": results}), f, indent=2, ensure_ascii=False)

    return results


# ============================================================
# 7. 主程序
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved MLP optimizer state_dicts on held-out residual/loss metrics and detailed trajectories."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Training output directory containing dataset_scale_ablation_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for test reports and figures. Default: <results-dir>/heldout_test_residual_loss_detailed.",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-test", type=int, default=4096, help="Number of random held-out test points.")
    parser.add_argument("--test-seed", type=int, default=20260617)
    parser.add_argument("--steps", type=int, default=50, help="Number of MLP/Newton iterations during testing.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Test cube radius = training sampling_radius * radius_scale.",
    )
    parser.add_argument(
        "--skip-detailed",
        action="store_true",
        help="Skip the detailed Adam lr=1e-4 trajectories for three test points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = discover_results_dir(args.results_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else results_dir / "heldout_test_residual_loss_detailed"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    experiment_files = find_experiment_files(results_dir)
    grid_specs = extract_unique_grid_specs(experiment_files)

    print(f"results_dir = {results_dir}")
    print(f"output_dir  = {output_dir}")
    print(f"device      = {device}")
    print(f"experiments = {len(experiment_files)}")
    print(f"unique training grids = {len(grid_specs)}")

    first_cfg = experiment_files[0].report.get("config", {})
    first_state = safe_torch_load_state_dict(experiment_files[0].model_path)
    global_dtype = infer_state_dtype(first_state)

    p_n_cpu = torch.tensor(first_cfg.get("p_n", [3.0, 4.0, 5.0]), dtype=global_dtype)
    v_n_cpu = torch.tensor(first_cfg.get("v_n", [0.5, -0.5, 0.0]), dtype=global_dtype)
    m = float(first_cfg.get("m", 1.0))
    g = float(first_cfg.get("g", 9.8))
    dt = float(first_cfg.get("dt", 0.01))
    y_star_cpu = torch.tensor(
        first_cfg.get(
            "y_star",
            (p_n_cpu + dt * v_n_cpu - dt**2 * torch.tensor([0.0, 0.0, g], dtype=global_dtype)).tolist(),
        ),
        dtype=global_dtype,
    )
    sampling_radius = float(first_cfg.get("sampling_radius_per_axis", first_cfg.get("sampling_radius", 0.01)))

    test_points_cpu, test_metadata = build_fixed_heldout_test_set_excluding_training_grids(
        y_star=y_star_cpu,
        radius=sampling_radius,
        num_random=int(args.num_test),
        seed=int(args.test_seed),
        radius_scale=float(args.radius_scale),
        dtype=global_dtype,
        grid_specs=grid_specs,
    )

    plot_training_and_test_distribution_overview(
        grid_specs=grid_specs,
        test_points_cpu=test_points_cpu,
        y_star=y_star_cpu.tolist(),
        p_n=p_n_cpu.tolist(),
        save_path=output_dir / "training_and_test_set_distribution_overview.png",
    )

    records: list[dict[str, Any]] = []
    for exp in experiment_files:
        cfg = exp.report.get("config", {})
        print(f"\n[eval] {exp.experiment_dir.name}")

        model, dtype, tensors = instantiate_model_from_report_and_state(
            experiment=exp,
            device=device,
        )
        p_n = tensors["p_n"]
        v_n = tensors["v_n"]
        m_i = float(cfg.get("m", m))
        g_i = float(cfg.get("g", g))
        dt_i = float(cfg.get("dt", dt))

        if not torch.allclose(p_n.to(global_dtype), p_n_cpu.to(global_dtype)) or not torch.allclose(v_n.to(global_dtype), v_n_cpu.to(global_dtype)):
            raise ValueError(
                f"Experiment {exp.experiment_dir.name} uses different p_n/v_n. "
                "This script assumes all saved models share the same physical problem."
            )
        if (m_i, g_i, dt_i) != (m, g, dt):
            raise ValueError(
                f"Experiment {exp.experiment_dir.name} uses different m/g/dt. "
                "This script assumes all saved models share the same physical problem."
            )

        test_eval = evaluate_model_on_initial_set(
            model=model,
            initial_points_cpu=test_points_cpu.to(dtype=dtype),
            p_n=p_n,
            v_n=v_n,
            y_star=y_star_cpu.to(dtype=dtype),
            m=m_i,
            g=g_i,
            dt=dt_i,
            steps=int(args.steps),
            batch_size=int(args.batch_size),
            device=device,
            dtype=dtype,
        )
        pn_eval = evaluate_model_on_initial_set(
            model=model,
            initial_points_cpu=p_n_cpu.reshape(1, 3).to(dtype=dtype),
            p_n=p_n,
            v_n=v_n,
            y_star=y_star_cpu.to(dtype=dtype),
            m=m_i,
            g=g_i,
            dt=dt_i,
            steps=int(args.steps),
            batch_size=1,
            device=device,
            dtype=dtype,
        )

        dataset_size = int(
            cfg.get(
                "actual_dataset_size",
                parse_dataset_size_from_name(exp.experiment_dir.name) or -1,
            )
        )

        record = {
            "experiment_name": cfg.get("experiment_name", exp.experiment_dir.name),
            "experiment_dir": str(exp.experiment_dir),
            "report_path": str(exp.report_path),
            "model_path": str(exp.model_path),
            "optimizer_name": cfg.get("optimizer_name", "unknown"),
            "learning_rate": float(cfg.get("learning_rate", float("nan"))),
            "target_dataset_size": int(cfg.get("target_dataset_size", dataset_size)),
            "dataset_size": dataset_size,
            "points_per_axis": int(cfg.get("points_per_axis", -1)),
            "axis_spacing": cfg.get("axis_spacing", None),
            "test_num_points": int(test_eval["num_points"]),
            "residual_mean_by_step": test_eval["residual_mean_by_step"],
            "residual_median_by_step": test_eval["residual_median_by_step"],
            "residual_p95_by_step": test_eval["residual_p95_by_step"],
            "residual_max_by_step": test_eval["residual_max_by_step"],
            "loss_gap_mean_by_step": test_eval["loss_gap_mean_by_step"],
            "loss_gap_median_by_step": test_eval["loss_gap_median_by_step"],
            "loss_gap_p95_by_step": test_eval["loss_gap_p95_by_step"],
            "loss_gap_max_by_step": test_eval["loss_gap_max_by_step"],
            "final_residual_mean": test_eval["final_residual_mean"],
            "final_residual_median": test_eval["final_residual_median"],
            "final_residual_p95": test_eval["final_residual_p95"],
            "final_residual_max": test_eval["final_residual_max"],
            "final_loss_gap_mean": test_eval["final_loss_gap_mean"],
            "final_loss_gap_median": test_eval["final_loss_gap_median"],
            "final_loss_gap_p95": test_eval["final_loss_gap_p95"],
            "final_loss_gap_max": test_eval["final_loss_gap_max"],
            "pn_residual_by_step": pn_eval["single_point_residual_by_step"],
            "pn_loss_gap_by_step": pn_eval["single_point_loss_gap_by_step"],
            "pn_final_residual": pn_eval["single_point_final_residual"],
            "pn_final_loss_gap": pn_eval["single_point_final_loss_gap"],
        }
        records.append(record)

        print(
            "  test residual mean={:.4e}, test loss mean={:.4e}, p_n residual={:.4e}, p_n loss={:.4e}".format(
                record["final_residual_mean"],
                record["final_loss_gap_mean"],
                record["pn_final_residual"],
                record["pn_final_loss_gap"],
            )
        )

    records = sorted(records, key=lambda r: (optimizer_key(r), int(r["dataset_size"])))

    detailed_results: list[dict[str, Any]] = []
    if not bool(args.skip_detailed):
        detailed_results = run_detailed_test_point_plots(
            experiment_files=experiment_files,
            test_points_cpu=test_points_cpu,
            p_n_cpu=p_n_cpu,
            v_n_cpu=v_n_cpu,
            y_star_cpu=y_star_cpu,
            m=m,
            g=g,
            dt=dt,
            device=device,
            steps=int(args.steps),
            output_dir=output_dir,
        )

    summary = {
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "test_set": test_metadata,
        "physics": {
            "p_n": p_n_cpu.tolist(),
            "v_n": v_n_cpu.tolist(),
            "m": m,
            "g": g,
            "dt": dt,
            "y_star": y_star_cpu.tolist(),
        },
        "evaluation": {
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "pointwise_metrics": {
                "residual": r"r_k(y_0)=||∇E(y^(k)(y_0))||_2",
                "loss_gap": r"ell_k(y_0)=E(y^(k)(y_0))-E(y*)",
            },
            "aggregate_metrics": {
                "test_set": "mean/median/p95/max over the fixed held-out test set T",
                "p_n": "single-point trajectory metric for the original physical initial state p_n",
            },
        },
        "experiments": records,
        "detailed_test_point_results": detailed_results,
    }

    summary_path = output_dir / "heldout_test_residual_loss_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary), f, indent=2, ensure_ascii=False)

    plot_final_residual_loss_vs_dataset_size(
        records,
        output_dir / "heldout_final_residual_loss_vs_training_dataset_size.png",
    )
    plot_mean_metric_trajectory_6_subplots(
        records=records,
        metric_key="residual_mean_by_step",
        ylabel=r"$\bar r_k=\frac{1}{|T|}\sum_{y_0\in T} r_k(y_0)$",
        title="Held-out test residual trajectories (mean over test set)",
        save_path=output_dir / "heldout_mean_residual_trajectory_6_optimizer_settings.png",
    )
    plot_mean_metric_trajectory_6_subplots(
        records=records,
        metric_key="loss_gap_mean_by_step",
        ylabel=r"$\bar \ell_k=\frac{1}{|T|}\sum_{y_0\in T} \ell_k(y_0)$",
        title="Held-out test energy loss-gap trajectories (mean over test set)",
        save_path=output_dir / "heldout_mean_loss_gap_trajectory_6_optimizer_settings.png",
    )

    print("\n✅ 测试完成")
    print(f"📄 测试 JSON: {summary_path}")
    print(f"🖼️ 训练/测试集分布总览图: {output_dir / 'training_and_test_set_distribution_overview.png'}")
    print(f"🖼️ residual/loss 汇总图: {output_dir / 'heldout_final_residual_loss_vs_training_dataset_size.png'}")
    print(f"🖼️ residual 迭代曲线: {output_dir / 'heldout_mean_residual_trajectory_6_optimizer_settings.png'}")
    print(f"🖼️ loss 迭代曲线: {output_dir / 'heldout_mean_loss_gap_trajectory_6_optimizer_settings.png'}")
    if not bool(args.skip_detailed):
        print(f"🖼️ 详细测试点图目录: {output_dir / 'detailed_test_points'}")


if __name__ == "__main__":
    main()
