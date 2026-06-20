r"""
用已保存的 MLP optimizer 参数批量测试 held-out 测试集残差。

适配训练脚本输出结构：
    <results_dir>/dataset_scale_ablation_summary.json
    <results_dir>/<experiment_name>/optimization_report.json
    <results_dir>/<experiment_name>/mlp_optimizer_state_dict.pt

测试集设计：
    在训练脚本使用的同一物理问题、同一 y_star、同一 sampling_radius 内，
    构造固定随机 held-out 点集：
        y0 ~ Uniform([y_star - R, y_star + R]^3)
    并额外加入原始 held-out 初值 p_n。

这样做的目的：
    1. 测试点与训练规则网格几乎必然不重合；
    2. 所有“训练集规模 × 优化器参数”模型使用完全相同的测试集；
    3. 不只看单个 p_n，而是看局部收敛域内的平均/中位数/95分位/max 残差。

典型用法：
    python test_saved_mlp_residuals.py --results-dir ./Pasted\ code\(15\) --device cuda:0

如果不指定 --results-dir，脚本会在当前目录和脚本同目录下自动寻找
最近的 dataset_scale_ablation_summary.json。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

PLOT_FLOOR = 1e-14


# ============================================================
# 1. 与训练脚本一致的网络和物理残差定义
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


# ============================================================
# 2. 文件发现、读取与安全转换
# ============================================================


@dataclass(frozen=True)
class ExperimentFile:
    experiment_dir: Path
    report_path: Path
    model_path: Path
    report: dict[str, Any]


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


# ============================================================
# 3. 测试集构造与模型评估
# ============================================================


def build_fixed_heldout_test_set(
    *,
    y_star: torch.Tensor,
    p_n: torch.Tensor,
    radius: float,
    num_random: int,
    seed: int,
    include_pn: bool,
    radius_scale: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    构造固定 held-out 测试集。

    训练集是 y_star 附近的规则网格；这里用连续随机点测试同一区域。
    随机连续采样与有限规则训练网格重合的概率为 0。
    """

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
    p_n_cpu = p_n.detach().cpu().to(dtype=dtype)

    random_offsets = (
        2.0 * torch.rand((num_random, 3), generator=generator, dtype=dtype) - 1.0
    ) * effective_radius
    test_points = y_star_cpu.unsqueeze(0) + random_offsets

    if include_pn:
        test_points = torch.cat([p_n_cpu.unsqueeze(0), test_points], dim=0)

    metadata = {
        "mode": "uniform_random_cube_near_y_star",
        "num_random_points": int(num_random),
        "include_pn_as_first_point": bool(include_pn),
        "num_total_points": int(test_points.shape[0]),
        "seed": int(seed),
        "sampling_center": "y_star",
        "base_sampling_radius": float(radius),
        "radius_scale": float(radius_scale),
        "effective_sampling_radius": effective_radius,
        "reason": (
            "Training uses deterministic regular grids near y_star. "
            "The test set uses continuous random held-out points in the same region, "
            "so all saved models are evaluated on the same non-grid initial states."
        ),
    }
    return test_points, metadata


@torch.no_grad()
def evaluate_model_on_test_set(
    *,
    model: MLPOptimizer,
    test_points_cpu: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
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
    history = torch.cat([p_n_device, v_n_device], dim=0)
    params = torch.tensor([m, g, dt], device=device, dtype=dtype)

    all_residuals: list[torch.Tensor] = []
    num_points = int(test_points_cpu.shape[0])

    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        y = test_points_cpu[start:end].to(device=device, dtype=dtype)
        batch_residuals = []

        for step in range(steps + 1):
            residual = stationarity_residual_norm(y, p_n_device, v_n_device, m, g, dt)
            batch_residuals.append(residual.detach().cpu())
            if step == steps:
                break
            delta = model(y, history, params)
            y = y + delta

        all_residuals.append(torch.stack(batch_residuals, dim=1))  # [B, steps+1]

    residuals = torch.cat(all_residuals, dim=0)  # [N, steps+1]
    finite_mask = torch.isfinite(residuals)

    def to_float_list(t: torch.Tensor) -> list[float]:
        return [float(x) for x in t.detach().cpu().tolist()]

    # nan-aware 统计：非有限值先转为 nan，便于 numpy 计算分位数。
    residuals_np = residuals.numpy().astype(float)
    residuals_np[~np.isfinite(residuals_np)] = np.nan

    mean_by_step = np.nanmean(residuals_np, axis=0)
    median_by_step = np.nanmedian(residuals_np, axis=0)
    p95_by_step = np.nanpercentile(residuals_np, 95, axis=0)
    max_by_step = np.nanmax(residuals_np, axis=0)

    final_values = residuals_np[:, -1]

    result = {
        "steps": int(steps),
        "num_test_points": int(num_points),
        "num_finite_residual_entries": int(finite_mask.sum().item()),
        "num_total_residual_entries": int(finite_mask.numel()),
        "mean_residual_by_step": to_float_list(torch.tensor(mean_by_step)),
        "median_residual_by_step": to_float_list(torch.tensor(median_by_step)),
        "p95_residual_by_step": to_float_list(torch.tensor(p95_by_step)),
        "max_residual_by_step": to_float_list(torch.tensor(max_by_step)),
        "final_mean_residual": float(np.nanmean(final_values)),
        "final_median_residual": float(np.nanmedian(final_values)),
        "final_p95_residual": float(np.nanpercentile(final_values, 95)),
        "final_max_residual": float(np.nanmax(final_values)),
        "final_num_nonfinite": int(np.count_nonzero(~np.isfinite(final_values))),
    }

    # 如果第一个点是 p_n，则额外记录它的残差轨迹。
    result["pn_residual_by_step_assuming_first_point"] = [
        float(x) if math.isfinite(float(x)) else None for x in residuals_np[0].tolist()
    ]
    result["pn_final_residual_assuming_first_point"] = result[
        "pn_residual_by_step_assuming_first_point"
    ][-1]

    return result


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
# 4. 绘图
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
    # 更接近训练脚本默认展示顺序：SGD 再 Adam，每组 lr 从大到小。
    preferred_order = {"sgd": 0, "adam": 1}
    return sorted(keys, key=lambda item: (preferred_order.get(item[0], 99), -item[1]))


def plot_final_residual_vs_dataset_size(records: Sequence[dict[str, Any]], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [
        ("final_mean_residual", "Final mean test residual"),
        ("final_median_residual", "Final median test residual"),
        ("final_p95_residual", "Final p95 test residual"),
    ]

    for key in unique_optimizer_keys(records):
        selected = sorted(
            [r for r in records if optimizer_key(r) == key],
            key=lambda r: int(r["dataset_size"]),
        )
        sizes = [int(r["dataset_size"]) for r in selected]
        for ax, (metric_name, title) in zip(axes, metrics):
            values = [finite_plot_value(r[metric_name]) for r in selected]
            ax.plot(sizes, values, marker="o", label=optimizer_label(key))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("Number of regular-grid training initial states")
            ax.set_ylabel(r"$\|\nabla E(y)\|_2$")
            ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle("Held-out random test set: final residual vs. training dataset size", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pn_final_residual_vs_dataset_size(records: Sequence[dict[str, Any]], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for key in unique_optimizer_keys(records):
        selected = sorted(
            [r for r in records if optimizer_key(r) == key],
            key=lambda r: int(r["dataset_size"]),
        )
        sizes = [int(r["dataset_size"]) for r in selected]
        values = [finite_plot_value(r["pn_final_residual_assuming_first_point"]) for r in selected]
        ax.plot(sizes, values, marker="o", label=optimizer_label(key))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(r"Original held-out $p_n$: final residual vs. training dataset size")
    ax.set_xlabel("Number of regular-grid training initial states")
    ax.set_ylabel(r"$\|\nabla E(y)\|_2$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_residual_trajectory_6_subplots(records: Sequence[dict[str, Any]], save_path: Path) -> None:
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
            curve = [finite_plot_value(v) for v in r["mean_residual_by_step"]]
            ax.plot(range(len(curve)), curve, marker="o", markersize=2.5, linewidth=1.2, label=f"N={int(r['dataset_size']):,}")
        ax.set_yscale("log")
        ax.set_title(optimizer_label(key))
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Mean test residual $\|\nabla E(y)\|_2$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axes_array[len(keys):]:
        ax.axis("off")

    fig.suptitle("Held-out random test residual trajectories", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_test_set_distribution(
    *,
    test_points_cpu: torch.Tensor,
    y_star: Sequence[float],
    p_n: Sequence[float],
    save_path: Path,
    max_points: int = 12000,
) -> None:
    points = test_points_cpu.detach().cpu().numpy()
    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).round().astype(int)
        points = points[idx]
    y_star_np = np.asarray(y_star, dtype=float)
    p_n_np = np.asarray(p_n, dtype=float)

    fig = plt.figure(figsize=(9, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, alpha=0.25, label="Held-out random test points")
    ax.scatter(y_star_np[0], y_star_np[1], y_star_np[2], marker="*", s=260, label=r"$y^*$")
    ax.scatter(p_n_np[0], p_n_np[1], p_n_np[2], marker="x", s=140, linewidths=2.0, label=r"$p_n$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Held-out test set distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 5. 主程序
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved MLP optimizer state_dicts on a fixed held-out test set."
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
        help="Directory for test reports and figures. Default: <results-dir>/heldout_test_residuals.",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-test", type=int, default=4096, help="Number of random held-out test points.")
    parser.add_argument("--test-seed", type=int, default=20260617)
    parser.add_argument("--steps", type=int, default=50, help="Number of MLP iterations during testing.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1.0,
        help="Test cube radius = training sampling_radius * radius_scale.",
    )
    parser.add_argument(
        "--no-include-pn",
        action="store_true",
        help="Do not prepend the original physical initial state p_n to the test set.",
    )
    return parser.parse_args()


def parse_dataset_size_from_name(name: str) -> int | None:
    match = re.search(r"num_samples_(\d+)", name)
    return int(match.group(1)) if match else None


def main() -> None:
    args = parse_args()
    results_dir = discover_results_dir(args.results_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else results_dir / "heldout_test_residuals"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    experiment_files = find_experiment_files(results_dir)

    print(f"results_dir = {results_dir}")
    print(f"output_dir  = {output_dir}")
    print(f"device      = {device}")
    print(f"experiments = {len(experiment_files)}")

    # 用第一组实验的 config 构造全局固定测试集。
    first_cfg = experiment_files[0].report.get("config", {})
    # state_dict dtype 用第一组来确定，通常是 torch.float64。
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

    test_points_cpu, test_metadata = build_fixed_heldout_test_set(
        y_star=y_star_cpu,
        p_n=p_n_cpu,
        radius=sampling_radius,
        num_random=int(args.num_test),
        seed=int(args.test_seed),
        include_pn=not bool(args.no_include_pn),
        radius_scale=float(args.radius_scale),
        dtype=global_dtype,
    )

    plot_test_set_distribution(
        test_points_cpu=test_points_cpu,
        y_star=y_star_cpu.tolist(),
        p_n=p_n_cpu.tolist(),
        save_path=output_dir / "heldout_test_set_distribution.png",
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

        # 如果不同实验不小心用了不同物理问题，直接报错，避免混合比较。
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

        eval_result = evaluate_model_on_test_set(
            model=model,
            test_points_cpu=test_points_cpu.to(dtype=dtype),
            p_n=p_n,
            v_n=v_n,
            m=m_i,
            g=g_i,
            dt=dt_i,
            steps=int(args.steps),
            batch_size=int(args.batch_size),
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
            **eval_result,
        }
        records.append(record)
        print(
            "  final mean={:.4e}, median={:.4e}, p95={:.4e}, p_n={}".format(
                record["final_mean_residual"],
                record["final_median_residual"],
                record["final_p95_residual"],
                record["pn_final_residual_assuming_first_point"],
            )
        )

    records = sorted(records, key=lambda r: (optimizer_key(r), int(r["dataset_size"])))

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
            "metric": "stationarity_residual_norm ||grad E(y)||_2",
        },
        "experiments": records,
    }

    summary_path = output_dir / "heldout_test_residual_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(summary), f, indent=2, ensure_ascii=False)

    plot_final_residual_vs_dataset_size(
        records,
        output_dir / "heldout_final_residual_vs_training_dataset_size.png",
    )
    plot_pn_final_residual_vs_dataset_size(
        records,
        output_dir / "pn_final_residual_vs_training_dataset_size.png",
    )
    plot_mean_residual_trajectory_6_subplots(
        records,
        output_dir / "heldout_mean_residual_trajectory_6_optimizer_settings.png",
    )

    print("\n✅ 测试完成")
    print(f"📄 测试 JSON: {summary_path}")
    print(f"🖼️ 总体残差图: {output_dir / 'heldout_final_residual_vs_training_dataset_size.png'}")
    print(f"🖼️ p_n 残差图: {output_dir / 'pn_final_residual_vs_training_dataset_size.png'}")
    print(f"🖼️ 6 子图残差轨迹: {output_dir / 'heldout_mean_residual_trajectory_6_optimizer_settings.png'}")


if __name__ == "__main__":
    main()
