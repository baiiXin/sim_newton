"""
自由落体单帧变分问题：MLP 网络结构消融实验
=============================================

实验目的
--------
固定物理问题、训练集、优化器与训练策略，只改变 MLP 隐藏层深度和宽度，
比较不同网络结构作为迭代型求解器时的：

1. 训练收敛速度；
2. 训练域内连续插值能力；
3. 训练域边界稳定性；
4. 轻度域外泛化能力；
5. 不同空间方向上的收敛一致性；
6. 超过训练展开步数后的长期迭代稳定性。

正式实验默认设置
----------------
- 数值精度：torch.float64；
- 设备：cuda:1；
- 训练集：以精确解 y_star 为中心的 22^3 = 10,648 点规则网格；
- 训练范围：每维 [y_star - 0.01, y_star + 0.01]；
- 优化器：Adam(lr=1e-3)；
- 训练模式：Full-Batch；
- 总训练轮数：50,000 epoch；
- 每个 epoch：一次 backward，一次 optimizer.step；
- 展开步数：K 从 1 开始，每 10,000 epoch 增加 1，最高 K=5；
- 训练损失：K 个展开步骤的 Full-Batch 平均变分能量之和；
- 输入：逐特征归一化；
- 输出：网络原始输出乘 dt，得到位置增量；
- 最后一层零初始化，使初始网络严格输出 0。

默认网络结构（10 组）
---------------------
深度对照：1x32、2x32、3x32、5x32、10x32
宽度对照：2x32、2x64、2x128、2x256
浅层宽网络：1x128、1x256

验证集（固定并由所有结构复用）
----------------------------
- 2,048 个训练域内 Sobol 点；
- 1,024 个训练域边界 Sobol 点；
- 合计 3,072 点。

每 1,000 epoch 使用固定 K=5 进行验证。以完整验证集上的 residual p95
作为第一排序指标，以 residual median 作为第二排序指标，保存验证最优模型。

测试集（固定并由所有结构复用）
----------------------------
- 4,096 个训练域内 Sobol 点；
- 2,048 个训练域边界 Sobol 点；
- 4,096 个近域外 Sobol 点；
- 416 个结构化方向点（26 个方向 x 16 个距离）；
- 1 个原始物理初值 p_n。

最终分别测试：
- 第 50,000 epoch 的最终模型；
- 验证集最优模型。

记录步数默认包括：0、1、2、3、4、5、10、20、50。

运行示例
--------
正式运行：
    python free_fall_mlp_architecture_ablation.py

只运行部分结构：
    python free_fall_mlp_architecture_ablation.py \
        --architecture-names 2x32 3x32 5x32

快速 CPU 检查：
    python free_fall_mlp_architecture_ablation.py \
        --device cpu --architecture-names 1x32 \
        --epochs 2 --eval-interval 1 --points-per-axis 2 \
        --validation-interior-size 16 --validation-boundary-size 8 \
        --test-interior-size 16 --test-boundary-size 8 \
        --test-exterior-size 16 --structured-radii-count 2 \
        --final-test-steps 2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 0. 默认设置
# ============================================================

TORCH_DTYPE = torch.float64
TORCH_DTYPE_NAME = "torch.float64"
torch.set_default_dtype(TORCH_DTYPE)

PLOT_FLOOR = 1e-14
MODEL_RANDOM_SEED = 42
DATASET_RANDOM_SEED = 20260617

DEFAULT_DEVICE = "cuda:1"
DEFAULT_POINTS_PER_AXIS = 22
DEFAULT_SAMPLING_RADIUS = 0.01
DEFAULT_EPOCHS = 50_000
DEFAULT_EVAL_INTERVAL = 1_000
DEFAULT_VALIDATION_STEPS = 5
DEFAULT_FINAL_TEST_STEPS = 50
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 10_000
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_SUMMARY_CURVE_POINTS = 1_000

DEFAULT_VALIDATION_INTERIOR_SIZE = 2_048
DEFAULT_VALIDATION_BOUNDARY_SIZE = 1_024
DEFAULT_TEST_INTERIOR_SIZE = 4_096
DEFAULT_TEST_BOUNDARY_SIZE = 2_048
DEFAULT_TEST_EXTERIOR_SIZE = 4_096
DEFAULT_STRUCTURED_RADII_COUNT = 16

INTERIOR_BOUNDARY_START_RATIO = 0.8
EXTERIOR_RADIUS_MULTIPLIER = 2.0
SUCCESS_THRESHOLDS = (1e-4, 1e-6, 1e-8)

ARCHITECTURE_CONFIGS: list[dict] = [
    {"name": "1x32", "hidden_dims": [32]},
    {"name": "2x32", "hidden_dims": [32, 32]},
    {"name": "3x32", "hidden_dims": [32, 32, 32]},
    {"name": "5x32", "hidden_dims": [32] * 5},
    {"name": "10x32", "hidden_dims": [32] * 10},
    {"name": "2x64", "hidden_dims": [64, 64]},
    {"name": "1x128", "hidden_dims": [128]},
    {"name": "2x128", "hidden_dims": [128, 128]},
    {"name": "1x256", "hidden_dims": [256]},
    {"name": "2x256", "hidden_dims": [256, 256]},
]


# ============================================================
# 1. 数据结构和通用函数
# ============================================================


@dataclass(frozen=True)
class RuntimeConfig:
    architecture_names: list[str]
    points_per_axis: int
    sampling_radius: float
    epochs: int
    eval_interval: int
    validation_steps: int
    final_test_steps: int
    initial_k: int
    k_increase_interval: int
    k_increase_amount: int
    max_k: int
    learning_rate: float
    device: str
    validation_interior_size: int
    validation_boundary_size: int
    test_interior_size: int
    test_boundary_size: int
    test_exterior_size: int
    structured_radii_count: int
    skip_individual_plots: bool


@dataclass(frozen=True)
class PhysicsConfig:
    m: float = 1.0
    g: float = 9.8
    dt: float = 0.01


def create_output_directory() -> Path:
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent / script_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return make_json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(data), file, indent=2, ensure_ascii=False)


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().tolist()


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def is_model_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def finite_plot_value(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return float("nan")
    return max(float(value), PLOT_FLOOR)


def get_k_for_epoch_index(epoch_index: int, config: RuntimeConfig) -> int:
    """epoch_index 从 0 开始。前 10,000 个 epoch 使用 K=1。"""

    return min(
        config.initial_k
        + (epoch_index // config.k_increase_interval) * config.k_increase_amount,
        config.max_k,
    )


def downsample_log(log: Sequence[dict], max_points: int = DEFAULT_SUMMARY_CURVE_POINTS) -> list[dict]:
    if not log:
        return []
    if len(log) <= max_points:
        return [dict(item) for item in log]
    indices = np.linspace(0, len(log) - 1, num=max_points, dtype=int)
    indices = sorted(set(indices.tolist() + [len(log) - 1]))
    return [dict(log[index]) for index in indices]


def validate_device(device_string: str) -> torch.device:
    device = torch.device(device_string)
    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device {device_string!r}, but CUDA is not available. "
            "Use --device cpu for a CPU check."
        )

    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested device cuda:{index}, but only {torch.cuda.device_count()} CUDA "
            "device(s) are visible. Check CUDA_VISIBLE_DEVICES or change --device."
        )
    return torch.device(f"cuda:{index}")


# ============================================================
# 2. 物理问题与网络
# ============================================================


def variational_energy(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
) -> torch.Tensor:
    residual = y - p_n - physics.dt * v_n
    kinetic = (physics.m / (2.0 * physics.dt**2)) * torch.sum(residual**2, dim=-1)
    potential = physics.m * physics.g * y[..., 2]
    return kinetic + potential


def stationarity_residual(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
) -> torch.Tensor:
    residual = (physics.m / physics.dt**2) * (y - p_n - physics.dt * v_n)
    gravity = torch.zeros_like(residual)
    gravity[..., 2] = physics.m * physics.g
    return residual + gravity


def stationarity_residual_norm(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
) -> torch.Tensor:
    return torch.linalg.vector_norm(
        stationarity_residual(y, p_n, v_n, physics),
        dim=-1,
    )


def exact_solution(
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
) -> torch.Tensor:
    gravity = torch.tensor(
        [0.0, 0.0, physics.g],
        dtype=p_n.dtype,
        device=p_n.device,
    )
    return p_n + physics.dt * v_n - physics.dt**2 * gravity


def newton_direction(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
) -> torch.Tensor:
    gradient = stationarity_residual(y, p_n, v_n, physics)
    return -(physics.dt**2 / physics.m) * gradient


class MLPOptimizer(nn.Module):
    """可配置深度与宽度的学习型迭代优化器。"""

    def __init__(
        self,
        hidden_dims: Sequence[int],
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        use_input_normalization: bool = True,
        use_output_dt_scaling: bool = True,
    ) -> None:
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer.")
        if any(int(width) <= 0 for width in hidden_dims):
            raise ValueError("Every hidden layer width must be positive.")

        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.use_input_normalization = bool(use_input_normalization)
        self.use_output_dt_scaling = bool(use_output_dt_scaling)

        layers: list[nn.Module] = []
        input_dim = 12
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 3))
        self.net = nn.Sequential(*layers)

        final_linear = self.net[-1]
        if not isinstance(final_linear, nn.Linear):
            raise RuntimeError("The final network layer must be nn.Linear.")
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

        self.register_buffer("input_mean", input_mean.detach().clone())
        self.register_buffer("input_std", input_std.detach().clone())

    @staticmethod
    def _expand_batch(feature: torch.Tensor, batch_size: int) -> torch.Tensor:
        if feature.ndim == 1:
            return feature.unsqueeze(0).expand(batch_size, -1)
        if feature.ndim == 2 and feature.shape[0] == batch_size:
            return feature
        raise ValueError(
            f"Feature shape {tuple(feature.shape)} is incompatible with batch size {batch_size}."
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
            raise ValueError(f"Expected y with shape [3] or [B, 3], got {tuple(y.shape)}.")

        batch_size = y.shape[0]
        history_batch = self._expand_batch(history, batch_size)
        params_batch = self._expand_batch(params, batch_size)
        inp = torch.cat([y, history_batch, params_batch], dim=-1)
        if self.use_input_normalization:
            inp = (inp - self.input_mean) / self.input_std
        delta = self.net(inp)
        if self.use_output_dt_scaling:
            delta = params_batch[:, 2:3] * delta
        return delta


# ============================================================
# 3. 训练、验证与测试数据
# ============================================================


def create_regular_training_grid(
    y_star: torch.Tensor,
    points_per_axis: int,
    radius: float,
) -> torch.Tensor:
    if points_per_axis <= 0:
        raise ValueError("points_per_axis must be positive.")
    if points_per_axis % 2 != 0:
        raise ValueError(
            "points_per_axis must be even so that the symmetric grid excludes y_star."
        )
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    axis = torch.linspace(
        -radius,
        radius,
        steps=points_per_axis,
        dtype=TORCH_DTYPE,
        device=y_star.device,
    )
    offsets = torch.cartesian_prod(axis, axis, axis)
    return y_star.unsqueeze(0) + offsets


def compute_training_input_normalizer(
    points_per_axis: int,
    radius: float,
    y_star: torch.Tensor,
    history: torch.Tensor,
    params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    y_std_value = radius * math.sqrt(
        (points_per_axis + 1.0) / (3.0 * (points_per_axis - 1.0))
    )
    input_mean = torch.cat([y_star, history, params], dim=0)
    input_std = torch.cat(
        [
            torch.full((3,), y_std_value, dtype=TORCH_DTYPE, device=y_star.device),
            torch.ones(9, dtype=TORCH_DTYPE, device=y_star.device),
        ],
        dim=0,
    )
    return input_mean, input_std


def sobol_cube_offsets(
    num_points: int,
    radius: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if num_points <= 0:
        return torch.empty((0, 3), dtype=TORCH_DTYPE, device=device)
    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    unit = engine.draw(num_points, dtype=TORCH_DTYPE)
    offsets = (2.0 * unit - 1.0) * radius
    return offsets.to(device)


def sobol_shell_offsets(
    num_points: int,
    inner_inf_radius: float,
    outer_inf_radius: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """从 L_inf 壳层 inner <= ||offset||_inf <= outer 中取固定数量 Sobol 点。"""

    if num_points <= 0:
        return torch.empty((0, 3), dtype=TORCH_DTYPE, device=device)
    if not (0.0 <= inner_inf_radius < outer_inf_radius):
        raise ValueError("Require 0 <= inner_inf_radius < outer_inf_radius.")

    engine = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=seed)
    selected: list[torch.Tensor] = []
    selected_count = 0

    # 边界壳层占总体积比例可能较低，因此循环分批拒绝采样。
    batch_size = max(1_024, num_points * 2)
    while selected_count < num_points:
        unit = engine.draw(batch_size, dtype=TORCH_DTYPE)
        offsets = (2.0 * unit - 1.0) * outer_inf_radius
        max_abs = offsets.abs().amax(dim=1)
        mask = (max_abs >= inner_inf_radius) & (max_abs <= outer_inf_radius)
        accepted = offsets[mask]
        if accepted.numel() == 0:
            batch_size *= 2
            continue
        selected.append(accepted)
        selected_count += accepted.shape[0]

    result = torch.cat(selected, dim=0)[:num_points]
    return result.to(device)


def create_structured_direction_offsets(
    radii_count: int,
    min_radius: float,
    max_radius: float,
    device: torch.device,
) -> torch.Tensor:
    if radii_count <= 0:
        raise ValueError("radii_count must be positive.")
    if not (0.0 < min_radius <= max_radius):
        raise ValueError("Require 0 < min_radius <= max_radius.")

    directions = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                if x == 0.0 and y == 0.0 and z == 0.0:
                    continue
                direction = torch.tensor([x, y, z], dtype=TORCH_DTYPE)
                direction = direction / torch.linalg.vector_norm(direction)
                directions.append(direction)

    direction_tensor = torch.stack(directions, dim=0)
    radii = torch.linspace(
        min_radius,
        max_radius,
        steps=radii_count,
        dtype=TORCH_DTYPE,
    )
    offsets = direction_tensor[:, None, :] * radii[None, :, None]
    return offsets.reshape(-1, 3).to(device)


def build_evaluation_datasets(
    y_star: torch.Tensor,
    p_n: torch.Tensor,
    config: RuntimeConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    radius = config.sampling_radius
    seed = DATASET_RANDOM_SEED
    device = y_star.device

    validation_sets = {
        "interior": y_star + sobol_cube_offsets(
            config.validation_interior_size,
            radius,
            seed + 1,
            device,
        ),
        "boundary": y_star + sobol_shell_offsets(
            config.validation_boundary_size,
            INTERIOR_BOUNDARY_START_RATIO * radius,
            radius,
            seed + 2,
            device,
        ),
    }

    test_sets = {
        "interior": y_star + sobol_cube_offsets(
            config.test_interior_size,
            radius,
            seed + 101,
            device,
        ),
        "boundary": y_star + sobol_shell_offsets(
            config.test_boundary_size,
            INTERIOR_BOUNDARY_START_RATIO * radius,
            radius,
            seed + 102,
            device,
        ),
        "near_exterior": y_star + sobol_shell_offsets(
            config.test_exterior_size,
            radius,
            EXTERIOR_RADIUS_MULTIPLIER * radius,
            seed + 103,
            device,
        ),
        "structured_directions": y_star + create_structured_direction_offsets(
            radii_count=config.structured_radii_count,
            min_radius=0.001,
            max_radius=EXTERIOR_RADIUS_MULTIPLIER * radius,
            device=device,
        ),
        "p_n": p_n.reshape(1, 3).clone(),
    }
    return validation_sets, test_sets


def combine_datasets(datasets: dict[str, torch.Tensor]) -> torch.Tensor:
    if not datasets:
        raise ValueError("datasets must not be empty.")
    return torch.cat(list(datasets.values()), dim=0)


def describe_datasets(
    validation_sets: dict[str, torch.Tensor],
    test_sets: dict[str, torch.Tensor],
    y_star: torch.Tensor,
) -> dict:
    def describe_group(group: dict[str, torch.Tensor]) -> dict:
        result = {}
        for name, points in group.items():
            offsets = points - y_star
            result[name] = {
                "num_points": int(points.shape[0]),
                "min_l2_distance_to_y_star": float(
                    torch.linalg.vector_norm(offsets, dim=1).min().item()
                ),
                "max_l2_distance_to_y_star": float(
                    torch.linalg.vector_norm(offsets, dim=1).max().item()
                ),
                "min_linf_distance_to_y_star": float(offsets.abs().amax(dim=1).min().item()),
                "max_linf_distance_to_y_star": float(offsets.abs().amax(dim=1).max().item()),
            }
        return result

    return {
        "validation": describe_group(validation_sets),
        "test": describe_group(test_sets),
    }


# ============================================================
# 4. 指标与批量展开
# ============================================================


def finite_statistics(values: torch.Tensor) -> dict:
    flat = values.detach().reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite_values = flat[finite_mask]
    finite_count = int(finite_mask.sum().item())
    total_count = int(flat.numel())

    result = {
        "count": total_count,
        "finite_count": finite_count,
        "finite_ratio": finite_count / total_count if total_count > 0 else 0.0,
        "mean": None,
        "median": None,
        "p95": None,
        "max": None,
        "min": None,
    }
    if finite_count == 0:
        return result

    result.update(
        {
            "mean": float(finite_values.mean().item()),
            "median": float(torch.quantile(finite_values, 0.50).item()),
            "p95": float(torch.quantile(finite_values, 0.95).item()),
            "max": float(finite_values.max().item()),
            "min": float(finite_values.min().item()),
        }
    )
    return result


def compute_state_metrics(
    y: torch.Tensor,
    y_star: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
    e_star: float,
    previous_residual: torch.Tensor | None,
    initial_residual: torch.Tensor,
) -> tuple[dict, torch.Tensor]:
    finite_rows = torch.isfinite(y).all(dim=1)

    position_error = torch.linalg.vector_norm(y - y_star, dim=1)
    residual_norm = stationarity_residual_norm(y, p_n, v_n, physics)
    energy_gap = variational_energy(y, p_n, v_n, physics) - e_star

    metrics = {
        "num_points": int(y.shape[0]),
        "finite_state_ratio": float(finite_rows.float().mean().item()),
        "position_error": finite_statistics(position_error),
        "residual_norm": finite_statistics(residual_norm),
        "energy_gap": finite_statistics(energy_gap),
        "success_rates": {
            f"position_error_lt_{threshold:.0e}": float(
                ((position_error < threshold) & torch.isfinite(position_error)).float().mean().item()
            )
            for threshold in SUCCESS_THRESHOLDS
        },
        "residual_increased_vs_initial_fraction": float(
            (
                (residual_norm > initial_residual)
                & torch.isfinite(residual_norm)
                & torch.isfinite(initial_residual)
            ).float().mean().item()
        ),
        "residual_increased_vs_previous_fraction": None,
    }

    if previous_residual is not None:
        metrics["residual_increased_vs_previous_fraction"] = float(
            (
                (residual_norm > previous_residual)
                & torch.isfinite(residual_norm)
                & torch.isfinite(previous_residual)
            ).float().mean().item()
        )

    return metrics, residual_norm


def rollout_dataset(
    model: MLPOptimizer,
    initial_points: torch.Tensor,
    history: torch.Tensor,
    params: torch.Tensor,
    y_star: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
    selected_steps: Sequence[int],
) -> dict:
    selected = sorted(set(int(step) for step in selected_steps if int(step) >= 0))
    if not selected:
        raise ValueError("selected_steps must contain at least one non-negative step.")

    max_step = max(selected)
    y = initial_points.clone()
    e_star = float(variational_energy(y_star, p_n, v_n, physics).item())
    initial_residual = stationarity_residual_norm(y, p_n, v_n, physics)
    previous_residual: torch.Tensor | None = None
    step_metrics: dict[str, dict] = {}

    model.eval()
    with torch.no_grad():
        for step in range(max_step + 1):
            if step in selected:
                metrics, _ = compute_state_metrics(
                    y=y,
                    y_star=y_star,
                    p_n=p_n,
                    v_n=v_n,
                    physics=physics,
                    e_star=e_star,
                    previous_residual=previous_residual,
                    initial_residual=initial_residual,
                )
                step_metrics[str(step)] = metrics

            if step == max_step:
                break

            previous_residual = stationarity_residual_norm(y, p_n, v_n, physics)
            y = y + model(y, history, params)

    return {
        "selected_steps": selected,
        "metrics_by_step": step_metrics,
    }


def rollout_point_trajectory(
    model: MLPOptimizer,
    initial_point: torch.Tensor,
    history: torch.Tensor,
    params: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
    num_steps: int,
) -> dict:
    y_star = exact_solution(p_n, v_n, physics)
    e_star = float(variational_energy(y_star, p_n, v_n, physics).item())
    y = initial_point.clone()
    trajectory = []

    model.eval()
    with torch.no_grad():
        for step in range(num_steps + 1):
            trajectory.append(
                {
                    "step": step,
                    "y": tensor_to_list(y),
                    "position_error": float(torch.linalg.vector_norm(y - y_star).item()),
                    "residual_norm": float(
                        stationarity_residual_norm(y, p_n, v_n, physics).item()
                    ),
                    "energy_gap": float(
                        variational_energy(y, p_n, v_n, physics).item() - e_star
                    ),
                }
            )
            if step < num_steps:
                y = y + model(y, history, params)

    return {"initial_point": tensor_to_list(initial_point), "trajectory": trajectory}


def newton_point_trajectory(
    initial_point: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
    num_steps: int,
) -> dict:
    y_star = exact_solution(p_n, v_n, physics)
    e_star = float(variational_energy(y_star, p_n, v_n, physics).item())
    y = initial_point.clone()
    trajectory = []

    with torch.no_grad():
        for step in range(num_steps + 1):
            trajectory.append(
                {
                    "step": step,
                    "y": tensor_to_list(y),
                    "position_error": float(torch.linalg.vector_norm(y - y_star).item()),
                    "residual_norm": float(
                        stationarity_residual_norm(y, p_n, v_n, physics).item()
                    ),
                    "energy_gap": float(
                        variational_energy(y, p_n, v_n, physics).item() - e_star
                    ),
                }
            )
            if step < num_steps:
                y = y + newton_direction(y, p_n, v_n, physics)

    return {"initial_point": tensor_to_list(initial_point), "trajectory": trajectory}


def evaluate_validation(
    model: MLPOptimizer,
    validation_sets: dict[str, torch.Tensor],
    history: torch.Tensor,
    params: torch.Tensor,
    y_star: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
    steps: int,
) -> dict:
    combined = combine_datasets(validation_sets)
    subset_results = {}
    for name, points in validation_sets.items():
        subset_results[name] = rollout_dataset(
            model=model,
            initial_points=points,
            history=history,
            params=params,
            y_star=y_star,
            p_n=p_n,
            v_n=v_n,
            physics=physics,
            selected_steps=[steps],
        )["metrics_by_step"][str(steps)]

    overall = rollout_dataset(
        model=model,
        initial_points=combined,
        history=history,
        params=params,
        y_star=y_star,
        p_n=p_n,
        v_n=v_n,
        physics=physics,
        selected_steps=[steps],
    )["metrics_by_step"][str(steps)]

    residual_stats = overall["residual_norm"]
    valid_for_selection = (
        overall["finite_state_ratio"] == 1.0
        and residual_stats["p95"] is not None
        and residual_stats["median"] is not None
        and math.isfinite(float(residual_stats["p95"]))
        and math.isfinite(float(residual_stats["median"]))
    )

    return {
        "steps": steps,
        "overall": overall,
        "subsets": subset_results,
        "selection_key": (
            [float(residual_stats["p95"]), float(residual_stats["median"])]
            if valid_for_selection
            else None
        ),
        "valid_for_selection": valid_for_selection,
    }


def evaluate_test_sets(
    model: MLPOptimizer,
    test_sets: dict[str, torch.Tensor],
    history: torch.Tensor,
    params: torch.Tensor,
    y_star: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
    selected_steps: Sequence[int],
) -> dict:
    return {
        name: rollout_dataset(
            model=model,
            initial_points=points,
            history=history,
            params=params,
            y_star=y_star,
            p_n=p_n,
            v_n=v_n,
            physics=physics,
            selected_steps=selected_steps,
        )
        for name, points in test_sets.items()
    }


# ============================================================
# 5. 绘图
# ============================================================


def plot_training_and_validation_curves(
    train_log: Sequence[dict],
    validation_log: Sequence[dict],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = [item["epoch"] for item in train_log]
    train_gaps = [finite_plot_value(item["training_gap_for_readability"]) for item in train_log]
    axes[0].plot(epochs, train_gaps)
    axes[0].set_yscale("log")
    axes[0].set_title("Full-Batch Training Energy-Sum Gap")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"$\sum_k [E(y^{(k)})-E(y^*)]$")
    axes[0].grid(True, alpha=0.3)

    validation_epochs = [item["epoch"] for item in validation_log]
    p95 = [
        finite_plot_value(item["validation"]["overall"]["residual_norm"]["p95"])
        for item in validation_log
    ]
    median = [
        finite_plot_value(item["validation"]["overall"]["residual_norm"]["median"])
        for item in validation_log
    ]
    axes[1].plot(validation_epochs, p95, marker="o", label="Residual p95")
    axes[1].plot(validation_epochs, median, marker="s", label="Residual median")
    axes[1].set_yscale("log")
    axes[1].set_title("Validation Residual after 5 Steps")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"$\|\nabla E(y)\|_2$")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    interior = [
        finite_plot_value(
            item["validation"]["subsets"]["interior"]["residual_norm"]["p95"]
        )
        for item in validation_log
    ]
    boundary = [
        finite_plot_value(
            item["validation"]["subsets"]["boundary"]["residual_norm"]["p95"]
        )
        for item in validation_log
    ]
    axes[2].plot(validation_epochs, interior, marker="o", label="Interior p95")
    axes[2].plot(validation_epochs, boundary, marker="s", label="Boundary p95")
    axes[2].set_yscale("log")
    axes[2].set_title("Validation Subset Residual p95")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel(r"$\|\nabla E(y)\|_2$")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pn_trajectory_comparison(
    final_trajectory: dict,
    best_trajectory: dict,
    newton_trajectory: dict,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for trajectory, label, marker in (
        (final_trajectory, "Final epoch model", "o"),
        (best_trajectory, "Best validation model", "s"),
        (newton_trajectory, "Newton", "^"),
    ):
        items = trajectory["trajectory"]
        steps = [item["step"] for item in items]
        residuals = [finite_plot_value(item["residual_norm"]) for item in items]
        errors = [finite_plot_value(item["position_error"]) for item in items]
        axes[0].plot(steps, residuals, marker=marker, markersize=3, label=label)
        axes[1].plot(steps, errors, marker=marker, markersize=3, label=label)

    axes[0].set_yscale("log")
    axes[0].set_title(r"Residual from Held-Out $p_n$")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\|\nabla E(y)\|_2$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_yscale("log")
    axes[1].set_title(r"Position Error from Held-Out $p_n$")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\|y-y^*\|_2$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_test_subset_curves(
    test_results: dict,
    save_path: Path,
) -> None:
    subset_names = ["interior", "boundary", "near_exterior", "structured_directions"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.reshape(-1)

    for ax, subset_name in zip(axes_flat, subset_names):
        subset = test_results[subset_name]
        steps = subset["selected_steps"]
        medians = [
            finite_plot_value(
                subset["metrics_by_step"][str(step)]["residual_norm"]["median"]
            )
            for step in steps
        ]
        p95s = [
            finite_plot_value(
                subset["metrics_by_step"][str(step)]["residual_norm"]["p95"]
            )
            for step in steps
        ]
        maxima = [
            finite_plot_value(
                subset["metrics_by_step"][str(step)]["residual_norm"]["max"]
            )
            for step in steps
        ]
        ax.plot(steps, medians, marker="o", label="median")
        ax.plot(steps, p95s, marker="s", label="p95")
        ax.plot(steps, maxima, marker="^", label="max")
        ax.set_yscale("log")
        ax.set_title(subset_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Residual $\|\nabla E(y)\|_2$")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_training_summary(
    summaries: Sequence[dict],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for summary in summaries:
        name = summary["architecture_name"]
        train_curve = summary["training_curve_for_summary"]
        validation_curve = summary["validation_curve_for_summary"]

        if train_curve:
            axes[0].plot(
                [item["epoch"] for item in train_curve],
                [finite_plot_value(item["training_gap_for_readability"]) for item in train_curve],
                label=name,
            )
        if validation_curve:
            axes[1].plot(
                [item["epoch"] for item in validation_curve],
                [
                    finite_plot_value(
                        item["validation"]["overall"]["residual_norm"]["p95"]
                    )
                    for item in validation_curve
                ],
                marker="o",
                markersize=3,
                label=name,
            )

    axes[0].set_yscale("log")
    axes[0].set_title("Training Gap by Architecture")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"$\sum_k [E(y^{(k)})-E(y^*)]$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_yscale("log")
    axes[1].set_title("Validation Residual p95 by Architecture")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"$\|\nabla E(y)\|_2$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_final_summary(
    summaries: Sequence[dict],
    final_step: int,
    save_path: Path,
) -> None:
    subset_names = ["interior", "boundary", "near_exterior", "structured_directions"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes_flat = axes.reshape(-1)

    names = [summary["architecture_name"] for summary in summaries]
    x = np.arange(len(names))

    for ax, subset_name in zip(axes_flat, subset_names):
        best_values = []
        final_values = []
        for summary in summaries:
            best_metrics = summary["best_model_test_summary"][subset_name]["metrics_by_step"][
                str(final_step)
            ]
            final_metrics = summary["final_model_test_summary"][subset_name]["metrics_by_step"][
                str(final_step)
            ]
            best_values.append(finite_plot_value(best_metrics["residual_norm"]["p95"]))
            final_values.append(finite_plot_value(final_metrics["residual_norm"]["p95"]))

        width = 0.38
        ax.bar(x - width / 2, best_values, width=width, label="Best validation model")
        ax.bar(x + width / 2, final_values, width=width, label="Final epoch model")
        ax.set_yscale("log")
        ax.set_title(f"{subset_name}: residual p95 at step {final_step}")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel(r"$\|\nabla E(y)\|_2$")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_performance_tradeoff(
    summaries: Sequence[dict],
    final_step: int,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    for summary in summaries:
        params = summary["num_trainable_parameters"]
        metric = summary["best_model_test_summary"]["near_exterior"]["metrics_by_step"][
            str(final_step)
        ]["residual_norm"]["p95"]
        ax.scatter(params, finite_plot_value(metric), s=70)
        ax.annotate(
            summary["architecture_name"],
            (params, finite_plot_value(metric)),
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Parameter Count vs. Near-Exterior Residual p95")
    ax.set_xlabel("Number of trainable parameters")
    ax.set_ylabel(r"Residual p95 at final step")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. 单个网络结构实验
# ============================================================


def run_architecture_experiment(
    architecture: dict,
    base_output_dir: Path,
    config: RuntimeConfig,
    training_grid: torch.Tensor,
    validation_sets: dict[str, torch.Tensor],
    test_sets: dict[str, torch.Tensor],
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    physics: PhysicsConfig,
) -> dict:
    architecture_name = architecture["name"]
    hidden_dims = architecture["hidden_dims"]
    output_dir = base_output_dir / architecture_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = training_grid.device
    y_star = exact_solution(p_n, v_n, physics)
    history = torch.cat([p_n, v_n], dim=0)
    params = torch.tensor(
        [physics.m, physics.g, physics.dt],
        dtype=TORCH_DTYPE,
        device=device,
    )
    e_star = float(variational_energy(y_star, p_n, v_n, physics).item())

    input_mean, input_std = compute_training_input_normalizer(
        points_per_axis=config.points_per_axis,
        radius=config.sampling_radius,
        y_star=y_star,
        history=history,
        params=params,
    )

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)

    model = MLPOptimizer(
        hidden_dims=hidden_dims,
        input_mean=input_mean,
        input_std=input_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    num_parameters = count_trainable_parameters(model)

    print("\n" + "=" * 90)
    print(f"网络结构: {architecture_name} | hidden_dims={hidden_dims}")
    print(f"参数量: {num_parameters:,}")
    print(f"输出目录: {output_dir}")
    print("=" * 90)

    train_log: list[dict] = []
    validation_log: list[dict] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    best_validation: dict | None = None
    best_selection_key: tuple[float, float] | None = None

    diverged = False
    divergence_epoch: int | None = None
    divergence_reason: str | None = None
    start_time = time.perf_counter()

    for epoch_index in range(config.epochs):
        epoch_number = epoch_index + 1
        k = get_k_for_epoch_index(epoch_index, config)
        model.train()
        optimizer.zero_grad(set_to_none=True)

        y = training_grid
        trajectory_loss = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        for _ in range(k):
            y = y + model(y, history, params)
            trajectory_loss = trajectory_loss + variational_energy(
                y, p_n, v_n, physics
            ).mean()

        if not bool(torch.isfinite(trajectory_loss)):
            diverged = True
            divergence_epoch = epoch_number
            divergence_reason = "non-finite full-batch trajectory loss"
        else:
            try:
                trajectory_loss.backward()
                optimizer.step()
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    diverged = True
                    divergence_epoch = epoch_number
                    divergence_reason = f"CUDA out of memory: {error}"
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

        if not diverged and not is_model_finite(model):
            diverged = True
            divergence_epoch = epoch_number
            divergence_reason = "non-finite model parameter after optimizer.step"

        if diverged:
            print(
                f"训练终止：epoch={divergence_epoch}, reason={divergence_reason}"
            )
            break

        trajectory_loss_value = float(trajectory_loss.item())
        training_gap = trajectory_loss_value - k * e_star
        train_log.append(
            {
                "epoch": epoch_number,
                "K": k,
                "trajectory_energy_sum": trajectory_loss_value,
                "training_gap_for_readability": training_gap,
            }
        )

        should_validate = (
            epoch_number % config.eval_interval == 0
            or epoch_number == config.epochs
        )
        if should_validate:
            validation = evaluate_validation(
                model=model,
                validation_sets=validation_sets,
                history=history,
                params=params,
                y_star=y_star,
                p_n=p_n,
                v_n=v_n,
                physics=physics,
                steps=config.validation_steps,
            )
            validation_log.append(
                {
                    "epoch": epoch_number,
                    "training_K": k,
                    "validation": validation,
                }
            )

            selection_key_list = validation["selection_key"]
            if selection_key_list is not None:
                selection_key = (selection_key_list[0], selection_key_list[1])
                if best_selection_key is None or selection_key < best_selection_key:
                    best_selection_key = selection_key
                    best_epoch = epoch_number
                    best_validation = copy.deepcopy(validation)
                    best_state_dict = state_dict_to_cpu(model)

            overall_residual = validation["overall"]["residual_norm"]
            elapsed = time.perf_counter() - start_time
            print(
                f"Epoch {epoch_number:5d} | K={k} | "
                f"train_gap={training_gap:.4e} | "
                f"val_residual_median={overall_residual['median']} | "
                f"val_residual_p95={overall_residual['p95']} | "
                f"best_epoch={best_epoch} | elapsed={elapsed:.1f}s"
            )

    if best_state_dict is None:
        # 极端情况下所有验证均无效，仍保存最终模型作为占位，但明确记录。
        best_state_dict = state_dict_to_cpu(model)
        best_epoch = train_log[-1]["epoch"] if train_log else 0
        best_validation = None

    final_state_dict = state_dict_to_cpu(model)
    torch.save(final_state_dict, output_dir / "final_epoch_model_state_dict.pt")
    torch.save(best_state_dict, output_dir / "best_validation_model_state_dict.pt")

    selected_steps = sorted(
        set(
            step
            for step in [0, 1, 2, 3, 4, 5, 10, 20, config.final_test_steps]
            if step <= config.final_test_steps
        )
    )

    # 最终模型测试。
    model.load_state_dict(final_state_dict)
    model.to(device)
    final_model_test = evaluate_test_sets(
        model=model,
        test_sets=test_sets,
        history=history,
        params=params,
        y_star=y_star,
        p_n=p_n,
        v_n=v_n,
        physics=physics,
        selected_steps=selected_steps,
    )
    final_pn_trajectory = rollout_point_trajectory(
        model=model,
        initial_point=p_n,
        history=history,
        params=params,
        p_n=p_n,
        v_n=v_n,
        physics=physics,
        num_steps=config.final_test_steps,
    )

    # 验证最优模型测试。
    model.load_state_dict(best_state_dict)
    model.to(device)
    best_model_test = evaluate_test_sets(
        model=model,
        test_sets=test_sets,
        history=history,
        params=params,
        y_star=y_star,
        p_n=p_n,
        v_n=v_n,
        physics=physics,
        selected_steps=selected_steps,
    )
    best_pn_trajectory = rollout_point_trajectory(
        model=model,
        initial_point=p_n,
        history=history,
        params=params,
        p_n=p_n,
        v_n=v_n,
        physics=physics,
        num_steps=config.final_test_steps,
    )
    newton_trajectory = newton_point_trajectory(
        initial_point=p_n,
        p_n=p_n,
        v_n=v_n,
        physics=physics,
        num_steps=config.final_test_steps,
    )

    experiment_report = {
        "architecture": {
            "name": architecture_name,
            "hidden_dims": hidden_dims,
            "num_hidden_layers": len(hidden_dims),
            "num_trainable_parameters": num_parameters,
        },
        "training_configuration": {
            "torch_dtype": TORCH_DTYPE_NAME,
            "device": str(device),
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
            "training_mode": "full_batch",
            "training_dataset_size": int(training_grid.shape[0]),
            "points_per_axis": config.points_per_axis,
            "sampling_radius": config.sampling_radius,
            "epochs_requested": config.epochs,
            "initial_K": config.initial_k,
            "K_increase_interval": config.k_increase_interval,
            "K_increase_amount": config.k_increase_amount,
            "max_K": config.max_k,
            "loss": "sum of stepwise mean variational energy over the full batch",
            "backpropagation": "full unroll without detach; one backward and one optimizer step per epoch",
            "input_normalization": True,
            "output_dt_scaling": True,
            "final_layer_zero_initialization": True,
            "model_random_seed": MODEL_RANDOM_SEED,
            "input_mean": tensor_to_list(input_mean),
            "input_std": tensor_to_list(input_std),
        },
        "physics": {
            **asdict(physics),
            "p_n": tensor_to_list(p_n),
            "v_n": tensor_to_list(v_n),
            "y_star": tensor_to_list(y_star),
            "E_star": e_star,
        },
        "training_status": {
            "diverged": diverged,
            "divergence_epoch": divergence_epoch,
            "divergence_reason": divergence_reason,
            "completed_epochs": len(train_log),
            "elapsed_seconds": time.perf_counter() - start_time,
        },
        "best_validation_checkpoint": {
            "epoch": best_epoch,
            "selection_rule": "minimize validation residual p95 after 5 steps, then residual median",
            "selection_key": list(best_selection_key) if best_selection_key is not None else None,
            "validation": best_validation,
        },
        "train_log": train_log,
        "validation_log": validation_log,
        "final_model_test": final_model_test,
        "best_model_test": best_model_test,
        "p_n_trajectories": {
            "final_model": final_pn_trajectory,
            "best_validation_model": best_pn_trajectory,
            "newton": newton_trajectory,
        },
    }
    save_json(experiment_report, output_dir / "architecture_experiment_report.json")

    if not config.skip_individual_plots:
        plot_training_and_validation_curves(
            train_log=train_log,
            validation_log=validation_log,
            save_path=output_dir / "training_and_validation_curves.png",
        )
        plot_pn_trajectory_comparison(
            final_trajectory=final_pn_trajectory,
            best_trajectory=best_pn_trajectory,
            newton_trajectory=newton_trajectory,
            save_path=output_dir / "p_n_trajectory_comparison.png",
        )
        plot_test_subset_curves(
            test_results=best_model_test,
            save_path=output_dir / "best_model_test_subset_residual_curves.png",
        )
        plot_test_subset_curves(
            test_results=final_model_test,
            save_path=output_dir / "final_model_test_subset_residual_curves.png",
        )

    final_step_key = str(config.final_test_steps)
    print(
        f"完成 {architecture_name}: best_epoch={best_epoch}, "
        f"best interior p95={best_model_test['interior']['metrics_by_step'][final_step_key]['residual_norm']['p95']}, "
        f"best exterior p95={best_model_test['near_exterior']['metrics_by_step'][final_step_key]['residual_norm']['p95']}"
    )

    return {
        "architecture_name": architecture_name,
        "hidden_dims": hidden_dims,
        "num_hidden_layers": len(hidden_dims),
        "num_trainable_parameters": num_parameters,
        "diverged": diverged,
        "divergence_epoch": divergence_epoch,
        "divergence_reason": divergence_reason,
        "completed_epochs": len(train_log),
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": (
            list(best_selection_key) if best_selection_key is not None else None
        ),
        "training_curve_for_summary": downsample_log(train_log),
        "validation_curve_for_summary": downsample_log(validation_log),
        "final_model_test_summary": final_model_test,
        "best_model_test_summary": best_model_test,
        "output_directory": str(output_dir),
    }


# ============================================================
# 7. 参数解析与主程序
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Free-fall MLP architecture ablation with fixed train/validation/test sets."
    )
    parser.add_argument(
        "--architecture-names",
        nargs="+",
        default=[item["name"] for item in ARCHITECTURE_CONFIGS],
        help="Architecture names to run. Available: "
        + ", ".join(item["name"] for item in ARCHITECTURE_CONFIGS),
    )
    parser.add_argument("--points-per-axis", type=int, default=DEFAULT_POINTS_PER_AXIS)
    parser.add_argument("--sampling-radius", type=float, default=DEFAULT_SAMPLING_RADIUS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--validation-steps", type=int, default=DEFAULT_VALIDATION_STEPS)
    parser.add_argument("--final-test-steps", type=int, default=DEFAULT_FINAL_TEST_STEPS)
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument(
        "--k-increase-interval",
        type=int,
        default=DEFAULT_K_INCREASE_INTERVAL,
    )
    parser.add_argument(
        "--k-increase-amount",
        type=int,
        default=DEFAULT_K_INCREASE_AMOUNT,
    )
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--validation-interior-size",
        type=int,
        default=DEFAULT_VALIDATION_INTERIOR_SIZE,
    )
    parser.add_argument(
        "--validation-boundary-size",
        type=int,
        default=DEFAULT_VALIDATION_BOUNDARY_SIZE,
    )
    parser.add_argument(
        "--test-interior-size",
        type=int,
        default=DEFAULT_TEST_INTERIOR_SIZE,
    )
    parser.add_argument(
        "--test-boundary-size",
        type=int,
        default=DEFAULT_TEST_BOUNDARY_SIZE,
    )
    parser.add_argument(
        "--test-exterior-size",
        type=int,
        default=DEFAULT_TEST_EXTERIOR_SIZE,
    )
    parser.add_argument(
        "--structured-radii-count",
        type=int,
        default=DEFAULT_STRUCTURED_RADII_COUNT,
    )
    parser.add_argument(
        "--skip-individual-plots",
        action="store_true",
        help="Skip per-architecture plots; JSON and global plots are still generated.",
    )
    return parser.parse_args()


def positive_int(value: int, name: str) -> int:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    available_names = {item["name"] for item in ARCHITECTURE_CONFIGS}
    requested_names = list(dict.fromkeys(args.architecture_names))
    unknown_names = [name for name in requested_names if name not in available_names]
    if unknown_names:
        raise ValueError(
            f"Unknown architecture names: {unknown_names}. Available: {sorted(available_names)}"
        )

    points_per_axis = positive_int(args.points_per_axis, "points_per_axis")
    if points_per_axis % 2 != 0:
        raise ValueError("points_per_axis must be even.")
    if args.sampling_radius <= 0.0:
        raise ValueError("sampling_radius must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    config = RuntimeConfig(
        architecture_names=requested_names,
        points_per_axis=points_per_axis,
        sampling_radius=float(args.sampling_radius),
        epochs=positive_int(args.epochs, "epochs"),
        eval_interval=positive_int(args.eval_interval, "eval_interval"),
        validation_steps=positive_int(args.validation_steps, "validation_steps"),
        final_test_steps=positive_int(args.final_test_steps, "final_test_steps"),
        initial_k=positive_int(args.initial_k, "initial_k"),
        k_increase_interval=positive_int(
            args.k_increase_interval, "k_increase_interval"
        ),
        k_increase_amount=positive_int(
            args.k_increase_amount, "k_increase_amount"
        ),
        max_k=positive_int(args.max_k, "max_k"),
        learning_rate=float(args.learning_rate),
        device=str(args.device),
        validation_interior_size=positive_int(
            args.validation_interior_size, "validation_interior_size"
        ),
        validation_boundary_size=positive_int(
            args.validation_boundary_size, "validation_boundary_size"
        ),
        test_interior_size=positive_int(args.test_interior_size, "test_interior_size"),
        test_boundary_size=positive_int(args.test_boundary_size, "test_boundary_size"),
        test_exterior_size=positive_int(args.test_exterior_size, "test_exterior_size"),
        structured_radii_count=positive_int(
            args.structured_radii_count, "structured_radii_count"
        ),
        skip_individual_plots=bool(args.skip_individual_plots),
    )
    if config.max_k < config.initial_k:
        raise ValueError("max_k must be greater than or equal to initial_k.")
    if config.final_test_steps < config.validation_steps:
        raise ValueError("final_test_steps must be at least validation_steps.")
    return config


def select_architectures(names: Iterable[str]) -> list[dict]:
    by_name = {item["name"]: item for item in ARCHITECTURE_CONFIGS}
    return [copy.deepcopy(by_name[name]) for name in names]


def main() -> None:
    config = validate_args(parse_args())
    device = validate_device(config.device)
    physics = PhysicsConfig()
    output_dir = create_output_directory()

    p_n = torch.tensor([3.0, 4.0, 5.0], dtype=TORCH_DTYPE, device=device)
    v_n = torch.tensor([0.5, -0.5, 0.0], dtype=TORCH_DTYPE, device=device)
    y_star = exact_solution(p_n, v_n, physics)

    training_grid = create_regular_training_grid(
        y_star=y_star,
        points_per_axis=config.points_per_axis,
        radius=config.sampling_radius,
    )
    validation_sets, test_sets = build_evaluation_datasets(
        y_star=y_star,
        p_n=p_n,
        config=config,
    )
    dataset_description = describe_datasets(
        validation_sets=validation_sets,
        test_sets=test_sets,
        y_star=y_star,
    )

    print("=" * 90)
    print("自由落体 MLP 网络结构消融实验")
    print(f"设备: {device}")
    print(f"精度: {TORCH_DTYPE_NAME}")
    print(f"输出目录: {output_dir}")
    print(f"训练集: {config.points_per_axis}^3 = {training_grid.shape[0]:,} 点")
    print(f"训练集显存: {training_grid.numel() * training_grid.element_size() / 1024**2:.3f} MiB")
    print(f"验证集: {sum(points.shape[0] for points in validation_sets.values()):,} 点")
    print(f"测试集: {sum(points.shape[0] for points in test_sets.values()):,} 点")
    print(f"网络结构: {config.architecture_names}")
    print("=" * 90)

    shared_config_report = {
        "runtime_config": asdict(config),
        "torch_dtype": TORCH_DTYPE_NAME,
        "device": str(device),
        "physics": {
            **asdict(physics),
            "p_n": tensor_to_list(p_n),
            "v_n": tensor_to_list(v_n),
            "y_star": tensor_to_list(y_star),
        },
        "training_dataset": {
            "type": "regular Cartesian grid centered at y_star",
            "points_per_axis": config.points_per_axis,
            "num_points": int(training_grid.shape[0]),
            "sampling_radius_per_axis": config.sampling_radius,
            "contains_y_star": False,
        },
        "evaluation_dataset_seed": DATASET_RANDOM_SEED,
        "dataset_description": dataset_description,
        "architecture_configs": select_architectures(config.architecture_names),
    }
    save_json(shared_config_report, output_dir / "shared_experiment_configuration.json")

    # 保存本轮所有网络共同使用的精确数据点，便于复现实验和后续逐点分析。
    torch.save(
        {
            "training_grid": training_grid.detach().cpu(),
            "validation_sets": {
                name: points.detach().cpu() for name, points in validation_sets.items()
            },
            "test_sets": {
                name: points.detach().cpu() for name, points in test_sets.items()
            },
            "p_n": p_n.detach().cpu(),
            "v_n": v_n.detach().cpu(),
            "y_star": y_star.detach().cpu(),
        },
        output_dir / "fixed_train_validation_test_sets.pt",
    )

    summaries = []
    for architecture in select_architectures(config.architecture_names):
        summary = run_architecture_experiment(
            architecture=architecture,
            base_output_dir=output_dir,
            config=config,
            training_grid=training_grid,
            validation_sets=validation_sets,
            test_sets=test_sets,
            p_n=p_n,
            v_n=v_n,
            physics=physics,
        )
        summaries.append(summary)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    overall_report = {
        "experiment_type": "fixed_dataset_mlp_architecture_ablation",
        "purpose": (
            "Change only MLP hidden-layer depth and width while fixing the physics, "
            "training grid, Adam learning rate, Full-Batch schedule, validation set, "
            "test set, normalization, output scaling, and random seeds."
        ),
        "shared_configuration": shared_config_report,
        "architecture_summaries": summaries,
    }
    save_json(overall_report, output_dir / "architecture_ablation_summary.json")

    plot_architecture_training_summary(
        summaries=summaries,
        save_path=output_dir / "architecture_training_and_validation_summary.png",
    )
    plot_architecture_final_summary(
        summaries=summaries,
        final_step=config.final_test_steps,
        save_path=output_dir / "architecture_final_test_summary.png",
    )
    plot_parameter_performance_tradeoff(
        summaries=summaries,
        final_step=config.final_test_steps,
        save_path=output_dir / "parameter_count_vs_near_exterior_performance.png",
    )

    print("\n" + "=" * 90)
    print("全部网络结构实验完成。")
    print(f"总汇总: {output_dir / 'architecture_ablation_summary.json'}")
    print(f"总图: {output_dir / 'architecture_training_and_validation_summary.png'}")
    print(f"总图: {output_dir / 'architecture_final_test_summary.png'}")
    print(f"总图: {output_dir / 'parameter_count_vs_near_exterior_performance.png'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
