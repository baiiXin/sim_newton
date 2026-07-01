#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨运动状态的单步抛体变分优化器实验

实验目标
--------
固定 m、g、dt，只改变运动状态 p_n 和 v_n，训练同一个 MLP 迭代求解器，
并测试它能否泛化到未见过的运动状态所定义的单步优化问题。

默认实验矩阵
------------
1. Identity + SGD(lr=1e-2)
2. ReLU     + SGD(lr=1e-2)
3. Identity + Adam(lr=1e-3)
4. ReLU     + Adam(lr=1e-3)

核心设置
--------
- 训练问题：100 个不同的 (p_n, v_n)
- 验证问题：20 个未见状态
- 测试问题：50 个未见状态
- 每个训练问题：1000 个初值
- 每个 epoch：随机无放回抽取 10 个问题；每个问题内部使用全部 1000 个初值
- 每个 epoch：一次 backward，一次 optimizer.step
- 最大 epoch：5000
- 每 1000 epoch 增加一次展开迭代次数，K=1->5
- 网络：12 -> 32 -> activation -> 3
- 输入逐特征归一化
- 不进行输出缩放
- 训练 loss：原始变分能量，不减 E(y*)，不使用监督标签
- 数值精度：torch.float32

运行示例
--------
python projectile_state_generalization.py
python projectile_state_generalization.py --device cuda:1
python projectile_state_generalization.py --output-dir ./outputs/state_generalization

说明
----
默认会顺序运行全部 4 组实验。输出目录中保存：
- 固定的数据划分与归一化统计量
- 每组实验的最佳/最后 checkpoint
- 训练与验证曲线
- 已见状态新初值、未见状态新初值的测试结果
- 参考状态 p_n 出发的 50 步轨迹
- 全部实验汇总 JSON 和对比图
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn


DTYPE = torch.float32


@dataclass(frozen=True)
class PhysicsConfig:
    mass: float = 1.0
    gravity: float = 9.8
    dt: float = 0.01


@dataclass(frozen=True)
class DataConfig:
    train_problem_count: int = 100
    validation_problem_count: int = 20
    test_problem_count: int = 50

    train_initial_count: int = 1000
    validation_initial_count: int = 512
    test_initial_count: int = 1024

    initial_radius: float = 0.01

    # p_n 的采样范围
    position_xy_min: float = -5.0
    position_xy_max: float = 5.0
    position_z_min: float = 0.0
    position_z_max: float = 10.0

    # v_n 三个分量使用相同范围
    velocity_min: float = -2.0
    velocity_max: float = 2.0

    state_sobol_seed: int = 20260622
    offset_sobol_seed: int = 20260623


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 5000
    problem_batch_size: int = 10
    k_increase_interval: int = 1000
    max_k: int = 5

    validation_interval: int = 100
    evaluation_steps: int = 50
    evaluation_problem_batch_size: int = 10

    hidden_dim: int = 32
    model_seed: int = 42
    problem_schedule_seed: int = 20260624
    print_interval: int = 100


@dataclass(frozen=True)
class ExperimentSpec:
    activation: str
    optimizer_name: str
    learning_rate: float

    @property
    def name(self) -> str:
        lr_text = f"{self.learning_rate:.0e}".replace("+", "")
        return f"{self.activation}_{self.optimizer_name}_lr_{lr_text}"


EXPERIMENTS: Tuple[ExperimentSpec, ...] = (
    ExperimentSpec("identity", "sgd", 1e-2),
    ExperimentSpec("relu", "sgd", 1e-2),
    ExperimentSpec("identity", "adam", 1e-3),
    ExperimentSpec("relu", "adam", 1e-3),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="固定 m、g、dt，仅泛化 p_n、v_n 的单步抛体优化器实验。"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="例如 auto、cpu、cuda:0、cuda:1。默认 auto。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录。默认在脚本同目录创建与脚本同名的文件夹。",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5000,
        help="训练 epoch 数。默认 5000。",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=100,
        help="验证间隔。默认每 100 epoch 验证一次。",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help=(
            "只运行指定实验，例如 "
            "--only identity_sgd_lr_1e-02 relu_adam_lr_1e-03"
        ),
    )
    return parser.parse_args()


def resolve_device(device_text: str) -> torch.device:
    if device_text == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    device = torch.device(device_text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA，但当前环境中 torch.cuda.is_available() 为 False。")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"指定设备 {device} 不存在；当前 CUDA 设备数为 {torch.cuda.device_count()}。"
            )
    return device


def default_output_dir() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.parent / script_path.stem


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable(asdict(value))
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(payload), file, ensure_ascii=False, indent=2)


def affine_map(unit_values: Tensor, low: float, high: float) -> Tensor:
    return low + (high - low) * unit_values


def generate_state_problems(
    data_cfg: DataConfig,
) -> Dict[str, Dict[str, Tensor]]:
    """
    用同一个 6 维 Sobol 序列一次生成全部状态，再严格切分 train/val/test。

    6 维顺序：
    [p_x, p_y, p_z, v_x, v_y, v_z]
    """
    total = (
        data_cfg.train_problem_count
        + data_cfg.validation_problem_count
        + data_cfg.test_problem_count
    )

    engine = torch.quasirandom.SobolEngine(
        dimension=6,
        scramble=True,
        seed=data_cfg.state_sobol_seed,
    )
    unit = engine.draw(total).to(dtype=DTYPE)

    positions = torch.empty(total, 3, dtype=DTYPE)
    velocities = torch.empty(total, 3, dtype=DTYPE)

    positions[:, 0] = affine_map(
        unit[:, 0], data_cfg.position_xy_min, data_cfg.position_xy_max
    )
    positions[:, 1] = affine_map(
        unit[:, 1], data_cfg.position_xy_min, data_cfg.position_xy_max
    )
    positions[:, 2] = affine_map(
        unit[:, 2], data_cfg.position_z_min, data_cfg.position_z_max
    )

    velocities[:, 0] = affine_map(
        unit[:, 3], data_cfg.velocity_min, data_cfg.velocity_max
    )
    velocities[:, 1] = affine_map(
        unit[:, 4], data_cfg.velocity_min, data_cfg.velocity_max
    )
    velocities[:, 2] = affine_map(
        unit[:, 5], data_cfg.velocity_min, data_cfg.velocity_max
    )

    train_end = data_cfg.train_problem_count
    validation_end = train_end + data_cfg.validation_problem_count

    return {
        "train": {
            "positions": positions[:train_end].clone(),
            "velocities": velocities[:train_end].clone(),
        },
        "validation": {
            "positions": positions[train_end:validation_end].clone(),
            "velocities": velocities[train_end:validation_end].clone(),
        },
        "test": {
            "positions": positions[validation_end:].clone(),
            "velocities": velocities[validation_end:].clone(),
        },
    }


def generate_relative_offsets(data_cfg: DataConfig) -> Dict[str, Tensor]:
    """
    使用一个 3 维 Sobol 序列连续抽样并切分，确保 train/val/test 初值扰动不重合。
    """
    total = (
        data_cfg.train_initial_count
        + data_cfg.validation_initial_count
        + data_cfg.test_initial_count
    )

    engine = torch.quasirandom.SobolEngine(
        dimension=3,
        scramble=True,
        seed=data_cfg.offset_sobol_seed,
    )
    unit = engine.draw(total).to(dtype=DTYPE)
    offsets = (2.0 * unit - 1.0) * data_cfg.initial_radius

    train_end = data_cfg.train_initial_count
    validation_end = train_end + data_cfg.validation_initial_count

    return {
        "train": offsets[:train_end].clone(),
        "validation": offsets[train_end:validation_end].clone(),
        "test": offsets[validation_end:].clone(),
    }


def exact_solution(
    positions: Tensor,
    velocities: Tensor,
    physics: PhysicsConfig,
) -> Tensor:
    result = positions + physics.dt * velocities
    result = result.clone()
    result[..., 2] -= (physics.dt**2) * physics.gravity
    return result


def variational_energy(
    y: Tensor,
    positions: Tensor,
    velocities: Tensor,
    physics: PhysicsConfig,
) -> Tensor:
    """
    原始变分能量：
        E(y) = m/(2 dt^2) ||y - p_n - dt v_n||^2 + m g y_z

    训练时直接使用这个函数的均值，不减 E(y*)。
    """
    inertial_delta = y - positions - physics.dt * velocities
    inertial = (
        physics.mass
        / (2.0 * physics.dt * physics.dt)
        * torch.sum(inertial_delta * inertial_delta, dim=-1)
    )
    potential = physics.mass * physics.gravity * y[..., 2]
    return inertial + potential


def residual_vector(
    y: Tensor,
    positions: Tensor,
    velocities: Tensor,
    physics: PhysicsConfig,
) -> Tensor:
    residual = (
        physics.mass
        / (physics.dt * physics.dt)
        * (y - positions - physics.dt * velocities)
    )
    residual = residual.clone()
    residual[..., 2] += physics.mass * physics.gravity
    return residual


def build_flat_problem_batch(
    positions: Tensor,
    velocities: Tensor,
    offsets: Tensor,
    physics: PhysicsConfig,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    输入：
        positions: [B, 3]
        velocities: [B, 3]
        offsets: [N, 3]

    返回扁平张量：
        y0, p_flat, v_flat, y_star_flat: [B*N, 3]
    """
    y_star = exact_solution(positions, velocities, physics)
    batch_size = positions.shape[0]
    initial_count = offsets.shape[0]

    y0 = y_star[:, None, :] + offsets[None, :, :]
    p_flat = (
        positions[:, None, :]
        .expand(batch_size, initial_count, 3)
        .reshape(-1, 3)
    )
    v_flat = (
        velocities[:, None, :]
        .expand(batch_size, initial_count, 3)
        .reshape(-1, 3)
    )
    y_star_flat = (
        y_star[:, None, :]
        .expand(batch_size, initial_count, 3)
        .reshape(-1, 3)
    )
    return y0.reshape(-1, 3), p_flat, v_flat, y_star_flat


def compute_input_normalization(
    train_positions: Tensor,
    train_velocities: Tensor,
    train_offsets: Tensor,
    physics: PhysicsConfig,
) -> Tuple[Tensor, Tensor]:
    """
    仅使用训练问题和训练初值计算 12 个输入特征的统计量。

    输入顺序：
        [y(3), p_n(3), v_n(3), m, g, dt]
    """
    y0, p_flat, v_flat, _ = build_flat_problem_batch(
        train_positions,
        train_velocities,
        train_offsets,
        physics,
    )
    sample_count = y0.shape[0]

    constants = torch.empty(sample_count, 3, dtype=DTYPE)
    constants[:, 0] = physics.mass
    constants[:, 1] = physics.gravity
    constants[:, 2] = physics.dt

    features = torch.cat([y0, p_flat, v_flat, constants], dim=1)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False)

    constant_mask = std < 1e-12
    std[constant_mask] = 1.0
    return mean, std


def activation_module(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "identity":
        return nn.Identity()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"不支持的激活函数：{name}")


class MLPOptimizer(nn.Module):
    def __init__(
        self,
        input_mean: Tensor,
        input_std: Tensor,
        activation: str,
        hidden_dim: int,
        physics: PhysicsConfig,
    ) -> None:
        super().__init__()
        self.register_buffer("input_mean", input_mean.clone().detach())
        self.register_buffer("input_std", input_std.clone().detach())

        self.physics = physics
        self.linear1 = nn.Linear(12, hidden_dim)
        self.activation = activation_module(activation)
        self.linear2 = nn.Linear(hidden_dim, 3)

        # 保留原实验的输出层零初始化。
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, y: Tensor, positions: Tensor, velocities: Tensor) -> Tensor:
        sample_count = y.shape[0]
        constants = y.new_empty(sample_count, 3)
        constants[:, 0] = self.physics.mass
        constants[:, 1] = self.physics.gravity
        constants[:, 2] = self.physics.dt

        features = torch.cat([y, positions, velocities, constants], dim=1)
        normalized = (features - self.input_mean) / self.input_std

        hidden = self.activation(self.linear1(normalized))
        # 不进行 dt 或其他输出缩放。
        delta_y = self.linear2(hidden)
        return delta_y


def create_optimizer(
    model: nn.Module,
    spec: ExperimentSpec,
) -> torch.optim.Optimizer:
    if spec.optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=spec.learning_rate)
    if spec.optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    raise ValueError(f"不支持的优化器：{spec.optimizer_name}")


def build_problem_schedule(
    problem_count: int,
    problem_batch_size: int,
    epochs: int,
    seed: int,
) -> Tensor:
    if problem_batch_size > problem_count:
        raise ValueError("problem_batch_size 不能大于训练问题总数。")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    schedule = torch.empty(epochs, problem_batch_size, dtype=torch.long)
    for epoch_index in range(epochs):
        schedule[epoch_index] = torch.randperm(
            problem_count,
            generator=generator,
        )[:problem_batch_size]
    return schedule


def current_unroll_steps(
    epoch: int,
    interval: int,
    max_k: int,
) -> int:
    return min(1 + (epoch - 1) // interval, max_k)


def clone_state_dict_to_cpu(model: nn.Module) -> Dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def finite_tensor_stats(values: Tensor) -> Dict[str, float | int]:
    flattened = values.detach().reshape(-1).to(dtype=torch.float64)
    finite_mask = torch.isfinite(flattened)
    finite_values = flattened[finite_mask]

    result: Dict[str, float | int] = {
        "count": int(flattened.numel()),
        "finite_count": int(finite_mask.sum().item()),
        "nonfinite_count": int((~finite_mask).sum().item()),
    }

    if finite_values.numel() == 0:
        result.update(
            {
                "mean": math.inf,
                "median": math.inf,
                "p95": math.inf,
                "max": math.inf,
                "min": math.inf,
            }
        )
        return result

    result.update(
        {
            "mean": float(finite_values.mean().item()),
            "median": float(torch.quantile(finite_values, 0.5).item()),
            "p95": float(torch.quantile(finite_values, 0.95).item()),
            "max": float(finite_values.max().item()),
            "min": float(finite_values.min().item()),
        }
    )
    return result


def rowwise_p95_with_nonfinite_penalty(values: Tensor) -> Tensor:
    """
    values: [problem_count, initial_count]

    某个问题只要出现非有限值，就把该问题的 p95 记为 +inf，
    防止验证统计掩盖发散问题。
    """
    values64 = values.to(dtype=torch.float64)
    result = torch.empty(values64.shape[0], dtype=torch.float64)

    for row_index in range(values64.shape[0]):
        row = values64[row_index]
        if not torch.isfinite(row).all():
            result[row_index] = math.inf
        else:
            result[row_index] = torch.quantile(row, 0.95)
    return result


@torch.no_grad()
def evaluate_problem_set(
    model: MLPOptimizer,
    positions_cpu: Tensor,
    velocities_cpu: Tensor,
    offsets_cpu: Tensor,
    physics: PhysicsConfig,
    device: torch.device,
    steps: int,
    problem_batch_size: int,
) -> Dict[str, Any]:
    model.eval()

    step_residual_chunks: List[List[Tensor]] = [
        [] for _ in range(steps + 1)
    ]
    step_gap_chunks: List[List[Tensor]] = [
        [] for _ in range(steps + 1)
    ]
    step_position_error_chunks: List[List[Tensor]] = [
        [] for _ in range(steps + 1)
    ]

    per_problem_final_residual_p95: List[Tensor] = []
    per_problem_final_gap_p95: List[Tensor] = []
    per_problem_final_position_error_p95: List[Tensor] = []

    problem_count = positions_cpu.shape[0]
    initial_count = offsets_cpu.shape[0]

    offsets = offsets_cpu.to(device=device, dtype=DTYPE)

    for start in range(0, problem_count, problem_batch_size):
        end = min(start + problem_batch_size, problem_count)

        positions = positions_cpu[start:end].to(device=device, dtype=DTYPE)
        velocities = velocities_cpu[start:end].to(device=device, dtype=DTYPE)

        y, p_flat, v_flat, y_star_flat = build_flat_problem_batch(
            positions,
            velocities,
            offsets,
            physics,
        )

        batch_problem_count = end - start

        def record_step(current_y: Tensor, step_index: int) -> None:
            residual_norm = torch.linalg.vector_norm(
                residual_vector(current_y, p_flat, v_flat, physics),
                dim=-1,
            )
            energy = variational_energy(current_y, p_flat, v_flat, physics)
            exact_energy = variational_energy(
                y_star_flat,
                p_flat,
                v_flat,
                physics,
            )
            gap = energy - exact_energy
            position_error = torch.linalg.vector_norm(
                current_y - y_star_flat,
                dim=-1,
            )

            step_residual_chunks[step_index].append(
                residual_norm.detach().cpu()
            )
            step_gap_chunks[step_index].append(gap.detach().cpu())
            step_position_error_chunks[step_index].append(
                position_error.detach().cpu()
            )

        record_step(y, 0)
        for step_index in range(1, steps + 1):
            y = y + model(y, p_flat, v_flat)
            record_step(y, step_index)

        final_residual = step_residual_chunks[-1][-1].reshape(
            batch_problem_count,
            initial_count,
        )
        final_gap = step_gap_chunks[-1][-1].reshape(
            batch_problem_count,
            initial_count,
        )
        final_position_error = step_position_error_chunks[-1][-1].reshape(
            batch_problem_count,
            initial_count,
        )

        per_problem_final_residual_p95.append(
            rowwise_p95_with_nonfinite_penalty(final_residual)
        )
        per_problem_final_gap_p95.append(
            rowwise_p95_with_nonfinite_penalty(final_gap)
        )
        per_problem_final_position_error_p95.append(
            rowwise_p95_with_nonfinite_penalty(final_position_error)
        )

    residual_step_stats = []
    gap_step_stats = []
    position_error_step_stats = []

    for step_index in range(steps + 1):
        residual_step_stats.append(
            finite_tensor_stats(torch.cat(step_residual_chunks[step_index]))
        )
        gap_step_stats.append(
            finite_tensor_stats(torch.cat(step_gap_chunks[step_index]))
        )
        position_error_step_stats.append(
            finite_tensor_stats(
                torch.cat(step_position_error_chunks[step_index])
            )
        )

    per_problem_residual_p95 = torch.cat(per_problem_final_residual_p95)
    per_problem_gap_p95 = torch.cat(per_problem_final_gap_p95)
    per_problem_position_error_p95 = torch.cat(
        per_problem_final_position_error_p95
    )

    final_residual = torch.cat(step_residual_chunks[-1])
    final_gap = torch.cat(step_gap_chunks[-1])
    final_position_error = torch.cat(step_position_error_chunks[-1])

    result = {
        "problem_count": problem_count,
        "initial_count_per_problem": initial_count,
        "evaluation_steps": steps,
        "final": {
            "residual": finite_tensor_stats(final_residual),
            "energy_gap": finite_tensor_stats(final_gap),
            "position_error": finite_tensor_stats(final_position_error),
        },
        "problem_level_final_p95": {
            "residual": finite_tensor_stats(per_problem_residual_p95),
            "energy_gap": finite_tensor_stats(per_problem_gap_p95),
            "position_error": finite_tensor_stats(
                per_problem_position_error_p95
            ),
            "residual_values": per_problem_residual_p95.tolist(),
            "energy_gap_values": per_problem_gap_p95.tolist(),
            "position_error_values": (
                per_problem_position_error_p95.tolist()
            ),
        },
        "step_statistics": {
            "residual": residual_step_stats,
            "energy_gap": gap_step_stats,
            "position_error": position_error_step_stats,
        },
    }
    return result


def validation_selection_key(metrics: Dict[str, Any]) -> Tuple[float, ...]:
    final_residual = metrics["final"]["residual"]
    problem_residual = metrics["problem_level_final_p95"]["residual"]

    return (
        float(final_residual["nonfinite_count"]),
        float(problem_residual["nonfinite_count"]),
        float(problem_residual["p95"]),
        float(problem_residual["median"]),
        float(final_residual["p95"]),
    )


@torch.no_grad()
def evaluate_reference_trajectory(
    model: MLPOptimizer,
    physics: PhysicsConfig,
    device: torch.device,
    steps: int,
) -> Dict[str, Any]:
    """
    使用原始参考运动状态：
        p_n = (3, 4, 5)
        v_n = (0.5, -0.5, 0)

    从 y^(0) = p_n 出发迭代 50 步。
    """
    model.eval()

    p = torch.tensor([[3.0, 4.0, 5.0]], dtype=DTYPE, device=device)
    v = torch.tensor([[0.5, -0.5, 0.0]], dtype=DTYPE, device=device)
    y_star = exact_solution(p, v, physics)
    y = p.clone()

    trajectory = [y.detach().cpu().squeeze(0).tolist()]
    residuals = []
    energy_gaps = []
    position_errors = []

    exact_energy = variational_energy(y_star, p, v, physics)

    def record(current_y: Tensor) -> None:
        residuals.append(
            float(
                torch.linalg.vector_norm(
                    residual_vector(current_y, p, v, physics),
                    dim=-1,
                ).item()
            )
        )
        energy_gaps.append(
            float(
                (
                    variational_energy(current_y, p, v, physics)
                    - exact_energy
                ).item()
            )
        )
        position_errors.append(
            float(
                torch.linalg.vector_norm(
                    current_y - y_star,
                    dim=-1,
                ).item()
            )
        )

    record(y)
    for _ in range(steps):
        y = y + model(y, p, v)
        trajectory.append(y.detach().cpu().squeeze(0).tolist())
        record(y)

    return {
        "position": p.detach().cpu().squeeze(0).tolist(),
        "velocity": v.detach().cpu().squeeze(0).tolist(),
        "exact_solution": y_star.detach().cpu().squeeze(0).tolist(),
        "trajectory": trajectory,
        "residual": residuals,
        "energy_gap": energy_gaps,
        "position_error": position_errors,
    }


def safe_positive(values: Sequence[float], floor: float = 1e-30) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    result = np.full_like(array, np.nan)
    result[finite] = np.maximum(array[finite], floor)
    return result


def plot_training_raw_loss(
    epochs: Sequence[int],
    values: Sequence[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values)
    plt.xlabel("Epoch")
    plt.ylabel("Raw variational energy loss")
    plt.title("Training raw variational energy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_training_gap(
    epochs: Sequence[int],
    values: Sequence[float],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, safe_positive(values))
    plt.xlabel("Epoch")
    plt.ylabel("Trajectory energy-gap sum")
    plt.title("Training energy gap (monitor only)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_validation_history(
    validation_history: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    epochs = [entry["epoch"] for entry in validation_history]
    p95_values = [
        entry["metrics"]["problem_level_final_p95"]["residual"]["p95"]
        for entry in validation_history
    ]
    median_values = [
        entry["metrics"]["problem_level_final_p95"]["residual"]["median"]
        for entry in validation_history
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, safe_positive(p95_values), label="problem p95 of p95")
    plt.plot(
        epochs,
        safe_positive(median_values),
        label="problem median of p95",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Final residual after evaluation rollout")
    plt.title("Validation problem-level residual")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_reference_metric(
    metric_values: Sequence[float],
    metric_name: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(metric_values)), safe_positive(metric_values))
    plt.xlabel("Iteration")
    plt.ylabel(metric_name)
    plt.title(f"Reference-state {metric_name}")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_summary(
    summaries: Sequence[Dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [entry["experiment"] for entry in summaries]
    medians = [
        entry["best_unseen_test"]["final"]["residual"]["median"]
        for entry in summaries
    ]
    p95_values = [
        entry["best_unseen_test"]["final"]["residual"]["p95"]
        for entry in summaries
    ]
    problem_p95_values = [
        entry["best_unseen_test"]["problem_level_final_p95"]["residual"]["p95"]
        for entry in summaries
    ]

    x = np.arange(len(labels))
    plt.figure(figsize=(11, 6))
    plt.plot(x, safe_positive(medians), marker="o", label="sample median")
    plt.plot(x, safe_positive(p95_values), marker="o", label="sample p95")
    plt.plot(
        x,
        safe_positive(problem_p95_values),
        marker="o",
        label="problem p95 of p95",
    )
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Final residual after 50 iterations")
    plt.title("Unseen-state test comparison")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_checkpoint(
    path: Path,
    state_dict: Dict[str, Tensor],
    spec: ExperimentSpec,
    input_mean: Tensor,
    input_std: Tensor,
    physics: PhysicsConfig,
    training_cfg: TrainingConfig,
) -> None:
    torch.save(
        {
            "model_state_dict": state_dict,
            "experiment": asdict(spec),
            "input_mean": input_mean.detach().cpu(),
            "input_std": input_std.detach().cpu(),
            "physics": asdict(physics),
            "training": asdict(training_cfg),
            "dtype": str(DTYPE),
        },
        path,
    )


def train_one_experiment(
    spec: ExperimentSpec,
    states: Dict[str, Dict[str, Tensor]],
    offsets: Dict[str, Tensor],
    input_mean_cpu: Tensor,
    input_std_cpu: Tensor,
    physics: PhysicsConfig,
    training_cfg: TrainingConfig,
    schedule: Tensor,
    device: torch.device,
    experiment_dir: Path,
) -> Dict[str, Any]:
    experiment_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(training_cfg.model_seed)

    model = MLPOptimizer(
        input_mean=input_mean_cpu.to(device=device, dtype=DTYPE),
        input_std=input_std_cpu.to(device=device, dtype=DTYPE),
        activation=spec.activation,
        hidden_dim=training_cfg.hidden_dim,
        physics=physics,
    ).to(device=device, dtype=DTYPE)

    optimizer = create_optimizer(model, spec)

    train_positions_cpu = states["train"]["positions"]
    train_velocities_cpu = states["train"]["velocities"]
    train_offsets_device = offsets["train"].to(device=device, dtype=DTYPE)

    raw_loss_history: List[float] = []
    gap_history: List[float] = []
    k_history: List[int] = []
    validation_history: List[Dict[str, Any]] = []

    best_key: Tuple[float, ...] | None = None
    best_state: Dict[str, Tensor] | None = None
    best_epoch: int | None = None
    stopped_reason: str | None = None

    start_time = time.time()

    for epoch in range(1, training_cfg.epochs + 1):
        model.train()
        k_steps = current_unroll_steps(
            epoch,
            training_cfg.k_increase_interval,
            training_cfg.max_k,
        )

        problem_indices = schedule[epoch - 1]
        positions = train_positions_cpu[problem_indices].to(
            device=device,
            dtype=DTYPE,
        )
        velocities = train_velocities_cpu[problem_indices].to(
            device=device,
            dtype=DTYPE,
        )

        y, p_flat, v_flat, y_star_flat = build_flat_problem_batch(
            positions,
            velocities,
            train_offsets_device,
            physics,
        )

        optimizer.zero_grad(set_to_none=True)

        raw_loss = torch.zeros((), device=device, dtype=DTYPE)
        monitoring_gap = torch.zeros((), device=device, dtype=DTYPE)

        exact_energy = variational_energy(
            y_star_flat,
            p_flat,
            v_flat,
            physics,
        )

        for _ in range(k_steps):
            y = y + model(y, p_flat, v_flat)
            current_energy = variational_energy(
                y,
                p_flat,
                v_flat,
                physics,
            )
            # 真正用于反向传播的 loss：原始变分能量。
            raw_loss = raw_loss + current_energy.mean()

            # 仅用于监控，不参与替换训练目标。
            monitoring_gap = monitoring_gap + (
                current_energy - exact_energy
            ).mean()

        if not torch.isfinite(raw_loss):
            stopped_reason = f"epoch {epoch}: raw loss is non-finite"
            print(f"[{spec.name}] {stopped_reason}")
            break

        raw_loss.backward()
        optimizer.step()

        parameter_finite = all(
            torch.isfinite(parameter).all()
            for parameter in model.parameters()
        )
        if not parameter_finite:
            stopped_reason = f"epoch {epoch}: model parameter is non-finite"
            print(f"[{spec.name}] {stopped_reason}")
            break

        raw_loss_history.append(float(raw_loss.detach().cpu().item()))
        gap_history.append(float(monitoring_gap.detach().cpu().item()))
        k_history.append(k_steps)

        should_validate = (
            epoch % training_cfg.validation_interval == 0
            or epoch == training_cfg.epochs
        )
        if should_validate:
            validation_metrics = evaluate_problem_set(
                model=model,
                positions_cpu=states["validation"]["positions"],
                velocities_cpu=states["validation"]["velocities"],
                offsets_cpu=offsets["validation"],
                physics=physics,
                device=device,
                steps=training_cfg.evaluation_steps,
                problem_batch_size=(
                    training_cfg.evaluation_problem_batch_size
                ),
            )
            key = validation_selection_key(validation_metrics)

            validation_entry = {
                "epoch": epoch,
                "k_steps": k_steps,
                "selection_key": list(key),
                "metrics": validation_metrics,
            }
            validation_history.append(validation_entry)

            if best_key is None or key < best_key:
                best_key = key
                best_state = clone_state_dict_to_cpu(model)
                best_epoch = epoch

        if (
            epoch == 1
            or epoch % training_cfg.print_interval == 0
            or epoch == training_cfg.epochs
        ):
            message = (
                f"[{spec.name}] "
                f"epoch={epoch:5d}/{training_cfg.epochs}, "
                f"K={k_steps}, "
                f"raw_loss={raw_loss_history[-1]:.8e}, "
                f"gap={gap_history[-1]:.8e}"
            )
            if validation_history and validation_history[-1]["epoch"] == epoch:
                val_p95 = validation_history[-1]["metrics"][
                    "problem_level_final_p95"
                ]["residual"]["p95"]
                message += f", val_problem_p95={val_p95:.8e}"
            print(message, flush=True)

    elapsed_seconds = time.time() - start_time

    last_state = clone_state_dict_to_cpu(model)
    if best_state is None:
        best_state = copy.deepcopy(last_state)
        best_epoch = len(raw_loss_history)
        best_key = (math.inf, math.inf, math.inf, math.inf, math.inf)

    save_checkpoint(
        experiment_dir / "last_model_state_dict.pt",
        last_state,
        spec,
        input_mean_cpu,
        input_std_cpu,
        physics,
        training_cfg,
    )
    save_checkpoint(
        experiment_dir / "best_validation_model_state_dict.pt",
        best_state,
        spec,
        input_mean_cpu,
        input_std_cpu,
        physics,
        training_cfg,
    )
    save_checkpoint(
        experiment_dir / "mlp_optimizer_state_dict.pt",
        best_state,
        spec,
        input_mean_cpu,
        input_std_cpu,
        physics,
        training_cfg,
    )

    completed_epochs = len(raw_loss_history)
    epoch_axis = list(range(1, completed_epochs + 1))

    plot_training_raw_loss(
        epoch_axis,
        raw_loss_history,
        experiment_dir / "training_raw_variational_energy.png",
    )
    plot_training_gap(
        epoch_axis,
        gap_history,
        experiment_dir / "training_energy_gap_monitor.png",
    )
    if validation_history:
        plot_validation_history(
            validation_history,
            experiment_dir / "validation_problem_residual.png",
        )

    # 最佳 checkpoint：已见训练状态 + 新初值
    model.load_state_dict(best_state)
    best_seen_test = evaluate_problem_set(
        model=model,
        positions_cpu=states["train"]["positions"],
        velocities_cpu=states["train"]["velocities"],
        offsets_cpu=offsets["test"],
        physics=physics,
        device=device,
        steps=training_cfg.evaluation_steps,
        problem_batch_size=training_cfg.evaluation_problem_batch_size,
    )

    # 最佳 checkpoint：未见测试状态 + 新初值
    best_unseen_test = evaluate_problem_set(
        model=model,
        positions_cpu=states["test"]["positions"],
        velocities_cpu=states["test"]["velocities"],
        offsets_cpu=offsets["test"],
        physics=physics,
        device=device,
        steps=training_cfg.evaluation_steps,
        problem_batch_size=training_cfg.evaluation_problem_batch_size,
    )

    best_reference = evaluate_reference_trajectory(
        model=model,
        physics=physics,
        device=device,
        steps=training_cfg.evaluation_steps,
    )

    plot_reference_metric(
        best_reference["residual"],
        "Residual norm",
        experiment_dir / "best_reference_residual.png",
    )
    plot_reference_metric(
        best_reference["energy_gap"],
        "Energy gap",
        experiment_dir / "best_reference_energy_gap.png",
    )
    plot_reference_metric(
        best_reference["position_error"],
        "Position error",
        experiment_dir / "best_reference_position_error.png",
    )

    # 最后 checkpoint：未见测试状态 + 新初值
    model.load_state_dict(last_state)
    last_unseen_test = evaluate_problem_set(
        model=model,
        positions_cpu=states["test"]["positions"],
        velocities_cpu=states["test"]["velocities"],
        offsets_cpu=offsets["test"],
        physics=physics,
        device=device,
        steps=training_cfg.evaluation_steps,
        problem_batch_size=training_cfg.evaluation_problem_batch_size,
    )

    report = {
        "experiment": asdict(spec),
        "experiment_name": spec.name,
        "device": str(device),
        "dtype": str(DTYPE),
        "physics": asdict(physics),
        "training": asdict(training_cfg),
        "completed_epochs": completed_epochs,
        "stopped_reason": stopped_reason,
        "elapsed_seconds": elapsed_seconds,
        "best_epoch": best_epoch,
        "best_selection_key": list(best_key),
        "raw_loss_history": raw_loss_history,
        "training_gap_history": gap_history,
        "k_history": k_history,
        "validation_history": validation_history,
        "best_checkpoint": {
            "seen_states_new_initials": best_seen_test,
            "unseen_states_new_initials": best_unseen_test,
            "reference_trajectory": best_reference,
        },
        "last_checkpoint": {
            "unseen_states_new_initials": last_unseen_test,
        },
    }
    save_json(experiment_dir / "optimization_report.json", report)

    concise_summary = {
        "experiment": spec.name,
        "best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "stopped_reason": stopped_reason,
        "best_seen_test": best_seen_test,
        "best_unseen_test": best_unseen_test,
        "last_unseen_test": last_unseen_test,
    }
    return concise_summary


def main() -> None:
    args = parse_args()

    physics = PhysicsConfig()
    data_cfg = DataConfig()
    training_cfg = TrainingConfig(
        epochs=args.epochs,
        validation_interval=args.validation_interval,
    )

    device = resolve_device(args.device)

    if args.output_dir is None:
        output_dir = default_output_dir()
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("跨运动状态的单步抛体变分优化器实验")
    print(f"PyTorch version : {torch.__version__}")
    print(f"Device          : {device}")
    print(f"Dtype           : {DTYPE}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    states = generate_state_problems(data_cfg)
    offsets = generate_relative_offsets(data_cfg)

    input_mean, input_std = compute_input_normalization(
        train_positions=states["train"]["positions"],
        train_velocities=states["train"]["velocities"],
        train_offsets=offsets["train"],
        physics=physics,
    )

    schedule = build_problem_schedule(
        problem_count=data_cfg.train_problem_count,
        problem_batch_size=training_cfg.problem_batch_size,
        epochs=training_cfg.epochs,
        seed=training_cfg.problem_schedule_seed,
    )

    # 固定数据划分，便于所有实验严格复用。
    torch.save(
        {
            "states": states,
            "relative_offsets": offsets,
            "input_mean": input_mean,
            "input_std": input_std,
            "problem_schedule": schedule,
            "physics": asdict(physics),
            "data_config": asdict(data_cfg),
            "training_config": asdict(training_cfg),
            "dtype": str(DTYPE),
        },
        output_dir / "fixed_dataset_split.pt",
    )
    save_json(
        output_dir / "fixed_dataset_split.json",
        {
            "physics": asdict(physics),
            "data_config": asdict(data_cfg),
            "training_config": asdict(training_cfg),
            "input_mean": input_mean,
            "input_std": input_std,
            "state_split": states,
            "relative_offsets": offsets,
            "notes": {
                "varying_quantities": ["p_n", "v_n"],
                "fixed_quantities": ["m", "g", "dt"],
                "training_loss": (
                    "raw variational energy; no E(y*) subtraction "
                    "in backward objective"
                ),
                "problem_batching": (
                    "10 problems sampled without replacement per epoch; "
                    "1000 initial points per selected problem are full-batch"
                ),
            },
        },
    )

    selected_experiments = list(EXPERIMENTS)
    if args.only:
        requested = set(args.only)
        selected_experiments = [
            spec for spec in EXPERIMENTS if spec.name in requested
        ]
        unknown = requested - {spec.name for spec in EXPERIMENTS}
        if unknown:
            available = ", ".join(spec.name for spec in EXPERIMENTS)
            raise ValueError(
                f"未知实验名称：{sorted(unknown)}。可选名称：{available}"
            )

    summaries: List[Dict[str, Any]] = []

    for experiment_index, spec in enumerate(selected_experiments, start=1):
        print()
        print("-" * 80)
        print(
            f"Running experiment {experiment_index}/"
            f"{len(selected_experiments)}: {spec.name}"
        )
        print("-" * 80)

        experiment_dir = output_dir / spec.name
        summary = train_one_experiment(
            spec=spec,
            states=states,
            offsets=offsets,
            input_mean_cpu=input_mean,
            input_std_cpu=input_std,
            physics=physics,
            training_cfg=training_cfg,
            schedule=schedule,
            device=device,
            experiment_dir=experiment_dir,
        )
        summaries.append(summary)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_json(
        output_dir / "state_generalization_summary.json",
        {
            "physics": asdict(physics),
            "data_config": asdict(data_cfg),
            "training_config": asdict(training_cfg),
            "device": str(device),
            "dtype": str(DTYPE),
            "experiments": summaries,
        },
    )

    if summaries:
        plot_summary(
            summaries,
            output_dir / "unseen_state_test_summary.png",
        )

    print()
    print("=" * 80)
    print("全部实验完成。")
    print(f"输出目录：{output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
