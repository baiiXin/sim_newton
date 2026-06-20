"""
自由落体单帧变分问题：规则网格 Mini-Batch 数据规模消融实验
float64，batch size=1000，50,000 次反向传播，固定单步展开 K=1
================================================================

实验目的
--------
以原始规则网格 Full-Batch 脚本为基准，只改变训练计数与取样方式：

1. 物理问题、网络结构、float64 精度、归一化、输出 dt 缩放保持不变；
2. 训练集仍是以精确解 y_star 为中心的三维规则 Cartesian 网格；
3. 停止条件不再是 epoch，而是成功执行 50,000 次 backward + optimizer.step；
4. 当训练集点数 N < 1000 时，每次更新使用整个训练集；
5. 当 N >= 1000 时，每次更新使用 batch size=1000；
6. mini-batch 使用循环无放回乱序：当前随机排列用完后重新打乱，
   边界不足 1000 时与下一轮排列拼接；
7. 每次参数更新在同一组网络参数下完整展开当前 K 步，步间不 detach，
   只调用一次 backward 和一次 optimizer.step；
8. loss 仍为 K 个展开步骤的 batch 平均隐式欧拉变分能量之和；
9. 所有 50,000 次反向传播都固定使用 K=1，不启用展开步数 curriculum。
10. 周期评估间隔按反向传播次数解释；
11. 最终仍从 held-out 物理初值 p_n 展开 50 步，与 Newton 对比；
12. 默认在 cuda:0 上运行，可通过 --device 指定其他 GPU。

规则网格规模
------------
    2^3   = 8
    4^3   = 64
    6^3   = 216
    10^3  = 1,000
    22^3  = 10,648
    46^3  = 97,336
    100^3 = 1,000,000

训练计数定义
------------
global_update_step 每增加 1，表示已经成功完成一次：
    optimizer.zero_grad()
    trajectory_loss.backward()
    optimizer.step()

数据集遍历轮数只作为统计量记录，不作为停止条件。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # 适配无显示器 Linux 环境

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 0. 默认实验参数
# ============================================================

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

PLOT_FLOOR = 1e-12

# float64 对照实验：恢复输入逐特征归一化与输出 dt 缩放。
USE_INPUT_NORMALIZATION = True
USE_OUTPUT_DT_SCALING = True

# 固定规则网格规模。每组均为 even_points_per_axis ** 3。
# 每个维度使用偶数个等距点，因此不会包含中心 y_star。
DEFAULT_TARGET_DATASET_SIZE_VALUES = [
    8,
    64,
    216,
    1_000,
    10_648,
    97_336,
    1_000_000,
]

# 固定局部采样范围：
# 每个维度都在 [y_star - R, y_star + R] 内均匀取点。
DEFAULT_SAMPLING_RADIUS = 0.01

# 训练停止条件：成功完成的反向传播/参数更新次数。
DEFAULT_UPDATE_STEPS = 50_000
DEFAULT_EVAL_INTERVAL_UPDATES = 100

# N < 1000 时使用 full-batch；N >= 1000 时目标 batch size 为 1000。
DEFAULT_BATCH_SIZE = 1_000

# 最终冻结测试仍统一展开 50 步，便于和原始实验直接比较。
DEFAULT_FINAL_TEST_STEPS = 50

# K 按 zero-based update_index 计算。
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL_UPDATES = 10_000
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 1

# 预生成完整规则网格时分块写入，控制临时张量峰值。
DEFAULT_GRID_PRECOMPUTE_CHUNK_SIZE = 1_000_000

# 固定随机种子：模型初始化与 mini-batch 乱序分别控制。
MODEL_RANDOM_SEED = 42
BATCH_RANDOM_SEED = 20260617

# 为控制输出文件体积，训练点分布图最多展示这些样本。
MAX_SCATTER_POINTS = 20_000

# float64 对照实验：完整测试 SGD/Adam × 三种学习率。
OPTIMIZER_CONFIGS = [
    {"optimizer_name": "sgd", "learning_rate": 1e-2},
    {"optimizer_name": "sgd", "learning_rate": 1e-3},
    {"optimizer_name": "sgd", "learning_rate": 1e-4},
    {"optimizer_name": "adam", "learning_rate": 1e-2},
    {"optimizer_name": "adam", "learning_rate": 1e-3},
    {"optimizer_name": "adam", "learning_rate": 1e-4},
]


# ============================================================
# 1. 数据结构与通用辅助函数
# ============================================================


@dataclass(frozen=True)
class RuntimeConfig:
    target_dataset_sizes: list[int]
    sampling_radius: float
    grid_precompute_chunk_size: int
    update_steps: int
    eval_interval_updates: int
    batch_size: int
    final_test_steps: int
    initial_k: int
    k_increase_interval_updates: int
    k_increase_amount: int
    max_k: int
    device: str
    skip_contour: bool


@dataclass(frozen=True)
class GridSpec:
    """一组规则三维网格的配置。"""

    target_num_points: int
    points_per_axis: int
    actual_num_points: int
    sampling_radius: float
    axis_spacing: float


class CyclingShuffledBatchSampler:
    """
    不依赖 epoch 的循环无放回 mini-batch 索引生成器。

    当前随机排列用完后重新打乱。若一个 batch 跨越排列末尾，
    则从下一轮新排列中补足，保证每次返回 batch_size 个索引。
    """

    def __init__(
        self,
        *,
        num_samples: int,
        batch_size: int,
        device: torch.device,
        seed: int,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > num_samples:
            raise ValueError(
                "CyclingShuffledBatchSampler requires batch_size <= num_samples."
            )

        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.device = device

        generator_device = device if device.type == "cuda" else torch.device("cpu")
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(int(seed))

        self._permutation: torch.Tensor | None = None
        self._cursor = 0
        self.completed_dataset_passes = 0

    @property
    def current_pass_fraction(self) -> float:
        return float(self._cursor) / float(self.num_samples)

    def _start_new_permutation(self) -> None:
        self._permutation = torch.randperm(
            self.num_samples,
            dtype=torch.int64,
            device=self.device,
            generator=self.generator,
        )
        self._cursor = 0

    def next_indices(self) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        remaining = self.batch_size

        while remaining > 0:
            if self._permutation is None:
                self._start_new_permutation()

            available = self.num_samples - self._cursor
            take = min(remaining, available)
            assert self._permutation is not None
            chunks.append(self._permutation[self._cursor : self._cursor + take])
            self._cursor += take
            remaining -= take

            if self._cursor == self.num_samples:
                self.completed_dataset_passes += 1
                self._permutation = None
                self._cursor = 0

        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=0)



def create_output_directory() -> Path:
    """在脚本同目录下创建同名输出目录。"""

    script_path = Path(__file__).resolve()
    output_dir = script_path.parent / script_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_positive_int_list(values: Iterable[int]) -> list[int]:
    """检查正整数列表，并去重排序。"""

    cleaned = sorted({int(value) for value in values})
    if not cleaned:
        raise ValueError("target_dataset_sizes must not be empty.")
    if cleaned[0] <= 0:
        raise ValueError("Every target dataset size must be positive.")
    return cleaned


def nearest_even_points_per_axis(target_num_points: int) -> int:
    """
    返回使 n^3 最接近目标总点数的偶数 n。

    三个维度使用相同的偶数采样点数。对于关于 0 对称的区间，
    偶数个轴向采样点不会包含 0，因此三维规则网格不会包含 y_star。
    """

    if target_num_points <= 0:
        raise ValueError("target_num_points must be positive.")

    root = target_num_points ** (1.0 / 3.0)
    lower = max(2, 2 * int(math.floor(root / 2.0)))
    upper = max(2, lower + 2)
    candidates = sorted({lower, upper})
    return min(candidates, key=lambda n: (abs(n**3 - target_num_points), n))


def make_grid_spec(target_num_points: int, sampling_radius: float) -> GridSpec:
    """根据目标总点数构造最接近的偶数轴向规则网格。"""

    if sampling_radius <= 0.0:
        raise ValueError("sampling_radius must be positive.")

    points_per_axis = nearest_even_points_per_axis(target_num_points)
    actual_num_points = points_per_axis**3
    axis_spacing = (2.0 * sampling_radius) / (points_per_axis - 1)
    return GridSpec(
        target_num_points=int(target_num_points),
        points_per_axis=points_per_axis,
        actual_num_points=actual_num_points,
        sampling_radius=float(sampling_radius),
        axis_spacing=float(axis_spacing),
    )


def make_grid_specs(
    target_dataset_sizes: Iterable[int],
    sampling_radius: float,
) -> list[GridSpec]:
    """构造五组或用户指定的规则网格配置，并检查实际网格规模不重复。"""

    specs = [
        make_grid_spec(target_num_points, sampling_radius)
        for target_num_points in ensure_positive_int_list(target_dataset_sizes)
    ]
    actual_sizes = [spec.actual_num_points for spec in specs]
    if len(set(actual_sizes)) != len(actual_sizes):
        raise ValueError(
            "Different target dataset sizes mapped to the same even grid size. "
            "Please choose more separated target values."
        )
    return specs


def get_k_for_update_step(update_index: int, config: RuntimeConfig) -> int:
    """根据 zero-based update_index 返回当前展开轨迹长度 K。"""

    return min(
        config.initial_k
        + (update_index // config.k_increase_interval_updates)
        * config.k_increase_amount,
        config.max_k,
    )


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    """将张量安全转换为普通 Python list。"""

    return tensor.detach().cpu().tolist()


def is_model_finite(model: nn.Module) -> bool:
    """检查网络参数是否全部为有限值。"""

    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def finite_plot_value(value: float) -> float:
    """将不可绘制的非有限值转换为 NaN，有限小值截断到绘图下限。"""

    value = float(value)
    if not math.isfinite(value):
        return float("nan")
    return max(value, PLOT_FLOOR)


def finite_rows(points: np.ndarray) -> np.ndarray:
    """只保留每一个分量均为有限值的坐标行。"""

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return points[np.isfinite(points).all(axis=1)]


def make_json_safe(value):
    """递归地将 NaN 和 Inf 转换为 None，保证输出 JSON 符合标准。"""

    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ============================================================
# 2. 物理问题、网络与优化器
# ============================================================


class MLPOptimizer(nn.Module):
    """
    学习型迭代优化器。

    输入：
        [y, p_n, v_n, m, g, dt]，共 12 维。

    输出：
        网络首先预测无缩放更新量，再乘以 dt 得到 delta_y。
    """

    def __init__(
        self,
        *,
        use_input_normalization: bool = True,
        use_output_dt_scaling: bool = True,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.use_input_normalization = use_input_normalization
        self.use_output_dt_scaling = use_output_dt_scaling

        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        # 与原实验保持一致：初始网络输出严格为 0。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        if input_mean is None:
            input_mean = torch.zeros(12, dtype=TORCH_DTYPE)
        if input_std is None:
            input_std = torch.ones(12, dtype=TORCH_DTYPE)

        self.register_buffer("input_mean", input_mean.clone().detach())
        self.register_buffer("input_std", input_std.clone().detach())

    @staticmethod
    def _expand_feature_for_batch(
        feature: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """将一维固定特征扩展为 [batch_size, feature_dim]。"""

        if feature.ndim == 1:
            return feature.unsqueeze(0).expand(batch_size, -1)
        if feature.ndim == 2 and feature.shape[0] == batch_size:
            return feature
        raise ValueError(
            "Feature shape is incompatible with the current batch: "
            f"feature.shape={tuple(feature.shape)}, batch_size={batch_size}."
        )

    def forward(
        self,
        y: torch.Tensor,
        history: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """支持单点 [3] 和批量点 [B, 3]。"""

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
    m: float = 1.0,
    g: float = 9.8,
    dt: float = 0.01,
) -> torch.Tensor:
    """
    隐式欧拉变分能量。

    支持：
        y.shape == [3]      -> 返回标量
        y.shape == [B, 3]   -> 返回 [B]

    E(y) = m / (2 dt^2) * ||y - p_n - dt v_n||^2 + m g y_z
    """

    residual = y - p_n - dt * v_n
    kinetic_term = (m / (2.0 * dt**2)) * torch.sum(residual**2, dim=-1)
    potential_term = m * g * y[..., 2]
    return kinetic_term + potential_term


def stationarity_residual(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float = 1.0,
    g: float = 9.8,
    dt: float = 0.01,
) -> torch.Tensor:
    """返回隐式欧拉变分问题的一阶驻点残差。"""

    residual = (m / dt**2) * (y - p_n - dt * v_n)
    gravity = torch.zeros_like(residual)
    gravity[..., 2] = m * g
    return residual + gravity


def stationarity_residual_norm(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float = 1.0,
    g: float = 9.8,
    dt: float = 0.01,
) -> torch.Tensor:
    """返回驻点残差二范数；支持单点与批量输入。"""

    return torch.linalg.vector_norm(
        stationarity_residual(y, p_n, v_n, m, g, dt),
        dim=-1,
    )


def newton_direction(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float = 1.0,
    g: float = 9.8,
    dt: float = 0.01,
) -> torch.Tensor:
    """当前严格凸二次问题的 Newton 更新方向。"""

    grad = stationarity_residual(y, p_n, v_n, m, g, dt)
    return -(dt**2 / m) * grad


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """根据实验配置创建 PyTorch 优化器。"""

    normalized_name = optimizer_name.lower()
    if normalized_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    if normalized_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    raise ValueError(
        f"Unsupported optimizer: {optimizer_name!r}. Expected 'sgd' or 'adam'."
    )


# ============================================================
# 3. 隐式规则网格训练集
# ============================================================


def flat_indices_to_grid_points(
    flat_indices: torch.Tensor,
    grid_spec: GridSpec,
    y_star: torch.Tensor,
) -> torch.Tensor:
    """
    将规则三维网格中的扁平索引转换为坐标，返回 [B, 3] 张量。

    返回张量与 y_star 位于同一设备。网格轴坐标使用
    linspace(-R, R, points_per_axis) 的等价公式。points_per_axis
    为偶数，因此任一坐标轴都不会采到 0。
    """

    if flat_indices.ndim != 1:
        raise ValueError("flat_indices must be a one-dimensional tensor.")

    device = y_star.device
    n = grid_spec.points_per_axis
    n_squared = n * n
    flat_indices = flat_indices.to(dtype=torch.int64, device=device)

    index_x = torch.div(flat_indices, n_squared, rounding_mode="floor")
    remainder = torch.remainder(flat_indices, n_squared)
    index_y = torch.div(remainder, n, rounding_mode="floor")
    index_z = torch.remainder(remainder, n)

    points = torch.empty(
        (flat_indices.shape[0], 3),
        dtype=TORCH_DTYPE,
        device=device,
    )
    spacing = grid_spec.axis_spacing
    radius = grid_spec.sampling_radius
    points[:, 0] = y_star[0] - radius + index_x.to(TORCH_DTYPE) * spacing
    points[:, 1] = y_star[1] - radius + index_y.to(TORCH_DTYPE) * spacing
    points[:, 2] = y_star[2] - radius + index_z.to(TORCH_DTYPE) * spacing
    return points


def precompute_regular_grid_on_device(
    *,
    grid_spec: GridSpec,
    y_star: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    分块生成完整规则网格，并缓存到 y_star 所在设备。

    最大默认网格含 10^6 个 float32 三维点，占用约 11.44 MiB。
    分块生成可避免同时创建过大的临时索引和坐标张量。
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    num_points = grid_spec.actual_num_points
    device = y_star.device
    training_grid = torch.empty(
        (num_points, 3),
        dtype=TORCH_DTYPE,
        device=device,
    )

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        flat_indices = torch.arange(
            start,
            end,
            dtype=torch.int64,
            device=device,
        )
        training_grid[start:end] = flat_indices_to_grid_points(
            flat_indices=flat_indices,
            grid_spec=grid_spec,
            y_star=y_star,
        )

    return training_grid



def sample_grid_points_for_plot(
    training_grid: torch.Tensor,
    max_points: int = MAX_SCATTER_POINTS,
) -> torch.Tensor:
    """从设备端缓存网格中均匀抽取少量点，仅用于绘图。"""

    num_points = training_grid.shape[0]
    num_shown = min(num_points, max_points)
    device = training_grid.device
    if num_shown == num_points:
        flat_indices = torch.arange(num_points, dtype=torch.int64, device=device)
    else:
        flat_indices = torch.linspace(
            0,
            num_points - 1,
            steps=num_shown,
            dtype=torch.float64,
            device=device,
        ).round().to(dtype=torch.int64)
    return training_grid.index_select(0, flat_indices)


def grid_contains_exact_center(grid_spec: GridSpec) -> bool:
    """偶数轴向点数时应恒为 False。"""

    return grid_spec.points_per_axis % 2 == 1


def compute_regular_grid_input_normalizer(
    *,
    grid_spec: GridSpec,
    y_star: torch.Tensor,
    history: torch.Tensor,
    params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    解析计算当前规则网格对应的逐特征输入归一化统计量。

    输入顺序为：
        [y, p_n, v_n, m, g, dt]

    只有 y 的三个坐标分量随训练样本变化。history 和 params 在当前
    固定物理问题中保持不变，因此其真实标准差为 0；与原实验一致，
    将这些固定特征的标准差替换为 1，避免除零。
    """

    n = grid_spec.points_per_axis
    radius = grid_spec.sampling_radius

    # 对 linspace(-R, R, n) 使用总体标准差（unbiased=False）。
    # 对称网格的均值为 0，方差为 R^2 * (n + 1) / (3 * (n - 1))。
    y_std_value = radius * math.sqrt((n + 1.0) / (3.0 * (n - 1.0)))

    input_mean = torch.cat(
        [
            y_star.detach().cpu(),
            history.detach().cpu(),
            params.detach().cpu(),
        ],
        dim=0,
    ).to(dtype=TORCH_DTYPE)

    input_std = torch.cat(
        [
            torch.full((3,), y_std_value, dtype=TORCH_DTYPE),
            torch.ones(9, dtype=TORCH_DTYPE),
        ],
        dim=0,
    )

    return input_mean, input_std


# ============================================================
# 4. 训练与评估
# ============================================================


def evaluate_reference_trajectory(
    model: MLPOptimizer,
    initial_y: torch.Tensor,
    history: torch.Tensor,
    params: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
    e_star: float,
    num_steps: int,
) -> dict:
    """冻结网络后，只从真实测试初值 p_n 出发展开轨迹。"""

    y = initial_y.clone()
    iterations = []

    for step in range(num_steps + 1):
        energy = float(variational_energy(y, p_n, v_n, m, g, dt).item())
        residual_norm = float(
            stationarity_residual_norm(y, p_n, v_n, m, g, dt).item()
        )
        iterations.append(
            {
                "step": step,
                "y": tensor_to_list(y),
                "energy": energy,
                "gap": energy - e_star,
                "residual_norm": residual_norm,
            }
        )

        if step == num_steps:
            break

        with torch.no_grad():
            delta = model(y, history, params)
            y = y + delta
            iterations[-1]["next_delta_norm"] = float(torch.norm(delta).item())

    return {
        "initial_y": tensor_to_list(initial_y),
        "iterations": iterations,
    }


def evaluate_newton_trajectory(
    initial_y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
    e_star: float,
    num_steps: int,
) -> dict:
    """从相同初值出发记录 Newton 轨迹。"""

    y = initial_y.clone()
    iterations = []

    for step in range(num_steps + 1):
        energy = float(variational_energy(y, p_n, v_n, m, g, dt).item())
        residual_norm = float(
            stationarity_residual_norm(y, p_n, v_n, m, g, dt).item()
        )
        iterations.append(
            {
                "step": step,
                "y": tensor_to_list(y),
                "energy": energy,
                "gap": energy - e_star,
                "residual_norm": residual_norm,
            }
        )

        if step == num_steps:
            break

        delta = newton_direction(y, p_n, v_n, m, g, dt)
        y = y + delta
        iterations[-1]["next_delta_norm"] = float(torch.norm(delta).item())

    return {
        "initial_y": tensor_to_list(initial_y),
        "iterations": iterations,
    }


def run_experiment(
    *,
    base_output_dir: Path,
    grid_spec: GridSpec,
    training_grid: torch.Tensor,
    optimizer_name: str,
    learning_rate: float,
    config: RuntimeConfig,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> dict:
    """运行一组“优化器 × 数据规模”实验，以参数更新次数作为停止条件。"""

    dataset_size = grid_spec.actual_num_points
    use_full_batch = dataset_size < config.batch_size
    effective_batch_size = dataset_size if use_full_batch else config.batch_size
    training_mode = (
        "full_batch_because_dataset_size_is_smaller_than_batch_size"
        if use_full_batch
        else "cycling_shuffled_minibatch_without_replacement"
    )

    experiment_name = (
        f"{optimizer_name}_lr_{learning_rate:.0e}_"
        f"grid_axis_{grid_spec.points_per_axis}_"
        f"num_samples_{dataset_size}_"
        f"batch_{effective_batch_size}_"
        f"updates_{config.update_steps}_"
        f"max_k_{config.max_k}"
    )
    output_dir = base_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🧪 实验组: {experiment_name}")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 80)

    requested_device = torch.device(config.device)
    if training_grid.device.type != requested_device.type:
        raise ValueError(
            "training_grid must already reside on the configured training device: "
            f"training_grid.device={training_grid.device}, "
            f"config.device={requested_device}."
        )
    if requested_device.index is not None and training_grid.device.index != requested_device.index:
        raise ValueError(
            "training_grid CUDA index does not match the configured device: "
            f"training_grid.device={training_grid.device}, "
            f"config.device={requested_device}."
        )

    device = training_grid.device
    if training_grid.shape != (dataset_size, 3):
        raise ValueError(
            "training_grid shape does not match grid_spec: "
            f"training_grid.shape={tuple(training_grid.shape)}, "
            f"expected={(dataset_size, 3)}."
        )

    p_n_device = p_n.to(device)
    v_n_device = v_n.to(device)
    y_star = p_n_device + dt * v_n_device - dt**2 * torch.tensor(
        [0.0, 0.0, g], dtype=TORCH_DTYPE, device=device
    )
    history = torch.cat([p_n_device, v_n_device])
    params = torch.tensor([m, g, dt], dtype=TORCH_DTYPE, device=device)

    e_star = float(variational_energy(y_star, p_n_device, v_n_device, m, g, dt).item())
    newton_solution = p_n_device + newton_direction(
        p_n_device, p_n_device, v_n_device, m, g, dt
    )

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)

    input_mean, input_std = compute_regular_grid_input_normalizer(
        grid_spec=grid_spec,
        y_star=y_star,
        history=history,
        params=params,
    )
    model = MLPOptimizer(
        use_input_normalization=USE_INPUT_NORMALIZATION,
        use_output_dt_scaling=USE_OUTPUT_DT_SCALING,
        input_mean=input_mean,
        input_std=input_std,
    ).to(device)
    optimizer = create_optimizer(model, optimizer_name, learning_rate)

    batch_sampler = None
    if not use_full_batch:
        batch_sampler = CyclingShuffledBatchSampler(
            num_samples=dataset_size,
            batch_size=effective_batch_size,
            device=device,
            seed=BATCH_RANDOM_SEED,
        )

    train_log: list[dict] = []
    eval_log: list[dict] = []
    diverged = False
    divergence_reason = None
    divergence_update_step = None
    cumulative_seen_samples = 0
    successful_update_steps = 0

    print(f"device={device}")
    print(f"torch_dtype={TORCH_DTYPE}")
    print(f"target_dataset_size={grid_spec.target_num_points}")
    print(f"actual_dataset_size={dataset_size}")
    print(f"points_per_axis={grid_spec.points_per_axis}")
    print(f"axis_spacing={grid_spec.axis_spacing:.8e}")
    print(f"training_mode={training_mode}")
    print(f"configured_batch_size={config.batch_size}")
    print(f"effective_batch_size={effective_batch_size}")
    print(f"total_backward_update_steps={config.update_steps}")
    print("termination_condition=successful backward + optimizer.step count")
    print("epoch_is_not_used_as_training_counter")
    print(f"training_grid_device={training_grid.device}")
    print(f"sampling_center=y_star={tensor_to_list(y_star)}")
    print(f"sampling_radius={config.sampling_radius}")
    print("sampling_distribution=regular_cartesian_grid")
    print("training_set_contains_y_star=False")
    print(f"use_input_normalization={USE_INPUT_NORMALIZATION}")
    print(f"use_output_dt_scaling={USE_OUTPUT_DT_SCALING}")
    print(f"input_mean={tensor_to_list(model.input_mean)}")
    print(f"input_std={tensor_to_list(model.input_std)}")
    print("loss=sum_of_stepwise_mean_variational_energy_over_current_batch")
    print(
        "trajectory_backpropagation=full_unroll_without_detach; "
        "one backward and one optimizer.step per global update step"
    )
    print(
        f"K schedule: initial={config.initial_k}, "
        f"increase +{config.k_increase_amount} every "
        f"{config.k_increase_interval_updates} update steps, max={config.max_k}"
    )

    for update_index in range(config.update_steps):
        global_update_step = update_index + 1
        model.train()
        k = get_k_for_update_step(update_index, config)

        if use_full_batch:
            training_batch = training_grid
            completed_dataset_passes = global_update_step
            current_pass_fraction = 0.0
        else:
            assert batch_sampler is not None
            batch_indices = batch_sampler.next_indices()
            training_batch = training_grid.index_select(0, batch_indices)
            completed_dataset_passes = batch_sampler.completed_dataset_passes
            current_pass_fraction = batch_sampler.current_pass_fraction

        actual_batch_size = int(training_batch.shape[0])
        cumulative_seen_samples += actual_batch_size

        y = training_batch
        optimizer.zero_grad(set_to_none=True)
        trajectory_loss = torch.zeros((), dtype=TORCH_DTYPE, device=device)
        for _ in range(k):
            delta = model(y, history, params)
            y = y + delta
            trajectory_loss = trajectory_loss + variational_energy(
                y, p_n_device, v_n_device, m, g, dt
            ).mean()

        if not bool(torch.isfinite(trajectory_loss)):
            diverged = True
            divergence_reason = "non-finite mini-batch trajectory loss"
            divergence_update_step = global_update_step
        else:
            try:
                trajectory_loss.backward()
                optimizer.step()
                successful_update_steps += 1
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    diverged = True
                    divergence_reason = f"CUDA out of memory during mini-batch training: {error}"
                    divergence_update_step = global_update_step
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

        if not diverged and not is_model_finite(model):
            diverged = True
            divergence_reason = "non-finite model parameter after optimizer.step"
            divergence_update_step = global_update_step

        if diverged:
            print(
                f"⚠️ 训练失败：global_update_step={divergence_update_step}, "
                f"reason={divergence_reason}"
            )
            break

        mean_training_trajectory_loss = float(trajectory_loss.item())
        training_gap_for_readability = mean_training_trajectory_loss - k * e_star
        equivalent_dataset_passes = cumulative_seen_samples / dataset_size

        train_log.append(
            {
                "global_update_step": global_update_step,
                "zero_based_update_index": update_index,
                "K": k,
                "training_mode": training_mode,
                "configured_batch_size": config.batch_size,
                "actual_batch_size": actual_batch_size,
                "successful_backward_update_steps": successful_update_steps,
                "cumulative_seen_samples": cumulative_seen_samples,
                "equivalent_dataset_passes": equivalent_dataset_passes,
                "completed_shuffled_dataset_passes": completed_dataset_passes,
                "current_shuffled_pass_fraction": current_pass_fraction,
                "mean_training_trajectory_loss": mean_training_trajectory_loss,
                "mean_training_gap_for_readability": training_gap_for_readability,
            }
        )

        should_evaluate = (
            global_update_step == 1
            or global_update_step % config.eval_interval_updates == 0
            or global_update_step == config.update_steps
        )
        if should_evaluate:
            model.eval()
            reference_trajectory = evaluate_reference_trajectory(
                model=model,
                initial_y=p_n_device,
                history=history,
                params=params,
                p_n=p_n_device,
                v_n=v_n_device,
                m=m,
                g=g,
                dt=dt,
                e_star=e_star,
                num_steps=config.max_k,
            )
            final_item = reference_trajectory["iterations"][-1]
            eval_log.append(
                {
                    "global_update_step": global_update_step,
                    "zero_based_update_index": update_index,
                    "K": k,
                    "evaluation_steps": config.max_k,
                    "actual_batch_size": actual_batch_size,
                    "cumulative_seen_samples": cumulative_seen_samples,
                    "equivalent_dataset_passes": equivalent_dataset_passes,
                    "reference_gap": final_item["gap"],
                    "reference_residual_norm": final_item["residual_norm"],
                    "reference_trajectory": reference_trajectory,
                }
            )
            print(
                f"Update {global_update_step:6d}/{config.update_steps} | "
                f"K={k} | batch={actual_batch_size} | "
                f"equiv_passes={equivalent_dataset_passes:.4f} | "
                f"train energy sum={mean_training_trajectory_loss:.8e} | "
                f"train gap={training_gap_for_readability:.4e} | "
                f"p_n gap({config.max_k} steps)={final_item['gap']:.4e} | "
                f"p_n residual={final_item['residual_norm']:.4e}"
            )

    print("✅ 训练完成。开始冻结网络并生成最终结果。")

    model.eval()
    final_mlp_trajectory = evaluate_reference_trajectory(
        model=model,
        initial_y=p_n_device,
        history=history,
        params=params,
        p_n=p_n_device,
        v_n=v_n_device,
        m=m,
        g=g,
        dt=dt,
        e_star=e_star,
        num_steps=config.final_test_steps,
    )
    final_newton_trajectory = evaluate_newton_trajectory(
        initial_y=p_n_device,
        p_n=p_n_device,
        v_n=v_n_device,
        m=m,
        g=g,
        dt=dt,
        e_star=e_star,
        num_steps=config.final_test_steps,
    )

    final_mlp_item = final_mlp_trajectory["iterations"][-1]
    final_newton_item = final_newton_trajectory["iterations"][-1]
    final_reference_is_finite = (
        math.isfinite(float(final_mlp_item["gap"]))
        and math.isfinite(float(final_mlp_item["residual_norm"]))
    )

    model_path = output_dir / "mlp_optimizer_state_dict.pt"
    torch.save(model.state_dict(), model_path)
    final_equivalent_dataset_passes = cumulative_seen_samples / dataset_size

    report = {
        "config": {
            "experiment_name": experiment_name,
            "torch_dtype": str(TORCH_DTYPE),
            "device": str(device),
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "target_dataset_size": grid_spec.target_num_points,
            "actual_dataset_size": dataset_size,
            "points_per_axis": grid_spec.points_per_axis,
            "axis_spacing": grid_spec.axis_spacing,
            "dataset_mode": "device_cached_regular_grid_update_count_training",
            "sampling_center": "y_star",
            "sampling_radius_per_axis": config.sampling_radius,
            "training_set_contains_y_star": False,
            "grid_precompute_chunk_size": config.grid_precompute_chunk_size,
            "training_counter": "successful_backward_and_optimizer_step_count",
            "epochs_used_as_stopping_condition": False,
            "configured_update_steps": config.update_steps,
            "successful_update_steps": successful_update_steps,
            "training_mode": training_mode,
            "configured_batch_size": config.batch_size,
            "effective_batch_size": effective_batch_size,
            "full_batch_condition": "actual_dataset_size < configured_batch_size",
            "batch_sampling": (
                "For N >= batch_size, use cyclic shuffled sampling without replacement. "
                "A batch crossing a permutation boundary is completed from the next shuffle."
            ),
            "batch_random_seed": BATCH_RANDOM_SEED,
            "cumulative_seen_samples": cumulative_seen_samples,
            "equivalent_dataset_passes": final_equivalent_dataset_passes,
            "training_unroll_mode": (
                "fixed" if config.max_k == config.initial_k else "curriculum"
            ),
            "training_unroll_schedule": (
                "K=min(initial_K + floor(update_index / interval) * amount, max_K)."
            ),
            "initial_K": config.initial_k,
            "K_increase_interval_updates": config.k_increase_interval_updates,
            "K_increase_amount": config.k_increase_amount,
            "max_K": config.max_k,
            "use_input_normalization": USE_INPUT_NORMALIZATION,
            "use_output_dt_scaling": USE_OUTPUT_DT_SCALING,
            "input_mean": tensor_to_list(model.input_mean),
            "input_std": tensor_to_list(model.input_std),
            "loss": (
                "For the current batch, sum the mean variational energy of all "
                "unrolled steps. No additional loss and no division by K."
            ),
            "backpropagation": (
                "Full unrolled trajectory backpropagation without detach; one backward "
                "and one optimizer.step per global update step."
            ),
            "evaluation_interval_updates": config.eval_interval_updates,
            "final_test_steps": config.final_test_steps,
            "evaluation_mode": "held_out_reference_initial_state_p_n_only",
            "p_n": tensor_to_list(p_n_device),
            "v_n": tensor_to_list(v_n_device),
            "m": m,
            "g": g,
            "dt": dt,
            "y_star": tensor_to_list(y_star),
            "newton_solution": tensor_to_list(newton_solution),
            "E_star": e_star,
            "model_random_seed": MODEL_RANDOM_SEED,
        },
        "train_log": train_log,
        "periodic_reference_evaluation": eval_log,
        "final_reference_comparison": {
            "mlp": final_mlp_trajectory,
            "newton": final_newton_trajectory,
        },
        "training_status": {
            "diverged": diverged,
            "divergence_reason": divergence_reason,
            "divergence_update_step": divergence_update_step,
            "successful_update_steps": successful_update_steps,
            "requested_update_steps": config.update_steps,
            "completed_requested_updates": (
                not diverged and successful_update_steps == config.update_steps
            ),
            "final_reference_rollout_is_finite": final_reference_is_finite,
        },
        "summary": {
            "final_reference_gap_after_fixed_steps": final_mlp_item["gap"],
            "final_reference_residual_norm_after_fixed_steps": final_mlp_item["residual_norm"],
            "newton_reference_gap_after_fixed_steps": final_newton_item["gap"],
            "newton_reference_residual_norm_after_fixed_steps": final_newton_item["residual_norm"],
        },
    }

    report_path = output_dir / "optimization_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(report), file, indent=2, ensure_ascii=False)

    plot_training_curves(
        train_log=train_log,
        eval_log=eval_log,
        save_path=output_dir / "training_and_reference_eval_curves.png",
    )
    plot_reference_residual_comparison(
        mlp_trajectory=final_mlp_trajectory,
        newton_trajectory=final_newton_trajectory,
        save_path=output_dir / "final_reference_residual_comparison.png",
    )
    plot_reference_trajectory_3d(
        mlp_trajectory=final_mlp_trajectory,
        newton_solution=tensor_to_list(newton_solution),
        initial_y=tensor_to_list(p_n_device),
        save_path=output_dir / "final_reference_trajectory_3d.png",
    )
    plot_training_dataset_sample(
        grid_spec=grid_spec,
        training_grid=training_grid,
        y_star=tensor_to_list(y_star),
        initial_y=tensor_to_list(p_n_device),
        save_path=output_dir / "training_dataset_sample.png",
    )
    if not config.skip_contour:
        plot_reference_energy_contour_2d(
            mlp_trajectory=final_mlp_trajectory,
            newton_trajectory=final_newton_trajectory,
            y_star=tensor_to_list(y_star),
            initial_y=tensor_to_list(p_n_device),
            p_n=tensor_to_list(p_n_device),
            v_n=tensor_to_list(v_n_device),
            m=m,
            g=g,
            dt=dt,
            save_path=output_dir / "final_reference_energy_contour_2d.png",
        )

    print(f"📄 报告: {report_path}")
    print(f"💾 网络参数: {model_path}")
    print(
        f"最终 p_n 测试：gap={final_mlp_item['gap']:.4e}, "
        f"residual={final_mlp_item['residual_norm']:.4e}"
    )

    return {
        "experiment_name": experiment_name,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "target_dataset_size": grid_spec.target_num_points,
        "dataset_size": dataset_size,
        "points_per_axis": grid_spec.points_per_axis,
        "axis_spacing": grid_spec.axis_spacing,
        "training_mode": training_mode,
        "configured_batch_size": config.batch_size,
        "effective_batch_size": effective_batch_size,
        "requested_update_steps": config.update_steps,
        "successful_update_steps": successful_update_steps,
        "cumulative_seen_samples": cumulative_seen_samples,
        "equivalent_dataset_passes": final_equivalent_dataset_passes,
        "initial_K": config.initial_k,
        "max_K": config.max_k,
        "sampling_radius": config.sampling_radius,
        "diverged": diverged,
        "divergence_reason": divergence_reason,
        "divergence_update_step": divergence_update_step,
        "final_reference_rollout_is_finite": final_reference_is_finite,
        "final_reference_gap_after_fixed_steps": final_mlp_item["gap"],
        "final_reference_residual_norm_after_fixed_steps": final_mlp_item["residual_norm"],
        "newton_reference_gap_after_fixed_steps": final_newton_item["gap"],
        "newton_reference_residual_norm_after_fixed_steps": final_newton_item["residual_norm"],
        "output_directory": str(output_dir),
    }


# ============================================================
# 5. 绘图
# ============================================================


def plot_training_curves(
    train_log: Sequence[dict],
    eval_log: Sequence[dict],
    save_path: Path,
) -> None:
    """按全局反向传播/参数更新次数绘制训练与 p_n 测试曲线。"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    train_steps = [item["global_update_step"] for item in train_log]
    train_gaps = [
        finite_plot_value(item["mean_training_gap_for_readability"])
        for item in train_log
    ]
    axes[0].plot(train_steps, train_gaps)
    axes[0].set_yscale("log")
    axes[0].set_title("Training Trajectory Energy-Sum Gap")
    axes[0].set_xlabel("Backward / optimizer update step")
    axes[0].set_ylabel(r"Batch mean $\sum_k [E(y^{(k)})-E(y^*)]$")
    axes[0].grid(True, alpha=0.3)

    eval_steps = [item["global_update_step"] for item in eval_log]
    eval_gaps = [finite_plot_value(item["reference_gap"]) for item in eval_log]
    axes[1].plot(eval_steps, eval_gaps, marker="o", markersize=3)
    axes[1].set_yscale("log")
    axes[1].set_title(r"Held-Out $p_n$ Test Gap")
    axes[1].set_xlabel("Backward / optimizer update step")
    axes[1].set_ylabel(r"$E(y)-E(y^*)$")
    axes[1].grid(True, alpha=0.3)

    eval_residuals = [
        finite_plot_value(item["reference_residual_norm"]) for item in eval_log
    ]
    axes[2].plot(eval_steps, eval_residuals, marker="s", markersize=3)
    axes[2].set_yscale("log")
    axes[2].set_title(r"Held-Out $p_n$ Test Residual")
    axes[2].set_xlabel("Backward / optimizer update step")
    axes[2].set_ylabel(r"$\|\nabla E(y)\|_2$")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_reference_residual_comparison(
    mlp_trajectory: dict,
    newton_trajectory: dict,
    save_path: Path,
) -> None:
    """绘制最终冻结评估中 MLP 与 Newton 的 residual 曲线。"""

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
    ax.set_title(r"Final Residual Comparison from Held-Out $p_n$")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Stationarity residual $\|\nabla E(y)\|_2$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    """让三维坐标轴具有相同尺度；自动忽略发散后的 NaN/Inf 坐标。"""

    points = finite_rows(points)
    if points.shape[0] == 0:
        points = np.zeros((1, 3), dtype=float)
    center = points.mean(axis=0)
    radius = max(float(np.ptp(points, axis=0).max()) / 2.0, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_reference_trajectory_3d(
    mlp_trajectory: dict,
    newton_solution: Sequence[float],
    initial_y: Sequence[float],
    save_path: Path,
) -> None:
    """绘制从 p_n 出发的最终三维测试轨迹。"""

    mlp_points = np.asarray(
        [item["y"] for item in mlp_trajectory["iterations"]], dtype=float
    )
    newton_point = np.asarray(newton_solution, dtype=float)
    initial_point = np.asarray(initial_y, dtype=float)

    mlp_points_finite = finite_rows(mlp_points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        mlp_points_finite[:, 0],
        mlp_points_finite[:, 1],
        mlp_points_finite[:, 2],
        "-o",
        linewidth=1.5,
        markersize=4,
        label="MLP test trajectory",
    )
    for step, point in enumerate(mlp_points_finite):
        ax.text(point[0], point[1], point[2], f"  {step}", fontsize=8)

    ax.scatter(
        initial_point[0],
        initial_point[1],
        initial_point[2],
        marker="x",
        s=140,
        linewidths=2.0,
        label=r"Held-out initial state $p_n$",
    )
    ax.scatter(
        newton_point[0],
        newton_point[1],
        newton_point[2],
        marker="*",
        s=320,
        label="Newton converged solution",
    )
    set_equal_3d_axes(
        ax,
        np.vstack([mlp_points, initial_point.reshape(1, 3), newton_point.reshape(1, 3)]),
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_title(r"Final MLP Iteration Trajectory from Held-Out $p_n$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_dataset_sample(
    grid_spec: GridSpec,
    training_grid: torch.Tensor,
    y_star: Sequence[float],
    initial_y: Sequence[float],
    save_path: Path,
) -> None:
    """绘制规则训练网格的抽样分布，避免将约 10^8 个点全部绘制。"""

    sample = sample_grid_points_for_plot(
        training_grid=training_grid,
        max_points=MAX_SCATTER_POINTS,
    )
    sample_np = sample.detach().cpu().numpy()
    y_star_np = np.asarray(y_star, dtype=float)
    initial_y_np = np.asarray(initial_y, dtype=float)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        sample_np[:, 0],
        sample_np[:, 1],
        sample_np[:, 2],
        s=5,
        alpha=0.25,
        label=(
            f"Regular-grid samples "
            f"(shown: {sample_np.shape[0]}/{grid_spec.actual_num_points})"
        ),
    )
    ax.scatter(
        y_star_np[0],
        y_star_np[1],
        y_star_np[2],
        marker="*",
        s=320,
        label=r"Sampling center $y^*$ (excluded from training)",
    )
    ax.scatter(
        initial_y_np[0],
        initial_y_np[1],
        initial_y_np[2],
        marker="x",
        s=140,
        linewidths=2.0,
        label=r"Held-out test initial state $p_n$",
    )
    set_equal_3d_axes(
        ax,
        np.vstack([sample_np, y_star_np.reshape(1, 3), initial_y_np.reshape(1, 3)]),
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_title(
        "Fixed Regular Cartesian Training Grid near y*\n"
        f"{grid_spec.points_per_axis} points per axis, "
        f"{grid_spec.actual_num_points} total points"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reference_energy_contour_2d(
    mlp_trajectory: dict,
    newton_trajectory: dict,
    y_star: Sequence[float],
    initial_y: Sequence[float],
    p_n: Sequence[float],
    v_n: Sequence[float],
    m: float,
    g: float,
    dt: float,
    save_path: Path,
) -> None:
    """绘制 x-z 平面中的能量差等高线与最终测试轨迹投影。"""

    mlp_points = np.asarray([item["y"] for item in mlp_trajectory["iterations"]])
    newton_points = np.asarray([item["y"] for item in newton_trajectory["iterations"]])
    y_star_np = np.asarray(y_star, dtype=float)
    initial_np = np.asarray(initial_y, dtype=float)
    p_n_np = np.asarray(p_n, dtype=float)
    v_n_np = np.asarray(v_n, dtype=float)

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
        ax.text(point[0], point[2], f"  {step}", fontsize=8)
    ax.scatter(initial_np[0], initial_np[2], marker="x", s=120, linewidths=2.0, label=r"$p_n$")
    ax.scatter(y_star_np[0], y_star_np[2], marker="*", s=260, label=r"$y^*$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$z$")
    ax.set_title(r"Projected Final-Test Trajectories on $E(y)-E(y^*)$ Contours")
    ax.legend()
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label(r"Energy gap $E(y)-E(y^*)$")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_scale_summary(
    summaries: Sequence[dict],
    save_path: Path,
) -> None:
    """绘制不同训练数据规模下，从 p_n 出发的最终收敛结果。"""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for optimizer_config in OPTIMIZER_CONFIGS:
        optimizer_name = optimizer_config["optimizer_name"]
        learning_rate = optimizer_config["learning_rate"]
        selected = sorted(
            [
                item
                for item in summaries
                if item["optimizer_name"] == optimizer_name
                and item["learning_rate"] == learning_rate
            ],
            key=lambda item: item["dataset_size"],
        )
        sizes = [item["dataset_size"] for item in selected]
        gaps = [
            finite_plot_value(item["final_reference_gap_after_fixed_steps"])
            for item in selected
        ]
        residuals = [
            finite_plot_value(item["final_reference_residual_norm_after_fixed_steps"])
            for item in selected
        ]
        label = f"{optimizer_name.upper()} lr={learning_rate:.0e}"
        axes[0].plot(sizes, gaps, marker="o", label=label)
        axes[1].plot(sizes, residuals, marker="s", label=label)

    axes[0].set_title(r"Held-Out $p_n$ Final Gap vs. Training Dataset Size")
    axes[0].set_ylabel(r"$E(y)-E(y^*)$")
    axes[1].set_title(r"Held-Out $p_n$ Final Residual vs. Training Dataset Size")
    axes[1].set_ylabel(r"$\|\nabla E(y)\|_2$")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of regular-grid training initial states")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. 主程序
# ============================================================


def parse_args() -> argparse.Namespace:
    """解析命令行参数。默认值即为正式实验配置。"""

    parser = argparse.ArgumentParser(
        description=(
            "Free-fall MLP optimizer ablation using a fixed number of "
            "backward/optimizer update steps."
        )
    )
    parser.add_argument(
        "--target-dataset-sizes",
        "--dataset-sizes",
        dest="target_dataset_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_DATASET_SIZE_VALUES,
    )
    parser.add_argument("--sampling-radius", type=float, default=DEFAULT_SAMPLING_RADIUS)
    parser.add_argument(
        "--grid-precompute-chunk-size",
        type=int,
        default=DEFAULT_GRID_PRECOMPUTE_CHUNK_SIZE,
    )
    parser.add_argument(
        "--update-steps",
        "--backward-steps",
        dest="update_steps",
        type=int,
        default=DEFAULT_UPDATE_STEPS,
    )
    parser.add_argument(
        "--eval-interval-updates",
        "--eval-interval",
        dest="eval_interval_updates",
        type=int,
        default=DEFAULT_EVAL_INTERVAL_UPDATES,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--final-test-steps", type=int, default=DEFAULT_FINAL_TEST_STEPS)
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument(
        "--k-increase-interval-updates",
        "--k-increase-interval",
        dest="k_increase_interval_updates",
        type=int,
        default=DEFAULT_K_INCREASE_INTERVAL_UPDATES,
    )
    parser.add_argument("--k-increase-amount", type=int, default=DEFAULT_K_INCREASE_AMOUNT)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Training device, for example: cpu, cuda:0, cuda:1.",
    )
    parser.add_argument("--skip-contour", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    """检查命令行配置并转换为不可变数据结构。"""

    target_dataset_sizes = ensure_positive_int_list(args.target_dataset_sizes)
    if args.sampling_radius <= 0.0:
        raise ValueError("sampling_radius must be positive.")
    if args.grid_precompute_chunk_size <= 0:
        raise ValueError("grid_precompute_chunk_size must be positive.")
    if args.update_steps <= 0:
        raise ValueError("update_steps must be positive.")
    if args.eval_interval_updates <= 0:
        raise ValueError("eval_interval_updates must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.final_test_steps <= 0:
        raise ValueError("final_test_steps must be positive.")
    if args.initial_k <= 0:
        raise ValueError("initial_k must be positive.")
    if args.k_increase_interval_updates <= 0:
        raise ValueError("k_increase_interval_updates must be positive.")
    if args.k_increase_amount <= 0:
        raise ValueError("k_increase_amount must be positive.")
    if args.max_k < args.initial_k:
        raise ValueError("max_k must be >= initial_k.")

    return RuntimeConfig(
        target_dataset_sizes=target_dataset_sizes,
        sampling_radius=float(args.sampling_radius),
        grid_precompute_chunk_size=int(args.grid_precompute_chunk_size),
        update_steps=int(args.update_steps),
        eval_interval_updates=int(args.eval_interval_updates),
        batch_size=int(args.batch_size),
        final_test_steps=int(args.final_test_steps),
        initial_k=int(args.initial_k),
        k_increase_interval_updates=int(args.k_increase_interval_updates),
        k_increase_amount=int(args.k_increase_amount),
        max_k=int(args.max_k),
        device=str(args.device),
        skip_contour=bool(args.skip_contour),
    )



def main() -> None:
    config = validate_args(parse_args())
    base_output_dir = create_output_directory()

    print(f"📁 总输出目录: {base_output_dir}")
    print(f"运行配置: {asdict(config)}")
    print(f"torch default dtype: {torch.get_default_dtype()}")

    # 固定物理问题。
    m = 1.0
    g = 9.8
    dt = 0.01
    p_n = torch.tensor([3.0, 4.0, 5.0], dtype=TORCH_DTYPE)
    v_n = torch.tensor([0.5, -0.5, 0.0], dtype=TORCH_DTYPE)
    y_star = p_n + dt * v_n - dt**2 * torch.tensor([0.0, 0.0, g])

    grid_specs = make_grid_specs(
        target_dataset_sizes=config.target_dataset_sizes,
        sampling_radius=config.sampling_radius,
    )
    print("规则网格配置：")
    for grid_spec in grid_specs:
        print(
            f"- target={grid_spec.target_num_points:,}, "
            f"points_per_axis={grid_spec.points_per_axis}, "
            f"actual={grid_spec.actual_num_points:,}, "
            f"axis_spacing={grid_spec.axis_spacing:.8e}"
        )
        if grid_contains_exact_center(grid_spec):
            raise RuntimeError("The regular grid unexpectedly contains y_star.")

    experiment_summaries = []
    device = torch.device(config.device)
    y_star_device = y_star.to(device)

    for grid_spec in grid_specs:
        print(
            f"\n预生成设备端规则网格：actual_N={grid_spec.actual_num_points:,}, "
            f"device={device}, chunk_size={config.grid_precompute_chunk_size:,}"
        )
        training_grid = precompute_regular_grid_on_device(
            grid_spec=grid_spec,
            y_star=y_star_device,
            chunk_size=config.grid_precompute_chunk_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            f"设备端网格已就绪：shape={tuple(training_grid.shape)}, "
            f"memory={training_grid.numel() * training_grid.element_size() / 1024**2:.2f} MB"
        )

        for optimizer_config in OPTIMIZER_CONFIGS:
            summary = run_experiment(
                base_output_dir=base_output_dir,
                grid_spec=grid_spec,
                training_grid=training_grid,
                optimizer_name=optimizer_config["optimizer_name"],
                learning_rate=optimizer_config["learning_rate"],
                config=config,
                p_n=p_n,
                v_n=v_n,
                m=m,
                g=g,
                dt=dt,
            )
            experiment_summaries.append(summary)

        del training_grid
        if device.type == "cuda":
            torch.cuda.empty_cache()

    overall_report = {
        "experiment_type": (
            "fixed_local_regular_cartesian_grid_dataset_size_ablation_"
            "with_update_count_based_minibatch_training"
        ),
        "purpose": (
            "Train every experiment for the same number of backward/optimizer "
            "updates. Use full-batch when N < batch size and cyclic shuffled "
            "mini-batches otherwise."
        ),
        "runtime_config": asdict(config),
        "model_random_seed": MODEL_RANDOM_SEED,
        "batch_random_seed": BATCH_RANDOM_SEED,
        "sampling_center": "y_star",
        "sampling_radius_per_axis": config.sampling_radius,
        "sampling_distribution": (
            "Cartesian product of three uniform linspace axes over [-R, R]; "
            "each axis uses an even number of points."
        ),
        "training_set_contains_y_star": False,
        "grid_specs": [asdict(spec) for spec in grid_specs],
        "training_counter": "successful_backward_and_optimizer_step_count",
        "epochs_used_as_stopping_condition": False,
        "configured_update_steps": config.update_steps,
        "configured_batch_size": config.batch_size,
        "training_mode": (
            "full_batch_if_N_lt_batch_size_else_"
            "cycling_shuffled_minibatch_without_replacement"
        ),
        "grid_precompute_chunk_size": config.grid_precompute_chunk_size,
        "use_input_normalization": USE_INPUT_NORMALIZATION,
        "use_output_dt_scaling": USE_OUTPUT_DT_SCALING,
        "loss": (
            "For the current batch, sum the mean variational energy of all "
            "unrolled steps. No additional loss and no division by K."
        ),
        "optimizer_configs": OPTIMIZER_CONFIGS,
        "experiments": experiment_summaries,
    }

    summary_path = base_output_dir / "dataset_scale_ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(overall_report), file, indent=2, ensure_ascii=False)

    summary_plot_path = base_output_dir / "dataset_scale_ablation_summary.png"
    plot_dataset_scale_summary(experiment_summaries, summary_plot_path)

    print("\n" + "=" * 80)
    print("✅ 所有数据规模实验完成。")
    print(f"📄 汇总 JSON: {summary_path}")
    print(f"🖼️ 汇总图片: {summary_plot_path}")
    for item in experiment_summaries:
        print(
            f"- {item['experiment_name']}: "
            f"target_N={item['target_dataset_size']}, "
            f"actual_N={item['dataset_size']}, "
            f"points_per_axis={item['points_per_axis']}, "
            f"training_mode={item['training_mode']}, "
            f"batch={item['effective_batch_size']}, "
            f"updates={item['successful_update_steps']}/{item['requested_update_steps']}, "
            f"max_K={item['max_K']}, "
            f"training_diverged={item['diverged']}, "
            f"final_rollout_finite={item['final_reference_rollout_is_finite']}, "
            f"最终 p_n gap={item['final_reference_gap_after_fixed_steps']:.4e}, "
            f"最终 p_n residual={item['final_reference_residual_norm_after_fixed_steps']:.4e}"
        )


if __name__ == "__main__":
    main()
