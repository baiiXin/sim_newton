"""
自由落体单帧变分问题：1×32 MLP 激活函数消融实验
===================================================

本脚本以“Full-Batch 数据规模消融 + 验证集选择最佳 checkpoint + 独立测试集评估”
为基础，只比较 1×32 MLP 隐藏层的激活函数。

正式实验设置
------------
1. 网络固定为 12 -> 32 -> activation -> 3；
2. 激活函数测试 identity、relu、leaky_relu、elu、tanh、gelu、silu；
3. 训练集目标规模固定为 8、1,000、100,000；规则网格实际规模为 8、1,000、97,336；
4. 优化器固定为 SGD(lr=1e-2) 与 Adam(lr=1e-4)；
5. 共 3 × 2 × 7 = 42 组实验；
6. 数值精度为 torch.float64，默认设备为 cuda:1；
7. 每个 epoch 使用完整训练规则网格做一次 Full-Batch 更新；
8. K 从 1 开始，每 10,000 epoch 增加 1，最高 K=5；
9. 每组完整训练 50,000 epoch，不执行 early stopping；
10. 所有实验共享固定的 1,024 点验证集和 3,072 点测试集；
11. 每 500 epoch 在验证集上固定展开 50 步，并用最终 residual p95 选择最佳 checkpoint；
12. 测试集仅在训练结束后使用；同时评估验证最优模型与最后 epoch 模型；
13. 每组实验重新使用同一个模型随机种子，使不同激活函数的线性层初始权重一致；
14. 最后一层始终零初始化，输入归一化和输出 dt 缩放保持不变。

默认运行
--------
    python free_fall_activation_function_ablation.py

快速 CPU 冒烟测试
-----------------
    python free_fall_activation_function_ablation.py \
        --device cpu --target-dataset-sizes 8 \
        --optimizer-configs adam:1e-4 \
        --activation-names identity relu \
        --epochs 2 --validation-interval 1 \
        --validation-size 16 --test-size 24 \
        --evaluation-steps 2 --skip-contour --skip-individual-plots
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 0. 默认实验参数
# ============================================================

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

PLOT_FLOOR = 1e-14
GRID_MATCH_TOL = 1e-10
MODEL_RANDOM_SEED = 42
HELDOUT_RANDOM_SEED = 20260617

USE_INPUT_NORMALIZATION = True
USE_OUTPUT_DT_SCALING = True

DEFAULT_TARGET_DATASET_SIZE_VALUES = [
    8,
    1_000,
    100_000,
]
DEFAULT_SAMPLING_RADIUS = 0.01
DEFAULT_EPOCHS = 50_000
DEFAULT_VALIDATION_INTERVAL = 500
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_EVALUATION_BATCH_SIZE = 8_192
DEFAULT_VALIDATION_SIZE = 1_024
DEFAULT_TEST_SIZE = 3_072
DEFAULT_HELDOUT_RADIUS_SCALE = 1.0

DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 10_000
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5
DEFAULT_GRID_PRECOMPUTE_CHUNK_SIZE = 1_000_000
DEFAULT_DEVICE = "cuda:1"
DEFAULT_SUMMARY_CURVE_POINTS = 1_000
MAX_SCATTER_POINTS = 8_000

ALL_OPTIMIZER_CONFIGS = [
    {"optimizer_name": "sgd", "learning_rate": 1e-2},
    {"optimizer_name": "adam", "learning_rate": 1e-4},
]

DEFAULT_ACTIVATION_NAMES = [
    "identity",
    "relu",
    "leaky_relu",
    "elu",
    "tanh",
    "gelu",
    "silu",
]


# ============================================================
# 1. 数据结构与通用函数
# ============================================================


@dataclass(frozen=True)
class RuntimeConfig:
    target_dataset_sizes: list[int]
    optimizer_configs: list[dict[str, Any]]
    activation_names: list[str]
    sampling_radius: float
    heldout_radius_scale: float
    grid_precompute_chunk_size: int
    epochs: int
    validation_interval: int
    evaluation_steps: int
    evaluation_batch_size: int
    validation_size: int
    test_size: int
    initial_k: int
    k_increase_interval: int
    k_increase_amount: int
    max_k: int
    device: str
    skip_contour: bool
    skip_individual_plots: bool


@dataclass(frozen=True)
class GridSpec:
    target_num_points: int
    points_per_axis: int
    actual_num_points: int
    sampling_radius: float
    axis_spacing: float


def create_output_directory() -> Path:
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent / script_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_json_safe(value: Any) -> Any:
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


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(data), file, indent=2, ensure_ascii=False)


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().cpu().tolist()


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def is_model_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def finite_plot_value(value: float | int | None) -> float:
    if value is None:
        return float("nan")
    value = float(value)
    if not math.isfinite(value):
        return float("nan")
    return max(value, PLOT_FLOOR)


def ensure_positive_int_list(values: Iterable[int]) -> list[int]:
    cleaned = sorted({int(value) for value in values})
    if not cleaned:
        raise ValueError("target_dataset_sizes must not be empty.")
    if cleaned[0] <= 0:
        raise ValueError("Every target dataset size must be positive.")
    return cleaned


def nearest_even_points_per_axis(target_num_points: int) -> int:
    if target_num_points <= 0:
        raise ValueError("target_num_points must be positive.")
    root = target_num_points ** (1.0 / 3.0)
    lower = max(2, 2 * int(math.floor(root / 2.0)))
    upper = max(2, lower + 2)
    return min({lower, upper}, key=lambda n: (abs(n**3 - target_num_points), n))


def make_grid_spec(target_num_points: int, sampling_radius: float) -> GridSpec:
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
    specs = [
        make_grid_spec(target, sampling_radius)
        for target in ensure_positive_int_list(target_dataset_sizes)
    ]
    actual_sizes = [spec.actual_num_points for spec in specs]
    if len(set(actual_sizes)) != len(actual_sizes):
        raise ValueError(
            "Different targets mapped to the same even regular-grid size. "
            "Choose more separated target values."
        )
    return specs


def get_k_for_epoch(epoch_index: int, config: RuntimeConfig) -> int:
    return min(
        config.initial_k
        + (epoch_index // config.k_increase_interval) * config.k_increase_amount,
        config.max_k,
    )


def downsample_log(records: Sequence[dict[str, Any]], max_points: int = DEFAULT_SUMMARY_CURVE_POINTS) -> list[dict[str, Any]]:
    if not records:
        return []
    if len(records) <= max_points:
        return copy.deepcopy(list(records))
    indices = np.linspace(0, len(records) - 1, num=max_points, dtype=int)
    indices = sorted(set(indices.tolist() + [len(records) - 1]))
    return [copy.deepcopy(records[index]) for index in indices]


def parse_optimizer_configs(values: Sequence[str] | None) -> list[dict[str, Any]]:
    if not values:
        return copy.deepcopy(ALL_OPTIMIZER_CONFIGS)

    parsed: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for raw in values:
        try:
            name_raw, lr_raw = raw.split(":", maxsplit=1)
            name = name_raw.strip().lower()
            learning_rate = float(lr_raw)
        except Exception as error:
            raise ValueError(
                f"Invalid optimizer config {raw!r}. Use forms such as adam:1e-3."
            ) from error
        if name not in {"sgd", "adam"}:
            raise ValueError(f"Unsupported optimizer {name!r}.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        key = (name, learning_rate)
        if key not in seen:
            parsed.append({"optimizer_name": name, "learning_rate": learning_rate})
            seen.add(key)
    return parsed


def parse_activation_names(values: Sequence[str] | None) -> list[str]:
    """解析激活函数列表，并保持用户给定顺序。"""

    allowed = set(DEFAULT_ACTIVATION_NAMES)
    if not values:
        return list(DEFAULT_ACTIVATION_NAMES)

    parsed: list[str] = []
    for raw in values:
        name = raw.strip().lower()
        if name not in allowed:
            raise ValueError(
                f"Unsupported activation {raw!r}. "
                f"Available activations: {', '.join(DEFAULT_ACTIVATION_NAMES)}."
            )
        if name not in parsed:
            parsed.append(name)
    if not parsed:
        raise ValueError("activation_names must not be empty.")
    return parsed


def create_activation(activation_name: str) -> nn.Module:
    """创建无额外可训练参数的隐藏层激活函数。"""

    name = activation_name.lower()
    if name == "identity":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "elu":
        return nn.ELU(alpha=1.0)
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {activation_name!r}")


# ============================================================
# 2. 网络与物理问题
# ============================================================


class MLPOptimizer(nn.Module):
    """1×32 学习型迭代器：12 -> 32 -> activation -> 3。"""

    def __init__(
        self,
        *,
        use_input_normalization: bool = True,
        use_output_dt_scaling: bool = True,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
        activation_name: str = "relu",
    ) -> None:
        super().__init__()
        self.use_input_normalization = bool(use_input_normalization)
        self.use_output_dt_scaling = bool(use_output_dt_scaling)
        self.activation_name = activation_name.lower()

        self.net = nn.Sequential(
            nn.Linear(12, 32),
            create_activation(self.activation_name),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        if input_mean is None:
            input_mean = torch.zeros(12, dtype=TORCH_DTYPE)
        if input_std is None:
            input_std = torch.ones(12, dtype=TORCH_DTYPE)
        self.register_buffer("input_mean", input_mean.clone().detach().to(TORCH_DTYPE))
        self.register_buffer("input_std", input_std.clone().detach().to(TORCH_DTYPE))

    @staticmethod
    def _expand_feature_for_batch(feature: torch.Tensor, batch_size: int) -> torch.Tensor:
        if feature.ndim == 1:
            return feature.unsqueeze(0).expand(batch_size, -1)
        if feature.ndim == 2 and feature.shape[0] == batch_size:
            return feature
        raise ValueError(
            f"Feature shape is incompatible: feature={tuple(feature.shape)}, batch={batch_size}."
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
            raise ValueError(f"Expected y shape [3] or [B,3], got {tuple(y.shape)}")

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
    kinetic = (m / (2.0 * dt**2)) * torch.sum(residual**2, dim=-1)
    potential = m * g * y[..., 2]
    return kinetic + potential


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
    return -(dt**2 / m) * stationarity_residual(y, p_n, v_n, m, g, dt)


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")


# ============================================================
# 3. 训练规则网格与归一化
# ============================================================


def flat_indices_to_grid_points(
    flat_indices: torch.Tensor,
    grid_spec: GridSpec,
    y_star: torch.Tensor,
) -> torch.Tensor:
    if flat_indices.ndim != 1:
        raise ValueError("flat_indices must be one-dimensional.")
    n = grid_spec.points_per_axis
    n2 = n * n
    flat_indices = flat_indices.to(dtype=torch.int64, device=y_star.device)
    ix = torch.div(flat_indices, n2, rounding_mode="floor")
    remainder = torch.remainder(flat_indices, n2)
    iy = torch.div(remainder, n, rounding_mode="floor")
    iz = torch.remainder(remainder, n)

    points = torch.empty(
        (flat_indices.shape[0], 3),
        dtype=TORCH_DTYPE,
        device=y_star.device,
    )
    lower = y_star - grid_spec.sampling_radius
    spacing = grid_spec.axis_spacing
    points[:, 0] = lower[0] + ix.to(TORCH_DTYPE) * spacing
    points[:, 1] = lower[1] + iy.to(TORCH_DTYPE) * spacing
    points[:, 2] = lower[2] + iz.to(TORCH_DTYPE) * spacing
    return points


def precompute_regular_grid_on_device(
    *,
    grid_spec: GridSpec,
    y_star: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    training_grid = torch.empty(
        (grid_spec.actual_num_points, 3),
        dtype=TORCH_DTYPE,
        device=y_star.device,
    )
    for start in range(0, grid_spec.actual_num_points, chunk_size):
        end = min(start + chunk_size, grid_spec.actual_num_points)
        flat_indices = torch.arange(start, end, dtype=torch.int64, device=y_star.device)
        training_grid[start:end] = flat_indices_to_grid_points(
            flat_indices,
            grid_spec,
            y_star,
        )
    return training_grid


def compute_regular_grid_input_normalizer(
    *,
    grid_spec: GridSpec,
    y_star: torch.Tensor,
    history: torch.Tensor,
    params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = grid_spec.points_per_axis
    radius = grid_spec.sampling_radius
    y_std_value = radius * math.sqrt((n + 1.0) / (3.0 * (n - 1.0)))
    input_mean = torch.cat(
        [y_star.detach().cpu(), history.detach().cpu(), params.detach().cpu()],
        dim=0,
    ).to(TORCH_DTYPE)
    input_std = torch.cat(
        [
            torch.full((3,), y_std_value, dtype=TORCH_DTYPE),
            torch.ones(9, dtype=TORCH_DTYPE),
        ],
        dim=0,
    )
    return input_mean, input_std


# ============================================================
# 4. 固定验证集 / 测试集
# ============================================================


def points_in_grid_mask(
    points: torch.Tensor,
    *,
    grid_spec: GridSpec,
    y_star: torch.Tensor,
    tol: float = GRID_MATCH_TOL,
) -> torch.Tensor:
    n = grid_spec.points_per_axis
    lower = y_star.to(dtype=points.dtype).unsqueeze(0) - grid_spec.sampling_radius
    coords = (points - lower) / grid_spec.axis_spacing
    rounded = torch.round(coords)
    close_to_integer = torch.abs(coords - rounded) <= tol
    in_range = (rounded >= 0) & (rounded <= n - 1)
    return torch.all(close_to_integer & in_range, dim=1)


def build_fixed_heldout_set_excluding_training_grids(
    *,
    y_star: torch.Tensor,
    radius: float,
    num_points: int,
    seed: int,
    radius_scale: float,
    grid_specs: Sequence[GridSpec],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    if radius <= 0.0 or radius_scale <= 0.0:
        raise ValueError("radius and radius_scale must be positive.")

    effective_radius = radius * radius_scale
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    y_star_cpu = y_star.detach().cpu().to(TORCH_DTYPE)

    chunks: list[torch.Tensor] = []
    collected = 0
    generated = 0
    rejected = 0
    base_batch = max(2_048, min(65_536, num_points * 2))

    for _ in range(10_000):
        if collected >= num_points:
            break
        remaining = num_points - collected
        current_batch = max(base_batch, remaining * 2)
        offsets = (
            2.0 * torch.rand(
                (current_batch, 3),
                generator=generator,
                dtype=TORCH_DTYPE,
            )
            - 1.0
        ) * effective_radius
        candidates = y_star_cpu.unsqueeze(0) + offsets
        keep = torch.ones(current_batch, dtype=torch.bool)
        for grid_spec in grid_specs:
            keep &= ~points_in_grid_mask(
                candidates,
                grid_spec=grid_spec,
                y_star=y_star_cpu,
            )
        kept = candidates[keep]
        generated += current_batch
        rejected += int((~keep).sum().item())
        if kept.shape[0] > remaining:
            kept = kept[:remaining]
        if kept.numel() > 0:
            chunks.append(kept)
            collected += int(kept.shape[0])

    if collected < num_points:
        raise RuntimeError(
            f"Unable to build enough held-out points: requested={num_points}, collected={collected}."
        )

    points = torch.cat(chunks, dim=0)
    metadata = {
        "mode": "uniform_random_cube_near_y_star_excluding_all_training_grids",
        "seed": seed,
        "num_points": num_points,
        "base_radius": radius,
        "radius_scale": radius_scale,
        "effective_radius": effective_radius,
        "generated_candidates": generated,
        "rejected_training_overlap": rejected,
        "strictly_excludes_all_training_points": True,
        "training_grids_checked": [asdict(spec) for spec in grid_specs],
    }
    return points, metadata


def split_heldout_points(
    heldout_points: torch.Tensor,
    validation_size: int,
    test_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected = validation_size + test_size
    if heldout_points.shape != (expected, 3):
        raise ValueError(
            f"heldout_points shape mismatch: got {tuple(heldout_points.shape)}, expected {(expected, 3)}."
        )
    validation = heldout_points[:validation_size].clone()
    test = heldout_points[validation_size:].clone()
    return validation, test


# ============================================================
# 5. 批量评估
# ============================================================


def _safe_nan_stat(values: np.ndarray, function, default: float = float("nan")) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(function(finite))


def _statistics_by_step(values: np.ndarray, prefix: str) -> dict[str, Any]:
    mean_by_step: list[float] = []
    median_by_step: list[float] = []
    p95_by_step: list[float] = []
    max_by_step: list[float] = []
    nonfinite_by_step: list[int] = []

    for step_index in range(values.shape[1]):
        column = values[:, step_index]
        finite = column[np.isfinite(column)]
        nonfinite_by_step.append(int(column.size - finite.size))
        if finite.size == 0:
            mean_by_step.append(float("nan"))
            median_by_step.append(float("nan"))
            p95_by_step.append(float("nan"))
            max_by_step.append(float("nan"))
        else:
            mean_by_step.append(float(np.mean(finite)))
            median_by_step.append(float(np.median(finite)))
            p95_by_step.append(float(np.percentile(finite, 95)))
            max_by_step.append(float(np.max(finite)))

    final_values = values[:, -1]
    final_finite = final_values[np.isfinite(final_values)]
    result = {
        f"{prefix}_mean_by_step": mean_by_step,
        f"{prefix}_median_by_step": median_by_step,
        f"{prefix}_p95_by_step": p95_by_step,
        f"{prefix}_max_by_step": max_by_step,
        f"{prefix}_num_nonfinite_by_step": nonfinite_by_step,
        f"final_{prefix}_num_nonfinite": int(final_values.size - final_finite.size),
    }
    if final_finite.size == 0:
        result.update(
            {
                f"final_{prefix}_mean": float("nan"),
                f"final_{prefix}_median": float("nan"),
                f"final_{prefix}_p95": float("nan"),
                f"final_{prefix}_max": float("nan"),
            }
        )
    else:
        result.update(
            {
                f"final_{prefix}_mean": float(np.mean(final_finite)),
                f"final_{prefix}_median": float(np.median(final_finite)),
                f"final_{prefix}_p95": float(np.percentile(final_finite, 95)),
                f"final_{prefix}_max": float(np.max(final_finite)),
            }
        )
    return result


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
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    model.eval()
    p_n_device = p_n.to(device=device, dtype=TORCH_DTYPE)
    v_n_device = v_n.to(device=device, dtype=TORCH_DTYPE)
    y_star_device = y_star.to(device=device, dtype=TORCH_DTYPE)
    history = torch.cat([p_n_device, v_n_device])
    params = torch.tensor([m, g, dt], device=device, dtype=TORCH_DTYPE)
    e_star = variational_energy(y_star_device, p_n_device, v_n_device, m, g, dt)

    all_residuals: list[torch.Tensor] = []
    all_gaps: list[torch.Tensor] = []
    num_points = int(initial_points_cpu.shape[0])

    for start in range(0, num_points, batch_size):
        end = min(start + batch_size, num_points)
        y = initial_points_cpu[start:end].to(device=device, dtype=TORCH_DTYPE)
        batch_residuals: list[torch.Tensor] = []
        batch_gaps: list[torch.Tensor] = []

        for step in range(steps + 1):
            residual = stationarity_residual_norm(y, p_n_device, v_n_device, m, g, dt)
            gap = variational_energy(y, p_n_device, v_n_device, m, g, dt) - e_star
            batch_residuals.append(residual.detach().cpu())
            batch_gaps.append(gap.detach().cpu())
            if step == steps:
                break
            y = y + model(y, history, params)

        all_residuals.append(torch.stack(batch_residuals, dim=1))
        all_gaps.append(torch.stack(batch_gaps, dim=1))

    residuals = torch.cat(all_residuals, dim=0).numpy().astype(float)
    gaps = torch.cat(all_gaps, dim=0).numpy().astype(float)
    residuals[~np.isfinite(residuals)] = np.nan
    gaps[~np.isfinite(gaps)] = np.nan

    result: dict[str, Any] = {
        "steps": int(steps),
        "num_points": num_points,
    }
    result.update(_statistics_by_step(residuals, "residual"))
    result.update(_statistics_by_step(gaps, "loss_gap"))

    if num_points == 1:
        result["single_point_residual_by_step"] = [
            float(value) if math.isfinite(float(value)) else None
            for value in residuals[0].tolist()
        ]
        result["single_point_loss_gap_by_step"] = [
            float(value) if math.isfinite(float(value)) else None
            for value in gaps[0].tolist()
        ]
        result["single_point_final_residual"] = result["single_point_residual_by_step"][-1]
        result["single_point_final_loss_gap"] = result["single_point_loss_gap_by_step"][-1]

    return result


def validation_selection_key(validation_result: dict[str, Any]) -> tuple[float, ...] | None:
    p95 = float(validation_result["final_residual_p95"])
    median = float(validation_result["final_residual_median"])
    gap_p95 = float(validation_result["final_loss_gap_p95"])
    if not (math.isfinite(p95) and math.isfinite(median) and math.isfinite(gap_p95)):
        return None
    nonfinite = float(validation_result["final_residual_num_nonfinite"])
    return (nonfinite, p95, median, gap_p95)


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
) -> dict[str, Any]:
    model.eval()
    p_n_device = p_n.to(device)
    v_n_device = v_n.to(device)
    y_star_device = y_star.to(device)
    y = initial_y.to(device).clone()
    history = torch.cat([p_n_device, v_n_device])
    params = torch.tensor([m, g, dt], dtype=TORCH_DTYPE, device=device)
    e_star = variational_energy(y_star_device, p_n_device, v_n_device, m, g, dt)

    iterations: list[dict[str, Any]] = []
    for step in range(steps + 1):
        energy = variational_energy(y, p_n_device, v_n_device, m, g, dt)
        residual = stationarity_residual_norm(y, p_n_device, v_n_device, m, g, dt)
        iterations.append(
            {
                "step": step,
                "y": tensor_to_list(y),
                "energy": float(energy.item()),
                "gap": float((energy - e_star).item()),
                "residual_norm": float(residual.item()),
            }
        )
        if step == steps:
            break
        delta = model(y, history, params)
        y = y + delta
        iterations[-1]["next_delta_norm"] = float(torch.linalg.vector_norm(delta).item())
    return {"initial_y": tensor_to_list(initial_y), "iterations": iterations}


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
) -> dict[str, Any]:
    p_n_device = p_n.to(device)
    v_n_device = v_n.to(device)
    y_star_device = y_star.to(device)
    y = initial_y.to(device).clone()
    e_star = variational_energy(y_star_device, p_n_device, v_n_device, m, g, dt)
    iterations: list[dict[str, Any]] = []
    for step in range(steps + 1):
        energy = variational_energy(y, p_n_device, v_n_device, m, g, dt)
        residual = stationarity_residual_norm(y, p_n_device, v_n_device, m, g, dt)
        iterations.append(
            {
                "step": step,
                "y": tensor_to_list(y),
                "energy": float(energy.item()),
                "gap": float((energy - e_star).item()),
                "residual_norm": float(residual.item()),
            }
        )
        if step == steps:
            break
        delta = newton_direction(y, p_n_device, v_n_device, m, g, dt)
        y = y + delta
        iterations[-1]["next_delta_norm"] = float(torch.linalg.vector_norm(delta).item())
    return {"initial_y": tensor_to_list(initial_y), "iterations": iterations}


# ============================================================
# 6. 绘图
# ============================================================


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


def sample_regular_grid_for_plot(
    grid_spec: GridSpec,
    y_star: np.ndarray,
    max_points: int = MAX_SCATTER_POINTS,
) -> np.ndarray:
    total = grid_spec.actual_num_points
    if total <= max_points:
        indices = np.arange(total, dtype=np.int64)
    else:
        indices = np.linspace(0, total - 1, max_points).round().astype(np.int64)
    n = grid_spec.points_per_axis
    n2 = n * n
    ix = indices // n2
    remainder = indices % n2
    iy = remainder // n
    iz = remainder % n
    lower = y_star - grid_spec.sampling_radius
    points = np.empty((indices.size, 3), dtype=float)
    points[:, 0] = lower[0] + ix * grid_spec.axis_spacing
    points[:, 1] = lower[1] + iy * grid_spec.axis_spacing
    points[:, 2] = lower[2] + iz * grid_spec.axis_spacing
    return points


def plot_dataset_distribution_overview(
    *,
    grid_specs: Sequence[GridSpec],
    validation_points: torch.Tensor,
    test_points: torch.Tensor,
    y_star: Sequence[float],
    p_n: Sequence[float],
    save_path: Path,
) -> None:
    y_star_np = np.asarray(y_star, dtype=float)
    p_n_np = np.asarray(p_n, dtype=float)
    validation_np = validation_points.detach().cpu().numpy()
    test_np = test_points.detach().cpu().numpy()

    num_plots = len(grid_specs) + 1
    num_cols = 4
    num_rows = math.ceil(num_plots / num_cols)
    fig = plt.figure(figsize=(5.2 * num_cols, 4.9 * num_rows))

    for index, grid_spec in enumerate(grid_specs):
        ax = fig.add_subplot(num_rows, num_cols, index + 1, projection="3d")
        train = sample_regular_grid_for_plot(grid_spec, y_star_np)
        ax.scatter(train[:, 0], train[:, 1], train[:, 2], s=4, alpha=0.28, label=f"train ({train.shape[0]}/{grid_spec.actual_num_points})")
        ax.scatter(p_n_np[0], p_n_np[1], p_n_np[2], marker="x", s=100, linewidths=2, label=r"$p_n$")
        ax.scatter(y_star_np[0], y_star_np[1], y_star_np[2], marker="*", s=180, label=r"$y^*$")
        set_equal_3d_axes(ax, np.vstack([train, p_n_np[None, :], y_star_np[None, :]]))
        ax.set_title(f"Training grid N={grid_spec.actual_num_points:,}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(fontsize=7)

    ax = fig.add_subplot(num_rows, num_cols, len(grid_specs) + 1, projection="3d")
    ax.scatter(validation_np[:, 0], validation_np[:, 1], validation_np[:, 2], s=5, alpha=0.35, label=f"validation N={validation_np.shape[0]}")
    ax.scatter(test_np[:, 0], test_np[:, 1], test_np[:, 2], s=5, alpha=0.25, label=f"test N={test_np.shape[0]}")
    ax.scatter(p_n_np[0], p_n_np[1], p_n_np[2], marker="x", s=100, linewidths=2, label=r"$p_n$")
    ax.scatter(y_star_np[0], y_star_np[1], y_star_np[2], marker="*", s=180, label=r"$y^*$")
    set_equal_3d_axes(ax, np.vstack([validation_np, test_np, p_n_np[None, :], y_star_np[None, :]]))
    ax.set_title("Fixed held-out split\n(excludes every training grid)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=7)

    fig.suptitle("Training / validation / test distributions", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_training_and_validation_curves(
    *,
    train_log: Sequence[dict[str, Any]],
    validation_log: Sequence[dict[str, Any]],
    best_epoch: int | None,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    train_epochs = [record["epoch"] for record in train_log]
    train_gaps = [finite_plot_value(record["training_gap_for_readability"]) for record in train_log]
    axes[0].plot(train_epochs, train_gaps)
    axes[0].set_yscale("log")
    axes[0].set_title("Training trajectory energy-sum gap")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training gap")

    val_epochs = [record["epoch"] for record in validation_log]
    val_p95 = [finite_plot_value(record["metrics"]["final_residual_p95"]) for record in validation_log]
    val_median = [finite_plot_value(record["metrics"]["final_residual_median"]) for record in validation_log]
    axes[1].plot(val_epochs, val_p95, marker="o", label="residual p95")
    axes[1].plot(val_epochs, val_median, marker="s", label="residual median")
    axes[1].set_yscale("log")
    axes[1].set_title("Validation residual after fixed rollout")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Residual")
    axes[1].legend()

    val_gap_p95 = [finite_plot_value(record["metrics"]["final_loss_gap_p95"]) for record in validation_log]
    val_gap_median = [finite_plot_value(record["metrics"]["final_loss_gap_median"]) for record in validation_log]
    axes[2].plot(val_epochs, val_gap_p95, marker="o", label="gap p95")
    axes[2].plot(val_epochs, val_gap_median, marker="s", label="gap median")
    axes[2].set_yscale("log")
    axes[2].set_title("Validation energy gap after fixed rollout")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Energy gap")
    axes[2].legend()

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, linestyle="--", alpha=0.7, label="best validation epoch")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_pn_comparison(
    *,
    best_trajectory: dict[str, Any],
    last_trajectory: dict[str, Any],
    newton_trajectory: dict[str, Any],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for trajectory, label, linestyle in [
        (best_trajectory, "best validation checkpoint", "-"),
        (last_trajectory, "last epoch checkpoint", "--"),
        (newton_trajectory, "Newton", ":"),
    ]:
        steps = [item["step"] for item in trajectory["iterations"]]
        residuals = [finite_plot_value(item["residual_norm"]) for item in trajectory["iterations"]]
        gaps = [finite_plot_value(item["gap"]) for item in trajectory["iterations"]]
        axes[0].plot(steps, residuals, linestyle=linestyle, marker="o", markersize=3, label=label)
        axes[1].plot(steps, gaps, linestyle=linestyle, marker="o", markersize=3, label=label)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_title(r"$p_n$ residual")
    axes[1].set_title(r"$p_n$ energy gap")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_reference_energy_contour_2d(
    *,
    trajectory: dict[str, Any],
    newton_trajectory: dict[str, Any],
    y_star: Sequence[float],
    p_n: Sequence[float],
    v_n: Sequence[float],
    m: float,
    g: float,
    dt: float,
    save_path: Path,
) -> None:
    mlp_points = finite_rows(np.asarray([item["y"] for item in trajectory["iterations"]], dtype=float))
    newton_points = finite_rows(np.asarray([item["y"] for item in newton_trajectory["iterations"]], dtype=float))
    y_star_np = np.asarray(y_star, dtype=float)
    p_n_np = np.asarray(p_n, dtype=float)
    v_n_np = np.asarray(v_n, dtype=float)
    projected = np.vstack([mlp_points[:, [0, 2]], newton_points[:, [0, 2]], y_star_np[[0, 2]][None, :], p_n_np[[0, 2]][None, :]])
    lower = projected.min(axis=0)
    upper = projected.max(axis=0)
    span = np.maximum(upper - lower, 2e-4)
    lower -= 0.2 * span
    upper += 0.2 * span
    x_values = np.linspace(lower[0], upper[0], 200)
    z_values = np.linspace(lower[1], upper[1], 200)
    x_grid, z_grid = np.meshgrid(x_values, z_values)
    points = np.broadcast_to(y_star_np.reshape(1, 1, 3), (200, 200, 3)).copy()
    points[..., 0] = x_grid
    points[..., 2] = z_grid
    residual = points - p_n_np - dt * v_n_np
    energy = (m / (2.0 * dt**2)) * np.sum(residual**2, axis=-1) + m * g * points[..., 2]
    star_residual = y_star_np - p_n_np - dt * v_n_np
    e_star = (m / (2.0 * dt**2)) * np.sum(star_residual**2) + m * g * y_star_np[2]
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
    ax.plot(mlp_points[:, 0], mlp_points[:, 2], "-o", markersize=3, label="best validation checkpoint")
    ax.plot(newton_points[:, 0], newton_points[:, 2], "--s", markersize=3, label="Newton")
    ax.scatter(p_n_np[0], p_n_np[2], marker="x", s=100, linewidths=2, label=r"$p_n$")
    ax.scatter(y_star_np[0], y_star_np[2], marker="*", s=180, label=r"$y^*$")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Best-checkpoint trajectory on energy-gap contours")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.colorbar(contour, ax=ax, label=r"$E(y)-E(y^*)$")
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def optimizer_key(record: dict[str, Any]) -> tuple[str, float]:
    return str(record["optimizer_name"]).lower(), float(record["learning_rate"])


def optimizer_label(key: tuple[str, float]) -> str:
    return f"{key[0].upper()} lr={key[1]:.0e}"


def activation_label(name: str) -> str:
    labels = {
        "identity": "Identity",
        "relu": "ReLU",
        "leaky_relu": "LeakyReLU(0.01)",
        "elu": "ELU",
        "tanh": "Tanh",
        "gelu": "GELU",
        "silu": "SiLU",
    }
    return labels.get(name, name)


def unique_optimizer_keys(records: Sequence[dict[str, Any]]) -> list[tuple[str, float]]:
    preferred = {"sgd": 0, "adam": 1}
    keys = {optimizer_key(record) for record in records}
    return sorted(keys, key=lambda item: (preferred.get(item[0], 99), -item[1]))


def unique_activation_names(records: Sequence[dict[str, Any]]) -> list[str]:
    present = {str(record["activation_name"]) for record in records}
    return [name for name in DEFAULT_ACTIVATION_NAMES if name in present]


def plot_final_heldout_summary(
    records: Sequence[dict[str, Any]],
    save_path: Path,
    checkpoint_field: str = "best_checkpoint_test",
) -> None:
    """每个优化器占两行；每条曲线表示一种激活函数。"""

    optimizer_keys = unique_optimizer_keys(records)
    residual_metrics = [
        ("final_residual_mean", "Test mean residual"),
        ("final_residual_median", "Test median residual"),
        ("final_residual_p95", "Test p95 residual"),
        ("pn_final_residual", r"$p_n$ residual"),
    ]
    gap_metrics = [
        ("final_loss_gap_mean", "Test mean energy gap"),
        ("final_loss_gap_median", "Test median energy gap"),
        ("final_loss_gap_p95", "Test p95 energy gap"),
        ("pn_final_loss_gap", r"$p_n$ energy gap"),
    ]
    fig, axes = plt.subplots(
        2 * len(optimizer_keys),
        4,
        figsize=(22, 9.5 * len(optimizer_keys)),
        squeeze=False,
    )

    for optimizer_index, key in enumerate(optimizer_keys):
        optimizer_records = [r for r in records if optimizer_key(r) == key]
        for activation_name in unique_activation_names(optimizer_records):
            selected = sorted(
                [
                    record
                    for record in optimizer_records
                    if record["activation_name"] == activation_name
                ],
                key=lambda record: int(record["dataset_size"]),
            )
            sizes = [int(record["dataset_size"]) for record in selected]
            label = activation_label(activation_name)
            for ax, (metric, title) in zip(
                axes[2 * optimizer_index], residual_metrics
            ):
                values = [
                    finite_plot_value(record[checkpoint_field][metric])
                    for record in selected
                ]
                ax.plot(sizes, values, marker="o", label=label)
                ax.set_title(f"{optimizer_label(key)}\n{title}")
            for ax, (metric, title) in zip(
                axes[2 * optimizer_index + 1], gap_metrics
            ):
                values = [
                    finite_plot_value(record[checkpoint_field][metric])
                    for record in selected
                ]
                ax.plot(sizes, values, marker="o", label=label)
                ax.set_title(f"{optimizer_label(key)}\n{title}")

    for ax in axes.reshape(-1):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Actual training dataset size")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Activation-function comparison on held-out test data\n"
        "(validation-selected checkpoints)",
        y=1.002,
        fontsize=15,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss_summary(
    records: Sequence[dict[str, Any]],
    save_path: Path,
) -> None:
    """每个“优化器 × 数据规模”一个子图，曲线表示激活函数。"""

    optimizer_keys = unique_optimizer_keys(records)
    dataset_sizes = sorted({int(record["dataset_size"]) for record in records})
    fig, axes = plt.subplots(
        len(optimizer_keys),
        len(dataset_sizes),
        figsize=(6 * len(dataset_sizes), 4.8 * len(optimizer_keys)),
        squeeze=False,
        sharex=True,
    )
    for row, key in enumerate(optimizer_keys):
        for col, dataset_size in enumerate(dataset_sizes):
            ax = axes[row, col]
            selected = [
                record
                for record in records
                if optimizer_key(record) == key
                and int(record["dataset_size"]) == dataset_size
            ]
            for record in sorted(
                selected,
                key=lambda item: DEFAULT_ACTIVATION_NAMES.index(item["activation_name"]),
            ):
                curve = record["training_curve_for_summary"]
                ax.plot(
                    [point["epoch"] for point in curve],
                    [
                        finite_plot_value(point["training_gap_for_readability"])
                        for point in curve
                    ],
                    label=activation_label(record["activation_name"]),
                )
            ax.set_yscale("log")
            ax.set_title(f"{optimizer_label(key)}\nN={dataset_size:,}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training gap")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
    fig.suptitle("Training loss by activation function", y=1.01, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_validation_summary(
    records: Sequence[dict[str, Any]],
    save_path: Path,
) -> None:
    """每个“优化器 × 数据规模”一个子图，比较验证 residual p95。"""

    optimizer_keys = unique_optimizer_keys(records)
    dataset_sizes = sorted({int(record["dataset_size"]) for record in records})
    fig, axes = plt.subplots(
        len(optimizer_keys),
        len(dataset_sizes),
        figsize=(6 * len(dataset_sizes), 4.8 * len(optimizer_keys)),
        squeeze=False,
        sharex=True,
    )
    for row, key in enumerate(optimizer_keys):
        for col, dataset_size in enumerate(dataset_sizes):
            ax = axes[row, col]
            selected = [
                record
                for record in records
                if optimizer_key(record) == key
                and int(record["dataset_size"]) == dataset_size
            ]
            for record in sorted(
                selected,
                key=lambda item: DEFAULT_ACTIVATION_NAMES.index(item["activation_name"]),
            ):
                curve = record["validation_curve_for_summary"]
                ax.plot(
                    [point["epoch"] for point in curve],
                    [
                        finite_plot_value(point["metrics"]["final_residual_p95"])
                        for point in curve
                    ],
                    marker="o",
                    markersize=2,
                    label=activation_label(record["activation_name"]),
                )
            ax.set_yscale("log")
            ax.set_title(f"{optimizer_label(key)}\nN={dataset_size:,}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation residual p95")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
    fig.suptitle(
        "Validation curves used for checkpoint selection",
        y=1.01,
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_activation_ranking(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按每个优化器和数据规模的测试 residual p95 给激活函数排序。"""

    rankings: list[dict[str, Any]] = []
    for key in unique_optimizer_keys(records):
        dataset_sizes = sorted(
            {
                int(record["dataset_size"])
                for record in records
                if optimizer_key(record) == key
            }
        )
        for dataset_size in dataset_sizes:
            selected = [
                record
                for record in records
                if optimizer_key(record) == key
                and int(record["dataset_size"]) == dataset_size
            ]
            ordered = sorted(
                selected,
                key=lambda record: (
                    finite_plot_value(
                        record["best_checkpoint_test"]["final_residual_p95"]
                    ),
                    finite_plot_value(
                        record["best_checkpoint_test"]["final_residual_median"]
                    ),
                ),
            )
            rankings.append(
                {
                    "optimizer_name": key[0],
                    "learning_rate": key[1],
                    "dataset_size": dataset_size,
                    "ranking_metric": "best-checkpoint held-out test residual p95",
                    "ranking": [
                        {
                            "rank": rank,
                            "activation_name": record["activation_name"],
                            "test_residual_p95": record["best_checkpoint_test"][
                                "final_residual_p95"
                            ],
                            "test_residual_median": record["best_checkpoint_test"][
                                "final_residual_median"
                            ],
                            "best_validation_epoch": record[
                                "best_validation_epoch"
                            ],
                        }
                        for rank, record in enumerate(ordered, start=1)
                    ],
                }
            )
    return rankings


# ============================================================
# 7. 单组训练实验
# ============================================================


def run_experiment(
    *,
    base_output_dir: Path,
    grid_spec: GridSpec,
    training_grid: torch.Tensor,
    validation_points_cpu: torch.Tensor,
    test_points_cpu: torch.Tensor,
    optimizer_name: str,
    learning_rate: float,
    activation_name: str,
    config: RuntimeConfig,
    p_n_cpu: torch.Tensor,
    v_n_cpu: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> dict[str, Any]:
    dataset_size = grid_spec.actual_num_points
    experiment_name = (
        f"{optimizer_name}_lr_{learning_rate:.0e}_"
        f"activation_{activation_name}_"
        f"grid_axis_{grid_spec.points_per_axis}_num_samples_{dataset_size}"
    )
    output_dir = base_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = training_grid.device
    p_n = p_n_cpu.to(device)
    v_n = v_n_cpu.to(device)
    y_star = p_n + dt * v_n - dt**2 * torch.tensor(
        [0.0, 0.0, g],
        dtype=TORCH_DTYPE,
        device=device,
    )
    history = torch.cat([p_n, v_n])
    params = torch.tensor([m, g, dt], dtype=TORCH_DTYPE, device=device)
    e_star = float(variational_energy(y_star, p_n, v_n, m, g, dt).item())

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
        activation_name=activation_name,
    ).to(device)
    optimizer = create_optimizer(model, optimizer_name, learning_rate)

    print("\n" + "=" * 88)
    print(f"实验：{experiment_name}")
    print(
        f"device={device}, dtype={TORCH_DTYPE}, "
        f"architecture=12->32->{activation_name}->3"
    )
    print(f"training_N={dataset_size:,}, validation_N={config.validation_size:,}, test_N={config.test_size:,}")
    print(
        f"optimizer={optimizer_name}, lr={learning_rate:.0e}, "
        f"activation={activation_name}"
    )
    print("no_early_stopping=True; validation_selects_best_checkpoint_only")
    print("=" * 88)

    train_log: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    best_epoch: int | None = None
    diverged = False
    divergence_epoch: int | None = None
    divergence_reason: str | None = None
    start_time = time.perf_counter()

    for epoch_index in range(config.epochs):
        epoch_number = epoch_index + 1
        k = get_k_for_epoch(epoch_index, config)
        model.train()
        y = training_grid
        optimizer.zero_grad(set_to_none=True)
        trajectory_loss = torch.zeros((), dtype=TORCH_DTYPE, device=device)

        for _ in range(k):
            y = y + model(y, history, params)
            trajectory_loss = trajectory_loss + variational_energy(
                y,
                p_n,
                v_n,
                m,
                g,
                dt,
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
            print(f"训练终止：epoch={divergence_epoch}, reason={divergence_reason}")
            break

        loss_value = float(trajectory_loss.item())
        training_gap = loss_value - k * e_star
        train_log.append(
            {
                "epoch": epoch_number,
                "K": k,
                "trajectory_energy_sum": loss_value,
                "training_gap_for_readability": training_gap,
            }
        )

        should_validate = (
            epoch_number % config.validation_interval == 0
            or epoch_number == config.epochs
        )
        if should_validate:
            validation_metrics = evaluate_model_on_initial_set(
                model=model,
                initial_points_cpu=validation_points_cpu,
                p_n=p_n_cpu,
                v_n=v_n_cpu,
                y_star=y_star.detach().cpu(),
                m=m,
                g=g,
                dt=dt,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                device=device,
            )
            current_key = validation_selection_key(validation_metrics)
            validation_log.append(
                {
                    "epoch": epoch_number,
                    "training_K": k,
                    "selection_key": list(current_key) if current_key is not None else None,
                    "metrics": validation_metrics,
                }
            )
            if current_key is not None and (best_key is None or current_key < best_key):
                best_key = current_key
                best_epoch = epoch_number
                best_validation_metrics = copy.deepcopy(validation_metrics)
                best_state_dict = state_dict_to_cpu(model)

            elapsed = time.perf_counter() - start_time
            print(
                f"Epoch {epoch_number:5d} | K={k} | train_gap={training_gap:.4e} | "
                f"val_res_p95={validation_metrics['final_residual_p95']:.4e} | "
                f"val_res_median={validation_metrics['final_residual_median']:.4e} | "
                f"best_epoch={best_epoch} | elapsed={elapsed:.1f}s"
            )

    last_state_dict = state_dict_to_cpu(model)
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(last_state_dict)
        best_epoch = train_log[-1]["epoch"] if train_log else 0
        best_validation_metrics = None

    torch.save(last_state_dict, output_dir / "last_model_state_dict.pt")
    torch.save(best_state_dict, output_dir / "best_validation_model_state_dict.pt")
    # 保留一个主模型文件名；其内容明确为验证集选择出的 checkpoint。
    torch.save(best_state_dict, output_dir / "mlp_optimizer_state_dict.pt")

    def evaluate_checkpoint(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        test_eval = evaluate_model_on_initial_set(
            model=model,
            initial_points_cpu=test_points_cpu,
            p_n=p_n_cpu,
            v_n=v_n_cpu,
            y_star=y_star.detach().cpu(),
            m=m,
            g=g,
            dt=dt,
            steps=config.evaluation_steps,
            batch_size=config.evaluation_batch_size,
            device=device,
        )
        pn_eval = evaluate_model_on_initial_set(
            model=model,
            initial_points_cpu=p_n_cpu.reshape(1, 3),
            p_n=p_n_cpu,
            v_n=v_n_cpu,
            y_star=y_star.detach().cpu(),
            m=m,
            g=g,
            dt=dt,
            steps=config.evaluation_steps,
            batch_size=1,
            device=device,
        )
        pn_trajectory = evaluate_single_trajectory(
            model=model,
            initial_y=p_n_cpu,
            p_n=p_n_cpu,
            v_n=v_n_cpu,
            y_star=y_star.detach().cpu(),
            m=m,
            g=g,
            dt=dt,
            steps=config.evaluation_steps,
            device=device,
        )
        return test_eval, pn_eval, pn_trajectory

    best_test_eval, best_pn_eval, best_pn_trajectory = evaluate_checkpoint(best_state_dict)
    last_test_eval, last_pn_eval, last_pn_trajectory = evaluate_checkpoint(last_state_dict)
    newton_trajectory = evaluate_newton_trajectory(
        initial_y=p_n_cpu,
        p_n=p_n_cpu,
        v_n=v_n_cpu,
        y_star=y_star.detach().cpu(),
        m=m,
        g=g,
        dt=dt,
        steps=config.evaluation_steps,
        device=device,
    )

    report = {
        "config": {
            "experiment_name": experiment_name,
            "torch_dtype": str(TORCH_DTYPE),
            "device": str(device),
            "architecture": f"12 -> 32 -> {activation_name} -> 3",
            "hidden_dims": [32],
            "activation_name": activation_name,
            "activation_has_trainable_parameters": False,
            "num_hidden_layers": 1,
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "target_dataset_size": grid_spec.target_num_points,
            "actual_dataset_size": dataset_size,
            "points_per_axis": grid_spec.points_per_axis,
            "axis_spacing": grid_spec.axis_spacing,
            "sampling_radius_per_axis": grid_spec.sampling_radius,
            "training_mode": "full_batch",
            "epochs_requested": config.epochs,
            "completed_epochs": len(train_log),
            "no_early_stopping": True,
            "validation_interval": config.validation_interval,
            "validation_steps": config.evaluation_steps,
            "validation_size": config.validation_size,
            "test_steps": config.evaluation_steps,
            "test_size": config.test_size,
            "checkpoint_selection": "minimum final-step validation residual p95; tie-break by residual median and energy-gap p95",
            "initial_K": config.initial_k,
            "K_increase_interval": config.k_increase_interval,
            "K_increase_amount": config.k_increase_amount,
            "max_K": config.max_k,
            "use_input_normalization": USE_INPUT_NORMALIZATION,
            "use_output_dt_scaling": USE_OUTPUT_DT_SCALING,
            "input_mean": tensor_to_list(model.input_mean),
            "input_std": tensor_to_list(model.input_std),
            "loss": "sum of stepwise mean variational energy over full batch",
            "backpropagation": "full unroll without detach; one backward and one optimizer step per epoch",
            "p_n": tensor_to_list(p_n_cpu),
            "v_n": tensor_to_list(v_n_cpu),
            "m": m,
            "g": g,
            "dt": dt,
            "y_star": tensor_to_list(y_star),
            "E_star": e_star,
            "model_random_seed": MODEL_RANDOM_SEED,
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
            "selection_key": list(best_key) if best_key is not None else None,
            "validation_metrics": best_validation_metrics,
        },
        "train_log": train_log,
        "validation_log": validation_log,
        "final_test": {
            "best_validation_checkpoint": {
                "heldout_test": best_test_eval,
                "p_n": best_pn_eval,
            },
            "last_epoch_checkpoint": {
                "heldout_test": last_test_eval,
                "p_n": last_pn_eval,
            },
        },
        "p_n_trajectories": {
            "best_validation_checkpoint": best_pn_trajectory,
            "last_epoch_checkpoint": last_pn_trajectory,
            "newton": newton_trajectory,
        },
    }
    save_json(report, output_dir / "optimization_report.json")

    if not config.skip_individual_plots:
        plot_training_and_validation_curves(
            train_log=train_log,
            validation_log=validation_log,
            best_epoch=best_epoch,
            save_path=output_dir / "training_and_validation_curves.png",
        )
        plot_pn_comparison(
            best_trajectory=best_pn_trajectory,
            last_trajectory=last_pn_trajectory,
            newton_trajectory=newton_trajectory,
            save_path=output_dir / "p_n_best_vs_last_vs_newton.png",
        )
        if not config.skip_contour:
            plot_reference_energy_contour_2d(
                trajectory=best_pn_trajectory,
                newton_trajectory=newton_trajectory,
                y_star=tensor_to_list(y_star),
                p_n=tensor_to_list(p_n_cpu),
                v_n=tensor_to_list(v_n_cpu),
                m=m,
                g=g,
                dt=dt,
                save_path=output_dir / "best_checkpoint_p_n_energy_contour_2d.png",
            )

    best_checkpoint_test_summary = {
        "final_residual_mean": best_test_eval["final_residual_mean"],
        "final_residual_median": best_test_eval["final_residual_median"],
        "final_residual_p95": best_test_eval["final_residual_p95"],
        "final_residual_max": best_test_eval["final_residual_max"],
        "final_loss_gap_mean": best_test_eval["final_loss_gap_mean"],
        "final_loss_gap_median": best_test_eval["final_loss_gap_median"],
        "final_loss_gap_p95": best_test_eval["final_loss_gap_p95"],
        "final_loss_gap_max": best_test_eval["final_loss_gap_max"],
        "pn_final_residual": best_pn_eval["single_point_final_residual"],
        "pn_final_loss_gap": best_pn_eval["single_point_final_loss_gap"],
    }
    last_checkpoint_test_summary = {
        "final_residual_mean": last_test_eval["final_residual_mean"],
        "final_residual_median": last_test_eval["final_residual_median"],
        "final_residual_p95": last_test_eval["final_residual_p95"],
        "final_residual_max": last_test_eval["final_residual_max"],
        "final_loss_gap_mean": last_test_eval["final_loss_gap_mean"],
        "final_loss_gap_median": last_test_eval["final_loss_gap_median"],
        "final_loss_gap_p95": last_test_eval["final_loss_gap_p95"],
        "final_loss_gap_max": last_test_eval["final_loss_gap_max"],
        "pn_final_residual": last_pn_eval["single_point_final_residual"],
        "pn_final_loss_gap": last_pn_eval["single_point_final_loss_gap"],
    }

    print(
        f"完成 {experiment_name}: best_epoch={best_epoch}, "
        f"test_residual_p95={best_test_eval['final_residual_p95']:.4e}, "
        f"p_n_residual={best_pn_eval['single_point_final_residual']:.4e}"
    )

    return {
        "experiment_name": experiment_name,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "activation_name": activation_name,
        "target_dataset_size": grid_spec.target_num_points,
        "dataset_size": dataset_size,
        "points_per_axis": grid_spec.points_per_axis,
        "diverged": diverged,
        "divergence_epoch": divergence_epoch,
        "divergence_reason": divergence_reason,
        "completed_epochs": len(train_log),
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": list(best_key) if best_key is not None else None,
        "best_validation_metrics": best_validation_metrics,
        "best_checkpoint_test": best_checkpoint_test_summary,
        "last_checkpoint_test": last_checkpoint_test_summary,
        "training_curve_for_summary": downsample_log(train_log),
        "validation_curve_for_summary": downsample_log(validation_log),
        "output_directory": str(output_dir),
    }


# ============================================================
# 8. 参数与主程序
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Free-fall 1x32 MLP activation-function ablation with validation-selected checkpoints."
    )
    parser.add_argument(
        "--target-dataset-sizes",
        "--dataset-sizes",
        dest="target_dataset_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_TARGET_DATASET_SIZE_VALUES,
    )
    parser.add_argument(
        "--optimizer-configs",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset, e.g. adam:1e-4 sgd:1e-2. Default: SGD 1e-2 and Adam 1e-4.",
    )
    parser.add_argument(
        "--activation-names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional subset of activations. Available: "
            + ", ".join(DEFAULT_ACTIVATION_NAMES)
        ),
    )
    parser.add_argument("--sampling-radius", type=float, default=DEFAULT_SAMPLING_RADIUS)
    parser.add_argument("--heldout-radius-scale", type=float, default=DEFAULT_HELDOUT_RADIUS_SCALE)
    parser.add_argument("--grid-precompute-chunk-size", type=int, default=DEFAULT_GRID_PRECOMPUTE_CHUNK_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--evaluation-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    parser.add_argument("--validation-size", type=int, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument("--k-increase-interval", type=int, default=DEFAULT_K_INCREASE_INTERVAL)
    parser.add_argument("--k-increase-amount", type=int, default=DEFAULT_K_INCREASE_AMOUNT)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--skip-contour", action="store_true")
    parser.add_argument("--skip-individual-plots", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    target_sizes = ensure_positive_int_list(args.target_dataset_sizes)
    optimizer_configs = parse_optimizer_configs(args.optimizer_configs)
    activation_names = parse_activation_names(args.activation_names)
    positive_fields = {
        "sampling_radius": args.sampling_radius,
        "heldout_radius_scale": args.heldout_radius_scale,
        "grid_precompute_chunk_size": args.grid_precompute_chunk_size,
        "epochs": args.epochs,
        "validation_interval": args.validation_interval,
        "evaluation_steps": args.evaluation_steps,
        "evaluation_batch_size": args.evaluation_batch_size,
        "validation_size": args.validation_size,
        "test_size": args.test_size,
        "initial_k": args.initial_k,
        "k_increase_interval": args.k_increase_interval,
        "k_increase_amount": args.k_increase_amount,
        "max_k": args.max_k,
    }
    for name, value in positive_fields.items():
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if args.max_k < args.initial_k:
        raise ValueError("max_k must be >= initial_k.")

    return RuntimeConfig(
        target_dataset_sizes=target_sizes,
        optimizer_configs=optimizer_configs,
        activation_names=activation_names,
        sampling_radius=float(args.sampling_radius),
        heldout_radius_scale=float(args.heldout_radius_scale),
        grid_precompute_chunk_size=int(args.grid_precompute_chunk_size),
        epochs=int(args.epochs),
        validation_interval=int(args.validation_interval),
        evaluation_steps=int(args.evaluation_steps),
        evaluation_batch_size=int(args.evaluation_batch_size),
        validation_size=int(args.validation_size),
        test_size=int(args.test_size),
        initial_k=int(args.initial_k),
        k_increase_interval=int(args.k_increase_interval),
        k_increase_amount=int(args.k_increase_amount),
        max_k=int(args.max_k),
        device=str(args.device),
        skip_contour=bool(args.skip_contour),
        skip_individual_plots=bool(args.skip_individual_plots),
    )


def validate_device(device: torch.device) -> None:
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested cuda:{index}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
        )


def main() -> None:
    config = validate_args(parse_args())
    output_dir = create_output_directory()
    device = torch.device(config.device)
    validate_device(device)

    print(f"输出目录: {output_dir}")
    print(f"运行配置: {asdict(config)}")
    print(f"torch default dtype: {torch.get_default_dtype()}")

    m = 1.0
    g = 9.8
    dt = 0.01
    p_n_cpu = torch.tensor([3.0, 4.0, 5.0], dtype=TORCH_DTYPE)
    v_n_cpu = torch.tensor([0.5, -0.5, 0.0], dtype=TORCH_DTYPE)
    y_star_cpu = p_n_cpu + dt * v_n_cpu - dt**2 * torch.tensor([0.0, 0.0, g], dtype=TORCH_DTYPE)

    grid_specs = make_grid_specs(config.target_dataset_sizes, config.sampling_radius)
    print("规则训练网格：")
    for spec in grid_specs:
        print(
            f"- target={spec.target_num_points:,}, axis={spec.points_per_axis}, "
            f"actual={spec.actual_num_points:,}, spacing={spec.axis_spacing:.8e}"
        )

    total_heldout = config.validation_size + config.test_size
    heldout_points_cpu, heldout_metadata = build_fixed_heldout_set_excluding_training_grids(
        y_star=y_star_cpu,
        radius=config.sampling_radius,
        num_points=total_heldout,
        seed=HELDOUT_RANDOM_SEED,
        radius_scale=config.heldout_radius_scale,
        grid_specs=grid_specs,
    )
    validation_points_cpu, test_points_cpu = split_heldout_points(
        heldout_points_cpu,
        config.validation_size,
        config.test_size,
    )
    torch.save(
        {
            "validation_points": validation_points_cpu,
            "test_points": test_points_cpu,
            "metadata": heldout_metadata,
        },
        output_dir / "fixed_validation_test_split.pt",
    )
    save_json(
        {
            "metadata": heldout_metadata,
            "split": {
                "validation_size": config.validation_size,
                "test_size": config.test_size,
                "validation_role": "checkpoint selection only; no gradient",
                "test_role": "final evaluation only; never used during training",
            },
        },
        output_dir / "fixed_validation_test_split.json",
    )
    plot_dataset_distribution_overview(
        grid_specs=grid_specs,
        validation_points=validation_points_cpu,
        test_points=test_points_cpu,
        y_star=tensor_to_list(y_star_cpu),
        p_n=tensor_to_list(p_n_cpu),
        save_path=output_dir / "training_validation_test_distribution_overview.png",
    )

    summaries: list[dict[str, Any]] = []
    y_star_device = y_star_cpu.to(device)
    for grid_spec in grid_specs:
        print(
            f"\n预生成训练网格 N={grid_spec.actual_num_points:,} on {device}..."
        )
        training_grid = precompute_regular_grid_on_device(
            grid_spec=grid_spec,
            y_star=y_star_device,
            chunk_size=config.grid_precompute_chunk_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        memory_mb = training_grid.numel() * training_grid.element_size() / 1024**2
        print(f"训练网格已就绪: shape={tuple(training_grid.shape)}, memory={memory_mb:.2f} MiB")

        for optimizer_config in config.optimizer_configs:
            for activation_name in config.activation_names:
                summary = run_experiment(
                    base_output_dir=output_dir,
                    grid_spec=grid_spec,
                    training_grid=training_grid,
                    validation_points_cpu=validation_points_cpu,
                    test_points_cpu=test_points_cpu,
                    optimizer_name=optimizer_config["optimizer_name"],
                    learning_rate=float(optimizer_config["learning_rate"]),
                    activation_name=activation_name,
                    config=config,
                    p_n_cpu=p_n_cpu,
                    v_n_cpu=v_n_cpu,
                    m=m,
                    g=g,
                    dt=dt,
                )
                summaries.append(summary)

        del training_grid
        if device.type == "cuda":
            torch.cuda.empty_cache()

    overall_report = {
        "experiment_type": "activation_function_ablation_with_validation_checkpoint_selection",
        "purpose": (
            "Fix the selected 1x32 architecture, three training-grid scales, and two optimizer "
            "settings; change only the hidden activation function. Select the best checkpoint "
            "with a fixed validation split and evaluate it on an independent test split."
        ),
        "runtime_config": asdict(config),
        "network": {
            "architecture_template": "12 -> 32 -> activation -> 3",
            "hidden_dims": [32],
            "activation_names": config.activation_names,
            "final_layer_zero_initialized": True,
            "same_linear_initialization_seed_across_activations": True,
        },
        "grid_specs": [asdict(spec) for spec in grid_specs],
        "heldout_metadata": heldout_metadata,
        "data_roles": {
            "training": "full-batch gradient updates",
            "validation": "checkpoint selection only; never backpropagated",
            "test": "final report only; never evaluated during training",
            "p_n": "separate physical reference initial state; not used for checkpoint selection",
        },
        "no_early_stopping": True,
        "num_experiments": len(summaries),
        "activation_rankings": make_activation_ranking(summaries),
        "experiments": summaries,
    }
    save_json(overall_report, output_dir / "activation_function_ablation_summary.json")

    plot_final_heldout_summary(
        summaries,
        output_dir / "activation_best_checkpoint_test_summary.png",
        checkpoint_field="best_checkpoint_test",
    )
    plot_training_loss_summary(
        summaries,
        output_dir / "activation_training_loss_summary.png",
    )
    plot_validation_summary(
        summaries,
        output_dir / "activation_validation_residual_p95_summary.png",
    )

    print("\n" + "=" * 88)
    print("所有实验完成。")
    print(f"汇总 JSON: {output_dir / 'activation_function_ablation_summary.json'}")
    for record in summaries:
        print(
            f"- {record['experiment_name']}: activation={record['activation_name']}, "
            f"best_epoch={record['best_validation_epoch']}, "
            f"test_p95={record['best_checkpoint_test']['final_residual_p95']:.4e}, "
            f"p_n={record['best_checkpoint_test']['pn_final_residual']:.4e}, "
            f"diverged={record['diverged']}"
        )


if __name__ == "__main__":
    main()
