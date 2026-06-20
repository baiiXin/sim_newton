"""
自由落体单帧变分问题：最简可用 1×32 线性迭代器组件消融
===========================================================

目标
----
在已经确定的最优网络结构基础上，固定：

- 网络：12 -> 32 -> 3，中间不使用激活函数；
- 训练集规模：8、1,000；
- 优化器：SGD(lr=1e-2)、Adam(lr=1e-4)；
- Full-Batch；
- 训练 5,000 epoch；
- K 从 1 开始，每 1,000 epoch 增加 1，最高 K=5；
- torch.float64；
- 默认设备 cuda:1。

只消融两个实现组件：

1. 输入逐特征归一化；
2. 网络输出乘 dt 的放缩。

四组组件配置：

- full：保留输入归一化和输出 dt 放缩；
- no_input_norm：去掉输入归一化；
- no_output_scaling：去掉输出 dt 放缩；
- raw：二者都去掉。

因此默认共运行：2 个数据规模 × 2 个优化器 × 4 个组件配置 = 16 组实验。

验证集与 checkpoint
-------------------

- 验证集和测试集分别使用独立、固定、可复现的 scrambled Sobol 点集；
- 两者都排除训练规则网格上的点；
- 验证集仅用于 checkpoint 选择，测试集只在训练完成后评估；
- 每 100 epoch 验证一次，固定展开 50 步；
- 只有进入最大 K=5 阶段的 checkpoint 才有资格成为最终最佳 checkpoint；
- checkpoint 不再只看第 50 步，而是综合步骤 1、2、5、10、20、50 的 residual p95；
- 主选择指标为这些步骤的 mean(log10 residual p95)，优先选择整条轨迹收敛更快的模型；
- 非有限值数量、最坏步骤、最终步 p95/median 和能量 gap p95 依次作为稳定性与并列判据；
- 不执行 early stopping，仍完整训练 5,000 epoch；
- 同时保存验证集选出的最佳模型和最后一个 epoch 模型。

默认运行
--------

    python free_fall_minimal_components_ablation.py

快速 CPU 冒烟测试
-----------------

    python free_fall_minimal_components_ablation.py \
        --device cpu \
        --target-dataset-sizes 8 \
        --optimizer-configs adam:1e-4 \
        --component-configs full raw \
        --epochs 5 \
        --validation-interval 1 \
        --evaluation-steps 5 \
        --checkpoint-steps 1 2 5 \
        --validation-size 16 \
        --test-size 32 \
        --initial-k 1 \
        --k-increase-interval 1 \
        --max-k 5 \
        --skip-individual-plots
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

PLOT_FLOOR = 1e-12
GRID_MATCH_TOL = 1e-5
MODEL_RANDOM_SEED = 42
VALIDATION_SOBOL_SEED = 20260618
TEST_SOBOL_SEED = 20260619

DEFAULT_TARGET_DATASET_SIZE_VALUES = [8, 1_000]
DEFAULT_SAMPLING_RADIUS = 0.01
DEFAULT_EPOCHS = 5_000
DEFAULT_VALIDATION_INTERVAL = 100
DEFAULT_EVALUATION_STEPS = 50
DEFAULT_CHECKPOINT_STEPS = [1, 2, 5, 10, 20, 50]
DEFAULT_EVALUATION_BATCH_SIZE = 8_192
DEFAULT_VALIDATION_SIZE = 1_024
DEFAULT_TEST_SIZE = 4_096
DEFAULT_HELDOUT_RADIUS_SCALE = 1.0

DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 1_000
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

COMPONENT_CONFIG_LIBRARY: dict[str, dict[str, Any]] = {
    "full": {
        "use_input_normalization": True,
        "use_output_dt_scaling": True,
        "description": "input normalization + output dt scaling",
    },
    "no_input_norm": {
        "use_input_normalization": False,
        "use_output_dt_scaling": True,
        "description": "remove input normalization; keep output dt scaling",
    },
    "no_output_scaling": {
        "use_input_normalization": True,
        "use_output_dt_scaling": False,
        "description": "keep input normalization; remove output dt scaling",
    },
    "raw": {
        "use_input_normalization": False,
        "use_output_dt_scaling": False,
        "description": "remove both input normalization and output dt scaling",
    },
}
DEFAULT_COMPONENT_CONFIG_NAMES = list(COMPONENT_CONFIG_LIBRARY)


# ============================================================
# 1. 数据结构与通用函数
# ============================================================


@dataclass(frozen=True)
class RuntimeConfig:
    target_dataset_sizes: list[int]
    optimizer_configs: list[dict[str, Any]]
    component_configs: list[dict[str, Any]]
    sampling_radius: float
    heldout_radius_scale: float
    grid_precompute_chunk_size: int
    epochs: int
    validation_interval: int
    evaluation_steps: int
    checkpoint_steps: list[int]
    evaluation_batch_size: int
    validation_size: int
    test_size: int
    initial_k: int
    k_increase_interval: int
    k_increase_amount: int
    max_k: int
    checkpoint_min_epoch: int
    device: str
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
        raise ValueError("The integer list must not be empty.")
    if cleaned[0] <= 0:
        raise ValueError("Every value must be positive.")
    return cleaned


def nearest_even_points_per_axis(target_num_points: int) -> int:
    if target_num_points <= 0:
        raise ValueError("target_num_points must be positive.")
    root = target_num_points ** (1.0 / 3.0)
    lower = max(2, 2 * int(math.floor(root / 2.0)))
    upper = max(2, lower + 2)
    return min({lower, upper}, key=lambda n: (abs(n**3 - target_num_points), n))


def make_grid_spec(target_num_points: int, sampling_radius: float) -> GridSpec:
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


def first_epoch_with_max_k(
    *,
    initial_k: int,
    increase_interval: int,
    increase_amount: int,
    max_k: int,
) -> int:
    if initial_k >= max_k:
        return 1
    num_increases = math.ceil((max_k - initial_k) / increase_amount)
    return num_increases * increase_interval + 1


def downsample_log(
    records: Sequence[dict[str, Any]],
    max_points: int = DEFAULT_SUMMARY_CURVE_POINTS,
) -> list[dict[str, Any]]:
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
                f"Invalid optimizer config {raw!r}. Use forms such as adam:1e-4."
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


def parse_component_configs(values: Sequence[str] | None) -> list[dict[str, Any]]:
    names = DEFAULT_COMPONENT_CONFIG_NAMES if not values else list(values)
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_name in names:
        name = raw_name.strip().lower()
        if name not in COMPONENT_CONFIG_LIBRARY:
            raise ValueError(
                f"Unsupported component config {raw_name!r}. Available: "
                + ", ".join(DEFAULT_COMPONENT_CONFIG_NAMES)
            )
        if name in seen:
            continue
        config = copy.deepcopy(COMPONENT_CONFIG_LIBRARY[name])
        config["name"] = name
        parsed.append(config)
        seen.add(name)
    return parsed


# ============================================================
# 2. 网络与物理问题
# ============================================================


class MLPOptimizer(nn.Module):
    """固定结构：12 -> 32 -> 3；隐藏层之间不使用激活函数。"""

    def __init__(
        self,
        *,
        use_input_normalization: bool,
        use_output_dt_scaling: bool,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.use_input_normalization = bool(use_input_normalization)
        self.use_output_dt_scaling = bool(use_output_dt_scaling)

        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.register_buffer("input_mean", input_mean.clone().detach().to(TORCH_DTYPE))
        self.register_buffer("input_std", input_std.clone().detach().to(TORCH_DTYPE))

    @staticmethod
    def _expand_feature_for_batch(feature: torch.Tensor, batch_size: int) -> torch.Tensor:
        if feature.ndim == 1:
            return feature.unsqueeze(0).expand(batch_size, -1)
        if feature.ndim == 2 and feature.shape[0] == batch_size:
            return feature
        raise ValueError(
            f"Feature shape is incompatible: feature={tuple(feature.shape)}, "
            f"batch={batch_size}."
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
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    if optimizer_name == "adam":
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
# 4. 固定 Sobol 验证集 / 测试集
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


def build_fixed_sobol_set_excluding_training_grids(
    *,
    y_star: torch.Tensor,
    radius: float,
    num_points: int,
    seed: int,
    radius_scale: float,
    grid_specs: Sequence[GridSpec],
    role: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if num_points <= 0:
        raise ValueError("num_points must be positive.")

    effective_radius = radius * radius_scale
    y_star_cpu = y_star.detach().cpu().to(TORCH_DTYPE)
    engine = torch.quasirandom.SobolEngine(
        dimension=3,
        scramble=True,
        seed=seed,
    )

    chunks: list[torch.Tensor] = []
    collected = 0
    generated = 0
    rejected = 0
    batch_size = max(2_048, num_points)

    for _ in range(1_000):
        if collected >= num_points:
            break
        unit_points = engine.draw(batch_size).to(TORCH_DTYPE)
        candidates = y_star_cpu.unsqueeze(0) + (2.0 * unit_points - 1.0) * effective_radius
        keep = torch.ones(candidates.shape[0], dtype=torch.bool)
        for grid_spec in grid_specs:
            keep &= ~points_in_grid_mask(
                candidates,
                grid_spec=grid_spec,
                y_star=y_star_cpu,
            )
        kept = candidates[keep]
        remaining = num_points - collected
        if kept.shape[0] > remaining:
            kept = kept[:remaining]
        if kept.numel() > 0:
            chunks.append(kept)
            collected += int(kept.shape[0])
        generated += int(candidates.shape[0])
        rejected += int((~keep).sum().item())

    if collected < num_points:
        raise RuntimeError(
            f"Unable to build enough Sobol points: requested={num_points}, "
            f"collected={collected}."
        )

    points = torch.cat(chunks, dim=0)
    metadata = {
        "role": role,
        "mode": "independent_scrambled_sobol_cube_excluding_all_training_grids",
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


# ============================================================
# 5. 批量评估与 checkpoint 选择
# ============================================================


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
            gap = torch.clamp(gap, min=0.0)
            batch_residuals.append(residual.detach().cpu())
            batch_gaps.append(gap.detach().cpu())
            if step < steps:
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
        result["single_point_residual_by_step"] = residuals[0].tolist()
        result["single_point_loss_gap_by_step"] = gaps[0].tolist()
        result["single_point_final_residual"] = float(residuals[0, -1])
        result["single_point_final_loss_gap"] = float(gaps[0, -1])

    return result


def checkpoint_selection_metrics(
    validation_result: dict[str, Any],
    checkpoint_steps: Sequence[int],
) -> tuple[tuple[float, ...] | None, dict[str, Any]]:
    p95_curve = validation_result["residual_p95_by_step"]
    median_curve = validation_result["residual_median_by_step"]
    nonfinite_curve = validation_result["residual_num_nonfinite_by_step"]

    selected_p95 = [float(p95_curve[step]) for step in checkpoint_steps]
    selected_median = [float(median_curve[step]) for step in checkpoint_steps]
    selected_nonfinite = [int(nonfinite_curve[step]) for step in checkpoint_steps]

    finite_p95 = all(math.isfinite(value) and value >= 0.0 for value in selected_p95)
    finite_median = all(math.isfinite(value) and value >= 0.0 for value in selected_median)
    final_gap_p95 = float(validation_result["final_loss_gap_p95"])

    details = {
        "checkpoint_steps": list(checkpoint_steps),
        "residual_p95_at_checkpoint_steps": selected_p95,
        "residual_median_at_checkpoint_steps": selected_median,
        "nonfinite_at_checkpoint_steps": selected_nonfinite,
        "total_nonfinite_at_checkpoint_steps": int(sum(selected_nonfinite)),
        "selection_score_mean_log10_residual_p95": None,
        "selection_score_max_log10_residual_p95": None,
    }

    if not (finite_p95 and finite_median and math.isfinite(final_gap_p95)):
        return None, details

    log_p95 = [math.log10(max(value, PLOT_FLOOR)) for value in selected_p95]
    mean_log_p95 = float(np.mean(log_p95))
    max_log_p95 = float(np.max(log_p95))
    details["selection_score_mean_log10_residual_p95"] = mean_log_p95
    details["selection_score_max_log10_residual_p95"] = max_log_p95

    key = (
        float(sum(selected_nonfinite)),
        mean_log_p95,
        max_log_p95,
        selected_p95[-1],
        selected_median[-1],
        final_gap_p95,
    )
    return key, details


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
                "gap": max(float((energy - e_star).item()), 0.0),
                "residual_norm": float(residual.item()),
            }
        )
        if step < steps:
            delta = model(y, history, params)
            y = y + delta
            iterations[-1]["next_delta_norm"] = float(
                torch.linalg.vector_norm(delta).item()
            )
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
                "gap": max(float((energy - e_star).item()), 0.0),
                "residual_norm": float(residual.item()),
            }
        )
        if step < steps:
            delta = newton_direction(y, p_n_device, v_n_device, m, g, dt)
            y = y + delta
            iterations[-1]["next_delta_norm"] = float(
                torch.linalg.vector_norm(delta).item()
            )
    return {"initial_y": tensor_to_list(initial_y), "iterations": iterations}


# ============================================================
# 6. 绘图
# ============================================================


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


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    radius = max(float(np.ptp(points, axis=0).max()) / 2.0, 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


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
    validation_np = validation_points.numpy()
    test_np = test_points.numpy()

    fig = plt.figure(figsize=(16, 5))
    for index, grid_spec in enumerate(grid_specs):
        ax = fig.add_subplot(1, 3, index + 1, projection="3d")
        train = sample_regular_grid_for_plot(grid_spec, y_star_np)
        ax.scatter(train[:, 0], train[:, 1], train[:, 2], s=8, alpha=0.35)
        ax.scatter(*p_n_np, marker="x", s=100, linewidths=2, label=r"$p_n$")
        ax.scatter(*y_star_np, marker="*", s=180, label=r"$y^*$")
        set_equal_3d_axes(ax, np.vstack([train, p_n_np, y_star_np]))
        ax.set_title(f"Training grid N={grid_spec.actual_num_points:,}")
        ax.legend(fontsize=8)

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.scatter(
        validation_np[:, 0], validation_np[:, 1], validation_np[:, 2],
        s=5, alpha=0.35, label=f"validation N={len(validation_np)}",
    )
    ax.scatter(
        test_np[:, 0], test_np[:, 1], test_np[:, 2],
        s=4, alpha=0.18, label=f"test N={len(test_np)}",
    )
    ax.scatter(*p_n_np, marker="x", s=100, linewidths=2, label=r"$p_n$")
    ax.scatter(*y_star_np, marker="*", s=180, label=r"$y^*$")
    set_equal_3d_axes(
        ax,
        np.vstack([validation_np, test_np, p_n_np, y_star_np]),
    )
    ax.set_title("Independent fixed Sobol sets")
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_training_and_validation_curves(
    *,
    train_log: Sequence[dict[str, Any]],
    validation_log: Sequence[dict[str, Any]],
    best_epoch: int | None,
    checkpoint_min_epoch: int,
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(
        [record["epoch"] for record in train_log],
        [finite_plot_value(record["training_gap_for_readability"]) for record in train_log],
    )
    axes[0].set_yscale("log")
    axes[0].set_title("Training trajectory energy-sum gap")
    axes[0].set_xlabel("Epoch")

    eligible = [record for record in validation_log if record["eligible_for_checkpoint"]]
    ineligible = [record for record in validation_log if not record["eligible_for_checkpoint"]]
    if ineligible:
        axes[1].plot(
            [record["epoch"] for record in ineligible],
            [
                float(record["selection_details"]["selection_score_mean_log10_residual_p95"])
                if record["selection_details"]["selection_score_mean_log10_residual_p95"] is not None
                else float("nan")
                for record in ineligible
            ],
            linestyle="--",
            alpha=0.45,
            label="diagnostic only",
        )
    if eligible:
        axes[1].plot(
            [record["epoch"] for record in eligible],
            [record["selection_details"]["selection_score_mean_log10_residual_p95"] for record in eligible],
            marker="o",
            markersize=3,
            label="eligible checkpoint",
        )
    axes[1].set_title("Validation trajectory selection score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mean log10 residual p95")
    axes[1].legend()

    for step in [1, 5, 20, 50]:
        if step > validation_log[0]["metrics"]["steps"]:
            continue
        axes[2].plot(
            [record["epoch"] for record in validation_log],
            [finite_plot_value(record["metrics"]["residual_p95_by_step"][step]) for record in validation_log],
            label=f"step {step}",
        )
    axes[2].set_yscale("log")
    axes[2].set_title("Validation residual p95 by rollout horizon")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    for ax in axes:
        ax.axvline(checkpoint_min_epoch, linestyle=":", alpha=0.7, label="max-K stage")
        if best_epoch is not None:
            ax.axvline(best_epoch, linestyle="--", alpha=0.7)
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


def optimizer_key(record: dict[str, Any]) -> tuple[str, float]:
    return str(record["optimizer_name"]), float(record["learning_rate"])


def optimizer_label(key: tuple[str, float]) -> str:
    return f"{key[0].upper()} lr={key[1]:.0e}"


def unique_optimizer_keys(records: Sequence[dict[str, Any]]) -> list[tuple[str, float]]:
    preferred = {"sgd": 0, "adam": 1}
    keys = {optimizer_key(record) for record in records}
    return sorted(keys, key=lambda item: (preferred.get(item[0], 99), -item[1]))


def component_label(name: str) -> str:
    labels = {
        "full": "norm + dt scale",
        "no_input_norm": "no norm",
        "no_output_scaling": "no dt scale",
        "raw": "raw",
    }
    return labels.get(name, name)


def plot_final_test_summary(records: Sequence[dict[str, Any]], save_path: Path) -> None:
    optimizer_keys = unique_optimizer_keys(records)
    fig, axes = plt.subplots(
        len(optimizer_keys),
        3,
        figsize=(18, 5 * len(optimizer_keys)),
        squeeze=False,
    )
    metrics = [
        ("final_residual_p95", "Test final residual p95"),
        ("final_residual_median", "Test final residual median"),
        ("pn_final_residual", r"$p_n$ final residual"),
    ]

    for row, key in enumerate(optimizer_keys):
        optimizer_records = [record for record in records if optimizer_key(record) == key]
        for component_name in DEFAULT_COMPONENT_CONFIG_NAMES:
            selected = sorted(
                [
                    record for record in optimizer_records
                    if record["component_config_name"] == component_name
                ],
                key=lambda record: int(record["dataset_size"]),
            )
            if not selected:
                continue
            sizes = [int(record["dataset_size"]) for record in selected]
            for ax, (metric, title) in zip(axes[row], metrics):
                values = [
                    finite_plot_value(record["best_checkpoint_test"][metric])
                    for record in selected
                ]
                ax.plot(sizes, values, marker="o", label=component_label(component_name))
                ax.set_title(f"{optimizer_label(key)}\n{title}")

    for ax in axes.reshape(-1):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training dataset size")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss_summary(records: Sequence[dict[str, Any]], save_path: Path) -> None:
    optimizer_keys = unique_optimizer_keys(records)
    dataset_sizes = sorted({int(record["dataset_size"]) for record in records})
    fig, axes = plt.subplots(
        len(optimizer_keys),
        len(dataset_sizes),
        figsize=(6 * len(dataset_sizes), 4.8 * len(optimizer_keys)),
        squeeze=False,
    )
    for row, key in enumerate(optimizer_keys):
        for col, dataset_size in enumerate(dataset_sizes):
            ax = axes[row, col]
            selected = [
                record for record in records
                if optimizer_key(record) == key
                and int(record["dataset_size"]) == dataset_size
            ]
            for record in selected:
                curve = record["training_curve_for_summary"]
                ax.plot(
                    [point["epoch"] for point in curve],
                    [finite_plot_value(point["training_gap_for_readability"]) for point in curve],
                    label=component_label(record["component_config_name"]),
                )
            ax.set_yscale("log")
            ax.set_title(f"{optimizer_label(key)}\nN={dataset_size:,}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training gap")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_component_ranking(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for key in unique_optimizer_keys(records):
        sizes = sorted(
            {
                int(record["dataset_size"])
                for record in records
                if optimizer_key(record) == key
            }
        )
        for dataset_size in sizes:
            selected = [
                record for record in records
                if optimizer_key(record) == key
                and int(record["dataset_size"]) == dataset_size
            ]
            ordered = sorted(
                selected,
                key=lambda record: (
                    finite_plot_value(record["best_checkpoint_test"]["final_residual_p95"]),
                    finite_plot_value(record["best_checkpoint_test"]["final_residual_median"]),
                ),
            )
            rankings.append(
                {
                    "optimizer_name": key[0],
                    "learning_rate": key[1],
                    "dataset_size": dataset_size,
                    "ranking_metric": "best-checkpoint independent test residual p95",
                    "ranking": [
                        {
                            "rank": rank,
                            "component_config_name": record["component_config_name"],
                            "test_residual_p95": record["best_checkpoint_test"]["final_residual_p95"],
                            "test_residual_median": record["best_checkpoint_test"]["final_residual_median"],
                            "best_validation_epoch": record["best_validation_epoch"],
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
    component_config: dict[str, Any],
    config: RuntimeConfig,
    p_n_cpu: torch.Tensor,
    v_n_cpu: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> dict[str, Any]:
    dataset_size = grid_spec.actual_num_points
    component_name = str(component_config["name"])
    experiment_name = (
        f"{optimizer_name}_lr_{learning_rate:.0e}_"
        f"components_{component_name}_"
        f"grid_axis_{grid_spec.points_per_axis}_num_samples_{dataset_size}"
    )
    output_dir = base_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = training_grid.device
    p_n = p_n_cpu.to(device)
    v_n = v_n_cpu.to(device)
    y_star = p_n + dt * v_n - dt**2 * torch.tensor(
        [0.0, 0.0, g], dtype=TORCH_DTYPE, device=device
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
        use_input_normalization=bool(component_config["use_input_normalization"]),
        use_output_dt_scaling=bool(component_config["use_output_dt_scaling"]),
        input_mean=input_mean,
        input_std=input_std,
    ).to(device)
    optimizer = create_optimizer(model, optimizer_name, learning_rate)

    print("\n" + "=" * 92)
    print(f"实验：{experiment_name}")
    print(
        f"device={device}, dtype={TORCH_DTYPE}, architecture=12->32->3, "
        f"activation=identity"
    )
    print(
        f"training_N={dataset_size:,}, validation_N={config.validation_size:,}, "
        f"test_N={config.test_size:,}"
    )
    print(
        f"optimizer={optimizer_name}, lr={learning_rate:.0e}, "
        f"components={component_name}"
    )
    print(
        f"checkpoint eligible from epoch {config.checkpoint_min_epoch}; "
        f"steps={config.checkpoint_steps}"
    )
    print("=" * 92)

    train_log: list[dict[str, Any]] = []
    validation_log: list[dict[str, Any]] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_selection_details: dict[str, Any] | None = None
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
                y, p_n, v_n, m, g, dt
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
        training_gap = max(loss_value - k * e_star, 0.0)
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
            or epoch_number == config.checkpoint_min_epoch
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
            current_key, selection_details = checkpoint_selection_metrics(
                validation_metrics,
                config.checkpoint_steps,
            )
            eligible = epoch_number >= config.checkpoint_min_epoch
            validation_log.append(
                {
                    "epoch": epoch_number,
                    "training_K": k,
                    "eligible_for_checkpoint": eligible,
                    "selection_key": list(current_key) if current_key is not None else None,
                    "selection_details": selection_details,
                    "metrics": validation_metrics,
                }
            )

            if (
                eligible
                and current_key is not None
                and (best_key is None or current_key < best_key)
            ):
                best_key = current_key
                best_epoch = epoch_number
                best_validation_metrics = copy.deepcopy(validation_metrics)
                best_selection_details = copy.deepcopy(selection_details)
                best_state_dict = state_dict_to_cpu(model)

            elapsed = time.perf_counter() - start_time
            score = selection_details["selection_score_mean_log10_residual_p95"]
            score_text = "nan" if score is None else f"{score:.4f}"
            print(
                f"Epoch {epoch_number:5d} | K={k} | train_gap={training_gap:.4e} | "
                f"val_score={score_text} | "
                f"val_final_p95={validation_metrics['final_residual_p95']:.4e} | "
                f"eligible={eligible} | best_epoch={best_epoch} | elapsed={elapsed:.1f}s"
            )

    last_state_dict = state_dict_to_cpu(model)
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(last_state_dict)
        best_epoch = train_log[-1]["epoch"] if train_log else 0
        best_validation_metrics = None
        best_selection_details = None

    torch.save(last_state_dict, output_dir / "last_model_state_dict.pt")
    torch.save(best_state_dict, output_dir / "best_validation_model_state_dict.pt")
    torch.save(best_state_dict, output_dir / "mlp_optimizer_state_dict.pt")

    def evaluate_checkpoint(
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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

    def compact_test_summary(
        test_eval: dict[str, Any],
        pn_eval: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "final_residual_mean": test_eval["final_residual_mean"],
            "final_residual_median": test_eval["final_residual_median"],
            "final_residual_p95": test_eval["final_residual_p95"],
            "final_residual_max": test_eval["final_residual_max"],
            "final_loss_gap_mean": test_eval["final_loss_gap_mean"],
            "final_loss_gap_median": test_eval["final_loss_gap_median"],
            "final_loss_gap_p95": test_eval["final_loss_gap_p95"],
            "final_loss_gap_max": test_eval["final_loss_gap_max"],
            "pn_final_residual": pn_eval["single_point_final_residual"],
            "pn_final_loss_gap": pn_eval["single_point_final_loss_gap"],
        }

    best_checkpoint_test_summary = compact_test_summary(best_test_eval, best_pn_eval)
    last_checkpoint_test_summary = compact_test_summary(last_test_eval, last_pn_eval)

    report = {
        "config": {
            "experiment_name": experiment_name,
            "torch_dtype": str(TORCH_DTYPE),
            "device": str(device),
            "architecture": "12 -> 32 -> 3",
            "hidden_dims": [32],
            "activation_name": "identity",
            "num_hidden_layers": 1,
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "component_config": component_config,
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
            "checkpoint_steps": config.checkpoint_steps,
            "checkpoint_min_epoch": config.checkpoint_min_epoch,
            "checkpoint_selection": (
                "eligible only in max-K stage; minimize total nonfinite count, then "
                "mean log10 residual p95 across selected rollout steps, worst selected "
                "log10 p95, final p95, final median, and final energy-gap p95"
            ),
            "validation_size": config.validation_size,
            "test_size": config.test_size,
            "initial_K": config.initial_k,
            "K_increase_interval": config.k_increase_interval,
            "K_increase_amount": config.k_increase_amount,
            "max_K": config.max_k,
            "input_mean": tensor_to_list(model.input_mean),
            "input_std": tensor_to_list(model.input_std),
            "loss": "sum of stepwise mean variational energy over full batch",
            "backpropagation": "full unroll without detach; one backward per epoch",
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
            "selection_details": best_selection_details,
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

    if not config.skip_individual_plots and validation_log:
        plot_training_and_validation_curves(
            train_log=train_log,
            validation_log=validation_log,
            best_epoch=best_epoch,
            checkpoint_min_epoch=config.checkpoint_min_epoch,
            save_path=output_dir / "training_and_validation_curves.png",
        )
        plot_pn_comparison(
            best_trajectory=best_pn_trajectory,
            last_trajectory=last_pn_trajectory,
            newton_trajectory=newton_trajectory,
            save_path=output_dir / "p_n_best_vs_last_vs_newton.png",
        )

    print(
        f"完成 {experiment_name}: best_epoch={best_epoch}, "
        f"test_residual_p95={best_test_eval['final_residual_p95']:.4e}, "
        f"p_n_residual={best_pn_eval['single_point_final_residual']:.4e}"
    )

    return {
        "experiment_name": experiment_name,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "component_config_name": component_name,
        "use_input_normalization": bool(component_config["use_input_normalization"]),
        "use_output_dt_scaling": bool(component_config["use_output_dt_scaling"]),
        "target_dataset_size": grid_spec.target_num_points,
        "dataset_size": dataset_size,
        "points_per_axis": grid_spec.points_per_axis,
        "diverged": diverged,
        "divergence_epoch": divergence_epoch,
        "divergence_reason": divergence_reason,
        "completed_epochs": len(train_log),
        "best_validation_epoch": best_epoch,
        "best_validation_selection_key": list(best_key) if best_key is not None else None,
        "best_validation_selection_details": best_selection_details,
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
        description=(
            "Free-fall 1x32 identity-MLP component ablation with Sobol validation "
            "and multi-horizon checkpoint selection."
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
    parser.add_argument(
        "--optimizer-configs",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset, e.g. adam:1e-4 sgd:1e-2.",
    )
    parser.add_argument(
        "--component-configs",
        type=str,
        nargs="+",
        default=None,
        help="Available: " + ", ".join(DEFAULT_COMPONENT_CONFIG_NAMES),
    )
    parser.add_argument("--sampling-radius", type=float, default=DEFAULT_SAMPLING_RADIUS)
    parser.add_argument("--heldout-radius-scale", type=float, default=DEFAULT_HELDOUT_RADIUS_SCALE)
    parser.add_argument("--grid-precompute-chunk-size", type=int, default=DEFAULT_GRID_PRECOMPUTE_CHUNK_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--evaluation-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=DEFAULT_CHECKPOINT_STEPS,
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    parser.add_argument("--validation-size", type=int, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument("--k-increase-interval", type=int, default=DEFAULT_K_INCREASE_INTERVAL)
    parser.add_argument("--k-increase-amount", type=int, default=DEFAULT_K_INCREASE_AMOUNT)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument(
        "--checkpoint-min-epoch",
        type=int,
        default=None,
        help=(
            "First epoch eligible for checkpoint selection. Default: automatically "
            "the first epoch whose training K reaches max_k."
        ),
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--skip-individual-plots", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    target_sizes = ensure_positive_int_list(args.target_dataset_sizes)
    optimizer_configs = parse_optimizer_configs(args.optimizer_configs)
    component_configs = parse_component_configs(args.component_configs)
    checkpoint_steps = ensure_positive_int_list(args.checkpoint_steps)

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
    if checkpoint_steps[-1] > args.evaluation_steps:
        raise ValueError("Every checkpoint step must be <= evaluation_steps.")

    automatic_min_epoch = first_epoch_with_max_k(
        initial_k=int(args.initial_k),
        increase_interval=int(args.k_increase_interval),
        increase_amount=int(args.k_increase_amount),
        max_k=int(args.max_k),
    )
    checkpoint_min_epoch = (
        automatic_min_epoch
        if args.checkpoint_min_epoch is None
        else int(args.checkpoint_min_epoch)
    )
    if checkpoint_min_epoch <= 0 or checkpoint_min_epoch > args.epochs:
        raise ValueError(
            "checkpoint_min_epoch must be between 1 and epochs. "
            f"Got {checkpoint_min_epoch} with epochs={args.epochs}."
        )

    return RuntimeConfig(
        target_dataset_sizes=target_sizes,
        optimizer_configs=optimizer_configs,
        component_configs=component_configs,
        sampling_radius=float(args.sampling_radius),
        heldout_radius_scale=float(args.heldout_radius_scale),
        grid_precompute_chunk_size=int(args.grid_precompute_chunk_size),
        epochs=int(args.epochs),
        validation_interval=int(args.validation_interval),
        evaluation_steps=int(args.evaluation_steps),
        checkpoint_steps=checkpoint_steps,
        evaluation_batch_size=int(args.evaluation_batch_size),
        validation_size=int(args.validation_size),
        test_size=int(args.test_size),
        initial_k=int(args.initial_k),
        k_increase_interval=int(args.k_increase_interval),
        k_increase_amount=int(args.k_increase_amount),
        max_k=int(args.max_k),
        checkpoint_min_epoch=checkpoint_min_epoch,
        device=str(args.device),
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
    y_star_cpu = p_n_cpu + dt * v_n_cpu - dt**2 * torch.tensor(
        [0.0, 0.0, g], dtype=TORCH_DTYPE
    )

    grid_specs = make_grid_specs(config.target_dataset_sizes, config.sampling_radius)
    print("规则训练网格：")
    for spec in grid_specs:
        print(
            f"- target={spec.target_num_points:,}, axis={spec.points_per_axis}, "
            f"actual={spec.actual_num_points:,}, spacing={spec.axis_spacing:.8e}"
        )

    validation_points_cpu, validation_metadata = build_fixed_sobol_set_excluding_training_grids(
        y_star=y_star_cpu,
        radius=config.sampling_radius,
        num_points=config.validation_size,
        seed=VALIDATION_SOBOL_SEED,
        radius_scale=config.heldout_radius_scale,
        grid_specs=grid_specs,
        role="checkpoint_selection_only",
    )
    test_points_cpu, test_metadata = build_fixed_sobol_set_excluding_training_grids(
        y_star=y_star_cpu,
        radius=config.sampling_radius,
        num_points=config.test_size,
        seed=TEST_SOBOL_SEED,
        radius_scale=config.heldout_radius_scale,
        grid_specs=grid_specs,
        role="final_evaluation_only",
    )

    torch.save(
        {
            "validation_points": validation_points_cpu,
            "test_points": test_points_cpu,
            "validation_metadata": validation_metadata,
            "test_metadata": test_metadata,
        },
        output_dir / "fixed_sobol_validation_test_sets.pt",
    )
    save_json(
        {
            "validation_metadata": validation_metadata,
            "test_metadata": test_metadata,
            "data_roles": {
                "validation": "checkpoint selection only; no gradient",
                "test": "final evaluation only; never used during training",
            },
        },
        output_dir / "fixed_sobol_validation_test_sets.json",
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
        print(f"\n预生成训练网格 N={grid_spec.actual_num_points:,} on {device}...")
        training_grid = precompute_regular_grid_on_device(
            grid_spec=grid_spec,
            y_star=y_star_device,
            chunk_size=config.grid_precompute_chunk_size,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        memory_mb = training_grid.numel() * training_grid.element_size() / 1024**2
        print(
            f"训练网格已就绪: shape={tuple(training_grid.shape)}, "
            f"memory={memory_mb:.2f} MiB"
        )

        for optimizer_config in config.optimizer_configs:
            for component_config in config.component_configs:
                summary = run_experiment(
                    base_output_dir=output_dir,
                    grid_spec=grid_spec,
                    training_grid=training_grid,
                    validation_points_cpu=validation_points_cpu,
                    test_points_cpu=test_points_cpu,
                    optimizer_name=str(optimizer_config["optimizer_name"]),
                    learning_rate=float(optimizer_config["learning_rate"]),
                    component_config=component_config,
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
        "experiment_type": "minimal_component_ablation_with_sobol_validation",
        "purpose": (
            "Fix the selected 1x32 identity architecture, two training-grid sizes, "
            "and two optimizer settings; ablate input normalization and output dt "
            "scaling while using independent Sobol validation/test sets and "
            "multi-horizon checkpoint selection."
        ),
        "runtime_config": asdict(config),
        "network": {
            "architecture": "12 -> 32 -> 3",
            "activation": "identity / no nonlinear activation",
            "final_layer_zero_initialized": True,
            "same_initialization_seed_across_component_configs": True,
        },
        "component_configs": config.component_configs,
        "grid_specs": [asdict(spec) for spec in grid_specs],
        "validation_metadata": validation_metadata,
        "test_metadata": test_metadata,
        "data_roles": {
            "training": "full-batch gradient updates",
            "validation": "multi-horizon checkpoint selection only",
            "test": "final report only",
            "p_n": "separate physical reference; not used for checkpoint selection",
        },
        "checkpoint_policy": {
            "no_early_stopping": True,
            "eligible_from_epoch": config.checkpoint_min_epoch,
            "reason": "only compare checkpoints from the final max-K training stage",
            "checkpoint_steps": config.checkpoint_steps,
            "primary_metric": "mean log10 validation residual p95 across checkpoint steps",
        },
        "num_experiments": len(summaries),
        "component_rankings": make_component_ranking(summaries),
        "experiments": summaries,
    }
    save_json(
        overall_report,
        output_dir / "minimal_component_ablation_summary.json",
    )

    plot_final_test_summary(
        summaries,
        output_dir / "minimal_components_best_checkpoint_test_summary.png",
    )
    plot_training_loss_summary(
        summaries,
        output_dir / "minimal_components_training_loss_summary.png",
    )

    print("\n" + "=" * 92)
    print("所有实验完成。")
    print(
        f"汇总 JSON: {output_dir / 'minimal_component_ablation_summary.json'}"
    )
    for record in summaries:
        print(
            f"- {record['experiment_name']}: "
            f"components={record['component_config_name']}, "
            f"best_epoch={record['best_validation_epoch']}, "
            f"test_p95={record['best_checkpoint_test']['final_residual_p95']:.4e}, "
            f"p_n={record['best_checkpoint_test']['pn_final_residual']:.4e}, "
            f"diverged={record['diverged']}"
        )


if __name__ == "__main__":
    main()
