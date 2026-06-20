"""
自由落体单帧变分问题：训练数据规模消融实验
================================================

实验目的
--------
只研究一个变量：固定局部采样区域内，训练初值数量增加以后，
MLP 学习型迭代器能否让真实测试初值 p_n 收敛到精确解 y_star。

本脚本刻意保持训练策略简单：
1. 物理问题固定；
2. 网络结构固定为 12 -> 32 -> 32 -> 3；
3. 输入不做归一化；
4. 网络输出不乘 dt，不做额外缩放；
5. loss 只使用隐式欧拉变分能量：对展开轨迹中的 K 步能量直接求和；
6. 每个 mini-batch 内完整展开 K 步轨迹，不在步间 detach；
7. 每个 mini-batch 结束后执行一次 backward 和 optimizer.step；
8. batch_size 固定为 256；
9. K 从 1 开始，每 200 个 epoch 增加 1，最终增加至 5；
10. 训练集围绕精确解 y_star 在立方体 [-0.1, 0.1]^3 中均匀随机采样，
    不包含精确解 y_star；
11. 最终只测试未参与训练的真实初值 p_n。

默认依次测试训练集规模：
    10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000

说明
----
- 这里的“epoch”使用标准定义：完整遍历一次固定训练集。
- 数据规模越大，一个 epoch 包含的 mini-batch 数量越多，计算量会显著增加。
- 10^7 样本、1000 epochs 的完整实验计算量非常大。脚本提供命令行参数，
  可先使用较小规模和较少 epoch 做快速检查，再运行正式实验。
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

TORCH_DTYPE = torch.float32
torch.set_default_dtype(TORCH_DTYPE)

PLOT_FLOOR = 1e-12

# 固定训练数据规模。每一个规模对应一组独立实验。
DEFAULT_DATASET_SIZE_VALUES = [
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
]

# 固定局部采样范围：
# y_train = y_star + offset, offset ~ Uniform([-R, R]^3)
DEFAULT_SAMPLING_RADIUS = 0.01
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 1000
DEFAULT_EVAL_INTERVAL = 100
DEFAULT_FINAL_TEST_STEPS = 50
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 200
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5

# 固定随机种子，确保不同数据规模之间可比较。
# 每一个较小数据集都是最大数据集的前缀，因此数据集规模扩大时保留已有样本。
DATASET_RANDOM_SEED = 123
MODEL_RANDOM_SEED = 42
SHUFFLE_RANDOM_SEED = 456

# 为控制输出文件体积，训练点分布图最多展示这些样本。
MAX_SCATTER_POINTS = 20_000

# 按照原脚本保留两个候选优化器，用相同数据分别训练。
OPTIMIZER_CONFIGS = [
    {
        "optimizer_name": "sgd",
        "learning_rate": 1e-2,
    },
    {
        "optimizer_name": "adam",
        "learning_rate": 1e-4,
    },
]


# ============================================================
# 1. 数据结构与通用辅助函数
# ============================================================


@dataclass(frozen=True)
class RuntimeConfig:
    dataset_sizes: list[int]
    sampling_radius: float
    batch_size: int
    epochs: int
    eval_interval: int
    final_test_steps: int
    initial_k: int
    k_increase_interval: int
    k_increase_amount: int
    max_k: int
    device: str
    skip_contour: bool


def create_output_directory() -> Path:
    """在脚本同目录下创建同名输出目录。"""

    script_path = Path(__file__).resolve()
    output_dir = script_path.parent / script_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_positive_int_list(values: Iterable[int]) -> list[int]:
    """检查数据规模参数，并去重排序。"""

    cleaned = sorted({int(value) for value in values})
    if not cleaned:
        raise ValueError("dataset_sizes must not be empty.")
    if cleaned[0] <= 0:
        raise ValueError("Every dataset size must be positive.")
    return cleaned


def get_k_for_epoch(epoch: int, config: RuntimeConfig) -> int:
    """根据 epoch 返回当前展开轨迹长度 K。"""

    return min(
        config.initial_k
        + (epoch // config.k_increase_interval) * config.k_increase_amount,
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
        直接输出位置更新量 delta_y，不乘 dt，不做其他缩放。
    """

    def __init__(self) -> None:
        super().__init__()

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
            return self.net(inp)

        if y.ndim != 2 or y.shape[-1] != 3:
            raise ValueError(f"Expected y shape [3] or [B, 3], got {tuple(y.shape)}")

        batch_size = y.shape[0]
        history_batch = self._expand_feature_for_batch(history, batch_size)
        params_batch = self._expand_feature_for_batch(params, batch_size)
        inp = torch.cat([y, history_batch, params_batch], dim=-1)
        return self.net(inp)


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
# 3. 固定随机训练集
# ============================================================


def make_master_training_dataset(
    y_star: torch.Tensor,
    max_num_points: int,
    sampling_radius: float,
    seed: int = DATASET_RANDOM_SEED,
) -> torch.Tensor:
    """
    构造最大规模的固定训练集，存放在 CPU。

    每个训练点满足：
        y = y_star + offset
        offset ~ Uniform([-sampling_radius, sampling_radius]^3)

    不允许训练集包含精确解 y_star。
    较小数据集使用最大训练集的前缀，因此规模扩大时保留已有样本。
    """

    if max_num_points <= 0:
        raise ValueError("max_num_points must be positive.")
    if sampling_radius <= 0.0:
        raise ValueError("sampling_radius must be positive.")

    y_star_cpu = y_star.detach().cpu().to(dtype=TORCH_DTYPE)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    offsets = (
        2.0
        * torch.rand(
            (max_num_points, 3),
            generator=generator,
            dtype=TORCH_DTYPE,
            device="cpu",
        )
        - 1.0
    ) * sampling_radius

    # 理论上随机浮点采样恰好得到 [0, 0, 0] 的概率极低；
    # 这里仍显式排除，保证实验定义严格成立。
    zero_mask = torch.all(offsets == 0.0, dim=1)
    while bool(zero_mask.any()):
        replacement_count = int(zero_mask.sum().item())
        offsets[zero_mask] = (
            2.0
            * torch.rand(
                (replacement_count, 3),
                generator=generator,
                dtype=TORCH_DTYPE,
                device="cpu",
            )
            - 1.0
        ) * sampling_radius
        zero_mask = torch.all(offsets == 0.0, dim=1)

    return y_star_cpu.unsqueeze(0) + offsets


def iterate_shuffled_minibatches(
    training_states: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> Iterable[torch.Tensor]:
    """
    将固定 CPU 训练集打乱后逐批返回。

    一个 epoch 会完整遍历一次 training_states。
    """

    num_points = training_states.shape[0]
    permutation = torch.randperm(num_points, generator=generator)

    for start in range(0, num_points, batch_size):
        indices = permutation[start : start + batch_size]
        yield training_states.index_select(0, indices)


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
    training_states_cpu: torch.Tensor,
    dataset_size: int,
    optimizer_name: str,
    learning_rate: float,
    config: RuntimeConfig,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> dict:
    """运行一组“优化器 × 数据规模”实验。"""

    experiment_name = (
        f"{optimizer_name}_lr_{learning_rate:.0e}_"
        f"num_samples_{dataset_size}"
    )
    output_dir = base_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🧪 实验组: {experiment_name}")
    print(f"📁 输出目录: {output_dir}")
    print("=" * 80)

    device = torch.device(config.device)
    p_n_device = p_n.to(device)
    v_n_device = v_n.to(device)

    y_star = p_n_device + dt * v_n_device - dt**2 * torch.tensor(
        [0.0, 0.0, g],
        dtype=TORCH_DTYPE,
        device=device,
    )
    history = torch.cat([p_n_device, v_n_device])
    params = torch.tensor([m, g, dt], dtype=TORCH_DTYPE, device=device)

    e_star = float(variational_energy(y_star, p_n_device, v_n_device, m, g, dt).item())
    newton_solution = p_n_device + newton_direction(
        p_n_device,
        p_n_device,
        v_n_device,
        m,
        g,
        dt,
    )

    # 固定训练集：仅使用最大主数据集的前 dataset_size 个样本。
    training_states = training_states_cpu[:dataset_size]

    torch.manual_seed(MODEL_RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)

    model = MLPOptimizer().to(device)
    optimizer = create_optimizer(model, optimizer_name, learning_rate)

    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(SHUFFLE_RANDOM_SEED)

    batches_per_epoch = math.ceil(dataset_size / config.batch_size)
    train_log = []
    eval_log = []
    diverged = False
    divergence_reason = None
    divergence_epoch = None
    divergence_batch_index = None

    print(f"device={device}")
    print(f"torch_dtype={TORCH_DTYPE}")
    print(f"dataset_size={dataset_size}")
    print(f"batch_size={config.batch_size}")
    print(f"batches_per_epoch={batches_per_epoch}")
    print(f"sampling_center=y_star={tensor_to_list(y_star)}")
    print(f"sampling_radius={config.sampling_radius}")
    print("sampling_distribution=uniform_cube")
    print("training_set_contains_y_star=False")
    print("use_input_normalization=False")
    print("use_output_dt_scaling=False")
    print("loss=sum_of_stepwise_mean_variational_energy_over_batch")
    print(
        "trajectory_backpropagation=full_unroll_without_detach; "
        "one backward and one optimizer.step per mini-batch"
    )
    print(
        f"K schedule: initial={config.initial_k}, "
        f"increase +{config.k_increase_amount} every "
        f"{config.k_increase_interval} epochs, max={config.max_k}"
    )

    for epoch in range(config.epochs):
        model.train()
        k = get_k_for_epoch(epoch, config)

        summed_sample_weighted_trajectory_loss = 0.0
        num_batches = 0
        num_seen_samples = 0

        for batch_index, batch_cpu in enumerate(
            iterate_shuffled_minibatches(
                training_states,
                config.batch_size,
                shuffle_generator,
            )
        ):
            y = batch_cpu.to(device=device, dtype=TORCH_DTYPE)
            current_batch_size = y.shape[0]

            optimizer.zero_grad(set_to_none=True)
            trajectory_loss = torch.zeros((), dtype=TORCH_DTYPE, device=device)

            # 同一组模型参数下完整展开 K 步；轨迹内部不 detach。
            for _ in range(k):
                delta = model(y, history, params)
                y = y + delta
                trajectory_loss = trajectory_loss + variational_energy(
                    y,
                    p_n_device,
                    v_n_device,
                    m,
                    g,
                    dt,
                ).mean()

            # 保持最朴素的定义：K 个展开步骤的变分能量直接求和。
            if not bool(torch.isfinite(trajectory_loss)):
                diverged = True
                divergence_reason = "non-finite mini-batch trajectory loss"
                divergence_epoch = epoch
                divergence_batch_index = batch_index
                break

            trajectory_loss.backward()
            optimizer.step()

            if not is_model_finite(model):
                diverged = True
                divergence_reason = "non-finite model parameter after optimizer.step"
                divergence_epoch = epoch
                divergence_batch_index = batch_index
                break

            summed_sample_weighted_trajectory_loss += (
                float(trajectory_loss.item()) * current_batch_size
            )
            num_batches += 1
            num_seen_samples += current_batch_size

        if diverged:
            print(
                f"⚠️ 训练发散：epoch={divergence_epoch}, "
                f"batch_index={divergence_batch_index}, "
                f"reason={divergence_reason}"
            )
            break

        if num_batches == 0:
            raise RuntimeError("No mini-batch was produced for the current epoch.")

        mean_training_trajectory_loss = (
            summed_sample_weighted_trajectory_loss / num_seen_samples
        )
        # 仅用于可读性：训练目标本身仍然是 K 步能量之和。
        training_gap_for_readability = mean_training_trajectory_loss - k * e_star

        train_log.append(
            {
                "epoch": epoch,
                "K": k,
                "num_batches": num_batches,
                "num_seen_samples": num_seen_samples,
                "mean_training_trajectory_loss": mean_training_trajectory_loss,
                "mean_training_gap_for_readability": training_gap_for_readability,
            }
        )

        if epoch % config.eval_interval == 0 or epoch == config.epochs - 1:
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
                    "epoch": epoch,
                    "K": k,
                    "evaluation_steps": config.max_k,
                    "reference_gap": final_item["gap"],
                    "reference_residual_norm": final_item["residual_norm"],
                    "reference_trajectory": reference_trajectory,
                }
            )

            print(
                f"Epoch {epoch:4d} | K={k} | "
                f"train trajectory energy sum={mean_training_trajectory_loss:.8e} | "
                f"train gap(readability)={training_gap_for_readability:.4e} | "
                f"p_n test gap({config.max_k} steps)={final_item['gap']:.4e} | "
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

    report = {
        "config": {
            "experiment_name": experiment_name,
            "torch_dtype": str(TORCH_DTYPE),
            "device": str(device),
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "dataset_size": dataset_size,
            "dataset_mode": "fixed_uniform_random_points_near_y_star",
            "sampling_center": "y_star",
            "sampling_radius": config.sampling_radius,
            "sampling_distribution": "Uniform([-R, R]^3)",
            "training_set_contains_y_star": False,
            "batch_size": config.batch_size,
            "batches_per_epoch": batches_per_epoch,
            "epochs": config.epochs,
            "initial_K": config.initial_k,
            "K_increase_interval": config.k_increase_interval,
            "K_increase_amount": config.k_increase_amount,
            "max_K": config.max_k,
            "use_input_normalization": False,
            "use_output_dt_scaling": False,
            "loss": (
                "For each mini-batch, sum the mean variational energy of all "
                "unrolled steps. No additional loss term and no division by K."
            ),
            "backpropagation": (
                "Full unrolled trajectory backpropagation without detach; "
                "one backward call and one optimizer.step per mini-batch."
            ),
            "evaluation_mode": "held_out_reference_initial_state_p_n_only",
            "p_n": tensor_to_list(p_n_device),
            "v_n": tensor_to_list(v_n_device),
            "m": m,
            "g": g,
            "dt": dt,
            "y_star": tensor_to_list(y_star),
            "newton_solution": tensor_to_list(newton_solution),
            "E_star": e_star,
            "dataset_random_seed": DATASET_RANDOM_SEED,
            "model_random_seed": MODEL_RANDOM_SEED,
            "shuffle_random_seed": SHUFFLE_RANDOM_SEED,
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
            "divergence_epoch": divergence_epoch,
            "divergence_batch_index": divergence_batch_index,
            "final_reference_rollout_is_finite": final_reference_is_finite,
        },
        "summary": {
            "final_reference_gap_after_fixed_steps": final_mlp_item["gap"],
            "final_reference_residual_norm_after_fixed_steps": final_mlp_item[
                "residual_norm"
            ],
            "newton_reference_gap_after_fixed_steps": final_newton_item["gap"],
            "newton_reference_residual_norm_after_fixed_steps": final_newton_item[
                "residual_norm"
            ],
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
        training_states=training_states,
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
        "dataset_size": dataset_size,
        "batch_size": config.batch_size,
        "batches_per_epoch": batches_per_epoch,
        "sampling_radius": config.sampling_radius,
        "diverged": diverged,
        "divergence_reason": divergence_reason,
        "divergence_epoch": divergence_epoch,
        "divergence_batch_index": divergence_batch_index,
        "final_reference_rollout_is_finite": final_reference_is_finite,
        "final_reference_gap_after_fixed_steps": final_mlp_item["gap"],
        "final_reference_residual_norm_after_fixed_steps": final_mlp_item[
            "residual_norm"
        ],
        "newton_reference_gap_after_fixed_steps": final_newton_item["gap"],
        "newton_reference_residual_norm_after_fixed_steps": final_newton_item[
            "residual_norm"
        ],
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
    """绘制训练 loss 和 p_n 周期性测试曲线。"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    train_epochs = [item["epoch"] for item in train_log]
    train_gaps = [
        finite_plot_value(item["mean_training_gap_for_readability"])
        for item in train_log
    ]
    axes[0].plot(train_epochs, train_gaps)
    axes[0].set_yscale("log")
    axes[0].set_title("Training Trajectory Energy-Sum Gap (Readability Only)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"Mean $\sum_k [E(y^{(k)})-E(y^*)]$")
    axes[0].grid(True, alpha=0.3)

    eval_epochs = [item["epoch"] for item in eval_log]
    eval_gaps = [finite_plot_value(item["reference_gap"]) for item in eval_log]
    axes[1].plot(eval_epochs, eval_gaps, marker="o")
    axes[1].set_yscale("log")
    axes[1].set_title(r"Held-Out $p_n$ Test Gap")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"$E(y)-E(y^*)$")
    axes[1].grid(True, alpha=0.3)

    eval_residuals = [
        finite_plot_value(item["reference_residual_norm"]) for item in eval_log
    ]
    axes[2].plot(eval_epochs, eval_residuals, marker="s")
    axes[2].set_yscale("log")
    axes[2].set_title(r"Held-Out $p_n$ Test Residual")
    axes[2].set_xlabel("Epoch")
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
    training_states: torch.Tensor,
    y_star: Sequence[float],
    initial_y: Sequence[float],
    save_path: Path,
) -> None:
    """绘制固定训练集的抽样分布，避免 10^7 点全部绘制造成图片过大。"""

    num_points = training_states.shape[0]
    if num_points <= MAX_SCATTER_POINTS:
        sample = training_states
    else:
        # 仅用于画图，均匀选择索引，不影响训练。
        indices = torch.linspace(0, num_points - 1, MAX_SCATTER_POINTS).long()
        sample = training_states.index_select(0, indices)

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
        label=f"Training samples (shown: {sample_np.shape[0]}/{num_points})",
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
    ax.set_title("Fixed Uniform-Random Training Dataset near y*")
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
            [item for item in summaries if item["optimizer_name"] == optimizer_name],
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
        ax.set_xlabel("Number of fixed training initial states")
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
        description="Free-fall MLP optimizer dataset-size ablation experiment."
    )
    parser.add_argument(
        "--dataset-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_DATASET_SIZE_VALUES,
        help="Fixed dataset sizes to evaluate.",
    )
    parser.add_argument("--sampling-radius", type=float, default=DEFAULT_SAMPLING_RADIUS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--final-test-steps", type=int, default=DEFAULT_FINAL_TEST_STEPS)
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument(
        "--k-increase-interval", type=int, default=DEFAULT_K_INCREASE_INTERVAL
    )
    parser.add_argument(
        "--k-increase-amount", type=int, default=DEFAULT_K_INCREASE_AMOUNT
    )
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device, for example: cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--skip-contour",
        action="store_true",
        help="Skip 2D energy contour images during quick tests.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    """检查命令行配置并转换为不可变数据结构。"""

    dataset_sizes = ensure_positive_int_list(args.dataset_sizes)
    if args.sampling_radius <= 0.0:
        raise ValueError("sampling_radius must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")
    if args.final_test_steps <= 0:
        raise ValueError("final_test_steps must be positive.")
    if args.initial_k <= 0:
        raise ValueError("initial_k must be positive.")
    if args.k_increase_interval <= 0:
        raise ValueError("k_increase_interval must be positive.")
    if args.k_increase_amount <= 0:
        raise ValueError("k_increase_amount must be positive.")
    if args.max_k < args.initial_k:
        raise ValueError("max_k must be greater than or equal to initial_k.")

    return RuntimeConfig(
        dataset_sizes=dataset_sizes,
        sampling_radius=float(args.sampling_radius),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        eval_interval=int(args.eval_interval),
        final_test_steps=int(args.final_test_steps),
        initial_k=int(args.initial_k),
        k_increase_interval=int(args.k_increase_interval),
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

    max_dataset_size = max(config.dataset_sizes)
    print(
        f"生成最大固定训练集：num_points={max_dataset_size}, "
        f"center=y_star={tensor_to_list(y_star)}, "
        f"radius={config.sampling_radius}"
    )
    master_training_dataset = make_master_training_dataset(
        y_star=y_star,
        max_num_points=max_dataset_size,
        sampling_radius=config.sampling_radius,
        seed=DATASET_RANDOM_SEED,
    )

    # 严格检查：训练集不能包含 y_star。
    contains_y_star = bool(
        torch.any(torch.all(master_training_dataset == y_star.unsqueeze(0), dim=1)).item()
    )
    if contains_y_star:
        raise RuntimeError("The generated training dataset unexpectedly contains y_star.")

    experiment_summaries = []
    for optimizer_config in OPTIMIZER_CONFIGS:
        for dataset_size in config.dataset_sizes:
            summary = run_experiment(
                base_output_dir=base_output_dir,
                training_states_cpu=master_training_dataset,
                dataset_size=dataset_size,
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

    overall_report = {
        "experiment_type": "fixed_local_uniform_random_dataset_size_ablation",
        "purpose": (
            "Keep the physical problem, sampling radius, network architecture, "
            "loss, batch size, optimizer configuration, and trajectory-unroll "
            "strategy fixed. Change only the number of fixed random training "
            "initial states near y_star, then evaluate whether the held-out "
            "physical initial state p_n converges."
        ),
        "runtime_config": asdict(config),
        "dataset_random_seed": DATASET_RANDOM_SEED,
        "model_random_seed": MODEL_RANDOM_SEED,
        "shuffle_random_seed": SHUFFLE_RANDOM_SEED,
        "sampling_center": "y_star",
        "sampling_radius": config.sampling_radius,
        "sampling_distribution": "Uniform([-R, R]^3)",
        "training_set_contains_y_star": False,
        "batch_size": config.batch_size,
        "use_input_normalization": False,
        "use_output_dt_scaling": False,
        "loss": (
            "For each mini-batch, sum the mean variational energy of all "
            "unrolled steps. No additional loss term and no division by K."
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
            f"N={item['dataset_size']}, "
            f"batch_size={item['batch_size']}, "
            f"training_diverged={item['diverged']}, "
            f"final_rollout_finite={item['final_reference_rollout_is_finite']}, "
            f"最终 p_n gap={item['final_reference_gap_after_fixed_steps']:.4e}, "
            f"最终 p_n residual={item['final_reference_residual_norm_after_fixed_steps']:.4e}"
        )


if __name__ == "__main__":
    main()
