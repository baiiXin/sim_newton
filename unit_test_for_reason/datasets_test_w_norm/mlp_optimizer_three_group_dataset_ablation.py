import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 适配无显示器 Linux 环境

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D


# ============================================================
# 0. 可调参数
# ============================================================

# True：绘制相对于原始初值 y0 的位移，更容易看清聚集在一起的点
# False：绘制绝对空间坐标
PLOT_RELATIVE_COORDINATES = False

# 每多少个 epoch 使用一种颜色
COLOR_BUCKET_SIZE = 200

# 引入 full_energy_experiment.py 中的结构改进：
# 1. 输入归一化
# 2. 输出按 dt 缩放
USE_NORMALIZATION = True
USE_DT_SCALING = True

# 数据集消融参数。三组实验的唯一区别是训练初值集合。
# 组 1：只有原始初值 y0。
# 组 2：从 y0 到理论最优解 y* 的 line 采样点，包含 y0 和 y*。
# 组 3：原始初值 y0，加上 y* 附近的随机扰动点；不额外加入 line。
LINE_NUM_POINTS = 11
LOCAL_NUM_POINTS = 32
LOCAL_STD_DT_UNITS = 1.0
LOCAL_RANDOM_SEED = 123

# 每组实验使用相同的网络初始化随机种子，使消融对比更公平。
MODEL_RANDOM_SEED = 42

ABLATION_EXPERIMENTS = [
    {
        "name": "01_initial_only",
        "description": "Only the original initial state y0.",
        "dataset_mode": "initial_only",
    },
    {
        "name": "02_initial_plus_line",
        "description": "Line samples from y0 to y_star, including both endpoints.",
        "dataset_mode": "initial_plus_line",
    },
    {
        "name": "03_initial_plus_endpoint_perturbations",
        "description": "The original initial state y0 plus random perturbations near y_star.",
        "dataset_mode": "initial_plus_endpoint_perturbations",
    },
]

# 训练与评估参数
EPOCHS = 1000
INITIAL_K = 1
K_INCREASE_INTERVAL = 200
MAX_K = 10
EVAL_INTERVAL = 100
EVAL_STEPS = 10
FINAL_TEST_STEPS = 15


# ============================================================
# 1. 输出目录
# ============================================================

def create_output_directory() -> Path:
    """
    在当前 Python 文件所在目录下创建同名子目录，并返回该目录。

    例如：
        脚本路径: /path/to/mlp_optimizer_two_initial_points.py
        输出目录: /path/to/mlp_optimizer_two_initial_points/
    """

    script_path = Path(__file__).resolve()
    output_dir = script_path.parent / script_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================
# 2. 模型与隐式欧拉变分能量
# ============================================================

class MLPOptimizer(nn.Module):
    """学习型迭代优化器。"""

    def __init__(
        self,
        use_normalization=False,
        use_dt_scaling=False,
        input_mean=None,
        input_std=None,
    ):
        super().__init__()

        self.use_normalization = use_normalization
        self.use_dt_scaling = use_dt_scaling

        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        if input_mean is None:
            input_mean = torch.zeros(12)
        if input_std is None:
            input_std = torch.ones(12)

        self.register_buffer("input_mean", input_mean.clone().detach())
        self.register_buffer("input_std", input_std.clone().detach())

    def forward(self, y, history, params):
        inp = torch.cat([y, history, params], dim=-1)

        if self.use_normalization:
            inp = (inp - self.input_mean) / self.input_std

        delta = self.net(inp)

        if self.use_dt_scaling:
            delta = params[2] * delta

        return delta


def variational_energy(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    隐式欧拉变分能量:

        E(y) =
            m / (2 * dt^2) * ||y - p_n - dt * v_n||^2
            + m * g * y_z
    """

    residual = y - p_n - dt * v_n

    kinetic_term = (m / (2.0 * dt**2)) * torch.sum(residual**2)
    potential_term = m * g * y[2]

    return kinetic_term + potential_term


def newton_direction(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    Newton 法方向:

        delta = -H^{-1} grad

    对于当前问题:

        H = (m / dt^2) I

        grad =
            (m / dt^2) * (y - p_n - dt * v_n)
            + [0, 0, m * g]^T

    由于目标函数是严格凸二次函数，Newton 法一步即可到达最优解。
    """

    residual = y - p_n - dt * v_n

    grad = (m / dt**2) * residual
    grad = grad.clone()  # 避免直接修改中间张量
    grad[2] += m * g

    hess_inv = dt**2 / m

    return -grad * hess_inv


def make_ablation_training_states(
    dataset_mode,
    y0,
    y_star,
    dt,
    num_line_points=LINE_NUM_POINTS,
    num_local_points=LOCAL_NUM_POINTS,
    local_std_dt_units=LOCAL_STD_DT_UNITS,
    seed=LOCAL_RANDOM_SEED,
):
    """
    根据数据集消融模式创建训练初值。

    initial_only:
        只有原始初值 y0。

    initial_plus_line:
        从 y0 到 y_star 均匀采样 num_line_points 个点。
        torch.linspace(0, 1, ...) 已经包含 y0，因此不会重复加入 y0。

    initial_plus_endpoint_perturbations:
        先加入原始初值 y0，再加入 num_local_points 个 y_star 附近扰动点。
        这一组不加入 line，也不额外加入精确的 y_star。
    """

    if dataset_mode == "initial_only":
        return [y0.clone()]

    if dataset_mode == "initial_plus_line":
        return [
            (1.0 - alpha) * y0 + alpha * y_star
            for alpha in torch.linspace(0.0, 1.0, num_line_points)
        ]

    if dataset_mode == "initial_plus_endpoint_perturbations":
        train_states = [y0.clone()]
        gen = torch.Generator(device=y0.device)
        gen.manual_seed(seed)

        for _ in range(num_local_points):
            noise = torch.randn(3, generator=gen, device=y0.device)
            train_states.append(y_star + dt * local_std_dt_units * noise)

        return train_states

    raise ValueError(f"未知的数据集消融模式: {dataset_mode}")


def compute_input_normalizer(train_states, history, params):
    x = torch.stack(
        [torch.cat([y, history, params], dim=-1) for y in train_states],
        dim=0,
    )
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


# ============================================================
# 3. 绘图辅助函数
# ============================================================

def to_plot_coordinates(points, reference_y0, relative=True):
    """
    将点转换为绘图坐标。

    relative=True:
        绘制 y - reference_y0

    relative=False:
        绘制绝对坐标 y
    """

    points_np = np.asarray(points, dtype=float).reshape(-1, 3)
    y0_np = np.asarray(reference_y0, dtype=float).reshape(1, 3)

    if relative:
        points_np = points_np - y0_np

    return points_np


def set_equal_3d_axes(ax, points):
    """
    让 3D 图的三个坐标轴具有相同尺度，避免轨迹形状被拉伸变形。
    """

    points_np = np.asarray(points, dtype=float).reshape(-1, 3)

    center = points_np.mean(axis=0)
    span = np.ptp(points_np, axis=0)

    # 防止所有点几乎重合时坐标轴范围退化
    radius = max(float(span.max()) / 2.0, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def set_axis_labels(ax, relative=True):
    """
    设置三维坐标轴名称。
    """

    if relative:
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(r"$\Delta y$")
        ax.set_zlabel(r"$\Delta z$")
    else:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")


# ============================================================
# 4. 单个初值的最终测试轨迹
# ============================================================

def plot_final_test_distribution(
    mlp_hist,
    newton_solution,
    initial_y,
    reference_y0,
    save_path,
    relative=True,
):
    """
    绘制训练结束后，MLP 优化器从指定初值出发的测试轨迹。

    标记规则:
        0, 1, 2, ... : MLP 迭代次序
        x            : 当前测试使用的初值
        *            : Newton 法收敛解
    """

    mlp_points = to_plot_coordinates(
        [item["y"] for item in mlp_hist["iterations"]],
        reference_y0=reference_y0,
        relative=relative,
    )

    newton_point = to_plot_coordinates(
        [newton_solution],
        reference_y0=reference_y0,
        relative=relative,
    )[0]

    initial_point = to_plot_coordinates(
        [initial_y],
        reference_y0=reference_y0,
        relative=relative,
    )[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        mlp_points[:, 0],
        mlp_points[:, 1],
        mlp_points[:, 2],
        "-o",
        linewidth=1.5,
        markersize=4,
        label="MLP test trajectory",
    )

    for step, point in enumerate(mlp_points):
        ax.text(
            point[0],
            point[1],
            point[2],
            f"  {step}",
            fontsize=9,
        )

    ax.scatter(
        initial_point[0],
        initial_point[1],
        initial_point[2],
        marker="x",
        s=140,
        c="black",
        linewidths=2.0,
        label="Test initial point",
    )

    ax.scatter(
        newton_point[0],
        newton_point[1],
        newton_point[2],
        marker="*",
        s=320,
        c="crimson",
        label="Newton converged solution",
    )

    ax.text(
        newton_point[0],
        newton_point[1],
        newton_point[2],
        "  * Newton",
        fontsize=10,
        color="crimson",
    )

    all_points = np.vstack(
        [
            mlp_points,
            initial_point.reshape(1, 3),
            newton_point.reshape(1, 3),
        ]
    )

    set_equal_3d_axes(ax, all_points)
    set_axis_labels(ax, relative=relative)

    ax.set_title("Final MLP Test Iteration Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"🖼️ 最终测试轨迹图已保存至: {save_path}")


# ============================================================
# 5. 绘制整个训练过程中的训练点和结果点
# ============================================================

def plot_training_points_and_results(
    visualization_log,
    newton_solution,
    training_initial_points,
    reference_y0,
    epochs,
    bucket_size,
    save_path,
    relative=True,
):
    """
    可视化整个训练过程中的点分布。

    每个 epoch 都从当前消融组中的全部训练初值分别出发，
    使用同一个 MLP 依次执行训练。

    左图:
        所有轨迹中的训练输入点 x^(0), ..., x^(K-1)

    右图:
        所有轨迹在每个 epoch 中的最终结果点 x^(K)

    颜色:
        每 bucket_size 个 epoch 使用一种颜色。

    特殊标记:
        黑色 x : 当前消融组中的训练初值
        红色 * : Newton 法收敛解

    实现说明:
        先按颜色区间聚合点，再绘制散点。这样可以避免为每个 epoch
        单独创建 Matplotlib scatter 对象，在训练轮数较多时显著减少
        保存图片所需的时间。
    """

    num_buckets = (epochs + bucket_size - 1) // bucket_size
    cmap = plt.get_cmap("tab10", num_buckets)

    fig = plt.figure(figsize=(19, 8))

    ax_train = fig.add_subplot(121, projection="3d")
    ax_result = fig.add_subplot(122, projection="3d")

    train_points_by_bucket = [[] for _ in range(num_buckets)]
    result_points_by_bucket = [[] for _ in range(num_buckets)]

    for epoch_item in visualization_log:
        bucket = epoch_item["bucket"]

        for trajectory in epoch_item["trajectories"]:
            train_points = to_plot_coordinates(
                trajectory["train_points"],
                reference_y0=reference_y0,
                relative=relative,
            )

            result_point = to_plot_coordinates(
                [trajectory["result_point"]],
                reference_y0=reference_y0,
                relative=relative,
            )

            train_points_by_bucket[bucket].append(train_points)
            result_points_by_bucket[bucket].append(result_point)

    all_train_points = []
    all_result_points = []

    for bucket in range(num_buckets):
        if not train_points_by_bucket[bucket]:
            continue

        color = cmap(bucket)
        bucket_train_points = np.vstack(train_points_by_bucket[bucket])
        bucket_result_points = np.vstack(result_points_by_bucket[bucket])

        ax_train.scatter(
            bucket_train_points[:, 0],
            bucket_train_points[:, 1],
            bucket_train_points[:, 2],
            marker="o",
            s=11,
            alpha=0.30,
            color=color,
        )

        ax_result.scatter(
            bucket_result_points[:, 0],
            bucket_result_points[:, 1],
            bucket_result_points[:, 2],
            marker="^",
            s=24,
            alpha=0.75,
            color=color,
        )

        all_train_points.append(bucket_train_points)
        all_result_points.append(bucket_result_points)

    newton_point = to_plot_coordinates(
        [newton_solution],
        reference_y0=reference_y0,
        relative=relative,
    )[0]

    initial_points_for_plot = to_plot_coordinates(
        training_initial_points,
        reference_y0=reference_y0,
        relative=relative,
    )

    for ax in (ax_train, ax_result):
        for initial_index, initial_point in enumerate(initial_points_for_plot):
            ax.scatter(
                initial_point[0],
                initial_point[1],
                initial_point[2],
                marker="x",
                s=120,
                c="black",
                linewidths=2.0,
            )

            ax.text(
                initial_point[0],
                initial_point[1],
                initial_point[2],
                f"  init {initial_index}",
                fontsize=9,
                color="black",
            )

        ax.scatter(
            newton_point[0],
            newton_point[1],
            newton_point[2],
            marker="*",
            s=360,
            c="crimson",
        )

        ax.text(
            newton_point[0],
            newton_point[1],
            newton_point[2],
            "  * Newton",
            fontsize=10,
            color="crimson",
        )

        set_axis_labels(ax, relative=relative)
        ax.grid(True, alpha=0.3)

    ax_train.set_title(
        r"All Training Input Points: "
        r"$x^{(0)}, x^{(1)}, \ldots, x^{(K-1)}$"
    )

    ax_result.set_title(
        r"All Epoch Result Points: $x^{(K)}$"
    )

    train_points_for_limits = np.vstack(
        all_train_points
        + [
            initial_points_for_plot,
            newton_point.reshape(1, 3),
        ]
    )

    result_points_for_limits = np.vstack(
        all_result_points
        + [
            initial_points_for_plot,
            newton_point.reshape(1, 3),
        ]
    )

    set_equal_3d_axes(ax_train, train_points_for_limits)
    set_equal_3d_axes(ax_result, result_points_for_limits)

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=r"Training input points $x^{(0)}, \ldots, x^{(K-1)}$",
            markerfacecolor="gray",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label=r"Epoch result points $x^{(K)}$",
            markerfacecolor="gray",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            label="Training initial states",
            linestyle="None",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Newton converged solution",
            markerfacecolor="crimson",
            markersize=14,
        ),
    ]

    color_handles = []

    for bucket in range(num_buckets):
        start_epoch = bucket * bucket_size
        end_epoch = min(
            (bucket + 1) * bucket_size - 1,
            epochs - 1,
        )

        color_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Epoch {start_epoch}-{end_epoch}",
                markerfacecolor=cmap(bucket),
                markersize=8,
            )
        )

    fig.legend(
        handles=marker_handles + color_handles,
        loc="center right",
        bbox_to_anchor=(1.00, 0.50),
    )

    fig.suptitle(
        "Training Data and Result Distribution by Epoch Range",
        fontsize=14,
    )

    plt.tight_layout(rect=[0.00, 0.00, 0.82, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"🖼️ 训练点和结果点分布图已保存至: {save_path}")

# ============================================================
# 6. 评估辅助函数
# ============================================================

def evaluate_mlp_steps(
    mlp,
    initial_y,
    history,
    params,
    p_n,
    v_n,
    m,
    g,
    dt,
    num_steps,
):
    """
    从给定初值出发，对训练后的 MLP 迭代 num_steps 次。
    """

    y = initial_y.clone()
    steps = []

    for step in range(1, num_steps + 1):
        with torch.no_grad():
            delta = mlp(y, history, params)

        y = y + delta

        energy = variational_energy(
            y,
            p_n,
            v_n,
            m,
            g,
            dt,
        ).item()

        steps.append(
            {
                "step": step,
                "y": y.tolist(),
                "loss": energy,
            }
        )

    return steps


def compare_mlp_and_newton(
    mlp,
    initial_y,
    history,
    params,
    p_n,
    v_n,
    m,
    g,
    dt,
    E_star,
    max_steps,
):
    """
    从同一个初值出发，对比 MLP 迭代器和 Newton 法。
    """

    E0 = variational_energy(
        initial_y,
        p_n,
        v_n,
        m,
        g,
        dt,
    ).item()

    mlp_hist = {
        "init_y": initial_y.tolist(),
        "history": [
            p_n.tolist(),
            v_n.tolist(),
        ],
        "params": params.tolist(),
        "E_star": E_star,
        "iterations": [
            {
                "step": 0,
                "y": initial_y.tolist(),
                "loss": E0,
            }
        ],
    }

    newton_hist = {
        "init_y": initial_y.tolist(),
        "history": [
            p_n.tolist(),
            v_n.tolist(),
        ],
        "params": params.tolist(),
        "E_star": E_star,
        "iterations": [
            {
                "step": 0,
                "y": initial_y.tolist(),
                "loss": E0,
            }
        ],
    }

    y_mlp = initial_y.clone()
    y_newton = initial_y.clone()

    for step in range(1, max_steps + 1):
        with torch.no_grad():
            delta_mlp = mlp(y_mlp, history, params)

        y_mlp = y_mlp + delta_mlp

        energy_mlp = variational_energy(
            y_mlp,
            p_n,
            v_n,
            m,
            g,
            dt,
        ).item()

        mlp_hist["iterations"].append(
            {
                "step": step,
                "y": y_mlp.tolist(),
                "loss": energy_mlp,
                "delta_norm": torch.norm(delta_mlp).item(),
            }
        )

        delta_newton = newton_direction(
            y_newton,
            p_n,
            v_n,
            m,
            g,
            dt,
        )

        y_newton = y_newton + delta_newton

        energy_newton = variational_energy(
            y_newton,
            p_n,
            v_n,
            m,
            g,
            dt,
        ).item()

        newton_hist["iterations"].append(
            {
                "step": step,
                "y": y_newton.tolist(),
                "loss": energy_newton,
                "delta_norm": torch.norm(delta_newton).item(),
            }
        )

    return {
        "mlp": mlp_hist,
        "newton": newton_hist,
    }


def print_final_comparison(case_index, comparison, E_star, max_rows=5):
    """
    打印一个初值对应的前若干步最终测试结果。
    """

    mlp_hist = comparison["mlp"]
    newton_hist = comparison["newton"]

    print(f"📊 初值 {case_index} 的最终迭代结果对比（前 {max_rows} 步）:")
    print(
        f"{'Step':<5} | "
        f"{'MLP Loss':<14} | "
        f"{'MLP Gap':<14} | "
        f"{'Newton Gap':<14} | "
        f"{'MLP y'}"
    )

    print("-" * 100)

    num_rows = min(max_rows, len(mlp_hist["iterations"]))

    for row_index in range(num_rows):
        mlp_item = mlp_hist["iterations"][row_index]
        newton_item = newton_hist["iterations"][row_index]

        mlp_gap = mlp_item["loss"] - E_star
        newton_gap = newton_item["loss"] - E_star

        y_str = str(
            [
                round(value, 6)
                for value in mlp_item["y"]
            ]
        )

        print(
            f"{mlp_item['step']:<5} | "
            f"{mlp_item['loss']:<14.8f} | "
            f"{mlp_gap:<14.4e} | "
            f"{newton_gap:<14.4e} | "
            f"{y_str}"
        )

    print()


# ============================================================
# 7. 绘制四宫格统计图
# ============================================================

def plot_summary_report(
    train_log,
    eval_log,
    reference_comparison,
    E_star,
    save_path,
):
    """
    绘制四宫格统计图。

    图 1 和图 2 对所有训练初值取最差 gap；
    图 3 和图 4 使用参考初值 y0 对比 MLP 与 Newton 法。
    """

    mlp_hist = reference_comparison["mlp"]
    newton_hist = reference_comparison["newton"]

    gap_mlp = [
        max(item["loss"] - E_star, 1e-12)
        for item in mlp_hist["iterations"]
    ]

    gap_newton = [
        max(item["loss"] - E_star, 1e-12)
        for item in newton_hist["iterations"]
    ]

    train_gap = [
        max(item["max_final_loss"] - E_star, 1e-12)
        for item in train_log
    ]

    eval_gap = [
        max(item["max_final_gap"], 1e-12)
        for item in eval_log
    ]

    mlp_norms = [
        item["delta_norm"]
        for item in mlp_hist["iterations"][1:]
    ]

    newton_norms = [
        item["delta_norm"]
        for item in newton_hist["iterations"][1:]
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    axes[0, 0].plot(
        [item["epoch"] for item in train_log],
        train_gap,
        color="steelblue",
    )

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Worst Training Gap over Training States")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Gap")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(
        [item["epoch"] for item in eval_log],
        eval_gap,
        marker="o",
        color="darkgreen",
    )

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Worst Periodic Eval Gap over Training States")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Gap")
    axes[0, 1].grid(True, alpha=0.3)

    test_steps = np.arange(len(gap_mlp))

    axes[1, 0].plot(
        test_steps,
        gap_mlp,
        label="MLP Optimizer",
        marker="o",
    )

    axes[1, 0].plot(
        test_steps,
        gap_newton,
        label="Newton Method",
        marker="s",
        linestyle="--",
        color="crimson",
    )

    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Final Comparison from Original Initial Point")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Gap")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(
        np.arange(len(mlp_norms)),
        mlp_norms,
        label=r"MLP $\|\Delta y\|$",
        marker="^",
    )

    axes[1, 1].plot(
        np.arange(len(newton_norms)),
        newton_norms,
        label=r"Newton $\|\Delta y\|$",
        marker="v",
        linestyle="--",
        color="crimson",
    )

    axes[1, 1].set_title("Update Step Magnitude from Original Initial Point")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel(r"$\|\Delta y\|_2$")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"🖼️ 常规统计图已保存至: {save_path}")


# ============================================================
# 8. 主程序
# ============================================================
# ============================================================
# 8. 单组实验
# ============================================================

def run_experiment(
    experiment,
    base_output_dir,
    p_n,
    v_n,
    m,
    g,
    dt,
):
    """
    运行一组数据集消融实验，并将该组全部结果写入独立子目录。

    三组实验保持网络结构、初始化种子、优化器和训练策略一致，
    只改变 training_initial_points。
    """

    experiment_name = experiment["name"]
    experiment_description = experiment["description"]
    dataset_mode = experiment["dataset_mode"]

    output_dir = base_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"🧪 数据集消融组: {experiment_name}")
    print(f"说明: {experiment_description}")
    print(f"📁 本组输出目录: {output_dir}")
    print("=" * 72)

    # 每组实验都重置随机种子，确保网络初始化一致。
    torch.manual_seed(MODEL_RANDOM_SEED)

    # --------------------------------------------------------
    # 8.1 构造物理问题与该组训练数据集
    # --------------------------------------------------------

    y0 = p_n.clone()

    y_star = (
        p_n
        + dt * v_n
        - dt**2 * torch.tensor([0.0, 0.0, g])
    )

    history = torch.cat([p_n, v_n])
    params = torch.tensor([m, g, dt])

    training_initial_points = make_ablation_training_states(
        dataset_mode=dataset_mode,
        y0=y0,
        y_star=y_star,
        dt=dt,
    )

    if USE_NORMALIZATION:
        input_mean, input_std = compute_input_normalizer(
            training_initial_points,
            history,
            params,
        )
    else:
        input_mean = torch.zeros(12)
        input_std = torch.ones(12)

    # --------------------------------------------------------
    # 8.2 创建模型
    # --------------------------------------------------------

    mlp = MLPOptimizer(
        use_normalization=USE_NORMALIZATION,
        use_dt_scaling=USE_DT_SCALING,
        input_mean=input_mean,
        input_std=input_std,
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    epochs = EPOCHS
    K = INITIAL_K

    train_log = []
    eval_log = []
    visualization_log = []
    training_point_rows = []
    result_point_rows = []
    micro_step_rows = []

    # --------------------------------------------------------
    # 8.3 理论最优解与 Newton 收敛解
    # --------------------------------------------------------

    E_star = variational_energy(
        y_star,
        p_n,
        v_n,
        m,
        g,
        dt,
    ).item()

    with torch.no_grad():
        newton_solution = (
            y0
            + newton_direction(
                y0,
                p_n,
                v_n,
                m,
                g,
                dt,
            )
        )
        initial_deltas = [
            mlp(initial_point, history, params)
            for initial_point in training_initial_points
        ]

    print("🚀 开始训练：隐式欧拉变分能量最小化")
    print(f"当前状态: p_n={p_n.tolist()}, v_n={v_n.tolist()}")
    print(f"理论最优解: y*={y_star.tolist()}")
    print(f"Newton 一步解: {newton_solution.tolist()}")
    print(f"理论最优能量: E*={E_star:.8f}")
    print(f"dataset_mode={dataset_mode}")
    print(f"训练初值总数: {len(training_initial_points)}")
    print(f"USE_NORMALIZATION={USE_NORMALIZATION}")
    print(f"USE_DT_SCALING={USE_DT_SCALING}")
    for initial_index, (initial_y, initial_delta) in enumerate(
        zip(training_initial_points[:5], initial_deltas[:5])
    ):
        print(f"训练初值 {initial_index}: {initial_y.tolist()}")
        print(f"初始化网络在初值 {initial_index} 上的输出: {initial_delta.tolist()}")
    if len(training_initial_points) > 5:
        print(f"其余训练初值已省略显示: {len(training_initial_points) - 5} 个")
    print(
        f"策略: 每 {K_INCREASE_INTERVAL} 个 epoch 增加一次 K，"
        "步间 detach，单步反向传播"
    )
    print(f"可视化: 每 {COLOR_BUCKET_SIZE} 个 epoch 使用一种颜色\n")

    # --------------------------------------------------------
    # 8.4 训练
    # --------------------------------------------------------

    for epoch in range(epochs):
        if (
            epoch > 0
            and epoch % K_INCREASE_INTERVAL == 0
            and K < MAX_K
        ):
            K += 1

        epoch_trajectories = []
        epoch_final_losses = []

        # 当前消融组中的全部轨迹共享同一个 MLP，按顺序执行参数更新。
        for initial_index, initial_y in enumerate(training_initial_points):
            y = initial_y.clone()
            trajectory_train_points = []

            for iteration_index in range(K):
                y_before = y.detach().clone()
                trajectory_train_points.append(y_before)
                training_point_rows.append([
                    epoch,
                    epoch // COLOR_BUCKET_SIZE,
                    initial_index,
                    iteration_index,
                    y_before[0].item(),
                    y_before[1].item(),
                    y_before[2].item(),
                ])

                delta = mlp(y, history, params)
                y = y + delta
                loss = variational_energy(y, p_n, v_n, m, g, dt)
                objective_gap = max(float(loss.item() - E_star), 0.0)

                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

                micro_step_rows.append([
                    epoch,
                    epoch // COLOR_BUCKET_SIZE,
                    initial_index,
                    iteration_index + 1,
                    float(loss.item()),
                    objective_gap,
                ])

                y = y.detach()

            epoch_result_point = y.detach().clone()
            result_point_rows.append([
                epoch,
                epoch // COLOR_BUCKET_SIZE,
                initial_index,
                epoch_result_point[0].item(),
                epoch_result_point[1].item(),
                epoch_result_point[2].item(),
            ])
            epoch_final_losses.append(loss.item())

            epoch_trajectories.append(
                {
                    "initial_index": initial_index,
                    "initial_y": initial_y.tolist(),
                    "train_points": [
                        point.tolist()
                        for point in trajectory_train_points
                    ],
                    "result_point": epoch_result_point.tolist(),
                    "final_loss": loss.item(),
                }
            )

        mean_final_loss = float(np.mean(epoch_final_losses))
        max_final_loss = float(np.max(epoch_final_losses))

        visualization_log.append(
            {
                "epoch": epoch,
                "bucket": epoch // COLOR_BUCKET_SIZE,
                "K": K,
                "trajectories": epoch_trajectories,
            }
        )

        train_log.append(
            {
                "epoch": epoch,
                "K": K,
                "final_losses": epoch_final_losses,
                "mean_final_loss": mean_final_loss,
                "max_final_loss": max_final_loss,
            }
        )

        # ----------------------------------------------------
        # 每隔固定 epoch，对当前组中的全部训练初值做固定步数评估
        # ----------------------------------------------------

        if epoch % EVAL_INTERVAL == 0 or epoch == epochs - 1:
            evaluations = []
            final_gaps = []

            for initial_index, initial_y in enumerate(training_initial_points):
                eval_steps = evaluate_mlp_steps(
                    mlp=mlp,
                    initial_y=initial_y,
                    history=history,
                    params=params,
                    p_n=p_n,
                    v_n=v_n,
                    m=m,
                    g=g,
                    dt=dt,
                    num_steps=EVAL_STEPS,
                )

                final_gap = eval_steps[-1]["loss"] - E_star
                final_gaps.append(final_gap)

                evaluations.append(
                    {
                        "initial_index": initial_index,
                        "initial_y": initial_y.tolist(),
                        "steps": eval_steps,
                        "final_gap": final_gap,
                    }
                )

            max_final_gap = float(np.max(final_gaps))
            mean_final_gap = float(np.mean(final_gaps))

            eval_log.append(
                {
                    "epoch": epoch,
                    "K": K,
                    "evaluations": evaluations,
                    "mean_final_gap": mean_final_gap,
                    "max_final_gap": max_final_gap,
                }
            )

            print(
                f"Epoch {epoch:4d} | "
                f"K={K:2d} | "
                f"Worst Eval Gap ({EVAL_STEPS} steps): {max_final_gap:.4e} | "
                f"Mean Eval Gap: {mean_final_gap:.4e}"
            )

    print("\n✅ 本组训练完成。开始最终对比评估...\n")

    training_point_rows = np.asarray(training_point_rows, dtype=np.float64)
    result_point_rows = np.asarray(result_point_rows, dtype=np.float64)
    micro_step_rows = np.asarray(micro_step_rows, dtype=np.float64)

    detailed_log_path = output_dir / "detailed_training_logs.npz"
    np.savez_compressed(
        detailed_log_path,
        training_points=training_point_rows,
        result_points=result_point_rows,
        micro_steps=micro_step_rows,
        y0=y0.detach().cpu().numpy(),
        y_star=y_star.detach().cpu().numpy(),
        newton_solution=newton_solution.detach().cpu().numpy(),
    )

    print(f"📁 压缩详细日志已保存至: {detailed_log_path}")

    # --------------------------------------------------------
    # 8.5 最终测试：统一从参考初值 y0 出发，MLP vs Newton
    # --------------------------------------------------------

    final_cases = []

    reference_initial_y = y0
    comparison = compare_mlp_and_newton(
        mlp=mlp,
        initial_y=reference_initial_y,
        history=history,
        params=params,
        p_n=p_n,
        v_n=v_n,
        m=m,
        g=g,
        dt=dt,
        E_star=E_star,
        max_steps=FINAL_TEST_STEPS,
    )

    final_cases.append(
        {
            "initial_index": 0,
            "initial_y": reference_initial_y.tolist(),
            "comparison": comparison,
        }
    )

    print_final_comparison(
        case_index=0,
        comparison=comparison,
        E_star=E_star,
        max_rows=5,
    )

    # --------------------------------------------------------
    # 8.6 保存 JSON 与网络参数
    # --------------------------------------------------------

    final_reference_gap = float(
        comparison["mlp"]["iterations"][-1]["loss"] - E_star
    )

    report = {
        "config": {
            "experiment_name": experiment_name,
            "experiment_description": experiment_description,
            "dataset_mode": dataset_mode,
            "output_directory": str(output_dir),
            "epochs": epochs,
            "initial_K": INITIAL_K,
            "K_increase_interval": K_INCREASE_INTERVAL,
            "max_K": MAX_K,
            "eval_interval": EVAL_INTERVAL,
            "eval_steps": EVAL_STEPS,
            "final_test_steps": FINAL_TEST_STEPS,
            "color_bucket_size": COLOR_BUCKET_SIZE,
            "plot_relative_coordinates": PLOT_RELATIVE_COORDINATES,
            "use_normalization": USE_NORMALIZATION,
            "use_dt_scaling": USE_DT_SCALING,
            "line_num_points": LINE_NUM_POINTS,
            "local_num_points": LOCAL_NUM_POINTS,
            "local_std_dt_units": LOCAL_STD_DT_UNITS,
            "local_random_seed": LOCAL_RANDOM_SEED,
            "model_random_seed": MODEL_RANDOM_SEED,
            "y0": y0.tolist(),
            "num_training_initial_points": len(training_initial_points),
            "training_initial_points": [
                point.tolist()
                for point in training_initial_points
            ],
            "p_n": p_n.tolist(),
            "v_n": v_n.tolist(),
            "m": m,
            "g": g,
            "dt": dt,
            "E_star": E_star,
            "y_star": y_star.tolist(),
            "newton_solution": newton_solution.tolist(),
        },
        "training_log": train_log,
        "periodic_evaluation": eval_log,
        "detailed_log_path": str(detailed_log_path),
        "final_comparison": {
            "cases": final_cases,
        },
        "summary": {
            "final_reference_gap_after_fixed_steps": final_reference_gap,
            "final_worst_periodic_eval_gap": eval_log[-1]["max_final_gap"],
            "final_mean_periodic_eval_gap": eval_log[-1]["mean_final_gap"],
        },
    }

    report_path = output_dir / "optimization_report.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    model_path = output_dir / "mlp_optimizer_state_dict.pt"
    torch.save(mlp.state_dict(), model_path)

    print(f"📁 数值结果已保存至: {report_path}")
    print(f"📁 网络参数已保存至: {model_path}")

    # --------------------------------------------------------
    # 8.7 四宫格统计图
    # --------------------------------------------------------

    plot_summary_report(
        train_log=train_log,
        eval_log=eval_log,
        reference_comparison=final_cases[0]["comparison"],
        E_star=E_star,
        save_path=output_dir / "optimization_report.png",
    )

    # --------------------------------------------------------
    # 8.8 参考初值 y0 对应的最终测试轨迹图
    # --------------------------------------------------------

    for case in final_cases:
        initial_index = case["initial_index"]
        initial_y = case["initial_y"]
        comparison = case["comparison"]

        plot_final_test_distribution(
            mlp_hist=comparison["mlp"],
            newton_solution=newton_solution.tolist(),
            initial_y=initial_y,
            reference_y0=y0.tolist(),
            save_path=(
                output_dir
                / f"final_test_distribution_initial_{initial_index}.png"
            ),
            relative=PLOT_RELATIVE_COORDINATES,
        )

    # --------------------------------------------------------
    # 8.9 整个训练过程中的训练点与结果点图
    # --------------------------------------------------------

    plot_training_points_and_results(
        visualization_log=visualization_log,
        newton_solution=newton_solution.tolist(),
        training_initial_points=[
            point.tolist()
            for point in training_initial_points
        ],
        reference_y0=y0.tolist(),
        epochs=epochs,
        bucket_size=COLOR_BUCKET_SIZE,
        save_path=(
            output_dir
            / "training_points_and_results_distribution.png"
        ),
        relative=PLOT_RELATIVE_COORDINATES,
    )

    print("✅ 本组结果已经生成完成。")
    print(f"📁 请查看本组输出目录: {output_dir}")

    return {
        "experiment_name": experiment_name,
        "experiment_description": experiment_description,
        "dataset_mode": dataset_mode,
        "output_directory": str(output_dir),
        "num_training_initial_points": len(training_initial_points),
        "final_reference_gap_after_fixed_steps": final_reference_gap,
        "final_worst_periodic_eval_gap": float(eval_log[-1]["max_final_gap"]),
        "final_mean_periodic_eval_gap": float(eval_log[-1]["mean_final_gap"]),
    }


# ============================================================
# 9. 主程序：依次运行三组数据集消融
# ============================================================

def main():
    base_output_dir = create_output_directory()
    print(f"📁 数据集消融总输出目录: {base_output_dir}")

    # 固定物理问题。三组实验只改变训练初值数据集。
    m = 1.0
    g = 9.8
    dt = 0.01
    p_n = torch.tensor([3.0, 4.0, 5.0])
    v_n = torch.tensor([0.5, -0.5, 0.0])

    experiment_summaries = []

    for experiment in ABLATION_EXPERIMENTS:
        summary = run_experiment(
            experiment=experiment,
            base_output_dir=base_output_dir,
            p_n=p_n,
            v_n=v_n,
            m=m,
            g=g,
            dt=dt,
        )
        experiment_summaries.append(summary)

    ablation_summary = {
        "base_output_directory": str(base_output_dir),
        "comparison_principle": (
            "All groups share the same physical problem, network architecture, "
            "model initialization seed, optimizer, and training schedule. "
            "Only the training initial-state dataset changes."
        ),
        "experiments": experiment_summaries,
    }

    summary_path = base_output_dir / "dataset_ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(ablation_summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print("✅ 三组数据集消融实验全部完成。")
    print(f"📁 总输出目录: {base_output_dir}")
    print(f"📄 汇总文件: {summary_path}")
    for item in experiment_summaries:
        print(
            f"- {item['experiment_name']}: "
            f"{item['num_training_initial_points']} 个训练初值，"
            f"最终参考 gap={item['final_reference_gap_after_fixed_steps']:.4e}"
        )


if __name__ == "__main__":
    main()
