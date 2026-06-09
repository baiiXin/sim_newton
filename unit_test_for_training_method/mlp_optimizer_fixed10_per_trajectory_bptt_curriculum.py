import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 适配无显示器 Linux 环境

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D


# 本脚本显式使用 float32，并比较两个候选优化器在固定扰动尺度下的性能。
TORCH_DTYPE = torch.float32
torch.set_default_dtype(TORCH_DTYPE)

# float32 下使用适中的对数坐标绘图下限，避免把数值底噪误读为有效精度。
PLOT_FLOOR = 1e-12


# ============================================================
# 0. 可调参数
# ============================================================

# True：绘制相对于原始初值 y0 的位移，更容易看清聚集在一起的点
# False：绘制绝对空间坐标
PLOT_RELATIVE_COORDINATES = False

# 每多少个 epoch 使用一种颜色
COLOR_BUCKET_SIZE = 200

# float32 固定扰动训练实验：比较两个候选优化器在同一个训练集上的表现。
#
# 固定条件：
# 1. 输入使用当前训练集统计量标准化；
# 2. 网络输出乘以 dt；
# 3. 训练集固定为“y0 附近 10 个随机扰动点”，不包含精确初值 y0；
# 4. 评估集只保留未参与训练的精确初值 y0；
# 5. 所有张量、模型参数和优化器状态均使用 torch.float32；
# 6. SGD 使用不带 momentum 的标准形式；
# 7. SGD 学习率固定为 1e-2；Adam 学习率固定为 1e-4。
#
# 本实验只使用一个扰动尺度：
#     sigma = 1e-2
#
# 训练策略：
# 1. 每个 epoch 对 10 个训练初值分别展开 K 步 MLP 迭代；
# 2. 不在轨迹内部的迭代步之间 detach，因此梯度能够穿过完整轨迹；
# 3. 对每条轨迹，累加该轨迹上 K 个步骤的 loss；
# 4. 每条轨迹分别执行一次 backward 和一次 optimizer.step；
# 5. 因此，每个 epoch 共执行 10 次参数更新；
# 6. K 从 5 开始，每 200 个 epoch 增加 5，1000 个 epoch 内依次为
#    5、10、15、20、25。
FIXED_NUM_PERTURBATION_POINTS = 10
LOCAL_RANDOM_SEED = 123
MODEL_RANDOM_SEED = 42

USE_NORMALIZATION = True
USE_DT_SCALING = True

PERTURBATION_STD_VALUES = [
    1e-2,
]

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

ABLATION_EXPERIMENTS = []
experiment_index = 1
for optimizer_config in OPTIMIZER_CONFIGS:
    optimizer_name = optimizer_config["optimizer_name"]
    learning_rate = optimizer_config["learning_rate"]

    for perturbation_std in PERTURBATION_STD_VALUES:
        ABLATION_EXPERIMENTS.append(
            {
                "name": (
                    f"{experiment_index:02d}_float32_{optimizer_name}_"
                    f"lr_{learning_rate:.0e}_perturbation_std_"
                    f"{perturbation_std:.0e}"
                ),
                "description": (
                    "Float32 training with full normalization, 10 initial-point "
                    f"perturbations, {optimizer_name.upper()} optimizer, "
                    f"lr={learning_rate:.0e}, absolute coordinate "
                    f"perturbation standard deviation sigma={perturbation_std:.0e}."
                ),
                "optimizer_name": optimizer_name,
                "learning_rate": learning_rate,
                "perturbation_std": perturbation_std,
                "num_local_points": FIXED_NUM_PERTURBATION_POINTS,
            }
        )
        experiment_index += 1

# 2D 能量等高线轨迹图参数。
# 默认展示 x-z 平面切片：未展示的 y 坐标固定在理论最优解 y_star 的 y 分量。
CONTOUR_PROJECTION_AXES = (0, 2)
CONTOUR_GRID_SIZE = 240
CONTOUR_LEVEL_COUNT = 28
CONTOUR_MARGIN_RATIO = 0.20
CONTOUR_MIN_AXIS_SPAN = 2e-4


# 训练与评估参数
EPOCHS = 1000
INITIAL_K = 5
K_INCREASE_INTERVAL = 200
K_INCREASE_AMOUNT = 5
MAX_K = 25
EVAL_INTERVAL = 100
EVAL_STEPS = 10
FINAL_TEST_STEPS = 50  # 最终冻结评估连续迭代 50 次


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


def stationarity_residual(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    计算隐式欧拉变分问题的一阶驻点方程残差：

        r(y) = grad E(y)
             = (m / dt^2) * (y - p_n - dt * v_n)
               + [0, 0, m * g]^T

    理论最优解 y_star 满足 r(y_star) = 0。
    因此 ||r(y)||_2 可以用于衡量迭代器是否真正收敛到方程解。
    """

    residual = (m / dt**2) * (y - p_n - dt * v_n)
    residual = residual.clone()
    residual[2] += m * g
    return residual


def stationarity_residual_norm(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """返回驻点方程残差的二范数。"""

    return torch.norm(
        stationarity_residual(y, p_n, v_n, m, g, dt)
    )


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


def make_training_states_near_initial(
    y0,
    perturbation_std,
    num_local_points,
    seed=LOCAL_RANDOM_SEED,
):
    """
    构造扰动范围消融实验使用的训练集。

    训练集包含：
        num_local_points 个 y0 附近的随机扰动点。

    注意：
        精确的原始初值 y0 不参与训练，但会保留在评估集中。

    每个扰动点满足：
        y = y0 + sigma * epsilon,
        epsilon ~ N(0, I),
        sigma = perturbation_std.

    perturbation_std 是绝对坐标标准差，不再额外乘以 dt。
    固定随机种子用于生成可复现的 10 个训练扰动点。
    """

    train_states = []

    gen = torch.Generator(device=y0.device)
    gen.manual_seed(seed)

    for _ in range(num_local_points):
        noise = torch.randn(3, generator=gen, device=y0.device)
        train_states.append(y0 + perturbation_std * noise)

    return train_states

def compute_input_normalizer(train_states, history, params):
    x = torch.stack(
        [torch.cat([y, history, params], dim=-1) for y in train_states],
        dim=0,
    )
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


def create_optimizer(model, optimizer_name, learning_rate):
    """根据当前消融组配置创建 PyTorch 优化器。"""

    normalized_name = optimizer_name.lower()

    if normalized_name == "sgd":
        # 使用不带 momentum 的标准 SGD，避免额外超参数干扰消融结果。
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
        )

    if normalized_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name!r}. "
        "Expected one of: 'sgd', 'adam'."
    )


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

    每个 epoch 都从当前消融组中的全部扰动训练初值分别出发，
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

        residual_norm = stationarity_residual_norm(
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
                "residual_norm": residual_norm,
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

    initial_residual_norm = stationarity_residual_norm(
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
                "residual_norm": initial_residual_norm,
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
                "residual_norm": initial_residual_norm,
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

        residual_norm_mlp = stationarity_residual_norm(
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
                "residual_norm": residual_norm_mlp,
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

        residual_norm_newton = stationarity_residual_norm(
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
                "residual_norm": residual_norm_newton,
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
        f"{'MLP Residual':<14} | "
        f"{'Newton Residual':<14} | "
        f"{'MLP y'}"
    )

    print("-" * 136)

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
            f"{mlp_item['residual_norm']:<14.4e} | "
            f"{newton_item['residual_norm']:<14.4e} | "
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

    图 1 对所有训练初值取最差 gap；图 2 对评估初值取最差 gap；
    图 3 和图 4 使用参考初值 y0 对比 MLP 与 Newton 法。
    """

    mlp_hist = reference_comparison["mlp"]
    newton_hist = reference_comparison["newton"]

    gap_mlp = [
        max(item["loss"] - E_star, PLOT_FLOOR)
        for item in mlp_hist["iterations"]
    ]

    gap_newton = [
        max(item["loss"] - E_star, PLOT_FLOOR)
        for item in newton_hist["iterations"]
    ]

    train_gap = [
        max(item["max_final_loss"] - E_star, PLOT_FLOOR)
        for item in train_log
    ]

    eval_gap = [
        max(item["max_final_gap"], PLOT_FLOOR)
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


def plot_final_residual_comparison(
    comparison,
    save_path,
    initial_index,
    is_reference_y0=False,
):
    """
    绘制一个评估初值对应的驻点方程 residual 下降曲线。

    每张图统一对比：
        1. 冻结训练后的 MLP 迭代器；
        2. 当前严格凸二次问题中的 Newton 法。

    当前评估集只保留未参与训练的原始初值 y0，因此 initial_index=0。
    """

    mlp_hist = comparison["mlp"]
    newton_hist = comparison["newton"]

    mlp_residual_norms = [
        max(item["residual_norm"], PLOT_FLOOR)
        for item in mlp_hist["iterations"]
    ]
    newton_residual_norms = [
        max(item["residual_norm"], PLOT_FLOOR)
        for item in newton_hist["iterations"]
    ]
    test_steps = np.arange(len(mlp_residual_norms))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        test_steps,
        mlp_residual_norms,
        label="MLP Optimizer",
        marker="o",
    )
    ax.plot(
        test_steps,
        newton_residual_norms,
        label="Newton Method",
        marker="s",
        linestyle="--",
        color="crimson",
    )

    ax.set_yscale("log")
    if is_reference_y0:
        title_suffix = "Initial 0: Original y0 (Held Out from Training)"
    else:
        title_suffix = f"Initial {initial_index}: Perturbed Training Initial State"

    ax.set_title(f"Final Residual Comparison\n{title_suffix}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Stationarity residual $\|\nabla E(y)\|_2$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"🖼️ 最终 residual 下降图已保存至: {save_path}")



# ============================================================
# 8. 最终测试轨迹的二维投影能量等高线图
# ============================================================

def plot_final_test_energy_contour_2d(
    comparison,
    newton_solution,
    initial_y,
    reference_y0,
    p_n,
    v_n,
    m,
    g,
    dt,
    save_path,
    projection_axes=CONTOUR_PROJECTION_AXES,
    relative=True,
):
    """
    绘制最终测试轨迹在二维坐标平面上的投影，并叠加能量等高线。

    默认投影到 x-z 平面：
        横轴：x
        纵轴：z

    未展示的坐标分量固定在理论最优解 y_star（即 Newton 一步解）上。
    因此，背景不是完整三维能量的无损压缩，而是穿过 y_star 的二维能量切片。

    轨迹本身则是完整三维 MLP / Newton 迭代轨迹在该平面上的投影。
    这张图与 final_test_distribution_initial_0.png 配套使用：
        - 3D 图展示空间轨迹；
        - 2D 图展示轨迹如何沿能量地形靠近最优点。
    """

    axis_names = ["x", "y", "z"]
    first_axis, second_axis = projection_axes
    if first_axis == second_axis:
        raise ValueError("projection_axes must contain two different axes.")

    mlp_points = np.asarray(
        [item["y"] for item in comparison["mlp"]["iterations"]],
        dtype=float,
    ).reshape(-1, 3)
    newton_points = np.asarray(
        [item["y"] for item in comparison["newton"]["iterations"]],
        dtype=float,
    ).reshape(-1, 3)

    newton_solution_np = np.asarray(newton_solution, dtype=float).reshape(3)
    initial_y_np = np.asarray(initial_y, dtype=float).reshape(3)
    reference_y0_np = np.asarray(reference_y0, dtype=float).reshape(3)
    p_n_np = np.asarray(p_n, dtype=float).reshape(3)
    v_n_np = np.asarray(v_n, dtype=float).reshape(3)

    # 对当前严格凸二次问题，Newton 一步解与理论最优解一致。
    y_star_np = newton_solution_np.copy()

    projected_points = np.vstack(
        [
            mlp_points[:, [first_axis, second_axis]],
            newton_points[:, [first_axis, second_axis]],
            initial_y_np[[first_axis, second_axis]].reshape(1, 2),
            y_star_np[[first_axis, second_axis]].reshape(1, 2),
        ]
    )

    lower = projected_points.min(axis=0)
    upper = projected_points.max(axis=0)
    span = np.maximum(upper - lower, CONTOUR_MIN_AXIS_SPAN)
    lower = lower - CONTOUR_MARGIN_RATIO * span
    upper = upper + CONTOUR_MARGIN_RATIO * span

    first_values = np.linspace(lower[0], upper[0], CONTOUR_GRID_SIZE)
    second_values = np.linspace(lower[1], upper[1], CONTOUR_GRID_SIZE)
    first_grid, second_grid = np.meshgrid(first_values, second_values)

    # 构造穿过 y_star 的二维切片。未展示坐标固定为 y_star 对应分量。
    slice_points = np.broadcast_to(
        y_star_np.reshape(1, 1, 3),
        (CONTOUR_GRID_SIZE, CONTOUR_GRID_SIZE, 3),
    ).copy()
    slice_points[..., first_axis] = first_grid
    slice_points[..., second_axis] = second_grid

    inertial_residual = slice_points - p_n_np - dt * v_n_np
    energy_grid = (
        (m / (2.0 * dt**2))
        * np.sum(inertial_residual**2, axis=-1)
        + m * g * slice_points[..., 2]
    )

    y_star_residual = y_star_np - p_n_np - dt * v_n_np
    E_star = (
        (m / (2.0 * dt**2)) * np.sum(y_star_residual**2)
        + m * g * y_star_np[2]
    )
    energy_gap_grid = np.maximum(energy_grid - E_star, PLOT_FLOOR)

    max_gap = float(np.max(energy_gap_grid))
    positive_values = energy_gap_grid[energy_gap_grid > PLOT_FLOOR]
    if positive_values.size == 0 or max_gap <= PLOT_FLOOR:
        min_level = PLOT_FLOOR
        max_level = PLOT_FLOOR * 10.0
    else:
        # 避免最中心的极小值让等高线动态范围过宽，导致主要区域难以阅读。
        min_positive = float(np.min(positive_values))
        min_level = max(min_positive, max_gap * 1e-8, PLOT_FLOOR)
        max_level = max(max_gap, min_level * 10.0)

    levels = np.geomspace(min_level, max_level, CONTOUR_LEVEL_COUNT)

    fig, ax = plt.subplots(figsize=(9, 7))

    contour = ax.contourf(
        first_grid,
        second_grid,
        energy_gap_grid,
        levels=levels,
        norm=matplotlib.colors.LogNorm(vmin=min_level, vmax=max_level),
        cmap="viridis",
        alpha=0.82,
        extend="both",
    )
    ax.contour(
        first_grid,
        second_grid,
        energy_gap_grid,
        levels=levels,
        norm=matplotlib.colors.LogNorm(vmin=min_level, vmax=max_level),
        colors="black",
        linewidths=0.35,
        alpha=0.45,
    )

    if relative:
        mlp_plot = mlp_points[:, [first_axis, second_axis]] - reference_y0_np[
            [first_axis, second_axis]
        ]
        newton_plot = newton_points[:, [first_axis, second_axis]] - reference_y0_np[
            [first_axis, second_axis]
        ]
        initial_plot = initial_y_np[[first_axis, second_axis]] - reference_y0_np[
            [first_axis, second_axis]
        ]
        y_star_plot = y_star_np[[first_axis, second_axis]] - reference_y0_np[
            [first_axis, second_axis]
        ]

        # contourf 已经使用绝对坐标绘制；仅将刻度显示转换为相对坐标。
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda value, _: f"{value - reference_y0_np[first_axis]:.4g}"
            )
        )
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda value, _: f"{value - reference_y0_np[second_axis]:.4g}"
            )
        )
        first_label = rf"$\Delta {axis_names[first_axis]}$"
        second_label = rf"$\Delta {axis_names[second_axis]}$"
    else:
        mlp_plot = mlp_points[:, [first_axis, second_axis]]
        newton_plot = newton_points[:, [first_axis, second_axis]]
        initial_plot = initial_y_np[[first_axis, second_axis]]
        y_star_plot = y_star_np[[first_axis, second_axis]]
        first_label = rf"${axis_names[first_axis]}$"
        second_label = rf"${axis_names[second_axis]}$"

    # 若使用相对坐标，轨迹点仍需以绝对坐标叠加到 contourf 上。
    if relative:
        mlp_draw = mlp_plot + reference_y0_np[[first_axis, second_axis]]
        newton_draw = newton_plot + reference_y0_np[[first_axis, second_axis]]
        initial_draw = initial_plot + reference_y0_np[[first_axis, second_axis]]
        y_star_draw = y_star_plot + reference_y0_np[[first_axis, second_axis]]
    else:
        mlp_draw = mlp_plot
        newton_draw = newton_plot
        initial_draw = initial_plot
        y_star_draw = y_star_plot

    ax.plot(
        mlp_draw[:, 0],
        mlp_draw[:, 1],
        "-o",
        linewidth=1.8,
        markersize=4,
        label="MLP projected trajectory",
    )
    ax.plot(
        newton_draw[:, 0],
        newton_draw[:, 1],
        "--s",
        linewidth=1.5,
        markersize=4,
        color="crimson",
        label="Newton projected trajectory",
    )

    for step, point in enumerate(mlp_draw):
        ax.text(point[0], point[1], f"  {step}", fontsize=8)

    ax.scatter(
        initial_draw[0],
        initial_draw[1],
        marker="x",
        s=120,
        c="black",
        linewidths=2.0,
        label="Test initial point",
    )
    ax.scatter(
        y_star_draw[0],
        y_star_draw[1],
        marker="*",
        s=260,
        c="crimson",
        label="Newton converged solution",
    )

    ax.set_xlabel(first_label)
    ax.set_ylabel(second_label)
    fixed_axis = ({0, 1, 2} - {first_axis, second_axis}).pop()
    ax.set_title(
        "Projected Final-Test Trajectories on Energy-Gap Contours\n"
        f"{axis_names[first_axis]}-{axis_names[second_axis]} slice, "
        f"{axis_names[fixed_axis]} fixed at y*"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)

    colorbar = fig.colorbar(contour, ax=ax)
    colorbar.set_label(r"Energy gap $E(y) - E(y^*)$")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"🖼️ 二维投影能量等高线轨迹图已保存至: {save_path}")


# ============================================================
# 9. 单组实验
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
    运行一组 float32 固定扰动训练实验，并将结果写入独立子目录。

    两类优化器分别固定采用 SGD(lr=1e-2) 与 Adam(lr=1e-4)。
    两组实验共享物理问题、10 个固定扰动训练点、网络结构、初始化种子、
    完整归一化方案、float32 精度和完整轨迹反向传播策略。

    数据划分：
        - 训练集：y0 附近的 10 个随机扰动点，不包含 y0；
        - 评估集：只保留未参与训练的精确初值 y0。

    训练策略：
        - K 从 5 开始，每 200 个 epoch 增加 5；
        - 每条轨迹内部完整展开，不在迭代步之间 detach；
        - 对每条轨迹累加 K 个步骤的 loss；
        - 每条轨迹分别执行一次 backward 和一次 optimizer.step；
        - 每个 epoch 共执行 10 次参数更新。
    """

    experiment_name = experiment["name"]
    experiment_description = experiment["description"]
    perturbation_std = experiment["perturbation_std"]
    num_local_points = experiment["num_local_points"]
    optimizer_name = experiment["optimizer_name"]
    learning_rate = experiment["learning_rate"]
    use_normalization = USE_NORMALIZATION
    use_dt_scaling = USE_DT_SCALING

    output_dir = base_output_dir / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"🧪 float32 优化器与扰动范围实验组: {experiment_name}")
    print(f"说明: {experiment_description}")
    print(f"📁 本组输出目录: {output_dir}")
    print("=" * 72)

    # 每组实验都重置随机种子，确保网络初始化一致。
    torch.manual_seed(MODEL_RANDOM_SEED)

    # --------------------------------------------------------
    # 9.1 构造物理问题与当前扰动尺度训练数据集
    # --------------------------------------------------------

    y0 = p_n.clone()

    y_star = (
        p_n
        + dt * v_n
        - dt**2 * torch.tensor([0.0, 0.0, g])
    )

    history = torch.cat([p_n, v_n])
    params = torch.tensor([m, g, dt])

    training_initial_points = make_training_states_near_initial(
        y0=y0,
        perturbation_std=perturbation_std,
        num_local_points=num_local_points,
        seed=LOCAL_RANDOM_SEED,
    )

    # 训练集不包含 y0。评估集只保留未参与训练的精确初值 y0。
    evaluation_initial_points = [
        y0.clone(),
    ]

    if use_normalization:
        input_mean, input_std = compute_input_normalizer(
            training_initial_points,
            history,
            params,
        )
    else:
        input_mean = torch.zeros(12)
        input_std = torch.ones(12)

    # --------------------------------------------------------
    # 9.2 创建模型
    # --------------------------------------------------------

    mlp = MLPOptimizer(
        use_normalization=use_normalization,
        use_dt_scaling=use_dt_scaling,
        input_mean=input_mean,
        input_std=input_std,
    )
    opt = create_optimizer(
        model=mlp,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
    )

    epochs = EPOCHS
    K = INITIAL_K

    train_log = []
    eval_log = []
    visualization_log = []
    training_point_rows = []
    result_point_rows = []
    micro_step_rows = []

    # --------------------------------------------------------
    # 9.3 理论最优解与 Newton 收敛解
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
    print("dataset_mode=fixed_10_initial_perturbations_only")
    print("evaluation_mode=held_out_y0_only")
    print(f"训练初值总数（不含 y0）: {len(training_initial_points)}")
    print(f"评估初值总数（仅 y0）: {len(evaluation_initial_points)}")
    print(f"perturbation_std={perturbation_std:.1e}")
    print(f"num_perturbation_points={num_local_points}")
    print(f"optimizer_name={optimizer_name}")
    print(f"learning_rate={learning_rate:.1e}")
    print(f"torch_dtype={TORCH_DTYPE}")
    print(f"use_normalization={use_normalization}")
    print(f"use_dt_scaling={use_dt_scaling}")
    for initial_index, (initial_y, initial_delta) in enumerate(
        zip(training_initial_points[:5], initial_deltas[:5])
    ):
        print(f"训练初值 {initial_index}: {initial_y.tolist()}")
        print(f"初始化网络在初值 {initial_index} 上的输出: {initial_delta.tolist()}")
    if len(training_initial_points) > 5:
        print(f"其余训练初值已省略显示: {len(training_initial_points) - 5} 个")
    print(
        f"策略: K 从 {INITIAL_K} 开始，每 {K_INCREASE_INTERVAL} 个 epoch "
        f"增加 {K_INCREASE_AMOUNT}；每条轨迹内部完整展开，不做步间 detach；"
        "每条轨迹汇总自身全部迭代步的 loss 后，"
        "分别执行一次 backward 和一次 optimizer.step"
    )
    print(f"可视化: 每 {COLOR_BUCKET_SIZE} 个 epoch 使用一种颜色\n")

    # --------------------------------------------------------
    # 9.4 训练
    # --------------------------------------------------------

    for epoch in range(epochs):
        if (
            epoch > 0
            and epoch % K_INCREASE_INTERVAL == 0
            and K < MAX_K
        ):
            K = min(K + K_INCREASE_AMOUNT, MAX_K)

        epoch_trajectories = []
        epoch_final_losses = []
        epoch_summed_trajectory_losses = []

        # 每条轨迹独立进行一次参数更新。轨迹内部不执行 detach，
        # 因此该轨迹所有步骤的 loss 都能够沿着完整展开的状态依赖关系
        # 反向传播。完成一条轨迹后，再处理下一条轨迹。
        for initial_index, initial_y in enumerate(training_initial_points):
            opt.zero_grad(set_to_none=True)

            y = initial_y.clone()
            trajectory_train_points = []
            trajectory_losses = []
            trajectory_total_loss = None

            for iteration_index in range(K):
                # detach 仅用于复制日志，不改变真正参与计算的 y。
                y_before_for_log = y.detach().clone()
                trajectory_train_points.append(y_before_for_log)
                training_point_rows.append([
                    epoch,
                    epoch // COLOR_BUCKET_SIZE,
                    initial_index,
                    iteration_index,
                    y_before_for_log[0].item(),
                    y_before_for_log[1].item(),
                    y_before_for_log[2].item(),
                ])

                delta = mlp(y, history, params)
                y = y + delta
                loss = variational_energy(y, p_n, v_n, m, g, dt)
                trajectory_losses.append(loss)
                trajectory_total_loss = (
                    loss
                    if trajectory_total_loss is None
                    else trajectory_total_loss + loss
                )

                objective_gap = max(float(loss.item() - E_star), 0.0)
                micro_step_rows.append([
                    epoch,
                    epoch // COLOR_BUCKET_SIZE,
                    initial_index,
                    iteration_index + 1,
                    float(loss.item()),
                    objective_gap,
                ])

            if trajectory_total_loss is None:
                raise RuntimeError("A training trajectory contains no steps.")

            # 当前轨迹的 K 个步骤共同形成一个目标函数。
            # 只在轨迹结束后执行一次反向传播和一次参数更新。
            trajectory_total_loss.backward()
            opt.step()

            epoch_result_point = y.detach().clone()
            final_loss = trajectory_losses[-1]
            summed_trajectory_loss = float(trajectory_total_loss.item())
            result_point_rows.append([
                epoch,
                epoch // COLOR_BUCKET_SIZE,
                initial_index,
                epoch_result_point[0].item(),
                epoch_result_point[1].item(),
                epoch_result_point[2].item(),
            ])
            epoch_final_losses.append(float(final_loss.item()))
            epoch_summed_trajectory_losses.append(summed_trajectory_loss)

            epoch_trajectories.append(
                {
                    "initial_index": initial_index,
                    "initial_y": initial_y.tolist(),
                    "train_points": [
                        point.tolist()
                        for point in trajectory_train_points
                    ],
                    "result_point": epoch_result_point.tolist(),
                    "final_loss": float(final_loss.item()),
                    "summed_trajectory_loss": summed_trajectory_loss,
                }
            )

        if not epoch_summed_trajectory_losses:
            raise RuntimeError("The fixed perturbation training set is empty.")

        summed_epoch_loss = float(np.sum(epoch_summed_trajectory_losses))
        mean_step_loss = summed_epoch_loss / (
            len(training_initial_points) * K
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
                "summed_epoch_loss": summed_epoch_loss,
                "mean_step_loss": mean_step_loss,
                "final_losses": epoch_final_losses,
                "mean_final_loss": mean_final_loss,
                "max_final_loss": max_final_loss,
            }
        )

        # ----------------------------------------------------
        # 每隔固定 epoch，从未参与训练的原始初值 y0 出发做固定步数评估。
        # ----------------------------------------------------

        if epoch % EVAL_INTERVAL == 0 or epoch == epochs - 1:
            evaluations = []
            final_gaps = []

            for initial_index, initial_y in enumerate(evaluation_initial_points):
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

    training_point_rows = np.asarray(training_point_rows, dtype=np.float32)
    result_point_rows = np.asarray(result_point_rows, dtype=np.float32)
    micro_step_rows = np.asarray(micro_step_rows, dtype=np.float32)

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
    # 9.5 最终测试：只从未参与训练的原始初值 y0 出发，对比 MLP 与 Newton
    # --------------------------------------------------------

    final_cases = []

    for initial_index, initial_y in enumerate(evaluation_initial_points):
        case_comparison = compare_mlp_and_newton(
            mlp=mlp,
            initial_y=initial_y,
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
                "initial_index": initial_index,
                "is_reference_y0": initial_index == 0,
                "initial_y": initial_y.tolist(),
                "comparison": case_comparison,
            }
        )

        print_final_comparison(
            case_index=initial_index,
            comparison=case_comparison,
            E_star=E_star,
            max_rows=5,
        )

    # final_cases[0] 始终对应未参与训练的原始初值 y0。
    reference_comparison = final_cases[0]["comparison"]

    # --------------------------------------------------------
    # 9.6 保存 JSON 与网络参数
    # --------------------------------------------------------

    final_reference_gap = float(
        reference_comparison["mlp"]["iterations"][-1]["loss"] - E_star
    )
    final_reference_residual_norm = float(
        reference_comparison["mlp"]["iterations"][-1]["residual_norm"]
    )

    report = {
        "config": {
            "experiment_name": experiment_name,
            "experiment_description": experiment_description,
            "dataset_mode": "fixed_10_initial_perturbations_only",
            "evaluation_mode": "held_out_y0_only",
            "training_strategy": (
                "Full unrolled trajectory backpropagation with one optimizer "
                "update per trajectory. For each fixed perturbed training "
                "trajectory, sum the energy losses from its K steps, then "
                "call backward once and optimizer.step once. Do not detach "
                "between steps inside a trajectory. Each epoch processes 10 "
                "trajectories sequentially and performs 10 parameter updates."
            ),
            "output_directory": str(output_dir),
            "epochs": epochs,
            "initial_K": INITIAL_K,
            "K_increase_interval": K_INCREASE_INTERVAL,
            "K_increase_amount": K_INCREASE_AMOUNT,
            "max_K": MAX_K,
            "eval_interval": EVAL_INTERVAL,
            "eval_steps": EVAL_STEPS,
            "final_test_steps": FINAL_TEST_STEPS,
            "color_bucket_size": COLOR_BUCKET_SIZE,
            "plot_relative_coordinates": PLOT_RELATIVE_COORDINATES,
            "contour_projection_axes": list(CONTOUR_PROJECTION_AXES),
            "contour_projection_axis_names": [
                ["x", "y", "z"][axis]
                for axis in CONTOUR_PROJECTION_AXES
            ],
            "contour_slice_definition": (
                "The 2D energy contour background is a slice through y_star; "
                "unshown coordinate components are fixed at y_star. The MLP "
                "and Newton trajectories are full 3D trajectories projected "
                "onto the selected coordinate plane."
            ),
            "torch_dtype": str(TORCH_DTYPE),
            "torch_default_dtype": str(torch.get_default_dtype()),
            "use_normalization": use_normalization,
            "use_dt_scaling": use_dt_scaling,
            "optimizer_name": optimizer_name,
            "learning_rate": learning_rate,
            "sgd_momentum": 0.0 if optimizer_name == "sgd" else None,
            "num_perturbation_points": num_local_points,
            "perturbation_std": perturbation_std,
            "perturbation_std_definition": (
                "Absolute coordinate standard deviation sigma in "
                "y = y0 + sigma * epsilon, epsilon ~ N(0, I)."
            ),
            "local_random_seed": LOCAL_RANDOM_SEED,
            "model_random_seed": MODEL_RANDOM_SEED,
            "y0": y0.tolist(),
            "num_training_initial_points": len(training_initial_points),
            "training_initial_points": [
                point.tolist()
                for point in training_initial_points
            ],
            "num_evaluation_initial_points": len(evaluation_initial_points),
            "evaluation_initial_points": [
                point.tolist()
                for point in evaluation_initial_points
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
            "final_reference_residual_norm_after_fixed_steps": (
                final_reference_residual_norm
            ),
            "newton_reference_residual_norm_after_fixed_steps": (
                reference_comparison["newton"]["iterations"][-1]["residual_norm"]
            ),
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
    # 9.7 四宫格统计图
    # --------------------------------------------------------

    plot_summary_report(
        train_log=train_log,
        eval_log=eval_log,
        reference_comparison=final_cases[0]["comparison"],
        E_star=E_star,
        save_path=output_dir / "optimization_report.png",
    )

    # --------------------------------------------------------
    # 9.8 为未参与训练的原始初值 y0 绘制 residual 与优化轨迹图
    # --------------------------------------------------------

    for case in final_cases:
        initial_index = case["initial_index"]
        initial_y = case["initial_y"]
        comparison = case["comparison"]
        is_reference_y0 = case["is_reference_y0"]

        plot_final_residual_comparison(
            comparison=comparison,
            save_path=(
                output_dir
                / f"final_residual_comparison_initial_{initial_index}.png"
            ),
            initial_index=initial_index,
            is_reference_y0=is_reference_y0,
        )

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


        # 评估集只包含 y0，因此只生成一张二维能量等高线背景图。
        if is_reference_y0:
            plot_final_test_energy_contour_2d(
                comparison=comparison,
                newton_solution=newton_solution.tolist(),
                initial_y=initial_y,
                reference_y0=y0.tolist(),
                p_n=p_n.tolist(),
                v_n=v_n.tolist(),
                m=m,
                g=g,
                dt=dt,
                save_path=(
                    output_dir
                    / f"final_test_energy_contour_2d_initial_{initial_index}.png"
                ),
                projection_axes=CONTOUR_PROJECTION_AXES,
                relative=PLOT_RELATIVE_COORDINATES,
            )

    # --------------------------------------------------------
    # 9.9 整个训练过程中的训练点与结果点图
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
        "dataset_mode": "fixed_10_initial_perturbations_only",
        "evaluation_mode": "held_out_y0_only",
        "training_strategy": "full_unrolled_sum_loss_one_backward_per_trajectory",
        "use_normalization": use_normalization,
        "use_dt_scaling": use_dt_scaling,
        "optimizer_name": optimizer_name,
        "learning_rate": learning_rate,
        "torch_dtype": str(TORCH_DTYPE),
        "sgd_momentum": 0.0 if optimizer_name == "sgd" else None,
        "perturbation_std": perturbation_std,
        "num_perturbation_points": num_local_points,
        "output_directory": str(output_dir),
        "num_training_initial_points": len(training_initial_points),
        "num_evaluation_initial_points": len(evaluation_initial_points),
        "final_reference_gap_after_fixed_steps": final_reference_gap,
        "final_reference_residual_norm_after_fixed_steps": (
            final_reference_residual_norm
        ),
        "newton_reference_residual_norm_after_fixed_steps": float(
            reference_comparison["newton"]["iterations"][-1]["residual_norm"]
        ),
        "final_worst_periodic_eval_gap": float(eval_log[-1]["max_final_gap"]),
        "final_mean_periodic_eval_gap": float(eval_log[-1]["mean_final_gap"]),
    }


# ============================================================
# 10. float32 固定扰动逐轨迹完整反向传播跨组汇总图
# ============================================================

def plot_float32_optimizer_perturbation_range_summary(
    experiment_summaries,
    save_path,
):
    """绘制两种优化器在固定扰动尺度下的最终指标。"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for optimizer_config in OPTIMIZER_CONFIGS:
        optimizer_name = optimizer_config["optimizer_name"]
        learning_rate = optimizer_config["learning_rate"]
        selected = [
            item
            for item in experiment_summaries
            if item["optimizer_name"] == optimizer_name
        ]
        ordered = sorted(
            selected,
            key=lambda item: item["perturbation_std"],
            reverse=True,
        )

        perturbation_stds = [item["perturbation_std"] for item in ordered]
        final_reference_gaps = [
            max(item["final_reference_gap_after_fixed_steps"], PLOT_FLOOR)
            for item in ordered
        ]
        final_worst_eval_gaps = [
            max(item["final_worst_periodic_eval_gap"], PLOT_FLOOR)
            for item in ordered
        ]
        final_reference_residuals = [
            max(
                item["final_reference_residual_norm_after_fixed_steps"],
                PLOT_FLOOR,
            )
            for item in ordered
        ]

        label = f"{optimizer_name.upper()} lr={learning_rate:.0e}"

        axes[0].plot(
            perturbation_stds,
            final_reference_gaps,
            marker="o",
            label=label,
        )
        axes[1].plot(
            perturbation_stds,
            final_worst_eval_gaps,
            marker="s",
            label=label,
        )
        axes[2].plot(
            perturbation_stds,
            final_reference_residuals,
            marker="^",
            label=label,
        )

    axes[0].set_title("Reference Objective Gap vs. Perturbation Range")
    axes[0].set_ylabel("Gap")

    axes[1].set_title("Worst Periodic Eval Gap vs. Perturbation Range")
    axes[1].set_ylabel("Gap")

    axes[2].set_title("Reference Residual vs. Perturbation Range")
    axes[2].set_ylabel(r"Stationarity residual $\|\nabla E(y)\|_2$")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"Perturbation standard deviation $\sigma$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"🖼️ float32 固定扰动完整轨迹反向传播汇总图已保存至: {save_path}")


# ============================================================
# 11. 主程序：依次运行两组 float32 固定扰动完整轨迹反向传播实验
# ============================================================

def main():
    base_output_dir = create_output_directory()
    print(f"📁 float32 固定扰动逐轨迹完整反向传播实验总输出目录: {base_output_dir}")
    print(f"torch default dtype: {torch.get_default_dtype()}")

    # 固定物理问题和扰动尺度。两组实验只改变训练优化器类型。
    m = 1.0
    g = 9.8
    dt = 0.01
    p_n = torch.tensor([3.0, 4.0, 5.0], dtype=TORCH_DTYPE)
    v_n = torch.tensor([0.5, -0.5, 0.0], dtype=TORCH_DTYPE)

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
        "experiment_type": "float32_fixed_perturbation_per_trajectory_full_unrolled_training",
        "torch_dtype": str(TORCH_DTYPE),
        "comparison_principle": (
            "Both groups use torch.float32 for physical tensors, network "
            "parameters, gradients, and optimizer states. They share the same "
            "physical problem, the same 10 fixed perturbed training states "
            "with sigma=1e-2, random seeds, network architecture, input "
            "normalization, output dt scaling, and curriculum schedule. The "
            "training set excludes the exact y0, and evaluation uses only the "
            "held-out exact y0. For each training trajectory, all K steps "
            "are fully unrolled without detach; that trajectory's step losses "
            "are summed before one backward call and one optimizer step. Each "
            "epoch therefore performs 10 sequential parameter updates. K "
            "starts at 5 and increases by 5 every 200 epochs. SGD uses lr=1e-2 "
            "without momentum; Adam uses lr=1e-4."
        ),
        "fixed_num_perturbation_points": FIXED_NUM_PERTURBATION_POINTS,
        "training_dataset_mode": "fixed_10_initial_perturbations_only",
        "evaluation_mode": "held_out_y0_only",
        "training_strategy": "full_unrolled_sum_loss_one_backward_per_trajectory",
        "initial_K": INITIAL_K,
        "K_increase_interval": K_INCREASE_INTERVAL,
        "K_increase_amount": K_INCREASE_AMOUNT,
        "max_K": MAX_K,
        "perturbation_std_values": PERTURBATION_STD_VALUES,
        "optimizer_configs": OPTIMIZER_CONFIGS,
        "contour_projection_axes": list(CONTOUR_PROJECTION_AXES),
        "experiments": experiment_summaries,
    }

    summary_path = (
        base_output_dir
        / "float32_fixed_perturbation_per_trajectory_full_unrolled_summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(ablation_summary, file, indent=2, ensure_ascii=False)

    summary_plot_path = (
        base_output_dir
        / "float32_fixed_perturbation_per_trajectory_full_unrolled_summary.png"
    )
    plot_float32_optimizer_perturbation_range_summary(
        experiment_summaries=experiment_summaries,
        save_path=summary_plot_path,
    )

    print("\n" + "=" * 72)
    print("✅ 两组 float32 固定扰动逐轨迹完整反向传播实验全部完成。")
    print(f"📁 总输出目录: {base_output_dir}")
    print(f"📄 汇总文件: {summary_path}")
    print(f"🖼️ 汇总图片: {summary_plot_path}")
    for item in experiment_summaries:
        print(
            f"- {item['experiment_name']}: "
            f"optimizer={item['optimizer_name'].upper()}，"
            f"lr={item['learning_rate']:.1e}，"
            f"sigma={item['perturbation_std']:.1e}，"
            f"dtype={item['torch_dtype']}，"
            f"最终参考 gap={item['final_reference_gap_after_fixed_steps']:.4e}，"
            f"MLP residual={item['final_reference_residual_norm_after_fixed_steps']:.4e}，"
            f"Newton residual={item['newton_reference_residual_norm_after_fixed_steps']:.4e}"
        )


if __name__ == "__main__":
    main()
