import json

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

# True：绘制相对于初值 y0 的位移，更容易看清聚集在一起的点
# False：绘制绝对空间坐标
PLOT_RELATIVE_COORDINATES = False

# 每多少个 epoch 使用一种颜色
COLOR_BUCKET_SIZE = 200


# ============================================================
# 1. 模型与隐式欧拉变分能量
# ============================================================

class MLPOptimizer(nn.Module):
    """
    学习型迭代优化器。

    输入:
        当前优化变量 y                      : 3D
        历史状态 history = [p_n, v_n]       : 6D
        物理参数 params = [m, g, dt]        : 3D

    输出:
        位置更新步长 delta_y                : 3D

    迭代形式:
        y^(k+1) = y^(k) + delta_y^(k)
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y, history, params):
        inp = torch.cat([y, history, params], dim=-1)
        return self.net(inp)


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


# ============================================================
# 2. 绘图辅助函数
# ============================================================

def to_plot_coordinates(points, y0, relative=True):
    """
    将点转换为绘图坐标。

    relative=True:
        绘制 y - y0

    relative=False:
        绘制绝对坐标 y
    """

    points_np = np.asarray(points, dtype=float).reshape(-1, 3)
    y0_np = np.asarray(y0, dtype=float).reshape(1, 3)

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
# 3. 绘制最终测试轨迹
# ============================================================

def plot_final_test_distribution(
    mlp_hist,
    newton_solution,
    y0,
    save_path="final_test_distribution.png",
    relative=True,
):
    """
    绘制训练结束后，MLP 优化器从初值出发的测试轨迹。

    标记规则:
        0, 1, 2, ... : MLP 迭代次序
        *            : Newton 法收敛解
    """

    mlp_points = to_plot_coordinates(
        [item["y"] for item in mlp_hist["iterations"]],
        y0=y0,
        relative=relative,
    )

    newton_point = to_plot_coordinates(
        [newton_solution],
        y0=y0,
        relative=relative,
    )[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # MLP 测试轨迹
    ax.plot(
        mlp_points[:, 0],
        mlp_points[:, 1],
        mlp_points[:, 2],
        "-o",
        linewidth=1.5,
        markersize=4,
        label="MLP test trajectory",
    )

    # 给 MLP 每一个测试点标记迭代序号
    for step, point in enumerate(mlp_points):
        ax.text(
            point[0],
            point[1],
            point[2],
            f"  {step}",
            fontsize=9,
        )

    # Newton 收敛点
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
# 4. 绘制整个训练过程中的训练点和结果点
# ============================================================

def plot_training_points_and_results(
    visualization_log,
    newton_solution,
    y0,
    epochs,
    bucket_size=200,
    save_path="training_points_and_results_distribution.png",
    relative=True,
):
    """
    可视化整个训练过程中的点分布。

    左图:
        所有训练输入点:
            x^(0), x^(1), ..., x^(K-1)

    右图:
        每个 epoch 的最终结果点:
            x^(K)

    颜色:
        每 bucket_size 个 epoch 使用一种颜色。

    特殊标记:
        黑色 x : 固定初值 x^(0)
        红色 * : Newton 法收敛解
    """

    num_buckets = (epochs + bucket_size - 1) // bucket_size
    cmap = plt.get_cmap("tab10", num_buckets)

    fig = plt.figure(figsize=(19, 8))

    ax_train = fig.add_subplot(121, projection="3d")
    ax_result = fig.add_subplot(122, projection="3d")

    all_train_points = []
    all_result_points = []

    for item in visualization_log:
        bucket = item["bucket"]
        color = cmap(bucket)

        train_points = to_plot_coordinates(
            item["train_points"],
            y0=y0,
            relative=relative,
        )

        result_point = to_plot_coordinates(
            [item["result_point"]],
            y0=y0,
            relative=relative,
        )

        # 左图：当前 epoch 参与训练的输入点
        ax_train.scatter(
            train_points[:, 0],
            train_points[:, 1],
            train_points[:, 2],
            marker="o",
            s=11,
            alpha=0.30,
            color=color,
        )

        # 右图：当前 epoch 完成全部 K 步之后得到的结果点
        ax_result.scatter(
            result_point[:, 0],
            result_point[:, 1],
            result_point[:, 2],
            marker="^",
            s=24,
            alpha=0.75,
            color=color,
        )

        all_train_points.append(train_points)
        all_result_points.append(result_point)

    newton_point = to_plot_coordinates(
        [newton_solution],
        y0=y0,
        relative=relative,
    )[0]

    initial_point = to_plot_coordinates(
        [y0],
        y0=y0,
        relative=relative,
    )[0]

    # 两张图中都画出初值和 Newton 解
    for ax in (ax_train, ax_result):
        ax.scatter(
            initial_point[0],
            initial_point[1],
            initial_point[2],
            marker="x",
            s=120,
            c="black",
            linewidths=2.0,
            label=r"Initial point $x^{(0)}$",
        )

        ax.scatter(
            newton_point[0],
            newton_point[1],
            newton_point[2],
            marker="*",
            s=360,
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

        set_axis_labels(ax, relative=relative)
        ax.grid(True, alpha=0.3)

    ax_train.set_title(
        r"All Training Input Points: "
        r"$x^{(0)}, x^{(1)}, \ldots, x^{(K-1)}$"
    )

    ax_result.set_title(
        r"All Epoch Result Points: $x^{(K)}$"
    )

    # 分别设置两张图的坐标轴范围
    train_points_for_limits = np.vstack(
        all_train_points
        + [
            initial_point.reshape(1, 3),
            newton_point.reshape(1, 3),
        ]
    )

    result_points_for_limits = np.vstack(
        all_result_points
        + [
            initial_point.reshape(1, 3),
            newton_point.reshape(1, 3),
        ]
    )

    set_equal_3d_axes(ax_train, train_points_for_limits)
    set_equal_3d_axes(ax_result, result_points_for_limits)

    # 图例 1：点形状的含义
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
            label=r"Epoch result point $x^{(K)}$",
            markerfacecolor="gray",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            label=r"Fixed initial point $x^{(0)}$",
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

    # 图例 2：颜色的含义
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
# 5. 主程序
# ============================================================

def main():
    torch.manual_seed(42)

    # --------------------------------------------------------
    # 5.1 物理参数
    # --------------------------------------------------------

    m = 1.0
    g = 9.8
    dt = 0.01

    # 固定当前状态
    p_n = torch.tensor([3.0, 4.0, 5.0])
    v_n = torch.tensor([0.5, -0.5, 0.0])

    # 优化变量初值
    y0 = p_n.clone()

    # 网络额外输入
    history = torch.cat([p_n, v_n])
    params = torch.tensor([m, g, dt])

    # --------------------------------------------------------
    # 5.2 创建模型
    # --------------------------------------------------------

    mlp = MLPOptimizer()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    epochs = 1000
    K = 1

    # 常规日志
    train_log = []
    eval_log = []

    # 保存整个训练过程中涉及到的点
    visualization_log = []

    # --------------------------------------------------------
    # 5.3 理论最优解与 Newton 收敛解
    # --------------------------------------------------------

    # 理论解析最优解
    y_star = (
        p_n
        + dt * v_n
        - dt**2 * torch.tensor([0.0, 0.0, g])
    )

    E_star = variational_energy(
        y_star,
        p_n,
        v_n,
        m,
        g,
        dt,
    ).item()

    # Newton 法从 y0 出发，一步计算收敛解
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

    print("🚀 开始训练：隐式欧拉变分能量最小化")
    print(f"当前状态: p_n={p_n.tolist()}, v_n={v_n.tolist()}")
    print(f"理论最优解: y*={y_star.tolist()}")
    print(f"Newton 一步解: {newton_solution.tolist()}")
    print(f"理论最优能量: E*={E_star:.8f}")
    print("策略: 每 100 个 epoch 增加一次 K，步间 detach，单步反向传播")
    print("可视化: 每 200 个 epoch 使用一种颜色\n")

    # --------------------------------------------------------
    # 5.4 训练
    # --------------------------------------------------------

    for epoch in range(epochs):
        if epoch > 0 and epoch % 200 == 0 and K < 10:
            K += 1

        # 每个 epoch 都从固定初值出发
        y = y0.clone()

        # 当前 epoch 中，网络实际接收的全部输入点
        epoch_train_points = []

        for k in range(K):
            # ----------------------------------------------
            # x^(k) 是当前第 k 次网络调用的输入
            # 因此将其记录为训练点
            # ----------------------------------------------
            epoch_train_points.append(
                y.detach().clone()
            )

            # 网络预测更新方向
            delta = mlp(
                y,
                history,
                params,
            )

            # 得到 x^(k+1)
            y = y + delta

            # 在更新后的点计算能量
            loss = variational_energy(
                y,
                p_n,
                v_n,
                m,
                g,
                dt,
            )

            # 单步反向传播和参数更新
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

            # 切断历史计算图
            y = y.detach()

        # 完成 K 次更新后:
        # y = x^(K)，将其保存为当前 epoch 的结果点
        epoch_result_point = y.detach().clone()

        visualization_log.append(
            {
                "epoch": epoch,
                "bucket": epoch // COLOR_BUCKET_SIZE,
                "K": K,
                "train_points": [
                    point.tolist()
                    for point in epoch_train_points
                ],
                "result_point": epoch_result_point.tolist(),
            }
        )

        train_log.append(
            {
                "epoch": epoch,
                "K": K,
                "final_loss": loss.item(),
            }
        )

        # ----------------------------------------------------
        # 每 100 个 epoch 进行一次固定 10 步评估
        # ----------------------------------------------------

        if epoch % 100 == 0 or epoch == epochs - 1:
            y_eval = y0.clone()
            eval_steps = []

            for i in range(10):
                with torch.no_grad():
                    delta_eval = mlp(
                        y_eval,
                        history,
                        params,
                    )

                y_eval = y_eval + delta_eval

                energy_eval = variational_energy(
                    y_eval,
                    p_n,
                    v_n,
                    m,
                    g,
                    dt,
                ).item()

                eval_steps.append(
                    {
                        "step": i + 1,
                        "y": y_eval.tolist(),
                        "loss": energy_eval,
                    }
                )

            eval_log.append(
                {
                    "epoch": epoch,
                    "K": K,
                    "steps": eval_steps,
                }
            )

            gap = eval_steps[-1]["loss"] - E_star

            print(
                f"Epoch {epoch:3d} | "
                f"K={K:2d} | "
                f"Eval Gap (10 steps): {gap:.4e}"
            )

    print("\n✅ 训练完成。开始最终对比评估...\n")

    # --------------------------------------------------------
    # 5.5 最终测试：MLP vs Newton
    # --------------------------------------------------------

    max_steps = 15

    mlp_hist = {
        "init_y": y0.tolist(),
        "history": [
            p_n.tolist(),
            v_n.tolist(),
        ],
        "params": params.tolist(),
        "E_star": E_star,
        "iterations": [],
    }

    newton_hist = {
        "init_y": y0.tolist(),
        "history": [
            p_n.tolist(),
            v_n.tolist(),
        ],
        "params": params.tolist(),
        "E_star": E_star,
        "iterations": [],
    }

    # Step 0
    E0 = variational_energy(
        y0,
        p_n,
        v_n,
        m,
        g,
        dt,
    ).item()

    mlp_hist["iterations"].append(
        {
            "step": 0,
            "y": y0.tolist(),
            "loss": E0,
        }
    )

    newton_hist["iterations"].append(
        {
            "step": 0,
            "y": y0.tolist(),
            "loss": E0,
        }
    )

    y_mlp = y0.clone()
    y_newton = y0.clone()

    mlp_losses = [E0]
    newton_losses = [E0]

    for i in range(max_steps):
        # MLP 迭代
        with torch.no_grad():
            delta_mlp = mlp(
                y_mlp,
                history,
                params,
            )

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
                "step": i + 1,
                "y": y_mlp.tolist(),
                "loss": energy_mlp,
            }
        )

        mlp_losses.append(energy_mlp)

        # Newton 迭代
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
                "step": i + 1,
                "y": y_newton.tolist(),
                "loss": energy_newton,
            }
        )

        newton_losses.append(energy_newton)

    # 打印前 5 步结果
    print("📊 最终迭代结果对比（前 5 步）:")
    print(
        f"{'Step':<5} | "
        f"{'MLP Loss':<14} | "
        f"{'MLP Gap':<14} | "
        f"{'Newton Gap':<14} | "
        f"{'MLP y'}"
    )

    print("-" * 100)

    for i in range(min(5, max_steps + 1)):
        mlp_item = mlp_hist["iterations"][i]
        newton_item = newton_hist["iterations"][i]

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

    # --------------------------------------------------------
    # 5.6 保存 JSON
    # --------------------------------------------------------

    report = {
        "config": {
            "epochs": epochs,
            "color_bucket_size": COLOR_BUCKET_SIZE,
            "plot_relative_coordinates": PLOT_RELATIVE_COORDINATES,
            "y0": y0.tolist(),
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
        "visualization_log": visualization_log,
        "final_comparison": {
            "mlp": mlp_hist,
            "newton": newton_hist,
        },
    }

    with open(
        "optimization_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n📁 数值结果已保存至: optimization_report.json")

    # --------------------------------------------------------
    # 5.7 原有的四宫格统计图
    # --------------------------------------------------------

    gap_mlp = [
        max(loss_value - E_star, 1e-12)
        for loss_value in mlp_losses
    ]

    gap_newton = [
        max(loss_value - E_star, 1e-12)
        for loss_value in newton_losses
    ]

    train_gap = [
        max(item["final_loss"] - E_star, 1e-12)
        for item in train_log
    ]

    eval_gap = [
        max(item["steps"][-1]["loss"] - E_star, 1e-12)
        for item in eval_log
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    # 图 1：训练 Gap
    axes[0, 0].plot(
        [item["epoch"] for item in train_log],
        train_gap,
        color="steelblue",
    )

    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Training Convergence Gap (E - E*)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Gap")
    axes[0, 0].grid(True, alpha=0.3)

    # 图 2：周期评估 Gap
    axes[0, 1].plot(
        [item["epoch"] for item in eval_log],
        eval_gap,
        marker="o",
        color="darkgreen",
    )

    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Periodic Eval Gap (10 steps)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Gap")
    axes[0, 1].grid(True, alpha=0.3)

    # 图 3：最终收敛对比
    test_steps = np.arange(max_steps + 1)

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
    axes[1, 0].set_title("Final Convergence Comparison (Gap)")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Gap")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 图 4：更新步长范数
    mlp_norms = []
    newton_norms = []

    y_mlp = y0.clone()
    y_newton = y0.clone()

    for _ in range(max_steps):
        with torch.no_grad():
            delta_mlp = mlp(
                y_mlp,
                history,
                params,
            )

        mlp_norms.append(
            torch.norm(delta_mlp).item()
        )

        y_mlp = y_mlp + delta_mlp

        delta_newton = newton_direction(
            y_newton,
            p_n,
            v_n,
            m,
            g,
            dt,
        )

        newton_norms.append(
            torch.norm(delta_newton).item()
        )

        y_newton = y_newton + delta_newton

    axes[1, 1].plot(
        np.arange(max_steps),
        mlp_norms,
        label=r"MLP $\|\Delta y\|$",
        marker="^",
    )

    axes[1, 1].plot(
        np.arange(max_steps),
        newton_norms,
        label=r"Newton $\|\Delta y\|$",
        marker="v",
        linestyle="--",
        color="crimson",
    )

    axes[1, 1].set_title("Update Step Magnitude")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel(r"$\|\Delta y\|_2$")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "optimization_report.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("🖼️ 常规统计图已保存至: optimization_report.png")

    # --------------------------------------------------------
    # 5.8 最终测试轨迹图
    # --------------------------------------------------------

    plot_final_test_distribution(
        mlp_hist=mlp_hist,
        newton_solution=newton_solution.tolist(),
        y0=y0.tolist(),
        save_path="final_test_distribution.png",
        relative=PLOT_RELATIVE_COORDINATES,
    )

    # --------------------------------------------------------
    # 5.9 整个训练过程中的训练点与结果点图
    # --------------------------------------------------------

    plot_training_points_and_results(
        visualization_log=visualization_log,
        newton_solution=newton_solution.tolist(),
        y0=y0.tolist(),
        epochs=epochs,
        bucket_size=COLOR_BUCKET_SIZE,
        save_path="training_points_and_results_distribution.png",
        relative=PLOT_RELATIVE_COORDINATES,
    )

    print("=" * 60)
    print("✅ 所有结果已经生成完成。")


if __name__ == "__main__":
    main()