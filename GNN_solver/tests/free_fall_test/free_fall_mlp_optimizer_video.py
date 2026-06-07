import os
import json
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # 适配无显示器 Linux
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ================= 1. 模型与隐式欧拉变分能量 =================
class MLPOptimizer(nn.Module):
    """
    输入: 当前优化变量 y(3) + 历史状态[p_n(3), v_n(3)] + 物理参数[m, g, dt](3)
    输出: 位置更新步长 delta_y(3)

    改动点 1：在模型内部做 base input 的数据标准化。
    改动点 2：网络预测 raw_delta，最终输出 delta_y = dt * raw_delta。
    注意：forward 调用方式不变，评估代码仍然使用 mlp(y, history, params)。
    """
    def __init__(self, input_mean=None, input_std=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        if input_mean is None:
            input_mean = torch.zeros(12)
        if input_std is None:
            input_std = torch.ones(12)

        self.register_buffer("input_mean", input_mean.clone().detach())
        self.register_buffer("input_std", input_std.clone().detach())

    def forward(self, y, history, params):
        # history = [p_n, v_n] (6D), params = [m, g, dt] (3D)
        inp = torch.cat([y, history, params], dim=-1)

        # base-only 标准化；不加入 residual / gradient 信息
        inp = (inp - self.input_mean) / self.input_std

        # dt-scaled output：让网络学习 O(1) 的 raw_delta，最终仍输出位置更新 delta_y
        raw_delta = self.net(inp)
        dt = params[2]
        return dt * raw_delta


def variational_energy(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    隐式欧拉变分能量:
    E(y) = (m/(2*dt^2)) * ||y - p_n - dt*v_n||^2 + m*g*y_z
    """
    residual = y - p_n - dt * v_n
    kinetic_term = (m / (2 * dt**2)) * torch.sum(residual**2)
    potential_term = m * g * y[2]  # y_z is the 3rd component
    return kinetic_term + potential_term


def variational_residual(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    变分最优性 residual:
    r(y) = grad E(y)
         = (m/dt^2) * (y - p_n - dt*v_n) + [0, 0, mg]^T

    最优点满足 r(y*) = 0
    """
    residual = y - p_n - dt * v_n
    grad = (m / dt**2) * residual

    gravity = torch.zeros_like(y)
    gravity[2] = m * g

    return grad + gravity


def residual_norm(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    residual 的 L2 范数，用于评估收敛性。
    """
    with torch.no_grad():
        r = variational_residual(y, p_n, v_n, m, g, dt)
        return torch.norm(r).item()


def newton_direction(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    牛顿法方向: -H^{-1} * grad
    H = (m/dt^2) * I, grad = (m/dt^2)*(y - p_n - dt*v_n) + [0,0,mg]^T
    """
    residual = y - p_n - dt * v_n
    grad = (m / dt**2) * residual
    grad[2] += m * g  # add gravity term to z-component
    hess_inv = (dt**2) / m  # scalar since Hessian is isotropic
    return -grad * hess_inv


def make_training_states(y0, y_star, dt, num_line_points=11, num_local_points=32,
                         local_std_dt_units=1.0, seed=123):
    """
    训练集扩充，但不加入 residual 信息。

    1. line anchors:
       y(alpha) = (1-alpha) y0 + alpha y*, alpha in [0, 1]
       作用：让网络不仅学 y0 -> y*，还学中间状态 -> y*，并且包含 y* 本身。

    2. local anchors:
       y = y* + dt * sigma * N(0, I)
       作用：让网络学习最优点附近任意 3D 小扰动都应该回到 y*。
    """
    train_states = []

    for alpha in torch.linspace(0.0, 1.0, num_line_points, dtype=y0.dtype, device=y0.device):
        train_states.append((1.0 - alpha) * y0 + alpha * y_star)

    if num_local_points > 0 and local_std_dt_units > 0:
        gen = torch.Generator(device=y0.device)
        gen.manual_seed(seed)
        for _ in range(num_local_points):
            noise = torch.randn(3, generator=gen, dtype=y0.dtype, device=y0.device)
            train_states.append(y_star + dt * local_std_dt_units * noise)

    return train_states


def compute_input_normalizer(train_states, history, params):
    """
    对 base input = [y, history, params] 做 dataset standardization。
    常量维度例如 p_n, v_n, m, g, dt 的 std 可能为 0，此时设为 1，避免除零。
    """
    inputs = []
    for y in train_states:
        inputs.append(torch.cat([y, history, params], dim=-1))

    x = torch.stack(inputs, dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)

    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


# ================= 2. 单步求解器：Newton 与 MLP Optimizer =================
def solve_one_step_newton(
    p_n,
    v_n,
    m=1.0,
    g=9.8,
    dt=0.01,
    residual_drop=1e-3,
    max_iters=5,
    abs_tol=1e-12
):
    """
    用 Newton 法求解单个隐式欧拉时间步。
    初值 y0 = p_n。
    停止条件：residual 下降三个数量级，或迭代达到 max_iters。
    """
    y = p_n.clone()

    r0 = residual_norm(y, p_n, v_n, m, g, dt)
    target = max(r0 * residual_drop, abs_tol)

    residual_hist = [r0]
    energy_hist = [variational_energy(y, p_n, v_n, m, g, dt).item()]

    num_iters = 0

    if r0 <= abs_tol:
        p_next = y.detach()
        v_next = (p_next - p_n) / dt
        return p_next, v_next.detach(), {
            "num_iters": num_iters,
            "residual_initial": r0,
            "residual_final": r0,
            "residual_hist": residual_hist,
            "energy_hist": energy_hist
        }

    for it in range(max_iters):
        d = newton_direction(y, p_n, v_n, m, g, dt)
        y = y + d

        r = residual_norm(y, p_n, v_n, m, g, dt)
        e = variational_energy(y, p_n, v_n, m, g, dt).item()

        residual_hist.append(r)
        energy_hist.append(e)

        num_iters = it + 1

        if r <= target:
            break

    p_next = y.detach()
    v_next = (p_next - p_n) / dt

    return p_next, v_next.detach(), {
        "num_iters": num_iters,
        "residual_initial": r0,
        "residual_final": residual_hist[-1],
        "residual_hist": residual_hist,
        "energy_hist": energy_hist
    }


def solve_one_step_mlp(
    mlp,
    p_n,
    v_n,
    params,
    m=1.0,
    g=9.8,
    dt=0.01,
    residual_drop=1e-3,
    max_iters=5,
    abs_tol=1e-12
):
    """
    用训练好的 MLP optimizer 求解单个隐式欧拉时间步。
    初值 y0 = p_n。
    停止条件：residual 下降三个数量级，或迭代达到 max_iters。
    """
    y = p_n.clone()
    history = torch.cat([p_n, v_n])

    r0 = residual_norm(y, p_n, v_n, m, g, dt)
    target = max(r0 * residual_drop, abs_tol)

    residual_hist = [r0]
    energy_hist = [variational_energy(y, p_n, v_n, m, g, dt).item()]

    num_iters = 0

    if r0 <= abs_tol:
        p_next = y.detach()
        v_next = (p_next - p_n) / dt
        return p_next, v_next.detach(), {
            "num_iters": num_iters,
            "residual_initial": r0,
            "residual_final": r0,
            "residual_hist": residual_hist,
            "energy_hist": energy_hist
        }

    for it in range(max_iters):
        with torch.no_grad():
            d = mlp(y, history, params)

        y = y + d

        r = residual_norm(y, p_n, v_n, m, g, dt)
        e = variational_energy(y, p_n, v_n, m, g, dt).item()

        residual_hist.append(r)
        energy_hist.append(e)

        num_iters = it + 1

        if r <= target:
            break

    p_next = y.detach()
    v_next = (p_next - p_n) / dt

    return p_next, v_next.detach(), {
        "num_iters": num_iters,
        "residual_initial": r0,
        "residual_final": residual_hist[-1],
        "residual_hist": residual_hist,
        "energy_hist": energy_hist
    }


# ================= 3. 自由落体序列 Rollout =================
def rollout_free_fall_sequence(
    mlp,
    p0,
    v0,
    params,
    m=1.0,
    g=9.8,
    dt=0.01,
    num_frames=300,
    residual_drop=1e-3,
    max_iters=5
):
    """
    分别用 Newton 和 MLP optimizer 滚动计算自由落体运动序列。

    num_frames 表示输出视频帧数。
    内部会执行 num_frames 个时间步。

    每个时间步：
        1. 从 y0 = p_n 开始迭代求 p_{n+1}
        2. 若 residual 下降 1e-3 或迭代达到 5 次，则停止
        3. 用 v_{n+1} = (p_{n+1} - p_n) / dt 更新速度
    """
    p_newton = p0.clone()
    v_newton = v0.clone()

    p_mlp = p0.clone()
    v_mlp = v0.clone()

    newton_positions = [p_newton.tolist()]
    mlp_positions = [p_mlp.tolist()]

    newton_velocities = [v_newton.tolist()]
    mlp_velocities = [v_mlp.tolist()]

    newton_step_info = []
    mlp_step_info = []

    for frame in range(num_frames):
        p_newton, v_newton, info_newton = solve_one_step_newton(
            p_newton,
            v_newton,
            m=m,
            g=g,
            dt=dt,
            residual_drop=residual_drop,
            max_iters=max_iters
        )

        p_mlp, v_mlp, info_mlp = solve_one_step_mlp(
            mlp,
            p_mlp,
            v_mlp,
            params,
            m=m,
            g=g,
            dt=dt,
            residual_drop=residual_drop,
            max_iters=max_iters
        )

        newton_positions.append(p_newton.tolist())
        mlp_positions.append(p_mlp.tolist())

        newton_velocities.append(v_newton.tolist())
        mlp_velocities.append(v_mlp.tolist())

        info_newton["frame"] = frame + 1
        info_mlp["frame"] = frame + 1

        newton_step_info.append(info_newton)
        mlp_step_info.append(info_mlp)

    return {
        "config": {
            "num_frames": num_frames,
            "dt": dt,
            "residual_drop": residual_drop,
            "max_iters": max_iters,
            "p0": p0.tolist(),
            "v0": v0.tolist(),
            "m": m,
            "g": g
        },
        "newton": {
            "positions": newton_positions,
            "velocities": newton_velocities,
            "step_info": newton_step_info
        },
        "mlp": {
            "positions": mlp_positions,
            "velocities": mlp_velocities,
            "step_info": mlp_step_info
        }
    }


# ================= 4. 视频渲染 =================
def render_free_fall_comparison_video(
    sequence_report,
    output_path="free_fall_newton_vs_mlp.mp4",
    fps=30
):
    """
    在同一个视频窗口中分成两个 3D 子窗口：
    左边 Newton，右边 MLP optimizer。
    """
    newton_positions = np.array(sequence_report["newton"]["positions"], dtype=np.float64)
    mlp_positions = np.array(sequence_report["mlp"]["positions"], dtype=np.float64)

    num_frames = sequence_report["config"]["num_frames"]
    dt = sequence_report["config"]["dt"]

    all_positions = np.concatenate([newton_positions, mlp_positions], axis=0)

    finite_mask = np.all(np.isfinite(all_positions), axis=1)
    if not np.all(finite_mask):
        print("⚠️ 检测到非有限位置值，渲染时将只使用有限值估计坐标范围。")
        all_positions_for_range = all_positions[finite_mask]
    else:
        all_positions_for_range = all_positions

    if all_positions_for_range.shape[0] == 0:
        raise RuntimeError("所有位置值都是非有限值，无法渲染视频。")

    mins = all_positions_for_range.min(axis=0)
    maxs = all_positions_for_range.max(axis=0)

    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    if span < 1e-6:
        span = 1.0

    margin = 0.1 * span
    half = 0.5 * span + margin

    xlim = (center[0] - half, center[0] + half)
    ylim = (center[1] - half, center[1] + half)
    zlim = (center[2] - half, center[2] + half)

    fig = plt.figure(figsize=(12, 6))
    ax_newton = fig.add_subplot(1, 2, 1, projection="3d")
    ax_mlp = fig.add_subplot(1, 2, 2, projection="3d")

    axes = [ax_newton, ax_mlp]

    for ax, title in zip(axes, ["Newton Method", "MLP Optimizer"]):
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=-60)

    newton_trail, = ax_newton.plot([], [], [], linewidth=2, label="trajectory")
    newton_point, = ax_newton.plot([], [], [], marker="o", markersize=8, label="point")
    newton_text = ax_newton.text2D(0.03, 0.93, "", transform=ax_newton.transAxes)

    mlp_trail, = ax_mlp.plot([], [], [], linewidth=2, label="trajectory")
    mlp_point, = ax_mlp.plot([], [], [], marker="o", markersize=8, label="point")
    mlp_text = ax_mlp.text2D(0.03, 0.93, "", transform=ax_mlp.transAxes)

    ax_newton.legend(loc="lower left")
    ax_mlp.legend(loc="lower left")

    def set_3d_line(line, points):
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

    def set_3d_point(point, p):
        point.set_data([p[0]], [p[1]])
        point.set_3d_properties([p[2]])

    def update(frame_id):
        # frame_id 从 0 开始；idx 从 1 开始，对应第 1 个求解后的状态
        idx = frame_id + 1
        t = idx * dt

        newton_curve = newton_positions[:idx + 1]
        mlp_curve = mlp_positions[:idx + 1]

        set_3d_line(newton_trail, newton_curve)
        set_3d_point(newton_point, newton_positions[idx])

        set_3d_line(mlp_trail, mlp_curve)
        set_3d_point(mlp_point, mlp_positions[idx])

        newton_info = sequence_report["newton"]["step_info"][idx - 1]
        mlp_info = sequence_report["mlp"]["step_info"][idx - 1]

        newton_text.set_text(
            f"frame={idx}\n"
            f"t={t:.2f}s\n"
            f"iters={newton_info['num_iters']}\n"
            f"res={newton_info['residual_final']:.2e}"
        )

        mlp_text.set_text(
            f"frame={idx}\n"
            f"t={t:.2f}s\n"
            f"iters={mlp_info['num_iters']}\n"
            f"res={mlp_info['residual_final']:.2e}"
        )

        return (
            newton_trail,
            newton_point,
            newton_text,
            mlp_trail,
            mlp_point,
            mlp_text
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False
    )

    plt.tight_layout()

    if output_path.endswith(".mp4"):
        if FFMpegWriter.isAvailable():
            writer = FFMpegWriter(fps=fps, bitrate=2400)
            anim.save(output_path, writer=writer)
            print(f"🎬 视频已保存至: {output_path}")
        else:
            fallback_path = output_path.replace(".mp4", ".gif")
            writer = PillowWriter(fps=fps)
            anim.save(fallback_path, writer=writer)
            print("⚠️ 当前环境未检测到 ffmpeg，已改存 GIF。")
            print(f"🎬 动画已保存至: {fallback_path}")
    else:
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        print(f"🎬 动画已保存至: {output_path}")

    plt.close(fig)


# ================= 5. 主程序 =================
def main():
    torch.manual_seed(42)

    # ----------------- 物理参数 -----------------
    m, g, dt = 1.0, 9.8, 0.01

    # 固定初值: 当前状态 (p_n, v_n)
    p_n = torch.tensor([3., 4., 5.])
    v_n = torch.tensor([0.5, -0.5, 0.0])  # 小初速度

    # 优化变量初值: y_0 = p_n (从当前位置开始优化下一位置)
    y0 = p_n.clone()

    # 网络输入拼接: history=[p_n, v_n], params=[m, g, dt]
    history = torch.cat([p_n, v_n])  # 6D
    params = torch.tensor([m, g, dt])  # 3D

    epochs = 1000
    K = 1
    train_log = []
    eval_log = []

    # 理论最优解与最优能量
    y_star = p_n + dt * v_n - dt**2 * torch.tensor([0., 0., g])
    E_star = variational_energy(y_star, p_n, v_n, m, g, dt).item()

    # ----------------- 训练集扩充 + base input 标准化 -----------------
    train_states = make_training_states(
        y0,
        y_star,
        dt,
        num_line_points=11,
        num_local_points=32,
        local_std_dt_units=1.0,
        seed=123
    )
    input_mean, input_std = compute_input_normalizer(train_states, history, params)

    mlp = MLPOptimizer(input_mean=input_mean, input_std=input_std)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    print("🚀 开始训练 (隐式欧拉变分能量最小化)")
    print(f"当前状态: p_n={p_n.tolist()}, v_n={v_n.tolist()}")
    print(f"理论最优: y*={y_star.tolist()}, E*={E_star:.6f}")
    print("策略: 每100 epoch K+=1, 步间 detach, 单步反向传播")
    print("新增: base input 数据标准化 + line/local 训练集扩充，不加入 residual 输入\n")

    for epoch in range(epochs):
        if epoch > 0 and epoch % 100 == 0 and K < 10:
            K += 1

        epoch_loss_sum = 0.0
        epoch_step_count = 0

        # === 保留原有训练逻辑：每个训练初值都执行 K 步，每步 backward/step/zero_grad/detach ===
        for y_init in train_states:
            y = y_init.clone()

            for _ in range(K):
                delta = mlp(y, history, params)  # 前向：网络预测位置更新
                y = y + delta                    # 应用更新
                loss = variational_energy(y, p_n, v_n, m, g, dt)  # 计算变分能量
                loss.backward()                  # 梯度回传
                opt.step()                       # 更新参数
                opt.zero_grad()                  # 清空梯度
                y = y.detach()                   # 切断历史计算图

                epoch_loss_sum += loss.item()
                epoch_step_count += 1

        mean_train_loss = epoch_loss_sum / max(epoch_step_count, 1)
        train_log.append({
            "epoch": epoch,
            "K": K,
            "final_loss": mean_train_loss,
            "num_train_states": len(train_states)
        })

        # 每100个epoch评估一次
        # 注意：评估方式保持不变，仍然从 y0 出发 rollout 10 步
        if epoch % 100 == 0 or epoch == epochs - 1:
            y_eval = y0.clone()
            eval_steps = []
            for i in range(10):
                with torch.no_grad():
                    d = mlp(y_eval, history, params)
                y_eval = y_eval + d
                e_val = variational_energy(y_eval, p_n, v_n, m, g, dt).item()
                r_val = residual_norm(y_eval, p_n, v_n, m, g, dt)
                eval_steps.append({
                    "step": i + 1,
                    "y": y_eval.tolist(),
                    "loss": e_val,
                    "residual_norm": r_val
                })

            eval_log.append({"epoch": epoch, "K": K, "steps": eval_steps})
            gap = eval_steps[-1]["loss"] - E_star
            print(f"Epoch {epoch:3d} | K={K:2d} | Eval Gap(10步): {gap:.4e}")

    print("\n✅ 训练完成。开始最终对比评估...\n")

    # ================= 6. 最终对比评估 =================
    max_steps = 15
    mlp_hist = {
        "init_y": y0.tolist(),
        "history": [p_n.tolist(), v_n.tolist()],
        "params": params.tolist(),
        "E_star": E_star,
        "iterations": []
    }
    newton_hist = {
        "init_y": y0.tolist(),
        "history": [p_n.tolist(), v_n.tolist()],
        "params": params.tolist(),
        "E_star": E_star,
        "iterations": []
    }

    # 记录初始状态 (Step 0)
    E0 = variational_energy(y0, p_n, v_n, m, g, dt).item()
    R0 = residual_norm(y0, p_n, v_n, m, g, dt)

    mlp_hist["iterations"].append({
        "step": 0,
        "y": y0.tolist(),
        "loss": E0,
        "residual_norm": R0
    })
    newton_hist["iterations"].append({
        "step": 0,
        "y": y0.tolist(),
        "loss": E0,
        "residual_norm": R0
    })

    y_mlp, y_new = y0.clone(), y0.clone()
    mlp_losses = [E0]
    newton_losses = [E0]
    mlp_residuals = [R0]
    newton_residuals = [R0]

    for i in range(max_steps):
        # MLP 迭代
        with torch.no_grad():
            d_m = mlp(y_mlp, history, params)
        y_mlp = y_mlp + d_m
        e_m = variational_energy(y_mlp, p_n, v_n, m, g, dt).item()
        r_m = residual_norm(y_mlp, p_n, v_n, m, g, dt)

        mlp_hist["iterations"].append({
            "step": i + 1,
            "y": y_mlp.tolist(),
            "loss": e_m,
            "residual_norm": r_m
        })
        mlp_losses.append(e_m)
        mlp_residuals.append(r_m)

        # 牛顿法迭代
        d_n = newton_direction(y_new, p_n, v_n, m, g, dt)
        y_new = y_new + d_n
        e_n = variational_energy(y_new, p_n, v_n, m, g, dt).item()
        r_n = residual_norm(y_new, p_n, v_n, m, g, dt)

        newton_hist["iterations"].append({
            "step": i + 1,
            "y": y_new.tolist(),
            "loss": e_n,
            "residual_norm": r_n
        })
        newton_losses.append(e_n)
        newton_residuals.append(r_n)

    # 打印前5步
    print("📊 最终迭代结果对比 (前5步):")
    print(f"{'Step':<5} | {'MLP Loss':<12} | {'MLP Residual':<14} | {'Newton Residual':<16} | {'MLP y':<25}")
    print("-" * 95)
    for i in range(min(5, max_steps + 1)):
        mlp_it = mlp_hist["iterations"][i]
        newton_it = newton_hist["iterations"][i]
        y_str = str([round(v, 4) for v in mlp_it["y"]])

        print(
            f"{mlp_it['step']:<5} | "
            f"{mlp_it['loss']:<12.6f} | "
            f"{mlp_it['residual_norm']:<14.4e} | "
            f"{newton_it['residual_norm']:<16.4e} | "
            f"{y_str:<25}"
        )

    # ================= 7. 保存单步优化结果 =================
    report = {
        "config": {
            "epochs": epochs,
            "y0": y0.tolist(),
            "p_n": p_n.tolist(),
            "v_n": v_n.tolist(),
            "m": m,
            "g": g,
            "dt": dt,
            "E_star": E_star,
            "normalization": {
                "input_mean": input_mean.tolist(),
                "input_std": input_std.tolist()
            },
            "training_set_expansion": {
                "num_line_points": 11,
                "num_local_points": 32,
                "local_std_dt_units": 1.0,
                "num_train_states": len(train_states)
            }
        },
        "training_log": train_log,
        "periodic_evaluation": eval_log,
        "final_comparison": {
            "mlp": mlp_hist,
            "newton": newton_hist
        }
    }

    with open("optimization_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n📁 数值结果已保存至: optimization_report.json")

    # ================= 8. 绘图：energy gap / residual =================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 绘制 Gap = Loss - E_star 以适配对数坐标（避免负数/零值）
    gap_mlp = [max(l - E_star, 1e-12) for l in mlp_losses]
    gap_newton = [max(l - E_star, 1e-12) for l in newton_losses]
    train_gap = [max(e["final_loss"] - E_star, 1e-12) for e in train_log]
    eval_gap = [max(e["steps"][-1]["loss"] - E_star, 1e-12) for e in eval_log]

    # 图1: 训练Gap曲线
    axes[0, 0].plot([e["epoch"] for e in train_log], train_gap, color="steelblue")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Training Convergence Gap (E - E*)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Gap")
    axes[0, 0].grid(True, alpha=0.3)

    # 图2: 周期评估Gap
    axes[0, 1].plot([e["epoch"] for e in eval_log], eval_gap, marker="o", color="darkgreen")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Periodic Eval Gap (10 steps)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Gap")
    axes[0, 1].grid(True, alpha=0.3)

    # 图3: 最终收敛对比（Energy Gap）
    steps = np.arange(max_steps + 1)
    axes[1, 0].plot(steps, gap_mlp, label="MLP Optimizer", marker="o")
    axes[1, 0].plot(steps, gap_newton, label="Newton Method", marker="s", linestyle="--", color="crimson")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Final Convergence Comparison (Energy Gap)")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Gap")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 图4: Residual Norm 对比
    res_mlp = [max(r, 1e-12) for r in mlp_residuals]
    res_newton = [max(r, 1e-12) for r in newton_residuals]
    axes[1, 1].plot(steps, res_mlp, label="MLP Optimizer", marker="o")
    axes[1, 1].plot(steps, res_newton, label="Newton Method", marker="s", linestyle="--", color="crimson")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Final Residual Norm Comparison")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel(r"||grad E(y)||_2")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optimization_report.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("🖼️  可视化图表已保存至: optimization_report.png")

    # ================= 9. 用求解器滚动计算自由落体序列并渲染视频 =================
    print("\n🎥 开始滚动计算 300 帧自由落体运动序列...")

    num_frames = 300
    residual_drop = 1e-3
    max_solver_iters = 5
    fps = 30

    sequence_report = rollout_free_fall_sequence(
        mlp=mlp,
        p0=p_n,
        v0=v_n,
        params=params,
        m=m,
        g=g,
        dt=dt,
        num_frames=num_frames,
        residual_drop=residual_drop,
        max_iters=max_solver_iters
    )

    with open("free_fall_sequence_report.json", "w") as f:
        json.dump(sequence_report, f, indent=2)

    print("📁 运动序列数值结果已保存至: free_fall_sequence_report.json")

    # 打印一些求解统计
    newton_iters = [s["num_iters"] for s in sequence_report["newton"]["step_info"]]
    mlp_iters = [s["num_iters"] for s in sequence_report["mlp"]["step_info"]]

    newton_final_res = [s["residual_final"] for s in sequence_report["newton"]["step_info"]]
    mlp_final_res = [s["residual_final"] for s in sequence_report["mlp"]["step_info"]]

    print("\n📊 300 帧单步求解统计:")
    print(f"Newton 平均迭代次数: {np.mean(newton_iters):.2f}")
    print(f"MLP    平均迭代次数: {np.mean(mlp_iters):.2f}")
    print(f"Newton 最后一帧 residual: {newton_final_res[-1]:.4e}")
    print(f"MLP    最后一帧 residual: {mlp_final_res[-1]:.4e}")

    render_free_fall_comparison_video(
        sequence_report,
        output_path="free_fall_newton_vs_mlp.mp4",
        fps=fps
    )

    print("=" * 50)


if __name__ == "__main__":
    main()
