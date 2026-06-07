import os
import json
import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # 适配无显示器 Linux
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# 1. 模型与隐式欧拉变分能量
# ============================================================
class MLPOptimizer(nn.Module):
    """
    输入:
        y(3) + history[p_n(3), v_n(3)] + params[m, g, dt](3) = 12D

    输出:
        delta_y(3)

    说明:
        1. 在模型内部做 base input 标准化。
        2. 网络预测 raw_delta，最终输出 delta_y = dt * raw_delta。
        3. forward 同时支持单样本:
              y:       [3]
              history: [6]
              params:  [3]
           和 batch:
              y:       [B, 3]
              history: [B, 6]
              params:  [B, 3]
    """
    def __init__(self, input_mean=None, input_std=None, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        if input_mean is None:
            input_mean = torch.zeros(12)
        if input_std is None:
            input_std = torch.ones(12)

        self.register_buffer("input_mean", input_mean.clone().detach())
        self.register_buffer("input_std", input_std.clone().detach())

    def forward(self, y, history, params):
        inp = torch.cat([y, history, params], dim=-1)
        inp = (inp - self.input_mean) / self.input_std

        raw_delta = self.net(inp)

        # 支持单样本和 batch。
        # 单样本 params[..., 2:3] shape=[1]，batch shape=[B,1]。
        dt = params[..., 2:3]
        return dt * raw_delta


def gravity_vector_like(y, m=1.0, g=9.8):
    gravity = torch.zeros_like(y)
    gravity[..., 2] = m * g
    return gravity


def variational_energy(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    隐式欧拉变分能量:
        E(y) = (m/(2*dt^2)) * ||y - p_n - dt*v_n||^2 + m*g*y_z

    支持:
        单样本: 返回 scalar tensor
        batch:  返回 [B] tensor
    """
    residual = y - p_n - dt * v_n
    kinetic_term = (m / (2 * dt**2)) * torch.sum(residual**2, dim=-1)
    potential_term = m * g * y[..., 2]
    return kinetic_term + potential_term


def variational_residual(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    变分最优性 residual:
        r(y) = grad E(y)
             = (m/dt^2) * (y - p_n - dt*v_n) + [0, 0, mg]^T

    最优点满足 r(y*) = 0。
    支持单样本和 batch。
    """
    residual = y - p_n - dt * v_n
    grad = (m / dt**2) * residual
    return grad + gravity_vector_like(y, m=m, g=g)


def residual_norm(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    residual 的 L2 范数。
    单样本返回 float；batch 返回 tensor [B]。
    """
    with torch.no_grad():
        r = variational_residual(y, p_n, v_n, m, g, dt)
        norm = torch.norm(r, dim=-1)
        if norm.ndim == 0:
            return norm.item()
        return norm


def newton_direction(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    牛顿法方向:
        d = -H^{-1} grad
        H = (m/dt^2) * I
    """
    grad = variational_residual(y, p_n, v_n, m, g, dt)
    hess_inv = (dt**2) / m
    return -grad * hess_inv


def implicit_euler_exact_solution(p_n, v_n, g=9.8, dt=0.01):
    """
    当前单点自由落体隐式欧拉变分问题的解析最优解。
    """
    gravity_acc = torch.zeros_like(p_n)
    gravity_acc[..., 2] = g
    return p_n + dt * v_n - dt**2 * gravity_acc


# ============================================================
# 2. 生成覆盖自由落体 rollout 分布的训练集
# ============================================================
def make_rollout_training_samples(
    p0,
    v0,
    params,
    m=1.0,
    g=9.8,
    dt=0.01,
    num_frames=300,
    num_line_points=11,
    num_local_points=8,
    local_std_dt_units=1.0,
    velocity_jitter_std=0.0,
    position_jitter_std=0.0,
    seed=123
):
    """
    生成覆盖 300 帧自由落体 rollout 分布的训练样本。

    关键修正:
        原脚本只在第一帧的固定 p_n、v_n 上训练。
        这里沿 Newton/解析轨迹生成每一帧的 p_n、v_n，
        然后为每个时间步构造多个 y_init，使 MLP 见到整个运动序列分布。

    每个训练样本包含:
        y_init:   当前优化变量初值或扰动点
        p_n:      当前时间步位置
        v_n:      当前时间步速度
        history:  [p_n, v_n]
        params:   [m, g, dt]
    """
    gen = torch.Generator(device=p0.device)
    gen.manual_seed(seed)

    y_list = []
    p_list = []
    v_list = []
    history_list = []
    params_list = []

    p = p0.clone()
    v = v0.clone()

    for _ in range(num_frames):
        # 可选扰动，用于增强泛化。默认关闭，保持和目标轨迹一致。
        p_train = p.clone()
        v_train = v.clone()

        if position_jitter_std > 0:
            p_train = p_train + position_jitter_std * torch.randn(3, generator=gen, dtype=p.dtype, device=p.device)

        if velocity_jitter_std > 0:
            v_train = v_train + velocity_jitter_std * torch.randn(3, generator=gen, dtype=v.dtype, device=v.device)

        y_star = implicit_euler_exact_solution(p_train, v_train, g=g, dt=dt)
        history = torch.cat([p_train, v_train])

        # line anchors: p_train -> y_star
        for alpha in torch.linspace(0.0, 1.0, num_line_points, dtype=p.dtype, device=p.device):
            y_init = (1.0 - alpha) * p_train + alpha * y_star

            y_list.append(y_init.detach())
            p_list.append(p_train.detach())
            v_list.append(v_train.detach())
            history_list.append(history.detach())
            params_list.append(params.detach())

        # local anchors: y_star 附近扰动
        for _ in range(num_local_points):
            noise = torch.randn(3, generator=gen, dtype=p.dtype, device=p.device)
            y_init = y_star + dt * local_std_dt_units * noise

            y_list.append(y_init.detach())
            p_list.append(p_train.detach())
            v_list.append(v_train.detach())
            history_list.append(history.detach())
            params_list.append(params.detach())

        # 用精确隐式欧拉更新，生成下一帧训练分布
        p_next = implicit_euler_exact_solution(p, v, g=g, dt=dt)
        v_next = (p_next - p) / dt

        p = p_next.detach()
        v = v_next.detach()

    return {
        "y_init": torch.stack(y_list, dim=0),
        "p_n": torch.stack(p_list, dim=0),
        "v_n": torch.stack(v_list, dim=0),
        "history": torch.stack(history_list, dim=0),
        "params": torch.stack(params_list, dim=0)
    }


def compute_input_normalizer_from_samples(samples):
    """
    对所有训练样本的 base input = [y, history, params] 做 dataset standardization。
    """
    x = torch.cat([samples["y_init"], samples["history"], samples["params"]], dim=-1)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


# ============================================================
# 3. 单步求解器：Newton 与 MLP Optimizer
# ============================================================
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

    停止条件:
        residual 下降三个数量级，或迭代达到 max_iters。
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

    停止条件:
        residual 下降三个数量级，或迭代达到 max_iters。
    """
    y = p_n.clone()
    history = torch.cat([p_n, v_n])

    r0 = residual_norm(y, p_n, v_n, m, g, dt)
    target = max(r0 * residual_drop, abs_tol)

    residual_hist = [r0]
    energy_hist = [variational_energy(y, p_n, v_n, m, g, dt).item()]
    delta_norm_hist = []

    num_iters = 0

    if r0 <= abs_tol:
        p_next = y.detach()
        v_next = (p_next - p_n) / dt
        return p_next, v_next.detach(), {
            "num_iters": num_iters,
            "residual_initial": r0,
            "residual_final": r0,
            "residual_hist": residual_hist,
            "energy_hist": energy_hist,
            "delta_norm_hist": delta_norm_hist
        }

    for it in range(max_iters):
        with torch.no_grad():
            d = mlp(y, history, params)

        y = y + d

        r = residual_norm(y, p_n, v_n, m, g, dt)
        e = variational_energy(y, p_n, v_n, m, g, dt).item()

        residual_hist.append(r)
        energy_hist.append(e)
        delta_norm_hist.append(torch.norm(d).item())

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
        "energy_hist": energy_hist,
        "delta_norm_hist": delta_norm_hist
    }


# ============================================================
# 4. 自由落体序列 Rollout
# ============================================================
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
    max_iters=5,
    print_debug=True
):
    """
    分别用 Newton 和 MLP optimizer 滚动计算自由落体运动序列。

    每个时间步:
        1. 从 y0 = p_n 开始迭代求 p_{n+1}
        2. 若 residual 下降 1e-3 或迭代达到 max_iters，则停止
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

    trajectory_error = []

    for frame in range(num_frames):
        old_p_mlp = p_mlp.clone()

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

        pos_err = torch.norm(p_mlp - p_newton).item()
        trajectory_error.append(pos_err)

        newton_positions.append(p_newton.tolist())
        mlp_positions.append(p_mlp.tolist())

        newton_velocities.append(v_newton.tolist())
        mlp_velocities.append(v_mlp.tolist())

        info_newton["frame"] = frame + 1
        info_newton["position_error_vs_other"] = pos_err

        info_mlp["frame"] = frame + 1
        info_mlp["position_error_vs_newton"] = pos_err
        info_mlp["step_displacement_norm"] = torch.norm(p_mlp - old_p_mlp).item()

        newton_step_info.append(info_newton)
        mlp_step_info.append(info_mlp)

        if print_debug and (frame < 5 or (frame + 1) % 50 == 0):
            dp = p_mlp - old_p_mlp
            print(
                f"[Rollout {frame + 1:03d}] "
                f"Newton z={p_newton[2].item(): .5f}, "
                f"MLP z={p_mlp[2].item(): .5f}, "
                f"|err|={pos_err:.3e}, "
                f"MLP dp={dp.tolist()}, "
                f"MLP res {info_mlp['residual_initial']:.2e}->{info_mlp['residual_final']:.2e}, "
                f"iters={info_mlp['num_iters']}"
            )

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
        },
        "trajectory_error_norm": trajectory_error
    }


# ============================================================
# 5. 视频渲染
# ============================================================
def render_free_fall_comparison_video(
    sequence_report,
    output_path="free_fall_newton_vs_mlp.mp4",
    fps=30,
    same_axes=True
):
    """
    在同一个视频窗口中分成两个 3D 子窗口:
        左边 Newton，右边 MLP optimizer。
    """
    newton_positions = np.array(sequence_report["newton"]["positions"], dtype=np.float64)
    mlp_positions = np.array(sequence_report["mlp"]["positions"], dtype=np.float64)

    num_frames = sequence_report["config"]["num_frames"]
    dt = sequence_report["config"]["dt"]

    if same_axes:
        all_positions = np.concatenate([newton_positions, mlp_positions], axis=0)
        limits = compute_axis_limits(all_positions)
        newton_limits = limits
        mlp_limits = limits
    else:
        newton_limits = compute_axis_limits(newton_positions)
        mlp_limits = compute_axis_limits(mlp_positions)

    fig = plt.figure(figsize=(12, 6))
    ax_newton = fig.add_subplot(1, 2, 1, projection="3d")
    ax_mlp = fig.add_subplot(1, 2, 2, projection="3d")

    setup_3d_axis(ax_newton, "Newton Method", newton_limits)
    setup_3d_axis(ax_mlp, "MLP Optimizer", mlp_limits)

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
        pos_err = sequence_report["trajectory_error_norm"][idx - 1]

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
            f"res={mlp_info['residual_final']:.2e}\n"
            f"|err|={pos_err:.2e}"
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


def compute_axis_limits(positions):
    finite_mask = np.all(np.isfinite(positions), axis=1)
    if not np.all(finite_mask):
        print("⚠️ 检测到非有限位置值，渲染时将只使用有限值估计坐标范围。")
        positions = positions[finite_mask]

    if positions.shape[0] == 0:
        raise RuntimeError("所有位置值都是非有限值，无法渲染。")

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)

    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    if span < 1e-6:
        span = 1.0

    margin = 0.1 * span
    half = 0.5 * span + margin

    return {
        "xlim": (center[0] - half, center[0] + half),
        "ylim": (center[1] - half, center[1] + half),
        "zlim": (center[2] - half, center[2] + half)
    }


def setup_3d_axis(ax, title, limits):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim(*limits["xlim"])
    ax.set_ylim(*limits["ylim"])
    ax.set_zlim(*limits["zlim"])

    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=-60)


# ============================================================
# 6. 训练与评估辅助函数
# ============================================================
def train_mlp_optimizer(
    mlp,
    train_samples,
    m=1.0,
    g=9.8,
    dt=0.01,
    epochs=600,
    batch_size=256,
    lr=1e-3,
    k_max=5,
    device="cpu"
):
    """
    mini-batch 训练 MLP optimizer。

    相比原始逐样本训练，这里做了两个修正:
        1. 训练样本覆盖 300 帧 rollout 分布。
        2. mini-batch 训练，避免样本数增大后过慢。

    仍然保留 learned optimizer 的 unroll 训练思想:
        每个 batch 从 y_init 出发，执行 K 步。
        每一步计算 energy，backward/step/zero_grad，然后 detach y。
    """
    mlp.train()
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    num_samples = train_samples["y_init"].shape[0]
    train_log = []

    for epoch in range(epochs):
        # K 从 1 逐渐增加到 k_max
        K = min(1 + epoch // max(epochs // k_max, 1), k_max)

        perm = torch.randperm(num_samples, device=device)

        epoch_loss_sum = 0.0
        epoch_res_sum = 0.0
        epoch_step_count = 0

        for start in range(0, num_samples, batch_size):
            idx = perm[start:start + batch_size]

            y = train_samples["y_init"][idx].clone()
            p_batch = train_samples["p_n"][idx]
            v_batch = train_samples["v_n"][idx]
            history_batch = train_samples["history"][idx]
            params_batch = train_samples["params"][idx]

            for _ in range(K):
                delta = mlp(y, history_batch, params_batch)
                y = y + delta

                energy = variational_energy(y, p_batch, v_batch, m=m, g=g, dt=dt)
                loss = energy.mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    r = variational_residual(y, p_batch, v_batch, m=m, g=g, dt=dt)
                    r_norm = torch.norm(r, dim=-1).mean().item()

                y = y.detach()

                epoch_loss_sum += loss.item()
                epoch_res_sum += r_norm
                epoch_step_count += 1

        mean_loss = epoch_loss_sum / max(epoch_step_count, 1)
        mean_res = epoch_res_sum / max(epoch_step_count, 1)

        log_item = {
            "epoch": epoch,
            "K": K,
            "mean_energy": mean_loss,
            "mean_residual_norm": mean_res,
            "num_samples": num_samples
        }
        train_log.append(log_item)

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | K={K:d} | "
                f"mean E={mean_loss:.6e} | "
                f"mean residual={mean_res:.6e}"
            )

    mlp.eval()
    return train_log


def evaluate_single_step_convergence(
    mlp,
    p_n,
    v_n,
    params,
    m=1.0,
    g=9.8,
    dt=0.01,
    max_steps=15
):
    """
    在单个时间步上比较 MLP optimizer 与 Newton 的 residual / energy 收敛。
    """
    y_star = implicit_euler_exact_solution(p_n, v_n, g=g, dt=dt)
    E_star = variational_energy(y_star, p_n, v_n, m=m, g=g, dt=dt).item()

    y_mlp = p_n.clone()
    y_new = p_n.clone()

    mlp_hist = []
    newton_hist = []

    for step in range(max_steps + 1):
        e_m = variational_energy(y_mlp, p_n, v_n, m=m, g=g, dt=dt).item()
        r_m = residual_norm(y_mlp, p_n, v_n, m=m, g=g, dt=dt)

        e_n = variational_energy(y_new, p_n, v_n, m=m, g=g, dt=dt).item()
        r_n = residual_norm(y_new, p_n, v_n, m=m, g=g, dt=dt)

        mlp_hist.append({
            "step": step,
            "y": y_mlp.tolist(),
            "loss": e_m,
            "energy_gap": e_m - E_star,
            "residual_norm": r_m
        })

        newton_hist.append({
            "step": step,
            "y": y_new.tolist(),
            "loss": e_n,
            "energy_gap": e_n - E_star,
            "residual_norm": r_n
        })

        if step < max_steps:
            with torch.no_grad():
                d_m = mlp(y_mlp, torch.cat([p_n, v_n]), params)
            y_mlp = y_mlp + d_m

            d_n = newton_direction(y_new, p_n, v_n, m=m, g=g, dt=dt)
            y_new = y_new + d_n

    return {
        "E_star": E_star,
        "y_star": y_star.tolist(),
        "mlp": mlp_hist,
        "newton": newton_hist
    }


def plot_training_and_single_step_report(train_log, single_step_report, output_path="optimization_report.png"):
    epochs = [x["epoch"] for x in train_log]
    train_res = [max(x["mean_residual_norm"], 1e-12) for x in train_log]
    train_energy = [x["mean_energy"] for x in train_log]

    mlp_hist = single_step_report["mlp"]
    newton_hist = single_step_report["newton"]

    steps = [x["step"] for x in mlp_hist]

    mlp_gap = [max(x["energy_gap"], 1e-12) for x in mlp_hist]
    newton_gap = [max(x["energy_gap"], 1e-12) for x in newton_hist]

    mlp_res = [max(x["residual_norm"], 1e-12) for x in mlp_hist]
    newton_res = [max(x["residual_norm"], 1e-12) for x in newton_hist]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, train_energy)
    axes[0, 0].set_title("Training Mean Energy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Mean Energy")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_res)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Training Mean Residual Norm")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel(r"Mean ||grad E(y)||_2")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, mlp_gap, label="MLP Optimizer", marker="o")
    axes[1, 0].plot(steps, newton_gap, label="Newton Method", marker="s", linestyle="--")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Single-step Energy Gap")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("E - E*")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, mlp_res, label="MLP Optimizer", marker="o")
    axes[1, 1].plot(steps, newton_res, label="Newton Method", marker="s", linestyle="--")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Single-step Residual Norm")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel(r"||grad E(y)||_2")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_diagnostics(sequence_report, output_path="free_fall_rollout_diagnostics.png"):
    newton_positions = np.array(sequence_report["newton"]["positions"], dtype=np.float64)
    mlp_positions = np.array(sequence_report["mlp"]["positions"], dtype=np.float64)

    frames = np.arange(newton_positions.shape[0])
    t = frames * sequence_report["config"]["dt"]

    err = np.array(sequence_report["trajectory_error_norm"], dtype=np.float64)
    err_frames = np.arange(1, len(err) + 1)

    mlp_final_res = np.array([s["residual_final"] for s in sequence_report["mlp"]["step_info"]], dtype=np.float64)
    newton_final_res = np.array([s["residual_final"] for s in sequence_report["newton"]["step_info"]], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(t, newton_positions[:, 2], label="Newton z")
    axes[0, 0].plot(t, mlp_positions[:, 2], label="MLP z", linestyle="--")
    axes[0, 0].set_title("Height over Time")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("z")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(err_frames, np.maximum(err, 1e-12))
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Trajectory Error ||p_mlp - p_newton||")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Error")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(err_frames, np.maximum(mlp_final_res, 1e-12), label="MLP")
    axes[1, 0].plot(err_frames, np.maximum(newton_final_res, 1e-12), label="Newton", linestyle="--")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Per-frame Final Residual")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel(r"Final ||grad E(y)||_2")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    mlp_iters = np.array([s["num_iters"] for s in sequence_report["mlp"]["step_info"]], dtype=np.float64)
    newton_iters = np.array([s["num_iters"] for s in sequence_report["newton"]["step_info"]], dtype=np.float64)

    axes[1, 1].plot(err_frames, mlp_iters, label="MLP")
    axes[1, 1].plot(err_frames, newton_iters, label="Newton", linestyle="--")
    axes[1, 1].set_title("Per-frame Solver Iterations")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Iterations")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. 主程序
# ============================================================
def main():
    torch.manual_seed(42)

    device = torch.device("cpu")

    # ----------------- 物理参数 -----------------
    m, g, dt = 1.0, 9.8, 0.01

    # 初值: 当前状态 (p_n, v_n)
    p_n = torch.tensor([3.0, 4.0, 5.0], device=device)
    v_n = torch.tensor([0.5, -0.5, 0.0], device=device)

    y0 = p_n.clone()
    params = torch.tensor([m, g, dt], device=device)

    # ----------------- 运动序列设置 -----------------
    num_frames = 300
    residual_drop = 1e-3
    max_solver_iters = 5
    fps = 30

    # ----------------- 训练设置 -----------------
    epochs = 600
    batch_size = 256
    lr = 1e-3
    k_max = 5

    print("🚀 开始生成覆盖 300 帧自由落体 rollout 分布的训练集")
    train_samples = make_rollout_training_samples(
        p0=p_n,
        v0=v_n,
        params=params,
        m=m,
        g=g,
        dt=dt,
        num_frames=num_frames,
        num_line_points=11,
        num_local_points=8,
        local_std_dt_units=1.0,
        velocity_jitter_std=0.0,
        position_jitter_std=0.0,
        seed=123
    )

    num_train_samples = train_samples["y_init"].shape[0]
    print(f"训练样本数: {num_train_samples}")

    input_mean, input_std = compute_input_normalizer_from_samples(train_samples)

    mlp = MLPOptimizer(
        input_mean=input_mean,
        input_std=input_std,
        hidden_dim=64
    ).to(device)

    print("\n🚀 开始训练 MLP optimizer")
    print(f"初始状态: p0={p_n.tolist()}, v0={v_n.tolist()}")
    print(f"dt={dt}, frames={num_frames}, residual_drop={residual_drop}, max_solver_iters={max_solver_iters}")
    print("关键修正: 训练数据覆盖整个 300 帧 rollout 分布，而不是只覆盖第一帧。\n")

    train_log = train_mlp_optimizer(
        mlp=mlp,
        train_samples=train_samples,
        m=m,
        g=g,
        dt=dt,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        k_max=k_max,
        device=device
    )

    print("\n✅ 训练完成。开始单步收敛对比评估...\n")

    # ----------------- 单步收敛评估 -----------------
    single_step_report = evaluate_single_step_convergence(
        mlp=mlp,
        p_n=p_n,
        v_n=v_n,
        params=params,
        m=m,
        g=g,
        dt=dt,
        max_steps=15
    )

    print("📊 单步迭代结果对比 (前5步):")
    print(f"{'Step':<5} | {'MLP Gap':<12} | {'MLP Res':<12} | {'Newton Gap':<12} | {'Newton Res':<12}")
    print("-" * 70)
    for i in range(5):
        mlp_it = single_step_report["mlp"][i]
        newton_it = single_step_report["newton"][i]
        print(
            f"{i:<5} | "
            f"{mlp_it['energy_gap']:<12.4e} | "
            f"{mlp_it['residual_norm']:<12.4e} | "
            f"{newton_it['energy_gap']:<12.4e} | "
            f"{newton_it['residual_norm']:<12.4e}"
        )

    plot_training_and_single_step_report(
        train_log=train_log,
        single_step_report=single_step_report,
        output_path="optimization_report.png"
    )
    print("🖼️  单步优化诊断图已保存至: optimization_report.png")

    # ----------------- 保存优化报告 -----------------
    optimization_report = {
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "k_max": k_max,
            "p0": p_n.tolist(),
            "v0": v_n.tolist(),
            "m": m,
            "g": g,
            "dt": dt,
            "num_frames_for_training_distribution": num_frames,
            "normalization": {
                "input_mean": input_mean.tolist(),
                "input_std": input_std.tolist()
            },
            "training_set_expansion": {
                "num_line_points": 11,
                "num_local_points": 8,
                "local_std_dt_units": 1.0,
                "num_train_samples": num_train_samples
            }
        },
        "training_log": train_log,
        "single_step_report": single_step_report
    }

    with open("optimization_report.json", "w", encoding="utf-8") as f:
        json.dump(optimization_report, f, indent=2, ensure_ascii=False)
    print("📁 优化数值报告已保存至: optimization_report.json")

    # ----------------- 300 帧自由落体 rollout -----------------
    print("\n🎥 开始滚动计算 300 帧自由落体运动序列...")

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
        max_iters=max_solver_iters,
        print_debug=True
    )

    with open("free_fall_sequence_report.json", "w", encoding="utf-8") as f:
        json.dump(sequence_report, f, indent=2, ensure_ascii=False)
    print("📁 运动序列数值结果已保存至: free_fall_sequence_report.json")

    plot_rollout_diagnostics(
        sequence_report,
        output_path="free_fall_rollout_diagnostics.png"
    )
    print("🖼️  运动序列诊断图已保存至: free_fall_rollout_diagnostics.png")

    # ----------------- 打印统计 -----------------
    newton_iters = [s["num_iters"] for s in sequence_report["newton"]["step_info"]]
    mlp_iters = [s["num_iters"] for s in sequence_report["mlp"]["step_info"]]

    newton_final_res = [s["residual_final"] for s in sequence_report["newton"]["step_info"]]
    mlp_final_res = [s["residual_final"] for s in sequence_report["mlp"]["step_info"]]

    traj_err = sequence_report["trajectory_error_norm"]

    print("\n📊 300 帧单步求解统计:")
    print(f"Newton 平均迭代次数: {np.mean(newton_iters):.2f}")
    print(f"MLP    平均迭代次数: {np.mean(mlp_iters):.2f}")
    print(f"Newton 最后一帧 residual: {newton_final_res[-1]:.4e}")
    print(f"MLP    最后一帧 residual: {mlp_final_res[-1]:.4e}")
    print(f"MLP vs Newton 最后一帧位置误差: {traj_err[-1]:.4e}")
    print(f"MLP vs Newton 最大位置误差: {np.max(traj_err):.4e}")

    # ----------------- 渲染视频 -----------------
    render_free_fall_comparison_video(
        sequence_report,
        output_path="free_fall_newton_vs_mlp.mp4",
        fps=fps,
        same_axes=True
    )

    print("=" * 60)
    print("✅ 全部完成")
    print("输出文件:")
    print("  - optimization_report.json")
    print("  - optimization_report.png")
    print("  - free_fall_sequence_report.json")
    print("  - free_fall_rollout_diagnostics.png")
    print("  - free_fall_newton_vs_mlp.mp4 或 free_fall_newton_vs_mlp.gif")
    print("=" * 60)


if __name__ == "__main__":
    main()
