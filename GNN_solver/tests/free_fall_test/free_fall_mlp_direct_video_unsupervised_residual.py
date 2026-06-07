import json
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # 适配无显示器 Linux
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# 1. 模型、能量、residual、Newton
# ============================================================
class MLPDirect(nn.Module):
    """
    非迭代版 Learned predictor:
        input  = history[p_n(3), v_n(3)] + params[m, g, dt](3)
        output = delta(3),  最终 y_pred = p_n + delta

    与 MLPOptimizer 的区别:
        - 不再以当前 iterate y 作为输入。
        - 不做任何迭代/line search/trust region:
          每个时间步只做 ONE forward pass 即给出预测位置。

    依旧是无监督训练: 只最小化 y_pred 处的变分一阶 residual,
    没有 target_delta, 没有 y_star label。

    输出做了 soft bound, 防止网络早期或 rollout 输出极端步长:
        delta = dt * raw_delta_limit * tanh(raw_delta / raw_delta_limit)
    """
    def __init__(self, input_mean=None, input_std=None, hidden_dim=128, raw_delta_limit=80.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )

        self.raw_delta_limit = raw_delta_limit

        if input_mean is None:
            input_mean = torch.zeros(9)
        if input_std is None:
            input_std = torch.ones(9)

        self.register_buffer("input_mean", input_mean.clone().detach())
        self.register_buffer("input_std", input_std.clone().detach())

    def forward(self, history, params):
        inp = torch.cat([history, params], dim=-1)
        inp = (inp - self.input_mean) / self.input_std

        raw_delta_unbounded = self.net(inp)

        raw_delta = self.raw_delta_limit * torch.tanh(
            raw_delta_unbounded / self.raw_delta_limit
        )

        dt = params[..., 2:3]
        return dt * raw_delta


def gravity_vector_like(y, m=1.0, g=9.8):
    gravity = torch.zeros_like(y)
    gravity[..., 2] = m * g
    return gravity


def variational_energy(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    隐式欧拉变分能量:
        E(y) = m/(2dt^2) ||y - p_n - dt v_n||^2 + m g y_z
    """
    residual = y - p_n - dt * v_n
    kinetic = (m / (2.0 * dt**2)) * torch.sum(residual**2, dim=-1)
    potential = m * g * y[..., 2]
    return kinetic + potential


def variational_residual(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    变分一阶 residual:
        r(y) = grad E(y)
             = (m/dt^2) * (y - p_n - dt*v_n) + [0, 0, mg]^T
    """
    residual = y - p_n - dt * v_n
    return (m / dt**2) * residual + gravity_vector_like(y, m=m, g=g)


def scaled_variational_residual(y, p_n, v_n, g=9.8, dt=0.01):
    """
    无监督训练使用的 scaled residual:
        (dt^2 / m) * grad E(y) = y - p_n - dt*v_n + dt^2*g*e_z
    """
    r = y - p_n - dt * v_n
    r = r.clone()
    r[..., 2] += dt**2 * g
    return r


def residual_norm(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    with torch.no_grad():
        r = variational_residual(y, p_n, v_n, m=m, g=g, dt=dt)
        n = torch.norm(r, dim=-1)
        if n.ndim == 0:
            return n.item()
        return n


def implicit_euler_exact_solution(p_n, v_n, g=9.8, dt=0.01):
    """
    仅用于 Newton 对比和评估, 不参与 MLP 训练标签。
    """
    a = torch.zeros_like(p_n)
    a[..., 2] = g
    return p_n + dt * v_n - dt**2 * a


def newton_direction(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    Newton direction = -H^{-1} grad。该问题 Hessian = m/dt^2 I, 一步到最优。
    """
    grad = variational_residual(y, p_n, v_n, m=m, g=g, dt=dt)
    return -grad * (dt**2 / m)


# ============================================================
# 2. 无监督训练样本 (与迭代版相同, 仅忽略 y_init 字段)
# ============================================================
def make_unsupervised_training_samples(
    p0,
    v0,
    params,
    g=9.8,
    dt=0.01,
    num_frames=300,
    num_random_samples=12000,
    trajectory_anchor_repeats=4,
    y_noise_dt_units=3.0,
    velocity_jitter_std=0.5,
    position_jitter_std=0.05,
    seed=123
):
    """
    生成训练用状态分布。直接版只使用 p_n, v_n, history, params。
    保留与迭代版相同的采样接口便于公平对比, y_init 字段在此版本中不参与训练。
    """
    gen = torch.Generator(device=p0.device)
    gen.manual_seed(seed)

    dtype = p0.dtype
    device = p0.device
    total_time = num_frames * dt

    y_list = []
    p_list = []
    v_list = []
    history_list = []
    params_list = []

    def append_sample(y, p, v):
        y_list.append(y.detach())
        p_list.append(p.detach())
        v_list.append(v.detach())
        history_list.append(torch.cat([p, v]).detach())
        params_list.append(params.detach())

    def state_at_time(t):
        p = p0.clone()
        v = v0.clone()

        p[0] = p0[0] + v0[0] * t
        p[1] = p0[1] + v0[1] * t
        p[2] = p0[2] + v0[2] * t - 0.5 * g * t * t

        v[0] = v0[0]
        v[1] = v0[1]
        v[2] = v0[2] - g * t

        return p, v

    # A. 轨迹锚点
    for frame in range(num_frames):
        t = torch.tensor(frame * dt, dtype=dtype, device=device)
        p, v = state_at_time(t)

        for _ in range(trajectory_anchor_repeats):
            append_sample(p.clone(), p.clone(), v.clone())

            noise = dt * y_noise_dt_units * torch.randn(3, generator=gen, dtype=dtype, device=device)
            append_sample(p + dt * v + noise, p.clone(), v.clone())

            alpha = torch.rand((), generator=gen, dtype=dtype, device=device)
            noise = dt * 0.5 * y_noise_dt_units * torch.randn(3, generator=gen, dtype=dtype, device=device)
            append_sample(p + alpha * dt * v + noise, p.clone(), v.clone())

    # B. 随机状态
    for _ in range(num_random_samples):
        t = total_time * torch.rand((), generator=gen, dtype=dtype, device=device)
        p, v = state_at_time(t)

        p = p + position_jitter_std * torch.randn(3, generator=gen, dtype=dtype, device=device)
        v = v + velocity_jitter_std * torch.randn(3, generator=gen, dtype=dtype, device=device)

        mode = int(torch.randint(0, 4, (), generator=gen, device=device).item())

        if mode == 0:
            y = p.clone()
        elif mode == 1:
            y = p + dt * v
        elif mode == 2:
            alpha = 1.5 * torch.rand((), generator=gen, dtype=dtype, device=device)
            y = p + alpha * dt * v
        else:
            alpha = 1.5 * torch.rand((), generator=gen, dtype=dtype, device=device)
            noise = dt * y_noise_dt_units * torch.randn(3, generator=gen, dtype=dtype, device=device)
            y = p + alpha * dt * v + noise

        y = y + dt * 0.25 * y_noise_dt_units * torch.randn(3, generator=gen, dtype=dtype, device=device)
        append_sample(y, p, v)

    return {
        "y_init": torch.stack(y_list, dim=0),  # 此版本不使用, 仅保留接口
        "p_n": torch.stack(p_list, dim=0),
        "v_n": torch.stack(v_list, dim=0),
        "history": torch.stack(history_list, dim=0),
        "params": torch.stack(params_list, dim=0)
    }


def compute_input_normalizer_direct(samples):
    x = torch.cat([samples["history"], samples["params"]], dim=-1)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


# ============================================================
# 3. 无监督 residual 训练 (单次前向, 不展开)
# ============================================================
def train_mlp_direct_unsupervised_residual(
    mlp,
    samples,
    m=1.0,
    g=9.8,
    dt=0.01,
    epochs=800,
    batch_size=512,
    lr=5e-4,
    delta_reg=1e-5,
    grad_clip=1.0,
    device="cpu"
):
    """
    非迭代直接预测的无监督训练。

    训练过程 (每个 batch 只前向一次):
        y_pred = p_n + MLP(history, params)
        loss   = mean || scaled_residual(y_pred) ||^2
               + delta_reg * mean ||delta||^2

    与迭代版本的区别:
        - 没有 unroll K, 因为网络本身不迭代。
        - 没有 detach 循环, 因为只前向一次。
    """
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-6)

    n = samples["p_n"].shape[0]
    train_log = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)

        loss_sum = 0.0
        scaled_res_sum = 0.0
        grad_res_sum = 0.0
        delta_sum = 0.0
        step_count = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]

            p = samples["p_n"][idx]
            v = samples["v_n"][idx]
            history = samples["history"][idx]
            params = samples["params"][idx]

            delta = mlp(history, params)
            y_pred = p + delta

            scaled_r = scaled_variational_residual(y_pred, p, v, g=g, dt=dt)
            scaled_r_norm = torch.norm(scaled_r, dim=-1)

            loss_res = torch.mean(scaled_r_norm**2)
            loss_delta = torch.mean(torch.sum(delta**2, dim=-1))

            loss = loss_res + delta_reg * loss_delta

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), grad_clip)
            opt.step()

            with torch.no_grad():
                grad_r = variational_residual(y_pred, p, v, m=m, g=g, dt=dt)
                grad_r_norm = torch.norm(grad_r, dim=-1).mean().item()
                d_norm = torch.norm(delta, dim=-1).mean().item()

            loss_sum += loss.item()
            scaled_res_sum += scaled_r_norm.mean().item()
            grad_res_sum += grad_r_norm
            delta_sum += d_norm
            step_count += 1

        log_item = {
            "epoch": epoch,
            "loss": loss_sum / max(step_count, 1),
            "mean_scaled_residual_position": scaled_res_sum / max(step_count, 1),
            "mean_grad_residual_norm": grad_res_sum / max(step_count, 1),
            "mean_delta_norm": delta_sum / max(step_count, 1)
        }
        train_log.append(log_item)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | "
                f"loss={log_item['loss']:.6e} | "
                f"scaled_res={log_item['mean_scaled_residual_position']:.6e} | "
                f"grad_res={log_item['mean_grad_residual_norm']:.6e} | "
                f"delta={log_item['mean_delta_norm']:.6e}"
            )

    return train_log


# ============================================================
# 4. 单步求解器
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
    y = p_n.clone()

    r0 = residual_norm(y, p_n, v_n, m=m, g=g, dt=dt)
    target = max(r0 * residual_drop, abs_tol)

    residual_hist = [r0]
    energy_hist = [variational_energy(y, p_n, v_n, m=m, g=g, dt=dt).item()]
    delta_norm_hist = []

    num_iters = 0

    for it in range(max_iters):
        if residual_hist[-1] <= target:
            break

        d = newton_direction(y, p_n, v_n, m=m, g=g, dt=dt)
        y = y + d

        r = residual_norm(y, p_n, v_n, m=m, g=g, dt=dt)
        e = variational_energy(y, p_n, v_n, m=m, g=g, dt=dt).item()

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


def solve_one_step_mlp_direct(
    mlp,
    p_n,
    v_n,
    params,
    m=1.0,
    g=9.8,
    dt=0.01
):
    """
    非迭代 MLP 单步求解器:
        y_next = p_n + MLP(history, params)

    单次前向, 没有 line search, 没有迭代, 没有 fallback。
    若网络输出非有限值, 则保持 p_n 不变。
    """
    history = torch.cat([p_n, v_n])

    r0 = residual_norm(p_n, p_n, v_n, m=m, g=g, dt=dt)
    e0 = variational_energy(p_n, p_n, v_n, m=m, g=g, dt=dt).item()

    with torch.no_grad():
        d = mlp(history, params)

    finite_pred = torch.all(torch.isfinite(d)).item()
    if not finite_pred:
        print("⚠️ MLPDirect produced non-finite delta; keeping p_n as prediction.")
        d = torch.zeros_like(p_n)

    y = p_n + d
    r_final = residual_norm(y, p_n, v_n, m=m, g=g, dt=dt)
    e_final = variational_energy(y, p_n, v_n, m=m, g=g, dt=dt).item()

    p_next = y.detach()
    v_next = (p_next - p_n) / dt

    return p_next, v_next.detach(), {
        "num_iters": 1,
        "residual_initial": r0,
        "residual_final": r_final,
        "residual_hist": [r0, r_final],
        "energy_hist": [e0, e_final],
        "delta_norm_hist": [torch.norm(d).item()],
        "finite_prediction": finite_pred
    }


# ============================================================
# 5. 评估与 rollout
# ============================================================
def evaluate_single_step_direct(
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
    Direct 网络只做一次前向, 因此 mlp_hist 只有两行:
      step 0: y = p_n (初值)
      step 1: y = p_n + MLP(history, params) (网络一次前向后)

    Newton 仍然展开 max_steps 步, 便于对照其收敛速度。
    """
    y_star = implicit_euler_exact_solution(p_n, v_n, g=g, dt=dt)
    E_star = variational_energy(y_star, p_n, v_n, m=m, g=g, dt=dt).item()

    history = torch.cat([p_n, v_n])

    y_init = p_n.clone()
    with torch.no_grad():
        d = mlp(history, params)
    y_pred = p_n + d

    mlp_hist = [
        {
            "step": 0,
            "y": y_init.tolist(),
            "loss": variational_energy(y_init, p_n, v_n, m=m, g=g, dt=dt).item(),
            "energy_gap": variational_energy(y_init, p_n, v_n, m=m, g=g, dt=dt).item() - E_star,
            "residual_norm": residual_norm(y_init, p_n, v_n, m=m, g=g, dt=dt)
        },
        {
            "step": 1,
            "y": y_pred.tolist(),
            "loss": variational_energy(y_pred, p_n, v_n, m=m, g=g, dt=dt).item(),
            "energy_gap": variational_energy(y_pred, p_n, v_n, m=m, g=g, dt=dt).item() - E_star,
            "residual_norm": residual_norm(y_pred, p_n, v_n, m=m, g=g, dt=dt)
        }
    ]

    y_newton = p_n.clone()
    newton_hist = []
    for step in range(max_steps + 1):
        e_n = variational_energy(y_newton, p_n, v_n, m=m, g=g, dt=dt).item()
        r_n = residual_norm(y_newton, p_n, v_n, m=m, g=g, dt=dt)
        newton_hist.append({
            "step": step,
            "y": y_newton.tolist(),
            "loss": e_n,
            "energy_gap": e_n - E_star,
            "residual_norm": r_n
        })
        if step < max_steps:
            d_n = newton_direction(y_newton, p_n, v_n, m=m, g=g, dt=dt)
            y_newton = y_newton + d_n

    return {
        "E_star": E_star,
        "y_star": y_star.tolist(),
        "mlp": mlp_hist,
        "newton": newton_hist
    }


def rollout_free_fall_sequence_direct(
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
            p_newton, v_newton,
            m=m, g=g, dt=dt,
            residual_drop=residual_drop,
            max_iters=max_iters
        )

        p_mlp, v_mlp, info_mlp = solve_one_step_mlp_direct(
            mlp, p_mlp, v_mlp, params,
            m=m, g=g, dt=dt
        )

        pos_err = torch.norm(p_mlp - p_newton).item()
        trajectory_error.append(pos_err)

        newton_positions.append(p_newton.tolist())
        mlp_positions.append(p_mlp.tolist())

        newton_velocities.append(v_newton.tolist())
        mlp_velocities.append(v_mlp.tolist())

        info_newton["frame"] = frame + 1
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
                f"MLP-direct z={p_mlp[2].item(): .5f}, "
                f"|err|={pos_err:.3e}, "
                f"MLP dp={dp.tolist()}, "
                f"MLP res {info_mlp['residual_initial']:.2e}->{info_mlp['residual_final']:.2e}"
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
# 6. 绘图与视频
# ============================================================
def plot_training_and_single_step_report_direct(train_log, single_step_report, output_path):
    epochs = [x["epoch"] for x in train_log]
    loss = [max(x["loss"], 1e-16) for x in train_log]
    scaled_res = [max(x["mean_scaled_residual_position"], 1e-16) for x in train_log]
    grad_res = [max(x["mean_grad_residual_norm"], 1e-16) for x in train_log]
    delta_norm = [max(x["mean_delta_norm"], 1e-16) for x in train_log]

    mlp_hist = single_step_report["mlp"]
    newton_hist = single_step_report["newton"]

    mlp_steps = [x["step"] for x in mlp_hist]
    newton_steps = [x["step"] for x in newton_hist]
    mlp_gap = [max(x["energy_gap"], 1e-16) for x in mlp_hist]
    newton_gap = [max(x["energy_gap"], 1e-16) for x in newton_hist]
    mlp_res = [max(x["residual_norm"], 1e-16) for x in mlp_hist]
    newton_res = [max(x["residual_norm"], 1e-16) for x in newton_hist]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, loss, label="loss")
    axes[0, 0].plot(epochs, scaled_res, label="scaled residual")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Unsupervised Residual Training (Direct, single-shot)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, grad_res, label="grad residual")
    axes[0, 1].plot(epochs, delta_norm, label="delta norm")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Training Diagnostics")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(mlp_steps, mlp_gap, label="MLP Direct (single shot)", marker="o")
    axes[1, 0].plot(newton_steps, newton_gap, label="Newton Method", marker="s", linestyle="--")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Single-step Energy Gap")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("E - E*")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(mlp_steps, mlp_res, label="MLP Direct (single shot)", marker="o")
    axes[1, 1].plot(newton_steps, newton_res, label="Newton Method", marker="s", linestyle="--")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Single-step Residual Norm")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel(r"||grad E(y)||_2")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_diagnostics(sequence_report, output_path):
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
    axes[0, 0].plot(t, mlp_positions[:, 2], label="MLP-direct z", linestyle="--")
    axes[0, 0].set_title("Height over Time")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("z")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(err_frames, np.maximum(err, 1e-16))
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Trajectory Error ||p_mlp - p_newton||")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Error")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(err_frames, np.maximum(mlp_final_res, 1e-16), label="MLP-direct")
    axes[1, 0].plot(err_frames, np.maximum(newton_final_res, 1e-16), label="Newton", linestyle="--")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Per-frame Final Residual")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel(r"Final ||grad E(y)||_2")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    mlp_iters = np.array([s["num_iters"] for s in sequence_report["mlp"]["step_info"]], dtype=np.float64)
    newton_iters = np.array([s["num_iters"] for s in sequence_report["newton"]["step_info"]], dtype=np.float64)

    axes[1, 1].plot(err_frames, mlp_iters, label="MLP-direct (always 1)")
    axes[1, 1].plot(err_frames, newton_iters, label="Newton", linestyle="--")
    axes[1, 1].set_title("Per-frame Solver Iterations")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Iterations")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_axis_limits(positions):
    finite_mask = np.all(np.isfinite(positions), axis=1)
    if not np.all(finite_mask):
        print("⚠️ 检测到非有限位置值, 渲染时将只使用有限值估计坐标范围。")
        positions = positions[finite_mask]

    if positions.shape[0] == 0:
        raise RuntimeError("所有位置值都是非有限值, 无法渲染。")

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)

    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)
    if span < 1e-6:
        span = 1.0

    half = 0.5 * span + 0.1 * span

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


def render_free_fall_comparison_video(
    sequence_report,
    output_path,
    fps=30,
    same_axes=True
):
    newton_positions = np.array(sequence_report["newton"]["positions"], dtype=np.float64)
    mlp_positions = np.array(sequence_report["mlp"]["positions"], dtype=np.float64)

    num_frames = sequence_report["config"]["num_frames"]
    dt = sequence_report["config"]["dt"]

    if same_axes:
        limits = compute_axis_limits(np.concatenate([newton_positions, mlp_positions], axis=0))
        newton_limits = limits
        mlp_limits = limits
    else:
        newton_limits = compute_axis_limits(newton_positions)
        mlp_limits = compute_axis_limits(mlp_positions)

    fig = plt.figure(figsize=(12, 6))
    ax_newton = fig.add_subplot(1, 2, 1, projection="3d")
    ax_mlp = fig.add_subplot(1, 2, 2, projection="3d")

    setup_3d_axis(ax_newton, "Newton Method", newton_limits)
    setup_3d_axis(ax_mlp, "MLP Direct (single shot)", mlp_limits)

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

        set_3d_line(newton_trail, newton_positions[:idx + 1])
        set_3d_point(newton_point, newton_positions[idx])

        set_3d_line(mlp_trail, mlp_positions[:idx + 1])
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
            f"iters={mlp_info['num_iters']} (forced)\n"
            f"res={mlp_info['residual_final']:.2e}\n"
            f"|err|={pos_err:.2e}"
        )

        return newton_trail, newton_point, newton_text, mlp_trail, mlp_point, mlp_text

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 / fps,
        blit=False
    )

    plt.tight_layout()

    if output_path.endswith(".mp4") and FFMpegWriter.isAvailable():
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(output_path, writer=writer)
        print(f"🎬 视频已保存至: {output_path}")
    else:
        fallback_path = output_path if output_path.endswith(".gif") else output_path.replace(".mp4", ".gif")
        writer = PillowWriter(fps=fps)
        anim.save(fallback_path, writer=writer)
        print(f"🎬 动画已保存至: {fallback_path}")

    plt.close(fig)


# ============================================================
# 7. 主程序
# ============================================================
def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    # 物理参数
    m, g, dt = 1.0, 9.8, 0.01

    # 初始状态
    p0 = torch.tensor([3.0, 4.0, 5.0], device=device)
    v0 = torch.tensor([0.5, -0.5, 0.0], device=device)
    params = torch.tensor([m, g, dt], device=device)

    # rollout 设置
    num_frames = 300
    residual_drop = 1e-3
    max_solver_iters = 5
    fps = 30

    # 训练设置 (与迭代版保持相同的 epoch/批量/lr 以便公平对比)
    epochs = 500
    batch_size = 512
    lr = 5e-4

    print("🚀 生成无监督训练样本 (Direct 版)")
    samples = make_unsupervised_training_samples(
        p0=p0,
        v0=v0,
        params=params,
        g=g,
        dt=dt,
        num_frames=num_frames,
        num_random_samples=12000,
        trajectory_anchor_repeats=4,
        y_noise_dt_units=3.0,
        velocity_jitter_std=0.5,
        position_jitter_std=0.05,
        seed=123
    )

    num_samples = samples["p_n"].shape[0]
    print(f"训练样本数: {num_samples}")

    input_mean, input_std = compute_input_normalizer_direct(samples)

    mlp = MLPDirect(
        input_mean=input_mean,
        input_std=input_std,
        hidden_dim=128,
        raw_delta_limit=80.0
    ).to(device)

    print("\n🚀 开始无监督 residual 训练 MLP Direct (single-shot)")
    print("训练目标: 单次前向后, 最小化预测点的变分一阶 residual。")
    print("没有迭代展开, 没有 target_delta, 没有 y_star label。\n")

    train_log = train_mlp_direct_unsupervised_residual(
        mlp=mlp,
        samples=samples,
        m=m,
        g=g,
        dt=dt,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        delta_reg=1e-5,
        grad_clip=1.0,
        device=device
    )

    print("\n✅ 训练完成。开始单步预测对比评估...\n")

    single_step_report = evaluate_single_step_direct(
        mlp=mlp,
        p_n=p0,
        v_n=v0,
        params=params,
        m=m,
        g=g,
        dt=dt,
        max_steps=15
    )

    print("📊 单步预测结果对比:")
    print(f"{'Step':<5} | {'MLP Gap':<12} | {'MLP Res':<12} | {'Newton Gap':<12} | {'Newton Res':<12}")
    print("-" * 70)
    for i in range(min(5, len(single_step_report["newton"]))):
        if i < len(single_step_report["mlp"]):
            mi = single_step_report["mlp"][i]
            mlp_gap_str = f"{mi['energy_gap']:<12.4e}"
            mlp_res_str = f"{mi['residual_norm']:<12.4e}"
        else:
            mlp_gap_str = f"{'(no iter)':<12}"
            mlp_res_str = f"{'(no iter)':<12}"
        ni = single_step_report["newton"][i]
        print(
            f"{i:<5} | "
            f"{mlp_gap_str} | "
            f"{mlp_res_str} | "
            f"{ni['energy_gap']:<12.4e} | "
            f"{ni['residual_norm']:<12.4e}"
        )

    plot_training_and_single_step_report_direct(
        train_log,
        single_step_report,
        output_path="optimization_report_direct.png"
    )
    print("🖼️  单步预测诊断图已保存至: optimization_report_direct.png")

    optimization_report = {
        "config": {
            "training_mode": "unsupervised_residual_minimization_direct_single_shot",
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "num_samples": num_samples,
            "p0": p0.tolist(),
            "v0": v0.tolist(),
            "m": m,
            "g": g,
            "dt": dt,
            "num_frames": num_frames,
            "normalization": {
                "input_mean": input_mean.tolist(),
                "input_std": input_std.tolist()
            }
        },
        "training_log": train_log,
        "single_step_report": single_step_report
    }

    with open("optimization_report_direct.json", "w", encoding="utf-8") as f:
        json.dump(optimization_report, f, indent=2, ensure_ascii=False)
    print("📁 优化数值报告已保存至: optimization_report_direct.json")

    print("\n🎥 开始滚动计算 300 帧自由落体运动序列 (Direct, single-shot)...")

    sequence_report = rollout_free_fall_sequence_direct(
        mlp=mlp,
        p0=p0,
        v0=v0,
        params=params,
        m=m,
        g=g,
        dt=dt,
        num_frames=num_frames,
        residual_drop=residual_drop,
        max_iters=max_solver_iters,
        print_debug=True
    )

    with open("free_fall_sequence_report_direct.json", "w", encoding="utf-8") as f:
        json.dump(sequence_report, f, indent=2, ensure_ascii=False)
    print("📁 运动序列数值结果已保存至: free_fall_sequence_report_direct.json")

    plot_rollout_diagnostics(
        sequence_report,
        output_path="free_fall_rollout_diagnostics_direct.png"
    )
    print("🖼️  运动序列诊断图已保存至: free_fall_rollout_diagnostics_direct.png")

    newton_iters = [s["num_iters"] for s in sequence_report["newton"]["step_info"]]
    mlp_iters = [s["num_iters"] for s in sequence_report["mlp"]["step_info"]]
    newton_final_res = [s["residual_final"] for s in sequence_report["newton"]["step_info"]]
    mlp_final_res = [s["residual_final"] for s in sequence_report["mlp"]["step_info"]]
    traj_err = sequence_report["trajectory_error_norm"]

    print("\n📊 300 帧单步求解统计:")
    print(f"Newton    平均迭代次数: {np.mean(newton_iters):.2f}")
    print(f"MLPDirect 固定迭代次数: {np.mean(mlp_iters):.2f}  (恒为 1)")
    print(f"Newton    最后一帧 residual: {newton_final_res[-1]:.4e}")
    print(f"MLPDirect 最后一帧 residual: {mlp_final_res[-1]:.4e}")
    print(f"MLPDirect vs Newton 最后一帧位置误差: {traj_err[-1]:.4e}")
    print(f"MLPDirect vs Newton 最大位置误差: {np.max(traj_err):.4e}")

    render_free_fall_comparison_video(
        sequence_report,
        output_path="free_fall_newton_vs_mlp_direct.mp4",
        fps=fps,
        same_axes=True
    )

    print("=" * 60)
    print("✅ 全部完成 (Direct, 非迭代版)")
    print("输出文件:")
    print("  - optimization_report_direct.json")
    print("  - optimization_report_direct.png")
    print("  - free_fall_sequence_report_direct.json")
    print("  - free_fall_rollout_diagnostics_direct.png")
    print("  - free_fall_newton_vs_mlp_direct.mp4 或 .gif")
    print("=" * 60)


if __name__ == "__main__":
    main()
