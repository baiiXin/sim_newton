import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

# ================= 1. 模型与隐式欧拉变分能量 =================
class MLPOptimizer(nn.Module):
    """
    输入: 仅保留优化变量 y (3D)
    输出: 位置更新步长 delta_y (3D)
    """
    def __init__(self):
        super().__init__()
        # 输入维度从 12 改为 3，移除 history 和 params
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, y):
        # 仅对变量 y 进行前向传播
        return self.net(y)

def variational_energy(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """隐式欧拉变分能量"""
    residual = y - p_n - dt * v_n
    kinetic_term = (m / (2 * dt**2)) * torch.sum(residual**2)
    potential_term = m * g * y[2]
    return kinetic_term + potential_term

def newton_direction(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """牛顿法方向: -H^{-1} * grad"""
    residual = y - p_n - dt * v_n
    grad = (m / dt**2) * residual
    grad = grad.clone()  # 避免 inplace 修改警告
    grad[2] += m * g
    hess_inv = (dt**2) / m
    return -grad * hess_inv

# ================= 2. 主程序 =================
def main():
    torch.manual_seed(42)
    m, g, dt = 1.0, 9.8, 0.01
    p_n = torch.tensor([3., 4., 5.])
    v_n = torch.tensor([0.5, -0.5, 0.0])
    y0 = p_n.clone()

    # 常量仍用于能量计算与牛顿法对比，但不再传入网络
    mlp = MLPOptimizer()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    epochs = 1000
    K = 1
    train_log = []
    eval_log = []

    y_star = p_n + dt * v_n - dt**2 * torch.tensor([0., 0., g])
    E_star = variational_energy(y_star, p_n, v_n, m, g, dt).item()

    print("🚀 开始训练 (隐式欧拉变分能量最小化)")
    print(f"当前状态: p_n={p_n.tolist()}, v_n={v_n.tolist()}")
    print(f"理论最优: y*={y_star.tolist()}, E*={E_star:.6f}")
    print("策略: 网络输入仅含变量 y，常量已移除\n")

    for epoch in range(epochs):
        if epoch > 0 and epoch % 100 == 0 and K < 10:
            K += 1

        y = y0.clone()
        for k in range(K):
            delta = mlp(y)  # 🔹 仅传入变量 y
            y = y + delta
            loss = variational_energy(y, p_n, v_n, m, g, dt)
            loss.backward()
            opt.step()
            opt.zero_grad()
            y = y.detach()

        train_log.append({"epoch": epoch, "K": K, "final_loss": loss.item()})

        if epoch % 100 == 0 or epoch == epochs - 1:
            y_eval = y0.clone()
            eval_steps = []
            for i in range(10):
                with torch.no_grad():
                    d = mlp(y_eval)  # 🔹 仅传入变量 y
                y_eval = y_eval + d
                e_val = variational_energy(y_eval, p_n, v_n, m, g, dt).item()
                eval_steps.append({"step": i+1, "y": y_eval.tolist(), "loss": e_val})

            eval_log.append({"epoch": epoch, "K": K, "steps": eval_steps})
            gap = eval_steps[-1]['loss'] - E_star
            print(f"Epoch {epoch:3d} | K={K:2d} | Eval Gap(10步): {gap:.4e}")

    print("\n✅ 训练完成。开始最终对比评估...\n")

    # ================= 3. 最终对比评估 =================
    max_steps = 15
    mlp_hist = {"init_y": y0.tolist(), "E_star": E_star, "iterations": []}
    newton_hist = {"init_y": y0.tolist(), "E_star": E_star, "iterations": []}

    E0 = variational_energy(y0, p_n, v_n, m, g, dt).item()
    mlp_hist["iterations"].append({"step": 0, "y": y0.tolist(), "loss": E0})
    newton_hist["iterations"].append({"step": 0, "y": y0.tolist(), "loss": E0})

    y_mlp, y_new = y0.clone(), y0.clone()
    mlp_losses = [E0]
    newton_losses = [E0]

    for i in range(max_steps):
        with torch.no_grad():
            d_m = mlp(y_mlp)  # 🔹 仅传入变量 y
        y_mlp = y_mlp + d_m
        e_m = variational_energy(y_mlp, p_n, v_n, m, g, dt).item()
        mlp_hist["iterations"].append({"step": i+1, "y": y_mlp.tolist(), "loss": e_m})
        mlp_losses.append(e_m)

        d_n = newton_direction(y_new, p_n, v_n, m, g, dt)
        y_new = y_new + d_n
        e_n = variational_energy(y_new, p_n, v_n, m, g, dt).item()
        newton_hist["iterations"].append({"step": i+1, "y": y_new.tolist(), "loss": e_n})
        newton_losses.append(e_n)

    print("📊 最终迭代结果对比 (前5步):")
    print(f"{'Step':<5} | {'MLP Loss':<12} | {'MLP y':<25} | {'Newton Gap':<12}")
    print("-" * 70)
    for i in range(min(5, max_steps+1)):
        mlp_it = mlp_hist["iterations"][i]
        y_str = str([round(v, 4) for v in mlp_it['y']])
        gap = mlp_it['loss'] - E_star
        print(f"{mlp_it['step']:<5} | {mlp_it['loss']:<12.6f} | {y_str:<25} | {gap:<12.4e}")

    # ================= 4. 保存结果 =================
    report = {
        "config": {"epochs": epochs, "y0": y0.tolist(), "p_n": p_n.tolist(),
                   "v_n": v_n.tolist(), "m": m, "g": g, "dt": dt, "E_star": E_star},
        "training_log": train_log,
        "periodic_evaluation": eval_log,
        "final_comparison": {"mlp": mlp_hist, "newton": newton_hist}
    }
    with open("optimization_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n📁 数值结果已保存至: optimization_report.json")

    # ================= 5. 绘图 =================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    gap_mlp = [max(l - E_star, 1e-12) for l in mlp_losses]
    gap_newton = [max(l - E_star, 1e-12) for l in newton_losses]
    train_gap = [max(e["final_loss"] - E_star, 1e-12) for e in train_log]
    eval_gap = [max(e["steps"][-1]["loss"] - E_star, 1e-12) for e in eval_log]

    axes[0,0].plot([e["epoch"] for e in train_log], train_gap, color='steelblue')
    axes[0,0].set_yscale('log')
    axes[0,0].set_title('Training Convergence Gap (E - E*)')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Gap')
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot([e["epoch"] for e in eval_log], eval_gap, marker='o', color='darkgreen')
    axes[0,1].set_yscale('log')
    axes[0,1].set_title('Periodic Eval Gap (10 steps)')
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Gap')
    axes[0,1].grid(True, alpha=0.3)

    steps = np.arange(max_steps + 1)
    axes[1,0].plot(steps, gap_mlp, label='MLP Optimizer', marker='o')
    axes[1,0].plot(steps, gap_newton, label='Newton Method', marker='s', linestyle='--', color='crimson')
    axes[1,0].set_yscale('log')
    axes[1,0].set_title('Final Convergence Comparison (Gap)')
    axes[1,0].set_xlabel('Iteration'); axes[1,0].set_ylabel('Gap')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    mlp_norms, newton_norms = [], []
    y_mlp, y_new = y0.clone(), y0.clone()
    for _ in range(max_steps):
        with torch.no_grad(): d_m = mlp(y_mlp)
        mlp_norms.append(torch.norm(d_m).item())
        y_mlp += d_m
        d_n = newton_direction(y_new, p_n, v_n, m, g, dt)
        newton_norms.append(torch.norm(d_n).item())
        y_new += d_n
    axes[1,1].plot(np.arange(max_steps), mlp_norms, label='MLP ||delta||', marker='^')
    axes[1,1].plot(np.arange(max_steps), newton_norms, label='Newton ||delta||', marker='v', linestyle='--', color='crimson')
    axes[1,1].set_title('Update Step Magnitude')
    axes[1,1].set_xlabel('Iteration'); axes[1,1].set_ylabel('||delta||_2')
    axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimization_report.png', dpi=300, bbox_inches='tight')
    print("🖼️  可视化图表已保存至: optimization_report.png")
    print("="*50)

if __name__ == "__main__":
    main()