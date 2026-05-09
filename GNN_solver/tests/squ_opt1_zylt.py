import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 适配无显示器 Linux
import matplotlib.pyplot as plt
import numpy as np
import json

# ================= 1. 模型与隐式欧拉变分能量 =================
class MLPOptimizer(nn.Module):
    """
    输入: 当前优化变量 y(3) + 历史状态[p_n(3), v_n(3)] + 物理参数[m, g, dt](3)
    输出: 位置更新步长 delta_y(3)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, y, history, params):
        # history = [p_n, v_n] (6D), params = [m, g, dt] (3D)
        inp = torch.cat([y, history, params], dim=-1)
        return self.net(inp)

def variational_energy(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    隐式欧拉变分能量:
    E(y) = (m/(2*dt^2)) * ||y - p_n - dt*v_n||^2 + m*g*y_z
    """
    residual = y - p_n - dt * v_n
    kinetic_term = (m / (2 * dt**2)) * torch.sum(residual**2)
    potential_term = m * g * y[2]  # y_z is the 3rd component
    return kinetic_term + potential_term

def newton_direction(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):
    """
    牛顿法方向: -H^{-1} * grad
    H = (m/dt^2) * I,  grad = (m/dt^2)*(y - p_n - dt*v_n) + [0,0,mg]^T
    """
    residual = y - p_n - dt * v_n
    grad = (m / dt**2) * residual
    grad[2] += m * g  # add gravity term to z-component
    hess_inv = (dt**2) / m  # scalar since Hessian is isotropic
    return -grad * hess_inv

# ================= 2. 主程序 =================
def main():
    torch.manual_seed(42)
    m, g, dt = 1.0, 9.8, 0.01
    
    # 固定初值: 当前状态 (p_n, v_n)
    p_n = torch.tensor([3., 4., 5.])
    v_n = torch.tensor([0.5, -0.5, 0.0])  # 小初速度
    
    # 优化变量初值: y_0 = p_n (从当前位置开始优化下一位置)
    y0 = p_n.clone()
    
    # 网络输入拼接: history=[p_n, v_n], params=[m, g, dt]
    history = torch.cat([p_n, v_n])  # 6D
    params = torch.tensor([m, g, dt])  # 3D
    
    mlp = MLPOptimizer()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
    epochs = 800
    K = 1
    train_log = []
    eval_log = []
    
    # 理论最优解与最优能量
    y_star = p_n + dt * v_n - dt**2 * torch.tensor([0., 0., g])
    E_star = variational_energy(y_star, p_n, v_n, m, g, dt).item()
    
    print("🚀 开始训练 (隐式欧拉变分能量最小化)")
    print(f"当前状态: p_n={p_n.tolist()}, v_n={v_n.tolist()}")
    print(f"理论最优: y*={y_star.tolist()}, E*={E_star:.6f}")
    print("策略: 每100 epoch K+=1, 步间 detach, 单步反向传播\n")
    
    for epoch in range(epochs):
        if epoch > 0 and epoch % 100 == 0:
            K += 1
            
        y = y0.clone()  # 每个epoch从固定初值重置
        
        # === 严格遵循您的伪代码 ===
        for k in range(K):
            delta = mlp(y, history, params)  # 前向：网络预测位置更新
            y = y + delta                     # 应用更新
            loss = variational_energy(y, p_n, v_n, m, g, dt)  # 计算变分能量
            loss.backward()                   # 梯度回传
            opt.step()                        # 更新参数
            opt.zero_grad()                   # 清空梯度
            y = y.detach()                    # 切断历史计算图
        # ============================
            
        train_log.append({"epoch": epoch, "K": K, "final_loss": loss.item()})
        
        # 每100个epoch评估一次
        if epoch % 100 == 0 or epoch == epochs - 1:
            y_eval = y0.clone()
            eval_steps = []
            for i in range(10):
                with torch.no_grad():
                    d = mlp(y_eval, history, params)
                y_eval = y_eval + d
                e_val = variational_energy(y_eval, p_n, v_n, m, g, dt).item()
                eval_steps.append({"step": i+1, "y": y_eval.tolist(), "loss": e_val})
                
            eval_log.append({"epoch": epoch, "K": K, "steps": eval_steps})
            gap = eval_steps[-1]['loss'] - E_star
            print(f"Epoch {epoch:3d} | K={K:2d} | Eval Gap(10步): {gap:.4e}")
            
    print("\n✅ 训练完成。开始最终对比评估...\n")
    
    # ================= 3. 最终对比评估 =================
    max_steps = 15
    mlp_hist = {"init_y": y0.tolist(), "history": [p_n.tolist(), v_n.tolist()], 
                "params": params.tolist(), "E_star": E_star, "iterations": []}
    newton_hist = {"init_y": y0.tolist(), "history": [p_n.tolist(), v_n.tolist()], 
                   "params": params.tolist(), "E_star": E_star, "iterations": []}
    
    # 记录初始状态 (Step 0)
    E0 = variational_energy(y0, p_n, v_n, m, g, dt).item()
    mlp_hist["iterations"].append({"step": 0, "y": y0.tolist(), "loss": E0})
    newton_hist["iterations"].append({"step": 0, "y": y0.tolist(), "loss": E0})
    
    y_mlp, y_new = y0.clone(), y0.clone()
    mlp_losses = [E0]
    newton_losses = [E0]
    
    for i in range(max_steps):
        # MLP 迭代
        with torch.no_grad():
            d_m = mlp(y_mlp, history, params)
        y_mlp = y_mlp + d_m
        e_m = variational_energy(y_mlp, p_n, v_n, m, g, dt).item()
        mlp_hist["iterations"].append({"step": i+1, "y": y_mlp.tolist(), "loss": e_m})
        mlp_losses.append(e_m)
        
        # 牛顿法迭代
        d_n = newton_direction(y_new, p_n, v_n, m, g, dt)
        y_new = y_new + d_n
        e_n = variational_energy(y_new, p_n, v_n, m, g, dt).item()
        newton_hist["iterations"].append({"step": i+1, "y": y_new.tolist(), "loss": e_n})
        newton_losses.append(e_n)
        
    # 打印前5步
    print("📊 最终迭代结果对比 (前5步):")
    print(f"{'Step':<5} | {'MLP Loss':<12} | {'MLP y':<25} | {'Newton Gap':<12}")
    print("-" * 70)
    for i in range(min(5, max_steps+1)):
        mlp_it = mlp_hist["iterations"][i]
        newton_it = newton_hist["iterations"][i]
        y_str = str([round(v,4) for v in mlp_it['y']])
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
    
    # 绘制 Gap = Loss - E_star 以适配对数坐标（避免负数/零值）
    gap_mlp = [max(l - E_star, 1e-12) for l in mlp_losses]
    gap_newton = [max(l - E_star, 1e-12) for l in newton_losses]
    train_gap = [max(e["final_loss"] - E_star, 1e-12) for e in train_log]
    eval_gap = [max(e["steps"][-1]["loss"] - E_star, 1e-12) for e in eval_log]
    
    # 图1: 训练Gap曲线
    axes[0,0].plot([e["epoch"] for e in train_log], train_gap, color='steelblue')
    axes[0,0].set_yscale('log')
    axes[0,0].set_title('Training Convergence Gap (E - E*)')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Gap')
    axes[0,0].grid(True, alpha=0.3)
    
    # 图2: 周期评估Gap
    axes[0,1].plot([e["epoch"] for e in eval_log], eval_gap, marker='o', color='darkgreen')
    axes[0,1].set_yscale('log')
    axes[0,1].set_title('Periodic Eval Gap (10 steps)')
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Gap')
    axes[0,1].grid(True, alpha=0.3)
    
    # 图3: 最终收敛对比（核心图）
    steps = np.arange(max_steps + 1)
    axes[1,0].plot(steps, gap_mlp, label='MLP Optimizer', marker='o')
    axes[1,0].plot(steps, gap_newton, label='Newton Method', marker='s', linestyle='--', color='crimson')
    axes[1,0].set_yscale('log')
    axes[1,0].set_title('Final Convergence Comparison (Gap)')
    axes[1,0].set_xlabel('Iteration'); axes[1,0].set_ylabel('Gap')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
    
    # 图4: 更新步长范数对比
    mlp_norms, newton_norms = [], []
    y_mlp, y_new = y0.clone(), y0.clone()
    for _ in range(max_steps):
        with torch.no_grad(): d_m = mlp(y_mlp, history, params)
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