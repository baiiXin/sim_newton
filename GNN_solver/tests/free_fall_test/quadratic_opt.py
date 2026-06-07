import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 无显示器环境强制使用非交互后端
import matplotlib.pyplot as plt
import numpy as np
import json

# ================= 1. 模型与目标函数 =================
class MLPOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 输出更新步长 delta
        )
        
    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

def objective(x, t):
    return torch.sum((x - t) ** 2)

def newton_direction(x, t):
    return t - x  # 理论最优步长

# ================= 2. 主程序 =================
def main():
    torch.manual_seed(42)
    
    # 固定初值与目标点
    x0 = torch.tensor([3., 4., 5.])
    t0 = torch.tensor([5., 4., 3.])
    
    mlp = MLPOptimizer()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    
    # 训练配置
    epochs = 1000          # epoch取长一点
    K = 1                 # 初始迭代次数
    train_log = []        # 记录每个epoch的最终Loss
    eval_log = []         # 每100 epoch的评估记录
    
    print("🚀 开始训练 (固定初值 x=[3,4,5], 目标 t=[0,0,0])")
    print("策略: 每100 epoch K+=1, 步间 detach, 单步反向传播\n")
    
    for epoch in range(epochs):
        # 每100个epoch增加一次迭代展开次数
        if epoch > 0 and epoch % 200 == 0 and K < 10:
            K += 1
            
        x = x0.clone()  # 每个epoch从固定初值重新开始，学习稳定轨迹
        
        # === 严格遵循您的伪代码 ===
        for k in range(K):
            delta = mlp(x, t0)           # 前向：网络预测步长
            x = x + delta                # 应用更新
            loss = objective(x, t0)      # 计算当前Loss
            loss.backward()              # 梯度回传至MLP参数
            opt.step()                   # 更新参数
            opt.zero_grad()              # 清空梯度(Python伪代码省略但PyTorch必需)
            x = x.detach()               # 切断历史计算图，防止图爆炸
        # ============================
            
        train_log.append({"epoch": epoch, "K": K, "final_loss": loss.item()})
        
        # 每100个epoch评估一次
        if epoch % 100 == 0 or epoch == epochs - 1:
            x_eval = x0.clone()
            eval_steps = []
            for i in range(10):
                with torch.no_grad():
                    d = mlp(x_eval, t0)
                x_eval = x_eval + d
                l = objective(x_eval, t0).item()
                eval_steps.append({"step": i+1, "x": x_eval.tolist(), "loss": l})
                
            eval_log.append({"epoch": epoch, "K": K, "steps": eval_steps})
            print(f"Epoch {epoch:3d} | K={K:2d} | Eval Loss(10步): {eval_steps[-1]['loss']:.4e}")
            
    print("\n✅ 训练完成。开始最终对比评估...\n")
    
    # ================= 3. 最终对比评估 =================
    max_steps = 15
    mlp_hist = {"init_x": x0.tolist(), "target": t0.tolist(), "iterations": []}
    newton_hist = {"init_x": x0.tolist(), "target": t0.tolist(), "iterations": []}
    
    # 记录初始状态 (Step 0)
    mlp_hist["iterations"].append({"step": 0, "x": x0.tolist(), "loss": objective(x0, t0).item()})
    newton_hist["iterations"].append({"step": 0, "x": x0.tolist(), "loss": objective(x0, t0).item()})
    
    x_mlp = x0.clone()
    x_new = x0.clone()
    mlp_losses_plot = [objective(x_mlp, t0).item()]
    newton_losses_plot = [objective(x_new, t0).item()]
    
    for i in range(max_steps):
        # MLP 迭代
        with torch.no_grad():
            d_m = mlp(x_mlp, t0)
        x_mlp = x_mlp + d_m
        l_m = objective(x_mlp, t0).item()
        mlp_hist["iterations"].append({"step": i+1, "x": x_mlp.tolist(), "loss": l_m})
        mlp_losses_plot.append(l_m)
        
        # 牛顿法迭代
        d_n = newton_direction(x_new, t0)
        x_new = x_new + d_n
        l_n = objective(x_new, t0).item()
        newton_hist["iterations"].append({"step": i+1, "x": x_new.tolist(), "loss": l_n})
        newton_losses_plot.append(l_n)
        
    # 打印最终15步详细结果
    print("📊 最终迭代结果对比 (前5步):")
    print(f"{'Step':<5} | {'MLP Loss':<15} | {'MLP x':<25} | {'Newton Loss':<15} | {'Newton x':<25}")
    print("-" * 95)
    for i in range(min(5, max_steps+1)):
        mlp_it = mlp_hist["iterations"][i]
        newton_it = newton_hist["iterations"][i]
        print(f"{mlp_it['step']:<5} | {mlp_it['loss']:<15.4e} | {str([round(v,3) for v in mlp_it['x']]):<25} | {newton_it['loss']:<15.4e} | {str([round(v,3) for v in newton_it['x']]):<25}")
        
    # ================= 4. 保存结果 =================
    report = {
        "config": {"epochs": epochs, "x0": x0.tolist(), "t0": t0.tolist()},
        "training_log": train_log,
        "periodic_evaluation": eval_log,
        "final_comparison": {"mlp": mlp_hist, "newton": newton_hist}
    }
    with open("optimization_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n📁 数值结果已保存至: optimization_report.json")
    
    # ================= 5. 绘图 =================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 训练Loss曲线
    epochs_arr = [e["epoch"] for e in train_log]
    loss_arr = [max(e["final_loss"], 1e-9) for e in train_log]
    axes[0,0].plot(epochs_arr, loss_arr, color='steelblue')
    axes[0,0].set_yscale('log')
    axes[0,0].set_title('Training Loss per Epoch')
    axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3)
    
    # 图2: 周期评估Loss (每次评估的第10步)
    eval_ep = [e["epoch"] for e in eval_log]
    eval_ls = [max(e["steps"][-1]["loss"], 1e-9) for e in eval_log]
    axes[0,1].plot(eval_ep, eval_ls, marker='o', color='darkgreen')
    axes[0,1].set_yscale('log')
    axes[0,1].set_title('Periodic Eval Loss (10 steps)')
    axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Loss')
    axes[0,1].grid(True, alpha=0.3)
    
    # 图3: 最终收敛对比 (核心图)
    steps = np.arange(max_steps + 1)
    axes[1,0].plot(steps, [max(l, 1e-9) for l in mlp_losses_plot], label='MLP Optimizer', marker='o')
    axes[1,0].plot(steps, [max(l, 1e-9) for l in newton_losses_plot], label='Newton Method', marker='s', linestyle='--', color='crimson')
    axes[1,0].set_yscale('log')
    axes[1,0].set_title('Final Convergence Comparison')
    axes[1,0].set_xlabel('Iteration'); axes[1,0].set_ylabel('Loss')
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
    
    # 图4: 更新步长范数对比
    mlp_norms, newton_norms = [], []
    x_mlp, x_new = x0.clone(), x0.clone()
    for _ in range(max_steps):
        with torch.no_grad(): d_m = mlp(x_mlp, t0)
        mlp_norms.append(torch.norm(d_m).item())
        x_mlp += d_m
        d_n = t0 - x_new
        newton_norms.append(torch.norm(d_n).item())
        x_new += d_n
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