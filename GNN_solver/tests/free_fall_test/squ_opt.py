import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 适配无显示器 Linux 环境
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# ================= 1. 模型与基础函数 =================
class MLPOptimizer(nn.Module):
    """
    学习型优化器：输入当前点 x(3) 与目标点 target(3)，输出更新步长 delta(3)
    不接收任何梯度/Hessian信息，仅靠坐标与目标位置隐式学习下降方向
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 输出直接作为增量加到 x 上
        )
        
    def forward(self, x, target):
        inp = torch.cat([x, target], dim=-1)  # 拼接坐标与目标点信息
        return self.net(inp)

def objective(x, target):
    """优化目标：最小化到目标点的欧式距离平方 (保持二阶可微且避免奇点)"""
    return torch.sum((x - target) ** 2)

def newton_direction(x, target):
    """牛顿法方向：对于 f(x)=||x-t||^2, H=2I, g=2(x-t), 牛顿步长 = t - x (理论单步收敛)"""
    return target - x

# ================= 2. 训练与评估主流程 =================
def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    mlp = MLPOptimizer()
    meta_opt = torch.optim.Adam(mlp.parameters(), lr=5e-4)
    
    # 训练配置
    epochs = 1000
    unroll_steps = 1  # 初始展开步数
    train_loss_log = []
    eval_log = []
    
    # 固定验证初值与目标点（原点）
    x_val_init = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
    target_val = torch.zeros(3, dtype=torch.float32)
    
    print("="*50)
    print("🚀 开始 Meta-Training MLP Optimizer")
    print(f"初值采样范围: x ~ U(-5, 5)^3, target ~ U(-2, 2)^3")
    print(f"策略: 每100 epoch 增加1次展开步数, 步间 detach 切断梯度")
    print("="*50)
    
    for epoch in range(epochs):
        # 每100个epoch增加一次迭代展开次数
        if epoch > 0 and epoch % 100 == 0:
            unroll_steps += 1
            
        # 随机采样训练样本
        x = torch.rand(3) * 10.0 - 5.0      # [-5, 5]
        target = torch.rand(3) * 4.0 - 2.0  # [-2, 2]
        
        meta_opt.zero_grad()
        
        # 展开迭代轨迹
        for _ in range(unroll_steps):
            delta = mlp(x, target)
            # 关键：不同迭代间 detach 切断计算图，防止图爆炸并模拟在线优化器行为
            x = x + delta  
            loss = objective(x, target)
            loss.backward()  # 每次 epoch 仅反向传播一次
            meta_opt.step()
            train_loss_log.append(loss.item())
            x = x.detach()
        
        # 每100 epoch 评估一次
        if epoch % 100 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                x_eval = x_val_init.clone()
                for _ in range(10):  # 固定迭代10次查看效果
                    x_eval = x_eval + mlp(x_eval, target_val)
                eval_loss = objective(x_eval, target_val).item()
                
            eval_log.append({"epoch": epoch, "unroll": unroll_steps, "loss": eval_loss})
            print(f"Epoch {epoch:3d} | Unroll: {unroll_steps} | "
                  f"Train Loss: {loss.item():.4e} | Eval(10steps) Loss: {eval_loss:.4e}")
            
    print("\n✅ MLP 训练完成。开始与牛顿法对比测试...")
    
    # ================= 3. 固定初值对比实验 =================
    max_steps = 15
    mlp_losses, newton_losses = [], []
    mlp_x, newton_x = x_val_init.clone(), x_val_init.clone()
    
    for i in range(max_steps + 1):
        mlp_losses.append(objective(mlp_x, target_val).item())
        newton_losses.append(objective(newton_x, target_val).item())
        
        if i < max_steps:
            with torch.no_grad():
                mlp_x = mlp_x + mlp(mlp_x, target_val)
                newton_x = newton_x + newton_direction(newton_x, target_val)
                
    # ================= 4. 保存结果到当前目录 =================
    # 保存结构化数据
    report_data = {
        "config": {"epochs": epochs, "init_x_range": [-5, 5], "target_range": [-2, 2]},
        "train_losses": train_loss_log,
        "evaluation": eval_log,
        "comparison": {
            "mlp_losses": mlp_losses,
            "newton_losses": newton_losses,
            "steps": max_steps,
            "init_point": x_val_init.tolist(),
            "target_point": target_val.tolist()
        }
    }
    with open("optimization_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print("📊 数值结果已保存至: optimization_report.json")
    
    # 绘制综合对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 训练Loss
    axes[0, 0].plot(train_loss_log, label='Meta-Training Loss', color='steelblue')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (Log Scale)')
    axes[0, 0].set_title('Training Loss Curve')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 子图2: 定期评估结果
    eval_epochs = [e["epoch"] for e in eval_log]
    eval_vals = [e["loss"] for e in eval_log]
    axes[0, 1].plot(eval_epochs, eval_vals, marker='o', color='darkgreen')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Eval Loss (10 steps on fixed init)')
    axes[0, 1].set_title('Periodic Evaluation Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 收敛对比 (MLP vs Newton)
    x_axis = np.arange(max_steps + 1)
    # 避免 log(0) 警告
    mlp_plot = [max(l, 1e-9) for l in mlp_losses]
    newton_plot = [max(l, 1e-9) for l in newton_losses]
    
    axes[1, 0].plot(x_axis, mlp_plot, label='MLP Optimizer (Learned)', marker='o', color='steelblue')
    axes[1, 0].plot(x_axis, newton_plot, label='Newton Method', marker='s', color='crimson', linestyle='--')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss ($||x - target||^2$)')
    axes[1, 0].set_title('Convergence Comparison (Fixed Init)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 子图4: 步长/方向范数对比
    mlp_norms, newton_norms = [], []
    mlp_x, newton_x = x_val_init.clone(), x_val_init.clone()
    for _ in range(max_steps):
        d_m = mlp(mlp_x, target_val).detach().numpy()
        d_n = newton_direction(newton_x, target_val).detach().numpy()
        mlp_norms.append(np.linalg.norm(d_m))
        newton_norms.append(np.linalg.norm(d_n))
        mlp_x = mlp_x + torch.from_numpy(d_m)
        newton_x = newton_x + torch.from_numpy(d_n)
        
    axes[1, 1].plot(np.arange(max_steps), mlp_norms, label='MLP Step Norm', marker='^', color='steelblue')
    axes[1, 1].plot(np.arange(max_steps), newton_norms, label='Newton Step Norm', marker='v', color='crimson', linestyle='--')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('||Direction||_2')
    axes[1, 1].set_title('Update Magnitude per Step')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('optimization_report.png', dpi=300, bbox_inches='tight')
    print("🖼️  可视化图表已保存至: optimization_report.png")
    print("="*50)

if __name__ == "__main__":
    main()