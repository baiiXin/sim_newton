# `baseline_normalized_augmented.py` 详细说明

## 一、脚本概述

本脚本是一个**自由落体单步隐式欧拉时间积分器**的学习实验。在 baseline（仅用 `y0 → y*` 单条直线训练的 MLP 优化器）的基础上，新增两个核心改动：

1. **输入数据标准化（Input Standardization）** — 在网络入口处对 `[y, history, params]` 做 dataset-level 的零均值/单位方差归一化。
2. **训练集扩充（Training Set Augmentation）** — 不再只用初值 `y0` 训练，而是构造**线性插值锚点**（`y0 → y*` 之间均匀取点）+**最优点局部扰动锚点**（`y* + dt·σ·N(0,I)`）作为多个训练起点。

脚本的核心目标是：训练一个 MLP，使其作为隐式欧拉变分能量 `E(y)` 的**迭代优化器**，在固定物理参数 `(p_n, v_n, m, g, dt)` 下，与解析牛顿法对比收敛行为（loss gap 与 residual norm）。

> 重要边界：脚本**显式不引入 residual 信息**作为网络输入。residual 仅出现在评估/收敛性度量中。这是与 `ablution_res_test.py` 这类 residual-aware 变体的关键区别。

---

## 二、物理与数学背景

### 2.1 隐式欧拉单步格式

对自由落体方程 $m \ddot{x} = -m g \hat{z}$，隐式欧拉离散化下一个位置 $y = x_{n+1}$ 满足：

$$
y - p_n - \Delta t \cdot v_n + \Delta t^2 \cdot g \hat{z} = 0
$$

等价于最小化变分能量

$$
E(y) = \frac{m}{2 \Delta t^2}\|y - p_n - \Delta t \cdot v_n\|^2 + m g \cdot y_z
$$

### 2.2 解析最优解

由 $\nabla E(y^*) = 0$，得到闭式解

$$
y^* = p_n + \Delta t \cdot v_n - \Delta t^2 \cdot g \hat{z}
$$

脚本中固定取 `p_n = [3, 4, 5]`，`v_n = [0.5, -0.5, 0]`，`m=1`，`g=9.8`，`dt=0.01`，对应 `y* ≈ [3.005, 3.995, 4.99902]`，`E*` 由 `variational_energy(y_star, ...)` 计算。

### 2.3 牛顿方向

由于 Hessian $H = (m/\Delta t^2) I$ 各向同性，牛顿步：

$$
\Delta y_{\mathrm{Newton}} = -\frac{\Delta t^2}{m} \nabla E(y)
$$

理论上一步即可收敛到 $y^*$（在浮点精度内）。这是脚本用来作为收敛上限/参考线的基准。

### 2.4 Residual

$$
r(y) = \nabla E(y) = \frac{m}{\Delta t^2}(y - p_n - \Delta t v_n) + m g \hat{z}
$$

`residual_norm` 返回 $\|r(y)\|_2$，作为收敛性的硬性指标（不依赖 `E*` 的数值精度）。

---

## 三、代码结构

### 3.1 模块清单

| 函数 / 类                       | 职责                                                                     |
| ------------------------------ | ----------------------------------------------------------------------- |
| `MLPOptimizer`                 | 网络主体；内部做输入标准化，并将网络输出乘 `dt` 作为位置更新。                |
| `variational_energy`           | 计算 $E(y)$，作为训练 loss 与评估目标。                                  |
| `newton_direction`             | 牛顿方向 $-H^{-1}\nabla E$，作为参考解法。                                |
| `variational_residual`         | 返回 $\nabla E(y)$ 向量。                                                |
| `residual_norm`                | 返回 $\|\nabla E(y)\|_2$（标量），评估收敛性。                           |
| `make_training_states`         | 构造扩充训练集：line anchors + local anchors。                            |
| `compute_input_normalizer`     | 在训练集上计算输入向量的均值与标准差，常量维度的 std 兜底为 1。             |
| `main`                         | 训练 + 评估 + 报告输出主流程。                                            |

### 3.2 `MLPOptimizer` 设计要点

- **网络**：`Linear(12, 32) → ReLU → Linear(32, 32) → ReLU → Linear(32, 3)`
- **输入** 12 维：`[y(3), p_n(3), v_n(3), m, g, dt]`
- **归一化**：`(inp - input_mean) / input_std`，`mean/std` 通过 `register_buffer` 注册，会随 `state_dict` 一起保存，不参与训练。
- **输出 dt-scaling**：网络预测 `raw_delta`，最终返回 `dt * raw_delta`。
  - **动机**：典型 $\Delta y \sim O(\Delta t^2 g)$，对 `dt=0.01` 量级约 $10^{-3}$。让网络直接预测会被 ReLU 饱和；让它学习 $O(1)$ 量级的 `raw_delta`，再乘 `dt` 还原。
  - 注意这里乘的是 `dt`（一阶），不是 `dt^2`（二阶）。这是经验权衡：既缩放到合理量级，又保留一定可学习裕度。

### 3.3 `forward` 签名约束

`mlp(y, history, params)` 三参数的调用形式**不变**，与 baseline / 其他变体兼容，确保评估代码不需要修改。

---

## 四、训练集扩充策略

`make_training_states(y0, y_star, dt, num_line_points=11, num_local_points=32, local_std_dt_units=1.0, seed=123)` 生成的训练初值由两部分组成：

### 4.1 Line anchors（线性插值锚点）

$$
y_\alpha = (1-\alpha) y_0 + \alpha y^*, \quad \alpha \in \{0, 0.1, \dots, 1.0\}
$$

共 `num_line_points=11` 个点（含 `y0` 和 `y*` 端点）。

**作用**：让网络学习从 `y0→y*` 路径上**任意中间状态**到 `y*` 的更新方向，而不是只学习起点 → 终点的单次跳跃。

### 4.2 Local anchors（最优点局部扰动锚点）

$$
y_{\mathrm{local}} = y^* + \Delta t \cdot \sigma \cdot \mathcal{N}(0, I_3)
$$

共 `num_local_points=32` 个点，`σ = local_std_dt_units = 1.0`，扰动半径量级 $\Delta t \cdot \sigma = 0.01$。

**作用**：让网络学到 `y*` 附近**各方向小扰动**的回归行为，提升迭代器在收敛后的稳定性（fixed point 性质）。

### 4.3 与 residual-aware 方法的区别

本脚本扩充训练集的方式是**纯几何采样**，**不利用** $\nabla E$ 的方向信息（这是 `ablution_res_test.py` 走的路线）。因此可作为"仅靠数据多样性 + 标准化"能走多远的对照基线。

---

## 五、训练流程

```
for epoch in range(1000):
    if epoch > 0 and epoch % 500 == 0 and K < 10:
        K += 1                       # 每 500 epoch 增加一步 K
    for y_init in train_states:      # 遍历 43 个训练初值
        y = y_init.clone()
        for k in range(K):
            delta = mlp(y, history, params)
            y = y + delta
            loss = variational_energy(y, ...)
            loss.backward()
            opt.step()
            opt.zero_grad()
            y = y.detach()           # 切断历史，单步 BPTT
```

**关键策略**：

- **K-step curriculum**：`K=1` 起步，每 500 个 epoch 增加 1 步（上限 10）。**注释里写的是"每 100 epoch K+=1"，但代码实际是每 500 epoch**——以代码为准。
- **单步反向传播**：每一步 `k` 计算完 loss 后立即 `backward + step + detach`，避免长链 BPTT 的梯度爆炸/消失。
- **优化器**：Adam，`lr=1e-3`，无学习率调度。
- **训练样本数**：43 = 11 line + 32 local。
- **每 epoch 总更新步数** = `43 * K`。

---

## 六、评估与对比

### 6.1 周期评估（每 100 epoch）

从 `y0` 出发 rollout 10 步，记录每步的 `y / loss / residual_norm`，写入 `eval_log`。

### 6.2 最终对比评估

固定 `max_steps = 15`，对相同 `y0`：

- MLP 路径：`y_{k+1} = y_k + mlp(y_k, history, params)`
- 牛顿路径：`y_{k+1} = y_k + newton_direction(y_k, ...)`

记录每步的 `y / loss / residual_norm`，写入 `final_comparison`。打印表格展示前 5 步。

### 6.3 更新步长对比

额外记录 `||delta_mlp||_2` 与 `||delta_Newton||_2` 用于判断 MLP 是否过早收敛或过冲。

---

## 七、输出文件

脚本在**当前工作目录**（不是脚本所在目录）下生成以下文件：

| 文件名                       | 内容                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------- |
| `optimization_report.json`   | 完整数值结果：`config / training_log / periodic_evaluation / final_comparison`        |
| `optimization_report.png`    | 4 子图：训练 Gap、周期评估 Gap、最终 Gap 对比、更新步长对比                              |
| `optimization_residual.png`  | MLP vs Newton 的 residual norm 收敛曲线（对数轴）                                       |

> 注意：路径是相对的（`open("optimization_report.json", "w")`），若不在脚本目录下运行，输出会落到调用方的 cwd 中。

### 7.1 JSON 结构

```jsonc
{
  "config": {
    "epochs": 1000,
    "y0": [...], "p_n": [...], "v_n": [...],
    "m": 1.0, "g": 9.8, "dt": 0.01,
    "E_star": <float>,
    "normalization": { "input_mean": [12], "input_std": [12] },
    "training_set_expansion": {
      "num_line_points": 11,
      "num_local_points": 32,
      "local_std_dt_units": 1.0,
      "num_train_states": 43
    }
  },
  "training_log": [{"epoch", "K", "final_loss", "num_train_states"}, ...],
  "periodic_evaluation": [{"epoch", "K", "steps": [{"step", "y", "loss", "residual_norm"}]}, ...],
  "final_comparison": {
    "mlp":    {"init_y", "history", "params", "E_star", "iterations": [...]},
    "newton": {"init_y", "history", "params", "E_star", "iterations": [...]}
  }
}
```

---

## 八、配置常量

下列常量在 `main()` 内**硬编码**，如需扫描请直接修改源文件：

| 名称                  | 当前值          | 说明                                  |
| --------------------- | --------------- | ------------------------------------- |
| `torch.manual_seed`   | 42              | 全局随机种子                          |
| `seed` (in `make_training_states`) | 123  | 局部扰动 RNG 种子（独立于全局）       |
| `m, g, dt`            | 1.0, 9.8, 0.01  | 物理参数                              |
| `p_n`                 | [3, 4, 5]       | 当前位置                              |
| `v_n`                 | [0.5, -0.5, 0]  | 当前速度                              |
| `y0`                  | `p_n.clone()`   | 优化变量初值                          |
| `epochs`              | 1000            | 训练轮数                              |
| `K` (init)            | 1               | rollout 步数初值                      |
| K 上调间隔            | 500 epoch       | `epoch % 500 == 0 and K < 10` 时 +1   |
| `num_line_points`     | 11              | 线性插值锚点数                        |
| `num_local_points`    | 32              | 局部扰动锚点数                        |
| `local_std_dt_units`  | 1.0             | 局部扰动的 σ（以 dt 为单位）          |
| `lr` (Adam)           | 1e-3            | 学习率                                |
| `max_steps`           | 15              | 最终对比评估的 rollout 步数           |

---

## 九、运行方式

```bash
# 进入希望落产物的目录（输出 JSON/PNG 都在 cwd）
cd /data/zhoucy/sim_newton/GNN_solver/tests/free_fall_test/

# 直接运行
python baseline_normalized_augmented.py
```

**依赖**：`torch`、`numpy`、`matplotlib`。脚本已设置 `matplotlib.use('Agg')`，无显示器服务器也可运行。

**预期运行时间**：单 CPU 约几十秒到 1~2 分钟（取决于硬件，1000 epoch × 43 states × K-step）。

---

## 十、与同目录其他脚本的关系

| 脚本                          | 与本脚本的差异                                                          |
| ----------------------------- | ----------------------------------------------------------------------- |
| `free_fall_opt.py`            | baseline：单初值 `y0` 训练，无标准化、无扩充                              |
| `free_fall_opt_dx.py`         | baseline + dt-scaling 输出                                              |
| `ablution_test.py`            | 消融实验：测试各种增强组合（含/不含标准化、含/不含扩充）                   |
| `ablution_res_test.py`        | 引入 **residual** 作为额外输入，研究 residual-aware 网络                  |
| 本脚本                        | base-only 标准化 + 训练集扩充（**不含 residual**）的固定配置版            |

---

## 十一、常见调参建议

- **训练发散 / loss 不下降**：先关闭训练集扩充（`num_local_points=0`），确认 baseline 是否能跑通；再逐步打开。
- **MLP 收敛远慢于牛顿**：本属正常。MLP 只学到接近一阶下降；牛顿是这个问题的精确解。看 residual 曲线是否下降至 `1e-3 ~ 1e-4` 量级即可。
- **想测试不同 `dt`**：除了改 `dt`，注意 `y* ≈ p_n` 当 `dt→0`，`E*→0`，gap 曲线的可读性会下降；建议同时调 `v_n / p_n` 量级。
- **想比较 standardization 是否真的有用**：把 `MLPOptimizer.__init__` 调用改为不传 `input_mean / input_std`（保持默认零均值单位方差，相当于不做标准化），对比 `optimization_report.json` 中的 `final_comparison.mlp.iterations[*].residual_norm`。

---

## 十二、已知限制

1. **物理参数固定**：模型只在 `(p_n=[3,4,5], v_n=[0.5,-0.5,0], dt=0.01)` 下训练，泛化到其他初值需要重新训练。
2. **三维各向同性问题**：本质上是 3D 二次型最小化，理论上线性模型已足够，MLP 主要价值在于**作为多场景泛化器的可行性验证**。
3. **输出文件相对路径**：`open("optimization_report.json")` 没有指定目录，跨目录调用会覆盖其他实验的产物。建议运行前 `cd` 到目标目录或外部封装为 wrapper 脚本。
4. **注释与代码不一致**：脚本注释写"每 100 epoch K+=1"，代码实际是 500。
