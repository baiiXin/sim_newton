# 基于 MLP 的单帧自由落体学习型迭代求解器实验

## 1. 实验概述

这段代码研究一个非常基础但具有代表性的学习型求解器问题：

> 对于一个固定的单时间步自由落体隐式欧拉问题，训练一个多层感知机（MLP），使其能够像迭代优化器一样，反复输出当前位置的修正量，并逐步逼近变分能量的最优解。

这里的神经网络不是直接预测最终答案，而是学习一个**可重复调用的迭代更新规则**。在测试阶段，同一个网络会被连续调用多次，形成一条优化轨迹：

$$
y^{(0)} \rightarrow y^{(1)} \rightarrow y^{(2)} \rightarrow \cdots \rightarrow y^{(K)}.
$$

脚本固定使用 `torch.float32`，并在完全相同的物理问题、训练数据、网络结构和训练策略下，对比两种训练网络参数的优化器：

| 实验组 | 参数优化器 | 学习率 | Momentum |
|---|---:|---:|---:|
| 1 | SGD | $10^{-2}$ | 0 |
| 2 | Adam | $10^{-4}$ | — |

注意区分两类“优化器”：

1. **MLP 学习型迭代器**：在物理问题中更新位置变量 $y$；
2. **SGD / Adam**：在训练过程中更新 MLP 的网络参数 $\theta$。

---

## 2. 问题定义：单帧自由落体的隐式欧拉变分问题

### 2.1 已知状态与待求变量

脚本研究的是一个质点在重力作用下的单时间步自由落体问题。当前时间步的状态为：

- 当前位置：$p_n \in \mathbb{R}^3$；
- 当前速度：$v_n \in \mathbb{R}^3$；
- 质量：$m$；
- 重力加速度：$g$；
- 时间步长：$\Delta t$。

需要求解下一时间步的位置：

$$
y = p_{n+1} \in \mathbb{R}^3.
$$

代码中固定使用：

```python
m = 1.0
g = 9.8
dt = 0.01
p_n = torch.tensor([3.0, 4.0, 5.0])
v_n = torch.tensor([0.5, -0.5, 0.0])
```

### 2.2 隐式欧拉对应的变分能量

代码将单步仿真写成如下能量最小化问题：

$$
\min_y E(y),
$$

其中：

$$
E(y)
=
\frac{m}{2\Delta t^2}
\left\|y-p_n-\Delta t v_n\right\|_2^2
+
m g y_z.
$$

第一项可以理解为惯性项，第二项是重力势能项。函数 `variational_energy(...)` 实现了这一目标函数。

### 2.3 一阶驻点方程

对能量求梯度，可以得到：

$$
\nabla E(y)
=
\frac{m}{\Delta t^2}
\left(y-p_n-\Delta t v_n\right)
+
\begin{bmatrix}
0 \\
0 \\
mg
\end{bmatrix}.
$$

最优解 $y^*$ 满足：

$$
\nabla E(y^*) = 0.
$$

因此：

$$
y^*
=
p_n + \Delta t v_n
-
\Delta t^2
\begin{bmatrix}
0 \\
0 \\
g
\end{bmatrix}.
$$

代入脚本中的参数：

$$
y^* = [3.005,\; 3.995,\; 4.99902].
$$

代码中的 `stationarity_residual(...)` 计算 $\nabla E(y)$，而 `stationarity_residual_norm(...)` 计算：

$$
\left\|\nabla E(y)\right\|_2.
$$

这个指标可以判断迭代结果是否真正接近方程解。

### 2.4 Newton 法基线

当前目标函数是严格凸二次函数，其 Hessian 为：

$$
H = \frac{m}{\Delta t^2} I.
$$

因此 Newton 更新为：

$$
\Delta y_{\text{Newton}}
=
-H^{-1}\nabla E(y).
$$

由于 Hessian 精确且不随 $y$ 改变，Newton 法只需要一步就能够到达理论最优解。函数 `newton_direction(...)` 实现了这个基线。

Newton 法在本实验中不是需要击败的复杂算法，而是一个可解释的“标准答案”，用于检查 MLP 的迭代轨迹是否合理。

---

## 3. 方法：学习一个可迭代调用的更新规则

### 3.1 迭代形式

MLP 不直接输出最终位置 $y^*$，而是输出当前位置的修正量：

$$
\Delta y^{(k)}
=
f_\theta\left(
 y^{(k)},\;
 p_n,\;
 v_n,\;
 m,\;
 g,\;
 \Delta t
\right).
$$

每次迭代执行：

$$
y^{(k+1)}
=
y^{(k)} + \Delta y^{(k)}.
$$

其中：

- $k$ 表示 MLP 迭代次数；
- $\theta$ 表示网络参数；
- 同一个网络 $f_\theta$ 会在一条轨迹中被重复调用多次。

从抽象角度看，这段代码希望验证：

> 一个结构简单的 MLP，能否通过训练获得局部收敛行为，并作为迭代型优化器使用？

### 3.2 与直接监督回归的区别

这段代码不是使用标签 $y^*$ 训练如下映射：

$$
(p_n, v_n, m, g, \Delta t) \mapsto y^*.
$$

它采用的是基于物理能量的训练方式。网络输出更新量后，将更新后的状态代入 $E(y)$，直接最小化物理目标函数：

$$
\mathcal{L} = E(y).
$$

因此，理论最优解主要用于评估，而不是作为监督标签直接送入损失函数。

---

## 4. 网络结构

### 4.1 输入特征

网络输入由三部分拼接得到：

```python
inp = torch.cat([y, history, params], dim=-1)
```

各部分含义如下：

| 输入部分 | 维数 | 含义 |
|---|---:|---|
| `y` | 3 | 当前迭代位置 $y^{(k)}$ |
| `history = [p_n, v_n]` | 6 | 当前时间步的历史状态 |
| `params = [m, g, dt]` | 3 | 物理参数与时间步长 |
| **总计** | **12** | MLP 的输入维数 |

### 4.2 MLP 主体

网络结构为：

```text
12 维输入
   ↓
Linear(12, 32)
   ↓
ReLU
   ↓
Linear(32, 32)
   ↓
ReLU
   ↓
Linear(32, 3)
   ↓
3 维更新量 Δy
```

对应代码：

```python
self.net = nn.Sequential(
    nn.Linear(12, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
)
```

这是一个两隐藏层 MLP：

| 层 | 输入维数 | 输出维数 | 激活函数 |
|---|---:|---:|---|
| 输入层到隐藏层 1 | 12 | 32 | ReLU |
| 隐藏层 1 到隐藏层 2 | 32 | 32 | ReLU |
| 隐藏层 2 到输出层 | 32 | 3 | 无 |

### 4.3 输出乘以时间步长

网络原始输出会乘以 `dt`：

```python
delta = params[2] * delta
```

即：

$$
\Delta y^{(k)}
=
\Delta t \cdot \widetilde{\Delta y}^{(k)}.
$$

这样做相当于给更新量加入一个与时间尺度一致的先验约束，避免网络一开始产生过大的位置跳跃。

### 4.4 最后一层零初始化

输出层使用零初始化：

```python
nn.init.zeros_(self.net[-1].weight)
nn.init.zeros_(self.net[-1].bias)
```

因此，在训练开始时：

$$
\Delta y^{(k)} = 0.
$$

初始网络等价于“不更新当前位置”。随后，训练过程逐渐学习有效的迭代方向和步长。

### 4.5 输入标准化

代码对输入执行逐特征标准化：

$$
\widehat{x}_j
=
\frac{x_j - \mu_j}{\sigma_j}.
$$

对应实现：

```python
inp = (inp - self.input_mean) / self.input_std
```

`input_mean` 和 `input_std` 都是 12 维向量，因此每个输入特征拥有独立的均值与标准差。它们通过 `register_buffer(...)` 保存到模型中，不参与梯度更新，但会随模型一起保存和加载。

需要注意：标准化统计量只根据 10 个初始扰动点计算。由于 `p_n、v_n、m、g、dt` 在当前实验中始终固定，这 9 个特征的方差接近 0，代码会将对应标准差替换为 1：

```python
std = torch.where(std < 1e-8, torch.ones_like(std), std)
```

因此，本实验真正发生变化的输入主要是当前位置 $y$ 的三个分量。

---

## 5. 数据集设计

### 5.1 固定的物理问题

脚本只构造一个物理优化问题：

- 固定 $p_n$；
- 固定 $v_n$；
- 固定 $m$；
- 固定 $g$；
- 固定 $\Delta t$。

训练集不会改变物理场景，也不会改变每个时间步的优化目标。它只改变**迭代求解器的初始猜测**。

### 5.2 原始初值

代码令：

```python
y0 = p_n.clone()
```

这里的 `y0` 不是新的物理状态，而是求解下一时刻位置 $y=p_{n+1}$ 时使用的初始猜测：

$$
y^{(0)} = y_0 = p_n.
$$

### 5.3 训练集：初值附近的 10 个随机扰动点

训练集包含 10 个局部扰动初值：

$$
y_i^{(0)}
=
y_0 + \sigma \varepsilon_i,
\qquad
\varepsilon_i \sim \mathcal{N}(0,I).
$$

脚本固定：

$$
\sigma = 10^{-2}.
$$

对应参数：

```python
FIXED_NUM_PERTURBATION_POINTS = 10
PERTURBATION_STD_VALUES = [1e-2]
LOCAL_RANDOM_SEED = 123
```

扰动是绝对坐标扰动，不再额外乘以 `dt`。固定随机种子保证 SGD 和 Adam 两组实验使用完全相同的 10 个训练初值。

### 5.4 评估集：未参与训练的精确初值

精确初值 `y0` 被刻意排除在训练集之外，只用于评估：

```python
evaluation_initial_points = [
    y0.clone(),
]
```

因此，测试集只有一个点：

$$
\mathcal{D}_{\text{eval}} = \{y_0\}.
$$

这个设计主要用于检查：网络在局部扰动点上训练后，能否回到未见过的中心点 `y0` 并保持合理的迭代行为。

### 5.5 训练数据并不只有 10 个状态

严格来说，10 个点只是每条轨迹的**初始状态**。在每个 epoch 中，MLP 会从每个初值出发继续生成中间状态：

$$
y_i^{(0)},
\;y_i^{(1)},
\;y_i^{(2)},
\;\ldots,
\;y_i^{(K)}.
$$

这些中间状态会随着网络参数不断变化，因此训练过程会在线生成大量轨迹点。代码会将它们记录到日志中，用于后续可视化。

---

## 6. 训练策略

### 6.1 完整展开的迭代轨迹

对于一个训练初值 $y_i^{(0)}$，代码连续调用 MLP 共 $K$ 次：

$$
y_i^{(k+1)}
=
y_i^{(k)}
+
f_\theta(y_i^{(k)}, p_n, v_n, m, g, \Delta t).
$$

每一步更新之后都计算能量：

$$
E\left(y_i^{(1)}\right),
E\left(y_i^{(2)}\right),
\ldots,
E\left(y_i^{(K)}\right).
$$

单条轨迹的训练损失为：

$$
\mathcal{L}_i(\theta)
=
\sum_{k=1}^{K}
E\left(y_i^{(k)}\right).
$$

这意味着网络不仅需要让最终状态接近最优解，也需要让早期迭代步骤尽快降低能量。

### 6.2 轨迹内部不执行 `detach`

在一条轨迹内部，状态依赖关系为：

$$
y_i^{(0)}
\rightarrow y_i^{(1)}
\rightarrow \cdots
\rightarrow y_i^{(K)}.
$$

代码不会在迭代步骤之间切断计算图。因此，梯度可以穿过完整轨迹，从后面的损失反向传播到前面的更新步骤：

```python
delta = mlp(y, history, params)
y = y + delta
loss = variational_energy(y, p_n, v_n, m, g, dt)
trajectory_total_loss = trajectory_total_loss + loss
```

日志记录时使用的：

```python
y_before_for_log = y.detach().clone()
```

只用于保存数值，不会影响真正参与训练的 `y`。

这种训练方式可以称为：

- 完整展开训练（full unrolling）；
- 穿过优化轨迹的反向传播；
- 类似于通过时间的反向传播（BPTT）。

### 6.3 每条轨迹单独更新一次网络参数

脚本不是先汇总 10 条轨迹，再统一更新一次参数。它会依次处理每个训练初值：

1. 从一个扰动初值出发，展开 $K$ 步；
2. 累加该轨迹的 $K$ 个能量；
3. 执行一次 `backward()`；
4. 执行一次 `optimizer.step()`；
5. 再处理下一个扰动初值。

即：

```python
for initial_y in training_initial_points:
    opt.zero_grad()
    # 展开 K 步并累加 loss
    trajectory_total_loss.backward()
    opt.step()
```

由于训练集有 10 个初值，因此每个 epoch 会执行 10 次网络参数更新。

需要注意：后处理的轨迹会使用已经被前面轨迹更新过的网络参数。因此，这是一种**固定顺序的逐轨迹在线更新**，而不是标准的批量梯度下降。

### 6.4 逐步增加展开长度

代码采用 curriculum learning 风格的训练策略。训练开始时只展开少量步骤，随后逐渐增加迭代长度：

| Epoch 范围 | 每条轨迹的 MLP 迭代次数 $K$ |
|---:|---:|
| 0–199 | 5 |
| 200–399 | 10 |
| 400–599 | 15 |
| 600–799 | 20 |
| 800–999 | 25 |

对应参数：

```python
EPOCHS = 1000
INITIAL_K = 5
K_INCREASE_INTERVAL = 200
K_INCREASE_AMOUNT = 5
MAX_K = 25
```

这种设计的动机是：先让网络学会短轨迹上的基本下降行为，再逐渐要求它在更长时间范围内保持稳定。

### 6.5 训练伪代码

```text
for optimizer_config in [SGD(lr=1e-2), Adam(lr=1e-4)]:
    初始化同一个 MLP
    构造同一组 10 个扰动训练初值
    将 y0 留作评估点

    K = 5
    for epoch in range(1000):
        每 200 个 epoch 将 K 增加 5，最大为 25

        for each perturbed initial state y_i^(0):
            清空梯度
            y = y_i^(0)
            trajectory_loss = 0

            for k in range(K):
                Δy = MLP(y, p_n, v_n, m, g, dt)
                y = y + Δy
                trajectory_loss += E(y)

            trajectory_loss.backward()
            optimizer.step()

        每 100 个 epoch：
            从未参与训练的 y0 出发，固定迭代 10 步并记录误差

    训练结束后：
        从 y0 出发，将 MLP 与 Newton 法连续对比 50 步
```

---

## 7. 两组实验的控制变量

两组实验共享以下条件：

- 物理问题完全相同；
- 训练集完全相同；
- 评估集完全相同；
- 扰动尺度均为 $\sigma=10^{-2}$；
- 网络结构相同；
- 网络初始化种子相同；
- 输入标准化方式相同；
- 输出均乘以 `dt`；
- 均使用 `torch.float32`；
- 均采用逐轨迹完整展开反向传播；
- 均使用相同的 $K$ 递增策略。

唯一主动改变的因素是：训练 MLP 参数时使用的优化器及其学习率。

| 实验名称 | 参数优化器 | 学习率 |
|---|---|---:|
| `01_float32_sgd_lr_1e-02_perturbation_std_1e-02` | SGD | $10^{-2}$ |
| `02_float32_adam_lr_1e-04_perturbation_std_1e-02` | Adam | $10^{-4}$ |

因此，这一版代码主要用于比较：

> 在完全相同的局部学习型迭代器训练任务上，SGD 与 Adam 哪一种更容易让 MLP 获得稳定、可重复调用的下降行为？

---

## 8. 评估方式

### 8.1 周期性评估

每隔 100 个 epoch，代码会冻结当前网络，从未参与训练的 `y0` 出发，连续迭代 10 步：

```python
EVAL_INTERVAL = 100
EVAL_STEPS = 10
```

记录最终能量误差：

$$
E(y^{(10)}) - E(y^*).
$$

### 8.2 最终评估

训练结束后，代码从同一个 `y0` 出发，将 MLP 和 Newton 法分别连续运行 50 步：

```python
FINAL_TEST_STEPS = 50
```

虽然 Newton 法第一步已经到达最优解，但继续记录其轨迹可以作为数值底噪参考。

### 8.3 主要指标

#### 指标 1：能量差

$$
\text{Gap}(y)
=
E(y)-E(y^*).
$$

它表示当前位置相对于理论最优解还剩多少目标函数误差。

#### 指标 2：驻点残差

$$
\text{Residual}(y)
=
\left\|\nabla E(y)\right\|_2.
$$

这个指标比单纯的能量值更直接地衡量当前位置是否满足离散方程。

#### 指标 3：更新量范数

$$
\left\|\Delta y^{(k)}\right\|_2.
$$

它用于观察迭代器是否逐渐减小步长，以及是否存在震荡或发散。

---

## 9. 输出文件与可视化

每组实验都会生成独立目录，并保存以下结果。

### 9.1 数值结果

| 文件 | 内容 |
|---|---|
| `optimization_report.json` | 配置、训练日志、周期评估、最终对比和汇总指标 |
| `detailed_training_logs.npz` | 训练轨迹点、每条轨迹的最终点、每个微步的能量与 gap |
| `mlp_optimizer_state_dict.pt` | 训练后的 MLP 参数与标准化缓冲区 |

### 9.2 单组实验图像

| 文件 | 内容 |
|---|---|
| `optimization_report.png` | 四宫格统计图：训练 gap、周期评估 gap、MLP 与 Newton 的最终 gap、更新量范数 |
| `final_residual_comparison_initial_0.png` | 从保留初值 `y0` 出发的 residual 下降曲线 |
| `final_test_distribution_initial_0.png` | MLP 最终测试轨迹的三维空间分布 |
| `final_test_energy_contour_2d_initial_0.png` | MLP 与 Newton 轨迹在二维能量等高线上的投影 |
| `training_points_and_results_distribution.png` | 整个训练过程中输入轨迹点与最终结果点的分布 |

二维能量等高线图默认投影到 $x$-$z$ 平面。未展示的 $y$ 坐标固定为理论最优解 $y^*$ 的对应分量，因此背景表示穿过最优点的二维能量切片。

### 9.3 跨实验汇总

主程序还会生成：

| 文件 | 内容 |
|---|---|
| `float32_fixed_perturbation_per_trajectory_full_unrolled_summary.json` | 两组实验的统一配置和最终指标 |
| `float32_fixed_perturbation_per_trajectory_full_unrolled_summary.png` | SGD 与 Adam 的最终性能对比图 |

---

## 10. 如何理解这段代码的实验边界

### 10.1 它验证了什么？

这段代码适合验证以下问题：

1. 一个小型 MLP 能否学习局部迭代更新规则？
2. 同一个网络被重复调用时，是否能够持续降低物理能量？
3. 完整展开反向传播是否能够训练出多步收敛行为？
4. 增加展开长度 $K$ 后，网络能否保持稳定？
5. SGD 与 Adam 对学习型迭代器的训练效果有何差异？

### 10.2 它尚未验证什么？

这段代码还不能证明网络已经成为通用仿真求解器，因为：

- 只研究一个固定时间步；
- 只使用一个固定的 $p_n$ 和 $v_n$；
- 只使用一个固定的 $m、g、dt$；
- 训练初值只位于 `y0` 附近的局部区域；
- 评估集只有一个中心点 `y0`；
- 没有测试多帧滚动误差；
- 没有测试新的速度、位置、时间步长或物理参数。

因此，更准确的结论是：

> 该脚本用于研究 MLP 在固定单帧自由落体问题附近，是否能够学习一个局部、可重复调用的迭代型更新算子。

### 10.3 一个值得注意的实现细节

脚本中的汇总图函数名称包含 `perturbation_range`，但当前实际只使用一个扰动尺度：

```python
PERTURBATION_STD_VALUES = [
    1e-2,
]
```

因此，当前汇总图中每种优化器只有一个横坐标点。它本质上是 SGD 与 Adam 的固定扰动尺度对比，而不是完整的扰动范围消融实验。

---

## 11. 总结

这段代码建立了一个最小化、可解释的学习型迭代求解器实验：

1. 将单帧自由落体隐式欧拉离散写成严格凸二次能量最小化问题；
2. 使用一个 `12 → 32 → 32 → 3` 的 MLP 输出位置更新量；
3. 在 `y0` 附近采样 10 个扰动初值，训练网络学习局部下降规则；
4. 对每条轨迹完整展开 $K$ 步，并对轨迹上所有能量求和；
5. 不在轨迹内部执行 `detach`，使梯度能够穿过完整迭代过程；
6. 每条轨迹单独执行一次网络参数更新，每个 epoch 共更新 10 次；
7. 逐渐将展开长度从 5 增加到 25；
8. 将训练后的 MLP 与一步收敛的 Newton 法进行对比；
9. 通过能量差、驻点残差、更新量范数和轨迹可视化评估收敛行为；
10. 对比 SGD 和 Adam 对学习型迭代器训练效果的影响。

这个实验的价值不在于解决自由落体本身，而在于用一个具有解析解的最小问题，检查学习型迭代求解器的基本训练机制是否正确，为后续扩展到多帧仿真、弹性体和布料等复杂问题提供基线。
