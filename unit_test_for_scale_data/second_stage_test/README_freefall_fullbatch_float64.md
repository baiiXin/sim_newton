# 自由落体单帧变分问题：规则网格 Full-Batch 数据规模消融实验

## 1. 项目简介

本脚本研究一个非常具体的问题：**神经网络能否学习一个局部迭代更新规则，使单帧自由落体隐式欧拉变分问题从给定初值逐步收敛到精确解。**

实验固定物理参数、网络结构、损失函数、训练策略和局部采样范围，仅改变精确解附近规则网格的密度，并比较不同训练数据规模下学习型迭代器的收敛表现。

默认实验使用：

- `torch.float64` 双精度；
- 三维规则网格 Full-Batch 训练；
- `SGD` 和 `Adam` 两类优化器；
- 每类优化器分别测试 `1e-2`、`1e-3`、`1e-4` 三种学习率；
- 7 档数据规模；
- 共计 `7 × 6 = 42` 组实验；
- 默认设备为 `cuda:0`。

> **核心定位**：这不是一个完整的自由落体轨迹预测器，也不是一个跨物理状态泛化的求解器。它是一个针对固定单帧优化问题的局部学习型迭代器实验。

---

## 2. 问题定义

### 2.1 单帧自由落体更新

已知当前时刻的位置和速度：

$$
\mathbf{p}_n \in \mathbb{R}^3, \qquad
\mathbf{v}_n \in \mathbb{R}^3,
$$

希望求下一帧的位置：

$$
\mathbf{y} = \mathbf{p}_{n+1}.
$$

脚本固定使用：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| $m$ | 质量 | `1.0` |
| $g$ | 重力加速度 | `9.8` |
| $\Delta t$ | 时间步长 | `0.01` |
| $\mathbf{p}_n$ | 当前帧位置 | `[3.0, 4.0, 5.0]` |
| $\mathbf{v}_n$ | 当前帧速度 | `[0.5, -0.5, 0.0]` |

### 2.2 隐式欧拉变分能量

脚本没有直接监督网络拟合解析解，而是定义隐式欧拉离散对应的变分能量：

$$
E(\mathbf{y})
=
\frac{m}{2\Delta t^2}
\left\|
\mathbf{y}-\mathbf{p}_n-\Delta t\mathbf{v}_n
\right\|_2^2
+
mg y_z.
$$

其中：

- 第一项约束下一帧位置不要偏离惯性预测 $\mathbf{p}_n + \Delta t\mathbf{v}_n$；
- 第二项是重力势能；
- 最优解是能量最小点。

对能量求梯度：

$$
\nabla E(\mathbf{y})
=
\frac{m}{\Delta t^2}
\left(
\mathbf{y}-\mathbf{p}_n-\Delta t\mathbf{v}_n
\right)
+
mg\mathbf{e}_z.
$$

令梯度为零，可得到精确解：

$$
\mathbf{y}^{*}
=
\mathbf{p}_n
+
\Delta t\mathbf{v}_n
-
\Delta t^2 g\mathbf{e}_z.
$$

在默认参数下：

$$
\mathbf{y}^{*}
=
[3.005,\; 3.995,\; 4.99902].
$$

### 2.3 为什么使用 Newton 方法作为基准

该能量函数是严格凸二次函数，Hessian 为常数：

$$
\nabla^2 E(\mathbf{y})
=
\frac{m}{\Delta t^2}\mathbf{I}.
$$

Newton 更新方向为：

$$
\Delta \mathbf{y}_{\text{Newton}}
=
-\left(\nabla^2 E\right)^{-1}\nabla E
=
-\frac{\Delta t^2}{m}\nabla E.
$$

由于目标函数是二次函数，Newton 方法理论上可从任意初值**一步到达精确解**。脚本将其作为理想优化器参考，用于对照 MLP 学习到的更新规则。

---

## 3. 实验究竟在学习什么

### 3.1 学习型迭代器，而不是直接回归器

网络接收当前迭代位置 $\mathbf{y}^{(k)}$，预测一个位置更新量：

$$
\Delta \mathbf{y}^{(k)}
=
\Delta t \cdot f_{\theta}(\mathbf{x}^{(k)}),
$$

然后更新：

$$
\mathbf{y}^{(k+1)}
=
\mathbf{y}^{(k)} + \Delta \mathbf{y}^{(k)}.
$$

因此，网络的角色类似一个可学习的局部优化算法：输入当前优化变量，输出下一次迭代应该移动的方向和步长。

### 3.2 一个重要区分：训练样本不是不同物理状态

训练网格中的每一个点表示：

$$
\mathbf{y}^{(0)}_i,
$$

即**同一个固定物理问题下，求解器可能遇到的不同迭代初值**。

训练集并没有改变：

- 当前帧位置 $\mathbf{p}_n$；
- 当前帧速度 $\mathbf{v}_n$；
- 质量 $m$；
- 重力加速度 $g$；
- 时间步长 $\Delta t$。

因此，该实验回答的是：

> 当求解器在固定问题的精确解附近，从不同局部初值出发时，MLP 能否学习一个稳定的迭代更新规则？增加局部采样密度是否有帮助？

该实验**不能**直接回答：

> 当物理初值、速度、材料参数或时间步长变化时，网络是否仍然能够泛化？

要研究后一个问题，需要让 `p_n`、`v_n`、`m`、`g`、`dt` 等特征也在训练集中变化。

---

## 4. 网络结构

### 4.1 输入特征

MLP 输入为 12 维向量：

$$
\mathbf{x}
=
[
\mathbf{y},
\mathbf{p}_n,
\mathbf{v}_n,
m,
g,
\Delta t
].
$$

各部分维度如下：

| 输入部分 | 维度 | 含义 |
|---|---:|---|
| `y` | 3 | 当前迭代位置 |
| `p_n` | 3 | 当前物理帧位置 |
| `v_n` | 3 | 当前物理帧速度 |
| `m, g, dt` | 3 | 质量、重力加速度和时间步长 |
| **总计** | **12** | MLP 输入维度 |

### 4.2 MLP 架构

网络结构固定为：

```text
12 → 32 → 32 → 3
```

具体形式：

```python
nn.Sequential(
    nn.Linear(12, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 3),
)
```

总参数量为：

$$
(12 \times 32 + 32)
+
(32 \times 32 + 32)
+
(32 \times 3 + 3)
=
1571.
$$

### 4.3 输出缩放

网络原始输出还会乘以时间步长：

$$
\Delta \mathbf{y}
=
\Delta t \cdot \operatorname{MLP}(\mathbf{x}).
$$

这样做将网络输出解释为与速度量级相近的更新，再通过 $\Delta t$ 转化为位置增量。

### 4.4 零初始化最后一层

代码将最后一层的权重和偏置初始化为零：

```python
nn.init.zeros_(self.net[-1].weight)
nn.init.zeros_(self.net[-1].bias)
```

因此，训练开始时网络对任意输入都输出：

$$
\Delta \mathbf{y}=\mathbf{0}.
$$

也就是说，初始模型不会改变当前状态，随后再通过能量损失逐步学习有效更新方向。

---

## 5. 输入归一化

### 5.1 逐特征归一化

输入使用逐特征标准化：

$$
\widehat{\mathbf{x}}
=
\frac{\mathbf{x}-\boldsymbol{\mu}}
{\boldsymbol{\sigma}}.
$$

归一化统计量不是通过遍历完整网格计算，而是根据规则网格的解析形式直接获得。

### 5.2 哪些特征会变化

在当前实验中，只有 `y` 随训练样本变化。其余特征：

```text
[p_n, v_n, m, g, dt]
```

在所有样本中均保持不变。

因此：

- `y` 的均值为 `y_star`；
- `y` 的标准差由规则网格解析计算；
- 固定特征的真实标准差为 0；
- 为避免除零，代码将固定特征的标准差替换为 1。

对于每轴使用 $n$ 个等距点、半径为 $R$ 的对称网格，`y` 每个坐标分量的总体标准差为：

$$
\sigma_y
=
R\sqrt{\frac{n+1}{3(n-1)}}.
$$

### 5.3 当前实验中归一化的实际含义

由于固定特征归一化后均为 0，网络在本实验中实际上主要学习：

$$
\frac{\mathbf{y}-\mathbf{y}^{*}}{\sigma_y}
\longmapsto
\Delta \mathbf{y}.
$$

换言之，它学习的是围绕精确解的局部误差修正规则。

---

## 6. 训练集设置

### 6.1 局部三维规则网格

训练数据以精确解 $\mathbf{y}^{*}$ 为中心，在三个坐标方向分别均匀采样：

$$
y_j \in [y_j^{*}-R,\; y_j^{*}+R],
\qquad R=0.01.
$$

三维训练集为三个一维采样轴的笛卡尔积：

$$
\mathcal{D}
=
\mathcal{A}_x
\times
\mathcal{A}_y
\times
\mathcal{A}_z.
$$

### 6.2 为什么每轴使用偶数个点

每个轴均使用偶数个对称采样点，因此偏移量 0 不会出现在一维轴上。于是三维笛卡尔积中也不会包含精确解：

$$
\mathbf{y}^{*}\notin\mathcal{D}.
$$

这样可以避免网络直接见到目标点本身。

### 6.3 默认数据规模

默认使用 7 档规则网格：

| 每轴点数 | 总样本数 | 轴向间距 | float64 网格缓存约占显存 |
|---:|---:|---:|---:|
| 2 | 8 | `2.00000000e-02` | `< 0.01 MiB` |
| 4 | 64 | `6.66666667e-03` | `< 0.01 MiB` |
| 6 | 216 | `4.00000000e-03` | `< 0.01 MiB` |
| 10 | 1,000 | `2.22222222e-03` | `0.02 MiB` |
| 22 | 10,648 | `9.52380952e-04` | `0.24 MiB` |
| 46 | 97,336 | `4.44444444e-04` | `2.23 MiB` |
| 100 | 1,000,000 | `2.02020202e-04` | `22.89 MiB` |

缓存显存只计算规则网格张量本身：

$$
N \times 3 \times 8\text{ bytes}.
$$

实际训练显存还包括网络激活、展开轨迹计算图、梯度和优化器状态，尤其会随样本数 $N$ 和展开步数 $K$ 增长。

### 6.4 自定义目标规模时的映射规则

命令行参数 `--target-dataset-sizes` 可以接收任意正整数。代码会寻找最接近目标规模的偶数轴向点数 $n$，然后使用：

$$
N_{\text{actual}}=n^3.
$$

因此，自定义的目标规模不一定等于实际规模。若两个目标值映射到同一个实际网格规模，脚本会报错，避免重复实验。

### 6.5 真实测试初值

最终测试从：

$$
\mathbf{y}^{(0)}_{\text{test}}=\mathbf{p}_n
$$

出发。

需要注意：`p_n` 是固定物理问题的当前帧位置，同时也是测试阶段采用的求解器迭代初值。它没有作为训练网格中的一个明确样本参与训练。

默认情况下：

$$
\mathbf{p}_n=[3,4,5],
$$

而：

$$
\mathbf{y}^{*}=[3.005,3.995,4.99902].
$$

二者距离较近，因此测试点位于局部采样区域内部，但不等于精确解，也通常不与规则网格中的某个训练点重合。

---

## 7. Full-Batch 训练策略

### 7.1 设备端缓存

对于每一个数据规模：

1. 在目标设备上分块生成完整规则网格；
2. 将网格缓存到显存或内存中；
3. 在该数据规模下，6 组优化器配置复用同一份缓存；
4. 完成 6 组实验后释放缓存。

分块预生成只用于降低构造网格时的临时张量峰值，并不会改变训练方式。

### 7.2 不使用 mini-batch

每个 epoch 直接对完整规则网格计算一次损失：

```text
batch size = 当前规则网格的全部样本数
```

脚本不执行：

- mini-batch 切分；
- 样本打乱；
- DataLoader；
- 多个 batch 的梯度累计。

每个 epoch 只有：

```text
1 次 backward
1 次 optimizer.step
```

### 7.3 多步展开

对于每个训练点 $\mathbf{y}_i^{(0)}$，网络连续迭代 $K$ 步：

$$
\mathbf{y}_i^{(k+1)}
=
\mathbf{y}_i^{(k)}
+
\Delta t\cdot f_{\theta}(\mathbf{x}_i^{(k)}).
$$

轨迹中间没有 `detach()`。因此，后续步骤的损失会通过完整展开轨迹反向传播到网络参数。

### 7.4 损失函数

训练目标是所有展开步骤的平均能量之和：

$$
\mathcal{L}(\theta)
=
\sum_{k=1}^{K}
\frac{1}{N}
\sum_{i=1}^{N}
E\left(\mathbf{y}_i^{(k)}\right).
$$

注意：

- 训练损失直接使用能量，不使用监督标签；
- 没有额外的残差损失；
- 没有直接约束网络输出逼近 Newton 方向；
- 没有除以 $K$；
- 没有对不同展开步骤使用不同权重。

记录日志时，脚本额外计算：

$$
\mathcal{L}_{\text{gap}}
=
\mathcal{L}-K E(\mathbf{y}^{*}),
$$

它仅用于更直观地绘图，不会参与反向传播。

### 7.5 Curriculum：逐步增加展开步数

默认训练共 10,000 个 epoch，展开步数 $K$ 按如下方式增加：

| Epoch 范围 | 展开步数 $K$ |
|---:|---:|
| `0 – 1999` | 1 |
| `2000 – 3999` | 2 |
| `4000 – 5999` | 3 |
| `6000 – 7999` | 4 |
| `8000 – 9999` | 5 |

这样做的目的，是让网络先学会一次局部更新，再逐步学习连续多次迭代后的稳定性。

---

## 8. 优化器与消融变量

### 8.1 默认优化器配置

每个数据规模均测试以下 6 组训练配置：

| 优化器 | 学习率 |
|---|---:|
| SGD | `1e-2` |
| SGD | `1e-3` |
| SGD | `1e-4` |
| Adam | `1e-2` |
| Adam | `1e-3` |
| Adam | `1e-4` |

这里的 SGD 和 Adam 是用于训练 MLP 参数 $\theta$ 的 PyTorch 优化器，不是测试阶段直接用于求解 $\mathbf{y}$ 的传统数值优化器。

### 8.2 数据规模消融中保持不变的因素

本实验希望重点观察局部训练数据密度的影响。默认情况下，以下因素保持固定：

- 单帧物理问题；
- $m$、$g$、$\Delta t$；
- $\mathbf{p}_n$、$\mathbf{v}_n$；
- 采样中心 $\mathbf{y}^{*}$；
- 采样半径 $R=0.01$；
- 网络结构；
- 双精度数值类型；
- 输入归一化方式；
- 输出乘以 `dt` 的缩放方式；
- Full-Batch 训练方式；
- Curriculum 展开策略；
- 模型随机种子 `42`。

严格来说，脚本同时比较了不同优化器和学习率，因此整体实验包含两个维度：

1. 规则网格数据规模；
2. MLP 参数训练优化器配置。

---

## 9. 评估指标

### 9.1 能量差 Gap

$$
\operatorname{gap}(\mathbf{y})
=
E(\mathbf{y})-E(\mathbf{y}^{*}).
$$

由于 $\mathbf{y}^{*}$ 是全局最小点，理论上：

$$
\operatorname{gap}(\mathbf{y})\ge 0.
$$

Gap 越接近 0，说明当前位置越接近最优解。

### 9.2 一阶驻点残差

$$
r(\mathbf{y})
=
\left\|\nabla E(\mathbf{y})\right\|_2.
$$

残差越接近 0，说明当前位置越接近变分问题的驻点。由于本问题严格凸，唯一驻点就是全局最小点。

### 9.3 周期性评估

训练期间每隔 `eval_interval=100` 个 epoch，脚本会冻结模型，并从测试初值：

$$
\mathbf{y}^{(0)}=\mathbf{p}_n
$$

出发展开 `max_k=5` 步，记录最终 Gap 和残差。

这使得不同 epoch 的模型可以在统一测试条件下比较。

### 9.4 最终评估

训练结束后，脚本从相同测试初值出发：

- 使用 MLP 连续迭代 `final_test_steps=50` 步；
- 使用 Newton 方法连续迭代 50 步；
- 记录每一步的位置、能量、Gap、残差和下一步更新量范数。

由于 Newton 方法理论上一步收敛，后续步骤主要用于统一输出格式和绘图。

---

## 10. 输出文件

脚本会在自身所在目录下创建一个与脚本同名的输出目录。例如：

```text
freefall_fullbatch_dataset_scale_ablation_float64.py
freefall_fullbatch_dataset_scale_ablation_float64/
```

### 10.1 每组实验的输出

每个“优化器 × 学习率 × 数据规模”组合会创建一个独立目录：

```text
<optimizer>_lr_<learning_rate>_grid_axis_<n>_num_samples_<N>/
```

其中包含：

| 文件 | 内容 |
|---|---|
| `mlp_optimizer_state_dict.pt` | 训练后的模型参数与归一化 buffer |
| `optimization_report.json` | 完整配置、训练日志、周期性测试记录、最终 MLP 与 Newton 轨迹 |
| `training_and_reference_eval_curves.png` | 训练 Gap、测试 Gap、测试残差随 epoch 变化曲线 |
| `final_reference_residual_comparison.png` | 最终 MLP 与 Newton 残差迭代曲线 |
| `final_reference_trajectory_3d.png` | MLP 从 `p_n` 出发的三维迭代轨迹 |
| `training_dataset_sample.png` | 训练规则网格抽样可视化 |
| `final_reference_energy_contour_2d.png` | `x-z` 平面能量差等高线和轨迹投影；使用 `--skip-contour` 时跳过 |

### 10.2 全部实验的汇总输出

输出根目录还会包含：

| 文件 | 内容 |
|---|---|
| `dataset_scale_ablation_summary.json` | 42 组实验的摘要配置和最终指标 |
| `dataset_scale_ablation_summary.png` | 不同数据规模下最终 Gap 和残差的对比图 |

---

## 11. 代码结构

脚本按照以下模块组织：

| 模块 | 主要内容 |
|---|---|
| `0. 默认实验参数` | dtype、数据规模、采样范围、epoch、Curriculum 和优化器配置 |
| `1. 数据结构与通用辅助函数` | 配置 dataclass、网格规模映射、JSON 安全输出、有限值检查 |
| `2. 物理问题、网络与优化器` | MLP、变分能量、驻点残差、Newton 方向、PyTorch 优化器构造 |
| `3. 隐式规则网格训练集` | 网格索引转坐标、设备端分块缓存、绘图抽样、解析归一化统计量 |
| `4. 训练与评估` | MLP 测试轨迹、Newton 轨迹、单组实验训练和日志保存 |
| `5. 绘图` | 训练曲线、残差对比、三维轨迹、训练集分布、能量等高线、规模汇总图 |
| `6. 主程序` | 参数解析、合法性检查、42 组实验循环、汇总输出 |

---

## 12. 运行环境

### 12.1 Python 依赖

需要安装：

```bash
pip install torch numpy matplotlib
```

建议使用带 CUDA 支持的 PyTorch 环境运行正式实验。

### 12.2 正式运行

假设脚本重命名为：

```text
freefall_fullbatch_dataset_scale_ablation_float64.py
```

在默认 `cuda:0` 上运行：

```bash
python freefall_fullbatch_dataset_scale_ablation_float64.py
```

指定其他 GPU：

```bash
python freefall_fullbatch_dataset_scale_ablation_float64.py --device cuda:1
```

### 12.3 快速检查

正式实验包含 42 组 Full-Batch 训练，每组默认 10,000 个 epoch，计算量很大。建议先使用少量数据和较少 epoch 做冒烟测试：

```bash
python freefall_fullbatch_dataset_scale_ablation_float64.py \
  --device cuda:0 \
  --dataset-sizes 8 64 216 \
  --epochs 20 \
  --eval-interval 5 \
  --final-test-steps 10 \
  --initial-k 1 \
  --k-increase-interval 10 \
  --max-k 2 \
  --skip-contour
```

CPU 也可用于小规模检查：

```bash
python freefall_fullbatch_dataset_scale_ablation_float64.py \
  --device cpu \
  --dataset-sizes 8 64 \
  --epochs 5 \
  --eval-interval 1 \
  --final-test-steps 5 \
  --skip-contour
```

---

## 13. 命令行参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--target-dataset-sizes` / `--dataset-sizes` | `8 64 216 1000 10648 97336 1000000` | 目标训练数据规模；实际规模映射到最接近的偶数三维规则网格 |
| `--sampling-radius` | `0.01` | 每个坐标轴相对 `y_star` 的采样半径 |
| `--grid-precompute-chunk-size` | `1000000` | 在设备端预生成规则网格时的分块大小 |
| `--epochs` | `10000` | 每组实验训练 epoch 数 |
| `--eval-interval` | `100` | 周期性测试间隔 |
| `--final-test-steps` | `50` | 训练结束后的最终迭代评估步数 |
| `--initial-k` | `1` | 初始训练展开步数 |
| `--k-increase-interval` | `2000` | 每隔多少 epoch 增加展开步数 |
| `--k-increase-amount` | `1` | 每次增加的展开步数 |
| `--max-k` | `5` | 最大训练展开步数 |
| `--device` | `cuda:0` | 训练设备，例如 `cpu`、`cuda:0`、`cuda:1` |
| `--skip-contour` | 关闭 | 跳过二维能量等高线图，加快快速测试 |

---

## 14. 如何阅读结果

建议优先观察以下三个问题。

### 14.1 网络是否真正收敛

查看：

```text
final_reference_residual_comparison.png
```

重点观察 MLP 的残差是否持续下降并逼近 0。若残差先下降后上升，说明 learned update 可能在多次迭代后不稳定。

### 14.2 增加数据规模是否改善测试表现

查看：

```text
dataset_scale_ablation_summary.png
```

横轴为训练网格样本数，纵轴分别为最终 Gap 和最终驻点残差。若曲线随数据规模增大明显下降，说明更密集的局部采样有助于学习稳定更新规则。

### 14.3 MLP 学到的轨迹与 Newton 有何差异

查看：

```text
final_reference_energy_contour_2d.png
final_reference_trajectory_3d.png
```

Newton 对该二次问题理论上一步收敛。MLP 若需要多步才能接近精确解，说明网络学习到的是近似修正方向；若发生振荡或发散，则说明多步迭代稳定性不足。

---

## 15. 当前实验的边界

为了正确解释结果，需要明确以下限制。

### 15.1 只研究一个固定物理问题

当前训练集中只有 `y` 在变化，`p_n`、`v_n` 和物理参数均固定。因此，网络并未学习一个通用自由落体求解器。

### 15.2 只研究精确解附近的局部区域

训练网格位于：

$$
[y_x^*-0.01,y_x^*+0.01]
\times
[y_y^*-0.01,y_y^*+0.01]
\times
[y_z^*-0.01,y_z^*+0.01].
$$

当前实验主要检验局部收敛能力，不检验远离精确解时的全局收敛域。

### 15.3 当前目标函数非常简单

自由落体单帧能量是严格凸二次函数，Newton 方法可以一步求解。该实验适合验证训练流程和 learned optimizer 的基本行为，但不足以证明网络能处理复杂非线性弹性体、接触或无穿透约束问题。

### 15.4 数据规模增加不仅影响信息量，也影响计算量

由于使用 Full-Batch 训练，每个 epoch 的计算和显存开销都会随样本数增长。展开步数 $K$ 增大时，还需要保存更长的反向传播计算图。

---

## 16. 代码审阅备注

脚本主体实现与实验目标一致，但原代码中的少量注释需要按实际实现理解：

1. `precompute_regular_grid_on_device()` 的 docstring 中写有“最大默认网格含 10^6 个 float32 三维点，占用约 11.44 MiB”。当前脚本实际使用 `torch.float64`，因此最大规则网格缓存约为 `22.89 MiB`。
2. 绘图函数中的个别注释提到避免绘制约 `10^8` 个点，但当前默认最大规模为 `10^6`。这不影响实际逻辑。
3. 最大规模训练的主要显存压力通常不是缓存网格本身，而是 Full-Batch 多步展开产生的计算图。

---

## 17. 后续可扩展方向

在当前局部单问题实验稳定后，可以逐步扩展：

1. 让 `p_n` 和 `v_n` 在训练集中变化，测试跨物理状态泛化；
2. 改变 `dt`、`m`、`g`，验证网络是否真正利用物理参数输入；
3. 比较 Full-Batch 与随机采样 mini-batch；
4. 比较单步训练、多步展开训练和带 `detach()` 的迭代训练；
5. 增加直接监督 Newton 更新方向的对照实验；
6. 将二次自由落体问题替换为非线性弹性势能、接触或无穿透约束问题。

---

## 18. 一句话总结

该脚本通过规则网格 Full-Batch 消融实验，研究一个 12→32→32→3 的 MLP 能否在固定单帧自由落体变分问题中，学习精确解附近的局部迭代更新规则，并考察训练网格密度、优化器和学习率对最终收敛性的影响。
