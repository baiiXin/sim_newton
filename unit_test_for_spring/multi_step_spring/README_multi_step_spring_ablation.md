# 两质点单弹簧学习型迭代求解器：八组消融实验

## 1. 实验目的

本实验用于判断：从原始失败版本修改到稳定成功版本时，究竟是哪一类改动起主要作用。

原始版本与成功版本之间同时改变了以下内容：

1. 网络输入从绝对状态改为变分残差；
2. 对残差进行质量预条件和无量纲化；
3. 网络线性层由有偏置改为无偏置；
4. 第一层由 PyTorch 默认初始化改为正交初始化；
5. 网络输出由直接位置增量改为无量纲输出再乘特征长度；
6. 训练目标由原始物理能量改为平移并按特征能量缩放的物理能量；
7. 加入全局梯度范数裁剪。

如果只比较原始版与成功版，无法判断以上哪一项是决定性因素。因此，脚本采用两条互补的消融路径：

- **正向重建**：从失败版出发，分别加入“训练稳定化改动”或“残差状态表示”；
- **反向删除**：从完整成功版出发，每次删除一个关键组件。

八组实验使用相同的物理问题、训练集、验证集、测试集、随机种子、优化器、学习率、训练轮数、展开深度课程、checkpoint 选择方式和评价指标。实验之间只有下文明确列出的配置发生变化。

---

## 2. 文件说明

- `multi_step_spring_ablation.py`：八组消融实验的完整运行脚本；
- `README_multi_step_spring_ablation.md`：实验设计、运行方式和结果判读说明。

运行脚本后，输出目录自动设置为：

```text
multi_step_spring_ablation/
```

该目录与 Python 脚本位于同一级目录。

---

## 3. 共享物理问题

系统由三维空间中的两个自由质点和一根非线性弹簧组成。每个物理时间步对应一个独立的六维变分优化问题。

待求位置为

$$
y=(y_1,y_2)\in\mathbb{R}^6.
$$

自由预测位置为

$$
q_i=p_i^n+\Delta t\,v_i^n-\Delta t^2g e_z.
$$

物理变分能量为

$$
E(y)=
\frac{m_1}{2\Delta t^2}\|y_1-q_1\|^2
+
\frac{m_2}{2\Delta t^2}\|y_2-q_2\|^2
+
\frac{k_s}{2}\left(\|y_2-y_1\|-\ell_0\right)^2
+C,
$$

其中 $C$ 与 $y$ 无关。

驻点残差定义为

$$
g(y)=\nabla_y E(y).
$$

所有学习型求解器都采用相同的迭代形式：

$$
y^{(k+1)}=y^{(k)}+\Delta y_\theta^{(k)}.
$$

同一组网络参数在所有内层迭代和所有物理时间步问题之间共享。

---

## 4. 三种输入状态

### 4.1 绝对状态输入

原始版本使用 17 维输入：

$$
z=
\left[
 y,\ q,\ m_1,\ m_2,\ \Delta t,\ k_s,\ \ell_0
\right]\in\mathbb{R}^{17}.
$$

均值和标准差只由训练集计算，并进行逐特征标准化：

$$
\hat z=\frac{z-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}.
$$

质量和固定物理参数在当前实验中是常数。对于标准差为零的常量特征，代码将除数设为 1；这些特征在减去均值后仍然严格为零。

### 4.2 有量纲的质量预条件残差

定义

$$
\widetilde r(y)=\Delta t^2M_c^{-1}g(y),
$$

其中

$$
M_c=\operatorname{diag}(m_1,m_1,m_1,m_2,m_2,m_2).
$$

$\widetilde r(y)$ 具有长度量纲，并且在驻点处严格为零。

### 4.3 无量纲质量预条件残差

选取特征长度

$$
s=5\times 10^{-2},
$$

定义无量纲输入

$$
u(y)=\frac{\Delta t^2M_c^{-1}g(y)}{s}.
$$

完整成功版网络输出无量纲修正，再映射回位置单位：

$$
\Delta y=s\,F_\theta(u).
$$

---

## 5. 八组实验总表

所有网络均使用恒等激活函数和一个宽度为 64 的隐藏层。所有实验的输出层权重均从零开始；存在输出偏置时，输出偏置也从零开始。

| 编号 | 实验目录名 | 输入 | 偏置 | 第一层初始化 | 输出形式 | 能量除以 $E_s$ | 梯度裁剪 | 主要问题 |
|---|---|---|---|---|---|---|---|---|
| A0 | `A0_original_raw_state` | 标准化绝对状态 | 有 | 默认 | 直接输出 $\Delta y$ | 否 | 否 | 复现原始失败基线 |
| A1 | `A1_raw_state_stabilized` | 标准化绝对状态 | 无 | 正交 | $\Delta y=sF_\theta$ | 是 | 是 | 不改状态表示，只加入其余稳定化措施能否成功 |
| A2 | `A2_residual_core_only` | $\Delta t^2M^{-1}g$ | 有 | 默认 | 直接输出 $\Delta y$ | 否 | 否 | 只改成残差输入是否已经带来主要改善 |
| A3 | `A3_stable_full` | $\Delta t^2M^{-1}g/s$ | 无 | 正交 | $\Delta y=sF_\theta$ | 是 | 是 | 完整成功版，所有反向删除实验的参照组 |
| A4 | `A4_stable_with_bias` | 同 A3 | 有 | 正交 | 同 A3 | 是 | 是 | 恢复偏置后，严格固定点是否被破坏 |
| A5 | `A5_stable_default_init` | 同 A3 | 无 | 默认 | 同 A3 | 是 | 是 | 正交初始化是否是决定因素 |
| A6 | `A6_stable_no_energy_scale` | 同 A3 | 无 | 正交 | 同 A3 | 否 | 是 | 去掉能量尺度后，训练梯度尺度是否失稳 |
| A7 | `A7_stable_no_gradient_clip` | 同 A3 | 无 | 正交 | 同 A3 | 是 | 否 | 梯度裁剪是否保护展开深度课程切换 |

下面逐组说明每一组和对照组之间的差异。

---

## 6. 八组实验的具体改变

## A0：`A0_original_raw_state`

这是失败版本基线。

配置为：

- 17 维标准化绝对状态输入；
- 网络结构为 `17 -> 64 -> Identity -> 6`；
- 两个线性层均带偏置；
- 第一层使用 PyTorch 默认初始化；
- 输出层权重和偏置初始化为零；
- 网络输出直接作为位置增量，不乘特征长度；
- 训练目标直接使用原始物理能量；
- 不进行能量尺度归一化；
- 不进行梯度裁剪。

它回答的问题是：**原始绝对状态仿射网络在统一数据和评价器下能够达到什么水平？**

A0 是整个消融实验的失败参照点。

---

## A1：`A1_raw_state_stabilized`

A1 保留 A0 的绝对状态输入，但加入成功版中的训练稳定化措施。

相对 A0 的改变为：

- 删除所有线性层偏置；
- 第一层改为正交初始化；
- 网络原始输出乘特征长度 $s$ 后作为位置增量；
- 训练能量减去固定的初始能量；
- 训练目标除以特征能量尺度 $E_s$；
- 加入全局梯度范数裁剪。

A1 **仍然不使用物理残差作为输入**。

它回答的问题是：

> 如果状态表示仍然是绝对位置和问题参数，仅依靠初始化、尺度处理和梯度裁剪，能否挽救原始网络？

关键对照是 **A1 与 A3**：两者都使用稳定化训练设置，主要区别是 A1 使用绝对状态，A3 使用无量纲残差。

如果 A1 明显失败而 A3 成功，说明主要作用来自状态表示，而不是一般性的训练技巧。

注意：A1 虽然没有偏置，但其输入在精确解处通常不为零，因此它不具备“零残差对应零更新”的结构性固定点保证。

---

## A2：`A2_residual_core_only`

A2 只将 A0 的绝对状态输入替换为有量纲的质量预条件残差：

$$
\widetilde r(y)=\Delta t^2M_c^{-1}g(y).
$$

其余部分尽量保持原始风格：

- 线性层带偏置；
- 第一层使用默认初始化；
- 输出直接作为位置增量；
- 使用原始物理能量；
- 不进行能量尺度归一化；
- 不进行梯度裁剪。

它回答的问题是：

> 只把网络状态改成局部物理残差，是否已经足以带来主要性能提升？

关键对照是 **A0 与 A2**。

如果 A2 相比 A0 改善多个数量级，说明残差状态是最主要的改动；其他措施更多是在提高训练稳定性、最终精度或长迭代固定点性质。

A2 有偏置，因此即使输入残差为零，也不能保证输出更新严格为零。

---

## A3：`A3_stable_full`

A3 是完整成功版本，也是所有反向删除实验的基准。

配置为：

- 输入为无量纲质量预条件残差：

$$
u=\frac{\Delta t^2M_c^{-1}g(y)}{s};
$$

- 网络结构为 `6 -> 64 -> Identity -> 6`；
- 两个线性层均无偏置；
- 第一层正交初始化；
- 输出层零初始化；
- 输出映射为：

$$
\Delta y=sF_\theta(u);
$$

- 使用平移后的物理能量；
- 除以特征能量尺度；
- 使用全局梯度范数裁剪。

由于输入在驻点处为零，并且网络完全无偏置，因此

$$
g(y^*)=0
\quad\Longrightarrow\quad
u(y^*)=0
\quad\Longrightarrow\quad
\Delta y=0.
$$

这意味着任何驻点都是学习迭代的严格固定点。

CSV 中所有 `log10_degradation_vs_A3` 指标都以 A3 为分母。

---

## A4：`A4_stable_with_bias`

A4 只相对 A3 恢复两个线性层的偏置，其他设置全部不变。

偏置仍然从零初始化，因此训练刚开始时网络与 A3 一样输出零；但训练后偏置可以变为非零。

此时即使

$$
g(y^*)=0,
$$

网络仍可能满足

$$
F_\theta(0)\neq 0.
$$

它回答的问题是：

> 严格固定点约束对于解附近的稳定迭代和最终双精度精度是否必要？

重点查看：

- `exact_state_final_drift_p95`；
- `exact_state_final_residual_p95`；
- 从精确解出发迭代 50 次后的漂移；
- A4 与 A3 的最终残差差异。

A4 可能在前几步表现正常，但在长迭代中逐渐离开精确解。因此不能只看一步下降比例。

---

## A5：`A5_stable_default_init`

A5 只将 A3 的第一层正交初始化替换为 PyTorch 默认初始化。

其他设置与 A3 完全一致：

- 无量纲残差输入；
- 无偏置；
- 输出层零初始化；
- 输出乘特征长度；
- 能量尺度归一化；
- 梯度裁剪。

它回答的问题是：

> 正交初始化是成功的必要条件，还是只改善早期收敛速度和方向均衡性？

重点比较：

- 最佳 checkpoint 出现的 epoch；
- 一步更新与理想修正的余弦；
- 前 1000 个 epoch 的验证曲线；
- 最终残差是否最终追平 A3。

如果 A5 最终接近 A3，但早期收敛更慢，则正交初始化是辅助性改动，而不是决定性改动。

---

## A6：`A6_stable_no_energy_scale`

A6 只取消 A3 中对特征能量尺度的除法。

A3 的特征能量尺度为

$$
E_s=\frac{m_{\mathrm{ref}}s^2}{\Delta t^2}.
$$

A6 仍然可以减去初始能量，但目标不再除以 $E_s$。

注意：减去与参数无关的初始能量不会改变反向传播梯度；真正被消融的是正标量 $E_s$ 对梯度幅值的调整。

A6 回答的问题是：

> 成功是否依赖合适的训练目标尺度，以及该尺度如何与梯度裁剪和 Adam 相互作用？

重点查看：

- `maximum_gradient_norm`；
- `gradient_clip_trigger_fraction`；
- 每次 $K$ 增加附近的训练和验证曲线；
- 是否频繁触发梯度裁剪；
- 最终残差是否比 A3 差。

因为 A6 仍保留梯度裁剪，如果其原始梯度长期远大于阈值，实际训练方向可能被连续归一化。这反映的是“能量尺度与裁剪的组合效应”。

---

## A7：`A7_stable_no_gradient_clip`

A7 只取消 A3 中的全局梯度范数裁剪，其他设置全部不变。

它回答的问题是：

> 在损失已经无量纲化之后，梯度裁剪是否仍是稳定训练所必需的？

重点查看：

- `maximum_gradient_norm`；
- $K$ 从 1 增加到 2、3、4、5 时是否出现损失尖峰；
- 是否出现非有限目标、非有限梯度或参数；
- 是否在某个展开阶段提前发散。

如果 A7 与 A3 基本一致，说明当前问题中梯度裁剪主要是保险措施。如果 A7 在课程切换处明显恶化，则裁剪是稳定多步展开训练的重要组成部分。

---

## 7. 训练设置

默认最大 epoch 已调整为

```text
5000
```

展开深度课程同步按比例调整为：

| Epoch 范围 | 训练展开深度 $K$ |
|---:|---:|
| 1–1000 | 1 |
| 1001–2000 | 2 |
| 2001–3000 | 3 |
| 3001–4000 | 4 |
| 4001–5000 | 5 |

对应默认参数为：

```python
DEFAULT_EPOCHS = 5_000
DEFAULT_INITIAL_K = 1
DEFAULT_K_INCREASE_INTERVAL = 1_000
DEFAULT_K_INCREASE_AMOUNT = 1
DEFAULT_MAX_K = 5
```

所有组共享：

- `torch.float64`；
- `Adam(lr=1e-3)`；
- full batch；
- 网络宽度 64；
- 恒等激活函数；
- 输出层零初始化；
- 同一个模型随机种子；
- 同一组 Sobol 数据；
- 每 500 epoch 验证一次；
- 固定迭代 50 次进行验证和测试；
- 不早停，只由验证集选择最佳 checkpoint；
- 相同的插值测试、外推测试、当前物理状态测试和精确解固定点测试。

---

## 8. 数据集保持不变

100 个物理时间步被视为 100 个相互独立的优化问题。

问题索引划分为：

- 训练问题：60 个；
- 验证问题：10 个；
- 插值测试问题：10 个；
- 外推测试问题：20 个。

每个训练问题默认生成 100 个初值，每个验证或测试问题默认生成 256 个初值。

每个问题内部使用 scrambled Sobol 点，并进行粒子交换增强。八组实验共享完全相同的数据张量，不会为不同消融组重新随机采样。

网络预测不会从一个物理时间步传播到下一个物理时间步。因此实验测试的是跨独立问题的优化器泛化能力，而不是连续仿真的长期误差累积。

---

## 9. 训练目标

每个 epoch 将当前展开深度下的所有更新完整展开，并通过全部迭代反向传播。

A0 和 A2 使用未缩放的物理能量。

稳定版本使用

$$
\widehat L
=
\sum_{j=1}^{K}
\frac{E(y^{(j)})-\operatorname{stopgrad}(E(y^{(0)}))}{E_s}.
$$

其中：

- 减去初始能量只改变日志数值，不改变参数梯度；
- 除以正数 $E_s$ 不改变极小点和梯度方向，只改变梯度幅值；
- 精确解不参与训练目标；
- 精确解只用于生成数据、评价误差、诊断和 checkpoint 选择。

---

## 10. 评价指标

所有实验使用同一套物理评价指标。

### 10.1 驻点残差

$$
R(y)=\|\nabla E(y)\|_2.
$$

重点使用 50 次迭代后的 p95，而不是仅使用均值。

### 10.2 精确解误差

$$
e(y)=\|y-y^*\|_2.
$$

### 10.3 能量差

$$
\Delta E(y)=E(y)-E(y^*).
$$

### 10.4 固定点漂移

将精确解 $y^*$ 作为初值，重复执行网络更新，并测量最终

$$
\|y^{(50)}-y^*\|_2.
$$

该指标对 A4 尤其重要。

### 10.5 一步质量诊断

脚本还记录：

- 更新方向与理想修正 $y^*-y$ 的余弦；
- 一步后误差改善的样本比例；
- 一步后残差改善的样本比例；
- 一步后能量下降的样本比例；
- 一步误差收缩比；
- 精确解处网络输出更新的均值和最大值。

### 10.6 相对 A3 的数量级退化

CSV 自动计算

$$
D_R=
\log_{10}
\frac{
\max(R_{\mathrm{ablation}},10^{-14})
}{
\max(R_{\mathrm{A3}},10^{-14})
}.
$$

解释如下：

- $D_R=0$：与 A3 相同量级；
- $D_R=1$：比 A3 差约 1 个数量级；
- $D_R=4$：比 A3 差约 4 个数量级；
- 负值：该指标优于 A3，但仍需同时检查是否存在其他指标恶化。

---

## 11. 运行方法

### 11.1 运行全部八组实验

脚本默认设备为 `cuda:1`：

```bash
python multi_step_spring_ablation.py
```

显式指定设备：

```bash
python multi_step_spring_ablation.py --device cuda:1
```

在 CPU 上进行小规模调试：

```bash
python multi_step_spring_ablation.py --device cpu
```

### 11.2 只运行部分实验

例如只运行四个最关键的正向重建组：

```bash
python multi_step_spring_ablation.py \
  --experiments \
  A0_original_raw_state \
  A1_raw_state_stabilized \
  A2_residual_core_only \
  A3_stable_full
```

只检查固定点约束：

```bash
python multi_step_spring_ablation.py \
  --experiments A3_stable_full A4_stable_with_bias
```

只检查损失尺度与梯度裁剪：

```bash
python multi_step_spring_ablation.py \
  --experiments \
  A3_stable_full \
  A6_stable_no_energy_scale \
  A7_stable_no_gradient_clip
```

### 11.3 快速调试命令

下面的命令只用于检查代码流程，不用于正式结论：

```bash
python multi_step_spring_ablation.py \
  --device cpu \
  --epochs 2 \
  --k-increase-interval 1 \
  --max-k 2 \
  --train-points-per-problem 4 \
  --eval-points-per-problem 4 \
  --evaluation-steps 2 \
  --validation-interval 1 \
  --diagnostic-interval 1 \
  --experiments A0_original_raw_state A3_stable_full \
  --skip-plots \
  --skip-newton-baseline
```

### 11.4 其他常用选项

```text
--skip-plots
```

跳过所有绘图，但仍保存模型、JSON 和 CSV。

```text
--skip-newton-baseline
```

跳过共享的全牛顿法基线。

```text
--save-datasets
```

额外保存生成后的所有数据张量。

---

## 12. 输出文件

根输出目录包含：

```text
runtime_config.json
reference_time_step_problems.json
dataset_metadata.json
ablation_summary.csv
ablation_summary.json
all_experiments_summary.json
ablation_final_metrics.png
ablation_current_state_residual.png
ablation_and_newton_final_metrics.png
ablation_current_state_with_newton.png
```

每个实验目录包含：

```text
best_validation_model_state_dict.pt
last_model_state_dict.pt
mlp_optimizer_state_dict.pt
optimization_report.json
experiment_summary.json
training_and_validation_curves.png
interpolation_test_rollout_metrics.png
extrapolation_test_rollout_metrics.png
special_state_metrics_vs_physical_time.png
```

如果某组训练发散，脚本会记录：

- 是否发散；
- 发散 epoch；
- 非有限目标、非有限梯度、非有限参数或 CUDA 显存错误等原因。

一组实验发散不会改变其他实验的配置。

---

## 13. `ablation_summary.csv` 的关键列

### 配置列

```text
experiment
label
comparison_role
changed_factor
input_mode
use_bias
first_layer_initialization
output_scale_mode
shift_energy
use_energy_scale
use_gradient_clip
```

### 最终指标列

```text
interpolation_residual_p95
extrapolation_residual_p95
interpolation_exact_error_p95
extrapolation_exact_error_p95
interpolation_energy_gap_p95
extrapolation_energy_gap_p95
exact_state_final_residual_p95
exact_state_final_drift_p95
current_state_final_residual_p95
```

### 训练稳定性列

```text
gradient_clip_trigger_fraction
maximum_gradient_norm
diverged
divergence_epoch
best_epoch
```

### 相对 A3 的退化列

```text
interp_residual_log10_degradation_vs_A3
extra_residual_log10_degradation_vs_A3
fixed_point_drift_log10_degradation_vs_A3
```

只有运行结果中包含 A3 时，才能计算相对 A3 的退化列。

---

## 14. 推荐的结果判读顺序

### 第一步：判断状态表示是否是主因

首先比较：

```text
A0_original_raw_state
A1_raw_state_stabilized
A2_residual_core_only
A3_stable_full
```

主要逻辑为：

1. **A1 仍差、A3 成功**：状态表示是主因；
2. **A2 已明显优于 A0**：残差输入具有较强的独立贡献；
3. **A2 接近 A3**：其余改动主要负责稳定性和最终高精度；
4. **A1 也成功**：不能把成功完全归因于残差输入，需要进一步分析训练尺度和结构约束。

### 第二步：判断固定点约束是否必要

比较：

```text
A3_stable_full
A4_stable_with_bias
```

优先查看精确解初值的 50 步漂移，而不是只看普通测试初值的一步误差。

### 第三步：判断初始化的作用

比较：

```text
A3_stable_full
A5_stable_default_init
```

如果 A5 只是收敛较慢但最终接近 A3，说明正交初始化是训练加速项。

### 第四步：判断能量尺度与梯度裁剪的作用

比较：

```text
A3_stable_full
A6_stable_no_energy_scale
A7_stable_no_gradient_clip
```

重点观察每个 $K$ 切换点：epoch 1001、2001、3001 和 4001 附近。

---

## 15. 三个解释边界

### 15.1 当前线性无偏置网络中，输入和输出的 $s$ 会抵消

对 A3 而言，恒等激活和无偏置结构给出

$$
\Delta y
=
sW_2W_1
\left(
\frac{\Delta t^2M^{-1}g(y)}{s}
\right)
=
W_2W_1\Delta t^2M^{-1}g(y).
$$

因此，当前实验不能把输入无量纲化中的 $s$ 和输出乘 $s$ 视为两个独立的表达能力来源。加入非线性激活函数后，$s$ 才会改变网络工作区间。

### 15.2 能量平移不改变参数梯度

$$
E(y)-\operatorname{stopgrad}(E(y^{(0)}))
$$

与 $E(y)$ 具有相同的参数梯度。A6 真正删除的是除以 $E_s$，不是能量平移本身。

### 15.3 当前质量预条件主要表现为统一缩放

本实验固定

$$
m_1=m_2=1,
\qquad
\Delta t=0.01,
$$

因此

$$
\Delta t^2M_c^{-1}=10^{-4}I.
$$

在当前数据上，它主要是统一尺度变换，而不是不同粒子或不同坐标之间的非均匀预条件。若要独立验证质量预条件的作用，需要在后续实验中改变 $m_1$、$m_2$ 或 $\Delta t$。

---

## 16. 实验结论应如何表述

完成实验后，建议按以下层次组织结论：

1. **决定性改动**：删除后导致多个数量级退化，或从失败转为成功；
2. **结构性改动**：主要影响固定点、长迭代稳定性和最终数值精度；
3. **训练稳定化改动**：主要影响梯度、课程切换和收敛速度；
4. **辅助性改动**：最终性能接近，但训练速度或早期方向质量发生变化。

不要只根据训练 loss 的绝对数值比较不同组。A0、A2 与其他组的目标尺度不同，训练 loss 本身不在同一量纲和数量级上。最终结论应以统一评价器输出的物理残差、精确解误差、能量差和固定点漂移为准。
