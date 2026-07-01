# 多 Motion 双质点弹簧学习型优化器实验

## 1. 实验目标

本脚本把原来的一条参考 motion 扩展为 32 条完整 motion，用于测试同一个学习型迭代求解器能否泛化到不同的：

- 质心平移速度；
- 高速水平运动；
- 整体向上抛；
- 初始弹簧拉伸和压缩；
- 径向相对速度；
- 切向相对速度；
- 多种困难因素的组合。

每条 motion 都生成 100 个物理时间步，但每个时间步仍被视为独立优化问题。网络预测不会传播到下一个物理时间步。

网络结构、损失和训练方法保持原实验不变，因此本实验主要测量一个固定线性残差预条件器在不同 motion 之间的覆盖能力。

## 2. Motion 划分

32 条 motion 按完整轨迹划分：

| 数据部分 | Motion 数量 | Motion 编号 |
|---|---:|---|
| 训练集 | 16 | 0–15 |
| 验证集 | 4 | 16–19 |
| 域内测试集 | 4 | 20–23 |
| 域外测试集 | 8 | 24–31 |

划分单位是完整 motion。验证和测试 motion 的任何时间步都不会进入训练集。

训练 motion 中包括：

- 原始 motion；
- 7 条人工设计的可解释锚点 motion；
- 8 条训练参数域内的 Sobol motion。

验证集和域内测试集分别使用独立 Sobol seed 在相同参数域采样。域外测试集包含高速平移、强压缩、强拉伸、快速径向运动、快速切向运动和组合困难情况。

所有 motion 的具体参数会保存到：

```text
motion_catalogue.json
```

## 3. Motion 参数化

每条 motion 使用质心和相对坐标生成。设

$$
d^0 = \rho n,
$$

其中 $\rho$ 是初始弹簧长度，$n$ 是单位方向。相对速度写成

$$
w = w_{\parallel}n+w_{\perp,1}t_1+w_{\perp,2}t_2.
$$

根据质量恢复两个质点的初始状态：

$$
p_1^0=c^0-\frac{m_2}{m_1+m_2}d^0,
\qquad
p_2^0=c^0+\frac{m_1}{m_1+m_2}d^0,
$$

$$
v_1^0=V-\frac{m_2}{m_1+m_2}w,
\qquad
v_2^0=V+\frac{m_1}{m_1+m_2}w.
$$

这种方式可以明确区分整体平移和真正改变弹簧求解结构的相对运动。

## 4. 时间步划分

训练 motion 的 100 个时间步进一步划分为：

- 16 个训练时间步：`TRAIN_TIME_INDICES`；
- 16 个已见 motion 的时间插值测试时间步；
- 第 80–99 帧作为已见 motion 的时间外推测试。

未见验证 motion 使用 10 个分层时间步。域内测试和域外测试 motion 使用每隔 5 帧采样的 20 个时间步。

因此最终分别报告：

1. 已见 motion、未见时间步的插值；
2. 已见 motion、后续时间步的外推；
3. 未见 motion、训练参数范围内的泛化；
4. 未见 motion、参数范围外的 OOD 泛化。

## 5. 训练数据预算

默认多 motion 训练集大小为

$$
16\ \text{motions}
\times 16\ \text{time steps}
\times 32\ \text{states}
=8192\ \text{states}.
$$

脚本同时训练一个等样本预算的单 motion 基线：

$$
1\ \text{motion}
\times16\ \text{time steps}
\times512\ \text{states}
=8192\ \text{states}.
$$

这样可以避免把“训练状态数量更多”误判成“motion 多样性有效”。

## 6. 初值采样

每个时间步仍以精确解为中心进行六维 Sobol 扰动和粒子交换增强。

原始采样半径为

$$
R_n^{\mathrm{raw}}=\lVert p^n-y_n^*\rVert_{\infty}.
$$

实际使用

$$
R_n=\operatorname{clip}
\left(R_n^{\mathrm{raw}},0.01,0.10\right).
$$

训练集中始终显式加入当前物理状态 $p^n$ 和精确状态 $y_n^*$。Sobol 点使用每个数据集共享的一条 scrambled Sobol 流，避免为数百个时间步反复初始化 Sobol 引擎。

## 7. 网络和训练设置

网络输入为无量纲质量预条件残差：

$$
u=\frac{\Delta t^2M^{-1}\nabla E(y)}{s}.
$$

网络输出映射为物理位置更新：

$$
\Delta y=s\,\mathrm{MLP}(u).
$$

默认配置：

- `torch.float64`；
- `cuda:1`；
- 无偏置网络；
- `Linear(6, 64) -> Identity -> Linear(64, 6)`；
- 第一层正交初始化；
- 输出层零初始化；
- Adam，学习率 $10^{-3}$；
- full batch；
- 50000 epochs；
- 展开步数 $K=1\rightarrow5$，每 10000 epochs 增加一次；
- 全局梯度范数裁剪阈值 1；
- 使用移位和正尺度归一化后的原始物理变分能量训练。

由于没有非线性激活函数，网络整体仍等价于一个固定的 $6\times6$ 线性残差预条件器。

## 8. Checkpoint 选择

Checkpoint 只使用 4 条未见验证 motion 选择。排序指标依次为：

1. 非有限 residual 数量；
2. 最差验证 motion 的最终 residual p95；
3. 验证集 pooled residual p95；
4. 最差验证 motion 的 exact-error p95；
5. pooled exact-error p95；
6. pooled energy-gap p95。

训练不会早停。最后一轮权重仍会保存，但完整测试只对验证集选出的最佳 checkpoint 执行，以避免把多 motion 测试成本重复一遍。

## 9. 评价输出

主要测试集均报告：

- residual；
- energy gap；
- exact-solution error；
- 两个质点各自的位置误差；
- mean、median、p95 和 max；
- 逐 motion 统计；
- 逐 motion 类别统计；
- 第 1、5、10、50 次求解器迭代结果；
- full Newton 对照；
- 当前物理状态测试；
- 精确解 fixed-point 测试。

脚本还保存每条 motion 的初始误差、初始 residual、Hessian 条件数和采样半径统计。

## 10. 运行方法

默认运行：

```bash
python multi_motion_spring_optimizer.py
```

默认使用 `cuda:1`。改为 `cuda:0`：

```bash
python multi_motion_spring_optimizer.py --device cuda:0
```

只训练多 motion 模型，不运行等预算单 motion 基线：

```bash
python multi_motion_spring_optimizer.py --skip-single-motion-baseline
```

不绘图：

```bash
python multi_motion_spring_optimizer.py --skip-plots
```

保存所有生成的数据张量：

```bash
python multi_motion_spring_optimizer.py --save-datasets
```

快速检查代码是否可运行：

```bash
python multi_motion_spring_optimizer.py \
  --device cpu \
  --epochs 1 \
  --validation-interval 1 \
  --evaluation-steps 1 \
  --report-steps 1 \
  --train-points-per-problem 4 \
  --eval-points-per-problem 4 \
  --diagnostic-interval 1 \
  --skip-plots
```

## 11. 输出目录

输出目录与脚本同名。例如脚本名为

```text
multi_motion_spring_optimizer.py
```

输出目录为

```text
multi_motion_spring_optimizer/
```

核心文件包括：

```text
runtime_config.json
motion_catalogue.json
motion_difficulty_statistics.json
reference_time_step_problems.json
dataset_metadata.json
all_experiments_summary.json
newton_baseline/
multi_motion/
single_motion_equal_budget_baseline/
```
