# Cloth Simulation Solver Experiments

本项目用于实验和整理布料仿真中的隐式欧拉求解器，包括传统牛顿法求解器和正在测试的 GNN 迭代求解器。

当前项目主要包含两个方向：

1. `cloth_simulation_newton`（待整理）
   传统牛顿法求解器，用作隐式欧拉变分能量优化问题的数值参考。

2. `GNN_solver`（正在测试）
   GNN 求解器实验代码，用于测试 GNN 是否可以作为隐式欧拉变分能量优化问题的迭代求解器。

目前主要测试代码位于：

```text
GNN_solver/_src
```

---

## 1. 项目目标

本项目的核心目标是：

> 测试 GNN 是否可以作为布料仿真中隐式欧拉变分能量优化问题的迭代求解器。

在隐式欧拉时间步中，传统方法通常需要求解一个变分能量优化问题。当前实验尝试使用神经网络在每次 solver iteration 中预测位置增量：

```python
delta_x = solver(...)
x_next = x_cur + delta_x
```

然后使用隐式欧拉能量作为训练目标：

```python
losses = ImplicitEulerLoss.forward(
    x=x_next,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
)
```

当前阶段不是直接追求泛化能力，而是先验证：

* GNN 是否能在单个时间步上学习到能量下降方向；
* 从不同初值出发时，GNN 迭代是否能稳定降低能量和 residual；
* GNN 是否有可能替代或辅助传统迭代求解器。

---

## 2. 当前目录结构

```text
repo/
├── cloth_simulation_newton/
│   └── 传统牛顿法求解器，待进一步整理
│
├── GNN_solver/
│   └── _src/
│       ├── GNN_solver.py
│       ├── loss_class.py
│       ├── train_common.py
│       ├── train_min500_one_iter_one_backward.py
│       └── 其他实验脚本
│
└── README.md
```

说明：

* `cloth_simulation_newton` 当前作为传统求解器参考；
* `GNN_solver` 是当前主要开发与测试目录；
* 当前训练、评估和日志实验主要在 `GNN_solver/_src` 下进行。

---

## 3. 当前 GNN 求解器思路

GNN 不直接预测最终位置，而是预测一次迭代的位置增量：

```python
delta_x = model(...)
x_next = x_cur + delta_x
```

GNN 本身不负责：

* 时间步推进；
* 计算 `x_hat`；
* 多次 solver iteration 循环；
* pinned vertices clamp；
* loss 计算。

这些逻辑由训练脚本或外部 solver wrapper 负责。

---

## 4. 当前网络结构概览

`GNN_solver` 当前使用类似 MeshGraphNets 的 Encode-Process-Decode 结构：

```text
node features ──► node encoder ──► node latent
                                      │
                                      ▼
edge features ──► edge encoder ──► processor ──► node decoder ──► delta_x
```

当前默认设置：

```python
latent_size = 128
message_passing_steps = 15
output_dim = 3
```

其中：

* `node_encoder` 编码节点特征；
* `edge_encoder` 编码边特征；
* `processor` 进行多轮 message passing；
* `decoder` 只解码节点特征，输出 `delta_x`。

当前网络保留 MLP 内部的 `LayerNorm`，但不使用输入输出的统计归一化器，例如 running mean / running std normalizer。

---

## 5. 当前输入特征设计

当前节点特征暂定为：

```python
node_feat = torch.cat([
    x_cur,        # 当前迭代位置
    x_hat,        # 惯性预测位置
    mass,         # nodal mass
    mu_lame,      # local material parameter
    lambda_lame,  # local material parameter
    k_bending,    # local bending stiffness
    dt,           # time step size
    pinned_flag,  # whether node is pinned
], dim=-1)
```

当前边特征暂定为：

```python
edge_feat = torch.cat([
    x_i - x_j,
    rest_i - rest_j,
    ||x_i - x_j||,
    ||rest_i - rest_j||,
    avg(mu_lame_i, mu_lame_j),
    avg(lambda_lame_i, lambda_lame_j),
    avg(k_bending_i, k_bending_j),
    dt,
], dim=-1)
```

后续需要通过实验判断这些特征是否足够。

---

## 6. 当前训练目标

当前训练方式是物理无监督训练，不使用 ground truth `x_next`。

训练 loss 来自隐式欧拉能量：

```python
losses = ImplicitEulerLoss.forward(
    x=x_pred,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
)
loss = losses["total"]
```

当前记录的 loss 项包括：

* `total`
* `inertia`
* `gravity`
* `elastic`
* `bending`

同时计算 residual 指标：

```python
res = loss_obj.residual(
    x=x_pred,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
)
```

主要观察：

* `residual_mean`
* `residual_max`

---

## 7. 当前最小训练实验

当前推荐的最小实验为：

```text
每个时间步只迭代 1 次；
每个时间步反向传播 1 次；
训练 500 epoch；
每个 epoch 都评估一次。
```

对应配置：

```python
num_epochs = 500
train_iters = 1
backward_mode = "iteration"
test_every = 1
```

训练时从两个初值分别出发：

```python
initial_states = {
    "x_prev": x_prev,
    "x_hat": x_hat,
}
```

因此每个 epoch 默认包含两个 optimizer step：

```text
x_prev 起步：1 step
x_hat  起步：1 step
```

---

## 8. 评估方式

每次评估时，从两个初值分别起步：

* `x_prev`
* `x_hat`

每个初值先记录第 0 次迭代，也就是不经过 GNN 更新，直接计算初值本身的 loss 和 residual：

```text
iter = 0
```

然后连续运行 15 次 GNN 迭代：

```text
iter = 1, 2, ..., 15
```

每次记录：

```text
phase, epoch, init_name, iter,
total_loss, inertia, gravity, elastic, bending,
residual_mean, residual_max
```

加入 `iter=0` 的原因是方便比较 GNN 更新前后是否真的降低能量和 residual。

---

## 9. 日志文件

当前建议保存两类日志。

### 9.1 评估日志

示例文件名：

```text
exp_min500_one_iter_one_backward_eval_every_epoch_eval_log.csv
```

字段：

```text
phase, epoch, init_name, iter,
total_loss, inertia, gravity, elastic, bending,
residual_mean, residual_max
```

### 9.2 训练 loss 日志

示例文件名：

```text
exp_min500_one_iter_one_backward_eval_every_epoch_train_loss_log.csv
```

字段：

```text
phase, epoch, backward_mode, train_iters, optimizer_steps, mean_train_loss
```

---

## 10. 推荐运行方式

进入当前主要测试目录：

```bash
cd GNN_solver/_src
```

运行最小实验：

```bash
python train_min500_one_iter_one_backward.py --device cuda
```

如果没有 GPU：

```bash
python train_min500_one_iter_one_backward.py --device cpu
```

自动选择设备：

```bash
python train_min500_one_iter_one_backward.py --device auto
```

---

## 11. 当前重点实验问题

当前阶段主要关注以下问题。

### 11.1 预训练形式是否可行

需要测试：

* 单次 GNN 迭代是否能降低能量；
* 从 `x_prev` 和 `x_hat` 起步是否都有效；
* 连续迭代 15 次时，loss 和 residual 是否稳定下降。

### 11.2 是否需要双精度

隐式欧拉能量和 residual 可能对数值精度敏感。需要测试：

* `float32`
* `float64`

切换到 `float64` 时需要检查：

* 训练脚本中的 `dtype`
* GNN forward 中是否存在强制 `torch.float32` 或 `.float()`
* loss 内部是否有强制 `float32` 的张量构造

### 11.3 MLP baseline

需要实现一个 MLP baseline，用于替换 GNN：

* 保持相同输入输出接口；
* 不做 message passing；
* 只基于节点局部特征预测 `delta_x`。

目的：判断当前 toy problem 是否过于简单，以及 GNN 是否真正利用了图结构。

### 11.4 输入特征是否需要调整

后续可能测试的特征包括：

* `x_cur - x_hat`
* `x_cur - x_prev`
* 当前速度估计
* inverse mass
* 外力或重力编码
* pinned / boundary type one-hot
* edge strain 或 relative stretch
* rest length normalized edge vector
* residual 或 gradient-like 信息

### 11.5 碰撞能量测试

如果当前无碰撞版本可以稳定降低 loss 和 residual，下一步加入碰撞能量项，例如：

* 点-面碰撞；
* 边-边碰撞；
* barrier energy；
* contact thickness / stiffness。

目标是测试 GNN 迭代器在更非线性、更局部、更刚性的能量项下是否仍然有效。

---

## 12. 当前需要重点观察的结果

训练和评估时建议重点检查：

1. `iter=0` 到 `iter=1` 是否下降；
2. `iter=1` 到 `iter=15` 是否持续下降；
3. 从 `x_prev` 起步和从 `x_hat` 起步是否表现一致；
4. `total_loss` 是否下降但 residual 不下降；
5. residual 是否下降但某些能量项异常；
6. 是否出现 NaN / Inf；
7. `float32` 与 `float64` 是否有明显差异；
8. 训练 loss 与评估 loss 是否趋势一致。

---

## 13. 已知注意事项

1. 当前训练集只有一个时间步和两个初值，容易过拟合。
   当前目标不是泛化，而是先验证 GNN 是否能作为能量下降迭代器。

2. 如果 `v_prev = 0`，则 `x_hat == x_prev`。
   为了区分两个初值，当前建议使用非零初速度。

3. 如果 `pinned_idx = None`，则不做 pinned vertex clamp。
   如果后续开启 pinned vertices，需要同时保证输入特征中 `pinned_flag` 正确。

4. 如果切换到 `float64`，需要保证数据、模型和 loss 内部张量 dtype 一致。

5. 评估时的 `iter=0` 是未经过 GNN 更新的初值参考。
   它不是训练迭代，只用于观察更新前后的变化。

---

## 14. 后续整理计划

后续建议逐步整理：

1. 将 `GNN_solver/_src` 下的实验脚本归类；
2. 为 `GNN_solver` 单独补充更详细的子目录 README；
3. 将 toy problem、训练配置和日志路径参数化；
4. 添加 MLP baseline；
5. 添加 float64 配置开关；
6. 对齐 `cloth_simulation_newton` 中的牛顿法参考结果；
7. 加入碰撞能量测试。

---

## 15. Codex 后续整理提示词

如果需要让 Codex 基于完整项目继续整理，可以使用以下提示词：

```text
你现在可以阅读整个 GitHub 项目。请重点关注 GNN_solver 目录，尤其是 GNN_solver/_src 下的代码；不要修改 cloth_simulation_newton，除非只是为了理解接口。

任务：整理并完善 GNN_solver 相关文档和入口脚本，使其准确反映当前代码状态。

背景：
本项目用于测试 GNN 是否可以作为 cloth simulation 中隐式欧拉变分能量优化问题的迭代求解器。GNN 不直接预测最终位置，而是每次 solver iteration 预测 delta_x，然后外部做 x_next = x_cur + delta_x。loss 使用 ImplicitEulerLoss，包括 inertia、gravity、elastic、bending 等能量项，并通过 residual 评估求解质量。

请完成以下工作：

1. 阅读 GNN_solver/_src 下的所有文件，确认当前实际文件名、类名和入口脚本名称。
2. 检查 GNN_solver.py 中的网络结构：
   - 是否使用 MeshGraphNets 风格的 Encode-Process-Decode；
   - 是否使用 PyG MetaLayer；
   - MLP 是否为 Linear/ReLU/.../Linear + optional LayerNorm；
   - encoder / processor / decoder 的 LayerNorm 设置；
   - message_passing_steps 是否为 15；
   - latent_size 是否为 128；
   - 输出是否为 delta_x。
3. 检查训练脚本：
   - 当前是否采用每个时间步只迭代 1 次、反向 1 次的最小训练方式；
   - 是否训练 500 epoch；
   - 是否每个 epoch 都评估；
   - 评估是否记录 iter=0,1,...,15；
   - 是否从 x_prev 和 x_hat 两个初值分别评估；
   - 是否分别保存 eval log 和 train loss log。
4. 检查 dtype：
   - 是否有硬编码 torch.float32 或 .float()；
   - 如果要支持 float64，应该在哪里修改；
   - README 中要说明如何切换双精度。
5. 检查 loss_class.py 或相关 loss 文件：
   - 是否与 GNN_solver 的 dtype/device 一致；
   - 是否存在内部强制 float32 的张量。
6. 如果代码与 README 描述不一致，请优先以代码为准修改 README。
7. 建议在根目录 README 中保留项目整体说明，在 GNN_solver 子目录 README 中保留更详细的网络结构、训练细节和实验说明。
8. 如有必要，可以添加一个最小入口脚本，要求：
   - 训练 500 epoch；
   - 每个 epoch 评估一次；
   - 每个时间步训练时只迭代 1 次并 backward 1 次；
   - 评估时记录 iter=0 到 iter=15；
   - 支持 --device cuda / cpu / auto；
   - 日志文件名不要覆盖其他实验。
9. 最后输出你修改了哪些文件，以及哪些地方还需要人工确认。
```
