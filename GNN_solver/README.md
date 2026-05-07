# GNN Solver for Implicit Euler Cloth Simulation

本目录用于测试 **GNN 是否可以作为隐式欧拉变分能量优化问题的迭代求解器**。

当前项目中有两个相关模块：

1. `cloth_simulation_newton`（待整理）
   传统牛顿法求解器，用作物理仿真与数值优化的参考实现。

2. `GNN_solver`（正在测试）
   GNN 迭代求解器实验代码。当前主要测试都在：

```text
/repo/GNN_solver/_src
```

本 README 只关注 `GNN_solver` 目录下的内容。

---

## 1. 研究目标

当前核心问题是：

> GNN 能否作为隐式欧拉时间步中变分能量优化问题的迭代求解器？

具体来说，给定当前时间步的状态：

* 上一时间步位置 `x_prev`
* 上一时间步速度 `v_prev`
* 惯性预测位置 `x_hat`
* 当前迭代位置 `x_cur`
* 网格拓扑 `edge_index`, `face_index`
* 材料参数与质量信息

希望 GNN 在每一次 solver iteration 中预测一个位置增量：

```python
delta_x = solver(...)
x_next = x_cur + delta_x
```

然后将 `x_next` 代入隐式欧拉能量：

```python
losses = ImplicitEulerLoss.forward(
    x=x_next,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
)
```

训练目标是让 GNN 的迭代更新逐步降低隐式欧拉变分能量，并降低对应 residual。

---

## 2. 当前阶段目标

当前主要测试以下问题：

### 2.1 预训练形式是否可行

当前优先验证：

* GNN 只迭代一次时，是否能学到合理的下降方向；
* 从不同初值出发时是否都能降低能量：

  * `x_prev`
  * `x_hat`
* 评估时连续迭代 15 次，观察 loss 与 residual 是否稳定下降。

### 2.2 数值精度影响

需要测试：

* `float32`
* `float64`

隐式欧拉能量与 residual 对数值精度比较敏感，后续需要比较双精度是否明显提升训练稳定性。

切换到 `float64` 时需要同时检查：

* 训练脚本中的 `dtype`
* `GNN_solver.py` 中是否存在强制 `.float()` 或 `torch.float32`
* `loss_class.py` 中是否存在强制 `float32` 的张量构造

推荐让网络 forward 中的输入 dtype 跟随模型参数 dtype，而不是写死 `torch.float32`。

### 2.3 MLP 替换 GNN

为了确认 GNN 结构本身是否必要，需要做一个 ablation：

* 保持输入输出接口不变；
* 将 message passing GNN 替换为普通 MLP；
* 对比同样训练方式下的 loss / residual 曲线。

如果 MLP 也能完成当前 toy problem，需要进一步提高任务难度或改进输入特征，以验证图结构信息是否真正发挥作用。

### 2.4 输入特征是否需要调整

当前 GNN 输入特征为暂定方案，后续需要通过实验判断是否足够。

节点特征当前包括：

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

边特征当前包括：

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

后续可能需要测试的特征包括：

* `x_cur - x_hat`
* `x_cur - x_prev`
* 当前速度估计
* 外力或重力编码
* inverse mass
* pinned / boundary type one-hot
* edge strain 或 relative stretch
* rest length normalized edge vector
* residual 或 gradient-like 信息

### 2.5 碰撞能量测试

如果当前无碰撞能量的隐式欧拉优化实验可行，下一步加入碰撞能量项，测试 GNN 迭代器是否仍然可以降低总能量与 residual。

---

## 3. 当前 GNN 求解器设计

### 3.1 输出语义

GNN 不直接输出新位置，而是输出单次迭代的位置增量：

```python
delta_x = model(...)
x_next = x_cur + delta_x
```

GNN 本身不负责：

* 时间步推进；
* 计算 `x_hat`；
* 多次迭代循环；
* pinned vertex clamp；
* loss 计算。

这些逻辑都放在训练脚本或外部 solver wrapper 中。

---

### 3.2 网络结构

当前网络参考 MeshGraphNets 的 Encode-Process-Decode 结构：

```text
node features ──► node encoder ──► node latent
                                      │
                                      ▼
edge features ──► edge encoder ──► 15 GraphNetBlock ──► node decoder ──► delta_x
```

默认超参数：

```python
latent_size = 128
num_layers = 2
message_passing_steps = 15
output_dim = 3
```

其中：

* `node_encoder`: 将节点输入特征编码到 128 维 latent space；
* `edge_encoder`: 将边输入特征编码到 128 维 latent space；
* `processor`: 15 层 message passing；
* `decoder`: 只解码节点 latent，输出 `delta_x: [N, 3]`。

---

### 3.3 MLP 结构

当前 MLP 与 MeshGraphNets 风格保持一致：

```python
Linear -> ReLU -> ... -> Linear -> optional LayerNorm
```

约定：

* encoder 与 processor 内部 MLP 使用 `LayerNorm`；
* decoder 不使用 `LayerNorm`；
* 不使用 OnlineNormalizer；
* 不对输入特征做统计归一化。

这里的“不归一化”指：不使用 running mean / running std 的输入输出 normalizer。
不是指移除 MLP 内部的 `LayerNorm`。

---

### 3.4 残差连接

每个 GraphNetBlock 的结构为：

```text
edge update -> node update -> residual add
```

边更新：

```python
new_edge_attr = edge_model(src, dst, edge_attr)
edge_attr = edge_attr + new_edge_attr
```

点更新：

```python
agg = scatter(edge_attr, receivers, reduce="sum")
new_x = node_model(torch.cat([x, agg], dim=-1))
x = x + new_x
```

注意这里有两层残差：

1. **网络内部 latent residual**

```python
edge_attr = edge_attr + new_edge_attr
x = x + new_x
```

2. **外部物理位置 residual**

```python
x_next = x_cur + delta_x
```

二者发生在不同空间，前者在 latent feature space，后者在 3D position space。

---

### 3.5 参数共享方式

当前参数共享方式为：

* 同一个 `edge_mlp` 作用于所有边，边之间共享参数；
* 同一个 `node_mlp` 作用于所有点，点之间共享参数；
* 不同 message passing step 之间不共享参数；
* 外部多次 solver iteration 调用的是同一个 GNN，因此外部迭代之间共享整套网络参数。

换句话说：

```text
15 个 GraphNetBlock 之间不共享参数；
训练或测试中的多次 solver iteration 共享同一个完整 GNN。
```

---

## 4. 当前训练目标

当前训练不是监督学习，没有 ground truth `x_next`。

训练目标来自物理能量：

```python
losses = ImplicitEulerLoss.forward(
    x=x_pred,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
)
loss = losses["total"]
```

其中 `losses` 通常包含：

* `total`
* `inertia`
* `gravity`
* `elastic`
* `bending`

同时使用：

```python
res = loss_obj.residual(
    x=x_pred,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
)
```

记录：

* `residual_mean`
* `residual_max`

---

## 5. 当前训练方式

目前推荐的最小训练方式为：

```text
每个时间步只迭代 1 次；
每个时间步反向传播 1 次；
训练 500 epoch；
每个 epoch 都评估一次。
```

即：

```python
train_iters = 1
backward_mode = "iteration"
num_epochs = 500
test_every = 1
```

训练时每个 epoch 会分别从两个初值出发：

```python
initial_states = {
    "x_prev": x_prev,
    "x_hat": x_hat,
}
```

因此每个 epoch 默认有两个 optimizer step：

```text
x_prev 起步：1 step
x_hat  起步：1 step
```

训练更新逻辑：

```python
for x_init in [x_prev, x_hat]:
    x_cur = x_init.clone()

    delta_x = solver(...)
    x_next = x_cur + delta_x
    x_next = clamp_pinned_vertices(x_next, x_prev, pinned_idx)

    losses = loss_obj.forward(
        x=x_next,
        x_prev=x_prev,
        v_prev=v_prev,
        dt=dt,
    )

    loss = losses["total"]
    loss.backward()
    optimizer.step()
```

---

## 6. 评估方式

每次评估时，从两个初值分别起步：

* `x_prev`
* `x_hat`

每个初值都记录第 0 次迭代，也就是直接使用初值计算的 loss 和 residual：

```text
iter = 0
```

然后连续运行 15 次 GNN 迭代：

```text
iter = 1, 2, ..., 15
```

每次迭代后记录：

```text
phase, epoch, init_name, iter,
total_loss, inertia, gravity, elastic, bending,
residual_mean, residual_max
```

加入 `iter=0` 的原因是：可以直接对比 GNN 更新前后的能量和 residual 是否下降。

---

## 7. 日志文件

当前建议保留两类日志。

### 7.1 评估日志

文件名示例：

```text
exp_min500_one_iter_one_backward_eval_every_epoch_eval_log.csv
```

字段：

```text
phase, epoch, init_name, iter,
total_loss, inertia, gravity, elastic, bending,
residual_mean, residual_max
```

### 7.2 训练 loss 日志

文件名示例：

```text
exp_min500_one_iter_one_backward_eval_every_epoch_train_loss_log.csv
```

字段：

```text
phase, epoch, backward_mode, train_iters, optimizer_steps, mean_train_loss
```

---

## 8. 推荐运行方式

在 `GNN_solver/_src` 目录下运行：

```bash
python train_min500_one_iter_one_backward.py --device cuda
```

如果没有 GPU，也可以：

```bash
python train_min500_one_iter_one_backward.py --device cpu
```

如果使用自动选择：

```bash
python train_min500_one_iter_one_backward.py --device auto
```

---

## 9. 当前需要重点观察的结果

训练和评估时重点检查：

1. `iter=0` 到 `iter=1` 是否下降；
2. `iter=1` 到 `iter=15` 是否持续下降；
3. 从 `x_prev` 起步和从 `x_hat` 起步是否表现一致；
4. `total_loss` 是否下降但 residual 不下降；
5. residual 是否下降但能量项异常；
6. 是否出现 NaN / Inf；
7. `float32` 与 `float64` 是否有明显差异；
8. 训练 loss 与评估 loss 是否趋势一致。

---

## 10. 后续实验计划

### 10.1 双精度测试

将整个任务切换到 `float64`：

* data tensor 使用 `torch.float64`
* model 使用 `.to(dtype=torch.float64)`
* loss 内部不应强制转换为 `float32`
* GNN forward 不应写死 `torch.float32`

目标是检查：

* loss 曲线是否更平滑；
* residual 是否更稳定下降；
* 是否减少 NaN / Inf。

### 10.2 MLP baseline

实现一个接口与 GNN 相同的 MLP solver：

```python
delta_x = mlp_solver(
    x_cur=x_cur,
    x_hat=x_hat,
    rest_pos=rest_pos,
    edge_index=edge_index,
    mass=mass,
    mu_lame=mu_lame,
    lambda_lame=lambda_lame,
    k_bending=k_bending,
    dt=dt,
    pinned_idx=pinned_idx,
)
```

但内部不做 message passing，仅基于节点局部特征预测 `delta_x`。

用于判断：

* 当前 toy problem 是否过于简单；
* GNN 是否真的利用了图结构和边特征；
* 输入特征是否已经泄露了足够信息，使 MLP 也能解决问题。

### 10.3 输入特征 ablation

建议分组测试：

1. 基础特征：`x_cur`, `x_hat`, material, mass, dt, pinned；
2. 加入 `x_cur - x_hat`；
3. 加入 `x_cur - x_prev`；
4. 加入 edge stretch；
5. 加入 normalized rest edge direction；
6. 加入 inverse mass；
7. 加入 force / residual-like feature。

### 10.4 多步自回归训练

如果单步训练可行，可以重新测试：

* 每个时间步迭代 10 次，每次 iteration 反向一次；
* 每个时间步 unroll 10 次，只在最终位置反向一次；
* 先单步预训练，再多步 fine-tune。

### 10.5 碰撞能量

如果无碰撞版本能稳定降低 loss 和 residual，下一步加入 collision energy：

* 点-面碰撞；
* 边-边碰撞；
* barrier energy；
* contact stiffness / thickness 参数；
* 碰撞 residual。

目标是测试 GNN 迭代器在更非线性、更局部、更刚性的能量项下是否仍能稳定工作。

---

## 11. 已知注意事项

1. 当前训练集只有一个时间步和两个初值，容易过拟合。
   当前目标不是泛化，而是先验证 GNN 是否能作为能量下降迭代器。

2. `x_prev` 与 `x_hat` 在某些 toy 设置下可能非常接近。
   如果 `v_prev = 0`，则二者完全相同。当前建议使用非零初速度以区分两个初值。

3. 如果 `pinned_idx = None`，则不做 pinned vertex clamp。
   如果后续开启 pinned vertices，需要同时保证输入特征中 `pinned_flag` 正确。

4. 如果切换到 `float64`，需要检查所有模块是否一致使用双精度。
   dtype 混用可能导致报错或隐式转换。

5. 评估时的 `iter=0` 是未经过 GNN 更新的初值参考。
   它不是训练迭代，只用于观察更新前后的变化。

---

# Codex 后续整理提示词

下面这段提示词可以交给 Codex，让它基于完整项目文件继续整理 `GNN_solver` 目录。

```text
你现在可以阅读整个 GitHub 项目，但请只关注 GNN_solver 目录，尤其是 GNN_solver/_src 下的代码。

任务：整理并完善 GNN_solver 目录下的 README 和实验入口，使其准确反映当前代码状态。

背景：
这个目录用于测试 GNN 是否可以作为 cloth simulation 中隐式欧拉变分能量优化问题的迭代求解器。GNN 不直接预测最终位置，而是每次 solver iteration 预测 delta_x，然后外部做 x_next = x_cur + delta_x。loss 使用 ImplicitEulerLoss，包括 inertia、gravity、elastic、bending 等能量项，并通过 residual 评估求解质量。

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
7. 在 README 中保留以下结构：
   - 研究目标
   - 当前网络结构
   - 输入特征
   - 训练方式
   - 评估方式
   - 日志文件
   - 推荐运行命令
   - 当前实验计划
   - 已知注意事项
8. 不要修改 cloth_simulation_newton 目录；只整理 GNN_solver 目录相关内容。
9. 如有必要，可以添加一个最小入口脚本，要求：
   - 训练 500 epoch；
   - 每个 epoch 评估一次；
   - 每个时间步训练时只迭代 1 次并 backward 1 次；
   - 评估时记录 iter=0 到 iter=15；
   - 支持 --device cuda / cpu / auto；
   - 日志文件名不要覆盖其他实验。
10. 最后输出你修改了哪些文件，以及哪些地方还需要人工确认。
```
