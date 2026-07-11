# Cloth 5x5 的 Metamizer 风格 Pool 训练

这个实验比较两种训练方式：

- 现有的 500-step 数据集训练流程。
- Metamizer 风格的在线训练 pool。

核心变化是：训练不再使用完整的 500-step 训练数据集。Pool 只从训练 motion 的初始状态初始化，然后让模型在自己的在线 rollout 状态分布上训练。

## 文件

```text
cloth13_train_metamizer_pool_models.py
cloth14_evaluate_pool_vs_existing_rollouts.py
```

- `cloth13`：从在线 pool 中训练 learned optimizer。
- `cloth14`：用连续 500 帧 rollout、每帧 50 次 inner iteration，对比 pool 训练模型和已有模型。

原来的 `cloth10` 到 `cloth12` 初始点采样数量消融实验保持不变。

## 训练语义

对每个训练 motion，pool 会创建五个环境：

```text
iterations_per_timestep = 1, 3, 5, 10, 30
```

默认有 16 个训练 motion，所以一共是：

```text
16 个 motion x 5 个 K-bucket = 80 个在线环境
```

一次参数更新的含义是：对每个在线环境都做一次 learned optimizer 更新：

```text
optimizer.step() == 一次神经网络更新，不是一次物理步
```

某个 K-bucket 只有在完成 K 次 learned update 后，才会推进一次物理环境：

```text
K = 1  -> 每个 epoch 1000 个物理步
K = 3  -> 每个 epoch 333 个物理步
K = 5  -> 每个 epoch 200 个物理步
K = 10 -> 每个 epoch 100 个物理步
K = 30 -> 每个 epoch 33 个物理步
```

默认训练计划是：

```text
epochs            = 50
updates_per_epoch = 1000
total updates     = 50,000
```

## 状态更新规则

learned optimizer 的更新形式和已有的 75D full-state 模型一致：

```text
input  = [current residual, previous residual, previous update]
output = 75D full-state displacement update
fixed vertices are gated/projected after the update
```

当一个环境完成自己的 K 次 inner update 后，物理状态按下面方式推进：

```text
x_{n+1} = y
v_{n+1} = (x_{n+1} - x_n) / dt
```

下一帧物理计算使用：

```text
y^(0) = x_n
```

这和 `cloth12` 的连续 rollout evaluator 保持一致。

## Loss

Pool 训练里的 loss 被故意设计得很简单：

```text
loss = mean(variational_energy_full(y_after_one_update, q, masses)) / physical_energy_scale
```

这里没有：

```text
no exact_y
no energy - exact_energy
no K-step unroll
no K-step average loss
```

也就是说，它不依赖精确解 `exact_y`，不最小化相对精确能量差，不做 K 步展开，也不对 K 步 loss 求平均。

## Pool reset 检查

每次 pool update 都会检查异常状态。只要对应环境出现坏状态，就把它 reset 回该 motion 的初始状态。

默认 reset 触发条件：

```text
non-finite y / energy / residual
abs(energy) > 1e8
residual > 1e8
max abs position > 1e3
min spring length < 1e-8
max spring length > 1e3
physical age >= 500 steps
```

训练日志会按原因记录 reset 次数。

## 命令

从 `cloth_5x5_500step_project/` 目录运行。

### Smoke test

```bash
python cloth13_train_metamizer_pool_models.py \
  --source-root cloth_5x5_500step_pipeline \
  --pool-root cloth_5x5_metamizer_pool_training \
  --activations identity \
  --epochs 2 \
  --updates-per-epoch 20 \
  --validation-interval 1 \
  --validation-rollout-length 5 \
  --validation-inner-steps 3 \
  --device cuda:0 \
  --overwrite
```

### 正式 pool 训练

```bash
python cloth13_train_metamizer_pool_models.py \
  --source-root cloth_5x5_500step_pipeline \
  --pool-root cloth_5x5_metamizer_pool_training \
  --activations identity relu tanh \
  --depths 1 \
  --widths 256 \
  --epochs 50 \
  --updates-per-epoch 1000 \
  --validation-interval 10 \
  --validation-rollout-length 100 \
  --validation-inner-steps 50 \
  --device cuda:0 \
  --resume
```

### 对比 pool 模型和已有 500-step 模型

默认评估使用 motion 20 到 31，rollout 长度 500，每帧 50 次 inner step：

```bash
python cloth14_evaluate_pool_vs_existing_rollouts.py \
  --source-root cloth_5x5_500step_pipeline \
  --pool-root cloth_5x5_metamizer_pool_training \
  --motion-indices 20 21 22 23 24 25 26 27 28 29 30 31 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

如果还要包含 `points_0032` 初始点消融模型和 baseline：

```bash
python cloth14_evaluate_pool_vs_existing_rollouts.py \
  --source-root cloth_5x5_500step_pipeline \
  --pool-root cloth_5x5_metamizer_pool_training \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --include-points-0032 \
  --baselines gd adam lbfgs newton \
  --motion-indices 20 21 22 23 24 25 26 27 28 29 30 31 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

如果要比较所有命名 motion：

```bash
python cloth14_evaluate_pool_vs_existing_rollouts.py \
  --motion-indices 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
                   16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

## 输出

训练输出：

```text
cloth_5x5_metamizer_pool_training/
└── models/
    └── activation_<activation>_depth_01_width_256_no_bias/
        ├── config.json
        ├── pool_manifest.json
        ├── train_log.csv
        ├── validation_metrics.json
        ├── latest_checkpoint.pt
        └── best_validation_model.pt
```

Rollout 对比输出：

```text
cloth_5x5_metamizer_pool_training/
└── rollout_evaluation/
    ├── all_motion_summary.csv
    ├── run_config.json
    └── motion_020/
        ├── full_500step_identity/curve.pt
        ├── pool_identity/curve.pt
        ├── ...
        ├── summary_metrics.csv
        ├── all_curves.pt
        └── figures/
            ├── rollout_x_iteration_vs_residual.png
            └── rollout_frame_vs_final_residual.png
```

每条 curve 都沿用 `cloth12` 的格式：

```text
residual_by_frame_and_iteration     # [completed_frames, inner_steps + 1]
final_residual_by_frame
global_residual
positions
velocities
reference_error_by_frame
```

## 主要解读

这个实验要回答的问题是：只从初始状态出发，并在模型自己产生的在线 residual 分布上训练 learned optimizer，是否能达到或超过用完整 500-step time-step 数据集训练出来的模型在连续 rollout 稳定性上的表现。
