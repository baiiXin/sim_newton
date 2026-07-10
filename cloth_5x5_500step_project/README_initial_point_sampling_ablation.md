# 初始点采样数量消融实验

这个实验只研究每个 motion/time-step problem 采样的初始状态数量：

```text
points_per_problem = {1, 8, 32, 64, 128, 1024}
```

其他设置全部固定。每一种 sample-count 实验都包含真实物理初始状态。其余状态是在已保存 reference solution 周围生成的 scrambled Sobol 采样。

## 1. 文件

```text
cloth10_prepare_initial_point_ablation.py
cloth11_train_initial_point_ablation.py
cloth12_evaluate_initial_point_ablation_rollouts.py
```

- `cloth10`：复用已有 reference trajectories，并为每个训练 problem 创建一条共享的、嵌套的 1024-state 序列。
- `cloth11`：对每个选定 prefix length 训练一个模型；epoch 数、minibatch 和 optimizer update 次数都固定。
- `cloth12`：在 motion 20-31 上评估选定 checkpoint，使用 500-frame 连续 rollout，每帧 50 次 inner iteration。

## 2. 数据语义

对每个训练 problem，共享 sample 轴的含义是：

```text
slot 0      : 真实物理初始状态 p_n
slots 1..   : exact_y 周围的 scrambled Sobol states
```

消融数据集是嵌套 prefix：

```text
points_0001 = slots [0:1]
points_0008 = slots [0:8]
points_0032 = slots [0:32]
points_0064 = slots [0:64]
points_0128 = slots [0:128]
points_1024 = slots [0:1024]
```

因此，每个实验都包含物理初始状态；增加 sample count 只是在扩大覆盖的初始状态区域。

下面这些已有文件会被复用，不会重新生成：

```text
cloth_5x5_500step_pipeline/data/reference/reference_problems.pt
cloth_5x5_500step_pipeline/data/reference/reference_motion_states.pt
cloth_5x5_500step_pipeline/data/reference/runtime_config.json
```

## 3. 训练语义

原始 time-problem minibatch 保持不变：

```text
16 training motions x 32 time steps per motion
```

每个 epoch 有 13 次 optimizer update：

```text
12 full windows: 16 motions x 32 time steps
1 tail window : 16 motions x 16 time steps
```

对一个 time-window minibatch，训练流程是：

```python
optimizer.zero_grad()
for sample_slot in range(points_per_problem):
    loss = loss_for_this_sample_slot()
    (loss / points_per_problem).backward()
clip_grad_norm_()
optimizer.step()
```

这带来的结果是：

- 每个 epoch 会且只会访问每个选中的 state 一次；
- 每个 sample-count 实验拥有相同数量的 optimizer update；
- GPU microbatch shape 与 sample count 无关；
- CUDA 峰值显存应基本保持固定；
- 总运行时间仍然会近似随 sample count 线性增长。

默认模型和训练设置：

```text
activation        = identity
depth             = 1
width             = 256
bias              = false
epochs            = 500
learning_rate     = 1e-3
gradient_clip     = 10
K curriculum      = 1, 3, 5, 10, 30
epochs_per_K      = 100
validation every  = 50 epochs
```

## 4. 验证 checkpoint 选择

验证使用原始 validation motions：

```text
motion 16, 17, 18, 19
```

每次 validation event 运行：

```text
4 motions x 300 rollout frames x 15 learned iterations per frame
```

每帧只使用第 15 次 iteration 后的 residual。这样会得到 1200 个 final residual。checkpoint selection metric 精确定义为：

```text
这 1200 个 final residual 的全局最大值
```

p95 值、每个 motion 的最大值以及最差 frame 只用于诊断保存，不参与 checkpoint 选择。

## 5. 测试 rollout

默认 test motions 是训练和验证之外的所有已有 motion：

```text
ID test : motion 20-23
OOD test: motion 24-31
```

每个 solver 的评估设置是：

```text
rollout length       = 500 frames
inner iterations     = 50 per frame
frame initial state  = solver 自己传播得到的物理状态
```

对每个 frame 和每个 model/baseline，保存的 curve 包含：

```text
initial_y_by_frame                  # y^(0)
solution_y_by_frame                 # y^(50)
residual_by_frame_and_iteration     # shape [completed_frames, 51]
final_residual_by_frame
positions                           # propagated frames, including frame 0
velocities
reference_error_by_frame
global_iteration
global_residual                     # iterations 1..50 flattened, no separators
```

已有 reference trajectory 不会重新运行。它保存的 solution 和 residual 只在每个 frame 的 iteration-50 endpoint 上绘图。

默认对比曲线：

```text
model_points_0001
model_points_0008
model_points_0032
model_points_0064
model_points_0128
model_points_1024
baseline_gd
baseline_adam
baseline_lbfgs
baseline_newton
reference endpoints
```

## 6. 输出结构

```text
cloth_5x5_initial_sample_ablation/
├── shared_reference/
├── shared_samples_1024/
│   ├── motion_000.pt
│   ├── ...
│   ├── motion_015.pt
│   └── manifest.json
├── points_0001/
│   ├── experiment.json
│   └── models/
├── points_0008/
├── points_0032/
├── points_0064/
├── points_0128/
├── points_1024/
└── rollout_evaluation/
    ├── all_motion_summary.csv
    ├── motion_020/
    │   ├── reference_len_500.pt
    │   ├── reference_endpoints.pt
    │   ├── model_points_0001/curve.pt
    │   ├── ...
    │   ├── baseline_newton/curve.pt
    │   ├── all_curves.pt
    │   ├── summary_metrics.csv
    │   └── figures/
    │       ├── rollout_x_iteration_vs_residual.png
    │       └── rollout_frame_vs_final_residual.png
    └── ...
```

## 7. 运行命令

从 `cloth_5x5_500step_project/` 目录运行。

### 准备共享的嵌套 samples

```bash
python cloth10_prepare_initial_point_ablation.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --sample-counts 1 8 32 64 128 1024 \
  --max-points 1024
```

### 小规模训练 smoke test

```bash
python cloth11_train_initial_point_ablation.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --sample-counts 1 \
  --epochs 2 \
  --validation-interval 1 \
  --validation-rollout-length 5 \
  --validation-inner-steps 2 \
  --device cuda:0 \
  --overwrite
```

### 训练正式消融实验

```bash
python cloth11_train_initial_point_ablation.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --sample-counts 1 8 32 64 128 1024 \
  --epochs 500 \
  --validation-interval 50 \
  --validation-rollout-length 300 \
  --validation-inner-steps 15 \
  --device cuda:0 \
  --resume
```

因为运行时间会随 sample count 增长，实验也可以分开启动，例如：

```bash
python cloth11_train_initial_point_ablation.py \
  --sample-counts 1024 \
  --device cuda:0 \
  --resume
```

### Rollout smoke test

```bash
python cloth12_evaluate_initial_point_ablation_rollouts.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --sample-counts 1 \
  --motion-indices 20 \
  --baselines gd newton \
  --rollout-length 5 \
  --inner-steps 3 \
  --device cuda:0 \
  --overwrite
```

### 运行正式的 12-motion rollout evaluation

```bash
python cloth12_evaluate_initial_point_ablation_rollouts.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --sample-counts 1 8 32 64 128 1024 \
  --motion-indices 20 21 22 23 24 25 26 27 28 29 30 31 \
  --baselines gd adam lbfgs newton \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

### 只从已保存 line data 重建图

```bash
python cloth12_evaluate_initial_point_ablation_rollouts.py \
  --source-root cloth_5x5_500step_pipeline \
  --ablation-root cloth_5x5_initial_sample_ablation \
  --motion-indices 20 21 22 23 24 25 26 27 28 29 30 31 \
  --rollout-length 500 \
  --inner-steps 50 \
  --plot-only
```

## 8. 解释结果时的注意事项

这个设计固定了 epochs、minibatch 定义和 optimizer-update 次数。相比把所有 states 放进一个巨大 batch，它能更干净地隔离 initial-state coverage 的影响。不过，它没有固定总浮点计算量：1024-point 模型每个 epoch 执行的 sample-slot forward/backward pass 大约是 32-point 模型的 32 倍。报告结果时应同时给出性能和 wall-clock cost。
