# Cloth 5×5 500-Step Learned Optimizer Pipeline

这个项目把原来的单文件布料 learned optimizer 实验拆成 9 个脚本，目标是把 **数据集构造、baseline 评估、模型训练、结果汇总、rollout 测试、渲染** 分离，方便后续频繁修改输入形式、网络结构、训练策略和测试方式。

默认实验设置：

- 布料：`5×5` 三角网格，左上角和左下角两个点固定。
- 状态：full-state `25×3 = 75D`。
- 内部物理：只对 `23` 个自由点求解，即 reduced `23×3 = 69D`。
- 网络输入：`[current residual, previous residual, previous update] = 3×75D = 225D`。
- 网络输出：`75D` 位移更新。
- 固定点处理：网络输出后对固定点位移 gate 为 0，并 hard projection 到固定位置。
- 时间长度：每个 motion `500` 个物理时间步。
- motion 数量：`32` 个。
- 训练时间：train motion 的 `0–399` 每个时间步都参与训练。
- seen extrapolation：train motion 的 `400–499`。
- 默认精度：`torch.float64`。
- 默认设备：`cuda:0`。

---

## 1. 文件说明

所有脚本需要放在同一个目录下，因为它们通过普通 Python import 互相调用。

```text
cloth01_generate_reference_and_samples.py   # 生成参考解和全量采样数据
cloth02_dataset_catalog.py                  # 聚合 train / validation / test 数据集
cloth03_solvers_and_models.py               # 公共物理、模型、solver、baseline 方法
cloth04_evaluate_baselines.py               # 评估 GD / Adam / L-BFGS / Newton baseline
cloth05_train_models.py                     # 训练 learned optimizer 并测试
cloth06_plot_summary.py                     # 汇总 baseline 和模型测试结果并绘图
cloth07_rollout_models.py                   # 连续 rollout 测试，支持断点复用
cloth08_render_rollouts.py                  # 渲染 rollout 结果
cloth09_render_reference_motions.py         # 渲染全部 motion 的参考解
```

其中：

- `cloth03_solvers_and_models.py` 是公共模块，不需要单独运行。
- 其他脚本按编号顺序运行。
- 脚本 7、8、9 可以在训练完成后按需运行。

---

## 2. 推荐目录结构

建议新建一个项目目录，例如：

```bash
mkdir -p cloth_5x5_500step_project
cd cloth_5x5_500step_project
```

把 9 个脚本放到该目录下。运行后默认输出目录为：

```text
cloth_5x5_500step_pipeline/
├── data/
│   ├── reference/
│   │   ├── reference_problems.pt
│   │   ├── reference_motion_states.pt
│   │   ├── runtime_config.json
│   │   ├── motion_catalogue.json
│   │   └── initial_state_figures/
│   ├── samples/
│   │   └── all_sampled_problems.pt
│   └── datasets/
│       ├── train.pt
│       ├── validation.pt
│       ├── seen_extrap.pt
│       ├── unseen_id.pt
│       ├── ood.pt
│       ├── current_state_seen_extrap.pt
│       ├── current_state_unseen_id.pt
│       ├── current_state_ood.pt
│       ├── train_batch_plan.json
│       └── dataset_manifest.json
├── baselines/
│   ├── parameter_selection.json
│   ├── baseline_metrics.json
│   ├── baseline_curves.pt
│   └── figures/
├── models/
│   └── <model_name>/
│       ├── config.json
│       ├── latest_checkpoint.pt
│       ├── best_validation_model.pt
│       ├── train_log.csv
│       ├── validation_metrics.json
│       ├── test_metrics.json
│       ├── test_curves.pt
│       └── figures/
├── results/
│   ├── summary_figures/
│   └── summary_tables/
├── rollouts/
│   └── motion_XXX/
│       ├── reference_len_500.pt
│       ├── baseline_gd/
│       ├── baseline_adam/
│       ├── baseline_lbfgs/
│       ├── baseline_newton/
│       └── <model_name>/
└── renders/
    ├── rollouts/
    └── reference_motions/
```

---

## 3. 环境依赖

核心依赖：

```bash
pip install numpy matplotlib torch
```

如果要保存 `.mp4` 视频，需要系统中能调用 `ffmpeg`。如果暂时不渲染视频，只保存图片帧，不需要配置 ffmpeg。

建议在服务器上使用已有的 PyTorch CUDA 环境，例如：

```bash
conda activate hood
```

默认代码使用：

```text
torch.float64
cuda:0
```

如果想换显卡，在命令里传：

```bash
--device cuda:1
```

---

## 4. 完整运行流程

### Step 1：生成 32×500 参考解和采样数据

```bash
python cloth01_generate_reference_and_samples.py \
  --output-dir cloth_5x5_500step_pipeline \
  --total-time-steps 500 \
  --points-per-problem 32
```

这个脚本会做三件事：

1. 生成 32 个 motion。
2. 每个 motion 连续生成 500 步 reference solution。
3. 每个 time-step problem 采样 `points_per_problem` 个初始状态。

主要输出：

```text
cloth_5x5_500step_pipeline/data/reference/reference_problems.pt
cloth_5x5_500step_pipeline/data/reference/reference_motion_states.pt
cloth_5x5_500step_pipeline/data/samples/all_sampled_problems.pt
cloth_5x5_500step_pipeline/data/reference/initial_state_figures/*.png
```

其中 `initial_state_figures/` 中每个 motion 一张图，画的是该 motion 的第 0 步初始状态。

可选参数：

```bash
--skip-samples    # 只生成参考解，不生成采样数据
--skip-plots      # 不绘制 32 张初始状态图
```

---

### Step 2：聚合数据集

```bash
python cloth02_dataset_catalog.py \
  --root cloth_5x5_500step_pipeline \
  --total-time-steps 500 \
  --train-time-stop 400 \
  --time-steps-per-motion-batch 32 \
  --eval-time-count 50
```

默认数据划分：

```text
train       : motion 0–15,  time 0–399
validation  : motion 16–19, time 0–499 均匀采 50 个时间步
seen_extrap : motion 0–15,  time 400–499
unseen_id   : motion 20–23, time 0–499 均匀采 50 个时间步
ood         : motion 24–31, time 0–499 均匀采 50 个时间步
```

训练 mini-batch 计划：

```text
一个 mini-batch = 16 个 train motion × 每个 motion 32 个 time-step problems
```

在 `0–399` 的训练时间范围内：

```text
12 个完整 batch: 16 × 32 time-step problems
1 个尾 batch:   16 × 16 time-step problems
```

主要输出：

```text
data/datasets/train.pt
data/datasets/validation.pt
data/datasets/seen_extrap.pt
data/datasets/unseen_id.pt
data/datasets/ood.pt
data/datasets/train_batch_plan.json
data/datasets/dataset_manifest.json
```

---

### Step 3：评估 baseline

```bash
python cloth04_evaluate_baselines.py \
  --root cloth_5x5_500step_pipeline \
  --device cuda:0
```

baseline 方法：

```text
GD
Adam
L-BFGS
Newton
```

脚本会先在 validation 上做参数选择：

- GD：选择 step size。
- Adam：选择 learning rate。
- L-BFGS：选择 learning rate 和 history size。
- Newton：无参数选择。

然后在以下数据集上统一评估：

```text
validation
seen_extrap
unseen_id
ood
```

主要输出：

```text
baselines/parameter_selection.json
baselines/baseline_metrics.json
baselines/baseline_curves.pt
baselines/figures/*.png
```

指标包括：

```text
residual_mean_by_iter
residual_max_by_iter
residual_sum_by_iter
final_residual_mean
final_residual_max
final_residual_sum
```

如果已经有 `parameter_selection.json`，想跳过参数选择，可以运行：

```bash
python cloth04_evaluate_baselines.py \
  --root cloth_5x5_500step_pipeline \
  --device cuda:0 \
  --skip-selection
```

---

### Step 4：训练 learned optimizer

默认训练：

```bash
python cloth05_train_models.py \
  --root cloth_5x5_500step_pipeline \
  --device cuda:0
```

默认模型网格：

```text
activation: identity, relu, tanh
depth: 1
width: 256
bias: False
```

默认训练设置：

```text
epochs = 500
validation_interval = 200
optimizer = Adam
learning_rate = 1e-3
gradient_clip_norm = 10
k_values = 1, 3, 5, 10, 30
epochs_per_k = 100
```

训练 batch 规则：

```text
一个 mini-batch = 16 个 train motion × 每个 motion 32 个 time-step problems
每个 time-step problem 使用它的全部采样初值
每个 epoch 覆盖完整 train set
```

主要输出：

```text
models/<model_name>/config.json
models/<model_name>/latest_checkpoint.pt
models/<model_name>/best_validation_model.pt
models/<model_name>/train_log.csv
models/<model_name>/validation_metrics.json
models/<model_name>/test_metrics.json
models/<model_name>/test_curves.pt
models/<model_name>/figures/*.png
```

只训练某一个模型配置，可以用：

```bash
python cloth05_train_models.py \
  --root cloth_5x5_500step_pipeline \
  --device cuda:0 \
  --activations identity \
  --depths 1 \
  --widths 256
```

如果要带 bias：

```bash
python cloth05_train_models.py \
  --root cloth_5x5_500step_pipeline \
  --device cuda:0 \
  --activations relu tanh \
  --depths 1 2 \
  --widths 256 512 \
  --use-bias
```

---

### Step 5：绘制 baseline 和模型汇总图

```bash
python cloth06_plot_summary.py \
  --root cloth_5x5_500step_pipeline
```

输入：

```text
baselines/baseline_metrics.json
models/*/test_metrics.json
```

输出：

```text
results/summary_figures/*.png
results/summary_tables/summary_metrics.csv
```

这个脚本只读取已有结果，不重新评估。

---

### Step 6：连续 rollout 测试

评估 baseline rollout：

```bash
python cloth07_rollout_models.py \
  --root cloth_5x5_500step_pipeline \
  --motion-index 3 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0 \
  --baselines gd adam lbfgs newton
```

评估 learned model rollout：

```bash
python cloth07_rollout_models.py \
  --root cloth_5x5_500step_pipeline \
  --motion-index 3 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0 \
  --model-dirs models/activation_identity_depth_01_width_0256_no_bias
```

同时评估 baseline 和模型：

```bash
python cloth07_rollout_models.py \
  --root cloth_5x5_500step_pipeline \
  --motion-index 3 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0 \
  --baselines gd adam lbfgs newton \
  --model-dirs models/activation_identity_depth_01_width_0256_no_bias
```

输出目录按 motion id 组织：

```text
rollouts/motion_003/
├── reference_len_500.pt
├── baseline_gd/
│   ├── rollout.pt
│   ├── metrics.json
│   └── status.json
├── baseline_adam/
├── baseline_lbfgs/
├── baseline_newton/
└── activation_identity_depth_01_width_0256_no_bias/
    ├── rollout.pt
    ├── metrics.json
    └── status.json
```

断点复用逻辑：

```text
如果同一个 motion + 同一个 solver/model 已经完成到目标 rollout_length：跳过。
如果已有结果但长度不足：从最后一帧继续 rollout。
如果加 --overwrite：删除旧结果，从头计算。
```

从头重跑：

```bash
python cloth07_rollout_models.py \
  --root cloth_5x5_500step_pipeline \
  --motion-index 3 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0 \
  --baselines gd adam lbfgs newton \
  --overwrite
```

---

### Step 7：渲染 rollout 结果

```bash
python cloth08_render_rollouts.py \
  --root cloth_5x5_500step_pipeline \
  --motion-index 3 \
  --frame-stride 5 \
  --save-frames \
  --make-video
```

只渲染指定 solver/model：

```bash
python cloth08_render_rollouts.py \
  --root cloth_5x5_500step_pipeline \
  --motion-index 3 \
  --solver-names baseline_gd activation_identity_depth_01_width_0256_no_bias \
  --frame-stride 5 \
  --save-frames \
  --make-video
```

输出：

```text
renders/rollouts/motion_003/<solver_name>/final_frame.png
renders/rollouts/motion_003/<solver_name>/metrics.png
renders/rollouts/motion_003/<solver_name>/frames/*.png
renders/rollouts/motion_003/<solver_name>/rollout.mp4
```

---

### Step 8：渲染全部 reference motion

渲染所有 32 个 motion：

```bash
python cloth09_render_reference_motions.py \
  --root cloth_5x5_500step_pipeline \
  --all \
  --frame-stride 5 \
  --make-video
```

只渲染指定 motion：

```bash
python cloth09_render_reference_motions.py \
  --root cloth_5x5_500step_pipeline \
  --motion-indices 0 1 2 3 \
  --frame-stride 5 \
  --save-frames \
  --make-video
```

输出：

```text
renders/reference_motions/motion_000/final_frame.png
renders/reference_motions/motion_000/frames/*.png
renders/reference_motions/motion_000/reference.mp4
...
```

---

## 5. 小规模 smoke test

正式 `32×500×points` 比较重。第一次建议先用小规模数据检查目录和数据流：

```bash
python cloth01_generate_reference_and_samples.py \
  --output-dir cloth_5x5_500step_debug \
  --total-time-steps 20 \
  --points-per-problem 4

python cloth02_dataset_catalog.py \
  --root cloth_5x5_500step_debug \
  --total-time-steps 20 \
  --train-time-stop 10 \
  --time-steps-per-motion-batch 5 \
  --eval-time-count 5
```

然后可以尝试：

```bash
python cloth04_evaluate_baselines.py \
  --root cloth_5x5_500step_debug \
  --device cuda:0 \
  --steps 5 \
  --selection-max-points 512

python cloth05_train_models.py \
  --root cloth_5x5_500step_debug \
  --device cuda:0 \
  --epochs 2 \
  --validation-interval 1 \
  --evaluation-steps 5 \
  --activations identity \
  --depths 1 \
  --widths 256
```

确认能跑通后，再切回正式目录：

```text
cloth_5x5_500step_pipeline
```

---

## 6. 常用检查命令

查看数据集大小：

```bash
python - <<'PY'
import torch
from pathlib import Path
root = Path('cloth_5x5_500step_pipeline')
for name in ['train', 'validation', 'seen_extrap', 'unseen_id', 'ood']:
    data = torch.load(root / 'data' / 'datasets' / f'{name}.pt', map_location='cpu')
    print(name, data['initial_y'].shape, data['metadata'])
PY
```

查看训练 batch plan：

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('cloth_5x5_500step_pipeline/data/datasets/train_batch_plan.json')
plan = json.loads(path.read_text())
print('num_batches_per_epoch =', plan['num_batches_per_epoch'])
print('first batch problems =', len(plan['problem_indices_by_batch'][0]))
print('last batch problems =', len(plan['problem_indices_by_batch'][-1]))
PY
```

查看 baseline 最终指标：

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path('cloth_5x5_500step_pipeline/baselines/baseline_metrics.json')
metrics = json.loads(path.read_text())
for dataset, methods in metrics.items():
    print('\n', dataset)
    for method, item in methods.items():
        print(method, item['final_residual_mean'], item['final_residual_max'])
PY
```

查看模型结果：

```bash
python - <<'PY'
import json
from pathlib import Path
for path in Path('cloth_5x5_500step_pipeline/models').glob('*/test_metrics.json'):
    print('\nMODEL:', path.parent.name)
    metrics = json.loads(path.read_text())
    for dataset, item in metrics.items():
        print(dataset, item['final_residual_mean'], item['final_residual_max'])
PY
```

---

## 7. 数据规模提醒

正式设置下：

```text
32 motions × 500 steps = 16000 time-step problems
```

如果 `points_per_problem = 32`：

```text
16000 × 32 = 512000 sampled states
```

因为保存的是 float64 full-state 数据，`all_sampled_problems.pt` 可能接近 GB 级。空间不够时，可以先降低：

```bash
--points-per-problem 8
```

---

## 8. 推荐正式运行顺序总览

```bash
python cloth01_generate_reference_and_samples.py \
  --output-dir cloth_5x5_500step_pipeline \
  --total-time-steps 500 \
  --points-per-problem 32

python cloth02_dataset_catalog.py \
  --root cloth_5x5_500step_pipeline \
  --total-time-steps 500 \
  --train-time-stop 400 \
  --time-steps-per-motion-batch 32 \
  --eval-time-count 50

python cloth04_evaluate_baselines.py \
  --root cloth_5x5_500step_pipeline \
  --device cuda:0

python cloth05_train_models.py \
  --root cloth_5x5_500step_pipeline \
  --device cuda:0

python cloth06_plot_summary.py \
  --root cloth_5x5_500step_pipeline

python cloth07_rollout_models.py \
  --root cloth_5x5_500step_pipeline \
  --motion-index 3 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0 \
  --baselines gd adam lbfgs newton \
  --model-dirs models/activation_identity_depth_01_width_0256_no_bias

python cloth08_render_rollouts.py \
  --root cloth_5x5_500step_pipeline \
  --motion-index 3 \
  --frame-stride 5 \
  --save-frames \
  --make-video

python cloth09_render_reference_motions.py \
  --root cloth_5x5_500step_pipeline \
  --all \
  --frame-stride 5 \
  --make-video
```

---

## 9. 设计原则

这个项目当前优先级是：

1. 数据、baseline、训练、rollout、渲染解耦。
2. 代码可读、流程清楚。
3. 不做过度 try/except 包装。
4. 不为了“工程完整性”牺牲实验逻辑透明度。
5. 后续方便替换输入、网络结构、训练方式和数据集构造策略。
