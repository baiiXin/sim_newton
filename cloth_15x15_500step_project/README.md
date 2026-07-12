# 15×15 布料 500 步神经迭代求解器实验项目

本目录将 `cloth_5x5_500step_project` 扩展到 **15×15 三角网格布料**，用于系统研究网络宽度、深度、激活函数、bias、训练初值数量和 training pool 训练方法。

本项目特别区分三个概念：

1. **物理时间步**：给定当前状态 $x_n,v_n$，构造下一时刻的隐式积分优化问题；
2. **内层迭代**：在同一个物理时间步问题中，从初值 $y^{(0)}=x_n$ 开始反复更新 $y$；
3. **连续 rollout**：把某一物理时间步求得的位置继续传播到下一物理时间步。

验证集和测试集中的“每个时间步只取一个初值”指的是：

- 每个 $(motion,t)$ 只构造一个初值 $y^{(0)}=x_n$；
- 默认在这个物理时间步问题内部运行 **50 次迭代**；
- 不把该问题的预测结果传播到下一个验证/测试问题。

因此验证/测试衡量的是**单个物理时间步内的迭代收敛能力**，而500步 rollout 衡量的是**跨物理时间步的长期稳定性**。

---

## 1. 15×15 物理与网络状态

- 网格：`15×15 = 225` 个顶点；
- 固定点：左上角和左下角，与5×5项目一致；
- 完整状态：`225×3 = 675D`；
- 自由顶点：223个；
- reduced物理状态：`223×3 = 669D`；
- 网络输入：

  $$
  [r^{(k)},r^{(k-1)},\Delta y^{(k-1)}]\in\mathbb{R}^{2025};
  $$

- 网络输出：`675D`位移修正；
- 固定点输出在应用前置零，并在更新后做hard projection；
- 默认数值精度：`torch.float64`；
- 默认设备：`cuda:0`。

`cloth03_solvers_and_models.py`复用原5×5项目中已经实现的物理、解析梯度和解析Hessian，只在模块载入后替换网格相关全局量，并增加GELU和SiLU。

---

## 2. Reference构造与motion过滤

### 2.1 为什么必须先生成完整reference

在连续reference轨迹中，第 $t$ 帧的reference位置会用于构造第 $t+1$ 帧的位置和速度。因此某一帧reference没有收敛时，后续帧也可能受到污染。

所以本项目采用：

1. 先生成全部32个motion的完整500步reference；
2. 每帧保存reference residual；
3. 每个motion绘制一条 residual vs. physical frame 曲线；
4. 渲染可疑motion；
5. 最后按**完整motion**排除，而不是删除孤立时间步。

原始reference不会因为排除选项而被删除，保证过滤决策可追溯、可修改。

### 2.2 Reference输出

```text
cloth_15x15_500step_pipeline/data/reference/
├── per_motion/
│   ├── motion_000.pt
│   ├── motion_001.pt
│   └── ...
├── reference_problems.pt
├── reference_motion_states.pt
├── runtime_config.json
├── motion_catalogue.json
├── initial_state_figures/
└── residual_audit/
    ├── motion_000_reference_residual.png
    ├── ...
    ├── all_motion_reference_residuals.png
    ├── reference_motion_summary.csv
    └── reference_audit.json
```

Reference按motion单独保存。任务中断后重新运行时，会复用已经完成的motion；只有传入 `--overwrite-reference` 才会重新计算。

### 2.3 推荐运行流程

先只生成reference：

```bash
python cloth01_generate_reference_and_samples.py \
  --output-dir cloth_15x15_500step_pipeline \
  --reference-only
```

渲染指定motion：

```bash
python cloth09_render_reference_motion.py \
  --root cloth_15x15_500step_pipeline \
  --motion-index 27 \
  --format mp4
```

在检查曲线和视频后，再决定排除列表。例如：

```bash
python cloth01_generate_reference_and_samples.py \
  --output-dir cloth_15x15_500step_pipeline \
  --samples-only \
  --exclude-motion-indices 27 31 \
  --points-per-problem 32
```

这里的 `27 31` 只是命令示例，不能在实际reference生成前预先认定这些motion失败。

### 2.4 阈值建议

不建议只看某个motion的最大residual。最终判断应综合：

- 是否出现NaN或Inf；
- residual p95和p99；
- 最大值是否只是孤立尖峰；
- 最坏帧之后的轨迹是否明显失真；
- 视觉上是否出现爆炸、极端拉伸或错误速度传播。

---

## 3. 训练集、验证集与测试集

原motion划分保持不变：

```text
train      : motion 0–15
validation : motion 16–19
test ID    : motion 20–23
test OOD   : motion 24–31
```

被人工排除的motion会从相应集合整体移除。

### 3.1 训练集

默认架构实验使用：

- train motion的物理时间步 `0–399`；
- 每个time-step problem默认32个扰动初值；
- 一个训练窗口包含所有训练motion的连续32个物理时间步；
- 训练中的unroll长度由 $K$ curriculum控制。

训练样本按motion分片保存，不再为每个初值重复保存 $q$、masses和exact solution。

### 3.2 验证集

验证集使用原验证motion的全部时间步：

```text
motion 16–19 × time 0–499 = 2000 个物理时间步问题
```

每个问题：

```text
初值                 : y^(0) = x_n
初始历史残差         : 0
初始历史更新         : 0
默认内层迭代次数     : 50
跨物理帧传播         : 否
```

会保存完整矩阵：

$$
R\in\mathbb{R}^{N\times 51},
$$

其中第0列是初值residual，后50列对应50次迭代后的residual。

Checkpoint默认按照验证集第50次迭代后的：

$$
\operatorname{P95}(R_{:,50})
$$

选择，越小越好。

同时保存mean、p50、p95、p99、max residual vs. iteration、每次迭代的非有限值数量以及最终改善比例。

### 3.3 测试集

测试采用完全相同的50次内层迭代协议：

```text
test_id_xn  : motion 20–23，全部500帧，共2000个问题
test_ood_xn : motion 24–31，全部500帧，共4000个问题
test_all_xn : motion 20–31，全部500帧，共6000个问题
```

测试数据不用于参数选择或checkpoint选择。

### 3.4 构造catalogue

```bash
python cloth02_dataset_catalog.py \
  --root cloth_15x15_500step_pipeline \
  --evaluation-iterations 50 \
  --exclude-motion-indices 27 31
```

---

## 4. Baseline协议

所有baseline必须和神经网络使用同一协议：

- 同一个验证/测试集合；
- 每个问题都从 $x_n$ 开始；
- 每个问题运行50次内层迭代；
- 各问题之间不共享优化器状态；
- 输出完整 residual vs. iteration 曲线。

Baseline包括：

```text
GD
Adam
L-BFGS
BFGS
Newton
```

### 4.1 Adam参数范围

原5×5代码中的Adam范围实际已经包含 `1e-5`，并不是从 `1e-4` 开始。15×15项目进一步扩展为：

```text
1e-8, 2e-8, 5e-8,
1e-7, 2e-7, 5e-7,
...
1e-1, 2e-1, 5e-1,
1.0
```

参数选择在验证集的均匀分层子集上进行，默认使用256个问题；选择指标是50次迭代后的final residual p95。选出的参数再在完整验证集和测试集上运行。

### 4.2 L-BFGS运行语义

原5×5实现调用单个PyTorch `LBFGS`对象，把一个batch的所有问题拼成一个大参数，并优化batch energy之和。这会导致：

- 不同物理问题共享同一组L-BFGS curvature history；
- two-loop recursion中的内积跨样本求和；
- 样本之间产生本不应存在的耦合；
- 横轴上的一次L-BFGS iteration不再严格对应每个问题的一次独立准牛顿更新。

15×15项目不再使用这个语义。

新L-BFGS实现为：

- 每个time-step problem拥有独立的 $s_i,y_i,\rho_i$ 历史；
- two-loop recursion按problem独立计算；
- 每个problem独立进行Armijo backtracking；
- history size和line-search初始步长在验证子集上选择；
- 每个横轴iteration对应一次独立准牛顿方向和一次line-search更新。

初始逆Hessian使用隐式积分惯性项对应的正对角预条件：

$$
H_0=\Delta t^2M^{-1}.
$$

### 4.3 Full BFGS

新增full BFGS：

- 每个问题维护独立的 $669\times669$ 逆Hessian近似；
- 使用标准inverse-BFGS更新；
- 使用独立Armijo backtracking；
- 曲率条件失败时跳过该次矩阵更新；
- 不与其他问题共享矩阵或曲率信息。

Full BFGS的内存和计算成本远高于L-BFGS。默认 `--bfgs-batch-size 2`，但支持在完整验证/测试集上运行。

### 4.4 Newton

Newton使用解析Hessian。15×15 reduced Hessian是 `669×669`，默认Newton batch size设为4。

### 4.5 运行baseline

```bash
python cloth08_evaluate_baselines.py \
  --root cloth_15x15_500step_pipeline \
  --steps 50 \
  --device cuda:0
```

主要输出：

```text
baselines/
├── parameter_selection.json
├── baseline_metrics.json
├── baseline_curves.pt
├── baseline_manifest.json
└── figures/
```

会生成验证集和测试集的mean、p95、max residual vs. iteration图。

---

## 5. 神经网络与Baseline对比图

模型训练完成、baseline评估完成后运行：

```bash
python cloth12_plot_model_vs_baselines.py \
  --root cloth_15x15_500step_pipeline \
  --experiment-dir PATH/TO/MODEL_EXPERIMENT_DIR \
  --model-label learned_optimizer
```

脚本会在同一张图上对比learned optimizer、GD、Adam、L-BFGS、BFGS和Newton，并分别绘制mean、p95和max residual vs. iteration。

`--experiment-dir` 每次只接收一个模型实验目录，目录中必须已经存在 `evaluation_curves.pt`。如果当前工作目录已经是 `cloth_15x15_500step_project/`，则路径应写成：

```bash
python cloth12_plot_model_vs_baselines.py \
  --root cloth_15x15_500step_pipeline \
  --experiment-dir cloth_15x15_500step_pipeline/experiments/STAGE/samples_XXXX/EXPERIMENT_NAME \
  --model-label learned_optimizer
```

---

## 6. 网络实验顺序

### Stage 1：ReLU宽度

固定 `activation=relu, depth=1, bias=False, samples=32`，测试：

```text
128, 256, 512, 1024, 2048, 4096
```

```bash
python cloth05_train_models.py \
  --root cloth_15x15_500step_pipeline \
  --stage width \
  --activations relu \
  --depths 1 \
  --widths 128 256 512 1024 2048 4096 \
  --bias-mode no-bias \
  --sample-count 32 \
  --evaluation-steps 50 \
  --device cuda:0 \
  --skip-completed

python cloth06_select_best.py \
  --root cloth_15x15_500step_pipeline \
  --stage width
```

### Stage 2：深度

使用Stage 1最佳宽度，测试 `1,2,3,5,7,10`。

```bash
python cloth05_train_models.py \
  --root cloth_15x15_500step_pipeline \
  --stage depth \
  --activations relu \
  --depths 1 2 3 5 7 10 \
  --widths BEST_WIDTH \
  --bias-mode no-bias \
  --sample-count 32 \
  --evaluation-steps 50 \
  --device cuda:0 \
  --skip-completed
```

### Stage 3：激活函数与bias

固定最佳宽度和深度，测试：

```text
activation ∈ {relu, gelu, silu, tanh, identity}
bias       ∈ {False, True}
```

```bash
python cloth05_train_models.py \
  --root cloth_15x15_500step_pipeline \
  --stage activation_bias \
  --activations relu gelu silu tanh identity \
  --depths BEST_DEPTH \
  --widths BEST_WIDTH \
  --bias-mode both \
  --sample-count 32 \
  --evaluation-steps 50 \
  --device cuda:0 \
  --skip-completed
```

Bias不仅增加参数量，还破坏“零residual必然产生零更新”的结构性质，因此应作为求解器结构消融解释。

“宽度→深度→激活与bias”属于贪心搜索。建议Stage 1和Stage 2后补充 `top-2 widths × top-2 depths` 共4组交叉实验。

### 6.1 单模型训练输出

每个 `cloth05_train_models.py` 模型实验目录会保存：

```text
experiments/STAGE/samples_XXXX/EXPERIMENT_NAME/
├── train_log.csv
├── validation_metrics.json
├── best_validation_model.pt
├── best_validation_summary.json
├── evaluation_metrics.json
├── evaluation_curves.pt
├── test_metrics.json
├── test_curves.pt
├── training_summary.json
├── completed.json
└── figures/
    ├── training_loss.png
    ├── validation_final_residual_overview.png
    ├── validation_xn_residual_vs_iteration.png
    ├── test_id_xn_residual_vs_iteration.png
    ├── test_ood_xn_residual_vs_iteration.png
    └── test_all_xn_residual_vs_iteration.png
```

`training_summary.json` 和 `completed.json` 中会显式记录：

```text
best_checkpoint_epoch
total_training_elapsed_seconds
```

其中 `total_training_elapsed_seconds` 是 `train_log.csv` 中所有epoch训练循环 `elapsed_seconds` 的总和，不包含最终validation/test评估绘图时间。

`figures/training_loss.png` 绘制训练loss随epoch变化；`figures/validation_final_residual_overview.png` 绘制每个验证节点处第50次内层迭代后的 `mean`、`p95`、`max` residual。两张图都会用黑色虚线标出 `best_checkpoint_epoch`。

对于已经完成的旧实验，重新运行带 `--skip-completed` 的训练命令时，会先从已有 `train_log.csv` 和 `validation_metrics.json` 补齐上述训练诊断文件，然后跳过重训。

---

## 7. 初值数量实验

测试数量：

```text
1, 8, 32, 128, 512, 1024
```

定义严格为：

```text
slot 0        = x_n
slot 1..1023  = 围绕exact_y的Sobol扰动
```

因此：

- 1点实验：只使用 $x_n$；
- 8点实验：$x_n$ + 7个扰动；
- 32点实验：$x_n$ + 31个扰动；
- 后续集合都是前一个集合的严格超集。

```bash
python cloth10_prepare_initial_point_ablation.py \
  --root cloth_15x15_500step_pipeline \
  --sample-counts 1 8 32 128 512 1024 \
  --max-points 1024
```

15×15下，16个训练motion、400帧、1024初值、675维float64的 `initial_y` 本身约为33 GiB，因此按motion和32帧时间窗口分片。

```bash
python cloth11_train_initial_point_ablation.py \
  --root cloth_15x15_500step_pipeline \
  --activation BEST_ACTIVATION \
  --depth BEST_DEPTH \
  --width BEST_WIDTH \
  --bias-mode no-bias \
  --sample-counts 1 8 32 128 512 1024 \
  --sample-chunk-size 8 \
  --device cuda:0 \
  --skip-completed
```

---

## 8. Training Pool实验

Training pool沿用Metamizer式语义：每个训练motion对应多个不同 $K$ 的live environment；每次参数更新只执行一次网络更新；environment完成自己的 $K$ 次内层更新后推进一个物理时间步；新物理帧重新从 $y^{(0)}=x_n$ 开始；不使用exact solution作为pool训练目标。

模型选择仍使用统一离线验证协议：全部验证motion、全部时间步、从 $x_n$ 开始、运行50次内层迭代。

```bash
python cloth13_train_pool.py \
  --root cloth_15x15_500step_pipeline \
  --activation BEST_ACTIVATION \
  --depth BEST_DEPTH \
  --width BEST_WIDTH \
  --evaluation-steps 50 \
  --device cuda:0
```

最佳模型带bias时增加 `--use-bias`。

---

## 9. 500步连续Rollout

Rollout与离线验证/测试不同：预测状态会传播到下一物理帧。

默认在原测试motion `20–31` 中去掉人工排除和reference非有限motion，然后选择reference residual p95最高的motion，连续rollout 500帧，每帧默认50次网络迭代。

MLP rollout：

```bash
python cloth07_rollout_hardest_motion.py \
  --root cloth_15x15_500step_pipeline \
  --solver mlp \
  --checkpoint PATH/TO/best_validation_model.pt \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

如果要指定motion，而不是自动选择hardest test motion，增加：

```bash
  --motion-index 26
```

Baseline rollout使用同一个入口。默认会读取 `baselines/parameter_selection.json` 中验证集选出的参数：

```bash
python cloth07_rollout_hardest_motion.py \
  --root cloth_15x15_500step_pipeline \
  --solver adam \
  --motion-index 26 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

可选solver：

```text
mlp, gd, adam, lbfgs, bfgs, newton
```

也可以手动覆盖baseline参数：

```bash
python cloth07_rollout_hardest_motion.py \
  --root cloth_15x15_500step_pipeline \
  --solver gd \
  --gd-step-size 5e-5 \
  --motion-index 26 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0

python cloth07_rollout_hardest_motion.py \
  --root cloth_15x15_500step_pipeline \
  --solver lbfgs \
  --initial-step 1.0 \
  --lbfgs-history-size 10 \
  --motion-index 26 \
  --rollout-length 500 \
  --inner-steps 50 \
  --device cuda:0
```

主要输出：

```text
rollouts/motion_XXX/EXPERIMENT_NAME/
├── curve.pt
├── curve.json
└── figures/
    ├── residual_vs_timestep.png
    └── worst_frame_residual_vs_iteration.png
```

`curve.pt` 保存连续rollout的positions、velocities、每帧每次内层迭代residual、每帧reference error和每帧耗时；`curve.json` 保存motion、checkpoint、输出路径和完成帧数摘要。

`figures/residual_vs_timestep.png` 绘制每个物理帧结束时的final residual；脚本会自动选择final residual最大的物理帧，并在 `figures/worst_frame_residual_vs_iteration.png` 中绘制该帧内部的 residual vs. inner iteration。

渲染rollout：

```bash
python cloth14_render_rollout.py \
  --root cloth_15x15_500step_pipeline \
  --rollout cloth_15x15_500step_pipeline/rollouts/motion_XXX/EXPERIMENT_NAME/curve.pt \
  --format mp4 \
  --fps 30
```

默认输出到同目录：

```text
rollouts/motion_XXX/EXPERIMENT_NAME/curve.mp4
```

渲染图中蓝色为模型rollout，灰色为reference对照，红色方块为固定点。标题显示当前帧、该物理帧最后一次内层迭代residual和相对reference的位置误差。

如果500帧渲染较慢，可以使用降采样：

```bash
python cloth14_render_rollout.py \
  --root cloth_15x15_500step_pipeline \
  --rollout cloth_15x15_500step_pipeline/rollouts/motion_XXX/EXPERIMENT_NAME/curve.pt \
  --format mp4 \
  --stride 5 \
  --fps 30
```

如果环境没有可用的FFmpeg，可改用GIF：

```bash
python cloth14_render_rollout.py \
  --root cloth_15x15_500step_pipeline \
  --rollout cloth_15x15_500step_pipeline/rollouts/motion_XXX/EXPERIMENT_NAME/curve.pt \
  --format gif \
  --stride 5
```

批量渲染尚未渲染的rollout时，可以不传 `--rollout`。脚本会扫描 `cloth_15x15_500step_pipeline/rollouts/**/curve.pt`，跳过已经存在同格式输出文件的结果，只渲染剩余项：

```bash
python cloth14_render_rollout.py \
  --root cloth_15x15_500step_pipeline \
  --format mp4 \
  --stride 5 \
  --fps 30
```

如需重新覆盖已有渲染，增加 `--overwrite`。

---

## 10. 显存测试

```bash
python cloth04_probe_memory.py \
  --root cloth_15x15_500step_pipeline \
  --activation relu \
  --depth 10 \
  --width 4096 \
  --sample-count 32 \
  --k 30 \
  --device cuda:0
```

如发生OOM，优先减小训练命令中的 `--sample-chunk-size`，而不是改变训练窗口或optimizer update次数。

---

## 11. 文件说明

```text
cloth_common.py                         公共I/O、曲线统计和50次迭代评估
cloth01_generate_reference_and_samples.py  reference、审计和普通训练样本
cloth02_dataset_catalog.py              完整验证/测试x_n数据集
cloth03_solvers_and_models.py           15×15物理兼容层
cloth04_probe_memory.py                 显存测试
cloth05_train_models.py                 统一模型训练入口
cloth06_select_best.py                  阶段最佳模型选择
cloth07_rollout_hardest_motion.py       500步连续rollout
cloth08_evaluate_baselines.py           GD/Adam/L-BFGS/BFGS/Newton
cloth09_render_reference_motion.py      按motion index渲染reference
cloth10_prepare_initial_point_ablation.py  x_n前缀初值数据
cloth11_train_initial_point_ablation.py 初值数量实验启动器
cloth12_plot_model_vs_baselines.py      模型与baseline曲线对比
cloth13_train_pool.py                   live training pool训练
cloth14_render_rollout.py               渲染连续rollout结果
```

---

## 12. 结果解释注意事项

1. Reference residual反映数值求解质量，不完全等价于视觉难度；必须同时检查曲线和渲染。
2. Motion排除规则必须在查看模型测试结果前固定。
3. 验证集用于baseline参数选择和网络checkpoint选择；测试曲线才是最终泛化指标。
4. 单个物理时间步内50次迭代表现好，不保证500步连续rollout稳定；两者必须分别报告。
5. Full BFGS在669维上非常昂贵，应单独记录运行时间和显存。
6. 宽度、深度和激活函数比较必须保持数据前缀、训练epoch、$K$ curriculum、随机种子和验证协议完全一致。
