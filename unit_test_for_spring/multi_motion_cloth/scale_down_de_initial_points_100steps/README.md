# 100-Step 3x69D History-Input Dataset-Style Evaluation

这里评估两个旧 100-step nonlinear 实验目录中的模型：

```text
unit_test_for_spring/multi_motion_cloth/nonlinear/fixed_left_edge_5x5_cloth_history_input_default_init_ablation
unit_test_for_spring/multi_motion_cloth/nonlinear/fixed_left_edge_5x5_cloth_degenerate_no_initial_perturbation_no_repetition
```

这两个脚本的 `MLPOptimizer` 定义一致：

```text
input  = [current residual, previous residual, previous update] = 3 x 69D
output = 69D free-state displacement update
```

区别主要是 `ModelSpec.experiment_name` 的目录名前缀，以及原训练数据语义不同。因此这里使用一个统一的 69D evaluator，并用同一套测试集评估两组模型。

## 测试集

默认测试集由：

```text
fixed_left_edge_5x5_cloth_history_input_default_init_ablation.py
```

按原始确定性逻辑重新生成。默认 split：

```text
seen_motion_temporal_interpolation
seen_motion_temporal_extrapolation
unseen_id_test
ood_test
```

每个 problem 使用原脚本的 `eval_points_per_problem = 128` 个 Sobol 初始状态。

## 运行

从仓库根目录运行：

```bash
python unit_test_for_spring/multi_motion_cloth/scale_down_de_initial_points_100steps/evaluate_100step_69d_models.py \
  --device cuda:0 \
  --sources history degenerate \
  --steps 50 \
  --batch-size 8192 \
  --overwrite
```

如果只想先测试一个小模型：

```bash
python unit_test_for_spring/multi_motion_cloth/scale_down_de_initial_points_100steps/evaluate_100step_69d_models.py \
  --device cpu \
  --sources history \
  --activations identity \
  --depths 1 \
  --widths 69 \
  --datasets unseen_id_test \
  --steps 1 \
  --batch-size 512 \
  --overwrite
```

如果当前目录是 `cloth_5x5_500step_project/`，脚本路径要退回上一层：

```bash
python ../unit_test_for_spring/multi_motion_cloth/scale_down_de_initial_points_100steps/evaluate_100step_69d_models.py \
  --device cuda:0 \
  --sources history degenerate \
  --steps 50 \
  --batch-size 8192 \
  --overwrite
```

## 输出

默认输出到本目录：

```text
unit_test_for_spring/multi_motion_cloth/scale_down_de_initial_points_100steps/
├── run_config.json
├── summary_metrics.csv
├── all_metrics.json
├── figures/
│   └── <dataset>_residual_mean_by_iteration.png
├── history/<model>/<dataset>/metrics.json
└── degenerate/<model>/<dataset>/metrics.json
```

## 只画 1x256 六个模型

评测完成后，如果只想画 `depth=1, width=256` 的三种激活函数，并比较两组来源共六个模型，运行：

```bash
python unit_test_for_spring/multi_motion_cloth/scale_down_de_initial_points_100steps/plot_1x256_six_models.py
```

输出：

```text
figures_1x256/
├── overview_1x256_residual_mean_by_step.png
├── seen_motion_temporal_interpolation_1x256_residual_mean_by_step.png
├── seen_motion_temporal_extrapolation_1x256_residual_mean_by_step.png
├── unseen_id_test_1x256_residual_mean_by_step.png
└── ood_test_1x256_residual_mean_by_step.png
```
