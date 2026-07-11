# Initial-Point Ablation Dataset-Style Evaluation

这里用和 `training_pool` 相同的 dataset-style 评测方式，评估：

```text
cloth_5x5_500step_project/cloth_5x5_initial_sample_ablation/
```

中的 sample-count 消融模型：

```text
points_0001
points_0008
points_0032
points_0064
points_0128
points_1024
```

当前这些目录里可用的模型都是：

```text
activation_identity_depth_01_width_256_no_bias
```

## 评测方式

这不是 continuous rollout。每个测试样本从 dataset 的 `initial_y` 独立开始，运行固定次数 learned optimizer iteration，统计：

```text
residual
energy_gap
exact_error
particle_mean_error
particle_max_error
spring_length_error
fixed_vertex_max_error
```

并输出 pooled、per-motion、per-problem、worst-motion 指标。

## 数据集

默认直接复用：

```text
cloth_5x5_500step_project/cloth_5x5_500step_pipeline/data/datasets/
```

默认测试 split 和原版测试一致：

```text
validation
seen_extrap
unseen_id
ood
```

## 运行

在仓库根目录运行：

```bash
conda run -n hood python unit_test_for_spring/multi_motion_cloth/scale_down_de_initial_points/evaluate_initial_point_ablation_tests.py \
  --device cuda:0 \
  --point-groups points_0001 points_0008 points_0032 points_0064 points_0128 points_1024 \
  --steps 50 \
  --batch-size 8192 \
  --overwrite
```

如果当前目录已经是 `cloth_5x5_500step_project/`，脚本路径要退回上一层：

```bash
python ../unit_test_for_spring/multi_motion_cloth/scale_down_de_initial_points/evaluate_initial_point_ablation_tests.py \
  --device cuda:0 \
  --point-groups points_0001 points_0008 points_0032 points_0064 points_0128 points_1024 \
  --steps 50 \
  --batch-size 8192 \
  --overwrite
```

如果当前机器没有 CUDA，可以用 `--device cpu` 做 smoke test，但完整评测会慢很多。

## 输出

默认输出到本目录：

```text
unit_test_for_spring/multi_motion_cloth/scale_down_de_initial_points/
├── run_config.json
├── summary_metrics.csv
├── all_metrics.json
├── figures/
│   └── <dataset>_residual_mean_by_iteration.png
└── points_XXXX/<model>/<dataset>/
    ├── metrics.json
    └── curves.pt
```

如果不想保存完整曲线，可以加：

```bash
--skip-curves
```
