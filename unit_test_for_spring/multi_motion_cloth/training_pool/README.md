# Training Pool Dataset-Style Evaluation

这里用于比较两组 75D full-state learned optimizer：

- `full_500step`：原版 500-step dataset 训练模型。
- `pool`：Metamizer-style pool 训练模型。

评测方式是原版测试集评估，不是 continuous rollout。每个测试样本从 dataset 里的 `initial_y` 独立开始，运行固定次数 learned optimizer iteration，统计 residual、energy gap、exact error、particle error、spring length error，并输出 pooled、per-motion、per-problem、worst-motion 指标。

## 数据集确认

默认直接复用：

```text
cloth_5x5_500step_project/cloth_5x5_500step_pipeline/data/datasets/
```

默认测试 split 和 `README_cloth_5x5_500step_pipeline.md` / `cloth05_train_models.py` 一致：

```text
validation
seen_extrap
unseen_id
ood
```

如果要额外测试 current-state 数据集，可以显式传：

```bash
--datasets validation seen_extrap unseen_id ood \
  current_state_seen_extrap current_state_unseen_id current_state_ood
```

## 正式运行

在仓库根目录运行：

```bash
conda run -n hood python unit_test_for_spring/multi_motion_cloth/training_pool/evaluate_pool_vs_full_dataset_tests.py \
  --device cuda:0 \
  --groups full_500step pool \
  --activations identity relu tanh \
  --depths 1 \
  --widths 256 \
  --steps 50 \
  --batch-size 8192 \
  --overwrite
```

当前脚本会自动处理原版模型路径：

- 优先查 `cloth_5x5_500step_pipeline/models/activation_*`
- 如果没有，则 fallback 到当前已有的 `cloth_5x5_500step_pipeline/models/old/activation_*`

## 输出

默认输出到本目录：

```text
unit_test_for_spring/multi_motion_cloth/training_pool/
├── run_config.json
├── summary_metrics.csv
├── all_metrics.json
├── figures/
│   └── <dataset>_residual_mean_by_iteration.png
├── full_500step/<model>/<dataset>/metrics.json
├── full_500step/<model>/<dataset>/curves.pt
└── pool/<model>/<dataset>/metrics.json
```

如果不想保存完整曲线，可以加：

```bash
--skip-curves
```
