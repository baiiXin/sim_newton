# 5×5 布料数据 Scale-up 独立流水线

这套脚本把原单文件实验拆成四个阶段：

1. `build_scaled_datasets.py`：只生成参考轨迹、训练数据集和共享测试基准；
2. `evaluate_baselines.py`：只在现成 benchmark 上选择并评估 Newton、GD、L-BFGS，结果永久缓存；
3. `train_scaled_mlp.py`：只读取训练集和 `validation_core` 训练网络；
4. `evaluate_scaled_mlp.py`：只读取 checkpoint 和 benchmark 测试网络，并合并已经缓存的 baseline 结果。

公共物理、数据格式、网络和评估函数集中在 `cloth_scale_common.py`。

---

## 1. 固定点表示

所有问题都统一使用完整的 25 个顶点：

- 状态、参考解和网络输出：`25 × 3 = 75` 维；
- 固定点 mask：25 维布尔量，仅用于物理约束、残差屏蔽和输出门控，不输入网络；
- 固定目标位置：75 维；
- 固定点不从网络历史状态中删除；
- 网络输出后将固定点更新显式置零；
- stationarity residual 只统计自由坐标，固定点上的约束反力不计入未收敛残差。

网络输入按照顶点交错排列。每个顶点有 9 个特征：

```text
[current residual (3), previous residual (3), previous update (3)]
```

因此总输入维度为 `25 × 9 = 225`，输出维度始终为 75。固定点 one-hot 已从网络输入中删除。


> 兼容性说明：该修改改变了第一层输入维度，旧的 250D one-hot checkpoint 不能继续训练或直接评估；数据集、参考轨迹和 baseline 缓存仍可复用。请从头训练新的 225D 网络。

当前这轮数据集的固定目标均为静止构型位置。`fixed_target` 已经保存，为后续加入固定点偏移和运动边界留好接口。

---

## 2. 首轮训练数据集

| 数据集 | 固定配置数 | Motion 数 | 时间点数 | 每问题状态数 | 物理问题数 | 训练样本数 | 主要问题 |
|---|---:|---:|---:|---:|---:|---:|---|
| D0 | 1 | 16 | 16 | 32 | 256 | 8,192 | 新 75D 表示能否复现旧分布结果 |
| D1-B | 1 | 64 | 16 | 8 | 1,024 | 8,192 | 总量相同，扩大 motion 覆盖是否更有效 |
| D1-L | 1 | 64 | 16 | 32 | 1,024 | 32,768 | motion 覆盖和总数据量同时扩大 |
| D2-B | 8 个双固定点配置 | 16 | 16 | 4 | 2,048 | 8,192 | 总量相同，扩大固定边界覆盖是否更有效 |
| D2-L | 8 个双固定点配置 | 16 | 16 | 32 | 2,048 | 65,536 | 固定边界覆盖和局部密度同时扩大 |
| D4-M | 24 个 mask，`k∈{1,2,3,5}` | 64 | 16 | 8 | 24,576 | 196,608 | motion 与固定条件联合 scale-up |

64 个训练 motion 是嵌套构造的：D0 的 16 个 motion 是 64 个 motion 的子集。D2 的 8 个双固定点配置也是 D4-M 的 24 个固定配置的子集。

每个问题默认保留当前物理状态和精确解两个显式状态，其余点使用参考解附近的 Sobol 采样。这与原脚本的采样哲学一致。后续若研究“初值类型”消融，应另建 Sobol-only、多尺度、结构模态和模型生成状态版本，不要改动现有数据集。

---

## 3. D4-M 的 24 个训练固定配置

- 6 个单固定点：四角、上边中点、左边中点；
- 8 个双固定点：同边远距、同边相邻、对角、对边、角点加边中点、非对称组合；
- 6 个三固定点：连续边界、分散角点、边中点和非对称组合；
- 4 个五固定点：完整上、下、左、右边。

固定点数量 4 被有意留出，作为 `count_ood` 测试。

---

## 4. 共享 benchmark

所有数据规模和所有网络共享同一套 benchmark，保证 scaling 曲线可比较。

| Split | 含义 |
|---|---|
| `validation_core` | 未见 motion，加上 legacy 和未见固定 mask 的混合验证集；用于 checkpoint 选择及 baseline 参数选择 |
| `state_id_legacy` | legacy 固定边界、训练 anchor motion、未见时间和初值 |
| `motion_generalization_legacy` | legacy 固定边界、完全未见的 ID motion |
| `boundary_generalization_seen_motion` | 未见固定 mask、所有训练集都见过的 anchor motion |
| `joint_generalization` | 固定 mask 与 motion 均未见 |
| `count_ood` | 固定点数量 4，训练数据中没有该数量 |
| `hard_ood` | 无固定点或内部顶点固定，并配合 OOD motion |

测试划分的单位是完整固定配置和完整 motion，而不是把同一轨迹附近的随机点拆到训练和测试两边。

---

## 5. 数据存储

每个训练数据目录包含：

```text
training/D0/
├── manifest.json
├── problems.pt
└── train.pt
```

`problems.pt` 每个物理问题只保存一次：

- `q`
- `masses`
- `exact_y`
- `current_y`
- `fixed_mask`
- `fixed_target`
- `sampling_radius`
- boundary / motion / time 索引
- 参考能量和参考 residual

`train.pt` 只保存：

- `initial_y`
- `problem_index`

因此不会为同一个问题的多个初值重复保存 `q、mass、exact_y、mask`。

---

## 6. 参考轨迹缓存

参考轨迹按 `(物理配置, 参考求解器设置, 固定配置, motion)` 缓存。D0、D1、D2、D4 之间会自动复用已有轨迹。

缓存 key 不只使用整数索引，还包含完整 boundary/motion 内容哈希，防止修改参数后误用旧文件。脚本被中断后，重新运行会跳过已完成轨迹。

CPU 可使用多进程：

```bash
python build_scaled_datasets.py \
  --datasets all \
  --build-benchmark \
  --output-root /data/zhoucy/sim_newton/unit_test_for_spring/scaled_cloth_data \
  --device cpu \
  --workers 16
```

使用 CUDA 生成参考轨迹时保持单进程：

```bash
python build_scaled_datasets.py \
  --datasets all \
  --build-benchmark \
  --output-root /data/zhoucy/sim_newton/unit_test_for_spring/scaled_cloth_data \
  --device cuda:0 \
  --workers 1
```

也可以逐组生成；缓存会复用：

```bash
python build_scaled_datasets.py --datasets D0 D1-B D2-B --build-benchmark --device cpu --workers 16
python build_scaled_datasets.py --datasets D1-L D2-L --device cpu --workers 16
python build_scaled_datasets.py --datasets D4-M --device cpu --workers 16
```

只构造部分 benchmark：

```bash
python build_scaled_datasets.py \
  --datasets D0 \
  --build-benchmark \
  --benchmark-splits validation_core state_id_legacy \
  --device cpu --workers 8
```

---

## 7. Baseline 只评估一次

```bash
python evaluate_baselines.py \
  --benchmark-root /data/zhoucy/sim_newton/unit_test_for_spring/scaled_cloth_data/benchmark_v1 \
  --output-root /data/zhoucy/sim_newton/unit_test_for_spring/scaled_cloth_data/baseline_results \
  --device cuda:0 \
  --steps 50
```

流程：

- GD 在 `validation_core` 上从候选步长中选择一次；
- L-BFGS 在 `validation_core` 上从 memory `{5,10,20}` 中选择一次；
- Newton、选定的 GD 和选定的 L-BFGS 在所有测试 split 上评估；
- 结果写入带 `baseline_id` 的独立目录；
- 后续网络评估只读取 JSON，不再运行 baseline。

这里的参考解 Newton 和 baseline Newton 是两件事：前者是带阻尼和线搜索的数据标签生成器；后者是从测试初值出发、固定迭代次数的标准 full-step Newton 对照。

---

## 8. 网络训练

例如在 D1-B 上只训练 identity / ReLU / Tanh，宽度 75、128、256，深度 1、2、5、10：

```bash
python train_scaled_mlp.py \
  --dataset-dir /data/zhoucy/sim_newton/unit_test_for_spring/scaled_cloth_data/training/D1-B \
  --benchmark-root /data/zhoucy/sim_newton/unit_test_for_spring/scaled_cloth_data/benchmark_v1 \
  --output-root /data/zhoucy/sim_newton/unit_test_for_spring/scaled_cloth_runs \
  --device cuda:0 \
  --activations identity relu tanh \
  --depths 1 2 5 10 \
  --widths 75 128 256 \
  --batch-size 8192
```

列出配置编号：

```bash
python train_scaled_mlp.py ... --list-configs
```

只运行一个配置：

```bash
python train_scaled_mlp.py ... --config-index 7
```

恢复中断训练：

```bash
python train_scaled_mlp.py ... --config-index 7 --resume
```

训练脚本默认为 mini-batch，因为 D4-M 已不适合所有配置都使用 full batch。默认训练为 500 epoch，K 每 100 epoch 按 `{1,3,5,10,30}` 增长。若要严格复刻旧脚本的 5000 epoch 调度：

```bash
--epochs 5000 --epochs-per-k 1000
```

注意：数据规模扩大后，“一个 epoch”包含的样本处理量也同步扩大，因此不能只用 epoch 数比较训练计算量。

---

## 9. 网络测试与已缓存 baseline 合并

```bash
python evaluate_scaled_mlp.py \
  --checkpoints \
    /path/to/model_A/best_validation_model.pt \
    /path/to/model_B/best_validation_model.pt \
  --benchmark-root /data/zhoucy/sim_newton/unit_test_for_spring/scaled_cloth_data/benchmark_v1 \
  --baseline-summary /data/zhoucy/sim_newton/unit_test_for_spring/scaled_cloth_data/baseline_results/<baseline_id>/summary.json \
  --output-root /data/zhoucy/sim_newton/unit_test_for_spring/scaled_cloth_evaluations \
  --device cuda:0 \
  --steps 50
```

输出：

- 每个 checkpoint 的完整 JSON；
- `combined_comparison.json`；
- `combined_comparison.csv`。

评估统计包括 pooled mean / median / p95 / max，以及 worst-boundary 和 worst-motion 的 p95 / max。

---

## 10. 冒烟测试

快速检查数据格式和脚本链路：

```bash
python build_scaled_datasets.py \
  --datasets D0 \
  --build-benchmark \
  --benchmark-splits validation_core \
  --smoke-test \
  --device cpu
```

`--smoke-test` 只构造一个 boundary、一个 motion、两个时间步和少量状态，不代表正式实验。
