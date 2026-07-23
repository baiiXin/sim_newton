# 双卡 Tensor Parallel MLP：显存测试与学习率选择记录

记录日期：2026-07-23  
实验目录：`cloth_tshirt/cloth_tshirt_pipeline/profiling/`

## 结论

正式训练采用以下配置：

| 项目 | 最终值 |
|---|---:|
| 模型 | full-state MLP，ReLU，depth 1，无 bias |
| width | `39936` |
| dtype | `float32` |
| 并行 | 2-rank PyTorch DTensor tensor parallel |
| GPU | `2 × NVIDIA GeForce RTX 3090 24 GB` |
| pool size | `512` |
| batch size | `32` |
| K buckets | `[1, 3, 10, 30]` |
| optimizer | Adam，`foreach=False` |
| **learning rate** | **`5e-8`** |
| gradient clip | global L2 norm `10.0` |

选择 `39936` 是因为它略宽于输入维数，同时在两张 24 GB GPU 上通过了 500-update 完整训练步显存测试。选择 `5e-8` 是因为它在稳定性优先的前提下，兼顾了能量下降、残差下降和有效学习速度。

不要使用最初的 `1e-4` 开始长训练。`1e-4` 虽然没有 OOM，但优化过程严重 overshoot；500-update 测试只能说明显存稳定，不能说明训练稳定。

## 1. 实验环境和网络规模

### 1.1 软件与硬件

| 项目 | 值 |
|---|---|
| Conda 环境 | `cloth_opter` |
| PyTorch | `2.13.0+cu130` |
| CUDA runtime | `13.0` |
| GPU 0 | NVIDIA GeForce RTX 3090，`23.5588 GiB` 可见显存 |
| GPU 1 | NVIDIA GeForce RTX 3090，`23.5568 GiB` 可见显存 |
| seed | `42` |

### 1.2 输入与参数规模

T-shirt 网格有 4424 个顶点：

```text
full_state_dim = 4424 × 3 = 13272
input_dim      = 3 × full_state_dim = 39816
```

输入由当前质量预条件残差、上一残差和上一更新拼接而成。最终隐藏宽度为 `39936`：

```text
width / input_dim = 39936 / 39816 = 1.0030139
```

depth-1、无 bias MLP 的参数量为：

```text
global parameters = 39816 × 39936 + 39936 × 13272
                  = 2,120,122,368

local parameters per rank = 1,060,061,184
```

float32 下，每张卡的本地权重、梯度和两个 Adam 状态的静态估算为 `15.796 GiB`。隐藏层使用 `ColwiseParallel`，输出层使用 `RowwiseParallel`；模型先在 meta device 上建立结构，再只物化本地 shard。

## 2. 显存候选实验

所有候选都执行完整训练步：

```text
pool ask
→ physics residual
→ tensor-parallel MLP forward
→ variational energy
→ backward
→ 双卡全局梯度裁剪
→ Adam
→ pool tell/reset checks
```

显存判据为：

```text
peak_reserved / total_memory <= 0.95
```

### 2.1 width 对比

| width | 全局参数量 | 每卡静态模型+梯度+Adam | pool/batch | 测量 updates | 峰值 allocated | 峰值 reserved | reserved 比例 | 吞吐 | 判定 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40960 | 2,174,484,480 | 16.201 GiB | 512/32 | 3 | 22.486 GiB | 22.639 GiB | 96.10% | 42.51 motions/s | 超过 95% 阈值 |
| **39936** | **2,120,122,368** | **15.796 GiB** | **512/32** | **500** | **21.929 GiB** | **22.080 GiB** | **93.73%** | **41.24 motions/s** | **通过** |

原始结果：

- [width 40960 memory probe](../cloth_tshirt_pipeline/profiling/tensor_parallel_width_40960/memory_probe.json)
- [width 39936, 500-update memory probe](../cloth_tshirt_pipeline/profiling/tp_width_39936_pool512_batch32/memory_probe.json)

`39936` 距离 95% reserved 上限约有 `0.30 GiB` 余量。余量不宽，因此正式训练必须独占两张 GPU，并保持 width、pool、batch 和 dtype 不变。

### 2.2 500-update soak 的解释

最终 soak 配置为：

```text
warmup updates   = 10
measured updates = 500
pool             = 512
batch            = 32
learning rate    = 1e-4
```

结果：

- 两个 rank 都成功完成；
- 两个 rank 的峰值 allocated/reserved 完全一致；
- 峰值 reserved 稳定在 `22.080 GiB`；
- 未发生 OOM 或非有限值异常；
- 500 个 measured updates 用时 `388.01 s`；
- 吞吐 `41.24 motions/s`。

但是末步 loss 为 `8.39e6`、residual ratio p95 为 `2.99e4`，梯度范数为 `2.20e8 → clip 10`。因此它只通过了内存和进程稳定性检查，优化配置没有通过。

## 3. 分片 checkpoint/resume 验收

使用 `cloth21_train_tensor_parallel_online.py` 完成了最短 GPU checkpoint smoke：

1. 训练到 update 2；
2. 写入 DCP 模型/Adam shards 和两个 rank-local runtime sidecar；
3. 使用 `--resume` 从 update 2 恢复；
4. 继续训练到 update 3；
5. `latest.json` 前进到 generation 1。

最终指针：

```json
{
  "checkpoint": "checkpoints/step_000000003_gen_000001",
  "checkpoint_generation": 1,
  "update_count": 3
}
```

验收结果：

| 检查项 | 结果 |
|---|---|
| checkpoint `COMPLETE` marker | 存在 |
| DCP model + Adam shards | 约 24 GiB |
| `runtime_rank_00.pt` | 约 182 MiB |
| `runtime_rank_01.pt` | 约 182 MiB |
| pool state 是否续接 | 是，最终 `total_environment_updates=96` |
| update 是否从 2 前进到 3 | 是 |
| plotting error | 无 |
| interrupted | `false` |

相关文件：

- [latest.json](../cloth_tshirt_pipeline/profiling/tp_checkpoint_smoke/latest.json)
- [completed.json](../cloth_tshirt_pipeline/profiling/tp_checkpoint_smoke/completed.json)
- [restored checkpoint manifest](../cloth_tshirt_pipeline/profiling/tp_checkpoint_smoke/checkpoints/step_000000003_gen_000001/manifest.json)

该 smoke 使用 `1e-4` 且跳过 validation，其目的只是在目标 GPU、NCCL 和磁盘文件系统上验证 DCP 保存/恢复，不用于选择学习率。

## 4. 学习率 sweep

### 4.1 固定实验协议

除 learning rate 和独立输出目录外，所有实验保持完全相同：

```text
width            = 39936
activation       = ReLU
depth            = 1
use_bias         = false
dtype            = float32
seed             = 42
pool_size        = 512
batch_size       = 32
k_buckets        = [1, 3, 10, 30]
warmup_updates   = 10
measured_updates = 500
gradient_clip    = 10.0
```

每个实验实际执行 510 次更新。表中的 loss、energy increase、residual ratio、update norm 和 last resets 是最后一个 batch 的指标。`cumulative resets` 根据以下关系推导：

```text
cumulative resets = total_sampled_motions - initial_pool_size
                  = total_sampled_motions - 512
```

510 次更新共处理：

```text
510 × 32 = 16320 batch rows
```

由于 max lifetime 是 500 个物理帧，而本轮短实验远未达到该寿命，累计重新采样主要反映几何/数值 reset。

### 4.2 完整结果

| LR | loss | energy increase | residual p50 | residual p95 | 末步 resets | 累计 resets | reset 率 | update norm mean | 吞吐 motions/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1e-6` | 594.14 | 56.25% | 3.5335 | 5.0973 | 14 | 7173 | 43.95% | 0.3617 | 74.12 |
| `3e-7` | 294.19 | 59.38% | 1.9194 | 2.2956 | 9 | 2812 | 17.23% | 0.2633 | 122.99 |
| `2e-7` | -37.53 | 28.12% | 0.9458 | 1.3514 | 2 | 1560 | 9.56% | 0.1344 | 151.65 |
| `1.5e-7` | -19.63 | 34.38% | 1.1889 | 1.5021 | 0 | 1400 | 8.58% | 0.1094 | 156.18 |
| `1.2e-7` | -37.41 | 0% | 0.8448 | 1.0088 | 0 | 440 | 2.70% | 0.0986 | 192.10 |
| `1e-7` | -25.07 | 15.62% | 0.9011 | 1.1157 | 0 | 84 | 0.515% | 0.0769 | 210.16 |
| `7e-8` | -42.73 | 0% | 0.9685 | 0.9871 | 0 | 10 | 0.061% | 0.1022 | 214.20 |
| `6e-8` | -37.17 | 0% | 0.9641 | 0.9876 | 0 | 10 | 0.061% | 0.0906 | 214.12 |
| **`5e-8`** | **-30.81** | **0%** | **0.9615** | **0.9830** | **0** | **7** | **0.043%** | **0.0771** | **214.84** |
| `4e-8` | -23.01 | 0% | 0.9635 | 0.9806 | 0 | 7 | 0.043% | 0.0619 | 214.07 |

所有结果的两个 rank 指标完全一致，峰值 reserved 都约为 `22.084 GiB`。

原始结果目录：

- [1e-6](../cloth_tshirt_pipeline/profiling/tp_lr1e-6_width39936_pool512_batch32/memory_probe.json)
- [3e-7](../cloth_tshirt_pipeline/profiling/tp_lr3e-7_width39936_pool512_batch32/memory_probe.json)
- [2e-7](../cloth_tshirt_pipeline/profiling/tp_lr2e-7_width39936_pool512_batch32/memory_probe.json)
- [1.5e-7](../cloth_tshirt_pipeline/profiling/tp_lr1p5e-7_width39936_pool512_batch32/memory_probe.json)
- [1.2e-7](../cloth_tshirt_pipeline/profiling/tp_lr1p2e-7_width39936_pool512_batch32/memory_probe.json)
- [1e-7](../cloth_tshirt_pipeline/profiling/tp_lr1e-7_width39936_pool512_batch32/memory_probe.json)
- [7e-8](../cloth_tshirt_pipeline/profiling/tp_lr7e-8_width39936_pool512_batch32/memory_probe.json)
- [6e-8](../cloth_tshirt_pipeline/profiling/tp_lr6e-8_width39936_pool512_batch32/memory_probe.json)
- [5e-8](../cloth_tshirt_pipeline/profiling/tp_lr5e-8_width39936_pool512_batch32/memory_probe.json)
- [4e-8](../cloth_tshirt_pipeline/profiling/tp_lr4e-8_width39936_pool512_batch32/memory_probe.json)

### 4.3 选择逻辑

选择顺序为：

1. 排除持续发生几何 reset、非有限值或能量上升的配置；
2. 在稳定配置中优先选择 residual p50/p95 都小于 1 的配置；
3. 比较累计 reset、能量下降强度和 update norm；
4. 给正式长训练保留一定的学习率稳定余量。

由此可得：

- `>=1.5e-7` 的累计 reset 过高，不能用于长训练；
- `1.2e-7` 虽然末步良好，但累计 reset 达到 440，说明稳定性不足；
- `1e-7` 仍有 84 次累计 reset，且末步 15.62% 样本能量上升；
- `4e-8` 到 `7e-8` 都进入稳定区域；
- `7e-8` 能量下降最强，但 update tail 和累计 reset 略高，靠近稳定区上沿；
- `4e-8` 与 `5e-8` 的累计 reset 同为 7，但 `5e-8` 的平均能量下降更强、residual p50 更好；
- 因此正式长训练选择 **`5e-8`**，`7e-8` 只保留为更激进的备选。

这里的“最佳”特指 seed 42、510-update 短期 sweep 下，按稳定性优先准则得到的工程最优值。最终泛化质量仍由正式训练中的 frozen validation 决定。

## 5. float32 下 `5e-8` 是否会被舍入

不会发生系统性的“大数吃小数”。`5e-8` 是 Adam learning rate，不是直接乘在 clip 后 RMS 梯度上作为 SGD 更新。

`5e-8` 实验最后一步的梯度数据为：

| 指标 | 数值 |
|---|---:|
| 全局梯度 L2 norm，clip 前 | 316.376 |
| 全局梯度 L2 norm，clip 后 | 10.0 |
| clip 比例 | 0.03161 |
| 单参数 RMS 梯度，clip 前 | `6.87e-3` |
| 单参数 RMS 梯度，clip 后 | `2.17e-4` |

Adam 使用：

```text
delta_weight ≈ -learning_rate × m_hat / (sqrt(v_hat) + eps)
```

一阶、二阶矩会对梯度尺度归一化，因此有效参数更新通常处于 `learning_rate × O(1)`，而不是 `learning_rate × raw_gradient`。

隐藏层初始化权重标准差约为 `0.007087`。这个量级的 float32 ULP 约为 `4.66e-10`，所以 `5e-8` 相当于约 107 个 ULP，可以明确写入权重。当前运行时还满足：

```text
torch.get_float32_matmul_precision()        = highest
torch.backends.cuda.matmul.allow_tf32       = False
```

Linear 没有使用 TF32 截短尾数。实验中模型实际产生的单坐标 RMS 更新约为 `6.69e-4 m`，单顶点位移 RMS 约为 `1.16e-3 m`，也直接证明整体更新没有消失。

个别梯度极小的参数在单步中仍可能舍入为零，这是正常现象；整体模型更新和训练信号没有被 float32 吞掉。

## 6. 正式训练命令

在 `cloth_opter` 环境和 `cloth_tshirt/` 目录运行：

```bash
python cloth21_train_tensor_parallel_online.py \
  --run-dir cloth_tshirt_pipeline/tensor_parallel/activation_relu_depth_01_width_39936_no_bias_lr5e-8/seed_42 \
  --fixed-data-dir fixed_data \
  --devices 0 1 \
  --dtype float32 \
  --seed 42 \
  --activation relu \
  --width 39936 \
  --no-use-bias \
  --pool-size 512 \
  --batch-size 32 \
  --k-buckets 1 3 10 30 \
  --learning-rate 5e-8 \
  --gradient-clip-norm 10.0 \
  --step-regularization-weight 0.0 \
  --max-updates 3000000 \
  --max-wall-hours 10 \
  --log-interval 100 \
  --checkpoint-interval 5000 \
  --keep-checkpoints 2 \
  --fast-validation-interval 10000 \
  --checkpoint-validation-interval 50000 \
  --fast-rollout-frames 32 \
  --checkpoint-rollout-frames 100 \
  --validation-batch-size 4
```

恢复训练时必须保持 mesh、dtype、网络、pool、K buckets、learning rate、gradient clip、step regularization、两 rank 拓扑和 PyTorch 版本不变。使用相同命令，把累计 wall-time 预算提高，并添加 `--resume`。例如第二段将总预算提高到 20 小时：

```text
--max-wall-hours 20 --resume
```

注意 `--max-wall-hours` 是包含 checkpoint 中已保存用时的累计预算，不是每次启动新增的时长。

## 7. 后续运行时检查

正式训练开始后，优先观察：

- `training_log.csv` 中的 `resets_total`、`resets_area` 和 `resets_edge`；
- `energy_increase_fraction` 是否长期接近 0；
- residual ratio p50/p95 是否保持在 1 附近并逐渐下降；
- update norm 是否出现持续上升；
- gradient norm 是否出现非有限值；
- fast/full frozen validation，而不是只看在线训练 batch；
- reserved memory 是否仍稳定在约 `22.08 GiB`。

如果正式训练后期累计 reset 明显上升，应优先暂停并检查曲线，不要因为 probe 的 510-update 结果稳定就假定 300 万 updates 全程一定稳定。
