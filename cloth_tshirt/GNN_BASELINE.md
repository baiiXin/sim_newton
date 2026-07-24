# T-shirt 共享权重 GNN 基线

本基线在保留现有 dense MLP 路径的同时，新增了一条并行的 GNN 路径。它复用当前的在线随机训练池、能量损失、冻结的验证集与测试集、失败检查，以及 single-motion 评估协议。

## 网络结构

对每个顶点，拼接三个世界坐标系下的三维向量：

- 当前的**原始**驻值残差；
- 上一次的原始残差；
- 上一次预测的位置增量。

得到的 9 维特征由两层 `9 -> 128 -> 128` ReLU MLP 编码。随后模型执行 5 轮消息传递，所有轮次共享同一个边 MLP 和同一个节点更新 MLP：

1. 每条无向网格边产生两条有向消息；
2. 每条消息以 `[接收节点隐状态, 发送节点隐状态]` 为输入，经过共享的 `256 -> 128 -> 128` ReLU MLP；
3. 每个顶点对收到的消息求和；
4. 节点更新以 `[节点隐状态, 消息和]` 为输入，经过共享的 `256 -> 128 -> 128` MLP：第一层后使用 ReLU，末层为线性输出；
5. 节点更新通过残差连接加回节点隐状态。

解码器为 `128 -> 128 -> 3`，只在第一个线性层后使用 ReLU。所有线性层均不含 bias。节点更新 MLP 的最后一层和解码器的最后一层使用零初始化。节点更新末层不能再接 ReLU，否则零初始化点的梯度恒为零，processor 将无法学习。

第一版基线有意保持简单，不包含：

- 残差的质量预条件；
- 输入中的固定点标记；
- 边属性；
- 输入归一化或输出缩放因子；
- 5 轮消息传递各自独立的 processor 参数。

固定顶点的输出仍会被硬门控为零，并由现有流程精确投影。

## 单元测试

在 `cloth_tshirt/` 目录运行：

```bash
python -m unittest -v test_tshirt_gnn.py
```

## 峰值显存测试

GNN 的 5 轮消息传递会在反向传播期间保留边和节点激活，不能使用 dense MLP 的显存结果估算训练 batch。开始长时间训练前，应在实际训练所用的 GPU、PyTorch 版本和 dtype 上扫描完整训练步的峰值显存：

```bash
python cloth06_probe_memory_and_throughput.py \
  --model-type gnn \
  --output-dir cloth_tshirt_gnn_pipeline/profiling/memory_probe \
  --fixed-data-dir fixed_data \
  --device cuda:0 \
  --dtype float64 \
  --seed 42 \
  --activation relu \
  --depth 2 \
  --width 128 \
  --no-use-bias \
  --pool-size 512 \
  --batch-sizes 4 8 16 32 \
  --warmup-updates 1 \
  --measured-updates 3 \
  --memory-headroom-fraction 0.85
```

每个候选 batch 都在全新的子进程中运行，执行完整的 `ask -> residual/GNN -> energy -> backward -> Adam -> tell/reset checks`。预热结束后会重置 CUDA 峰值统计，再对测量阶段计时并记录：

- `peak_allocated_gib`：PyTorch 张量实际占用的最大显存；
- `peak_reserved_gib`：PyTorch CUDA 缓存分配器保留的最大显存；
- `peak_allocated_fraction` / `peak_reserved_fraction`：上述数值占 GPU 总显存的比例；
- `motions_per_second`：完整训练步的实测吞吐量；
- `status`：`success`、`oom`、`failed` 或 `worker_crashed`。

汇总结果写入 `memory_probe.json` 和 `memory_probe.csv`。只有 `peak_reserved_fraction <= 0.85` 的成功结果会参与推荐；其中吞吐量最高的 batch 写入 `recommended_training_config.json`。正式训练应使用其中的 `recommended_batch_size`，并保持 `model_type=gnn`、dtype、width、pool size 与测试一致。不同 GPU 上的峰值显存不同，因此文档不写死某个硬件的测量值。

`--model-type gnn` 会强制检查本基线的 ReLU、depth 2、无 bias 和 5 轮消息传递配置，避免误用 dense MLP 的显存数据。若 batch 4 仍然 OOM，可先减小 `--pool-size` 以降低常驻训练池显存，或改用 `float32` 后重新测试；pool size 和每个 batch 都必须能被 4 个 K buckets 整除。

## 训练

5 轮消息传递会为反向传播保留大量边和节点激活，因此应从峰值显存测试推荐的小 batch 开始。以下示例使用 batch 4：

```bash
python cloth17_train_gnn_online.py \
  --device cuda:0 \
  --dtype float64 \
  --pool-size 512 \
  --batch-size 4 \
  --max-wall-hours 10
```

默认输出根目录为 `cloth_tshirt_gnn_pipeline/`。默认实验名会记录原始残差输入、5 轮共享权重消息传递、宽度 128、两层 MLP 和无 bias 配置。

## 完整冻结验证与测试

```bash
python cloth18_evaluate_gnn_checkpoint.py \
  --checkpoint cloth_tshirt_gnn_pipeline/gnn_raw_residual_mp05_width_0128_depth_02_no_bias/seed_42/best_validation_model.pt \
  --rollout-frames 500 \
  --inner-steps 50
```

## Single-motion 网络 rollout

```bash
python cloth19_rollout_gnn_single_motion.py \
  --mode network \
  --split typical \
  --motion-index 0 \
  --checkpoint /path/to/best_validation_model.pt \
  --rollout-frames 500 \
  --inner-steps 50
```
