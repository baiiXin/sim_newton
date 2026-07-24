# T-shirt 在线随机 dynamics 学习优化器

这个目录把 `cloth_15x15_500step_project_scale_up` 扩展到 4424 顶点的 T-shirt 网格，使用
`t-shirt/tshirt_from_garment_meshes.obj`。固定模型只生成一次；训练 motion 在环境 reset 时在线随机生成且不落盘；验证集、测试集与 4 个典型 single motions 从同一 dynamics 分布独立采样后冻结。

当前版本仍是无碰撞的隐式薄壳基线，接口和高频速度分布已经为后续加入身体/自碰撞保留，但本次没有悄悄引入碰撞能量。

## 已固定的实验约定

| 范畴 | 约定 |
|---|---|
| 网格 | 4424 vertices / 8710 triangles / 12994 interior hinges |
| 固定点 | 左右肩各取前、后表面一个顶点，共 4 点 |
| 材料 | seed 42，从 HOOD 范围抽一组后永久固定 |
| dynamics | Haar-uniform 刚体旋转、小幅位置扰动、刚体 + 平滑 + 逐顶点高频速度 |
| 训练数据 | pool reset 时在线采样，不保存 individual motions；checkpoint 保存 RNG/pool 状态 |
| 固定评估 | validation 32、test 64、typical single motions 4 |
| 网络 | full-state MLP；输入为当前质量预条件残差、上一残差、上一更新；输出完整位置增量并硬门控固定点 |
| 默认网络 | ReLU、depth 1、width 2048、无 bias、float64、Adam `1e-4`、gradient clip 10 |
| 双卡正式网络 | ReLU、depth 1、width 39936、无 bias、float32、2-rank tensor parallel、pool 512、batch 32 |
| K buckets | `[1, 3, 10, 30]` |
| 快速验证 | 每物理帧固定 15 次 learned updates，不提前停止 |
| checkpoint validation / 最终 validation / test | 每物理帧固定 50 次，不提前停止；`1e-3` 只作为统计阈值 |
| single motion | baseline 与 network 均每物理帧固定 50 次，不提前停止 |
| baseline | 固定步长原始 GD；质量预条件线搜索 GD；3×3 Hessian block 预条件线搜索 GD |
| 线搜索 | 最多 12 次 objective trials，含初始步长的尝试 |

冻结验证/测试初值是有意的：如果测试时也在线重采样，模型之间和训练时刻之间会混入 Monte-Carlo 抽样噪声，回归结果不可复现，少量异常样本还会显著改变失败率。冻结的是独立随机抽取的初值，不是参考解或轨迹；训练仍保持无限在线采样。

## 每个物理帧的优化目标与能量

每个物理帧将自由顶点的 implicit Euler 更新写成一个无约束变分问题；固定点在求值前直接投影到目标位置：

$$
\begin{aligned}
q_i &= x_i^n + \Delta t\,v_i^n + \Delta t^2 g, \\
\mathcal{E}(y;q)
&= \sum_{i\in\mathcal{V}_{\mathrm{free}}}
   \frac{m_i}{2\Delta t^2}\lVert y_i-q_i\rVert_2^2
   + E_m(y) + E_b(y).
\end{aligned}
$$

膜项采用 NVIDIA Newton VBD 的 stable Neo-Hookean density，并按静止三角形面积和厚度积分：

$$
\begin{aligned}
E_m(y) &= t\sum_f A_f^0\,\psi(F_f),
&F_f &= D_{s,f}D_{m,f}^{-1}, \\
\psi(F) &= \frac{\mu}{2}(I_C-2)
 + \frac{\lambda_{\mathrm{NH}}}{2}(J-\alpha)^2
 - \frac{\lambda_{\mathrm{NH}}}{2}(1-\alpha)^2, \\
I_C &= \operatorname{tr}(F^{\mathsf T}F)=\lVert F\rVert_F^2,
&J &= \sqrt{\det(F^{\mathsf T}F)}, \\
\lambda_{\mathrm{NH}} &= \lambda+\mu,
&\alpha &= 1+\frac{\mu}{\lambda_{\mathrm{NH}}}.
\end{aligned}
$$

这里的 $A_f^0$、$D_{m,f}^{-1}$ 都由 OBJ 静止构型确定，$t$ 是布料厚度。最后一个常数项只把静止构型的膜能数值平移到零，不改变力、Hessian、优化方向或解。弯曲能为

$$
\begin{aligned}
E_b(y) &= \sum_h \frac{1}{2}k_b\,\ell_h^0
\left[\operatorname{wrap}(\theta_h-\theta_h^0)\right]^2, \\
\operatorname{wrap}(\phi) &= \operatorname{atan2}(\sin\phi,\cos\phi).
\end{aligned}
$$

每条内边的静止长度 $\ell_h^0$ 和静止二面角 $\theta_h^0$ 都直接由 OBJ 计算，因此弯曲项在原始曲面也是零应力。材料范围来自 [HOOD post-CVPR 配置](https://github.com/Dolorousrtur/HOOD/blob/master/configs/postcvpr.yaml)，Neo-Hookean 与二面角实现参照 [NVIDIA Newton VBD cloth kernels](https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/vbd/particle_vbd_kernels.py)。

## 1. 构建固定模型和固定评估初值

在本目录运行：

```bash
python cloth01_build_fixed_model_and_datasets.py
```

脚本会生成 `fixed_data/model_spec.json`、拓扑 cache、32/64/4 个冻结初值及其图像。训练 motion 不会被生成或保存。写入 NPZ 使用原子替换，并在复用已有文件前强制解压检查，避免截断 archive 被误认为有效。

只重画已有初值、不重新采样：

```bash
python cloth10_plot_initial_states.py --splits validation test typical
```

典型 motion 0 是衣服面水平、初速度为零、只受重力后摆动；另外三个分别是原始竖直释放、高频速度场、随机 pose + 混合速度。

## 2. 先测完整训练步的显存

```bash
python cloth06_probe_memory_and_throughput.py \
  --output-dir cloth_tshirt_pipeline/profiling/memory_probe \
  --fixed-data-dir fixed_data \
  --device cuda:0 \
  --dtype float64 \
  --seed 42 \
  --model-type mlp \
  --activation relu \
  --depth 1 \
  --width 2048 \
  --no-use-bias \
  --pool-size 512 \
  --batch-sizes 4 8 16 32 64 128 \
  --warmup-updates 1 \
  --measured-updates 3 \
  --memory-headroom-fraction 0.85
```

显存测试会显式接收以下网络和训练规模参数：

| 参数 | 默认值 | 含义与约束 |
|---|---:|---|
| `--model-type` | `mlp` | 待测试的网络类型，可选 `mlp/gnn`；GNN 基线固定使用 ReLU、depth 2、无 bias 和 5 轮消息传递 |
| `--activation` | `relu` | 隐藏层激活，可选 `identity/relu/gelu/silu/tanh` |
| `--depth` | `1` | 隐藏线性层数量，必须为正整数 |
| `--width` | `2048` | 每个隐藏层的宽度，必须为正整数 |
| `--use-bias` / `--no-use-bias` | `--no-use-bias` | 是否为全部线性层启用 bias |
| `--pool-size` | `512` | 常驻 GPU 的在线训练环境数；必须被 4 个 K buckets 整除 |
| `--batch-sizes` | `4 8 16 32 64 128` | 待扫描的训练 batch；每项必须被 4 整除且不大于 pool size |

每个候选 batch size 都在独立子进程中，用完全相同的网络和 pool 配置执行完整的 `ask → residual/model → energy → backward → Adam → tell/reset checks`。`--warmup-updates` 不计时，随后用 `--measured-updates` 统计吞吐和峰值；只有 `peak_reserved / total_memory <= memory_headroom_fraction` 的结果才参与推荐。

完整 CLI 配置会打印到终端，并写入 `memory_probe.json` 和 `recommended_training_config.json` 的 `configuration` 字段；`memory_probe.json`/`memory_probe.csv` 的每个结果也会重复记录网络规格、pool size 和该次实际 batch size。推荐文件按显存阈值内 `motions_per_second` 最高的结果给出 `recommended_batch_size`，并一并保存网络类型、宽度、深度、激活、bias、消息传递轮数和 pool size，便于确认显存测试与正式训练完全一致。

### 双卡 tensor parallel：确定 `1 × 39936` MLP

T-shirt 的完整状态维数为 `4424 × 3 = 13272`，MLP 拼接当前残差、上一残差和上一更新后的输入维数为 `39816`。`39936 / 39816 = 1.003`，既略宽于输入，又能被两个 tensor-parallel rank 整除。在 `cloth_opter` 环境中复现实测配置：

```bash
python cloth20_probe_dual_gpu_tensor_parallel.py \
  --output-dir cloth_tshirt_pipeline/profiling/tp_width_39936_pool512_batch32 \
  --fixed-data-dir fixed_data \
  --devices 0 1 \
  --dtype float32 \
  --seed 42 \
  --activation relu \
  --width 39936 \
  --no-use-bias \
  --pool-size 512 \
  --batch-size 32 \
  --warmup-updates 1 \
  --measured-updates 10 \
  --memory-headroom-fraction 0.95
```

脚本由当前 Python 自动启动两个 `torchrun` rank，并使用 PyTorch DTensor 的 `ColwiseParallel` 切分隐藏层、`RowwiseParallel` 切分输出层。模型先在 `meta` device 上建立结构，再只物化每张卡自己的权重 shard，不会先在任一卡创建完整网络。两个 rank 的在线 pool batch 由 rank 0 广播，确保 tensor-parallel 两侧看到完全相同的输入。

`1 × 39936` 共 `2,120,122,368` 个参数，每张卡持有 `1,060,061,184` 个参数；float32 下每卡权重、梯度和两个 Adam 状态的静态估算为 `15.80 GiB`。探测仍执行完整的 `ask → physics residual/model → energy → backward → 全局梯度裁剪 → Adam → tell/reset`，分别记录两张卡的峰值到 `worker_rank_00.json`、`worker_rank_01.json`，汇总写入 `memory_probe.json`。

在本仓库记录的 `2 × RTX 3090 24 GB`、PyTorch `2.13.0+cu130` 环境中，上述配置实测最大峰值 reserved 为 `22.08 GiB`，约占单卡显存 `93.73%`，吞吐为 `42.20 motions/s`。相同 pool/batch 下的 width `40960` 达到 `22.64 GiB`、约 `96.1%`，超过 `95%` 阈值，因此正式配置固定为 `39936`，不要把探测通过解释成还能继续增加 batch 或 width。

为避免完整大矩阵正交 QR 在切分前制造不可接受的临时显存，探测和正式训练共用 `cloth_tensor_parallel.py`：按全局 fan-in 缩放分别初始化本地隐藏层 shard，输出层仍为零初始化。正式训练前建议让同一配置连续跑至少 500 updates，检查 reserved 是否稳定、两 rank 指标是否一致、无 OOM/NaN、pool reset 原因是否合理；这一步通过后不要再改网络、pool、batch、dtype 或 PyTorch 版本。

## 3. 在线训练

```bash
python cloth05_train_online.py \
  --device cuda:0 \
  --dtype float64 \
  --activation relu \
  --depth 1 \
  --width 2048 \
  --no-use-bias \
  --pool-size 512 \
  --batch-size 32 \
  --max-wall-hours 10
```

显存扫描使用复数参数 `--batch-sizes` 测多个候选值；正式训练使用单数参数 `--batch-size`。应把上例的 `32` 替换为 `recommended_training_config.json` 中的 `recommended_batch_size`，并保持 activation、depth、width、bias、dtype 和 pool size 与显存扫描一致。

训练正常结束或收到 `Ctrl-C` 时都会保存 `latest_checkpoint.pt`；后者可用 `--resume` 恢复，包括在线采样 RNG 和 pool。训练结束会自动调用 `cloth11_plot_training_progress.py`，统一绘制 loss、残差比、梯度裁剪、pool reset 和两种验证曲线。

快速验证使用全部 32 个冻结 validation motions、32 帧、每帧固定 15 步；checkpoint validation 使用同样 32 个初值、100 帧、每帧固定 50 步。只有后者按“失败数优先”的字典序选择 `best_validation_model.pt`。达到残差比 `1e-3` 只记入 `converged_frame_count/fraction`，不会结束该帧的迭代。

### 双卡 `1 × 39936` 正式训练

显存候选、500-update soak、checkpoint/resume smoke、完整学习率 sweep、float32 精度分析和最终参数选择记录见 [`document/TENSOR_PARALLEL_MEMORY_AND_LR_SELECTION.md`](document/TENSOR_PARALLEL_MEMORY_AND_LR_SELECTION.md)。

`cloth21_train_tensor_parallel_online.py` 是与上面 probe 同实现的正式入口。它由当前 Python 自动启动两个 `torchrun` worker，因此应先激活实测使用的 `cloth_opter` 环境，然后从本目录运行：

```bash
python cloth21_train_tensor_parallel_online.py \
  --fixed-data-dir fixed_data \
  --devices 0 1 \
  --dtype float32 \
  --seed 42 \
  --activation relu \
  --width 39936 \
  --no-use-bias \
  --pool-size 512 \
  --batch-size 32 \
  --max-wall-hours 10
```

默认输出目录为 `cloth_tshirt_pipeline/tensor_parallel/activation_relu_depth_01_width_39936_no_bias/seed_42`。两个 rank 都执行同一份 physics、pool、loss 和冻结 validation，rank 0 广播每个训练 batch；大权重、梯度和 Adam 状态由 DTensor 分片。每个日志区间会交叉检查两 rank 的 loss、残差比和 pool 计数，validation 也会检查两侧数值一致，只有 rank 0 写 CSV、图和 validation 结果。

快速 validation 默认每 10000 updates 运行；完整 checkpoint validation 默认每 50000 updates 运行并按稳定性优先的字典序选 best。`Ctrl-C`/`SIGTERM` 会先在两个 rank 间同步停止请求，完成当前安全边界后写可恢复 checkpoint。若作业调度器会直接发送 `SIGKILL`，无法执行收尾保存，只能回到上一个周期 checkpoint。

#### 分片 checkpoint 与恢复

正式入口使用 PyTorch Distributed Checkpoint（DCP），不会在任一卡或 CPU 上聚合完整的 21 亿参数状态：

- `latest.json` 原子指向最新的完整训练 checkpoint；
- `checkpoints/step_XXXXXXXXX_gen_XXXXXX/distributed/` 保存各 rank 的模型和 Adam shard；
- 同目录的 `runtime_rank_00.pt`、`runtime_rank_01.pt` 保存各 rank 的在线 pool、采样 RNG、Torch RNG、update、累计用时和 best 选择状态；
- 只有写完 `manifest.json` 和 `COMPLETE` 后，临时目录才会原子发布并更新 `latest.json`；不完整目录不会被恢复；
- 默认只保留最近 2 个完整 checkpoint；`best.json` 指向完整 validation 选出的 model-only DCP，并只保留 1 份 best。

一个完整 checkpoint 约含 `7.90 GiB` 权重和 `15.80 GiB` Adam 状态，另有两个 pool sidecar；model-only best 约 `7.90 GiB`。默认保留策略在轮换瞬间需要同时容纳旧的 2 份和正在写的新 checkpoint，建议至少预留约 `80 GiB` 可用磁盘。可用 `--keep-checkpoints 1` 降低占用，但会减少回退余量。

从 `latest.json` 恢复时必须使用同一 run dir 和相同的 mesh、dtype、网络、pool/K buckets、两 rank 拓扑以及完全相同的 PyTorch 版本：

```bash
python cloth21_train_tensor_parallel_online.py \
  --run-dir cloth_tshirt_pipeline/tensor_parallel/activation_relu_depth_01_width_39936_no_bias/seed_42 \
  --fixed-data-dir fixed_data --devices 0 1 \
  --dtype float32 --activation relu --width 39936 --no-use-bias \
  --pool-size 512 --batch-size 32 \
  --max-wall-hours 20 --resume
```

`--max-wall-hours` 是包含已保存累计用时的总预算；例如第一段预算 10 小时已经用完，续跑时要把它提高到 20 小时。也可用 `--resume-checkpoint /absolute/path/to/checkpoints/step_...` 明确回退。DCP 目录不是旧单卡脚本所用的 `.pt` 文件，现有 `cloth07_evaluate_checkpoint.py` 和 `cloth13_inference.py` 不能直接读取；训练期间的双卡 validation 已由 `cloth21` 完成，后续独立评估需要继续通过两 rank DTensor loader 运行。

500-update soak 通过后、正式长任务前，再用目标 GPU 做一次最短的 checkpoint/resume 验收（关闭耗时 validation）：

```bash
python cloth21_train_tensor_parallel_online.py \
  --run-dir cloth_tshirt_pipeline/profiling/tp_checkpoint_smoke \
  --fixed-data-dir fixed_data --devices 0 1 \
  --max-updates 2 --checkpoint-interval 2 --keep-checkpoints 1 \
  --skip-initial-validation --skip-final-validation --no-save-best-model

python cloth21_train_tensor_parallel_online.py \
  --run-dir cloth_tshirt_pipeline/profiling/tp_checkpoint_smoke \
  --fixed-data-dir fixed_data --devices 0 1 \
  --max-updates 3 --checkpoint-interval 2 --keep-checkpoints 1 \
  --skip-initial-validation --skip-final-validation --no-save-best-model --resume
```

第二条命令应明确打印从 update 2 恢复，并继续写到 update 3；这同时验证当前两张 GPU、NCCL、DTensor、Adam shard 和目标磁盘文件系统，而不只是验证 Python 接口。恢复时若日志中存在晚于 checkpoint 的未保存 update，原日志会先备份到 `resume_audit/`，再截断到一致的 update，避免重复曲线。

## 4. 完整验证和测试

```bash
python cloth07_evaluate_checkpoint.py \
  --checkpoint cloth_tshirt_pipeline/activation_relu_depth_01_width_2048_no_bias/seed_42/best_validation_model.pt \
  --rollout-frames 500 --inner-steps 50
```

最终 validation/test 的每个有效物理帧都执行完整 50 次更新，即使残差已经下降三个数量级也不提前退出。非有限更新或几何失败仍立即标记为失败，避免无意义地继续传播坏状态。

除残差、失败率、几何比和能量外，统一报告：

`single_step_le_two_orders_frame_count = count(residual_after_one_step / residual_initial >= 1e-2)`。

也就是 learned optimizer 单步下降没有超过两个数量级的帧数；数值越小越好。

## 5. single-motion：baseline 与 network 分开运行和复用

先运行一次三种 baseline：

```bash
python cloth09_rollout_single_motion.py \
  --mode baseline --split typical --motion-index 0 \
  --rollout-frames 500 --inner-steps 50 --line-search-max-trials 12
```

再单独测试任意网络：

```bash
python cloth09_rollout_single_motion.py \
  --mode network --split typical --motion-index 0 \
  --checkpoint /path/to/best_validation_model.pt
```

两种模式各自写 `manifest.json`、`metrics.json`、逐帧 `curves.npz` 和稀疏 `trajectory.npz`。配置 hash 完全匹配时自动跳过；baseline cache 不属于任何网络目录，因此所有模型可复用。

single-motion 评估同样固定使用完整 50 次更新。`residual_ratio_tolerance` 仅用于收敛帧统计，不用于提前通过。

扫描 root 下全部已训练模型、只启动缺失测试，并在全部完成后画四张比较图：

```bash
python cloth12_scan_single_motion_rollouts.py \
  --root cloth_tshirt_pipeline \
  --checkpoint-kind best \
  --split typical --motion-index 0
```

四张图分别比较逐帧最终残差比、单步残差比、累计 objective/residual evaluations 和终端指标。

## 6. 交互 inference 与 Polyscope MP4

安装渲染依赖后运行：

```bash
pip install -r requirements.txt
python cloth13_inference.py --root cloth_tshirt_pipeline
```

脚本会扫描 root 下所有 `best_validation_model.pt` 和 `latest_checkpoint.pt`，在终端中列出三种 GD 与已训练网络供选择。直接回车默认选择 **3×3 Hessian block 预条件线搜索 GD**；非交互运行也使用这个默认项。也可跳过菜单：

```bash
# 指定网络 checkpoint
python cloth13_inference.py \
  --solver network --checkpoint /path/to/best_validation_model.pt

# 用户设定初始 dynamics；高频速度 RMS 是精确请求值而不是采样上界
python cloth13_inference.py \
  --pose random --seed 7 \
  --translation-velocity 0.5 0 0 \
  --angular-velocity 0 2 0 \
  --smooth-velocity-rms 1.0 \
  --high-frequency-velocity-rms 2.5 \
  --position-perturb-rms-edge-fraction 0.08
```

默认 dynamics 是衣服面水平、零初速度，在四个肩部固定点和重力作用下荡落；默认每物理帧最多 50 次迭代，残差降至该帧初值的 `1e-3` 时提前收敛，rollout 500 帧。若要让 inference 也强制跑满 50 次，添加 `--fixed-inner-steps`。位置扰动仍经过三角形面积、奇异值和条件数筛查。

每次运行保存：

- 精确复现实次 dynamics 的 `initial_state.npz`；
- `curves.npz`、`residuals.csv` 和 `metrics.json`；
- 完整 `trajectory.npz` 与 `inference_manifest.json`；
- `residual_vs_frames.png`；
- Polyscope 渲染、H.264 编码的 `motion.mp4`。

无 `DISPLAY` 的 Linux 机器会自动使用 Polyscope EGL headless backend；可用 `--egl-device-index` 选择 GPU。交互桌面可用 `--no-headless`。参考 [Polyscope headless rendering](https://polyscope.run/features/headless_rendering/) 和 [screenshot API](https://polyscope.run/py/features/screenshots/)。

## 7. 一键流程

```bash
python cloth08_run_end_to_end.py --device cuda:0 --dtype float64
```

顺序为：固定数据检查/构建 → 显存探测 → 使用推荐 batch 训练 → validation/test 完整评估 → 水平重力 single-motion baseline/network 扫描。

## 8. 测试

```bash
python -m unittest -v \
  test_tshirt_sampling.py test_tshirt_physics.py \
  test_training_pool.py test_pipeline_contracts.py
```

采样/网格测试只依赖 NumPy；物理和训练测试需要 PyTorch。项目默认 `float64`，建议先在目标 CUDA/PyTorch 环境运行测试和显存探测，再开始长训练。

## 9. Newton VBD reference

在已经安装 Newton/Warp 的 `cloth_opter` 环境运行 frozen typical motion 0：

```bash
conda run --no-capture-output -n cloth_opter python cloth14_vbd_reference.py
```

默认使用 `dt=0.01`、每步 10 次 VBD 迭代并运行 6000 个物理步；有 CUDA 时自动选择
`cuda:0`，否则回退 CPU。VBD 自碰撞默认开启。稀疏轨迹、诊断、精确配置和与当前
无碰撞 `float64` 模型的差异说明写入 `vbd_reference/`。

渲染已有 VBD 轨迹：

```bash
conda run --no-capture-output -n cloth_opter python cloth15_render_vbd_reference.py
```

该脚本通过 Polyscope EGL 渲染并使用 H.264 编码 `vbd_reference/motion.mp4`，同时绘制
已保存的速度与几何质量诊断。只有输入中确实存在 residual 字段时才会绘制 residual 曲线。
