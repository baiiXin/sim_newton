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
| K buckets | `[1, 3, 10, 30]` |
| 快速验证 | 每物理帧固定 15 次 learned updates，不提前停止 |
| checkpoint validation / 最终 validation / test | 每物理帧固定 50 次，不提前停止；`1e-3` 只作为统计阈值 |
| single motion | baseline 与 network 均每物理帧固定 50 次，不提前停止 |
| baseline | 固定步长原始 GD；质量预条件线搜索 GD；3×3 Hessian block 预条件线搜索 GD |
| 线搜索 | 最多 12 次 objective trials，含初始步长的尝试 |

冻结验证/测试初值是有意的：如果测试时也在线重采样，模型之间和训练时刻之间会混入 Monte-Carlo 抽样噪声，回归结果不可复现，少量异常样本还会显著改变失败率。冻结的是独立随机抽取的初值，不是参考解或轨迹；训练仍保持无限在线采样。

## 能量

膜能采用 NVIDIA Newton VBD 使用的 stable Neo-Hookean surface density：

\[
\psi(F)=\frac{\mu}{2}(I_C-2)+\frac{\lambda+\mu}{2}(J-\alpha)^2
-\frac{\lambda+\mu}{2}(1-\alpha)^2,
\qquad
\alpha=1+\frac{\mu}{\lambda+\mu}.
\]

最后一项只是把 OBJ 静止构型的能量数值平移到零。它不是根据当前构型重新拟合材料参数，也不改变力、Hessian、优化方向或解。OBJ 静止构型另外决定每个三角形的 `Dm^{-1}`。弯曲能为

\[
E_b=\sum_h \frac{1}{2} k_b\,\ell_h^0\,\operatorname{wrap}(\theta_h-\theta_h^0)^2,
\]

其中每条内边的静止二面角 `theta0` 直接由 OBJ 计算，因此弯曲项在原始曲面也是零应力。材料范围来自 [HOOD post-CVPR 配置](https://github.com/Dolorousrtur/HOOD/blob/master/configs/postcvpr.yaml)，Neo-Hookean 与二面角实现参照 [NVIDIA Newton VBD cloth kernels](https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/vbd/particle_vbd_kernels.py)。

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
  --device cuda:0 --dtype float64 \
  --batch-sizes 4 8 16 32 64 128
```

每个 batch size 都在独立子进程中执行完整的 `ask → residual/model → energy → backward → Adam → tell/reset checks`。结果包含 peak allocated/reserved memory，并只在 85% 显存余量内推荐吞吐率最高的 batch。

## 3. 在线训练

```bash
python cloth05_train_online.py \
  --device cuda:0 --dtype float64 \
  --pool-size 512 --batch-size 32 \
  --max-wall-hours 10
```

训练正常结束或收到 `Ctrl-C` 时都会保存 `latest_checkpoint.pt`；后者可用 `--resume` 恢复，包括在线采样 RNG 和 pool。训练结束会自动调用 `cloth11_plot_training_progress.py`，统一绘制 loss、残差比、梯度裁剪、pool reset 和两种验证曲线。

快速验证使用全部 32 个冻结 validation motions、32 帧、每帧固定 15 步；checkpoint validation 使用同样 32 个初值、100 帧、每帧固定 50 步。只有后者按“失败数优先”的字典序选择 `best_validation_model.pt`。达到残差比 `1e-3` 只记入 `converged_frame_count/fraction`，不会结束该帧的迭代。

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
