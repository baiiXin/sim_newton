# Newton VBD reference: typical 0

本目录由 `cloth14_vbd_reference.py` 生成。参考轨迹使用冻结数据中的
`typical_00_horizontal_gravity_release`：衣服面水平、初速度为零、4 个肩部顶点固定，
在重力下释放。

## 实际运行配置

- 环境：conda `cloth_opter`
- Newton `1.4.0`，Warp `1.15.0`
- 设备：`cuda:0`
- 物理步数：6000，`dt=0.01` 秒，总仿真时间 60 秒
- 每步 VBD 迭代：10
- 自碰撞：True，半径 0.00047 m，margin 0.000705 m
- 稀疏轨迹间隔：每 10 步保存一次，共 601 帧

## 参数映射

- `density = areal_density = 0.551579541482` kg/m²。
- Newton 1.4.0 VBD 使用与项目 README 相同形式的 stable Neo-Hookean 膜能。
  Newton 的面积积分参数采用二维系数，因此设置
  `tri_ke = thickness * lame_mu = 13.7395304023` N/m、
  `tri_ka = thickness * lame_lambda = 37.8989509017` N/m。
- `edge_ke = bending_stiffness = 6.4999970376e-05`；VBD 的二面角项同样乘静止边长。
- 项目基线没有材料阻尼，故 `tri_kd=edge_kd=0`。

## 与当前项目模型不完全一致的内容

1. Newton/Warp 内部使用 `float32`；当前学习优化器基线默认 `float64`。
2. 本结果保留 Newton VBD 的 Hessian 投影、顶点着色/Gauss-Seidel 更新和默认碰撞检测节奏，
   因而不是项目里三种 GD 基线的逐迭代复刻。
3. 项目基线用 `wrap(theta-theta_rest)` 计算弹性二面角差；Newton 1.4.0 的 VBD 弹性弯曲核
   直接使用 `theta-theta_rest`（只有阻尼的逐步角度差会 wrap）。跨过 `-pi/pi` 分支时两者会不同。
4. VBD 自碰撞已开启；当前项目能量基线明确没有碰撞项。自碰撞半径取布厚，margin 为其
   1.5 倍。没有添加 README 未定义的人体、地面或其他刚体碰撞体。
5. 四个固定点在 Newton 中通过零质量和清除 `ParticleFlags.ACTIVE` 实现；其目标位置恒定。
6. `trajectory.npz` 是每 10 步采样一次的稀疏轨迹，不含全部 6001 个状态。

## 结果健康检查

- 全部保存状态有限：True。
- 终态最大固定点误差：5.1990417e-09 m。
- 终态速度：mean=0.0003245496 m/s，max=0.011439363 m/s。
- 终态三角形面积比范围：[0.55401356, 1.6927546]。
- 终态边长比范围：[0.25468818, 6.3122903]。

状态没有 NaN/Inf，但边长比极值表明局部变形明显；加之上面的碰撞和弯曲角分支差异，
该结果适合作为 Newton VBD 行为参考，不应视为当前无碰撞 `float64` 优化目标的高精度真值。

## 文件

- `trajectory.npz`：`steps`、`times`、`positions`、`velocities`、`faces`、`fixed_indices`。
- `diagnostics.csv`：每个保存帧的速度、位移、面积/边长比和固定点误差。
- `metrics.json`：完成状态、运行耗时和终态摘要。
- `manifest.json`：输入哈希、软件版本及完整参数映射。
- `resume_checkpoint.npz`：断点续跑状态；从断点恢复会重建 VBD 碰撞检测器的瞬态内部状态。
- `motion.mp4`：Polyscope EGL 渲染的 1280×720、30 fps、H.264 视频；601 帧对应
  0–6000 物理步，约 20.03 秒播放时间。
- `motion_final.png`：视频终态帧。
- `diagnostics_vs_time_step.png`：已有速度、面积比和边长比随物理步的变化。
- `render_manifest.json`：渲染、编码和 residual 可用性记录。

## 渲染与 residual

渲染命令：

```bash
conda run --no-capture-output -n cloth_opter python cloth15_render_vbd_reference.py
```

本次 VBD 仿真没有保存每次 VBD 迭代的梯度范数或求解 residual；`trajectory.npz` 和
`diagnostics.csv` 中也没有 residual 字段。因此没有生成 `residual_vs_time_step.png`，并且没有
把速度、面积比或边长比重新命名为 residual。可用量已单独绘制在
`diagnostics_vs_time_step.png`。若以后仿真输出加入名称包含 `residual` 的诊断列，渲染脚本会
额外生成 residual 曲线。

运行命令：

```bash
conda run --no-capture-output -n cloth_opter python cloth14_vbd_reference.py
```
