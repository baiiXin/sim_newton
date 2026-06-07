# `free_fall_mlp_optimizer_video.py` 详细说明

## 一、脚本概述

本脚本在 `baseline_normalized_augmented.py` 的基础上，**追加了一个多帧序列 rollout 模块和 3D 视频渲染模块**，用于把"训练好的 MLP optimizer"和"解析牛顿法"做时间积分上的可视化对比。

它分为两段任务：

1. **单步隐式欧拉变分能量最小化训练**（与基线脚本一致）
   - 在固定 `(p_n, v_n, m, g, dt)` 下，训练一个 MLP 作为隐式欧拉变分能量 `E(y)` 的迭代优化器。
   - 训练集：`y0 → y*` 的线性插值锚点 + `y*` 周围的局部高斯扰动锚点。
   - 输入做 dataset-level 标准化，输出乘以 `dt`（让网络学 O(1) 的 `raw_delta`）。
   - 不引入 residual 作为输入信息。

2. **多帧自由落体 rollout + 视频对比**（脚本新增的部分）
   - 用训练好的 MLP 与解析牛顿法**各自滚动 300 个时间步**，每步以前一帧的 `(p_n, v_n)` 作为隐式欧拉的当前状态，重新做单步求解。
   - 把两条 3D 轨迹并排画在同一个视频里（左 Newton，右 MLP），输出 `free_fall_newton_vs_mlp.mp4`。

> 关键边界：训练阶段使用的是**固定的** `p_n = [3, 4, 5]`、`v_n = [0.5, -0.5, 0]`。Rollout 阶段每一帧的 `(p_n, v_n)` 都会变化。

---

## 二、物理与数学背景

### 2.1 隐式欧拉单步格式

对自由落体方程 $m\ddot{x} = -mg\hat{z}$，隐式欧拉的下一位置 $y = x_{n+1}$ 满足

$$
y - p_n - \Delta t\,v_n + \Delta t^2 g\,\hat{z} = 0
$$

等价于最小化

$$
E(y) = \frac{m}{2\Delta t^2}\|y - p_n - \Delta t v_n\|^2 + m g\,y_z
$$

### 2.2 解析最优解

$$
y^* = p_n + \Delta t v_n - \Delta t^2 g\,\hat{z}
$$

### 2.3 牛顿方向

由于 Hessian 各向同性 $H = (m/\Delta t^2)I$，一步牛顿即可命中 $y^*$（浮点精度内）。

### 2.4 多步 rollout 中的速度更新

每个时间步求出 $p_{n+1}$ 后，用差分恢复速度：

$$
v_{n+1} = \frac{p_{n+1} - p_n}{\Delta t}
$$

并把 $(p_{n+1}, v_{n+1})$ 作为下一帧的 $(p_n, v_n)$。

---

## 三、代码结构

### 3.1 模块清单

| 函数 / 类                              | 职责                                                                     |
| ------------------------------------- | ----------------------------------------------------------------------- |
| `MLPOptimizer`                        | 网络主体；内部做输入标准化，并将网络输出乘以 `dt`。                          |
| `variational_energy`                  | 计算 $E(y)$，训练 loss 与评估目标。                                       |
| `variational_residual` / `residual_norm` | 返回 $\nabla E(y)$ 与其 L2 范数。                                       |
| `newton_direction`                    | 牛顿方向 $-H^{-1}\nabla E$。                                              |
| `make_training_states`                | 构造扩充训练集（line + local）。                                          |
| `compute_input_normalizer`            | 计算输入向量的均值/标准差。                                                |
| **`solve_one_step_newton`** *(新增)*  | 用牛顿法求解**一个**时间步：从 $y_0 = p_n$ 起迭代，residual 下降 $10^{-3}$ 或 5 次封顶。 |
| **`solve_one_step_mlp`** *(新增)*     | 用 MLP 求解一个时间步，停止准则与上同。                                     |
| **`rollout_free_fall_sequence`** *(新增)* | 并行滚动 Newton 和 MLP，输出 300 帧位置/速度序列与每步统计。               |
| **`render_free_fall_comparison_video`** *(新增)* | 把两条轨迹并排画在两个 3D 子图里，存成 `.mp4`（无 ffmpeg 时回退到 `.gif`）。 |
| `main`                                | 训练 → 单步对比评估 → rollout → 视频渲染。                                  |

### 3.2 `MLPOptimizer` 设计要点

- **网络**：`Linear(12, 32) → ReLU → Linear(32, 32) → ReLU → Linear(32, 3)`
- **输入** 12 维：`[y(3), p_n(3), v_n(3), m, g, dt]`
- **标准化**：在 `forward` 内部用 `(inp - input_mean) / input_std` 做归一化；常量维度（训练时 `p_n, v_n, m, g, dt` 全部恒定）的 std 兜底为 1。
- **输出缩放**：`delta_y = dt * raw_delta`，让网络学 O(1) 的 `raw_delta`，输出仍然是位置更新。

### 3.3 单步求解器停止条件（两端一致）

- `residual ≤ max(r0 * 1e-3, 1e-12)`（下降三个数量级）；
- 或迭代达 `max_iters = 5` 次。

> 注意：Newton 法理论上 1 步即收敛；MLP 是数据驱动近似，多次迭代可视为它对自身预测做迭代修正。

### 3.4 Rollout 主循环

```
for frame in range(num_frames):
    p_newton, v_newton, _ = solve_one_step_newton(p_newton, v_newton, ...)
    p_mlp,    v_mlp,    _ = solve_one_step_mlp   (mlp, p_mlp, v_mlp, params, ...)
```

Newton 路径和 MLP 路径**各自独立演化**（互不影响），方便对比。

### 3.5 视频渲染要点

- 把两条轨迹的所有位置合并后取统一坐标范围，左右两个子图共享同一可视区。
- 每帧叠加文字：`frame`、`t`、`iters`、`res`（求解器收敛信息）。
- 优先用 `FFMpegWriter` 存 `.mp4`，否则回退 `PillowWriter` 存 `.gif`。

---

## 四、产物清单

`main()` 执行后会在当前工作目录写出：

| 文件                                | 内容                                                   |
| ----------------------------------- | ----------------------------------------------------- |
| `optimization_report.json`          | 训练日志 + 周期评估 + 单步迭代最终对比。               |
| `optimization_report.png`           | 4 子图：训练 gap、周期评估 gap、最终收敛 gap、residual。 |
| `free_fall_sequence_report.json`    | 300 帧 Newton / MLP 的位置、速度、每步统计。           |
| `free_fall_newton_vs_mlp.mp4`       | 左 Newton / 右 MLP 的并排 3D 轨迹动画（30 fps）。      |

---

## 五、训练超参（默认）

| 项                       | 取值                                       |
| ------------------------ | ------------------------------------------ |
| 物理常数                 | `m=1, g=9.8, dt=0.01`                       |
| 训练初值 / 状态           | `p_n=[3,4,5], v_n=[0.5,-0.5,0]`              |
| 训练集扩充               | 11 个 line anchors + 32 个 local anchors    |
| `local_std_dt_units`     | 1.0（局部扰动尺度 = `dt`）                 |
| 优化器                   | Adam，`lr=1e-3`                             |
| Epoch 数                 | 1000                                        |
| K（每个 init 的步数）    | 从 1 起，每 100 epoch +1，封顶 10           |
| Rollout 帧数             | 300                                         |
| Rollout `max_iters`      | 5                                           |
| `residual_drop`          | `1e-3`                                      |
| 视频 fps                 | 30                                          |

---

## 六、与基线脚本的关系

本脚本与 `baseline_normalized_augmented.py` 共享完全相同的训练逻辑和单步评估逻辑。**唯一新增的部分**是：

1. `solve_one_step_newton` / `solve_one_step_mlp`：把单步迭代封装成「带停机准则的求解器」。
2. `rollout_free_fall_sequence`：在时间维度上把 MLP 当作真正的"隐式欧拉求解器"使用，做 300 步 rollout。
3. `render_free_fall_comparison_video`：3D 并排视频。

---

## 七、注意事项与已知行为

- 训练阶段的所有 `(p_n, v_n)` 都是**固定的训练值**；rollout 阶段每帧的 `(p_n, v_n)` 都在改变，这与训练分布**不同**。
- 这种"单点训练 + 全时长 rollout"的设计天然存在**分布外（OOD）问题**，是视频中 MLP 轨迹与 Newton 轨迹快速发散的根本原因（见下方"为什么 MLP 质点几乎不动"的分析）。
- 输出乘 `dt` 的设计本身不会让 MLP 不动；不动是因为网络在 OOD 输入下回退到了"已在最优点附近，输出 ≈ 0"的训练记忆。
