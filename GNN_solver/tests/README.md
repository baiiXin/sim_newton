# ImplicitEulerLoss README

`ImplicitEulerLoss` 是一个用于训练 / 评估 GNN 仿真求解器的隐式欧拉（Implicit Euler）能量型 loss 计算类。

它的设计目标是：给定一个网络预测的下一步位置 `x_pred = x^{t+1}`，计算该位置是否满足隐式欧拉离散方程对应的 incremental potential。该 loss 可以直接作为 GNN 求解器的训练目标，也可以用 `residual()` 评估当前解的收敛性。

当前版本是一个**最小无碰撞版本**，用于先跑通完整 pipeline：

```text
几何预处理
→ inertia / gravity / elastic / bending loss
→ total loss
→ residual = grad(total loss, x)
→ 用作 GNN solver 的训练目标
```

---

## 1. 当前版本功能

当前实现包含以下 loss 项：

| Loss             | 状态     | 说明                                              |
| ---------------- | -------- | ------------------------------------------------- |
| `inertia_loss`   | 已实现   | 隐式欧拉中的惯性势能                              |
| `gravity_loss`   | 已实现   | 重力势能                                          |
| `elastic_loss`   | 已实现   | St. Venant-Kirchhoff 三角形 FEM stretching energy |
| `bending_loss`   | 已实现   | 简单二面角弯曲能量                                |
| `collision_loss` | 暂未实现 | 后续接入 VBD collision loss                       |
| `friction_loss`  | 暂未实现 | 当前版本暂不考虑                                  |

当前版本支持：

- 单个 mesh；
- PyTorch autograd；
- CPU / CUDA；
- `float32` / `float64`；
- pinned vertices；
- residual 评估；
- 与 GNN solver 直接对接。

当前版本暂不支持：

- batch；
- collision / self-collision；
- friction；
- 厚度；
- bending 几何缩放权重；
- per-face / per-edge 材料参数；
- face winding 自动修正。

---

## 2. 当前假设

为了先跑通最小 pipeline，当前实现采用以下假设：

```text
1. mesh 是三角形表面网格。
2. face_index 的 winding 一致。
3. density 是面密度，不是体密度。
4. 不显式建模 thickness。
5. elastic 使用 2D effective Lamé 参数 mu 和 lambda_。
6. bending 使用简单二面角弹簧，不乘几何缩放权重。
7. material 参数是全局常数，不随时间变化。
8. forward 只接收随时间变化的空间运动状态。
9. pinned 顶点位置建议在外部强制设置。
10. residual 中 pinned 顶点的残差会被置零。
```

---

## 3. 数学背景

连续时间下，离散网格系统满足牛顿第二定律：

```math
M \ddot{x}(t) = f(x(t))
```

如果力来自势能 `E(x)`：

```math
f(x) = -\nabla_x E(x)
```

则系统可以写成：

```math
M \ddot{x}(t) + \nabla_x E(x(t)) = 0
```

隐式欧拉离散为：

```math
v^{t+1} = v^t + \Delta t\, M^{-1} f(x^{t+1})
```

```math
x^{t+1} = x^t + \Delta t\, v^{t+1}
```

消去速度后得到：

```math
\frac{1}{\Delta t^2}M\left(x^{t+1} - \hat{x}\right)
+ \nabla_x E(x^{t+1}) = 0
```

其中：

```math
\hat{x} = x^t + \Delta t\, v^t
```

这个方程等价于最小化 incremental potential：

```math
\Pi(x)
=
\frac{1}{2\Delta t^2}(x-\hat{x})^T M (x-\hat{x})
+ E(x)
```

当前类计算的 `total loss` 就是这个 incremental potential 的无碰撞版本：

```math
L_{total}
=
L_{inertia}
+
L_{gravity}
+
L_{elastic}
+
L_{bending}
```

理想隐式欧拉解满足：

```math
\nabla_x L_{total}(x) = 0
```

因此 `residual()` 定义为：

```math
r(x) = \nabla_x L_{total}(x)
```

---

## 4. 类接口总览

```python
loss_obj = ImplicitEulerLoss(
    rest_pos=rest_pos,
    edge_index=edge_index,
    face_index=face_index,
    density=density,
    mu=mu,
    lambda_=lambda_,
    k_bending=k_bending,
    gravity=(0.0, 0.0, -9.81),
    dt=None,
    pinned_idx=pinned_idx,
)
```

### 4.1 初始化参数

| 参数         | 形状 / 类型      | 说明                                            |
| ------------ | ---------------- | ----------------------------------------------- |
| `rest_pos`   | `[N, 3]`         | rest pose / material-space 顶点坐标             |
| `edge_index` | `[E, 2]`         | 无向边索引，当前最小版本中保存但不直接用于 loss |
| `face_index` | `[F, 3]`         | 三角片索引，假设 winding 一致                   |
| `density`    | scalar           | 面密度，用于计算 lumped vertex mass             |
| `mu`         | scalar           | StVK Lamé shear parameter                       |
| `lambda_`    | scalar           | StVK Lamé first parameter                       |
| `k_bending`  | scalar           | 简单二面角 bending stiffness                    |
| `gravity`    | `[3]`            | 重力加速度方向和大小                            |
| `dt`         | scalar or `None` | 可选固定时间步；若为 `None`，forward 时必须传入 |
| `pinned_idx` | `[P]` or `None`  | 固定点索引                                      |
| `eps`        | scalar           | 数值稳定用 epsilon                              |

### 4.2 初始化后保存的几何 / 物理量

| 成员变量      | 形状        | 说明                                       |
| ------------- | ----------- | ------------------------------------------ |
| `rest_pos`    | `[N, 3]`    | rest pose 坐标                             |
| `edge_index`  | `[E, 2]`    | 边索引                                     |
| `face_index`  | `[F, 3]`    | 三角片索引                                 |
| `face_area`   | `[F]`       | 每个 rest triangle 面积                    |
| `vertex_area` | `[N]`       | lumped vertex area                         |
| `mass`        | `[N]`       | `density * vertex_area`                    |
| `Dm_inv`      | `[F, 2, 2]` | 每个三角形 rest material shape matrix 的逆 |
| `hinges`      | `[H, 4]`    | bending hinge，格式 `[i, j, k, l]`         |
| `theta0`      | `[H]`       | rest pose 下每个 hinge 的二面角            |
| `free_mask`   | `[N]`       | 非 pinned 顶点 mask                        |

---

## 5. Forward 接口

```python
losses = loss_obj.forward(
    x=x_pred,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
)
```

### 5.1 输入

| 参数     | 形状             | 说明                                            |
| -------- | ---------------- | ----------------------------------------------- |
| `x`      | `[N, 3]`         | 当前预测 / 待优化位置，即 `x^{t+1}`             |
| `x_prev` | `[N, 3]`         | 上一时间步位置，即 `x^t`                        |
| `v_prev` | `[N, 3]`         | 上一时间步速度，即 `v^t`                        |
| `dt`     | scalar or `None` | 时间步长；如果初始化时设置过 `dt`，这里可以不传 |

### 5.2 输出

```python
{
    "total": loss_total,
    "inertia": loss_inertia,
    "gravity": loss_gravity,
    "elastic": loss_elastic,
    "bending": loss_bending,
}
```

每个值都是 scalar tensor。

---

## 6. Residual 接口

```python
res = loss_obj.residual(
    x=x_pred,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
    normalize_by_mass=False,
    create_graph=False,
)
```

### 6.1 residual 定义

默认 residual 是：

```math
r(x)=\nabla_x L_{total}(x)
```

如果：

```python
normalize_by_mass=True
```

则返回质量归一化 residual：

```math
r_i \leftarrow \frac{r_i}{m_i}
```

### 6.2 输出

```python
{
    "vector": residual_vec,          # [N, 3]
    "norm_per_vertex": norm_i,      # [N]
    "mean": mean_norm,              # scalar, only free vertices
    "max": max_norm,                # scalar, only free vertices
    "l2": l2_norm,                  # scalar, only free vertices
}
```

pinned 顶点的 residual 会被置零。

---

## 7. 每种 loss 的计算细节

### 7.1 Inertia loss

惯性参考位置：

```math
\hat{x}_i = x_i^t + \Delta t\, v_i^t
```

惯性势能：

```math
L_{inertia}
=
\sum_{i \in \mathcal{F}}
\frac{m_i}{2\Delta t^2}
\left\|x_i - \hat{x}_i\right\|^2
```

其中：

- `m_i` 是 lumped vertex mass；
- `\mathcal{F}` 是自由顶点集合；
- pinned 顶点不参与 inertia loss。

代码接口：

```python
loss_inertia = loss_obj.inertia_loss(
    x=x,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
)
```

物理含义：

- 如果没有任何力，最小化 inertia loss 会得到匀速运动：

```math
x^{t+1}=x^t+\Delta t\,v^t
```

- inertia loss 的梯度是隐式欧拉方程中的惯性残差项：

```math
\nabla_{x_i} L_{inertia}
=
\frac{m_i}{\Delta t^2}(x_i-\hat{x}_i)
```

---

### 7.2 Gravity loss

重力为：

```math
f_{g,i}=m_i g
```

因为力等于势能负梯度：

```math
f_g = -\nabla_x L_{gravity}
```

所以重力势能为：

```math
L_{gravity}
=
-\sum_{i \in \mathcal{F}} m_i g^T x_i
```

代码接口：

```python
loss_gravity = loss_obj.gravity_loss(x)
```

注意符号：

- 如果 `z` 轴向上，`gravity=(0, 0, -9.81)`；
- 则：

```math
L_{gravity}=\sum_i m_i 9.81 z_i
```

高度越高，重力势能越大；最小化它会让顶点向下运动。

只考虑 inertia + gravity 时，解析最优解为：

```math
x^{t+1}=x^t+\Delta t v^t + \Delta t^2 g
```

---

### 7.3 Elastic / stretching loss

当前版本使用三角形 FEM 的 St. Venant-Kirchhoff membrane energy。

对每个三角形：

```math
f=(i,j,k)
```

当前位置下：

```math
D_s=
\begin{bmatrix}
 x_j-x_i & x_k-x_i
\end{bmatrix}
\in \mathbb{R}^{3\times 2}
```

初始化时预计算 rest shape matrix 的逆：

```math
D_m^{-1}\in \mathbb{R}^{2\times 2}
```

变形梯度：

```math
F = D_s D_m^{-1}
\in \mathbb{R}^{3\times 2}
```

Green strain：

```math
G=\frac{1}{2}(F^T F-I)
\in \mathbb{R}^{2\times 2}
```

StVK 能量密度：

```math
\psi(G)
=
\mu \|G\|_F^2
+
\frac{\lambda}{2}\operatorname{tr}(G)^2
```

单个三角形 elastic energy：

```math
L_{elastic,f}=A_f\psi(G_f)
```

总 elastic loss：

```math
L_{elastic}
=
\sum_f
A_f
\left[
\mu \|G_f\|_F^2
+
\frac{\lambda}{2}\operatorname{tr}(G_f)^2
\right]
```

代码接口：

```python
loss_elastic = loss_obj.elastic_loss(x)
```

当前简化：

- 不乘 thickness；
- `mu` 和 `lambda_` 是全局常数；
- 所有 triangle 都参与 elastic loss；
- 当前版本假设 face winding 一致。

Sanity checks：

- rest pose 下 elastic loss 为 0；
- 刚体平移下 elastic loss 为 0；
- 刚体旋转下 elastic loss 应为 0；
- 拉伸后 elastic loss > 0。

---

### 7.4 Bending loss

当前版本使用最简单的二面角弹簧。

一个 bending hinge 格式为：

```math
h=(i,j,k,l)
```

其中：

- `(i,j)` 是共享边；
- `k` 是第一个三角形的对顶点；
- `l` 是第二个三角形的对顶点。

当前构型下共享边：

```math
e=x_j-x_i
```

单位边方向：

```math
\hat{e}=\frac{e}{\|e\|}
```

两侧三角形法向：

```math
n_0 = \operatorname{normalize}\left((x_j-x_i)\times(x_k-x_i)\right)
```

```math
n_1 = \operatorname{normalize}\left((x_l-x_i)\times(x_j-x_i)\right)
```

有符号二面角：

```math
\theta
=
\operatorname{atan2}
\left(
\hat{e}^T(n_0\times n_1),
 n_0^T n_1
\right)
```

初始化时用 `rest_pos` 计算：

```math
\theta_0
```

角度差使用 wrapped difference：

```math
\Delta\theta
=
\operatorname{atan2}
\left(
\sin(\theta-\theta_0),
\cos(\theta-\theta_0)
\right)
```

弯曲能量：

```math
L_{bending}
=
\sum_h
\frac{1}{2}k_b(\Delta\theta_h)^2
```

代码接口：

```python
loss_bending = loss_obj.bending_loss(x)
```

当前简化：

- 不乘边长 / 面积缩放权重；
- `k_bending` 是全局常数；
- 所有 hinge 都参与 bending loss；
- 边界边不产生 hinge；
- 当前版本假设 face winding 一致。

Sanity checks：

- rest pose 下 bending loss 为 0；
- 刚体平移下 bending loss 为 0；
- 刚体旋转下 bending loss 应为 0；
- 折弯后 bending loss > 0。

---

## 8. Pinned vertices 处理

当前策略：

```text
1. pinned 顶点索引在初始化时传入 pinned_idx。
2. 类内部生成 free_mask。
3. inertia / gravity 只对 free vertices 求和。
4. elastic / bending 当前最小版本中不跳过单元。
5. residual 输出时 pinned 顶点 residual 置零。
6. 训练 / 优化时，建议在外部强制 pinned 顶点位置。
```

推荐外部强制 pinned 顶点位置：

```python
def clamp_pinned_vertices(x, reference_x, pinned_idx):
    if pinned_idx is None:
        return x
    x = x.clone()
    x[pinned_idx] = reference_x[pinned_idx]
    return x
```

训练时：

```python
x_pred = solver(...)
x_pred = clamp_pinned_vertices(x_pred, x_prev, pinned_idx)
losses = loss_obj.forward(x_pred, x_prev, v_prev, dt)
```

---

## 9. 与 GNN solver 对接

你的 GNN solver 可以输出以下任意一种量，只要最终转换成 `x_pred: [N, 3]` 即可。

### 9.1 GNN 直接输出下一步位置

```python
x_pred = gnn_solver(...)
losses = loss_obj.forward(x_pred, x_prev, v_prev, dt)
train_loss = losses["total"]
```

### 9.2 GNN 输出位移

```python
dx_pred = gnn_solver(...)
x_pred = x_prev + dx_pred
losses = loss_obj.forward(x_pred, x_prev, v_prev, dt)
```

### 9.3 GNN 输出加速度

```python
a_pred = gnn_solver(...)
x_pred = x_prev + dt * v_prev + dt * dt * a_pred
losses = loss_obj.forward(x_pred, x_prev, v_prev, dt)
```

### 9.4 标准训练接口

```python
optimizer.zero_grad()

x_pred = solver(x_prev, v_prev, edge_index)
x_pred = clamp_pinned_vertices(x_pred, x_prev, pinned_idx)

losses = loss_obj.forward(
    x=x_pred,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
)

train_loss = losses["total"]
train_loss.backward()
optimizer.step()
```

### 9.5 评估 residual

```python
with torch.no_grad():
    x_pred = solver(x_prev, v_prev, edge_index)
    x_pred = clamp_pinned_vertices(x_pred, x_prev, pinned_idx)

res = loss_obj.residual(
    x=x_pred,
    x_prev=x_prev,
    v_prev=v_prev,
    dt=dt,
)

print("residual mean:", res["mean"])
print("residual max:", res["max"])
print("residual l2:", res["l2"])
```

注意：`residual()` 内部需要 autograd，所以即使外部使用 `torch.no_grad()` 得到 `x_pred`，传入 `residual()` 时它也会创建局部可微副本。

---

## 10. 最小使用示例

下面是一个最小 dummy solver 测试。它模拟 GNN 输出 `x_pred`，并用 `ImplicitEulerLoss` 作为训练目标。

```python
import torch
from loss_class import ImplicitEulerLoss


class DummyGNNSolver(torch.nn.Module):
    def __init__(self, num_vertices, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.delta = torch.nn.Parameter(
            torch.zeros(num_vertices, 3, dtype=dtype, device=device)
        )

    def forward(self, x_prev, v_prev, edge_index):
        return x_prev + self.delta


def clamp_pinned_vertices(x, reference_x, pinned_idx):
    if pinned_idx is None:
        return x
    x = x.clone()
    x[pinned_idx] = reference_x[pinned_idx]
    return x


def main():
    dtype = torch.float64
    device = "cpu"

    rest_pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=dtype, device=device)

    face_index = torch.tensor([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=torch.long, device=device)

    edge_index = torch.tensor([
        [0, 1],
        [1, 2],
        [0, 2],
        [1, 3],
        [2, 3],
    ], dtype=torch.long, device=device)

    pinned_idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    loss_obj = ImplicitEulerLoss(
        rest_pos=rest_pos,
        edge_index=edge_index,
        face_index=face_index,
        density=2.0,
        mu=10.0,
        lambda_=10.0,
        k_bending=0.1,
        gravity=(0.0, 0.0, -9.81),
        pinned_idx=pinned_idx,
    )

    solver = DummyGNNSolver(
        num_vertices=rest_pos.shape[0],
        dtype=dtype,
        device=device,
    )

    dt = torch.tensor(0.05, dtype=dtype, device=device)
    x_prev = rest_pos.clone()
    v_prev = torch.zeros_like(rest_pos)

    optimizer = torch.optim.LBFGS(
        solver.parameters(),
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()

        x_pred = solver(x_prev, v_prev, edge_index)
        x_pred = clamp_pinned_vertices(x_pred, x_prev, pinned_idx)

        losses = loss_obj.forward(
            x=x_pred,
            x_prev=x_prev,
            v_prev=v_prev,
            dt=dt,
        )

        losses["total"].backward()
        return losses["total"]

    optimizer.step(closure)

    with torch.no_grad():
        x_pred = solver(x_prev, v_prev, edge_index)
        x_pred = clamp_pinned_vertices(x_pred, x_prev, pinned_idx)

        losses = loss_obj.forward(
            x=x_pred,
            x_prev=x_prev,
            v_prev=v_prev,
            dt=dt,
        )

    res = loss_obj.residual(
        x=x_pred,
        x_prev=x_prev,
        v_prev=v_prev,
        dt=dt,
    )

    print("total loss:", losses["total"])
    print("inertia:", losses["inertia"])
    print("gravity:", losses["gravity"])
    print("elastic:", losses["elastic"])
    print("bending:", losses["bending"])
    print("residual mean:", res["mean"])
    print("residual max:", res["max"])
    print("x_pred:")
    print(x_pred)


if __name__ == "__main__":
    main()
```

---

## 11. 建议测试项

当前版本建议保留以下测试：

```text
1. face_area 是否正确。
2. vertex_area 是否正确。
3. mass 是否正确。
4. Dm_inv 是否正确且 finite。
5. hinge 构造是否正确。
6. theta0 是否正确。
7. pinned/free mask 是否正确。
8. 退化三角形是否报错。
9. rest pose 下 elastic/bending 是否为 0。
10. 刚体平移下 elastic/bending 是否为 0。
11. 拉伸后 elastic 是否为正。
12. 弯折后 bending 是否为正。
13. inertia + gravity 解析最优点梯度是否为 0。
14. forward 输出 key 和 total 是否正确。
15. residual 是否正确计算。
16. pinned residual 是否置零。
17. 用 dummy solver 优化时 total loss 和 residual 是否下降。
```

---

## 12. 常见问题

### Q1：为什么 `forward` 不接收材料参数？

因为当前设计中，几何和物理参数都随模型确定，不随时间变化。因此：

```text
rest_pos / topology / density / mu / lambda_ / k_bending / gravity / pinned_idx
```

都在初始化时传入。`forward` 只接收随时间变化的运动状态：

```text
x, x_prev, v_prev, dt
```

---

### Q2：为什么 `mass` 在初始化时计算？

因为当前版本中 `density` 是固定面密度，质量由 rest geometry 和 density 决定：

```math
m_i = \rho A_i
```

其中：

```math
A_i=\sum_{f\ni i}\frac{A_f}{3}
```

它不随时间变化，所以在初始化阶段计算一次即可。

---

### Q3：为什么 gravity loss 可能是负数？

重力势能是：

```math
L_{gravity}=-\sum_i m_i g^T x_i
```

如果顶点向重力方向运动，重力势能会下降。因此它可以是负数。总 incremental potential 仍然可以正常优化。

---

### Q4：为什么 residual 数值可能很大？

惯性 residual 中有：

```math
\frac{m_i}{\Delta t^2}(x_i-\hat{x}_i)
```

当 `dt` 很小时，`1/dt^2` 很大，所以即使位置误差只有 `0.01` 或 `0.1`，residual 也可能很大。这是正常的。

---

### Q5：为什么 pinned 点还要在外部 clamp？

loss 类会在 inertia / gravity 中忽略 pinned 顶点，并在 residual 中把 pinned 顶点残差置零。但网络输出本身仍可能改变 pinned 顶点位置。为了保证物理边界条件严格满足，建议在外部做：

```python
x_pred[pinned_idx] = x_prev[pinned_idx]
```

---

## 13. 后续扩展计划

建议按以下顺序扩展：

1. 接入 VBD collision loss；
1. 增加 self-collision；
1. bending 加入几何缩放权重：

```math
w_h = \frac{\|e_h^0\|^2}{A_{0,h}+A_{1,h}}
```

4. 引入 thickness；
4. 支持 per-face `mu/lambda_`；
4. 支持 per-edge / per-hinge `k_bending`；
4. 支持 batch；
4. 支持 face winding 自动检查与修正；
4. 加入 friction；
4. 对部分 loss 提供解析 gradient / Hessian，用于更高效的求解或评估。

