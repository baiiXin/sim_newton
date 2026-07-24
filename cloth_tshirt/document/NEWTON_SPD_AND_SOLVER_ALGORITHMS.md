# T-shirt 隐式时间步、Newton、分块 SPD 与 VBD 算法说明

记录日期：2026-07-24
对应实现：

- `cloth02_batched_physics.py`
- `cloth25_rollout_newton_single_motion.py`
- `cloth26_rollout_newton_best_iterate.py`
- `cloth14_vbd_reference.py`
- `cloth03_training_pool.py`

## 1. 文档范围和最重要的结论

本文说明当前 typical 0 single-motion 实验中涉及的五类方法：

1. `raw_best`：完整 Hessian 的矩阵无关 Newton-MINRES，不带阻尼、不带线搜索；
2. `newton_linesearch_best`：同一个 Newton-MINRES 方向，加 Armijo 回溯线搜索；
3. `spd_block_linesearch_best`：每顶点 `3×3` 分块 SPD 近似，加 Armijo 回溯线搜索；
4. NVIDIA Newton VBD reference：外部 VBD 求解器，每顶点局部求解并包含自碰撞；
5. learned MLP optimizer：学习更新量，不显式求解 Hessian。

前三种方法还共用“每个物理帧保存残差最小的有效迭代结果”的 safeguard。

需要特别注意：

> `spd_block_linesearch_best` 没有组装完整全局 Hessian，也没有对完整 Hessian
> 做正定投影。它只组装每个顶点的 `3×3` 对角块，忽略不同顶点之间的 Hessian
> 非对角耦合，然后分别对这些小块做特征值下限投影。

因此，该方法更准确的名称是：

```text
block-diagonal SPD quasi-Newton / simultaneous block-Jacobi step
```

而不是“完整 Hessian 的 projected Newton”。

## 2. 记号与固定模型

| 记号 | 含义 |
|---|---|
| \(N\) | 顶点数，当前为 4424 |
| \(F\) | 三角形数，当前为 8710 |
| \(H\) | 内部 hinge 数，当前为 12994 |
| \(h\) | 时间步长，当前为 \(0.01\) s |
| \(x_n\in\mathbb R^{N\times3}\) | 第 \(n\) 个物理帧的位置 |
| \(v_n\in\mathbb R^{N\times3}\) | 第 \(n\) 个物理帧的速度 |
| \(y\in\mathbb R^{N\times3}\) | 当前隐式时间步内待优化的位置 |
| \(m_i\) | 顶点 \(i\) 的 lumped mass |
| \(g_0=(0,-9.81,0)\) | 重力加速度 |
| \(\bar x_i\) | 固定顶点的目标位置 |
| \(P(\cdot)\) | 将固定顶点覆盖为 \(\bar x_i\) 的位置投影 |
| \(G\) | free-DOF gate；自由坐标为 1，固定坐标为 0 |

当前模型有 4 个固定顶点。所有残差、更新和速度都经过 \(G\)，固定顶点不参与
自由度求解。

## 3. 每个物理时间步求解的变分问题

### 3.1 隐式 Euler 预测位置

代码首先计算

$$
q_n=x_n+h\,v_n+h^2g_0.
$$

重力被吸收到 \(q_n\) 中，因此变分能量里不再单独写重力势能。

### 3.2 增量势能

每个时间步求解

$$
x_{n+1}
=
\operatorname*{arg\,min}_{P(y)=y}
\Phi_n(y),
$$

其中

$$
\Phi_n(y)
=
\frac{1}{2h^2}
\sum_{i\in\mathcal F}m_i\lVert y_i-q_{n,i}\rVert_2^2
+E_{\mathrm{mem}}(y)
+E_{\mathrm{bend}}(y).
$$

\(\mathcal F\) 是自由顶点集合。当前 PyTorch 变分能量没有自碰撞项。

驻点残差和 Hessian 定义为

$$
r(y)=G\nabla\Phi_n(y),\qquad
\mathcal H(y)=G\nabla^2\Phi_n(y)G.
$$

代码报告的 residual 是所有自由坐标上的全局 L2 范数：

$$
R(y)=\lVert r(y)\rVert_2.
$$

### 3.3 时间步结束后的状态更新

选定本帧结果 \(y_\star\) 后：

$$
x_{n+1}=P(y_\star),
\qquad
v_{n+1}=\frac{x_{n+1}-x_n}{h}.
$$

固定顶点速度随后被 \(G\) 清零。

## 4. 膜能与弯曲能

### 4.1 Stable Neo-Hookean 膜能

对三角形 \(t=(i_0,i_1,i_2)\)，定义

$$
D_s=
\begin{bmatrix}
y_{i_1}-y_{i_0} & y_{i_2}-y_{i_0}
\end{bmatrix}
\in\mathbb R^{3\times2}.
$$

静止构型的 intrinsic matrix 逆记为

$$
A=D_m^{-1}\in\mathbb R^{2\times2}.
$$

表面形变梯度为

$$
F=D_sA=
\begin{bmatrix}
f_0&f_1
\end{bmatrix}\in\mathbb R^{3\times2}.
$$

定义

$$
I_C=\lVert f_0\rVert^2+\lVert f_1\rVert^2,
$$

$$
J=
\sqrt{
\lVert f_0\rVert^2\lVert f_1\rVert^2
-(f_0^\mathsf Tf_1)^2
}
=\lVert f_0\times f_1\rVert.
$$

代码使用

$$
\lambda_s=\lambda+\mu,
\qquad
\alpha=1+\frac{\mu}{\lambda_s},
$$

以及 stable Neo-Hookean density

$$
\psi(F)
=
\frac{\mu}{2}(I_C-2)
+\frac{\lambda_s}{2}(J-\alpha)^2
-\frac{\lambda_s}{2}(1-\alpha)^2.
$$

最后一个常数只把静止能量平移到零，不改变梯度和 Hessian。膜能为

$$
E_{\mathrm{mem}}(y)
=
\sum_{t=1}^{F} A_t^0\,t_c\,\psi(F_t),
$$

其中 \(A_t^0\) 是静止三角形面积，\(t_c\) 是布料厚度。

当前参数为：

```text
mu              = 29233.04340904943
lambda          = 80636.0657482915
thickness       = 0.00047
```

### 4.2 二面角弯曲能

对 hinge \(e=(i_0,i_1,i_2,i_3)\)，\(i_2,i_3\) 是共享边的端点，
\(i_0,i_1\) 是两侧三角形的对顶点。定义

$$
e_v=y_{i_3}-y_{i_2},
$$

$$
n_0=(y_{i_2}-y_{i_0})\times(y_{i_3}-y_{i_0}),
$$

$$
n_1=(y_{i_3}-y_{i_1})\times(y_{i_2}-y_{i_1}).
$$

归一化后：

$$
s=(\hat n_0\times\hat n_1)^\mathsf T\hat e_v,
\qquad
c=\hat n_0^\mathsf T\hat n_1,
$$

$$
\theta=\operatorname{atan2}(s,c).
$$

相对静止角的差值用 wrapped angle：

$$
\Delta\theta
=
\operatorname{atan2}
\left(
\sin(\theta-\theta^0),
\cos(\theta-\theta^0)
\right).
$$

弯曲能为

$$
E_{\mathrm{bend}}(y)
=
\sum_{e=1}^{H}
\frac{1}{2}k_b\,\ell_e^0\,(\Delta\theta_e)^2.
$$

当前 \(k_b=6.499997037604635\times10^{-5}\)。

## 5. 完整梯度、完整 Hessian 与“组装”

### 5.1 数学上的完整组装

设 \(S_e\) 把全局顶点向量抽取为 element \(e\) 的局部向量。完整梯度可写成

$$
\nabla\Phi
=
\frac{M}{h^2}(y-q)
+\sum_t S_t^\mathsf T\nabla E_t
+\sum_e S_e^\mathsf T\nabla E_e.
$$

完整 Hessian 为

$$
\nabla^2\Phi
=
\frac{M}{h^2}
+\sum_t S_t^\mathsf T\nabla^2E_tS_t
+\sum_e S_e^\mathsf T\nabla^2E_eS_e.
$$

一个三角形的局部 Hessian 是 \(9\times9\)，一个 hinge 的局部 Hessian 是
\(12\times12\)。完整组装会同时产生：

- 顶点自身的 \(3\times3\) 对角块；
- 同一个 element 内不同顶点之间的 \(3\times3\) 非对角块。

### 5.2 当前 raw Newton 没有显式组装全局矩阵

`raw_best` 和 `newton_linesearch_best` 使用 PyTorch autograd 得到精确梯度，
再用二次自动微分计算 Hessian-vector product：

$$
\operatorname{HVP}(z)
=
\mathcal H(y)z
=
\frac{\partial}{\partial y}
\left(r(y)^\mathsf Tz\right).
$$

因此：

- 使用的是完整膜能和完整弯曲能 Hessian；
- 包含同一 element 内不同顶点之间的耦合；
- 包含弯曲能的二阶角度项；
- 不生成全局 dense/COO Hessian；
- MINRES 只要求一个 `HVP(z)` 函数。

### 5.3 分块 SPD 中“组装”的准确含义

分块方法只保留全局 Hessian 的每顶点对角块：

$$
B=\operatorname{blockdiag}(B_1,\ldots,B_N),
\qquad B_i\in\mathbb R^{3\times3}.
$$

数学形式为

$$
B_i
=
\frac{m_i}{h^2}I_3
+\sum_{t\ni i}K^{\mathrm{mem}}_{t,i}
+\sum_{e\ni i}K^{\mathrm{bend}}_{e,i}.
$$

所谓组装，就是把每个 triangle/hinge 计算出的局部 `3×3` 块，根据局部顶点到
全局顶点的索引做 scatter-add：

```text
local block K(element, local_vertex)
              |
              v
B[global_vertex_index] += K
```

没有组装任何 \(B_{ij},i\ne j\) 的非对角块。

## 6. 膜能 `3×3` 局部块的具体公式

这一节对应 `_membrane_block_hessian()`，也是当前 SPD 组装最关键的部分。

令

$$
A=D_m^{-1}
=
\begin{bmatrix}
A_{00}&A_{01}\\
A_{10}&A_{11}
\end{bmatrix}.
$$

对于三角形的三个局部顶点 \(a=0,1,2\)，顶点位移 \(\delta x_a\) 引起
\(f_0,f_1\) 的变化为

$$
\delta f_0=a_a\,\delta x_a,
\qquad
\delta f_1=b_a\,\delta x_a,
$$

其中

$$
(a_0,b_0)
=
(-(A_{00}+A_{10}),-(A_{01}+A_{11})),
$$

$$
(a_1,b_1)=(A_{00},A_{01}),
\qquad
(a_2,b_2)=(A_{10},A_{11}).
$$

首先计算 \(J\) 对两个形变梯度列的导数：

$$
g_0=\frac{\partial J}{\partial f_0}
=
\frac{
\lVert f_1\rVert^2f_0-(f_0^\mathsf Tf_1)f_1
}{J},
$$

$$
g_1=\frac{\partial J}{\partial f_1}
=
\frac{
\lVert f_0\rVert^2f_1-(f_0^\mathsf Tf_1)f_0
}{J}.
$$

对局部顶点 \(a\)，定义

$$
d_a=a_ag_0+b_ag_1,
$$

$$
c_a=a_af_1-b_af_0.
$$

再定义 stable Neo-Hookean kernel 中的标量

$$
\sigma=\lambda_s(J-\alpha),
$$

$$
\rho=\frac{\max(\sigma,0)}{J},
\qquad
c_1=\lambda_s-\rho,
$$

$$
\eta_a
=
\mu(a_a^2+b_a^2)
+\rho
\left(
a_a^2\lVert f_1\rVert^2
+b_a^2\lVert f_0\rVert^2
-2a_ab_a f_0^\mathsf Tf_1
\right).
$$

三角形对该局部顶点贡献的 `3×3` 块为

$$
K^{\mathrm{mem}}_{t,a}
=
A_t^0t_c
\left[
\eta_aI_3
+c_1d_ad_a^\mathsf T
-\rho c_ac_a^\mathsf T
\right].
$$

然后把它组装到

$$
B_{i_a}\mathrel{+}=K^{\mathrm{mem}}_{t,a}.
$$

代码一次处理所有三角形，并用 `scatter_add_` 完成局部顶点到全局顶点的并行
组装。

## 7. 弯曲 `3×3` 局部块的具体公式

精确弯曲 Hessian 的局部块包含

$$
\frac{\partial^2 E_e}{\partial x_a^2}
=
k_b\ell_e^0
\left[
\frac{\partial\theta}{\partial x_a}
\frac{\partial\theta}{\partial x_a}^\mathsf T
+\Delta\theta
\frac{\partial^2\theta}{\partial x_a^2}
\right].
$$

当前分块组装使用 Gauss-Newton/PSD 部分：

$$
K^{\mathrm{bend}}_{e,a}
=
k_b\ell_e^0\,
d_{\theta,a}d_{\theta,a}^\mathsf T,
\qquad
d_{\theta,a}=\frac{\partial\theta}{\partial x_a}.
$$

它省略：

1. \(\Delta\theta\,\partial^2\theta/\partial x_a^2\)；
2. hinge 中不同顶点之间的
   \(d_{\theta,a}d_{\theta,b}^\mathsf T,\ a\ne b\)。

角度导数按以下恒等式解析计算。对任意归一化向量
\(\hat u=u/\lVert u\rVert\)：

$$
d\hat u
=
\frac{I-\hat u\hat u^\mathsf T}{\lVert u\rVert}\,du.
$$

又因为 \(\theta=\operatorname{atan2}(s,c)\)，且理想情况下
\(s^2+c^2=1\)：

$$
d\theta=c\,ds-s\,dc.
$$

代码依次构造 \(dn_0,dn_1\)，再计算 \(ds,dc\)，得到 hinge 四个顶点各自的
\(d_{\theta,a}\)。最后同样 scatter-add：

$$
B_{i_a}\mathrel{+}=K^{\mathrm{bend}}_{e,a}.
$$

## 8. `3×3` 块的 SPD 谱投影

### 8.1 投影前的块

组装完成后：

$$
\widetilde B_i
=
\frac{m_i}{h^2}I_3
+\sum_{t\ni i}K^{\mathrm{mem}}_{t,i}
+\sum_{e\ni i}K^{\mathrm{bend}}_{e,i}.
$$

先强制对称：

$$
\widetilde B_i
\leftarrow
\frac{1}{2}
(\widetilde B_i+\widetilde B_i^\mathsf T).
$$

### 8.2 特征值下限

对每个顶点单独做

$$
\widetilde B_i
=
Q_i\operatorname{diag}(\lambda_{i1},\lambda_{i2},\lambda_{i3})Q_i^\mathsf T.
$$

定义该块的尺度

$$
s_i=\frac{1}{3}\sum_{j=1}^{3}|\lambda_{ij}|.
$$

当前代码中的谱下限为

$$
\delta_i=10^{-9}+10^{-6}s_i.
$$

投影后的特征值：

$$
\widehat\lambda_{ij}=\max(\lambda_{ij},\delta_i).
$$

最终

$$
B_i
=
Q_i
\operatorname{diag}(\widehat\lambda_{i1},
\widehat\lambda_{i2},
\widehat\lambda_{i3})
Q_i^\mathsf T.
$$

固定顶点直接设置为 \(B_i=I_3\)，其实际更新还会被 \(G\) 清零。

这个投影保证在数值分解成功时

$$
z^\mathsf TB_iz>0,\qquad \forall z\ne0.
$$

### 8.3 组装和投影伪代码

```text
function ASSEMBLE_BLOCK_SPD(y):
    # B has shape [N, 3, 3]
    for vertex i:
        B[i] = (m[i] / h^2) * I3

    # Membrane assembly: every triangle contributes to its 3 vertices.
    for triangle t in parallel:
        F = Ds(y[t]) * inverse_Dm[t]
        compute f0, f1, J, g0, g1, rho, c1

        for local_vertex a in {0, 1, 2}:
            get coefficients (a_a, b_a)
            d = a_a * g0 + b_a * g1
            c = a_a * f1 - b_a * f0
            eta = membrane_identity_coefficient(...)
            K = rest_area[t] * thickness * (
                    eta * I3 + c1 * outer(d, d) - rho * outer(c, c)
                )
            scatter_add(B[triangle[t, a]], K)

    # Bending assembly: every hinge contributes to its 4 vertices.
    for hinge e in parallel:
        compute theta and dtheta_dx[0:4]
        for local_vertex a in {0, 1, 2, 3}:
            K = bending_stiffness * rest_hinge_length[e] \
                * outer(dtheta_dx[a], dtheta_dx[a])
            scatter_add(B[hinge[e, a]], K)

    for vertex i in parallel:
        B[i] = 0.5 * (B[i] + transpose(B[i]))
        eigenvalues, Q = eigh(B[i])
        scale = mean(abs(eigenvalues))
        floor = 1e-9 + 1e-6 * scale
        eigenvalues = maximum(eigenvalues, floor)
        B[i] = Q * diag(eigenvalues) * transpose(Q)

        if vertex i is fixed:
            B[i] = I3

    return B
```

计算量和存储量均为线性规模：

```text
assembly work  = O(N + F + H)
block storage  = O(9N)
eigendecompose = N 个独立的 3×3 eigh
```

## 9. 方法一：raw Newton-MINRES

### 9.1 Newton 方程

在第 \(k\) 次内迭代：

$$
\mathcal H(y_k)s_k=-r(y_k).
$$

\(\mathcal H\) 可能不定，因此使用 MINRES，而不是要求正定的 CG。

当前配置：

```text
MINRES maximum iterations = 500
relative tolerance        = 1e-2
absolute tolerance        = 1e-10
preconditioner            = block3x3
```

线性停止条件为

$$
\lVert -r-\mathcal Hs\rVert
\le
\max
\left(
10^{-10},
10^{-2}\lVert r\rVert
\right).
$$

### 9.2 `block3x3` 在这里仅是预条件器

raw Newton 中使用

$$
M^{-1}z=B^{-1}z
$$

预条件 MINRES，但被求解的方程仍然是

$$
\mathcal Hs=-r.
$$

因此，SPD block 不替代完整 Hessian，只影响 Krylov 收敛速度。在 `eigh` 因局部块
病态而失败时，代码回退到 \(M=I\)。这个回退仍然不改变 Newton 方程。

### 9.3 更新

raw 方法不做阻尼和线搜索：

$$
y_{k+1}=P(y_k+s_k).
$$

如果 MINRES 不收敛、breakdown，或者 full step 非有限，本物理帧停止继续内迭代，
然后使用本帧之前保存的最佳有效迭代。

### 9.4 伪代码

```text
function RAW_NEWTON_STEP(y, q):
    r = free_gradient(Phi(y, q))

    function HVP(z):
        return free_gate(autograd_gradient(dot(r, z), y))

    try:
        B = ASSEMBLE_BLOCK_SPD(y)
        PRECONDITION(z) = solve_3x3_blocks(B, z)
    catch local_eigendecomposition_failure:
        PRECONDITION(z) = z

    solve HVP(s) = -r with preconditioned MINRES

    if MINRES did not reach tolerance or broke down:
        return SOLVER_ISSUE

    return project_fixed(y + s)
```

## 10. 方法二：Newton-MINRES + Armijo 线搜索

Newton 方向与第 9 节完全相同。区别是不用完整步长直接更新，而是先检查它是否为
下降方向：

$$
\sigma_k=r(y_k)^\mathsf Ts_k.
$$

如果 \(\sigma_k\ge0\) 或非有限，线搜索立即报告 `non_descent_direction`。

否则从 \(\alpha=1\) 开始，要求

$$
\Phi(P(y_k+\alpha s_k))
\le
\Phi(y_k)+c_1\alpha\,r(y_k)^\mathsf Ts_k,
$$

当前

$$
c_1=10^{-4}.
$$

若不满足则

$$
\alpha\leftarrow0.5\alpha,
$$

最多尝试 12 次。

```text
function ARMIJO(y, step, gradient, energy):
    slope = dot(gradient, step)
    if slope is nonfinite or slope >= 0:
        return REJECT("non_descent_direction")

    alpha = 1
    repeat at most 12 times:
        candidate = project_fixed(y + alpha * step)
        if candidate is finite and
           Phi(candidate) <= energy + 1e-4 * alpha * slope:
            return ACCEPT(candidate, alpha)
        alpha = 0.5 * alpha

    return REJECT("armijo_rejected")
```

线搜索不修改物理目标 \(\Phi\)，只改变到达驻点的数值路径。

## 11. 方法三：分块 SPD + Armijo 线搜索

### 11.1 搜索方向

该方法不调用完整 Hessian HVP，也不运行 MINRES。它直接使用第 8 节的分块矩阵：

$$
B(y_k)s_k=-r(y_k).
$$

因为 \(B\) 是 block diagonal，每个顶点独立求解：

$$
s_{k,i}=-B_i^{-1}r_i.
$$

所有顶点的 \(s_{k,i}\) 同时计算，因此这是 simultaneous block-Jacobi
quasi-Newton direction。

若 \(B\) 严格 SPD 且 \(r\ne0\)，则

$$
r^\mathsf Ts
=
-r^\mathsf TB^{-1}r
<0,
$$

所以理论上一定是下降方向。之后仍使用第 10 节的 Armijo 线搜索，以处理浮点误差
和非线性。

### 11.2 完整伪代码

```text
function BLOCK_SPD_STEP(y, q):
    gradient = free_gradient(Phi(y, q))
    B = ASSEMBLE_BLOCK_SPD(y)

    for free vertex i in parallel:
        step[i] = -solve(B[i], gradient[i])
    for fixed vertex i:
        step[i] = 0

    return ARMIJO(
        y=y,
        step=step,
        gradient=gradient,
        energy=Phi(y, q),
    )
```

### 11.3 它改变了什么，没有改变什么

它没有改变：

- \(q_n\)；
- 膜能和弯曲能公式；
- 固定约束；
- 最终想求解的驻点条件 \(r(y)=0\)。

它改变了：

- 内迭代方向，不再是完整 Newton 方向；
- 收敛速度；
- 在只运行 50 次内迭代时得到的近似解；
- 非凸问题中可能进入的吸引域。

因此，“SPD 没有修改物理能量”不等于“有限迭代轨迹一定与完整 Newton 相同”。
只有当不同求解器充分收敛到同一个驻点时，它们才会给出相同的离散时间步结果。

谱下限也不是物理黏性阻尼。物理阻尼需要写进力、速度项或能量；这里的谱下限只用于
构造数值搜索矩阵 \(B\)。

## 12. 惯性初值与原来的当前位置初值

每个新物理时间步有两种初值：

### 12.1 原来的当前位置初值

$$
y_0=P(x_n).
$$

如果本帧所有迭代都失败并选择初值，则

$$
x_{n+1}=x_n,
\qquad
v_{n+1}=0.
$$

这会把速度清零。连续失败时容易形成数值“锁死”。

### 12.2 现在的惯性初值

$$
y_0=P(x_n+h\,v_n).
$$

注意它不是 \(q_n\)：这里按实验要求不包含 \(h^2g_0\)。

如果本帧最终仍选择初值，则自由顶点满足

$$
x_{n+1}=x_n+h\,v_n,
\qquad
v_{n+1}=v_n.
$$

因此 solver issue 不再自动把已有速度清零，时间积分仍会沿当前惯性方向前进。

修改初值不修改 \(\Phi_n\)，但非凸问题中它可能改变收敛 basin；有限迭代结果也会变化。

## 13. 每帧最佳有效迭代 safeguard

三个 `cloth26` variant 都不是无条件采用最后一次迭代，而是把初值和每个成功
candidate 都作为候选。

候选必须通过：

- 位置和 residual 有限；
- residual \(\le10^{12}\)；
- 最大绝对坐标 \(\le10^4\)；
- triangle area ratio 在 \([10^{-3},10^3]\)；
- edge ratio 在 \([10^{-3},10^3]\)；
- 固定点误差 \(\le10^{-9}\)。

在所有有效候选中选择 residual 最小者：

$$
y_\star
=
\operatorname*{arg\,min}_{y_k\in\mathcal V}
R(y_k),
$$

其中 \(\mathcal V\) 包含有效初值和有效迭代。

```text
function ONE_PHYSICAL_FRAME(method, x, v):
    q = x + h * v + h^2 * gravity
    y = project_fixed(x + h * v)       # inertia initial guess

    best_y = y
    best_residual = residual_norm(y)
    best_valid = passes_failure_checks(y)

    repeat at most 50 inner iterations:
        candidate = METHOD_STEP(method, y, q)
        if linear solve / line search reports an issue:
            break

        y = candidate
        candidate_residual = residual_norm(y)
        candidate_valid = passes_failure_checks(y)

        if candidate_valid and
           (not best_valid or candidate_residual < best_residual):
            best_y = y
            best_residual = candidate_residual
            best_valid = true

    x_next = project_fixed(best_y)
    v_next = free_gate((x_next - x) / h)
    return x_next, v_next
```

“solver issue 帧”只表示某次内迭代因以下原因提前停止：

- `linear_nonconvergence`；
- `linear_breakdown`；
- `nonfinite_full_step`；
- `non_descent_direction`；
- `armijo_rejected`。

它不等于整帧没有输出。该帧仍然会采用已经记录的最佳有效候选。

## 14. residual ratio、`ratio_p95` 与 convergence

对物理帧 \(n\)：

$$
\operatorname{ratio}_n
=
\frac{R(y_{\star,n})}
{\max(R(y_{0,n}),10^{-30})}.
$$

当使用 inertia 初值时，分母就是

$$
R(P(x_n+h\,v_n)).
$$

500 帧的

$$
\operatorname{ratio\_p95}
=
Q_{0.95}
\left(
\operatorname{ratio}_0,\ldots,\operatorname{ratio}_{499}
\right)
$$

使用 NumPy `method="linear"`。排序后的位置为

$$
(500-1)\times0.95=474.05,
$$

即在 0-based 第 474 和 475 个值之间线性插值，也就是通常所说的第 475 和第
476 个顺序统计量之间。

`ratio_p95=0.08` 表示约 95% 的物理帧都把初始驻点残差降到了 8% 以下。它不说明
轨迹是物理 ground truth，也不能替代下落角度、速度、能量、接触和几何检查。

当前 convergence 判据更严格：

$$
R(y_\star)
\le
\max
\left(
10^{-10},
10^{-3}R(y_0)
\right).
$$

所以 ratio 明显小于 1 只表示残差下降，不一定被统计为 converged。

## 15. 三个 Newton variant 的对照

| 项目 | `raw_best` | `newton_linesearch_best` | `spd_block_linesearch_best` |
|---|---|---|---|
| 梯度 | 完整 autograd | 完整 autograd | 完整 autograd |
| Hessian | 完整、矩阵无关 HVP | 完整、矩阵无关 HVP | 不使用完整 Hessian |
| 线性求解 | MINRES | MINRES | 每顶点 `3×3 solve` |
| block SPD 用途 | MINRES 预条件器 | MINRES 预条件器 | 实际搜索矩阵 |
| 顶点间 Hessian 耦合 | 有 | 有 | 无 |
| 弯曲二阶角度项 | 有 | 有 | 无 |
| Armijo | 无 | 有 | 有 |
| Levenberg/LM 阻尼 | 无 | 无 | 无 |
| 最佳有效迭代 | 有 | 有 | 有 |
| inertia 初值 | 新实验使用 | 新实验使用 | 新实验使用 |

## 16. NVIDIA Newton VBD reference 与本项目 SPD 的区别

`cloth14_vbd_reference.py` 调用外部 Newton/Warp `SolverVBD`，默认：

```text
dt                    = 0.01
iterations per frame  = 10
self-contact          = enabled
```

VBD 的结构性伪代码是：

```text
for each physical frame:
    build inertial prediction
    perform collision detection / contact update

    repeat VBD sweeps:
        for graph color:
            for vertices i of this color in parallel:
                assemble vertex-local gradient g_i
                assemble vertex-local 3x3 Hessian approximation H_ii
                    from inertia
                    + incident triangles
                    + incident bending hinges
                    + active contacts
                solve H_ii * delta_i = -g_i
                update x_i immediately

    update velocity from position difference
```

它与 `spd_block_linesearch_best` 的主要区别：

1. VBD 按 graph color 更新，前一颜色的新位置会被后续颜色看到，接近 block
   Gauss-Seidel；
2. 当前 PyTorch SPD 一次组装所有 \(B_i\) 并同时更新，属于 block Jacobi；
3. VBD reference 有自碰撞，PyTorch 变分能量没有；
4. VBD reference 当前每帧只有 10 sweep；
5. 当前 PyTorch SPD 有全局 Armijo 和最佳 residual safeguard。

因此不能把二者的轨迹差异只归因于“是否 SPD”。自碰撞、Jacobi/Gauss-Seidel
更新顺序和每帧迭代数都不同。

## 17. Learned MLP optimizer 的公式

MLP 不显式构造 Hessian。先计算质量预条件 residual：

$$
\widetilde r_i
=
\frac{h^2}{m_i}r_i.
$$

将当前 residual、上一 residual 和上一更新拼接：

$$
z_k=
\frac{1}{\ell}
\begin{bmatrix}
\widetilde r_k\\
\widetilde r_{k-1}\\
\Delta y_{k-1}
\end{bmatrix},
\qquad
\ell=0.05.
$$

对当前 width 39936、depth 1、ReLU、无 bias 网络：

$$
h_k=\operatorname{ReLU}(W_1z_k),
$$

$$
\Delta y_k=\ell W_2h_k,
$$

$$
y_{k+1}=P(y_k+G\Delta y_k).
$$

```text
function MLP_STEP(y, q, previous_residual, previous_update):
    residual = mass_precondition(free_gradient(Phi(y, q)))
    features = concat(residual, previous_residual, previous_update) / 0.05
    hidden = ReLU(W1 * features)
    delta = 0.05 * W2 * hidden
    delta = free_gate(delta)
    return project_fixed(y + delta), residual, delta
```

MLP 的方向既不是 Newton 方向，也没有由 SPD 保证的下降性质。训练 loss 鼓励一步后的
变分能量下降，但推理时仍需用 residual、几何失败率和长 rollout 验证。

## 18. 关于“数值方法是否改变物理问题”

| 修改 | 是否修改 \(\Phi_n\) | 有限迭代时能否改变轨迹 |
|---|---:|---:|
| 从 \(x_n\) 改为 \(x_n+hv_n\) 初值 | 否 | 能 |
| Armijo 线搜索 | 否 | 能 |
| block SPD 替代完整 Newton 方向 | 否 | 能 |
| block SPD 仅作为 MINRES 预条件器 | 否 | 理想精确线性求解时不能；实际容差下可能间接影响 |
| 在线性系统中加入数值正则 \(\gamma I\) | 否 | 能 |
| 在能量或力中加入速度阻尼 | 是 | 能 |
| 增加自碰撞能/约束 | 是 | 能 |

这里“没有修改物理问题”只表示目标函数公式未变，不代表在固定 50 次内迭代和非凸
条件下会得到相同的时间轨迹。

## 19. 实现位置索引

| 内容 | 实现 |
|---|---|
| \(q_n\)、能量、残差、状态更新 | `cloth02_batched_physics.py::TShirtPhysics` |
| stable NH 膜块 | `TShirtPhysics._membrane_block_hessian` |
| 弯曲角导数和 PSD 块 | `TShirtPhysics._bending_angle_derivatives`、`_bending_block_hessian` |
| scatter-add、eigh 与谱下限 | `TShirtPhysics.block_diagonal_hessian` |
| Hessian-vector product | `cloth25_rollout_newton_single_motion.py::_newton_step` |
| MINRES | `cloth25_rollout_newton_single_motion.py::_minimum_residual` |
| inertia 初值 | `cloth26_rollout_newton_best_iterate.py::_initial_iterate` |
| SPD block step | `cloth26_rollout_newton_best_iterate.py::_block_spd_step` |
| Armijo | `cloth26_rollout_newton_best_iterate.py::_armijo_line_search` |
| 最佳有效迭代与每帧推进 | `cloth26_rollout_newton_best_iterate.py::run` |
| 外部 VBD 配置 | `cloth14_vbd_reference.py::build_vbd` |
| learned MLP update | `cloth03_training_pool.py::LearnedOptimizerMLP` |
