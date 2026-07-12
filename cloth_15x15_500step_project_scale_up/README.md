# 15×15 布料大规模训练池项目

本目录在 `cloth_15x15_500step_project` 的基础上，实现一个**不依赖预生成 reference trajectory** 的大规模布料神经迭代求解器训练流程。

项目将场景构造、批量物理、在线训练池、显存测试、双验证、checkpoint 选择和最终测试拆成独立模块。默认使用 `torch.float64` 和 `cuda:0`。

> 当前仓库已经完成代码、单元测试和 CPU 集成测试。RTX 3090 的真实显存、吞吐量、6 小时训练结果和最终测试指标，需要在目标服务器上执行本文末尾的一键命令后生成。

---

## 1. 项目目标

主要研究问题是：

1. 当训练 motion 从几十条扩大到 1024、2048、3072 条后，学习型迭代求解器是否获得更好的组合泛化和长时间稳定性；
2. 网络能否在不同初始形状、预应变、速度、固定点、动态 Dirichlet 边界和材料参数之间共享迭代规律；
3. 在不生成高成本 reference trajectory 的情况下，能否通过在线能量下降训练和连续 rollout 验证筛选稳定模型；
4. 不同每帧迭代次数 `K={1,3,10,30}` 下，精度、稳定性和计算预算如何变化。

项目第一版不包含：

- 接触和碰撞；
- 独立弯曲能；
- 阻尼模型；
- 风场；
- reference trajectory error；
- 时间步长扰动；
- Sobol motion 采样。

因此当前最终评估应称为：

> **reference-free optimizer stability benchmark**，即不依赖参考轨迹的优化器稳定性评估。

它可以判断 residual、能量、约束和几何是否健康，但不能直接证明轨迹位置误差足够小。

---

## 2. 目录结构

```text
cloth_15x15_500step_project_scale_up/
├── scenario_templates.py
├── scenario_geometry.py
├── scenario_builder.py
├── scenario_catalogue.py
├── scenario_audit.py
├── validation_protocol.py
│
├── cloth01_build_scenario_catalogue.py
├── cloth02_batched_physics.py
├── cloth03_training_pool.py
├── cloth04_reference_free_validation.py
├── cloth05_train_scale_up.py
├── cloth06_probe_memory_and_throughput.py
├── cloth07_evaluate_best_checkpoint.py
├── cloth08_run_end_to_end.py
│
├── test_scenario_catalogue.py
├── test_batched_physics.py
├── test_training_pool.py
├── test_final_evaluation.py
└── README.md
```

各脚本职责如下：

| 脚本 | 功能 |
|---|---|
| `cloth01_build_scenario_catalogue.py` | 构造并审计 C1/C2/C3、validation 和 test catalogue |
| `cloth02_batched_physics.py` | 逐 environment 的固定点、材料和动态边界批量物理 |
| `cloth03_training_pool.py` | 675D learned optimizer、在线训练池、loss 和训练 step |
| `cloth04_reference_free_validation.py` | 连续 rollout 验证、失败检测、曲线与结果保存 |
| `cloth05_train_scale_up.py` | 正式训练、双验证、checkpoint 和 resume |
| `cloth06_probe_memory_and_throughput.py` | 独立子进程显存与吞吐量扫描 |
| `cloth07_evaluate_best_checkpoint.py` | best checkpoint 的长 validation 和分组 test |
| `cloth08_run_end_to_end.py` | 显存测试、6 小时训练和最终评估的一键入口 |

---

## 3. Scenario catalogue

### 3.1 场景维度

训练场景由六个主要离散轴和一个朝向轴组成。

#### 初始形状：7 种

- 平面；
- `uv` 正/负鞍形；
- `u²-v²` 正/负鞍形；
- 向上凸包；
- 向下凸包。

#### 初始预应变：7 种

- 无应变；
- x 拉伸/压缩；
- y 拉伸/压缩；
- xy 双向拉伸/压缩。

训练倍率为：

```text
0.85 / 1.00 / 1.15
```

magnitude OOD 使用：

```text
0.70 / 1.30
```

#### 初始速度：19 种

- 静止；
- x/y/z 六个方向的中速和高速平移；
- 绕 x/y/z 三个轴的正反旋转。

#### 固定点：22 种

- 单点 9 种：四角、四边中点、中心；
- 两点 12 种：对角、同边两角、对边中点、中心加边中点；
- 四点 1 种：四角固定。

**所有 train、validation 和 test scenario 至少包含一个固定点。项目中不使用无固定点布料。**

固定点通过归一化坐标或拓扑规则定义，可映射到 5×5、15×15 或其他规则网格。

#### Dirichlet 边界运动：7 种

- 静态；
- 水平圆周正向/反向；
- 竖直圆周正向/反向；
- 正向/反向扭布。

所有动态边界满足：

$$
x_D(0)=x_{\mathrm{initial}},
$$

因此第 0 帧不会发生固定点位置跳变。固定点速度使用 $\dot x_D(t)$，不再统一强制为零。

单固定点与扭布不兼容，catalogue 生成器会自动过滤非法组合。

#### 材料：8 种训练预设

材料 schema 为：

```text
areal_density
stretch_stiffness
shear_stiffness
bending_stiffness
damping
```

当前版本实际启用：

- 面密度；
- 水平/竖直结构边刚度倍率；
- 三角网格对角边刚度倍率。

`bending_stiffness` 和 `damping` 暂时为 `None`。

#### 朝向

全局朝向不进入第一版训练集，仅在 orientation OOD 中出现，用来检查重力方向与边界关系改变后的泛化能力。

---

### 3.2 C1、C2、C3 的构造

完整训练 catalogue 为 3072 条，按连续前缀严格嵌套：

```text
0–255       anchors
256–1023    pairwise coverage
1024–2047   common multifactor
2048–3071   hard in-domain
```

因此：

```text
C1 = train[0:1024]
C2 = train[0:2048]
C3 = train[0:3072]
```

即：

$$
C_1\subset C_2\subset C_3.
$$

#### Anchors：256 条

包含：

- baseline；
- 每个离散轴的单因素变化；
- 基础 `shape × strain` 组合；
- 常见 `fixed mask × Dirichlet` 组合；
- 确定性补齐组合。

#### Pairwise：768 条

六个核心轴在过滤非法边界组合后，共有 1904 个合法二因素取值对。生成器使用确定性候选遍历和贪心覆盖，不调用随机数，C1 最终覆盖：

```text
1904 / 1904
```

#### Common multifactor：1024 条

在 C1 基础上追加复杂度分数不低于 3 的常规多因素组合。

#### Hard in-domain：1024 条

在 C2 基础上追加复杂度分数不低于 6 的困难训练域内组合。它们同时包含多个强因素，但仍只使用训练域参数，不属于 OOD。

---

### 3.3 Validation 与 test

Validation 固定为 128 条训练域内未见组合。C1、C2、C3 必须共用同一套 validation。

Test 固定为 256 条：

```text
ID combination             96
magnitude OOD              48
boundary/material OOD      48
orientation OOD            32
hard combined OOD          32
合计                       256
```

测试集不参与：

- loss 设计；
- 学习率选择；
- 训练停止；
- checkpoint 选择；
- batch size 选择。

---

## 4. 批量物理接口

旧项目将两个固定点消元为 669D reduced state，无法在同一 batch 中支持不同固定点数量。

本项目统一使用完整状态：

$$
y\in\mathbb R^{B\times225\times3},
$$

即每个环境使用 675D。

固定点处理方式：

1. 网络输入保留固定点位置和历史状态；
2. 网络输出通过逐环境 free mask 门控；
3. 每次更新后 hard projection 到动态 Dirichlet target；
4. 固定点 residual 清零；
5. 固定点与自由点之间的弹簧能仍正常计算。

### 4.1 拓扑

15×15 三角网格包含：

```text
225 vertices
616 spring edges
196 diagonal/shear edges
```

### 4.2 质量

使用三角形面积集总：

$$
A_i=\sum_{T\ni i}\frac{A_T}{3},
\qquad
m_i=\rho_A A_i.
$$

这使质量定义在不同网格分辨率间保持一致的连续体含义。

### 4.3 隐式 Euler 目标

每个环境的变分能量为：

$$
E(y)=
\sum_{i\in\mathcal F}
\frac{m_i}{2\Delta t^2}\|y_i-q_i\|^2
+
\sum_e\frac{k_e}{2}
\left(\|y_j-y_i\|-\ell_e\right)^2.
$$

其中 $\mathcal F$ 是该环境自己的自由点集合。

单元测试使用 float64 棄查：

$$
\nabla_yE(y)=r(y),
$$

解析 residual 与自动微分梯度一致。

---

## 5. Learned optimizer

默认网络：

```text
activation = identity
depth      = 1
width      = 256
bias       = False
dtype      = float64
```

输入为：

$$
[\tilde r_k,\tilde r_{k-1},\Delta y_{k-1}]
\in\mathbb R^{2025},
$$

输出为：

$$
\Delta y_k\in\mathbb R^{675}.
$$

当前 residual 使用逐环境质量预条件：

$$
\tilde r_k=\Delta t^2M^{-1}r_k.
$$

隐藏层正交初始化，输出层零初始化，所以初始模型输出严格为零。

---

## 6. 在线训练池

默认正式实验：

```text
training catalogue          C2 = 2048 scenarios
live environments           512
mini-batch size             32
K buckets                   1 / 3 / 10 / 30
environments per K          128
batch samples per K         8
environment lifetime        64 physical frames
learning rate               1e-3
gradient clipping           10
wall-clock limit            6 hours
```

一次 `optimizer.step()` 只给 batch 中 32 个环境各执行一次 learned update。

某个环境累计完成自身的 K 次 learned update 后，才推进一个物理帧。新物理帧从：

```text
自由点：y^(0) = x_n
固定点：y^(0) = x_D(t_{n+1})
```

开始。

训练池使用确定性调度：

- scenario 使用互质步长环形遍历；
- 每个 K bucket 内循环取样；
- batch 始终保持 K 均衡；
- 环境达到寿命或发生异常后分配 catalogue 中的新 scenario。

训练总量不使用模糊的 epoch 描述，而报告：

- optimizer updates；
- environment updates；
- completed physical frames；
- unique scenarios seen；
- wall-clock time。

若完成 $U$ 个 optimizer updates，batch 为 32，则：

$$
N_{\mathrm{environment\ updates}}=32U.
$$

最终 6 小时能完成多少 $U$，由 RTX 3090 实测吞吐量决定。

---

## 7. 训练 loss

训练目标为逐环境归一化的一步能量变化：

$$
\mathcal L=
\frac1B\sum_i
\frac{
E_i(y_i^{k+1})-
\operatorname{stopgrad}(E_i(y_i^k))
}{S_i}.
$$

归一化尺度：

$$
S_i=
\frac{\bar m_iL_i^2}{\Delta t^2}
+
\frac1{N_e}\sum_e k_{i,e}\ell_e^2.
$$

它用于降低质量、刚度和尺度差异导致的 batch 支配问题。

可选更新正则项为：

$$
\lambda_{\mathrm{step}}
\frac{\|\Delta y\|^2}{N_{\mathrm{free}}L^2},
$$

默认 `lambda_step=0`。

训练不使用：

- exact solution；
- reference error；
- K-step unroll；
- K-step 平均 loss。

---

## 8. 双验证与 checkpoint

### 8.1 Fast monitor

```text
motions           32
rollout frames    32
inner steps       10
interval          2000 optimizer updates
select checkpoint 否
```

用途是快速观察训练趋势和提前发现发散。

### 8.2 Checkpoint validation

```text
motions           128
rollout frames    100
inner steps       10
interval          10000 optimizer updates
select checkpoint 是
```

只有这套验证参与 `best_validation_model.pt` 的选择。

两套验证都保存：

```text
history.csv
per_motion.csv
curves.pt
figures/
runs/update_XXXXXXXXX/
```

### 8.3 Checkpoint 排名

使用稳定性优先的字典序：

$$
\left(
N_{\mathrm{fail}},
-\operatorname{P05}(T_{\mathrm{survival}}),
\operatorname{P95}\left(\frac{r_K}{r_0+\varepsilon}\right),
f_{\Delta E>0}
\right).
$$

比较顺序：

1. 失败 motion 更少；
2. survival frame p05 更高；
3. trajectory residual ratio p95 更低；
4. 能量上升比例更低。

Fast monitor 永远不会修改 best checkpoint。

---

## 9. Checkpoint 与日志

### 9.1 Checkpoint 文件

```text
latest_checkpoint.pt
best_validation_model.pt
periodic/checkpoint_update_XXXXXXXXX.pt
```

`latest` 和 `periodic` 保存完整恢复状态：

- 模型；
- Adam；
- pool 中的 p/v/q/y；
- target positions；
- previous residual/update；
- scenario IDs；
- K 和 inner iteration；
- physical frame、environment age；
- scenario/batch scheduler cursor；
- reset counters；
- Python、NumPy、PyTorch 和 CUDA RNG；
- catalogue fingerprint；
- update 数和累计训练时间。

`best_validation_model.pt` 不保存完整 pool，只用于最终评估。

### 9.2 训练日志

`train_log.csv` 包含：

- loss；
- normalized energy change；
- energy increase fraction；
- residual before/after；
- residual ratio p50/p95；
- update norm；
- gradient norm before/after clipping；
- 各类 reset；
- 各 K 完成的 physical frames；
- scenario 覆盖；
- optimizer/environment updates per second；
- CUDA peak allocated/reserved memory。

---

## 10. 最大显存与吞吐量测试

显存脚本在独立 Python 子进程中分别测试：

```text
batch size = 32 / 64 / 128 / 256 / 512
```

每个配置执行完整链路：

```text
ask
→ model forward
→ energy loss
→ backward
→ gradient clipping
→ Adam.step
→ tell
```

默认：

```text
warm-up updates     20
measured updates    100
reserved memory 上限 85%
```

输出：

```text
profiling/memory_probe/
├── memory_probe.csv
├── memory_probe.json
└── recommended_training_config.json
```

脚本会记录峰值显存、update mean/p50/p95、environment updates/s，以及根据实测吞吐量估算的 6 小时训练规模。

默认正式训练仍使用 batch 32。只有显式添加 `--use-recommended-batch` 时，一键脚本才使用吞吐量推荐值。

---

## 11. 最终评估

最终评估只加载训练阶段已经选出的 best checkpoint，不会重新选择模型。

Validation：

```text
128 scenarios
500 physical frames
K = 1 / 3 / 10 / 30
```

Test：

```text
256 scenarios
500 physical frames
K = 1 / 3 / 10 / 30
```

Test 同时报告：

- 全体 256 条；
- ID combination；
- magnitude OOD；
- boundary/material OOD；
- orientation OOD；
- hard combined OOD。

输出：

```text
final_evaluation/
├── summary.csv
├── summary.json
├── test_group_summary.csv
├── test_group_summary.json
├── evaluation_manifest.json
├── figures/
└── validation/
    ├── final_validation_k1/
    ├── final_validation_k3/
    ├── final_validation_k10/
    ├── final_validation_k30/
    ├── test_all_k1/
    ├── test_all_k3/
    ├── test_all_k10/
    └── test_all_k30/
```

主要指标：

- failed motion count；
- survival rate 和 survival frame p05；
- residual ratio p95；
- final residual p95/max；
- energy increase fraction；
- min/max edge ratio；
- maximum constraint error。

---

## 12. 一键运行

进入项目目录：

```bash
cd /data/zhoucy/sim_newton/cloth_15x15_500step_project_scale_up
```

执行默认完整流程：

```bash
python cloth08_run_end_to_end.py \
  --device cuda:0 \
  --overwrite
```

该命令依次执行：

1. C2、pool 512、batch 32/64/128/256/512 的显存与吞吐量测试；
2. C2、pool 512、batch 32、float64 的 6 小时正式训练；
3. best checkpoint 的 validation/test 500 帧、K={1,3,10,30} 最终评估。

只查看将要执行的命令，不真正运行：

```bash
python cloth08_run_end_to_end.py \
  --device cuda:0 \
  --dry-run
```

使用显存测试推荐 batch：

```bash
python cloth08_run_end_to_end.py \
  --device cuda:0 \
  --use-recommended-batch \
  --overwrite
```

从 `latest_checkpoint.pt` 继续训练：

```bash
python cloth08_run_end_to_end.py \
  --device cuda:0 \
  --skip-memory-probe \
  --resume
```

仅做最终评估：

```bash
python cloth08_run_end_to_end.py \
  --device cuda:0 \
  --skip-memory-probe \
  --skip-training
```

流水线日志保存在：

```text
cloth_15x15_scale_up_pipeline/pipeline_logs/
├── 01_memory_probe.log
├── 02_training.log
└── 03_final_evaluation.log
```

总状态保存在：

```text
cloth_15x15_scale_up_pipeline/pipeline_state.json
```

---

## 13. 分步运行

### 13.1 构造 catalogue

```bash
python cloth01_build_scenario_catalogue.py \
  --output-dir cloth_15x15_scale_up_pipeline/data/scenarios
```

只审计：

```bash
python cloth01_build_scenario_catalogue.py --audit-only
```

### 13.2 显存测试

```bash
python cloth06_probe_memory_and_throughput.py \
  --device cuda:0
```

### 13.3 正式训练

```bash
python cloth05_train_scale_up.py \
  --root cloth_15x15_scale_up_pipeline \
  --catalogue c2 \
  --device cuda:0 \
  --dtype float64 \
  --pool-size 512 \
  --batch-size 32 \
  --k-buckets 1 3 10 30 \
  --max-wall-hours 6 \
  --overwrite
```

### 13.4 最终评估

```bash
python cloth07_evaluate_best_checkpoint.py \
  --run-dir cloth_15x15_scale_up_pipeline/experiments/train_c2/activation_identity_depth_01_width_0256_no_bias/seed_42 \
  --device cuda:0 \
  --validation-frames 500 \
  --test-frames 500 \
  --inner-steps 1 3 10 30
```

---

## 14. C1/C2/C3 数据规模对比

三个数据规模实验应从相同随机种子独立训练，不能把 C1 checkpoint 继续训练成 C2，否则数据量和训练时长会耦合。

```bash
python cloth08_run_end_to_end.py --catalogue c1 --device cuda:0 --overwrite
python cloth08_run_end_to_end.py --catalogue c2 --device cuda:0 --overwrite
python cloth08_run_end_to_end.py --catalogue c3 --device cuda:0 --overwrite
```

默认每个实验各限制 6 小时，因此三组约需要 18 GPU hours，最终应比较：

- wall-clock 相同条件下的 optimizer updates；
- environment updates；
- scenario 覆盖；
- validation stability；
- 五类 test 的分组表现。

---

## 15. 测试

运行全部测试：

```bash
python -m unittest -v \
  test_scenario_catalogue.py \
  test_batched_physics.py \
  test_training_pool.py \
  test_final_evaluation.py
```

GitHub Actions 在 Python 3.10 和 3.12 下执行：

- 全目录编译；
- 所有命令行入口检查；
- catalogue 单元测试；
- batched physics 数值测试；
- training pool、loss、反传和恢复测试；
- final evaluation 分组汇总测试；
- CPU 显存脚本子进程 smoke test；
- 一键流水线 dry-run；
- 完整 catalogue 审计。

---

## 16. 结果解释注意事项

1. 当前没有 reference，因此 residual 下降和能量下降不等价于真实轨迹误差小；
2. failed motion 会在 checkpoint 选择中受到最高优先级惩罚；
3. 测试集必须在所有训练和选择完成后运行；
4. C1/C2/C3 比较必须使用相同 validation/test、模型、随机种子和训练时长；
5. batch size 改变会同时改变吞吐量和梯度统计，正式报告应记录实际 batch；
6. 若出现大量非 lifetime reset，应先检查 energy/residual/edge/position 分类，而不是直接增加训练时间；
7. 当前材料中的弯曲和阻尼字段只是预留接口，不能将其描述为已实现的物理项。
