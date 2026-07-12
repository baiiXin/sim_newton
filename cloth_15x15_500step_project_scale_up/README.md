# 15×15 布料 Scale-up Training Pool 项目

本目录用于在 `cloth_15x15_500step_project` 基础上扩展物理场景规模。第一阶段只构造 **scenario catalogue**，不生成 reference，也不进行每个物理时间步附近的初值采样。

## 当前阶段目标

- `C1 = 1024` 个训练 scenario；
- `C2 = 2048` 个训练 scenario；
- `C3 = 3072` 个训练 scenario；
- 三个训练集合严格嵌套；
- `128` 个 validation scenario；
- `256` 个 test scenario；
- 所有场景来自手工定义的离散模板和确定性组合算法；
- 不使用随机采样、Sobol motion 或预生成 reference。

## 场景维度

### 初始形状：7 种训练模板

- 平面；
- 四种鞍形；
- 向上凸包；
- 向下凸包。

扭转被合并进鞍形初始形变，不再设置独立随机 twist 参数。

### 拉伸和压缩：7 种训练模板

- 无形变；
- x/y 单轴拉伸；
- x/y 单轴压缩；
- 双向拉伸；
- 双向压缩。

训练倍率为 `0.85 / 1.00 / 1.15`，测试 OOD 使用 `0.70 / 1.30`。

### 初始速度：19 种训练模板

- 静止；
- x/y/z 六个方向的中速和高速平移；
- 绕 x/y/z 三个轴的正反旋转。

测试 OOD 额外使用极高速平移和更高角速度。

### 固定点：22 种训练模板

- 单点：四角、四边中点、中心点，共 9 种；
- 两点：对角、同边两角、对边中点、中心+边中点，共 12 种；
- 四点：四个角点，共 1 种。

固定点通过归一化 `(u,v)` 或拓扑规则定义，可直接映射到 5×5、15×15 或更高分辨率。

### Dirichlet 边界运动：7 种

- 静态；
- 水平圆周正/反；
- 竖直圆周正/反；
- 正/反扭布。

所有运动满足 `x_D(0)=x_initial`，不会在第 0 帧产生位置跳变。单固定点不允许使用扭布模式。

### 材料：8 种训练预设

schema 保留：

```text
areal_density
stretch_stiffness
shear_stiffness
bending_stiffness
optional_damping
```

当前弹簧模型实际使用面密度、结构边刚度倍率和对角边刚度倍率；`bending_stiffness` 与 `damping` 暂时为 `None`，避免假装当前模型已经实现独立弯曲和阻尼。

## 训练 catalogue 分块

```text
0–255       anchors：基准、单因素和基础组合
256–1023    pairwise：覆盖所有合法二因素组合
1024–2047   common_multifactor：常规多因素组合
2048–3071   hard_in_domain：训练范围内困难组合
```

前 1024 条已经覆盖训练离散轴上所有合法 pairwise combinations。后续集合严格追加，因此可以直接研究 motion 数量的 scaling curve。

## 测试集组成

```text
ID combination             96
magnitude OOD              48
boundary/material OOD      48
orientation OOD            32
hard combined OOD          32
合计                       256
```

全局朝向不放入第一版训练集，但保留 32 个 orientation OOD scenario，用实验判断它是否确实不重要。

## 运行

在本目录中执行：

```bash
python cloth01_build_scenario_catalogue.py \
  --output-dir cloth_15x15_scale_up_pipeline/data/scenarios
```

只做内存构造和审计：

```bash
python cloth01_build_scenario_catalogue.py --audit-only
```

单元测试：

```bash
python -m unittest -v test_scenario_catalogue.py
```

## 输出

```text
cloth_15x15_scale_up_pipeline/data/scenarios/
├── manifest.json
├── audit.json
├── train_c1_1024.json
├── train_c1_1024.csv
├── train_c2_2048.json
├── train_c2_2048.csv
├── train_c3_3072.json
├── train_c3_3072.csv
├── validation_128.json
├── validation_128.csv
├── test_256.json
└── test_256.csv
```

Catalogue 只保存模板 ID 和参数引用，不保存每条场景的 225×3 初始位置数组。运行时由 `build_initial_state()` 按分辨率生成初始位置、速度、fixed mask 和材料参数，避免冗余存储并保留跨分辨率扩展能力。

## 审计内容

`audit.json` 检查：

- 各 split 数量；
- C1/C2/C3 是否严格嵌套；
- split 内和 split 间是否重复；
- fixed mask 与 Dirichlet motion 是否兼容；
- C1 是否覆盖全部合法 pairwise combinations；
- 所有初始几何是否有限、是否存在退化边；
- moving Dirichlet 是否在 `t=0` 发生位置跳变。

## 下一阶段

Catalogue 审计通过后，再实现：

1. 每个 environment 独立的 batched fixed mask；
2. 每个 environment 独立的材料参数；
3. 512 live environments、batch size 32 的 mini-batch training pool；
4. 逐场景归一化的一步能量下降 loss；
5. 不依赖 reference 的连续 rollout validation。
