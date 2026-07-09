# unit_test_for_5x5_cloth 目录说明与评价

生成时间：2026-07-02  
当前工作目录：`/data/zhoucy/sim_newton/unit_test_for_5x5_cloth`

## 目录概览

这个目录是一个独立的 5x5 固定左边缘三角布料 learned optimizer 实验实现。代码目标是测试一个 MLP/GNN 风格的学习型迭代求解器能否作为隐式 Euler 变分能量优化问题的迭代 solver。当前目录内没有 `GNN_solver/` 子目录，实际核心代码集中在 `cloth5x5/`。

顶层内容如下：

| 路径 | 作用 |
| --- | --- |
| `fixed_left_edge_5x5_cloth_multi_motion_train_compare.py` | 主训练与对比实验入口。封装 confirmed experiment：多 motion、多 time-step 数据，训练 learned optimizer，并与 gradient descent 和 full Newton 对比。 |
| `fixed_left_edge_5x5_cloth_multi_motion_rollout.py` | 连续 rollout 入口。加载训练产物，对 reference、MLP、GD、Newton 做 500-frame 物理轨迹对比。 |
| `cloth5x5/` | 核心 Python 包，包含配置、拓扑、物理能量、参考解生成、数据集、模型、训练循环、评估、绘图和 rollout 逻辑。 |
| `tests/` | 轻量单元测试，覆盖设备解析、拓扑计数、固定点重建、能量梯度一致性、Newton 单步能量下降等。 |
| `.agents/`, `.codex/`, `.git/` | 代理/工作区/Git 元数据。 |
| `__pycache__/`, `cloth5x5/__pycache__/`, `tests/__pycache__/` | Python 缓存产物。 |

## 核心模块说明

### `cloth5x5/constants.py`

定义 5x5 网格、固定顶点、自由顶点、三角网格拓扑、弹簧边、训练超参数、时间划分、随机种子和数值容差。当前布料有：

- 25 个粒子；
- 左上角和左下角 2 个固定顶点；
- 23 个自由粒子；
- 69 维自由状态；
- 56 条弹簧边；
- 32 个三角面；
- 默认 dtype 为 `torch.float64`。

### `cloth5x5/config.py`

定义运行配置、物理配置、motion spec、数据集 bundle 等 dataclass。`default_physical_config()` 构造默认布料初始位置、质量、重力、时间步长、弹簧刚度和 rest length。

### `cloth5x5/physics.py`

实现隐式 Euler 变分能量、stationarity residual、解析 Hessian、Newton update、gradient descent update、参考解求解和物理状态推进。这是实验可信度的核心模块。代码中能量、梯度和 Hessian 都围绕消元后的 23 个自由顶点实现。

### `cloth5x5/model.py`

定义 `MLPOptimizer`。模型输入是 mass-preconditioned stationarity residual，输出 `delta`，外部通过 `y_next = y + delta` 应用更新。结构是 69 -> 69 -> Identity -> 69，第一层正交初始化，输出层零初始化。这个设计比较适合作为 learned iterative optimizer 的最小可控基线。

### `cloth5x5/dataset.py`

围绕 reference solution 生成 Sobol 初始点数据。数据集记录 initial state、`q`、mass、exact state、problem/motion/time 索引和元数据。训练点可以显式包含当前物理状态和 exact state，这有助于诊断近解行为。

### `cloth5x5/motions.py`

构造 32 个完整 motion，并按 complete motion 划分 train/validation/id test/OOD test。包含手工 anchor、Sobol in-domain motion 和手工 OOD motion。这个划分比随机混合 time-step 更合理，因为能防止同一 motion 泄漏到训练和测试。

### `cloth5x5/train_loop.py`

实现 full-batch 训练循环。训练按 epoch 使用逐步增长的 K 次 learned optimizer rollout，目标来自变分能量；保存 last/best checkpoint、训练日志、诊断日志、验证日志和最终评估报告。评估结果同时对 learned、gradient descent、full Newton 做对比。

### `cloth5x5/evaluate.py`

评估每个 solver 在每次迭代后的 residual、energy gap、exact error、粒子误差、弹簧长度误差等指标。它显式记录 step 0，这符合“迭代求解器需要看初始状态和每步变化”的要求。

### `tests/`

测试量不大，但抓住了几个关键基础性质：

- 拓扑数量是否符合 5x5 三角布料；
- 固定顶点重建是否正确；
- autograd 梯度是否匹配手写 stationarity residual；
- spring length 是否正；
- Newton 单步是否不增加能量；
- `resolve_device("auto")` 和 `resolve_device("cpu")` 的行为。

## 工程评价

整体评价：这是一个实验目标明确、边界较清楚的小型研究代码库。它不是通用布料仿真框架，而是围绕“5x5 固定左边缘布料 learned iterative solver”组织的可复现实验。模块拆分合理，物理、数据、模型、训练、评估和绘图分离得比较清楚。

优点：

- 实验定义具体：5x5、固定左边缘、32 个完整 motion、100 个 time step、train/validation/test/OOD split 都写在代码里。
- learned solver 的语义清楚：模型输出 `delta`，外部执行加法更新，符合迭代求解器形式。
- 有强基线：gradient descent 和 full Newton 都在同一评估框架里比较。
- 评估指标比较完整：不仅看均值，也看 p95、max、worst-motion p95 和 worst-motion max，有利于暴露边界失败。
- 使用 `float64`，对这类小规模物理优化问题更稳妥。
- `evaluate_solver_on_dataset()` 记录 step 0，便于观察初始误差和每轮下降趋势。
- 有单元测试覆盖物理梯度一致性，这是物理优化代码里很关键的检查。

主要风险和不足：

- 运行依赖没有在目录中显式声明。我尝试做最小运行检查时，导入 `cloth5x5` 因缺少 `matplotlib` 失败。建议补充 `requirements.txt` 或环境说明。
- `cloth5x5/__init__.py` 顶层立即 `import matplotlib` 并设置 backend，这会让不需要绘图的导入也依赖 matplotlib。更稳妥的方式是把绘图依赖限制在 plotting/viz 相关模块中。
- `cloth5x5/motions.py` 中 `generate_in_domain_sobol_motion_specs()` 使用 `TORCH_DTYPE`，但当前文件导入列表没有包含 `TORCH_DTYPE`。这在真正执行 `build_motion_catalogue()` 时可能触发 `NameError`，但当前环境先被 `matplotlib` 缺失挡住了。
- 训练循环当前是 full unroll 后一次 optimizer step，梯度会穿过 K 次模型更新。这个选择不是错，但作为“迭代求解器训练”应在实验报告或代码注释中明确说明，因为它和“每一步单独优化并 detach rollout state”的训练语义不同。
- `DEFAULT_DEVICE = "cuda:0"` 对无 GPU 环境不友好。虽然代码支持 `--device auto/cpu/cuda`，但默认值倾向 CUDA，跑脚本时可能直接失败。
- 测试主要覆盖基础物理和设备解析，尚未覆盖数据集构造、motion split 无泄漏、评估 step 0 日志完整性、训练报告保存完整性和 rollout 输入输出契约。

## 建议

1. 增加依赖文件，例如 `requirements.txt`，至少列出 `torch`、`numpy`、`matplotlib`。
2. 将 `matplotlib` 的 backend 设置从 `cloth5x5/__init__.py` 移到绘图入口，避免核心物理/数据模块被绘图依赖阻塞。
3. 修复 `cloth5x5/motions.py` 对 `TORCH_DTYPE` 的缺失导入，并增加一个 `build_motion_catalogue(default_physical_config())` 的单元测试。
4. 对训练梯度流策略增加明确说明：当前是 K-step unrolled training；如果后续改成每步一个 optimizer step，应在每步后 detach rollout state。
5. 增加针对评估日志的测试，确认 selected report steps 包含 `0`，并确认训练报告同时保存 train log、diagnostic log、validation log 和 evaluation。
6. 考虑把默认 device 改为 `auto`，同时保留显式 `--device cuda:0`。

## 已做验证

执行过以下检查：

```bash
pwd
rg --files -g '!*__pycache__*' -g '!*.pyc'
find . -maxdepth 2 -type d
git status --short
python3 -m py_compile cloth5x5/*.py fixed_left_edge_5x5_cloth_multi_motion_train_compare.py fixed_left_edge_5x5_cloth_multi_motion_rollout.py tests/*.py
```

结果：

- 当前目录确认为 `/data/zhoucy/sim_newton/unit_test_for_5x5_cloth`。
- `python3 -m py_compile ...` 通过。
- `python` 命令不存在，需使用 `python3`。
- 最小运行检查 `build_motion_catalogue(default_physical_config())` 未完成，因为导入包时缺少 `matplotlib`。

