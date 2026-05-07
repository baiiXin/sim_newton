1. cloth_simulation_newton（待整理）
    牛顿法求解器
2. GNN_solver（正在进行测试）
    GNN求解器

目前的测试在/repo/GNN_solver/_src目录下进行。

主要目标是测试GNN是不是能作为仿真中隐式欧拉变分能量优化问题的迭代求解器使用。
1. 预训练的形式是否可行
    - 换双精度
    - MLP替换GNN
    - 输入特征是不是需要调整？
2. 如果可行，继续加碰撞能量测试
