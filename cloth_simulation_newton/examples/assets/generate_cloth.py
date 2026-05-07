import numpy as np

def generate_cloth_mesh(a, b, c, d, h1, h2, z):
    '''
    功能: 生成矩形布料网格，保证法向量统一向上 (+Z)
    输入:
        a, b: x方向范围 [a, b]
        c, d: y方向范围 [c, d]
        h1: x方向步长
        h2: y方向步长
        z: 平面的高度
    输出:
        P: 质点坐标 (N_points x 3)
        T: 三角形索引 (N_triangles x 3)
    '''
    # 1. 确定网格维度 (使用 round 防止浮点误差)
    Nx = int(np.round((b - a) / h1)) + 1
    Ny = int(np.round((d - c) / h2)) + 1
    
    # 2. 生成坐标点 (P)
    x = np.linspace(a, b, Nx)
    y = np.linspace(c, d, Ny)
    
    # meshgrid 生成网格，注意 indexing='xy'
    # 行索引对应 y，列索引对应 x
    X_grid, Y_grid = np.meshgrid(x, y, indexing='xy')
    
    # 展平坐标，形成点列表
    # 顺序：先排第一行(y=c)的所有x，再排第二行...
    P_xy = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    N_points = P_xy.shape[0]
    
    # 添加 z 坐标
    P = np.column_stack([P_xy, np.full(N_points, z)])
    
    # 3. 生成三角形拓扑 (T)
    # 我们使用向量化方法生成索引，而不是双重循环
    
    # 获取网格左下角的点索引 (即每个正方形格子的左下角点)
    # 只需要前 Ny-1 行和前 Nx-1 列的点作为基准
    # 生成一个 (Ny-1, Nx-1) 的矩阵，存的是点索引
    ids = np.arange(Ny * Nx).reshape((Ny, Nx))
    node_indices = ids[:-1, :-1].ravel()  # 对应所有格子的左下角点 P0
    
    # 定义每个格子四个角的相对偏移
    # P2(左上) --- P3(右上)
    # |             |
    # P0(左下) --- P1(右下)
    
    P0 = node_indices
    P1 = node_indices + 1
    P2 = node_indices + Nx
    P3 = node_indices + Nx + 1
    
    # 构建三角形，保证逆时针 (Counter-Clockwise) 以使法向量朝上 (+Z)
    # 划分方式：对角线连接 P1-P2
    
    # 三角形 1: (P0, P1, P2) -> 向量 P0->P1 (x+), P0->P2 (y+), Cross -> z+
    T1 = np.column_stack([P0, P1, P2])
    
    # 三角形 2: (P1, P3, P2) -> 向量 P1->P3 (y+), P1->P2 (x-), 这里的顺序要注意
    # 验证: (P1, P3, P2)
    # Vec1 = P3-P1 = (0, dy, 0)
    # Vec2 = P2-P1 = (-dx, dy, 0)
    # Cross = (0, 0, dx*dy) -> z+ (正确)
    T2 = np.column_stack([P1, P3, P2])
    
    # 合并并重塑
    T = np.vstack([T1, T2])
    
    return P, T

def generate_unique_springs(P, T, stiff_k):
    '''
    功能: 从三角网格中提取去重后的边 (弹簧)
    输入:
        P: 质点坐标
        T: 三角形索引
        stiff_k: 刚度系数
    输出:
        num_springs: 弹簧数量
        spring_indices: 弹簧连接索引 (N_springs x 2)
        spring_lengths: 弹簧原长
        spring_stiffness: 弹簧刚度数组
    '''
    # 1. 提取所有三角形的三条边
    # 边: [0,1], [1,2], [2,0]
    edges_raw = np.vstack([
        T[:, [0, 1]],
        T[:, [1, 2]],
        T[:, [2, 0]]
    ])
    
    # 2. 边内部排序 (关键步骤！)
    # 将 [5, 1] 变成 [1, 5]，确保无向边的唯一性
    edges_sorted = np.sort(edges_raw, axis=1)
    
    # 3. 去重
    # unique 函数去除重复行
    unique_edges = np.unique(edges_sorted, axis=0)
    
    num_springs = unique_edges.shape[0]
    
    # 4. 计算长度
    # 获取端点坐标
    p_start = P[unique_edges[:, 0]]
    p_end = P[unique_edges[:, 1]]
    
    # 向量计算距离
    lengths = np.linalg.norm(p_end - p_start, axis=1)
    
    # 5. 刚度数组
    k_array = np.full(num_springs, stiff_k)
    
    return num_springs, unique_edges, lengths, k_array

# ================= 使用示例 =================
if __name__ == "__main__":
    # 参数设置
    a, b = 0.0, 1.0   # X范围
    c, d = 0.0, 1.0   # Y范围
    h1 = 0.5          # X步长 (生成 3 个点: 0, 0.5, 1.0)
    h2 = 0.5          # Y步长 (生成 3 个点: 0, 0.5, 1.0)
    z_val = 5.0       # 高度
    k_val = 100.0     # 刚度
    
    print("--- 1. 生成网格 ---")
    P, T = generate_cloth_mesh(a, b, c, d, h1, h2, z_val)
    print(f"质点数量: {P.shape[0]}")
    print(f"三角形数量: {T.shape[0]}")
    
    # 验证拓扑
    print("\n前9个三角形索引:\n", T[:9])
    print("\n前9个三角形索引:\n", P[:9])
    
    print("\n--- 2. 生成并去重弹簧 ---")
    Ns, Springs, Lens, Ks = generate_unique_springs(P, T, k_val)
    
    print(f"去重后的弹簧(边)数量: {Ns}")
    print("前16个弹簧:\n", Springs[:16])
    print("前16个长度:\n", Lens[:16])
    print("前16个刚度:\n", Ks[:16])
    print(Ns)
    
    # 简单验证逻辑:
    # 2x2的格子(3x3=9个点)，应该有:
    # 横边: 2行 * 3段 = 6? 不，是 3行 * 2段 = 6根
    # 竖边: 3列 * 2段 = 6根
    # 斜边: 2*2个格子，每个格子1根对角线 = 4根
    # 总共应该 = 16 根弹簧
    # (具体数量取决于三角形划分方向，这里每个方格被切成2个三角形，共有一条对角线)