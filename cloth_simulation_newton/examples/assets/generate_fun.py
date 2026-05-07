import numpy as np

def generate_spring(Mass_P, Mass_T, stiff_k):
    '''
    功能: 根据质点网格生成弹簧网格
    输入:
        Mass_P: 质点的坐标矩阵，大小为 Nm x 3
        Mass_T: 三角形的索引矩阵，大小为 3 x Nt，其中 Nt 是三角形的数量
        stiff_k: 弹簧的刚度
    输出:
        NS: 弹簧的数量
        Spring_ele: 弹簧的索引矩阵，大小为 NS x 2，其中 NS 是弹簧的数量
        Spring_len: 弹簧的长度，大小为 NS
        Spring_stiff_k: 弹簧的刚度，大小为 NS
    '''
    
    Ne = Mass_T.shape[0]
    # 修正数组初始化方式
    Spring_ele_0 = np.zeros((3*Ne, 3), dtype=int)
    for i in range(Ne):
        ele = np.array([
            [Mass_T[i][0], Mass_T[i][1], i],
            [Mass_T[i][0], Mass_T[i][2], i],
            [Mass_T[i][1], Mass_T[i][2], i]
        ])
        Spring_ele_0[3*i:3*i+3] = ele  # 修正切片范围

    Spring_ele_0 = Spring_ele_0[np.lexsort((Spring_ele_0[:,2], Spring_ele_0[:,1], Spring_ele_0[:,0]))]
    Spring_ele_0 = np.vstack([Spring_ele_0, [-1,-1,-1]])  # 修正添加结束标记的方式
    Ns = Spring_ele_0.shape[0]
    Spring_ele = []
    for i in range(Ns-1):  # 修正循环范围
        if Spring_ele_0[i][0] == Spring_ele_0[i+1][0] and Spring_ele_0[i][1] == Spring_ele_0[i+1][1]:
            continue
        else:
            Spring_ele.append([Spring_ele_0[i][0], Spring_ele_0[i][1]])

    NS = len(Spring_ele)
    Spring_len = np.zeros(NS)
    Spring_stiff_k =  stiff_k  
    for i in range(NS):
        Spring_len[i] = np.linalg.norm(Mass_P[Spring_ele[i][0]]-Mass_P[Spring_ele[i][1]])     

    return NS, np.array(Spring_ele), Spring_len, Spring_stiff_k


# 生成质点 √
def generate_mass(a, b, c, d, h1, h2, z, mass_m): 
    '''
    功能: 根据矩形区域和高度生成质点网格
    输入:
        a, b, c, d: 矩形区域的四个端点坐标
        h1, h2: 矩形区域的两个方向的步长
        z: 矩形区域的高度
        mass_m: 每个质点的质量
    输出:
        Nm: 质点的数量
        P: 质点的坐标矩阵，大小为 Nm x 3
        T: 三角形的索引矩阵，大小为 3 x Nt，其中 Nt 是三角形的数量
        V: 三角形的顶点坐标矩阵，大小为 Nt x 3 x 3
        mass_m: 每个质点的质量
    '''

    # Nx-N1; Ny-N2
    N1 = int((b - a) / h1)
    N2 = int((d - c) / h2)
    
    Nx = N1 + 1
    Ny = N2 + 1
    X = np.arange(a, b + h1, h1)
    Y = np.arange(c, d + h2, h2)
    
    # 三角，线性元
    Nm = (N1 + 1) * (N2 + 1)
    
    # 预先计算所有需要的数组大小
    total_points = Nx * Ny
    total_triangles = 2 * (Nx-1) * (Ny-1)
    
    # 使用更高效的内存布局 (Fortran顺序)
    V = np.zeros((3, total_points), dtype=np.float32, order='F').T
    T = np.zeros((3, total_triangles), dtype=np.int32, order='F')

    # 结点坐标P - 向量化版本(增加z坐标)
    X_grid, Y_grid = np.meshgrid(X, Y, indexing='xy')  # 改为'xy'而不是'ij'
    P_xy = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
    P = np.column_stack([P_xy, np.full(P_xy.shape[0], z)])  # 增加z坐标
    
    # 三角元T - Python版本
    
    step_T_node = 0
    step_P = Ny
    for i in range(1, Nx):
        for j in range(Ny-1):
            step_T_node += 1
            step_P += 1
            T[:, 2*step_T_node-2] = [step_P-Ny-1, step_P-1, step_P-Ny]  # 所有索引减1
            T[:, 2*step_T_node-1] = [step_P-Ny, step_P-1, step_P]  # 所有索引减1
        step_P += 1
        
    # 先转置T数组，再对每个三角形的顶点索引进行排序
    T_transposed = T.T  # 转置操作
    T_sorted = np.sort(T_transposed, axis=1)
    
    return Nm, P, T_sorted, V, mass_m  # 返回排序后的三角形索引
   
