import numpy as np


# 生成单位正二十面体并进行细分，得到单位球面的三角网格
def _create_unit_icosphere(subdivisions=2):
    # 正二十面体顶点坐标中用到的黄金比例
    t = (1.0 + np.sqrt(5.0)) / 2.0

    # 初始 12 个顶点
    V = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=np.float64,
    )

    # 初始 20 个三角面片（索引）
    F = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int32,
    )

    # 先归一化到单位球面上
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    # 计算边中点的函数，并利用 cache 共享中点，避免重复顶点
    def midpoint(i, j, cache):
        key = tuple(sorted((i, j)))
        if key in cache:
            return cache[key]
        m = (V[i] + V[j]) * 0.5
        m /= np.linalg.norm(m)
        cache[key] = len(V_list)
        V_list.append(m)
        return cache[key]

    # 细分若干次：每次把一个三角面细分成 4 个小三角面
    for _ in range(subdivisions):
        cache = {}
        V_list = list(V)
        F_new = []

        for tri in F:
            i, j, k = tri
            a = midpoint(i, j, cache)
            b = midpoint(j, k, cache)
            c = midpoint(k, i, cache)

            F_new += [
                [i, a, c],
                [j, b, a],
                [k, c, b],
                [a, b, c],
            ]

        V = np.array(V_list, dtype=np.float64)
        F = np.array(F_new, dtype=np.int32)

    return V, F


# 根据细分层数 + 球心 + 半径生成球面网格
def create_icosphere(subdivisions=2, center=(0.0, 0.0, 0.0), radius=1.0):
    # 先生成单位球面网格
    V, F = _create_unit_icosphere(subdivisions=subdivisions)
    # 再缩放到指定半径，并平移到指定球心
    V = radius * V + np.asarray(center, dtype=np.float64)
    return V, F


def main():
    V, F = create_icosphere(subdivisions=3, radius=1.0)
    print(V.shape, F.shape)


if __name__ == "__main__":
    main()
