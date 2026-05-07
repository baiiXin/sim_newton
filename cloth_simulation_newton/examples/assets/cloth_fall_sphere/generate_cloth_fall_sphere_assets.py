import numpy as np

from generate_cloth import generate_cloth_mesh
from generate_sphere import create_icosphere

import os

# 根据目标三角形数量，选择最接近的正二十面体细分层数
def _choose_subdivisions_for_target_triangles(target_tri_num):
    if target_tri_num is None or target_tri_num <= 0:
        return 2
    best_s = 0
    best_err = None
    for s in range(0, 6):
        tri_num = 20 * (4**s)
        err = abs(tri_num - target_tri_num)
        if best_err is None or err < best_err:
            best_err = err
            best_s = s
    return best_s


def generate_cloth_fall_sphere_assets(
    # cloth
    cloth_rects = [(-1.0, 1.0, -1.0, 1.0)],
    h1 = 0.5,
    h2 = 0.5,
    z  = 5.0,
    cloth_layers = int(1),
    z_offset = 0.5,
    # sphere
    sphere_center = (0.0, 0.0, 2.0),
    sphere_radius = 1.0,
    sphere_target_triangles = 200,
    # ground
    add_ground=True,
    ground_rect = None,
    ground_z = 0.0,
):
    # cloth_rects: [(a,b,c,d), ...] 多个布料矩形区域
    # sphere_center: 球心坐标 (cx,cy,cz)
    # sphere_radius: 球半径
    # sphere_target_triangles: 目标三角片个数（近似）
    # h1, h2: 布料网格在 x,y 方向的步长
    # z: 第一层布料的 z 高度
    # cloth_layers: 布料层数
    # z_offset: 相邻两层布料在 z 方向的偏移
    # add_ground: 是否在 z=0 添加一个大地面

    all_vertices = []
    all_triangles = []
    fixed_index = []
    mesh = {}

    vertex_offset = 0  # 累积顶点偏移量，用于把局部索引转换为全局索引
    tri_offset = 0     # 累积三角形偏移量，用于记录在全局 triangles 中的起始行
    cloth_counter = 0  # 布块计数，cloth1, cloth2, ...

    ### 1.cloth
    for layer_idx in range(cloth_layers):
        z_layer = z + layer_idx * z_offset
        for rect in cloth_rects:
            a, b, c, d = rect
            # 调用通用的 generate_cloth_mesh 生成当前层、当前矩形的布料网格
            pos_cloth, tri_cloth = generate_cloth_mesh(
                a, b, c, d, h1, h2, z_layer
            )

            pos_cloth = np.asarray(pos_cloth, dtype=np.float64)
            tri_cloth = np.asarray(tri_cloth, dtype=np.int32)

            tri_count = tri_cloth.shape[0]
            pos_count = pos_cloth.shape[0]

            # 将局部顶点索引平移到全局顶点索引
            triangles_global = tri_cloth + vertex_offset

            cloth_counter += 1
            cloth_name = f"cloth{cloth_counter}"
            mesh[cloth_name] = {
                "start_pos_index": int(vertex_offset),   # 顶点在全局 vertices 中的起始索引
                "pos_num": int(pos_count),               # 该布料的顶点数量
                "start_tri_index": int(tri_offset),      # 该布料三角形在全局 triangles 中的起始行
                "triangle_num": int(tri_count),          # 该布料的三角形数量
                "vertices": pos_cloth.copy(),               # 该布料自身的顶点坐标（局部）
                "triangles": tri_cloth.copy(),              # 该布料自身的三角拓扑（局部）
            }

            all_vertices.append(pos_cloth)
            all_triangles.append(triangles_global)

            vertex_offset += pos_count
            tri_offset += tri_count

    ### 2. sphere
    subdivisions = _choose_subdivisions_for_target_triangles(sphere_target_triangles)
    sphere_V, sphere_F = create_icosphere(
        subdivisions=subdivisions,
        center=np.asarray(sphere_center, dtype=np.float64),
        radius=float(sphere_radius),
    )

    sphere_V = np.asarray(sphere_V, dtype=np.float64)
    sphere_F = np.asarray(sphere_F, dtype=np.int32)

    sphere_pos_count = sphere_V.shape[0]
    sphere_tri_count = sphere_F.shape[0]

    # 球的三角形索引平移到全局
    sphere_triangles_global = sphere_F + vertex_offset

    mesh["sphere"] = {
        "start_pos_index": int(vertex_offset),     # 球的顶点在全局 vertices 中的起始索引
        "pos_num": int(sphere_pos_count),
        "start_tri_index": int(tri_offset),        # 球的三角形在全局 triangles 中的起始行
        "triangle_num": int(sphere_tri_count),
        "vertices": sphere_V.copy(),               # 球自身的顶点坐标（局部）
        "triangles": sphere_F.copy(),              # 球自身的三角拓扑（局部）
    }

    # 固定点
    sphere_fixed_indices = np.arange(vertex_offset, vertex_offset + sphere_pos_count, dtype=np.int32)
    fixed_index.append(sphere_fixed_indices)

    all_vertices.append(sphere_V)
    all_triangles.append(sphere_triangles_global)

    vertex_offset += sphere_pos_count
    tri_offset += sphere_tri_count

    ### 3. ground
    if add_ground and len(cloth_rects) > 0:
        if ground_rect is None:
            a0, b0, c0, d0 = cloth_rects[0]
            width_x = b0 - a0
            width_y = d0 - c0
            center_x = 0.5 * (a0 + b0)
            center_y = 0.5 * (c0 + d0)
            half_x = 1.5 * width_x
            half_y = 1.5 * width_y

            gx_min = center_x - half_x
            gx_max = center_x + half_x
            gy_min = center_y - half_y
            gy_max = center_y + half_y
        else:
            gx_min, gx_max, gy_min, gy_max = ground_rect

        # 以第一块布料为基准，在其中心位置放置一个面积放大 3 倍的地面
        ground_vertices = np.array(
            [
                [gx_min, gy_min, ground_z],
                [gx_max, gy_min, ground_z],
                [gx_max, gy_max, ground_z],
                [gx_min, gy_max, ground_z],
            ],
            dtype=np.float64,
        )

        # 仅由两个三角形组成的矩形地面（局部三角拓扑）
        ground_triangles = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ],
            dtype=np.int32,
        )

        ground_pos_count = ground_vertices.shape[0]
        ground_tri_count = ground_triangles.shape[0]

        # 地面的三角形索引平移到全局
        ground_triangles_global = ground_triangles + vertex_offset

        mesh["ground"] = {
            "start_pos_index": int(vertex_offset),               # 顶点在全局 vertices 中的起始索引
            "pos_num": int(ground_pos_count),
            "start_tri_index": int(tri_offset),                  # 地面三角形在全局 triangles 中的起始行
            "triangle_num": int(ground_tri_count),
            "vertices": ground_vertices.copy(),                  # 地面自身的顶点坐标（局部）
            "triangles": ground_triangles.copy(),          # 地面自身的三角拓扑（局部）
        }

        # 固定点
        ground_fixed_indices = np.arange(vertex_offset, vertex_offset + ground_pos_count, dtype=np.int32)
        fixed_index.append(ground_fixed_indices)

        all_vertices.append(ground_vertices)
        all_triangles.append(ground_triangles_global)

        vertex_offset += ground_pos_count
        tri_offset += ground_tri_count

    # 将所有图形的顶点拼接成一个大的顶点数组
    vertices = np.vstack(all_vertices).astype(np.float64, copy=False)
    # 将所有图形的三角形拼接成一个大的三角数组
    triangles = np.vstack(all_triangles).astype(np.int32, copy=False)

    # 在 mesh 顶层记录全局顶点和三角形，便于统一使用
    mesh["vertices_all"] = vertices
    mesh["triangles_all"] = triangles

    # 固定点
    fixed_index = np.hstack(fixed_index).astype(np.int32, copy=False)
    mesh["fixed_index"] = fixed_index

    return vertices, triangles, mesh


def main():
    # cloth
    cloth_rects = [(-17.0, 17.0, -17.0, 17.0)]
    h1 = 0.5
    h2 = 0.5
    z  = 15.0
    cloth_layers = 1
    z_offset = 0.5

    # sphere
    sphere_center = (0.0, 0.0, 8.0)
    sphere_radius = 5.0
    target_triangles = 2000

    # ground
    add_ground = True
    ground_rect = (-60.0, 60.0, -60.0, 60.0)
    ground_z = 0.0

    # generate assets
    vertices_all, triangles_all, mesh = generate_cloth_fall_sphere_assets(
        # cloth
        cloth_rects=cloth_rects,
        h1=h1,
        h2=h2,
        z=z,
        cloth_layers=cloth_layers,
        z_offset=z_offset,
        # sphere
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        sphere_target_triangles=target_triangles,
        # ground
        add_ground=add_ground,
        ground_rect=ground_rect,
        ground_z=ground_z,
    )

    #print("mesh:", mesh)
    print("mesh:", mesh.keys())

    print("vertices_all.shape", vertices_all.shape, type(vertices_all))
    print("triangles_all.shape", triangles_all.shape, type(triangles_all))
    print("mesh['fixed_index'].shape", mesh["fixed_index"].shape, type(mesh["fixed_index"]))

    print('\nsave assets !!!')
    # save assets
    name = "cloth_fall_sphere_1_unit.npz"
    save_dir = os.path.dirname(__file__)
    path = os.path.join(save_dir, name)
    np.savez(path, mesh=mesh)

    data = np.load(path, allow_pickle=True)
    print("data['mesh'].item().keys()", data["mesh"].item().keys())



if __name__ == "__main__":
    main()
