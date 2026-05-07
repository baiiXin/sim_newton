import warp as wp
import numpy as np
import os
import argparse
from warp.sim.collide import TriMeshCollisionDetector

def list_input_files(input_dir):
    files = [f for f in os.listdir(input_dir) if f.startswith("cloth_data_") and f.endswith(".npy")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
    return files

def choose_file_interactive(files, input_dir):
    print("可选数据文件:")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")
    print("请输入索引或文件名:")
    s = input().strip()
    if os.path.isabs(s) or s.endswith(".npy"):
        p = s if os.path.isabs(s) else os.path.join(input_dir, s)
        if os.path.exists(p):
            return p
    if s.isdigit():
        i = int(s)
        i = max(0, min(i, len(files) - 1))
        return os.path.join(input_dir, files[i])
    return os.path.join(input_dir, files[0])

def load_pair(data_path, input_dir):
    base = os.path.splitext(os.path.basename(data_path))[0]
    if not base.startswith("cloth_data_"):
        raise ValueError("数据文件名需为 cloth_data_*.npy")
    suffix = base[len("cloth_data_"):]
    topy_path = os.path.join(input_dir, f"cloth_topy_{suffix}.npy")
    if not os.path.exists(topy_path):
        raise FileNotFoundError("未找到对应的 cloth_topy_*.npy")
    data = np.load(data_path)
    tris = np.load(topy_path).astype(np.int32)
    return data, tris

def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def choose_frame_interactive(N):
    print(f"可选帧索引范围: 0 到 {N-1}")
    print("请输入帧索引(数字)，或输入 first/last：")
    s = input().strip().lower()
    if s in ("", "last"):
        return N - 1
    if s == "first":
        return 0
    try:
        i = int(s)
    except Exception:
        i = N - 1
    return clamp(i, 0, N - 1)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "..", "render", "input")
    files = list_input_files(input_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.list:
        for i, f in enumerate(files):
            print(f"[{i}] {f}")
        return
    if not files and args.file is None:
        raise FileNotFoundError("input 目录无 cloth_data_*.npy")
    data_file = args.file
    if data_file is None or args.choose:
        data_file = choose_file_interactive(files, input_dir)
    data, triangles = load_pair(data_file, input_dir)
    N, M, _ = data.shape
    if args.frame is None:
        frame = choose_frame_interactive(N)
    else:
        frame = clamp(int(args.frame), 0, N - 1)
    verts_np = data[frame].astype(np.float32)
    device = wp.get_device(args.device)
    vertices = [wp.vec3(float(verts_np[i, 0]), float(verts_np[i, 1]), float(verts_np[i, 2])) for i in range(M)]
    builder = wp.sim.ModelBuilder()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vertices=vertices,
        indices=triangles.reshape(-1),
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=0.02,
        tri_ke=1.0e5,
        tri_ka=1.0e5,
        tri_kd=2.0e-6,
        edge_ke=10,
    )
    try:
        builder.color()
    except Exception:
        pass
    model = builder.finalize(device=device)
    vbd_integrator = wp.sim.VBDIntegrator(model)
    collision_detector = TriMeshCollisionDetector(model)
    collision_detector.refit(verts_np)
    # 碰撞信息
    collision_detector.vertex_triangle_collision_detection(0.02)
    collision_detector.edge_edge_collision_detection(0.02)
    print('collision_info:\n', collision_detector.collision_info)
    # 检测三角形三角形碰撞
    collision_detector.triangle_triangle_intersection_detection()
    print("model.tri_indices", model.tri_indices.shape)
    print("model.edge_indices", model.edge_indices.shape)
    print("frame", frame)
    print("triangle_intersecting_triangles:")
    print(collision_detector.triangle_intersecting_triangles.numpy())
    counts = collision_detector.triangle_intersecting_triangles_count.numpy()
    print("triangle_intersecting_triangles_count:")
    print(counts)
    print("resize_flags:")
    print(collision_detector.resize_flags.numpy())
    print("Total intersections:", int(counts.sum()))

if __name__ == "__main__":
    main()


