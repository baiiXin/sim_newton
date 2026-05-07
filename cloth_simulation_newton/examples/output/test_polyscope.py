# 文件用途：使用 Polyscope 对布料仿真数据进行可视化并导出视频/帧
# 注释分块：导入、数据加载、网格构建、三角面加载、可视化主流程

# 导入依赖
import os
import math
import numpy as np
import argparse
import trimesh


# 在候选路径列表中查找首个存在的文件路径
def _find_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# 加载布料顶点数据（优先完整帧序列，退路为单帧）
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

def load_cloth_data_and_topy_for(data_file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "data")
    suffix = os.path.splitext(os.path.basename(data_file))[0][len("cloth_data_"):]
    topy_file = os.path.join(input_dir, f"cloth_topy_{suffix}.npy")
    data = np.load(data_file)
    topy = np.load(topy_file) if os.path.exists(topy_file) else None
    # 打印加载信息
    print(f"loaded {data_file} with shape {data.shape}")
    if topy is not None:
        print(f"loaded {topy_file} with shape {topy.shape}")
    return data, topy

# 根据顶点数 nm 构造 n×n 网格的两三角剖分
def build_grid_triangles(nm):
    n = int(round(math.sqrt(nm)))
    if n * n != nm:
        # 顶点数量无法组成正方形网格时返回 None
        return None
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            # 每个网格单元拆分成两个三角形
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return np.array(faces, dtype=np.int32)

# 加载三角面索引（磁盘优先，否则基于网格构建）
def load_triangles(nm, topy):
    if topy is not None:
        return topy
    return build_grid_triangles(nm)


# 主流程：初始化 Polyscope、注册对象、环绕拍摄并导出
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "data")
    files = list_input_files(input_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args()
    if args.list:
        for i, f in enumerate(files):
            print(f"[{i}] {f}")
        return
    if not files and args.file is None:
        raise FileNotFoundError("no cloth_data_*.npy in input")
    data_file = args.file
    if data_file is None or args.choose:
        data_file = choose_file_interactive(files, input_dir)
    cloth_data, topy = load_cloth_data_and_topy_for(data_file)
    N, M, _ = cloth_data.shape
    triangles = load_triangles(M, topy)
    # 修复法向
    #mesh_cloth1 = trimesh.Trimesh(vertices=cloth_data[0], faces=triangles, process=False)
    #mesh_cloth1.fix_normals()
    #triangles = mesh_cloth1.faces
    try:
        import cv2
    except Exception:
        cv2 = None
    import polyscope as ps
    if os.environ.get("DISPLAY", "") == "":
        ps.set_allow_headless_backends(True)

    # 3) 初始化 Polyscope 渲染环境
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    TARGET_W, TARGET_H = 1920, 1080
    ps.set_window_size(TARGET_W, TARGET_H)

    # 4) 注册对象：优先表面网格，否则点云
    if triangles is not None:
        mesh = ps.register_surface_mesh("cloth", cloth_data[0], triangles, color=(0.5, 0.7, 1.0), smooth_shade=True)
    else:
        mesh = ps.register_point_cloud("cloth_points", cloth_data[0])
        try:
            mesh.set_radius(0.02, relative=True)
        except Exception:
            pass

    # 5) 输出配置：视频或帧序列
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cloth_polyscope.mp4")
    fps = 300
    frame_size = (1920, 1080)

    try:
        import imageio
    except Exception:
        imageio = None
    writer_cv2 = None
    writer_imageio = None
    if cv2 is not None:
        writer_cv2 = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    elif imageio is not None:
        writer_imageio = imageio.get_writer(output_path, fps=float(fps))
    else:
        raise RuntimeError("opencv 和 imageio 不可用，无法写出 mp4")
    verts0 = cloth_data[0]
    vmin = np.min(verts0, axis=0)
    vmax = np.max(verts0, axis=0)
    center = (vmin + vmax) * 0.5
    extent = vmax - vmin
    radius = float(np.linalg.norm(extent))
    if radius < 1e-8:
        radius = 1.0
    cam_pos = center + np.array([0.6 * radius, 0.0 * radius, 1.2 * radius], dtype=np.float64)
    ps.look_at(cam_pos, center)

    t = 0
    max_frames = int(N)
    is_recording = True

    def callback():
        nonlocal t, is_recording

        if t >= max_frames:
            if is_recording:
                is_recording = False
                if writer_cv2 is not None:
                    writer_cv2.release()
                if writer_imageio is not None:
                    writer_imageio.close()
                print(f"\nDone! Video saved as:\n{output_path}")
                if args.show:
                    ps.set_user_callback(None)
            return

        verts = cloth_data[t]
        if hasattr(mesh, "update_vertex_positions"):
            mesh.update_vertex_positions(verts)
        else:
            mesh.update_point_positions(verts)

        screenshot = ps.screenshot_to_buffer(transparent_bg=False)
        if screenshot is None:
            t += 1
            return

        if screenshot.ndim == 3 and screenshot.shape[-1] >= 3:
            rgb = screenshot[..., :3]
        else:
            rgb = screenshot

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)

        h, w = rgb.shape[:2]
        if (w, h) != frame_size:
            if cv2 is not None:
                rgb = cv2.resize(rgb, frame_size)

        if writer_cv2 is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer_cv2.write(bgr)
        elif writer_imageio is not None:
            writer_imageio.append_data(rgb)

        print(f"Processing frame: {t+1} / {max_frames}", end="\r")
        t += 1

    if args.show:
        ps.set_user_callback(callback)
        ps.show()
    else:
        while is_recording:
            callback()

if __name__ == "__main__":
    main()
