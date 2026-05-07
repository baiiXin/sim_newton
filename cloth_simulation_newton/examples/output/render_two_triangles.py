import os
import math
import numpy as np
import argparse

def _find_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_cloth_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "data")
    files = [f for f in os.listdir(input_dir) if f.startswith("cloth_data_") and f.endswith(".npy")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
    if not files:
        raise FileNotFoundError("no cloth_data_*.npy in input")
    data_path = os.path.join(input_dir, files[0])
    return np.load(data_path)

def find_topy_for_data(data_filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "data")
    suffix = os.path.splitext(os.path.basename(data_filename))[0][len("cloth_data_"):]
    topy_path = os.path.join(input_dir, f"cloth_topy_{suffix}.npy")
    return topy_path if os.path.exists(topy_path) else None

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
        raise FileNotFoundError("no cloth_data_*.npy in data")
    data_file = args.file
    if data_file is None or args.choose:
        data_file = choose_file_interactive(files, input_dir)
    cloth_data = np.load(data_file)
    topy_path = find_topy_for_data(data_file)
    triangles = np.load(topy_path) if (topy_path is not None) else None
    N, M, _ = cloth_data.shape
    idxA = np.array([0, 1, 2], dtype=np.int32)
    idxB = np.array([3, 4, 5], dtype=np.int32)
    import polyscope as ps
    try:
        import cv2
    except Exception:
        cv2 = None
    is_headless = os.environ.get("DISPLAY", "") == ""
    if is_headless:
        ps.set_allow_headless_backends(True)
        ps.init("openGL3_egl")
    else:
        ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    meshA = ps.register_surface_mesh(
        "large_triangle",
        cloth_data[0, idxA],
        np.array([[0, 1, 2]], dtype=np.int32),
        color=(1.0, 0.2, 0.2),
        smooth_shade=False,
    )
    meshB = ps.register_surface_mesh(
        "small_triangle",
        cloth_data[0, idxB],
        np.array([[0, 1, 2]], dtype=np.int32),
        color=(0.2, 0.6, 1.0),
        smooth_shade=False,
    )
    
    center = np.array([3.0, 3.0, 0.0])
    R = 5.0
    cam_pos = center + np.array([R, 0.0, 0.5 * R])
    ps.look_at(cam_pos, center)
    output_dir = os.path.join(script_dir, "video")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "two_triangles.mp4")
    fps = 100
    frame_size = (1280, 720)
    tmp_path = os.path.join(output_dir, "frame_tmp.png")
    try:
        import imageio.v2 as imageio
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
    for frame in range(N):
        vertsA = cloth_data[frame, idxA]
        vertsB = cloth_data[frame, idxB]
        meshA.update_vertex_positions(vertsA)
        meshB.update_vertex_positions(vertsB)
        ps.look_at(cam_pos, center.tolist())
        ps.screenshot(tmp_path)
        if writer_cv2 is not None:
            img = cv2.imread(tmp_path)
            if img is not None:
                h, w = img.shape[:2]
                if (w, h) != frame_size:
                    img = cv2.resize(img, frame_size)
                writer_cv2.write(img)
        elif writer_imageio is not None:
            im = imageio.imread(tmp_path)
            writer_imageio.append_data(im)
    if writer_cv2 is not None:
        writer_cv2.release()
    if writer_imageio is not None:
        writer_imageio.close()
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass
    if args.show:
        try:
            if hasattr(ps, "is_headless") and ps.is_headless():
                return
            ps.show()
        except Exception:
            pass

if __name__ == "__main__":
    main()
