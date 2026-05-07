import os
import re
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(path):
    arr = np.load(path)
    if arr.ndim == 2 and arr.shape[1] == 3:
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected shape [N, M, 3]")
    return arr


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def parse_select(s, N):
    if s is None:
        return list(range(N))
    t = s.strip()
    t = t.strip("[]() ")
    if t.lower() in ("all", "*"):
        return list(range(N))
    if re.fullmatch(r"\d+", t):
        i = int(t)
        i = clamp(i, 0, N - 1)
        return [i]
    parts = re.split(r"[-:,]", t)
    parts = [p for p in parts if p.strip() != ""]
    if len(parts) == 2 and all(re.fullmatch(r"\d+", p.strip()) for p in parts):
        a = clamp(int(parts[0].strip()), 0, N - 1)
        b = clamp(int(parts[1].strip()), 0, N - 1)
        if a <= b:
            return list(range(a, b + 1))
        else:
            return list(range(b, a + 1))
    nums = re.findall(r"\d+", t)
    if len(nums) == 1:
        i = clamp(int(nums[0]), 0, N - 1)
        return [i]
    if len(nums) >= 2:
        a = clamp(int(nums[0]), 0, N - 1)
        b = clamp(int(nums[1]), 0, N - 1)
        if a <= b:
            return list(range(a, b + 1))
        else:
            return list(range(b, a + 1))
    raise ValueError("invalid select pattern")


def build_grid_triangles(nm):
    n = int(round(math.sqrt(nm)))
    if n * n != nm:
        return None
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = v0 + 1
            v2 = v0 + n
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    return np.array(faces, dtype=np.int32)

def load_triangles_from_data(data_path, nm):
    base = os.path.splitext(os.path.basename(data_path))[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    if base.startswith("cloth_data_"):
        suffix = base[len("cloth_data_"):]
        topy_path = os.path.join(input_dir, f"cloth_topy_{suffix}.npy")
        if os.path.exists(topy_path):
            return np.load(topy_path)
    return build_grid_triangles(nm)


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
    if re.fullmatch(r"\d+", s):
        i = int(s)
        i = clamp(i, 0, len(files) - 1)
        return os.path.join(input_dir, files[i])
    return os.path.join(input_dir, files[0])


def compute_center_radius(data_subset):
    flat = data_subset.reshape(-1, 3)
    mn = np.min(flat, axis=0)
    mx = np.max(flat, axis=0)
    center = (mn + mx) / 2.0
    extent = mx - mn
    radius = float(np.linalg.norm(extent))
    if radius < 1e-6:
        radius = 1.0
    return center, radius

def compute_bounds(data_subset):
    flat = data_subset.reshape(-1, 3)
    mn = np.min(flat, axis=0)
    mx = np.max(flat, axis=0)
    size = mx - mn
    pad = 0.05 * size
    pad = np.where(pad < 1e-6, 0.01, pad)
    return mn - pad, mx + pad


def render_image_polyscope(data, idx, out_dir, base_name, width, height, tris, show=False):
    os.makedirs(out_dir, exist_ok=True)
    try:
        import polyscope as ps
    except Exception:
        v0 = data[idx]
        mn, mx = compute_bounds(v0[None, ...])
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(v0[:, 0], v0[:, 1], v0[:, 2], s=2, c="k")
        ax.set_xlim(float(mn[0]), float(mx[0]))
        ax.set_ylim(float(mn[1]), float(mx[1]))
        ax.set_zlim(float(mn[2]), float(mx[2]))
        ax.view_init(elev=20, azim=45)
        out_path = os.path.join(out_dir, f"{base_name}_{idx:06d}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(out_path)
        return out_path
    import polyscope as ps
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    v0 = data[idx]
    if tris is not None:
        mesh = ps.register_surface_mesh("cloth", v0, tris, color=(0.5, 0.7, 1.0), smooth_shade=True)
    else:
        mesh = ps.register_point_cloud("cloth_points", v0)
        try:
            mesh.set_radius(0.02, relative=True)
        except Exception:
            pass
    center, radius = compute_center_radius(data[[idx]])
    R = max(1.5, 1.2 * radius)
    views = [
        (0.0, 10.0),
        (90.0, 10.0),
        (180.0, 10.0),
        (270.0, 10.0),
        (150.0, 10.0),
        (0.0, 60.0),
    ]
    tmp_files = []
    for az_deg, el_deg in views:
        az = math.radians(az_deg)
        el = math.radians(el_deg)
        x = R * math.cos(el) * math.cos(az)
        y = R * math.cos(el) * math.sin(az)
        z = R * math.sin(el)
        cam_pos = center + np.array([x, y, z])
        ps.look_at(cam_pos, center.tolist())
        tmp_path = os.path.join(out_dir, f"tmp_ps_{int(az_deg)}_{int(el_deg)}.png")
        ps.screenshot(tmp_path)
        tmp_files.append(tmp_path)
    out_path = os.path.join(out_dir, f"{base_name}_{idx:06d}.png")
    try:
        import cv2
        imgs = []
        for p in tmp_files:
            im = cv2.imread(p)
            if im is None:
                continue
            im = cv2.resize(im, (int(width), int(height)))
            imgs.append(im)
        while len(imgs) < 6:
            imgs.append(np.zeros((int(height), int(width), 3), dtype=np.uint8))
        row1 = np.hstack(imgs[0:3])
        row2 = np.hstack(imgs[3:6])
        mosaic = np.vstack([row1, row2])
        cv2.imwrite(out_path, mosaic)
    except Exception:
        try:
            import imageio
            imgs = []
            for p in tmp_files:
                im = imageio.imread(p)
                if im is None:
                    continue
                # resize via numpy nearest if needed
                h, w = im.shape[:2]
                if (w, h) != (int(width), int(height)):
                    try:
                        import PIL.Image as PILImage
                        im = np.array(PILImage.fromarray(im).resize((int(width), int(height))))
                    except Exception:
                        im = im[: int(height), : int(width)]
                imgs.append(im)
            while len(imgs) < 6:
                imgs.append(np.zeros((int(height), int(width), 3), dtype=np.uint8))
            row1 = np.hstack(imgs[0:3])
            row2 = np.hstack(imgs[3:6])
            mosaic = np.vstack([row1, row2])
            imageio.imwrite(out_path, mosaic)
        except Exception:
            out_path = tmp_files[0]
    for p in tmp_files:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    print(out_path)
    if show:
        try:
            import polyscope as ps
            ps.show()
        except Exception:
            pass
    return out_path


def render_video_polyscope(data, indices, out_dir, base_name, fps, width, height, tris, show=False, mosaic_last=False):
    os.makedirs(out_dir, exist_ok=True)
    try:
        import polyscope as ps
    except Exception:
        return render_video_matplot(data, indices, out_dir, base_name, fps, width, height)
    try:
        import cv2
    except Exception:
        cv2 = None
    try:
        import imageio
    except Exception:
        imageio = None
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    v0 = data[indices[0]]
    if tris is not None:
        mesh = ps.register_surface_mesh("cloth", v0, tris, color=(0.5, 0.7, 1.0), smooth_shade=True)
    else:
        mesh = ps.register_point_cloud("cloth_points", v0)
        try:
            mesh.set_radius(0.02, relative=True)
        except Exception:
            pass
    subset = data[indices]
    center, radius = compute_center_radius(subset)
    R = max(1.5, 1.2 * radius)
    az = math.radians(150.0)
    el = math.radians(10.0)
    x = R * math.cos(el) * math.cos(az)
    y = R * math.cos(el) * math.sin(az)
    z = R * math.sin(el)
    cam_pos = center + np.array([x, y, z])
    ps.look_at(cam_pos, center.tolist())
    tmp_path = os.path.join(out_dir, "frame_tmp.png")
    out_path = os.path.join(out_dir, f"{base_name}_frames_{indices[0]:06d}_{indices[-1]:06d}.mp4")
    writer_cv2 = None
    writer_imageio = None
    if cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_cv2 = cv2.VideoWriter(out_path, fourcc, float(fps), (int(width), int(height)))
    elif imageio is not None:
        writer_imageio = imageio.get_writer(out_path, fps=float(fps))
    else:
        raise RuntimeError("opencv 或 imageio 不可用，无法写出 mp4")
    for idx in indices:
        verts = data[idx]
        if hasattr(mesh, "update_vertex_positions"):
            mesh.update_vertex_positions(verts)
        else:
            mesh.update_point_positions(verts)
        ps.screenshot(tmp_path)
        if writer_cv2 is not None:
            img = cv2.imread(tmp_path)
            if img is not None:
                h, w = img.shape[:2]
                if (w, h) != (int(width), int(height)):
                    img = cv2.resize(img, (int(width), int(height)))
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
    if mosaic_last:
        try:
            render_image_polyscope(data, indices[-1], out_dir, base_name + "_last_mosaic", width, height, tris, show=False)
        except Exception:
            pass
    if show:
        try:
            import polyscope as ps
            if hasattr(mesh, "update_vertex_positions"):
                mesh.update_vertex_positions(data[indices[-1]])
            else:
                mesh.update_point_positions(data[indices[-1]])
            ps.show()
        except Exception:
            pass
    print(f"video: {out_path}")
    return out_path

def render_video_matplot(data, indices, out_dir, base_name, fps, width, height):
    try:
        import cv2
    except Exception:
        cv2 = None
    try:
        import imageio
    except Exception:
        imageio = None
    subset = data[indices]
    mn, mx = compute_bounds(subset)
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    tmp_path = os.path.join(out_dir, "frame_tmp.png")
    out_path = os.path.join(out_dir, f"{base_name}_frames_{indices[0]:06d}_{indices[-1]:06d}.mp4")
    writer_cv2 = None
    writer_imageio = None
    if cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_cv2 = cv2.VideoWriter(out_path, fourcc, float(fps), (int(width), int(height)))
    elif imageio is not None:
        writer_imageio = imageio.get_writer(out_path, fps=float(fps))
    else:
        raise RuntimeError("opencv 或 imageio 不可用，无法写出 mp4")
    for idx in indices:
        ax.cla()
        P = data[idx]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, c="k")
        ax.set_xlim(float(mn[0]), float(mx[0]))
        ax.set_ylim(float(mn[1]), float(mx[1]))
        ax.set_zlim(float(mn[2]), float(mx[2]))
        ax.view_init(elev=20, azim=45)
        fig.tight_layout()
        fig.savefig(tmp_path)
        if writer_cv2 is not None:
            img = cv2.imread(tmp_path)
            if img is not None:
                h, w = img.shape[:2]
                if (w, h) != (int(width), int(height)):
                    img = cv2.resize(img, (int(width), int(height)))
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
    plt.close(fig)
    print(f"video: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    default_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(default_dir, "input")
    files = list_input_files(input_dir)
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--choose", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--select", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["video", "image"], default="video")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--mosaic_last", action="store_true")
    args = parser.parse_args()
    if args.list:
        for i, f in enumerate(files):
            print(f"[{i}] {f}")
        return
    data_file = args.file
    if data_file is None or args.choose:
        if not files:
            raise FileNotFoundError("input 目录无可用 cloth_data_*.npy")
        data_file = choose_file_interactive(files, input_dir)
    data = load_data(data_file)
    N, M, _ = data.shape
    print(f"file: {os.path.abspath(data_file)}")
    print(f"shape: N={N}, M={M}, D=3")
    if args.info:
        return

    sel_str = args.select
    if sel_str is None:
        print("选择帧范围，如 [0,100] 或 [10,15] 或 [123]")
        sel_str = input().strip()
    indices = parse_select(sel_str, N)
    out_dir = os.path.join(default_dir, "output")
    base_name = os.path.splitext(os.path.basename(data_file))[0]
    tris = load_triangles_from_data(data_file, M)

    if args.mode == "image":
        indices = indices[:1]
        render_image_polyscope(data, indices[0], out_dir, base_name, args.width, args.height, tris, show=args.show)
    else:
        render_video_polyscope(data, indices, out_dir, base_name, args.fps, args.width, args.height, tris, show=args.show, mosaic_last=args.mosaic_last)


if __name__ == "__main__":
    main()