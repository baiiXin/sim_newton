#!/usr/bin/env python3
import os
import math
import numpy as np


def main():
    from generate_cloth_fall_sphere_assets import generate_cloth_fall_sphere_assets

    vertices_all, triangles_all, mesh = generate_cloth_fall_sphere_assets(
        cloth_rects=[(-2.0, 2.0, -2.0, 2.0)],
        h1=0.5,
        h2=0.5,
        z=5.0,
        mass_m=0.001,
        cloth_layers=5,
        z_offset=0.5,
        sphere_center=(0.0, 0.0, 2.0),
        sphere_radius=0.5,
        sphere_target_triangles=200,
        add_ground=True,
        ground_rect=(-6.0, 6.0, -6.0, 6.0),
        ground_z=0.0,
    )

    cloth_keys = [k for k in mesh.keys() if k.startswith("cloth")]
    cloth_keys = sorted(cloth_keys, key=lambda s: int(s[5:]))

    import polyscope as ps
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

    palette = [
        (0.5, 0.7, 1.0),
        (0.7, 0.3, 0.9),
        (0.2, 1.0, 0.8),
        (1.0, 0.6, 0.0),
        (0.6, 1.0, 0.4),
    ]

    for i, key in enumerate(cloth_keys):
        info = mesh[key]
        verts = info["vertices"]
        tris = info["triangles"]
        ps.register_surface_mesh(
            key,
            verts,
            tris,
            color=palette[i % len(palette)],
            smooth_shade=True,
        )

    sphere_info = mesh["sphere"]
    ground_info = mesh["ground"]

    ps.register_surface_mesh(
        "sphere",
        sphere_info["vertices"],
        sphere_info["triangles"],
        color=(1.0, 0.4, 0.4),
        smooth_shade=True,
    )
    ps.register_surface_mesh(
        "ground",
        ground_info["vertices"],
        ground_info["triangles"],
        color=(0.5, 0.5, 0.5),
        smooth_shade=False,
    )

    fixed_index = mesh.get("fixed_index", None)
    if fixed_index is not None:
        verts_fixed = vertices_all[fixed_index]
        pc = ps.register_point_cloud(
            "fixed_points",
            verts_fixed,
        )
        try:
            pc.set_radius(0.02, relative=True)
            pc.set_color((1.0, 1.0, 0.0))
        except Exception:
            pass

    all_min = np.min(vertices_all, axis=0)
    all_max = np.max(vertices_all, axis=0)
    center = (all_min + all_max) / 2.0
    extent = all_max - all_min
    radius = float(np.linalg.norm(extent))
    if radius < 1e-6:
        radius = 1.0
    R = 1.2 * radius
    height = 0.4 * radius

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "cloth_fall_sphere_polyscope.mp4")
    tmp_path = os.path.join(script_dir, "frame_tmp.png")
    fps = 60
    frame_size = (1280, 720)
    n_frames = 240

    writer_cv2 = None
    writer_imageio = None
    if cv2 is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_cv2 = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    elif imageio is not None:
        writer_imageio = imageio.get_writer(out_path, fps=float(fps))

    angles = np.linspace(0.0, 2.0 * math.pi, num=n_frames, endpoint=False)
    for theta in angles:
        x = center[0] + R * math.cos(theta)
        y = center[1] + R * math.sin(theta)
        z = center[2] + height
        cam_pos = np.array([x, y, z])
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
            if os.path.exists(tmp_path):
                im = imageio.imread(tmp_path)
                writer_imageio.append_data(im)

    if writer_cv2 is not None:
        writer_cv2.release()
    if writer_imageio is not None:
        writer_imageio.close()
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()

