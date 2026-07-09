"""Script 9: render all reference motions.

Inputs:
    data/reference/reference_motion_states.pt
    data/reference/motion_catalogue.json

Outputs:
    renders/reference_motions/motion_XXX/final_frame.png
    optional frame images and mp4 video

Run:
    python cloth09_render_reference_motions.py --root cloth_5x5_500step_pipeline --motion-indices 0 1 2 --make-video
    python cloth09_render_reference_motions.py --root cloth_5x5_500step_pipeline --all --save-frames
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch

from cloth03_solvers_and_models import SPRING_EDGES


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_equal_axes(ax, points: torch.Tensor) -> None:
    arr = points.reshape(-1, 3).cpu().numpy()
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins) + 1e-12)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_cloth(ax, positions: torch.Tensor) -> None:
    p = positions.cpu()
    for i, j in SPRING_EDGES:
        ax.plot(
            [p[i, 0], p[j, 0]],
            [p[i, 1], p[j, 1]],
            [p[i, 2], p[j, 2]],
            linewidth=0.9,
        )
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=12)


def render_single_frame(positions: torch.Tensor, all_positions: torch.Tensor, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    draw_cloth(ax, positions)
    set_equal_axes(ax, all_positions)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_motion(
    *,
    motion_index: int,
    motion_name: str,
    positions: torch.Tensor,
    output_dir: Path,
    frame_stride: int,
    save_frames: bool,
    make_video: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    last_frame = positions.shape[0] - 1
    render_single_frame(
        positions[last_frame],
        positions,
        f"motion {motion_index:03d}: {motion_name}\nfinal frame {last_frame}",
        output_dir / "final_frame.png",
    )

    if save_frames:
        frame_dir = output_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(0, last_frame + 1, frame_stride):
            render_single_frame(
                positions[frame],
                positions,
                f"motion {motion_index:03d}: {motion_name}\nframe {frame}",
                frame_dir / f"frame_{frame:04d}.png",
            )

    if make_video:
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found; skip mp4 video")
            return
        video_path = output_dir / "reference.mp4"
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        writer = FFMpegWriter(fps=24)
        with writer.saving(fig, video_path, dpi=160):
            for frame in range(0, last_frame + 1, frame_stride):
                ax.clear()
                draw_cloth(ax, positions[frame])
                set_equal_axes(ax, positions)
                ax.set_title(f"motion {motion_index:03d}: {motion_name}\nframe {frame}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.view_init(elev=22, azim=-60)
                writer.grab_frame()
        plt.close(fig)


def make_overview(reference: dict[str, Any], motion_names: dict[int, str], output_dir: Path) -> None:
    positions = reference["positions"]
    motion_ids = reference["motion_index"].tolist()
    fig = plt.figure(figsize=(16, 12))
    for plot_id, motion_index in enumerate(motion_ids, start=1):
        ax = fig.add_subplot(4, 8, plot_id, projection="3d")
        final_positions = positions[plot_id - 1, -1]
        draw_cloth(ax, final_positions)
        set_equal_axes(ax, positions[plot_id - 1])
        ax.set_title(f"{motion_index:02d}\n{motion_names.get(int(motion_index), '')[:18]}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    fig.savefig(output_dir / "overview_final_frames.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render reference motions.")
    parser.add_argument("--root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--motion-indices", type=int, nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--make-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference = torch.load(args.root / "data" / "reference" / "reference_motion_states.pt", map_location="cpu")
    catalogue = load_json(args.root / "data" / "reference" / "motion_catalogue.json")
    motion_names = {int(motion["index"]): motion["name"] for motion in catalogue["motions"]}
    motion_ids = [int(x) for x in reference["motion_index"].tolist()]

    selected = motion_ids if args.all or not args.motion_indices else args.motion_indices
    output_root = args.root / "renders" / "reference_motions"
    output_root.mkdir(parents=True, exist_ok=True)
    make_overview(reference, motion_names, output_root)

    for motion_index in selected:
        row = motion_ids.index(int(motion_index))
        motion_name = motion_names.get(int(motion_index), f"motion_{motion_index:03d}")
        out = output_root / f"motion_{motion_index:03d}"
        print(f"render reference motion {motion_index:03d}: {motion_name}")
        render_motion(
            motion_index=int(motion_index),
            motion_name=motion_name,
            positions=reference["positions"][row],
            output_dir=out,
            frame_stride=max(1, args.frame_stride),
            save_frames=args.save_frames,
            make_video=args.make_video,
        )
    print(f"saved reference renders to {output_root}")


if __name__ == "__main__":
    main()
