"""Script 8: render rollout results from Script 7.

Inputs:
    rollouts/motion_XXX/reference_len_*.pt
    rollouts/motion_XXX/<solver>/rollout.pt

Outputs:
    renders/rollouts/motion_XXX/<solver>/final_frame.png
    renders/rollouts/motion_XXX/<solver>/metrics.png
    optional frame images and mp4 video

Run:
    python cloth08_render_rollouts.py --root cloth_5x5_500step_pipeline --motion-index 3 --save-frames --make-video
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


def find_reference_path(motion_dir: Path) -> Path:
    candidates = sorted(motion_dir.glob("reference_len_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no reference_len_*.pt under {motion_dir}")
    return candidates[-1]


def set_equal_axes(ax, points: torch.Tensor) -> None:
    arr = points.reshape(-1, 3).cpu().numpy()
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins) + 1e-12)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_cloth(ax, positions: torch.Tensor, *, label: str, linewidth: float = 1.0, alpha: float = 1.0) -> None:
    p = positions.cpu()
    for i, j in SPRING_EDGES:
        ax.plot(
            [p[i, 0], p[j, 0]],
            [p[i, 1], p[j, 1]],
            [p[i, 2], p[j, 2]],
            linewidth=linewidth,
            alpha=alpha,
        )
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=12, alpha=alpha, label=label)


def plot_metrics(rollout: dict[str, Any], out_path: Path) -> None:
    residual = np.asarray(rollout.get("residual_by_step", []), dtype=float)
    error = np.asarray(rollout.get("reference_error_by_step", []), dtype=float)
    if residual.size == 0 and error.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    if residual.size:
        ax.plot(np.arange(1, residual.size + 1), np.maximum(residual, 1e-16), label="residual")
    if error.size:
        ax.plot(np.arange(1, error.size + 1), np.maximum(error, 1e-16), label="reference error")
    ax.set_yscale("log")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("value")
    ax.set_title("rollout metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_frame(predicted: torch.Tensor, reference: torch.Tensor, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    draw_cloth(ax, reference, label="reference", linewidth=0.8, alpha=0.55)
    draw_cloth(ax, predicted, label="prediction", linewidth=1.0, alpha=0.95)
    set_equal_axes(ax, torch.cat([predicted.reshape(-1, 3), reference.reshape(-1, 3)], dim=0))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-60)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_solver(
    *,
    solver_dir: Path,
    reference_positions: torch.Tensor,
    output_dir: Path,
    frame_stride: int,
    save_frames: bool,
    make_video: bool,
) -> None:
    rollout = torch.load(solver_dir / "rollout.pt", map_location="cpu")
    predicted = rollout["positions"]
    steps = min(predicted.shape[0], reference_positions.shape[0]) - 1
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metrics(rollout, output_dir / "metrics.png")
    render_frame(
        predicted[steps],
        reference_positions[steps],
        f"{solver_dir.name}: final frame {steps}",
        output_dir / "final_frame.png",
    )

    frame_dir = output_dir / "frames"
    if save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)
        for frame in range(0, steps + 1, frame_stride):
            render_frame(
                predicted[frame],
                reference_positions[frame],
                f"{solver_dir.name}: frame {frame}",
                frame_dir / f"frame_{frame:04d}.png",
            )

    if make_video:
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found; skip mp4 video")
            return
        video_path = output_dir / "rollout.mp4"
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        writer = FFMpegWriter(fps=24)
        with writer.saving(fig, video_path, dpi=160):
            for frame in range(0, steps + 1, frame_stride):
                ax.clear()
                draw_cloth(ax, reference_positions[frame], label="reference", linewidth=0.8, alpha=0.55)
                draw_cloth(ax, predicted[frame], label="prediction", linewidth=1.0, alpha=0.95)
                set_equal_axes(ax, torch.cat([predicted.reshape(-1, 3), reference_positions.reshape(-1, 3)], dim=0))
                ax.set_title(f"{solver_dir.name}: frame {frame}")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.view_init(elev=22, azim=-60)
                writer.grab_frame()
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render rollout outputs.")
    parser.add_argument("--root", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--motion-index", type=int, required=True)
    parser.add_argument("--solver-names", nargs="*", default=[])
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--make-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_dir = args.root / "rollouts" / f"motion_{args.motion_index:03d}"
    reference = torch.load(find_reference_path(motion_dir), map_location="cpu")
    reference_positions = reference["positions"]

    solver_dirs = [p for p in sorted(motion_dir.iterdir()) if p.is_dir() and (p / "rollout.pt").exists()]
    if args.solver_names:
        name_set = set(args.solver_names)
        solver_dirs = [p for p in solver_dirs if p.name in name_set]

    render_root = args.root / "renders" / "rollouts" / f"motion_{args.motion_index:03d}"
    for solver_dir in solver_dirs:
        print(f"render {solver_dir.name}")
        render_solver(
            solver_dir=solver_dir,
            reference_positions=reference_positions,
            output_dir=render_root / solver_dir.name,
            frame_stride=max(1, args.frame_stride),
            save_frames=args.save_frames,
            make_video=args.make_video,
        )
    print(f"saved rollout renders to {render_root}")


if __name__ == "__main__":
    main()
