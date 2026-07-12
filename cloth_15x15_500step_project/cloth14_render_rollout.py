"""Render a stored continuous rollout from cloth07_rollout_hardest_motion.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
import numpy as np
import torch

from cloth_common import load_json


def axis_box(*arrays: np.ndarray) -> tuple[np.ndarray, float]:
    valid = [array.reshape(-1, 3) for array in arrays if array.size]
    stacked = np.concatenate(valid, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) * 0.58, 1e-3)
    return center, radius


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a stored 15x15 rollout.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("cloth_15x15_500step_pipeline"),
        help="pipeline root, used for spring edges and optional reference overlay",
    )
    parser.add_argument(
        "--rollout",
        type=Path,
        required=True,
        help="path to curve.pt produced by cloth07_rollout_hardest_motion.py",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-reference", action="store_true")
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive")

    payload = torch.load(args.rollout, map_location="cpu")
    runtime = load_json(args.root / "data" / "reference" / "runtime_config.json")
    edges = [tuple(edge) for edge in runtime["spring_edges"]]
    fixed = runtime["fixed_vertex_indices"]

    motion_index = int(payload["motion_index"])
    positions = payload["positions"][:: args.stride].numpy()
    residual = payload.get("residual_by_frame_and_iteration", torch.empty(0))
    error = payload.get("reference_error_by_frame", torch.empty(0))
    residual = residual.numpy() if torch.is_tensor(residual) else np.asarray(residual)
    error = error.numpy() if torch.is_tensor(error) else np.asarray(error)

    reference = np.empty((0, 0, 3), dtype=positions.dtype)
    if not args.no_reference:
        states = torch.load(
            args.root / "data" / "reference" / "reference_motion_states.pt",
            map_location="cpu",
        )
        ids = [int(value) for value in states["motion_index"].tolist()]
        if motion_index in ids:
            row = ids.index(motion_index)
            reference = states["positions"][row, : len(payload["positions"])][
                :: args.stride
            ].numpy()

    center, radius = axis_box(positions, reference)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ref_lines = [
        ax.plot([], [], [], color="0.6", alpha=0.35, linewidth=0.25)[0]
        for _ in edges
    ]
    rollout_lines = [
        ax.plot([], [], [], color="tab:blue", linewidth=0.45)[0]
        for _ in edges
    ]
    ref_points = ax.scatter([], [], [], s=4, color="0.55", alpha=0.35)
    rollout_points = ax.scatter([], [], [], s=6, color="tab:blue")
    pins = ax.scatter([], [], [], s=45, marker="s", color="tab:red")

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-60)

    def update(frame: int):
        original_frame = frame * args.stride
        x = positions[frame]
        if reference.size:
            ref = reference[min(frame, len(reference) - 1)]
            for line, (i, j) in zip(ref_lines, edges):
                line.set_data_3d(
                    [ref[i, 0], ref[j, 0]],
                    [ref[i, 1], ref[j, 1]],
                    [ref[i, 2], ref[j, 2]],
                )
            ref_points._offsets3d = (ref[:, 0], ref[:, 1], ref[:, 2])
        for line, (i, j) in zip(rollout_lines, edges):
            line.set_data_3d(
                [x[i, 0], x[j, 0]],
                [x[i, 1], x[j, 1]],
                [x[i, 2], x[j, 2]],
            )
        rollout_points._offsets3d = (x[:, 0], x[:, 1], x[:, 2])
        pins._offsets3d = (x[fixed, 0], x[fixed, 1], x[fixed, 2])

        if original_frame == 0 or residual.size == 0:
            title_tail = "initial"
        else:
            metric_index = min(original_frame - 1, residual.shape[0] - 1)
            final_residual = float(residual[metric_index, -1])
            frame_error = float(error[metric_index]) if metric_index < len(error) else float("nan")
            title_tail = f"residual={final_residual:.3e} error={frame_error:.3e}"
        ax.set_title(f"motion {motion_index:03d} rollout frame {original_frame:03d}\n{title_tail}")
        return [*ref_lines, *rollout_lines, ref_points, rollout_points, pins]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=1000 / args.fps,
        blit=False,
    )
    output = args.output or args.rollout.with_suffix(f".{args.format}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps) if args.format == "mp4" else PillowWriter(fps=args.fps)
    animation.save(output, writer=writer, dpi=140)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
