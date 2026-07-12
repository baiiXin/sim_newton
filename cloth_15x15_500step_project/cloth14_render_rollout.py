"""Render stored continuous rollouts from cloth07_rollout_hardest_motion.py."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def default_output_path(rollout: Path, format_name: str) -> Path:
    return rollout.with_suffix(f".{format_name}")


def rollout_files(scan_root: Path) -> list[Path]:
    return sorted(path for path in scan_root.glob("**/curve.pt") if path.is_file())


def load_reference(root: Path, motion_index: int, length: int, stride: int) -> np.ndarray:
    states = torch.load(
        root / "data" / "reference" / "reference_motion_states.pt",
        map_location="cpu",
    )
    ids = [int(value) for value in states["motion_index"].tolist()]
    if motion_index not in ids:
        return np.empty((0, 0, 3), dtype=np.float64)
    row = ids.index(motion_index)
    return states["positions"][row, :length][::stride].numpy()


def render_one(
    *,
    root: Path,
    rollout: Path,
    output: Path,
    stride: int,
    fps: int,
    format_name: str,
    no_reference: bool,
) -> Path:
    if stride <= 0:
        raise ValueError("--stride must be positive")

    payload: dict[str, Any] = torch.load(rollout, map_location="cpu")
    runtime = load_json(root / "data" / "reference" / "runtime_config.json")
    edges = [tuple(edge) for edge in runtime["spring_edges"]]
    fixed = runtime["fixed_vertex_indices"]

    motion_index = int(payload["motion_index"])
    positions = payload["positions"][::stride].numpy()
    residual = payload.get("residual_by_frame_and_iteration", torch.empty(0))
    error = payload.get("reference_error_by_frame", torch.empty(0))
    residual = residual.numpy() if torch.is_tensor(residual) else np.asarray(residual)
    error = error.numpy() if torch.is_tensor(error) else np.asarray(error)

    reference = np.empty((0, 0, 3), dtype=positions.dtype)
    if not no_reference:
        reference = load_reference(root, motion_index, len(payload["positions"]), stride)

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

    label = str(payload.get("solver_info", {}).get("solver", payload.get("solver", "rollout")))

    def update(frame: int):
        original_frame = frame * stride
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
        ax.set_title(
            f"{label}: motion {motion_index:03d} rollout frame {original_frame:03d}\n{title_tail}"
        )
        return [*ref_lines, *rollout_lines, ref_points, rollout_points, pins]

    animation = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=1000 / fps,
        blit=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps) if format_name == "mp4" else PillowWriter(fps=fps)
    animation.save(output, writer=writer, dpi=140)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render stored 15x15 rollouts.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("cloth_15x15_500step_pipeline"),
        help="pipeline root, used for spring edges and optional reference overlay",
    )
    parser.add_argument(
        "--rollout",
        type=Path,
        default=None,
        help="path to one curve.pt; omit to scan --scan-root",
    )
    parser.add_argument(
        "--scan-root",
        type=Path,
        default=None,
        help="directory scanned for **/curve.pt when --rollout is omitted",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rollout is not None:
        output = args.output or default_output_path(args.rollout, args.format)
        if output.exists() and not args.overwrite:
            print(f"skip existing render {output}")
            return
        print(
            render_one(
                root=args.root,
                rollout=args.rollout,
                output=output,
                stride=args.stride,
                fps=args.fps,
                format_name=args.format,
                no_reference=args.no_reference,
            )
        )
        return

    if args.output is not None:
        raise ValueError("--output is only valid with --rollout")
    scan_root = args.scan_root or (args.root / "rollouts")
    files = rollout_files(scan_root)
    rendered = 0
    skipped = 0
    for rollout in files:
        output = default_output_path(rollout, args.format)
        if output.exists() and not args.overwrite:
            skipped += 1
            print(f"skip existing render {output}")
            continue
        print(
            render_one(
                root=args.root,
                rollout=rollout,
                output=output,
                stride=args.stride,
                fps=args.fps,
                format_name=args.format,
                no_reference=args.no_reference,
            )
        )
        rendered += 1
    print(f"rendered={rendered} skipped={skipped} scanned={len(files)}")


if __name__ == "__main__":
    main()
