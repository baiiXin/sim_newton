"""Interactive T-shirt inference with configurable dynamics and Polyscope MP4 output."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch

from cloth02_batched_physics import FrozenMotionBatch, load_physics
from cloth05_train_online import load_model_checkpoint
from cloth09_rollout_single_motion import (
    SingleMotionSettings,
    run_solver_rollout,
    save_solver_rollout,
)
from tshirt_config import (
    DEFAULT_DYNAMICS,
    DEFAULT_FIXED_DATA_DIR,
    DEFAULT_TRAIN_SEED,
    PROJECT_DIR,
    write_json,
)
from tshirt_mesh import load_tshirt_mesh
from tshirt_sampling import build_inference_motion


DEFAULT_ROOT = Path("cloth_tshirt_pipeline")


@dataclass(frozen=True)
class SolverChoice:
    kind: str
    label: str
    checkpoint: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fixed-data-dir", type=Path, default=DEFAULT_FIXED_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument(
        "--solver",
        choices=("interactive", "block3x3", "mass", "fixed", "network"),
        default="interactive",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--include-periodic", action="store_true")
    parser.add_argument("--list-models", action="store_true")

    dynamics = parser.add_argument_group("initial dynamics")
    dynamics.add_argument("--pose", choices=("horizontal", "vertical", "random"), default="horizontal")
    dynamics.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    dynamics.add_argument(
        "--translation-velocity", type=float, nargs=3, metavar=("VX", "VY", "VZ"),
        default=(0.0, 0.0, 0.0),
    )
    dynamics.add_argument(
        "--angular-velocity", type=float, nargs=3, metavar=("WX", "WY", "WZ"),
        default=(0.0, 0.0, 0.0),
    )
    dynamics.add_argument("--smooth-velocity-rms", type=float, default=0.0)
    dynamics.add_argument("--high-frequency-velocity-rms", type=float, default=0.0)
    dynamics.add_argument("--position-perturb-rms-edge-fraction", type=float, default=0.0)
    dynamics.add_argument("--velocity-clip", type=float, default=12.0)

    solve = parser.add_argument_group("implicit solve")
    solve.add_argument("--inner-steps", type=int, default=50)
    solve.add_argument("--convergence-ratio", type=float, default=1e-3)
    solve.add_argument("--absolute-residual-tolerance", type=float, default=1e-10)
    solve.add_argument(
        "--fixed-inner-steps",
        action="store_true",
        help="ignore the convergence threshold and always use --inner-steps",
    )
    solve.add_argument("--rollout-frames", type=int, default=500)
    solve.add_argument("--fixed-gd-step-size", type=float, default=5e-5)
    solve.add_argument("--mass-ls-step-size", type=float, default=1.0)
    solve.add_argument("--block-ls-step-size", type=float, default=1.0)
    solve.add_argument("--line-search-max-trials", type=int, default=12)

    render = parser.add_argument_group("Polyscope MP4 rendering")
    render.add_argument("--fps", type=int, default=30)
    render.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="auto-detected from DISPLAY by default; headless Linux uses Polyscope EGL",
    )
    render.add_argument("--egl-device-index", type=int, default=-1)
    render.add_argument("--video-crf", type=int, default=18)
    return parser.parse_args()


def _seed_dir(checkpoint: Path) -> Path:
    return checkpoint.parent.parent if checkpoint.parent.name == "periodic" else checkpoint.parent


def _checkpoint_label(checkpoint: Path) -> str:
    seed_dir = _seed_dir(checkpoint)
    experiment = seed_dir.parent.name
    if checkpoint.name == "best_validation_model.pt":
        kind = "best"
    elif checkpoint.name == "latest_checkpoint.pt":
        kind = "latest"
    else:
        kind = checkpoint.stem
    return f"network: {experiment}/{seed_dir.name}/{kind}"


def scan_trained_models(root: Path, include_periodic: bool) -> list[SolverChoice]:
    root = Path(root)
    paths = list(root.glob("**/best_validation_model.pt"))
    paths.extend(root.glob("**/latest_checkpoint.pt"))
    if include_periodic:
        paths.extend(root.glob("**/periodic/checkpoint_update_*.pt"))
    return [
        SolverChoice("network", _checkpoint_label(path), path.resolve())
        for path in sorted(set(path.resolve() for path in paths))
    ]


def available_choices(root: Path, include_periodic: bool) -> list[SolverChoice]:
    return [
        SolverChoice("block3x3", "GD: 3x3 Hessian-block preconditioned + line search [default]"),
        SolverChoice("mass", "GD: mass preconditioned + line search"),
        SolverChoice("fixed", "GD: raw fixed step"),
        *scan_trained_models(root, include_periodic),
    ]


def choose_solver(args: argparse.Namespace) -> SolverChoice:
    if args.checkpoint is not None:
        return SolverChoice("network", _checkpoint_label(args.checkpoint), args.checkpoint.resolve())
    choices = available_choices(args.root, args.include_periodic)
    if args.list_models:
        for index, choice in enumerate(choices):
            suffix = f" ({choice.checkpoint})" if choice.checkpoint else ""
            print(f"[{index}] {choice.label}{suffix}")
        raise SystemExit(0)
    if args.solver != "interactive":
        if args.solver == "network":
            networks = [choice for choice in choices if choice.kind == "network"]
            if not networks:
                raise FileNotFoundError(f"no trained checkpoints found under {Path(args.root).resolve()}")
            if not sys.stdin.isatty():
                return networks[0]
            choices = networks
        else:
            return next(choice for choice in choices if choice.kind == args.solver)
    if not sys.stdin.isatty():
        return choices[0]
    print("Available inference solvers/models:")
    for index, choice in enumerate(choices):
        print(f"  [{index}] {choice.label}")
    raw = input("Select solver/model [0]: ").strip()
    index = 0 if not raw else int(raw)
    if index < 0 or index >= len(choices):
        raise ValueError(f"selection must be in [0, {len(choices) - 1}]")
    return choices[index]


def _solver_name(choice: SolverChoice) -> str:
    return {
        "block3x3": "gd_block3x3_ls",
        "mass": "gd_mass_ls",
        "fixed": "gd_fixed",
        "network": "network",
    }[choice.kind]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _resolve_mesh_path(fixed_data_dir: Path, mesh_path: str) -> Path:
    configured = Path(mesh_path)
    candidates = (
        configured,
        Path(fixed_data_dir).resolve().parent / configured,
        PROJECT_DIR / configured,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not resolve fixed-model OBJ; tried: {attempted}")


def _write_residual_csv(path: Path, curves: dict[str, np.ndarray]) -> None:
    fields = ["frame", *curves]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        length = len(next(iter(curves.values())))
        for frame in range(length):
            row: dict[str, Any] = {"frame": frame}
            for key, values in curves.items():
                value = values[frame]
                row[key] = value.item() if hasattr(value, "item") else value
            writer.writerow(row)


def plot_residual_vs_frames(curves: dict[str, np.ndarray], output: Path, threshold: float) -> None:
    import matplotlib.pyplot as plt

    frames = np.arange(len(curves["initial_residual"]))
    valid = np.isfinite(curves["initial_residual"]) & np.isfinite(curves["final_residual"])
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].semilogy(frames[valid], curves["initial_residual"][valid], label="initial residual")
    axes[0].semilogy(frames[valid], curves["final_residual"][valid], label="final residual")
    axes[0].set(ylabel="residual L2 norm", title="Implicit-solve residual vs. physical frame")
    ratio_valid = valid & (curves["residual_ratio"] > 0.0)
    axes[1].semilogy(frames[ratio_valid], curves["residual_ratio"][ratio_valid], label="final / initial")
    axes[1].axhline(threshold, color="black", ls="--", lw=1, label=f"threshold={threshold:g}")
    axes[1].set(xlabel="physical frame", ylabel="residual ratio")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def render_polyscope_mp4(
    *,
    positions: np.ndarray,
    faces: np.ndarray,
    fixed_indices: tuple[int, ...],
    output: Path,
    fps: int,
    headless: bool,
    egl_device_index: int,
    crf: int,
) -> dict[str, Any]:
    try:
        import imageio_ffmpeg
        import polyscope as ps
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Polyscope video rendering requires `pip install polyscope imageio-ffmpeg`"
        ) from error

    ps.set_program_name("T-shirt inference")
    ps.set_use_prefs_file(False)
    ps.set_build_gui(False)
    ps.set_verbosity(0)
    if headless:
        ps.set_egl_device_index(egl_device_index)
        ps.init("openGL3_egl")
    else:
        ps.set_allow_headless_backends(True)
        ps.init()
    mesh = ps.register_surface_mesh(
        "T-shirt",
        positions[0],
        faces,
        color=(0.18, 0.48, 0.82),
        edge_color=(0.08, 0.12, 0.18),
        edge_width=0.35,
        smooth_shade=True,
        material="candy",
    )
    fixed = np.asarray(fixed_indices, dtype=np.int64)
    ps.register_point_cloud(
        "fixed shoulders",
        positions[0, fixed],
        color=(0.90, 0.12, 0.10),
        radius=0.015,
    )
    minimum = np.min(positions, axis=(0, 1))
    maximum = np.max(positions, axis=(0, 1))
    center = 0.5 * (minimum + maximum)
    radius = max(float(np.max(maximum - minimum)), 1e-3)
    camera = center + radius * np.asarray((1.45, 0.75, 1.65))
    ps.look_at(tuple(camera), tuple(center))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tshirt_polyscope_") as directory:
        frames = Path(directory)
        for index, vertex_positions in enumerate(positions):
            mesh.update_vertex_positions(vertex_positions)
            ps.screenshot(
                str(frames / f"frame_{index:06d}.png"),
                transparent_bg=False,
                include_UI=False,
            )
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(frames / "frame_%06d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Polyscope/ffmpeg did not produce a non-empty MP4")
    return {
        "renderer": "Polyscope",
        "headless": headless,
        "egl_device_index": egl_device_index if headless else None,
        "fps": fps,
        "rendered_frame_count": int(positions.shape[0]),
        "ffmpeg": ffmpeg,
        "mp4": str(output.resolve()),
    }


def main() -> None:
    args = parse_args()
    if args.inner_steps <= 0 or args.rollout_frames <= 0 or args.fps <= 0:
        raise ValueError("inner steps, rollout frames, and fps must be positive")
    if not 0.0 < args.convergence_ratio < 1.0:
        raise ValueError("convergence ratio must be in (0, 1)")
    if not 1 <= args.line_search_max_trials <= 12:
        raise ValueError("line-search trials must be in [1, 12]")
    choice = choose_solver(args)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    physics = load_physics(fixed_data_dir=args.fixed_data_dir, device=args.device, dtype=dtype)
    mesh = load_tshirt_mesh(_resolve_mesh_path(args.fixed_data_dir, physics.model.mesh_path))
    if mesh.sha256 != physics.model.mesh_sha256:
        raise ValueError("inference OBJ hash differs from the fixed model")
    state = build_inference_motion(
        mesh,
        physics.model,
        DEFAULT_DYNAMICS,
        seed=args.seed,
        pose=args.pose,
        translation_velocity=args.translation_velocity,
        angular_velocity=args.angular_velocity,
        smooth_velocity_rms=args.smooth_velocity_rms,
        high_frequency_velocity_rms=args.high_frequency_velocity_rms,
        position_perturb_rms_edge_fraction=args.position_perturb_rms_edge_fraction,
        velocity_clip=args.velocity_clip,
    )
    motion = FrozenMotionBatch(
        motion_ids=(state.motion_id,),
        positions=torch.as_tensor(state.positions[None], dtype=dtype, device=args.device),
        velocities=torch.as_tensor(state.velocities[None], dtype=dtype, device=args.device),
        seeds=torch.as_tensor((args.seed,), dtype=torch.long, device=args.device),
    )
    model = None
    checkpoint_metadata: dict[str, Any] | None = None
    if choice.kind == "network":
        if choice.checkpoint is None:
            raise ValueError("network selection has no checkpoint")
        model, _, checkpoint_metadata = load_model_checkpoint(choice.checkpoint, physics=physics)
    settings = SingleMotionSettings(
        rollout_frames=args.rollout_frames,
        inner_steps=args.inner_steps,
        residual_ratio_tolerance=args.convergence_ratio,
        absolute_residual_tolerance=args.absolute_residual_tolerance,
        fixed_gd_step_size=args.fixed_gd_step_size,
        mass_ls_step_size=args.mass_ls_step_size,
        block_ls_step_size=args.block_ls_step_size,
        line_search_max_trials=args.line_search_max_trials,
        trajectory_stride=1,
        early_stop=not args.fixed_inner_steps,
    )
    solver = _solver_name(choice)
    print(f"running inference with {choice.label}")
    result = run_solver_rollout(
        solver=solver,
        physics=physics,
        motion=motion,
        settings=settings,
        model=model,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(args.root) / "inference" /
        f"{timestamp}_{_safe_name(args.pose)}_{_safe_name(choice.kind)}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "initial_state.npz",
        motion_id=np.asarray(state.motion_id),
        positions=state.positions,
        velocities=state.velocities,
        fixed_indices=np.asarray(physics.model.fixed_indices, dtype=np.int64),
    )
    save_solver_rollout(result, output)
    _write_residual_csv(output / "residuals.csv", result.curves)
    plot_residual_vs_frames(
        result.curves,
        output / "residual_vs_frames.png",
        args.convergence_ratio,
    )
    headless = (not bool(os.environ.get("DISPLAY"))) if args.headless is None else args.headless
    render_metadata = render_polyscope_mp4(
        positions=result.trajectory_positions,
        faces=mesh.faces,
        fixed_indices=physics.model.fixed_indices,
        output=output / "motion.mp4",
        fps=args.fps,
        headless=headless,
        egl_device_index=args.egl_device_index,
        crf=args.video_crf,
    )
    write_json(
        output / "inference_manifest.json",
        {
            "completed": True,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "environment": "runs on the machine/container which invokes this script",
            "solver_choice": asdict(choice),
            "checkpoint_update": (
                None if checkpoint_metadata is None else checkpoint_metadata.get("update_count")
            ),
            "dynamics": state.metadata,
            "solve_settings": asdict(settings),
            "summary": result.summary,
            "outputs": {
                "initial_state": str((output / "initial_state.npz").resolve()),
                "curves": str((output / "curves.npz").resolve()),
                "residual_csv": str((output / "residuals.csv").resolve()),
                "residual_plot": str((output / "residual_vs_frames.png").resolve()),
                "trajectory": str((output / "trajectory.npz").resolve()),
                "video": str((output / "motion.mp4").resolve()),
            },
            "render": render_metadata,
        },
    )
    print(f"inference outputs written to {output}")
    print(f"MP4: {output / 'motion.mp4'}")


if __name__ == "__main__":
    main()
