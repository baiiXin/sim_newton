from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
import torch


def load_base_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Base training script not found: {script_path}")
    module_name = f"spring_train_module_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuous 100-frame rollout comparison between the trained MLP solver "
            "and the full Newton solver for the two-particle single-spring problem."
        )
    )
    parser.add_argument(
        "--base-script-path",
        type=Path,
        default=Path("/data/zhoucy/sim_newton/unit_test_for_spring/multi_step_spring/two_particle_spring_residual_optimizer_with_newton.py"),
        help="Path to the training/evaluation script that defines the model and physics.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/data/zhoucy/sim_newton/unit_test_for_spring/multi_step_spring/two_particle_spring_residual_optimizer_with_newton"),
        help="Output directory produced by the training script.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="multi_problem",
        choices=["multi_problem", "single_problem_baseline"],
        help="Which trained experiment checkpoint to load.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_validation_model_state_dict.pt",
        help="Checkpoint file name inside project_root/experiment_name/.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path. Overrides project-root / experiment-name / checkpoint-name.",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="extrapolation_test",
        choices=["interpolation_test", "extrapolation_test"],
        help="Which test split to draw the start time-point from.",
    )
    parser.add_argument(
        "--start-problem-index",
        type=int,
        default=None,
        help="Global physical problem index to start from. Must belong to the chosen test split."
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of physical frames to solve continuously after the chosen start state.",
    )
    parser.add_argument(
        "--inner-iterations",
        type=int,
        default=50,
        help="Number of inner solver iterations per physical frame for both MLP and Newton.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Torch device used for the rollout evaluation.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Frame rate of the output animation.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=170,
        help="DPI used when saving the animation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to a same-named directory next to this script.",
    )
    return parser.parse_args()


def ensure_device(module, device: torch.device) -> None:
    if device.type == "cuda":
        module.validate_device(device)


def choose_start_problem(module, split, test_split: str, start_problem_index: int | None) -> int:
    if test_split == "interpolation_test":
        allowed = tuple(split.interpolation_test_indices)
    elif test_split == "extrapolation_test":
        allowed = tuple(split.extrapolation_test_indices)
    else:
        raise ValueError(f"Unsupported test split: {test_split}")

    if not allowed:
        raise RuntimeError(f"No problem indices found in split {test_split}.")

    if start_problem_index is None:
        return int(allowed[0])

    if int(start_problem_index) not in allowed:
        raise ValueError(
            f"start-problem-index={start_problem_index} is not in {test_split}. "
            f"Allowed indices are: {allowed}"
        )
    return int(start_problem_index)


def predictor_from_state(p: torch.Tensor, v: torch.Tensor, physical) -> torch.Tensor:
    gravity = torch.tensor([0.0, 0.0, physical.g], dtype=p.dtype, device=p.device)
    p1 = p[0:3]
    p2 = p[3:6]
    v1 = v[0:3]
    v2 = v[3:6]
    q1 = p1 + physical.dt * v1 - physical.dt**2 * gravity
    q2 = p2 + physical.dt * v2 - physical.dt**2 * gravity
    return torch.cat([q1, q2], dim=0)


def solve_one_frame_mlp(module, model, p: torch.Tensor, v: torch.Tensor, masses: torch.Tensor, physical, inner_iterations: int):
    q = predictor_from_state(p, v, physical)
    y = p.reshape(1, 6).clone()
    q_batch = q.reshape(1, 6)
    masses_batch = masses.reshape(1, 2)

    with torch.no_grad():
        for _ in range(inner_iterations):
            delta = model(y, q_batch, masses_batch, physical=physical)
            y = y + delta
        y_final = y.squeeze(0)
        residual = module.stationarity_residual_norm(
            y.reshape(1, 6),
            q_batch,
            masses_batch,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        ).item()
    v_next = (y_final - p) / physical.dt
    return y_final, v_next, q, residual


def solve_one_frame_newton(module, p: torch.Tensor, v: torch.Tensor, masses: torch.Tensor, physical, inner_iterations: int):
    q = predictor_from_state(p, v, physical)
    y = p.reshape(1, 6).clone()
    q_batch = q.reshape(1, 6)
    masses_batch = masses.reshape(1, 2)

    with torch.no_grad():
        for _ in range(inner_iterations):
            y, _ = module.apply_newton_update(
                y,
                q_batch,
                masses_batch,
                physical,
                residual_tolerance=module.NEWTON_RESIDUAL_TOLERANCE,
            )
        y_final = y.squeeze(0)
        residual = module.stationarity_residual_norm(
            y.reshape(1, 6),
            q_batch,
            masses_batch,
            dt=physical.dt,
            spring_k=physical.spring_k,
            rest_length=physical.rest_length,
        ).item()
    v_next = (y_final - p) / physical.dt
    return y_final, v_next, q, residual


def rollout(module, model, start_problem, physical, num_frames: int, inner_iterations: int, device: torch.device):
    masses = start_problem.masses.to(device=device, dtype=module.TORCH_DTYPE)

    p0 = start_problem.p_n.to(device=device, dtype=module.TORCH_DTYPE)
    v0 = start_problem.v_n.to(device=device, dtype=module.TORCH_DTYPE)

    mlp_positions = [p0.detach().cpu().clone()]
    newton_positions = [p0.detach().cpu().clone()]
    mlp_velocities = [v0.detach().cpu().clone()]
    newton_velocities = [v0.detach().cpu().clone()]
    mlp_residuals = [float('nan')]
    newton_residuals = [float('nan')]

    p_mlp = p0.clone()
    v_mlp = v0.clone()
    p_newton = p0.clone()
    v_newton = v0.clone()

    for _ in range(num_frames):
        p_mlp, v_mlp, _, residual_mlp = solve_one_frame_mlp(
            module, model, p_mlp, v_mlp, masses, physical, inner_iterations
        )
        p_newton, v_newton, _, residual_newton = solve_one_frame_newton(
            module, p_newton, v_newton, masses, physical, inner_iterations
        )

        mlp_positions.append(p_mlp.detach().cpu().clone())
        newton_positions.append(p_newton.detach().cpu().clone())
        mlp_velocities.append(v_mlp.detach().cpu().clone())
        newton_velocities.append(v_newton.detach().cpu().clone())
        mlp_residuals.append(float(residual_mlp))
        newton_residuals.append(float(residual_newton))

    return {
        "mlp_positions": torch.stack(mlp_positions, dim=0).numpy().astype(float),
        "newton_positions": torch.stack(newton_positions, dim=0).numpy().astype(float),
        "mlp_velocities": torch.stack(mlp_velocities, dim=0).numpy().astype(float),
        "newton_velocities": torch.stack(newton_velocities, dim=0).numpy().astype(float),
        "mlp_residuals": np.asarray(mlp_residuals, dtype=float),
        "newton_residuals": np.asarray(newton_residuals, dtype=float),
    }


def compute_difference_metrics(mlp_positions: np.ndarray, newton_positions: np.ndarray) -> dict[str, np.ndarray]:
    diff = mlp_positions - newton_positions
    p1_diff = np.linalg.norm(diff[:, 0:3], axis=1)
    p2_diff = np.linalg.norm(diff[:, 3:6], axis=1)
    total_diff = np.linalg.norm(diff[:, 0:6], axis=1)
    center_diff = np.linalg.norm(
        0.5 * (mlp_positions[:, 0:3] + mlp_positions[:, 3:6])
        - 0.5 * (newton_positions[:, 0:3] + newton_positions[:, 3:6]),
        axis=1,
    )
    spring_length_mlp = np.linalg.norm(mlp_positions[:, 3:6] - mlp_positions[:, 0:3], axis=1)
    spring_length_newton = np.linalg.norm(newton_positions[:, 3:6] - newton_positions[:, 0:3], axis=1)
    spring_length_diff = np.abs(spring_length_mlp - spring_length_newton)
    return {
        "p1_diff": p1_diff,
        "p2_diff": p2_diff,
        "total_diff": total_diff,
        "center_diff": center_diff,
        "spring_length_diff": spring_length_diff,
        "spring_length_mlp": spring_length_mlp,
        "spring_length_newton": spring_length_newton,
    }


def compute_axis_limits(mlp_positions: np.ndarray, newton_positions: np.ndarray):
    combined = np.concatenate([mlp_positions, newton_positions], axis=0)
    x_values = combined[:, [0, 3]].reshape(-1)
    z_values = combined[:, [2, 5]].reshape(-1)

    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    z_min, z_max = float(np.min(z_values)), float(np.max(z_values))
    x_margin = max(0.08, 0.08 * (x_max - x_min + 1e-12))
    z_margin = max(0.08, 0.08 * (z_max - z_min + 1e-12))
    return (x_min - x_margin, x_max + x_margin), (z_min - z_margin, z_max + z_margin)


def save_static_overview(
    output_path: Path,
    frame_times: np.ndarray,
    mlp_positions: np.ndarray,
    newton_positions: np.ndarray,
    diff_metrics: dict[str, np.ndarray],
    mlp_residuals: np.ndarray,
    newton_residuals: np.ndarray,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].plot(mlp_positions[:, 0], mlp_positions[:, 2], label='MLP particle 1')
    axes[0].plot(mlp_positions[:, 3], mlp_positions[:, 5], label='MLP particle 2')
    axes[0].plot(newton_positions[:, 0], newton_positions[:, 2], '--', label='Newton particle 1')
    axes[0].plot(newton_positions[:, 3], newton_positions[:, 5], '--', label='Newton particle 2')
    axes[0].set_title('x-z trajectories')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('z')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(frame_times, diff_metrics['p1_diff'], label='||Δp1||')
    axes[1].plot(frame_times, diff_metrics['p2_diff'], label='||Δp2||')
    axes[1].plot(frame_times, diff_metrics['total_diff'], label='||Δp|| total', linewidth=2.0)
    axes[1].plot(frame_times, diff_metrics['spring_length_diff'], label='|Δ spring length|', linestyle=':')
    axes[1].set_yscale('log')
    axes[1].set_title('MLP vs Newton position differences')
    axes[1].set_xlabel('physical time')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(frame_times[1:], mlp_residuals[1:], label='MLP residual')
    axes[2].plot(frame_times[1:], newton_residuals[1:], label='Newton residual')
    axes[2].set_yscale('log')
    axes[2].set_title('Per-frame final residual after inner solves')
    axes[2].set_xlabel('physical time')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def build_animation(
    output_gif_path: Path,
    output_mp4_path: Path,
    frame_times: np.ndarray,
    mlp_positions: np.ndarray,
    newton_positions: np.ndarray,
    diff_metrics: dict[str, np.ndarray],
    mlp_residuals: np.ndarray,
    newton_residuals: np.ndarray,
    fps: int,
    dpi: int,
    start_problem_index: int,
    experiment_name: str,
    inner_iterations: int,
):
    num_frames = mlp_positions.shape[0]
    (x_lo, x_hi), (z_lo, z_hi) = compute_axis_limits(mlp_positions, newton_positions)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 6.2))
    ax_mlp, ax_newton, ax_diff = axes

    def init_motion_axis(ax, title: str):
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(z_lo, z_hi)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)

    init_motion_axis(ax_mlp, 'MLP continuous rollout')
    init_motion_axis(ax_newton, 'Newton continuous rollout')

    spring_line_mlp, = ax_mlp.plot([], [], '-o', linewidth=2.0, markersize=7)
    spring_line_newton, = ax_newton.plot([], [], '-o', linewidth=2.0, markersize=7)
    trail_p1_mlp, = ax_mlp.plot([], [], '-', linewidth=1.2, alpha=0.8)
    trail_p2_mlp, = ax_mlp.plot([], [], '-', linewidth=1.2, alpha=0.8)
    trail_p1_newton, = ax_newton.plot([], [], '-', linewidth=1.2, alpha=0.8)
    trail_p2_newton, = ax_newton.plot([], [], '-', linewidth=1.2, alpha=0.8)

    info_mlp = ax_mlp.text(0.02, 0.98, '', transform=ax_mlp.transAxes, va='top', ha='left', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    info_newton = ax_newton.text(0.02, 0.98, '', transform=ax_newton.transAxes, va='top', ha='left', fontsize=9,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_diff.set_title('Differences that are hard to see in the motion view')
    ax_diff.set_xlabel('physical time')
    ax_diff.set_ylabel('difference magnitude')
    ax_diff.set_yscale('log')
    ax_diff.grid(True, alpha=0.3)

    line_p1, = ax_diff.plot(frame_times, diff_metrics['p1_diff'], label='||Δp1||')
    line_p2, = ax_diff.plot(frame_times, diff_metrics['p2_diff'], label='||Δp2||')
    line_total, = ax_diff.plot(frame_times, diff_metrics['total_diff'], linewidth=2.0, label='||Δp|| total')
    line_length, = ax_diff.plot(frame_times, diff_metrics['spring_length_diff'], linestyle=':', label='|Δ spring length|')
    cursor = ax_diff.axvline(frame_times[0], linestyle='--', alpha=0.8)
    marker_p1, = ax_diff.plot([frame_times[0]], [max(diff_metrics['p1_diff'][0], 1e-18)], 'o')
    marker_p2, = ax_diff.plot([frame_times[0]], [max(diff_metrics['p2_diff'][0], 1e-18)], 'o')
    marker_total, = ax_diff.plot([frame_times[0]], [max(diff_metrics['total_diff'][0], 1e-18)], 'o')
    marker_length, = ax_diff.plot([frame_times[0]], [max(diff_metrics['spring_length_diff'][0], 1e-18)], 'o')
    ax_diff.legend(fontsize=8, loc='upper left')
    diff_text = ax_diff.text(0.02, 0.98, '', transform=ax_diff.transAxes, va='top', ha='left', fontsize=9,
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(
        f"Continuous rollout from test problem {start_problem_index} | model={experiment_name} | inner iterations/frame={inner_iterations}",
        y=0.98,
        fontsize=13,
    )

    def update(frame_idx: int):
        p_mlp = mlp_positions[frame_idx]
        p_newton = newton_positions[frame_idx]

        spring_line_mlp.set_data([p_mlp[0], p_mlp[3]], [p_mlp[2], p_mlp[5]])
        spring_line_newton.set_data([p_newton[0], p_newton[3]], [p_newton[2], p_newton[5]])

        trail_p1_mlp.set_data(mlp_positions[: frame_idx + 1, 0], mlp_positions[: frame_idx + 1, 2])
        trail_p2_mlp.set_data(mlp_positions[: frame_idx + 1, 3], mlp_positions[: frame_idx + 1, 5])
        trail_p1_newton.set_data(newton_positions[: frame_idx + 1, 0], newton_positions[: frame_idx + 1, 2])
        trail_p2_newton.set_data(newton_positions[: frame_idx + 1, 3], newton_positions[: frame_idx + 1, 5])

        info_mlp.set_text(
            f"frame = {frame_idx}\n"
            f"time = {frame_times[frame_idx]:.3f}s\n"
            f"residual = {mlp_residuals[frame_idx]:.3e}\n"
            f"p1_y = {p_mlp[1]:.5f}, p2_y = {p_mlp[4]:.5f}"
        )
        info_newton.set_text(
            f"frame = {frame_idx}\n"
            f"time = {frame_times[frame_idx]:.3f}s\n"
            f"residual = {newton_residuals[frame_idx]:.3e}\n"
            f"p1_y = {p_newton[1]:.5f}, p2_y = {p_newton[4]:.5f}"
        )

        current_time = frame_times[frame_idx]
        cursor.set_xdata([current_time, current_time])

        def safe_value(values: np.ndarray, idx: int) -> float:
            return float(max(values[idx], 1e-18))

        marker_p1.set_data([current_time], [safe_value(diff_metrics['p1_diff'], frame_idx)])
        marker_p2.set_data([current_time], [safe_value(diff_metrics['p2_diff'], frame_idx)])
        marker_total.set_data([current_time], [safe_value(diff_metrics['total_diff'], frame_idx)])
        marker_length.set_data([current_time], [safe_value(diff_metrics['spring_length_diff'], frame_idx)])

        diff_text.set_text(
            f"||Δp1|| = {diff_metrics['p1_diff'][frame_idx]:.3e}\n"
            f"||Δp2|| = {diff_metrics['p2_diff'][frame_idx]:.3e}\n"
            f"||Δp|| total = {diff_metrics['total_diff'][frame_idx]:.3e}\n"
            f"|Δ spring length| = {diff_metrics['spring_length_diff'][frame_idx]:.3e}"
        )

        return (
            spring_line_mlp,
            spring_line_newton,
            trail_p1_mlp,
            trail_p2_mlp,
            trail_p1_newton,
            trail_p2_newton,
            info_mlp,
            info_newton,
            cursor,
            marker_p1,
            marker_p2,
            marker_total,
            marker_length,
            diff_text,
        )

    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 / max(fps, 1), blit=False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    anim.save(output_gif_path, writer=PillowWriter(fps=fps), dpi=dpi)

    if shutil.which('ffmpeg') is not None:
        anim.save(output_mp4_path, writer=FFMpegWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    def convert(value: Any):
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return convert(value.tolist())
        if isinstance(value, torch.Tensor):
            return convert(value.detach().cpu().tolist())
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value
    with path.open('w', encoding='utf-8') as f:
        json.dump(convert(summary), f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    module = load_base_module(args.base_script_path)

    device = torch.device(args.device)
    ensure_device(module, device)

    physical = module.default_physical_config()
    problems = module.generate_reference_sequence(physical, module.DEFAULT_TOTAL_TIME_STEPS)
    split = module.build_problem_split(module.DEFAULT_TOTAL_TIME_STEPS)
    start_problem_index = choose_start_problem(module, split, args.test_split, args.start_problem_index)
    start_problem = problems[start_problem_index]

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = args.project_root / args.experiment_name / args.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run the training script first or pass --checkpoint-path explicitly."
        )

    model = module.MLPOptimizer(residual_length_scale=module.DEFAULT_RESIDUAL_LENGTH_SCALE)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = rollout(
        module=module,
        model=model,
        start_problem=start_problem,
        physical=physical,
        num_frames=args.num_frames,
        inner_iterations=args.inner_iterations,
        device=device,
    )

    frame_times = np.arange(args.num_frames + 1, dtype=float) * physical.dt
    diff_metrics = compute_difference_metrics(result['mlp_positions'], result['newton_positions'])

    base_name = (
        f"{args.experiment_name}_{args.test_split}_start_{start_problem_index}_"
        f"frames_{args.num_frames}_inner_{args.inner_iterations}"
    )
    gif_path = output_dir / f"{base_name}.gif"
    mp4_path = output_dir / f"{base_name}.mp4"
    overview_path = output_dir / f"{base_name}_overview.png"
    data_path = output_dir / f"{base_name}_trajectories.npz"
    summary_path = output_dir / f"{base_name}_summary.json"

    build_animation(
        output_gif_path=gif_path,
        output_mp4_path=mp4_path,
        frame_times=frame_times,
        mlp_positions=result['mlp_positions'],
        newton_positions=result['newton_positions'],
        diff_metrics=diff_metrics,
        mlp_residuals=result['mlp_residuals'],
        newton_residuals=result['newton_residuals'],
        fps=args.fps,
        dpi=args.dpi,
        start_problem_index=start_problem_index,
        experiment_name=args.experiment_name,
        inner_iterations=args.inner_iterations,
    )

    save_static_overview(
        overview_path,
        frame_times,
        result['mlp_positions'],
        result['newton_positions'],
        diff_metrics,
        result['mlp_residuals'],
        result['newton_residuals'],
    )

    np.savez(
        data_path,
        frame_times=frame_times,
        mlp_positions=result['mlp_positions'],
        newton_positions=result['newton_positions'],
        mlp_velocities=result['mlp_velocities'],
        newton_velocities=result['newton_velocities'],
        mlp_residuals=result['mlp_residuals'],
        newton_residuals=result['newton_residuals'],
        p1_diff=diff_metrics['p1_diff'],
        p2_diff=diff_metrics['p2_diff'],
        total_diff=diff_metrics['total_diff'],
        center_diff=diff_metrics['center_diff'],
        spring_length_diff=diff_metrics['spring_length_diff'],
    )

    summary = {
        'base_script_path': args.base_script_path,
        'project_root': args.project_root,
        'experiment_name': args.experiment_name,
        'checkpoint_path': checkpoint_path,
        'device': str(device),
        'test_split': args.test_split,
        'start_problem_index': start_problem_index,
        'start_problem_time': start_problem.time,
        'num_frames': args.num_frames,
        'inner_iterations': args.inner_iterations,
        'dt': physical.dt,
        'outputs': {
            'gif': gif_path,
            'mp4': mp4_path if mp4_path.exists() else None,
            'overview_png': overview_path,
            'trajectory_npz': data_path,
        },
        'final_metrics': {
            'final_p1_diff': float(diff_metrics['p1_diff'][-1]),
            'final_p2_diff': float(diff_metrics['p2_diff'][-1]),
            'final_total_diff': float(diff_metrics['total_diff'][-1]),
            'max_total_diff': float(np.max(diff_metrics['total_diff'])),
            'mean_total_diff': float(np.mean(diff_metrics['total_diff'])),
            'max_spring_length_diff': float(np.max(diff_metrics['spring_length_diff'])),
            'final_mlp_residual': float(result['mlp_residuals'][-1]),
            'final_newton_residual': float(result['newton_residuals'][-1]),
        },
    }
    save_summary(summary_path, summary)

    print('\nContinuous rollout comparison completed.')
    print(f'Start test problem index: {start_problem_index} (physical time = {start_problem.time:.3f}s)')
    print(f'Loaded checkpoint: {checkpoint_path}')
    print(f'GIF saved to: {gif_path}')
    if mp4_path.exists():
        print(f'MP4 saved to: {mp4_path}')
    print(f'Overview figure saved to: {overview_path}')
    print(f'Raw trajectory data saved to: {data_path}')
    print(f'Summary JSON saved to: {summary_path}')
    print(f'Final total position difference: {diff_metrics["total_diff"][-1]:.6e}')
    print(f'Max total position difference: {np.max(diff_metrics["total_diff"]):.6e}')


if __name__ == '__main__':
    main()
