"""Script 1: generate 500-step reference trajectories and per-problem samples.

Outputs are written under --output-dir:
    data/reference/reference_problems.pt
    data/reference/reference_motion_states.pt
    data/reference/runtime_config.json
    data/reference/motion_catalogue.json
    data/samples/all_sampled_problems.pt
    data/reference/initial_state_figures/motion_XXX_initial_state.png

Run:
    python cloth01_generate_reference_and_samples.py --output-dir cloth_5x5_500step_pipeline
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cloth03_solvers_and_models import (
    DEFAULT_SAMPLING_RADIUS_MAX,
    DEFAULT_SAMPLING_RADIUS_MIN,
    DEFAULT_TOTAL_TIME_STEPS,
    FIXED_VERTEX_INDICES,
    FREE_STATE_DIM,
    FULL_STATE_DIM,
    GRID_COLS,
    GRID_ROWS,
    NUM_PARTICLES,
    SPATIAL_DIM,
    SPRING_EDGES,
    TORCH_DTYPE,
    TRAIN_SOBOL_SEED,
    TRIANGLE_FACES,
    TimeStepProblem,
    build_motion_catalogue,
    build_problem_dataset,
    default_physical_config,
    full_state_from_free_state,
    full_state_from_positions,
    generate_reference_sequence_for_motion,
    project_fixed_vertices,
)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def tensor_stack(records: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([x.detach().cpu() for x in records], dim=0).contiguous()


def problem_records_to_tensor_dict(problems: list[TimeStepProblem], physical) -> dict[str, Any]:
    q_full = []
    exact_y_full = []
    for problem in problems:
        q_full.append(
            project_fixed_vertices(
                full_state_from_free_state(problem.q_free.reshape(1, -1), physical), physical
            ).squeeze(0)
        )
        exact_y_full.append(
            project_fixed_vertices(
                full_state_from_free_state(problem.exact_y_free.reshape(1, -1), physical), physical
            ).squeeze(0)
        )

    return {
        "p_n_full": tensor_stack([full_state_from_positions(p.p_n_full) for p in problems]),
        "v_n_full": tensor_stack([full_state_from_positions(p.v_n_full) for p in problems]),
        "q": tensor_stack(q_full),
        "exact_y": tensor_stack(exact_y_full),
        "q_free": tensor_stack([p.q_free for p in problems]),
        "exact_y_free": tensor_stack([p.exact_y_free for p in problems]),
        "masses": tensor_stack([p.free_masses for p in problems]),
        "problem_index": torch.tensor([p.index for p in problems], dtype=torch.long),
        "motion_index": torch.tensor([p.motion_index for p in problems], dtype=torch.long),
        "time_index": torch.tensor([p.local_time_index for p in problems], dtype=torch.long),
        "time": torch.tensor([p.time for p in problems], dtype=TORCH_DTYPE),
        "raw_sampling_radius": torch.tensor([p.raw_sampling_radius for p in problems], dtype=TORCH_DTYPE),
        "sampling_radius": torch.tensor([p.sampling_radius for p in problems], dtype=TORCH_DTYPE),
        "exact_energy": torch.tensor([p.exact_energy for p in problems], dtype=TORCH_DTYPE),
        "exact_residual": torch.tensor([p.exact_residual for p in problems], dtype=TORCH_DTYPE),
        "metadata": {
            "state_dim": FULL_STATE_DIM,
            "free_state_dim": FREE_STATE_DIM,
            "num_problems": len(problems),
            "problem_unit": "one motion at one time step",
            "state_representation": "full 75D positions with fixed vertices projected",
        },
    }


def build_reference_motion_states(
    problems: list[TimeStepProblem], total_time_steps: int, physical
) -> dict[str, Any]:
    by_motion: dict[int, list[TimeStepProblem]] = {}
    for problem in problems:
        by_motion.setdefault(problem.motion_index, []).append(problem)

    position_sequences = []
    velocity_sequences = []
    for motion_index in sorted(by_motion):
        sequence = sorted(by_motion[motion_index], key=lambda p: p.local_time_index)
        first = sequence[0]
        positions = [first.p_n_full.detach().cpu()]
        velocities = [first.v_n_full.detach().cpu()]
        for problem in sequence:
            y_full = project_fixed_vertices(
                full_state_from_free_state(problem.exact_y_free.reshape(1, -1), physical), physical
            ).reshape(NUM_PARTICLES, SPATIAL_DIM)
            positions.append(y_full.detach().cpu())
            if len(positions) >= 2:
                velocity = (positions[-1] - positions[-2]) / physical.dt
                velocity[list(FIXED_VERTEX_INDICES), :] = 0.0
                velocities.append(velocity)
        if len(positions) != total_time_steps + 1:
            raise RuntimeError(f"motion {motion_index} has {len(positions)} frames")
        position_sequences.append(torch.stack(positions, dim=0))
        velocity_sequences.append(torch.stack(velocities, dim=0))

    return {
        "positions": torch.stack(position_sequences, dim=0).contiguous(),
        "velocities": torch.stack(velocity_sequences, dim=0).contiguous(),
        "motion_index": torch.tensor(sorted(by_motion), dtype=torch.long),
        "metadata": {
            "shape": "[num_motions, total_time_steps + 1, 25, 3]",
            "total_time_steps": total_time_steps,
            "dt": physical.dt,
        },
    }


def build_all_samples(
    problems: list[TimeStepProblem], *, points_per_problem: int, seed: int, physical
) -> dict[str, Any]:
    initial_y = []
    q = []
    masses = []
    exact_y = []
    problem_index = []
    motion_index = []
    time_index = []

    for n, problem in enumerate(problems):
        dataset = build_problem_dataset(
            problem=problem,
            size=points_per_problem,
            seed=seed + 100_003 * int(problem.motion_index) + 1009 * int(problem.local_time_index),
            role=f"sample_m{problem.motion_index:02d}_t{problem.local_time_index:03d}",
            physical=physical,
            include_explicit_train_points=False,
        )
        initial_y.append(dataset.initial_y)
        q.append(dataset.q)
        masses.append(dataset.masses)
        exact_y.append(dataset.exact_y)
        problem_index.append(dataset.problem_index)
        motion_index.append(dataset.motion_index)
        time_index.append(dataset.time_index)
        if n == 0 or (n + 1) % 500 == 0:
            print(f"sampled {n + 1:5d}/{len(problems)} time-step problems")

    return {
        "initial_y": torch.cat(initial_y, dim=0).contiguous(),
        "q": torch.cat(q, dim=0).contiguous(),
        "masses": torch.cat(masses, dim=0).contiguous(),
        "exact_y": torch.cat(exact_y, dim=0).contiguous(),
        "problem_index": torch.cat(problem_index, dim=0).contiguous(),
        "motion_index": torch.cat(motion_index, dim=0).contiguous(),
        "time_index": torch.cat(time_index, dim=0).contiguous(),
        "metadata": {
            "points_per_problem": points_per_problem,
            "num_problems": len(problems),
            "num_samples": len(problems) * points_per_problem,
            "seed": seed,
            "sampling": "Sobol samples in a Linf cube around each reference solution",
            "explicit_current_or_exact_points": False,
        },
    }


def plot_initial_state(motion, physical, save_path: Path) -> None:
    positions = torch.tensor(motion.p0, dtype=TORCH_DTYPE)
    velocities = torch.tensor(motion.v0, dtype=TORCH_DTYPE)
    fixed = list(FIXED_VERTEX_INDICES)
    positions[fixed, :] = torch.tensor(physical.fixed_positions, dtype=TORCH_DTYPE)
    velocities[fixed, :] = 0.0

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    for i, j in SPRING_EDGES:
        xs = [positions[i, 0].item(), positions[j, 0].item()]
        ys = [positions[i, 1].item(), positions[j, 1].item()]
        zs = [positions[i, 2].item(), positions[j, 2].item()]
        ax.plot(xs, ys, zs, linewidth=0.8)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=18)
    ax.scatter(positions[fixed, 0], positions[fixed, 1], positions[fixed, 2], s=55, marker="s")

    free = [i for i in range(NUM_PARTICLES) if i not in fixed]
    velocity_scale = 0.08
    ax.quiver(
        positions[free, 0], positions[free, 1], positions[free, 2],
        velocities[free, 0] * velocity_scale,
        velocities[free, 1] * velocity_scale,
        velocities[free, 2] * velocity_scale,
        length=1.0,
        normalize=False,
        linewidth=0.6,
    )
    ax.set_title(f"motion {motion.index:02d}: {motion.name}\ntime step 0 initial state")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 32x500 cloth reference data and samples.")
    parser.add_argument("--output-dir", type=Path, default=Path("cloth_5x5_500step_pipeline"))
    parser.add_argument("--total-time-steps", type=int, default=DEFAULT_TOTAL_TIME_STEPS)
    parser.add_argument("--points-per-problem", type=int, default=32)
    parser.add_argument("--sampling-radius-min", type=float, default=DEFAULT_SAMPLING_RADIUS_MIN)
    parser.add_argument("--sampling-radius-max", type=float, default=DEFAULT_SAMPLING_RADIUS_MAX)
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SOBOL_SEED)
    parser.add_argument("--skip-samples", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_time_steps <= 0 or args.points_per_problem <= 0:
        raise ValueError("total-time-steps and points-per-problem must be positive")

    root = args.output_dir
    reference_dir = root / "data" / "reference"
    samples_dir = root / "data" / "samples"
    reference_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    physical = default_physical_config()
    motions, motion_split = build_motion_catalogue(physical)

    print(f"output directory: {root.resolve()}")
    print(f"motions: {len(motions)}; time steps per motion: {args.total_time_steps}")

    if not args.skip_plots:
        plot_dir = reference_dir / "initial_state_figures"
        for motion in motions:
            plot_initial_state(motion, physical, plot_dir / f"motion_{motion.index:03d}_initial_state.png")
        print(f"saved initial state figures to {plot_dir}")

    all_problems: list[TimeStepProblem] = []
    for motion in motions:
        all_problems.extend(
            generate_reference_sequence_for_motion(
                physical=physical,
                motion=motion,
                total_steps=args.total_time_steps,
                sampling_radius_min=args.sampling_radius_min,
                sampling_radius_max=args.sampling_radius_max,
            )
        )

    torch.save(problem_records_to_tensor_dict(all_problems, physical), reference_dir / "reference_problems.pt")
    torch.save(
        build_reference_motion_states(all_problems, args.total_time_steps, physical),
        reference_dir / "reference_motion_states.pt",
    )

    save_json(
        {
            "physical_config": asdict(physical),
            "motion_split": asdict(motion_split),
            "motions": [asdict(motion) for motion in motions],
            "grid_rows": GRID_ROWS,
            "grid_cols": GRID_COLS,
            "num_particles": NUM_PARTICLES,
            "spatial_dim": SPATIAL_DIM,
            "full_state_dim": FULL_STATE_DIM,
            "free_state_dim": FREE_STATE_DIM,
            "fixed_vertex_indices": list(FIXED_VERTEX_INDICES),
            "spring_edges": [list(edge) for edge in SPRING_EDGES],
            "triangle_faces": [list(face) for face in TRIANGLE_FACES],
            "total_time_steps": args.total_time_steps,
            "points_per_problem": args.points_per_problem,
            "sampling_radius_min": args.sampling_radius_min,
            "sampling_radius_max": args.sampling_radius_max,
        },
        reference_dir / "runtime_config.json",
    )
    save_json(
        {"motions": [asdict(motion) for motion in motions], "motion_split": asdict(motion_split)},
        reference_dir / "motion_catalogue.json",
    )

    if not args.skip_samples:
        samples = build_all_samples(
            all_problems,
            points_per_problem=args.points_per_problem,
            seed=args.sample_seed,
            physical=physical,
        )
        torch.save(samples, samples_dir / "all_sampled_problems.pt")
        print(f"saved samples: {samples['metadata']}")

    print("done")


if __name__ == "__main__":
    main()
