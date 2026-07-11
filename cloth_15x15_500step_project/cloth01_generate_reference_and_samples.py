"""Generate/audit 15x15 reference trajectories and compact training sample shards.

Recommended two-pass workflow:
1. `--reference-only`: generate all 32 raw reference motions and inspect residuals.
2. Render suspicious motions, decide exclusions, then run `--samples-only` with
   `--exclude-motion-indices ...` to generate only usable training shards.

Reference data are never deleted when a motion is excluded. Exclusion is applied
only to downstream sample/dataset construction so the audit remains reproducible.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    NUM_FREE_PARTICLES,
    NUM_PARTICLES,
    SPATIAL_DIM,
    SPRING_EDGES,
    TORCH_DTYPE,
    TRAIN_SOBOL_SEED,
    TRIANGLE_FACES,
    TimeStepProblem,
    build_motion_catalogue,
    default_physical_config,
    full_state_from_free_state,
    full_state_from_positions,
    generate_reference_sequence_for_motion,
    generate_sobol_points,
    project_fixed_vertices,
)
from cloth_common import resolve_exclusions, save_json, write_csv

TRAIN_MOTIONS = tuple(range(0, 16))


def stack_cpu(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([v.detach().cpu() for v in values], dim=0).contiguous()


def problems_to_reference(problems: list[TimeStepProblem], physical) -> dict[str, Any]:
    q_full: list[torch.Tensor] = []
    exact_full: list[torch.Tensor] = []
    for problem in problems:
        q_full.append(project_fixed_vertices(
            full_state_from_free_state(problem.q_free.reshape(1, -1), physical), physical
        ).squeeze(0))
        exact_full.append(project_fixed_vertices(
            full_state_from_free_state(problem.exact_y_free.reshape(1, -1), physical), physical
        ).squeeze(0))
    return {
        "p_n_full": stack_cpu([full_state_from_positions(p.p_n_full) for p in problems]),
        "v_n_full": stack_cpu([full_state_from_positions(p.v_n_full) for p in problems]),
        "q": stack_cpu(q_full),
        "exact_y": stack_cpu(exact_full),
        "q_free": stack_cpu([p.q_free for p in problems]),
        "exact_y_free": stack_cpu([p.exact_y_free for p in problems]),
        "masses": stack_cpu([p.free_masses for p in problems]),
        "problem_index": torch.tensor([p.index for p in problems], dtype=torch.long),
        "motion_index": torch.tensor([p.motion_index for p in problems], dtype=torch.long),
        "time_index": torch.tensor([p.local_time_index for p in problems], dtype=torch.long),
        "time": torch.tensor([p.time for p in problems], dtype=TORCH_DTYPE),
        "raw_sampling_radius": torch.tensor([p.raw_sampling_radius for p in problems], dtype=TORCH_DTYPE),
        "sampling_radius": torch.tensor([p.sampling_radius for p in problems], dtype=TORCH_DTYPE),
        "exact_energy": torch.tensor([p.exact_energy for p in problems], dtype=TORCH_DTYPE),
        "exact_residual": torch.tensor([p.exact_residual for p in problems], dtype=TORCH_DTYPE),
        "metadata": {
            "grid": [GRID_ROWS, GRID_COLS],
            "state_dim": FULL_STATE_DIM,
            "free_state_dim": FREE_STATE_DIM,
            "num_problems": len(problems),
            "problem_unit": "one motion at one physical time step",
            "state_representation": f"full {FULL_STATE_DIM}D positions with fixed vertices projected",
        },
    }


def build_motion_states(reference: dict[str, Any], total_steps: int, physical) -> dict[str, Any]:
    positions_all: list[torch.Tensor] = []
    velocities_all: list[torch.Tensor] = []
    residual_all: list[torch.Tensor] = []
    energy_all: list[torch.Tensor] = []
    radius_all: list[torch.Tensor] = []
    motion_ids = sorted(set(int(v) for v in reference["motion_index"].tolist()))
    for motion_index in motion_ids:
        rows = torch.nonzero(reference["motion_index"] == motion_index, as_tuple=False).flatten()
        rows = rows[torch.argsort(reference["time_index"].index_select(0, rows))]
        if rows.numel() != total_steps:
            raise RuntimeError(f"motion {motion_index} has {rows.numel()} problems; expected {total_steps}")
        p0 = reference["p_n_full"][rows[0]].reshape(NUM_PARTICLES, SPATIAL_DIM)
        v0 = reference["v_n_full"][rows[0]].reshape(NUM_PARTICLES, SPATIAL_DIM)
        exact = reference["exact_y"].index_select(0, rows).reshape(total_steps, NUM_PARTICLES, SPATIAL_DIM)
        positions = torch.cat([p0.unsqueeze(0), exact], dim=0)
        velocities = torch.empty_like(positions)
        velocities[0] = v0
        velocities[1:] = (positions[1:] - positions[:-1]) / physical.dt
        velocities[:, list(FIXED_VERTEX_INDICES), :] = 0.0
        positions_all.append(positions)
        velocities_all.append(velocities)
        residual_all.append(reference["exact_residual"].index_select(0, rows))
        energy_all.append(reference["exact_energy"].index_select(0, rows))
        radius_all.append(reference["sampling_radius"].index_select(0, rows))
    return {
        "positions": torch.stack(positions_all).contiguous(),
        "velocities": torch.stack(velocities_all).contiguous(),
        "exact_residual": torch.stack(residual_all).contiguous(),
        "exact_energy": torch.stack(energy_all).contiguous(),
        "sampling_radius": torch.stack(radius_all).contiguous(),
        "motion_index": torch.tensor(motion_ids, dtype=torch.long),
        "metadata": {
            "shape_positions": [len(motion_ids), total_steps + 1, NUM_PARTICLES, SPATIAL_DIM],
            "shape_residual": [len(motion_ids), total_steps],
            "dt": physical.dt,
            "residual_semantics": "stationarity residual of stored reference solution at every frame",
        },
    }


def audit_reference(states: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    residual = states["exact_residual"].numpy().astype(float)
    motion_ids = states["motion_index"].tolist()
    rows: list[dict[str, Any]] = []
    for row_index, motion_index in enumerate(motion_ids):
        values = residual[row_index]
        finite = values[np.isfinite(values)]
        rows.append({
            "motion_index": int(motion_index),
            "num_frames": int(values.size),
            "num_nonfinite": int((~np.isfinite(values)).sum()),
            "residual_mean": float(np.mean(finite)) if finite.size else float("inf"),
            "residual_p95": float(np.percentile(finite, 95)) if finite.size else float("inf"),
            "residual_p99": float(np.percentile(finite, 99)) if finite.size else float("inf"),
            "residual_max": float(np.max(finite)) if finite.size else float("inf"),
            "worst_frame": int(np.nanargmax(values)) if finite.size else None,
            "final_residual": float(values[-1]),
        })

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(np.arange(values.size), np.maximum(values, 1e-30))
        ax.set_yscale("log")
        ax.set_xlabel("physical frame")
        ax.set_ylabel("reference stationarity residual")
        ax.set_title(f"motion {int(motion_index):03d}: reference residual")
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"motion_{int(motion_index):03d}_reference_residual.png", dpi=180)
        plt.close(fig)

    write_csv(rows, output_dir / "reference_motion_summary.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    for row_index, motion_index in enumerate(motion_ids):
        ax.plot(np.arange(residual.shape[1]), np.maximum(residual[row_index], 1e-30), label=str(motion_index), linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("physical frame")
    ax.set_ylabel("reference stationarity residual")
    ax.set_title("all reference motions")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(ncol=4, fontsize=7, title="motion")
    fig.tight_layout()
    fig.savefig(output_dir / "all_motion_reference_residuals.png", dpi=200)
    plt.close(fig)

    hardest_max = max(rows, key=lambda r: (r["num_nonfinite"] > 0, r["residual_max"]))
    hardest_p95 = max(rows, key=lambda r: (r["num_nonfinite"] > 0, r["residual_p95"]))
    audit = {
        "ranking_rows": rows,
        "hardest_by_max": hardest_max,
        "hardest_by_p95": hardest_p95,
        "threshold_not_applied": True,
        "recommended_manual_workflow": (
            "render suspicious motions, choose a residual/convergence threshold, then pass complete motion indices "
            "to --exclude-motion-indices; do not remove isolated frames"
        ),
    }
    save_json(audit, output_dir / "reference_audit.json")
    return audit


def plot_initial_state(motion, physical, path: Path) -> None:
    positions = torch.tensor(motion.p0, dtype=TORCH_DTYPE)
    velocities = torch.tensor(motion.v0, dtype=TORCH_DTYPE)
    fixed = list(FIXED_VERTEX_INDICES)
    positions[fixed] = torch.tensor(physical.fixed_positions, dtype=TORCH_DTYPE)
    velocities[fixed] = 0.0
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i, j in SPRING_EDGES:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            [positions[i, 2], positions[j, 2]],
            linewidth=0.35,
        )
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=5)
    ax.scatter(positions[fixed, 0], positions[fixed, 1], positions[fixed, 2], s=45, marker="s")
    free = [i for i in range(NUM_PARTICLES) if i not in fixed]
    ax.quiver(
        positions[free, 0], positions[free, 1], positions[free, 2],
        velocities[free, 0], velocities[free, 1], velocities[free, 2],
        length=0.04, normalize=False, linewidth=0.25,
    )
    ax.set_title(f"motion {motion.index:03d}: {motion.name}\ninitial state")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def rows_for_motion(reference: dict[str, Any], motion_index: int, time_stop: int) -> torch.Tensor:
    mask = (reference["motion_index"] == motion_index) & (reference["time_index"] < time_stop)
    rows = torch.nonzero(mask, as_tuple=False).flatten()
    rows = rows[torch.argsort(reference["time_index"].index_select(0, rows))]
    if rows.numel() != time_stop:
        raise RuntimeError(f"motion {motion_index}: expected {time_stop} rows, found {rows.numel()}")
    return rows


def generate_training_sample_shards(
    *,
    reference: dict[str, Any],
    physical,
    output_dir: Path,
    motion_indices: tuple[int, ...],
    time_stop: int,
    points_per_problem: int,
    seed: int,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for motion_index in motion_indices:
        path = output_dir / f"motion_{motion_index:03d}.pt"
        if path.exists() and not overwrite:
            print(f"skip existing sample shard {path}")
            records.append({"motion_index": motion_index, "path": str(path), "reused": True})
            continue
        rows = rows_for_motion(reference, motion_index, time_stop)
        initial = torch.empty((time_stop, points_per_problem, FULL_STATE_DIM), dtype=TORCH_DTYPE)
        for local, source_row in enumerate(rows.tolist()):
            center = reference["exact_y_free"][source_row].to(TORCH_DTYPE)
            radius = float(reference["sampling_radius"][source_row].item())
            time_index = int(reference["time_index"][source_row].item())
            points, _ = generate_sobol_points(
                count=points_per_problem,
                center=center,
                radius=radius,
                seed=seed + 100_003 * motion_index + 1009 * time_index,
                physical=physical,
                explicit_points=(),
            )
            initial[local] = project_fixed_vertices(full_state_from_free_state(points, physical), physical).cpu()
            if local == 0 or (local + 1) % 50 == 0 or local + 1 == time_stop:
                print(f"motion {motion_index:03d}: sampled {local + 1:03d}/{time_stop} x {points_per_problem}")
        payload = {
            "initial_y": initial.contiguous(),
            "q": reference["q"].index_select(0, rows).contiguous(),
            "masses": reference["masses"].index_select(0, rows).contiguous(),
            "exact_y": reference["exact_y"].index_select(0, rows).contiguous(),
            "problem_index": reference["problem_index"].index_select(0, rows).contiguous(),
            "motion_index": reference["motion_index"].index_select(0, rows).contiguous(),
            "time_index": reference["time_index"].index_select(0, rows).contiguous(),
            "metadata": {
                "format": "motion_shard_v1",
                "motion_index": motion_index,
                "time_range": [0, time_stop - 1],
                "points_per_problem": points_per_problem,
                "nested_prefixes": True,
                "physical_xn_included": False,
                "sample_semantics": "scrambled Sobol perturbations around exact_y in stored Linf radius",
            },
        }
        torch.save(payload, path)
        records.append({"motion_index": motion_index, "path": str(path), "reused": False})
    save_json({
        "format": "motion_shards_v1",
        "motion_indices": list(motion_indices),
        "time_range": [0, time_stop - 1],
        "points_per_problem": points_per_problem,
        "physical_xn_included": False,
        "records": records,
    }, output_dir / "manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and audit 15x15 cloth reference data.")
    parser.add_argument("--output-dir", type=Path, default=Path("cloth_15x15_500step_pipeline"))
    parser.add_argument("--total-time-steps", type=int, default=DEFAULT_TOTAL_TIME_STEPS)
    parser.add_argument("--train-time-stop", type=int, default=400)
    parser.add_argument("--points-per-problem", type=int, default=32)
    parser.add_argument("--sampling-radius-min", type=float, default=DEFAULT_SAMPLING_RADIUS_MIN)
    parser.add_argument("--sampling-radius-max", type=float, default=DEFAULT_SAMPLING_RADIUS_MAX)
    parser.add_argument("--sample-seed", type=int, default=TRAIN_SOBOL_SEED)
    parser.add_argument("--exclude-motion-indices", type=int, nargs="*", default=[])
    parser.add_argument("--exclusion-file", type=Path, default=None)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--samples-only", action="store_true")
    parser.add_argument("--skip-initial-plots", action="store_true")
    parser.add_argument("--overwrite-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_only and args.samples_only:
        raise ValueError("--reference-only and --samples-only are mutually exclusive")
    if args.total_time_steps <= 0 or args.train_time_stop <= 0 or args.points_per_problem <= 0:
        raise ValueError("time-step and point counts must be positive")
    exclusions = resolve_exclusions(args.exclude_motion_indices, args.exclusion_file)
    root = args.output_dir
    reference_dir = root / "data" / "reference"
    sample_dir = root / "data" / "samples"
    reference_dir.mkdir(parents=True, exist_ok=True)
    physical = default_physical_config()
    motions, split = build_motion_catalogue(physical)

    reference_path = reference_dir / "reference_problems.pt"
    states_path = reference_dir / "reference_motion_states.pt"
    if not args.samples_only:
        if not args.skip_initial_plots:
            for motion in motions:
                plot_initial_state(motion, physical, reference_dir / "initial_state_figures" / f"motion_{motion.index:03d}.png")
        all_problems: list[TimeStepProblem] = []
        for motion in motions:
            all_problems.extend(generate_reference_sequence_for_motion(
                physical=physical,
                motion=motion,
                total_steps=args.total_time_steps,
                sampling_radius_min=args.sampling_radius_min,
                sampling_radius_max=args.sampling_radius_max,
            ))
        reference = problems_to_reference(all_problems, physical)
        states = build_motion_states(reference, args.total_time_steps, physical)
        torch.save(reference, reference_path)
        torch.save(states, states_path)
        save_json({
            "physical_config": asdict(physical),
            "motion_split": asdict(split),
            "motions": [asdict(m) for m in motions],
            "grid_rows": GRID_ROWS,
            "grid_cols": GRID_COLS,
            "num_particles": NUM_PARTICLES,
            "num_free_particles": NUM_FREE_PARTICLES,
            "full_state_dim": FULL_STATE_DIM,
            "free_state_dim": FREE_STATE_DIM,
            "fixed_vertex_indices": list(FIXED_VERTEX_INDICES),
            "spring_edges": [list(e) for e in SPRING_EDGES],
            "triangle_faces": [list(f) for f in TRIANGLE_FACES],
            "total_time_steps": args.total_time_steps,
            "train_time_stop": args.train_time_stop,
            "default_points_per_problem": args.points_per_problem,
            "sampling_radius_min": args.sampling_radius_min,
            "sampling_radius_max": args.sampling_radius_max,
        }, reference_dir / "runtime_config.json")
        save_json({"motions": [asdict(m) for m in motions], "motion_split": asdict(split)}, reference_dir / "motion_catalogue.json")
        audit_reference(states, reference_dir / "residual_audit")
        print(f"saved raw reference data under {reference_dir}")
    else:
        if not reference_path.exists() or not states_path.exists():
            raise FileNotFoundError("run --reference-only first")
        reference = torch.load(reference_path, map_location="cpu")

    if not args.reference_only:
        train_motions = tuple(i for i in TRAIN_MOTIONS if i not in exclusions)
        if not train_motions:
            raise ValueError("all training motions were excluded")
        generate_training_sample_shards(
            reference=reference,
            physical=physical,
            output_dir=sample_dir,
            motion_indices=train_motions,
            time_stop=args.train_time_stop,
            points_per_problem=args.points_per_problem,
            seed=args.sample_seed,
            overwrite=args.overwrite_samples,
        )
        save_json({
            "excluded_motion_indices": list(exclusions),
            "included_train_motion_indices": list(train_motions),
            "reason": "manual reference convergence audit",
        }, root / "data" / "motion_exclusions.json")
        print(f"saved compact training sample shards under {sample_dir}")


if __name__ == "__main__":
    main()
