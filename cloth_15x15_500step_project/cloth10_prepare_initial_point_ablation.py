"""Prepare nested {1,8,32,128,512,1024} initial-point prefixes.

For every physical time-step problem, sample slot 0 is exactly x_n. Therefore:
- the 1-point experiment contains only x_n;
- every larger experiment contains the same x_n plus N-1 Sobol perturbations;
- all experiment sets are strict nested prefixes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cloth03_solvers_and_models import (
    FULL_STATE_DIM,
    TORCH_DTYPE,
    TRAIN_SOBOL_SEED,
    full_state_from_free_state,
    generate_sobol_points,
    physical_config_from_dict,
    project_fixed_vertices,
)
from cloth_common import load_json, resolve_exclusions, save_json

DEFAULT_COUNTS = (1, 8, 32, 128, 512, 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("cloth_15x15_500step_pipeline")
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--sample-counts", type=int, nargs="+", default=list(DEFAULT_COUNTS)
    )
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--train-time-stop", type=int, default=400)
    parser.add_argument("--time-window", type=int, default=32)
    parser.add_argument("--seed", type=int, default=TRAIN_SOBOL_SEED)
    parser.add_argument("--exclude-motion-indices", type=int, nargs="*", default=[])
    parser.add_argument("--exclusion-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    counts = tuple(args.sample_counts)
    if (
        tuple(sorted(set(counts))) != counts
        or counts[-1] > args.max_points
        or counts[0] != 1
    ):
        raise ValueError(
            "sample-counts must be sorted unique prefixes, begin with 1, and not exceed max-points"
        )

    runtime = load_json(args.root / "data" / "reference" / "runtime_config.json")
    physical = physical_config_from_dict(runtime["physical_config"])
    reference = torch.load(
        args.root / "data" / "reference" / "reference_problems.pt",
        map_location="cpu",
    )
    exclusion_file = args.exclusion_file or (
        args.root / "data" / "motion_exclusions.json"
    )
    excluded = set(
        resolve_exclusions(args.exclude_motion_indices, exclusion_file)
    )
    motions = [index for index in range(16) if index not in excluded]
    output_root = args.output_root or (
        args.root / "data" / "initial_point_ablation" / "max_1024"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for motion_index in motions:
        for time_start in range(0, args.train_time_stop, args.time_window):
            time_stop = min(time_start + args.time_window, args.train_time_stop)
            path = (
                output_root
                / f"motion_{motion_index:03d}"
                / f"time_{time_start:03d}_{time_stop - 1:03d}.pt"
            )
            if path.exists() and not args.overwrite:
                records.append(
                    {
                        "motion_index": motion_index,
                        "time_start": time_start,
                        "time_stop": time_stop,
                        "path": str(path.relative_to(output_root)),
                        "reused": True,
                    }
                )
                continue

            mask = (
                (reference["motion_index"] == motion_index)
                & (reference["time_index"] >= time_start)
                & (reference["time_index"] < time_stop)
            )
            rows = torch.nonzero(mask, as_tuple=False).flatten()
            rows = rows[
                torch.argsort(reference["time_index"].index_select(0, rows))
            ]
            initial = torch.empty(
                (time_stop - time_start, args.max_points, FULL_STATE_DIM),
                dtype=TORCH_DTYPE,
            )
            for local_index, row in enumerate(rows.tolist()):
                time_index = int(reference["time_index"][row])
                initial[local_index, 0] = project_fixed_vertices(
                    reference["p_n_full"][row].reshape(1, -1), physical
                ).squeeze(0)
                remaining = args.max_points - 1
                if remaining > 0:
                    points, _ = generate_sobol_points(
                        count=remaining,
                        center=reference["exact_y_free"][row],
                        radius=float(reference["sampling_radius"][row]),
                        seed=(
                            args.seed
                            + 100_003 * motion_index
                            + 1009 * time_index
                        ),
                        physical=physical,
                        explicit_points=(),
                    )
                    initial[local_index, 1:] = project_fixed_vertices(
                        full_state_from_free_state(points, physical), physical
                    ).cpu()
                print(
                    f"motion={motion_index:03d} time={time_index:03d} "
                    f"points={args.max_points} (slot 0 = x_n)"
                )

            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "initial_y": initial.contiguous(),
                    "metadata": {
                        "format": "window_shard_v2_xn_prefix",
                        "motion_index": motion_index,
                        "time_start": time_start,
                        "time_stop": time_stop,
                        "max_points": args.max_points,
                        "slot_0": "x_n",
                        "remaining_slots": "Sobol perturbations around exact_y",
                        "nested_prefixes": list(counts),
                    },
                },
                path,
            )
            records.append(
                {
                    "motion_index": motion_index,
                    "time_start": time_start,
                    "time_stop": time_stop,
                    "path": str(path.relative_to(output_root)),
                    "reused": False,
                }
            )

    save_json(
        {
            "format": "window_shards_v1",
            "format_detail": "slot_0_is_x_n_then_sobol_perturbations",
            "max_points": args.max_points,
            "points_per_problem": args.max_points,
            "sample_counts": list(counts),
            "motion_indices": motions,
            "train_time_range": [0, args.train_time_stop - 1],
            "time_window": args.time_window,
            "physical_xn_included": True,
            "physical_xn_slot": 0,
            "one_point_experiment": "x_n only",
            "nested_prefixes": True,
            "approx_initial_y_storage_gib": (
                len(motions)
                * args.train_time_stop
                * args.max_points
                * FULL_STATE_DIM
                * 8
                / 2**30
            ),
            "records": records,
        },
        output_root / "manifest.json",
    )
    print(output_root / "manifest.json")


if __name__ == "__main__":
    main()
