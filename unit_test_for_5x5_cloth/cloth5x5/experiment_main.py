from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from .config import RuntimeConfig, default_physical_config
from .constants import (
    DEFAULT_DEVICE,
    DEFAULT_DIAGNOSTIC_INTERVAL,
    DEFAULT_EPOCHS,
    DEFAULT_EVALUATION_BATCH_SIZE,
    DEFAULT_EVALUATION_STEPS,
    DEFAULT_EVAL_POINTS_PER_PROBLEM,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_INITIAL_K,
    DEFAULT_K_INCREASE_AMOUNT,
    DEFAULT_K_INCREASE_INTERVAL,
    DEFAULT_MAX_K,
    DEFAULT_REPORT_STEPS,
    DEFAULT_RESIDUAL_LENGTH_SCALE,
    DEFAULT_SAMPLING_RADIUS_MAX,
    DEFAULT_SAMPLING_RADIUS_MIN,
    DEFAULT_TOTAL_TIME_STEPS,
    DEFAULT_TRAIN_POINTS_PER_PROBLEM,
    DEFAULT_VALIDATION_INTERVAL,
    FIXED_VERTEX_INDICES,
    GRID_COLS,
    GRID_ROWS,
    OOD_TEST_SOBOL_SEED,
    SEEN_EXTRAPOLATION_TEST_SOBOL_SEED,
    SEEN_EXTRAPOLATION_TIME_INDICES,
    SEEN_INTERPOLATION_TEST_SOBOL_SEED,
    SEEN_INTERPOLATION_TIME_INDICES,
    SPRING_EDGES,
    TRAIN_SOBOL_SEED,
    TRAIN_TIME_INDICES,
    TRIANGLE_FACES,
    UNSEEN_ID_TEST_SOBOL_SEED,
    UNSEEN_TEST_TIME_INDICES,
    VALIDATION_SOBOL_SEED,
    VALIDATION_TIME_INDICES,
)
from .dataset import (
    build_dataset_for_motion_times,
    build_special_state_dataset,
    dataset_to_serializable_dict,
)
from .evaluate import (
    evaluate_solver_on_dataset,
    plot_gradient_descent_step_size_selection,
    select_gradient_descent_step_size,
)
from .io import create_output_directory, resolve_device, save_json, validate_device
from .motions import build_motion_catalogue
from .plotting import plot_reference_motion_overview, problem_to_record, run_physics_checks, select_hard_ood_case
from .reference import generate_all_reference_sequences, problem_lookup
from .train_loop import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5x5 triangular-cloth multi-motion learned optimizer")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--total-time-steps", type=int, default=DEFAULT_TOTAL_TIME_STEPS)
    parser.add_argument("--train-points-per-problem", type=int, default=DEFAULT_TRAIN_POINTS_PER_PROBLEM)
    parser.add_argument("--eval-points-per-problem", type=int, default=DEFAULT_EVAL_POINTS_PER_PROBLEM)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--diagnostic-interval", type=int, default=DEFAULT_DIAGNOSTIC_INTERVAL)
    parser.add_argument("--evaluation-steps", type=int, default=DEFAULT_EVALUATION_STEPS)
    parser.add_argument("--evaluation-batch-size", type=int, default=DEFAULT_EVALUATION_BATCH_SIZE)
    parser.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    parser.add_argument("--k-increase-interval", type=int, default=DEFAULT_K_INCREASE_INTERVAL)
    parser.add_argument("--k-increase-amount", type=int, default=DEFAULT_K_INCREASE_AMOUNT)
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--report-steps", type=int, nargs="+", default=list(DEFAULT_REPORT_STEPS))
    parser.add_argument("--residual-length-scale", type=float, default=DEFAULT_RESIDUAL_LENGTH_SCALE)
    parser.add_argument("--gradient-clip-norm", type=float, default=DEFAULT_GRADIENT_CLIP_NORM)
    parser.add_argument("--sampling-radius-min", type=float, default=DEFAULT_SAMPLING_RADIUS_MIN)
    parser.add_argument("--sampling-radius-max", type=float, default=DEFAULT_SAMPLING_RADIUS_MAX)
    parser.add_argument("--skip-single-motion-baseline", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--save-datasets", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> RuntimeConfig:
    positive_ints = {
        "total_time_steps": args.total_time_steps,
        "train_points_per_problem": args.train_points_per_problem,
        "eval_points_per_problem": args.eval_points_per_problem,
        "epochs": args.epochs,
        "validation_interval": args.validation_interval,
        "diagnostic_interval": args.diagnostic_interval,
        "evaluation_steps": args.evaluation_steps,
        "evaluation_batch_size": args.evaluation_batch_size,
        "initial_k": args.initial_k,
        "k_increase_interval": args.k_increase_interval,
        "k_increase_amount": args.k_increase_amount,
        "max_k": args.max_k,
    }
    for name, value in positive_ints.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.total_time_steps) != 100:
        raise ValueError("The confirmed experiment requires exactly 100 time steps per motion")
    if int(args.initial_k) > int(args.max_k):
        raise ValueError("initial_k cannot exceed max_k")
    if float(args.sampling_radius_min) <= 0 or float(args.sampling_radius_max) < float(args.sampling_radius_min):
        raise ValueError("Invalid sampling-radius clamp")
    report_steps = tuple(
        sorted(
            set(
                [int(s) for s in args.report_steps if 0 < int(s) <= int(args.evaluation_steps)]
                + [int(args.evaluation_steps)]
            )
        )
    )
    resolved_device = resolve_device(str(args.device))
    return RuntimeConfig(
        total_time_steps=int(args.total_time_steps),
        train_points_per_problem=int(args.train_points_per_problem),
        eval_points_per_problem=int(args.eval_points_per_problem),
        epochs=int(args.epochs),
        validation_interval=int(args.validation_interval),
        diagnostic_interval=int(args.diagnostic_interval),
        evaluation_steps=int(args.evaluation_steps),
        evaluation_batch_size=int(args.evaluation_batch_size),
        initial_k=int(args.initial_k),
        k_increase_interval=int(args.k_increase_interval),
        k_increase_amount=int(args.k_increase_amount),
        max_k=int(args.max_k),
        report_steps=report_steps,
        residual_length_scale=float(args.residual_length_scale),
        gradient_clip_norm=float(args.gradient_clip_norm),
        sampling_radius_min=float(args.sampling_radius_min),
        sampling_radius_max=float(args.sampling_radius_max),
        device=str(resolved_device),
        run_single_motion_baseline=not bool(args.skip_single_motion_baseline),
        skip_plots=bool(args.skip_plots),
        save_datasets=bool(args.save_datasets),
    )


def main(*, script_file: Path) -> None:
    config = validate_args(parse_args())
    physical = default_physical_config()
    motions, motion_split = build_motion_catalogue(physical)
    output_dir = create_output_directory(script_file=script_file)
    device = torch.device(config.device)
    validate_device(device)

    physics_checks = run_physics_checks(physical, motions[0])
    print(f"Output directory: {output_dir}")
    print(f"Physics checks: {physics_checks}")
    if not config.skip_plots:
        plot_reference_motion_overview(motions, output_dir / "motion_catalogue_overview.png")

    problems = generate_all_reference_sequences(physical, motions, config)
    lookup = problem_lookup(problems)
    problems_by_index = {p.index: p for p in problems}

    save_json(
        {
            "runtime_config": asdict(config),
            "physical_config": asdict(physical),
            "motion_split": asdict(motion_split),
            "motions": [asdict(m) for m in motions],
            "fixed_vertex_indices": list(FIXED_VERTEX_INDICES),
            "fixed_positions": [list(p) for p in physical.fixed_positions],
            "grid_rows": GRID_ROWS,
            "grid_cols": GRID_COLS,
            "spring_edges": [list(e) for e in SPRING_EDGES],
            "triangle_faces": [list(f) for f in TRIANGLE_FACES],
            "physics_checks": physics_checks,
        },
        output_dir / "runtime_config.json",
    )
    save_json(
        {"problems": [problem_to_record(p) for p in problems]},
        output_dir / "reference_time_step_problems.json",
    )
    save_json(
        {"motions": [asdict(m) for m in motions], "motion_split": asdict(motion_split)},
        output_dir / "motion_catalogue.json",
    )

    multi_training = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.train_motion_indices,
        time_indices=TRAIN_TIME_INDICES,
        points_per_problem=config.train_points_per_problem,
        base_seed=TRAIN_SOBOL_SEED,
        role="multi_motion_training",
        physical=physical,
        include_explicit_train_points=True,
    )
    single_points_per_problem = config.train_points_per_problem * len(motion_split.train_motion_indices)
    single_training = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=(0,),
        time_indices=TRAIN_TIME_INDICES,
        points_per_problem=single_points_per_problem,
        base_seed=TRAIN_SOBOL_SEED + 1_000_000,
        role="single_motion_equal_budget_training",
        physical=physical,
        include_explicit_train_points=True,
    )
    if len(multi_training) != len(single_training):
        raise AssertionError("Single-motion baseline must have the same number of training states")

    validation = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.validation_motion_indices,
        time_indices=VALIDATION_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=VALIDATION_SOBOL_SEED,
        role="unseen_motion_validation",
        physical=physical,
        include_explicit_train_points=False,
    )
    seen_interp = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.train_motion_indices,
        time_indices=SEEN_INTERPOLATION_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=SEEN_INTERPOLATION_TEST_SOBOL_SEED,
        role="seen_motion_temporal_interpolation",
        physical=physical,
        include_explicit_train_points=False,
    )
    seen_extrap = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.train_motion_indices,
        time_indices=SEEN_EXTRAPOLATION_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=SEEN_EXTRAPOLATION_TEST_SOBOL_SEED,
        role="seen_motion_temporal_extrapolation",
        physical=physical,
        include_explicit_train_points=False,
    )
    unseen_id = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.id_test_motion_indices,
        time_indices=UNSEEN_TEST_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=UNSEEN_ID_TEST_SOBOL_SEED,
        role="unseen_id_test",
        physical=physical,
        include_explicit_train_points=False,
    )
    ood_test = build_dataset_for_motion_times(
        lookup=lookup,
        motion_indices=motion_split.ood_test_motion_indices,
        time_indices=UNSEEN_TEST_TIME_INDICES,
        points_per_problem=config.eval_points_per_problem,
        base_seed=OOD_TEST_SOBOL_SEED,
        role="ood_test",
        physical=physical,
        include_explicit_train_points=False,
    )
    current_seen = build_special_state_dataset(
        lookup=lookup,
        motion_indices=motion_split.train_motion_indices,
        time_indices=SEEN_INTERPOLATION_TIME_INDICES,
        state="current",
        role="current_state_seen_motion",
    )
    current_id = build_special_state_dataset(
        lookup=lookup,
        motion_indices=motion_split.id_test_motion_indices,
        time_indices=UNSEEN_TEST_TIME_INDICES,
        state="current",
        role="current_state_unseen_id",
    )
    current_ood = build_special_state_dataset(
        lookup=lookup,
        motion_indices=motion_split.ood_test_motion_indices,
        time_indices=UNSEEN_TEST_TIME_INDICES,
        state="current",
        role="current_state_ood",
    )
    evaluation_datasets = {
        "seen_motion_temporal_interpolation": seen_interp,
        "seen_motion_temporal_extrapolation": seen_extrap,
        "unseen_id_test": unseen_id,
        "ood_test": ood_test,
        "current_state_seen_motion": current_seen,
        "current_state_unseen_id": current_id,
        "current_state_ood": current_ood,
    }

    hard_case = select_hard_ood_case(ood_test, problems_by_index, physical)
    save_json(hard_case, output_dir / "hard_case_selection.json")

    gd_step_size, gd_selection = select_gradient_descent_step_size(
        validation=validation,
        physical=physical,
        config=config,
        device=device,
    )
    save_json(gd_selection, output_dir / "gradient_descent_step_selection.json")
    if not config.skip_plots:
        plot_gradient_descent_step_size_selection(
            gd_selection,
            output_dir / "gradient_descent_step_size_selection.png",
        )
    print(f"Selected gradient-descent step size: {gd_step_size:.3e}")

    shared_baselines: dict[str, dict] = {}
    for name, dataset in evaluation_datasets.items():
        print(f"Evaluating GD and Newton on {name} ...")
        shared_baselines[name] = {
            "gradient_descent": evaluate_solver_on_dataset(
                solver="gradient_descent",
                dataset_cpu=dataset,
                physical=physical,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                report_steps=config.report_steps,
                device=device,
                gd_step_size=gd_step_size,
            ),
            "full_newton": evaluate_solver_on_dataset(
                solver="full_newton",
                dataset_cpu=dataset,
                physical=physical,
                steps=config.evaluation_steps,
                batch_size=config.evaluation_batch_size,
                report_steps=config.report_steps,
                device=device,
            ),
        }
    save_json(shared_baselines, output_dir / "shared_gd_newton_baselines.json")

    if config.save_datasets:
        torch.save(
            {
                "multi_motion_training": dataset_to_serializable_dict(multi_training),
                "single_motion_equal_budget_training": dataset_to_serializable_dict(single_training),
                "validation": dataset_to_serializable_dict(validation),
                **{name: dataset_to_serializable_dict(dataset) for name, dataset in evaluation_datasets.items()},
            },
            output_dir / "generated_datasets.pt",
        )

    reports = [
        run_experiment(
            experiment_name="multi_motion",
            training_cpu=multi_training,
            validation_cpu=validation,
            evaluation_datasets=evaluation_datasets,
            output_dir=output_dir,
            config=config,
            physical=physical,
            gd_step_size=gd_step_size,
            shared_baselines=shared_baselines,
        )
    ]
    if config.run_single_motion_baseline:
        reports.append(
            run_experiment(
                experiment_name="single_motion_equal_budget_baseline",
                training_cpu=single_training,
                validation_cpu=validation,
                evaluation_datasets=evaluation_datasets,
                output_dir=output_dir,
                config=config,
                physical=physical,
                gd_step_size=gd_step_size,
                shared_baselines=shared_baselines,
            )
        )

    summary = {
        "experiment_type": "fixed_left_edge_5x5_cloth_multi_motion_generalization",
        "runtime_config": asdict(config),
        "physical_config": asdict(physical),
        "motion_split": asdict(motion_split),
        "motions": [asdict(m) for m in motions],
        "physics_checks": physics_checks,
        "gradient_descent_selection": gd_selection,
        "hard_case_selection": hard_case,
        "shared_baselines": shared_baselines,
        "experiments": reports,
        "metric_policy": {
            "pooled_statistics": ["mean", "median", "p95", "max"],
            "motion_boundary_statistics": ["worst-motion p95", "worst-motion max"],
        },
    }
    save_json(summary, output_dir / "all_experiments_summary.json")
    print("\nCompleted all experiments.")
    print(f"Summary: {output_dir / 'all_experiments_summary.json'}")
    print(
        "500-frame rollout input: "
        + str(output_dir / "multi_motion" / "best_validation_model_state_dict.pt")
    )
