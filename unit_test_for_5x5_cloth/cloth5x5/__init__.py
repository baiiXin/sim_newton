"""Fixed-left-edge 5x5 triangular-cloth multi-motion learned optimizer package."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from .config import (
    DatasetBundle,
    MotionSpec,
    MotionSplit,
    PhysicalConfig,
    RuntimeConfig,
    TimeStepProblem,
    default_physical_config,
    finite_plot_value,
    get_k_for_epoch,
    physical_config_from_dict,
)
from .constants import *  # noqa: F403
from .dataset import (
    build_dataset_for_motion_times,
    build_problem_dataset,
    build_special_state_dataset,
    concatenate_datasets,
    dataset_to_serializable_dict,
    generate_sobol_points,
)
from .evaluate import (
    evaluate_solver_on_dataset,
    select_gradient_descent_step_size,
    validation_selection_key,
)
from .io import (
    create_output_directory,
    load_json,
    resolve_device,
    save_json,
    state_dict_to_cpu,
    validate_device,
)
from .model import MLPOptimizer, apply_model_update, physical_energy_scale
from .motions import build_motion_catalogue, make_motion_spec
from .physics import (
    advance_physical_state,
    apply_gradient_descent_update,
    apply_newton_update,
    free_state_from_full,
    full_positions_from_free,
    make_q_free,
    reshape_free,
    solve_reference_solution,
    spring_lengths_from_free,
    stationarity_residual,
    stationarity_residual_norm,
    variational_energy,
    variational_hessian,
)
from .reference import generate_all_reference_sequences, generate_reference_sequence_for_motion, problem_lookup
from .solvers import apply_solver_step, run_solver_steps
from .train_loop import one_step_diagnostics, run_experiment

__all__ = [
    "DatasetBundle",
    "MLPOptimizer",
    "MotionSpec",
    "MotionSplit",
    "PhysicalConfig",
    "RuntimeConfig",
    "TimeStepProblem",
    "advance_physical_state",
    "apply_gradient_descent_update",
    "apply_model_update",
    "apply_newton_update",
    "apply_solver_step",
    "build_dataset_for_motion_times",
    "build_motion_catalogue",
    "build_problem_dataset",
    "build_special_state_dataset",
    "concatenate_datasets",
    "create_output_directory",
    "dataset_to_serializable_dict",
    "default_physical_config",
    "evaluate_solver_on_dataset",
    "finite_plot_value",
    "free_state_from_full",
    "full_positions_from_free",
    "generate_all_reference_sequences",
    "generate_reference_sequence_for_motion",
    "generate_sobol_points",
    "get_k_for_epoch",
    "load_json",
    "make_motion_spec",
    "make_q_free",
    "one_step_diagnostics",
    "physical_config_from_dict",
    "physical_energy_scale",
    "problem_lookup",
    "reshape_free",
    "resolve_device",
    "run_experiment",
    "run_solver_steps",
    "save_json",
    "select_gradient_descent_step_size",
    "solve_reference_solution",
    "spring_lengths_from_free",
    "state_dict_to_cpu",
    "stationarity_residual",
    "stationarity_residual_norm",
    "validate_device",
    "validation_selection_key",
    "variational_energy",
    "variational_hessian",
]
