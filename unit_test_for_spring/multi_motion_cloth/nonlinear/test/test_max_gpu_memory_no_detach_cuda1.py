#!/usr/bin/env python3
"""Maximum-GPU-memory preflight for full-gradient history input.

This test deliberately keeps the autograd graph through both the previous
residual and the previous update.  It imports the current default-initialized
history-input experiment and replaces only ``apply_model_update`` at runtime,
so the tested update is::

    next_state.previous_residual = current_residual
    next_state.previous_update = delta

with no ``detach()`` on either tensor.

The production experiment file itself is not modified by this preflight.

Original preflight description:

The default run targets cuda:1 and tests the largest experiment settings:

* full training batch: 8192 states
* history-input MLP: depth=10, width=256, no bias, float64
* unrolled learned-optimizer steps: K=30
* activations: identity, ReLU, and Tanh (one isolated subprocess each)
* one full-batch Newton update, which exercises the batched 69x69 Hessian solve

Each case runs in its own subprocess so that CUDA caching/fragmentation from one
case cannot affect another case.  The script writes JSON and CSV summaries and
returns a non-zero exit code when any requested case runs out of memory.

Example
-------
python test_max_gpu_memory_no_detach_cuda1.py

To select a different visible GPU:
python test_max_gpu_memory_no_detach_cuda1.py --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch


GIB = 1024**3
DEFAULT_EXPERIMENT_SCRIPT = (
    Path(__file__).resolve().parent
    / "fixed_left_edge_5x5_cloth_history_input_default_init_ablation.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Test maximum CUDA memory with no detach on previous residual "
                     "or previous update.")
    )
    parser.add_argument("--experiment-script", type=Path, default=DEFAULT_EXPERIMENT_SCRIPT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--activations",
        nargs="+",
        choices=("identity", "relu", "tanh"),
        default=("identity", "relu", "tanh"),
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--residual-length-scale", type=float, default=5e-2)
    parser.add_argument("--synthetic-radius", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--skip-newton",
        action="store_true",
        help="Skip the full-batch Newton Hessian/solve memory test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "max_gpu_memory_no_detach_test_cuda1",
    )

    # Internal worker arguments. Users normally do not set these.
    parser.add_argument("--worker-case", choices=("training", "newton"), default=None)
    parser.add_argument("--worker-activation", default=None)
    parser.add_argument("--worker-result", type=Path, default=None)
    return parser.parse_args()


def validate_positive(args: argparse.Namespace) -> None:
    for name in ("batch_size", "k", "depth", "width"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.gradient_clip_norm <= 0:
        raise ValueError("--gradient-clip-norm must be positive")
    if args.residual_length_scale <= 0:
        raise ValueError("--residual-length-scale must be positive")
    if args.synthetic_radius <= 0:
        raise ValueError("--synthetic-radius must be positive")


def load_experiment_module(path: Path):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment script not found: {path}")
    module_name = f"cloth_memory_target_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import experiment script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def enable_full_gradient_history(exp) -> None:
    """Replace the detached state transition with a fully differentiable one."""

    def apply_model_update_no_detach(
        model,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        physical,
        optimizer_state,
    ):
        delta, current_residual = model(
            y, q, masses, optimizer_state, physical=physical
        )
        next_state = exp.LearnedOptimizerState(
            previous_residual=current_residual,
            previous_update=delta,
        )
        return y + delta, delta, next_state

    exp.apply_model_update = apply_model_update_no_detach


def history_gradient_connectivity_probe(
    exp,
    *,
    model,
    tensors: dict[str, torch.Tensor],
    physical,
) -> dict[str, Any]:
    """Confirm that history tensors keep graph connections after two steps."""
    y = tensors["initial_y"][:2]
    q = tensors["q"][:2]
    masses = tensors["masses"][:2]
    state0 = exp.LearnedOptimizerState.zeros_like(y)
    y1, delta1, state1 = exp.apply_model_update(
        model, y, q, masses, physical, state0
    )
    y2, delta2, state2 = exp.apply_model_update(
        model, y1, q, masses, physical, state1
    )
    probe_loss = y2.square().mean() + delta2.square().mean()
    probe_grads = torch.autograd.grad(
        probe_loss,
        tuple(model.parameters()),
        allow_unused=True,
        retain_graph=False,
        create_graph=False,
    )
    finite_grad_count = sum(
        grad is not None and bool(torch.isfinite(grad).all())
        for grad in probe_grads
    )
    result = {
        "step1_previous_residual_requires_grad": bool(
            state1.previous_residual.requires_grad
        ),
        "step1_previous_update_requires_grad": bool(
            state1.previous_update.requires_grad
        ),
        "step2_previous_residual_requires_grad": bool(
            state2.previous_residual.requires_grad
        ),
        "step2_previous_update_requires_grad": bool(
            state2.previous_update.requires_grad
        ),
        "step2_previous_residual_has_grad_fn": state2.previous_residual.grad_fn
        is not None,
        "step2_previous_update_has_grad_fn": state2.previous_update.grad_fn
        is not None,
        "finite_parameter_gradient_tensors": int(finite_grad_count),
        "parameter_tensor_count": len(probe_grads),
    }
    if not result["step1_previous_update_requires_grad"]:
        raise RuntimeError(
            "Full-gradient history check failed: first previous update is detached"
        )
    if not (
        result["step2_previous_residual_requires_grad"]
        and result["step2_previous_update_requires_grad"]
        and result["step2_previous_residual_has_grad_fn"]
        and result["step2_previous_update_has_grad_fn"]
    ):
        raise RuntimeError(
            "Full-gradient history check failed after the second iteration"
        )
    if finite_grad_count == 0:
        raise RuntimeError(
            "Full-gradient history probe did not reach any model parameter"
        )
    return result


def device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("This preflight requires a CUDA device")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def validate_cuda_device(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable: torch.cuda.is_available() is False")
    index = device_index(device)
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise RuntimeError(f"Requested {device}, but only {count} CUDA device(s) are visible")
    torch.cuda.set_device(index)


def gib(value: int | float) -> float:
    return float(value) / GIB


def memory_snapshot(device: torch.device) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "device_name": props.name,
        "total_memory_bytes": int(total_bytes),
        "total_memory_gib": gib(total_bytes),
        "free_memory_bytes": int(free_bytes),
        "free_memory_gib": gib(free_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "allocated_gib": gib(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "reserved_gib": gib(torch.cuda.memory_reserved(device)),
    }


def synchronize_and_clean(device: torch.device, *, empty_cache: bool = True) -> None:
    torch.cuda.synchronize(device)
    gc.collect()
    if empty_cache:
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)


def build_synthetic_training_tensors(
    exp,
    *,
    batch_size: int,
    radius: float,
    seed: int,
    device: torch.device,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Build physically valid tensors with exactly the production batch shapes."""
    physical = exp.default_physical_config()
    dtype = exp.TORCH_DTYPE

    base_full = torch.tensor(physical.p0, dtype=dtype, device="cpu")
    base_free = exp.free_state_from_full(base_full)
    zero_velocity = torch.zeros_like(base_full)
    q_base = exp.make_q_free(base_full, zero_velocity, physical)
    free_masses = torch.tensor(
        [physical.masses[i] for i in exp.FREE_VERTEX_INDICES],
        dtype=dtype,
        device="cpu",
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perturbation = (
        2.0
        * torch.rand(
            (batch_size, exp.FREE_STATE_DIM),
            dtype=dtype,
            generator=generator,
            device="cpu",
        )
        - 1.0
    ) * radius
    q_perturbation = (
        2.0
        * torch.rand(
            (batch_size, exp.FREE_STATE_DIM),
            dtype=dtype,
            generator=generator,
            device="cpu",
        )
        - 1.0
    ) * (0.25 * radius)

    initial_y = base_free.unsqueeze(0) + perturbation
    q = q_base.unsqueeze(0) + q_perturbation
    masses = free_masses.unsqueeze(0).expand(batch_size, -1).clone()
    exact_y = base_free.unsqueeze(0).expand(batch_size, -1).clone()

    # Keep the same resident tensor categories as DatasetBundle.to(device).
    problem_index = torch.arange(batch_size, dtype=torch.long) % 256
    motion_index = torch.arange(batch_size, dtype=torch.long) % 16
    time_index = torch.arange(batch_size, dtype=torch.long) % 100

    tensors = {
        "initial_y": initial_y.to(device=device),
        "q": q.to(device=device),
        "masses": masses.to(device=device),
        "exact_y": exact_y.to(device=device),
        "problem_index": problem_index.to(device=device),
        "motion_index": motion_index.to(device=device),
        "time_index": time_index.to(device=device),
    }
    return physical, tensors


def execute_training_step(
    exp,
    *,
    model,
    optimizer,
    tensors: dict[str, torch.Tensor],
    physical,
    k: int,
    gradient_clip_norm: float,
    residual_length_scale: float,
) -> dict[str, float]:
    initial_y = tensors["initial_y"]
    q = tensors["q"]
    masses = tensors["masses"]
    exact_y = tensors["exact_y"]

    # In the production script these are computed once and remain resident.
    initial_energy = exp.variational_energy(initial_y, q, masses, physical).detach()
    exact_energy = exp.variational_energy(exact_y, q, masses, physical).detach()
    energy_scale = exp.physical_energy_scale(masses, physical, residual_length_scale)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    y = initial_y
    state = exp.LearnedOptimizerState.zeros_like(y)
    objective_sum = torch.zeros((), dtype=exp.TORCH_DTYPE, device=y.device)
    energy_gap_sum = torch.zeros((), dtype=exp.TORCH_DTYPE, device=y.device)
    final_step_objective = torch.zeros((), dtype=exp.TORCH_DTYPE, device=y.device)

    for _ in range(k):
        y, _, state = exp.apply_model_update(model, y, q, masses, physical, state)
        energy = exp.variational_energy(y, q, masses, physical)
        step_objective = ((energy - initial_energy) / energy_scale).mean()
        step_energy_gap = (energy - exact_energy).mean()
        objective_sum = objective_sum + step_objective
        energy_gap_sum = energy_gap_sum + step_energy_gap
        final_step_objective = step_objective

    objective_mean = objective_sum / float(k)
    energy_gap_mean = energy_gap_sum / float(k)
    if not bool(torch.isfinite(objective_mean)):
        raise RuntimeError("The synthetic maximum-memory objective became non-finite")
    objective_mean.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    if not bool(torch.isfinite(grad_norm)):
        raise RuntimeError("The synthetic maximum-memory gradient became non-finite")
    optimizer.step()

    return {
        "objective_mean": float(objective_mean.detach().item()),
        "energy_gap_mean": float(energy_gap_mean.detach().item()),
        "final_step_objective": float(final_step_objective.detach().item()),
        "gradient_norm_before_clip": float(grad_norm.detach().item()),
    }


def add_peak_fields(record: dict[str, Any], device: torch.device) -> None:
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    total = int(record["before_test"]["total_memory_bytes"])
    record.update(
        peak_allocated_bytes=peak_allocated,
        peak_allocated_gib=gib(peak_allocated),
        peak_reserved_bytes=peak_reserved,
        peak_reserved_gib=gib(peak_reserved),
        peak_allocated_fraction_of_total=peak_allocated / max(total, 1),
        peak_reserved_fraction_of_total=peak_reserved / max(total, 1),
        reserved_headroom_bytes=max(total - peak_reserved, 0),
        reserved_headroom_gib=gib(max(total - peak_reserved, 0)),
    )


def run_training_worker(args: argparse.Namespace, exp, device: torch.device) -> dict[str, Any]:
    activation = str(args.worker_activation)
    physical, tensors = build_synthetic_training_tensors(
        exp,
        batch_size=args.batch_size,
        radius=args.synthetic_radius,
        seed=args.seed,
        device=device,
    )
    spec = exp.ModelSpec(
        activation=activation,
        depth=args.depth,
        width=args.width,
        use_bias=exp.USE_BIAS,
    )
    torch.manual_seed(exp.MODEL_RANDOM_SEED)
    torch.cuda.manual_seed_all(exp.MODEL_RANDOM_SEED)
    model = exp.MLPOptimizer(args.residual_length_scale, spec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=exp.LEARNING_RATE)

    connectivity = history_gradient_connectivity_probe(
        exp, model=model, tensors=tensors, physical=physical
    )
    model.zero_grad(set_to_none=True)
    synchronize_and_clean(device)

    record: dict[str, Any] = {
        "case": "learned_training",
        "activation": activation,
        "batch_size": args.batch_size,
        "K": args.k,
        "depth": args.depth,
        "width": args.width,
        "parameter_count": model.parameter_count,
        "dtype": str(exp.TORCH_DTYPE),
        "use_bias": bool(exp.USE_BIAS),
        "history_gradient_policy": (
            "full gradient through previous residual and previous update; no detach"
        ),
        "history_gradient_connectivity": connectivity,
    }
    record["after_setup"] = memory_snapshot(device)

    # K=1 warm-up creates the Adam moment tensors before the measured K=30 pass.
    warmup_metrics = execute_training_step(
        exp,
        model=model,
        optimizer=optimizer,
        tensors=tensors,
        physical=physical,
        k=1,
        gradient_clip_norm=args.gradient_clip_norm,
        residual_length_scale=args.residual_length_scale,
    )
    optimizer.zero_grad(set_to_none=True)
    synchronize_and_clean(device)
    record["warmup_metrics"] = warmup_metrics
    record["before_test"] = memory_snapshot(device)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    metrics = execute_training_step(
        exp,
        model=model,
        optimizer=optimizer,
        tensors=tensors,
        physical=physical,
        k=args.k,
        gradient_clip_norm=args.gradient_clip_norm,
        residual_length_scale=args.residual_length_scale,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    record["status"] = "success"
    record["passed"] = True
    record["elapsed_seconds"] = elapsed
    record["metrics"] = metrics
    add_peak_fields(record, device)
    record["after_test"] = memory_snapshot(device)
    return record


def run_newton_worker(args: argparse.Namespace, exp, device: torch.device) -> dict[str, Any]:
    physical, tensors = build_synthetic_training_tensors(
        exp,
        batch_size=args.batch_size,
        radius=args.synthetic_radius,
        seed=args.seed + 1,
        device=device,
    )
    record: dict[str, Any] = {
        "case": "full_newton_update",
        "activation": None,
        "batch_size": args.batch_size,
        "K": 1,
        "state_dimension": int(exp.FREE_STATE_DIM),
        "hessian_shape": [args.batch_size, exp.FREE_STATE_DIM, exp.FREE_STATE_DIM],
        "dtype": str(exp.TORCH_DTYPE),
    }
    record["after_setup"] = memory_snapshot(device)

    # Small warm-up initializes CUDA linear-algebra workspaces without allocating
    # the production-size Hessian.
    with torch.no_grad():
        exp.apply_newton_update(
            tensors["initial_y"][:1],
            tensors["q"][:1],
            tensors["masses"][:1],
            physical,
        )
    synchronize_and_clean(device)
    record["before_test"] = memory_snapshot(device)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad():
        y_next, delta = exp.apply_newton_update(
            tensors["initial_y"],
            tensors["q"],
            tensors["masses"],
            physical,
        )
        # Force both returned tensors to be materialized before reading the peak.
        delta_norm = float(torch.linalg.vector_norm(delta, dim=-1).mean().item())
        finite = bool(torch.isfinite(y_next).all() and torch.isfinite(delta).all())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if not finite:
        raise RuntimeError("Full-batch Newton update produced non-finite values")

    record["status"] = "success"
    record["passed"] = True
    record["elapsed_seconds"] = elapsed
    record["mean_delta_norm"] = delta_norm
    add_peak_fields(record, device)
    record["after_test"] = memory_snapshot(device)
    return record


def failure_record(
    *,
    args: argparse.Namespace,
    case: str,
    activation: str | None,
    device: torch.device,
    exc: BaseException,
) -> dict[str, Any]:
    text = f"{type(exc).__name__}: {exc}"
    is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text.lower()
    record: dict[str, Any] = {
        "case": "learned_training" if case == "training" else "full_newton_update",
        "activation": activation,
        "batch_size": args.batch_size,
        "K": args.k if case == "training" else 1,
        "depth": args.depth if case == "training" else None,
        "width": args.width if case == "training" else None,
        "status": "oom" if is_oom else "failed",
        "passed": False,
        "error": text,
        "traceback": traceback.format_exc(),
    }
    if torch.cuda.is_available():
        try:
            record["failure_memory"] = memory_snapshot(device)
            record["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            record["peak_allocated_gib"] = gib(torch.cuda.max_memory_allocated(device))
            record["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
            record["peak_reserved_gib"] = gib(torch.cuda.max_memory_reserved(device))
        except Exception:
            pass
    return record


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)


def run_worker(args: argparse.Namespace) -> int:
    if args.worker_result is None:
        raise ValueError("--worker-result is required in worker mode")
    device = torch.device(args.device)
    record: dict[str, Any]
    try:
        validate_cuda_device(device)
        exp = load_experiment_module(args.experiment_script)
        enable_full_gradient_history(exp)
        if args.worker_case == "training":
            if args.worker_activation not in {"identity", "relu", "tanh"}:
                raise ValueError("A valid --worker-activation is required")
            record = run_training_worker(args, exp, device)
        elif args.worker_case == "newton":
            record = run_newton_worker(args, exp, device)
        else:
            raise ValueError(f"Unsupported worker case: {args.worker_case}")
    except BaseException as exc:
        record = failure_record(
            args=args,
            case=str(args.worker_case),
            activation=args.worker_activation,
            device=device,
            exc=exc,
        )
    write_json(record, args.worker_result)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0 if record.get("passed") else 2


def worker_command(
    args: argparse.Namespace,
    *,
    case: str,
    activation: str | None,
    result_path: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--experiment-script",
        str(args.experiment_script.expanduser().resolve()),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--k",
        str(args.k),
        "--depth",
        str(args.depth),
        "--width",
        str(args.width),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--residual-length-scale",
        str(args.residual_length_scale),
        "--synthetic-radius",
        str(args.synthetic_radius),
        "--seed",
        str(args.seed),
        "--worker-case",
        case,
        "--worker-result",
        str(result_path),
    ]
    if activation is not None:
        command.extend(["--worker-activation", activation])
    return command


def flatten_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    before = record.get("before_test", {}) or {}
    after = record.get("after_test", {}) or {}
    return {
        "case": record.get("case"),
        "activation": record.get("activation"),
        "status": record.get("status"),
        "passed": record.get("passed"),
        "batch_size": record.get("batch_size"),
        "K": record.get("K"),
        "depth": record.get("depth"),
        "width": record.get("width"),
        "parameter_count": record.get("parameter_count"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "total_memory_gib": before.get("total_memory_gib"),
        "free_memory_before_gib": before.get("free_memory_gib"),
        "allocated_before_gib": before.get("allocated_gib"),
        "reserved_before_gib": before.get("reserved_gib"),
        "peak_allocated_gib": record.get("peak_allocated_gib"),
        "peak_reserved_gib": record.get("peak_reserved_gib"),
        "peak_reserved_fraction_of_total": record.get("peak_reserved_fraction_of_total"),
        "reserved_headroom_gib": record.get("reserved_headroom_gib"),
        "allocated_after_gib": after.get("allocated_gib"),
        "reserved_after_gib": after.get("reserved_gib"),
        "error": record.get("error"),
    }


def print_summary(records: Sequence[dict[str, Any]], device: str) -> None:
    print("\n" + "=" * 118)
    print(f"Full-gradient-history maximum-memory preflight summary on {device}")
    print("=" * 118)
    header = (
        f"{'case':<23} {'activation':<10} {'status':<9} "
        f"{'peak alloc':>12} {'peak reserv':>12} {'headroom':>11} {'time':>9}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        peak_alloc = record.get("peak_allocated_gib")
        peak_res = record.get("peak_reserved_gib")
        headroom = record.get("reserved_headroom_gib")
        elapsed = record.get("elapsed_seconds")
        print(
            f"{str(record.get('case')):<23} "
            f"{str(record.get('activation') or '-'):<10} "
            f"{str(record.get('status')):<9} "
            f"{(f'{peak_alloc:.3f} GiB' if isinstance(peak_alloc, (int, float)) else '-'):>12} "
            f"{(f'{peak_res:.3f} GiB' if isinstance(peak_res, (int, float)) else '-'):>12} "
            f"{(f'{headroom:.3f} GiB' if isinstance(headroom, (int, float)) else '-'):>11} "
            f"{(f'{elapsed:.2f}s' if isinstance(elapsed, (int, float)) else '-'):>9}"
        )
    successful = [r for r in records if r.get("passed")]
    failed = [r for r in records if not r.get("passed")]
    if successful:
        worst = max(successful, key=lambda r: float(r.get("peak_reserved_gib", -1.0)))
        print(
            "\nLargest successful peak: "
            f"{worst.get('case')} / {worst.get('activation') or '-'} = "
            f"{worst.get('peak_reserved_gib'):.3f} GiB reserved "
            f"({100.0 * worst.get('peak_reserved_fraction_of_total', 0.0):.1f}% of GPU)."
        )
    if failed:
        print(f"Result: FAIL — {len(failed)} requested case(s) did not fit or failed.")
    else:
        print("Result: PASS — every requested maximum-memory case completed without CUDA OOM.")


def run_parent(args: argparse.Namespace) -> int:
    args.experiment_script = args.experiment_script.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.experiment_script.is_file():
        raise FileNotFoundError(f"Experiment script not found: {args.experiment_script}")

    cases: list[tuple[str, str | None]] = [
        ("training", activation) for activation in args.activations
    ]
    if not args.skip_newton:
        cases.append(("newton", None))

    records: list[dict[str, Any]] = []
    for case, activation in cases:
        label = f"{case}_{activation}" if activation else case
        result_path = args.output_dir / f"{label}.json"
        if result_path.exists():
            result_path.unlink()
        print("\n" + "#" * 118)
        print(
            f"Running isolated case: {label}; device={args.device}; "
            f"batch={args.batch_size}; K={args.k}; depth={args.depth}; width={args.width}"
        )
        print("#" * 118)
        completed = subprocess.run(
            worker_command(
                args,
                case=case,
                activation=activation,
                result_path=result_path,
            ),
            check=False,
        )
        if result_path.exists():
            with result_path.open("r", encoding="utf-8") as file:
                record = json.load(file)
        else:
            record = {
                "case": "learned_training" if case == "training" else "full_newton_update",
                "activation": activation,
                "status": "worker_crashed",
                "passed": False,
                "returncode": completed.returncode,
                "error": "Worker exited without writing its result JSON.",
            }
        record["worker_returncode"] = completed.returncode
        records.append(record)

    successful = [record for record in records if record.get("passed")]
    worst = (
        max(successful, key=lambda record: float(record.get("peak_reserved_gib", -1.0)))
        if successful
        else None
    )
    summary = {
        "experiment_script": str(args.experiment_script),
        "history_gradient_policy": (
            "full gradient through previous residual and previous update; no detach"
        ),
        "device": args.device,
        "requested": {
            "batch_size": args.batch_size,
            "K": args.k,
            "depth": args.depth,
            "width": args.width,
            "activations": list(args.activations),
            "gradient_clip_norm": args.gradient_clip_norm,
            "residual_length_scale": args.residual_length_scale,
            "synthetic_radius": args.synthetic_radius,
            "newton_included": not args.skip_newton,
        },
        "all_passed": all(record.get("passed") for record in records),
        "num_cases": len(records),
        "num_passed": sum(bool(record.get("passed")) for record in records),
        "num_failed": sum(not bool(record.get("passed")) for record in records),
        "worst_successful_case": worst,
        "records": records,
    }
    write_json(summary, args.output_dir / "max_gpu_memory_summary.json")

    rows = [flatten_for_csv(record) for record in records]
    csv_path = args.output_dir / "max_gpu_memory_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print_summary(records, args.device)
    print(f"\nJSON summary: {args.output_dir / 'max_gpu_memory_summary.json'}")
    print(f"CSV summary:  {csv_path}")
    return 0 if summary["all_passed"] else 2


def main() -> None:
    args = parse_args()
    validate_positive(args)
    if args.worker_case is not None:
        raise SystemExit(run_worker(args))
    raise SystemExit(run_parent(args))


if __name__ == "__main__":
    main()
