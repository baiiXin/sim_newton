#!/usr/bin/env python3
"""Maximum CUDA-memory test for the requested large MLP ablation.

Requested experiment grid
-------------------------
activation x hidden width x hidden depth x bias
    {ReLU, Tanh} x {512, 1024} x {1, 2} x {False, True}

This standalone script measures the most memory-demanding *training* operation,
not merely inference.  Each worker performs one full-batch epoch with:

* fixed-left-edge 5x5 triangular cloth,
* 8192 states,
* torch.float64,
* K=30 differentiable learned-optimizer steps,
* the same three-channel history input as the training script,
* mean-over-K variational-energy objective,
* backward(), gradient clipping at 10, and Adam.step().

The parent process launches one isolated worker per configuration.  Therefore an
out-of-memory failure in one configuration does not terminate the remaining
checks, and CUDA memory is fully released when that worker exits.

Default behavior tests the four maximum-size candidates:
    width=1024, depth=2, activation in {relu,tanh}, bias in {False,True}.
Use --all-configs to test the complete 16-configuration grid.

The CUDA device is intentionally fixed to cuda:1.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn


# =============================================================================
# 0. Fixed experiment constants
# =============================================================================

GRID_ROWS = 5
GRID_COLS = 5
SPATIAL_DIM = 3
NUM_PARTICLES = GRID_ROWS * GRID_COLS
FIXED_VERTEX_INDICES = (0, (GRID_ROWS - 1) * GRID_COLS)
FREE_VERTEX_INDICES = tuple(
    index for index in range(NUM_PARTICLES)
    if index not in set(FIXED_VERTEX_INDICES)
)
NUM_FREE_PARTICLES = len(FREE_VERTEX_INDICES)
FREE_STATE_DIM = NUM_FREE_PARTICLES * SPATIAL_DIM  # 23 * 3 = 69
HISTORY_INPUT_CHANNELS = 3
MODEL_INPUT_DIM = HISTORY_INPUT_CHANNELS * FREE_STATE_DIM  # 207

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

DEVICE_STRING = "cuda:1"
DEFAULT_BATCH_SIZE = 8192
DEFAULT_K = 30
DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_GRADIENT_CLIP_NORM = 10.0
LEARNING_RATE = 1e-3
MODEL_RANDOM_SEED = 42
DATA_RANDOM_SEED = 20260706
DISTANCE_EPS = 1e-12

REQUESTED_ACTIVATIONS = ("relu", "tanh")
REQUESTED_WIDTHS = (512, 1024)
REQUESTED_DEPTHS = (1, 2)
REQUESTED_BIASES = (False, True)


# =============================================================================
# 1. Cloth topology and physical model
# =============================================================================


def grid_index(row: int, col: int) -> int:
    return row * GRID_COLS + col


def build_triangular_cloth_edges() -> tuple[tuple[int, int], ...]:
    """Build horizontal, vertical, and one alternating diagonal per cell."""
    edge_set: set[tuple[int, int]] = set()

    def add_edge(a: int, b: int) -> None:
        edge_set.add((min(a, b), max(a, b)))

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS - 1):
            add_edge(grid_index(row, col), grid_index(row, col + 1))
    for row in range(GRID_ROWS - 1):
        for col in range(GRID_COLS):
            add_edge(grid_index(row, col), grid_index(row + 1, col))
    for row in range(GRID_ROWS - 1):
        for col in range(GRID_COLS - 1):
            tl = grid_index(row, col)
            tr = grid_index(row, col + 1)
            bl = grid_index(row + 1, col)
            br = grid_index(row + 1, col + 1)
            if (row + col) % 2 == 0:
                add_edge(tl, br)
            else:
                add_edge(bl, tr)

    edges = tuple(sorted(edge_set))
    expected = (
        GRID_ROWS * (GRID_COLS - 1)
        + (GRID_ROWS - 1) * GRID_COLS
        + (GRID_ROWS - 1) * (GRID_COLS - 1)
    )
    if len(edges) != expected:
        raise RuntimeError(f"Expected {expected} edges, found {len(edges)}")
    return edges


SPRING_EDGES = build_triangular_cloth_edges()
NUM_SPRINGS = len(SPRING_EDGES)


@dataclass(frozen=True)
class PhysicalConfig:
    masses: tuple[float, ...]
    dt: float
    spring_stiffness: tuple[float, ...]
    rest_lengths: tuple[float, ...]
    p0: tuple[tuple[float, float, float], ...]


def default_physical_config() -> PhysicalConfig:
    spacing = 0.5
    height = 1.2
    p0 = tuple(
        (col * spacing, -row * spacing, height)
        for row in range(GRID_ROWS)
        for col in range(GRID_COLS)
    )
    rest_lengths = tuple(math.dist(p0[i], p0[j]) for i, j in SPRING_EDGES)
    return PhysicalConfig(
        masses=tuple(1.0 for _ in range(NUM_PARTICLES)),
        dt=0.01,
        spring_stiffness=tuple(2500.0 for _ in range(NUM_SPRINGS)),
        rest_lengths=rest_lengths,
        p0=p0,
    )


def reshape_free(y: torch.Tensor) -> torch.Tensor:
    if y.shape[-1] != FREE_STATE_DIM:
        raise ValueError(
            f"Expected final dimension {FREE_STATE_DIM}, got {tuple(y.shape)}"
        )
    return y.reshape(*y.shape[:-1], NUM_FREE_PARTICLES, SPATIAL_DIM)


def full_positions_from_free(
    y: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    free = reshape_free(y)
    leading_shape = free.shape[:-2]
    base = torch.as_tensor(physical.p0, dtype=y.dtype, device=y.device)
    view_shape = (*([1] * len(leading_shape)), NUM_PARTICLES, SPATIAL_DIM)
    full = base.reshape(view_shape).expand(
        *leading_shape, NUM_PARTICLES, SPATIAL_DIM
    ).clone()
    full[..., list(FREE_VERTEX_INDICES), :] = free
    return full


def spring_lengths_from_free(
    y: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    full = full_positions_from_free(y, physical)
    edges = torch.as_tensor(SPRING_EDGES, dtype=torch.long, device=y.device)
    vectors = full[..., edges[:, 1], :] - full[..., edges[:, 0], :]
    return torch.linalg.vector_norm(vectors, dim=-1)


def variational_energy(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    free = reshape_free(y)
    q_free = reshape_free(q)
    inertial = (masses / (2.0 * physical.dt**2)) * torch.sum(
        (free - q_free) ** 2,
        dim=-1,
    )
    lengths = spring_lengths_from_free(y, physical)
    stiffness = torch.as_tensor(
        physical.spring_stiffness,
        dtype=y.dtype,
        device=y.device,
    )
    rest = torch.as_tensor(
        physical.rest_lengths,
        dtype=y.dtype,
        device=y.device,
    )
    spring = 0.5 * stiffness * (lengths - rest) ** 2
    return torch.sum(inertial, dim=-1) + torch.sum(spring, dim=-1)


def stationarity_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    free = reshape_free(y)
    q_free = reshape_free(q)
    full = full_positions_from_free(y, physical)

    grad_free = (masses[..., :, None] / physical.dt**2) * (free - q_free)
    full_grad = torch.zeros_like(full)
    full_grad[..., list(FREE_VERTEX_INDICES), :] = grad_free

    edges = torch.as_tensor(SPRING_EDGES, dtype=torch.long, device=y.device)
    edge_vectors = full[..., edges[:, 1], :] - full[..., edges[:, 0], :]
    lengths = torch.linalg.vector_norm(
        edge_vectors,
        dim=-1,
        keepdim=True,
    ).clamp_min(DISTANCE_EPS)
    stiffness = torch.as_tensor(
        physical.spring_stiffness,
        dtype=y.dtype,
        device=y.device,
    )
    rest = torch.as_tensor(
        physical.rest_lengths,
        dtype=y.dtype,
        device=y.device,
    )
    parameter_shape = [1] * (edge_vectors.ndim - 2) + [NUM_SPRINGS, 1]
    edge_grad = (
        stiffness.reshape(parameter_shape)
        * (1.0 - rest.reshape(parameter_shape) / lengths)
        * edge_vectors
    )
    full_grad = full_grad.clone()
    full_grad.index_add_(-2, edges[:, 0], -edge_grad)
    full_grad.index_add_(-2, edges[:, 1], edge_grad)
    return full_grad[..., list(FREE_VERTEX_INDICES), :].reshape(
        *y.shape[:-1],
        FREE_STATE_DIM,
    )


def mass_preconditioned_residual(
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
) -> torch.Tensor:
    residual = stationarity_residual(y, q, masses, physical)
    mass_per_coordinate = masses.repeat_interleave(SPATIAL_DIM, dim=-1)
    return physical.dt**2 * residual / mass_per_coordinate


# =============================================================================
# 2. Requested MLP and one-step history
# =============================================================================


@dataclass(frozen=True)
class ModelSpec:
    activation: str
    width: int
    depth: int
    use_bias: bool

    @property
    def name(self) -> str:
        bias_name = "bias" if self.use_bias else "no_bias"
        return (
            f"{self.activation}_width_{self.width:04d}_"
            f"depth_{self.depth}_{bias_name}"
        )


@dataclass(frozen=True)
class LearnedOptimizerState:
    previous_residual: torch.Tensor
    previous_update: torch.Tensor

    @classmethod
    def zeros_like(cls, y: torch.Tensor) -> "LearnedOptimizerState":
        zeros = torch.zeros_like(y)
        return cls(
            previous_residual=zeros,
            previous_update=zeros.clone(),
        )


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class MLPOptimizer(nn.Module):
    def __init__(
        self,
        residual_length_scale: float,
        model_spec: ModelSpec,
    ) -> None:
        super().__init__()
        self.model_spec = model_spec
        self.activation = make_activation(model_spec.activation)

        hidden_layers: list[nn.Linear] = []
        input_dim = MODEL_INPUT_DIM
        for _ in range(model_spec.depth):
            hidden_layers.append(
                nn.Linear(input_dim, model_spec.width, bias=model_spec.use_bias)
            )
            input_dim = model_spec.width
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(
            model_spec.width,
            FREE_STATE_DIM,
            bias=model_spec.use_bias,
        )
        self.register_buffer(
            "residual_length_scale",
            torch.tensor(float(residual_length_scale), dtype=TORCH_DTYPE),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        optimizer_state: LearnedOptimizerState,
        *,
        physical: PhysicalConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_residual = mass_preconditioned_residual(y, q, masses, physical)
        h = torch.cat(
            [
                current_residual / self.residual_length_scale,
                optimizer_state.previous_residual / self.residual_length_scale,
                optimizer_state.previous_update / self.residual_length_scale,
            ],
            dim=-1,
        )
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        delta = self.residual_length_scale * self.output_layer(h)
        return delta, current_residual


def apply_model_update(
    model: MLPOptimizer,
    y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    optimizer_state: LearnedOptimizerState,
) -> tuple[torch.Tensor, LearnedOptimizerState]:
    delta, current_residual = model(
        y,
        q,
        masses,
        optimizer_state,
        physical=physical,
    )
    next_state = LearnedOptimizerState(
        previous_residual=current_residual.detach(),
        previous_update=delta.detach(),
    )
    return y + delta, next_state


# =============================================================================
# 3. Synthetic full-batch construction
# =============================================================================


def make_training_batch(
    batch_size: int,
    device: torch.device,
    physical: PhysicalConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create finite, nondegenerate states with the real training tensor shapes."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(DATA_RANDOM_SEED)

    base_full = torch.tensor(physical.p0, dtype=TORCH_DTYPE)
    base_free = base_full[list(FREE_VERTEX_INDICES), :].reshape(1, FREE_STATE_DIM)

    # The exact rest configuration is used as q.  Initial states are perturbed in
    # a 1e-2 L-infinity cube, matching the lower end of the real sampling range.
    perturbation = (
        2.0
        * torch.rand(
            (batch_size, FREE_STATE_DIM),
            generator=generator,
            dtype=TORCH_DTYPE,
        )
        - 1.0
    ) * 1e-2
    initial_y = (base_free + perturbation).to(device)
    q = base_free.expand(batch_size, -1).clone().to(device)
    masses = torch.ones(
        (batch_size, NUM_FREE_PARTICLES),
        dtype=TORCH_DTYPE,
        device=device,
    )

    if not bool(torch.all(spring_lengths_from_free(initial_y, physical) > 1e-6)):
        raise RuntimeError("Synthetic memory-test batch contains a degenerate spring")
    return initial_y, q, masses


def physical_energy_scale(
    masses: torch.Tensor,
    physical: PhysicalConfig,
    residual_length_scale: float,
) -> float:
    return (
        float(masses.mean().item())
        * residual_length_scale**2
        / physical.dt**2
    )


# =============================================================================
# 4. Worker measurement
# =============================================================================


def gib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0**3)


def validate_cuda1(device: torch.device) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type != "cuda" or device.index != 1:
        raise RuntimeError(f"This script must use cuda:1, got {device}")
    if torch.cuda.device_count() <= 1:
        raise RuntimeError(
            f"cuda:1 was requested, but only {torch.cuda.device_count()} CUDA device(s) are visible"
        )


def run_worker(
    *,
    spec: ModelSpec,
    batch_size: int,
    k_steps: int,
    result_path: Path,
) -> int:
    device = torch.device(DEVICE_STRING)
    record: dict[str, Any] = {
        **asdict(spec),
        "experiment_name": spec.name,
        "device": DEVICE_STRING,
        "dtype": "torch.float64",
        "batch_size": batch_size,
        "K": k_steps,
        "status": "error",
        "parameter_count": None,
        "objective": None,
        "gradient_norm_before_clip": None,
        "elapsed_seconds": None,
        "total_memory_gib": None,
        "free_memory_before_gib": None,
        "allocated_before_gib": None,
        "reserved_before_gib": None,
        "peak_allocated_gib": None,
        "peak_reserved_gib": None,
        "peak_allocated_fraction_of_total": None,
        "peak_reserved_fraction_of_total": None,
        "error": "",
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }

    try:
        validate_cuda1(device)
        torch.cuda.set_device(device)
        torch.manual_seed(MODEL_RANDOM_SEED)
        torch.cuda.manual_seed_all(MODEL_RANDOM_SEED)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        free_before, total_memory = torch.cuda.mem_get_info(device)
        record["total_memory_gib"] = gib(total_memory)
        record["free_memory_before_gib"] = gib(free_before)

        # Reset before allocating data/model/optimizer so the reported peak is
        # the complete process-local memory footprint of the training step.
        torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()

        physical = default_physical_config()
        initial_y, q, masses = make_training_batch(
            batch_size,
            device,
            physical,
        )
        model = MLPOptimizer(DEFAULT_RESIDUAL_LENGTH_SCALE, spec).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        record["parameter_count"] = model.parameter_count
        record["allocated_before_gib"] = gib(torch.cuda.memory_allocated(device))
        record["reserved_before_gib"] = gib(torch.cuda.memory_reserved(device))

        initial_energy = variational_energy(initial_y, q, masses, physical).detach()
        energy_scale = physical_energy_scale(
            masses,
            physical,
            DEFAULT_RESIDUAL_LENGTH_SCALE,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        y = initial_y
        optimizer_state = LearnedOptimizerState.zeros_like(y)
        objective_sum = torch.zeros((), dtype=TORCH_DTYPE, device=device)

        for _ in range(k_steps):
            y, optimizer_state = apply_model_update(
                model,
                y,
                q,
                masses,
                physical,
                optimizer_state,
            )
            energy = variational_energy(y, q, masses, physical)
            objective_sum = objective_sum + (
                (energy - initial_energy) / energy_scale
            ).mean()

        objective = objective_sum / float(k_steps)
        if not bool(torch.isfinite(objective)):
            raise RuntimeError(
                "The synthetic maximum-memory step produced a non-finite objective"
            )

        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            DEFAULT_GRADIENT_CLIP_NORM,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(
                "The synthetic maximum-memory step produced a non-finite gradient norm"
            )
        optimizer.step()  # Includes first-step Adam state allocation.
        torch.cuda.synchronize(device)

        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        elapsed = time.perf_counter() - start_time

        record.update(
            status="success",
            objective=float(objective.detach().item()),
            gradient_norm_before_clip=float(gradient_norm.detach().item()),
            elapsed_seconds=elapsed,
            peak_allocated_gib=gib(peak_allocated),
            peak_reserved_gib=gib(peak_reserved),
            peak_allocated_fraction_of_total=float(peak_allocated / total_memory),
            peak_reserved_fraction_of_total=float(peak_reserved / total_memory),
        )
        exit_code = 0

    except torch.cuda.OutOfMemoryError as exc:
        record["status"] = "oom"
        record["error"] = f"{type(exc).__name__}: {exc}"
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        exit_code = 2
    except Exception as exc:  # noqa: BLE001 - preserve complete worker failure.
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        exit_code = 1

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return exit_code


# =============================================================================
# 5. Parent orchestration and reports
# =============================================================================


def requested_specs(all_configs: bool) -> list[ModelSpec]:
    if not all_configs:
        return [
            ModelSpec(activation, 1024, 2, use_bias)
            for activation in REQUESTED_ACTIVATIONS
            for use_bias in REQUESTED_BIASES
        ]
    return [
        ModelSpec(activation, width, depth, use_bias)
        for activation in REQUESTED_ACTIVATIONS
        for width in REQUESTED_WIDTHS
        for depth in REQUESTED_DEPTHS
        for use_bias in REQUESTED_BIASES
    ]


def write_summary_csv(records: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "activation",
        "width",
        "depth",
        "use_bias",
        "status",
        "parameter_count",
        "batch_size",
        "K",
        "objective",
        "gradient_norm_before_clip",
        "elapsed_seconds",
        "total_memory_gib",
        "free_memory_before_gib",
        "allocated_before_gib",
        "reserved_before_gib",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "peak_allocated_fraction_of_total",
        "peak_reserved_fraction_of_total",
        "error",
        "returncode",
        "worker_stderr_tail",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def print_summary(records: Sequence[dict[str, Any]]) -> None:
    print("\n" + "=" * 112)
    print("Maximum full-batch CUDA memory test summary")
    print("=" * 112)
    header = (
        f"{'activation':<10} {'width':>6} {'depth':>5} {'bias':>6} "
        f"{'status':>9} {'params':>12} {'peak alloc':>12} "
        f"{'peak reserv':>12} {'time(s)':>9}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        peak_alloc = record.get("peak_allocated_gib")
        peak_reserv = record.get("peak_reserved_gib")
        elapsed = record.get("elapsed_seconds")
        print(
            f"{str(record.get('activation')):<10} "
            f"{int(record.get('width', 0)):>6d} "
            f"{int(record.get('depth', 0)):>5d} "
            f"{str(record.get('use_bias')):>6} "
            f"{str(record.get('status')):>9} "
            f"{int(record.get('parameter_count') or 0):>12,d} "
            f"{(f'{peak_alloc:.3f} GiB' if isinstance(peak_alloc, (int, float)) else '-'):>12} "
            f"{(f'{peak_reserv:.3f} GiB' if isinstance(peak_reserv, (int, float)) else '-'):>12} "
            f"{(f'{elapsed:.2f}' if isinstance(elapsed, (int, float)) else '-'):>9}"
        )

    successful = [r for r in records if r.get("status") == "success"]
    if successful:
        largest = max(
            successful,
            key=lambda r: float(r.get("peak_reserved_gib") or -1.0),
        )
        print("\nLargest successful peak reservation:")
        print(
            f"  {largest['activation']}, width={largest['width']}, "
            f"depth={largest['depth']}, bias={largest['use_bias']}: "
            f"{largest['peak_reserved_gib']:.3f} GiB reserved, "
            f"{largest['peak_allocated_gib']:.3f} GiB allocated."
        )


def run_parent(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else script_path.with_suffix("").with_name(script_path.stem + "_results")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = output_dir / "worker_results"
    worker_dir.mkdir(parents=True, exist_ok=True)

    specs = requested_specs(args.all_configs)
    print(f"CUDA device: {DEVICE_STRING}")
    print(f"Batch size: {args.batch_size:,}")
    print(f"Unrolled training steps K: {args.k_steps}")
    print(f"Configurations to test: {len(specs)}")
    print(f"Output directory: {output_dir}")

    records: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        result_path = worker_dir / f"{spec.name}.json"
        command = [
            sys.executable,
            str(script_path),
            "--worker",
            "--activation",
            spec.activation,
            "--width",
            str(spec.width),
            "--depth",
            str(spec.depth),
            "--use-bias",
            "1" if spec.use_bias else "0",
            "--batch-size",
            str(args.batch_size),
            "--k-steps",
            str(args.k_steps),
            "--result-path",
            str(result_path),
        ]
        print("\n" + "-" * 100)
        print(f"[{index}/{len(specs)}] Testing {spec.name}")
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )

        if result_path.exists():
            with result_path.open("r", encoding="utf-8") as file:
                record = json.load(file)
        else:
            record = {
                **asdict(spec),
                "experiment_name": spec.name,
                "status": "worker_failed_without_result",
                "error": "Worker exited before writing its JSON result",
            }
        record["returncode"] = completed.returncode
        record["worker_stderr_tail"] = completed.stderr[-4000:]
        records.append(record)

        status = record.get("status")
        if status == "success":
            print(
                f"success: peak allocated={record['peak_allocated_gib']:.3f} GiB, "
                f"peak reserved={record['peak_reserved_gib']:.3f} GiB"
            )
        else:
            print(f"{status}: {record.get('error', '')[:1000]}")
            if completed.stderr:
                print("worker stderr tail:")
                print(completed.stderr[-2000:])

    summary_json = output_dir / "maximum_memory_test_summary.json"
    summary_csv = output_dir / "maximum_memory_test_summary.csv"
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)
    write_summary_csv(records, summary_csv)
    print_summary(records)
    print(f"\nJSON summary: {summary_json}")
    print(f"CSV summary:  {summary_csv}")

    return 0 if all(r.get("status") == "success" for r in records) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Test all 16 requested configurations instead of only the four maximum-size candidates.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--k-steps", type=int, default=DEFAULT_K)
    parser.add_argument("--output-dir", type=str, default="")

    # Internal worker arguments.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--activation",
        choices=REQUESTED_ACTIVATIONS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--width", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--depth", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--use-bias", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--result-path", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.k_steps <= 0:
        parser.error("--k-steps must be positive")
    if args.worker:
        missing = [
            name
            for name in ("activation", "width", "depth", "use_bias", "result_path")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(f"Worker is missing required arguments: {missing}")
    return args


def main() -> int:
    args = parse_args()
    if args.worker:
        spec = ModelSpec(
            activation=args.activation,
            width=args.width,
            depth=args.depth,
            use_bias=bool(args.use_bias),
        )
        return run_worker(
            spec=spec,
            batch_size=args.batch_size,
            k_steps=args.k_steps,
            result_path=Path(args.result_path),
        )
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
