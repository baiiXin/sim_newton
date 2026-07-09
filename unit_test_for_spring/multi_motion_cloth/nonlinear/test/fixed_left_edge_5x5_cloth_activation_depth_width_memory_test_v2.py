"""
CUDA peak-memory test for the planned 5x5 cloth activation/depth/width ablation.

Planned architecture grid
-------------------------
activations: identity, relu, tanh
hidden depths: 1, 2, 5, 10
hidden widths: 69, 128, 256
bias policy: identity uses no bias; relu/tanh use bias in every Linear layer
training unroll schedule: this tester targets the worst stage, K=30 by default
precision: float64

What this script measures
-------------------------
For each requested architecture and candidate micro-batch size, a fresh child
process performs the memory-critical part of one real training update:

1. Construct valid 5x5 triangular-cloth states.
2. Unroll the learned optimizer K times.
3. Evaluate the real implicit-Euler variational energy at every step.
4. Average the energy-change objective over batch and K.
5. Run backward(), gradient clipping, and Adam.step().
6. Record CUDA peak allocated/reserved memory.

Each candidate runs in an isolated subprocess. A CUDA OOM therefore does not
leave allocator fragments that contaminate later candidates.

The default scan tests the three activations at the largest planned network
(depth=10, width=256), because this is the relevant worst-case architecture.
To scan the full architecture grid, pass:

    --depths 1 2 5 10 --widths 69 128 256

Example
-------
python fixed_left_edge_5x5_cloth_activation_depth_width_memory_test.py \
    --device cuda:0 \
    --batch-sizes 64 128 256 512 1024 2048 4096 8192

Outputs
-------
<output-dir>/memory_test_results.json
<output-dir>/memory_test_results.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn


# =============================================================================
# 1. Cloth topology and physical energy (matched to the training experiment)
# =============================================================================

GRID_ROWS = 5
GRID_COLS = 5
SPATIAL_DIM = 3
NUM_PARTICLES = GRID_ROWS * GRID_COLS
FIXED_VERTEX_INDICES = (0, (GRID_ROWS - 1) * GRID_COLS)
FREE_VERTEX_INDICES = tuple(
    i for i in range(NUM_PARTICLES) if i not in set(FIXED_VERTEX_INDICES)
)
NUM_FREE_PARTICLES = len(FREE_VERTEX_INDICES)
FREE_STATE_DIM = NUM_FREE_PARTICLES * SPATIAL_DIM  # 23 * 3 = 69
DISTANCE_EPS = 1e-12
TORCH_DTYPE = torch.float64


def grid_index(row: int, col: int) -> int:
    return row * GRID_COLS + col


def build_triangular_cloth_edges() -> tuple[tuple[int, int], ...]:
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
            add_edge(tl, br) if (row + col) % 2 == 0 else add_edge(bl, tr)

    edges = tuple(sorted(edge_set))
    expected = (
        GRID_ROWS * (GRID_COLS - 1)
        + (GRID_ROWS - 1) * GRID_COLS
        + (GRID_ROWS - 1) * (GRID_COLS - 1)
    )
    if len(edges) != expected:
        raise RuntimeError(f"Expected {expected} springs, got {len(edges)}")
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
    height = 1.20
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
        raise ValueError(f"Expected final dimension {FREE_STATE_DIM}, got {tuple(y.shape)}")
    return y.reshape(*y.shape[:-1], NUM_FREE_PARTICLES, SPATIAL_DIM)


def full_positions_from_free(y: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    free = reshape_free(y)
    leading_shape = free.shape[:-2]
    base = torch.as_tensor(physical.p0, dtype=y.dtype, device=y.device)
    view_shape = (*([1] * len(leading_shape)), NUM_PARTICLES, SPATIAL_DIM)
    full = base.reshape(view_shape).expand(
        *leading_shape, NUM_PARTICLES, SPATIAL_DIM
    ).clone()
    full[..., list(FREE_VERTEX_INDICES), :] = free
    return full


def spring_edge_tensor(device: torch.device) -> torch.Tensor:
    return torch.as_tensor(SPRING_EDGES, dtype=torch.long, device=device)


def spring_lengths_from_free(y: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    full = full_positions_from_free(y, physical)
    edges = spring_edge_tensor(y.device)
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
        (free - q_free) ** 2, dim=-1
    )
    lengths = spring_lengths_from_free(y, physical)
    stiffness = torch.as_tensor(
        physical.spring_stiffness, dtype=y.dtype, device=y.device
    )
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
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

    edges = spring_edge_tensor(y.device)
    edge_vectors = full[..., edges[:, 1], :] - full[..., edges[:, 0], :]
    lengths = torch.linalg.vector_norm(
        edge_vectors, dim=-1, keepdim=True
    ).clamp_min(DISTANCE_EPS)
    stiffness = torch.as_tensor(
        physical.spring_stiffness, dtype=y.dtype, device=y.device
    )
    rest = torch.as_tensor(physical.rest_lengths, dtype=y.dtype, device=y.device)
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
        *y.shape[:-1], FREE_STATE_DIM
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
# 2. Planned activation/depth/width model
# =============================================================================


def make_activation(name: str) -> nn.Module:
    if name == "identity":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class AblationMLPOptimizer(nn.Module):
    """MLP where depth means the number of hidden Linear layers."""

    def __init__(
        self,
        *,
        activation_name: str,
        hidden_depth: int,
        hidden_width: int,
        residual_length_scale: float,
        stress_output_init_std: float,
    ) -> None:
        super().__init__()
        if hidden_depth <= 0:
            raise ValueError("hidden_depth must be positive")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if residual_length_scale <= 0.0:
            raise ValueError("residual_length_scale must be positive")

        self.activation_name = activation_name
        self.hidden_depth = int(hidden_depth)
        self.hidden_width = int(hidden_width)
        self.use_bias = activation_name != "identity"

        hidden_layers: list[nn.Linear] = []
        in_features = FREE_STATE_DIM
        for _ in range(hidden_depth):
            hidden_layers.append(
                nn.Linear(in_features, hidden_width, bias=self.use_bias)
            )
            in_features = hidden_width
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = nn.Linear(
            hidden_width, FREE_STATE_DIM, bias=self.use_bias
        )
        self.activation = make_activation(activation_name)

        gain = nn.init.calculate_gain(
            "linear" if activation_name == "identity" else activation_name
        )
        for layer in self.hidden_layers:
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Formal training will still use a zero-initialized output layer. For a
        # memory stress test, a tiny nonzero weight exercises the full K-step
        # graph after the network has begun training. Set --output-init-std 0
        # to reproduce the exact epoch-0 initialization instead.
        if stress_output_init_std > 0.0:
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=stress_output_init_std)
        else:
            nn.init.zeros_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

        self.register_buffer(
            "residual_length_scale",
            torch.tensor(float(residual_length_scale), dtype=TORCH_DTYPE),
        )

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        masses: torch.Tensor,
        *,
        physical: PhysicalConfig,
    ) -> torch.Tensor:
        u = mass_preconditioned_residual(y, q, masses, physical)
        h = u / self.residual_length_scale
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.residual_length_scale * self.output_layer(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 3. Synthetic but physically valid stress batch
# =============================================================================


def make_stress_batch(
    *,
    batch_size: int,
    physical: PhysicalConfig,
    device: torch.device,
    seed: int,
    perturbation_radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if perturbation_radius <= 0.0:
        raise ValueError("perturbation_radius must be positive")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    base_full = torch.as_tensor(physical.p0, dtype=TORCH_DTYPE, device=device)
    base_free = base_full[list(FREE_VERTEX_INDICES), :].reshape(1, FREE_STATE_DIM)

    y_noise = (
        2.0
        * torch.rand(
            (batch_size, FREE_STATE_DIM),
            dtype=TORCH_DTYPE,
            device=device,
            generator=generator,
        )
        - 1.0
    ) * perturbation_radius
    q_noise = (
        2.0
        * torch.rand(
            (batch_size, FREE_STATE_DIM),
            dtype=TORCH_DTYPE,
            device=device,
            generator=generator,
        )
        - 1.0
    ) * (0.5 * perturbation_radius)

    initial_y = (base_free + y_noise).contiguous()
    q = (base_free + q_noise).contiguous()
    masses = torch.ones(
        (batch_size, NUM_FREE_PARTICLES), dtype=TORCH_DTYPE, device=device
    )

    if not bool(torch.all(spring_lengths_from_free(initial_y, physical) > DISTANCE_EPS)):
        raise RuntimeError("Stress batch unexpectedly contains a degenerate spring")
    return initial_y, q, masses


# =============================================================================
# 4. One full memory-critical training update
# =============================================================================


def train_step(
    *,
    model: AblationMLPOptimizer,
    optimizer: torch.optim.Optimizer,
    initial_y: torch.Tensor,
    q: torch.Tensor,
    masses: torch.Tensor,
    physical: PhysicalConfig,
    unroll_steps: int,
    residual_length_scale: float,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    y = initial_y
    initial_energy = variational_energy(y, q, masses, physical).detach()
    energy_scale = (
        float(masses.mean().item())
        * residual_length_scale**2
        / physical.dt**2
    )

    objective = torch.zeros((), dtype=TORCH_DTYPE, device=initial_y.device)
    for _ in range(unroll_steps):
        y = y + model(y, q, masses, physical=physical)
        energy = variational_energy(y, q, masses, physical)
        objective = objective + torch.mean((energy - initial_energy) / energy_scale)
    objective = objective / float(unroll_steps)

    if not bool(torch.isfinite(objective)):
        raise FloatingPointError("Objective became non-finite before backward")
    objective.backward()
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip_norm
    )
    grad_norm = float(grad_norm_tensor.item())
    if not math.isfinite(grad_norm):
        raise FloatingPointError("Gradient norm became non-finite")
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("A model parameter became non-finite")
    return float(objective.detach().item()), grad_norm


# =============================================================================
# 5. Worker process
# =============================================================================


def bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024.0**3)


def parse_cuda_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("This script measures CUDA memory and requires a CUDA device")
    return 0 if device.index is None else int(device.index)


def worker_run(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "activation": args.activation,
        "depth": int(args.depth),
        "width": int(args.width),
        "bias": args.activation != "identity",
        "batch_size": int(args.worker_batch_size),
        "K": int(args.k),
        "dtype": "float64",
        "device": args.device,
        "status": "error",
    }

    # Keep every CUDA-touching operation inside the try block. Some PyTorch
    # versions fail during set_device() when an unsupported allocator option is
    # present; the old script exited before it could write a diagnostic JSON.
    device = torch.device(args.device)
    try:
        cuda_index = parse_cuda_index(device)
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        if cuda_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{cuda_index}, but only {torch.cuda.device_count()} "
                "CUDA devices are visible"
            )

        torch.cuda.set_device(cuda_index)
        torch.set_default_dtype(TORCH_DTYPE)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        physical = default_physical_config()

        torch.cuda.empty_cache()
        gc.collect()
        free_before, total_memory = torch.cuda.mem_get_info(device)

        model = AblationMLPOptimizer(
            activation_name=args.activation,
            hidden_depth=args.depth,
            hidden_width=args.width,
            residual_length_scale=args.residual_length_scale,
            stress_output_init_std=args.output_init_std,
        ).to(device=device, dtype=TORCH_DTYPE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        initial_y, q, masses = make_stress_batch(
            batch_size=args.worker_batch_size,
            physical=physical,
            device=device,
            seed=args.seed + 17,
            perturbation_radius=args.perturbation_radius,
        )

        for _ in range(args.warmup_steps):
            train_step(
                model=model,
                optimizer=optimizer,
                initial_y=initial_y,
                q=q,
                masses=masses,
                physical=physical,
                unroll_steps=args.k,
                residual_length_scale=args.residual_length_scale,
                gradient_clip_norm=args.gradient_clip_norm,
            )
            torch.cuda.synchronize(device)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)

        start_time = time.perf_counter()
        objective, grad_norm = train_step(
            model=model,
            optimizer=optimizer,
            initial_y=initial_y,
            q=q,
            masses=masses,
            physical=physical,
            unroll_steps=args.k,
            residual_length_scale=args.residual_length_scale,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time

        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        end_allocated = torch.cuda.memory_allocated(device)
        end_reserved = torch.cuda.memory_reserved(device)
        free_after, _ = torch.cuda.mem_get_info(device)

        result.update(
            status="success",
            parameter_count=count_parameters(model),
            objective=objective,
            gradient_norm_before_clip=grad_norm,
            elapsed_seconds=elapsed,
            total_memory_bytes=int(total_memory),
            free_memory_before_bytes=int(free_before),
            free_memory_after_bytes=int(free_after),
            baseline_allocated_bytes=int(baseline_allocated),
            baseline_reserved_bytes=int(baseline_reserved),
            peak_allocated_bytes=int(peak_allocated),
            peak_reserved_bytes=int(peak_reserved),
            end_allocated_bytes=int(end_allocated),
            end_reserved_bytes=int(end_reserved),
            total_memory_gib=bytes_to_gib(total_memory),
            free_memory_before_gib=bytes_to_gib(free_before),
            peak_allocated_gib=bytes_to_gib(peak_allocated),
            peak_reserved_gib=bytes_to_gib(peak_reserved),
            peak_reserved_fraction_of_total=float(peak_reserved / total_memory),
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update(status="oom", error=str(exc))
        try:
            result["peak_allocated_gib"] = bytes_to_gib(
                torch.cuda.max_memory_allocated(device)
            )
            result["peak_reserved_gib"] = bytes_to_gib(
                torch.cuda.max_memory_reserved(device)
            )
        except Exception:
            pass
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            result.update(status="oom", error=str(exc))
        else:
            result.update(status="error", error=f"RuntimeError: {exc}")
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
    return result


def worker_main(args: argparse.Namespace) -> None:
    path = Path(args.result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = worker_run(args)
    except BaseException as exc:
        # Last-resort protection: always leave a result file, even if failure
        # happens during CUDA/PyTorch initialization outside normal exceptions.
        result = {
            "activation": args.activation,
            "depth": int(args.depth),
            "width": int(args.width),
            "bias": args.activation != "identity",
            "batch_size": int(args.worker_batch_size),
            "K": int(args.k),
            "dtype": "float64",
            "device": args.device,
            "status": "fatal_error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


# =============================================================================
# 6. Parent scan and reports
# =============================================================================


def run_worker_subprocess(
    *,
    args: argparse.Namespace,
    activation: str,
    depth: int,
    width: int,
    batch_size: int,
    result_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--result-path",
        str(result_path),
        "--device",
        args.device,
        "--activation",
        activation,
        "--depth",
        str(depth),
        "--width",
        str(width),
        "--worker-batch-size",
        str(batch_size),
        "--k",
        str(args.k),
        "--warmup-steps",
        str(args.warmup_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--residual-length-scale",
        str(args.residual_length_scale),
        "--perturbation-radius",
        str(args.perturbation_radius),
        "--output-init-std",
        str(args.output_init_std),
        "--seed",
        str(args.seed),
    ]
    env = os.environ.copy()
    if args.allocator_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.allocator_conf

    completed = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if not result_path.exists():
        stdout_tail = completed.stdout[-4000:]
        stderr_tail = completed.stderr[-4000:]
        diagnostic = stderr_tail.strip() or stdout_tail.strip() or "no child output"
        return {
            "activation": activation,
            "depth": depth,
            "width": width,
            "bias": activation != "identity",
            "batch_size": batch_size,
            "K": args.k,
            "status": "process_error",
            "error": (
                f"Worker exited with return code {completed.returncode} before "
                f"writing its result file. Child diagnostic: {diagnostic}"
            ),
            "returncode": completed.returncode,
            "stdout": stdout_tail,
            "stderr": stderr_tail,
        }
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    result["worker_returncode"] = completed.returncode
    if completed.stdout.strip():
        result["worker_stdout_tail"] = completed.stdout[-2000:]
    if completed.stderr.strip() and result.get("status") not in {"oom"}:
        result["worker_stderr_tail"] = completed.stderr[-4000:]
    return result


def architecture_key(result: dict[str, Any]) -> str:
    return (
        f"{result['activation']}_depth{int(result['depth']):02d}_"
        f"width{int(result['width']):03d}"
    )


def summarize_results(
    results: Sequence[dict[str, Any]], safety_fraction: float
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in results:
        grouped.setdefault(architecture_key(record), []).append(record)

    per_architecture: dict[str, Any] = {}
    safe_maxima: list[int] = []
    successful_maxima: list[int] = []
    for key, records in grouped.items():
        records = sorted(records, key=lambda item: int(item["batch_size"]))
        successful = [r for r in records if r.get("status") == "success"]
        safe = [
            r
            for r in successful
            if float(r.get("peak_reserved_fraction_of_total", 1.0))
            <= safety_fraction
        ]
        maximum_successful = (
            max(int(r["batch_size"]) for r in successful) if successful else None
        )
        recommended_safe = (
            max(int(r["batch_size"]) for r in safe) if safe else None
        )
        if maximum_successful is not None:
            successful_maxima.append(maximum_successful)
        if recommended_safe is not None:
            safe_maxima.append(recommended_safe)
        per_architecture[key] = {
            "activation": records[0]["activation"],
            "depth": int(records[0]["depth"]),
            "width": int(records[0]["width"]),
            "bias": bool(records[0]["bias"]),
            "maximum_successful_batch_size": maximum_successful,
            "recommended_safe_batch_size": recommended_safe,
            "safety_fraction": safety_fraction,
        }

    return {
        "per_architecture": per_architecture,
        "global_maximum_common_successful_batch_size": (
            min(successful_maxima) if len(successful_maxima) == len(grouped) else None
        ),
        "global_recommended_safe_batch_size": (
            min(safe_maxima) if len(safe_maxima) == len(grouped) else None
        ),
        "recommendation_rule": (
            "For each architecture, choose the largest successful candidate whose "
            f"peak reserved memory is <= {safety_fraction:.0%} of total GPU memory; "
            "the global recommendation is the minimum across tested architectures."
        ),
    }


def write_csv(results: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "activation",
        "depth",
        "width",
        "bias",
        "batch_size",
        "K",
        "status",
        "parameter_count",
        "objective",
        "gradient_norm_before_clip",
        "elapsed_seconds",
        "total_memory_gib",
        "free_memory_before_gib",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "peak_reserved_fraction_of_total",
        "error",
        "returncode",
        "worker_returncode",
        "worker_stderr_tail",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in results:
            writer.writerow(record)


def parent_main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    activations = list(dict.fromkeys(args.activations))
    depths = sorted(set(int(v) for v in args.depths))
    widths = sorted(set(int(v) for v in args.widths))
    batch_sizes = sorted(set(int(v) for v in args.batch_sizes))

    for name in activations:
        if name not in {"identity", "relu", "tanh"}:
            raise ValueError(f"Unsupported activation: {name}")
    if any(v <= 0 for v in depths + widths + batch_sizes):
        raise ValueError("Depths, widths, and batch sizes must be positive")
    if not (0.0 < args.safety_fraction <= 1.0):
        raise ValueError("safety_fraction must lie in (0, 1]")

    configurations = [
        (activation, depth, width)
        for activation in activations
        for depth in depths
        for width in widths
    ]
    print("=" * 96)
    print("5x5 cloth CUDA memory scan")
    print(f"device={args.device}, dtype=float64, K={args.k}")
    print(f"architectures={len(configurations)}, candidate batches={batch_sizes}")
    print("bias rule: identity=False, relu/tanh=True")
    print(f"allocator conf: {args.allocator_conf or '(unchanged)'}")
    print("=" * 96)

    all_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cloth_memory_test_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for config_index, (activation, depth, width) in enumerate(configurations, start=1):
            label = f"{activation}, depth={depth}, width={width}"
            print(f"\n[{config_index}/{len(configurations)}] {label}")
            for batch_size in batch_sizes:
                result_path = temp_dir / (
                    f"{activation}_d{depth}_w{width}_b{batch_size}.json"
                )
                result = run_worker_subprocess(
                    args=args,
                    activation=activation,
                    depth=depth,
                    width=width,
                    batch_size=batch_size,
                    result_path=result_path,
                )
                all_results.append(result)
                status = result.get("status", "unknown")
                if status == "success":
                    print(
                        f"  batch={batch_size:5d}: SUCCESS | "
                        f"peak allocated={result['peak_allocated_gib']:.2f} GiB | "
                        f"peak reserved={result['peak_reserved_gib']:.2f} GiB | "
                        f"time={result['elapsed_seconds']:.2f}s"
                    )
                else:
                    message = str(result.get("error", ""))
                    if len(message) > 180:
                        message = message[:177] + "..."
                    print(f"  batch={batch_size:5d}: {status.upper()} | {message}")
                    if status == "oom" and not args.continue_after_oom:
                        print("  Larger batch sizes skipped for this architecture.")
                        break

    summary = summarize_results(all_results, args.safety_fraction)
    payload = {
        "test_definition": {
            "state_dimension": FREE_STATE_DIM,
            "free_particles": NUM_FREE_PARTICLES,
            "num_springs": NUM_SPRINGS,
            "dtype": "float64",
            "device": args.device,
            "K": args.k,
            "activations": activations,
            "depths": depths,
            "widths": widths,
            "bias_policy": "identity: no bias; relu/tanh: bias in every Linear layer",
            "batch_sizes": batch_sizes,
            "warmup_steps": args.warmup_steps,
            "output_init_std_for_stress_test": args.output_init_std,
            "loss": "mean over batch and K of dimensionless energy change",
            "allocator_conf": args.allocator_conf,
            "safety_fraction": args.safety_fraction,
        },
        "summary": summary,
        "results": all_results,
    }

    json_path = output_dir / "memory_test_results.json"
    csv_path = output_dir / "memory_test_results.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    write_csv(all_results, csv_path)

    print("\n" + "=" * 96)
    print("Scan complete")
    print(
        "Global maximum common successful batch: "
        f"{summary['global_maximum_common_successful_batch_size']}"
    )
    print(
        "Global recommended safe batch: "
        f"{summary['global_recommended_safe_batch_size']}"
    )
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")
    print("=" * 96)


# =============================================================================
# 7. CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CUDA peak-memory scan for the 5x5 cloth MLP ablation"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["identity", "relu", "tanh"],
        help="Architectures to scan in parent mode.",
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[10],
        help="Default only tests the deepest planned model.",
    )
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        default=[256],
        help="Default only tests the widest planned model.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--residual-length-scale", type=float, default=5e-2)
    parser.add_argument("--perturbation-radius", type=float, default=5e-2)
    parser.add_argument(
        "--output-init-std",
        type=float,
        default=1e-4,
        help=(
            "Tiny nonzero output weight used only to stress the full trained graph. "
            "Use 0 to reproduce exact zero-output initialization."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-fraction", type=float, default=0.85)
    parser.add_argument(
        "--allocator-conf",
        default="",
        help=(
            "Optional PYTORCH_CUDA_ALLOC_CONF passed to workers. Empty by default "
            "for compatibility with older PyTorch versions."
        ),
    )
    parser.add_argument(
        "--continue-after-oom",
        action="store_true",
        help="Continue trying larger candidates after an OOM.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().with_suffix("").parent / (Path(__file__).stem + "_results")),
    )

    # Internal worker arguments. Users normally do not set these directly.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--activation", default="identity", help=argparse.SUPPRESS)
    parser.add_argument("--depth", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=256, help=argparse.SUPPRESS)
    parser.add_argument("--worker-batch-size", type=int, default=64, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.k <= 0:
        raise ValueError("K must be positive")
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")
    if args.worker:
        if not args.result_path:
            raise ValueError("--result-path is required in worker mode")
        worker_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()
