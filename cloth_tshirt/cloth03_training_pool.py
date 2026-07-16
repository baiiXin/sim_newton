"""Learned optimizer and live online-randomized T-shirt training pool."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from cloth02_batched_physics import TShirtPhysics
from tshirt_config import DEFAULT_DYNAMICS, DEFAULT_OBJ_PATH, DynamicsDistribution
from tshirt_mesh import load_tshirt_mesh
from tshirt_sampling import sample_random_motion


DEFAULT_POOL_SIZE = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_K_BUCKETS = (1, 3, 10, 30)
DEFAULT_ENVIRONMENT_LIFETIME_FRAMES = 500
DEFAULT_RESIDUAL_LENGTH_SCALE = 5e-2
DEFAULT_GRADIENT_CLIP_NORM = 10.0
DEFAULT_LEARNING_RATE = 1e-4


@dataclass(frozen=True)
class ModelSpec:
    activation: str = "relu"
    depth: int = 1
    width: int = 2048
    use_bias: bool = False

    @property
    def experiment_name(self) -> str:
        bias = "bias" if self.use_bias else "no_bias"
        return (
            f"activation_{self.activation}_depth_{self.depth:02d}_"
            f"width_{self.width:04d}_{bias}"
        )


def _activation(name: str) -> nn.Module:
    choices: dict[str, nn.Module] = {
        "identity": nn.Identity(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
    }
    if name not in choices:
        raise ValueError(f"Unsupported activation: {name}")
    return choices[name]


def _activation_gain(name: str) -> float:
    if name == "identity":
        return 1.0
    if name in {"relu", "gelu", "silu"}:
        return sqrt(2.0)
    if name == "tanh":
        return 5.0 / 3.0
    raise ValueError(f"Unsupported activation: {name}")


class LearnedOptimizerMLP(nn.Module):
    """Full-state history MLP retained from the 15x15 reference project."""

    def __init__(
        self,
        *,
        physics: TShirtPhysics,
        residual_length_scale: float = DEFAULT_RESIDUAL_LENGTH_SCALE,
        model_spec: ModelSpec = ModelSpec(),
        initialize: bool = True,
    ) -> None:
        super().__init__()
        if residual_length_scale <= 0.0:
            raise ValueError("residual_length_scale must be positive")
        if model_spec.depth <= 0 or model_spec.width <= 0:
            raise ValueError("model depth and width must be positive")
        self.physics = physics
        self.full_state_dim = 3 * physics.num_vertices
        self.model_spec = model_spec
        self.activation = _activation(model_spec.activation)
        input_dim = 3 * self.full_state_dim
        hidden: list[nn.Linear] = []
        for _ in range(model_spec.depth):
            hidden.append(
                nn.Linear(
                    input_dim,
                    model_spec.width,
                    bias=model_spec.use_bias,
                    dtype=physics.dtype,
                    device=physics.device,
                )
            )
            input_dim = model_spec.width
        self.hidden_layers = nn.ModuleList(hidden)
        self.output_layer = nn.Linear(
            model_spec.width,
            self.full_state_dim,
            bias=model_spec.use_bias,
            dtype=physics.dtype,
            device=physics.device,
        )
        if initialize:
            gain = _activation_gain(model_spec.activation)
            for layer in self.hidden_layers:
                nn.init.orthogonal_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.output_layer.weight)
            if self.output_layer.bias is not None:
                nn.init.zeros_(self.output_layer.bias)
        self.register_buffer(
            "residual_length_scale",
            torch.tensor(float(residual_length_scale), dtype=physics.dtype, device=physics.device),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def current_preconditioned_residual(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        fixed_targets: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.physics.stationarity_residual(y, q, fixed_targets)
        preconditioned = self.physics.mass_preconditioned_residual(residual)
        return preconditioned.reshape(preconditioned.shape[0], -1).detach()

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        fixed_targets: torch.Tensor,
        *,
        previous_residual: torch.Tensor | None = None,
        previous_update: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.physics.check_state(y, "y")
        current = self.current_preconditioned_residual(y, q, fixed_targets)
        if previous_residual is None:
            previous_residual = torch.zeros_like(current)
        if previous_update is None:
            previous_update = torch.zeros_like(current)
        expected = (y.shape[0], self.full_state_dim)
        for name, value in (
            ("current_residual", current),
            ("previous_residual", previous_residual),
            ("previous_update", previous_update),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        hidden = torch.cat((current, previous_residual, previous_update), dim=-1)
        hidden = hidden / self.residual_length_scale
        for layer in self.hidden_layers:
            hidden = self.activation(layer(hidden))
        raw_delta = self.residual_length_scale * self.output_layer(hidden)
        gate = self.physics.free_update_gate(y.shape[0], dtype=raw_delta.dtype).reshape(y.shape[0], -1)
        return raw_delta * gate, current


def apply_model_update(
    model: LearnedOptimizerMLP,
    y: torch.Tensor,
    q: torch.Tensor,
    fixed_targets: torch.Tensor,
    *,
    previous_residual: torch.Tensor | None = None,
    previous_update: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta, current = model(
        y,
        q,
        fixed_targets,
        previous_residual=previous_residual,
        previous_update=previous_update,
    )
    y_next = y.reshape(y.shape[0], -1) + delta
    y_next = model.physics.project_positions(y_next.reshape_as(y), fixed_targets)
    return y_next, delta, current


def energy_scale(physics: TShirtPhysics) -> torch.Tensor:
    free = (~physics.fixed_mask).to(physics.dtype)
    mean_free_mass = torch.sum(physics.vertex_masses * free) / free.sum().clamp_min(1.0)
    rest_edge_lengths = torch.linalg.vector_norm(
        physics.rest_positions[physics.edges[:, 1]] - physics.rest_positions[physics.edges[:, 0]],
        dim=-1,
    )
    length = rest_edge_lengths.mean()
    inertial = mean_free_mass * length.square() / (physics.dt * physics.dt)
    membrane = (
        (physics.model.material.lame_mu + physics.model.material.lame_lambda)
        * physics.model.material.thickness
        * physics.face_areas.mean()
    )
    bending = physics.model.material.bending_stiffness * physics.hinge_rest_lengths.mean()
    return (inertial + membrane + bending).clamp_min(torch.finfo(physics.dtype).tiny)


@dataclass
class EnergyLossResult:
    loss: torch.Tensor
    normalized_change: torch.Tensor
    energy_before: torch.Tensor
    energy_after: torch.Tensor
    scale: torch.Tensor
    step_regularizer: torch.Tensor


def normalized_one_step_energy_loss(
    *,
    physics: TShirtPhysics,
    y_before: torch.Tensor,
    y_after: torch.Tensor,
    q: torch.Tensor,
    fixed_targets: torch.Tensor,
    delta: torch.Tensor,
    step_regularization_weight: float = 0.0,
) -> EnergyLossResult:
    energy_before = physics.variational_energy(y_before, q, fixed_targets)
    energy_after = physics.variational_energy(y_after, q, fixed_targets)
    scale = energy_scale(physics)
    normalized_change = (energy_after - energy_before.detach()) / scale
    delta_points = delta.reshape(y_before.shape)
    free = physics.free_update_gate(y_before.shape[0], dtype=y_before.dtype)
    free_count = (~physics.fixed_mask).sum().clamp_min(1).to(y_before.dtype)
    rest_edge_lengths = torch.linalg.vector_norm(
        physics.rest_positions[physics.edges[:, 1]] - physics.rest_positions[physics.edges[:, 0]],
        dim=-1,
    )
    characteristic_length = rest_edge_lengths.mean().clamp_min(1e-30)
    regularizer = torch.sum((delta_points * free).square(), dim=(-2, -1)) / (
        free_count * characteristic_length.square()
    )
    loss = normalized_change.mean()
    if step_regularization_weight:
        loss = loss + float(step_regularization_weight) * regularizer.mean()
    return EnergyLossResult(loss, normalized_change, energy_before, energy_after, scale, regularizer)


@dataclass
class PoolBatch:
    row_indices: torch.Tensor
    y: torch.Tensor
    q: torch.Tensor
    fixed_targets: torch.Tensor
    previous_residual: torch.Tensor
    previous_update: torch.Tensor
    k_values: torch.Tensor
    physical_steps: torch.Tensor
    motion_seeds: torch.Tensor


class OnlineTrainingPool:
    """Balanced K-bucket pool whose reset states are sampled online."""

    def __init__(
        self,
        *,
        physics: TShirtPhysics,
        seed: int,
        pool_size: int = DEFAULT_POOL_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        k_buckets: Sequence[int] = DEFAULT_K_BUCKETS,
        max_lifetime_physical_steps: int = DEFAULT_ENVIRONMENT_LIFETIME_FRAMES,
        dynamics: DynamicsDistribution = DEFAULT_DYNAMICS,
        max_energy: float = 1e12,
        max_residual: float = 1e12,
        max_abs_position: float = 1e4,
        min_simulation_area_ratio: float = 0.05,
        max_simulation_area_ratio: float = 20.0,
        min_simulation_edge_ratio: float = 0.05,
        max_simulation_edge_ratio: float = 20.0,
    ) -> None:
        self.physics = physics
        self.device = physics.device
        self.dtype = physics.dtype
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.dynamics = dynamics
        self.sampler_mesh = load_tshirt_mesh(DEFAULT_OBJ_PATH)
        if self.sampler_mesh.sha256 != physics.model.mesh_sha256:
            raise ValueError("Sampler OBJ hash does not match fixed model")
        self.pool_size = int(pool_size)
        self.batch_size = int(batch_size)
        self.k_buckets = tuple(int(value) for value in k_buckets)
        if not self.k_buckets or any(value <= 0 for value in self.k_buckets):
            raise ValueError("K buckets must be non-empty and positive")
        if self.pool_size % len(self.k_buckets) or self.batch_size % len(self.k_buckets):
            raise ValueError("pool_size and batch_size must be divisible by number of K buckets")
        if self.batch_size > self.pool_size:
            raise ValueError("batch_size must not exceed pool_size")
        self.rows_per_k = self.pool_size // len(self.k_buckets)
        self.batch_per_k = self.batch_size // len(self.k_buckets)
        self.max_lifetime_physical_steps = int(max_lifetime_physical_steps)
        self.max_energy = float(max_energy)
        self.max_residual = float(max_residual)
        self.max_abs_position = float(max_abs_position)
        self.min_simulation_area_ratio = float(min_simulation_area_ratio)
        self.max_simulation_area_ratio = float(max_simulation_area_ratio)
        self.min_simulation_edge_ratio = float(min_simulation_edge_ratio)
        self.max_simulation_edge_ratio = float(max_simulation_edge_ratio)

        k_values: list[int] = []
        self.rows_by_k: dict[int, torch.Tensor] = {}
        for bucket_index, k_value in enumerate(self.k_buckets):
            start = bucket_index * self.rows_per_k
            rows = torch.arange(start, start + self.rows_per_k, device=self.device)
            self.rows_by_k[k_value] = rows
            k_values.extend([k_value] * self.rows_per_k)
        self.k = torch.as_tensor(k_values, dtype=torch.long, device=self.device)
        self.batch_cursors = {value: 0 for value in self.k_buckets}
        shape = (self.pool_size, physics.num_vertices, 3)
        self.p = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v = torch.zeros_like(self.p)
        self.q = torch.zeros_like(self.p)
        self.y = torch.zeros_like(self.p)
        self.fixed_targets = torch.zeros_like(self.p)
        self.previous_residual = torch.zeros(
            (self.pool_size, 3 * physics.num_vertices), dtype=self.dtype, device=self.device
        )
        self.previous_update = torch.zeros_like(self.previous_residual)
        self.inner_iteration = torch.zeros(self.pool_size, dtype=torch.long, device=self.device)
        self.physical_step = torch.zeros_like(self.inner_iteration)
        self.age_physical_step = torch.zeros_like(self.inner_iteration)
        self.motion_seeds = torch.zeros_like(self.inner_iteration)
        self.total_environment_updates = 0
        self.total_completed_physical_frames = 0
        self.total_sampled_motions = 0
        self.reset_counts = {
            "resets_total": 0,
            "resets_nonfinite": 0,
            "resets_energy": 0,
            "resets_residual": 0,
            "resets_position": 0,
            "resets_area": 0,
            "resets_edge": 0,
            "resets_lifetime": 0,
        }
        self.sampling_stats = {
            "position_sampling_attempts_total": 0,
            "high_frequency_velocity_rms_sum": 0.0,
            "velocity_rms_sum": 0.0,
        }
        self._assign_new_motions(torch.arange(self.pool_size, device=self.device))

    @torch.no_grad()
    def _assign_new_motions(self, rows: torch.Tensor) -> None:
        rows = rows.to(device=self.device, dtype=torch.long)
        if rows.numel() == 0:
            return
        seeds = self.rng.integers(
            0, np.iinfo(np.int64).max, size=int(rows.numel()), dtype=np.int64
        )
        states = [
            sample_random_motion(
                self.sampler_mesh,
                self.physics.model,
                self.dynamics,
                seed=int(seed),
                motion_id=f"train_online_{self.total_sampled_motions + offset:012d}",
                split="train_online",
            )
            for offset, seed in enumerate(seeds)
        ]
        positions = torch.as_tensor(
            np.stack([state.positions for state in states]), dtype=self.dtype, device=self.device
        )
        velocities = torch.as_tensor(
            np.stack([state.velocities for state in states]), dtype=self.dtype, device=self.device
        )
        q = self.physics.make_q(positions, velocities)
        targets = positions.clone()
        y = self.physics.project_positions(positions, targets)
        self.p.index_copy_(0, rows, positions)
        self.v.index_copy_(0, rows, velocities)
        self.q.index_copy_(0, rows, q)
        self.y.index_copy_(0, rows, y)
        self.fixed_targets.index_copy_(0, rows, targets)
        self.previous_residual.index_fill_(0, rows, 0.0)
        self.previous_update.index_fill_(0, rows, 0.0)
        self.inner_iteration.index_fill_(0, rows, 0)
        self.physical_step.index_fill_(0, rows, 0)
        self.age_physical_step.index_fill_(0, rows, 0)
        self.motion_seeds.index_copy_(
            0, rows, torch.as_tensor(seeds, dtype=torch.long, device=self.device)
        )
        self.total_sampled_motions += len(states)
        self.sampling_stats["position_sampling_attempts_total"] += sum(
            int(state.metadata["position_sampling_attempts"]) for state in states
        )
        self.sampling_stats["high_frequency_velocity_rms_sum"] += sum(
            float(state.metadata["high_frequency_velocity_rms_requested"]) for state in states
        )
        self.sampling_stats["velocity_rms_sum"] += sum(
            float(state.metadata["velocity_rms"]) for state in states
        )

    def _take_cyclic_rows(self, k_value: int) -> torch.Tensor:
        rows = self.rows_by_k[k_value]
        cursor = self.batch_cursors[k_value]
        offsets = (torch.arange(self.batch_per_k, device=self.device) + cursor) % rows.numel()
        selected = rows.index_select(0, offsets)
        self.batch_cursors[k_value] = (cursor + self.batch_per_k) % int(rows.numel())
        return selected

    def next_batch_indices(self) -> torch.Tensor:
        return torch.cat([self._take_cyclic_rows(value) for value in self.k_buckets])

    def ask(self) -> PoolBatch:
        rows = self.next_batch_indices()
        return PoolBatch(
            row_indices=rows,
            y=self.y.index_select(0, rows).detach().clone(),
            q=self.q.index_select(0, rows).detach().clone(),
            fixed_targets=self.fixed_targets.index_select(0, rows).detach().clone(),
            previous_residual=self.previous_residual.index_select(0, rows).detach().clone(),
            previous_update=self.previous_update.index_select(0, rows).detach().clone(),
            k_values=self.k.index_select(0, rows),
            physical_steps=self.physical_step.index_select(0, rows),
            motion_seeds=self.motion_seeds.index_select(0, rows),
        )

    @torch.no_grad()
    def tell(
        self,
        batch: PoolBatch,
        *,
        y_next: torch.Tensor,
        delta: torch.Tensor,
        current_residual: torch.Tensor,
        energy_after: torch.Tensor,
        residual_after: torch.Tensor,
    ) -> dict[str, Any]:
        rows = batch.row_indices
        y_next = self.physics.check_state(y_next, "y_next").detach()
        delta = delta.detach()
        current_residual = current_residual.detach()
        energy_after = energy_after.detach()
        residual_after = residual_after.detach()
        finite_y = torch.isfinite(y_next).flatten(start_dim=1).all(dim=1)
        nonfinite = ~finite_y | ~torch.isfinite(energy_after) | ~torch.isfinite(residual_after)
        energy_bad = torch.isfinite(energy_after) & (energy_after.abs() > self.max_energy)
        residual_bad = torch.isfinite(residual_after) & (residual_after > self.max_residual)
        position_bad = finite_y & (y_next.abs().amax(dim=(-2, -1)) > self.max_abs_position)
        area_ratio = self.physics.triangle_area_ratios(y_next)
        edge_ratio = self.physics.edge_length_ratios(y_next)
        area_bad = (
            area_ratio.amin(dim=-1) < self.min_simulation_area_ratio
        ) | (area_ratio.amax(dim=-1) > self.max_simulation_area_ratio)
        edge_bad = (
            edge_ratio.amin(dim=-1) < self.min_simulation_edge_ratio
        ) | (edge_ratio.amax(dim=-1) > self.max_simulation_edge_ratio)
        bad = nonfinite | energy_bad | residual_bad | position_bad | area_bad | edge_bad

        good_local = torch.nonzero(~bad, as_tuple=False).flatten()
        if good_local.numel():
            good_rows = rows.index_select(0, good_local)
            self.y.index_copy_(0, good_rows, y_next.index_select(0, good_local))
            self.previous_update.index_copy_(0, good_rows, delta.index_select(0, good_local))
            self.previous_residual.index_copy_(0, good_rows, current_residual.index_select(0, good_local))
            self.inner_iteration.index_add_(0, good_rows, torch.ones_like(good_rows))

        completed_local = torch.nonzero(
            (~bad) & (self.inner_iteration.index_select(0, rows) >= batch.k_values),
            as_tuple=False,
        ).flatten()
        completed_rows = rows.index_select(0, completed_local)
        if completed_local.numel():
            old_p = self.p.index_select(0, completed_rows)
            solved = self.y.index_select(0, completed_rows)
            targets = self.fixed_targets.index_select(0, completed_rows)
            p_next, v_next = self.physics.advance_state(old_p, solved, targets)
            self.p.index_copy_(0, completed_rows, p_next)
            self.v.index_copy_(0, completed_rows, v_next)
            self.q.index_copy_(0, completed_rows, self.physics.make_q(p_next, v_next))
            self.y.index_copy_(0, completed_rows, self.physics.project_positions(p_next, targets))
            self.previous_residual.index_fill_(0, completed_rows, 0.0)
            self.previous_update.index_fill_(0, completed_rows, 0.0)
            self.inner_iteration.index_fill_(0, completed_rows, 0)
            self.physical_step.index_add_(0, completed_rows, torch.ones_like(completed_rows))
            self.age_physical_step.index_add_(0, completed_rows, torch.ones_like(completed_rows))
            self.total_completed_physical_frames += int(completed_rows.numel())

        lifetime = self.age_physical_step.index_select(0, rows) >= self.max_lifetime_physical_steps
        reset_local = torch.nonzero(bad | lifetime, as_tuple=False).flatten()
        reset_rows = rows.index_select(0, reset_local)
        if reset_rows.numel():
            self._assign_new_motions(reset_rows)

        def count(mask: torch.Tensor) -> int:
            return int(mask.sum().item())

        counts = {
            "resets_total": count(bad | lifetime),
            "resets_nonfinite": count(nonfinite),
            "resets_energy": count(energy_bad),
            "resets_residual": count(residual_bad),
            "resets_position": count(position_bad),
            "resets_area": count(area_bad),
            "resets_edge": count(edge_bad),
            "resets_lifetime": count(lifetime),
        }
        for key, value in counts.items():
            self.reset_counts[key] += value
        self.total_environment_updates += int(rows.numel())
        counts.update(
            {
                "completed_physical_frames": int(completed_rows.numel()),
                "total_sampled_motions": int(self.total_sampled_motions),
            }
        )
        return counts

    def manifest(self) -> dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "batch_size": self.batch_size,
            "k_buckets": list(self.k_buckets),
            "rows_per_k": self.rows_per_k,
            "batch_per_k": self.batch_per_k,
            "max_lifetime_physical_steps": self.max_lifetime_physical_steps,
            "train_seed": self.seed,
            "dynamics_distribution": asdict(self.dynamics),
            "training_samples_persisted": False,
            "sample_policy": "new independent motion at every environment reset",
        }

    def state_dict(self) -> dict[str, Any]:
        tensor_names = (
            "p", "v", "q", "y", "fixed_targets", "previous_residual",
            "previous_update", "inner_iteration", "physical_step",
            "age_physical_step", "motion_seeds",
        )
        state = {name: getattr(self, name).detach().cpu() for name in tensor_names}
        state.update(
            {
                "model_mesh_sha256": self.physics.model.mesh_sha256,
                "rng_state": self.rng.bit_generator.state,
                "batch_cursors": dict(self.batch_cursors),
                "total_environment_updates": self.total_environment_updates,
                "total_completed_physical_frames": self.total_completed_physical_frames,
                "total_sampled_motions": self.total_sampled_motions,
                "reset_counts": dict(self.reset_counts),
                "sampling_stats": dict(self.sampling_stats),
            }
        )
        return state

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("model_mesh_sha256") != self.physics.model.mesh_sha256:
            raise ValueError("Pool checkpoint belongs to a different fixed model")
        tensor_names = (
            "p", "v", "q", "y", "fixed_targets", "previous_residual",
            "previous_update", "inner_iteration", "physical_step",
            "age_physical_step", "motion_seeds",
        )
        for name in tensor_names:
            destination = getattr(self, name)
            source = torch.as_tensor(state[name], dtype=destination.dtype, device=self.device)
            if tuple(source.shape) != tuple(destination.shape):
                raise ValueError(f"Pool checkpoint {name} shape mismatch")
            destination.copy_(source)
        self.rng.bit_generator.state = state["rng_state"]
        self.batch_cursors = {int(key): int(value) for key, value in state["batch_cursors"].items()}
        self.total_environment_updates = int(state["total_environment_updates"])
        self.total_completed_physical_frames = int(state["total_completed_physical_frames"])
        self.total_sampled_motions = int(state["total_sampled_motions"])
        self.reset_counts = {str(key): int(value) for key, value in state["reset_counts"].items()}
        self.sampling_stats = {
            str(key): float(value) if "sum" in str(key) else int(value)
            for key, value in state["sampling_stats"].items()
        }


def gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def training_step(
    *,
    model: LearnedOptimizerMLP,
    optimizer: torch.optim.Optimizer,
    pool: OnlineTrainingPool,
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM,
    step_regularization_weight: float = 0.0,
) -> dict[str, Any]:
    batch = pool.ask()
    optimizer.zero_grad(set_to_none=True)
    residual_before = pool.physics.stationarity_residual_norm(
        batch.y, batch.q, batch.fixed_targets
    )
    y_next, delta, current = apply_model_update(
        model,
        batch.y,
        batch.q,
        batch.fixed_targets,
        previous_residual=batch.previous_residual,
        previous_update=batch.previous_update,
    )
    loss_result = normalized_one_step_energy_loss(
        physics=pool.physics,
        y_before=batch.y,
        y_after=y_next,
        q=batch.q,
        fixed_targets=batch.fixed_targets,
        delta=delta,
        step_regularization_weight=step_regularization_weight,
    )
    loss_result.loss.backward()
    grad_before = gradient_norm(list(model.parameters()))
    if gradient_clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
    grad_after = gradient_norm(list(model.parameters()))
    optimizer.step()
    residual_after = pool.physics.stationarity_residual_norm(
        y_next.detach(), batch.q, batch.fixed_targets
    )
    pool_stats = pool.tell(
        batch,
        y_next=y_next,
        delta=delta,
        current_residual=current,
        energy_after=loss_result.energy_after,
        residual_after=residual_after,
    )
    eps = torch.finfo(residual_after.dtype).eps
    ratio = residual_after / (residual_before + eps)
    update_norm = torch.linalg.vector_norm(delta.detach(), dim=-1)
    metrics: dict[str, Any] = {
        "loss": float(loss_result.loss.detach().cpu()),
        "normalized_energy_change_mean": float(loss_result.normalized_change.mean().detach().cpu()),
        "normalized_energy_change_p95": float(torch.quantile(loss_result.normalized_change.detach(), 0.95).cpu()),
        "energy_increase_fraction": float((loss_result.normalized_change.detach() > 0).double().mean().cpu()),
        "residual_before_mean": float(residual_before.mean().detach().cpu()),
        "residual_after_mean": float(residual_after.mean().detach().cpu()),
        "residual_ratio_p50": float(torch.quantile(ratio.detach(), 0.50).cpu()),
        "residual_ratio_p95": float(torch.quantile(ratio.detach(), 0.95).cpu()),
        "update_norm_mean": float(update_norm.mean().cpu()),
        "update_norm_p95": float(torch.quantile(update_norm, 0.95).cpu()),
        "gradient_norm_before_clip": grad_before,
        "gradient_norm_after_clip": grad_after,
        "batch_size": pool.batch_size,
    }
    metrics.update(pool_stats)
    return metrics
