"""Mini-batch live training pool and learned optimizer for scale-up cloth scenarios.

The pool keeps a fixed number of live environments, while scenario definitions live
in a larger deterministic catalogue. Each optimizer step selects a balanced
mini-batch across K buckets. One selected environment receives exactly one learned
update; it advances one physical frame after K updates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import gcd, sqrt
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from cloth02_batched_physics import (
    BatchedClothParameters,
    advance_state,
    build_batched_parameters,
    dirichlet_targets,
    flatten_positions,
    free_update_gate,
    make_q,
    project_positions,
    spring_lengths,
    stationarity_residual,
    stationarity_residual_norm,
    variational_energy,
)
from scenario_templates import ScenarioSpec


DEFAULT_POOL_SIZE = 2048
DEFAULT_BATCH_SIZE = 256
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
    if name == "identity":
        return nn.Identity()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def _activation_gain(name: str) -> float:
    if name == "identity":
        return 1.0
    if name in {"relu", "gelu", "silu"}:
        return sqrt(2.0)
    if name == "tanh":
        return 5.0 / 3.0
    raise ValueError(f"Unsupported activation: {name}")


class LearnedOptimizerMLP(nn.Module):
    """History-input MLP with per-environment mass preconditioning and fixed gating."""

    def __init__(
        self,
        *,
        full_state_dim: int,
        residual_length_scale: float = DEFAULT_RESIDUAL_LENGTH_SCALE,
        model_spec: ModelSpec = ModelSpec(),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if full_state_dim <= 0:
            raise ValueError("full_state_dim must be positive")
        if residual_length_scale <= 0:
            raise ValueError("residual_length_scale must be positive")
        if model_spec.depth <= 0 or model_spec.width <= 0:
            raise ValueError("model depth and width must be positive")
        self.full_state_dim = int(full_state_dim)
        self.model_spec = model_spec
        self.activation = _activation(model_spec.activation)

        layers: list[nn.Linear] = []
        input_dim = 3 * self.full_state_dim
        for _ in range(model_spec.depth):
            layers.append(nn.Linear(input_dim, model_spec.width, bias=model_spec.use_bias, dtype=dtype))
            input_dim = model_spec.width
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(
            model_spec.width,
            self.full_state_dim,
            bias=model_spec.use_bias,
            dtype=dtype,
        )
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
            torch.tensor(float(residual_length_scale), dtype=dtype),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def current_preconditioned_residual(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        params: BatchedClothParameters,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        residual = stationarity_residual(y, q, params, target_positions)
        residual_points = residual.reshape(params.batch_size, params.num_vertices, 3)
        preconditioned = (
            params.dt**2
            * residual_points
            / params.masses.clamp_min(torch.finfo(params.dtype).tiny).unsqueeze(-1)
        )
        preconditioned = preconditioned * free_update_gate(params).to(params.dtype)
        return flatten_positions(preconditioned)

    def forward(
        self,
        y: torch.Tensor,
        q: torch.Tensor,
        params: BatchedClothParameters,
        *,
        target_positions: torch.Tensor,
        previous_residual: torch.Tensor | None = None,
        previous_update: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.current_preconditioned_residual(
            y,
            q,
            params,
            target_positions,
        )
        if previous_residual is None:
            previous_residual = torch.zeros_like(current)
        if previous_update is None:
            previous_update = torch.zeros_like(current)
        expected = (params.batch_size, self.full_state_dim)
        for name, value in (
            ("current_residual", current),
            ("previous_residual", previous_residual),
            ("previous_update", previous_update),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        h = torch.cat((current, previous_residual, previous_update), dim=-1)
        h = h / self.residual_length_scale
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        raw_delta = self.residual_length_scale * self.output_layer(h)
        gated_delta = raw_delta * free_update_gate(params, flattened=True).to(raw_delta.dtype)
        return gated_delta, current


def apply_model_update(
    model: LearnedOptimizerMLP,
    y: torch.Tensor,
    q: torch.Tensor,
    params: BatchedClothParameters,
    *,
    target_positions: torch.Tensor,
    previous_residual: torch.Tensor | None = None,
    previous_update: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_flat = y.reshape(params.batch_size, -1)
    delta, current = model(
        y_flat,
        q.reshape(params.batch_size, -1),
        params,
        target_positions=target_positions,
        previous_residual=previous_residual,
        previous_update=previous_update,
    )
    y_next = project_positions(
        y_flat + delta,
        params,
        target_positions,
    )
    return y_next, delta, current


def scenario_catalogue_fingerprint(scenarios: Sequence[ScenarioSpec]) -> str:
    payload = [
        asdict(scenario)
        for scenario in scenarios
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def per_environment_energy_scale(
    params: BatchedClothParameters,
    *,
    minimum: float = 1e-30,
) -> torch.Tensor:
    """Return a dimensional energy scale for each heterogeneous environment."""
    free = (~params.fixed_mask).to(params.dtype)
    free_count = free.sum(dim=-1).clamp_min(1.0)
    mean_free_mass = (params.masses * free).sum(dim=-1) / free_count
    characteristic_length = params.rest_lengths.mean(dim=-1)
    inertial_scale = (
        mean_free_mass
        * characteristic_length.square()
        / (params.dt**2)
    )
    spring_scale = (
        params.spring_stiffness
        * params.rest_lengths.square()
    ).mean(dim=-1)
    return (inertial_scale + spring_scale).clamp_min(float(minimum))


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
    y_before: torch.Tensor,
    y_after: torch.Tensor,
    q: torch.Tensor,
    delta: torch.Tensor,
    params: BatchedClothParameters,
    target_positions: torch.Tensor,
    step_regularization_weight: float = 0.0,
) -> EnergyLossResult:
    energy_before = variational_energy(
        y_before,
        q,
        params,
        target_positions,
    )
    energy_after = variational_energy(
        y_after,
        q,
        params,
        target_positions,
    )
    scale = per_environment_energy_scale(params)
    normalized_change = (
        energy_after - energy_before.detach()
    ) / scale
    delta_points = delta.reshape(params.batch_size, params.num_vertices, 3)
    free = free_update_gate(params).to(params.dtype)
    free_count = (~params.fixed_mask).sum(dim=-1).clamp_min(1)
    length = params.rest_lengths.mean(dim=-1).clamp_min(1e-30)
    step_regularizer = (
        torch.sum((delta_points * free).square(), dim=(-2, -1))
        / (free_count.to(params.dtype) * length.square())
    )
    loss = normalized_change.mean()
    if step_regularization_weight:
        loss = loss + float(step_regularization_weight) * step_regularizer.mean()
    return EnergyLossResult(
        loss=loss,
        normalized_change=normalized_change,
        energy_before=energy_before,
        energy_after=energy_after,
        scale=scale,
        step_regularizer=step_regularizer,
    )


def _coprime_step(total: int, preferred: int) -> int:
    if total <= 0:
        raise ValueError("total must be positive")
    step = int(preferred) % total
    if step == 0:
        step = 1
    while gcd(step, total) != 1:
        step += 1
    return step


@dataclass
class PoolBatch:
    row_indices: torch.Tensor
    params: BatchedClothParameters
    y: torch.Tensor
    q: torch.Tensor
    target_positions: torch.Tensor
    previous_residual: torch.Tensor
    previous_update: torch.Tensor
    k_values: torch.Tensor
    scenario_indices: torch.Tensor
    physical_steps: torch.Tensor


class LiveTrainingPool:
    """Deterministic mini-batch live pool with balanced K-bucket scheduling."""

    def __init__(
        self,
        *,
        scenarios: Sequence[ScenarioSpec],
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
        pool_size: int = DEFAULT_POOL_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        k_buckets: Sequence[int] = DEFAULT_K_BUCKETS,
        max_lifetime_physical_steps: int = DEFAULT_ENVIRONMENT_LIFETIME_FRAMES,
        scenario_offset: int = 0,
        scenario_step: int = 65537,
        max_energy: float = 1e12,
        max_residual: float = 1e12,
        max_abs_position: float = 1e4,
        min_spring_length: float = 1e-8,
        max_spring_length: float = 1e4,
    ) -> None:
        if not scenarios:
            raise ValueError("scenarios must not be empty")
        self.scenarios = tuple(scenarios)
        self.device = torch.device(device)
        self.dtype = dtype
        self.pool_size = int(pool_size)
        self.batch_size = int(batch_size)
        self.k_buckets = tuple(int(value) for value in k_buckets)
        if any(value <= 0 for value in self.k_buckets):
            raise ValueError("all K buckets must be positive")
        if self.pool_size <= 0 or self.batch_size <= 0:
            raise ValueError("pool_size and batch_size must be positive")
        if self.pool_size % len(self.k_buckets) != 0:
            raise ValueError("pool_size must be divisible by number of K buckets")
        if self.batch_size % len(self.k_buckets) != 0:
            raise ValueError("batch_size must be divisible by number of K buckets")
        if self.batch_size > self.pool_size:
            raise ValueError("batch_size must not exceed pool_size")
        self.rows_per_k = self.pool_size // len(self.k_buckets)
        self.batch_per_k = self.batch_size // len(self.k_buckets)
        if self.batch_per_k > self.rows_per_k:
            raise ValueError("batch asks for more rows per K than the pool owns")
        self.max_lifetime_physical_steps = int(max_lifetime_physical_steps)
        if self.max_lifetime_physical_steps <= 0:
            raise ValueError("max_lifetime_physical_steps must be positive")
        self.max_energy = float(max_energy)
        self.max_residual = float(max_residual)
        self.max_abs_position = float(max_abs_position)
        self.min_spring_length = float(min_spring_length)
        self.max_spring_length = float(max_spring_length)

        self.parameter_bank = build_batched_parameters(
            self.scenarios,
            dtype=dtype,
            device=self.device,
        )
        self.catalogue_fingerprint = scenario_catalogue_fingerprint(self.scenarios)
        self.scenario_offset = int(scenario_offset) % len(self.scenarios)
        self.scenario_step = _coprime_step(len(self.scenarios), int(scenario_step))
        self.scenario_cursor = 0
        self.total_scenario_assignments = 0
        self.seen_scenarios = torch.zeros(
            len(self.scenarios),
            dtype=torch.bool,
            device=self.device,
        )

        k_values: list[int] = []
        self.rows_by_k: dict[int, torch.Tensor] = {}
        for bucket_index, k_value in enumerate(self.k_buckets):
            start = bucket_index * self.rows_per_k
            rows = torch.arange(
                start,
                start + self.rows_per_k,
                dtype=torch.long,
                device=self.device,
            )
            self.rows_by_k[k_value] = rows
            k_values.extend([k_value] * self.rows_per_k)
        self.k = torch.as_tensor(k_values, dtype=torch.long, device=self.device)
        self.batch_cursors = {k: 0 for k in self.k_buckets}

        n = self.parameter_bank.num_vertices
        self.scenario_indices = torch.zeros(self.pool_size, dtype=torch.long, device=self.device)
        self.p = torch.zeros(self.pool_size, n, 3, dtype=dtype, device=self.device)
        self.v = torch.zeros_like(self.p)
        self.q = torch.zeros_like(self.p)
        self.y = torch.zeros_like(self.p)
        self.target_positions = torch.zeros_like(self.p)
        self.previous_residual = torch.zeros(self.pool_size, 3 * n, dtype=dtype, device=self.device)
        self.previous_update = torch.zeros_like(self.previous_residual)
        self.inner_iteration = torch.zeros(self.pool_size, dtype=torch.long, device=self.device)
        self.physical_step = torch.zeros(self.pool_size, dtype=torch.long, device=self.device)
        self.age_physical_step = torch.zeros(self.pool_size, dtype=torch.long, device=self.device)
        self.total_environment_updates = 0
        self.total_completed_physical_frames = 0
        self.reset_counts = {
            "resets_total": 0,
            "resets_nonfinite": 0,
            "resets_energy": 0,
            "resets_residual": 0,
            "resets_position": 0,
            "resets_spring": 0,
            "resets_lifetime": 0,
        }
        self._assign_new_scenarios(
            torch.arange(self.pool_size, dtype=torch.long, device=self.device)
        )

    def _next_scenario_indices(self, count: int) -> torch.Tensor:
        values = [
            (
                self.scenario_offset
                + (self.scenario_cursor + offset) * self.scenario_step
            )
            % len(self.scenarios)
            for offset in range(count)
        ]
        self.scenario_cursor += count
        self.total_scenario_assignments += count
        result = torch.as_tensor(values, dtype=torch.long, device=self.device)
        self.seen_scenarios[result] = True
        return result

    @torch.no_grad()
    def _assign_new_scenarios(self, rows: torch.Tensor) -> None:
        rows = rows.to(device=self.device, dtype=torch.long)
        if rows.numel() == 0:
            return
        scenario_indices = self._next_scenario_indices(int(rows.numel()))
        params = self.parameter_bank.index_select(scenario_indices)
        self.scenario_indices.index_copy_(0, rows, scenario_indices)
        self.p.index_copy_(0, rows, params.initial_positions)
        self.v.index_copy_(0, rows, params.initial_velocities)
        q = make_q(params.initial_positions, params.initial_velocities, params)
        next_time = torch.full(
            (rows.numel(),),
            params.dt,
            dtype=params.dtype,
            device=params.device,
        )
        targets, _ = dirichlet_targets(params, next_time)
        y = project_positions(params.initial_positions, params, targets)
        self.q.index_copy_(0, rows, q)
        self.target_positions.index_copy_(0, rows, targets)
        self.y.index_copy_(0, rows, y)
        self.previous_residual.index_fill_(0, rows, 0.0)
        self.previous_update.index_fill_(0, rows, 0.0)
        self.inner_iteration.index_fill_(0, rows, 0)
        self.physical_step.index_fill_(0, rows, 0)
        self.age_physical_step.index_fill_(0, rows, 0)

    def _take_cyclic_rows(self, k_value: int) -> torch.Tensor:
        rows = self.rows_by_k[k_value]
        cursor = self.batch_cursors[k_value]
        offsets = (
            torch.arange(self.batch_per_k, device=self.device)
            + int(cursor)
        ) % rows.numel()
        selected = rows.index_select(0, offsets)
        self.batch_cursors[k_value] = (
            int(cursor) + self.batch_per_k
        ) % rows.numel()
        return selected

    def next_batch_indices(self) -> torch.Tensor:
        return torch.cat(
            [self._take_cyclic_rows(k) for k in self.k_buckets],
            dim=0,
        )

    def ask(self) -> PoolBatch:
        rows = self.next_batch_indices()
        scenario_indices = self.scenario_indices.index_select(0, rows)
        params = self.parameter_bank.index_select(scenario_indices)
        return PoolBatch(
            row_indices=rows,
            params=params,
            y=self.y.index_select(0, rows).detach().clone(),
            q=self.q.index_select(0, rows).detach().clone(),
            target_positions=self.target_positions.index_select(0, rows).detach().clone(),
            previous_residual=self.previous_residual.index_select(0, rows).detach().clone(),
            previous_update=self.previous_update.index_select(0, rows).detach().clone(),
            k_values=self.k.index_select(0, rows),
            scenario_indices=scenario_indices,
            physical_steps=self.physical_step.index_select(0, rows),
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
        params = batch.params
        y_points = y_next.reshape(params.batch_size, params.num_vertices, 3).detach()
        delta = delta.detach()
        current_residual = current_residual.detach()
        energy_after = energy_after.detach()
        residual_after = residual_after.detach()

        lengths = spring_lengths(y_points, params, batch.target_positions)
        finite_y = torch.isfinite(y_points).flatten(start_dim=1).all(dim=1)
        nonfinite = (
            ~finite_y
            | ~torch.isfinite(energy_after)
            | ~torch.isfinite(residual_after)
        )
        energy_bad = torch.isfinite(energy_after) & (
            energy_after.abs() > self.max_energy
        )
        residual_bad = torch.isfinite(residual_after) & (
            residual_after > self.max_residual
        )
        position_bad = finite_y & (
            y_points.abs().amax(dim=(-2, -1)) > self.max_abs_position
        )
        spring_bad = (
            lengths.amin(dim=-1) < self.min_spring_length
        ) | (
            lengths.amax(dim=-1) > self.max_spring_length
        )
        bad = nonfinite | energy_bad | residual_bad | position_bad | spring_bad

        good_rows_local = torch.nonzero(~bad, as_tuple=False).flatten()
        if good_rows_local.numel():
            good_rows = rows.index_select(0, good_rows_local)
            self.y.index_copy_(0, good_rows, y_points.index_select(0, good_rows_local))
            self.previous_update.index_copy_(
                0,
                good_rows,
                delta.index_select(0, good_rows_local),
            )
            self.previous_residual.index_copy_(
                0,
                good_rows,
                current_residual.index_select(0, good_rows_local),
            )
            self.inner_iteration.index_add_(
                0,
                good_rows,
                torch.ones_like(good_rows, dtype=self.inner_iteration.dtype),
            )

        completed_local = torch.nonzero(
            (~bad)
            & (
                self.inner_iteration.index_select(0, rows)
                >= batch.k_values
            ),
            as_tuple=False,
        ).flatten()
        completed_rows = rows.index_select(0, completed_local)
        if completed_local.numel():
            completed_params = params.index_select(completed_local)
            current_p = self.p.index_select(0, completed_rows)
            solved = self.y.index_select(0, completed_rows)
            next_steps = self.physical_step.index_select(0, completed_rows) + 1
            next_time = next_steps.to(completed_params.dtype) * completed_params.dt
            p_next, v_next = advance_state(
                current_p,
                solved,
                completed_params,
                next_time=next_time,
            )
            self.p.index_copy_(0, completed_rows, p_next)
            self.v.index_copy_(0, completed_rows, v_next)
            self.physical_step.index_copy_(0, completed_rows, next_steps)
            self.age_physical_step.index_add_(
                0,
                completed_rows,
                torch.ones_like(completed_rows, dtype=self.age_physical_step.dtype),
            )
            q_next = make_q(p_next, v_next, completed_params)
            solve_time = (
                next_steps.to(completed_params.dtype) + 1.0
            ) * completed_params.dt
            targets_next, _ = dirichlet_targets(completed_params, solve_time)
            y0_next = project_positions(p_next, completed_params, targets_next)
            self.q.index_copy_(0, completed_rows, q_next)
            self.target_positions.index_copy_(0, completed_rows, targets_next)
            self.y.index_copy_(0, completed_rows, y0_next)
            self.previous_residual.index_fill_(0, completed_rows, 0.0)
            self.previous_update.index_fill_(0, completed_rows, 0.0)
            self.inner_iteration.index_fill_(0, completed_rows, 0)
            self.total_completed_physical_frames += int(completed_rows.numel())

        lifetime_local = torch.nonzero(
            self.age_physical_step.index_select(0, rows)
            >= self.max_lifetime_physical_steps,
            as_tuple=False,
        ).flatten()
        lifetime = torch.zeros_like(bad)
        if lifetime_local.numel():
            lifetime[lifetime_local] = True
        reset_local = torch.nonzero(bad | lifetime, as_tuple=False).flatten()
        reset_rows = rows.index_select(0, reset_local)
        if reset_rows.numel():
            self._assign_new_scenarios(reset_rows)

        def count(mask: torch.Tensor) -> int:
            return int(mask.sum().item())

        counts = {
            "resets_total": count(bad | lifetime),
            "resets_nonfinite": count(nonfinite),
            "resets_energy": count(energy_bad),
            "resets_residual": count(residual_bad),
            "resets_position": count(position_bad),
            "resets_spring": count(spring_bad),
            "resets_lifetime": count(lifetime),
        }
        for key, value in counts.items():
            self.reset_counts[key] += value
        self.total_environment_updates += int(rows.numel())
        counts.update(
            {
                "completed_physical_frames": int(completed_rows.numel()),
                "unique_scenarios_seen": int(self.seen_scenarios.sum().item()),
                "total_scenario_assignments": int(self.total_scenario_assignments),
            }
        )
        for k_value in self.k_buckets:
            select = self.k == k_value
            counts[f"physical_frames_k{k_value}"] = int(
                self.physical_step[select].sum().item()
            )
        return counts

    def manifest(self) -> dict[str, Any]:
        return {
            "pool_size": self.pool_size,
            "batch_size": self.batch_size,
            "catalogue_size": len(self.scenarios),
            "catalogue_fingerprint": self.catalogue_fingerprint,
            "k_buckets": list(self.k_buckets),
            "rows_per_k": self.rows_per_k,
            "batch_per_k": self.batch_per_k,
            "max_lifetime_physical_steps": self.max_lifetime_physical_steps,
            "scenario_scheduler": {
                "mode": "deterministic_coprime_ring",
                "offset": self.scenario_offset,
                "step": self.scenario_step,
            },
            "batch_scheduler": "deterministic_balanced_cyclic_K_buckets",
            "semantics": {
                "optimizer_step": "one learned update for batch_size live environments",
                "physical_frame": "advance an environment after its assigned K learned updates",
                "new_frame_initial_guess": "free vertices from x_n; fixed vertices projected to x_D(t_{n+1})",
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "catalogue_fingerprint": self.catalogue_fingerprint,
            "scenario_indices": self.scenario_indices.detach().cpu(),
            "p": self.p.detach().cpu(),
            "v": self.v.detach().cpu(),
            "q": self.q.detach().cpu(),
            "y": self.y.detach().cpu(),
            "target_positions": self.target_positions.detach().cpu(),
            "previous_residual": self.previous_residual.detach().cpu(),
            "previous_update": self.previous_update.detach().cpu(),
            "inner_iteration": self.inner_iteration.detach().cpu(),
            "physical_step": self.physical_step.detach().cpu(),
            "age_physical_step": self.age_physical_step.detach().cpu(),
            "scenario_cursor": int(self.scenario_cursor),
            "total_scenario_assignments": int(self.total_scenario_assignments),
            "seen_scenarios": self.seen_scenarios.detach().cpu(),
            "batch_cursors": dict(self.batch_cursors),
            "total_environment_updates": int(self.total_environment_updates),
            "total_completed_physical_frames": int(self.total_completed_physical_frames),
            "reset_counts": dict(self.reset_counts),
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("catalogue_fingerprint") != self.catalogue_fingerprint:
            raise ValueError("Checkpoint catalogue fingerprint does not match current catalogue")
        tensor_names = (
            "scenario_indices",
            "p",
            "v",
            "q",
            "y",
            "target_positions",
            "previous_residual",
            "previous_update",
            "inner_iteration",
            "physical_step",
            "age_physical_step",
            "seen_scenarios",
        )
        for name in tensor_names:
            destination = getattr(self, name)
            source = torch.as_tensor(
                state[name],
                dtype=destination.dtype,
                device=destination.device,
            )
            if tuple(source.shape) != tuple(destination.shape):
                raise ValueError(
                    f"Pool checkpoint {name} has shape {tuple(source.shape)}, "
                    f"expected {tuple(destination.shape)}"
                )
            destination.copy_(source)
        self.scenario_cursor = int(state["scenario_cursor"])
        self.total_scenario_assignments = int(state["total_scenario_assignments"])
        self.batch_cursors = {
            int(key): int(value)
            for key, value in state["batch_cursors"].items()
        }
        self.total_environment_updates = int(state["total_environment_updates"])
        self.total_completed_physical_frames = int(
            state["total_completed_physical_frames"]
        )
        self.reset_counts = {
            str(key): int(value)
            for key, value in state["reset_counts"].items()
        }


def gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    norms = [
        parameter.grad.detach().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def training_step(
    *,
    model: LearnedOptimizerMLP,
    optimizer: torch.optim.Optimizer,
    pool: LiveTrainingPool,
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM,
    step_regularization_weight: float = 0.0,
) -> dict[str, Any]:
    batch = pool.ask()
    optimizer.zero_grad(set_to_none=True)
    residual_before = stationarity_residual_norm(
        batch.y,
        batch.q,
        batch.params,
        batch.target_positions,
    )
    y_next, delta, current = apply_model_update(
        model,
        batch.y,
        batch.q,
        batch.params,
        target_positions=batch.target_positions,
        previous_residual=batch.previous_residual,
        previous_update=batch.previous_update,
    )
    loss_result = normalized_one_step_energy_loss(
        y_before=batch.y,
        y_after=y_next,
        q=batch.q,
        delta=delta,
        params=batch.params,
        target_positions=batch.target_positions,
        step_regularization_weight=step_regularization_weight,
    )
    loss_result.loss.backward()
    grad_before = gradient_norm(list(model.parameters()))
    if gradient_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(gradient_clip_norm),
        )
    grad_after = gradient_norm(list(model.parameters()))
    optimizer.step()
    residual_after = stationarity_residual_norm(
        y_next.detach(),
        batch.q,
        batch.params,
        batch.target_positions,
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
    residual_ratio = residual_after / (residual_before + eps)
    update_norm = torch.linalg.vector_norm(delta.detach(), dim=-1)
    metrics: dict[str, Any] = {
        "loss": float(loss_result.loss.detach().cpu()),
        "normalized_energy_change_mean": float(
            loss_result.normalized_change.detach().mean().cpu()
        ),
        "normalized_energy_change_p95": float(
            torch.quantile(loss_result.normalized_change.detach(), 0.95).cpu()
        ),
        "energy_increase_fraction": float(
            (loss_result.normalized_change.detach() > 0).double().mean().cpu()
        ),
        "residual_before_mean": float(residual_before.detach().mean().cpu()),
        "residual_after_mean": float(residual_after.detach().mean().cpu()),
        "residual_ratio_p50": float(torch.quantile(residual_ratio.detach(), 0.50).cpu()),
        "residual_ratio_p95": float(torch.quantile(residual_ratio.detach(), 0.95).cpu()),
        "update_norm_mean": float(update_norm.mean().cpu()),
        "update_norm_p95": float(torch.quantile(update_norm, 0.95).cpu()),
        "gradient_norm_before_clip": grad_before,
        "gradient_norm_after_clip": grad_after,
        "batch_size": int(pool.batch_size),
    }
    metrics.update(pool_stats)
    return metrics
