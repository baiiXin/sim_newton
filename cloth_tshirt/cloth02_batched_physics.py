"""Batched implicit-Euler physics for the fixed T-shirt model.

The membrane term is the stable Neo-Hookean shell energy used by NVIDIA
Newton's VBD implementation.  The additive rest-energy constant is subtracted;
this changes neither forces nor Hessians.  Bending uses a rest-angle dihedral
energy so the curved OBJ pose is stress free.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tshirt_config import DEFAULT_FIXED_DATA_DIR, FixedModelSpec, load_model_spec


Tensor = torch.Tensor


@dataclass
class FrozenMotionBatch:
    motion_ids: tuple[str, ...]
    positions: Tensor
    velocities: Tensor
    seeds: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.positions.shape[0])


@dataclass
class TShirtPhysics:
    model: FixedModelSpec
    rest_positions: Tensor
    faces: Tensor
    edges: Tensor
    face_areas: Tensor
    inv_dm: Tensor
    vertex_masses: Tensor
    hinge_indices: Tensor
    hinge_rest_angles: Tensor
    hinge_rest_lengths: Tensor
    fixed_mask: Tensor

    @property
    def device(self) -> torch.device:
        return self.rest_positions.device

    @property
    def dtype(self) -> torch.dtype:
        return self.rest_positions.dtype

    @property
    def num_vertices(self) -> int:
        return int(self.rest_positions.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def num_hinges(self) -> int:
        return int(self.hinge_indices.shape[0])

    @property
    def dt(self) -> float:
        return float(self.model.dt)

    @property
    def gravity(self) -> Tensor:
        return torch.as_tensor(self.model.gravity, dtype=self.dtype, device=self.device)

    def check_state(self, value: Tensor, name: str) -> Tensor:
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3 or value.shape[1:] != (self.num_vertices, 3):
            raise ValueError(
                f"{name} must have shape [B,{self.num_vertices},3] or "
                f"[{self.num_vertices},3], got {tuple(value.shape)}"
            )
        return value

    def free_update_gate(self, batch_size: int, *, dtype: torch.dtype | None = None) -> Tensor:
        dtype = self.dtype if dtype is None else dtype
        return (~self.fixed_mask).to(dtype).view(1, self.num_vertices, 1).expand(batch_size, -1, 3)

    def project_positions(self, positions: Tensor, fixed_targets: Tensor) -> Tensor:
        positions = self.check_state(positions, "positions")
        fixed_targets = self.check_state(fixed_targets, "fixed_targets")
        if fixed_targets.shape[0] not in (1, positions.shape[0]):
            raise ValueError("fixed_targets batch does not broadcast to positions")
        mask = self.fixed_mask.view(1, self.num_vertices, 1)
        return torch.where(mask, fixed_targets, positions)

    def make_q(self, positions: Tensor, velocities: Tensor) -> Tensor:
        positions = self.check_state(positions, "positions")
        velocities = self.check_state(velocities, "velocities")
        return positions + self.dt * velocities + (self.dt * self.dt) * self.gravity.view(1, 1, 3)

    def membrane_energy(self, positions: Tensor) -> Tensor:
        positions = self.check_state(positions, "positions")
        triangles = positions[:, self.faces]  # [B,F,3,3]
        ds = torch.stack(
            (triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0]),
            dim=-1,
        )  # [B,F,3,2]
        deformation = ds @ self.inv_dm.unsqueeze(0)
        f0 = deformation[..., 0]
        f1 = deformation[..., 1]
        f0_sq = torch.sum(f0 * f0, dim=-1)
        f1_sq = torch.sum(f1 * f1, dim=-1)
        f01 = torch.sum(f0 * f1, dim=-1)
        j_sq = torch.clamp(f0_sq * f1_sq - f01 * f01, min=1e-20)
        j_surface = torch.sqrt(j_sq)
        i_c = f0_sq + f1_sq

        mu = float(self.model.material.lame_mu)
        lambda_nh = float(self.model.material.lame_lambda + self.model.material.lame_mu)
        alpha = 1.0 + mu / lambda_nh
        rest_constant = 0.5 * lambda_nh * (1.0 - alpha) ** 2
        density = (
            0.5 * mu * (i_c - 2.0)
            + 0.5 * lambda_nh * (j_surface - alpha) ** 2
            - rest_constant
        )
        scale = self.face_areas * float(self.model.material.thickness)
        return torch.sum(density * scale.unsqueeze(0), dim=-1)

    def dihedral_angles(self, positions: Tensor) -> Tensor:
        positions = self.check_state(positions, "positions")
        hinge = self.hinge_indices
        x0 = positions[:, hinge[:, 0]]
        x1 = positions[:, hinge[:, 1]]
        x2 = positions[:, hinge[:, 2]]
        x3 = positions[:, hinge[:, 3]]
        edge = x3 - x2
        n0 = torch.linalg.cross(x2 - x0, x3 - x0, dim=-1)
        n1 = torch.linalg.cross(x3 - x1, x2 - x1, dim=-1)
        eps = torch.finfo(positions.dtype).eps
        edge_hat = edge / torch.linalg.vector_norm(edge, dim=-1, keepdim=True).clamp_min(eps)
        n0_hat = n0 / torch.linalg.vector_norm(n0, dim=-1, keepdim=True).clamp_min(eps)
        n1_hat = n1 / torch.linalg.vector_norm(n1, dim=-1, keepdim=True).clamp_min(eps)
        sine = torch.sum(torch.linalg.cross(n0_hat, n1_hat, dim=-1) * edge_hat, dim=-1)
        cosine = torch.sum(n0_hat * n1_hat, dim=-1)
        return torch.atan2(sine, cosine)

    def bending_energy(self, positions: Tensor) -> Tensor:
        theta = self.dihedral_angles(positions)
        delta = theta - self.hinge_rest_angles.unsqueeze(0)
        # Wrapped difference avoids the artificial 2*pi jump of atan2.
        delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        stiffness = float(self.model.material.bending_stiffness)
        return torch.sum(
            0.5 * stiffness * self.hinge_rest_lengths.unsqueeze(0) * delta * delta,
            dim=-1,
        )

    def internal_energy_components(self, positions: Tensor) -> dict[str, Tensor]:
        return {
            "membrane": self.membrane_energy(positions),
            "bending": self.bending_energy(positions),
        }

    def variational_energy_components(
        self,
        positions: Tensor,
        q: Tensor,
        fixed_targets: Tensor,
    ) -> dict[str, Tensor]:
        positions = self.project_positions(positions, fixed_targets)
        q = self.check_state(q, "q")
        free = (~self.fixed_mask).to(self.dtype).view(1, self.num_vertices)
        inertial = 0.5 * self.vertex_masses.view(1, self.num_vertices) / (self.dt * self.dt)
        inertial = torch.sum(
            inertial * free * torch.sum((positions - q) ** 2, dim=-1),
            dim=-1,
        )
        internal = self.internal_energy_components(positions)
        return {"inertial": inertial, **internal}

    def variational_energy(self, positions: Tensor, q: Tensor, fixed_targets: Tensor) -> Tensor:
        components = self.variational_energy_components(positions, q, fixed_targets)
        return components["inertial"] + components["membrane"] + components["bending"]

    def stationarity_residual(
        self,
        positions: Tensor,
        q: Tensor,
        fixed_targets: Tensor,
        *,
        create_graph: bool = False,
    ) -> Tensor:
        # Validation/baseline callers often use torch.no_grad(); the residual
        # still needs a short local autograd tape because it is dE/dy.
        with torch.enable_grad():
            positions = self.check_state(positions, "positions")
            if not positions.requires_grad:
                positions = positions.detach().requires_grad_(True)
            energy = self.variational_energy(positions, q, fixed_targets)
            (gradient,) = torch.autograd.grad(
                energy.sum(),
                positions,
                create_graph=create_graph,
                retain_graph=create_graph,
            )
        gate = self.free_update_gate(gradient.shape[0], dtype=gradient.dtype)
        return gradient * gate

    def stationarity_residual_norm(
        self,
        positions: Tensor,
        q: Tensor,
        fixed_targets: Tensor,
    ) -> Tensor:
        residual = self.stationarity_residual(positions, q, fixed_targets)
        return torch.linalg.vector_norm(residual.reshape(residual.shape[0], -1), dim=-1)

    def mass_preconditioned_residual(self, residual: Tensor) -> Tensor:
        residual = self.check_state(residual, "residual")
        inverse_mass = 1.0 / self.vertex_masses.clamp_min(torch.finfo(self.dtype).tiny)
        output = (self.dt * self.dt) * residual * inverse_mass.view(1, self.num_vertices, 1)
        return output * self.free_update_gate(output.shape[0], dtype=output.dtype)

    def edge_length_ratios(self, positions: Tensor) -> Tensor:
        positions = self.check_state(positions, "positions")
        current = torch.linalg.vector_norm(
            positions[:, self.edges[:, 1]] - positions[:, self.edges[:, 0]], dim=-1
        )
        rest = torch.linalg.vector_norm(
            self.rest_positions[self.edges[:, 1]] - self.rest_positions[self.edges[:, 0]], dim=-1
        )
        return current / rest.clamp_min(torch.finfo(self.dtype).tiny).unsqueeze(0)

    def triangle_area_ratios(self, positions: Tensor) -> Tensor:
        positions = self.check_state(positions, "positions")
        triangles = positions[:, self.faces]
        area = 0.5 * torch.linalg.vector_norm(
            torch.linalg.cross(
                triangles[:, :, 1] - triangles[:, :, 0],
                triangles[:, :, 2] - triangles[:, :, 0],
                dim=-1,
            ),
            dim=-1,
        )
        return area / self.face_areas.clamp_min(torch.finfo(self.dtype).tiny).unsqueeze(0)

    def advance_state(
        self,
        previous_positions: Tensor,
        solved_positions: Tensor,
        fixed_targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        previous_positions = self.check_state(previous_positions, "previous_positions")
        next_positions = self.project_positions(solved_positions, fixed_targets)
        next_velocities = (next_positions - previous_positions) / self.dt
        next_velocities = next_velocities * self.free_update_gate(
            next_velocities.shape[0], dtype=next_velocities.dtype
        )
        return next_positions.detach(), next_velocities.detach()

    def _membrane_block_hessian(self, positions: Tensor) -> Tensor:
        """Per-vertex 3x3 PSD membrane blocks from Newton's VBD kernel."""

        positions = self.check_state(positions, "positions")
        batch = positions.shape[0]
        triangles = positions[:, self.faces]
        ds = torch.stack(
            (triangles[:, :, 1] - triangles[:, :, 0], triangles[:, :, 2] - triangles[:, :, 0]),
            dim=-1,
        )
        deformation = ds @ self.inv_dm.unsqueeze(0)
        f0 = deformation[..., 0]
        f1 = deformation[..., 1]
        f0_sq = torch.sum(f0 * f0, dim=-1)
        f1_sq = torch.sum(f1 * f1, dim=-1)
        f01 = torch.sum(f0 * f1, dim=-1)
        j_sq = torch.clamp(f0_sq * f1_sq - f01 * f01, min=1e-20)
        j_surface = torch.sqrt(j_sq)
        inv_j = 1.0 / j_surface
        g0 = inv_j.unsqueeze(-1) * (f1_sq.unsqueeze(-1) * f0 - f01.unsqueeze(-1) * f1)
        g1 = inv_j.unsqueeze(-1) * (f0_sq.unsqueeze(-1) * f1 - f01.unsqueeze(-1) * f0)
        mu = float(self.model.material.lame_mu)
        lambda_nh = float(self.model.material.lame_lambda + self.model.material.lame_mu)
        alpha = 1.0 + mu / lambda_nh
        stress_scalar = lambda_nh * (j_surface - alpha)
        r_value = torch.clamp(stress_scalar, min=0.0) * inv_j
        c1 = lambda_nh - r_value
        identity = torch.eye(3, dtype=self.dtype, device=self.device).view(1, 1, 3, 3)
        blocks = torch.zeros(
            (batch, self.num_vertices, 3, 3), dtype=self.dtype, device=self.device
        )
        dm = self.inv_dm
        coefficients = (
            (-(dm[:, 0, 0] + dm[:, 1, 0]), -(dm[:, 0, 1] + dm[:, 1, 1])),
            (dm[:, 0, 0], dm[:, 0, 1]),
            (dm[:, 1, 0], dm[:, 1, 1]),
        )
        scale = (
            self.face_areas * float(self.model.material.thickness)
        ).view(1, self.num_faces, 1, 1)
        for local, (df0, df1) in enumerate(coefficients):
            a = df0.view(1, self.num_faces)
            b = df1.view(1, self.num_faces)
            d_j = g0 * a.unsqueeze(-1) + g1 * b.unsqueeze(-1)
            cross_column = f1 * a.unsqueeze(-1) - f0 * b.unsqueeze(-1)
            identity_coefficient = mu * (a * a + b * b) + r_value * (
                a * a * f1_sq + b * b * f0_sq - 2.0 * a * b * f01
            )
            local_block = identity_coefficient[..., None, None] * identity
            local_block = local_block + c1[..., None, None] * (
                d_j.unsqueeze(-1) * d_j.unsqueeze(-2)
            )
            local_block = local_block - r_value[..., None, None] * (
                cross_column.unsqueeze(-1) * cross_column.unsqueeze(-2)
            )
            local_block = local_block * scale
            index = self.faces[:, local].view(1, self.num_faces, 1, 1).expand(batch, -1, 3, 3)
            blocks.scatter_add_(1, index, local_block)
        return blocks

    @staticmethod
    def _skew(value: Tensor) -> Tensor:
        zeros = torch.zeros_like(value[..., 0])
        x, y, z = value.unbind(dim=-1)
        return torch.stack(
            (
                zeros, -z, y,
                z, zeros, -x,
                -y, x, zeros,
            ),
            dim=-1,
        ).reshape(value.shape[:-1] + (3, 3))

    @staticmethod
    def _normalized_derivative(length: Tensor, unit: Tensor, derivative: Tensor) -> Tensor:
        identity = torch.eye(3, dtype=unit.dtype, device=unit.device)
        projection = identity - unit.unsqueeze(-1) * unit.unsqueeze(-2)
        return (projection @ derivative) / length[..., None, None]

    def _bending_angle_derivatives(self, positions: Tensor) -> tuple[Tensor, ...]:
        positions = self.check_state(positions, "positions")
        h = self.hinge_indices
        x0 = positions[:, h[:, 0]]
        x1 = positions[:, h[:, 1]]
        x2 = positions[:, h[:, 2]]
        x3 = positions[:, h[:, 3]]
        x02, x03 = x2 - x0, x3 - x0
        x13, x12 = x3 - x1, x2 - x1
        edge = x3 - x2
        n0 = torch.linalg.cross(x02, x03, dim=-1)
        n1 = torch.linalg.cross(x13, x12, dim=-1)
        eps = 1e-12 if self.dtype == torch.float64 else 1e-6
        n0_length = torch.linalg.vector_norm(n0, dim=-1).clamp_min(eps)
        n1_length = torch.linalg.vector_norm(n1, dim=-1).clamp_min(eps)
        edge_length = torch.linalg.vector_norm(edge, dim=-1).clamp_min(eps)
        n0_hat = n0 / n0_length.unsqueeze(-1)
        n1_hat = n1 / n1_length.unsqueeze(-1)
        edge_hat = edge / edge_length.unsqueeze(-1)
        sine = torch.sum(torch.linalg.cross(n0_hat, n1_hat, dim=-1) * edge_hat, dim=-1)
        cosine = torch.sum(n0_hat * n1_hat, dim=-1)
        skew_edge = self._skew(edge)
        zero = torch.zeros_like(skew_edge)
        normal_derivatives = (
            (self._normalized_derivative(n0_length, n0_hat, skew_edge), zero),
            (zero, self._normalized_derivative(n1_length, n1_hat, -skew_edge)),
            (
                self._normalized_derivative(n0_length, n0_hat, -self._skew(x03)),
                self._normalized_derivative(n1_length, n1_hat, self._skew(x13)),
            ),
            (
                self._normalized_derivative(n0_length, n0_hat, self._skew(x02)),
                self._normalized_derivative(n1_length, n1_hat, -self._skew(x12)),
            ),
        )
        skew_n0 = self._skew(n0_hat)
        skew_n1 = self._skew(n1_hat)
        output: list[Tensor] = []
        for dn0, dn1 in normal_derivatives:
            d_sine_matrix = skew_n0 @ dn1 - skew_n1 @ dn0
            d_sine = (d_sine_matrix.transpose(-1, -2) @ edge_hat.unsqueeze(-1)).squeeze(-1)
            d_cosine = (
                (dn0.transpose(-1, -2) @ n1_hat.unsqueeze(-1)).squeeze(-1)
                + (dn1.transpose(-1, -2) @ n0_hat.unsqueeze(-1)).squeeze(-1)
            )
            output.append(d_sine * cosine.unsqueeze(-1) - d_cosine * sine.unsqueeze(-1))
        return tuple(output)

    def _bending_block_hessian(self, positions: Tensor) -> Tensor:
        positions = self.check_state(positions, "positions")
        batch = positions.shape[0]
        blocks = torch.zeros(
            (batch, self.num_vertices, 3, 3), dtype=self.dtype, device=self.device
        )
        derivatives = self._bending_angle_derivatives(positions)
        scale = (
            float(self.model.material.bending_stiffness) * self.hinge_rest_lengths
        ).view(1, self.num_hinges, 1, 1)
        for local, derivative in enumerate(derivatives):
            local_block = scale * (derivative.unsqueeze(-1) * derivative.unsqueeze(-2))
            index = self.hinge_indices[:, local].view(1, self.num_hinges, 1, 1).expand(batch, -1, 3, 3)
            blocks.scatter_add_(1, index, local_block)
        return blocks

    def block_diagonal_hessian(
        self,
        positions: Tensor,
        *,
        relative_eigenvalue_floor: float = 1e-6,
        absolute_eigenvalue_floor: float = 1e-9,
    ) -> Tensor:
        """Return an SPD 3x3 block-Jacobi approximation of d²E/dy²."""

        positions = self.check_state(positions, "positions")
        batch = positions.shape[0]
        identity = torch.eye(3, dtype=self.dtype, device=self.device).view(1, 1, 3, 3)
        inertial = self.vertex_masses / (self.dt * self.dt)
        blocks = inertial.view(1, self.num_vertices, 1, 1) * identity
        blocks = blocks.expand(batch, -1, -1, -1).clone()
        blocks += self._membrane_block_hessian(positions)
        blocks += self._bending_block_hessian(positions)
        blocks = 0.5 * (blocks + blocks.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(blocks)
        scale = torch.mean(torch.abs(eigenvalues), dim=-1, keepdim=True)
        floor = absolute_eigenvalue_floor + relative_eigenvalue_floor * scale
        eigenvalues = torch.maximum(eigenvalues, floor)
        blocks = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
        fixed = self.fixed_mask.view(1, self.num_vertices, 1, 1)
        return torch.where(fixed, identity.expand_as(blocks), blocks)

    def block_hessian_preconditioned_residual(self, positions: Tensor, residual: Tensor) -> Tensor:
        positions = self.check_state(positions, "positions")
        residual = self.check_state(residual, "residual")
        blocks = self.block_diagonal_hessian(positions)
        direction = torch.linalg.solve(blocks, residual.unsqueeze(-1)).squeeze(-1)
        return direction * self.free_update_gate(direction.shape[0], dtype=direction.dtype)


def load_physics(
    *,
    fixed_data_dir: Path = DEFAULT_FIXED_DATA_DIR,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> TShirtPhysics:
    fixed_data_dir = Path(fixed_data_dir)
    model = load_model_spec(fixed_data_dir / "model_spec.json")
    with np.load(fixed_data_dir / "topology_cache.npz") as cache:
        arrays = {name: np.asarray(cache[name]) for name in cache.files}
    to_float = lambda name: torch.as_tensor(arrays[name], dtype=dtype, device=device)
    to_long = lambda name: torch.as_tensor(arrays[name], dtype=torch.long, device=device)
    fixed_mask = torch.zeros(model.num_vertices, dtype=torch.bool, device=device)
    fixed_mask[torch.as_tensor(model.fixed_indices, dtype=torch.long, device=device)] = True
    return TShirtPhysics(
        model=model,
        rest_positions=to_float("rest_positions"),
        faces=to_long("faces"),
        edges=to_long("edges"),
        face_areas=to_float("face_areas"),
        inv_dm=to_float("inv_dm"),
        vertex_masses=to_float("vertex_masses"),
        hinge_indices=to_long("hinge_indices"),
        hinge_rest_angles=to_float("hinge_rest_angles"),
        hinge_rest_lengths=to_float("hinge_rest_lengths"),
        fixed_mask=fixed_mask,
    )


def load_frozen_motion_batch(
    path: Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> FrozenMotionBatch:
    with np.load(path) as data:
        motion_ids = tuple(str(value) for value in data["motion_ids"].tolist())
        positions = np.asarray(data["positions"])
        velocities = np.asarray(data["velocities"])
        seeds = np.asarray(data["seeds"])
    return FrozenMotionBatch(
        motion_ids=motion_ids,
        positions=torch.as_tensor(positions, dtype=dtype, device=device),
        velocities=torch.as_tensor(velocities, dtype=dtype, device=device),
        seeds=torch.as_tensor(seeds, dtype=torch.long, device=device),
    )
