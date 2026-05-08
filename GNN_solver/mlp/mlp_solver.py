from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def _as_column_or_matrix(value, n: int, *, name: str, dtype, device) -> Tensor:
    """
    Accept scalar, [N], [N, 1], or [N, K].
    Return [N, 1] for scalar/[N], or keep [N, K].
    """
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=dtype, device=device)
    else:
        value = value.to(dtype=dtype, device=device)

    if value.ndim == 0:
        return value.view(1, 1).expand(n, 1)

    if value.ndim == 1:
        if value.shape[0] != n:
            raise ValueError(
                f"{name} must have shape [N], got {tuple(value.shape)} with N={n}"
            )
        return value[:, None]

    if value.ndim == 2:
        if value.shape[0] != n:
            raise ValueError(
                f"{name} must have shape [N, K], got {tuple(value.shape)} with N={n}"
            )
        return value

    raise ValueError(
        f"{name} must be scalar, [N], [N, 1], or [N, K], got {tuple(value.shape)}"
    )


def build_pinned_flag(
    num_nodes: int,
    pinned_idx=None,
    pinned_flag=None,
    *,
    dtype,
    device,
) -> Tensor:
    """
    Return pinned flag with shape [N, 1].

    The MLP uses this as an input feature only.
    Pinned vertices are still clamped outside the model by the training loop.
    """
    if pinned_flag is not None:
        return _as_column_or_matrix(
            pinned_flag,
            num_nodes,
            name="pinned_flag",
            dtype=dtype,
            device=device,
        )

    flag = torch.zeros(num_nodes, 1, dtype=dtype, device=device)
    if pinned_idx is not None:
        pinned_idx = torch.as_tensor(pinned_idx, dtype=torch.long, device=device)
        flag[pinned_idx] = 1.0
    return flag


def build_mlp_node_features(
    *,
    x_cur: Tensor,
    x_hat: Tensor,
    rest_pos: Tensor,
    mass: Tensor,
    mu_lame: Tensor,
    lambda_lame: Tensor,
    k_bending: Tensor,
    dt: Tensor,
    pinned_idx=None,
    pinned_flag=None,
) -> Tensor:
    """
    Per-node feature layout:

        [x_cur,
         x_hat,
         rest_pos,
         mass,
         mu_lame,
         lambda_lame,
         k_bending,
         dt,
         pinned_flag]

    Default dimension:
        3 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 1 = 15
    """
    if x_cur.shape != x_hat.shape:
        raise ValueError(
            f"x_cur and x_hat must have the same shape, got {x_cur.shape} and {x_hat.shape}"
        )

    if x_cur.shape != rest_pos.shape:
        raise ValueError(
            f"x_cur and rest_pos must have the same shape, got {x_cur.shape} and {rest_pos.shape}"
        )

    if x_cur.ndim != 2 or x_cur.shape[-1] != 3:
        raise ValueError(f"x_cur must have shape [N, 3], got {tuple(x_cur.shape)}")

    n = x_cur.shape[0]
    dtype = x_cur.dtype
    device = x_cur.device

    mass = _as_column_or_matrix(mass, n, name="mass", dtype=dtype, device=device)
    mu_lame = _as_column_or_matrix(mu_lame, n, name="mu_lame", dtype=dtype, device=device)
    lambda_lame = _as_column_or_matrix(
        lambda_lame,
        n,
        name="lambda_lame",
        dtype=dtype,
        device=device,
    )
    k_bending = _as_column_or_matrix(
        k_bending,
        n,
        name="k_bending",
        dtype=dtype,
        device=device,
    )
    dt_feat = _as_column_or_matrix(dt, n, name="dt", dtype=dtype, device=device)
    pin_feat = build_pinned_flag(
        n,
        pinned_idx=pinned_idx,
        pinned_flag=pinned_flag,
        dtype=dtype,
        device=device,
    )

    return torch.cat(
        [
            x_cur,
            x_hat,
            rest_pos,
            mass,
            mu_lame,
            lambda_lame,
            k_bending,
            dt_feat,
            pin_feat,
        ],
        dim=-1,
    )


class MLP(nn.Module):
    """
    Simple fully connected network.

    This is intentionally much simpler than the GNN baseline.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        layers = []

        prev_dim = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MLPIterationSolver(nn.Module):
    """
    Global MLP baseline for one implicit-Euler solver iteration.

    It has the same external forward interface as GNNIterationSolver:

        delta_x = solver(
            x_cur=x_cur,
            x_hat=x_hat,
            rest_pos=rest_pos,
            edge_index=edge_index,
            mass=mass,
            mu_lame=mu_lame,
            lambda_lame=lambda_lame,
            k_bending=k_bending,
            dt=dt,
            pinned_idx=pinned_idx,
        )

    Notes:
        - edge_index is accepted for interface compatibility but not used.
        - The model flattens all node features into one global vector.
        - The output is reshaped back to [N, 3].
        - Pinned vertices are still clamped outside the model, same as the GNN setup.
    """

    def __init__(
        self,
        *,
        num_vertices: int,
        node_feature_dim: int = 15,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        out_dim_per_vertex: int = 3,
    ) -> None:
        super().__init__()

        if num_vertices <= 0:
            raise ValueError("num_vertices must be positive")

        self.num_vertices = num_vertices
        self.node_feature_dim = node_feature_dim
        self.out_dim_per_vertex = out_dim_per_vertex

        in_dim = num_vertices * node_feature_dim
        out_dim = num_vertices * out_dim_per_vertex

        self.mlp = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )

    def forward(
        self,
        *,
        x_cur: Tensor,
        x_hat: Tensor,
        rest_pos: Tensor,
        edge_index: Tensor,
        mass: Tensor,
        mu_lame: Tensor,
        lambda_lame: Tensor,
        k_bending: Tensor,
        dt: Tensor,
        pinned_idx=None,
        pinned_flag=None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        del edge_index, batch

        if x_cur.ndim != 2 or x_cur.shape != (self.num_vertices, 3):
            raise ValueError(
                f"x_cur must have shape [{self.num_vertices}, 3], got {tuple(x_cur.shape)}"
            )

        x_hat = x_hat.to(dtype=x_cur.dtype, device=x_cur.device)
        rest_pos = rest_pos.to(dtype=x_cur.dtype, device=x_cur.device)

        node_feat = build_mlp_node_features(
            x_cur=x_cur,
            x_hat=x_hat,
            rest_pos=rest_pos,
            mass=mass,
            mu_lame=mu_lame,
            lambda_lame=lambda_lame,
            k_bending=k_bending,
            dt=dt,
            pinned_idx=pinned_idx,
            pinned_flag=pinned_flag,
        )

        if node_feat.shape[-1] != self.node_feature_dim:
            raise ValueError(
                f"node feature dim mismatch: got {node_feat.shape[-1]}, "
                f"expected {self.node_feature_dim}"
            )

        flat_feat = node_feat.reshape(1, self.num_vertices * self.node_feature_dim)

        flat_delta = self.mlp(flat_feat)

        delta_x = flat_delta.reshape(
            self.num_vertices,
            self.out_dim_per_vertex,
        )

        return delta_x


if __name__ == "__main__":
    dtype = torch.float32
    device = "cpu"

    rest_pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )

    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [0, 2],
            [1, 3],
            [2, 3],
        ],
        dtype=torch.long,
        device=device,
    )

    num_vertices = rest_pos.shape[0]

    x_cur = rest_pos.clone()
    x_hat = rest_pos.clone()

    mass = torch.ones(num_vertices, dtype=dtype, device=device)
    mu_lame = torch.full((num_vertices,), 10.0, dtype=dtype, device=device)
    lambda_lame = torch.full((num_vertices,), 10.0, dtype=dtype, device=device)
    k_bending = torch.full((num_vertices,), 0.1, dtype=dtype, device=device)
    dt = torch.tensor(0.03, dtype=dtype, device=device)
    pinned_idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    solver = MLPIterationSolver(
        num_vertices=num_vertices,
        node_feature_dim=15,
        hidden_dim=128,
        num_hidden_layers=3,
    ).to(device=device, dtype=dtype)

    delta_x = solver(
        x_cur=x_cur,
        x_hat=x_hat,
        rest_pos=rest_pos,
        edge_index=edge_index,
        mass=mass,
        mu_lame=mu_lame,
        lambda_lame=lambda_lame,
        k_bending=k_bending,
        dt=dt,
        pinned_idx=pinned_idx,
    )

    print("delta_x shape:", tuple(delta_x.shape))
    print("delta_x dtype:", delta_x.dtype)
    print(delta_x)