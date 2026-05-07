from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MetaLayer
from torch_geometric.utils import scatter


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Same structure as the reference MeshGraphNets-style MLP.

    Structure:
        Linear -> ReLU -> ... -> Linear -> optional LayerNorm(out_dim)

    Notes:
        - No input normalizer is used before encoders.
        - LayerNorm inside MLP is kept, matching the reference structure.
    """

    def __init__(
        self,
        in_dim: int,
        latent_size: int,
        out_dim: int,
        num_layers: int,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        widths = [latent_size] * num_layers + [out_dim]

        layers = []
        prev = in_dim
        for i, width in enumerate(widths):
            layers.append(nn.Linear(prev, width))
            if i != len(widths) - 1:
                layers.append(nn.ReLU())
            prev = width

        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.mlp(x))


def _as_column_or_matrix(value, n: int, *, name: str, dtype, device) -> Tensor:
    """
    Accepts scalar, [N], [N, 1], or [N, K].
    Returns [N, 1] for scalar/[N], or keeps [N, K].
    """
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=dtype, device=device)
    else:
        value = value.to(dtype=dtype, device=device)

    if value.ndim == 0:
        return value.view(1, 1).expand(n, 1)

    if value.ndim == 1:
        if value.shape[0] != n:
            raise ValueError(f"{name} must have shape [N], got {tuple(value.shape)} with N={n}")
        return value[:, None]

    if value.ndim == 2:
        if value.shape[0] != n:
            raise ValueError(f"{name} must have shape [N, K], got {tuple(value.shape)} with N={n}")
        return value

    raise ValueError(f"{name} must be scalar, [N], [N,1], or [N,K], got {tuple(value.shape)}")


def build_pinned_flag(
    num_nodes: int,
    pinned_idx=None,
    pinned_flag=None,
    *,
    dtype,
    device,
) -> Tensor:
    """
    Returns pinned flag with shape [N, 1].

    The model uses this only as input information. It does not clamp internally.
    Clamp outside the model if needed.
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


def edge_index_to_pyg(edge_index: Tensor) -> Tensor:
    """
    Accepts either [E, 2] or [2, E].
    Returns PyG-style [2, E].
    """
    if edge_index.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {tuple(edge_index.shape)}")

    if edge_index.shape[0] == 2:
        return edge_index.long()

    if edge_index.shape[1] == 2:
        return edge_index.t().contiguous().long()

    raise ValueError(f"edge_index must have shape [E,2] or [2,E], got {tuple(edge_index.shape)}")


def make_bidirectional_edges(edge_index: Tensor, *, remove_duplicates: bool = True) -> Tensor:
    """
    Converts input edges to directed bidirectional edge_index [2, E_dir].
    External code may pass [E, 2]; internally we use [2, E].
    """
    edge_index = edge_index_to_pyg(edge_index)
    src, dst = edge_index[0], edge_index[1]

    rev = torch.stack([dst, src], dim=0)
    out = torch.cat([edge_index, rev], dim=1)

    if remove_duplicates:
        # unique over columns; order is not important for scatter aggregation.
        out = torch.unique(out.t(), dim=0).t().contiguous()

    return out


# -----------------------------------------------------------------------------
# Feature builders
# -----------------------------------------------------------------------------

def build_node_features(
    *,
    x_cur: Tensor,
    x_hat: Tensor,
    mass: Tensor,
    mu_lame: Tensor,
    lambda_lame: Tensor,
    k_bending: Tensor,
    dt: Tensor,
    pinned_idx=None,
    pinned_flag=None,
) -> Tensor:
    """
    Node feature layout:
        [x_cur, x_hat, mass, mu_lame, lambda_lame, k_bending, dt, pinned_flag]

    Expected base dimension when all scalar per-node fields are scalar/[N]/[N,1]:
        3 + 3 + 1 + 1 + 1 + 1 + 1 + 1 = 12
    """
    if x_cur.shape != x_hat.shape:
        raise ValueError(f"x_cur and x_hat must have the same shape, got {x_cur.shape} and {x_hat.shape}")
    if x_cur.ndim != 2 or x_cur.shape[-1] != 3:
        raise ValueError(f"x_cur must have shape [N,3], got {tuple(x_cur.shape)}")

    n = x_cur.shape[0]
    dtype = x_cur.dtype
    device = x_cur.device

    mass = _as_column_or_matrix(mass, n, name="mass", dtype=dtype, device=device)
    mu_lame = _as_column_or_matrix(mu_lame, n, name="mu_lame", dtype=dtype, device=device)
    lambda_lame = _as_column_or_matrix(lambda_lame, n, name="lambda_lame", dtype=dtype, device=device)
    k_bending = _as_column_or_matrix(k_bending, n, name="k_bending", dtype=dtype, device=device)
    dt_feat = _as_column_or_matrix(dt, n, name="dt", dtype=dtype, device=device)
    pin_feat = build_pinned_flag(n, pinned_idx, pinned_flag, dtype=dtype, device=device)

    return torch.cat(
        [x_cur, x_hat, mass, mu_lame, lambda_lame, k_bending, dt_feat, pin_feat],
        dim=-1,
    )


def build_edge_features(
    *,
    x_cur: Tensor,
    rest_pos: Tensor,
    edge_index_dir: Tensor,
    mu_lame: Tensor,
    lambda_lame: Tensor,
    k_bending: Tensor,
    dt: Tensor,
    eps: float = 1.0e-12,
) -> Tensor:
    """
    Edge feature layout for each directed edge i -> j:
        [x_i - x_j,
         rest_i - rest_j,
         ||x_i - x_j||,
         ||rest_i - rest_j||,
         avg(mu_lame_i, mu_lame_j),
         avg(lambda_lame_i, lambda_lame_j),
         avg(k_bending_i, k_bending_j),
         dt]

    Expected base dimension when material fields are scalar per node:
        3 + 3 + 1 + 1 + 1 + 1 + 1 + 1 = 12
    """
    if x_cur.shape != rest_pos.shape:
        raise ValueError(f"x_cur and rest_pos must have the same shape, got {x_cur.shape} and {rest_pos.shape}")
    if edge_index_dir.shape[0] != 2:
        raise ValueError(f"edge_index_dir must have shape [2,E], got {tuple(edge_index_dir.shape)}")

    n = x_cur.shape[0]
    e = edge_index_dir.shape[1]
    dtype = x_cur.dtype
    device = x_cur.device

    src, dst = edge_index_dir[0], edge_index_dir[1]

    cur_vec = x_cur[src] - x_cur[dst]
    rest_vec = rest_pos[src] - rest_pos[dst]
    cur_len = torch.linalg.norm(cur_vec, dim=-1, keepdim=True).clamp_min(eps)
    rest_len = torch.linalg.norm(rest_vec, dim=-1, keepdim=True).clamp_min(eps)

    mu_lame = _as_column_or_matrix(mu_lame, n, name="mu_lame", dtype=dtype, device=device)
    lambda_lame = _as_column_or_matrix(lambda_lame, n, name="lambda_lame", dtype=dtype, device=device)
    k_bending = _as_column_or_matrix(k_bending, n, name="k_bending", dtype=dtype, device=device)

    mu_edge = 0.5 * (mu_lame[src] + mu_lame[dst])
    lambda_edge = 0.5 * (lambda_lame[src] + lambda_lame[dst])
    kb_edge = 0.5 * (k_bending[src] + k_bending[dst])

    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, dtype=dtype, device=device)
    else:
        dt = dt.to(dtype=dtype, device=device)

    if dt.ndim == 0:
        dt_edge = dt.view(1, 1).expand(e, 1)
    elif dt.ndim == 1 and dt.shape[0] == e:
        dt_edge = dt[:, None]
    elif dt.ndim == 2 and dt.shape[0] == e:
        dt_edge = dt
    else:
        raise ValueError(f"dt must be scalar, [E], or [E,K] for edge features, got {tuple(dt.shape)}")

    return torch.cat(
        [cur_vec, rest_vec, cur_len, rest_len, mu_edge, lambda_edge, kb_edge, dt_edge],
        dim=-1,
    )


# -----------------------------------------------------------------------------
# MeshGraphNets-style encode-process-decode network
# -----------------------------------------------------------------------------

class EdgeModel(nn.Module):
    def __init__(self, latent_size: int, num_layers: int) -> None:
        super().__init__()
        self.edge_mlp = MLP(
            in_dim=latent_size * 3,
            latent_size=latent_size,
            out_dim=latent_size,
            num_layers=num_layers,
            layer_norm=True,
        )

    def forward(
        self,
        src: Tensor,
        dst: Tensor,
        edge_attr: Tensor,
        u: Optional[Tensor],
        batch: Optional[Tensor],
    ) -> Tensor:
        del u, batch
        out = torch.cat([src, dst, edge_attr], dim=-1)
        return self.edge_mlp(out)


class NodeModel(nn.Module):
    def __init__(self, latent_size: int, num_layers: int) -> None:
        super().__init__()
        self.node_mlp = MLP(
            in_dim=latent_size * 2,
            latent_size=latent_size,
            out_dim=latent_size,
            num_layers=num_layers,
            layer_norm=True,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        u: Optional[Tensor],
        batch: Optional[Tensor],
    ) -> Tensor:
        del u, batch
        senders, receivers = edge_index
        del senders
        agg = scatter(edge_attr, receivers, dim=0, dim_size=x.size(0), reduce="sum")
        out = torch.cat([x, agg], dim=-1)
        return self.node_mlp(out)


class GraphNetBlock(nn.Module):
    """
    One processor step: edge update -> node update -> residual add.

    This matches the reference residual convention:
        new_x, new_edge_attr = MetaLayer(...)
        x = x + new_x
        edge_attr = edge_attr + new_edge_attr
    """

    def __init__(self, latent_size: int, num_layers: int) -> None:
        super().__init__()
        self.meta = MetaLayer(
            edge_model=EdgeModel(latent_size, num_layers),
            node_model=NodeModel(latent_size, num_layers),
            global_model=None,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        new_x, new_edge_attr, _ = self.meta(
            x,
            edge_index,
            edge_attr=edge_attr,
            u=None,
            batch=batch,
        )
        x = x + new_x
        edge_attr = edge_attr + new_edge_attr
        return x, edge_attr


class EncodeProcessDecode(nn.Module):
    """
    MeshGraphNets-style encode-process-decode network.

    Differences from the uploaded reference cloth network:
        - no OnlineNormalizer / input normalizer before encoders,
        - feature construction is for implicit Euler iteration,
        - decoder returns delta_x directly.
    """

    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        latent_size: int = 128,
        num_layers: int = 2,
        message_passing_steps: int = 15,
    ) -> None:
        super().__init__()
        self.node_encoder = MLP(
            in_dim=node_input_size,
            latent_size=latent_size,
            out_dim=latent_size,
            num_layers=num_layers,
            layer_norm=True,
        )
        self.edge_encoder = MLP(
            in_dim=edge_input_size,
            latent_size=latent_size,
            out_dim=latent_size,
            num_layers=num_layers,
            layer_norm=True,
        )
        self.processor = nn.ModuleList(
            [GraphNetBlock(latent_size, num_layers) for _ in range(message_passing_steps)]
        )
        self.decoder = MLP(
            in_dim=latent_size,
            latent_size=latent_size,
            out_dim=output_size,
            num_layers=num_layers,
            layer_norm=False,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for block in self.processor:
            x, edge_attr = block(x, edge_index, edge_attr, batch=batch)

        return self.decoder(x)


# -----------------------------------------------------------------------------
# Iterative delta-x solver network
# -----------------------------------------------------------------------------

class IterativeDeltaGNN(nn.Module):
    """
    GNN that returns delta_x for one solver iteration.

    This module does NOT:
        - perform time stepping,
        - compute x_hat,
        - update x_cur across iterations,
        - clamp pinned vertices,
        - compute loss.

    External expected usage:
        delta_x = model(...)
        x_next_iter = x_cur + delta_x
        x_next_iter = clamp_pinned_vertices(x_next_iter, reference_x, pinned_idx)
    """

    def __init__(
        self,
        node_in_dim: int = 12,
        edge_in_dim: int = 12,
        latent_size: int = 128,
        num_layers: int = 2,
        message_passing_steps: int = 15,
        out_dim: int = 3,
        remove_duplicate_edges: bool = True,
    ) -> None:
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.message_passing_steps = message_passing_steps
        self.out_dim = out_dim
        self.remove_duplicate_edges = remove_duplicate_edges

        self.learned_model = EncodeProcessDecode(
            node_input_size=node_in_dim,
            edge_input_size=edge_in_dim,
            output_size=out_dim,
            latent_size=latent_size,
            num_layers=num_layers,
            message_passing_steps=message_passing_steps,
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
        """
        Returns:
            delta_x: [N, 3]
        """
        # Project convention: float32.
        x_cur = x_cur.to(dtype=torch.float32)
        x_hat = x_hat.to(dtype=x_cur.dtype, device=x_cur.device)
        rest_pos = rest_pos.to(dtype=x_cur.dtype, device=x_cur.device)
        edge_index = edge_index.to(device=x_cur.device)
        if batch is not None:
            batch = batch.to(device=x_cur.device)

        edge_index_dir = make_bidirectional_edges(
            edge_index,
            remove_duplicates=self.remove_duplicate_edges,
        )

        node_feat = build_node_features(
            x_cur=x_cur,
            x_hat=x_hat,
            mass=mass,
            mu_lame=mu_lame,
            lambda_lame=lambda_lame,
            k_bending=k_bending,
            dt=dt,
            pinned_idx=pinned_idx,
            pinned_flag=pinned_flag,
        )

        edge_feat = build_edge_features(
            x_cur=x_cur,
            rest_pos=rest_pos,
            edge_index_dir=edge_index_dir,
            mu_lame=mu_lame,
            lambda_lame=lambda_lame,
            k_bending=k_bending,
            dt=dt,
        )

        if node_feat.shape[-1] != self.node_in_dim:
            raise ValueError(
                f"node feature dim mismatch: got {node_feat.shape[-1]}, expected {self.node_in_dim}. "
                "Adjust node_in_dim or feature builder."
            )
        if edge_feat.shape[-1] != self.edge_in_dim:
            raise ValueError(
                f"edge feature dim mismatch: got {edge_feat.shape[-1]}, expected {self.edge_in_dim}. "
                "Adjust edge_in_dim or feature builder."
            )

        delta_x = self.learned_model(
            x=node_feat,
            edge_index=edge_index_dir,
            edge_attr=edge_feat,
            batch=batch,
        )
        return delta_x


class GNNIterationSolver(nn.Module):
    """
    Thin wrapper whose forward returns delta_x only.

    This is intentionally not a time-stepper. You can call it multiple times
    inside one physical time step if you want iterative refinement.
    """

    def __init__(
        self,
        node_in_dim: int = 12,
        edge_in_dim: int = 12,
        latent_size: int = 128,
        num_layers: int = 2,
        message_passing_steps: int = 15,
    ) -> None:
        super().__init__()
        self.gnn = IterativeDeltaGNN(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            latent_size=latent_size,
            num_layers=num_layers,
            message_passing_steps=message_passing_steps,
            out_dim=3,
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
        return self.gnn(
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
            pinned_flag=pinned_flag,
            batch=batch,
        )


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cpu"
    dtype = torch.float32

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

    # External format [E, 2] is accepted.
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

    n = rest_pos.shape[0]
    x_cur = rest_pos.clone()
    x_hat = rest_pos.clone()

    mass = torch.ones(n, dtype=dtype, device=device)
    mu_lame = torch.full((n,), 10.0, dtype=dtype, device=device)
    lambda_lame = torch.full((n,), 10.0, dtype=dtype, device=device)
    k_bending = torch.full((n,), 0.1, dtype=dtype, device=device)
    dt = torch.tensor(0.05, dtype=dtype, device=device)
    pinned_idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    model = GNNIterationSolver(
        node_in_dim=12,
        edge_in_dim=12,
        latent_size=128,
        num_layers=2,
        message_passing_steps=15,
    ).to(device=device, dtype=dtype)

    delta_x = model(
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
