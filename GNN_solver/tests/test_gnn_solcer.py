import torch
from loss_class import ImplicitEulerLoss


def make_square_mesh(dtype=torch.float64, device="cpu"):
    rest_pos = torch.tensor([
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [1.0, 1.0, 0.0],  # 3
    ], dtype=dtype, device=device)

    face_index = torch.tensor([
        [0, 1, 2],
        [1, 3, 2],
    ], dtype=torch.long, device=device)

    edge_index = torch.tensor([
        [0, 1],
        [1, 2],
        [0, 2],
        [1, 3],
        [2, 3],
    ], dtype=torch.long, device=device)

    return rest_pos, edge_index, face_index


class DummyGNNSolver(torch.nn.Module):
    """
    Minimal stand-in for your future GNN solver.

    It learns a per-vertex displacement delta and outputs:

        x_pred = x_prev + delta

    Your real GNN can replace this class as long as it outputs [N, 3].
    """

    def __init__(self, num_vertices, dtype=torch.float64, device="cpu"):
        super().__init__()
        self.delta = torch.nn.Parameter(
            torch.zeros(num_vertices, 3, dtype=dtype, device=device)
        )

    def forward(self, x_prev, v_prev, edge_index):
        return x_prev + self.delta


def clamp_pinned_vertices(x, reference_x, pinned_idx):
    if pinned_idx is None:
        return x

    x = x.clone()
    x[pinned_idx] = reference_x[pinned_idx]
    return x


def test_loss_as_gnn_solver_objective():
    print("\n[TEST] Use ImplicitEulerLoss as GNN solver objective")

    dtype = torch.float64
    device = "cpu"

    rest_pos, edge_index, face_index = make_square_mesh(
        dtype=dtype,
        device=device,
    )

    pinned_idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    loss_obj = ImplicitEulerLoss(
        rest_pos=rest_pos,
        edge_index=edge_index,
        face_index=face_index,
        density=2.0,
        mu=10.0,
        lambda_=10.0,
        k_bending=0.1,
        gravity=(0.0, 0.0, -9.81),
        pinned_idx=pinned_idx,
    )

    solver = DummyGNNSolver(
        num_vertices=rest_pos.shape[0],
        dtype=dtype,
        device=device,
    )

    dt = torch.tensor(0.05, dtype=dtype, device=device)

    x_prev = rest_pos.clone()
    v_prev = torch.zeros_like(rest_pos)

    with torch.no_grad():
        x_init = solver(x_prev, v_prev, edge_index)
        x_init = clamp_pinned_vertices(x_init, x_prev, pinned_idx)

        losses_before = loss_obj.forward(
            x=x_init,
            x_prev=x_prev,
            v_prev=v_prev,
            dt=dt,
        )

        res_before = loss_obj.residual(
            x=x_init,
            x_prev=x_prev,
            v_prev=v_prev,
            dt=dt,
        )

    print("  before total loss:", losses_before["total"])
    print("  before residual mean:", res_before["mean"])
    print("  before residual max:", res_before["max"])

    optimizer = torch.optim.LBFGS(
        solver.parameters(),
        max_iter=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
    )

    def closure():
        optimizer.zero_grad()

        x_pred = solver(x_prev, v_prev, edge_index)
        x_pred = clamp_pinned_vertices(x_pred, x_prev, pinned_idx)

        losses = loss_obj.forward(
            x=x_pred,
            x_prev=x_prev,
            v_prev=v_prev,
            dt=dt,
        )

        losses["total"].backward()
        return losses["total"]

    optimizer.step(closure)

    with torch.no_grad():
        x_final = solver(x_prev, v_prev, edge_index)
        x_final = clamp_pinned_vertices(x_final, x_prev, pinned_idx)

        losses_after = loss_obj.forward(
            x=x_final,
            x_prev=x_prev,
            v_prev=v_prev,
            dt=dt,
        )

        res_after = loss_obj.residual(
            x=x_final,
            x_prev=x_prev,
            v_prev=v_prev,
            dt=dt,
        )

    print("  after total loss:", losses_after["total"])
    print("  after residual mean:", res_after["mean"])
    print("  after residual max:", res_after["max"])
    print("  optimized x:")
    print(x_final)

    assert losses_after["total"] < losses_before["total"]
    assert res_after["mean"] < res_before["mean"]

    torch.testing.assert_close(
        res_after["vector"][pinned_idx],
        torch.zeros_like(res_after["vector"][pinned_idx]),
        atol=1e-12,
        rtol=1e-12,
    )

    print("  GNN solver objective interface OK")


if __name__ == "__main__":
    test_loss_as_gnn_solver_objective()