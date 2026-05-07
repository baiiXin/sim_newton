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


def make_loss_obj(rest_pos, edge_index, face_index, pinned_idx=None):
    return ImplicitEulerLoss(
        rest_pos=rest_pos,
        edge_index=edge_index,
        face_index=face_index,
        density=2.0,
        mu=1.0,
        lambda_=1.0,
        k_bending=0.1,
        gravity=(0.0, 0.0, -9.81),
        pinned_idx=pinned_idx,
    )


def test_geometry_precompute():
    print("\n[TEST] Geometry precomputation")

    rest_pos, edge_index, face_index = make_square_mesh()
    loss_obj = make_loss_obj(rest_pos, edge_index, face_index)

    expected_face_area = torch.tensor([0.5, 0.5], dtype=rest_pos.dtype)
    torch.testing.assert_close(loss_obj.face_area, expected_face_area)
    print("  face_area OK:", loss_obj.face_area)

    expected_vertex_area = torch.tensor([
        1.0 / 6.0,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 6.0,
    ], dtype=rest_pos.dtype)

    torch.testing.assert_close(loss_obj.vertex_area, expected_vertex_area)
    print("  vertex_area OK:", loss_obj.vertex_area)

    expected_mass = 2.0 * expected_vertex_area
    torch.testing.assert_close(loss_obj.mass, expected_mass)
    print("  mass OK:", loss_obj.mass)

    expected_Dm_inv_0 = torch.eye(2, dtype=rest_pos.dtype)
    torch.testing.assert_close(loss_obj.Dm_inv[0], expected_Dm_inv_0)
    print("  Dm_inv[0] OK:")
    print(loss_obj.Dm_inv[0])

    assert torch.isfinite(loss_obj.Dm_inv).all()
    print("  Dm_inv finite OK")

    assert loss_obj.hinges.shape[0] == 1
    print("  hinges OK:", loss_obj.hinges)

    torch.testing.assert_close(
        loss_obj.theta0,
        torch.zeros_like(loss_obj.theta0),
        atol=1e-12,
        rtol=1e-12,
    )
    print("  theta0 OK:", loss_obj.theta0)


def test_pinned_mask():
    print("\n[TEST] Pinned / free mask")

    rest_pos, edge_index, face_index = make_square_mesh()

    pinned_idx = torch.tensor([0, 3], dtype=torch.long)

    loss_obj = make_loss_obj(
        rest_pos,
        edge_index,
        face_index,
        pinned_idx=pinned_idx,
    )

    expected_free_mask = torch.tensor(
        [False, True, True, False],
        dtype=torch.bool,
    )

    torch.testing.assert_close(loss_obj.free_mask.cpu(), expected_free_mask)
    print("  free_mask OK:", loss_obj.free_mask)


def test_degenerate_triangle_check():
    print("\n[TEST] Degenerate triangle check")

    bad_rest_pos = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=torch.float64)

    bad_faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    bad_edges = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.long)

    try:
        _ = ImplicitEulerLoss(
            rest_pos=bad_rest_pos,
            edge_index=bad_edges,
            face_index=bad_faces,
            density=1.0,
            mu=1.0,
            lambda_=1.0,
            k_bending=0.1,
        )
        raise RuntimeError("Degenerate triangle test failed: no error was raised.")
    except ValueError as e:
        print("  degenerate triangle check OK:", e)


def test_rest_elastic_and_bending_zero():
    print("\n[TEST] Rest pose elastic / bending loss")

    rest_pos, edge_index, face_index = make_square_mesh()
    loss_obj = make_loss_obj(rest_pos, edge_index, face_index)

    x = rest_pos.clone()

    loss_elastic = loss_obj.elastic_loss(x)
    loss_bending = loss_obj.bending_loss(x)

    torch.testing.assert_close(
        loss_elastic,
        torch.zeros((), dtype=rest_pos.dtype),
        atol=1e-12,
        rtol=1e-12,
    )

    torch.testing.assert_close(
        loss_bending,
        torch.zeros((), dtype=rest_pos.dtype),
        atol=1e-12,
        rtol=1e-12,
    )

    print("  elastic rest OK:", loss_elastic)
    print("  bending rest OK:", loss_bending)


def test_rigid_translation_invariance():
    print("\n[TEST] Rigid translation invariance")

    rest_pos, edge_index, face_index = make_square_mesh()
    loss_obj = make_loss_obj(rest_pos, edge_index, face_index)

    translation = torch.tensor([2.0, -1.0, 3.0], dtype=rest_pos.dtype)
    x = rest_pos + translation

    loss_elastic = loss_obj.elastic_loss(x)
    loss_bending = loss_obj.bending_loss(x)

    torch.testing.assert_close(
        loss_elastic,
        torch.zeros((), dtype=rest_pos.dtype),
        atol=1e-12,
        rtol=1e-12,
    )

    torch.testing.assert_close(
        loss_bending,
        torch.zeros((), dtype=rest_pos.dtype),
        atol=1e-12,
        rtol=1e-12,
    )

    print("  elastic translation OK:", loss_elastic)
    print("  bending translation OK:", loss_bending)


def test_positive_deformation_losses():
    print("\n[TEST] Positive deformation losses")

    rest_pos, edge_index, face_index = make_square_mesh()
    loss_obj = make_loss_obj(rest_pos, edge_index, face_index)

    x_stretch = rest_pos.clone()
    x_stretch[:, 0] *= 1.2

    loss_elastic = loss_obj.elastic_loss(x_stretch)
    assert loss_elastic > 0
    print("  elastic stretch positive OK:", loss_elastic)

    x_bend = rest_pos.clone()
    x_bend[3, 2] = 0.5

    loss_bending = loss_obj.bending_loss(x_bend)
    assert loss_bending > 0
    print("  bending deformation positive OK:", loss_bending)


def test_inertia_gravity_optimum_gradient():
    print("\n[TEST] Inertia + gravity analytic optimum")

    rest_pos, edge_index, face_index = make_square_mesh()
    loss_obj = make_loss_obj(rest_pos, edge_index, face_index)

    dt = torch.tensor(0.01, dtype=rest_pos.dtype)

    x_prev = rest_pos.clone()
    v_prev = torch.zeros_like(rest_pos)

    x_opt = x_prev + dt * v_prev + dt * dt * loss_obj.gravity

    x_var = x_opt.clone().detach().requires_grad_(True)

    loss = (
        loss_obj.inertia_loss(x_var, x_prev, v_prev, dt)
        + loss_obj.gravity_loss(x_var)
    )

    grad = torch.autograd.grad(loss, x_var)[0]
    free_grad = grad[loss_obj.free_mask]

    torch.testing.assert_close(
        free_grad,
        torch.zeros_like(free_grad),
        atol=1e-10,
        rtol=1e-10,
    )

    print("  inertia + gravity grad norm OK:", free_grad.norm())


def test_forward_output():
    print("\n[TEST] Forward output")

    rest_pos, edge_index, face_index = make_square_mesh()
    loss_obj = make_loss_obj(rest_pos, edge_index, face_index)

    dt = torch.tensor(0.01, dtype=rest_pos.dtype)

    x_prev = rest_pos.clone()
    v_prev = torch.zeros_like(rest_pos)
    x = rest_pos.clone()

    losses = loss_obj.forward(
        x=x,
        x_prev=x_prev,
        v_prev=v_prev,
        dt=dt,
    )

    expected_keys = {
        "total",
        "inertia",
        "gravity",
        "elastic",
        "bending",
    }

    assert set(losses.keys()) == expected_keys

    total_check = (
        losses["inertia"]
        + losses["gravity"]
        + losses["elastic"]
        + losses["bending"]
    )

    torch.testing.assert_close(losses["total"], total_check)

    for name, value in losses.items():
        assert torch.is_tensor(value)
        assert value.shape == torch.Size([])
        assert torch.isfinite(value)

    print("  forward keys OK:", list(losses.keys()))
    print("  total:", losses["total"])
    print("  inertia:", losses["inertia"])
    print("  gravity:", losses["gravity"])
    print("  elastic:", losses["elastic"])
    print("  bending:", losses["bending"])

def test_residual_inertia_gravity_optimum():
    print("\n[TEST] Residual at inertia + gravity analytic optimum")

    rest_pos, edge_index, face_index = make_square_mesh()

    # Disable elastic and bending for this analytic test.
    loss_obj = ImplicitEulerLoss(
        rest_pos=rest_pos,
        edge_index=edge_index,
        face_index=face_index,
        density=2.0,
        mu=0.0,
        lambda_=0.0,
        k_bending=0.0,
        gravity=(0.0, 0.0, -9.81),
    )

    dt = torch.tensor(0.01, dtype=rest_pos.dtype)

    x_prev = rest_pos.clone()
    v_prev = torch.zeros_like(rest_pos)

    x_opt = x_prev + dt * v_prev + dt * dt * loss_obj.gravity

    res = loss_obj.residual(
        x=x_opt,
        x_prev=x_prev,
        v_prev=v_prev,
        dt=dt,
    )

    torch.testing.assert_close(
        res["vector"][loss_obj.free_mask],
        torch.zeros_like(res["vector"][loss_obj.free_mask]),
        atol=1e-10,
        rtol=1e-10,
    )

    print("  residual mean OK:", res["mean"])
    print("  residual max OK:", res["max"])
    print("  residual l2 OK:", res["l2"])

def test_residual_pinned_zero():
    print("\n[TEST] Residual pinned vertices are zero")

    rest_pos, edge_index, face_index = make_square_mesh()

    pinned_idx = torch.tensor([0, 3], dtype=torch.long)

    loss_obj = ImplicitEulerLoss(
        rest_pos=rest_pos,
        edge_index=edge_index,
        face_index=face_index,
        density=2.0,
        mu=1.0,
        lambda_=1.0,
        k_bending=0.1,
        gravity=(0.0, 0.0, -9.81),
        pinned_idx=pinned_idx,
    )

    dt = torch.tensor(0.01, dtype=rest_pos.dtype)

    x_prev = rest_pos.clone()
    v_prev = torch.zeros_like(rest_pos)

    # Deliberately deform x so residual is generally nonzero.
    x = rest_pos.clone()
    x[1, 0] += 0.1
    x[2, 2] += 0.2

    res = loss_obj.residual(
        x=x,
        x_prev=x_prev,
        v_prev=v_prev,
        dt=dt,
    )

    torch.testing.assert_close(
        res["vector"][pinned_idx],
        torch.zeros_like(res["vector"][pinned_idx]),
        atol=1e-12,
        rtol=1e-12,
    )

    assert res["mean"] > 0

    print("  pinned residual zero OK")
    print("  residual mean:", res["mean"])
    print("  residual max:", res["max"])
    print("  residual l2:", res["l2"])

def test_minimize_total_loss_with_lbfgs():
    print("\n[TEST] Minimize total loss with LBFGS")

    rest_pos, edge_index, face_index = make_square_mesh()

    # Use pinned vertices to avoid the whole square simply falling freely.
    pinned_idx = torch.tensor([0, 1], dtype=torch.long)

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

    dt = torch.tensor(0.01, dtype=rest_pos.dtype)

    x_prev = rest_pos.clone()
    v_prev = torch.zeros_like(rest_pos)

    # Start from a deliberately perturbed state.
    x = rest_pos.clone()
    x[2, 2] -= 0.05
    x[3, 2] -= 0.08
    x = x.detach().clone().requires_grad_(True)

    res_before = loss_obj.residual(
        x=x.detach(),
        x_prev=x_prev,
        v_prev=v_prev,
        dt=dt,
    )

    loss_before = loss_obj.forward(
        x=x,
        x_prev=x_prev,
        v_prev=v_prev,
        dt=dt,
    )["total"].detach()

    print("  before loss:", loss_before)
    print("  before residual mean:", res_before["mean"])
    print("  before residual max:", res_before["max"])

    optimizer = torch.optim.LBFGS(
        [x],
        max_iter=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
    )

    def closure():
        optimizer.zero_grad()

        # Enforce pinned vertices externally.
        with torch.no_grad():
            x[pinned_idx] = rest_pos[pinned_idx]

        losses = loss_obj.forward(
            x=x,
            x_prev=x_prev,
            v_prev=v_prev,
            dt=dt,
        )

        losses["total"].backward()
        return losses["total"]

    optimizer.step(closure)

    with torch.no_grad():
        x[pinned_idx] = rest_pos[pinned_idx]

    res_after = loss_obj.residual(
        x=x.detach(),
        x_prev=x_prev,
        v_prev=v_prev,
        dt=dt,
    )

    loss_after = loss_obj.forward(
        x=x.detach(),
        x_prev=x_prev,
        v_prev=v_prev,
        dt=dt,
    )["total"]

    print("  after loss:", loss_after)
    print("  after residual mean:", res_after["mean"])
    print("  after residual max:", res_after["max"])
    print("  optimized x:")
    print(x.detach())

    assert loss_after < loss_before
    assert res_after["mean"] < res_before["mean"]

    torch.testing.assert_close(
        res_after["vector"][pinned_idx],
        torch.zeros_like(res_after["vector"][pinned_idx]),
        atol=1e-12,
        rtol=1e-12,
    )

    print("  LBFGS optimization OK")

def run_all_tests():
    test_geometry_precompute()
    test_pinned_mask()
    test_degenerate_triangle_check()
    test_rest_elastic_and_bending_zero()
    test_rigid_translation_invariance()
    test_positive_deformation_losses()
    test_inertia_gravity_optimum_gradient()
    test_forward_output()
    test_residual_inertia_gravity_optimum()
    test_residual_pinned_zero()
    test_minimize_total_loss_with_lbfgs()

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all_tests()