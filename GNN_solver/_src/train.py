from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor

from loss_class import ImplicitEulerLoss
from GNN_solver import GNNIterationSolver


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def clamp_pinned_vertices(x: Tensor, reference_x: Tensor, pinned_idx: Tensor | None) -> Tensor:
    """
    Keep pinned vertices fixed.

    This is intentionally outside the GNN. The GNN predicts delta_x only.
    """
    if pinned_idx is None:
        return x
    x = x.clone()
    x[pinned_idx] = reference_x[pinned_idx]
    return x


def compute_x_hat(x_prev: Tensor, v_prev: Tensor, dt: Tensor) -> Tensor:
    """
    Inertial prediction used as a fixed input feature during one time step.

    If your ImplicitEulerLoss uses a different inertia convention, change it here.
    The GNN itself does not perform time stepping.
    """
    return x_prev + dt * v_prev


def run_iterations(
    *,
    solver: GNNIterationSolver,
    x_init: Tensor,
    x_hat: Tensor,
    rest_pos: Tensor,
    edge_index: Tensor,
    mass: Tensor,
    mu_lame: Tensor,
    lambda_lame: Tensor,
    k_bending: Tensor,
    dt: Tensor,
    pinned_idx: Tensor | None,
    reference_x: Tensor,
    num_iters: int,
) -> Tensor:
    """
    Autoregressively apply the learned delta solver.

    At each iteration:
        delta_x = solver(x_cur, x_hat, ...)
        x_cur = x_cur + delta_x
        x_cur[pinned] = reference_x[pinned]
    """
    x_cur = x_init.clone()

    for _ in range(num_iters):
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
        x_cur = x_cur + delta_x
        x_cur = clamp_pinned_vertices(x_cur, reference_x, pinned_idx)

    return x_cur


def init_csv_log(log_path: Path) -> None:
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase",
                "epoch",
                "init_name",
                "iter",
                "total_loss",
                "inertia",
                "gravity",
                "elastic",
                "bending",
                "residual_mean",
                "residual_max",
            ]
        )


def append_eval_row(
    log_path: Path,
    *,
    phase: str,
    epoch: int,
    init_name: str,
    iter_id: int,
    losses: Dict[str, Tensor],
    residual: Dict[str, Tensor],
) -> None:
    def scalar(v):
        if torch.is_tensor(v):
            return float(v.detach().cpu())
        return float(v)

    row = [
        phase,
        epoch,
        init_name,
        iter_id,
        scalar(losses["total"]),
        scalar(losses["inertia"]),
        scalar(losses["gravity"]),
        scalar(losses["elastic"]),
        scalar(losses["bending"]),
        scalar(residual["mean"]),
        scalar(residual["max"]),
    ]

    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    print(
        f"[{phase:>8s}] "
        f"epoch={epoch:05d} "
        f"init={init_name:>6s} "
        f"iter={iter_id:02d} "
        f"loss={row[4]:.8e} "
        f"res_mean={row[9]:.8e} "
        f"res_max={row[10]:.8e}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_15_iterations(
    *,
    solver: GNNIterationSolver,
    loss_obj: ImplicitEulerLoss,
    phase: str,
    epoch: int,
    initial_states: Dict[str, Tensor],
    x_hat: Tensor,
    x_prev: Tensor,
    v_prev: Tensor,
    rest_pos: Tensor,
    edge_index: Tensor,
    mass: Tensor,
    mu_lame: Tensor,
    lambda_lame: Tensor,
    k_bending: Tensor,
    dt: Tensor,
    pinned_idx: Tensor | None,
    log_path: Path,
    test_iters: int = 15,
) -> None:
    """
    For each initial value, iterate 15 times.
    After each iteration, compute total loss and residual, then print and log them.
    """
    solver.eval()

    for init_name, x_init in initial_states.items():
        x_cur = x_init.clone()

        for iter_id in range(1, test_iters + 1):
            with torch.no_grad():
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
                x_cur = x_cur + delta_x
                x_cur = clamp_pinned_vertices(x_cur, x_prev, pinned_idx)

                losses = loss_obj.forward(
                    x=x_cur,
                    x_prev=x_prev,
                    v_prev=v_prev,
                    dt=dt,
                )

            # Keep residual outside torch.no_grad(), matching the pattern in your demo.
            residual = loss_obj.residual(
                x=x_cur,
                x_prev=x_prev,
                v_prev=v_prev,
                dt=dt,
            )

            append_eval_row(
                log_path,
                phase=phase,
                epoch=epoch,
                init_name=init_name,
                iter_id=iter_id,
                losses=losses,
                residual=residual,
            )

    solver.train()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_phase(
    *,
    solver: GNNIterationSolver,
    loss_obj: ImplicitEulerLoss,
    optimizer: torch.optim.Optimizer,
    phase: str,
    num_epochs: int,
    train_iters: int,
    test_every: int,
    initial_states: Dict[str, Tensor],
    x_hat: Tensor,
    x_prev: Tensor,
    v_prev: Tensor,
    rest_pos: Tensor,
    edge_index: Tensor,
    mass: Tensor,
    mu_lame: Tensor,
    lambda_lame: Tensor,
    k_bending: Tensor,
    dt: Tensor,
    pinned_idx: Tensor | None,
    log_path: Path,
) -> None:
    """
    One training phase.

    Training objective:
        For each epoch, start from both initial states: x_prev and x_hat.

        For each initial state, run train_iters autoregressive iterations.
        IMPORTANT: every iteration computes its own ImplicitEulerLoss and performs
        its own backward + optimizer step.

    This means:
        pretrain: train_iters = 1
            each epoch has 2 optimizer steps, one for x_prev and one for x_hat.

        finetune: train_iters = 10
            each epoch has 20 optimizer steps, 10 from x_prev and 10 from x_hat.

    After each optimizer step, x_cur is detached before the next iteration.
    This avoids backpropagating through already-updated parameters from earlier
    autoregressive iterations.
    """
    for epoch in range(1, num_epochs + 1):
        solver.train()

        epoch_loss_sum = 0.0
        epoch_step_count = 0

        for _, x_init in initial_states.items():
            x_cur = x_init.clone()

            for _ in range(train_iters):
                optimizer.zero_grad()

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

                x_next = x_cur + delta_x
                x_next = clamp_pinned_vertices(x_next, x_prev, pinned_idx)

                losses = loss_obj.forward(
                    x=x_next,
                    x_prev=x_prev,
                    v_prev=v_prev,
                    dt=dt,
                )

                loss = losses["total"]
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.detach().cpu())
                epoch_step_count += 1

                # Continue autoregressive rollout from the latest predicted state,
                # but do not keep the old computation graph after the optimizer step.
                x_cur = x_next.detach()

        mean_epoch_loss = epoch_loss_sum / max(epoch_step_count, 1)

        if epoch % test_every == 0:
            print(
                f"\n=== {phase} epoch {epoch}/{num_epochs} | "
                f"train_iters={train_iters} | "
                f"optimizer_steps={epoch_step_count} | "
                f"mean_train_loss={mean_epoch_loss:.8e} ===",
                flush=True,
            )
            evaluate_15_iterations(
                solver=solver,
                loss_obj=loss_obj,
                phase=phase,
                epoch=epoch,
                initial_states=initial_states,
                x_hat=x_hat,
                x_prev=x_prev,
                v_prev=v_prev,
                rest_pos=rest_pos,
                edge_index=edge_index,
                mass=mass,
                mu_lame=mu_lame,
                lambda_lame=lambda_lame,
                k_bending=k_bending,
                dt=dt,
                pinned_idx=pinned_idx,
                log_path=log_path,
                test_iters=15,
            )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    dtype = torch.float32
    device = "cpu"

    # -------------------------------------------------------------------------
    # Toy one-step dataset
    # -------------------------------------------------------------------------
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

    face_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=torch.long,
        device=device,
    )

    # External format [E, 2]. The GNN internally converts it to [2, E] and adds reverse edges.
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

    pinned_idx = torch.tensor([0, 1], dtype=torch.long, device=device)

    density = 2.0
    mu_value = 10.0
    lambda_value = 10.0
    k_bending_value = 0.1

    loss_obj = ImplicitEulerLoss(
        rest_pos=rest_pos,
        edge_index=edge_index,
        face_index=face_index,
        density=density,
        mu=mu_value,
        lambda_=lambda_value,
        k_bending=k_bending_value,
        gravity=(0.0, 0.0, -9.81),
        pinned_idx=pinned_idx,
    )

    dt = torch.tensor(0.03, dtype=dtype, device=device)
    x_prev = rest_pos.clone()
    v_prev = torch.zeros_like(rest_pos)
    x_hat = compute_x_hat(x_prev, v_prev, dt)

    # The toy example has v_prev = 0, so x_hat == x_prev.
    # In real training data, these two initial states may differ.
    initial_states = {
        "x_prev": x_prev.clone(),
        "x_hat": x_hat.clone(),
    }

    # -------------------------------------------------------------------------
    # GNN input physical features
    # -------------------------------------------------------------------------
    num_vertices = rest_pos.shape[0]

    # Placeholder nodal mass. Replace this with your real lumped nodal mass if available.
    # Shape: [N]
    mass = torch.ones(num_vertices, dtype=dtype, device=device) * density

    # Local material parameters. Here they are constant per node.
    mu_lame = torch.full((num_vertices,), mu_value, dtype=dtype, device=device)
    lambda_lame = torch.full((num_vertices,), lambda_value, dtype=dtype, device=device)
    k_bending = torch.full((num_vertices,), k_bending_value, dtype=dtype, device=device)

    # -------------------------------------------------------------------------
    # Model and optimizer
    # -------------------------------------------------------------------------
    solver = GNNIterationSolver(
        node_in_dim=12,
        edge_in_dim=12,
        latent_size=128,
        num_layers=2,
        message_passing_steps=15,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(
        solver.parameters(),
        lr=1.0e-4,
        weight_decay=0.0,
    )

    log_path = Path("gnn_iterative_eval_log.csv")
    init_csv_log(log_path)
    print(f"Evaluation log will be written to: {log_path.resolve()}", flush=True)

    # Optional: evaluate the untrained model first.
    print("\n=== initial evaluation before training ===", flush=True)
    evaluate_15_iterations(
        solver=solver,
        loss_obj=loss_obj,
        phase="initial",
        epoch=0,
        initial_states=initial_states,
        x_hat=x_hat,
        x_prev=x_prev,
        v_prev=v_prev,
        rest_pos=rest_pos,
        edge_index=edge_index,
        mass=mass,
        mu_lame=mu_lame,
        lambda_lame=lambda_lame,
        k_bending=k_bending,
        dt=dt,
        pinned_idx=pinned_idx,
        log_path=log_path,
        test_iters=15,
    )

    # -------------------------------------------------------------------------
    # Phase 1: pretraining
    #   one GNN iteration per time step, 10000 epochs
    # -------------------------------------------------------------------------
    train_phase(
        solver=solver,
        loss_obj=loss_obj,
        optimizer=optimizer,
        phase="pretrain",
        num_epochs=10_000,
        train_iters=1,
        test_every=100,
        initial_states=initial_states,
        x_hat=x_hat,
        x_prev=x_prev,
        v_prev=v_prev,
        rest_pos=rest_pos,
        edge_index=edge_index,
        mass=mass,
        mu_lame=mu_lame,
        lambda_lame=lambda_lame,
        k_bending=k_bending,
        dt=dt,
        pinned_idx=pinned_idx,
        log_path=log_path,
    )

    # -------------------------------------------------------------------------
    # Phase 2: autoregressive fine-tuning
    #   ten GNN iterations per time step, 1000 epochs
    # -------------------------------------------------------------------------
    train_phase(
        solver=solver,
        loss_obj=loss_obj,
        optimizer=optimizer,
        phase="finetune",
        num_epochs=1_000,
        train_iters=10,
        test_every=100,
        initial_states=initial_states,
        x_hat=x_hat,
        x_prev=x_prev,
        v_prev=v_prev,
        rest_pos=rest_pos,
        edge_index=edge_index,
        mass=mass,
        mu_lame=mu_lame,
        lambda_lame=lambda_lame,
        k_bending=k_bending,
        dt=dt,
        pinned_idx=pinned_idx,
        log_path=log_path,
    )

    torch.save(
        {
            "model_state_dict": solver.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "dt": dt.detach().cpu(),
            "rest_pos": rest_pos.detach().cpu(),
            "edge_index": edge_index.detach().cpu(),
            "face_index": face_index.detach().cpu(),
            "pinned_idx": pinned_idx.detach().cpu(),
        },
        "gnn_iterative_solver_final.pt",
    )
    print("\nSaved checkpoint to: gnn_iterative_solver_final.pt", flush=True)


if __name__ == "__main__":
    main()
