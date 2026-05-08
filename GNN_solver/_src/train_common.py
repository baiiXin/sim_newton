from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch
from torch import Tensor

from loss_class import ImplicitEulerLoss
from GNN_solver import GNNIterationSolver


BackwardMode = Literal["iteration", "time_step"]


@dataclass
class PhaseConfig:
    phase: str
    num_epochs: int
    train_iters: int
    backward_mode: BackwardMode


@dataclass
class ProblemBundle:
    solver: GNNIterationSolver
    loss_obj: ImplicitEulerLoss
    optimizer: torch.optim.Optimizer
    initial_states: Dict[str, Tensor]
    x_hat: Tensor
    x_prev: Tensor
    v_prev: Tensor
    rest_pos: Tensor
    edge_index: Tensor
    face_index: Tensor
    mass: Tensor
    mu_lame: Tensor
    lambda_lame: Tensor
    k_bending: Tensor
    dt: Tensor
    pinned_idx: Optional[Tensor]
    device: torch.device
    dtype: torch.dtype


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def clamp_pinned_vertices(x: Tensor, reference_x: Tensor, pinned_idx: Optional[Tensor]) -> Tensor:
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


def build_problem(
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype = torch.float32,
    lr: float = 1.0e-2,
    weight_decay: float = 0.0,
    seed: int = 0,
) -> ProblemBundle:
    """
    Build the one-time-step toy problem and the GNN solver.

    Replace this function later when you switch from the toy mesh to your real dataset.
    """
    torch.manual_seed(seed)

    device = resolve_device(device) if isinstance(device, str) else torch.device(device)

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

    # initial velocity
    initial_velocity = torch.tensor(
        [
            [-0.0, -0.0, 0.0],
            [ 0.0, -0.0, 0.0],
            [-0.0,  0.0, 0.0],
            [ 0.0,  0.0, 0.0],
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

    # External format [E, 2].
    # The GNN internally converts it to [2, E] and adds reverse edges.
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

    pinned_idx = None # torch.tensor([], dtype=torch.long, device=device)

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
    v_prev = initial_velocity.clone()
    x_hat = compute_x_hat(x_prev, v_prev, dt)

    # In this toy case v_prev = 0, so x_hat == x_prev.
    # In real data these two initial states can differ.
    initial_states = {
        "x_prev": x_prev.clone(),
        "x_hat": x_hat.clone(),
    }

    num_vertices = rest_pos.shape[0]

    # Placeholder nodal mass. Replace with real lumped nodal mass if available.
    mass = torch.ones(num_vertices, dtype=dtype, device=device) * density

    # Local material parameters. Here they are constant per node.
    mu_lame = torch.full((num_vertices,), mu_value, dtype=dtype, device=device)
    lambda_lame = torch.full((num_vertices,), lambda_value, dtype=dtype, device=device)
    k_bending = torch.full((num_vertices,), k_bending_value, dtype=dtype, device=device)

    solver = GNNIterationSolver(
        node_in_dim=12,
        edge_in_dim=12,
        latent_size=128,
        num_layers=2,
        message_passing_steps=15,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(
        solver.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    return ProblemBundle(
        solver=solver,
        loss_obj=loss_obj,
        optimizer=optimizer,
        initial_states=initial_states,
        x_hat=x_hat,
        x_prev=x_prev,
        v_prev=v_prev,
        rest_pos=rest_pos,
        edge_index=edge_index,
        face_index=face_index,
        mass=mass,
        mu_lame=mu_lame,
        lambda_lame=lambda_lame,
        k_bending=k_bending,
        dt=dt,
        pinned_idx=pinned_idx,
        device=device,
        dtype=dtype,
    )


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


def init_train_loss_log(log_path: Path) -> None:
    with log_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "phase",
                "epoch",
                "backward_mode",
                "train_iters",
                "optimizer_steps",
                "mean_train_loss",
            ]
        )


def append_train_loss_row(
    log_path: Path,
    *,
    phase: str,
    epoch: int,
    backward_mode: BackwardMode,
    train_iters: int,
    optimizer_steps: int,
    mean_train_loss: float,
) -> None:
    with log_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                phase,
                epoch,
                backward_mode,
                train_iters,
                optimizer_steps,
                mean_train_loss,
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
        f"[{phase:>18s}] "
        f"epoch={epoch:05d} "
        f"init={init_name:>6s} "
        f"iter={iter_id:02d} "
        f"loss={row[4]:.8e} "
        f"res_mean={row[9]:.8e} "
        f"res_max={row[10]:.8e}",
        flush=True,
    )


def _compute_losses_and_residual(
    *,
    bundle: ProblemBundle,
    x: Tensor,
) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Compute loss and residual for a given x.

    Loss does not need gradients during evaluation.
    Residual is intentionally computed outside torch.no_grad(), matching the
    original pattern, in case residual internally uses autograd.
    """
    with torch.no_grad():
        losses = bundle.loss_obj.forward(
            x=x,
            x_prev=bundle.x_prev,
            v_prev=bundle.v_prev,
            dt=bundle.dt,
        )

    residual = bundle.loss_obj.residual(
        x=x,
        x_prev=bundle.x_prev,
        v_prev=bundle.v_prev,
        dt=bundle.dt,
    )
    return losses, residual


def evaluate_15_iterations(
    *,
    bundle: ProblemBundle,
    phase: str,
    epoch: int,
    log_path: Path,
    test_iters: int = 15,
    include_iter0: bool = True,
) -> None:
    """
    Evaluation protocol:
        For each initial value:
          - optionally log iteration 0, i.e. direct loss/residual of the initial value,
          - then iterate test_iters times,
          - after each iteration, compute total loss and residual, then print and log.

    With include_iter0=True and test_iters=15, CSV contains iter=0,1,...,15.
    """
    solver = bundle.solver
    solver.eval()

    for init_name, x_init in bundle.initial_states.items():
        x_cur = x_init.clone()
        x_cur = clamp_pinned_vertices(x_cur, bundle.x_prev, bundle.pinned_idx)

        if include_iter0:
            losses, residual = _compute_losses_and_residual(bundle=bundle, x=x_cur)
            append_eval_row(
                log_path,
                phase=phase,
                epoch=epoch,
                init_name=init_name,
                iter_id=0,
                losses=losses,
                residual=residual,
            )

        for iter_id in range(1, test_iters + 1):
            with torch.no_grad():
                delta_x = solver(
                    x_cur=x_cur,
                    x_hat=bundle.x_hat,
                    rest_pos=bundle.rest_pos,
                    edge_index=bundle.edge_index,
                    mass=bundle.mass,
                    mu_lame=bundle.mu_lame,
                    lambda_lame=bundle.lambda_lame,
                    k_bending=bundle.k_bending,
                    dt=bundle.dt,
                    pinned_idx=bundle.pinned_idx,
                )
                x_cur = x_cur + delta_x
                x_cur = clamp_pinned_vertices(x_cur, bundle.x_prev, bundle.pinned_idx)

            losses, residual = _compute_losses_and_residual(bundle=bundle, x=x_cur)

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


def train_phase_iteration_backward(
    *,
    bundle: ProblemBundle,
    phase: str,
    num_epochs: int,
    train_iters: int,
    test_every: int,
    log_path: Path,
    train_loss_log_path: Optional[Path] = None,
) -> None:
    """
    Training mode A: every solver iteration has its own loss/backward/optimizer.step.

    With two initial states:
        optimizer steps per epoch = 2 * train_iters
    """
    solver = bundle.solver
    optimizer = bundle.optimizer
    loss_obj = bundle.loss_obj

    for epoch in range(1, num_epochs + 1):
        solver.train()

        epoch_loss_sum = 0.0
        epoch_step_count = 0

        for _, x_init in bundle.initial_states.items():
            x_cur = x_init.clone()

            for _ in range(train_iters):
                optimizer.zero_grad()

                delta_x = solver(
                    x_cur=x_cur,
                    x_hat=bundle.x_hat,
                    rest_pos=bundle.rest_pos,
                    edge_index=bundle.edge_index,
                    mass=bundle.mass,
                    mu_lame=bundle.mu_lame,
                    lambda_lame=bundle.lambda_lame,
                    k_bending=bundle.k_bending,
                    dt=bundle.dt,
                    pinned_idx=bundle.pinned_idx,
                )

                x_next = x_cur + delta_x
                x_next = clamp_pinned_vertices(x_next, bundle.x_prev, bundle.pinned_idx)

                losses = loss_obj.forward(
                    x=x_next,
                    x_prev=bundle.x_prev,
                    v_prev=bundle.v_prev,
                    dt=bundle.dt,
                )

                loss = losses["total"]
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.detach().cpu())
                epoch_step_count += 1

                # Continue rollout from latest state, but do not keep the old graph.
                x_cur = x_next.detach()

        mean_loss = epoch_loss_sum / max(epoch_step_count, 1)

        if train_loss_log_path is not None:
            append_train_loss_row(
                train_loss_log_path,
                phase=phase,
                epoch=epoch,
                backward_mode="iteration",
                train_iters=train_iters,
                optimizer_steps=epoch_step_count,
                mean_train_loss=mean_loss,
            )

        if epoch % test_every == 0:
            print(
                f"\n=== {phase} epoch {epoch}/{num_epochs} | "
                f"mode=iteration_backward | "
                f"train_iters={train_iters} | "
                f"optimizer_steps={epoch_step_count} | "
                f"mean_train_loss={mean_loss:.8e} ===",
                flush=True,
            )
            evaluate_15_iterations(
                bundle=bundle,
                phase=phase,
                epoch=epoch,
                log_path=log_path,
                test_iters=15,
                include_iter0=True,
            )


def train_phase_time_step_backward(
    *,
    bundle: ProblemBundle,
    phase: str,
    num_epochs: int,
    train_iters: int,
    test_every: int,
    log_path: Path,
    train_loss_log_path: Optional[Path] = None,
) -> None:
    """
    Training mode B: one backward/optimizer.step per time step.

    For each initial state:
        1. unroll train_iters autoregressive GNN iterations,
        2. compute ImplicitEulerLoss on the final x,
        3. backward once and step once.

    With two initial states:
        optimizer steps per epoch = 2
    """
    solver = bundle.solver
    optimizer = bundle.optimizer
    loss_obj = bundle.loss_obj

    for epoch in range(1, num_epochs + 1):
        solver.train()

        epoch_loss_sum = 0.0
        epoch_step_count = 0

        for _, x_init in bundle.initial_states.items():
            optimizer.zero_grad()

            x_cur = x_init.clone()

            for _ in range(train_iters):
                delta_x = solver(
                    x_cur=x_cur,
                    x_hat=bundle.x_hat,
                    rest_pos=bundle.rest_pos,
                    edge_index=bundle.edge_index,
                    mass=bundle.mass,
                    mu_lame=bundle.mu_lame,
                    lambda_lame=bundle.lambda_lame,
                    k_bending=bundle.k_bending,
                    dt=bundle.dt,
                    pinned_idx=bundle.pinned_idx,
                )
                x_cur = x_cur + delta_x
                x_cur = clamp_pinned_vertices(x_cur, bundle.x_prev, bundle.pinned_idx)

            losses = loss_obj.forward(
                x=x_cur,
                x_prev=bundle.x_prev,
                v_prev=bundle.v_prev,
                dt=bundle.dt,
            )

            loss = losses["total"]
            loss.backward()
            optimizer.step()

            epoch_loss_sum += float(loss.detach().cpu())
            epoch_step_count += 1

        mean_loss = epoch_loss_sum / max(epoch_step_count, 1)

        if train_loss_log_path is not None:
            append_train_loss_row(
                train_loss_log_path,
                phase=phase,
                epoch=epoch,
                backward_mode="time_step",
                train_iters=train_iters,
                optimizer_steps=epoch_step_count,
                mean_train_loss=mean_loss,
            )

        if epoch % test_every == 0:
            print(
                f"\n=== {phase} epoch {epoch}/{num_epochs} | "
                f"mode=time_step_backward | "
                f"train_iters={train_iters} | "
                f"optimizer_steps={epoch_step_count} | "
                f"mean_train_loss={mean_loss:.8e} ===",
                flush=True,
            )
            evaluate_15_iterations(
                bundle=bundle,
                phase=phase,
                epoch=epoch,
                log_path=log_path,
                test_iters=15,
                include_iter0=True,
            )


def run_phase(
    *,
    bundle: ProblemBundle,
    phase_config: PhaseConfig,
    test_every: int,
    log_path: Path,
    train_loss_log_path: Optional[Path] = None,
) -> None:
    if phase_config.backward_mode == "iteration":
        train_phase_iteration_backward(
            bundle=bundle,
            phase=phase_config.phase,
            num_epochs=phase_config.num_epochs,
            train_iters=phase_config.train_iters,
            test_every=test_every,
            log_path=log_path,
            train_loss_log_path=train_loss_log_path,
        )
    elif phase_config.backward_mode == "time_step":
        train_phase_time_step_backward(
            bundle=bundle,
            phase=phase_config.phase,
            num_epochs=phase_config.num_epochs,
            train_iters=phase_config.train_iters,
            test_every=test_every,
            log_path=log_path,
            train_loss_log_path=train_loss_log_path,
        )
    else:
        raise ValueError(f"Unknown backward_mode: {phase_config.backward_mode}")


def save_checkpoint(bundle: ProblemBundle, checkpoint_path: Path, *, experiment_name: str) -> None:
    torch.save(
        {
            "experiment_name": experiment_name,
            "model_state_dict": bundle.solver.state_dict(),
            "optimizer_state_dict": bundle.optimizer.state_dict(),
            "dt": bundle.dt.detach().cpu(),
            "rest_pos": bundle.rest_pos.detach().cpu(),
            "edge_index": bundle.edge_index.detach().cpu(),
            "face_index": bundle.face_index.detach().cpu(),
            "pinned_idx": None if bundle.pinned_idx is None else bundle.pinned_idx.detach().cpu(),
        },
        checkpoint_path,
    )


def run_experiment(
    *,
    experiment_name: str,
    phases: List[PhaseConfig],
    device: str = "auto",
    lr: float = 1.0e-4,
    weight_decay: float = 0.0,
    seed: int = 0,
    test_every: int = 100,
    initial_eval: bool = True,
    output_dir: Path | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> None:
    """
    Run a full experiment.

    Evaluation logs are written to:
        {output_dir}/{experiment_name}_eval_log.csv

    Per-epoch training losses are written to:
        {output_dir}/{experiment_name}_train_loss_log.csv

    Checkpoints are written to:
        {output_dir}/{experiment_name}_final.pt

    When output_dir is None, files are written to the current working directory
    to preserve existing training script output naming.
    """
    bundle = build_problem(
        device=device,
        dtype=dtype,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
    )

    output_root = Path(".") if output_dir is None else Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = output_root / f"{experiment_name}_eval_log.csv"
    train_loss_log_path = output_root / f"{experiment_name}_train_loss_log.csv"
    checkpoint_path = output_root / f"{experiment_name}_final.pt"

    init_csv_log(log_path)
    init_train_loss_log(train_loss_log_path)

    print(f"Device: {bundle.device}", flush=True)
    print(f"Evaluation log: {log_path.resolve()}", flush=True)
    print(f"Train loss log: {train_loss_log_path.resolve()}", flush=True)
    print(f"Checkpoint: {checkpoint_path.resolve()}", flush=True)

    if initial_eval:
        print("\n=== initial evaluation before training ===", flush=True)
        evaluate_15_iterations(
            bundle=bundle,
            phase="initial",
            epoch=0,
            log_path=log_path,
            test_iters=15,
            include_iter0=True,
        )

    for phase_config in phases:
        run_phase(
            bundle=bundle,
            phase_config=phase_config,
            test_every=test_every,
            log_path=log_path,
            train_loss_log_path=train_loss_log_path,
        )

    save_checkpoint(bundle, checkpoint_path, experiment_name=experiment_name)
    print(f"\nSaved checkpoint to: {checkpoint_path}", flush=True)


def build_curriculum_iteration_phases(
    *,
    total_epochs: int,
    curriculum_every: int = 1_000,
    start_train_iters: int = 1,
    max_train_iters: Optional[int] = None,
) -> List[PhaseConfig]:
    """
    Build iteration-backward curriculum phases.

    The first phase trains with train_iters=start_train_iters. After each
    curriculum_every epochs, train_iters increases by one. Each solver iteration
    still uses its own backward/optimizer.step, so rollout state is detached
    before the next iteration by train_phase_iteration_backward.
    """
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if curriculum_every <= 0:
        raise ValueError("curriculum_every must be positive")
    if start_train_iters <= 0:
        raise ValueError("start_train_iters must be positive")
    if max_train_iters is not None and max_train_iters < start_train_iters:
        raise ValueError("max_train_iters must be >= start_train_iters")

    phases: List[PhaseConfig] = []
    remaining_epochs = total_epochs
    phase_index = 0

    while remaining_epochs > 0:
        phase_epochs = min(curriculum_every, remaining_epochs)
        train_iters = start_train_iters + phase_index
        if max_train_iters is not None:
            train_iters = min(train_iters, max_train_iters)

        phases.append(
            PhaseConfig(
                phase=f"curriculum_iter_{train_iters:02d}",
                num_epochs=phase_epochs,
                train_iters=train_iters,
                backward_mode="iteration",
            )
        )

        remaining_epochs -= phase_epochs
        phase_index += 1

    return phases


def run_curriculum_iteration_experiment(
    *,
    experiment_name: str = "exp_curriculum_iter_every1000",
    total_epochs: int = 10_000,
    curriculum_every: int = 1_000,
    start_train_iters: int = 1,
    max_train_iters: Optional[int] = None,
    device: str = "auto",
    lr: float = 1.0e-4,
    weight_decay: float = 0.0,
    seed: int = 0,
    test_every: int = 100,
    initial_eval: bool = True,
    output_dir: Path | str | None = None,
) -> None:
    """
    Run curriculum multi-step training and write results under _src/result.

    Default schedule for 10_000 epochs:
        epochs 0001-1000: train_iters=1
        epochs 1001-2000: train_iters=2
        ...
        epochs 9001-10000: train_iters=10
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "result"

    phases = build_curriculum_iteration_phases(
        total_epochs=total_epochs,
        curriculum_every=curriculum_every,
        start_train_iters=start_train_iters,
        max_train_iters=max_train_iters,
    )

    print("Curriculum iteration schedule:", flush=True)
    for phase in phases:
        print(
            f"  {phase.phase}: epochs={phase.num_epochs}, "
            f"train_iters={phase.train_iters}, mode={phase.backward_mode}",
            flush=True,
        )

    run_experiment(
        experiment_name=experiment_name,
        phases=phases,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        test_every=test_every,
        initial_eval=initial_eval,
        output_dir=output_dir,
    )


def parse_curriculum_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run curriculum iteration-backward training. The number of "
            "training solver iterations starts at 1 and increases every "
            "--curriculum-every epochs."
        )
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'",
    )
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-every", type=int, default=100)
    parser.add_argument("--no-initial-eval", action="store_true")
    parser.add_argument("--experiment-name", default="exp_curriculum_iter_every1000")
    parser.add_argument("--total-epochs", type=int, default=10_000)
    parser.add_argument("--curriculum-every", type=int, default=1_000)
    parser.add_argument("--start-train-iters", type=int, default=1)
    parser.add_argument("--max-train-iters", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "result",
        help="Directory for eval logs, train loss logs, and checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_curriculum_args()
    run_curriculum_iteration_experiment(
        experiment_name=args.experiment_name,
        total_epochs=args.total_epochs,
        curriculum_every=args.curriculum_every,
        start_train_iters=args.start_train_iters,
        max_train_iters=args.max_train_iters,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        test_every=args.test_every,
        initial_eval=not args.no_initial_eval,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
