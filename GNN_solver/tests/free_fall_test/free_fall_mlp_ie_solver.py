import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class Motion:
    motion_id: int
    x0: torch.Tensor      # [3]
    v0: torch.Tensor      # [3]
    g: torch.Tensor       # [3]
    dt: float
    mass: float

    @property
    def y(self) -> torch.Tensor:
        # Inertial target in implicit Euler: y = x_n + dt * v_n
        return self.x0 + self.dt * self.v0

    @property
    def exact_x(self) -> torch.Tensor:
        # Minimizer of E(x) = m/(2dt^2)||x-y||^2 - m g^T x
        return self.y + (self.dt ** 2) * self.g


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_motions(num_motions: int, dt: float, mass: float, device: torch.device) -> List[Motion]:
    """Construct similar free-fall one-step motions with slightly different initial velocities."""
    g = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32, device=device)
    x0 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    motions: List[Motion] = []
    for i in range(num_motions):
        # Velocities are intentionally close to each other.
        c = i - (num_motions - 1) / 2.0
        vx = 0.15 + 0.03 * c
        vy = -0.05 + 0.01 * math.sin(i)
        vz = 2.00 + 0.04 * c
        v0 = torch.tensor([vx, vy, vz], dtype=torch.float32, device=device)
        motions.append(Motion(i, x0.clone(), v0, g, dt, mass))
    return motions


def energy(motion: Motion, x: torch.Tensor) -> torch.Tensor:
    """Implicit Euler variational energy for one point under gravity."""
    y = motion.y
    m = motion.mass
    dt = motion.dt
    inertia = 0.5 * m / (dt * dt) * torch.sum((x - y) ** 2)
    potential = -m * torch.dot(motion.g, x)
    return inertia + potential


def residual(motion: Motion, x: torch.Tensor) -> torch.Tensor:
    """Gradient of the implicit Euler energy. The exact solution has residual 0."""
    m = motion.mass
    dt = motion.dt
    return m / (dt * dt) * (x - motion.y) - m * motion.g


def newton_exact_step(motion: Motion, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """One Newton step for this quadratic energy. It reaches the exact minimizer."""
    grad = residual(motion, x)
    h_diag = motion.mass / (motion.dt * motion.dt)
    dx = -grad / h_diag
    return x + dx, dx


def exact_energy_gap(motion: Motion, x: torch.Tensor) -> torch.Tensor:
    """E(x) - E(x*) >= 0; same minimizer as raw energy, easier to read as a loss."""
    return energy(motion, x) - energy(motion, motion.exact_x)


class SolverMLP(nn.Module):
    def __init__(self, in_dim: int = 14, hidden_dim: int = 128, out_scale: float = 0.20):
        super().__init__()
        self.out_scale = out_scale
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # Bound dx to avoid very large unstable jumps early in training.
        return self.out_scale * torch.tanh(self.net(feat))


def build_feature(motion: Motion, x_cur: torch.Tensor, iter_id: int, max_test_iters: int) -> torch.Tensor:
    """Feature vector for the MLP solver."""
    r = residual(motion, x_cur)
    feat = torch.cat([
        x_cur,                    # 3
        motion.y,                 # 3
        motion.v0 / 5.0,          # 3, simple normalization
        r / 50.0,                 # 3, residual is a useful solver feature
        torch.tensor([motion.dt], dtype=x_cur.dtype, device=x_cur.device),
        torch.tensor([iter_id / max(1, max_test_iters - 1)], dtype=x_cur.dtype, device=x_cur.device),
    ])
    return feat.unsqueeze(0)


def vec_to_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().reshape(-1)]


def vec_str(x: torch.Tensor) -> str:
    vals = vec_to_list(x)
    return "[" + ", ".join(f"{v:+.6f}" for v in vals) + "]"


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_initial_info(motions: List[Motion]) -> List[Dict]:
    rows: List[Dict] = []
    print("\n[Initial residuals from x_init = x0]")
    for motion in motions:
        x_init = motion.x0.clone()
        r0 = residual(motion, x_init)
        x_star, newton_dx = newton_exact_step(motion, x_init)
        gap0 = exact_energy_gap(motion, x_init)
        row = {
            "motion_id": motion.motion_id,
            "v0_x": float(motion.v0[0].cpu()),
            "v0_y": float(motion.v0[1].cpu()),
            "v0_z": float(motion.v0[2].cpu()),
            "x_init_x": float(x_init[0].cpu()),
            "x_init_y": float(x_init[1].cpu()),
            "x_init_z": float(x_init[2].cpu()),
            "exact_x_x": float(motion.exact_x[0].cpu()),
            "exact_x_y": float(motion.exact_x[1].cpu()),
            "exact_x_z": float(motion.exact_x[2].cpu()),
            "init_residual_norm": float(torch.linalg.norm(r0).cpu()),
            "init_energy_gap": float(gap0.cpu()),
            "newton_exact_dx_x": float(newton_dx[0].cpu()),
            "newton_exact_dx_y": float(newton_dx[1].cpu()),
            "newton_exact_dx_z": float(newton_dx[2].cpu()),
        }
        rows.append(row)
        print(
            f"  motion {motion.motion_id:02d} | v0={vec_str(motion.v0)} | "
            f"init_res={row['init_residual_norm']:.6e} | exact_dx={vec_str(newton_dx)}"
        )
    return rows


@torch.no_grad()
def test_solver(
    model: SolverMLP,
    motion: Motion,
    epoch: int,
    test_iters: int,
) -> Tuple[List[Dict], Dict]:
    model.eval()
    rows: List[Dict] = []
    x_cur = motion.x0.clone()

    for it in range(test_iters):
        r_before = residual(motion, x_cur)
        feat = build_feature(motion, x_cur, it, test_iters)
        dx = model(feat).squeeze(0)
        x_next = x_cur + dx
        r_after = residual(motion, x_next)
        _, newton_dx = newton_exact_step(motion, x_cur)
        gap = exact_energy_gap(motion, x_next)
        err = torch.linalg.norm(x_next - motion.exact_x)

        row = {
            "epoch": epoch,
            "motion_id": motion.motion_id,
            "iter": it + 1,
            "residual_before_norm": float(torch.linalg.norm(r_before).cpu()),
            "residual_after_norm": float(torch.linalg.norm(r_after).cpu()),
            "energy_gap": float(gap.cpu()),
            "error_to_exact_norm": float(err.cpu()),
            "mlp_dx_x": float(dx[0].cpu()),
            "mlp_dx_y": float(dx[1].cpu()),
            "mlp_dx_z": float(dx[2].cpu()),
            "newton_dx_x": float(newton_dx[0].cpu()),
            "newton_dx_y": float(newton_dx[1].cpu()),
            "newton_dx_z": float(newton_dx[2].cpu()),
            "x_next_x": float(x_next[0].cpu()),
            "x_next_y": float(x_next[1].cpu()),
            "x_next_z": float(x_next[2].cpu()),
        }
        rows.append(row)
        x_cur = x_next.clone()

    summary = rows[-1]
    return rows, summary


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_seed(args.seed)

    motions = make_motions(args.num_motions, args.dt, args.mass, device)
    test_motion = motions[args.test_motion_id]
    train_motions = motions if not args.holdout_test else [m for m in motions if m.motion_id != args.test_motion_id]

    model = SolverMLP(hidden_dim=args.hidden_dim, out_scale=args.dx_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    init_rows = collect_initial_info(motions)
    train_rows: List[Dict] = []
    test_rows: List[Dict] = []

    print("\n[Training]")
    print(
        f"  train motions: {[m.motion_id for m in train_motions]} | "
        f"test motion: {test_motion.motion_id} | "
        f"holdout_test={args.holdout_test}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_iters = 1 + (epoch - 1) // args.iter_increase_every
        epoch_losses: List[float] = []
        epoch_residuals: List[float] = []

        for motion in train_motions:
            # Each motion only solves one frame, starting from x_n.
            x_cur = motion.x0.clone().detach()
            init_r_norm = torch.linalg.norm(residual(motion, x_cur)).item()
            init_gap = exact_energy_gap(motion, x_cur).item()

            for it in range(train_iters):
                r_before = residual(motion, x_cur.detach())
                _, newton_dx = newton_exact_step(motion, x_cur.detach())

                feat = build_feature(motion, x_cur.detach(), it, args.test_iters)
                dx = model(feat).squeeze(0)
                x_pred = x_cur.detach() + dx

                # Variational energy objective. The shifted form has the same minimizer
                # as E(x), but is non-negative and easier to log.
                loss = exact_energy_gap(motion, x_pred)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                with torch.no_grad():
                    r_after = residual(motion, x_pred)
                    err = torch.linalg.norm(x_pred - motion.exact_x)
                    train_rows.append({
                        "epoch": epoch,
                        "motion_id": motion.motion_id,
                        "train_iter_budget": train_iters,
                        "iter": it + 1,
                        "loss_energy_gap": float(loss.detach().cpu()),
                        "init_residual_norm": float(init_r_norm),
                        "init_energy_gap": float(init_gap),
                        "residual_before_norm": float(torch.linalg.norm(r_before).cpu()),
                        "residual_after_norm": float(torch.linalg.norm(r_after).cpu()),
                        "error_to_exact_norm": float(err.cpu()),
                        "mlp_dx_x": float(dx.detach()[0].cpu()),
                        "mlp_dx_y": float(dx.detach()[1].cpu()),
                        "mlp_dx_z": float(dx.detach()[2].cpu()),
                        "newton_dx_x": float(newton_dx[0].cpu()),
                        "newton_dx_y": float(newton_dx[1].cpu()),
                        "newton_dx_z": float(newton_dx[2].cpu()),
                        "x_pred_x": float(x_pred.detach()[0].cpu()),
                        "x_pred_y": float(x_pred.detach()[1].cpu()),
                        "x_pred_z": float(x_pred.detach()[2].cpu()),
                    })
                    epoch_losses.append(float(loss.detach().cpu()))
                    epoch_residuals.append(float(torch.linalg.norm(r_after).cpu()))

                    # Detach between iterations because this experiment requires
                    # one backward pass per solver iteration, not one large BPTT graph.
                    x_cur = x_pred.detach()

        if epoch % args.test_every == 0 or epoch == 1 or epoch == args.epochs:
            rows, summary = test_solver(model, test_motion, epoch, args.test_iters)
            test_rows.extend(rows)
            mean_loss = sum(epoch_losses) / max(1, len(epoch_losses))
            mean_res = sum(epoch_residuals) / max(1, len(epoch_residuals))
            print(
                f"  epoch {epoch:04d} | train_iters={train_iters:02d} | "
                f"train_loss={mean_loss:.6e} | train_res={mean_res:.6e} | "
                f"test_final_res={summary['residual_after_norm']:.6e} | "
                f"test_final_err={summary['error_to_exact_norm']:.6e}"
            )
            print(
                f"    test last dx={summary['mlp_dx_x']:+.6f}, "
                f"{summary['mlp_dx_y']:+.6f}, {summary['mlp_dx_z']:+.6f} | "
                f"Newton dx at last state={summary['newton_dx_x']:+.6f}, "
                f"{summary['newton_dx_y']:+.6f}, {summary['newton_dx_z']:+.6f}"
            )

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "init_residuals.csv"), init_rows)
    write_csv(os.path.join(args.out_dir, "train_log.csv"), train_rows)
    write_csv(os.path.join(args.out_dir, "test_log.csv"), test_rows)
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, os.path.join(args.out_dir, "free_fall_mlp.pt"))

    print("\n[Done]")
    print(f"  saved: {os.path.join(args.out_dir, 'init_residuals.csv')}")
    print(f"  saved: {os.path.join(args.out_dir, 'train_log.csv')}")
    print(f"  saved: {os.path.join(args.out_dir, 'test_log.csv')}")
    print(f"  saved: {os.path.join(args.out_dir, 'free_fall_mlp.pt')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--num_motions", type=int, default=10)
    parser.add_argument("--test_motion_id", type=int, default=0)
    parser.add_argument("--holdout_test", action="store_true", help="If set, exclude the test motion from training.")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dx_scale", type=float, default=0.20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--iter_increase_every", type=int, default=100)
    parser.add_argument("--test_every", type=int, default=100)
    parser.add_argument("--test_iters", type=int, default=15)
    parser.add_argument("--out_dir", type=str, default="./free_fall_mlp_logs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
