#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Free-fall implicit-Euler MLP optimizer experiments.

This script is based on the original single-case script:
- MLP predicts delta_y.
- The external loop applies y <- y + delta_y.
- Training minimizes the implicit Euler variational energy.
- Evaluation compares against the analytic optimum and Newton step.

It answers three questions:
1) How much does adding residual input help?
2) Can the network overfit the simplest single-motion case?
3) After overfitting / learning the simple rule, can it generalize in a minimal 9-train / 1-test split?

Usage:
    python experiment_residual_overfit_generalization.py

Optional:
    python experiment_residual_overfit_generalization.py --epochs 2000 --outdir results_residual_test
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")  # for headless Linux
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 1. Physics: implicit Euler variational energy
# ============================================================

@dataclass
class MotionCase:
    name: str
    p_n: Tuple[float, float, float]
    v_n: Tuple[float, float, float]
    m: float = 1.0
    g: float = 9.8
    dt: float = 0.01

    def tensors(self, device: torch.device, dtype: torch.dtype):
        p = torch.tensor(self.p_n, device=device, dtype=dtype)
        v = torch.tensor(self.v_n, device=device, dtype=dtype)
        return p, v, self.m, self.g, self.dt


def y_star_for_case(p_n: torch.Tensor, v_n: torch.Tensor, g: float, dt: float) -> torch.Tensor:
    """Analytic minimizer: y* = p_n + dt * v_n - dt^2 * g * e_z."""
    gravity_vec = torch.tensor([0.0, 0.0, g], device=p_n.device, dtype=p_n.dtype)
    return p_n + dt * v_n - (dt ** 2) * gravity_vec


def variational_energy(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float = 1.0,
    g: float = 9.8,
    dt: float = 0.01,
) -> torch.Tensor:
    """
    E(y) = m/(2 dt^2) ||y - p_n - dt v_n||^2 + m g y_z
    """
    residual = y - p_n - dt * v_n
    kinetic_term = (m / (2.0 * dt ** 2)) * torch.sum(residual ** 2)
    potential_term = m * g * y[2]
    return kinetic_term + potential_term


def newton_direction(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float = 1.0,
    g: float = 9.8,
    dt: float = 0.01,
) -> torch.Tensor:
    """
    For this quadratic free-fall energy, Newton reaches y* in one step.
    delta_newton = y* - y.
    """
    y_star = y_star_for_case(p_n, v_n, g, dt)
    return y_star - y


def energy_gap(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float = 1.0,
    g: float = 9.8,
    dt: float = 0.01,
) -> torch.Tensor:
    y_star = y_star_for_case(p_n, v_n, g, dt)
    e = variational_energy(y, p_n, v_n, m, g, dt)
    e_star = variational_energy(y_star, p_n, v_n, m, g, dt)
    return e - e_star


# ============================================================
# 2. Network: optional residual input
# ============================================================

class MLPOptimizer(nn.Module):
    """
    Base input:
        y(3) + p_n(3) + v_n(3) + [m, g, dt](3) = 12D

    Optional residual input:
        none                 -> 12D
        kinematic_residual   -> + (y - p_n - dt v_n), 15D
        optimality_residual  -> + (y - y_star), 15D
                              This equals dt^2/m times the energy gradient.
                              For this quadratic problem, Newton direction is exactly -optimality_residual.
    """
    def __init__(self, residual_mode: str = "none", hidden_dim: int = 64):
        super().__init__()
        assert residual_mode in {"none", "kinematic_residual", "optimality_residual"}
        self.residual_mode = residual_mode
        input_dim = 12 if residual_mode == "none" else 15

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Important for optimizer learning:
        # Start from delta = 0 rather than a random large step.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def make_input(
        self,
        y: torch.Tensor,
        p_n: torch.Tensor,
        v_n: torch.Tensor,
        m: float,
        g: float,
        dt: float,
    ) -> torch.Tensor:
        params = torch.tensor([m, g, dt], device=y.device, dtype=y.dtype)
        base = [y, p_n, v_n, params]

        if self.residual_mode == "kinematic_residual":
            residual = y - p_n - dt * v_n
            base.append(residual)
        elif self.residual_mode == "optimality_residual":
            y_star = y_star_for_case(p_n, v_n, g, dt)
            residual = y - y_star
            base.append(residual)

        return torch.cat(base, dim=-1)

    def forward(
        self,
        y: torch.Tensor,
        p_n: torch.Tensor,
        v_n: torch.Tensor,
        m: float,
        g: float,
        dt: float,
    ) -> torch.Tensor:
        inp = self.make_input(y, p_n, v_n, m, g, dt)
        return self.net(inp)


# ============================================================
# 3. Dataset
# ============================================================

def make_motion_cases() -> List[MotionCase]:
    """
    Ten tiny free-fall motions.
    The default split uses the first 9 for training and the last one for testing.
    """
    return [
        MotionCase("motion_00", (3.0, 4.0, 5.0),   (0.5, -0.5, 0.0)),
        MotionCase("motion_01", (2.0, 1.0, 4.0),   (0.2,  0.1, 0.3)),
        MotionCase("motion_02", (-1.0, 2.0, 3.0),  (-0.3, 0.4, 0.1)),
        MotionCase("motion_03", (0.0, 0.0, 2.0),   (0.0,  0.5, -0.2)),
        MotionCase("motion_04", (1.5, -2.0, 6.0),  (0.7, -0.1, 0.2)),
        MotionCase("motion_05", (-2.5, 1.5, 3.5),  (-0.4, -0.3, 0.4)),
        MotionCase("motion_06", (4.0, -1.0, 5.5),  (0.1,  0.6, -0.1)),
        MotionCase("motion_07", (-3.0, -2.0, 4.5), (0.6,  0.2, 0.0)),
        MotionCase("motion_08", (2.5, 3.0, 2.5),   (-0.2, 0.3, 0.5)),
        MotionCase("motion_09_test", (0.8, -3.0, 5.2), (0.35, -0.45, 0.25)),
    ]


# ============================================================
# 4. Training and evaluation
# ============================================================

def train_one_model(
    residual_mode: str,
    train_cases: List[MotionCase],
    *,
    epochs: int,
    lr: float,
    max_k: int,
    k_increase_every: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[MLPOptimizer, List[Dict]]:
    torch.manual_seed(seed)
    model = MLPOptimizer(residual_mode=residual_mode).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_log = []
    for epoch in range(epochs):
        K = min(1 + epoch // k_increase_every, max_k)

        total_loss_value = 0.0
        total_gap_value = 0.0

        # simple deterministic order; enough for this small diagnostic
        for case in train_cases:
            p_n, v_n, m, g, dt = case.tensors(device, dtype)
            y = p_n.clone()  # same as original script: start from current position

            for _ in range(K):
                delta = model(y, p_n, v_n, m, g, dt)
                y_next = y + delta

                loss = variational_energy(y_next, p_n, v_n, m, g, dt)
                gap = energy_gap(y_next, p_n, v_n, m, g, dt)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total_loss_value += float(loss.detach().cpu())
                total_gap_value += float(gap.detach().cpu())
                y = y_next.detach()

        train_log.append({
            "epoch": epoch,
            "K": K,
            "mean_loss": total_loss_value / max(1, len(train_cases) * K),
            "mean_gap": total_gap_value / max(1, len(train_cases) * K),
        })

    return model, train_log


@torch.no_grad()
def evaluate_case(
    model: MLPOptimizer,
    case: MotionCase,
    *,
    eval_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict:
    p_n, v_n, m, g, dt = case.tensors(device, dtype)
    y_star = y_star_for_case(p_n, v_n, g, dt)
    e_star = variational_energy(y_star, p_n, v_n, m, g, dt)

    y = p_n.clone()
    iterations = []

    # step 0
    e0 = variational_energy(y, p_n, v_n, m, g, dt)
    iterations.append({
        "step": 0,
        "y": y.detach().cpu().tolist(),
        "loss": float(e0.cpu()),
        "gap": float((e0 - e_star).cpu()),
        "dist_to_star": float(torch.norm(y - y_star).cpu()),
        "delta_norm": None,
        "newton_delta_error": None,
    })

    for step in range(1, eval_steps + 1):
        delta = model(y, p_n, v_n, m, g, dt)
        delta_newton = newton_direction(y, p_n, v_n, m, g, dt)
        y = y + delta

        e = variational_energy(y, p_n, v_n, m, g, dt)
        iterations.append({
            "step": step,
            "y": y.detach().cpu().tolist(),
            "loss": float(e.cpu()),
            "gap": float((e - e_star).cpu()),
            "dist_to_star": float(torch.norm(y - y_star).cpu()),
            "delta_norm": float(torch.norm(delta).cpu()),
            "newton_delta_error": float(torch.norm(delta - delta_newton).cpu()),
        })

    # Does the learned optimizer output zero when already at optimum?
    delta_at_star = model(y_star, p_n, v_n, m, g, dt)

    return {
        "case": asdict(case),
        "y_star": y_star.detach().cpu().tolist(),
        "E_star": float(e_star.cpu()),
        "iterations": iterations,
        "final_gap": iterations[-1]["gap"],
        "final_dist_to_star": iterations[-1]["dist_to_star"],
        "first_step_newton_error": iterations[1]["newton_delta_error"],
        "delta_at_star_norm": float(torch.norm(delta_at_star).cpu()),
    }


def summarize_run(
    run_name: str,
    residual_mode: str,
    train_cases: List[MotionCase],
    test_cases: List[MotionCase],
    *,
    epochs: int,
    lr: float,
    max_k: int,
    k_increase_every: int,
    eval_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Dict:
    model, train_log = train_one_model(
        residual_mode,
        train_cases,
        epochs=epochs,
        lr=lr,
        max_k=max_k,
        k_increase_every=k_increase_every,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    train_eval = [
        evaluate_case(model, c, eval_steps=eval_steps, device=device, dtype=dtype)
        for c in train_cases
    ]
    test_eval = [
        evaluate_case(model, c, eval_steps=eval_steps, device=device, dtype=dtype)
        for c in test_cases
    ]

    def mean_metric(items: List[Dict], key: str) -> float:
        return float(np.mean([item[key] for item in items])) if items else float("nan")

    summary = {
        "run_name": run_name,
        "residual_mode": residual_mode,
        "num_train_cases": len(train_cases),
        "num_test_cases": len(test_cases),
        "epochs": epochs,
        "lr": lr,
        "max_k": max_k,
        "k_increase_every": k_increase_every,
        "eval_steps": eval_steps,
        "seed": seed,
        "train_mean_final_gap": mean_metric(train_eval, "final_gap"),
        "train_mean_final_dist_to_star": mean_metric(train_eval, "final_dist_to_star"),
        "train_mean_first_step_newton_error": mean_metric(train_eval, "first_step_newton_error"),
        "train_mean_delta_at_star_norm": mean_metric(train_eval, "delta_at_star_norm"),
        "test_mean_final_gap": mean_metric(test_eval, "final_gap"),
        "test_mean_final_dist_to_star": mean_metric(test_eval, "final_dist_to_star"),
        "test_mean_first_step_newton_error": mean_metric(test_eval, "first_step_newton_error"),
        "test_mean_delta_at_star_norm": mean_metric(test_eval, "delta_at_star_norm"),
    }

    return {
        "summary": summary,
        "train_log": train_log,
        "train_eval": train_eval,
        "test_eval": test_eval,
    }


# ============================================================
# 5. I/O and plots
# ============================================================

def save_summary_csv(path: Path, summaries: List[Dict]) -> None:
    if not summaries:
        return
    keys = list(summaries[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)


def plot_training_curves(outdir: Path, reports: List[Dict]) -> None:
    plt.figure(figsize=(9, 5))
    for report in reports:
        log = report["train_log"]
        label = report["summary"]["run_name"]
        xs = [row["epoch"] for row in log]
        ys = [max(row["mean_gap"], 1e-14) for row in log]
        plt.plot(xs, ys, label=label)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean training gap: E - E*")
    plt.title("Training curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_final_bar(outdir: Path, summaries: List[Dict], metric: str) -> None:
    labels = [s["run_name"] for s in summaries]
    values = [max(float(s[metric]), 1e-14) for s in summaries]

    plt.figure(figsize=(10, 5))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.yscale("log")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(metric)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"{metric}.png", dpi=220, bbox_inches="tight")
    plt.close()


def print_summary_table(summaries: List[Dict]) -> None:
    cols = [
        "run_name",
        "train_mean_final_gap",
        "train_mean_delta_at_star_norm",
        "test_mean_final_gap",
        "test_mean_delta_at_star_norm",
    ]
    print("\n========== SUMMARY ==========")
    print(" | ".join(f"{c:>30s}" for c in cols))
    print("-" * (33 * len(cols)))
    for row in summaries:
        print(" | ".join(
            f"{row[c]:>30.4e}" if isinstance(row[c], float) else f"{str(row[c]):>30s}"
            for c in cols
        ))
    print("=============================\n")


# ============================================================
# 6. Main experiment matrix
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--k-increase-every", type=int, default=150)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--outdir", type=str, default="results_residual_overfit_generalization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = make_motion_cases()
    one_case = [cases[0]]
    train_9 = cases[:9]
    test_1 = [cases[9]]

    residual_modes = ["none", "kinematic_residual", "optimality_residual"]
    reports = []

    # Experiment A/B:
    # One-motion overfit + residual ablation.
    for mode in residual_modes:
        run_name = f"overfit_1_motion__{mode}"
        print(f"[RUN] {run_name}")
        reports.append(summarize_run(
            run_name,
            mode,
            one_case,
            one_case,  # test same as train for overfit diagnostic
            epochs=args.epochs,
            lr=args.lr,
            max_k=args.max_k,
            k_increase_every=args.k_increase_every,
            eval_steps=args.eval_steps,
            device=device,
            dtype=dtype,
            seed=args.seed,
        ))

    # Experiment C:
    # Minimal generalization: train 9 motions, test 1 held-out motion.
    for mode in residual_modes:
        run_name = f"generalize_9_train_1_test__{mode}"
        print(f"[RUN] {run_name}")
        reports.append(summarize_run(
            run_name,
            mode,
            train_9,
            test_1,
            epochs=args.epochs,
            lr=args.lr,
            max_k=args.max_k,
            k_increase_every=args.k_increase_every,
            eval_steps=args.eval_steps,
            device=device,
            dtype=dtype,
            seed=args.seed,
        ))

    # Save everything.
    full_report = {
        "args": vars(args),
        "device": str(device),
        "dtype": str(dtype),
        "reports": reports,
    }

    with (outdir / "full_report.json").open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    summaries = [r["summary"] for r in reports]
    save_summary_csv(outdir / "summary.csv", summaries)
    plot_training_curves(outdir, reports)
    plot_final_bar(outdir, summaries, "train_mean_final_gap")
    plot_final_bar(outdir, summaries, "test_mean_final_gap")
    plot_final_bar(outdir, summaries, "train_mean_delta_at_star_norm")
    plot_final_bar(outdir, summaries, "test_mean_delta_at_star_norm")

    print_summary_table(summaries)
    print(f"[DONE] Results saved to: {outdir.resolve()}")
    print("Key files:")
    print(f"  - {outdir / 'summary.csv'}")
    print(f"  - {outdir / 'full_report.json'}")
    print(f"  - {outdir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
