#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overfit-first diagnostic for the free-fall implicit-Euler learned optimizer.

Goal:
    Before testing residual ablation or 9/1 generalization, verify that the network
    can overfit one single motion and one single quadratic implicit-Euler energy.

Main changes compared with the earlier experiment:
    1. Focus only on one motion.
    2. Fix K=1 by default: learn the one-step optimizer first.
    3. Train on multiple y states from the SAME motion:
         y0, points on the line y0 -> y*, and y* itself.
       This is important because otherwise the model never sees y=y*, so there is
       no reason for it to output delta=0 at the optimum.
    4. Use scaled residual input:
         r_scaled = (y - y*) / dt
       Then the ideal raw output is simply:
         raw_delta = -r_scaled
         delta_y   = dt * raw_delta
    5. Use the exact quadratic gap:
         gap = m/(2 dt^2) ||y_next - y*||^2
       This avoids numerical cancellation from subtracting E - E*.

Usage:
    python overfit_first_freefall.py

Recommended:
    python overfit_first_freefall.py --epochs 5000 --lr 3e-3
    python overfit_first_freefall.py --epochs 8000 --lr 1e-3 --dtype float64
"""

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


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
    gravity_vec = torch.tensor([0.0, 0.0, g], device=p_n.device, dtype=p_n.dtype)
    return p_n + dt * v_n - (dt ** 2) * gravity_vec


def variational_energy(y, p_n, v_n, m, g, dt):
    residual = y - p_n - dt * v_n
    kinetic = (m / (2.0 * dt ** 2)) * torch.sum(residual ** 2)
    potential = m * g * y[2]
    return kinetic + potential


def exact_quadratic_gap(y: torch.Tensor, y_star: torch.Tensor, m: float, dt: float) -> torch.Tensor:
    """
    Since this energy is quadratic with Hessian m/dt^2 I:
        E(y) - E(y*) = m/(2 dt^2) ||y - y*||^2
    This is more stable than computing E(y)-E(y*) directly.
    """
    return (m / (2.0 * dt ** 2)) * torch.sum((y - y_star) ** 2)


def newton_delta(y: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
    return y_star - y


class OverfitMLPOptimizer(nn.Module):
    """
    The model predicts a raw step u, and the actual update is:
        delta_y = dt * u

    This makes the target raw step O(1), because:
        delta_y* = y* - y
        u*       = (y* - y) / dt = - (y - y*) / dt
    """
    def __init__(self, feature_mode: str, hidden_dim: int = 64):
        super().__init__()
        assert feature_mode in {
            "base_raw",
            "base_plus_scaled_opt_residual",
            "scaled_opt_residual_only",
        }
        self.feature_mode = feature_mode

        if feature_mode == "base_raw":
            input_dim = 12
        elif feature_mode == "base_plus_scaled_opt_residual":
            input_dim = 15
        elif feature_mode == "scaled_opt_residual_only":
            input_dim = 3
        else:
            raise ValueError(feature_mode)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

        # Start from zero update. This is safer for learned optimizer training.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def make_features(self, y, p_n, v_n, m, g, dt):
        y_star = y_star_for_case(p_n, v_n, g, dt)
        r_scaled = (y - y_star) / dt

        if self.feature_mode == "scaled_opt_residual_only":
            return r_scaled

        # Light normalization for base features.
        pos_scale = 5.0
        vel_scale = 1.0
        g_scale = 10.0
        dt_scale = 0.01
        params = torch.tensor([m, g / g_scale, dt / dt_scale], device=y.device, dtype=y.dtype)
        base = torch.cat([y / pos_scale, p_n / pos_scale, v_n / vel_scale, params], dim=-1)

        if self.feature_mode == "base_raw":
            return base
        return torch.cat([base, r_scaled], dim=-1)

    def forward(self, y, p_n, v_n, m, g, dt):
        features = self.make_features(y, p_n, v_n, m, g, dt)
        raw_delta = self.net(features)
        delta_y = dt * raw_delta
        return delta_y


def make_anchor_states(p_n, y_star, dt, *, num_line_points, local_perturb_std_dt_units, num_local_perturbs, seed):
    """
    Build training y states for ONE motion.

    Basic line anchors:
        y(alpha) = (1-alpha) y0 + alpha y*, alpha in [0, 1]

    Local perturbations around y*:
        y = y* + dt * std * randn(3)

    Including y* itself is crucial if we want delta_at_star -> 0.
    """
    assert num_line_points >= 2
    anchors = []
    for alpha in torch.linspace(0.0, 1.0, num_line_points, device=p_n.device, dtype=p_n.dtype):
        anchors.append((1.0 - alpha) * p_n + alpha * y_star)

    if num_local_perturbs > 0 and local_perturb_std_dt_units > 0:
        gen = torch.Generator(device=p_n.device)
        gen.manual_seed(seed)
        for _ in range(num_local_perturbs):
            noise = torch.randn(3, generator=gen, device=p_n.device, dtype=p_n.dtype)
            anchors.append(y_star + dt * local_perturb_std_dt_units * noise)
    return anchors


def train_overfit_one_mode(feature_mode, case, *, epochs, lr, hidden_dim, num_line_points,
                           local_perturb_std_dt_units, num_local_perturbs, device, dtype, seed):
    torch.manual_seed(seed)
    p_n, v_n, m, g, dt = case.tensors(device, dtype)
    y_star = y_star_for_case(p_n, v_n, g, dt)

    anchors = make_anchor_states(
        p_n, y_star, dt,
        num_line_points=num_line_points,
        local_perturb_std_dt_units=local_perturb_std_dt_units,
        num_local_perturbs=num_local_perturbs,
        seed=seed,
    )

    model = OverfitMLPOptimizer(feature_mode=feature_mode, hidden_dim=hidden_dim).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        dist_after = []
        delta_errors = []

        for y in anchors:
            delta = model(y, p_n, v_n, m, g, dt)
            y_next = y + delta
            gap = exact_quadratic_gap(y_next, y_star, m, dt)
            losses.append(gap)
            with torch.no_grad():
                dist_after.append(torch.norm(y_next - y_star))
                delta_errors.append(torch.norm(delta - newton_delta(y, y_star)))

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            log.append({
                "epoch": epoch,
                "mean_gap": float(loss.detach().cpu()),
                "max_dist_after_step": float(torch.stack(dist_after).max().cpu()),
                "mean_delta_newton_error": float(torch.stack(delta_errors).mean().cpu()),
            })

    metrics = evaluate_overfit_model(model, case, anchors, device=device, dtype=dtype)
    return {
        "feature_mode": feature_mode,
        "case": asdict(case),
        "config": {
            "epochs": epochs,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "num_line_points": num_line_points,
            "local_perturb_std_dt_units": local_perturb_std_dt_units,
            "num_local_perturbs": num_local_perturbs,
            "seed": seed,
        },
        "training_log": log,
        "metrics": metrics,
    }


@torch.no_grad()
def evaluate_overfit_model(model, case, anchors, *, device, dtype):
    p_n, v_n, m, g, dt = case.tensors(device, dtype)
    y_star = y_star_for_case(p_n, v_n, g, dt)

    def eval_state(y):
        delta = model(y, p_n, v_n, m, g, dt)
        y_next = y + delta
        return {
            "input_y": y.detach().cpu().tolist(),
            "delta": delta.detach().cpu().tolist(),
            "delta_norm": float(torch.norm(delta).cpu()),
            "newton_delta": newton_delta(y, y_star).detach().cpu().tolist(),
            "newton_delta_error": float(torch.norm(delta - newton_delta(y, y_star)).cpu()),
            "after_step_y": y_next.detach().cpu().tolist(),
            "after_step_dist_to_star": float(torch.norm(y_next - y_star).cpu()),
            "after_step_gap": float(exact_quadratic_gap(y_next, y_star, m, dt).cpu()),
        }

    init_eval = eval_state(p_n)
    star_eval = eval_state(y_star)
    anchor_evals = [eval_state(y) for y in anchors]

    rollout = []
    y = p_n.clone()
    for step in range(10):
        row = eval_state(y)
        row["step"] = step
        rollout.append(row)
        delta = model(y, p_n, v_n, m, g, dt)
        y = y + delta

    max_anchor_dist = max(row["after_step_dist_to_star"] for row in anchor_evals)
    max_anchor_gap = max(row["after_step_gap"] for row in anchor_evals)
    max_anchor_delta_error = max(row["newton_delta_error"] for row in anchor_evals)

    success_threshold = 1e-4
    success = (
        init_eval["after_step_dist_to_star"] < success_threshold
        and init_eval["after_step_gap"] < success_threshold
        and star_eval["delta_norm"] < success_threshold
        and max_anchor_dist < success_threshold
    )

    return {
        "y_star": y_star.detach().cpu().tolist(),
        "initial_state_eval": init_eval,
        "star_state_eval": star_eval,
        "max_anchor_after_step_dist_to_star": max_anchor_dist,
        "max_anchor_after_step_gap": max_anchor_gap,
        "max_anchor_delta_newton_error": max_anchor_delta_error,
        "rollout_from_initial": rollout,
        "success_threshold": success_threshold,
        "success": success,
    }


def save_summary_csv(outdir, reports):
    rows = []
    for report in reports:
        m = report["metrics"]
        init_eval = m["initial_state_eval"]
        star_eval = m["star_state_eval"]
        rows.append({
            "feature_mode": report["feature_mode"],
            "success": m["success"],
            "initial_after_step_gap": init_eval["after_step_gap"],
            "initial_after_step_dist_to_star": init_eval["after_step_dist_to_star"],
            "initial_delta_newton_error": init_eval["newton_delta_error"],
            "delta_at_star_norm": star_eval["delta_norm"],
            "max_anchor_after_step_gap": m["max_anchor_after_step_gap"],
            "max_anchor_after_step_dist_to_star": m["max_anchor_after_step_dist_to_star"],
            "max_anchor_delta_newton_error": m["max_anchor_delta_newton_error"],
        })

    path = outdir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training(outdir, reports):
    plt.figure(figsize=(9, 5))
    for report in reports:
        xs = [row["epoch"] for row in report["training_log"]]
        ys = [max(row["mean_gap"], 1e-16) for row in report["training_log"]]
        plt.plot(xs, ys, label=report["feature_mode"])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean anchor gap after one step")
    plt.title("Overfit-first training curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close()


def print_summary(reports):
    print("\n========== OVERFIT-FIRST SUMMARY ==========")
    print(
        f"{'feature_mode':>32s} | "
        f"{'success':>7s} | "
        f"{'init_gap':>12s} | "
        f"{'init_dist':>12s} | "
        f"{'delta@star':>12s} | "
        f"{'max_anchor_dist':>16s}"
    )
    print("-" * 105)
    for report in reports:
        m = report["metrics"]
        init_eval = m["initial_state_eval"]
        star_eval = m["star_state_eval"]
        print(
            f"{report['feature_mode']:>32s} | "
            f"{str(m['success']):>7s} | "
            f"{init_eval['after_step_gap']:>12.4e} | "
            f"{init_eval['after_step_dist_to_star']:>12.4e} | "
            f"{star_eval['delta_norm']:>12.4e} | "
            f"{m['max_anchor_after_step_dist_to_star']:>16.4e}"
        )
    print("===========================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-line-points", type=int, default=11)
    parser.add_argument("--local-perturb-std-dt-units", type=float, default=0.0)
    parser.add_argument("--num-local-perturbs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--outdir", type=str, default="results_overfit_first")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["base_raw", "base_plus_scaled_opt_residual", "scaled_opt_residual_only"],
        choices=["base_raw", "base_plus_scaled_opt_residual", "scaled_opt_residual_only"],
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    case = MotionCase(
        name="single_motion_default",
        p_n=(3.0, 4.0, 5.0),
        v_n=(0.5, -0.5, 0.0),
        m=1.0,
        g=9.8,
        dt=0.01,
    )

    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Case: {case}")

    reports = []
    for mode in args.modes:
        print(f"\n[RUN] feature_mode={mode}")
        report = train_overfit_one_mode(
            mode,
            case,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_line_points=args.num_line_points,
            local_perturb_std_dt_units=args.local_perturb_std_dt_units,
            num_local_perturbs=args.num_local_perturbs,
            device=device,
            dtype=dtype,
            seed=args.seed,
        )
        reports.append(report)

    full_report = {
        "args": vars(args),
        "device": str(device),
        "dtype": str(dtype),
        "reports": reports,
    }

    with (outdir / "full_report.json").open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    save_summary_csv(outdir, reports)
    plot_training(outdir, reports)
    print_summary(reports)

    print(f"[DONE] Saved to: {outdir.resolve()}")
    print(f"  - {outdir / 'summary.csv'}")
    print(f"  - {outdir / 'full_report.json'}")
    print(f"  - {outdir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
