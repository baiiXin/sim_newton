#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair comparison with line + local anchors:
Is residual input necessary for learning a local optimizer field?

Compared with compare_residual_necessity_fair.py:
    - Training is no longer only on the 1D line y0 -> y*.
    - Training states include:
        1) line anchors from y0 to y*
        2) local 3D perturbations around y*
    - Loss is still energy-only.
    - No Newton-step / delta MSE supervision is used.
    - Output is still dt-scaled:
          delta_y = dt * raw_delta
    - All modes use data standardization fitted on their own training features.
    - The ONLY conceptual difference is feature mode:
        base_only:
            [y, p_n, v_n, m, g, dt]
        residual_only:
            [(y - y*) / dt]
        base_plus_residual:
            [y, p_n, v_n, m, g, dt, (y - y*) / dt]

This script answers:
    1. Can base-only learn not just a line, but a local 3D optimizer field?
    2. Does residual make local learning faster or more accurate?
    3. Does residual help held-out local perturbations and extrapolation?

Usage:
    python compare_residual_necessity_line_local.py

Recommended:
    python compare_residual_necessity_line_local.py --epochs 3000 --lr 1e-3
    python compare_residual_necessity_line_local.py --epochs 5000 --lr 1e-3 --dtype float64
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


# ============================================================
# 1. Motion and quadratic implicit-Euler energy
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
    e_z = torch.tensor([0.0, 0.0, 1.0], device=p_n.device, dtype=p_n.dtype)
    return p_n + dt * v_n - (dt ** 2) * g * e_z


def exact_quadratic_gap(y: torch.Tensor, y_star: torch.Tensor, m: float, dt: float) -> torch.Tensor:
    """
    For this quadratic free-fall energy:
        E(y) - E(y*) = m/(2 dt^2) ||y - y*||^2
    """
    return (m / (2.0 * dt ** 2)) * torch.sum((y - y_star) ** 2)


def newton_delta(y: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
    """
    Since the energy is quadratic with constant Hessian, Newton reaches y* in one step.
    """
    return y_star - y


# ============================================================
# 2. Feature modes
# ============================================================

class FeatureBuilder:
    def __init__(self, mode: str):
        assert mode in {"base_only", "residual_only", "base_plus_residual"}
        self.mode = mode

    def raw_features(
        self,
        y: torch.Tensor,
        p_n: torch.Tensor,
        v_n: torch.Tensor,
        m: float,
        g: float,
        dt: float,
    ) -> torch.Tensor:
        y_star = y_star_for_case(p_n, v_n, g, dt)
        scaled_residual = (y - y_star) / dt

        if self.mode == "residual_only":
            return scaled_residual

        params = torch.tensor([m, g, dt], device=y.device, dtype=y.dtype)
        base = torch.cat([y, p_n, v_n, params], dim=-1)

        if self.mode == "base_only":
            return base

        return torch.cat([base, scaled_residual], dim=-1)


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def fit_standardizer(features: List[torch.Tensor], eps: float = 1e-8) -> Standardizer:
    x = torch.stack(features, dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std < eps, torch.ones_like(std), std)
    return Standardizer(mean=mean, std=std)


# ============================================================
# 3. MLP learned optimizer
# ============================================================

class MLPOptimizer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

        # Start from zero update.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 4. Training/evaluation states
# ============================================================

def make_line_anchors(
    p_n: torch.Tensor,
    y_star: torch.Tensor,
    num_line_points: int,
) -> List[torch.Tensor]:
    states = []
    for alpha in torch.linspace(0.0, 1.0, num_line_points, device=p_n.device, dtype=p_n.dtype):
        states.append((1.0 - alpha) * p_n + alpha * y_star)
    return states


def make_local_states(
    center: torch.Tensor,
    dt: float,
    *,
    num_samples: int,
    std_dt_units: float,
    seed: int,
) -> List[torch.Tensor]:
    """
    y = center + dt * std_dt_units * N(0, I)
    """
    if num_samples <= 0:
        return []

    gen = torch.Generator(device=center.device)
    gen.manual_seed(seed)

    states = []
    for _ in range(num_samples):
        noise = torch.randn(3, generator=gen, device=center.device, dtype=center.dtype)
        states.append(center + dt * std_dt_units * noise)
    return states


def make_extrapolation_states(
    p_n: torch.Tensor,
    y_star: torch.Tensor,
    *,
    num_points: int,
    alpha_min: float,
    alpha_max: float,
) -> List[torch.Tensor]:
    states = []
    for alpha in torch.linspace(alpha_min, alpha_max, num_points, device=p_n.device, dtype=p_n.dtype):
        states.append((1.0 - alpha) * p_n + alpha * y_star)
    return states


def merge_states(*groups: List[torch.Tensor]) -> List[torch.Tensor]:
    states = []
    for group in groups:
        states.extend(group)
    return states


# ============================================================
# 5. Evaluation
# ============================================================

@torch.no_grad()
def eval_states(
    model: MLPOptimizer,
    builder: FeatureBuilder,
    standardizer: Standardizer,
    states: List[torch.Tensor],
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float,
    g: float,
    dt: float,
) -> Dict:
    y_star = y_star_for_case(p_n, v_n, g, dt)

    if len(states) == 0:
        return {
            "mean_dist": float("nan"),
            "max_dist": float("nan"),
            "mean_gap": float("nan"),
            "max_gap": float("nan"),
            "mean_delta_newton_error": float("nan"),
            "max_delta_newton_error": float("nan"),
            "mean_delta_norm": float("nan"),
            "max_delta_norm": float("nan"),
        }

    dists = []
    gaps = []
    delta_errors = []
    delta_norms = []

    for y in states:
        x_raw = builder.raw_features(y, p_n, v_n, m, g, dt)
        x = standardizer.transform(x_raw)
        raw_delta = model(x)
        delta = dt * raw_delta
        y_next = y + delta

        dists.append(torch.norm(y_next - y_star))
        gaps.append(exact_quadratic_gap(y_next, y_star, m, dt))
        delta_errors.append(torch.norm(delta - newton_delta(y, y_star)))
        delta_norms.append(torch.norm(delta))

    dists_t = torch.stack(dists)
    gaps_t = torch.stack(gaps)
    delta_errors_t = torch.stack(delta_errors)
    delta_norms_t = torch.stack(delta_norms)

    return {
        "mean_dist": float(dists_t.mean().detach().cpu()),
        "max_dist": float(dists_t.max().detach().cpu()),
        "mean_gap": float(gaps_t.mean().detach().cpu()),
        "max_gap": float(gaps_t.max().detach().cpu()),
        "mean_delta_newton_error": float(delta_errors_t.mean().detach().cpu()),
        "max_delta_newton_error": float(delta_errors_t.max().detach().cpu()),
        "mean_delta_norm": float(delta_norms_t.mean().detach().cpu()),
        "max_delta_norm": float(delta_norms_t.max().detach().cpu()),
    }


# ============================================================
# 6. Training for one mode
# ============================================================

def train_one_mode(
    mode: str,
    case: MotionCase,
    args,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict:
    torch.manual_seed(args.seed)

    p_n, v_n, m, g, dt = case.tensors(device, dtype)
    y_star = y_star_for_case(p_n, v_n, g, dt)

    # Training anchors.
    train_line_states = make_line_anchors(p_n, y_star, args.num_line_points)
    train_local_states = make_local_states(
        y_star,
        dt,
        num_samples=args.train_local_samples,
        std_dt_units=args.train_local_std_dt_units,
        seed=args.seed + 10,
    )
    train_states = merge_states(train_line_states, train_local_states)

    # Held-out evaluation states.
    heldout_local_states = make_local_states(
        y_star,
        dt,
        num_samples=args.heldout_local_samples,
        std_dt_units=args.heldout_local_std_dt_units,
        seed=args.seed + 1000,
    )
    wide_local_states = make_local_states(
        y_star,
        dt,
        num_samples=args.wide_local_samples,
        std_dt_units=args.wide_local_std_dt_units,
        seed=args.seed + 2000,
    )
    extrap_states = make_extrapolation_states(
        p_n,
        y_star,
        num_points=args.extrap_num_points,
        alpha_min=args.extrap_alpha_min,
        alpha_max=args.extrap_alpha_max,
    )

    builder = FeatureBuilder(mode)

    raw_train_features = [
        builder.raw_features(y, p_n, v_n, m, g, dt) for y in train_states
    ]
    standardizer = fit_standardizer(raw_train_features)

    model = MLPOptimizer(
        input_dim=raw_train_features[0].numel(),
        hidden_dim=args.hidden_dim,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = []
    first_train_success_epoch = -1
    first_heldout_success_epoch = -1

    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)

        losses = []
        for y in train_states:
            x_raw = builder.raw_features(y, p_n, v_n, m, g, dt)
            x = standardizer.transform(x_raw)
            raw_delta = model(x)
            delta = dt * raw_delta
            y_next = y + delta
            losses.append(exact_quadratic_gap(y_next, y_star, m, dt))

        train_loss = torch.stack(losses).mean()
        train_loss.backward()
        optimizer.step()

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            train_line_eval = eval_states(model, builder, standardizer, train_line_states, p_n, v_n, m, g, dt)
            train_local_eval = eval_states(model, builder, standardizer, train_local_states, p_n, v_n, m, g, dt)
            train_all_eval = eval_states(model, builder, standardizer, train_states, p_n, v_n, m, g, dt)
            init_eval = eval_states(model, builder, standardizer, [p_n], p_n, v_n, m, g, dt)
            star_eval = eval_states(model, builder, standardizer, [y_star], p_n, v_n, m, g, dt)
            heldout_local_eval = eval_states(model, builder, standardizer, heldout_local_states, p_n, v_n, m, g, dt)
            wide_local_eval = eval_states(model, builder, standardizer, wide_local_states, p_n, v_n, m, g, dt)
            extrap_eval = eval_states(model, builder, standardizer, extrap_states, p_n, v_n, m, g, dt)

            train_success = (
                train_line_eval["max_dist"] < args.success_threshold
                and train_local_eval["max_dist"] < args.success_threshold
                and star_eval["max_delta_norm"] < args.success_threshold
            )
            heldout_success = heldout_local_eval["max_dist"] < args.success_threshold

            if train_success and first_train_success_epoch < 0:
                first_train_success_epoch = epoch
            if heldout_success and first_heldout_success_epoch < 0:
                first_heldout_success_epoch = epoch

            log.append({
                "epoch": epoch,
                "train_loss": float(train_loss.detach().cpu()),
                "train_all_max_dist": train_all_eval["max_dist"],
                "train_line_max_dist": train_line_eval["max_dist"],
                "train_local_max_dist": train_local_eval["max_dist"],
                "init_dist": init_eval["max_dist"],
                "delta_at_star": star_eval["max_delta_norm"],
                "heldout_local_max_dist": heldout_local_eval["max_dist"],
                "wide_local_max_dist": wide_local_eval["max_dist"],
                "extrap_max_dist": extrap_eval["max_dist"],
                "train_success": train_success,
                "heldout_success": heldout_success,
            })

    # Final evaluation.
    train_line_eval = eval_states(model, builder, standardizer, train_line_states, p_n, v_n, m, g, dt)
    train_local_eval = eval_states(model, builder, standardizer, train_local_states, p_n, v_n, m, g, dt)
    train_all_eval = eval_states(model, builder, standardizer, train_states, p_n, v_n, m, g, dt)
    init_eval = eval_states(model, builder, standardizer, [p_n], p_n, v_n, m, g, dt)
    star_eval = eval_states(model, builder, standardizer, [y_star], p_n, v_n, m, g, dt)
    heldout_local_eval = eval_states(model, builder, standardizer, heldout_local_states, p_n, v_n, m, g, dt)
    wide_local_eval = eval_states(model, builder, standardizer, wide_local_states, p_n, v_n, m, g, dt)
    extrap_eval = eval_states(model, builder, standardizer, extrap_states, p_n, v_n, m, g, dt)

    success_train = (
        train_line_eval["max_dist"] < args.success_threshold
        and train_local_eval["max_dist"] < args.success_threshold
        and star_eval["max_delta_norm"] < args.success_threshold
    )
    success_heldout_local = heldout_local_eval["max_dist"] < args.success_threshold
    success_wide_local = wide_local_eval["max_dist"] < args.success_threshold
    success_extrap = extrap_eval["max_dist"] < args.success_threshold

    summary = {
        "mode": mode,

        "success_train": success_train,
        "success_heldout_local": success_heldout_local,
        "success_wide_local": success_wide_local,
        "success_extrap": success_extrap,

        "first_train_success_epoch": first_train_success_epoch,
        "first_heldout_success_epoch": first_heldout_success_epoch,

        "init_dist": init_eval["max_dist"],
        "init_gap": init_eval["max_gap"],
        "init_delta_newton_error": init_eval["max_delta_newton_error"],

        "delta_at_star_norm": star_eval["max_delta_norm"],
        "star_after_step_dist": star_eval["max_dist"],

        "train_line_mean_dist": train_line_eval["mean_dist"],
        "train_line_max_dist": train_line_eval["max_dist"],
        "train_line_mean_gap": train_line_eval["mean_gap"],
        "train_line_max_gap": train_line_eval["max_gap"],

        "train_local_mean_dist": train_local_eval["mean_dist"],
        "train_local_max_dist": train_local_eval["max_dist"],
        "train_local_mean_gap": train_local_eval["mean_gap"],
        "train_local_max_gap": train_local_eval["max_gap"],

        "train_all_mean_dist": train_all_eval["mean_dist"],
        "train_all_max_dist": train_all_eval["max_dist"],
        "train_all_mean_gap": train_all_eval["mean_gap"],
        "train_all_max_gap": train_all_eval["max_gap"],

        "heldout_local_mean_dist": heldout_local_eval["mean_dist"],
        "heldout_local_max_dist": heldout_local_eval["max_dist"],
        "heldout_local_mean_gap": heldout_local_eval["mean_gap"],
        "heldout_local_max_gap": heldout_local_eval["max_gap"],

        "wide_local_mean_dist": wide_local_eval["mean_dist"],
        "wide_local_max_dist": wide_local_eval["max_dist"],
        "wide_local_mean_gap": wide_local_eval["mean_gap"],
        "wide_local_max_gap": wide_local_eval["max_gap"],

        "extrap_mean_dist": extrap_eval["mean_dist"],
        "extrap_max_dist": extrap_eval["max_dist"],
        "extrap_mean_gap": extrap_eval["mean_gap"],
        "extrap_max_gap": extrap_eval["max_gap"],
    }

    return {
        "mode": mode,
        "summary": summary,
        "training_log": log,
        "case": asdict(case),
        "feature_mean": standardizer.mean.detach().cpu().tolist(),
        "feature_std": standardizer.std.detach().cpu().tolist(),
        "config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "num_line_points": args.num_line_points,
            "train_local_samples": args.train_local_samples,
            "train_local_std_dt_units": args.train_local_std_dt_units,
            "heldout_local_samples": args.heldout_local_samples,
            "heldout_local_std_dt_units": args.heldout_local_std_dt_units,
            "wide_local_samples": args.wide_local_samples,
            "wide_local_std_dt_units": args.wide_local_std_dt_units,
            "extrap_num_points": args.extrap_num_points,
            "extrap_alpha_min": args.extrap_alpha_min,
            "extrap_alpha_max": args.extrap_alpha_max,
            "success_threshold": args.success_threshold,
            "seed": args.seed,
        },
    }


# ============================================================
# 7. Output
# ============================================================

def save_summary_csv(outdir: Path, reports: List[Dict]) -> None:
    rows = [r["summary"] for r in reports]
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(outdir: Path, reports: List[Dict]) -> None:
    specs = [
        ("train_all_max_dist", "Train all max distance", "train_all_max_dist.png"),
        ("train_line_max_dist", "Train line max distance", "train_line_max_dist.png"),
        ("train_local_max_dist", "Train local max distance", "train_local_max_dist.png"),
        ("heldout_local_max_dist", "Held-out local max distance", "heldout_local_max_dist.png"),
        ("wide_local_max_dist", "Wide local max distance", "wide_local_max_dist.png"),
        ("delta_at_star", "||delta(y*)||", "delta_at_star.png"),
        ("extrap_max_dist", "Extrapolation line max distance", "extrap_max_dist.png"),
    ]

    for key, ylabel, filename in specs:
        plt.figure(figsize=(9, 5))
        for report in reports:
            xs = [row["epoch"] for row in report["training_log"]]
            ys = [max(row[key], 1e-16) for row in report["training_log"]]
            plt.plot(xs, ys, label=report["mode"])

        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=220, bbox_inches="tight")
        plt.close()


def print_summary(reports: List[Dict]) -> None:
    print("\n========== LINE + LOCAL RESIDUAL NECESSITY SUMMARY ==========")
    print(
        f"{'mode':>22s} | "
        f"{'train?':>6s} | "
        f"{'held?':>6s} | "
        f"{'wide?':>6s} | "
        f"{'extrap?':>7s} | "
        f"{'ep_train':>8s} | "
        f"{'ep_held':>7s} | "
        f"{'init':>10s} | "
        f"{'d@star':>10s} | "
        f"{'line':>10s} | "
        f"{'local':>10s} | "
        f"{'heldout':>10s} | "
        f"{'wide':>10s} | "
        f"{'extrap':>10s}"
    )
    print("-" * 165)

    for report in reports:
        s = report["summary"]
        print(
            f"{s['mode']:>22s} | "
            f"{str(s['success_train']):>6s} | "
            f"{str(s['success_heldout_local']):>6s} | "
            f"{str(s['success_wide_local']):>6s} | "
            f"{str(s['success_extrap']):>7s} | "
            f"{s['first_train_success_epoch']:>8d} | "
            f"{s['first_heldout_success_epoch']:>7d} | "
            f"{s['init_dist']:>10.3e} | "
            f"{s['delta_at_star_norm']:>10.3e} | "
            f"{s['train_line_max_dist']:>10.3e} | "
            f"{s['train_local_max_dist']:>10.3e} | "
            f"{s['heldout_local_max_dist']:>10.3e} | "
            f"{s['wide_local_max_dist']:>10.3e} | "
            f"{s['extrap_max_dist']:>10.3e}"
        )

    print("==============================================================\n")


# ============================================================
# 8. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--success-threshold", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--outdir", type=str, default="results_compare_residual_necessity_line_local")

    # Training states.
    parser.add_argument("--num-line-points", type=int, default=11)
    parser.add_argument("--train-local-samples", type=int, default=128)
    parser.add_argument("--train-local-std-dt-units", type=float, default=1.0)

    # Held-out evaluation states.
    parser.add_argument("--heldout-local-samples", type=int, default=256)
    parser.add_argument("--heldout-local-std-dt-units", type=float, default=1.0)
    parser.add_argument("--wide-local-samples", type=int, default=256)
    parser.add_argument("--wide-local-std-dt-units", type=float, default=2.0)

    # Extrapolation along the original line.
    parser.add_argument("--extrap-num-points", type=int, default=31)
    parser.add_argument("--extrap-alpha-min", type=float, default=-1.0)
    parser.add_argument("--extrap-alpha-max", type=float, default=2.0)

    parser.add_argument(
        "--modes",
        nargs="+",
        default=["base_only", "residual_only", "base_plus_residual"],
        choices=["base_only", "residual_only", "base_plus_residual"],
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
    print("Training recipe: K=1, line + local anchors, energy-only, dt-scaled output, data standardization.")
    print("Only feature mode changes.")

    reports = []
    for mode in args.modes:
        print(f"\n[RUN] mode={mode}")
        reports.append(train_one_mode(mode, case, args, device, dtype))

    full_report = {
        "args": vars(args),
        "device": str(device),
        "dtype": str(dtype),
        "reports": reports,
    }

    with (outdir / "full_report.json").open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    save_summary_csv(outdir, reports)
    plot_curves(outdir, reports)
    print_summary(reports)

    print(f"[DONE] Saved to: {outdir.resolve()}")
    print(f"  - {outdir / 'summary.csv'}")
    print(f"  - {outdir / 'full_report.json'}")
    print(f"  - {outdir / 'train_all_max_dist.png'}")
    print(f"  - {outdir / 'train_line_max_dist.png'}")
    print(f"  - {outdir / 'train_local_max_dist.png'}")
    print(f"  - {outdir / 'heldout_local_max_dist.png'}")
    print(f"  - {outdir / 'wide_local_max_dist.png'}")
    print(f"  - {outdir / 'delta_at_star.png'}")
    print(f"  - {outdir / 'extrap_max_dist.png'}")


if __name__ == "__main__":
    main()
