#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair comparison: Is residual input necessary?

All modes use the SAME successful training recipe from the previous base-only result:
    - one motion
    - K = 1
    - line anchors from y0 to y*
    - energy-only loss, no delta/Newton supervision
    - output parameterization: delta_y = dt * raw_delta
    - data standardization of input features

The ONLY difference between modes is the feature set:
    1. base_only:
        [y, p_n, v_n, m, g, dt]
    2. residual_only:
        [(y - y*) / dt]
    3. base_plus_residual:
        [y, p_n, v_n, m, g, dt, (y - y*) / dt]

Usage:
    python compare_residual_necessity_fair.py
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
    e_z = torch.tensor([0.0, 0.0, 1.0], device=p_n.device, dtype=p_n.dtype)
    return p_n + dt * v_n - (dt ** 2) * g * e_z


def exact_quadratic_gap(y: torch.Tensor, y_star: torch.Tensor, m: float, dt: float) -> torch.Tensor:
    return (m / (2.0 * dt ** 2)) * torch.sum((y - y_star) ** 2)


def newton_delta(y: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
    return y_star - y


class FeatureBuilder:
    def __init__(self, mode: str):
        assert mode in {"base_only", "residual_only", "base_plus_residual"}
        self.mode = mode

    def raw_features(self, y, p_n, v_n, m, g, dt):
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
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_line_anchors(p_n, y_star, num_line_points: int):
    anchors = []
    for alpha in torch.linspace(0.0, 1.0, num_line_points, device=p_n.device, dtype=p_n.dtype):
        anchors.append((1.0 - alpha) * p_n + alpha * y_star)
    return anchors


def make_local_eval_states(y_star, dt, num_samples: int, std_dt_units: float, seed: int):
    gen = torch.Generator(device=y_star.device)
    gen.manual_seed(seed)
    states = []
    for _ in range(num_samples):
        noise = torch.randn(3, generator=gen, device=y_star.device, dtype=y_star.dtype)
        states.append(y_star + dt * std_dt_units * noise)
    return states


def make_extrapolation_states(p_n, y_star, num_points: int, alpha_min: float, alpha_max: float):
    states = []
    for alpha in torch.linspace(alpha_min, alpha_max, num_points, device=p_n.device, dtype=p_n.dtype):
        states.append((1.0 - alpha) * p_n + alpha * y_star)
    return states


@torch.no_grad()
def eval_states(model, builder, standardizer, states, p_n, v_n, m, g, dt):
    y_star = y_star_for_case(p_n, v_n, g, dt)

    dists, gaps, delta_errors, delta_norms = [], [], [], []
    for y in states:
        x = standardizer.transform(builder.raw_features(y, p_n, v_n, m, g, dt))
        delta = dt * model(x)
        y_next = y + delta

        dists.append(torch.norm(y_next - y_star))
        gaps.append(exact_quadratic_gap(y_next, y_star, m, dt))
        delta_errors.append(torch.norm(delta - newton_delta(y, y_star)))
        delta_norms.append(torch.norm(delta))

    dists = torch.stack(dists)
    gaps = torch.stack(gaps)
    delta_errors = torch.stack(delta_errors)
    delta_norms = torch.stack(delta_norms)

    return {
        "mean_dist": float(dists.mean().cpu()),
        "max_dist": float(dists.max().cpu()),
        "mean_gap": float(gaps.mean().cpu()),
        "max_gap": float(gaps.max().cpu()),
        "mean_delta_newton_error": float(delta_errors.mean().cpu()),
        "max_delta_newton_error": float(delta_errors.max().cpu()),
        "mean_delta_norm": float(delta_norms.mean().cpu()),
        "max_delta_norm": float(delta_norms.max().cpu()),
    }


def train_one_mode(mode, case, args, device, dtype):
    torch.manual_seed(args.seed)

    p_n, v_n, m, g, dt = case.tensors(device, dtype)
    y_star = y_star_for_case(p_n, v_n, g, dt)

    train_states = make_line_anchors(p_n, y_star, args.num_line_points)
    local_states = make_local_eval_states(y_star, dt, args.local_eval_samples, args.local_eval_std_dt_units, args.seed + 1000)
    extrap_states = make_extrapolation_states(
        p_n, y_star, args.extrap_num_points, args.extrap_alpha_min, args.extrap_alpha_max
    )

    builder = FeatureBuilder(mode)
    train_raw_features = [builder.raw_features(y, p_n, v_n, m, g, dt) for y in train_states]
    standardizer = fit_standardizer(train_raw_features)

    model = MLPOptimizer(input_dim=train_raw_features[0].numel(), hidden_dim=args.hidden_dim).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = []
    first_success_epoch = -1

    for epoch in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        losses = []
        for y in train_states:
            x = standardizer.transform(builder.raw_features(y, p_n, v_n, m, g, dt))
            delta = dt * model(x)
            y_next = y + delta
            losses.append(exact_quadratic_gap(y_next, y_star, m, dt))

        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            train_eval = eval_states(model, builder, standardizer, train_states, p_n, v_n, m, g, dt)
            init_eval = eval_states(model, builder, standardizer, [p_n], p_n, v_n, m, g, dt)
            star_eval = eval_states(model, builder, standardizer, [y_star], p_n, v_n, m, g, dt)
            local_eval = eval_states(model, builder, standardizer, local_states, p_n, v_n, m, g, dt)
            extrap_eval = eval_states(model, builder, standardizer, extrap_states, p_n, v_n, m, g, dt)

            success = (
                init_eval["max_dist"] < args.success_threshold
                and star_eval["max_delta_norm"] < args.success_threshold
                and train_eval["max_dist"] < args.success_threshold
            )
            if success and first_success_epoch < 0:
                first_success_epoch = epoch

            log.append({
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "train_max_dist": train_eval["max_dist"],
                "init_dist": init_eval["max_dist"],
                "delta_at_star": star_eval["max_delta_norm"],
                "local_max_dist": local_eval["max_dist"],
                "extrap_max_dist": extrap_eval["max_dist"],
                "train_success": success,
            })

    train_eval = eval_states(model, builder, standardizer, train_states, p_n, v_n, m, g, dt)
    init_eval = eval_states(model, builder, standardizer, [p_n], p_n, v_n, m, g, dt)
    star_eval = eval_states(model, builder, standardizer, [y_star], p_n, v_n, m, g, dt)
    local_eval = eval_states(model, builder, standardizer, local_states, p_n, v_n, m, g, dt)
    extrap_eval = eval_states(model, builder, standardizer, extrap_states, p_n, v_n, m, g, dt)

    summary = {
        "mode": mode,
        "success_train_line": (
            init_eval["max_dist"] < args.success_threshold
            and star_eval["max_delta_norm"] < args.success_threshold
            and train_eval["max_dist"] < args.success_threshold
        ),
        "success_local": local_eval["max_dist"] < args.success_threshold,
        "success_extrap": extrap_eval["max_dist"] < args.success_threshold,
        "first_success_epoch": first_success_epoch,

        "init_gap": init_eval["max_gap"],
        "init_dist": init_eval["max_dist"],
        "init_delta_newton_error": init_eval["max_delta_newton_error"],

        "delta_at_star_norm": star_eval["max_delta_norm"],
        "star_after_step_dist": star_eval["max_dist"],

        "train_mean_dist": train_eval["mean_dist"],
        "train_max_dist": train_eval["max_dist"],
        "train_mean_gap": train_eval["mean_gap"],
        "train_max_gap": train_eval["max_gap"],

        "local_mean_dist": local_eval["mean_dist"],
        "local_max_dist": local_eval["max_dist"],
        "local_mean_gap": local_eval["mean_gap"],
        "local_max_gap": local_eval["max_gap"],

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
    }


def save_summary_csv(outdir, reports):
    rows = [r["summary"] for r in reports]
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(outdir, reports):
    specs = [
        ("train_max_dist", "Train-line max distance", "train_line_max_dist.png"),
        ("delta_at_star", "||delta(y*)||", "delta_at_star.png"),
        ("local_max_dist", "Local y* perturbation max distance", "local_max_dist.png"),
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


def print_summary(reports):
    print("\n========== FAIR RESIDUAL NECESSITY SUMMARY ==========")
    print(
        f"{'mode':>22s} | {'train?':>6s} | {'local?':>6s} | {'extrap?':>7s} | "
        f"{'first_ep':>8s} | {'init_dist':>11s} | {'delta@star':>11s} | "
        f"{'train_max':>11s} | {'local_max':>11s} | {'extrap_max':>11s}"
    )
    print("-" * 125)
    for report in reports:
        s = report["summary"]
        print(
            f"{s['mode']:>22s} | {str(s['success_train_line']):>6s} | "
            f"{str(s['success_local']):>6s} | {str(s['success_extrap']):>7s} | "
            f"{s['first_success_epoch']:>8d} | {s['init_dist']:>11.4e} | "
            f"{s['delta_at_star_norm']:>11.4e} | {s['train_max_dist']:>11.4e} | "
            f"{s['local_max_dist']:>11.4e} | {s['extrap_max_dist']:>11.4e}"
        )
    print("=====================================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-line-points", type=int, default=11)
    parser.add_argument("--success-threshold", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--outdir", type=str, default="results_compare_residual_necessity_fair")

    parser.add_argument("--local-eval-samples", type=int, default=128)
    parser.add_argument("--local-eval-std-dt-units", type=float, default=1.0)
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
    print("Training recipe: K=1, line anchors, energy-only, dt-scaled output, data standardization.")
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
    print(f"  - {outdir / 'train_line_max_dist.png'}")
    print(f"  - {outdir / 'delta_at_star.png'}")
    print(f"  - {outdir / 'local_max_dist.png'}")
    print(f"  - {outdir / 'extrap_max_dist.png'}")


if __name__ == "__main__":
    main()
