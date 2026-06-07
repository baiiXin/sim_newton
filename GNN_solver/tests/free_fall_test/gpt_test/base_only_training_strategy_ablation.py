#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base-only training-strategy ablation for the free-fall implicit-Euler learned optimizer.

Question:
    Without adding residual/gradient information to the input, can training strategy alone make
    the network reach the same overfit quality as the residual-input model?

Strict rule:
    The network input never contains residual, gradient, y-y*, or equivalent residual features.
    Input is always only:
        y, p_n, v_n, m, g, dt

What changes across strategies:
    1. Feature normalization.
    2. Output parameterization: direct delta_y vs delta_y = dt * raw_delta.
    3. Training states: only y0 vs anchors along y0 -> y*, including y*.
    4. Loss: energy gap only vs energy gap + supervised Newton-step loss.

Usage:
    python base_only_training_strategy_ablation.py

Recommended:
    python base_only_training_strategy_ablation.py --epochs 5000 --lr 1e-3
    python base_only_training_strategy_ablation.py --epochs 8000 --lr 1e-3 --dtype float64
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

# Small CPU MLPs are usually faster and more predictable with one thread.
torch.set_num_threads(1)


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


def exact_quadratic_gap_batch(y: torch.Tensor, y_star: torch.Tensor, m: float, dt: float) -> torch.Tensor:
    return (m / (2.0 * dt ** 2)) * torch.sum((y - y_star) ** 2, dim=-1)


class BaseOnlyMLP(nn.Module):
    """Base-only input: [y, p_n, v_n, m, g, dt]. No residual input."""
    def __init__(self, normalization_mode: str, output_mode: str, hidden_dim: int, depth: int, activation: str):
        super().__init__()
        assert normalization_mode in {"raw", "fixed", "dataset"}
        assert output_mode in {"direct_delta", "dt_scaled_raw_delta"}
        assert activation in {"relu", "tanh", "silu"}
        self.normalization_mode = normalization_mode
        self.output_mode = output_mode

        self.register_buffer("feature_mean", torch.zeros(12))
        self.register_buffer("feature_std", torch.ones(12))

        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}[activation]
        layers = []
        for i in range(depth):
            layers += [nn.Linear(12 if i == 0 else hidden_dim, hidden_dim), act_cls()]
        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)

        # Start from zero update.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def raw_features(self, y: torch.Tensor, p_n: torch.Tensor, v_n: torch.Tensor, m: float, g: float, dt: float):
        n = y.shape[0]
        p = p_n.view(1, 3).expand(n, 3)
        v = v_n.view(1, 3).expand(n, 3)
        params = torch.tensor([m, g, dt], device=y.device, dtype=y.dtype).view(1, 3).expand(n, 3)
        return torch.cat([y, p, v, params], dim=-1)

    def fit_dataset_normalizer(self, states: torch.Tensor, p_n: torch.Tensor, v_n: torch.Tensor, m: float, g: float, dt: float):
        """Fit mean/std over base-only raw features. This is preprocessing, not residual input."""
        with torch.no_grad():
            feat = self.raw_features(states, p_n, v_n, m, g, dt)
            mean = feat.mean(dim=0)
            std = feat.std(dim=0, unbiased=False)
            std = torch.clamp(std, min=1e-6)
            self.feature_mean.copy_(mean)
            self.feature_std.copy_(std)

    def make_features(self, y: torch.Tensor, p_n: torch.Tensor, v_n: torch.Tensor, m: float, g: float, dt: float):
        raw = self.raw_features(y, p_n, v_n, m, g, dt)
        if self.normalization_mode == "raw":
            return raw
        if self.normalization_mode == "dataset":
            return (raw - self.feature_mean.view(1, -1)) / self.feature_std.view(1, -1)

        # Fixed normalization. Still base-only; no residual feature is introduced.
        scale = torch.tensor(
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 10.0, 0.01],
            device=y.device,
            dtype=y.dtype,
        ).view(1, 12)
        return raw / scale

    def forward(self, y: torch.Tensor, p_n: torch.Tensor, v_n: torch.Tensor, m: float, g: float, dt: float):
        raw = self.net(self.make_features(y, p_n, v_n, m, g, dt))
        if self.output_mode == "direct_delta":
            return raw
        return dt * raw

@dataclass
class Strategy:
    name: str
    normalization_mode: str  # raw, fixed, dataset
    output_mode: str
    train_state_mode: str
    loss_mode: str
    lambda_delta: float = 1.0
    hidden_dim: int = 128
    depth: int = 3
    activation: str = "tanh"


def build_strategies() -> List[Strategy]:
    return [
        Strategy("A_raw_direct_energy_y0_only", "raw", "direct_delta", "y0_only", "energy_only", hidden_dim=64, depth=2, activation="relu"),
        Strategy("B_fixednorm_dt_energy_y0_only", "fixed", "dt_scaled_raw_delta", "y0_only", "energy_only"),
        Strategy("C_fixednorm_dt_energy_line", "fixed", "dt_scaled_raw_delta", "line_anchors", "energy_only"),
        Strategy("D_fixednorm_dt_energy_line_local", "fixed", "dt_scaled_raw_delta", "line_plus_local", "energy_only"),
        Strategy("E_fixednorm_energy_plus_delta_mse_line", "fixed", "dt_scaled_raw_delta", "line_anchors", "energy_plus_delta_mse", lambda_delta=1.0),
        Strategy("F_fixednorm_delta_mse_strong_line_local", "fixed", "dt_scaled_raw_delta", "line_plus_local", "energy_plus_delta_mse", lambda_delta=10.0),
        Strategy("G_datastd_dt_energy_line", "dataset", "dt_scaled_raw_delta", "line_anchors", "energy_only"),
        Strategy("H_datastd_energy_plus_delta_mse_line", "dataset", "dt_scaled_raw_delta", "line_anchors", "energy_plus_delta_mse", lambda_delta=1.0),
        Strategy("I_datastd_delta_mse_strong_line_local", "dataset", "dt_scaled_raw_delta", "line_plus_local", "energy_plus_delta_mse", lambda_delta=10.0),
    ]


def make_train_states(strategy: Strategy, p_n: torch.Tensor, y_star: torch.Tensor, dt: float, args) -> torch.Tensor:
    if strategy.train_state_mode == "y0_only":
        return p_n.view(1, 3).clone()

    alphas = torch.linspace(0.0, 1.0, args.num_line_points, device=p_n.device, dtype=p_n.dtype).view(-1, 1)
    states = (1.0 - alphas) * p_n.view(1, 3) + alphas * y_star.view(1, 3)

    if strategy.train_state_mode == "line_plus_local":
        gen = torch.Generator(device=p_n.device)
        gen.manual_seed(args.seed)
        noise = torch.randn(args.num_local_perturbs, 3, generator=gen, device=p_n.device, dtype=p_n.dtype)
        local = y_star.view(1, 3) + dt * args.local_perturb_std_dt_units * noise
        states = torch.cat([states, local], dim=0)

    return states


def train_strategy(strategy: Strategy, case: MotionCase, args, device, dtype) -> Dict:
    torch.manual_seed(args.seed)
    p_n, v_n, m, g, dt = case.tensors(device, dtype)
    y_star = y_star_for_case(p_n, v_n, g, dt)
    states = make_train_states(strategy, p_n, y_star, dt, args)

    model = BaseOnlyMLP(strategy.normalization_mode, strategy.output_mode, strategy.hidden_dim, strategy.depth, strategy.activation).to(device=device, dtype=dtype)
    if strategy.normalization_mode == "dataset":
        model.fit_dataset_normalizer(states, p_n, v_n, m, g, dt)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)

    log = []
    target_delta = y_star.view(1, 3) - states
    for epoch in range(args.epochs):
        delta = model(states, p_n, v_n, m, g, dt)
        y_next = states + delta
        gap = exact_quadratic_gap_batch(y_next, y_star.view(1, 3), m, dt)

        if strategy.loss_mode == "energy_only":
            loss = gap.mean()
        else:
            delta_mse_scaled = torch.mean(((delta - target_delta) / dt) ** 2)
            loss = gap.mean() + strategy.lambda_delta * delta_mse_scaled

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                delta_err = torch.norm(delta - target_delta, dim=-1)
                dist_after = torch.norm(y_next - y_star.view(1, 3), dim=-1)
                log.append({
                    "epoch": epoch,
                    "loss": float(loss.cpu()),
                    "mean_gap": float(gap.mean().cpu()),
                    "max_dist_after": float(dist_after.max().cpu()),
                    "mean_delta_error": float(delta_err.mean().cpu()),
                    "lr": float(sched.get_last_lr()[0]),
                })

    metrics = evaluate_model(model, case, states, device, dtype)
    return {
        "strategy": asdict(strategy),
        "case": asdict(case),
        "num_train_states": int(states.shape[0]),
        "training_log": log,
        "metrics": metrics,
    }


@torch.no_grad()
def evaluate_model(model: BaseOnlyMLP, case: MotionCase, states: torch.Tensor, device, dtype) -> Dict:
    p_n, v_n, m, g, dt = case.tensors(device, dtype)
    y_star = y_star_for_case(p_n, v_n, g, dt)

    def eval_batch(y: torch.Tensor) -> Dict:
        delta = model(y, p_n, v_n, m, g, dt)
        y_next = y + delta
        target = y_star.view(1, 3) - y
        return {
            "delta_norm": torch.norm(delta, dim=-1),
            "delta_newton_error": torch.norm(delta - target, dim=-1),
            "after_step_dist_to_star": torch.norm(y_next - y_star.view(1, 3), dim=-1),
            "after_step_gap": exact_quadratic_gap_batch(y_next, y_star.view(1, 3), m, dt),
        }

    init = eval_batch(p_n.view(1, 3))
    star = eval_batch(y_star.view(1, 3))
    train = eval_batch(states)

    # Rollout from initial.
    rollout = []
    y = p_n.view(1, 3).clone()
    for step in range(10):
        row = eval_batch(y)
        rollout.append({
            "step": step,
            "after_step_gap": float(row["after_step_gap"][0].cpu()),
            "after_step_dist_to_star": float(row["after_step_dist_to_star"][0].cpu()),
            "delta_newton_error": float(row["delta_newton_error"][0].cpu()),
        })
        y = y + model(y, p_n, v_n, m, g, dt)

    threshold = 1e-4
    success = (
        float(init["after_step_dist_to_star"][0].cpu()) < threshold
        and float(init["after_step_gap"][0].cpu()) < threshold
        and float(star["delta_norm"][0].cpu()) < threshold
        and float(train["after_step_dist_to_star"].max().cpu()) < threshold
    )

    return {
        "y_star": y_star.cpu().tolist(),
        "initial_eval": {
            "after_step_gap": float(init["after_step_gap"][0].cpu()),
            "after_step_dist_to_star": float(init["after_step_dist_to_star"][0].cpu()),
            "delta_newton_error": float(init["delta_newton_error"][0].cpu()),
        },
        "star_eval": {
            "delta_norm": float(star["delta_norm"][0].cpu()),
        },
        "max_train_after_step_gap": float(train["after_step_gap"].max().cpu()),
        "max_train_after_step_dist_to_star": float(train["after_step_dist_to_star"].max().cpu()),
        "max_train_delta_newton_error": float(train["delta_newton_error"].max().cpu()),
        "rollout_from_initial": rollout,
        "success_threshold": threshold,
        "success": bool(success),
    }


def save_summary_csv(outdir: Path, reports: List[Dict]) -> None:
    rows = []
    for r in reports:
        s = r["strategy"]
        m = r["metrics"]
        rows.append({
            "strategy": s["name"],
            "success": m["success"],
            "normalization_mode": s["normalization_mode"],
            "output_mode": s["output_mode"],
            "train_state_mode": s["train_state_mode"],
            "loss_mode": s["loss_mode"],
            "lambda_delta": s["lambda_delta"],
            "num_train_states": r["num_train_states"],
            "initial_after_step_gap": m["initial_eval"]["after_step_gap"],
            "initial_after_step_dist_to_star": m["initial_eval"]["after_step_dist_to_star"],
            "initial_delta_newton_error": m["initial_eval"]["delta_newton_error"],
            "delta_at_star_norm": m["star_eval"]["delta_norm"],
            "max_train_after_step_gap": m["max_train_after_step_gap"],
            "max_train_after_step_dist_to_star": m["max_train_after_step_dist_to_star"],
            "max_train_delta_newton_error": m["max_train_delta_newton_error"],
        })
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training(outdir: Path, reports: List[Dict]) -> None:
    plt.figure(figsize=(11, 6))
    for r in reports:
        xs = [e["epoch"] for e in r["training_log"]]
        ys = [max(e["mean_gap"], 1e-16) for e in r["training_log"]]
        plt.plot(xs, ys, label=r["strategy"]["name"])
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean train-state gap after one step")
    plt.title("Base-only training-strategy ablation")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=220, bbox_inches="tight")
    plt.close()


def print_summary(reports: List[Dict]) -> None:
    print("\n========== BASE-ONLY STRATEGY SUMMARY ==========")
    print(f"{'strategy':>40s} | {'success':>7s} | {'init_gap':>11s} | {'init_dist':>11s} | {'delta@star':>11s} | {'max_train_dist':>14s}")
    print("-" * 112)
    for r in reports:
        m = r["metrics"]
        print(f"{r['strategy']['name']:>40s} | {str(m['success']):>7s} | {m['initial_eval']['after_step_gap']:>11.4e} | {m['initial_eval']['after_step_dist_to_star']:>11.4e} | {m['star_eval']['delta_norm']:>11.4e} | {m['max_train_after_step_dist_to_star']:>14.4e}")
    print("================================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-line-points", type=int, default=11)
    parser.add_argument("--num-local-perturbs", type=int, default=20)
    parser.add_argument("--local-perturb-std-dt-units", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--outdir", default="results_base_only_strategy_ablation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    case = MotionCase("single_motion_default", (3.0, 4.0, 5.0), (0.5, -0.5, 0.0), 1.0, 9.8, 0.01)
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Case: {case}")
    print("Strict rule: model input is base-only [y, p_n, v_n, m, g, dt]. No residual input.\n")

    reports = []
    for strategy in build_strategies():
        print(f"[RUN] {strategy.name}")
        reports.append(train_strategy(strategy, case, args, device, dtype))

    with (outdir / "full_report.json").open("w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "device": str(device),
            "dtype": str(dtype),
            "strict_rule": "model input is always base-only [y, p_n, v_n, m, g, dt]",
            "reports": reports,
        }, f, indent=2)
    save_summary_csv(outdir, reports)
    plot_training(outdir, reports)
    print_summary(reports)
    print(f"[DONE] Saved to: {outdir.resolve()}")
    print(f"  - {outdir / 'summary.csv'}")
    print(f"  - {outdir / 'full_report.json'}")
    print(f"  - {outdir / 'training_curves.png'}")

if __name__ == "__main__":
    main()
