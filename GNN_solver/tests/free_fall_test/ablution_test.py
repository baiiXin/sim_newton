import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from copy import deepcopy

# ============================================================
# Ablation Study:
#   A: Baseline
#   B: Normalization Only
#   C: Coverage Only
#   D: Full
#
# 核心目标：
#   研究 fixed-point stability 到底来源于：
#       1. normalization
#       2. training-state coverage
#
# 输出：
#   1. rollout convergence
#   2. fixed-point residual
#   3. vector field near optimum
#   4. update magnitude
# ============================================================


# ============================================================
# 1. MLP
# ============================================================

class MLPOptimizer(nn.Module):

    def __init__(
        self,
        use_normalization=False,
        use_dt_scaling=False,
        input_mean=None,
        input_std=None
    ):
        super().__init__()

        self.use_normalization = use_normalization
        self.use_dt_scaling = use_dt_scaling

        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

        if input_mean is None:
            input_mean = torch.zeros(12)

        if input_std is None:
            input_std = torch.ones(12)

        self.register_buffer("input_mean", input_mean.clone().detach())
        self.register_buffer("input_std", input_std.clone().detach())

    def forward(self, y, history, params):

        inp = torch.cat([y, history, params], dim=-1)

        if self.use_normalization:
            inp = (inp - self.input_mean) / self.input_std

        delta = self.net(inp)

        if self.use_dt_scaling:
            dt = params[2]
            delta = dt * delta

        return delta


# ============================================================
# 2. Physics / Energy
# ============================================================

def variational_energy(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):

    residual = y - p_n - dt * v_n

    kinetic = (m / (2 * dt**2)) * torch.sum(residual**2)
    potential = m * g * y[2]

    return kinetic + potential


def newton_direction(y, p_n, v_n, m=1.0, g=9.8, dt=0.01):

    residual = y - p_n - dt * v_n

    grad = (m / dt**2) * residual
    grad[2] += m * g

    hess_inv = (dt**2) / m

    return -grad * hess_inv


# ============================================================
# 3. Dataset Construction
# ============================================================

def make_training_states(
    y0,
    y_star,
    dt,
    num_line_points=11,
    num_local_points=32,
    local_std_dt_units=1.0,
    seed=123
):

    train_states = []

    # line states
    for alpha in torch.linspace(0.0, 1.0, num_line_points):

        y = (1.0 - alpha) * y0 + alpha * y_star
        train_states.append(y)

    # local states
    if num_local_points > 0:

        gen = torch.Generator(device=y0.device)
        gen.manual_seed(seed)

        for _ in range(num_local_points):

            noise = torch.randn(3, generator=gen)

            y = y_star + dt * local_std_dt_units * noise

            train_states.append(y)

    return train_states


def compute_input_normalizer(train_states, history, params):

    xs = []

    for y in train_states:
        xs.append(torch.cat([y, history, params], dim=-1))

    x = torch.stack(xs, dim=0)

    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)

    std = torch.where(std < 1e-8, torch.ones_like(std), std)

    return mean, std


# ============================================================
# 4. Train
# ============================================================

def train_model(
    name,
    use_normalization,
    use_dt_scaling,
    use_coverage,
    epochs=1000,
    lr=1e-3,
):

    torch.manual_seed(42)

    # -------------------------
    # Physics config
    # -------------------------

    m, g, dt = 1.0, 9.8, 0.01

    p_n = torch.tensor([3., 4., 5.])
    v_n = torch.tensor([0.5, -0.5, 0.0])

    y0 = p_n.clone()

    history = torch.cat([p_n, v_n])
    params = torch.tensor([m, g, dt])

    y_star = p_n + dt * v_n - dt**2 * torch.tensor([0., 0., g])

    E_star = variational_energy(
        y_star,
        p_n,
        v_n,
        m,
        g,
        dt
    ).item()

    # -------------------------
    # dataset
    # -------------------------

    if use_coverage:

        train_states = make_training_states(
            y0,
            y_star,
            dt,
            num_line_points=11,
            num_local_points=32,
            local_std_dt_units=1.0
        )

    else:
        train_states = [y0]

    # -------------------------
    # normalization
    # -------------------------

    if use_normalization:

        input_mean, input_std = compute_input_normalizer(
            train_states,
            history,
            params
        )

    else:

        input_mean = torch.zeros(12)
        input_std = torch.ones(12)

    # -------------------------
    # model
    # -------------------------

    mlp = MLPOptimizer(
        use_normalization=use_normalization,
        use_dt_scaling=use_dt_scaling,
        input_mean=input_mean,
        input_std=input_std
    )

    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    # ========================================================
    # training
    # ========================================================

    train_curve = []

    K = 1

    for epoch in range(epochs):

        if epoch > 0 and epoch % 100 == 0 and K < 10:
            K += 1

        epoch_loss = 0.0

        for y_init in train_states:

            y = y_init.clone()

            for k in range(K):

                delta = mlp(y, history, params)

                y = y + delta

                loss = variational_energy(
                    y,
                    p_n,
                    v_n,
                    m,
                    g,
                    dt
                )

                loss.backward()

                opt.step()

                opt.zero_grad()

                y = y.detach()

                epoch_loss += loss.item()

        train_curve.append(epoch_loss)

    # ========================================================
    # rollout evaluation
    # ========================================================

    max_steps = 20

    y = y0.clone()

    rollout_y = []
    rollout_loss = []
    rollout_gap = []
    rollout_delta_norm = []

    for step in range(max_steps):

        loss = variational_energy(
            y,
            p_n,
            v_n,
            m,
            g,
            dt
        ).item()

        rollout_y.append(y.clone())
        rollout_loss.append(loss)
        rollout_gap.append(loss - E_star)
        rollout_delta_norm.append(torch.norm(delta).item())

        with torch.no_grad():

            delta = mlp(y, history, params)

        y = y + delta

    # ========================================================
    # fixed-point residual
    # ========================================================

    with torch.no_grad():

        delta_star = mlp(y_star, history, params)

    fixed_point_residual = torch.norm(delta_star).item()

    # ========================================================
    # vector field around optimum
    # ========================================================

    field_x = []
    field_z = []

    field_u = []
    field_v = []

    radius = 0.03

    xs = np.linspace(-radius, radius, 21)
    zs = np.linspace(-radius, radius, 21)

    for dx in xs:
        for dz in zs:

            y_probe = y_star.clone()

            y_probe[0] += dx
            y_probe[2] += dz

            with torch.no_grad():

                d = mlp(y_probe, history, params)

            field_x.append(y_probe[0].item())
            field_z.append(y_probe[2].item())

            field_u.append(d[0].item())
            field_v.append(d[2].item())

    result = {
        "name": name,
        "model": mlp,
        "train_curve": train_curve,
        "rollout_gap": rollout_gap,
        "rollout_delta_norm": rollout_delta_norm,
        "fixed_point_residual": fixed_point_residual,
        "field_x": np.array(field_x),
        "field_z": np.array(field_z),
        "field_u": np.array(field_u),
        "field_v": np.array(field_v),
        "y_star": y_star,
        "E_star": E_star
    }

    return result


# ============================================================
# 5. Main
# ============================================================

def main():

    configs = [

        {
            "name": "A_Baseline",
            "use_normalization": False,
            "use_dt_scaling": False,
            "use_coverage": False
        },

        {
            "name": "B_NormalizationOnly",
            "use_normalization": True,
            "use_dt_scaling": True,
            "use_coverage": False
        },

        {
            "name": "C_CoverageOnly",
            "use_normalization": False,
            "use_dt_scaling": False,
            "use_coverage": True
        },

        {
            "name": "D_Full",
            "use_normalization": True,
            "use_dt_scaling": True,
            "use_coverage": True
        },
    ]

    results = []

    for cfg in configs:

        print("=" * 60)
        print("Training:", cfg["name"])

        res = train_model(**cfg)

        results.append(res)

        print(
            f"Fixed-point residual: "
            f"{res['fixed_point_residual']:.6e}"
        )

        print(
            f"Final rollout gap: "
            f"{res['rollout_gap'][-1]:.6e}"
        )

    # ========================================================
    # Visualization
    # ========================================================

    fig = plt.figure(figsize=(18, 14))

    # --------------------------------------------------------
    # 1. rollout gap
    # --------------------------------------------------------

    ax1 = plt.subplot(2, 2, 1)

    for res in results:

        gap = np.maximum(
            np.array(res["rollout_gap"]),
            1e-12
        )

        ax1.plot(
            gap,
            marker='o',
            label=res["name"]
        )

    ax1.set_yscale('log')
    ax1.set_title("Rollout Gap")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("E - E*")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --------------------------------------------------------
    # 2. delta norm
    # --------------------------------------------------------

    ax2 = plt.subplot(2, 2, 2)

    for res in results:

        dn = np.maximum(
            np.array(res["rollout_delta_norm"]),
            1e-12
        )

        ax2.plot(
            dn,
            marker='o',
            label=res["name"]
        )

    ax2.set_yscale('log')
    ax2.set_title("Update Magnitude")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("||delta||")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --------------------------------------------------------
    # 3. fixed-point residual bar
    # --------------------------------------------------------

    ax3 = plt.subplot(2, 2, 3)

    names = [r["name"] for r in results]

    residuals = [
        max(r["fixed_point_residual"], 1e-12)
        for r in results
    ]

    ax3.bar(names, residuals)

    ax3.set_yscale('log')
    ax3.set_title("Fixed-point Residual")
    ax3.set_ylabel("||delta(y*)||")

    # --------------------------------------------------------
    # 4. vector field
    # --------------------------------------------------------

    ax4 = plt.subplot(2, 2, 4)

    # 只画 baseline 和 full
    baseline = results[0]
    full = results[-1]

    stride = 8

    ax4.quiver(
        baseline["field_x"][::stride],
        baseline["field_z"][::stride],
        baseline["field_u"][::stride],
        baseline["field_v"][::stride],
        angles='xy',
        scale_units='xy',
        scale=1.0,
        alpha=0.7,
        label='Baseline'
    )

    ax4.quiver(
        full["field_x"][::stride],
        full["field_z"][::stride],
        full["field_u"][::stride],
        full["field_v"][::stride],
        angles='xy',
        scale_units='xy',
        scale=1.0,
        alpha=0.7,
        label='Full'
    )

    ax4.scatter(
        full["y_star"][0].item(),
        full["y_star"][2].item(),
        s=120,
        marker='*',
        label='y*'
    )

    ax4.set_title("Vector Field Near Optimum")
    ax4.set_xlabel("x")
    ax4.set_ylabel("z")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    plt.savefig(
        "ablation_study.png",
        dpi=300,
        bbox_inches='tight'
    )

    print("\nSaved figure: ablation_study.png")

    # ========================================================
    # JSON report
    # ========================================================

    report = {}

    for res in results:

        report[res["name"]] = {

            "fixed_point_residual":
                res["fixed_point_residual"],

            "final_rollout_gap":
                res["rollout_gap"][-1],

            "rollout_gap":
                res["rollout_gap"],

            "rollout_delta_norm":
                res["rollout_delta_norm"]
        }

    with open("ablation_report.json", "w") as f:

        json.dump(report, f, indent=2)

    print("Saved report: ablation_report.json")

    print("\nDone.")


if __name__ == "__main__":
    main()