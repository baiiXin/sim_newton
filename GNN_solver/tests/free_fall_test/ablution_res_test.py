import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import json

# ============================================================
# Full Ablation Study
#
# Dimensions:
#
#   1. Normalization
#   2. Coverage
#   3. Loss Type
#
# Total:
#   2 x 2 x 2 = 8 experiments
#
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

        self.register_buffer(
            "input_mean",
            input_mean.clone().detach()
        )

        self.register_buffer(
            "input_std",
            input_std.clone().detach()
        )

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
# 2. Physics
# ============================================================

def variational_energy(
    y,
    p_n,
    v_n,
    m=1.0,
    g=9.8,
    dt=0.01
):

    residual = y - p_n - dt * v_n

    kinetic = (m / (2 * dt**2)) * torch.sum(residual**2)

    potential = m * g * y[2]

    return kinetic + potential


def energy_residual(
    y,
    p_n,
    v_n,
    m=1.0,
    g=9.8,
    dt=0.01
):

    r = (m / dt**2) * (y - p_n - dt * v_n)

    r[2] += m * g

    return r


def residual_loss(
    y,
    p_n,
    v_n,
    m=1.0,
    g=9.8,
    dt=0.01
):

    r = energy_residual(
        y,
        p_n,
        v_n,
        m,
        g,
        dt
    )

    return torch.sum(r**2)


def newton_direction(
    y,
    p_n,
    v_n,
    m=1.0,
    g=9.8,
    dt=0.01
):

    r = energy_residual(
        y,
        p_n,
        v_n,
        m,
        g,
        dt
    )

    hess_inv = (dt**2) / m

    return -hess_inv * r


# ============================================================
# 3. Dataset
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

    # --------------------------------------------------------
    # line states
    # --------------------------------------------------------

    for alpha in torch.linspace(0.0, 1.0, num_line_points):

        y = (1.0 - alpha) * y0 + alpha * y_star

        train_states.append(y)

    # --------------------------------------------------------
    # local states
    # --------------------------------------------------------

    if num_local_points > 0:

        gen = torch.Generator(device=y0.device)

        gen.manual_seed(seed)

        for _ in range(num_local_points):

            noise = torch.randn(
                3,
                generator=gen
            )

            y = y_star + dt * local_std_dt_units * noise

            train_states.append(y)

    return train_states


def compute_input_normalizer(
    train_states,
    history,
    params
):

    xs = []

    for y in train_states:

        xs.append(
            torch.cat([y, history, params], dim=-1)
        )

    x = torch.stack(xs, dim=0)

    mean = x.mean(dim=0)

    std = x.std(dim=0, unbiased=False)

    std = torch.where(
        std < 1e-8,
        torch.ones_like(std),
        std
    )

    return mean, std


# ============================================================
# 4. Training
# ============================================================

def train_model(
    name,
    use_normalization,
    use_dt_scaling,
    use_coverage,
    loss_type,
    epochs=1000,
    lr=1e-3
):

    torch.manual_seed(42)

    # ========================================================
    # Physics config
    # ========================================================

    m = 1.0
    g = 9.8
    dt = 0.01

    p_n = torch.tensor([3., 4., 5.])

    v_n = torch.tensor([0.5, -0.5, 0.0])

    y0 = p_n.clone()

    history = torch.cat([p_n, v_n])

    params = torch.tensor([m, g, dt])

    y_star = (
        p_n
        + dt * v_n
        - dt**2 * torch.tensor([0., 0., g])
    )

    E_star = variational_energy(
        y_star,
        p_n,
        v_n,
        m,
        g,
        dt
    ).item()

    # ========================================================
    # dataset
    # ========================================================

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

    # ========================================================
    # normalization
    # ========================================================

    if use_normalization:

        input_mean, input_std = compute_input_normalizer(
            train_states,
            history,
            params
        )

    else:

        input_mean = torch.zeros(12)

        input_std = torch.ones(12)

    # ========================================================
    # model
    # ========================================================

    mlp = MLPOptimizer(
        use_normalization=use_normalization,
        use_dt_scaling=use_dt_scaling,
        input_mean=input_mean,
        input_std=input_std
    )

    opt = torch.optim.Adam(
        mlp.parameters(),
        lr=lr
    )

    # ========================================================
    # training
    # ========================================================

    train_curve = []

    K = 1

    for epoch in range(epochs):

        if epoch > 0 and epoch % 200 == 0 and K < 10:
            K += 1

        epoch_loss = 0.0

        for y_init in train_states:

            y = y_init.clone()

            for k in range(K):

                delta = mlp(y, history, params)

                y = y + delta

                # --------------------------------------------
                # loss type
                # --------------------------------------------

                if loss_type == "energy":

                    loss = variational_energy(
                        y,
                        p_n,
                        v_n,
                        m,
                        g,
                        dt
                    )

                elif loss_type == "residual":

                    loss = residual_loss(
                        y,
                        p_n,
                        v_n,
                        m,
                        g,
                        dt
                    )

                else:
                    raise ValueError(loss_type)

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

    rollout_energy_gap = []

    rollout_residual_norm = []

    rollout_delta_norm = []

    rollout_distance_to_star = []

    for step in range(max_steps):

        # ----------------------------------------------------
        # evaluate current state
        # ----------------------------------------------------

        energy = variational_energy(
            y,
            p_n,
            v_n,
            m,
            g,
            dt
        ).item()

        gap = energy - E_star

        r = energy_residual(
            y,
            p_n,
            v_n,
            m,
            g,
            dt
        )

        residual_norm = torch.norm(r).item()

        dist = torch.norm(y - y_star).item()

        # ----------------------------------------------------
        # delta
        # ----------------------------------------------------

        with torch.no_grad():

            delta = mlp(y, history, params)

        delta_norm = torch.norm(delta).item()

        # ----------------------------------------------------
        # record
        # ----------------------------------------------------

        rollout_energy_gap.append(gap)

        rollout_residual_norm.append(residual_norm)

        rollout_delta_norm.append(delta_norm)

        rollout_distance_to_star.append(dist)

        # ----------------------------------------------------
        # update
        # ----------------------------------------------------

        y = y + delta

    # ========================================================
    # fixed-point residual
    # ========================================================

    with torch.no_grad():

        delta_star = mlp(
            y_star,
            history,
            params
        )

    fixed_point_residual = torch.norm(
        delta_star
    ).item()

    # ========================================================
    # vector field
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

                d = mlp(
                    y_probe,
                    history,
                    params
                )

            field_x.append(y_probe[0].item())

            field_z.append(y_probe[2].item())

            field_u.append(d[0].item())

            field_v.append(d[2].item())

    # ========================================================
    # return
    # ========================================================

    result = {

        "name": name,

        "loss_type": loss_type,

        "use_normalization": use_normalization,

        "use_coverage": use_coverage,

        "fixed_point_residual":
            fixed_point_residual,

        "rollout_energy_gap":
            rollout_energy_gap,

        "rollout_residual_norm":
            rollout_residual_norm,

        "rollout_delta_norm":
            rollout_delta_norm,

        "rollout_distance_to_star":
            rollout_distance_to_star,

        "field_x":
            np.array(field_x),

        "field_z":
            np.array(field_z),

        "field_u":
            np.array(field_u),

        "field_v":
            np.array(field_v),

        "y_star":
            y_star
    }

    return result


# ============================================================
# 5. Main
# ============================================================

def main():

    configs = [

        # ====================================================
        # BASE
        # ====================================================

        {
            "name": "A1_Base_Energy",
            "use_normalization": False,
            "use_dt_scaling": False,
            "use_coverage": False,
            "loss_type": "energy"
        },

        {
            "name": "A2_Base_Residual",
            "use_normalization": False,
            "use_dt_scaling": False,
            "use_coverage": False,
            "loss_type": "residual"
        },

        # ====================================================
        # NORMALIZATION ONLY
        # ====================================================

        {
            "name": "B1_Norm_Energy",
            "use_normalization": True,
            "use_dt_scaling": True,
            "use_coverage": False,
            "loss_type": "energy"
        },

        {
            "name": "B2_Norm_Residual",
            "use_normalization": True,
            "use_dt_scaling": True,
            "use_coverage": False,
            "loss_type": "residual"
        },

        # ====================================================
        # COVERAGE ONLY
        # ====================================================

        {
            "name": "C1_Coverage_Energy",
            "use_normalization": False,
            "use_dt_scaling": False,
            "use_coverage": True,
            "loss_type": "energy"
        },

        {
            "name": "C2_Coverage_Residual",
            "use_normalization": False,
            "use_dt_scaling": False,
            "use_coverage": True,
            "loss_type": "residual"
        },

        # ====================================================
        # FULL
        # ====================================================

        {
            "name": "D1_Full_Energy",
            "use_normalization": True,
            "use_dt_scaling": True,
            "use_coverage": True,
            "loss_type": "energy"
        },

        {
            "name": "D2_Full_Residual",
            "use_normalization": True,
            "use_dt_scaling": True,
            "use_coverage": True,
            "loss_type": "residual"
        },
    ]

    results = []

    # ========================================================
    # Run
    # ========================================================

    for cfg in configs:

        print("=" * 70)

        print("Training:", cfg["name"])

        res = train_model(**cfg)

        results.append(res)

        print(
            f"Fixed-point residual: "
            f"{res['fixed_point_residual']:.6e}"
        )

        print(
            f"Final energy gap: "
            f"{res['rollout_energy_gap'][-1]:.6e}"
        )

        print(
            f"Final residual norm: "
            f"{res['rollout_residual_norm'][-1]:.6e}"
        )

    # ========================================================
    # Visualization
    # ========================================================

    fig = plt.figure(figsize=(22, 18))

    # ========================================================
    # 1. Energy Gap
    # ========================================================

    ax1 = plt.subplot(3, 2, 1)

    for res in results:

        curve = np.maximum(
            np.array(res["rollout_energy_gap"]),
            1e-12
        )

        ax1.plot(
            curve,
            marker='o',
            label=res["name"]
        )

    ax1.set_yscale('log')

    ax1.set_title("Energy Gap")

    ax1.set_xlabel("Iteration")

    ax1.set_ylabel("E - E*")

    ax1.grid(True, alpha=0.3)

    ax1.legend(fontsize=8)

    # ========================================================
    # 2. Residual Norm
    # ========================================================

    ax2 = plt.subplot(3, 2, 2)

    for res in results:

        curve = np.maximum(
            np.array(res["rollout_residual_norm"]),
            1e-12
        )

        ax2.plot(
            curve,
            marker='o',
            label=res["name"]
        )

    ax2.set_yscale('log')

    ax2.set_title("Residual Norm")

    ax2.set_xlabel("Iteration")

    ax2.set_ylabel("||r(y)||")

    ax2.grid(True, alpha=0.3)

    ax2.legend(fontsize=8)

    # ========================================================
    # 3. Delta Norm
    # ========================================================

    ax3 = plt.subplot(3, 2, 3)

    for res in results:

        curve = np.maximum(
            np.array(res["rollout_delta_norm"]),
            1e-12
        )

        ax3.plot(
            curve,
            marker='o',
            label=res["name"]
        )

    ax3.set_yscale('log')

    ax3.set_title("Update Magnitude")

    ax3.set_xlabel("Iteration")

    ax3.set_ylabel("||delta||")

    ax3.grid(True, alpha=0.3)

    ax3.legend(fontsize=8)

    # ========================================================
    # 4. Distance to y*
    # ========================================================

    ax4 = plt.subplot(3, 2, 4)

    for res in results:

        curve = np.maximum(
            np.array(res["rollout_distance_to_star"]),
            1e-12
        )

        ax4.plot(
            curve,
            marker='o',
            label=res["name"]
        )

    ax4.set_yscale('log')

    ax4.set_title("Distance to Optimum")

    ax4.set_xlabel("Iteration")

    ax4.set_ylabel("||y - y*||")

    ax4.grid(True, alpha=0.3)

    ax4.legend(fontsize=8)

    # ========================================================
    # 5. Fixed-point residual
    # ========================================================

    ax5 = plt.subplot(3, 2, 5)

    names = [r["name"] for r in results]

    vals = [
        max(r["fixed_point_residual"], 1e-12)
        for r in results
    ]

    ax5.bar(names, vals)

    ax5.set_yscale('log')

    ax5.set_title("Fixed-point Residual")

    ax5.set_ylabel("||delta(y*)||")

    ax5.tick_params(axis='x', rotation=45)

    # ========================================================
    # 6. Vector field
    # ========================================================

    ax6 = plt.subplot(3, 2, 6)

    baseline = results[0]

    full = results[-1]

    stride = 8

    ax6.quiver(
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

    ax6.quiver(
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

    ax6.scatter(
        full["y_star"][0].item(),
        full["y_star"][2].item(),

        s=120,
        marker='*',

        label='y*'
    )

    ax6.set_title("Vector Field Near Optimum")

    ax6.set_xlabel("x")

    ax6.set_ylabel("z")

    ax6.grid(True, alpha=0.3)

    ax6.legend()

    plt.tight_layout()

    plt.savefig(
        "full_ablation_study.png",
        dpi=300,
        bbox_inches='tight'
    )

    print("\nSaved figure: full_ablation_study.png")

    # ========================================================
    # JSON report
    # ========================================================

    report = {}

    for res in results:

        report[res["name"]] = {

            "loss_type":
                res["loss_type"],

            "use_normalization":
                res["use_normalization"],

            "use_coverage":
                res["use_coverage"],

            "fixed_point_residual":
                res["fixed_point_residual"],

            "final_energy_gap":
                res["rollout_energy_gap"][-1],

            "final_residual_norm":
                res["rollout_residual_norm"][-1],

            "energy_gap_curve":
                res["rollout_energy_gap"],

            "residual_norm_curve":
                res["rollout_residual_norm"],

            "delta_norm_curve":
                res["rollout_delta_norm"],

            "distance_to_star_curve":
                res["rollout_distance_to_star"]
        }

    with open("full_ablation_report.json", "w") as f:

        json.dump(report, f, indent=2)

    print("Saved report: full_ablation_report.json")

    print("\nDone.")


if __name__ == "__main__":
    main()