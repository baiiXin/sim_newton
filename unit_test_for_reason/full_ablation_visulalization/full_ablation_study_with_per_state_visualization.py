import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 适配无显示器 Linux 环境

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.lines import Line2D


# ============================================================
# Full Ablation Study with Detailed Training Visualization
#
# Dimensions:
#   1. Normalization (+ dt scaling in the original design)
#   2. Coverage
#   3. Loss Type
#
# Total:
#   2 x 2 x 2 = 8 experiments
#
# Extra outputs for each MLP:
#   1. Overall training-input / result-point distribution plot
#   2. Overall training-loss plot
#   3. One additional pair of plots for each training initial state
#   4. Compressed detailed numerical logs (.npz)
# ============================================================


# ============================================================
# 0. Global config
# ============================================================

OUTPUT_DIR = Path("ablation_outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"
LOGS_DIR = OUTPUT_DIR / "logs"

COLOR_BUCKET_SIZE = 200
PLOT_RELATIVE_COORDINATES = True


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_model_plot_dir(name):
    model_dir = PLOTS_DIR / name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


# ============================================================
# 1. MLP
# ============================================================

class MLPOptimizer(nn.Module):

    def __init__(
        self,
        use_normalization=False,
        use_dt_scaling=False,
        input_mean=None,
        input_std=None,
    ):
        super().__init__()

        self.use_normalization = use_normalization
        self.use_dt_scaling = use_dt_scaling

        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        if input_mean is None:
            input_mean = torch.zeros(12)

        if input_std is None:
            input_std = torch.ones(12)

        self.register_buffer(
            "input_mean",
            input_mean.clone().detach(),
        )

        self.register_buffer(
            "input_std",
            input_std.clone().detach(),
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
    dt=0.01,
):
    residual = y - p_n - dt * v_n

    kinetic = (m / (2.0 * dt**2)) * torch.sum(residual**2)
    potential = m * g * y[2]

    return kinetic + potential


def energy_residual(
    y,
    p_n,
    v_n,
    m=1.0,
    g=9.8,
    dt=0.01,
):
    r = (m / dt**2) * (y - p_n - dt * v_n)
    r = r.clone()
    r[2] += m * g

    return r


def residual_loss(
    y,
    p_n,
    v_n,
    m=1.0,
    g=9.8,
    dt=0.01,
):
    r = energy_residual(
        y,
        p_n,
        v_n,
        m,
        g,
        dt,
    )

    return torch.sum(r**2)


def newton_direction(
    y,
    p_n,
    v_n,
    m=1.0,
    g=9.8,
    dt=0.01,
):
    r = energy_residual(
        y,
        p_n,
        v_n,
        m,
        g,
        dt,
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
    seed=123,
):
    train_states = []

    # --------------------------------------------------------
    # line states: y0 -> y_star
    # --------------------------------------------------------

    for alpha in torch.linspace(0.0, 1.0, num_line_points):
        y = (1.0 - alpha) * y0 + alpha * y_star
        train_states.append(y)

    # --------------------------------------------------------
    # local states around y_star
    # --------------------------------------------------------

    if num_local_points > 0:
        gen = torch.Generator(device=y0.device)
        gen.manual_seed(seed)

        for _ in range(num_local_points):
            noise = torch.randn(
                3,
                generator=gen,
            )

            y = y_star + dt * local_std_dt_units * noise
            train_states.append(y)

    return train_states


def compute_input_normalizer(
    train_states,
    history,
    params,
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
        std,
    )

    return mean, std


# ============================================================
# 4. Metric helpers
# ============================================================

def objective_for_plot(
    loss_type,
    y,
    p_n,
    v_n,
    E_star,
    m,
    g,
    dt,
):
    """
    Return the objective used for visualization.

    energy model:
        E(y) - E*          (same gradient as E(y), easier to interpret)

    residual model:
        ||r(y)||^2         (exact residual training objective)
    """

    if loss_type == "energy":
        value = variational_energy(
            y,
            p_n,
            v_n,
            m,
            g,
            dt,
        ).item() - E_star

        # Avoid tiny negative values caused by floating-point roundoff.
        return max(float(value), 0.0)

    if loss_type == "residual":
        return float(
            residual_loss(
                y,
                p_n,
                v_n,
                m,
                g,
                dt,
            ).item()
        )

    raise ValueError(f"Unknown loss type: {loss_type}")


def frozen_terminal_metrics(
    mlp,
    train_states,
    K,
    history,
    params,
    p_n,
    v_n,
    y_star,
    E_star,
    loss_type,
    m,
    g,
    dt,
):
    """
    Evaluate the current network after an epoch without changing parameters.

    For each training initial state, run exactly K network updates with frozen
    parameters and evaluate the terminal point x^(K).

    Returns both overall aggregate metrics and one metric dictionary per
    initial state.
    """

    terminal_objectives = []
    terminal_energy_gaps = []
    terminal_residual_norms = []
    terminal_distances = []
    per_state_metrics = []

    with torch.no_grad():
        for init_state_id, y_init in enumerate(train_states):
            y = y_init.clone()

            for _ in range(K):
                y = y + mlp(y, history, params)

            energy_gap = (
                variational_energy(
                    y,
                    p_n,
                    v_n,
                    m,
                    g,
                    dt,
                ).item()
                - E_star
            )

            energy_gap = max(float(energy_gap), 0.0)

            r = energy_residual(
                y,
                p_n,
                v_n,
                m,
                g,
                dt,
            )

            residual_norm = torch.norm(r).item()
            distance_to_star = torch.norm(y - y_star).item()

            if loss_type == "energy":
                terminal_objective = energy_gap
            elif loss_type == "residual":
                terminal_objective = residual_norm**2
            else:
                raise ValueError(loss_type)

            state_dict = {
                "init_state_id": int(init_state_id),
                "terminal_objective": float(terminal_objective),
                "terminal_energy_gap": float(energy_gap),
                "terminal_residual_norm": float(residual_norm),
                "terminal_distance_to_star": float(distance_to_star),
            }

            per_state_metrics.append(state_dict)
            terminal_objectives.append(terminal_objective)
            terminal_energy_gaps.append(energy_gap)
            terminal_residual_norms.append(residual_norm)
            terminal_distances.append(distance_to_star)

    return {
        "mean_terminal_objective": float(np.mean(terminal_objectives)),
        "worst_terminal_objective": float(np.max(terminal_objectives)),
        "mean_terminal_energy_gap": float(np.mean(terminal_energy_gaps)),
        "worst_terminal_energy_gap": float(np.max(terminal_energy_gaps)),
        "mean_terminal_residual_norm": float(np.mean(terminal_residual_norms)),
        "worst_terminal_residual_norm": float(np.max(terminal_residual_norms)),
        "mean_terminal_distance_to_star": float(np.mean(terminal_distances)),
        "worst_terminal_distance_to_star": float(np.max(terminal_distances)),
        "per_state": per_state_metrics,
    }


# ============================================================
# 5. Plot helpers
# ============================================================

def to_plot_coordinates(points, y0, relative=True):
    points_np = np.asarray(points, dtype=float).reshape(-1, 3)
    y0_np = np.asarray(y0, dtype=float).reshape(1, 3)

    if relative:
        points_np = points_np - y0_np

    return points_np


def set_equal_3d_axes(ax, points):
    points_np = np.asarray(points, dtype=float).reshape(-1, 3)

    center = points_np.mean(axis=0)
    span = np.ptp(points_np, axis=0)
    radius = max(float(span.max()) / 2.0, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def set_axis_labels(ax, relative=True):
    if relative:
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(r"$\Delta y$")
        ax.set_zlabel(r"$\Delta z$")
    else:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")


def safe_log_curve(values, floor=1e-16):
    return np.maximum(np.asarray(values, dtype=float), floor)


def plot_training_point_distribution(
    name,
    training_point_positions,
    training_point_buckets,
    result_point_positions,
    result_point_buckets,
    y0,
    newton_solution,
    epochs,
    bucket_size=200,
    relative=True,
    save_path=None,
    figure_title=None,
    reference_point=None,
    reference_point_label=None,
):
    """
    Create two 3D subplots:
        left : all training input points x^(0), ..., x^(K-1)
        right: all training result points x^(K)

    The points are the actual states visited while training. Since opt.step()
    is called after every micro-step, later points may be produced by updated
    network parameters.
    """

    train_pts = to_plot_coordinates(
        training_point_positions,
        y0=y0,
        relative=relative,
    )

    result_pts = to_plot_coordinates(
        result_point_positions,
        y0=y0,
        relative=relative,
    )

    newton_pt = to_plot_coordinates(
        [newton_solution],
        y0=y0,
        relative=relative,
    )[0]

    if reference_point is None:
        reference_point = y0
    if reference_point_label is None:
        reference_point_label = r"Reference initial point $y_0$"

    reference_pt = to_plot_coordinates(
        [reference_point],
        y0=y0,
        relative=relative,
    )[0]

    train_buckets = np.asarray(training_point_buckets, dtype=int)
    result_buckets = np.asarray(result_point_buckets, dtype=int)

    num_buckets = (epochs + bucket_size - 1) // bucket_size
    cmap = plt.get_cmap("tab10", num_buckets)

    fig = plt.figure(figsize=(19, 8))
    ax_train = fig.add_subplot(121, projection="3d")
    ax_result = fig.add_subplot(122, projection="3d")

    for bucket in range(num_buckets):
        train_mask = train_buckets == bucket
        result_mask = result_buckets == bucket

        if np.any(train_mask):
            ax_train.scatter(
                train_pts[train_mask, 0],
                train_pts[train_mask, 1],
                train_pts[train_mask, 2],
                marker="o",
                s=8,
                alpha=0.22,
                color=cmap(bucket),
            )

        if np.any(result_mask):
            ax_result.scatter(
                result_pts[result_mask, 0],
                result_pts[result_mask, 1],
                result_pts[result_mask, 2],
                marker="^",
                s=17,
                alpha=0.55,
                color=cmap(bucket),
            )

    for ax in (ax_train, ax_result):
        ax.scatter(
            reference_pt[0],
            reference_pt[1],
            reference_pt[2],
            marker="x",
            s=120,
            c="black",
            linewidths=2.0,
            label=reference_point_label,
        )

        ax.scatter(
            newton_pt[0],
            newton_pt[1],
            newton_pt[2],
            marker="*",
            s=360,
            c="crimson",
            label="Newton converged solution",
        )

        ax.text(
            newton_pt[0],
            newton_pt[1],
            newton_pt[2],
            "  * Newton",
            fontsize=10,
            color="crimson",
        )

        set_axis_labels(ax, relative=relative)
        ax.grid(True, alpha=0.3)

    ax_train.set_title(
        r"All Training Input Points: "
        r"$x^{(0)}, x^{(1)}, \ldots, x^{(K-1)}$"
    )

    ax_result.set_title(
        r"All Training Result Points: $x^{(K)}$"
    )

    set_equal_3d_axes(
        ax_train,
        np.vstack(
            [
                train_pts,
                reference_pt.reshape(1, 3),
                newton_pt.reshape(1, 3),
            ]
        ),
    )

    set_equal_3d_axes(
        ax_result,
        np.vstack(
            [
                result_pts,
                reference_pt.reshape(1, 3),
                newton_pt.reshape(1, 3),
            ]
        ),
    )

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=r"Training inputs $x^{(0)}, \ldots, x^{(K-1)}$",
            markerfacecolor="gray",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label=r"Training result $x^{(K)}$",
            markerfacecolor="gray",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            label=reference_point_label,
            linestyle="None",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Newton converged solution",
            markerfacecolor="crimson",
            markersize=14,
        ),
    ]

    color_handles = []

    for bucket in range(num_buckets):
        start_epoch = bucket * bucket_size
        end_epoch = min(
            (bucket + 1) * bucket_size - 1,
            epochs - 1,
        )

        color_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Epoch {start_epoch}-{end_epoch}",
                markerfacecolor=cmap(bucket),
                markersize=8,
            )
        )

    fig.legend(
        handles=marker_handles + color_handles,
        loc="center right",
        bbox_to_anchor=(1.00, 0.50),
    )

    coordinate_note = "relative to y0" if relative else "absolute coordinates"

    if figure_title is None:
        figure_title = f"{name}: Training Data and Result Distribution ({coordinate_note})"

    fig.suptitle(
        figure_title,
        fontsize=14,
    )

    plt.tight_layout(rect=[0.00, 0.00, 0.82, 0.95])

    if save_path is None:
        save_path = PLOTS_DIR / f"{name}_points.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return str(save_path)


def plot_training_loss_curves(name, history, loss_type, save_path=None, figure_title=None):
    """
    Create a three-panel plot for one MLP or one specific training state.

    Panel 1:
        Mean training objective per actual backpropagation micro-step.

    Panel 2:
        - Last backprop objective: the exact final objective encountered by the
          training loop in an epoch.
        - Frozen-model worst terminal objective.
        - Frozen-model mean terminal objective.

    Panel 3:
        Frozen-model terminal metrics, using common physical quantities.
    """

    epochs = np.asarray(history["epoch"], dtype=int)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 15),
    )

    axes[0].plot(
        epochs,
        safe_log_curve(history["mean_training_objective"]),
        label="Mean training objective per backprop step",
    )

    axes[0].set_yscale("log")
    axes[0].set_title("Mean Training Objective per Optimizer Step")
    axes[0].set_xlabel("Epoch")

    if loss_type == "energy":
        axes[0].set_ylabel(r"Mean $E(x)-E^*$")
    else:
        axes[0].set_ylabel(r"Mean $\|r(x)\|^2$")

    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        safe_log_curve(history["last_backprop_objective"]),
        label="Last backprop objective (original order-dependent statistic)",
    )

    axes[1].plot(
        epochs,
        safe_log_curve(history["worst_terminal_objective"]),
        label="Frozen-model worst terminal objective",
    )

    axes[1].plot(
        epochs,
        safe_log_curve(history["mean_terminal_objective"]),
        linestyle="--",
        label="Frozen-model mean terminal objective",
    )

    axes[1].set_yscale("log")
    axes[1].set_title("Terminal / End-of-Epoch Objective")
    axes[1].set_xlabel("Epoch")

    if loss_type == "energy":
        axes[1].set_ylabel(r"$E(x)-E^*$")
    else:
        axes[1].set_ylabel(r"$\|r(x)\|^2$")

    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        epochs,
        safe_log_curve(history["mean_terminal_energy_gap"]),
        label="Mean terminal energy gap",
    )

    axes[2].plot(
        epochs,
        safe_log_curve(history["worst_terminal_energy_gap"]),
        linestyle="--",
        label="Worst terminal energy gap",
    )

    axes[2].plot(
        epochs,
        safe_log_curve(history["mean_terminal_residual_norm"]),
        label="Mean terminal residual norm",
    )

    axes[2].plot(
        epochs,
        safe_log_curve(history["worst_terminal_residual_norm"]),
        linestyle="--",
        label="Worst terminal residual norm",
    )

    axes[2].plot(
        epochs,
        safe_log_curve(history["mean_terminal_distance_to_star"]),
        label="Mean terminal distance to y*",
    )

    axes[2].plot(
        epochs,
        safe_log_curve(history["worst_terminal_distance_to_star"]),
        linestyle="--",
        label="Worst terminal distance to y*",
    )

    axes[2].set_yscale("log")
    axes[2].set_title("Frozen-model Terminal Metrics after Each Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Metric value")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2)

    if figure_title is None:
        figure_title = f"{name}: Training Loss and Frozen-model Evaluation"

    fig.suptitle(
        figure_title,
        fontsize=14,
    )

    plt.tight_layout(rect=[0.00, 0.00, 1.00, 0.97])

    if save_path is None:
        save_path = PLOTS_DIR / f"{name}_training_loss.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return str(save_path)


# ============================================================
# 6. Training
# ============================================================

def train_model(
    name,
    use_normalization,
    use_dt_scaling,
    use_coverage,
    loss_type,
    epochs=1000,
    lr=1e-3,
):
    torch.manual_seed(42)

    # ========================================================
    # Physics config
    # ========================================================

    m = 1.0
    g = 9.8
    dt = 0.01

    p_n = torch.tensor([3.0, 4.0, 5.0])
    v_n = torch.tensor([0.5, -0.5, 0.0])

    y0 = p_n.clone()

    history = torch.cat([p_n, v_n])
    params = torch.tensor([m, g, dt])

    y_star = (
        p_n
        + dt * v_n
        - dt**2 * torch.tensor([0.0, 0.0, g])
    )

    E_star = variational_energy(
        y_star,
        p_n,
        v_n,
        m,
        g,
        dt,
    ).item()

    # For this strictly convex quadratic problem, one Newton step reaches y*.
    newton_solution = (
        y0
        + newton_direction(
            y0,
            p_n,
            v_n,
            m,
            g,
            dt,
        )
    )

    # ========================================================
    # Dataset
    # ========================================================

    if use_coverage:
        train_states = make_training_states(
            y0,
            y_star,
            dt,
            num_line_points=11,
            num_local_points=32,
            local_std_dt_units=1.0,
        )
    else:
        train_states = [y0]

    # ========================================================
    # Normalization
    # ========================================================

    if use_normalization:
        input_mean, input_std = compute_input_normalizer(
            train_states,
            history,
            params,
        )
    else:
        input_mean = torch.zeros(12)
        input_std = torch.ones(12)

    # ========================================================
    # Model
    # ========================================================

    mlp = MLPOptimizer(
        use_normalization=use_normalization,
        use_dt_scaling=use_dt_scaling,
        input_mean=input_mean,
        input_std=input_std,
    )

    opt = torch.optim.Adam(
        mlp.parameters(),
        lr=lr,
    )

    model_plot_dir = ensure_model_plot_dir(name)

    # ========================================================
    # Training logs
    # ========================================================

    history_keys = [
        "epoch",
        "K",
        "mean_training_objective",
        "last_backprop_objective",
        "mean_terminal_objective",
        "worst_terminal_objective",
        "mean_terminal_energy_gap",
        "worst_terminal_energy_gap",
        "mean_terminal_residual_norm",
        "worst_terminal_residual_norm",
        "mean_terminal_distance_to_star",
        "worst_terminal_distance_to_star",
    ]

    training_history = {key: [] for key in history_keys}

    per_state_history = {
        init_state_id: {key: [] for key in history_keys}
        for init_state_id in range(len(train_states))
    }

    # Detailed logs are saved as compact arrays rather than giant JSON lists.
    # Training point row:
    #   [epoch, bucket, init_state_id, iteration, x, y, z]
    # Result point row:
    #   [epoch, bucket, init_state_id, x, y, z]
    # Micro-step row:
    #   [epoch, bucket, init_state_id, iteration,
    #    raw_backprop_loss, objective_for_plot]
    training_point_rows = []
    result_point_rows = []
    micro_step_rows = []

    # ========================================================
    # Training
    # ========================================================

    K = 1

    for epoch in range(epochs):
        if epoch > 0 and epoch % 200 == 0 and K < 10:
            K += 1

        bucket = epoch // COLOR_BUCKET_SIZE

        objective_sum = 0.0
        objective_count = 0
        last_backprop_objective = None

        per_state_objective_sum = [0.0 for _ in range(len(train_states))]
        per_state_objective_count = [0 for _ in range(len(train_states))]
        per_state_last_backprop_objective = [None for _ in range(len(train_states))]

        for init_state_id, y_init in enumerate(train_states):
            y = y_init.clone()

            for k in range(K):
                # --------------------------------------------
                # Save x^(k): actual network input during training.
                # --------------------------------------------
                y_before = y.detach().clone()

                training_point_rows.append(
                    [
                        epoch,
                        bucket,
                        init_state_id,
                        k,
                        y_before[0].item(),
                        y_before[1].item(),
                        y_before[2].item(),
                    ]
                )

                # --------------------------------------------
                # Forward and update state: x^(k) -> x^(k+1)
                # --------------------------------------------
                delta = mlp(y, history, params)
                y = y + delta

                # --------------------------------------------
                # Backprop objective
                # --------------------------------------------
                if loss_type == "energy":
                    loss = variational_energy(
                        y,
                        p_n,
                        v_n,
                        m,
                        g,
                        dt,
                    )
                elif loss_type == "residual":
                    loss = residual_loss(
                        y,
                        p_n,
                        v_n,
                        m,
                        g,
                        dt,
                    )
                else:
                    raise ValueError(loss_type)

                raw_backprop_loss = float(loss.item())

                if loss_type == "energy":
                    current_objective_for_plot = max(
                        raw_backprop_loss - E_star,
                        0.0,
                    )
                else:
                    current_objective_for_plot = raw_backprop_loss

                # --------------------------------------------
                # Update model parameters immediately.
                # --------------------------------------------
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)

                y = y.detach()

                objective_sum += current_objective_for_plot
                objective_count += 1
                last_backprop_objective = current_objective_for_plot

                per_state_objective_sum[init_state_id] += current_objective_for_plot
                per_state_objective_count[init_state_id] += 1
                per_state_last_backprop_objective[init_state_id] = current_objective_for_plot

                micro_step_rows.append(
                    [
                        epoch,
                        bucket,
                        init_state_id,
                        k + 1,
                        raw_backprop_loss,
                        current_objective_for_plot,
                    ]
                )

            # ----------------------------------------------
            # Save x^(K): actual result after the K training updates
            # for this initial state.
            # ----------------------------------------------
            result_point_rows.append(
                [
                    epoch,
                    bucket,
                    init_state_id,
                    y[0].item(),
                    y[1].item(),
                    y[2].item(),
                ]
            )

        if objective_count == 0 or last_backprop_objective is None:
            raise RuntimeError("No optimization micro-step was executed.")

        mean_training_objective = objective_sum / objective_count

        # ----------------------------------------------------
        # Freeze current MLP after the epoch and evaluate all
        # training initial states using the same parameters.
        # ----------------------------------------------------
        terminal_metrics = frozen_terminal_metrics(
            mlp=mlp,
            train_states=train_states,
            K=K,
            history=history,
            params=params,
            p_n=p_n,
            v_n=v_n,
            y_star=y_star,
            E_star=E_star,
            loss_type=loss_type,
            m=m,
            g=g,
            dt=dt,
        )

        training_history["epoch"].append(epoch)
        training_history["K"].append(K)
        training_history["mean_training_objective"].append(
            float(mean_training_objective)
        )
        training_history["last_backprop_objective"].append(
            float(last_backprop_objective)
        )

        per_state_terminal_metrics = terminal_metrics.pop("per_state")

        for key, value in terminal_metrics.items():
            training_history[key].append(float(value))

        for init_state_id in range(len(train_states)):
            if per_state_objective_count[init_state_id] == 0:
                raise RuntimeError(f"No micro-step executed for init_state_id={init_state_id}.")
            if per_state_last_backprop_objective[init_state_id] is None:
                raise RuntimeError(f"Missing last backprop objective for init_state_id={init_state_id}.")

            state_hist = per_state_history[init_state_id]
            state_hist["epoch"].append(epoch)
            state_hist["K"].append(K)
            state_hist["mean_training_objective"].append(
                float(per_state_objective_sum[init_state_id] / per_state_objective_count[init_state_id])
            )
            state_hist["last_backprop_objective"].append(
                float(per_state_last_backprop_objective[init_state_id])
            )

            state_metrics = per_state_terminal_metrics[init_state_id]
            state_hist["mean_terminal_objective"].append(float(state_metrics["terminal_objective"]))
            state_hist["worst_terminal_objective"].append(float(state_metrics["terminal_objective"]))
            state_hist["mean_terminal_energy_gap"].append(float(state_metrics["terminal_energy_gap"]))
            state_hist["worst_terminal_energy_gap"].append(float(state_metrics["terminal_energy_gap"]))
            state_hist["mean_terminal_residual_norm"].append(float(state_metrics["terminal_residual_norm"]))
            state_hist["worst_terminal_residual_norm"].append(float(state_metrics["terminal_residual_norm"]))
            state_hist["mean_terminal_distance_to_star"].append(float(state_metrics["terminal_distance_to_star"]))
            state_hist["worst_terminal_distance_to_star"].append(float(state_metrics["terminal_distance_to_star"]))

    # ========================================================
    # Rollout evaluation from y0
    # ========================================================

    max_steps = 20
    y = y0.clone()

    rollout_energy_gap = []
    rollout_residual_norm = []
    rollout_delta_norm = []
    rollout_distance_to_star = []

    for _ in range(max_steps):
        # Evaluate the current state before applying the next update.
        energy = variational_energy(
            y,
            p_n,
            v_n,
            m,
            g,
            dt,
        ).item()

        gap = max(float(energy - E_star), 0.0)

        r = energy_residual(
            y,
            p_n,
            v_n,
            m,
            g,
            dt,
        )

        residual_norm = torch.norm(r).item()
        dist = torch.norm(y - y_star).item()

        with torch.no_grad():
            delta = mlp(y, history, params)

        delta_norm = torch.norm(delta).item()

        rollout_energy_gap.append(gap)
        rollout_residual_norm.append(residual_norm)
        rollout_delta_norm.append(delta_norm)
        rollout_distance_to_star.append(dist)

        y = y + delta

    # ========================================================
    # Fixed-point residual
    # ========================================================

    with torch.no_grad():
        delta_star = mlp(
            y_star,
            history,
            params,
        )

    fixed_point_residual = torch.norm(delta_star).item()

    # ========================================================
    # Vector field
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
                    params,
                )

            field_x.append(y_probe[0].item())
            field_z.append(y_probe[2].item())
            field_u.append(d[0].item())
            field_v.append(d[2].item())

    # ========================================================
    # Convert logs to compact arrays
    # ========================================================

    training_point_rows = np.asarray(training_point_rows, dtype=np.float64)
    result_point_rows = np.asarray(result_point_rows, dtype=np.float64)
    micro_step_rows = np.asarray(micro_step_rows, dtype=np.float64)

    # ========================================================
    # Save detailed compressed logs for this MLP
    # ========================================================

    detailed_log_path = LOGS_DIR / f"{name}_detailed_logs.npz"

    np.savez_compressed(
        detailed_log_path,
        training_points=training_point_rows,
        result_points=result_point_rows,
        micro_steps=micro_step_rows,
        newton_solution=newton_solution.detach().cpu().numpy(),
        y_star=y_star.detach().cpu().numpy(),
        y0=y0.detach().cpu().numpy(),
        # Column descriptions stored as arrays for convenience.
        training_point_columns=np.asarray(
            ["epoch", "bucket", "init_state_id", "iteration", "x", "y", "z"]
        ),
        result_point_columns=np.asarray(
            ["epoch", "bucket", "init_state_id", "x", "y", "z"]
        ),
        micro_step_columns=np.asarray(
            [
                "epoch",
                "bucket",
                "init_state_id",
                "iteration",
                "raw_backprop_loss",
                "objective_for_plot",
            ]
        ),
    )

    # ========================================================
    # Per-model plots: overall + one group for each initial state
    # ========================================================

    overall_point_plot_path = plot_training_point_distribution(
        name=name,
        training_point_positions=training_point_rows[:, 4:7],
        training_point_buckets=training_point_rows[:, 1].astype(int),
        result_point_positions=result_point_rows[:, 3:6],
        result_point_buckets=result_point_rows[:, 1].astype(int),
        y0=y0.tolist(),
        newton_solution=newton_solution.tolist(),
        epochs=epochs,
        bucket_size=COLOR_BUCKET_SIZE,
        relative=PLOT_RELATIVE_COORDINATES,
        save_path=model_plot_dir / "overall_points.png",
        figure_title=f"{name}: Overall Training Data and Result Distribution",
        reference_point=y0.tolist(),
        reference_point_label=r"Reference initial point $y_0$",
    )

    overall_loss_plot_path = plot_training_loss_curves(
        name=name,
        history=training_history,
        loss_type=loss_type,
        save_path=model_plot_dir / "overall_training_loss.png",
        figure_title=f"{name}: Overall Training Loss and Frozen-model Evaluation",
    )

    per_state_plot_paths = {}
    per_state_loss_plot_paths = {}

    train_state_positions = [state.detach().cpu().numpy().tolist() for state in train_states]

    for init_state_id, state_position in enumerate(train_state_positions):
        train_mask = training_point_rows[:, 2].astype(int) == init_state_id
        result_mask = result_point_rows[:, 2].astype(int) == init_state_id

        state_name = f"state_{init_state_id:02d}"
        state_title_prefix = f"{name}: init_state_id={init_state_id}"
        state_label = rf"Training initial state $s_{{{init_state_id}}}$"

        per_state_plot_paths[state_name] = plot_training_point_distribution(
            name=name,
            training_point_positions=training_point_rows[train_mask, 4:7],
            training_point_buckets=training_point_rows[train_mask, 1].astype(int),
            result_point_positions=result_point_rows[result_mask, 3:6],
            result_point_buckets=result_point_rows[result_mask, 1].astype(int),
            y0=y0.tolist(),
            newton_solution=newton_solution.tolist(),
            epochs=epochs,
            bucket_size=COLOR_BUCKET_SIZE,
            relative=PLOT_RELATIVE_COORDINATES,
            save_path=model_plot_dir / f"{state_name}_points.png",
            figure_title=f"{state_title_prefix}: Training Data and Result Distribution",
            reference_point=state_position,
            reference_point_label=state_label,
        )

        per_state_loss_plot_paths[state_name] = plot_training_loss_curves(
            name=name,
            history=per_state_history[init_state_id],
            loss_type=loss_type,
            save_path=model_plot_dir / f"{state_name}_training_loss.png",
            figure_title=f"{state_title_prefix}: Training Loss and Frozen-model Evaluation",
        )

    # ========================================================
    # Return summary and plotting data
    # ========================================================

    return {
        "name": name,
        "loss_type": loss_type,
        "use_normalization": use_normalization,
        "use_dt_scaling": use_dt_scaling,
        "use_coverage": use_coverage,
        "num_training_states": len(train_states),
        "fixed_point_residual": fixed_point_residual,
        "rollout_energy_gap": rollout_energy_gap,
        "rollout_residual_norm": rollout_residual_norm,
        "rollout_delta_norm": rollout_delta_norm,
        "rollout_distance_to_star": rollout_distance_to_star,
        "training_history": training_history,
        "field_x": np.asarray(field_x),
        "field_z": np.asarray(field_z),
        "field_u": np.asarray(field_u),
        "field_v": np.asarray(field_v),
        "y_star": y_star,
        "newton_solution": newton_solution,
        "overall_point_plot_path": overall_point_plot_path,
        "overall_loss_plot_path": overall_loss_plot_path,
        "per_state_point_plot_paths": per_state_plot_paths,
        "per_state_loss_plot_paths": per_state_loss_plot_paths,
        "per_state_history": per_state_history,
        "training_state_positions": train_state_positions,
        "detailed_log_path": str(detailed_log_path),
    }


# ============================================================
# 7. Overview plot
# ============================================================

def plot_overview(results):
    fig = plt.figure(figsize=(22, 18))

    # ========================================================
    # 1. Energy Gap
    # ========================================================

    ax1 = plt.subplot(3, 2, 1)

    for res in results:
        curve = safe_log_curve(res["rollout_energy_gap"])

        ax1.plot(
            curve,
            marker="o",
            label=res["name"],
        )

    ax1.set_yscale("log")
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
        curve = safe_log_curve(res["rollout_residual_norm"])

        ax2.plot(
            curve,
            marker="o",
            label=res["name"],
        )

    ax2.set_yscale("log")
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
        curve = safe_log_curve(res["rollout_delta_norm"])

        ax3.plot(
            curve,
            marker="o",
            label=res["name"],
        )

    ax3.set_yscale("log")
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
        curve = safe_log_curve(res["rollout_distance_to_star"])

        ax4.plot(
            curve,
            marker="o",
            label=res["name"],
        )

    ax4.set_yscale("log")
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
        max(r["fixed_point_residual"], 1e-16)
        for r in results
    ]

    ax5.bar(names, vals)
    ax5.set_yscale("log")
    ax5.set_title("Fixed-point Residual")
    ax5.set_ylabel("||delta(y*)||")
    ax5.tick_params(axis="x", rotation=45)

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
        angles="xy",
        scale_units="xy",
        scale=1.0,
        alpha=0.7,
        label="Baseline",
    )

    ax6.quiver(
        full["field_x"][::stride],
        full["field_z"][::stride],
        full["field_u"][::stride],
        full["field_v"][::stride],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        alpha=0.7,
        label="Full",
    )

    ax6.scatter(
        full["newton_solution"][0].item(),
        full["newton_solution"][2].item(),
        s=150,
        marker="*",
        c="crimson",
        label="Newton solution = y*",
    )

    ax6.set_title("Vector Field Near Optimum")
    ax6.set_xlabel("x")
    ax6.set_ylabel("z")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()

    save_path = OUTPUT_DIR / "full_ablation_study.png"

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return str(save_path)


# ============================================================
# 8. JSON report
# ============================================================

def save_json_report(results, overview_plot_path):
    report = {
        "overview_plot": overview_plot_path,
        "notes": {
            "mean_training_objective": (
                "Mean objective across all actual backpropagation micro-steps "
                "within an epoch. This is normalized by the number of initial "
                "states and K."
            ),
            "last_backprop_objective": (
                "The exact final objective encountered by the training loop in "
                "an epoch. It preserves the original order-dependent statistic; "
                "it is not mathematically guaranteed to be the worst training "
                "state."
            ),
            "worst_terminal_objective": (
                "After each epoch, freeze the current MLP, roll out every "
                "training initial state for K steps, and take the maximum "
                "terminal objective. This is the strict worst-case metric."
            ),
            "point_distribution": (
                "Detailed training input points x^(0)...x^(K-1) and result "
                "points x^(K) are stored in compressed NPZ files."
            ),
            "per_state_outputs": (
                "For every experiment, the script saves an overall pair of plots "
                "plus one pair of plots for each training initial state."
            ),
        },
        "experiments": {},
    }

    for res in results:
        report["experiments"][res["name"]] = {
            "loss_type": res["loss_type"],
            "use_normalization": res["use_normalization"],
            "use_dt_scaling": res["use_dt_scaling"],
            "use_coverage": res["use_coverage"],
            "num_training_states": res["num_training_states"],
            "fixed_point_residual": res["fixed_point_residual"],
            "final_energy_gap": res["rollout_energy_gap"][-1],
            "final_residual_norm": res["rollout_residual_norm"][-1],
            "energy_gap_curve": res["rollout_energy_gap"],
            "residual_norm_curve": res["rollout_residual_norm"],
            "delta_norm_curve": res["rollout_delta_norm"],
            "distance_to_star_curve": res["rollout_distance_to_star"],
            "training_history": res["training_history"],
            "overall_point_plot": res["overall_point_plot_path"],
            "overall_training_loss_plot": res["overall_loss_plot_path"],
            "per_state_point_plots": res["per_state_point_plot_paths"],
            "per_state_training_loss_plots": res["per_state_loss_plot_paths"],
            "per_state_history": res["per_state_history"],
            "training_state_positions": res["training_state_positions"],
            "detailed_log": res["detailed_log_path"],
        }

    report_path = OUTPUT_DIR / "full_ablation_report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return str(report_path)


# ============================================================
# 9. Main
# ============================================================

def main():
    ensure_output_dirs()

    configs = [
        # ====================================================
        # BASE
        # ====================================================
        {
            "name": "A1_Base_Energy",
            "use_normalization": False,
            "use_dt_scaling": False,
            "use_coverage": False,
            "loss_type": "energy",
        },
        {
            "name": "A2_Base_Residual",
            "use_normalization": False,
            "use_dt_scaling": False,
            "use_coverage": False,
            "loss_type": "residual",
        },
        # ====================================================
        # NORMALIZATION ONLY IN THE ORIGINAL EXPERIMENT DESIGN
        # Note: use_dt_scaling is also enabled, matching the
        # uploaded script.
        # ====================================================
        {
            "name": "B1_Norm_Energy",
            "use_normalization": True,
            "use_dt_scaling": True,
            "use_coverage": False,
            "loss_type": "energy",
        },
        {
            "name": "B2_Norm_Residual",
            "use_normalization": True,
            "use_dt_scaling": True,
            "use_coverage": False,
            "loss_type": "residual",
        },
        # ====================================================
        # COVERAGE ONLY
        # ====================================================
        {
            "name": "C1_Coverage_Energy",
            "use_normalization": False,
            "use_dt_scaling": False,
            "use_coverage": True,
            "loss_type": "energy",
        },
        {
            "name": "C2_Coverage_Residual",
            "use_normalization": False,
            "use_dt_scaling": False,
            "use_coverage": True,
            "loss_type": "residual",
        },
        # ====================================================
        # FULL
        # ====================================================
        {
            "name": "D1_Full_Energy",
            "use_normalization": True,
            "use_dt_scaling": True,
            "use_coverage": True,
            "loss_type": "energy",
        },
        {
            "name": "D2_Full_Residual",
            "use_normalization": True,
            "use_dt_scaling": True,
            "use_coverage": True,
            "loss_type": "residual",
        },
    ]

    results = []

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

        print(
            f"Saved overall point plot: "
            f"{res['overall_point_plot_path']}"
        )

        print(
            f"Saved overall loss plot: "
            f"{res['overall_loss_plot_path']}"
        )

        print(
            f"Saved per-state point plots: "
            f"{len(res['per_state_point_plot_paths'])}"
        )

        print(
            f"Saved per-state loss plots: "
            f"{len(res['per_state_loss_plot_paths'])}"
        )

        print(
            f"Saved detailed log: "
            f"{res['detailed_log_path']}"
        )

    overview_plot_path = plot_overview(results)
    report_path = save_json_report(results, overview_plot_path)

    print("\nSaved overview figure:", overview_plot_path)
    print("Saved report:", report_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
