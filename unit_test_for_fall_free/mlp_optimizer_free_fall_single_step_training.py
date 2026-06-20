"""
Learned iterative optimizer experiment on a one-second free-fall trajectory.

Experiment design
-----------------
Test trajectory:
    - Initial position: [3, 4, 50]
    - Initial velocity: [1, -1, 0]
    - dt = 0.01 s
    - 100 analytic states at t = 0.00, 0.01, ..., 0.99 s
      (the initial state is included; the t = 1.00 s endpoint is excluded)

Training data:
    - The 100 analytic states form 100 physical-state groups.
    - For each group, perturb the optimization initial guess y^(0) = p_n ten times.
    - Total: 100 groups x 10 perturbed optimization initials = 1000 samples.
    - Input normalization statistics are computed directly from these 1000 samples.

Training schedule:
    - At each epoch, randomly choose one of the 100 groups.
    - Sequentially train on the ten perturbed initials in that group.
    - For every sample and every optimizer iteration:
        * compute one loss,
        * backpropagate once,
        * update network parameters once,
        * detach the state before the next optimizer iteration.
    - Use exactly K = 1 learned-optimizer update per sample.
    - No iterative learned-optimizer rollout is used during training.

Evaluation:
    - Preserve all visited training points and final points for visualization.
    - On all 100 analytic test states, compare one-step MLP and Newton residuals.
    - Recursively simulate 101 motion frames and save an MLP/Newton comparison video.
    - Run both SGD(lr=1e-2) and Adam(lr=1e-4).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Suitable for headless Linux servers.

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 0. Global configuration
# ============================================================

TORCH_DTYPE = torch.float32
torch.set_default_dtype(TORCH_DTYPE)

# This floor is only for logarithmic visualization. It does not claim float32
# can reliably resolve energy gaps or residuals at this numerical level.
PLOT_FLOOR = 1e-12

# Reproducibility
DATASET_RANDOM_SEED = 123
GROUP_SAMPLING_RANDOM_SEED = 456
MODEL_RANDOM_SEED = 42

# Physical problem
MASS = 1.0
GRAVITY = 9.8
DT = 0.01
INITIAL_POSITION = (3.0, 4.0, 50.0)
INITIAL_VELOCITY = (1.0, -1.0, 0.0)

# Dataset
NUM_TEST_FRAMES = 100  # t = 0.00, 0.01, ..., 0.99 seconds
NUM_PERTURBATIONS_PER_GROUP = 10
TRAIN_PERTURBATION_STD = 1e-2

# Learned optimizer
USE_NORMALIZATION = True
USE_DT_SCALING = True

# Training curriculum
EPOCHS = 1000
INITIAL_K = 1
K_INCREASE_INTERVAL = 200
K_INCREASE_AMOUNT = 1
MAX_K = 1
COLOR_BUCKET_SIZE = 200
PERIODIC_EVAL_INTERVAL = 100

# Final evaluation
FINAL_TEST_STEPS = 1
MOTION_SOLVER_STEPS = 1
SAVE_VIDEO = True
VIDEO_FPS = 20

# Visualization
PLOT_RELATIVE_COORDINATES = False

OPTIMIZER_CONFIGS = [
    {
        "optimizer_name": "sgd",
        "learning_rate": 1e-2,
    },
    {
        "optimizer_name": "adam",
        "learning_rate": 1e-4,
    },
]


@dataclass(frozen=True)
class ExperimentConfig:
    optimizer_name: str
    learning_rate: float

    @property
    def name(self) -> str:
        return (
            f"float32_{self.optimizer_name}_lr_{self.learning_rate:.0e}_"
            f"trajectory_groups_100x10"
        )


# ============================================================
# 1. Output directory
# ============================================================


def create_output_directory() -> Path:
    """Create a sibling output directory named after this Python script."""

    script_path = Path(__file__).resolve()
    output_dir = script_path.parent / script_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================
# 2. Learned optimizer and physical objective
# ============================================================


class MLPOptimizer(nn.Module):
    """Small learned iterative optimizer: 12 input channels -> 3D update."""

    def __init__(
        self,
        use_normalization: bool,
        use_dt_scaling: bool,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
    ) -> None:
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

        # Begin from the conservative rule delta_y = 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.register_buffer("input_mean", input_mean.clone().detach())
        self.register_buffer("input_std", input_std.clone().detach())

    def forward(
        self,
        y: torch.Tensor,
        history: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        inp = torch.cat([y, history, params], dim=-1)

        if self.use_normalization:
            inp = (inp - self.input_mean) / self.input_std

        delta = self.net(inp)

        if self.use_dt_scaling:
            delta = params[2] * delta

        return delta


def variational_energy(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float = MASS,
    g: float = GRAVITY,
    dt: float = DT,
) -> torch.Tensor:
    """Implicit-Euler variational energy for free fall."""

    residual = y - p_n - dt * v_n
    kinetic_term = (m / (2.0 * dt**2)) * torch.sum(residual**2)
    potential_term = m * g * y[2]
    return kinetic_term + potential_term


def stationarity_residual(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float = MASS,
    g: float = GRAVITY,
    dt: float = DT,
) -> torch.Tensor:
    """Gradient of the implicit-Euler variational energy."""

    residual = (m / dt**2) * (y - p_n - dt * v_n)
    residual = residual.clone()
    residual[2] += m * g
    return residual


def stationarity_residual_norm(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float = MASS,
    g: float = GRAVITY,
    dt: float = DT,
) -> torch.Tensor:
    return torch.norm(stationarity_residual(y, p_n, v_n, m, g, dt))


def newton_direction(
    y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    m: float = MASS,
    g: float = GRAVITY,
    dt: float = DT,
) -> torch.Tensor:
    """Newton direction. For this quadratic objective, one step is exact."""

    grad = stationarity_residual(y, p_n, v_n, m, g, dt)
    hessian_inverse = dt**2 / m
    return -hessian_inverse * grad


def implicit_euler_exact_solution(
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    g: float = GRAVITY,
    dt: float = DT,
) -> torch.Tensor:
    """Closed-form minimizer of the current implicit-Euler objective."""

    gravity_vector = torch.tensor(
        [0.0, 0.0, g],
        dtype=p_n.dtype,
        device=p_n.device,
    )
    return p_n + dt * v_n - dt**2 * gravity_vector


# ============================================================
# 3. Dataset generation and normalization
# ============================================================


def analytic_free_fall_states(
    initial_position: Sequence[float],
    initial_velocity: Sequence[float],
    dt: float,
    num_frames: int,
    g: float = GRAVITY,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate analytic free-fall states at t = 0, dt, ..., (num_frames - 1) dt.
    """

    times = torch.arange(num_frames, dtype=TORCH_DTYPE) * dt
    p0 = torch.tensor(initial_position, dtype=TORCH_DTYPE)
    v0 = torch.tensor(initial_velocity, dtype=TORCH_DTYPE)
    acceleration = torch.tensor([0.0, 0.0, -g], dtype=TORCH_DTYPE)

    positions = (
        p0.unsqueeze(0)
        + times.unsqueeze(1) * v0.unsqueeze(0)
        + 0.5 * times.square().unsqueeze(1) * acceleration.unsqueeze(0)
    )
    velocities = v0.unsqueeze(0) + times.unsqueeze(1) * acceleration.unsqueeze(0)

    return times, positions, velocities


def build_training_groups(
    test_positions: torch.Tensor,
    perturbation_std: float,
    num_perturbations_per_group: int,
    seed: int,
) -> torch.Tensor:
    """
    Perturb each analytic frame ten times.

    Returns:
        Tensor with shape [num_groups, num_perturbations_per_group, 3].
    """

    generator = torch.Generator(device=test_positions.device)
    generator.manual_seed(seed)

    noise = torch.randn(
        test_positions.shape[0],
        num_perturbations_per_group,
        3,
        generator=generator,
        dtype=test_positions.dtype,
        device=test_positions.device,
    )
    return test_positions.unsqueeze(1) + perturbation_std * noise


def build_normalizer_from_training_data(
    training_initials: torch.Tensor,
    test_positions: torch.Tensor,
    test_velocities: torch.Tensor,
    params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute normalization directly from all 1000 training samples.

    Each input row is [y_initial, p_n, v_n, m, g, dt]. Fixed channels receive
    standard deviation 1 to avoid division by zero.
    """

    num_groups, num_per_group, _ = training_initials.shape

    p_repeated = test_positions.unsqueeze(1).expand(-1, num_per_group, -1)
    v_repeated = test_velocities.unsqueeze(1).expand(-1, num_per_group, -1)
    params_repeated = params.view(1, 1, 3).expand(num_groups, num_per_group, -1)

    rows = torch.cat(
        [training_initials, p_repeated, v_repeated, params_repeated],
        dim=-1,
    ).reshape(-1, 12)

    mean = rows.mean(dim=0)
    std = rows.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


def save_dataset(
    output_dir: Path,
    test_times: torch.Tensor,
    test_positions: torch.Tensor,
    test_velocities: torch.Tensor,
    training_initials: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
) -> Path:
    path = output_dir / "free_fall_dataset.npz"
    np.savez_compressed(
        path,
        test_times=test_times.detach().cpu().numpy(),
        test_positions=test_positions.detach().cpu().numpy(),
        test_velocities=test_velocities.detach().cpu().numpy(),
        training_initials=training_initials.detach().cpu().numpy(),
        input_mean=input_mean.detach().cpu().numpy(),
        input_std=input_std.detach().cpu().numpy(),
    )
    return path


# ============================================================
# 4. Optimizer construction
# ============================================================


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    normalized_name = optimizer_name.lower()

    if normalized_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)

    if normalized_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name!r}. Expected 'sgd' or 'adam'."
    )


# ============================================================
# 5. Evaluation helpers
# ============================================================


def run_mlp_iterations(
    mlp: MLPOptimizer,
    initial_y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    params: torch.Tensor,
    num_steps: int,
) -> Dict[str, List]:
    """Run a frozen MLP optimizer and record step 0 through step num_steps."""

    history = torch.cat([p_n, v_n])
    y = initial_y.clone()

    ys = [y.detach().cpu().tolist()]
    residuals = [float(stationarity_residual_norm(y, p_n, v_n).item())]
    losses = [float(variational_energy(y, p_n, v_n).item())]

    for _ in range(num_steps):
        with torch.no_grad():
            delta = mlp(y, history, params)
            y = y + delta

        ys.append(y.detach().cpu().tolist())
        residuals.append(float(stationarity_residual_norm(y, p_n, v_n).item()))
        losses.append(float(variational_energy(y, p_n, v_n).item()))

    return {
        "ys": ys,
        "residuals": residuals,
        "losses": losses,
    }


def run_newton_iterations(
    initial_y: torch.Tensor,
    p_n: torch.Tensor,
    v_n: torch.Tensor,
    num_steps: int,
) -> Dict[str, List]:
    """Run Newton iterations and record step 0 through step num_steps."""

    y = initial_y.clone()

    ys = [y.detach().cpu().tolist()]
    residuals = [float(stationarity_residual_norm(y, p_n, v_n).item())]
    losses = [float(variational_energy(y, p_n, v_n).item())]

    for _ in range(num_steps):
        with torch.no_grad():
            y = y + newton_direction(y, p_n, v_n)

        ys.append(y.detach().cpu().tolist())
        residuals.append(float(stationarity_residual_norm(y, p_n, v_n).item()))
        losses.append(float(variational_energy(y, p_n, v_n).item()))

    return {
        "ys": ys,
        "residuals": residuals,
        "losses": losses,
    }


def evaluate_all_test_frames(
    mlp: MLPOptimizer,
    test_positions: torch.Tensor,
    test_velocities: torch.Tensor,
    params: torch.Tensor,
    num_steps: int,
) -> Dict[str, np.ndarray]:
    """
    Evaluate 100 physical states x num_steps learned-optimizer iterations.

    For every physical frame, the optimization initial guess is y^(0) = p_n.
    """

    mlp_residuals = []
    newton_residuals = []
    mlp_positions = []
    newton_positions = []
    implicit_targets = []

    for p_n, v_n in zip(test_positions, test_velocities):
        initial_y = p_n.clone()

        mlp_history = run_mlp_iterations(
            mlp=mlp,
            initial_y=initial_y,
            p_n=p_n,
            v_n=v_n,
            params=params,
            num_steps=num_steps,
        )
        newton_history = run_newton_iterations(
            initial_y=initial_y,
            p_n=p_n,
            v_n=v_n,
            num_steps=num_steps,
        )

        mlp_residuals.append(mlp_history["residuals"])
        newton_residuals.append(newton_history["residuals"])
        mlp_positions.append(mlp_history["ys"])
        newton_positions.append(newton_history["ys"])
        implicit_targets.append(
            implicit_euler_exact_solution(p_n, v_n).detach().cpu().tolist()
        )

    return {
        "mlp_residuals": np.asarray(mlp_residuals, dtype=np.float32),
        "newton_residuals": np.asarray(newton_residuals, dtype=np.float32),
        "mlp_positions": np.asarray(mlp_positions, dtype=np.float32),
        "newton_positions": np.asarray(newton_positions, dtype=np.float32),
        "implicit_targets": np.asarray(implicit_targets, dtype=np.float32),
    }


def summarize_residual_matrix(residuals: np.ndarray) -> Dict[str, List[float]]:
    return {
        "mean": residuals.mean(axis=0).astype(float).tolist(),
        "max": residuals.max(axis=0).astype(float).tolist(),
        "min": residuals.min(axis=0).astype(float).tolist(),
    }


def simulate_recursive_motion(
    mlp: MLPOptimizer,
    params: torch.Tensor,
    initial_position: Sequence[float],
    initial_velocity: Sequence[float],
    num_updates: int,
    mlp_solver_steps: int,
) -> Dict[str, np.ndarray]:
    """
    Simulate num_updates implicit-Euler motion updates recursively.

    Both MLP and Newton begin from the same physical state. At each time step,
    the learned optimization starts from y^(0) = current position. Newton uses
    one exact Newton update for the same variational problem.
    """

    p_mlp = torch.tensor(initial_position, dtype=TORCH_DTYPE)
    v_mlp = torch.tensor(initial_velocity, dtype=TORCH_DTYPE)
    p_newton = p_mlp.clone()
    v_newton = v_mlp.clone()

    mlp_positions = [p_mlp.detach().cpu().tolist()]
    mlp_velocities = [v_mlp.detach().cpu().tolist()]
    newton_positions = [p_newton.detach().cpu().tolist()]
    newton_velocities = [v_newton.detach().cpu().tolist()]

    for _ in range(num_updates):
        history_mlp = torch.cat([p_mlp, v_mlp])
        y_mlp = p_mlp.clone()

        for _ in range(mlp_solver_steps):
            with torch.no_grad():
                y_mlp = y_mlp + mlp(y_mlp, history_mlp, params)

        next_p_mlp = y_mlp
        next_v_mlp = (next_p_mlp - p_mlp) / DT

        y_newton = p_newton.clone()
        with torch.no_grad():
            next_p_newton = y_newton + newton_direction(y_newton, p_newton, v_newton)
        next_v_newton = (next_p_newton - p_newton) / DT

        p_mlp = next_p_mlp.detach()
        v_mlp = next_v_mlp.detach()
        p_newton = next_p_newton.detach()
        v_newton = next_v_newton.detach()

        mlp_positions.append(p_mlp.cpu().tolist())
        mlp_velocities.append(v_mlp.cpu().tolist())
        newton_positions.append(p_newton.cpu().tolist())
        newton_velocities.append(v_newton.cpu().tolist())

    analytic_times, analytic_positions, analytic_velocities = analytic_free_fall_states(
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        dt=DT,
        num_frames=num_updates + 1,
        g=GRAVITY,
    )

    return {
        "times": analytic_times.detach().cpu().numpy(),
        "analytic_positions": analytic_positions.detach().cpu().numpy(),
        "analytic_velocities": analytic_velocities.detach().cpu().numpy(),
        "mlp_positions": np.asarray(mlp_positions, dtype=np.float32),
        "mlp_velocities": np.asarray(mlp_velocities, dtype=np.float32),
        "newton_positions": np.asarray(newton_positions, dtype=np.float32),
        "newton_velocities": np.asarray(newton_velocities, dtype=np.float32),
    }


# ============================================================
# 6. Plotting helpers
# ============================================================


def set_equal_3d_axes(ax, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    center = points.mean(axis=0)
    span = np.ptp(points, axis=0)
    radius = max(float(span.max()) / 2.0, 1e-6)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_training_points_and_results(
    training_points: np.ndarray,
    result_points: np.ndarray,
    analytic_test_positions: np.ndarray,
    implicit_targets: np.ndarray,
    epochs: int,
    save_path: Path,
) -> None:
    """
    Plot all visited learned-optimizer input points and every sample's final point.

    training_points columns:
        epoch, bucket, group_index, sample_index, iteration_index, x, y, z
    result_points columns:
        epoch, bucket, group_index, sample_index, x, y, z
    """

    num_buckets = (epochs + COLOR_BUCKET_SIZE - 1) // COLOR_BUCKET_SIZE
    cmap = plt.get_cmap("tab10", max(num_buckets, 1))

    fig = plt.figure(figsize=(20, 8))
    ax_train = fig.add_subplot(121, projection="3d")
    ax_result = fig.add_subplot(122, projection="3d")

    for bucket in range(num_buckets):
        train_selected = training_points[training_points[:, 1] == bucket]
        result_selected = result_points[result_points[:, 1] == bucket]
        color = cmap(bucket)

        if train_selected.size:
            ax_train.scatter(
                train_selected[:, 5],
                train_selected[:, 6],
                train_selected[:, 7],
                s=8,
                alpha=0.22,
                color=color,
            )

        if result_selected.size:
            ax_result.scatter(
                result_selected[:, 4],
                result_selected[:, 5],
                result_selected[:, 6],
                marker="^",
                s=14,
                alpha=0.46,
                color=color,
            )

    for ax in (ax_train, ax_result):
        ax.plot(
            analytic_test_positions[:, 0],
            analytic_test_positions[:, 1],
            analytic_test_positions[:, 2],
            linewidth=1.8,
            label="Analytic test trajectory",
        )
        ax.plot(
            implicit_targets[:, 0],
            implicit_targets[:, 1],
            implicit_targets[:, 2],
            linestyle="--",
            linewidth=1.5,
            label="Implicit-Euler targets",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True, alpha=0.3)

    ax_train.set_title(r"All Training Input Points $y^{(0)},\ldots,y^{(K-1)}$")
    ax_result.set_title(r"Final Result Points $y^{(K)}$ for Every Trained Sample")

    all_points = np.vstack(
        [
            training_points[:, 5:8],
            result_points[:, 4:7],
            analytic_test_positions,
            implicit_targets,
        ]
    )
    set_equal_3d_axes(ax_train, all_points)
    set_equal_3d_axes(ax_result, all_points)

    color_handles = []
    for bucket in range(num_buckets):
        start_epoch = bucket * COLOR_BUCKET_SIZE
        end_epoch = min((bucket + 1) * COLOR_BUCKET_SIZE - 1, epochs - 1)
        color_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(bucket),
                markersize=8,
                label=f"Epoch {start_epoch}-{end_epoch}",
            )
        )

    fig.legend(
        handles=color_handles,
        loc="center right",
        bbox_to_anchor=(0.995, 0.50),
    )
    fig.suptitle("Training Data and Result Distribution by Epoch Range", fontsize=14)
    plt.tight_layout(rect=[0.00, 0.00, 0.86, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_periodic_evaluation(
    periodic_eval_log: List[Dict],
    save_path: Path,
) -> None:
    epochs = [item["epoch"] for item in periodic_eval_log]
    mean_residuals = [item["mlp_final_mean_residual"] for item in periodic_eval_log]
    max_residuals = [item["mlp_final_max_residual"] for item in periodic_eval_log]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, mean_residuals, marker="o", label="MLP mean residual")
    ax.plot(epochs, max_residuals, marker="s", label="MLP max residual")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Stationarity residual $\|\nabla E(y)\|_2$")
    ax.set_title("Periodic Frozen Evaluation on 100 Analytic Test Frames")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_residual_comparison(
    evaluation: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Plot the requested 100-frame one-step residual comparison."""

    mlp = np.maximum(evaluation["mlp_residuals"], PLOT_FLOOR)
    newton = np.maximum(evaluation["newton_residuals"], PLOT_FLOOR)
    steps = np.arange(mlp.shape[1])

    fig = plt.figure(figsize=(17, 6))
    ax_curve = fig.add_subplot(121)
    ax_heatmap = fig.add_subplot(122)

    # Individual frames are intentionally faint. Mean and worst-case curves carry
    # the main comparison signal.
    for frame_index in range(mlp.shape[0]):
        ax_curve.plot(steps, mlp[frame_index], linewidth=0.7, alpha=0.13)

    ax_curve.plot(
        steps,
        mlp.mean(axis=0),
        marker="o",
        linewidth=2.2,
        label="MLP mean over 100 frames",
    )
    ax_curve.plot(
        steps,
        mlp.max(axis=0),
        marker="^",
        linewidth=2.0,
        label="MLP worst frame",
    )
    ax_curve.plot(
        steps,
        newton.mean(axis=0),
        marker="s",
        linestyle="--",
        linewidth=2.0,
        label="Newton mean over 100 frames",
    )
    ax_curve.plot(
        steps,
        newton.max(axis=0),
        marker="v",
        linestyle="--",
        linewidth=1.8,
        label="Newton worst frame",
    )
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("Learned-optimizer / Newton iteration")
    ax_curve.set_ylabel(r"Stationarity residual $\|\nabla E(y)\|_2$")
    ax_curve.set_title("Residual Comparison: 100 Test Frames x 1 Update")
    ax_curve.legend()
    ax_curve.grid(True, alpha=0.3)

    heatmap = ax_heatmap.imshow(
        np.log10(mlp),
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    ax_heatmap.set_xlabel("MLP iteration")
    ax_heatmap.set_ylabel("Analytic test-frame index")
    ax_heatmap.set_title(r"MLP residual heatmap: $\log_{10}\|\nabla E(y)\|_2$")
    colorbar = fig.colorbar(heatmap, ax=ax_heatmap)
    colorbar.set_label(r"$\log_{10}$ residual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_motion_trajectory(
    motion: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    analytic = motion["analytic_positions"]
    mlp = motion["mlp_positions"]
    newton = motion["newton_positions"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(analytic[:, 0], analytic[:, 1], analytic[:, 2], label="Analytic motion")
    ax.plot(mlp[:, 0], mlp[:, 1], mlp[:, 2], label="MLP recursive simulation")
    ax.plot(
        newton[:, 0],
        newton[:, 1],
        newton[:, 2],
        linestyle="--",
        label="Newton implicit-Euler simulation",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Recursive Free-Fall Motion: Analytic vs. MLP vs. Newton")
    ax.legend()
    ax.grid(True, alpha=0.3)
    set_equal_3d_axes(ax, np.vstack([analytic, mlp, newton]))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_motion_video(
    motion: Dict[str, np.ndarray],
    output_dir: Path,
    fps: int,
) -> Path:
    """
    Save a 101-frame comparison video.

    Prefer MP4 through ffmpeg. Fall back to GIF through Pillow when ffmpeg is
    unavailable on the current machine.
    """

    times = motion["times"]
    analytic = motion["analytic_positions"]
    mlp = motion["mlp_positions"]
    newton = motion["newton_positions"]
    all_points = np.vstack([analytic, mlp, newton])

    center = all_points.mean(axis=0)
    span = np.ptp(all_points, axis=0)
    radius = max(float(span.max()) / 2.0, 1e-6)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Recursive Free-Fall Motion")
    ax.grid(True, alpha=0.3)

    analytic_line, = ax.plot([], [], [], linewidth=1.5, label="Analytic motion")
    mlp_line, = ax.plot([], [], [], linewidth=1.8, label="MLP recursive simulation")
    newton_line, = ax.plot(
        [], [], [], linestyle="--", linewidth=1.8, label="Newton implicit-Euler simulation"
    )
    analytic_marker, = ax.plot([], [], [], marker="o", linestyle="None")
    mlp_marker, = ax.plot([], [], [], marker="^", linestyle="None")
    newton_marker, = ax.plot([], [], [], marker="s", linestyle="None")
    time_text = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)
    ax.legend()

    def update(frame_index: int):
        end = frame_index + 1

        analytic_line.set_data(analytic[:end, 0], analytic[:end, 1])
        analytic_line.set_3d_properties(analytic[:end, 2])
        mlp_line.set_data(mlp[:end, 0], mlp[:end, 1])
        mlp_line.set_3d_properties(mlp[:end, 2])
        newton_line.set_data(newton[:end, 0], newton[:end, 1])
        newton_line.set_3d_properties(newton[:end, 2])

        analytic_marker.set_data([analytic[frame_index, 0]], [analytic[frame_index, 1]])
        analytic_marker.set_3d_properties([analytic[frame_index, 2]])
        mlp_marker.set_data([mlp[frame_index, 0]], [mlp[frame_index, 1]])
        mlp_marker.set_3d_properties([mlp[frame_index, 2]])
        newton_marker.set_data([newton[frame_index, 0]], [newton[frame_index, 1]])
        newton_marker.set_3d_properties([newton[frame_index, 2]])

        time_text.set_text(f"Frame {frame_index:03d} / {len(times) - 1:03d}    t={times[frame_index]:.2f} s")

        return (
            analytic_line,
            mlp_line,
            newton_line,
            analytic_marker,
            mlp_marker,
            newton_marker,
            time_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=1000.0 / fps,
        blit=False,
    )

    if writers.is_available("ffmpeg"):
        save_path = output_dir / "motion_comparison_101_frames.mp4"
        writer = FFMpegWriter(fps=fps, metadata={"artist": "OpenAI"}, bitrate=2400)
    else:
        save_path = output_dir / "motion_comparison_101_frames.gif"
        writer = PillowWriter(fps=fps)

    animation.save(save_path, writer=writer)
    plt.close(fig)
    return save_path


def plot_optimizer_comparison(
    summaries: List[Dict],
    save_path: Path,
) -> None:
    """Cross-optimizer summary using final frozen 100-frame evaluation."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for summary in summaries:
        steps = np.arange(len(summary["final_test_mlp_residual_mean_by_step"]))
        label = f"{summary['optimizer_name'].upper()} lr={summary['learning_rate']:.0e}"
        axes[0].plot(
            steps,
            np.maximum(summary["final_test_mlp_residual_mean_by_step"], PLOT_FLOOR),
            marker="o",
            label=label,
        )
        axes[1].plot(
            steps,
            np.maximum(summary["final_test_mlp_residual_max_by_step"], PLOT_FLOOR),
            marker="s",
            label=label,
        )

    axes[0].set_title("MLP Mean Residual over 100 Test Frames")
    axes[1].set_title("MLP Worst Residual over 100 Test Frames")

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("MLP iteration")
        ax.set_ylabel(r"Stationarity residual $\|\nabla E(y)\|_2$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 7. Single experiment
# ============================================================


def run_experiment(
    config: ExperimentConfig,
    base_output_dir: Path,
    test_times: torch.Tensor,
    test_positions: torch.Tensor,
    test_velocities: torch.Tensor,
    training_initials: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    params: torch.Tensor,
) -> Dict:
    output_dir = base_output_dir / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print(f"Experiment: {config.name}")
    print(f"Output directory: {output_dir}")
    print("=" * 78)

    torch.manual_seed(MODEL_RANDOM_SEED)

    mlp = MLPOptimizer(
        use_normalization=USE_NORMALIZATION,
        use_dt_scaling=USE_DT_SCALING,
        input_mean=input_mean,
        input_std=input_std,
    )
    optimizer = create_optimizer(
        model=mlp,
        optimizer_name=config.optimizer_name,
        learning_rate=config.learning_rate,
    )

    group_generator = torch.Generator(device=test_positions.device)
    group_generator.manual_seed(GROUP_SAMPLING_RANDOM_SEED)

    training_point_rows: List[List[float]] = []
    result_point_rows: List[List[float]] = []
    micro_step_rows: List[List[float]] = []
    selected_group_rows: List[List[int]] = []
    train_log: List[Dict] = []
    periodic_eval_log: List[Dict] = []

    k = INITIAL_K

    for epoch in range(EPOCHS):
        if (
            epoch > 0
            and epoch % K_INCREASE_INTERVAL == 0
            and k < MAX_K
        ):
            k = min(k + K_INCREASE_AMOUNT, MAX_K)

        bucket = epoch // COLOR_BUCKET_SIZE
        group_index = int(
            torch.randint(
                low=0,
                high=NUM_TEST_FRAMES,
                size=(1,),
                generator=group_generator,
            ).item()
        )
        selected_group_rows.append([epoch, group_index, k])

        p_n = test_positions[group_index]
        v_n = test_velocities[group_index]
        history = torch.cat([p_n, v_n])
        e_star = float(variational_energy(
            implicit_euler_exact_solution(p_n, v_n), p_n, v_n
        ).item())

        sample_final_losses = []
        sample_final_gaps = []
        sample_final_residuals = []

        # The chosen physical-state group contains exactly ten perturbed
        # optimization initial guesses. Train on them sequentially.
        for sample_index, initial_y in enumerate(training_initials[group_index]):
            y = initial_y.clone()

            for iteration_index in range(k):
                y_before = y.detach().clone()
                training_point_rows.append(
                    [
                        epoch,
                        bucket,
                        group_index,
                        sample_index,
                        iteration_index,
                        float(y_before[0].item()),
                        float(y_before[1].item()),
                        float(y_before[2].item()),
                    ]
                )

                optimizer.zero_grad(set_to_none=True)
                delta = mlp(y, history, params)
                y_next = y + delta
                loss = variational_energy(y_next, p_n, v_n)

                # One backward pass and one network update per learned-optimizer
                # iteration. The state is detached immediately afterward.
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    residual_norm = float(
                        stationarity_residual_norm(y_next, p_n, v_n).item()
                    )
                    objective_gap = max(float(loss.item()) - e_star, 0.0)

                micro_step_rows.append(
                    [
                        epoch,
                        bucket,
                        group_index,
                        sample_index,
                        iteration_index + 1,
                        float(loss.item()),
                        objective_gap,
                        residual_norm,
                    ]
                )

                y = y_next.detach()

            final_loss = float(variational_energy(y, p_n, v_n).item())
            final_gap = max(final_loss - e_star, 0.0)
            final_residual = float(stationarity_residual_norm(y, p_n, v_n).item())

            result_point_rows.append(
                [
                    epoch,
                    bucket,
                    group_index,
                    sample_index,
                    float(y[0].item()),
                    float(y[1].item()),
                    float(y[2].item()),
                ]
            )
            sample_final_losses.append(final_loss)
            sample_final_gaps.append(final_gap)
            sample_final_residuals.append(final_residual)

        epoch_item = {
            "epoch": epoch,
            "K": k,
            "selected_group_index": group_index,
            "selected_time": float(test_times[group_index].item()),
            "mean_final_loss": float(np.mean(sample_final_losses)),
            "max_final_loss": float(np.max(sample_final_losses)),
            "mean_final_gap": float(np.mean(sample_final_gaps)),
            "max_final_gap": float(np.max(sample_final_gaps)),
            "mean_final_residual": float(np.mean(sample_final_residuals)),
            "max_final_residual": float(np.max(sample_final_residuals)),
        }
        train_log.append(epoch_item)

        if epoch % PERIODIC_EVAL_INTERVAL == 0 or epoch == EPOCHS - 1:
            frozen_eval = evaluate_all_test_frames(
                mlp=mlp,
                test_positions=test_positions,
                test_velocities=test_velocities,
                params=params,
                num_steps=FINAL_TEST_STEPS,
            )
            final_mlp_residuals = frozen_eval["mlp_residuals"][:, -1]
            final_newton_residuals = frozen_eval["newton_residuals"][:, -1]

            periodic_item = {
                "epoch": epoch,
                "K": k,
                "mlp_final_mean_residual": float(final_mlp_residuals.mean()),
                "mlp_final_max_residual": float(final_mlp_residuals.max()),
                "newton_final_mean_residual": float(final_newton_residuals.mean()),
                "newton_final_max_residual": float(final_newton_residuals.max()),
            }
            periodic_eval_log.append(periodic_item)

            print(
                f"Epoch {epoch:4d} | K={k:2d} | group={group_index:3d} | "
                f"train max residual={epoch_item['max_final_residual']:.4e} | "
                f"test mean residual={periodic_item['mlp_final_mean_residual']:.4e} | "
                f"test max residual={periodic_item['mlp_final_max_residual']:.4e}"
            )

    training_points_np = np.asarray(training_point_rows, dtype=np.float32)
    result_points_np = np.asarray(result_point_rows, dtype=np.float32)
    micro_steps_np = np.asarray(micro_step_rows, dtype=np.float32)
    selected_groups_np = np.asarray(selected_group_rows, dtype=np.int32)

    implicit_targets = torch.stack(
        [
            implicit_euler_exact_solution(p_n, v_n)
            for p_n, v_n in zip(test_positions, test_velocities)
        ],
        dim=0,
    ).detach().cpu().numpy()

    training_log_path = output_dir / "detailed_training_logs.npz"
    np.savez_compressed(
        training_log_path,
        training_points=training_points_np,
        result_points=result_points_np,
        micro_steps=micro_steps_np,
        selected_groups=selected_groups_np,
    )

    plot_training_points_and_results(
        training_points=training_points_np,
        result_points=result_points_np,
        analytic_test_positions=test_positions.detach().cpu().numpy(),
        implicit_targets=implicit_targets,
        epochs=EPOCHS,
        save_path=output_dir / "training_points_and_results_distribution.png",
    )

    plot_periodic_evaluation(
        periodic_eval_log=periodic_eval_log,
        save_path=output_dir / "periodic_test_residuals.png",
    )

    final_evaluation = evaluate_all_test_frames(
        mlp=mlp,
        test_positions=test_positions,
        test_velocities=test_velocities,
        params=params,
        num_steps=FINAL_TEST_STEPS,
    )
    final_eval_path = output_dir / "final_test_100_frames_1_step.npz"
    np.savez_compressed(final_eval_path, **final_evaluation)

    plot_final_residual_comparison(
        evaluation=final_evaluation,
        save_path=output_dir / "final_test_residual_comparison_100_frames_1_step.png",
    )

    motion = simulate_recursive_motion(
        mlp=mlp,
        params=params,
        initial_position=INITIAL_POSITION,
        initial_velocity=INITIAL_VELOCITY,
        num_updates=NUM_TEST_FRAMES,
        mlp_solver_steps=MOTION_SOLVER_STEPS,
    )
    motion_path = output_dir / "motion_comparison_101_frames.npz"
    np.savez_compressed(motion_path, **motion)

    plot_motion_trajectory(
        motion=motion,
        save_path=output_dir / "motion_comparison_101_frames.png",
    )

    video_path = None
    if SAVE_VIDEO:
        video_path = save_motion_video(
            motion=motion,
            output_dir=output_dir,
            fps=VIDEO_FPS,
        )

    model_path = output_dir / "mlp_optimizer_state_dict.pt"
    torch.save(mlp.state_dict(), model_path)

    mlp_residual_summary = summarize_residual_matrix(final_evaluation["mlp_residuals"])
    newton_residual_summary = summarize_residual_matrix(
        final_evaluation["newton_residuals"]
    )

    final_mlp_positions = motion["mlp_positions"]
    final_newton_positions = motion["newton_positions"]
    final_analytic_positions = motion["analytic_positions"]

    report = {
        "config": {
            **asdict(config),
            "experiment_name": config.name,
            "torch_dtype": str(TORCH_DTYPE),
            "mass": MASS,
            "gravity": GRAVITY,
            "dt": DT,
            "initial_position": list(INITIAL_POSITION),
            "initial_velocity": list(INITIAL_VELOCITY),
            "num_test_frames": NUM_TEST_FRAMES,
            "test_time_range": [
                float(test_times[0].item()),
                float(test_times[-1].item()),
            ],
            "test_endpoint_at_1s_excluded": True,
            "num_groups": int(training_initials.shape[0]),
            "num_perturbations_per_group": int(training_initials.shape[1]),
            "num_training_samples": int(training_initials.shape[0] * training_initials.shape[1]),
            "train_perturbation_std": TRAIN_PERTURBATION_STD,
            "normalization_source": "all_1000_training_samples",
            "use_normalization": USE_NORMALIZATION,
            "use_dt_scaling": USE_DT_SCALING,
            "epochs": EPOCHS,
            "initial_K": INITIAL_K,
            "K_increase_interval": K_INCREASE_INTERVAL,
            "K_increase_amount": K_INCREASE_AMOUNT,
            "max_K": MAX_K,
            "color_bucket_size": COLOR_BUCKET_SIZE,
            "periodic_eval_interval": PERIODIC_EVAL_INTERVAL,
            "final_test_steps": FINAL_TEST_STEPS,
            "motion_solver_steps": MOTION_SOLVER_STEPS,
            "training_rule": (
                "At every epoch randomly choose one physical-state group. "
                "Train sequentially on its ten perturbed optimization initials. "
                "For every sample and every optimizer iteration, compute one "
                "loss, call backward once, call optimizer.step once, then detach "
                "the iterative state before the next learned-optimizer step."
            ),
        },
        "input_normalizer": {
            "mean": input_mean.detach().cpu().tolist(),
            "std": input_std.detach().cpu().tolist(),
        },
        "training_log": train_log,
        "periodic_evaluation": periodic_eval_log,
        "final_test_residuals": {
            "mlp": mlp_residual_summary,
            "newton": newton_residual_summary,
        },
        "recursive_motion": {
            "final_mlp_position": final_mlp_positions[-1].astype(float).tolist(),
            "final_newton_position": final_newton_positions[-1].astype(float).tolist(),
            "final_analytic_position": final_analytic_positions[-1].astype(float).tolist(),
            "mlp_final_position_error_vs_newton": float(
                np.linalg.norm(final_mlp_positions[-1] - final_newton_positions[-1])
            ),
            "mlp_final_position_error_vs_analytic": float(
                np.linalg.norm(final_mlp_positions[-1] - final_analytic_positions[-1])
            ),
            "newton_final_position_error_vs_analytic": float(
                np.linalg.norm(final_newton_positions[-1] - final_analytic_positions[-1])
            ),
        },
        "artifacts": {
            "training_log_npz": str(training_log_path),
            "final_evaluation_npz": str(final_eval_path),
            "motion_npz": str(motion_path),
            "model_path": str(model_path),
            "video_path": str(video_path) if video_path is not None else None,
        },
    }

    report_path = output_dir / "optimization_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Saved report: {report_path}")
    if video_path is not None:
        print(f"Saved 101-frame motion video: {video_path}")

    return {
        "experiment_name": config.name,
        "optimizer_name": config.optimizer_name,
        "learning_rate": config.learning_rate,
        "output_directory": str(output_dir),
        "final_test_mlp_residual_mean_by_step": mlp_residual_summary["mean"],
        "final_test_mlp_residual_max_by_step": mlp_residual_summary["max"],
        "final_test_newton_residual_mean_by_step": newton_residual_summary["mean"],
        "final_test_newton_residual_max_by_step": newton_residual_summary["max"],
        "mlp_final_position_error_vs_newton": report["recursive_motion"][
            "mlp_final_position_error_vs_newton"
        ],
        "mlp_final_position_error_vs_analytic": report["recursive_motion"][
            "mlp_final_position_error_vs_analytic"
        ],
        "newton_final_position_error_vs_analytic": report["recursive_motion"][
            "newton_final_position_error_vs_analytic"
        ],
        "video_path": str(video_path) if video_path is not None else None,
    }


# ============================================================
# 8. Main program
# ============================================================


def main() -> None:
    base_output_dir = create_output_directory()
    print(f"Base output directory: {base_output_dir}")
    print(f"Torch default dtype: {torch.get_default_dtype()}")

    test_times, test_positions, test_velocities = analytic_free_fall_states(
        initial_position=INITIAL_POSITION,
        initial_velocity=INITIAL_VELOCITY,
        dt=DT,
        num_frames=NUM_TEST_FRAMES,
        g=GRAVITY,
    )
    training_initials = build_training_groups(
        test_positions=test_positions,
        perturbation_std=TRAIN_PERTURBATION_STD,
        num_perturbations_per_group=NUM_PERTURBATIONS_PER_GROUP,
        seed=DATASET_RANDOM_SEED,
    )
    params = torch.tensor([MASS, GRAVITY, DT], dtype=TORCH_DTYPE)
    input_mean, input_std = build_normalizer_from_training_data(
        training_initials=training_initials,
        test_positions=test_positions,
        test_velocities=test_velocities,
        params=params,
    )

    dataset_path = save_dataset(
        output_dir=base_output_dir,
        test_times=test_times,
        test_positions=test_positions,
        test_velocities=test_velocities,
        training_initials=training_initials,
        input_mean=input_mean,
        input_std=input_std,
    )

    print(f"Saved shared dataset: {dataset_path}")
    print(f"Test states: {test_positions.shape[0]}")
    print(
        "Training samples: "
        f"{training_initials.shape[0]} groups x "
        f"{training_initials.shape[1]} perturbations = "
        f"{training_initials.shape[0] * training_initials.shape[1]}"
    )

    summaries = []
    for item in OPTIMIZER_CONFIGS:
        config = ExperimentConfig(
            optimizer_name=item["optimizer_name"],
            learning_rate=float(item["learning_rate"]),
        )
        summary = run_experiment(
            config=config,
            base_output_dir=base_output_dir,
            test_times=test_times,
            test_positions=test_positions,
            test_velocities=test_velocities,
            training_initials=training_initials,
            input_mean=input_mean,
            input_std=input_std,
            params=params,
        )
        summaries.append(summary)

    plot_optimizer_comparison(
        summaries=summaries,
        save_path=base_output_dir / "optimizer_comparison_final_residuals.png",
    )

    summary_report = {
        "experiment_type": "free_fall_trajectory_learned_iterative_optimizer",
        "shared_dataset_path": str(dataset_path),
        "torch_dtype": str(TORCH_DTYPE),
        "optimizer_configs": OPTIMIZER_CONFIGS,
        "experiments": summaries,
    }
    summary_path = base_output_dir / "experiment_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary_report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print("All optimizer experiments completed.")
    print(f"Summary report: {summary_path}")
    print(f"Cross-optimizer residual plot: {base_output_dir / 'optimizer_comparison_final_residuals.png'}")
    for summary in summaries:
        print(
            f"- {summary['experiment_name']}: "
            f"MLP final mean residual="
            f"{summary['final_test_mlp_residual_mean_by_step'][-1]:.4e}, "
            f"MLP final max residual="
            f"{summary['final_test_mlp_residual_max_by_step'][-1]:.4e}, "
            f"MLP motion final error vs Newton="
            f"{summary['mlp_final_position_error_vs_newton']:.4e}"
        )


if __name__ == "__main__":
    main()
