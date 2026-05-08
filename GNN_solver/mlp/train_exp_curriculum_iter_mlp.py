from __future__ import annotations

from pathlib import Path

from train_common import PhaseConfig, run_experiment


# -----------------------------------------------------------------------------
# Fixed experiment config
# -----------------------------------------------------------------------------

EXPERIMENT_NAME = "exp_mlp_curriculum_iter_every1000"

# Training schedule:
#   epochs 0001-1000:  train_iters = 1
#   epochs 1001-2000:  train_iters = 2
#   ...
#   epochs 9001-10000: train_iters = 10
TOTAL_EPOCHS = 10_000
CURRICULUM_EVERY = 1_000
START_TRAIN_ITERS = 1
MAX_TRAIN_ITERS = 10

# Optimization / evaluation config
DEVICE = "auto"          # or "cuda:0", "cpu"
LR = 1.0e-4
WEIGHT_DECAY = 0.0
SEED = 0
TEST_EVERY = 100
INITIAL_EVAL = True

# Save all results to GNN_solver/mlp/result
OUTPUT_DIR = Path(__file__).resolve().parent / "result"


def build_curriculum_iteration_phases():
    """
    Build curriculum phases.

    Each phase uses backward_mode="iteration", meaning:
        every solver iteration has its own loss.backward() and optimizer.step().

    This is the detached per-iteration training mode, not full time-step
    unrolled backpropagation.
    """
    phases = []

    current_train_iters = START_TRAIN_ITERS
    remaining_epochs = TOTAL_EPOCHS

    while remaining_epochs > 0:
        phase_epochs = min(CURRICULUM_EVERY, remaining_epochs)

        phases.append(
            PhaseConfig(
                phase=f"curriculum_iter_{current_train_iters:02d}",
                num_epochs=phase_epochs,
                train_iters=current_train_iters,
                backward_mode="iteration",
            )
        )

        remaining_epochs -= phase_epochs

        if current_train_iters < MAX_TRAIN_ITERS:
            current_train_iters += 1

    return phases


def main() -> None:
    phases = build_curriculum_iteration_phases()

    print("MLP curriculum iteration-backward experiment")
    print(f"Experiment name: {EXPERIMENT_NAME}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("Schedule:")
    for phase in phases:
        print(
            f"  {phase.phase}: "
            f"epochs={phase.num_epochs}, "
            f"train_iters={phase.train_iters}, "
            f"backward_mode={phase.backward_mode}"
        )

    run_experiment(
        experiment_name=EXPERIMENT_NAME,
        phases=phases,
        device=DEVICE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
        test_every=TEST_EVERY,
        initial_eval=INITIAL_EVAL,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()