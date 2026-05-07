from __future__ import annotations

import argparse

from train_common import PhaseConfig, run_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-every", type=int, default=100)
    parser.add_argument("--no-initial-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_experiment(
        experiment_name="exp03_pre1000_fine2000_timestep_backward",
        phases=[
            PhaseConfig(
                phase="pretrain_iter",
                num_epochs=1_000,
                train_iters=1,
                backward_mode="iteration",
            ),
            PhaseConfig(
                phase="finetune_timestep",
                num_epochs=2_000,
                train_iters=10,
                backward_mode="time_step",
            ),
        ],
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        test_every=args.test_every,
        initial_eval=not args.no_initial_eval,
    )


if __name__ == "__main__":
    main()
