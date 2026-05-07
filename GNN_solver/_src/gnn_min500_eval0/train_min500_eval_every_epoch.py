from __future__ import annotations

import argparse

from train_common import PhaseConfig, run_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'")
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    # Minimal defaults:
    #   500 epochs,
    #   evaluate after every epoch,
    #   10 autoregressive solver iterations per training sample,
    #   backward once per solver iteration.
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument(
        "--backward-mode",
        choices=["iteration", "time_step"],
        default="iteration",
        help="'iteration': backward each solver iteration; 'time_step': backward once after unroll",
    )
    parser.add_argument("--no-initial-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_experiment(
        experiment_name="exp_min500_eval_every_epoch",
        phases=[
            PhaseConfig(
                phase="train500",
                num_epochs=args.epochs,
                train_iters=args.train_iters,
                backward_mode=args.backward_mode,
            ),
        ],
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        test_every=1,
        initial_eval=not args.no_initial_eval,
    )


if __name__ == "__main__":
    main()
