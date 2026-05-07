# Cloth Simulation Solver Experiments

This repository collects experiments around iterative solvers for cloth simulation. The core research question is whether a Graph Neural Network (GNN) can act as an iterative solver for the implicit Euler variational energy optimization problem in cloth simulation.

## Project directions

The repository currently contains two main directions:

- `cloth_simulation_newton/`: a traditional Newton-method cloth solver. This area still needs cleanup and is kept as a numerical reference.
- `GNN_solver/`: the actively tested GNN solver direction. It is used to test whether a GNN can serve as an iterative solver for the implicit Euler variational energy optimization problem.

The current primary test directory is:

```text
GNN_solver/_src
```

## Current GNN solver idea

The GNN is used as an iteration operator, not as a direct final-position predictor.

- The GNN does **not** directly predict the final cloth positions.
- At each solver iteration, the GNN predicts `delta_x`.
- External code applies the state update:

  ```python
  x_next = x_cur + delta_x
  ```

- The training loss comes from `ImplicitEulerLoss`.

In other words, the current experiment asks whether repeated GNN-predicted updates can reduce the implicit Euler energy and residual in the same spirit as an iterative numerical solver.

## Directory structure

```text
repo/
├── cloth_simulation_newton/
│   └── Traditional Newton solver; pending cleanup; numerical reference
│
├── GNN_solver/
│   └── _src/
│       ├── GNN_solver.py
│       ├── loss_class.py
│       ├── train_common.py
│       ├── train_min500_one_iter_one_backward.py
│       └── Other current experiment utilities/scripts
│
└── README.md
```

Notes:

- `GNN_solver/_src` is the current main location for training, evaluation, and experiment scripts.
- `cloth_simulation_newton` is not the current cleanup target, but it remains important as the future Newton-method reference.

## Current minimal training experiment

The current recommended minimal experiment is intentionally small:

- Each time step runs only **1** GNN solver iteration during training.
- Each time step performs **1** backward pass.
- Training runs for **500 epochs**.
- Evaluation runs once per epoch.
- Evaluation starts from both initial states:
  - `x_prev`
  - `x_hat`

This setup is meant to first test whether the network can behave as an energy-decreasing iterator, not whether it generalizes across many scenes or time steps.

## Current evaluation protocol

For each evaluation pass and each initial state, evaluation records the initial state before applying the GNN:

```text
iter = 0
```

`iter=0` is the loss and residual computed directly at the initial state. It is a reference point, not a GNN update result.

After that, evaluation runs 15 GNN iterations and records:

```text
iter = 1, 2, ..., 15
```

Each recorded row includes:

- `total_loss`
- `inertia`
- `gravity`
- `elastic`
- `bending`
- `residual_mean`
- `residual_max`

The main purpose is to check whether loss and residual decrease from `iter=0` to `iter=1`, and whether repeated GNN iterations remain stable through `iter=15`.

## Recommended run command

From the repository root, enter the current main test directory:

```bash
cd GNN_solver/_src
```

Run the current minimal 500-epoch experiment on CUDA:

```bash
python train_min500_one_iter_one_backward.py --device cuda
```

Other supported device modes are:

```bash
python train_min500_one_iter_one_backward.py --device cpu
python train_min500_one_iter_one_backward.py --device auto
```

Use `--device auto` when you want the script to choose an available device automatically.

## Current experiment status and interpretation

The current setup is a minimal, controlled experiment:

- The training set currently has only one time step and two initial states.
- The goal is to validate whether the GNN can act as an energy-decreasing iteration operator.
- This is **not** yet a generalization test.
- If `v_prev = 0`, then `x_hat == x_prev`; in that case, the two nominal initial states are identical.

When reviewing results, pay special attention to:

1. Whether `iter=0` to `iter=1` decreases `total_loss` and residual.
2. Whether `iter=1` through `iter=15` remain stable.
3. Whether behavior is consistent from both `x_prev` and `x_hat`.
4. Whether any energy component or residual metric becomes unstable, NaN, or Inf.

## Notes on dtype and precision

Future experiments may test whether `float64` is more stable than `float32` for this optimization problem.

If switching to `float64`, make sure dtype is consistent across:

- input data;
- model parameters and intermediate tensors;
- `ImplicitEulerLoss` internals;
- any constants created inside training, evaluation, or loss code.

Avoid mixing dtypes accidentally. A partial dtype conversion can make the experiment difficult to interpret.

## Current follow-up experiment plan

Planned follow-up work:

- Test whether `float64` is more stable.
- Add an MLP baseline.
- Test whether input features need adjustment.
- If the no-collision energy experiment works, continue by adding collision energy tests.
- Later, align the GNN solver results with the Newton-method reference results in `cloth_simulation_newton`.

## Important caveats

- The current training set has only one time step and two initial states. The goal is to first verify energy-decreasing iteration behavior, not generalization.
- If `v_prev = 0`, then `x_hat == x_prev`.
- If switching to `float64`, ensure data, model, and loss internals use consistent dtype.
- `iter=0` is an evaluation reference item, not the result of a GNN update.
