# AGENTS.md

This file defines repository-specific guidance for Codex agents working in this repository.

## Scope and default focus

- By default, focus only on the `GNN_solver/` directory.
- Do not modify `cloth_simulation_newton/` unless the user explicitly asks for changes there.
- It is OK to read files under `cloth_simulation_newton/` as reference material when needed.

## Project goal

The goal of this project is to test whether a GNN can act as an iterative solver for the implicit Euler variational energy optimization problem in cloth simulation.

## Solver and loss assumptions

- The GNN outputs `delta_x`.
- Code outside the GNN applies the state update as:

  ```python
  x_next = x_cur + delta_x
  ```

- Training loss comes from `ImplicitEulerLoss`.

## Code modification rules

- Prefer small, focused changes that are easy to review.
- Do not delete old experiment scripts.
- Do not overwrite or replace existing log naming conventions.
- Preserve `--device auto/cpu/cuda` support.
- When dtype support matters, avoid hard-coding `torch.float32` and avoid forcing tensors through `.float()`.
- Evaluation must record `iter=0`.
- Training scripts must save both eval logs and train loss logs.


## GNN iterative-solver training guidance

When modifying multi-iteration training or evaluation, keep the distinction between an iterative optimizer and a one-step predictor explicit:

- The desired solver behavior is stable iterative energy reduction, not only one-step movement toward a low-energy state.
- Multi-step training should make gradient flow intentional and explicit. If each solver iteration has its own optimizer step, detach the rollout state before the next iteration so gradients from later iterations do not backpropagate through previous updates. If training through the full unrolled time step, document that choice and watch for exploding, vanishing, or conflicting gradients.
- Add or preserve diagnostics for:
  - near-optimum zero-step behavior, i.e. `delta_x` should shrink near a converged solution;
  - descent-direction statistics, e.g. whether `grad(E)^T delta_x < 0`;
  - energy after every iteration, including `iter=0`;
  - line-search or step-size ablations when direction quality and step length need to be separated.
- Prefer staged/curriculum experiments before long fully-unrolled training runs, for example single-step training before 2-step, 4-step, and 10-step training.
- If adding teacher supervision experiments, compare against simple gradient-descent and Newton-style directions before changing the network architecture.

## Validation before PR

Before opening a PR, run at least:

```bash
python -m py_compile GNN_solver/_src/*.py
```

If a smoke test cannot be run, explain the reason in the PR description.

## PR description

PR descriptions should include the following sections:

- `Summary`
- `Validation`
