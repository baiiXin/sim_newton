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
