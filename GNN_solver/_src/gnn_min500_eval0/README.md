# Minimal 500-epoch entry with eval iteration 0

Files:

- `train_common.py`
- `train_min500_eval_every_epoch.py`

Put them beside:

- `GNN_solver.py`
- `loss_class.py`

Run:

```bash
python train_min500_eval_every_epoch.py --device cuda
```

Default behavior:

- train 500 epochs,
- evaluate every epoch,
- evaluation starts from both `x_prev` and `x_hat`,
- evaluation logs iteration 0 before any GNN update,
- then logs iterations 1 through 15,
- default training uses 10 autoregressive iterations and backward once per solver iteration.

Output files:

- `exp_min500_eval_every_epoch_eval_log.csv`
- `exp_min500_eval_every_epoch_train_loss_log.csv`
- `exp_min500_eval_every_epoch_final.pt`

Optional variants:

```bash
python train_min500_eval_every_epoch.py --device cuda --train-iters 1
python train_min500_eval_every_epoch.py --device cuda --backward-mode time_step
```
