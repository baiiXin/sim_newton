# GNN implicit Euler training variants

Place these files in the same directory as:

- `GNN_solver.py`
- `loss_class.py`

Then run, for example:

```bash
conda activate hood
cd /data/zhoucy/sim_newton/GNN_solver/_src
python train_01_no_pretrain_finetune5000_iter_backward.py --device cuda
python train_02_pre1000_fine2000_iter_backward.py --device cuda
python train_03_pre1000_fine2000_timestep_backward.py --device cuda
python train_04_pre1000_fine1000_iter_then1000_timestep.py --device cuda
```

`--device auto` is the default and uses CUDA if available.

Each experiment writes distinct files:

- `{experiment_name}_eval_log.csv`
- `{experiment_name}_final.pt`

The evaluation protocol is the same for all variants:

- every `--test-every` epochs, default 10,
- start from both initial states: `x_prev` and `x_hat`,
- run 15 iterations,
- after every iteration compute and log total loss and residual.


1. train_01_no_pretrain_finetune5000_iter_backward.py
   不做预训练
   后训练 5000 epoch
   后训练每个 iteration 都 backward + optimizer.step

2. train_02_pre1000_fine2000_iter_backward.py
   预训练 1000 epoch
   后训练 2000 epoch
   后训练每个 iteration 都 backward + optimizer.step

3. train_03_pre1000_fine2000_timestep_backward.py
   预训练 1000 epoch
   后训练 2000 epoch
   后训练每个 time step 只 backward 一次
   即 unroll 10 次后，对最终 x 计算 loss，再 backward + step

4. train_04_pre1000_fine1000_iter_then1000_timestep.py
   预训练 1000 epoch
   后训练先做 1000 epoch 每 iteration backward
   再做 1000 epoch 每 time step backward