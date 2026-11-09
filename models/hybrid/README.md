# Hybrid Network (data + physics loss)

Implementation of the hybrid network, the third method compared with the data-driven NN and the PINN. It learns $(x, y, t) \rightarrow (u, v, p)$ from both sources at once: the finite-difference (FD) training data, exactly as the data-driven NN sees it, and the governing equations, initial condition and boundary conditions, exactly as the PINN sees them. Every other setting is shared with the two other networks, so the three can be compared run for run.

Source code is maintained in ```src/```:

- `config.py` — training settings, all taken from `models/common/config.py` (5 hidden layers of 64 tanh neurons, Adam with learning rate $10^{-3}$, 20,000 steps, seeds 1–5, data split) plus the data batch size (4,096, as the data NN) and the data-loss weight (1)
- `model.py` — the network (the same builder as the other two methods, `models/common/network.py`)
- `train.py` — training loop and run record; the physics terms and training points are imported unchanged from `models/pinn/src/losses.py` and `models/pinn/src/config.py`

There is no `evaluate.py`: runs are scored with the same evaluator as the other networks (`experiments/analysis/evaluate_all.py`).

## Loss

$$ L = \lambda_{data} L_{data} + \lambda_{ic} L_{ic} + \lambda_{bc} L_{bc} + \lambda_c \overline{r_c^2} + \lambda_m \left( \overline{r_x^2} + \overline{r_y^2} \right) $$

- **Data part** (as the data NN): FD train split, a new mini-batch of 4,096 rows every step, shuffled once per epoch with the run seed; $L_{data}$ is the mean over $u, v, p$ of the mean squared error divided by the train-split variance of that output. For the same prediction this gives the same value as the data NN's loss on standardized targets.
- **Physics part** (as the PINN): the same 20,000 collocation points, 4,096 initial-condition points and 1,000 points per wall (sampling seed 0), the full set every step, residuals by automatic differentiation.
- **Weights**: all five start at 1. The PINN's one-time check at step 2,000 is applied to all five components with the same rule: if the largest weighted term exceeds the smallest by more than a factor of 100, every weight is rescaled so that each weighted term equals their median.

## Differences from the other two networks

- **From the PINN:** only the added data loss.
- **From the data NN:** the added physics loss, and the output units. The network outputs $u, v, p$ in physical units (as the PINN), so the residuals are taken directly from the output; the data NN outputs standardized values.

The second difference is separated by a control run (`--data-only`, below). The shared settings, the data loss and the batch order are checked by `python -m tests.test_hybrid`.

## Training

From the repository root:

```
python models/hybrid/src/train.py --device gpu --seed 1 [--re 400]
```

Options:

- `--data-only` — control run: the hybrid code with the data loss only (physics terms not computed, no weight check), physical outputs. Hybrid vs control isolates the physics loss; control vs data NN isolates the output units.
- `--steps 100000` — longer training (as S5)
- `--shared-gpu` — the GPU is shared with another job; the run is marked `"timing_valid": false` and its times are not used as cost results
- `--run-id`, `--exp-dir`, `--tag` — override the run id, the output folder, or add a suffix to the run id

The batches used for the results:

```
python experiments/run_batch.py --hybrid           # seeds 1-3 at Re 10, 100, 400, 1000; seeds 4-5 at Re 100, 400; 100k steps at Re 400
python experiments/run_batch.py --hybrid-control   # data-only control, seeds 1-3 at Re 100 and 400
python experiments/run_batch.py --day              # seeds 4-5 at Re 10 and 1000, controls at Re 10 and 1000, more 100k runs
```

Run folders:

- main runs: `experiments/experiment_00X_re<Re>/hybrid/hybrid_re<Re>_gpu_s<seed>/`
- control: `experiments/experiment_00X_re<Re>/control/hybrid/hybrid_re<Re>_gpu_s<seed>_dataonly/`
- 100,000 steps: `experiments/experiment_00X_re<Re>/extended/hybrid/hybrid_re<Re>_gpu_s<seed>_steps100k/`

## Saved Outputs

The same files as the other networks: `model.keras`, `history.csv` (each of the six loss terms and the total every 10 steps), `eval_history.csv` (relative errors against FD every 500 steps, for monitoring only), `predictions.npz`, `metrics.json` (written by `evaluate_all.py`), `performance.json`, `gpu_monitor.csv` and `config.json`. `config.json` records the starting weights, the weighted terms and their imbalance ratio at the step-2,000 check, and whether the check changed the weights.
