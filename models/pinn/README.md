# Physics-Informed Neural Network (PINN)

Implementation of the PINN used to incorporate the physical governing equations into neural-network training. It learns $(x, y, t) \rightarrow (u, v, p)$ from the incompressible Navier-Stokes equations, the initial condition and the boundary conditions only; it never sees FD data.

Source code is maintained in ```src/```:

- `config.py` — sampling (20,000 collocation points, 4,096 initial-condition points, 1,000 points per wall), loss weights (all 1) and the one-time imbalance check at step 2,000
- `losses.py` — initial, boundary and PDE residual losses; causal weighting of the residual in time
- `model.py` — the network (same architecture as the data-driven NN)
- `train.py` — training loop and run record

The loss is

$$ L = \lambda_{ic} L_{ic} + \lambda_{bc} L_{bc} + \lambda_c \overline{r_c^2} + \lambda_m \left( \overline{r_x^2} + \overline{r_y^2} \right) $$

with the residuals computed by automatic differentiation.

## Training

From the repository root:

```
python models/pinn/src/train.py --device gpu --seed 1 [--re 400]
```

Options used by the studies:

- `--n-collocation 10000` — number of collocation points (S2)
- `--hidden-layers 6 --neurons 100` — network size (S3)
- `--steps 100000` — longer training (S5)
- `--causal` — causal weighting of the PDE residual (S4, Wang, Sankaran and Perdikaris 2022): the time interval is split into 32 chunks and chunk $i$ is weighted by $w_i = \exp(-\varepsilon \sum_{k<i} L_k)$, with $\varepsilon$ raised through $10^{-2}, 10^{-1}, 1, 10, 100$ whenever the smallest weight exceeds 0.99
- `--init-from <model.keras>` — start from the weights of a trained network with the same architecture instead of random weights; the optimizer starts fresh (S6, $Re$ continuation: the $Re = 400$ PINN started from the $Re = 100$ PINN of the same seed)
- `--shared-gpu` — the GPU is shared with another job; the run is marked `"timing_valid": false` and its times are not used as cost results
- `--run-id`, `--exp-dir`, `--tag` — override the run id, the output folder, or add a suffix to the run id

The causal weights are checked by `python -m tests.test_causal_weights`.

## Saved Outputs

The same files as the data-driven NN: `model.keras`, `history.csv` (each loss term every 10 steps), `eval_history.csv` (relative errors against FD every 500 steps, for monitoring only), `predictions.npz`, `metrics.json`, `performance.json` and `config.json`.
