# Data-Driven Neural Network

Implementation of the data-driven neural network used for comparison with the numerical solver and PINN. It learns $(x, y, t) \rightarrow (u, v, p)$ directly from the finite-difference (FD) dataset and never sees the governing equations.

Source code is maintained in ```src/```:

- `config.py` — training settings (batch 4,096, Adam with learning rate $10^{-3}$, 20,000 steps); the network size is shared with the PINN in `models/common/config.py` (5 hidden layers of 64 tanh neurons)
- `model.py` — the network
- `train.py` — training loop and run record
- `evaluate.py` — scoring of a single run

The dataset is built from the $64 \times 64$ FD transient with `python -m models.common.dataset --re 100 --build`.

## Training

From the repository root:

```
python models/data_driven_nn/src/train.py --device gpu --seed 1 [--re 400]
```

Options used by the studies:

- `--train-fraction 0.1` — train on a fraction of the training rows (S1)
- `--hidden-layers 6 --neurons 100` — network size (S3)
- `--steps 100000` — longer training (S5)
- `--shared-gpu` — the GPU is shared with another job; the run is marked `"timing_valid": false` and its times are not used as cost results
- `--run-id`, `--exp-dir` — override the run id and the output folder

## Saved Outputs

Each training run creates a new folder inside `experiments/`. Each run saves:

- `model.keras` — trained TensorFlow model
- `history.csv` — training loss every 10 steps
- `eval_history.csv` — validation loss and relative errors every 500 steps
- `predictions.npz` — model predictions on the full space-time grid
- `metrics.json` — evaluation metrics
- `performance.json` — training and inference time, peak RAM and VRAM
- `config.json` — configuration used for the run

When comparing models, shared configuration values should be verified to ensure the runs use equivalent experimental conditions.
