# Data-Driven Neural Network

Implementation of the data-driven neural network used for comparison with the numerical solver and PINN.

Source code is maintained in ```src/```

## Saved Outputs

Each training run will create a new subfolder inside experiments/. Each run will save:

- `model.keras` — trained TensorFlow model
- `history.csv` — training history
- `predictions.npz` — model predictions
- `metrics.json` — evaluation metrics
- `config.json` — configuration used for the run

When comparing models, shared configuration values should be verified to ensure the runs use equivalent experimental conditions.