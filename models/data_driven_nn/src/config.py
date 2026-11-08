# Data-driven neural network: learns (x, y, t) -> (u, v, p) from FD data only.
# Shared values come from models/common/config.py so both networks match.
from models.common import config as C

# Training settings
## Fixed number of optimizer steps (identical to the PINN for comparability)
TRAINING_STEPS = C.TRAINING_STEPS
## Mini-batch of FD space-time points per step
BATCH_SIZE = 4096
LEARNING_RATE = C.LEARNING_RATE
OPTIMIZER = C.OPTIMIZER
## Loss: mean of the three per-output MSEs on standardized targets
##   L_data = (MSE_u + MSE_v + MSE_p) / 3
## Standardizing (train-split mean and std) keeps p from dominating u and v
LOSS = "mean of per-output MSE on standardized u, v, p"

# Model settings
HIDDEN_LAYERS = C.HIDDEN_LAYERS
NEURONS_PER_LAYER = C.NEURONS_PER_LAYER
ACTIVATION = C.ACTIVATION

# Reproducibility
## Seeds 1-5 are the repeated runs; the data split always uses seed 0
SEEDS = [1, 2, 3, 4, 5]
SPLIT_SEED = C.SPLIT_SEED

# File paths / saved outputs
## Training data: experiments/data/dataset_re<Re>_n64.npz (from the FD transient)
## Runs: experiments/experiment_001_re100/data_driven_nn/dnn_re<Re>_<device>_s<seed>/
METHOD = "data_driven_nn"


def run_id(reynolds_number, device, seed):
    return f"dnn_re{reynolds_number:g}_{device}_s{seed}"
