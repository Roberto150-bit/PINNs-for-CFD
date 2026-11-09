# Hybrid network: learns (x, y, t) -> (u, v, p) from BOTH the FD training data
# (exactly what the data-driven NN sees) and the physics loss (exactly what the
# PINN sees). Everything else is held equal to the other two networks, so the
# three can be compared run for run: same architecture, optimizer, learning
# rate, number of steps, seeds, data split, data batch size and physics points.
from models.common import config as C

# Training settings (identical to the data NN and the PINN)
TRAINING_STEPS = C.TRAINING_STEPS
LEARNING_RATE = C.LEARNING_RATE
OPTIMIZER = C.OPTIMIZER

# Model settings (identical)
HIDDEN_LAYERS = C.HIDDEN_LAYERS
NEURONS_PER_LAYER = C.NEURONS_PER_LAYER
ACTIVATION = C.ACTIVATION

# Data part (identical to the data NN): train split of the FD dataset, a new
# shuffled mini-batch of 4,096 points every step, loss = mean of the per-output
# MSEs on standardized u, v, p (train-split mean and std).
BATCH_SIZE = 4096
SPLIT_SEED = C.SPLIT_SEED

# Physics part (identical to the PINN): the same fixed collocation, initial and
# boundary points (taken from models/pinn/src/config.py), full set every step.

# Loss
##   L = λ_data L_data + λ_ic L_ic + λ_bc L_bc + λ_c mean(r_c²) + λ_m (mean(r_x²) + mean(r_y²))
## All weights start at 1, as in the PINN. The PINN's one-time weight check at
## step 2,000 is applied to all five components (data included), same rule.
LAMBDA_DATA = 1.0

# Reproducibility
SEEDS = [1, 2, 3, 4, 5]

# Runs: experiments/experiment_00X_re<Re>/hybrid/hybrid_re<Re>_<device>_s<seed>/
METHOD = "hybrid"


def run_id(reynolds_number, device, seed):
    return f"hybrid_re{reynolds_number:g}_{device}_s{seed}"
