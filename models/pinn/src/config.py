# Physics-informed neural network: learns (x, y, t) -> (u, v, p) from the
# Navier–Stokes equations, the initial condition and the boundary conditions
# only. It never sees FD data during training (FD is used only to monitor
# and evaluate).
from models.common import config as C

# Training settings
## Same number of optimizer steps as the data-driven NN
TRAINING_STEPS = C.TRAINING_STEPS
LEARNING_RATE = C.LEARNING_RATE
OPTIMIZER = C.OPTIMIZER

# Model settings
HIDDEN_LAYERS = C.HIDDEN_LAYERS
NEURONS_PER_LAYER = C.NEURONS_PER_LAYER
ACTIVATION = C.ACTIVATION

# Training points (all fixed for the whole run; full set used every step)
## Interior collocation points (x, y, t), uniform in (0,1)^2 x (0, T]
N_COLLOCATION = 20000
## Initial-condition points at t = 0, interior of the cavity: u = v = 0
N_INITIAL = 4096
## Boundary points per wall, uniform in position along the wall and in t.
## Lid: u = tanh(10 t), v = 0. Other walls: u = v = 0.
## The two lid corners (0, 1) and (1, 1) belong to both the lid and a
## wall, so positions are drawn strictly inside each wall.
N_BOUNDARY_PER_WALL = 1000
## Sampling uses one fixed seed for every run, like the data split, so
## runs differ only in the network initialization seed
SAMPLING_SEED = 0

# Loss
##   L = λ_ic L_ic + λ_bc L_bc + λ_c mean(r_c²) + λ_m (mean(r_x²) + mean(r_y²))
##   L_ic = MSE_u + MSE_v at t = 0;  L_bc = MSE_u + MSE_v on the walls
LAMBDA_IC = 1.0
LAMBDA_BC = 1.0
LAMBDA_CONTINUITY = 1.0
LAMBDA_MOMENTUM = 1.0
## Weight check: after this step, if one component exceeds another by more
## than this factor, the weights are rebalanced once (recorded in config.json)
LAMBDA_CHECK_STEP = 2000
LAMBDA_IMBALANCE_FACTOR = 100.0

# Reproducibility
SEEDS = [1, 2, 3, 4, 5]

# File paths / saved outputs
## Runs: experiments/experiment_001_re100/pinn/pinn_re<Re>_<device>_s<seed>/
METHOD = "pinn"


def run_id(reynolds_number, device, seed):
    return f"pinn_re{reynolds_number:g}_{device}_s{seed}"
