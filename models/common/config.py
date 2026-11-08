"""Settings shared by every method so that runs are directly comparable."""

# Problem: 2D incompressible lid-driven cavity, unit square, ρ = 1
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 5.0            # time window covered by the networks
RHO = 1.0
LID_VELOCITY = 1.0
LID_RAMP_RATE = 10.0               # u_lid(t) = tanh(10 t), as in the FD reference


def kinematic_viscosity(reynolds_number):
    # Re = U L / ν with U = L = 1
    return 1.0 / reynolds_number


# Reference grid and snapshots (from the FD transient run)
GRID_N = 64                        # 64 x 64 points
SNAPSHOT_INTERVAL = 0.1            # 51 snapshots, t = 0.0, 0.1, ..., 5.0
TEST_B_TIMES = [0.5 * k for k in range(1, 11)]   # held-out whole snapshots
SPLIT_SEED = 0
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)             # train / validation / test A

# Network shared by the data-driven NN and the PINN
HIDDEN_LAYERS = 5
NEURONS_PER_LAYER = 64
ACTIVATION = "tanh"
KERNEL_INITIALIZER = "glorot_uniform"
N_INPUTS = 3                       # (x, y, t), scaled to [-1, 1]
N_OUTPUTS = 3                      # (u, v, p)

# Optimization shared by both networks
OPTIMIZER = "adam"
LEARNING_RATE = 1e-3
TRAINING_STEPS = 20000             # identical for both methods
EVAL_EVERY = 500                   # test-A relative L2 + wall clock
HISTORY_EVERY = 10                 # loss components to history.csv

# Paths
EXPERIMENT_DIR = "experiments/experiment_001_re100"
DATA_DIR = "experiments/data"


def fd_reference_run_id(reynolds_number, n=GRID_N):
    return f"fd_re{reynolds_number:g}_n{n}_transient"


def dataset_path(reynolds_number, n=GRID_N):
    return f"{DATA_DIR}/dataset_re{reynolds_number:g}_n{n}.npz"
