# Computational domain dimensions
## x ∈ [0, 1]
X_MIN = 0.0
X_MAX = 1.0

## y ∈ [0, 1]
Y_MIN = 0.0
Y_MAX = 1.0


# Grid size
## Number of grid points in the x and y directions
NX = 21
NY = 21


# Time step
## Simulated time advanced per iteration
DT = 0.001


# Number of time steps
## Number of solver iterations
## Total simulated time = DT * NT
NT = 6000

# Convergence criterion
# Maximum allowed velocity change between consecutive time steps
VELOCITY_TOLERANCE = 1e-5

# Pressure solver 
## Number of iterations used to solve the pressure Poisson equation
## Number of subtime steps
PRESSURE_ITERATIONS = 50


# Fluid density
## ρ represents the fluid's mass per unit volume
RHO = 1.0


# Kinematic viscosity
## ν represents how strongly momentum diffuses through the fluid
NU = 0.01


# Initial conditions
## Initial horizontal velocity (u)
## Initial vertical velocity (v)
## Initial pressure (p)
U_INITIAL = 0.0
V_INITIAL = 0.0
P_INITIAL = 0.0


# Boundary conditions
## Left, right, and bottom walls: no-slip -> u = 0, v = 0
WALL_VELOCITY = 0.0

## Top wall: moves horizontally -> u = 1, v = 0
LID_VELOCITY = 1.0


# Numerical options
## The defaults below reproduce the original 21 x 21 solver exactly.
## Validated alternatives are selected per run from run_solver.py.

## Advection scheme for u ∂u/∂x + v ∂u/∂y (and the same for v)
## "backward": first-order backward differences (original)
## "central":  second-order central differences
ADVECTION_SCHEME = "backward"

## Pressure Poisson solver
## "jacobi": PRESSURE_ITERATIONS Jacobi sweeps per time step (original)
## "direct": sparse LU factorization of the Laplacian, built once and
##           solved exactly every time step
PRESSURE_SOLVER = "jacobi"

## Lid start-up
## False: impulsive start, lid velocity jumps to LID_VELOCITY at t = 0 (original)
## True:  smooth start, u_lid(t) = LID_VELOCITY * tanh(LID_RAMP_RATE * t)
##        (reaches 0.995 * LID_VELOCITY by t = 0.3; the steady state is unchanged)
LID_RAMP = False
LID_RAMP_RATE = 10.0

## Pressure boundary condition on the lid
## "lid_dirichlet": p = 0 along the whole lid (original); zero normal gradient
##                  on the other three walls
## "neumann":       zero normal gradient on all four walls, with the pressure
##                  constant fixed by requiring zero mean over the interior
PRESSURE_BC = "lid_dirichlet"

## Time-stepping scheme
## "pressure_source": pressure Poisson equation with source
##                    ρ[(1/Δt)∇·u − (∂u/∂x)² − 2(∂u/∂y)(∂v/∂x) − (∂v/∂y)²],
##                    then an explicit momentum update (original)
## "projection":      fractional-step (Chorin) projection; requires the
##                    direct pressure solver
## "incremental_projection": projection with an incremental pressure
##                    correction; its steady state satisfies discrete
##                    continuity and momentum exactly, independent of Δt
TIME_SCHEME = "pressure_source"
