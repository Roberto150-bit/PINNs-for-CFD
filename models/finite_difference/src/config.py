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