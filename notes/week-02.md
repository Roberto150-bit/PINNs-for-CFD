# Week 2

## Goal

Implement the finite-difference baseline and begin the shared
architecture for the neural-network models.

## Mathematical Notes

The one-dimensional central-difference approximation is:

$$
\frac{\partial u}{\partial x}
\approx
\frac{u_{i+1} - u_{i-1}}{2\Delta x}
$$

For the second derivative:

$$
\frac{\partial^2 u}{\partial x^2}
\approx
\frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}
$$

### Why this matters

I am using the finite-difference solution as the numerical baseline
against which the neural-network approaches will eventually be compared.

## Implementation Notes

The initial boundary-condition implementation is:

```python
u[:, 0] = 0
u[:, -1] = 0

## Finite-Difference Solver — Baseline Run (NT = 500)

Configuration:
- Grid: 21 × 21
- DT: 0.001
- NT: 500
- Total simulated time: 0.5
- Density: 1.0
- Kinematic viscosity: 0.01
- Lid velocity: 1.0
- Reynolds number: 100

Results:
- Solver completed without NaN or infinite values.
- Runtime: 0.8251 s
- Final velocity change: 3.427430e-04
- Velocity field showed the expected clockwise lid-driven cavity circulation.
- Pressure field showed the strongest pressure variation near the upper corners.
- The solution remained numerically stable, but convergence has not yet been established.

### Convergence Check — NT = 1000

The simulation was repeated with NT increased from 500 to 1000 while all other parameters remained unchanged.

Results:
- Total simulated time: 1.0
- Final velocity change: 1.653634e-04
- Solver completed without NaN or infinite values.
- Runtime: 0.7296 s

Comparison:
- NT = 500: 3.427430e-04
- NT = 1000: 1.653634e-04
- The final velocity change decreased by approximately 52%.

Interpretation:
- The decreasing velocity change indicates that the solution is progressing toward a steady state.
- Additional simulation time is needed before claiming convergence.

### Convergence Check — NT = 2000

Results:
- Total simulated time: 2.0
- Final velocity change: 7.936554e-05
- Runtime: 2.1918 s
- No NaN or infinite values were detected.

Convergence trend:
- NT = 500:  3.427430e-04
- NT = 1000: 1.653634e-04
- NT = 2000: 7.936554e-05

The velocity change consistently decreased as the simulated time increased. This indicates that the numerical solution is approaching a steady state while remaining numerically stable.