# Goal and Progress

Main goal: Implement the finite-difference solver.

The baseline problem uses a $1 \times 1$ cavity with a moving top wall. The fluid begins at rest, with the top wall moving horizontally at a velocity of $1.0$ while the other three walls remain stationary. I used a $21 \times 21$ grid, density $\rho = 1.0$, kinematic viscosity $\nu = 0.01$, and a time step of $\Delta t = 0.001$.

Using the lid velocity $U=1$, cavity length $L=1$, and kinematic viscosity $\nu=0.01$, the Reynolds number is:

$$
Re = \frac{UL}{\nu}
=
\frac{(1)(1)}{0.01}
=
100
$$

I chose to keep the problem at $Re=100$ because it gives a relatively simple laminar lid-driven cavity case for establishing the numerical baseline before comparing it with the neural-network approaches.

The $21 \times 21$ grid was kept deliberately small for the first baseline. It is enough to represent the main circulation inside the cavity while keeping the solver quick to run during implementation and debugging. I am not treating this as a grid-independent solution; grid refinement can be tested later.

## Mathematical and Numerical Reasoning

I started from the standard incompressible Navier-Stokes equations for two-dimensional flow. The main task was taking the derivatives in those continuous equations and turning them into calculations that can be performed on the grid.

With 21 points across a domain from 0 to 1, there are 20 intervals, so:

$$
\Delta x
=
\Delta y
=
\frac{1-0}{21-1}
=
0.05
$$

The finite-difference formulas are standard numerical approximations for derivatives. Instead of having a continuous function available everywhere, the solver only has values at neighboring grid points.

For example:

$$
\frac{\partial u}{\partial x}
\approx
\frac{u_{i,j+1}-u_{i,j-1}}{2\Delta x}
$$

This estimates the change in $u$ in the $x$ direction using the values on either side of the current grid point.

For a second derivative:

$$
\frac{\partial^2u}{\partial x^2}
\approx
\frac{
u_{i,j+1}
-
2u_{i,j}
+
u_{i,j-1}
}{
\Delta x^2
}
$$

This is used in the viscous-diffusion terms because those terms depend on how the velocity changes relative to the surrounding values.

The incompressibility condition is:

$$
\nabla \cdot \mathbf{V}=0
$$

or:

$$
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
=
0
$$

This means fluid should not accumulate or disappear at a point. The velocity update by itself does not automatically maintain this condition, so pressure has to be connected to the velocity field.

The pressure source term $b$ and pressure Poisson equation come from the standard incompressible Navier-Stokes formulation. I did not derive these equations from scratch; the work here was understanding what each term was doing and then converting the derivatives into finite-difference calculations.

The pressure source term used is:

$$
b
=
\rho
\left[
\frac{1}{\Delta t}
\left(
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
\right)
-
\left(
\frac{\partial u}{\partial x}
\right)^2
-
2
\frac{\partial u}{\partial y}
\frac{\partial v}{\partial x}
-
\left(
\frac{\partial v}{\partial y}
\right)^2
\right]
$$

The current velocity field is used to construct $b$, and then $b$ becomes the source term in:

$$
\nabla^2p=b
$$

The pressure field is updated repeatedly using neighboring pressure values. I kept:

```text
PRESSURE_ITERATIONS = 50
```

for the baseline. This is currently a fixed numerical choice rather than a value I have tested independently for sensitivity.

Once pressure is updated, the horizontal and vertical momentum equations calculate the next velocity field. Each velocity update combines the previous velocity with advection, the pressure gradient, and viscous diffusion. After that update, the wall boundary conditions are reapplied so the top lid remains at its prescribed velocity and the other walls remain stationary.

## Solver Testing and Tuning

The first complete run used a maximum of 500 time steps. With $\Delta t=0.001$, this represents a maximum simulated time of:

$$
500(0.001)=0.5
$$

The solver completed without `NaN` or infinite values, with a final velocity change of `3.427430e-04` and a runtime of approximately `0.8251 s`.

The velocity visualization showed the expected clockwise circulation produced by the moving lid, and the pressure field showed stronger pressure variation near the upper corners. The solver was stable, but that did not mean the flow had reached a steady state yet.

So I increased the maximum number of time steps while keeping the grid, time step, viscosity, density, and boundary conditions unchanged. The trend was:

```text
NT = 500     3.427430e-04
NT = 1000    1.653634e-04
NT = 2000    7.936554e-05
NT = 4000    2.360300e-05
NT = 5000    1.413040e-05
```

The velocity change kept decreasing as more simulated time was allowed, showing that the flow was continuing toward a steady state rather than only remaining numerically stable.

To measure this, I used the largest change in either velocity component between consecutive time steps:

$$
\epsilon
=
\max
\left(
\max|u^{n+1}-u^n|,
\max|v^{n+1}-v^n|
\right)
$$

Here, $\epsilon$ is the numerical change being measured. In the code, this is called `velocity_change`.

For the baseline, I used:

$$
\epsilon < 10^{-5}
$$

as the stopping condition. This means the solver stops once the largest velocity change anywhere on the grid falls below the selected tolerance.

At 5000 steps, the final velocity change was still slightly above the tolerance, so I increased the maximum to 6000. The solver then stopped automatically at step 5681:

```text
Maximum NT:            6000
Steps completed:       5681
Final velocity change: 9.996184e-06
Runtime:               6.5968 s
```

Since:

$$
9.996184\times10^{-6}
<
1\times10^{-5}
$$

the selected steady-state convergence condition was reached.

This also changed how I think about `NT`. It is now the maximum number of time steps the solver is allowed to run, not necessarily the number it has to complete. In this case, the solver stopped at 5681 because the convergence condition was reached before 6000.

## Current Baseline

The Week 2 finite-difference baseline is currently:

```text
Domain:                 1 × 1
Grid:                   21 × 21
dx:                     0.05
dy:                     0.05
dt:                     0.001
Maximum time steps:     6000
Pressure iterations:    50
Density:                1.0
Kinematic viscosity:    0.01
Lid velocity:           1.0
Reynolds number:        100
Convergence tolerance:  1e-5
```

The solver now runs the full lid-driven cavity problem, produces velocity and pressure fields, and stops automatically once the selected steady-state convergence condition is reached.