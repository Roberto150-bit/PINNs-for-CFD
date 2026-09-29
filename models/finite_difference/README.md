# Finite-Difference Model

Implementation of the finite-difference computational fluid dynamic  solver used as the traditional numerical baseline for the thesis.

The solver models two-dimensional incompressible lid-driven cavity flow. Its numerical solution provides a baseline for later comparison with the data-driven neural network and physics-informed neural network (PINN) models.

Source code is maintained in `src/`.

## Problem

The solver models the two-dimensional lid-driven cavity problem.

A fluid is enclosed inside a square domain:

$$
0 \leq x \leq 1,
\qquad
0 \leq y \leq 1
$$

The velocity field has two components:

- $u$ — horizontal velocity in the $x$ direction
- $v$ — vertical velocity in the $y$ direction

The left, right, and bottom walls are stationary:

$$
u = 0,
\qquad
v = 0
$$

The top wall moves horizontally to the right:

$$
u = 1,
\qquad
v = 0
$$

The moving wall transfers momentum into the initially stationary fluid, producing circulation within the cavity.

### Fluid Assumptions

The model does not represent a specific real-world fluid such as water or air. Instead, it uses a simplified fluid with constant properties suitable for the lid-driven cavity benchmark.

The fluid is assumed to be:

- **Incompressible**: Density does not change during the simulation.
- **Newtonian**: Viscous behavior is described using a constant viscosity.
- **Constant density**: Fluid mass per unit volume remains fixed and uniform throughout ($\rho = 1$).
- **Constant kinematic viscosity**: The ratio of flow resistance to density is fixed uniformly ($\nu = 0.01$).

These normalized values allow the numerical behavior of the solver to be studied without introducing additional material-specific properties.


## Model Configuration

The current baseline configuration is:

| Quantity | Symbol / Code | Value | Meaning |
|---|---|---:|---|
| Domain | $x,y$ | $1 \times 1$ | Physical region represented by the simulation |
| Grid size | `NX`, `NY` | $21 \times 21$ | Number of numerical grid points |
| Grid spacing | $\Delta x$, $\Delta y$ | $0.05$ | Distance between neighboring grid points |
| Time step | $\Delta t$ / `DT` | $0.001$ | Simulated time advanced during one iteration |
| Maximum time steps | `NT` | $6000$ | Maximum number of time updates allowed |
| Density | $\rho$ / `RHO` | $1.0$ | Fluid mass per unit volume |
| Kinematic viscosity | $\nu$ / `NU` | $0.01$ | Controls viscous momentum diffusion |
| Lid velocity | $U$ | $1.0$ | Horizontal velocity of the moving top wall |
| Pressure iterations | `PRESSURE_ITERATIONS` | $50$ | Jacobi iterations performed per time step |
| Velocity tolerance | $\epsilon$ | $10^{-5}$ | Threshold used to identify a steady solution |

`NT` is a maximum rather than a required number of iterations. The solver can stop earlier when the velocity convergence criterion is satisfied.

---

## Mathematical Notation

The following notation is used throughout this README:

- $x$, $y$ — spatial coordinates
- $t$ — simulated time
- $u$, $v$ — horizontal and vertical velocity components
- $p$ — pressure
- $\rho$ — fluid density
- $\nu$ — kinematic viscosity
- $\Delta x$, $\Delta y$ — grid spacing
- $\Delta t$ — time-step size
- $\nabla$ — the **nabla** (or del) operator, used to represent spatial derivatives
- $\mathbf{V} = (u,v)$ — the two-dimensional velocity vector

For example,

$$
\nabla \cdot \mathbf{V}
$$

represents the **divergence of the velocity field**. It measures whether fluid is locally spreading outward from or accumulating toward a point.

For an incompressible flow, this divergence must equal zero.

---

## Governing Equations

The solver approximates the two-dimensional incompressible Navier-Stokes equations.

### Conservation of Mass

Because the fluid is incompressible,

$$
\nabla \cdot \mathbf{V} = 0
$$

which in two dimensions becomes:

$$
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
=
0
$$

This is the **continuity equation**. It requires the numerical velocity field to remain divergence-free.

### Conservation of Momentum

The horizontal momentum equation is:

$$
\frac{\partial u}{\partial t}
+
u\frac{\partial u}{\partial x}
+
v\frac{\partial u}{\partial y}
=
-\frac{1}{\rho}\frac{\partial p}{\partial x}
+
\nu
\left(
\frac{\partial^2u}{\partial x^2}
+
\frac{\partial^2u}{\partial y^2}
\right)
$$

The vertical momentum equation is:

$$
\frac{\partial v}{\partial t}
+
u\frac{\partial v}{\partial x}
+
v\frac{\partial v}{\partial y}
=
-\frac{1}{\rho}\frac{\partial p}{\partial y}
+
\nu
\left(
\frac{\partial^2v}{\partial x^2}
+
\frac{\partial^2v}{\partial y^2}
\right)
$$

These equations describe how velocity changes due to three effects:

- **Advection** — the moving fluid transports momentum.
- **Pressure gradients** — differences in pressure accelerate the fluid.
- **Viscous diffusion** — viscosity spreads momentum between neighboring regions of the fluid.

The finite-difference solver replaces the derivatives in these equations with numerical approximations that can be evaluated on the computational grid.