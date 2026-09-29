# Finite-Difference Solver Configuration

This document describes the physical and numerical parameters defined in: `src/config.py`

The purpose of this file is to document what each configuration variable means, why its current value is being used, and how changing it would affect the simulation.

The current configuration defines the baseline lid-driven cavity case used by the finite-difference solver.

## Configuration Summary

| Parameter | Code Variable | Current Value | Meaning |
|---|---|---:|---|
| Horizontal domain | `X_MIN`, `X_MAX` | $0.0$, $1.0$ | Lower and upper bounds of the cavity in the $x$ direction |
| Vertical domain | `Y_MIN`, `Y_MAX` | $0.0$, $1.0$ | Lower and upper bounds of the cavity in the $y$ direction |
| Horizontal grid points | `NX` | $21$ | Number of numerical points used across the cavity |
| Vertical grid points | `NY` | $21$ | Number of numerical points used vertically |
| Time step | `DT` | $0.001$ | Amount of simulated time advanced per solver iteration |
| Maximum time steps | `NT` | $6000$ | Maximum number of time iterations allowed |
| Pressure iterations | `PRESSURE_ITERATIONS` | $50$ | Number of pressure updates performed during each time step |
| Fluid density | `RHO` | $1.0$ | Constant normalized density of the fluid |
| Kinematic viscosity | `NU` | $0.01$ | Controls how strongly momentum spreads through the fluid |
| Initial horizontal velocity | `U_INITIAL` | $0.0$ | Starting velocity in the $x$ direction |
| Initial vertical velocity | `V_INITIAL` | $0.0$ | Starting velocity in the $y$ direction |
| Initial pressure | `P_INITIAL` | $0.0$ | Starting numerical estimate of the pressure field |
| Stationary wall velocity | `WALL_VELOCITY` | $0.0$ | Velocity assigned to the stationary walls |
| Moving lid velocity | `LID_VELOCITY` | $1.0$ | Horizontal velocity assigned to the top wall |
| Velocity tolerance | `VELOCITY_TOLERANCE` | $10^{-5}$ | Maximum velocity change allowed before the solution is considered steady |

The configuration includes two general types of values:

- Physical parameters: define the cavity and the fluid being modeled.
- Numerical parameters: control how the computer approximates and advances that problem.


## Computational Domain

The cavity occupies:

$$
0 \leq x \leq 1
$$

$$
0 \leq y \leq 1
$$

which is defined by:

```python
X_MIN = 0.0
X_MAX = 1.0

Y_MIN = 0.0
Y_MAX = 1.0
```

Here, $x$ and $y$ are the spatial coordinates of the two-dimensional cavity.

The resulting geometry of the cavity has a width and height of:

$$
L_x = X_{\max} - X_{\min} = 1
$$

$$
L_y = Y_{\max} - Y_{\min} = 1
$$

Which is a normalized $1 \times 1$ square cavity.

The domain describes the size and shape of the modeled region. Changing the domain changes the physical problem itself. For example, a $2 \times 1$ domain would represent a rectangular cavity rather than a higher-resolution version of the current cavity.

The $1 \times 1$ domain is kept fixed for the baseline case.


## Grid Resolution

The cavity is represented numerically using:

```python
NX = 21
NY = 21
```

`NX` gives the number of grid points in the horizontal direction and `NY` gives the number in the vertical direction.

The current grid therefore contains:

$$
21 \times 21 = 441
$$

total grid points.

The grid size is different from the domain size. The domain remains $1 \times 1$, while the grid determines how many numerical points are used to represent that domain.

For a uniform grid:

$$
\Delta x
=
\frac{X_{\max}-X_{\min}}{N_x-1}
$$

$$
\Delta y
=
\frac{Y_{\max}-Y_{\min}}{N_y-1}
$$

For the current configuration:

$$
\Delta x
=
\Delta y
=
\frac{1}{21-1}
=
0.05
$$

The $21 \times 21$ grid was selected as the initial baseline because it keeps the simulation inexpensive while the solver is being developed and tested.

Later, values such as $41 \times 41$ or $81 \times 81$ can be tested while keeping the domain unchanged. This increases the numerical resolution without changing the physical cavity.

## Time Step

The simulation uses:

```python
DT = 0.001
```

which means:

$$
\Delta t = 0.001
$$

`DT` determines how far the numerical solution advances in simulated time during each solver iteration.

The size of the time step affects how large each numerical update can be. Larger values move the simulation forward more quickly but can make an explicit numerical method less stable. Smaller values provide more gradual updates but require more iterations to cover the same simulated duration.

The value ($\Delta t = 0.001$) was selected as a conservative starting value for the current grid and flow conditions.

The baseline runs completed without producing `NaN` or infinite values at this time step.

This confirms that the value is usable for the current configuration, although different time-step sizes can later be compared to determine how sensitive the numerical solution is to $\Delta t$.

## Maximum Time Steps

The maximum number of solver iterations is:

```python
NT = 6000
```

`NT` places an upper limit on how many time steps the simulation can perform.

Combined with:

$$
\Delta t = 0.001
$$

the largest simulated time allowed is:

$$
t_{\max}
=
NT\Delta t
$$

$$
t_{\max}
=
6000(0.001)
=
6.0
$$

The solver does not necessarily execute all $6000$ steps. It also checks how much the velocity field changes after each iteration and can stop earlier when the solution becomes sufficiently steady.

During baseline testing, the maximum number of steps was gradually increased until the convergence condition could be reached.

With:

```text
NT = 6000
```

the solver reached the current convergence criterion after:

```text
5681 time steps
```

Therefore, $6000$ is used as an upper limit that gives the current baseline enough time to reach a steady solution.

## Fluid Model

The current configuration does not represent a specific fluid such as water or air.

Instead, the solver uses a normalized fluid with constant properties.

The model assumes that the fluid is:

- incompressible,
- Newtonian,
- constant in density,
- constant in kinematic viscosity.

The governing equations and the consequences of these assumptions are explained in the finite-difference solver [README.md](README.md).


## Fluid Density

Density is defined as:

```python
RHO = 1.0
```

or:

$$
\rho = 1
$$

Density represents mass per unit volume.

The current model uses a normalized constant value rather than the dimensional density of a particular real fluid. Because the modeled flow is incompressible, the density remains constant throughout the simulation. `RHO` appears in the pressure and velocity calculations, so changing it would alter the physical parameters of the problem.


## Kinematic Viscosity

Kinematic viscosity is defined as:

```python
NU = 0.01
```

or:

$$
\nu = 0.01
$$

Kinematic viscosity describes how strongly momentum spreads through the fluid because of viscosity.

The value is connected to the Reynolds number:

$$
Re
=
\frac{UL}{\nu}
$$

Reynolds number measures the ratio of inertial forces to viscous forces within a fluid flow. It indicates the fluid properties if it is smooth and predicatable it is called laminar, or if it swirls and crashes it is called turbulent.

For the current baseline:

$$
U=1
$$

$$
L=1
$$

$$
\nu=0.01
$$

which gives:

$$
Re
=
\frac{(1)(1)}{0.01}
=
100
$$

Therefore, `NU = 0.01` is used because it gives the baseline:

$$
Re=100
$$

when combined with the unit cavity and unit lid velocity.

Changing `NU` while keeping the other values fixed would change the Reynolds number and therefore change the flow being modeled.

## Initial Horizontal and Vertical Velocity

The velocity field has two components:

- $u$ — horizontal velocity
- $v$ — vertical velocity

The initial values are:

```python
U_INITIAL = 0.0
V_INITIAL = 0.0
```

so the fluid begins with:

$$
u(x,y,0)=0
$$

$$
v(x,y,0)=0
$$

throughout the cavity interior. The fluid therefore begins at rest. The moving top boundary is applied before the first numerical update, so the flow begins developing from the motion of the lid.

## Initial Pressure

The pressure field begins with:

```python
P_INITIAL = 0.0
```

or:

$$
p(x,y,0)=0
$$

This value is the starting numerical estimate for the pressure field.

It does not mean that the physical fluid has zero absolute pressure.

As the simulation runs, the pressure field is updated according to the current flow so that the required pressure differences develop throughout the cavity.

## Stationary Wall Velocity

The stationary walls use:

```python
WALL_VELOCITY = 0.0
```

This value is applied to the:

- left wall,
- right wall,
- bottom wall.

The velocity condition is:

$$
u = 0,
\qquad
v = 0
$$

which means the fluid velocity at these stationary boundaries is also zero.

## Moving Lid Velocity

The moving top wall uses:

```python
LID_VELOCITY = 1.0
```

which gives:

$$
u = 1,
\qquad
v = 0
$$

at the top boundary.

The lid therefore moves horizontally while having no vertical motion.

This moving boundary drives the fluid inside the cavity and produces the circulating flow.

The value:

$$
U=1
$$

is also used as the characteristic velocity when calculating the Reynolds number.

## Pressure Iterations

Pressure is updated several times during every main simulation time step.

The current configuration uses:

```python
PRESSURE_ITERATIONS = 50
```

This means that for each velocity time step, the solver performs $50$ pressure updates before continuing.

Conceptually:

```text
Time step
    ├── pressure update
    ├── pressure update
    ├── ...
    └── pressure update
```

for a total of $50$ pressure updates.

This is different from `NT`:

- `NT` controls the maximum number of main simulation time steps.
- `PRESSURE_ITERATIONS` controls how many times the pressure field is refined within each time step.

More pressure updates generally require more computation but can provide a more developed pressure solution before the velocity field is updated.

The current value of $50$ is the baseline setting used by this implementation.

The exact pressure-solving method and the equations behind these updates are explained in the main [README.md](README.md), where the pressure solution itself is introduced.

## Velocity Convergence Tolerance

The stopping tolerance is:

```python
VELOCITY_TOLERANCE = 1e-5
```

After every time step, the solver compares the newly calculated velocity field with the velocity field from the previous step.

The largest change is measured as:

$$
\epsilon
=
\max
\left(
\max\left|u^{n+1}-u^n\right|,
\max\left|v^{n+1}-v^n\right|
\right)
$$

The solution is considered sufficiently steady when:

$$
\epsilon < 10^{-5}
$$

For the current baseline, this condition was reached after:

```text
5681 time steps
```

with:

$$
\epsilon
=
9.996184 \times 10^{-6}
$$

The tolerance and `NT` serve different purposes:

- `VELOCITY_TOLERANCE` determines when the solution can stop because it has become sufficiently steady.
- `NT` prevents the simulation from continuing indefinitely if that condition is not reached.

## Derived Values

Some useful values are calculated from the configuration rather than written directly into `config.py`.

| Quantity | Current Value | Derived From |
|---|---:|---|
| Cavity width | $1$ | `X_MAX - X_MIN` |
| Cavity height | $1$ | `Y_MAX - Y_MIN` |
| Total grid points | $441$ | `NX × NY` |
| Grid spacing $\Delta x$ | $0.05$ | Domain width and `NX` |
| Grid spacing $\Delta y$ | $0.05$ | Domain height and `NY` |
| Reynolds number | $100$ | $U$, $L$, and $\nu$ |
| Maximum simulated time | $6.0$ | `NT × DT` |
| Baseline convergence step | $5681$ | Observed solver result |

These values help describe the complete baseline without requiring additional configuration variables.

## Parameters for Later Testing

The current settings define the first working baseline rather than a final optimized numerical configuration.

Future experiments can test the effect of changing:

- `NX` and `NY` — spatial resolution
- `DT` — time-step size
- `PRESSURE_ITERATIONS` — pressure refinement
- `VELOCITY_TOLERANCE` — convergence strictness
- `NU` — viscosity and Reynolds number

The baseline configuration should remain unchanged while individual parameters are tested so that their effects can be compared consistently.