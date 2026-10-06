# Finite-Difference Solver

This contains the finite-difference computational fluid dynamics (CFD) solver used as the traditional numerical baseline for the project. The solver models two-dimensional incompressible lid-driven cavity flow using finite-difference approximations of the Navier-Stokes equations.

The current baseline uses:

| Quantity | Value |
|---|---:|
| Domain | $1 \times 1$ |
| Grid | $21 \times 21$ |
| Reynolds number | $Re = 100$ |
| Time step | $\Delta t = 0.001$ |
| Maximum time steps | $6000$ |
| Pressure iterations per time step | $50$ |
| Velocity convergence tolerance | $10^{-5}$ |

The complete parameter definitions and reasoning behind these values are documented in [CONFIGURATION.md](CONFIGURATION.md).

Source code is located in: `src/`

### Solver Overview

The numerical solver evolves three fields across the computational grid:

- $u$ — horizontal velocity
- $v$ — vertical velocity
- $p$ — pressure

At each time step, the solver:

1. Saves the current velocity field.
2. Builds the pressure source term ($b$) from the current velocities.
3. Refines the pressure field using the pressure Poisson equation.
4. Uses the updated pressure to calculate new horizontal and vertical velocities.
5. Reapplies the velocity boundary conditions.
6. Measures how much the velocity field changed.
7. Stops if the convergence tolerance has been reached; otherwise begins the next time step.

The current configuration performs 50 pressure updates inside each main time step. This process continues until the flow becomes sufficiently steady or the maximum number of time steps is reached.

## 1. Computational Grid

The physical setup is a continuous $1 \times 1$ domain:

$$
0 \leq x \leq 1,
\qquad
0 \leq y \leq 1
$$

A numerical solver cannot calculate the flow at every possible location in this continuous space. Instead, the domain is represented using a finite set of grid points. For the current baseline:

```python
NX = 21
NY = 21
```

so the solver uses $21$ points in the horizontal and vertical direction. The grid coordinates are created in `create_grid()`:

```python
x = np.linspace(X_MIN, X_MAX, NX)
y = np.linspace(Y_MIN, Y_MAX, NY)
```

`np.linspace()` creates evenly spaced values between the two domain boundaries. For the $x$ and $y$ direction, this produces coordinates of the form:

```python
[0.00, 0.05, 0.10, 0.15, ..., 0.95, 1.00]
```

The spacing values between grid points are required later by the finite-difference equations, which approximate spatial derivatives by comparing values at neighboring grid points over a known distance. The distance between neighboring grid points is calculated as:

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

For the current $21 \times 21$ grid:

$$
\Delta x
=
\Delta y
=
\frac{1 - 0}{21 - 1}
=
0.05
$$

The function therefore returns:

```python
return x, y, dx, dy
```

These values define the numerical locations where the flow variables will later be stored and updated.

## 2. Initial Fields

The velocity and pressure fields are initialized using the values from `config.py`. Once the computational grid has been created, the solver needs an initial value for each flow variable at every grid point. The current implementation initializes three two-dimensional fields:

```python
u = np.full((NY, NX), U_INITIAL)
v = np.full((NY, NX), V_INITIAL)
p = np.full((NY, NX), P_INITIAL)
```

Each array has the same shape as the computational grid:

```python
(NY, NX) = (21, 21)
```

This means that every grid point stores:

- one horizontal velocity value, $u$
- one vertical velocity value, $v$
- one pressure value, $p$

For the current baseline:

```python
U_INITIAL = 0.0
V_INITIAL = 0.0
P_INITIAL = 0.0
```

The velocity field begins at rest, while the pressure field begins from a uniform numerical estimate.

`np.full()` is used because it creates an entire two-dimensional array and fills every grid location with the same starting value.

The velocity boundary conditions are then applied before the first time step. This changes the top wall to its prescribed moving-lid velocity while the interior of the fluid remains initially stationary.

The initialized fields are returned as:

```python
return u, v, p
```

These arrays become the starting state used by the numerical solver.

## 3. Building the Pressure Source Term

At this point, the solver already has:

- the computational grid,
- the horizontal velocity field $u$,
- the vertical velocity field $v$,
- the current pressure field $p$.

The next step is to determine how the current velocity field should influence the pressure calculation. For incompressible flow, the velocity field must satisfy:

$$
\nabla \cdot \mathbf{V}=0
$$

In two dimensions:

$$
\frac{\partial u}{\partial x}
+
\frac{\partial v}{\partial y}
=
0
$$

This condition means that the flow cannot create a net accumulation or loss of fluid at a point. The numerical velocity field will not automatically satisfy this condition after every update. Therefore, before pressure can be solved, the solver first summarizes information from the current velocity field into a new array called the pressure source term, $b$. This term is calculated from the spatial changes in $u$ and $v$:

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

The equations above contain derivatives such as:

$$
\frac{\partial u}{\partial x}
$$

but the solver does not have a continuous function for $u$. It only knows the velocity values stored at the grid points. For this we use a technique called central-difference approximation to simplify the spatial derivatives:

$$
\frac{\partial v}{\partial y}
\approx
\frac{
v_{i+1,j}
-
v_{i-1,j}
}{
2\Delta y
}
$$

$$
\frac{\partial u}{\partial y}
\approx
\frac{
u_{i+1,j}
-
u_{i-1,j}
}{
2\Delta y
}
$$

$$
\frac{\partial v}{\partial x}
\approx
\frac{
v_{i,j+1}
-
v_{i,j-1}
}{
2\Delta x
}
$$

$$
\frac{\partial u}{\partial x}
\approx
\frac{
u_{i,j+1}
-
u_{i,j-1}
}{
2\Delta x
}
$$

The previously calculated `dx` and `dy` are what give the distance between these neighboring grid points.

### In the source code

The pressure source array is first created with the same dimensions as the computational grid:

```python
b = np.zeros((NY, NX))
```

It initially contains zeros. The source term is then calculated only for the interior of the grid:

```python
b[1:-1, 1:-1]
```

For example, the mathematical approximation:

$$
\frac{\partial u}{\partial x}
\approx
\frac{
u_{i,j+1}
-
u_{i,j-1}
}{
2\Delta x
}
$$

appears in the code as:

```python
(u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
```

The NumPy slices represent neighboring grid locations:

```text
u[1:-1, 0:-2]   left neighbor
u[1:-1, 1:-1]   current interior location
u[1:-1, 2:]     right neighbor
```

The complete calculation combines these finite-difference derivatives into `b`. Afterward:

```python
return b
```

passes the completed pressure source field to the pressure solver.

### Only Interior Points Are Calculated

Central differences require a neighboring grid point on both sides of the current location. Interior points have those neighbors. A point directly on a wall does not have another grid point outside the cavity, so the same central-difference calculation cannot be applied there. The pressure behavior directly on the walls is handled later through the pressure boundary conditions.

## 4. Solving the Pressure Field

The solver can now use $b$ to calculate the pressure field $p$. The relationship is given by the pressure Poisson equation:

$$
\nabla^2 p=b
$$

For pressure in two dimensions:

$$
\nabla^2p
=
\frac{\partial^2p}{\partial x^2}
+
\frac{\partial^2p}{\partial y^2}
=
b
$$

The left side describes how pressure varies relative to surrounding pressure values, while the right side contains the source information calculated from the current velocity field. Similar to the velocity derivatives, the second derivatives of pressure must be approximated using neighboring grid points.


$$
\frac{\partial^2p}{\partial x^2}
\approx
\frac{
p_{i,j+1}
-
2p_{i,j}
+
p_{i,j-1}
}{
\Delta x^2
}
$$

and:

$$
\frac{\partial^2p}{\partial y^2}
\approx
\frac{
p_{i+1,j}
-
2p_{i,j}
+
p_{i-1,j}
}{
\Delta y^2
}
$$

Substituting these approximations into the pressure Poisson equation and solving for the pressure at the current point gives:

$$
p_{i,j}
=
\frac{
\left(p_{i,j+1}+p_{i,j-1}\right)\Delta y^2
+
\left(p_{i+1,j}+p_{i-1,j}\right)\Delta x^2
}{
2\left(\Delta x^2+\Delta y^2\right)
}
-
\frac{
\Delta x^2\Delta y^2
}{
2\left(\Delta x^2+\Delta y^2\right)
}
b_{i,j}
$$

This equation says that the new pressure at one interior grid point depends on:

- the pressure to its left,
- the pressure to its right,
- the pressure above it,
- the pressure below it,
- the local source value $b_{i,j}$.

Although, the pressure at one grid point depends on the pressure at surrounding points. Those surrounding values are themselves only estimates of the final pressure field. The solver therefore cannot obtain the complete pressure field from a single update. Instead, it begins with the current pressure estimate:

```python
pn = p.copy()
```

and uses that field to calculate a new estimate. The process is repeated. The current configuration performs:

```text
PRESSURE_ITERATIONS = 50
```

pressure updates during every main simulation time step. This repeated-update method is known as Jacobi iteration. The important feature of Jacobi iteration in this implementation is that every new pressure value in one iteration is calculated from the same previous pressure field, `pn`.

After every pressure update, the boundary conditions are reapplied. On the left and right walls:

$$
\frac{\partial p}{\partial x}=0
$$

and on the bottom wall:

$$
\frac{\partial p}{\partial y}=0
$$

A zero pressure gradient means the pressure does not change in the direction perpendicular to that wall. In the code, this is enforced by copying the neighboring interior value.

For example:

```python
p[:, 0] = p[:, 1]
```

sets the pressure on the left wall equal to the pressure immediately beside it. 
The top wall uses: $p=0$ as the reference pressure:

```python
p[-1, :] = 0.0
```

The completed pressure field is then returned:

```python
return p
```

and becomes an input to the velocity update in the next stage of the solver.

## 5. Velocity Update

The pressure field has now been updated for the current time step. The solver can use that pressure, together with the previous velocity field, to calculate the next values of $u$ and $v$. For each interior grid point, the velocity update follows the same general structure:

$$
\text{Velocity}_{new} 
=
\text{Velocity}_{previous}
-
\text{Advection}
-
\text{Pressure-gradient}
+
\text{Viscous-diffusion}
$$

Advection and viscous diffusion act in both the $x$ and $y$ directions, so each velocity update contains contributions from both directions.


### Horizontal Velocity

The horizontal velocity $u$ is governed by the horizontal momentum equation:

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

The finite-difference solver converts this equation into an update for each interior grid point:

$$
\begin{aligned}
u_{i,j}^{n+1}
={}&
u_{i,j}^{n}
\\
&-
u_{i,j}^{n}
\frac{\Delta t}{\Delta x}
\left(
u_{i,j}^{n}-u_{i,j-1}^{n}
\right)
\\
&-
v_{i,j}^{n}
\frac{\Delta t}{\Delta y}
\left(
u_{i,j}^{n}-u_{i-1,j}^{n}
\right)
\\
&-
\frac{\Delta t}{2\rho\Delta x}
\left(
p_{i,j+1}-p_{i,j-1}
\right)
\\
&+
\nu\frac{\Delta t}{\Delta x^2}
\left(
u_{i,j+1}^{n}
-2u_{i,j}^{n}
+u_{i,j-1}^{n}
\right)
\\
&+
\nu\frac{\Delta t}{\Delta y^2}
\left(
u_{i+1,j}^{n}
-2u_{i,j}^{n}
+u_{i-1,j}^{n}
\right)
\end{aligned}
$$

This can be read one contribution at a time.

#### 1. Previous horizontal velocity

$$
u_{i,j}^{n}
$$

This is the horizontal velocity already present at the grid point.

#### 2. Horizontal advection

$$
-
u_{i,j}^{n}
\frac{\Delta t}{\Delta x}
\left(
u_{i,j}^{n}-u_{i,j-1}^{n}
\right)
$$

This represents horizontal motion carrying horizontal momentum through the fluid. The spatial derivative is approximated using the current point and the neighboring point to the left.

#### 3. Vertical advection

$$
-
v_{i,j}^{n}
\frac{\Delta t}{\Delta y}
\left(
u_{i,j}^{n}-u_{i-1,j}^{n}
\right)
$$

The horizontal velocity can also be transported by fluid moving vertically. Together, these two terms represent the advection of horizontal momentum.

#### 4. Pressure-gradient contribution

$$
-
\frac{\Delta t}{2\rho\Delta x}
\left(
p_{i,j+1}-p_{i,j-1}
\right)
$$

A difference in pressure between the left and right sides of a grid point can accelerate the fluid horizontally. The solver estimates this pressure gradient using the pressure values on both sides of the point.

#### 5. Viscous diffusion in the $x$ direction

$$
+
\nu\frac{\Delta t}{\Delta x^2}
\left(
u_{i,j+1}^{n}
-2u_{i,j}^{n}
+u_{i,j-1}^{n}
\right)
$$

This compares the current horizontal velocity with its left and right neighbors.

#### 6. Viscous diffusion in the $y$ direction

$$
+
\nu\frac{\Delta t}{\Delta y^2}
\left(
u_{i+1,j}^{n}
-2u_{i,j}^{n}
+u_{i-1,j}^{n}
\right)
$$

This performs the same type of calculation using the neighboring velocities above and below the current point. Viscous diffusion tends to smooth differences in velocity between neighboring parts of the fluid. After these contributions are combined, the solver has the new horizontal velocity field $u^{n+1}$.

### Vertical Velocity

The vertical component follows the same structure.

Its momentum equation is:

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

The implemented finite-difference update is:

$$
\begin{aligned}
v_{i,j}^{n+1}
={}&
v_{i,j}^{n}
\\
&-
u_{i,j}^{n}
\frac{\Delta t}{\Delta x}
\left(
v_{i,j}^{n}-v_{i,j-1}^{n}
\right)
\\
&-
v_{i,j}^{n}
\frac{\Delta t}{\Delta y}
\left(
v_{i,j}^{n}-v_{i-1,j}^{n}
\right)
\\
&-
\frac{\Delta t}{2\rho\Delta y}
\left(
p_{i+1,j}-p_{i-1,j}
\right)
\\
&+
\nu\frac{\Delta t}{\Delta x^2}
\left(
v_{i,j+1}^{n}
-2v_{i,j}^{n}
+v_{i,j-1}^{n}
\right)
\\
&+
\nu\frac{\Delta t}{\Delta y^2}
\left(
v_{i+1,j}^{n}
-2v_{i,j}^{n}
+v_{i-1,j}^{n}
\right)
\end{aligned}
$$

The calculation contains the same effects as the horizontal update. The main difference is the pressure term. For $v$, it uses the pressure change in the $y$ direction:

$$
\frac{\partial p}{\partial y}
$$

because it is calculating vertical acceleration. In the code, both fields are updated together:

```python
return u, v
```

The resulting velocity fields describe the new interior flow state for the current time step.

## 6. Velocity Boundary Conditions

After the velocity equations are evaluated, the wall conditions are reapplied. This moving boundary introduces momentum into the cavity and produces the primary circulating flow. The boundary conditions are reapplied after every velocity update so that numerical calculations inside the domain do not alter the prescribed wall behavior.

The left, right, and bottom walls remain stationary:

$$
u = 0,
\qquad
v = 0
$$

The moving top wall is:

$$
u = 1,
\qquad
v = 0
$$

## 7. Convergence Check

The solver is looking for a steady-state solution, meaning the velocity field is no longer changing significantly from one time step to the next. To measure this, the solver compares the new velocity field with the velocity field from the previous time step.

The convergence measure is written as:

$$
\epsilon
=
\max
\left(
\max\left|u^{n+1}-u^n\right|,
\max\left|v^{n+1}-v^n\right|
\right)
$$

For the current baseline:

$$
\epsilon
<
10^{-5}
$$

The symbol $\epsilon$ is commonly used to represent a small numerical error, difference, or convergence measure. In this solver, it corresponds to the code variable `velocity_change`.

## 8. Complete Solver Loop

The main numerical loop in `solver.py` can be summarized as:

```text
Create grid
        ↓
Initialize u, v, p
        ↓
Apply velocity boundary conditions
        ↓
┌─────────────────────────────────────┐
│ Save previous u and v               │
│                                     │
│ Build pressure source b             │
│              ↓                      │
│ Solve pressure p                    │
│              ↓                      │
│ Update horizontal velocity u        │
│              ↓                      │
│ Update vertical velocity v          │
│              ↓                      │
│ Reapply velocity boundary conditions│
│              ↓                      │
│ Measure velocity change             │
│              ↓                      │
│ Check convergence                   │
└─────────────────────────────────────┘
        ↓
Return x, y, u, v, p
```

The pressure solution and velocity solution are therefore coupled. The velocity field determines the pressure source term, pressure influences the next velocity update, and the process repeats until the flow becomes sufficiently steady.

## 9. Running the Solver

From:

```text
models/finite_difference/src/
```

run:

```bash
python run_solver.py
```

A successful converged run reports output similar to:

```text
Solver completed without NaN or infinite values.
Final velocity change: 9.996184e-06
Steps completed: 5681
Convergence criteria reached.
Runtime: ...
```

The numerical solution is saved to:

```text
solver_results.npz
```

`solver_results.npz` contains:

| Array | Meaning |
|---|---|
| `x` | Horizontal grid coordinates |
| `y` | Vertical grid coordinates |
| `u` | Horizontal velocity field |
| `v` | Vertical velocity field |
| `p` | Pressure field |

Saving the numerical fields separately allows the solution to be analyzed or visualized without rerunning the CFD simulation.

## 10. Visualization

After generating the numerical solution, run:

```bash
python visualization.py
```

The visualization script produces:

```text
velocity_field.png
pressure_field.png
```

The velocity plot shows the direction and relative magnitude of the flow throughout the cavity. The pressure plot shows the pressure distribution generated by the moving-lid flow.

## 11. Source Files

### `src/config.py`

Contains the physical and numerical configuration used by the solver.

See [`CONFIGURATION.md`](CONFIGURATION.md) for the complete parameter documentation.

### `src/solver.py`

Contains the numerical CFD implementation:

```text
create_grid()
initialize_variables()
build_pressure_source()
solve_pressure()
update_velocity()
apply_velocity_boundary_conditions()
run_solver()
```

### `src/run_solver.py`

Runs the simulation and:

- checks for `NaN` and infinite values,
- reports the final velocity change,
- reports the number of completed time steps,
- reports whether convergence was reached,
- records runtime,
- saves the resulting numerical fields.

### `src/visualization.py`

Loads the saved numerical solution and creates the velocity and pressure plots.