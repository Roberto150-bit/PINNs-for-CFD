import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from config import (
    X_MIN,
    X_MAX,
    Y_MIN,
    Y_MAX,
    NX,
    NY,
    DT,
    NT,
    VELOCITY_TOLERANCE,
    PRESSURE_ITERATIONS,
    RHO,
    NU,
    U_INITIAL,
    V_INITIAL,
    P_INITIAL,
    WALL_VELOCITY,
    LID_VELOCITY,
    ADVECTION_SCHEME,
    PRESSURE_SOLVER,
    LID_RAMP,
    LID_RAMP_RATE,
    PRESSURE_BC,
    TIME_SCHEME,
)

# Arrays are stored as [row, column] = [j, i] = [y index, x index],
# so every field has shape (ny, nx).


def create_grid(nx=NX, ny=NY):
    x = np.linspace(X_MIN, X_MAX, nx)   # horizontal grid-point
    y = np.linspace(Y_MIN, Y_MAX, ny)   # vertical grid-point

    dx = (X_MAX - X_MIN) / (nx - 1)     # spacing between x-points
    dy = (Y_MAX - Y_MIN) / (ny - 1)     # spacing between y-points

    return x, y, dx, dy


def initialize_variables(nx=NX, ny=NY):
    u = np.full((ny, nx), U_INITIAL)    # horizontal velocity at every grid point
    v = np.full((ny, nx), V_INITIAL)    # vertical velocity at every grid point
    p = np.full((ny, nx), P_INITIAL)    # pressure at every grid point

    return u, v, p


def lid_velocity(t, lid_ramp=LID_RAMP):
    # Horizontal velocity of the top wall at time t
    # Impulsive start: the lid moves at full speed from t = 0
    # Ramped start: u_lid(t) = U * tanh(k t), which is 0 at t = 0 and smooth,
    # so the same start-up can be represented by a neural network
    if lid_ramp:
        return LID_VELOCITY * np.tanh(LID_RAMP_RATE * t)
    return LID_VELOCITY


def build_pressure_source(u, v, dx, dy, dt=DT, rho=RHO):
    # Builds the source term for the pressure Poisson equation
    # Incompressible flow requires velocity divergence: ∇ · u = 0
    ny, nx = u.shape
    b = np.zeros((ny, nx))

    # Calculate only at interior grid points using central differences
    b[1:-1, 1:-1] = rho * (
        (1 / dt)
        * (
            # Velocity divergence
            (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
            + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
        )
        # Horizontal velocity gradient contribution
        - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2

        # Interaction between x- and y-velocity gradients
        - 2
        * (
            (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy)
            * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)
        )

        # Vertical velocity gradient contribution
        - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2
    )

    return b


def apply_pressure_boundary_conditions(p, pressure_bc=PRESSURE_BC):
    # Pressure boundary conditions
    p[:, -1] = p[:, -2]   # Right wall: zero pressure gradient
    p[:, 0] = p[:, 1]     # Left wall: zero pressure gradient
    p[0, :] = p[1, :]     # Bottom wall: zero pressure gradient
    if pressure_bc == "lid_dirichlet":
        p[-1, :] = 0.0        # Top wall: reference pressure
    else:
        p[-1, :] = p[-2, :]   # Top wall: zero pressure gradient

    return p


def solve_pressure(p, b, dx, dy, iterations=PRESSURE_ITERATIONS,
                   pressure_bc=PRESSURE_BC):
    # Improves the pressure approximation with Jacobi iterations
    for _ in range(iterations):
        pn = p.copy()

        # Update pressure at interior grid points
        p[1:-1, 1:-1] = (
            (
                # Pressure from left/right neighbors
                (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2

                # Pressure from top/bottom neighbors
                + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2
            )
            / (2 * (dx**2 + dy**2))

            # Correction from the pressure source term
            - (
                dx**2 * dy**2
                / (2 * (dx**2 + dy**2))
            )
            * b[1:-1, 1:-1]
        )

        p = apply_pressure_boundary_conditions(p, pressure_bc)

    if pressure_bc == "neumann":
        # Pressure is defined only up to a constant: fix its mean at zero
        p -= p[1:-1, 1:-1].mean()

    return p


def build_pressure_matrix(nx, ny, dx, dy, pressure_bc=PRESSURE_BC):
    # Sparse 5-point Laplacian acting on the interior pressure unknowns,
    # with the same boundary conditions the Jacobi sweeps converge to:
    #   left, right, bottom walls: p_wall = p_neighbor (zero normal gradient)
    #   top wall (lid), "lid_dirichlet": p = 0 (fixes the pressure constant)
    #   top wall (lid), "neumann":       p_wall = p_neighbor as on the other walls
    # Solving A p = b exactly gives the fixed point of the Jacobi iteration.
    # The matrix depends only on the grid, so it is factorized once with
    # sparse LU and the factorization is reused at every time step.
    #
    # With zero-gradient conditions on all four walls the Laplacian is
    # singular (any constant can be added to p). The system is then bordered
    # with one extra row and column,
    #     [ A   1 ] [ p ]   [ b ]
    #     [ 1ᵀ  0 ] [ λ ] = [ 0 ],
    # which selects the zero-mean solution; λ absorbs the small
    # incompatibility of the discrete source with the Neumann conditions.
    mx, my = nx - 2, ny - 2                 # interior points in x and y
    n = mx * my
    cx, cy = 1.0 / dx**2, 1.0 / dy**2

    def index(i, j):
        # Interior point (i, j), i = 0..mx-1, j = 0..my-1, to unknown number
        return j * mx + i

    rows, cols, vals = [], [], []
    for j in range(my):
        for i in range(mx):
            k = index(i, j)
            diag = -2.0 * (cx + cy)

            # West neighbor; at the left wall p_wall = p_interior
            if i > 0:
                rows.append(k); cols.append(index(i - 1, j)); vals.append(cx)
            else:
                diag += cx
            # East neighbor; at the right wall p_wall = p_interior
            if i < mx - 1:
                rows.append(k); cols.append(index(i + 1, j)); vals.append(cx)
            else:
                diag += cx
            # South neighbor; at the bottom wall p_wall = p_interior
            if j > 0:
                rows.append(k); cols.append(index(i, j - 1)); vals.append(cy)
            else:
                diag += cy
            # North neighbor; at the lid either p = 0 (adds nothing) or
            # p_wall = p_interior
            if j < my - 1:
                rows.append(k); cols.append(index(i, j + 1)); vals.append(cy)
            elif pressure_bc == "neumann":
                diag += cy

            rows.append(k); cols.append(k); vals.append(diag)

    if pressure_bc == "neumann":
        # Bordering row and column of ones (zero-mean constraint)
        for k in range(n):
            rows.append(k); cols.append(n); vals.append(1.0)
            rows.append(n); cols.append(k); vals.append(1.0)
        A = sp.csc_matrix((vals, (rows, cols)), shape=(n + 1, n + 1))
    else:
        A = sp.csc_matrix((vals, (rows, cols)), shape=(n, n))
    return spla.splu(A)


def solve_pressure_direct(p, b, pressure_lu, pressure_bc=PRESSURE_BC):
    # Exact solution of the discrete pressure Poisson equation
    ny, nx = p.shape
    rhs = b[1:-1, 1:-1].ravel()
    if pressure_bc == "neumann":
        rhs = np.append(rhs, 0.0)
    p_interior = pressure_lu.solve(rhs)[: (ny - 2) * (nx - 2)]
    p[1:-1, 1:-1] = p_interior.reshape(ny - 2, nx - 2)

    return apply_pressure_boundary_conditions(p, pressure_bc)


def advection_terms(f, u, v, dx, dy, scheme=ADVECTION_SCHEME):
    # Advective rate of change of field f at interior points:
    #   u ∂f/∂x + v ∂f/∂y
    # "backward": (f_i - f_{i-1}) / dx. This is upwind only where the
    #             velocity is positive; inside the vortex u and v are
    #             negative, where it becomes downwind. First-order accurate.
    # "central":  (f_{i+1} - f_{i-1}) / (2 dx). Second-order accurate.
    #             Stable without oscillations while the cell Reynolds number
    #             |u| dx / ν stays below about 2 (or modestly above with the
    #             viscous damping on these grids).
    if scheme == "backward":
        return (
            u[1:-1, 1:-1] / dx * (f[1:-1, 1:-1] - f[1:-1, 0:-2])
            + v[1:-1, 1:-1] / dy * (f[1:-1, 1:-1] - f[0:-2, 1:-1])
        )
    if scheme == "central":
        return (
            u[1:-1, 1:-1] * (f[1:-1, 2:] - f[1:-1, 0:-2]) / (2 * dx)
            + v[1:-1, 1:-1] * (f[2:, 1:-1] - f[0:-2, 1:-1]) / (2 * dy)
        )
    raise ValueError(f"Unknown advection scheme: {scheme}")


def update_velocity(u, v, p, dx, dy, dt=DT, nu=NU, rho=RHO,
                    advection=ADVECTION_SCHEME):
    # Save previous velocity fields
    un = u.copy()
    vn = v.copy()

    if advection == "backward":
        # Original explicit update, kept term-for-term so that the default
        # configuration reproduces the original results bit for bit

        # Update horizontal velocity at interior grid points
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]

            # Horizontal advection
            - un[1:-1, 1:-1] * dt / dx
            * (un[1:-1, 1:-1] - un[1:-1, 0:-2])

            # Vertical advection
            - vn[1:-1, 1:-1] * dt / dy
            * (un[1:-1, 1:-1] - un[0:-2, 1:-1])

            # Pressure-gradient force
            - dt / (2 * rho * dx)
            * (p[1:-1, 2:] - p[1:-1, 0:-2])

            # Viscous diffusion in x
            + nu * dt / dx**2
            * (
                un[1:-1, 2:]
                - 2 * un[1:-1, 1:-1]
                + un[1:-1, 0:-2]
            )

            # Viscous diffusion in y
            + nu * dt / dy**2
            * (
                un[2:, 1:-1]
                - 2 * un[1:-1, 1:-1]
                + un[0:-2, 1:-1]
            )
        )

        # Update vertical velocity at interior grid points
        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]

            # Horizontal advection
            - un[1:-1, 1:-1] * dt / dx
            * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])

            # Vertical advection
            - vn[1:-1, 1:-1] * dt / dy
            * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])

            # Pressure-gradient force
            - dt / (2 * rho * dy)
            * (p[2:, 1:-1] - p[0:-2, 1:-1])

            # Viscous diffusion in x
            + nu * dt / dx**2
            * (
                vn[1:-1, 2:]
                - 2 * vn[1:-1, 1:-1]
                + vn[1:-1, 0:-2]
            )

            # Viscous diffusion in y
            + nu * dt / dy**2
            * (
                vn[2:, 1:-1]
                - 2 * vn[1:-1, 1:-1]
                + vn[0:-2, 1:-1]
            )
        )

        return u, v

    # Same explicit Euler update with a selectable advection scheme
    u[1:-1, 1:-1] = un[1:-1, 1:-1] + dt * (
        # Advection
        - advection_terms(un, un, vn, dx, dy, advection)

        # Pressure-gradient force
        - (p[1:-1, 2:] - p[1:-1, 0:-2]) / (2 * rho * dx)

        # Viscous diffusion
        + nu * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) / dx**2
        + nu * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) / dy**2
    )

    v[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt * (
        # Advection
        - advection_terms(vn, un, vn, dx, dy, advection)

        # Pressure-gradient force
        - (p[2:, 1:-1] - p[0:-2, 1:-1]) / (2 * rho * dy)

        # Viscous diffusion
        + nu * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) / dx**2
        + nu * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) / dy**2
    )

    return u, v


def projection_step(u, v, p, dx, dy, dt, nu, rho, advection, pressure_lu,
                    pressure_bc, lid_u, incremental=False):
    # Fractional-step projection.
    # Non-incremental (Chorin):
    #   1. intermediate velocity from advection and diffusion only
    #        u* = uⁿ + Δt (−(u·∇)u + ν ∇²u)
    #   2. pressure that removes the divergence of u*
    #        ∇²p = (ρ / Δt) ∇·u*
    #   3. projection
    #        uⁿ⁺¹ = u* − (Δt / ρ) ∇p
    # Incremental pressure correction:
    #   1. u* = uⁿ + Δt (−(u·∇)u + ν ∇²u − ∇pⁿ / ρ)
    #   2. pressure increment φ:  ∇²φ = (ρ / Δt) ∇·u*
    #   3. uⁿ⁺¹ = u* − (Δt / ρ) ∇φ,   pⁿ⁺¹ = pⁿ + φ
    #   At a steady state φ = 0, so the discrete continuity equation
    #   ∇·u = 0 and the momentum balance hold exactly, independently of Δt.
    un = u.copy()
    vn = v.copy()
    us = u.copy()
    vs = v.copy()

    us[1:-1, 1:-1] = un[1:-1, 1:-1] + dt * (
        - advection_terms(un, un, vn, dx, dy, advection)
        + nu * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) / dx**2
        + nu * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) / dy**2
    )
    vs[1:-1, 1:-1] = vn[1:-1, 1:-1] + dt * (
        - advection_terms(vn, un, vn, dx, dy, advection)
        + nu * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) / dx**2
        + nu * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) / dy**2
    )
    if incremental:
        # Previous pressure gradient included in the intermediate velocity
        us[1:-1, 1:-1] -= dt / rho * (p[1:-1, 2:] - p[1:-1, 0:-2]) / (2 * dx)
        vs[1:-1, 1:-1] -= dt / rho * (p[2:, 1:-1] - p[0:-2, 1:-1]) / (2 * dy)
    us, vs = apply_velocity_boundary_conditions(us, vs, lid_u)

    # Pressure Poisson equation with the divergence of u* as its source
    b = np.zeros_like(p)
    b[1:-1, 1:-1] = rho / dt * (
        (us[1:-1, 2:] - us[1:-1, 0:-2]) / (2 * dx)
        + (vs[2:, 1:-1] - vs[0:-2, 1:-1]) / (2 * dy)
    )
    phi = solve_pressure_direct(np.zeros_like(p), b, pressure_lu, pressure_bc)

    # Projection onto (approximately) divergence-free velocity
    u = us.copy()
    v = vs.copy()
    u[1:-1, 1:-1] = us[1:-1, 1:-1] - dt / rho * (phi[1:-1, 2:] - phi[1:-1, 0:-2]) / (2 * dx)
    v[1:-1, 1:-1] = vs[1:-1, 1:-1] - dt / rho * (phi[2:, 1:-1] - phi[0:-2, 1:-1]) / (2 * dy)

    if incremental:
        p = apply_pressure_boundary_conditions(p + phi, pressure_bc)
    else:
        p = phi

    return u, v, p, b


def apply_velocity_boundary_conditions(u, v, lid_u=LID_VELOCITY):
    # Left wall: no-slip
    u[:, 0] = WALL_VELOCITY
    v[:, 0] = WALL_VELOCITY

    # Right wall: no-slip
    u[:, -1] = WALL_VELOCITY
    v[:, -1] = WALL_VELOCITY

    # Bottom wall: no-slip
    u[0, :] = WALL_VELOCITY
    v[0, :] = WALL_VELOCITY

    # Top wall: moving lid (applied last, so the two top corners take the
    # lid velocity)
    u[-1, :] = lid_u
    v[-1, :] = WALL_VELOCITY

    return u, v


def divergence_rms(u, v, dx, dy):
    # Root-mean-square of the discrete velocity divergence ∂u/∂x + ∂v/∂y at
    # interior points, using second-order central differences.
    # Measures how well the pressure step enforces incompressibility.
    div = (
        (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
        + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
    )
    return float(np.sqrt(np.mean(div**2)))


def pressure_residual_rms(p, b, dx, dy):
    # Root-mean-square residual of the discrete pressure Poisson equation
    # ∇²p - b at interior points (boundary values already applied).
    # Zero to round-off for the direct solver; measures how far a fixed
    # number of Jacobi sweeps is from the exact discrete solution.
    lap = (
        (p[1:-1, 2:] - 2 * p[1:-1, 1:-1] + p[1:-1, 0:-2]) / dx**2
        + (p[2:, 1:-1] - 2 * p[1:-1, 1:-1] + p[0:-2, 1:-1]) / dy**2
    )
    return float(np.sqrt(np.mean((lap - b[1:-1, 1:-1]) ** 2)))


def run_solver(
    nx=NX,
    ny=NY,
    dt=DT,
    nt=NT,
    nu=NU,
    rho=RHO,
    tolerance=VELOCITY_TOLERANCE,
    advection=ADVECTION_SCHEME,
    pressure_solver=PRESSURE_SOLVER,
    pressure_iterations=PRESSURE_ITERATIONS,
    lid_ramp=LID_RAMP,
    pressure_bc=PRESSURE_BC,
    time_scheme=TIME_SCHEME,
    snapshot_every=None,
    stop_when_steady=True,
    history_every=None,
):
    # Advances the cavity flow from rest.
    #   snapshot_every: if set, store (u, v, p) every this many steps,
    #                   starting with the initial state at t = 0
    #   stop_when_steady: stop once the largest velocity change in one step
    #                     falls below `tolerance`
    #   history_every: if set, record step, time, velocity change and
    #                  divergence RMS every this many steps
    # With all arguments left at their defaults this is the original solver.

    # Create grid and initialize flow variables
    x, y, dx, dy = create_grid(nx, ny)
    u, v, p = initialize_variables(nx, ny)

    pressure_lu = None
    if pressure_solver == "direct":
        pressure_lu = build_pressure_matrix(nx, ny, dx, dy, pressure_bc)
    elif pressure_solver != "jacobi":
        raise ValueError(f"Unknown pressure solver: {pressure_solver}")
    if pressure_bc not in ("lid_dirichlet", "neumann"):
        raise ValueError(f"Unknown pressure boundary condition: {pressure_bc}")
    if time_scheme not in ("pressure_source", "projection", "incremental_projection"):
        raise ValueError(f"Unknown time scheme: {time_scheme}")
    if time_scheme != "pressure_source" and pressure_solver != "direct":
        raise ValueError("The projection scheme uses the direct pressure solver")

    # Apply wall velocities before the first time step
    u, v = apply_velocity_boundary_conditions(u, v, lid_velocity(0.0, lid_ramp))

    snapshots = None
    if snapshot_every:
        snapshots = {"t": [0.0], "u": [u.copy()], "v": [v.copy()], "p": [p.copy()]}
    history = []

    # Advance the solution through time
    for step in range(nt):
        # Save velocities from the previous time step
        u_previous = u.copy()
        v_previous = v.copy()

        # Lid velocity at the new time level t = (step + 1) dt
        lid_u = lid_velocity((step + 1) * dt, lid_ramp)

        if time_scheme in ("projection", "incremental_projection"):
            u, v, p, b = projection_step(u, v, p, dx, dy, dt, nu, rho, advection,
                                         pressure_lu, pressure_bc, lid_u,
                                         incremental=(time_scheme == "incremental_projection"))
        else:
            b = build_pressure_source(u, v, dx, dy, dt, rho)
            if pressure_solver == "direct":
                p = solve_pressure_direct(p, b, pressure_lu, pressure_bc)
            else:
                p = solve_pressure(p, b, dx, dy, pressure_iterations, pressure_bc)
            u, v = update_velocity(u, v, p, dx, dy, dt, nu, rho, advection)

        # Boundary values at the new time level
        u, v = apply_velocity_boundary_conditions(u, v, lid_u)

        # Largest velocity change between consecutive time steps
        velocity_change = max(
            np.max(np.abs(u - u_previous)),
            np.max(np.abs(v - v_previous)),
        )

        if history_every and (step + 1) % history_every == 0:
            history.append({
                "step": step + 1,
                "t": (step + 1) * dt,
                "velocity_change": float(velocity_change),
                "divergence_rms": divergence_rms(u, v, dx, dy),
                "divergence_rms_core": divergence_rms(u[3:-3, 3:-3], v[3:-3, 3:-3], dx, dy),
                "pressure_residual_rms": pressure_residual_rms(p, b, dx, dy),
            })

        if snapshot_every and (step + 1) % snapshot_every == 0:
            snapshots["t"].append((step + 1) * dt)
            snapshots["u"].append(u.copy())
            snapshots["v"].append(v.copy())
            snapshots["p"].append(p.copy())

        if not np.isfinite(velocity_change):
            break

        # Stop early if the velocity field is sufficiently steady
        if stop_when_steady and velocity_change < tolerance:
            break

    if snapshot_every or history_every:
        return x, y, u, v, p, velocity_change, step + 1, snapshots, history
    return x, y, u, v, p, velocity_change, step + 1
