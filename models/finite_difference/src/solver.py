import numpy as np

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
    LID_VELOCITY
)


def create_grid():
    x = np.linspace(X_MIN, X_MAX, NX)   # horizontal grid-point
    y = np.linspace(Y_MIN, Y_MAX, NY)   # vertical grid-point

    dx = (X_MAX - X_MIN) / (NX - 1)     # spacing between x-points
    dy = (Y_MAX - Y_MIN) / (NY - 1)     # spacing between y-points

    return x, y, dx, dy


def initialize_variables():
    u = np.full((NY, NX), U_INITIAL)    # horizontal velocity at every grid point
    v = np.full((NY, NX), V_INITIAL)    # vertical velocity at every grid point
    p = np.full((NY, NX), P_INITIAL)    # pressure at every grid point

    return u, v, p


def build_pressure_source(u, v, dx, dy):
    # Builds the source term for the pressure Poisson equation
    # Incompressible flow requires velocity divergence: ∇ · u = 0
    b = np.zeros((NY, NX))

    # Calculate only at interior grid points using central differences
    b[1:-1, 1:-1] = RHO * (
        (1 / DT)
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


def solve_pressure(p, b, dx, dy):
    # Improves the pressure approximation
    for _ in range(PRESSURE_ITERATIONS):
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

        # Pressure boundary conditions
        p[:, -1] = p[:, -2]   # Right wall: zero pressure gradient
        p[:, 0] = p[:, 1]     # Left wall: zero pressure gradient
        p[0, :] = p[1, :]     # Bottom wall: zero pressure gradient
        p[-1, :] = 0.0        # Top wall: reference pressure
    
    return p


def update_velocity(u, v, p, dx, dy):
    # Save previous velocity fields
    un = u.copy()
    vn = v.copy()

    # Update horizontal velocity at interior grid points
    u[1:-1, 1:-1] = (
        un[1:-1, 1:-1]

        # Horizontal advection
        - un[1:-1, 1:-1] * DT / dx
        * (un[1:-1, 1:-1] - un[1:-1, 0:-2])

        # Vertical advection
        - vn[1:-1, 1:-1] * DT / dy
        * (un[1:-1, 1:-1] - un[0:-2, 1:-1])

        # Pressure-gradient force
        - DT / (2 * RHO * dx)
        * (p[1:-1, 2:] - p[1:-1, 0:-2])

        # Viscous diffusion in x
        + NU * DT / dx**2
        * (
            un[1:-1, 2:]
            - 2 * un[1:-1, 1:-1]
            + un[1:-1, 0:-2]
        )

        # Viscous diffusion in y
        + NU * DT / dy**2
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
        - un[1:-1, 1:-1] * DT / dx
        * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])

        # Vertical advection
        - vn[1:-1, 1:-1] * DT / dy
        * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])

        # Pressure-gradient force
        - DT / (2 * RHO * dy)
        * (p[2:, 1:-1] - p[0:-2, 1:-1])

        # Viscous diffusion in x
        + NU * DT / dx**2
        * (
            vn[1:-1, 2:]
            - 2 * vn[1:-1, 1:-1]
            + vn[1:-1, 0:-2]
        )

        # Viscous diffusion in y
        + NU * DT / dy**2
        * (
            vn[2:, 1:-1]
            - 2 * vn[1:-1, 1:-1]
            + vn[0:-2, 1:-1]
        )
    )

    return u, v


def apply_velocity_boundary_conditions(u, v):
    # Left wall: no-slip
    u[:, 0] = WALL_VELOCITY
    v[:, 0] = WALL_VELOCITY

    # Right wall: no-slip
    u[:, -1] = WALL_VELOCITY
    v[:, -1] = WALL_VELOCITY

    # Bottom wall: no-slip
    u[0, :] = WALL_VELOCITY
    v[0, :] = WALL_VELOCITY

    # Top wall: moving lid
    u[-1, :] = LID_VELOCITY
    v[-1, :] = WALL_VELOCITY

    return u, v


def run_solver():
    # Create grid and initialize flow variables
    x, y, dx, dy = create_grid()
    u, v, p = initialize_variables()

    # Apply wall velocities before the first time step
    u, v = apply_velocity_boundary_conditions(u, v)

    # Advance the solution through time
    for step in range(NT):
        # Save velocities from the previous time step
        u_previous = u.copy()
        v_previous = v.copy()

        b = build_pressure_source(u, v, dx, dy)
        p = solve_pressure(p, b, dx, dy)
        u, v = update_velocity(u, v, p, dx, dy)
        u, v = apply_velocity_boundary_conditions(u, v)

        # Largest velocity change between consecutive time steps
        velocity_change = max(
            np.max(np.abs(u - u_previous)),
            np.max(np.abs(v - v_previous)),
        )

        # Stop early if the velocity field is sufficiently steady
        if velocity_change < VELOCITY_TOLERANCE:
            break

    return x, y, u, v, p, velocity_change, step + 1