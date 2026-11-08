"""Training points and loss terms of the PINN.

All loss functions take `fields`, a callable (x, y, t) -> (u, v, p) on
column tensors of physical coordinates, so they can be tested on analytic
fields as well as on the network.
"""
import numpy as np
import tensorflow as tf

from models.common import config as C
from models.common.residuals import navier_stokes_residuals


def lid_velocity(t):
    # Smooth start, identical to the FD reference: u_lid(t) = tanh(10 t)
    return C.LID_VELOCITY * tf.tanh(C.LID_RAMP_RATE * t)


def sample_points(n_collocation, n_initial, n_per_wall, seed):
    """Fixed training points (float32 column arrays)."""
    rng = np.random.default_rng(seed)
    T0, T1 = C.T_MIN, C.T_MAX

    def col(a):
        return a.astype("float32")[:, None]

    def open_unit(n):
        # Uniform in the open interval (0, 1): excludes the corners exactly
        u = rng.uniform(0.0, 1.0, n)
        return np.where(u == 0.0, 0.5, u)

    pts = {}
    pts["col"] = [col(open_unit(n_collocation)), col(open_unit(n_collocation)),
                  col(rng.uniform(T0, T1, n_collocation))]
    pts["ic"] = [col(open_unit(n_initial)), col(open_unit(n_initial)), col(np.zeros(n_initial))]

    walls = []
    for name in ("lid", "bottom", "left", "right"):
        s = open_unit(n_per_wall)
        t = rng.uniform(T0, T1, n_per_wall)
        if name == "lid":
            x, y = s, np.ones(n_per_wall)
        elif name == "bottom":
            x, y = s, np.zeros(n_per_wall)
        elif name == "left":
            x, y = np.zeros(n_per_wall), s
        else:
            x, y = np.ones(n_per_wall), s
        walls.append((x, y, t, np.full(n_per_wall, name == "lid")))
    bx, by, bt, is_lid = (np.concatenate([w[k] for w in walls]) for k in range(4))
    pts["bc"] = [col(bx), col(by), col(bt)]
    pts["bc_is_lid"] = col(is_lid.astype("float64"))
    return pts


def initial_loss(fields, x, y, t):
    u, v, _ = fields(x, y, t)
    return tf.reduce_mean(tf.square(u)) + tf.reduce_mean(tf.square(v))


def boundary_loss(fields, x, y, t, is_lid):
    u, v, _ = fields(x, y, t)
    u_target = is_lid * lid_velocity(t)          # tanh(10 t) on the lid, 0 elsewhere
    return tf.reduce_mean(tf.square(u - u_target)) + tf.reduce_mean(tf.square(v))


def residual_losses(fields, x, y, t, nu):
    r_c, r_x, r_y = navier_stokes_residuals(fields, x, y, t, nu)
    return (tf.reduce_mean(tf.square(r_c)), tf.reduce_mean(tf.square(r_x)),
            tf.reduce_mean(tf.square(r_y)))
