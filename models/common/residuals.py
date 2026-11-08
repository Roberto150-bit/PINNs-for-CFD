"""Residuals of the unsteady incompressible Navier–Stokes equations (ρ = 1).

    r_c = u_x + v_y
    r_x = u_t + u u_x + v u_y + p_x − ν (u_xx + u_yy)
    r_y = v_t + u v_x + v v_y + p_y − ν (v_xx + v_yy)

`fields` is any callable (x, y, t) -> (u, v, p) on column tensors of
physical coordinates. Derivatives are taken by automatic differentiation
with respect to the physical coordinates, so when `fields` applies the
[-1, 1] input scaling internally the chain rule factors (2 / L) are
included automatically.

Used by the PINN loss (inside tf.function) and by the evaluator, which
computes the same residuals for both networks.
"""
import tensorflow as tf

from models.common.dataset import scale_inputs


def network_fields(model):
    # Wraps a network that takes scaled (x, y, t) and returns (u, v, p)
    def fields(x, y, t):
        X = tf.concat([x, y, t], axis=1)
        out = model(scale_inputs_tf(X))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
    return fields


def scale_inputs_tf(X):
    return tf.cast(scale_inputs(X), X.dtype)


def navier_stokes_residuals(fields, x, y, t, nu):
    """Returns (r_c, r_x, r_y), each of shape (N, 1)."""
    with tf.GradientTape(persistent=True) as outer:
        outer.watch([x, y])
        with tf.GradientTape(persistent=True) as inner:
            inner.watch([x, y, t])
            u, v, p = fields(x, y, t)
        u_x, u_y, u_t = inner.gradient(u, [x, y, t])
        v_x, v_y, v_t = inner.gradient(v, [x, y, t])
        p_x, p_y = inner.gradient(p, [x, y])
        del inner
    # Second derivatives for the viscous terms
    u_xx = outer.gradient(u_x, x)
    u_yy = outer.gradient(u_y, y)
    v_xx = outer.gradient(v_x, x)
    v_yy = outer.gradient(v_y, y)
    del outer

    r_c = u_x + v_y
    r_x = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    r_y = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    return r_c, r_x, r_y


def taylor_green_fields(nu):
    # Exact decaying Taylor–Green vortex (ρ = 1):
    #   u = −cos x sin y e^(−2νt),  v = sin x cos y e^(−2νt),
    #   p = −¼ (cos 2x + cos 2y) e^(−4νt)
    def fields(x, y, t):
        F = tf.exp(-2.0 * nu * t)
        u = -tf.cos(x) * tf.sin(y) * F
        v = tf.sin(x) * tf.cos(y) * F
        p = -0.25 * (tf.cos(2.0 * x) + tf.cos(2.0 * y)) * F * F
        return u, v, p
    return fields
