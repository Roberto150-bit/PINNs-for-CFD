"""G10: Navier–Stokes residuals vanish for the exact Taylor–Green vortex.

The exact field must give residual RMS below 1e-5 in float32; a perturbed
field must give a clearly nonzero residual. Also checks the chain rule
through the [-1, 1] input scaling: a network-style wrapper that scales the
inputs before evaluating the exact field gives the same residuals.

    python -m tests.test_residuals
"""
import numpy as np

from models.common.network import configure_tensorflow


def main():
    tf = configure_tensorflow("cpu", seed=0)
    from models.common.residuals import navier_stokes_residuals, taylor_green_fields
    from models.common.dataset import INPUT_LOW, INPUT_HIGH

    nu = 0.01
    rng = np.random.default_rng(0)
    n = 50000
    pts = rng.uniform([0, 0, 0], [2 * np.pi, 2 * np.pi, 5.0], size=(n, 3)).astype("float32")
    x, y, t = (tf.constant(pts[:, k:k + 1]) for k in range(3))

    def rms(r):
        return float(tf.sqrt(tf.reduce_mean(tf.square(r))))

    exact = taylor_green_fields(nu)
    rc, rx, ry = navier_stokes_residuals(exact, x, y, t, nu)
    exact_rms = [rms(rc), rms(rx), rms(ry)]
    ok_exact = max(exact_rms) < 1e-5

    def perturbed(x, y, t):
        u, v, p = exact(x, y, t)
        return u + 0.01 * tf.sin(3.0 * x) * tf.sin(2.0 * y), v, p

    pc, px, py = navier_stokes_residuals(perturbed, x, y, t, nu)
    pert_rms = [rms(pc), rms(px), rms(py)]
    ok_pert = pert_rms[0] > 1e-3 and pert_rms[1] > 1e-3

    # Chain rule through input scaling: evaluate the exact field from scaled
    # inputs, mapping back to physical coordinates inside the graph
    low = tf.constant(INPUT_LOW, tf.float32)
    high = tf.constant(INPUT_HIGH, tf.float32)

    def via_scaling(x, y, t):
        X = tf.concat([x, y, t], axis=1)
        Xs = 2.0 * (X - low) / (high - low) - 1.0          # what the network sees
        Xp = (Xs + 1.0) / 2.0 * (high - low) + low          # inside the "network"
        return exact(Xp[:, 0:1], Xp[:, 1:2], Xp[:, 2:3])

    sc, sx, sy = navier_stokes_residuals(via_scaling, x, y, t, nu)
    scaled_rms = [rms(sc), rms(sx), rms(sy)]
    ok_scaled = max(scaled_rms) < 1e-5

    print(f"[{'PASS' if ok_exact else 'FAIL'}] G10 exact Taylor–Green residual RMS (r_c, r_x, r_y) = "
          + ", ".join(f"{r:.2e}" for r in exact_rms) + "  (< 1e-5, float32, 50,000 points)")
    print(f"[{'PASS' if ok_pert else 'FAIL'}] G10 perturbed field residual RMS = "
          + ", ".join(f"{r:.2e}" for r in pert_rms) + "  (clearly nonzero)")
    print(f"[{'PASS' if ok_scaled else 'FAIL'}] residuals through input scaling = "
          + ", ".join(f"{r:.2e}" for r in scaled_rms))
    raise SystemExit(0 if (ok_exact and ok_pert and ok_scaled) else 1)


if __name__ == "__main__":
    main()
