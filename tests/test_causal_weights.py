"""Causal weighting of the PINN residual (study S4) on cases with known answers.

    python -m tests.test_causal_weights
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "models", "pinn", "src"))

import numpy as np  # noqa: E402

from models.common.network import configure_tensorflow  # noqa: E402


def main():
    tf = configure_tensorflow("cpu", seed=0)
    import losses
    from models.common import config as C

    checks = []

    # 1. Chunk ids: 4 chunks over [0, 5]
    t = np.array([[0.0], [1.2], [1.25], [2.6], [4.99], [5.0]], dtype="float32")
    ids = losses.time_chunk_ids(t, 4)
    checks.append(("chunk ids", list(ids) == [0, 0, 1, 2, 3, 3], f"{list(ids)} (expected [0, 0, 1, 2, 3, 3])"))

    # 2. Chunk means
    vals = tf.constant([1.0, 3.0, 5.0, 7.0, 2.0, 4.0])
    means = losses.chunk_means(vals, tf.constant(ids), 4).numpy()
    checks.append(("chunk means", np.allclose(means, [2.0, 5.0, 7.0, 3.0]), f"{means}"))

    # 3. Weights: w_1 = 1, w_i = exp(-eps * sum of earlier losses)
    L = tf.constant([0.1, 0.2, 0.3, 0.4])
    w = losses.causal_weights(L, 2.0).numpy()
    expected = np.exp(-2.0 * np.array([0.0, 0.1, 0.3, 0.6]))
    checks.append(("weights formula", np.allclose(w, expected), f"{w} vs {expected}"))
    checks.append(("weights non-increasing", bool(np.all(np.diff(w) <= 0)), f"{w}"))

    # 4. eps = 0 gives all weights 1, so the loss is the plain mean of chunk losses
    w0 = losses.causal_weights(L, 0.0).numpy()
    checks.append(("eps = 0 -> weights 1", np.allclose(w0, 1.0), f"{w0}"))

    # 5. No gradient flows through the weights
    v = tf.Variable([0.1, 0.2, 0.3, 0.4])
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(losses.causal_weights(v, 5.0))
    g = tape.gradient(loss, v)
    checks.append(("weights carry no gradient", g is None, f"gradient {g}"))

    # 6. Full loss on an analytic field whose residual grows with t:
    #    u = v = 0, p = x * t^2 gives r_c = 0, r_x = dp/dx = t^2, r_y = 0
    n = 4000
    rng = np.random.default_rng(0)
    x = tf.constant(rng.uniform(0, 1, (n, 1)).astype("float32"))
    y = tf.constant(rng.uniform(0, 1, (n, 1)).astype("float32"))
    tt = rng.uniform(C.T_MIN, C.T_MAX, (n, 1)).astype("float32")
    chunk_ids = tf.constant(losses.time_chunk_ids(tt, 8))
    tt = tf.constant(tt)

    def field(x, y, t):
        # zero velocity written so autodiff still sees x, y and t
        zero = 0.0 * (x ** 2 + y ** 2) * t
        return zero, zero, x * t ** 2 + 0.0 * y

    l_c, l_x, l_y, l_pde, w_min = losses.causal_residual_losses(field, x, y, tt, 0.01, chunk_ids, 8,
                                                                tf.constant(0.0), 1.0, 1.0)
    r2 = tt.numpy()[:, 0] ** 4
    ids = chunk_ids.numpy()
    chunk = np.array([r2[ids == k].mean() for k in range(8)])
    checks.append(("analytic residual L_x = mean(t^4)", abs(float(l_x) - r2.mean()) < 1e-3 * r2.mean(),
                   f"{float(l_x):.4f} vs {r2.mean():.4f}"))
    checks.append(("eps = 0: L_pde = mean of chunk means", abs(float(l_pde) - chunk.mean()) < 1e-3 * chunk.mean(),
                   f"{float(l_pde):.4f} vs {chunk.mean():.4f}; min weight {float(w_min):.3f}"))
    _, _, _, l_pde_big, w_min_big = losses.causal_residual_losses(field, x, y, tt, 0.01, chunk_ids, 8,
                                                                  tf.constant(1.0), 1.0, 1.0)
    w_ref = np.exp(-1.0 * np.concatenate([[0.0], np.cumsum(chunk)[:-1]]))
    checks.append(("eps = 1: L_pde = mean(w_i L_i)", abs(float(l_pde_big) - (w_ref * chunk).mean()) < 1e-3,
                   f"{float(l_pde_big):.5f} vs {(w_ref * chunk).mean():.5f}; min weight {float(w_min_big):.2e} "
                   f"vs {w_ref.min():.2e}"))

    ok = True
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
