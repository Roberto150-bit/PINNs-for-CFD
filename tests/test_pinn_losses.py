"""G11: PINN boundary and initial-condition losses are exactly zero on fields
built to satisfy them, and clearly nonzero otherwise.

    python -m tests.test_pinn_losses
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "models", "pinn", "src"))

from models.common.network import configure_tensorflow  # noqa: E402


def main():
    tf = configure_tensorflow("cpu", seed=0)
    import losses
    import config as pinn_config

    pts = losses.sample_points(pinn_config.N_COLLOCATION, pinn_config.N_INITIAL,
                               pinn_config.N_BOUNDARY_PER_WALL, pinn_config.SAMPLING_SEED)
    bc = [tf.constant(a) for a in pts["bc"]]
    is_lid = tf.constant(pts["bc_is_lid"].astype("float32"))
    ic = [tf.constant(a) for a in pts["ic"]]
    bx, by = pts["bc"][0][:, 0], pts["bc"][1][:, 0]

    # Field satisfying every boundary condition (corners excluded):
    #   u = tanh(10 t) on the lid y = 1, 0 on the other walls; v = 0 on all walls
    def bc_field(x, y, t):
        lid = tf.cast(y >= 1.0, tf.float32)
        u = lid * tf.tanh(10.0 * t) + x * (1 - x) * y * (1 - y) * tf.sin(t)
        v = x * (1 - x) * y * (1 - y) * tf.cos(3 * t)
        return u, v, tf.sin(x + y + t)

    # Field that is zero at t = 0 but nonzero elsewhere
    def ic_field(x, y, t):
        return t * tf.cos(x), tf.sin(t) * y, tf.ones_like(x)

    def wrong_field(x, y, t):
        return tf.ones_like(x) * 0.3, tf.ones_like(x) * 0.2, tf.zeros_like(x)

    l_bc = float(losses.boundary_loss(bc_field, *bc, is_lid))
    l_bc_wrong = float(losses.boundary_loss(wrong_field, *bc, is_lid))
    l_ic = float(losses.initial_loss(ic_field, *ic))
    l_ic_wrong = float(losses.initial_loss(wrong_field, *ic))

    checks = [
        ("G11 BC loss on a field satisfying the BCs", l_bc == 0.0, f"{l_bc:.3e} (expected 0)"),
        ("G11 BC loss on a field violating them", l_bc_wrong > 1e-2, f"{l_bc_wrong:.3e}"),
        ("G11 IC loss on a field that is 0 at t = 0", l_ic == 0.0, f"{l_ic:.3e} (expected 0)"),
        ("G11 IC loss on a nonzero field", l_ic_wrong > 1e-2, f"{l_ic_wrong:.3e}"),
        ("boundary points: 1,000 per wall, 4 walls", len(bx) == 4000, len(bx)),
        ("corners (0,1), (1,1) excluded", not any(((bx == 0) | (bx == 1)) & (by == 1)), "lid corners absent"),
        ("lid points flagged", int(pts["bc_is_lid"].sum()) == 1000 and bool((by[pts["bc_is_lid"][:, 0] == 1] == 1).all()),
         int(pts["bc_is_lid"].sum())),
        ("initial points at t = 0, interior", bool((pts["ic"][2] == 0).all()
                                                   and (pts["ic"][0] > 0).all() and (pts["ic"][0] < 1).all()),
         len(pts["ic"][0])),
        ("collocation points: 20,000 interior", len(pts["col"][0]) == 20000
         and bool((pts["col"][2] > 0).all() and (pts["col"][2] <= 5).all()), len(pts["col"][0])),
    ]
    ok = True
    for name, passed, detail in checks:
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
