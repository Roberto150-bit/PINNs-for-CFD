"""G14: the shared evaluator scores the FD reference against itself as
exactly zero error, and FD with 1% multiplicative Gaussian noise as
relative L2 ≈ 0.01.

    python -m tests.test_evaluator
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "analysis"))
import evaluate_all as E  # noqa: E402


def main():
    ref, data = E.load_reference(100.0)
    raw = np.load(os.path.join(E.C.EXPERIMENT_DIR, "finite_difference",
                               E.C.fd_reference_run_id(100.0), "solution.npz"))
    ok = True

    def report(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    same = E.prepare(raw)
    vals = []
    for split, rows in (("test_a", data["idx_test_a"]), ("test_b", data["idx_test_b"]), ("full", None)):
        vals += list(E.accuracy(same, ref, rows).values())
    snap = E.per_snapshot_rel_l2(same, ref)
    vals += [v for k in ("u", "v", "p") for v in snap[k] if v is not None]
    report("G14 FD vs itself: every MSE and relative L2 exactly 0", all(v == 0.0 for v in vals),
           f"{len(vals)} values, max {max(vals):.1e}")

    rng = np.random.default_rng(0)
    noisy = {k: raw[k] * (1 + 0.01 * rng.standard_normal(raw[k].shape)) for k in ("u", "v", "p")}
    acc = E.accuracy(E.prepare(noisy), ref)
    rl2 = [acc[f"relative_l2_{k}"] for k in ("u", "v", "p")]
    report("G14 FD + 1% noise: relative L2 ≈ 0.01 (u, v, p)", all(abs(r - 0.01) < 0.001 for r in rl2),
           ", ".join(f"{r:.5f}" for r in rl2))

    # Pressure-constant invariance: adding a constant per snapshot changes nothing
    shifted = dict(raw)
    shifted = {k: raw[k] for k in ("u", "v", "p")}
    shifted["p"] = raw["p"] + np.arange(raw["p"].shape[0])[:, None, None] * 3.0
    acc_s = E.accuracy(E.prepare(shifted), ref)
    report("pressure compared up to a per-snapshot constant", acc_s["relative_l2_p"] < 1e-12,
           f"{acc_s['relative_l2_p']:.1e}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
