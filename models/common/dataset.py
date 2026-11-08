"""Space-time dataset built from the 64 x 64 finite-difference transient.

Each FD snapshot (u, v, p) on the 64 x 64 grid at t = 0.0, 0.1, ..., 5.0
becomes 4,096 rows of inputs (x, y, t) and targets (u, v, p):
51 x 64 x 64 = 208,896 rows. Row order is time-major, then y, then x:
    row = k * 64 * 64 + j * 64 + i   for snapshot k, grid row j, column i.

Splits (identical for every run; saved with the data):
  test B : the 10 whole snapshots t = 0.5, 1.0, ..., 5.0 (time interpolation)
  rest   : the other 41 snapshots, shuffled with seed 0 and split 70/15/15
           into train / validation / test A

Inputs are scaled to [-1, 1] by `scale_inputs`; models always receive
scaled inputs, and physical derivatives are taken through the scaling.

Build and check (from the repository root):
    python -m models.common.dataset --re 100 --build
    python -m models.common.dataset --re 100 --check
"""
import argparse
import hashlib
import json
import os

import numpy as np

from models.common import config as C

INPUT_LOW = np.array([C.X_MIN, C.Y_MIN, C.T_MIN])
INPUT_HIGH = np.array([C.X_MAX, C.Y_MAX, C.T_MAX])


def scale_inputs(X):
    # Physical (x, y, t) -> [-1, 1]^3. Works on NumPy arrays and TF tensors.
    return 2.0 * (X - INPUT_LOW) / (INPUT_HIGH - INPUT_LOW) - 1.0


def input_scale_factors():
    # d(scaled)/d(physical) for x, y, t: the chain-rule factors
    return 2.0 / (INPUT_HIGH - INPUT_LOW)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fd_solution_path(re, n=C.GRID_N):
    return os.path.join(C.EXPERIMENT_DIR, "finite_difference",
                        C.fd_reference_run_id(re, n), "solution.npz")


def flatten_fields(x, y, t, u, v, p):
    # (nt, ny, nx) fields -> rows of (x, y, t) and (u, v, p)
    T, Y, X = np.meshgrid(t, y, x, indexing="ij")
    inputs = np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1)
    targets = np.stack([u.ravel(), v.ravel(), p.ravel()], axis=1)
    return inputs, targets


def make_splits(t, n_points_per_snapshot):
    # Test B: whole snapshots at TEST_B_TIMES
    snapshot_ids = np.arange(len(t))
    test_b_snaps = np.array([int(np.argmin(np.abs(t - tb))) for tb in C.TEST_B_TIMES])
    assert np.allclose(t[test_b_snaps], C.TEST_B_TIMES)
    rest_snaps = np.setdiff1d(snapshot_ids, test_b_snaps)

    def rows_of(snaps):
        return (snaps[:, None] * n_points_per_snapshot
                + np.arange(n_points_per_snapshot)[None, :]).ravel()

    idx_test_b = rows_of(test_b_snaps)
    rest = rows_of(rest_snaps)

    rng = np.random.default_rng(C.SPLIT_SEED)
    rest = rng.permutation(rest)
    n_train = int(C.SPLIT_FRACTIONS[0] * len(rest))
    n_val = int(C.SPLIT_FRACTIONS[1] * len(rest))
    idx_train = np.sort(rest[:n_train])
    idx_val = np.sort(rest[n_train:n_train + n_val])
    idx_test_a = np.sort(rest[n_train + n_val:])
    return idx_train, idx_val, idx_test_a, np.sort(idx_test_b)


def build(re):
    src = fd_solution_path(re)
    d = np.load(src)
    x, y, t, u, v, p = (d[k] for k in ("x", "y", "t", "u", "v", "p"))
    inputs, targets = flatten_fields(x, y, t, u, v, p)
    idx_train, idx_val, idx_test_a, idx_test_b = make_splits(t, len(x) * len(y))

    out = C.dataset_path(re)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, inputs=inputs, targets=targets, x=x, y=y, t=t, u=u, v=v, p=p,
             idx_train=idx_train, idx_val=idx_val, idx_test_a=idx_test_a,
             idx_test_b=idx_test_b)
    meta = {
        "reynolds_number": re,
        "source_run_id": C.fd_reference_run_id(re),
        "source_file": src,
        "source_sha256": sha256(src),
        "rows": int(len(inputs)),
        "row_order": "time-major, then y, then x: k*ny*nx + j*nx + i",
        "split_seed": C.SPLIT_SEED,
        "split_fractions": C.SPLIT_FRACTIONS,
        "test_b_times": C.TEST_B_TIMES,
        "counts": {"train": int(len(idx_train)), "val": int(len(idx_val)),
                   "test_a": int(len(idx_test_a)), "test_b": int(len(idx_test_b))},
        "input_scaling": "scaled = 2 (X - low) / (high - low) - 1",
        "input_low": INPUT_LOW.tolist(), "input_high": INPUT_HIGH.tolist(),
        "dataset_sha256": sha256(out),
    }
    with open(out.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return out


def load(re):
    return dict(np.load(C.dataset_path(re)))


def check(re):
    # Dataset gates: G5 (fields) and G6 (splits). Returns True if all pass.
    d = load(re)
    ok = True

    def report(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    n = C.GRID_N
    for k in ("u", "v", "p"):
        report(f"G5 {k} shape", d[k].shape == (51, n, n), d[k].shape)
    interior0 = max(np.abs(d[k][0, 1:-1, 1:-1]).max() for k in ("u", "v", "p"))
    report("G5 interior of t=0 is zero (u, v, p)", interior0 == 0.0, f"max |value| = {interior0:.3e}")
    lid_expected = C.LID_VELOCITY * np.tanh(C.LID_RAMP_RATE * d["t"])
    lid_err = np.abs(d["u"][:, -1, :] - lid_expected[:, None]).max()
    report("G5 lid row u = tanh(10 t)", lid_err < 1e-12, f"max |error| = {lid_err:.3e}")
    report("G5 rows", len(d["inputs"]) == 208896, len(d["inputs"]))

    counts = {k: len(d[f"idx_{k}"]) for k in ("train", "val", "test_a", "test_b")}
    expected = {"train": 117555, "val": 25190, "test_a": 25191, "test_b": 40960}
    report("G6 split counts", counts == expected, counts)
    sets = {k: set(d[f"idx_{k}"].tolist()) for k in counts}
    names = list(sets)
    overlap = sum(len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:])
    report("G6 no overlap", overlap == 0, f"shared rows = {overlap}")
    report("G6 splits cover all rows", sum(counts.values()) == len(d["inputs"]),
           sum(counts.values()))
    train_times = np.unique(np.round(d["inputs"][d["idx_train"], 2], 6))
    clash = np.intersect1d(train_times, np.round(C.TEST_B_TIMES, 6))
    report("G6 test-B times absent from train", len(clash) == 0, f"clashing times = {clash.tolist()}")
    scaled = scale_inputs(d["inputs"])
    report("G6 scaled inputs in [-1, 1]", scaled.min() >= -1 - 1e-12 and scaled.max() <= 1 + 1e-12,
           f"min {scaled.min():.6f}, max {scaled.max():.6f}")
    # Row-order check: a random row matches the field it came from
    rng = np.random.default_rng(1)
    rows = rng.integers(0, len(d["inputs"]), 1000)
    k, rem = np.divmod(rows, n * n)
    j, i = np.divmod(rem, n)
    match = (np.allclose(d["inputs"][rows, 0], d["x"][i]) and np.allclose(d["inputs"][rows, 1], d["y"][j])
             and np.allclose(d["inputs"][rows, 2], d["t"][k])
             and np.allclose(d["targets"][rows, 0], d["u"][k, j, i])
             and np.allclose(d["targets"][rows, 2], d["p"][k, j, i]))
    report("row order (x, y, t) <-> field[k, j, i]", match, "1000 random rows")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.build:
        build(args.re)
    if args.check or not args.build:
        raise SystemExit(0 if check(args.re) else 1)


if __name__ == "__main__":
    main()
