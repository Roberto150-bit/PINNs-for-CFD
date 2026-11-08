"""Build a 64 x 64 reference field from a finer FD run whose grid contains it.

    python models/finite_difference/src/subsample_reference.py --re 1000 --source-n 127

On n points per side the grid spacing is 1/(n-1). A grid of 2*63 + 1 = 127
points therefore contains every point of the 64 x 64 grid (spacing 1/63):
taking every second point gives exactly the 64 x 64 coordinates. The finer
run is solved on its own grid and only its stored fields are subsampled, so
nothing is interpolated.

Writes <experiment dir>/finite_difference/fd_re<Re>_n<source>to64_transient/
(solution.npz plus a config.json that records the source run and stride).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
from models.common import config as C  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, required=True)
    parser.add_argument("--source-n", type=int, default=127)
    args = parser.parse_args()

    n_out = C.GRID_N
    stride = (args.source_n - 1) // (n_out - 1)
    if stride * (n_out - 1) != args.source_n - 1:
        raise SystemExit(f"a {args.source_n}-point grid does not contain the {n_out}-point grid")

    fd_dir = os.path.join(C.experiment_dir(args.re), "finite_difference")
    src_id = f"fd_re{args.re:g}_n{args.source_n}_transient"
    src_dir = os.path.join(fd_dir, src_id)
    out_id = f"fd_re{args.re:g}_n{args.source_n}to{n_out}_transient"
    out_dir = os.path.join(fd_dir, out_id)
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(os.path.join(src_dir, "solution.npz"))
    sub = {k: d[k][..., ::stride, ::stride] for k in ("u", "v", "p")}
    x, y = d["x"][::stride], d["y"][::stride]
    # The subsampled coordinates must be the 64 x 64 grid coordinates
    target = np.linspace(C.X_MIN, C.X_MAX, n_out)
    assert np.allclose(x, target, atol=1e-12) and np.allclose(y, target, atol=1e-12)
    np.savez(os.path.join(out_dir, "solution.npz"), x=x, y=y, t=d["t"], **sub)

    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    cfg.update({
        "run_id": out_id,
        "grid": {"nx": n_out, "ny": n_out, "dx": float(x[1] - x[0])},
        "derived_from": {"run_id": src_id, "stride": stride, "solved_on": [args.source_n, args.source_n],
                         "source_sha256": sha256(os.path.join(src_dir, "solution.npz")),
                         "method": "every second grid point of the finer solution (no interpolation)"},
    })
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    # Cost of producing this field is the cost of the source run
    shutil.copy(os.path.join(src_dir, "performance.json"), os.path.join(out_dir, "performance.json"))
    print(f"wrote {out_dir}: fields {sub['u'].shape} from {src_id} (stride {stride})")


if __name__ == "__main__":
    main()
