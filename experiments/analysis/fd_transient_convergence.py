"""Convergence check for the FD transient (no published transient benchmark).

Compares the 51 snapshots t = 0.0, 0.1, ..., 5.0 of
  - the 64 x 64 run at Δt against the same grid at Δt/2 (time convergence);
  - the 64 x 64 run against the 128 x 128 run interpolated to the 64 x 64
    points with a bicubic spline (grid convergence).
Pressure is compared after removing each snapshot's mean.

Usage:
    python experiments/analysis/fd_transient_convergence.py --re 100
Writes experiments/results/fd_transient_convergence_re<Re>.json and .csv.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline


def load(exp_dir, run_id):
    return np.load(os.path.join(exp_dir, "finite_difference", run_id, "solution.npz"))


def mean_removed(p):
    return p - p.mean(axis=(1, 2), keepdims=True)


def to_grid(src, x_dst, y_dst, field):
    out = np.empty((field.shape[0], len(y_dst), len(x_dst)))
    for k in range(field.shape[0]):
        out[k] = RectBivariateSpline(src["y"], src["x"], field[k], kx=3, ky=3)(y_dst, x_dst)
    return out


def compare(a, b_fields):
    # Max |difference| per snapshot for u, v and mean-removed p
    rows = []
    for k, t in enumerate(a["t"]):
        rows.append({
            "t": round(float(t), 6),
            "max_abs_du": float(np.abs(a["u"][k] - b_fields["u"][k]).max()),
            "max_abs_dv": float(np.abs(a["v"][k] - b_fields["v"][k]).max()),
            "max_abs_dp": float(np.abs(mean_removed(a["p"])[k] - mean_removed(b_fields["p"])[k]).max()),
            "rms_du": float(np.sqrt(np.mean((a["u"][k] - b_fields["u"][k]) ** 2))),
            "rms_dv": float(np.sqrt(np.mean((a["v"][k] - b_fields["v"][k]) ** 2))),
            "rms_dp": float(np.sqrt(np.mean((mean_removed(a["p"])[k] - mean_removed(b_fields["p"])[k]) ** 2))),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--exp-dir", default="experiments/experiment_001_re100")
    parser.add_argument("--out-dir", default="experiments/results")
    args = parser.parse_args()
    re = f"{args.re:g}"

    ref = load(args.exp_dir, f"fd_re{re}_n64_transient")
    half = load(args.exp_dir, f"fd_re{re}_n64_transient_dthalf")
    fine = load(args.exp_dir, f"fd_re{re}_n128_transient")
    assert np.allclose(ref["t"], half["t"]) and np.allclose(ref["t"], fine["t"])

    time_rows = compare(ref, {k: half[k] for k in ("u", "v", "p")})
    fine_on_64 = {k: to_grid(fine, ref["x"], ref["y"], fine[k]) for k in ("u", "v", "p")}
    grid_rows = compare(ref, fine_on_64)

    def summary(rows):
        last = rows[-1]
        return {
            "max_over_snapshots": max(max(r["max_abs_du"], r["max_abs_dv"]) for r in rows),
            "max_over_snapshots_p": max(r["max_abs_dp"] for r in rows),
            "at_t_end": max(last["max_abs_du"], last["max_abs_dv"]),
            "at_t_end_p": last["max_abs_dp"],
            "rms_at_t_end_u": last["rms_du"], "rms_at_t_end_v": last["rms_dv"],
            "rms_at_t_end_p": last["rms_dp"],
        }

    out = {
        "reynolds_number": args.re,
        "time_convergence": {"runs": [f"fd_re{re}_n64_transient", f"fd_re{re}_n64_transient_dthalf"],
                             "summary_velocity": summary(time_rows), "per_snapshot": time_rows},
        "grid_convergence": {"runs": [f"fd_re{re}_n64_transient", f"fd_re{re}_n128_transient"],
                             "summary_velocity": summary(grid_rows), "per_snapshot": grid_rows},
    }
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"fd_transient_convergence_re{re}")
    with open(stem + ".json", "w") as f:
        json.dump(out, f, indent=2)
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["max_abs_du", "max_abs_dv", "max_abs_dp", "rms_du", "rms_dv", "rms_dp"]
        w.writerow(["t"] + [f"dt_{c}" for c in cols] + [f"grid_{c}" for c in cols])
        for a, b in zip(time_rows, grid_rows):
            w.writerow([a["t"]] + [a[c] for c in cols] + [b[c] for c in cols])

    for name in ("time_convergence", "grid_convergence"):
        s = out[name]["summary_velocity"]
        print(f"{name}: max over snapshots (u,v) {s['max_over_snapshots']:.3e}; at t=5 (u,v) "
              f"{s['at_t_end']:.3e}; p max {s['max_over_snapshots_p']:.3e}, p at t=5 {s['at_t_end_p']:.3e}; "
              f"RMS at t=5 u {s['rms_at_t_end_u']:.3e} v {s['rms_at_t_end_v']:.3e} p {s['rms_at_t_end_p']:.3e}")
    for r in time_rows[::10]:
        print(f"  dt  t={r['t']:.1f} du={r['max_abs_du']:.2e} dv={r['max_abs_dv']:.2e} dp={r['max_abs_dp']:.2e}")
    for r in grid_rows[::10]:
        print(f"  grid t={r['t']:.1f} du={r['max_abs_du']:.2e} dv={r['max_abs_dv']:.2e} dp={r['max_abs_dp']:.2e}")


if __name__ == "__main__":
    main()
