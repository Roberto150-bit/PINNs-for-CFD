"""Validate steady finite-difference solutions against published benchmarks.

For each steady FD run (grids 32, 64, 128) this compares
  - u along the vertical centerline x = 0.5 and v along the horizontal
    centerline y = 0.5 with Marchi et al. (2009), Table 6, at the 15 tabulated
    stations (FD fields interpolated with a bicubic spline);
  - the primary-vortex center, taken as the location of the minimum stream
    function, with Marchi et al. (2009) and Hou et al. (1995).

The stream function follows Marchi et al.: ψ(x, y) = ∫_0^y u(x, s) ds,
integrated upward from the bottom wall (trapezoidal rule), so ψ < 0 in the
clockwise primary vortex. The grid minimum is refined to sub-grid precision
with a quadratic fit through the 3 x 3 neighborhood.

Usage:
    python experiments/analysis/fd_validation.py --re 100
Writes experiments/results/fd_validation_re<Re>.json and .csv.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import cumulative_trapezoid

BENCH_DIR = "experiments/benchmarks"


def read_commented_csv(path):
    with open(path) as f:
        rows = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(rows))


def load_marchi(re):
    rows = [r for r in read_commented_csv(os.path.join(BENCH_DIR, "marchi2009.csv"))
            if float(r["reynolds_number"]) == re]
    u = sorted((float(r["y"]), float(r["value"])) for r in rows if r["quantity"] == "u")
    v = sorted((float(r["x"]), float(r["value"])) for r in rows if r["quantity"] == "v")
    scalars = {r["quantity"]: float(r["value"]) for r in rows if r["quantity"] not in ("u", "v")}
    return np.array(u), np.array(v), scalars


def load_hou(re):
    rows = [r for r in read_commented_csv(os.path.join(BENCH_DIR, "hou1995.csv"))
            if float(r["reynolds_number"]) == re]
    return float(rows[0]["x"]), float(rows[0]["y"])


def stream_function(y, u):
    # ψ(x, y) = ∫_0^y u dy along each column; u has shape (ny, nx)
    return cumulative_trapezoid(u, y, axis=0, initial=0.0)


def vortex_center(x, y, psi):
    # Grid minimum of ψ, then a quadratic fit in each direction through the
    # minimum and its two neighbors for sub-grid location
    j, i = np.unravel_index(np.argmin(psi), psi.shape)
    xc, yc = x[i], y[j]
    if 0 < i < len(x) - 1:
        f0, f1, f2 = psi[j, i - 1], psi[j, i], psi[j, i + 1]
        denom = f0 - 2 * f1 + f2
        if denom != 0:
            xc = x[i] + 0.5 * (f0 - f2) / denom * (x[1] - x[0])
    if 0 < j < len(y) - 1:
        f0, f1, f2 = psi[j - 1, i], psi[j, i], psi[j + 1, i]
        denom = f0 - 2 * f1 + f2
        if denom != 0:
            yc = y[j] + 0.5 * (f0 - f2) / denom * (y[1] - y[0])
    return {"grid_x": float(x[i]), "grid_y": float(y[j]), "x": float(xc), "y": float(yc),
            "psi_min": float(psi[j, i])}


def centerline_profiles(x, y, u, v, y_stations, x_stations):
    # Fields are (ny, nx); the spline takes (y, x) order
    su = RectBivariateSpline(y, x, u, kx=3, ky=3)
    sv = RectBivariateSpline(y, x, v, kx=3, ky=3)
    u_line = su(y_stations, [0.5]).ravel()
    v_line = sv([0.5], x_stations).ravel()
    return u_line, v_line


def validate(re, grids, exp_dir):
    mu, mv, ms = load_marchi(re)
    hou = load_hou(re)
    marchi_center = (ms["x(psimin)"], ms["y(psimin)"])
    results = []
    for n in grids:
        run_id = f"fd_re{re:g}_n{n}_steady"
        run_dir = os.path.join(exp_dir, "finite_difference", run_id)
        d = np.load(os.path.join(run_dir, "solution.npz"))
        x, y, u, v = d["x"], d["y"], d["u"][-1], d["v"][-1]
        u_line, v_line = centerline_profiles(x, y, u, v, mu[:, 0], mv[:, 0])
        psi = stream_function(y, u)
        c = vortex_center(x, y, psi)
        dx = float(x[1] - x[0])
        res = {
            "run_id": run_id, "n": n, "dx": dx,
            "u_centerline_max_abs_dev": float(np.max(np.abs(u_line - mu[:, 1]))),
            "v_centerline_max_abs_dev": float(np.max(np.abs(v_line - mv[:, 1]))),
            "u_centerline_rms_dev": float(np.sqrt(np.mean((u_line - mu[:, 1]) ** 2))),
            "v_centerline_rms_dev": float(np.sqrt(np.mean((v_line - mv[:, 1]) ** 2))),
            "psi_min": c["psi_min"], "psi_min_marchi": ms["psimin"],
            "vortex_x": c["x"], "vortex_y": c["y"],
            "vortex_grid_x": c["grid_x"], "vortex_grid_y": c["grid_y"],
            "vortex_dist_marchi": float(np.hypot(c["x"] - marchi_center[0], c["y"] - marchi_center[1])),
            "vortex_dist_hou": float(np.hypot(c["x"] - hou[0], c["y"] - hou[1])),
            "vortex_dist_marchi_grid": float(np.hypot(c["grid_x"] - marchi_center[0],
                                                      c["grid_y"] - marchi_center[1])),
            "vortex_dist_hou_grid": float(np.hypot(c["grid_x"] - hou[0], c["grid_y"] - hou[1])),
            "u_centerline": u_line.tolist(), "v_centerline": v_line.tolist(),
        }
        results.append(res)
    out = {
        "reynolds_number": re,
        "marchi_u_stations_y": mu[:, 0].tolist(), "marchi_u": mu[:, 1].tolist(),
        "marchi_v_stations_x": mv[:, 0].tolist(), "marchi_v": mv[:, 1].tolist(),
        "marchi_vortex_center": marchi_center, "hou_vortex_center": hou,
        "runs": results,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--exp-dir", default="experiments/experiment_001_re100")
    parser.add_argument("--out-dir", default="experiments/results")
    args = parser.parse_args()

    out = validate(args.re, args.grids, args.exp_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"fd_validation_re{args.re:g}")
    with open(stem + ".json", "w") as f:
        json.dump(out, f, indent=2)
    keys = [k for k in out["runs"][0] if not k.endswith("centerline")]
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(out["runs"])

    print(f"Re = {args.re:g}; Marchi center {out['marchi_vortex_center']}, Hou center {out['hou_vortex_center']}")
    for r in out["runs"]:
        print(f"  n={r['n']:4d} dx={r['dx']:.4f}  max|du|={r['u_centerline_max_abs_dev']:.4e} "
              f"max|dv|={r['v_centerline_max_abs_dev']:.4e}  center=({r['vortex_x']:.4f},{r['vortex_y']:.4f}) "
              f"grid=({r['vortex_grid_x']:.4f},{r['vortex_grid_y']:.4f}) psi_min={r['psi_min']:.5f} "
              f"dist M={r['vortex_dist_marchi']:.4f} H={r['vortex_dist_hou']:.4f}")


if __name__ == "__main__":
    main()
