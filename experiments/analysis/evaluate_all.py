"""One evaluator for all three methods.

For every run folder in the experiment (FD solution.npz, network
predictions.npz) this computes, against the 64 x 64 FD reference
(fd_re<Re>_n64_transient):

Accuracy     MSE and relative L2 of u, v, p on test A, test B and the full
             field. Pressure is compared after subtracting each snapshot's
             mean (pressure is defined only up to a constant).
Over time    relative L2 of u, v, p per snapshot (51 values)
Physics      networks: RMS of r_c, r_x, r_y by automatic differentiation
             at 50,000 fresh random interior points (fixed seed)
             all methods: continuity ∂u/∂x + ∂v/∂y by the same
             second-order central stencil on the 64 x 64 grid, every
             snapshot (full interior and a core region 3 points from walls)
Constraints  RMS boundary error on the 64 x 64 wall nodes (lid corners
             excluded) over all snapshots, and RMS initial-condition error
             (u, v at t = 0, interior nodes)
Cost         copied from each run's performance.json

Writes metrics.json into each run folder and
experiments/results/experiment_results.csv (one row per run).

    python experiments/analysis/evaluate_all.py [--re 100] [--only RUN_ID ...]
"""
import argparse
import csv
import glob
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from models.common import config as C  # noqa: E402

PHYSICS_POINTS = 50000
PHYSICS_SEED = 2024
CORE = 3


# ---------- field metrics (pure NumPy; also used by the tests) ----------

def mean_removed(p):
    return p - p.mean(axis=(1, 2), keepdims=True)


def prepare(fields):
    # (u, v, p) arrays of shape (nt, ny, nx); pressure mean removed per snapshot
    return {"u": fields["u"], "v": fields["v"], "p": mean_removed(fields["p"])}


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def accuracy(pred, ref, rows=None):
    # pred, ref: prepared dicts; rows: flat row indices (dataset order) or None
    out = {}
    for k in ("u", "v", "p"):
        a, b = pred[k].ravel(), ref[k].ravel()
        if rows is not None:
            a, b = a[rows], b[rows]
        out[f"mse_{k}"] = mse(a, b)
        out[f"relative_l2_{k}"] = rel_l2(a, b)
    return out


def per_snapshot_rel_l2(pred, ref):
    # None where the reference snapshot is identically zero (t = 0: fluid at rest)
    return {k: [rel_l2(pred[k][i], ref[k][i]) if np.linalg.norm(ref[k][i]) > 0 else None
                for i in range(ref[k].shape[0])]
            for k in ("u", "v", "p")}


def stencil_continuity(u, v, x, y):
    dx, dy = x[1] - x[0], y[1] - y[0]
    div = ((u[:, 1:-1, 2:] - u[:, 1:-1, :-2]) / (2 * dx)
           + (v[:, 2:, 1:-1] - v[:, :-2, 1:-1]) / (2 * dy))
    c = CORE - 1
    core = div[:, c:-c, c:-c]
    return {
        "rms_all": float(np.sqrt(np.mean(div ** 2))),
        "rms_core": float(np.sqrt(np.mean(core ** 2))),
        "rms_per_snapshot": np.sqrt(np.mean(div ** 2, axis=(1, 2))).tolist(),
        "rms_core_per_snapshot": np.sqrt(np.mean(core ** 2, axis=(1, 2))).tolist(),
    }


def constraint_errors(u, v, t):
    # Wall nodes of the 64 x 64 grid over all snapshots, lid corners excluded
    lid_u = C.LID_VELOCITY * np.tanh(C.LID_RAMP_RATE * t)[:, None]
    lid_err_u = u[:, -1, 1:-1] - lid_u
    lid_err_v = v[:, -1, 1:-1]
    walls_u = np.concatenate([u[:, 0, :], u[:, :-1, 0], u[:, :-1, -1]], axis=1)
    walls_v = np.concatenate([v[:, 0, :], v[:, :-1, 0], v[:, :-1, -1]], axis=1)
    bc = np.concatenate([lid_err_u.ravel(), lid_err_v.ravel(), walls_u.ravel(), walls_v.ravel()])
    ic = np.concatenate([u[0, 1:-1, 1:-1].ravel(), v[0, 1:-1, 1:-1].ravel()])
    return {"boundary_rms": float(np.sqrt(np.mean(bc ** 2))),
            "initial_rms": float(np.sqrt(np.mean(ic ** 2)))}


# ---------- network physics by automatic differentiation ----------

def load_network_fields(run_dir):
    """Callable (x, y, t) -> physical (u, v, p) for a saved network run."""
    import tensorflow as tf
    from models.common.residuals import scale_inputs_tf
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    net = tf.keras.models.load_model(os.path.join(run_dir, "model.keras"), compile=False)
    if cfg["method"] == "data_driven_nn":
        mean = tf.constant(np.array(cfg["data"]["target_mean"], dtype="float32"))
        std = tf.constant(np.array(cfg["data"]["target_std"], dtype="float32"))
    else:
        mean = tf.zeros(3)
        std = tf.ones(3)

    def fields(x, y, t):
        out = net(scale_inputs_tf(tf.concat([x, y, t], axis=1))) * std + mean
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]
    return fields


def autodiff_physics(run_dir, nu, batch=10000):
    import tensorflow as tf
    from models.common.residuals import navier_stokes_residuals
    fields = load_network_fields(run_dir)
    rng = np.random.default_rng(PHYSICS_SEED)
    pts = rng.uniform([C.X_MIN, C.Y_MIN, C.T_MIN], [C.X_MAX, C.Y_MAX, C.T_MAX],
                      size=(PHYSICS_POINTS, 3)).astype("float32")
    sq = np.zeros(3)
    for i in range(0, PHYSICS_POINTS, batch):
        x, y, t = (tf.constant(pts[i:i + batch, k:k + 1]) for k in range(3))
        r = navier_stokes_residuals(fields, x, y, t, nu)
        sq += [float(tf.reduce_sum(tf.square(ri))) for ri in r]
    rms = np.sqrt(sq / PHYSICS_POINTS)
    return {"continuity_residual": float(rms[0]), "x_momentum_residual": float(rms[1]),
            "y_momentum_residual": float(rms[2]), "points": PHYSICS_POINTS, "seed": PHYSICS_SEED}


# ---------- driver ----------

def find_runs(exp_dir):
    runs = []
    for method in ("finite_difference", "data_driven_nn", "pinn", "hybrid"):
        for d in sorted(glob.glob(os.path.join(exp_dir, method, "*"))):
            if os.path.isfile(os.path.join(d, "config.json")):
                runs.append(d)
    return runs


def evaluate_run(run_dir, ref, data, nu, with_physics=True):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    method = cfg.get("method") or cfg.get("numerical_method")
    field_file = "solution.npz" if method == "finite_difference" else "predictions.npz"
    path = os.path.join(run_dir, field_file)
    perf_path = os.path.join(run_dir, "performance.json")
    perf = json.load(open(perf_path)) if os.path.exists(perf_path) else {}
    result = {"run_id": cfg["run_id"], "method": method, "config": cfg}

    if cfg.get("training", {}).get("timing_run"):
        result["status"] = "timing run (cost only)"
        result["performance"] = perf
        return result
    if not os.path.exists(path):
        result["status"] = "missing field file"
        return result
    f = np.load(path)
    if f["u"].shape != ref["u"].shape:
        result["status"] = f"not on the reference grid {f['u'].shape}"
        result["performance"] = perf
        return result

    pred = prepare(f)
    result["status"] = "evaluated"
    result["accuracy"] = {
        "test_a": accuracy(pred, ref, data["idx_test_a"]),
        "test_b": accuracy(pred, ref, data["idx_test_b"]),
        "full_field": accuracy(pred, ref),
    }
    result["per_snapshot_relative_l2"] = {"t": data["t"].tolist(), **per_snapshot_rel_l2(pred, ref)}
    result["stencil_continuity"] = stencil_continuity(f["u"], f["v"], f["x"], f["y"])
    result["constraints"] = constraint_errors(f["u"], f["v"], f["t"])
    if method != "finite_difference" and with_physics:
        result["physical_consistency"] = autodiff_physics(run_dir, nu)
    result["performance"] = perf
    return result


def flat_row(r, re):
    cfg = r["config"]
    perf = r.get("performance", {})
    timing = perf.get("timing", {})
    mem = perf.get("memory", {})
    sto = perf.get("storage", {})
    inf = perf.get("inference", {})
    row = {
        "experiment_id": os.path.basename(C.experiment_dir(re)), "problem": "lid_driven_cavity",
        "reynolds_number": re, "grid_nx": C.GRID_N, "grid_ny": C.GRID_N, "time_dependent": True,
        "method": r["method"], "run_id": r["run_id"], "status": r["status"],
        "device": cfg.get("device"), "seed": cfg.get("seed"),
        "deterministic_ops": cfg.get("deterministic_ops"),
        "timing_run": bool(cfg.get("training", {}).get("timing_run")),
    }
    for split in ("test_a", "test_b", "full_field"):
        acc = r.get("accuracy", {}).get(split, {})
        for k in ("u", "v", "p"):
            row[f"{split}_mse_{k}"] = acc.get(f"mse_{k}")
            row[f"{split}_relative_l2_{k}"] = acc.get(f"relative_l2_{k}")
    # week-03 column names refer to the full field
    for k in ("u", "v", "p"):
        row[f"mse_{k}"] = row[f"full_field_mse_{k}"]
        row[f"relative_l2_{k}"] = row[f"full_field_relative_l2_{k}"]
    phys = r.get("physical_consistency", {})
    row["continuity_residual"] = phys.get("continuity_residual")
    row["x_momentum_residual"] = phys.get("x_momentum_residual")
    row["y_momentum_residual"] = phys.get("y_momentum_residual")
    sc = r.get("stencil_continuity", {})
    row["stencil_continuity_rms"] = sc.get("rms_all")
    row["stencil_continuity_rms_core"] = sc.get("rms_core")
    con = r.get("constraints", {})
    row["boundary_rms"] = con.get("boundary_rms")
    row["initial_rms"] = con.get("initial_rms")
    row.update({
        "total_runtime_seconds": timing.get("total_runtime_seconds"),
        "training_time_seconds": timing.get("training_time_seconds"),
        "solve_time_seconds": timing.get("solve_time_seconds"),
        "time_per_step_seconds": timing.get("time_per_step_seconds"),
        "first_step_seconds": timing.get("first_step_seconds"),
        "steps": timing.get("steps"),
        "throughput_samples_per_second": timing.get("throughput_samples_per_second"),
        "inference_time_seconds": timing.get("inference_time_seconds"),
        "inference_single_point_seconds": inf.get("single_point_seconds"),
        "peak_ram_bytes": mem.get("peak_ram_bytes"),
        "peak_vram_bytes": mem.get("peak_vram_bytes"),
        "free_ram_at_start_bytes": mem.get("free_ram_at_start_bytes"),
        "gpu_utilization_mean_percent": (perf.get("gpu_monitor") or {}).get("gpu_utilization_mean_percent"),
        "model_size_bytes": sto.get("model_size_bytes"),
        "output_size_bytes": sto.get("output_size_bytes"),
        "total_run_size_bytes": sto.get("total_run_size_bytes"),
    })
    return row


def load_reference(re):
    from models.common import dataset
    data = dataset.load(re)
    ref_dir = os.path.join(C.experiment_dir(re), "finite_difference", C.fd_reference_run_id(re))
    ref_raw = np.load(os.path.join(ref_dir, "solution.npz"))
    return prepare(ref_raw), data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=float, default=100.0)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="device for the autodiff residuals")
    parser.add_argument("--exp-dir", default=None,
                        help="folder of runs to score (default: the experiment folder for --re)")
    parser.add_argument("--out", default=None,
                        help="default: experiments/results/experiment_results.csv (Re 100), "
                             "experiment_results_re<Re>.csv otherwise")
    args = parser.parse_args()
    if args.out is None:
        args.out = ("experiments/results/experiment_results.csv" if args.re == 100.0
                    else f"experiments/results/experiment_results_re{args.re:g}.csv")

    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    ref, data = load_reference(args.re)
    nu = C.kinematic_viscosity(args.re)
    rows = []
    for run_dir in find_runs(args.exp_dir or C.experiment_dir(args.re)):
        run_id = os.path.basename(run_dir)
        if args.only and run_id not in args.only:
            continue
        r = evaluate_run(run_dir, ref, data, nu)
        metrics = {k: v for k, v in r.items() if k not in ("config", "performance")}
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        rows.append(flat_row(r, args.re))
        acc = r.get("accuracy", {}).get("test_a", {})
        print(f"{run_id:36s} {r['status']:24s} " + (
            f"testA relL2 u {acc['relative_l2_u']:.4f} v {acc['relative_l2_v']:.4f} p {acc['relative_l2_p']:.4f}"
            if acc else ""), flush=True)

    if not args.only:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
