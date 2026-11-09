"""Aggregate run results into summary tables and the poster's numbers.

Inputs (all generated):
  experiments/results/experiment_results.csv   (evaluate_all.py)
  each run's metrics.json / performance.json / eval_history.csv
  experiments/results/fd_validation_re100.json, fd_transient_convergence_re100.json
  experiments/experiment_001_re100/determinism_cost/

Outputs:
  experiments/results/summary_by_group.csv   mean ± std per (method, device)
  experiments/results/aggregate.json         every derived result with its run ids
  experiments/results/poster_numbers.json    the numbers quoted in the poster text

Derived cost results
  total cost to a usable 64 x 64 x 51 field:
      FD = solve time;  data NN = FD solve + training;  PINN = training
  break-even field evaluations n*: smallest n for which
      cost_to_train + n · inference  <  n · FD solve,
      i.e. n* = cost_to_train / (FD solve − inference), when inference < FD solve
  FD scaling exponent: least-squares slope of log(solve time) vs log(N²)

    python experiments/analysis/aggregate.py
"""
import csv
import glob
import json
import os
import statistics
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from models.common import config as C  # noqa: E402

RESULTS = "experiments/results"
EXP = C.EXPERIMENT_DIR
# Network methods and the short names used in poster_numbers.json keys
METHODS = (("data_driven_nn", "dnn"), ("pinn", "pinn"), ("hybrid", "hybrid"))
SHORT = dict(METHODS)

METRIC_COLS = (
    [f"{s}_relative_l2_{k}" for s in ("test_a", "test_b", "full_field") for k in "uvp"]
    + [f"{s}_mse_{k}" for s in ("test_a", "test_b", "full_field") for k in "uvp"]
    + ["continuity_residual", "x_momentum_residual", "y_momentum_residual",
       "stencil_continuity_rms", "stencil_continuity_rms_core", "boundary_rms", "initial_rms",
       "training_time_seconds", "time_per_step_seconds", "first_step_seconds",
       "throughput_samples_per_second", "inference_time_seconds", "inference_single_point_seconds",
       "peak_ram_bytes", "peak_vram_bytes", "model_size_bytes", "output_size_bytes",
       "total_run_size_bytes", "gpu_utilization_mean_percent", "solve_time_seconds"]
)


def read_rows():
    with open(os.path.join(RESULTS, "experiment_results.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("", "None"):
                r[k] = None
            elif k in METRIC_COLS:
                r[k] = float(v)
    return rows


def mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, 0
    return (statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0, len(vals))


def group_key(r):
    if r["method"] == "finite_difference":
        return None
    kind = "timing" if r["timing_run"] == "True" else "full"
    return (r["method"], r["device"], kind)


def summarize(rows):
    groups = {}
    for r in rows:
        k = group_key(r)
        if k is not None:
            groups.setdefault(k, []).append(r)
    out = {}
    for (method, device, kind), rs in sorted(groups.items()):
        entry = {"method": method, "device": device, "kind": kind,
                 "runs": sorted(r["run_id"] for r in rs), "n": len(rs)}
        for col in METRIC_COLS:
            m, s, n = mean_std([r.get(col) for r in rs])
            entry[col] = {"mean": m, "std": s, "n": n}
        out[f"{method}|{device}|{kind}"] = entry
    return out


def fd_timing():
    # Solve time per grid: median of the three timing repeats (history logging off)
    res = {}
    for n in (32, 64, 128):
        runs = sorted(glob.glob(os.path.join(EXP, "finite_difference", f"fd_re100_n{n}_transient_timing_r*")))
        times, ram = [], []
        for d in runs:
            p = json.load(open(os.path.join(d, "performance.json")))
            times.append(p["timing"]["solve_time_seconds"])
            ram.append(p["memory"]["peak_ram_bytes"])
        if times:
            res[n] = {"runs": [os.path.basename(d) for d in runs], "solve_seconds": times,
                      "median_solve_seconds": statistics.median(times),
                      "median_peak_ram_bytes": statistics.median(ram),
                      "output_size_bytes": os.path.getsize(os.path.join(runs[0], "solution.npz"))}
    if len(res) >= 2:
        ns = np.array(sorted(res))
        t = np.array([res[n]["median_solve_seconds"] for n in ns])
        slope = np.polyfit(np.log(ns.astype(float) ** 2), np.log(t), 1)[0]
        res["scaling_exponent_vs_unknowns"] = float(slope)
        res["scaling_note"] = "slope of log(solve time) vs log(N^2) for N = grid points per side, fixed 5,000 steps"
    return res


def determinism_cost():
    out = {}
    for method, rid in (("data_driven_nn", "dnn"), ("pinn", "pinn")):
        per = {}
        for tag in ("det", "nondet"):
            p = os.path.join(EXP, "determinism_cost", method, f"{rid}_re100_gpu_s1_{tag}2000", "performance.json")
            if os.path.exists(p):
                per[tag] = json.load(open(p))["timing"]["time_per_step_seconds"]
        if len(per) == 2:
            out[method] = {"time_per_step_deterministic": per["det"],
                           "time_per_step_nondeterministic": per["nondet"],
                           "slowdown_ratio": per["det"] / per["nondet"],
                           "runs": [f"{rid}_re100_gpu_s1_det2000", f"{rid}_re100_gpu_s1_nondet2000"]}
    return out


def accuracy_vs_time(rows):
    # Test-A relative L2 vs training wall-clock, per run (from eval_history.csv)
    curves = {}
    for r in rows:
        if r["method"] == "finite_difference" or r["status"] != "evaluated":
            continue
        path = os.path.join(EXP, r["method"], r["run_id"], "eval_history.csv")
        with open(path) as f:
            h = list(csv.DictReader(f))
        curves[r["run_id"]] = {k: [float(x[k]) for x in h] for k in
                               ("step", "train_time_seconds", "rel_l2_u", "rel_l2_v", "rel_l2_p")}
    return curves


def cost_to_solution(summary, fd):
    fd64 = fd.get(64, {}).get("median_solve_seconds")
    out = {"fd_solve_seconds_n64": fd64, "fd_runs": fd.get(64, {}).get("runs")}
    for key, g in summary.items():
        method, device, kind = key.split("|")
        if kind != "full":
            continue
        train = g["training_time_seconds"]["mean"]
        infer = g["inference_time_seconds"]["mean"]
        if train is None:
            continue
        # Methods trained on FD data also pay for the simulation that made it
        total = train + (fd64 if method in ("data_driven_nn", "hybrid") else 0.0)
        entry = {"training_seconds_mean": train, "inference_seconds_mean": infer,
                 "total_cost_to_field_seconds": total, "runs": g["runs"]}
        if fd64 is not None and infer is not None and infer < fd64:
            entry["break_even_evaluations"] = total / (fd64 - infer)
        else:
            entry["break_even_evaluations"] = None
            entry["break_even_note"] = "inference is not faster than an FD solve: never repays"
        out[f"{method}|{device}"] = entry
    return out


def poster_numbers(summary, fd, det, cost, validation, transient):
    def g(method, device, col, stat="mean", kind="full"):
        e = summary.get(f"{method}|{device}|{kind}")
        return None if e is None else e[col][stat]

    def runs(method, device, kind="full"):
        e = summary.get(f"{method}|{device}|{kind}")
        return [] if e is None else e["runs"]

    nums = {}

    def put(key, value, run_ids, desc):
        nums[key] = {"value": value, "runs": run_ids, "description": desc}

    for m, short in METHODS:
        for split in ("test_a", "test_b"):
            for k in "uvp":
                col = f"{split}_relative_l2_{k}"
                put(f"{short}_gpu_{col}_mean", g(m, "gpu", col), runs(m, "gpu"),
                    f"{m} GPU runs: mean relative L2 of {k} on {split}")
                put(f"{short}_gpu_{col}_std", g(m, "gpu", col, "std"), runs(m, "gpu"),
                    f"{m} GPU runs: std over seeds")
        put(f"{short}_gpu_n_runs", len(runs(m, "gpu")) or None, runs(m, "gpu"), f"{m} GPU: number of seeds (Re 100)")
        for col in ("continuity_residual", "x_momentum_residual", "y_momentum_residual",
                    "stencil_continuity_rms_core", "boundary_rms", "initial_rms",
                    "training_time_seconds", "time_per_step_seconds", "inference_time_seconds",
                    "inference_single_point_seconds", "peak_vram_bytes", "peak_ram_bytes",
                    "model_size_bytes"):
            put(f"{short}_gpu_{col}_mean", g(m, "gpu", col), runs(m, "gpu"), f"{m} GPU: mean {col}")
            put(f"{short}_gpu_{col}_std", g(m, "gpu", col, "std"), runs(m, "gpu"), f"{m} GPU: std {col}")
    put("dnn_cpu_training_time_seconds_mean", g("data_driven_nn", "cpu", "training_time_seconds"),
        runs("data_driven_nn", "cpu"), "data NN CPU: mean training time")
    put("dnn_cpu_time_per_step_seconds_mean", g("data_driven_nn", "cpu", "time_per_step_seconds"),
        runs("data_driven_nn", "cpu"), "data NN CPU: mean time per step")
    put("dnn_cpu_inference_time_seconds_mean", g("data_driven_nn", "cpu", "inference_time_seconds"),
        runs("data_driven_nn", "cpu"), "data NN CPU: mean full-field inference time")
    put("dnn_cpu_peak_ram_bytes_mean", g("data_driven_nn", "cpu", "peak_ram_bytes"),
        runs("data_driven_nn", "cpu"), "data NN CPU: mean peak RSS")
    put("pinn_cpu_time_per_step_seconds_mean",
        g("pinn", "cpu", "time_per_step_seconds", kind="timing"), runs("pinn", "cpu", "timing"),
        "PINN CPU timing runs (2,000 steps): mean time per step")
    put("pinn_cpu_peak_ram_bytes_mean", g("pinn", "cpu", "peak_ram_bytes", kind="timing"),
        runs("pinn", "cpu", "timing"), "PINN CPU timing runs: mean peak RSS")
    fd_rows = {r["run_id"]: r for r in read_rows() if r["method"] == "finite_difference"}
    ref_id = C.fd_reference_run_id(100.0)
    if ref_id in fd_rows:
        put("fd_stencil_continuity_rms_core", fd_rows[ref_id]["stencil_continuity_rms_core"], [ref_id],
            "FD reference: core-region continuity RMS (central stencil)")
    for n in (32, 64, 128):
        if n in fd:
            put(f"fd_n{n}_solve_seconds_median", fd[n]["median_solve_seconds"], fd[n]["runs"],
                f"FD {n}x{n} transient (5,000 steps) solve time, median of repeats")
            put(f"fd_n{n}_peak_ram_bytes", fd[n]["median_peak_ram_bytes"], fd[n]["runs"],
                f"FD {n}x{n} transient: peak RSS, median of repeats")
    if "scaling_exponent_vs_unknowns" in fd:
        put("fd_scaling_exponent", fd["scaling_exponent_vs_unknowns"],
            [r for n in (32, 64, 128) if n in fd for r in fd[n]["runs"]], fd["scaling_note"])
    for key, e in cost.items():
        if "|" not in key:
            continue
        m, dev = key.split("|")
        short = SHORT[m]
        put(f"{short}_{dev}_total_cost_to_field_seconds", e["total_cost_to_field_seconds"], e["runs"],
            f"{m} {dev}: total cost to a usable field")
        put(f"{short}_{dev}_break_even_evaluations", e["break_even_evaluations"], e["runs"],
            f"{m} {dev}: field evaluations before inference repays training vs rerunning FD")
    for m, e in det.items():
        put(f"{'dnn' if m == 'data_driven_nn' else 'pinn'}_determinism_slowdown", e["slowdown_ratio"],
            e["runs"], f"{m}: time per step with / without enable_op_determinism (GPU, 2,000 steps)")
    if validation:
        for r in validation["runs"]:
            n = r["n"]
            put(f"fd_val_n{n}_u_max_dev", r["u_centerline_max_abs_dev"], [r["run_id"]],
                f"FD {n}: max |u - Marchi| on x = 0.5")
            put(f"fd_val_n{n}_v_max_dev", r["v_centerline_max_abs_dev"], [r["run_id"]],
                f"FD {n}: max |v - Marchi| on y = 0.5")
            put(f"fd_val_n{n}_vortex_x", r["vortex_x"], [r["run_id"]], f"FD {n}: vortex center x")
            put(f"fd_val_n{n}_vortex_y", r["vortex_y"], [r["run_id"]], f"FD {n}: vortex center y")
            put(f"fd_val_n{n}_vortex_dist_marchi", r["vortex_dist_marchi"], [r["run_id"]],
                f"FD {n}: vortex center distance to Marchi")
            put(f"fd_val_n{n}_vortex_dist_hou", r["vortex_dist_hou"], [r["run_id"]],
                f"FD {n}: vortex center distance to Hou")
            put(f"fd_val_n{n}_psi_min", r["psi_min"], [r["run_id"]], f"FD {n}: minimum stream function")
    if transient:
        tc = transient["time_convergence"]
        put("fd_dt_halving_max_diff_t5", tc["summary_velocity"]["at_t_end"], tc["runs"],
            "FD 64: max |velocity difference| between dt and dt/2 at t = 5")
    for key, e in re400_numbers().items():
        nums[key] = e
    for key, e in extended_numbers().items():
        nums[key] = e
    for key, e in spike_numbers().items():
        nums[key] = e
    for key, e in continuation_numbers().items():
        nums[key] = e
    for key, e in control_numbers().items():
        nums[key] = e
    return nums


def control_numbers():
    """Hybrid control (D23): hybrid code with the data loss only, physical outputs. Compared
    with the hybrid and the data NN runs of the same seeds at the same Re.
    Keys: ctrl_re<Re>_<dataonly|hybrid|dnn>_<u|cont>_mean, ctrl_re<Re>_n_runs."""
    out = {}
    for re in ("10", "100", "400", "1000"):
        path = os.path.join(RESULTS, f"control_re{re}.csv")
        main = os.path.join(RESULTS, "experiment_results.csv" if re == "100" else f"experiment_results_re{re}.csv")
        if not (os.path.exists(path) and os.path.exists(main)):
            continue
        with open(path) as f:
            ctrl = [r for r in csv.DictReader(f) if r["status"] == "evaluated" and r["test_a_relative_l2_u"]]
        with open(main) as f:
            base = {r["run_id"]: r for r in csv.DictReader(f)}
        groups = {"dataonly": [], "hybrid": [], "dnn": []}
        for r in ctrl:
            hyb_id = r["run_id"].replace("_dataonly", "")
            dnn_id = hyb_id.replace("hybrid_", "dnn_")
            if hyb_id in base and dnn_id in base:
                groups["dataonly"].append(r)
                groups["hybrid"].append(base[hyb_id])
                groups["dnn"].append(base[dnn_id])
        if not groups["dataonly"]:
            continue
        ids = sorted(x["run_id"] for g in groups.values() for x in g)
        out[f"ctrl_re{re}_n_runs"] = {"value": len(groups["dataonly"]), "runs": ids,
                                      "description": f"hybrid data-only control at Re {re}: runs with same-seed hybrid and data NN"}
        for name, rows in groups.items():
            for short, col in (("u", "test_a_relative_l2_u"), ("cont", "continuity_residual")):
                out[f"ctrl_re{re}_{name}_{short}_mean"] = {
                    "value": statistics.mean(float(x[col]) for x in rows), "runs": ids,
                    "description": f"Re {re}, seeds of the control runs: {name} mean {col}"}
    return out


def continuation_numbers():
    """S6 warm start: PINN at Re 400 initialized from the trained Re 100 PINN (same seed).
    Mean test-A relative L2 of the warm-started runs and of the standard runs with the same
    seeds. Keys: s6_re400_<warm|base>_<u|v|p>_mean, s6_re400_n_runs."""
    path = os.path.join(RESULTS, "continuation_re400.csv")
    base_path = os.path.join(RESULTS, "experiment_results_re400.csv")
    if not (os.path.exists(path) and os.path.exists(base_path)):
        return {}
    with open(path) as f:
        warm = [r for r in csv.DictReader(f) if r["status"] == "evaluated" and r["test_a_relative_l2_u"]]
    with open(base_path) as f:
        base = {r["run_id"]: r for r in csv.DictReader(f)}
    pairs = []
    for r in warm:
        base_id = r["run_id"].replace("_from_re100", "")
        if base_id in base and base[base_id]["test_a_relative_l2_u"]:
            pairs.append((r, base[base_id]))
    if not pairs:
        return {}
    ids = sorted(x["run_id"] for p in pairs for x in p)
    out = {"s6_re400_n_runs": {"value": len(pairs), "runs": ids,
                               "description": "S6: warm-started Re 400 PINN runs with a same-seed standard run"}}
    for k in "uvp":
        col = f"test_a_relative_l2_{k}"
        out[f"s6_re400_warm_{k}_mean"] = {
            "value": statistics.mean(float(w[col]) for w, _ in pairs), "runs": ids,
            "description": f"S6: Re 400 PINN started from the trained Re 100 PINN, mean {col}"}
        out[f"s6_re400_base_{k}_mean"] = {
            "value": statistics.mean(float(b[col]) for _, b in pairs), "runs": ids,
            "description": f"S6: standard Re 400 PINN runs with the same seeds, mean {col}"}
    return out


def re400_numbers():
    """GPU runs at every Reynolds number other than 100 (experiment_results_re<Re>.csv):
    mean, min and max of test-A relative L2 and continuity residual per method.
    Keys: re<Re>_<dnn|pinn>_gpu_<column>_<mean|min|max> (e.g. re400_..., re10_..., re1000_...)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, "experiment_results_re*.csv"))):
        re = os.path.basename(path)[len("experiment_results_re"):-4]
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if r["device"] == "gpu" and r["timing_run"] in ("False", "")]
        for m, short in METHODS:
            sel = [r for r in rows if r["method"] == m and r["run_id"].endswith(tuple(f"_s{s}" for s in range(1, 6)))]
            ids = sorted(r["run_id"] for r in sel)
            for col in ["test_a_relative_l2_" + k for k in "uvp"] + ["continuity_residual"]:
                vals = [float(r[col]) for r in sel if r[col] not in ("", "None")]
                if vals:
                    out[f"re{re}_{short}_gpu_{col}_mean"] = {
                        "value": statistics.mean(vals), "runs": ids,
                        "description": f"Re {re} {m} GPU runs: mean {col}"}
                    out[f"re{re}_{short}_gpu_{col}_min"] = {
                        "value": min(vals), "runs": ids, "description": f"Re {re} {m} GPU runs: min {col}"}
                    out[f"re{re}_{short}_gpu_{col}_max"] = {
                        "value": max(vals), "runs": ids, "description": f"Re {re} {m} GPU runs: max {col}"}
    return out


def spike_numbers():
    """Final-step spikes at Re 100 (main GPU runs): a run 'ends on a spike' when its final
    test-A error in u is more than 2x the best value it reached during training (the spike
    rule fixed before any hybrid result). Keys: spike_re100_<short>_final_n and, for the
    worst such run per method, spike_re100_<short>_worst_best / _worst_final / _worst_run."""
    out = {}
    with open(os.path.join(RESULTS, "experiment_results.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["device"] == "gpu" and r["timing_run"] == "False"
                and r["status"] == "evaluated" and r["method"] != "finite_difference"]
    for m, short in METHODS:
        runs = sorted(r["run_id"] for r in rows if r["method"] == m)
        if not runs:
            continue
        hits = []
        for rid in runs:
            with open(os.path.join(EXP, m, rid, "eval_history.csv")) as f:
                h = [float(x["rel_l2_u"]) for x in csv.DictReader(f)]
            best, final = min(h), h[-1]
            if final > 2 * best:
                hits.append((final / best, rid, best, final))
        out[f"spike_re100_{short}_final_n"] = {"value": len(hits), "runs": runs,
                                               "description": f"{m} Re 100: runs whose final u error > 2x their best"}
        if hits:
            _, rid, best, final = max(hits)
            out[f"spike_re100_{short}_worst_best"] = {"value": best, "runs": [rid],
                                                      "description": f"{rid}: best test-A u error during training"}
            out[f"spike_re100_{short}_worst_final"] = {"value": final, "runs": [rid],
                                                       "description": f"{rid}: test-A u error at the final step"}
    return out


def extended_numbers():
    """S5 training length: test-A relative L2 of u at 20,000 steps and at the last step
    of each completed 100,000-step run, averaged per (Re, method)."""
    out = {}
    for re, folder in ((100.0, "experiment_001_re100"), (400.0, "experiment_002_re400"),
                       (10.0, "experiment_003_re10"), (1000.0, "experiment_004_re1000")):
        for m, short in METHODS:
            at20, at40, atend, ids = [], [], [], []
            for run in sorted(glob.glob(f"experiments/{folder}/extended/{m}/*_steps100k")):
                if not os.path.exists(os.path.join(run, "performance.json")):
                    continue
                with open(os.path.join(run, "eval_history.csv")) as f:
                    hist = list(csv.DictReader(f))
                by_step = {int(h["step"]): float(h["rel_l2_u"]) for h in hist}
                if 20000 not in by_step:
                    continue
                at20.append(by_step[20000])
                atend.append(by_step[max(by_step)])
                at40.append(by_step.get(40000))
                ids.append(os.path.basename(run))
            if ids:
                tag = f"s5_re{re:g}_{short}"
                if all(v is not None for v in at40):
                    # equal-cost reference for the S6 warm start (20k + 20k = 40,000 steps in total)
                    out[f"{tag}_u_at_40k_mean"] = {"value": statistics.mean(at40), "runs": ids,
                                                   "description": f"S5 Re {re:g} {m}: test-A rel. L2 u at step 40,000"}
                out[f"{tag}_u_at_20k_mean"] = {"value": statistics.mean(at20), "runs": ids,
                                               "description": f"S5 Re {re:g} {m}: test-A rel. L2 u at step 20,000"}
                out[f"{tag}_u_at_100k_mean"] = {"value": statistics.mean(atend), "runs": ids,
                                                "description": f"S5 Re {re:g} {m}: test-A rel. L2 u at step 100,000"}
                out[f"{tag}_n_runs"] = {"value": len(ids), "runs": ids,
                                        "description": f"S5 Re {re:g} {m}: completed 100k-step runs"}
    return out


def main():
    rows = read_rows()
    summary = summarize(rows)
    fd = fd_timing()
    det = determinism_cost()
    cost = cost_to_solution(summary, fd)
    vpath = os.path.join(RESULTS, "fd_validation_re100.json")
    tpath = os.path.join(RESULTS, "fd_transient_convergence_re100.json")
    validation = json.load(open(vpath)) if os.path.exists(vpath) else None
    transient = json.load(open(tpath)) if os.path.exists(tpath) else None

    with open(os.path.join(RESULTS, "summary_by_group.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "device", "kind", "n", "runs"] + [f"{c}_{s}" for c in METRIC_COLS for s in ("mean", "std")])
        for e in summary.values():
            w.writerow([e["method"], e["device"], e["kind"], e["n"], " ".join(e["runs"])]
                       + [e[c][s] for c in METRIC_COLS for s in ("mean", "std")])

    agg = {"summary": summary, "fd_timing": {str(k): v for k, v in fd.items()},
           "determinism_cost": det, "cost_to_solution": cost,
           "accuracy_vs_time": accuracy_vs_time(rows)}
    with open(os.path.join(RESULTS, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=1)
    nums = poster_numbers(summary, fd, det, cost, validation, transient)
    with open(os.path.join(RESULTS, "poster_numbers.json"), "w") as f:
        json.dump(nums, f, indent=1)

    for key, e in summary.items():
        print(f"{key:32s} n={e['n']} testA relL2 u {e['test_a_relative_l2_u']['mean']} "
              f"train {e['training_time_seconds']['mean']}")
    print("FD timing:", {k: v["median_solve_seconds"] if isinstance(v, dict) else v for k, v in fd.items()})
    print("cost:", json.dumps(cost, indent=1))
    print("determinism:", det)


if __name__ == "__main__":
    main()
