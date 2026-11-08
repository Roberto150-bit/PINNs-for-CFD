"""Training and error analysis of the two networks.

Figures go to experiments/analysis/plots/analysis_<name>.png (300 dpi, drawn
at the printed width of 7.5 in, text >= 24 pt), each with a sidecar
analysis_<name>.values.json holding the numbers that were plotted and the run
ids they come from. Nothing is fitted or smoothed: every plotted value is a
logged value, or a mean / min / max over seeds of logged values.

    python experiments/analysis/training_analysis.py               # every figure
    python experiments/analysis/training_analysis.py --only reynolds_trend

Figures
    loss_components     PINN loss terms vs step (Re 100 and Re 400)
    generalization      data NN train vs validation loss vs step
    repeatability       test-A error vs step, every seed
    error_vs_t          per-snapshot error vs time t, at each Reynolds number
    error_map_re<Re>    where in space the error is (mean |u error| over t and seeds)
    s1_data, s2_collocation, s3_size   sensitivity studies (Re 100)
    reynolds_trend      test-A error and PDE residual vs Reynolds number
Also writes experiments/results/reynolds_trend.csv.
"""
import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

from models.common import config as C  # noqa: E402

OUT = "experiments/analysis/plots"
RESULTS = "experiments/results"
DPI = 300
WIDTH = 7.5            # inches: one poster column

# Same method colors as the other figures
COLORS = {"finite_difference": "#2a78d6", "data_driven_nn": "#eb6834", "pinn": "#1baf7a"}
LABELS = {"finite_difference": "FD", "data_driven_nn": "Data NN", "pinn": "PINN"}
SHORT = {"data_driven_nn": "dnn", "pinn": "pinn"}
INK, INK2, GRIDLINE = "#0b0b0b", "#52514e", "#d9d8d4"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#f7f9fc", "#9ec5f4", "#2a78d6", "#104281"])
# PINN loss terms: categorical slots 4-8 plus a distinct line style each, so the
# terms stay distinguishable without color
TERMS = [("loss_ic", "initial", "#eda100", "-"),
         ("loss_bc", "boundary", "#e87ba4", "--"),
         ("loss_continuity", "continuity", "#008300", ":"),
         ("loss_x_momentum", "x-mom.", "#6250d6", "-."),
         ("loss_y_momentum", "y-mom.", "#e34948", (0, (5, 1, 1, 1, 1, 1)))]
TREND_RE = (10.0, 100.0, 400.0, 1000.0)
TREND_SEEDS = (1, 2, 3)          # seeds run at every Reynolds number

plt.rcParams.update({
    "font.size": 24, "axes.titlesize": 24, "axes.labelsize": 24, "xtick.labelsize": 24,
    "ytick.labelsize": 24, "legend.fontsize": 24, "font.family": "DejaVu Sans",
    "legend.handlelength": 1.4, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.25,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRIDLINE, "grid.linewidth": 0.8, "lines.linewidth": 2.5,
    "savefig.dpi": DPI, "figure.dpi": 100,
})


# ---------------- helpers ----------------

def plain_log_axis(axis, subs=(1.0, 2.0, 5.0)):
    """Log axis labelled in plain decimals (0.02, 0.1, ...)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def run_dir(re, method, seed):
    return os.path.join(C.experiment_dir(re), method, f"{SHORT[method]}_re{re:g}_gpu_s{seed}")


def existing_runs(re, method, seeds):
    """Main GPU runs that finished (performance.json written)."""
    out = []
    for s in seeds:
        d = run_dir(re, method, s)
        if os.path.isfile(os.path.join(d, "performance.json")):
            out.append(d)
    return out


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def results_csv(re):
    name = "experiment_results.csv" if re == 100.0 else f"experiment_results_re{re:g}.csv"
    path = os.path.join(RESULTS, name)
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def band(curves):
    a = np.vstack(curves)
    return a.mean(axis=0), a.min(axis=0), a.max(axis=0)


def save(fig, name, values):
    os.makedirs(OUT, exist_ok=True)
    png = os.path.join(OUT, f"analysis_{name}.png")
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    with open(os.path.join(OUT, f"analysis_{name}.values.json"), "w") as f:
        json.dump(values, f, indent=1)
    print(f"wrote {png}")


def tolist(a):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, dtype=float)]


def rid(d):
    return os.path.basename(d.rstrip("/"))


# ---------------- figures ----------------

def fig_loss_components(res=(100.0, 400.0), seeds=(1, 2, 3, 4, 5), every=100):
    """PINN loss terms vs step: mean over seeds, min-max band (raw values at every
    `every`-th step; the history is logged every 10 steps)."""
    fig, axes = plt.subplots(len(res), 1, figsize=(WIDTH, 4.2 * len(res)), sharex=True, sharey=True,
                             layout="constrained")
    values = {"note": f"raw logged values at every {every}th step; mean and min-max over seeds"}
    for ax, re in zip(np.atleast_1d(axes), res):
        dirs = existing_runs(re, "pinn", seeds)
        hist = [read_csv(os.path.join(d, "history.csv")) for d in dirs]
        step = hist[0]["step"]
        keep = (step % every == 0) | (step == 1)
        values[f"re{re:g}"] = {"runs": [rid(d) for d in dirs], "step": tolist(step[keep])}
        for key, label, color, ls in TERMS:
            mean, lo, hi = band([h[key][keep] for h in hist])
            ax.fill_between(step[keep], lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(step[keep], mean, color=color, ls=ls, label=label)
            values[f"re{re:g}"][key] = {"mean": tolist(mean), "min": tolist(lo), "max": tolist(hi)}
        ax.set_yscale("log")
        plain_log_axis(ax.yaxis, subs=(1.0,))
        ax.set_title(f"PINN, Re = {re:g}", loc="left")
        ax.set_ylabel("Loss term")
    np.atleast_1d(axes)[-1].set_xlabel("Training step")
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=22, frameon=False)
    save(fig, "loss_components", values)


def fig_generalization(res=(100.0, 400.0), seeds=(1, 2, 3, 4, 5)):
    """Data NN: training loss (mean of the logged mini-batch losses in the 500
    steps before each evaluation) vs validation loss (full validation set)."""
    fig, axes = plt.subplots(len(res), 1, figsize=(WIDTH, 4.2 * len(res)), sharex=True, sharey=True,
                             layout="constrained")
    values = {"note": "train = mean of logged mini-batch losses over the 500 steps up to each "
                      "evaluation step; validation = full validation split; both standardized MSE"}
    for ax, re in zip(np.atleast_1d(axes), res):
        dirs = existing_runs(re, "data_driven_nn", seeds)
        trains, vals, gaps = [], [], []
        for d in dirs:
            h = read_csv(os.path.join(d, "history.csv"))
            e = read_csv(os.path.join(d, "eval_history.csv"))
            steps = e["step"][e["step"] >= C.EVAL_EVERY]
            tr = np.array([h["loss"][(h["step"] > s - C.EVAL_EVERY) & (h["step"] <= s)].mean() for s in steps])
            va = e["val_loss"][e["step"] >= C.EVAL_EVERY]
            trains.append(tr), vals.append(va), gaps.append(va / tr)
        tm, tlo, thi = band(trains)
        vm, vlo, vhi = band(vals)
        c = COLORS["data_driven_nn"]
        ax.fill_between(steps, tlo, thi, color=c, alpha=0.18, lw=0)
        ax.plot(steps, tm, color=c, ls="-", label="training")
        ax.fill_between(steps, vlo, vhi, color=INK2, alpha=0.15, lw=0)
        ax.plot(steps, vm, color=INK2, ls="--", label="validation")
        ax.set_yscale("log")
        plain_log_axis(ax.yaxis, subs=(1.0, 2.0, 5.0))
        ax.set_title(f"Data NN, Re = {re:g}", loc="left")
        ax.set_ylabel("Loss (MSE)")
        gm = np.vstack(gaps).mean(axis=0)
        values[f"re{re:g}"] = {"runs": [rid(d) for d in dirs], "step": tolist(steps),
                               "train": {"mean": tolist(tm), "min": tolist(tlo), "max": tolist(thi)},
                               "validation": {"mean": tolist(vm), "min": tolist(vlo), "max": tolist(vhi)},
                               "val_over_train_mean": tolist(gm),
                               "final_val_over_train_per_seed": [float(g[-1]) for g in gaps]}
    np.atleast_1d(axes)[-1].set_xlabel("Training step")
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    save(fig, "generalization", values)


def fig_repeatability(res=(100.0, 400.0), seeds=(1, 2, 3, 4, 5)):
    """Test-A relative L2 of u vs step for every seed (one thin line per run)."""
    fig, axes = plt.subplots(len(res), 1, figsize=(WIDTH, 4.2 * len(res)), sharex=True, sharey=True,
                             layout="constrained")
    values = {}
    for ax, re in zip(np.atleast_1d(axes), res):
        values[f"re{re:g}"] = {}
        for method in ("data_driven_nn", "pinn"):
            for k, d in enumerate(existing_runs(re, method, seeds)):
                e = read_csv(os.path.join(d, "eval_history.csv"))
                ax.plot(e["step"], e["rel_l2_u"], color=COLORS[method], lw=1.6, alpha=0.85,
                        label=LABELS[method] if k == 0 else None)
                values[f"re{re:g}"][rid(d)] = {"step": tolist(e["step"]), "rel_l2_u": tolist(e["rel_l2_u"])}
        ax.set_yscale("log")
        plain_log_axis(ax.yaxis, subs=(1.0, 3.0))
        ax.set_title(f"Re = {re:g}, one line per seed", loc="left")
        ax.set_ylabel("Rel. L2, u")
    np.atleast_1d(axes)[-1].set_xlabel("Training step")
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    save(fig, "repeatability", values)


def fig_error_vs_t(seeds=TREND_SEEDS):
    """Per-snapshot relative L2 of u vs time t, one panel per Reynolds number
    (mean over seeds, min-max band). t = 0 is left out: the FD field is zero."""
    res = [re for re in TREND_RE if existing_runs(re, "pinn", seeds) or existing_runs(re, "data_driven_nn", seeds)]
    ncol = 2 if len(res) > 1 else 1
    nrow = int(np.ceil(len(res) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(WIDTH, 3.4 * nrow + 1.0), sharex=True, sharey=True,
                             layout="constrained", squeeze=False)
    values = {}
    for ax, re in zip(axes.flat, res):
        values[f"re{re:g}"] = {}
        for method in ("data_driven_nn", "pinn"):
            dirs = [d for d in existing_runs(re, method, seeds) if os.path.isfile(os.path.join(d, "metrics.json"))]
            if not dirs:
                continue
            curves = []
            for d in dirs:
                m = json.load(open(os.path.join(d, "metrics.json")))["per_snapshot_relative_l2"]
                t = np.array(m["t"])
                curves.append(np.array(m["u"], dtype=float))
            keep = t > 0
            mean, lo, hi = band([c[keep] for c in curves])
            ax.fill_between(t[keep], lo, hi, color=COLORS[method], alpha=0.2, lw=0)
            ax.plot(t[keep], mean, color=COLORS[method], label=LABELS[method])
            values[f"re{re:g}"][method] = {"runs": [rid(d) for d in dirs], "t": tolist(t[keep]),
                                           "mean": tolist(mean), "min": tolist(lo), "max": tolist(hi)}
        ax.set_yscale("log")
        plain_log_axis(ax.yaxis, subs=(1.0,))
        ax.set_title(f"Re = {re:g}", loc="left")
    for ax in axes.flat[len(res):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time t")
    for ax in axes[:, 0]:
        ax.set_ylabel("Rel. L2, u")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    save(fig, "error_vs_t", values)


def fig_error_map(re, seeds=TREND_SEEDS):
    """Mean |u_network - u_FD| over all snapshots and seeds, on the 64 x 64 grid."""
    ref_path = os.path.join(C.experiment_dir(re), "finite_difference", C.fd_reference_run_id(re), "solution.npz")
    if not (os.path.isfile(ref_path) and all(existing_runs(re, m, seeds) for m in SHORT)):
        print(f"error_map_re{re:g}: skipped (reference or runs not available yet)")
        return
    ref = np.load(ref_path)
    maps, values = {}, {"re": re}
    for method in ("data_driven_nn", "pinn"):
        dirs = existing_runs(re, method, seeds)
        if not dirs:
            return
        errs = [np.abs(np.load(os.path.join(d, "predictions.npz"))["u"] - ref["u"]).mean(axis=0) for d in dirs]
        maps[method] = np.mean(errs, axis=0)
        m = maps[method]
        j, i = np.unravel_index(np.argmax(m), m.shape)
        values[method] = {"runs": [rid(d) for d in dirs], "mean_abs_error": float(m.mean()),
                          "max_abs_error": float(m.max()),
                          "argmax_xy": [float(ref["x"][i]), float(ref["y"][j])],
                          "top_rows_mean": float(m[-8:].mean()), "rest_mean": float(m[:-8].mean())}
    vmax = max(float(np.percentile(m, 99)) for m in maps.values())
    values["color_max_99th_percentile"] = vmax
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.6), layout="constrained", sharey=True)
    for ax, method in zip(axes, ("data_driven_nn", "pinn")):
        im = ax.pcolormesh(ref["x"], ref["y"], maps[method], cmap=SEQUENTIAL, vmin=0, vmax=vmax,
                           shading="nearest", rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(LABELS[method])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(False)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    cb = fig.colorbar(im, ax=axes, location="bottom", shrink=0.8, extend="max", aspect=30)
    cb.set_label(f"Mean |u error|, Re = {re:g}")
    cb.ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    save(fig, f"error_map_re{re:g}", values)


def sensitivity_rows():
    path = os.path.join(RESULTS, "sensitivity_re100.csv")
    rows = list(csv.DictReader(open(path, newline="")))
    base = [r for r in results_csv(100.0) if r["run_id"] in
            {f"{s}_re100_gpu_s{k}" for s in ("dnn", "pinn") for k in (1, 2, 3)}]
    return rows, base


def seed_plot(ax, xs, ys_per_x, color, label, marker="o"):
    """Every seed as a dot, the mean as a line."""
    means = []
    for x, ys in zip(xs, ys_per_x):
        ax.scatter([x] * len(ys), ys, s=70, color=color, alpha=0.6, edgecolor="white", linewidth=1, zorder=3)
        means.append(float(np.mean(ys)))
    ax.plot(xs, means, color=color, marker=marker, ms=10, label=label, zorder=4)
    return means


def fig_sensitivity():
    rows, base = sensitivity_rows()
    val = lambda rs: [float(r["test_a_relative_l2_u"]) for r in rs]  # noqa: E731
    pick = lambda pat: [r for r in rows if r["run_id"].endswith(pat)]  # noqa: E731
    base_d = [r for r in base if r["run_id"].startswith("dnn")]
    base_p = [r for r in base if r["run_id"].startswith("pinn")]

    # S1: data NN vs train fraction
    fig, ax = plt.subplots(figsize=(WIDTH, 4.6), layout="constrained")
    xs = [1, 10, 100]
    groups = [pick("_frac1"), pick("_frac10"), base_d]
    m = seed_plot(ax, xs, [val(g) for g in groups], COLORS["data_driven_nn"], "Data NN")
    pm = float(np.mean(val(base_p)))
    ax.axhline(pm, color=COLORS["pinn"], ls="--", label="PINN, no data")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xticks(xs, ["1%", "10%", "100%"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    plain_log_axis(ax.yaxis, subs=(1.0, 2.0, 5.0))
    ax.set_xlabel("Training data used")
    ax.set_ylabel("Rel. L2, u")
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    save(fig, "s1_data", {"x_percent": xs, "runs": [[r["run_id"] for r in g] for g in groups],
                          "rel_l2_u": [val(g) for g in groups], "mean": m,
                          "pinn_base_mean": pm, "pinn_base_runs": [r["run_id"] for r in base_p]})

    # S2: PINN vs collocation points
    fig, ax = plt.subplots(figsize=(WIDTH, 4.6), layout="constrained")
    xs = [5000, 10000, 20000]
    groups = [pick("_col5000"), pick("_col10000"), base_p]
    m = seed_plot(ax, xs, [val(g) for g in groups], COLORS["pinn"], "Test-A error, u")
    ax2_vals = [[float(r["continuity_residual"]) for r in g] for g in groups]
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xticks(xs, ["5k", "10k", "20k"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    plain_log_axis(ax.yaxis, subs=(1.0, 2.0, 5.0))
    ax.set_xlabel("PINN collocation points")
    ax.set_ylabel("Rel. L2, u")
    save(fig, "s2_collocation", {"x": xs, "runs": [[r["run_id"] for r in g] for g in groups],
                                 "rel_l2_u": [val(g) for g in groups], "mean": m,
                                 "continuity_residual (not plotted)": ax2_vals})

    # S3: both methods vs parameter count
    fig, ax = plt.subplots(figsize=(WIDTH, 4.6), layout="constrained")
    vals = {}
    for method, short, bg in (("data_driven_nn", "dnn", base_d), ("pinn", "pinn", base_p)):
        groups = [[r for r in pick("_arch4x50") if r["run_id"].startswith(short)], bg,
                  [r for r in pick("_arch6x100") if r["run_id"].startswith(short)]]
        xs = [8003, 17091, 51203]
        m = seed_plot(ax, xs, [val(g) for g in groups], COLORS[method], LABELS[method])
        vals[method] = {"runs": [[r["run_id"] for r in g] for g in groups], "rel_l2_u": [val(g) for g in groups],
                        "mean": m}
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xticks(xs, ["4×50\n8k", "5×64\n17k", "6×100\n51k"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    plain_log_axis(ax.yaxis, subs=(1.0, 2.0, 5.0))
    ax.set_xlabel("Network size (parameters)")
    ax.set_ylabel("Rel. L2, u")
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    save(fig, "s3_size", {"x_parameters": xs, **vals})


def reynolds_rows(seeds=TREND_SEEDS):
    keep = {f"{SHORT[m]}_re{re:g}_gpu_s{s}" for re in TREND_RE for m in SHORT for s in seeds}
    rows = []
    for re in TREND_RE:
        for r in results_csv(re):
            if r["run_id"] in keep and r["status"] == "evaluated":
                rows.append(r)
    return rows


TREND_COLS = ["reynolds_number", "method", "run_id", "seed", "test_a_relative_l2_u", "test_a_relative_l2_v",
              "test_a_relative_l2_p", "test_b_relative_l2_u", "continuity_residual", "x_momentum_residual",
              "y_momentum_residual", "boundary_rms", "initial_rms"]


def fig_reynolds_trend(seeds=TREND_SEEDS):
    rows = reynolds_rows(seeds)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "reynolds_trend.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TREND_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {RESULTS}/reynolds_trend.csv ({len(rows)} runs)")

    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 8.4), sharex=True, layout="constrained")
    values = {"seeds": list(seeds)}
    for ax, col, ylabel in ((axes[0], "test_a_relative_l2_u", "Rel. L2, u"),
                            (axes[1], "continuity_residual", "Continuity\nresidual")):
        values[col] = {}
        for method in ("data_driven_nn", "pinn"):
            res = [re for re in TREND_RE if any(float(r["reynolds_number"]) == re and r["method"] == method
                                                for r in rows)]
            ys = [[float(r[col]) for r in rows if float(r["reynolds_number"]) == re and r["method"] == method]
                  for re in res]
            if not res:
                continue
            m = seed_plot(ax, res, ys, COLORS[method], LABELS[method] if col == "test_a_relative_l2_u" else None)
            values[col][method] = {"re": res, "per_seed": ys, "mean": m,
                                   "runs": [[r["run_id"] for r in rows if float(r["reynolds_number"]) == re
                                             and r["method"] == method] for re in res]}
        ax.set_yscale("log")
        plain_log_axis(ax.yaxis, subs=(1.0,))
        ax.set_ylabel(ylabel)
    axes[1].set_xscale("log")
    axes[1].set_xticks(list(TREND_RE), [f"{re:g}" for re in TREND_RE])
    axes[1].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[1].set_xlabel("Reynolds number")
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    save(fig, "reynolds_trend", values)


FIGURES = {
    "loss_components": fig_loss_components,
    "generalization": fig_generalization,
    "repeatability": fig_repeatability,
    "error_vs_t": fig_error_vs_t,
    "error_map": lambda: [fig_error_map(re) for re in TREND_RE],
    "sensitivity": fig_sensitivity,
    "reynolds_trend": fig_reynolds_trend,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=list(FIGURES), default=None)
    args = parser.parse_args()
    for name in args.only or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
