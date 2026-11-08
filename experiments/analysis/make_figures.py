"""Regenerate every figure from the run records.

Figures are written at their printed poster size, 300 dpi, text >= 24 pt,
with one fixed color per method (colorblind-safe categorical slots 1-3):
    FD = blue, data-driven NN = orange, PINN = aqua.
Each figure's file name lists the runs it is drawn from, and a sidecar
<name>.values.json records the numbers actually plotted (read back from
the plot), which `--check` compares with experiment_results.csv /
aggregate.json.

    python experiments/analysis/make_figures.py              # all figures + check
    python experiments/analysis/make_figures.py --only accuracy
    python experiments/analysis/make_figures.py --selftest   # synthetic data, known answers
"""
import argparse
import csv
import glob
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

from models.common import config as C  # noqa: E402

OUT = "experiments/analysis/plots"
RESULTS = "experiments/results"
EXP = C.EXPERIMENT_DIR
DPI = 300
WIDTH = 7.5            # inches: one poster column

# One fixed color per method on every figure
COLORS = {"finite_difference": "#2a78d6", "data_driven_nn": "#eb6834", "pinn": "#1baf7a"}
LABELS = {"finite_difference": "FD", "data_driven_nn": "Data NN", "pinn": "PINN"}
GRID_SHADES = {32: "#86b6ef", 64: "#2a78d6", 128: "#104281"}   # one-hue ramp for grid levels
INK, INK2, GRIDLINE = "#0b0b0b", "#52514e", "#d9d8d4"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#f0efec", "#b42b2a"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#f7f9fc", "#9ec5f4", "#2a78d6", "#104281"])

plt.rcParams.update({
    # Figures are drawn at their printed size (7.5 in = one poster column), so
    # 24 pt here is 24 pt on the printed poster
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
    """Log axis labelled at 1, 2, 5 per decade in plain decimals (0.02, 0.1, ...),
    with unlabelled minor ticks, so labels stay short at 24 pt."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def run_span(run_ids):
    """Compact run-id label for file names: dnn_re100_gpu_s1-s5."""
    groups = {}
    for r in sorted(run_ids):
        stem, _, seed = r.rpartition("_s")
        if seed.isdigit():
            groups.setdefault(stem, []).append(int(seed))
        else:
            groups.setdefault(r, [])
    parts = []
    for stem, seeds in groups.items():
        parts.append(f"{stem}_s{min(seeds)}-s{max(seeds)}" if len(seeds) > 1 else
                     (f"{stem}_s{seeds[0]}" if seeds else stem))
    return "__".join(parts)


def save(fig, name, runs, plotted):
    os.makedirs(OUT, exist_ok=True)
    stem = f"{name}__{run_span(runs)}" if runs else name
    path = os.path.join(OUT, stem + ".png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    with open(os.path.join(OUT, stem + ".values.json"), "w") as f:
        json.dump({"figure": name, "runs": sorted(runs), "plotted": plotted}, f, indent=1)
    return path


def bar_heights(ax):
    return [float(p.get_height()) for p in ax.patches if isinstance(p, Rectangle) and p.get_width() > 0]


def load_csv():
    with open(os.path.join(RESULTS, "experiment_results.csv")) as f:
        return list(csv.DictReader(f))


def load_agg():
    return json.load(open(os.path.join(RESULTS, "aggregate.json")))


def group(agg, method, device="gpu", kind="full"):
    return agg["summary"].get(f"{method}|{device}|{kind}")


# ---------------- figures ----------------

def fig_accuracy(data):
    """Relative L2 of u, v, p per method: mean ± std over seeds, test A and test B.

    data = {method: {"test_a": {"u": (mean, std), ...}, "test_b": {...}, "runs": [...]}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.6), sharey=True, constrained_layout=True)
    methods = [m for m in ("data_driven_nn", "pinn") if m in data]
    width = 0.8 / len(methods)
    plotted = {}
    for ax, split, title in zip(axes, ("test_a", "test_b"), ("Test A", "Test B")):
        for i, m in enumerate(methods):
            means = [data[m][split][k][0] for k in "uvp"]
            stds = [data[m][split][k][1] for k in "uvp"]
            xs = np.arange(3) + (i - (len(methods) - 1) / 2) * width
            bars = ax.bar(xs, means, width * 0.92, yerr=stds, color=COLORS[m], label=LABELS[m],
                          capsize=6, error_kw={"elinewidth": 2, "ecolor": INK})
            plotted[f"{split}|{m}"] = [float(b.get_height()) for b in bars]
        ax.set_xticks(range(3), ["u", "v", "p"])
        ax.set_yscale("log")
        plain_log_axis(ax.yaxis)
        ax.set_title(title, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Relative L2")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="outside lower center", ncols=2)
    return fig, plotted


def fig_error_vs_time(data):
    """Relative L2 of u per snapshot vs t, mean over seeds. data = {method: {"t": [...], "u": [...]}}"""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.3), constrained_layout=True)
    plotted = {}
    for m in ("data_driven_nn", "pinn"):
        if m not in data:
            continue
        t, e = np.array(data[m]["t"]), np.array(data[m]["u"], dtype=float)
        (line,) = ax.plot(t, e, color=COLORS[m], label=LABELS[m], marker="o", markersize=5)
        plotted[m] = [float(v) for v in line.get_ydata()]
    for i, tb in enumerate(C.TEST_B_TIMES):
        ax.axvline(tb, color=INK2, linewidth=1.2, linestyle=":", zorder=0,
                   label="Test-B times" if i == 0 else None)
    ax.set_yscale("log")
    plain_log_axis(ax.yaxis)
    ax.set_xlabel("t")
    ax.set_ylabel("Rel. L2, u")
    fig.legend(frameon=False, loc="outside lower center", ncols=3, handlelength=1.0,
               columnspacing=0.8, fontsize=22)
    return fig, plotted


def fig_residuals(data):
    """RMS PDE residuals by method (autodiff, 50,000 points), mean ± std.

    data = {method: {"continuity": (m, s), "x_momentum": (m, s), "y_momentum": (m, s)}}
    """
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0), constrained_layout=True)
    names = ["continuity", "x_momentum", "y_momentum"]
    methods = [m for m in ("data_driven_nn", "pinn") if m in data]
    width = 0.8 / len(methods)
    plotted = {}
    for i, m in enumerate(methods):
        means = [data[m][n][0] for n in names]
        stds = [data[m][n][1] for n in names]
        xs = np.arange(3) + (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(xs, means, width * 0.92, yerr=stds, color=COLORS[m], label=LABELS[m], capsize=6,
                      error_kw={"elinewidth": 2, "ecolor": INK})
        plotted[m] = [float(b.get_height()) for b in bars]
    ax.set_xticks(range(3), ["Contin.", "x-mom.", "y-mom."])
    ax.set_yscale("log")
    plain_log_axis(ax.yaxis, subs=(1.0,))
    ax.set_ylabel("RMS resid.")
    ax.grid(axis="x", visible=False)
    fig.legend(frameon=False, loc="outside lower center", ncols=2)
    return fig, plotted


def fig_snapshots(fields, times):
    """u at selected times for FD, data NN, PINN, plus |error| maps vs FD.

    fields = {method: {"x", "y", "t", "u"}}; error maps for the two networks.
    """
    methods = ["finite_difference", "data_driven_nn", "pinn"]
    rows = methods + ["err_data_driven_nn", "err_pinn"]
    fig, axes = plt.subplots(len(rows), len(times), figsize=(WIDTH, 7.0), constrained_layout=True)
    ref = fields["finite_difference"]
    vmax = max(abs(ref["u"]).max(), 1.0)
    plotted = {}
    # Error colour scale: 99th percentile of |error| over both networks and the shown
    # times, so the lid-corner spikes (largest errors) do not wash out the interior
    shown = [int(np.argmin(np.abs(ref["t"] - tt))) for tt in times]
    errs = np.concatenate([np.abs(fields[m]["u"][shown] - ref["u"][shown]).ravel()
                           for m in ("data_driven_nn", "pinn")])
    err_max = float(np.percentile(errs, 99))
    for r, row in enumerate(rows):
        for c, tt in enumerate(times):
            ax = axes[r, c]
            k = int(np.argmin(np.abs(ref["t"] - tt)))
            if row.startswith("err_"):
                m = row[4:]
                img = np.abs(fields[m]["u"][k] - ref["u"][k])
                im_err = ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], cmap=SEQUENTIAL,
                                   vmin=0, vmax=err_max)
                label = f"{'NN' if m == 'data_driven_nn' else 'PINN'}\nerror"
            else:
                img = fields[row]["u"][k]
                im = ax.imshow(img, origin="lower", extent=[0, 1, 0, 1], cmap=DIVERGING,
                               vmin=-vmax, vmax=vmax)
                label = {"finite_difference": "FD", "data_driven_nn": "Data\nNN", "pinn": "PINN"}[row]
            plotted[f"{row}|t={tt}"] = {"max": float(img.max()), "min": float(img.min())}
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(f"t = {tt:g}", color=INK)
            if c == 0:
                ax.set_ylabel(label, color=INK)
    fig.colorbar(im, ax=axes[:3, :].ravel().tolist(), shrink=0.9, label="u")
    fig.colorbar(im_err, ax=axes[3:, :].ravel().tolist(), shrink=0.9, extend="max",
                 label="|u error|")
    return fig, plotted


def fig_accuracy_vs_time(curves):
    """Test-A relative L2 of u vs training time; one line per run, color by method,
    solid = GPU, dashed = CPU. curves = {run_id: {"method", "device", "time": [], "err": []}}"""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.7), constrained_layout=True)
    plotted = {}
    seen = set()
    for run_id, c in sorted(curves.items()):
        style = "-" if c["device"] == "gpu" else "--"
        label = f"{LABELS[c['method']]} {c['device'].upper()}"
        (line,) = ax.plot(c["time"], c["err"], style, color=COLORS[c["method"]], linewidth=2.5,
                          alpha=0.9, label=None if label in seen else label)
        seen.add(label)
        plotted[run_id] = [float(v) for v in line.get_ydata()]
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_axis(ax.xaxis, subs=(1.0,))
    plain_log_axis(ax.yaxis, subs=(1.0, 3.0))
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Rel. L2, u")
    fig.legend(frameon=False, loc="outside lower center", ncols=2)
    return fig, plotted


def fig_centerlines(val):
    """FD centerline u(0.5, y) and v(x, 0.5) at grids 32/64/128 vs Marchi et al. (2009)."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 5.2), constrained_layout=True)
    plotted = {}
    ys, xs = np.array(val["marchi_u_stations_y"]), np.array(val["marchi_v_stations_x"])
    for r in val["runs"]:
        n = r["n"]
        (l1,) = axes[0].plot(r["u_centerline"], ys, color=GRID_SHADES[n], label=f"FD {n}²")
        (l2,) = axes[1].plot(xs, r["v_centerline"], color=GRID_SHADES[n], label=f"FD {n}²")
        plotted[f"u|{n}"] = [float(v) for v in l1.get_xdata()]
        plotted[f"v|{n}"] = [float(v) for v in l2.get_ydata()]
    axes[0].plot(val["marchi_u"], ys, "o", mfc="white", mec=INK, mew=1.5, markersize=8, label="Marchi")
    axes[1].plot(xs, val["marchi_v"], "o", mfc="white", mec=INK, mew=1.5, markersize=8, label="Marchi")
    axes[0].set_xlabel("u(0.5, y)")
    axes[0].set_ylabel("y")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v(x, 0.5)")
    axes[1].set_xticks([0, 0.5, 1])
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="outside lower center", ncols=2,
               columnspacing=1.0, handletextpad=0.4)
    return fig, plotted


def fig_grid_convergence(val):
    """Max centerline deviation from Marchi vs grid spacing (log-log)."""
    fig, ax = plt.subplots(figsize=(WIDTH, 4.0), constrained_layout=True)
    dx = np.array([r["dx"] for r in val["runs"]])
    du = np.array([r["u_centerline_max_abs_dev"] for r in val["runs"]])
    dv = np.array([r["v_centerline_max_abs_dev"] for r in val["runs"]])
    (lu,) = ax.plot(dx, du, "o-", color=GRID_SHADES[64], markersize=10, label="u(0.5, y)")
    (lv,) = ax.plot(dx, dv, "s--", color=GRID_SHADES[128], markersize=10, label="v(x, 0.5)")
    ref = du[-1] * (dx / dx[-1]) ** 2
    ax.plot(dx, ref, ":", color=INK2, linewidth=2, label="slope 2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_axis(ax.xaxis)
    plain_log_axis(ax.yaxis)
    ax.set_xlabel("Grid spacing Δx")
    ax.set_ylabel("Max |FD − Marchi|")
    ax.legend(frameon=False, loc="upper left")
    return fig, {"u": [float(v) for v in lu.get_ydata()], "v": [float(v) for v in lv.get_ydata()],
                 "dx": [float(v) for v in lu.get_xdata()]}


def fig_vortex_center(val):
    """Primary-vortex center (ψ minimum) for FD grids vs Marchi and Hou."""
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4), constrained_layout=True)
    plotted = {}
    for r in val["runs"]:
        pts = ax.plot(r["vortex_x"], r["vortex_y"], "o", color=GRID_SHADES[r["n"]], markersize=14,
                      label=f"FD {r['n']}²")
        plotted[str(r["n"])] = [float(pts[0].get_xdata()[0]), float(pts[0].get_ydata()[0])]
    mx, my = val["marchi_vortex_center"]
    hx, hy = val["hou_vortex_center"]
    ax.plot(mx, my, "D", mfc="white", mec=INK, mew=2.5, markersize=14, label="Marchi")
    ax.plot(hx, hy, "s", mfc="white", mec=INK2, mew=2.5, markersize=14, label="Hou")
    ax.set_xlim(mx - 0.02, mx + 0.02)
    ax.set_ylim(my - 0.02, my + 0.02)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    return fig, plotted


def fig_schematic():
    """Cavity with boundary conditions and the lid ramp (no data)."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 3.8), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=4, ec=INK))
    ax.add_patch(Rectangle((0, 1), 1, 0.05, color=COLORS["finite_difference"]))
    ax.annotate("", (0.85, 1.14), (0.15, 1.14), arrowprops={"arrowstyle": "-|>", "lw": 3, "color": INK})
    ax.text(0.5, 1.22, "u = tanh(10t)", ha="center", fontsize=24, color=INK)
    ax.text(0.5, -0.16, "u = v = 0", ha="center", fontsize=24, color=INK)
    ax.text(-0.1, 0.5, "u = v = 0", ha="center", va="center", rotation=90, fontsize=24, color=INK)
    ax.text(1.1, 0.5, "u = v = 0", ha="center", va="center", rotation=90, fontsize=24, color=INK)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.62 + 0.2 * np.cos(th), 0.68 + 0.17 * np.sin(th), color=INK2, lw=2, ls="--")
    ax.text(0.5, 0.25, "Re = 100", ha="center", fontsize=24, color=INK)
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.22, 1.36)
    ax.set_aspect("equal")
    ax.axis("off")
    t = np.linspace(0, 1, 300)
    ax2.plot(t, np.tanh(C.LID_RAMP_RATE * t), color=COLORS["finite_difference"])
    ax2.set_xlabel("t")
    ax2.set_ylabel("lid u")
    return fig, {}


def fig_pipelines():
    """FD → data → data NN; physics → PINN; one shared evaluator."""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.5), constrained_layout=True)
    ax.axis("off")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.4)

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc="white", ec=color, lw=4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=24, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=30, lw=3, color=INK2))

    box(0.2, 6.0, 3.0, 2.0, "FD\nsolver", COLORS["finite_difference"])
    box(4.6, 6.0, 3.0, 2.0, "FD\ndata", COLORS["finite_difference"])
    box(9.0, 6.0, 2.6, 2.0, "Data\nNN", COLORS["data_driven_nn"])
    box(0.2, 0.4, 3.0, 2.0, "NS + IC\n+ BC", INK2)
    box(9.0, 0.4, 2.6, 2.0, "PINN", COLORS["pinn"])
    box(12.4, 3.2, 2.4, 2.0, "Eval.", INK)
    arrow(3.4, 7.0, 4.6, 7.0)
    arrow(7.8, 7.0, 9.0, 7.0)
    arrow(3.4, 1.4, 9.0, 1.4)
    arrow(11.8, 6.6, 12.6, 5.4)
    arrow(11.8, 1.8, 12.6, 3.0)
    ax.annotate("", (12.4, 4.2), (1.7, 5.9), arrowprops={"arrowstyle": "-|>", "lw": 2, "ls": ":",
                                                         "color": COLORS["finite_difference"],
                                                         "connectionstyle": "arc3,rad=0.12"})
    return fig, {}


# ---------------- data loading for the real figures ----------------

def gpu_runs(rows, method):
    return [r for r in rows if r["method"] == method and r["device"] == "gpu"
            and r["status"] == "evaluated" and r["timing_run"] == "False"]


def accuracy_data(agg):
    out = {}
    for m in ("data_driven_nn", "pinn"):
        g = group(agg, m)
        if g is None:
            continue
        out[m] = {s: {k: (g[f"{s}_relative_l2_{k}"]["mean"], g[f"{s}_relative_l2_{k}"]["std"]) for k in "uvp"}
                  for s in ("test_a", "test_b")}
        out[m]["runs"] = g["runs"]
    return out


def residual_data(agg):
    out = {}
    for m in ("data_driven_nn", "pinn"):
        g = group(agg, m)
        if g is None:
            continue
        out[m] = {n: (g[f"{n}_residual"]["mean"], g[f"{n}_residual"]["std"])
                  for n in ("continuity", "x_momentum", "y_momentum")}
        out[m]["runs"] = g["runs"]
    return out


def error_vs_time_data(rows):
    out = {}
    for m in ("data_driven_nn", "pinn"):
        runs = gpu_runs(rows, m)
        if not runs:
            continue
        per = []
        for r in runs:
            met = json.load(open(os.path.join(EXP, m, r["run_id"], "metrics.json")))
            ps = met["per_snapshot_relative_l2"]
            per.append(ps["u"])
        t = ps["t"]
        keep = [i for i in range(len(t)) if per[0][i] is not None]
        out[m] = {"t": [t[i] for i in keep],
                  "u": [float(np.mean([p[i] for p in per])) for i in keep],
                  "runs": [r["run_id"] for r in runs]}
    return out


def curve_data(agg, rows):
    meta = {r["run_id"]: r for r in rows}
    out = {}
    for run_id, c in agg["accuracy_vs_time"].items():
        r = meta[run_id]
        out[run_id] = {"method": r["method"], "device": r["device"],
                       "time": c["train_time_seconds"], "err": c["rel_l2_u"]}
    return out


# ---------------- self test on synthetic data ----------------

def selftest():
    ok = True

    def check(name, got, expected):
        nonlocal ok
        good = np.allclose(np.array(got, dtype=float), np.array(expected, dtype=float), rtol=1e-12)
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] selftest {name}")

    acc = {"data_driven_nn": {"test_a": {"u": (0.02, 0.001), "v": (0.03, 0.002), "p": (0.1, 0.01)},
                              "test_b": {"u": (0.04, 0.001), "v": (0.05, 0.002), "p": (0.2, 0.01)}},
           "pinn": {"test_a": {"u": (0.3, 0.01), "v": (0.4, 0.02), "p": (0.5, 0.03)},
                    "test_b": {"u": (0.31, 0.01), "v": (0.41, 0.02), "p": (0.51, 0.03)}}}
    fig, p = fig_accuracy(acc)
    plt.close(fig)
    check("accuracy bars", p["test_a|pinn"] + p["test_b|data_driven_nn"], [0.3, 0.4, 0.5, 0.04, 0.05, 0.2])
    res = {"data_driven_nn": {"continuity": (0.5, 0.1), "x_momentum": (5.0, 1.0), "y_momentum": (2.0, 0.3)},
           "pinn": {"continuity": (0.01, 0.001), "x_momentum": (0.02, 0.002), "y_momentum": (0.03, 0.003)}}
    fig, p = fig_residuals(res)
    plt.close(fig)
    check("residual bars", p["data_driven_nn"] + p["pinn"], [0.5, 5.0, 2.0, 0.01, 0.02, 0.03])
    t = np.linspace(0.1, 5, 50)
    ev = {"data_driven_nn": {"t": t, "u": 0.01 + 0 * t}, "pinn": {"t": t, "u": np.exp(-t)}}
    fig, p = fig_error_vs_time(ev)
    plt.close(fig)
    check("error-vs-time line", p["pinn"], np.exp(-t))
    cur = {"a_s1": {"method": "pinn", "device": "gpu", "time": [1, 10, 100], "err": [1, 0.5, 0.25]},
           "b_s1": {"method": "data_driven_nn", "device": "cpu", "time": [2, 20], "err": [0.3, 0.1]}}
    fig, p = fig_accuracy_vs_time(cur)
    plt.close(fig)
    check("accuracy-vs-time lines", p["a_s1"] + p["b_s1"], [1, 0.5, 0.25, 0.3, 0.1])
    x = np.linspace(0, 1, 8)
    tt = np.array([0.5, 2.0, 5.0])
    base = np.ones((3, 8, 8)) * 0.2
    flds = {"finite_difference": {"x": x, "y": x, "t": tt, "u": base},
            "data_driven_nn": {"x": x, "y": x, "t": tt, "u": base + 0.1},
            "pinn": {"x": x, "y": x, "t": tt, "u": base - 0.05}}
    fig, p = fig_snapshots(flds, [0.5, 2.0, 5.0])
    plt.close(fig)
    check("error map max (data NN 0.1, PINN 0.05)",
          [p["err_data_driven_nn|t=2.0"]["max"], p["err_pinn|t=5.0"]["max"]], [0.1, 0.05])
    val = {"marchi_u_stations_y": [0.25, 0.5, 0.75], "marchi_u": [-0.1, -0.2, 0.1],
           "marchi_v_stations_x": [0.25, 0.5, 0.75], "marchi_v": [0.1, 0.05, -0.2],
           "marchi_vortex_center": (0.6, 0.73), "hou_vortex_center": (0.61, 0.74),
           "runs": [{"n": n, "dx": 1 / (n - 1), "u_centerline": [0.1 * n, 0, 0], "v_centerline": [0, 0, n],
                     "u_centerline_max_abs_dev": 1 / n, "v_centerline_max_abs_dev": 2 / n,
                     "vortex_x": 0.6 + 1 / n, "vortex_y": 0.73} for n in (32, 64, 128)]}
    fig, p = fig_grid_convergence(val)
    plt.close(fig)
    check("grid convergence points", p["u"] + p["v"], [1 / 32, 1 / 64, 1 / 128, 2 / 32, 2 / 64, 2 / 128])
    fig, p = fig_centerlines(val)
    plt.close(fig)
    check("centerline curves", p["u|64"] + p["v|128"], [6.4, 0, 0, 0, 0, 128])
    fig, p = fig_vortex_center(val)
    plt.close(fig)
    check("vortex centers", p["32"] + p["128"], [0.6 + 1 / 32, 0.73, 0.6 + 1 / 128, 0.73])
    span = run_span(["dnn_re100_gpu_s1", "dnn_re100_gpu_s3", "pinn_re100_gpu_s2"])
    good = span == "dnn_re100_gpu_s1-s3__pinn_re100_gpu_s2"
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] selftest run-id label: {span}")
    return ok


# ---------------- G15 check ----------------

def check_against_results(rows, agg):
    """Every plotted number must equal the value in experiment_results.csv / aggregate.json."""
    ok = True
    by_id = {r["run_id"]: r for r in rows}

    def close(a, b):
        return np.isclose(float(a), float(b), rtol=1e-9, atol=0)

    for path in sorted(glob.glob(os.path.join(OUT, "*.values.json"))):
        v = json.load(open(path))
        name, plotted = v["figure"], v["plotted"]
        good = True
        if name == "accuracy":
            for key, heights in plotted.items():
                split, m = key.split("|")
                runs = [r for r in v["runs"] if by_id[r]["method"] == m]
                for k, h in zip("uvp", heights):
                    col = [float(by_id[r][f"{split}_relative_l2_{k}"]) for r in runs]
                    good &= close(h, np.mean(col))
        elif name == "residuals":
            for m, heights in plotted.items():
                runs = [r for r in v["runs"] if by_id[r]["method"] == m]
                for n, h in zip(("continuity", "x_momentum", "y_momentum"), heights):
                    col = [float(by_id[r][f"{n}_residual"]) for r in runs]
                    good &= close(h, np.mean(col))
        elif name == "accuracy_vs_time":
            for run_id, ys in plotted.items():
                good &= np.allclose(ys, agg["accuracy_vs_time"][run_id]["rel_l2_u"], rtol=1e-12)
        elif name == "error_vs_time":
            for m, ys in plotted.items():
                runs = [r for r in v["runs"] if by_id[r]["method"] == m]
                per = [json.load(open(os.path.join(EXP, m, r, "metrics.json")))["per_snapshot_relative_l2"]["u"]
                       for r in runs]
                exp = [np.mean([p[i] for p in per]) for i in range(len(per[0])) if per[0][i] is not None]
                good &= np.allclose(ys, exp, rtol=1e-12)
        elif name in ("fd_centerlines", "fd_grid_convergence", "fd_vortex_center"):
            val = json.load(open(os.path.join(RESULTS, "fd_validation_re100.json")))
            runs = {str(r["n"]): r for r in val["runs"]}
            if name == "fd_grid_convergence":
                good &= np.allclose(plotted["u"], [r["u_centerline_max_abs_dev"] for r in val["runs"]])
                good &= np.allclose(plotted["v"], [r["v_centerline_max_abs_dev"] for r in val["runs"]])
            elif name == "fd_vortex_center":
                for n, (x, y) in plotted.items():
                    good &= close(x, runs[n]["vortex_x"]) and close(y, runs[n]["vortex_y"])
            else:
                for key, ys in plotted.items():
                    comp, n = key.split("|")
                    good &= np.allclose(ys, runs[n][f"{comp}_centerline"])
        else:
            print(f"[INFO] {name}: no tabulated numbers to compare (field maps / diagrams)")
            continue
        ok &= bool(good)
        print(f"[{'PASS' if good else 'FAIL'}] G15 {os.path.basename(path)}")
    return ok


# ---------------- main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    have_results = os.path.exists(os.path.join(RESULTS, "experiment_results.csv")) and \
        os.path.exists(os.path.join(RESULTS, "aggregate.json"))
    rows = load_csv() if have_results else []
    agg = load_agg() if have_results else {"summary": {}, "accuracy_vs_time": {}}
    val = json.load(open(os.path.join(RESULTS, "fd_validation_re100.json")))
    want = set(args.only) if args.only else None

    def wanted(name):
        return want is None or name in want

    for old in glob.glob(os.path.join(OUT, "*")):
        base = os.path.basename(old).split("__")[0].split(".")[0]
        if want is None or base in want:
            os.remove(old)

    made = []
    if wanted("accuracy"):
        d = accuracy_data(agg)
        fig, p = fig_accuracy(d)
        made.append(save(fig, "accuracy", sum((d[m]["runs"] for m in d), []), p))
    if wanted("residuals"):
        d = residual_data(agg)
        fig, p = fig_residuals(d)
        made.append(save(fig, "residuals", sum((d[m]["runs"] for m in d), []), p))
    if wanted("error_vs_time"):
        d = error_vs_time_data(rows)
        fig, p = fig_error_vs_time(d)
        made.append(save(fig, "error_vs_time", sum((d[m]["runs"] for m in d), []), p))
    if wanted("accuracy_vs_time"):
        d = curve_data(agg, rows)
        fig, p = fig_accuracy_vs_time(d)
        made.append(save(fig, "accuracy_vs_time", list(d), p))
    if wanted("snapshots"):
        ref_id = C.fd_reference_run_id(100.0)
        flds = {"finite_difference": dict(np.load(os.path.join(EXP, "finite_difference", ref_id, "solution.npz")))}
        used = [ref_id]
        for m, prefix in (("data_driven_nn", "dnn"), ("pinn", "pinn")):
            rid = f"{prefix}_re100_gpu_s1"
            flds[m] = dict(np.load(os.path.join(EXP, m, rid, "predictions.npz")))
            used.append(rid)
        fig, p = fig_snapshots(flds, [0.5, 2.0, 5.0])
        made.append(save(fig, "snapshots", used, p))
    if wanted("fd_centerlines"):
        fig, p = fig_centerlines(val)
        made.append(save(fig, "fd_centerlines", [r["run_id"] for r in val["runs"]], p))
    if wanted("fd_grid_convergence"):
        fig, p = fig_grid_convergence(val)
        made.append(save(fig, "fd_grid_convergence", [r["run_id"] for r in val["runs"]], p))
    if wanted("fd_vortex_center"):
        fig, p = fig_vortex_center(val)
        made.append(save(fig, "fd_vortex_center", [r["run_id"] for r in val["runs"]], p))
    if wanted("schematic"):
        fig, p = fig_schematic()
        made.append(save(fig, "schematic", [], p))
    if wanted("pipelines"):
        fig, p = fig_pipelines()
        made.append(save(fig, "pipelines", [], p))
    for m in made:
        print("wrote", m)
    raise SystemExit(0 if check_against_results(rows, agg) else 1)


if __name__ == "__main__":
    main()
