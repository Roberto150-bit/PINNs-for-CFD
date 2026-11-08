"""Build poster/poster_draft.pptx on the SCE Project Day template.

Every number in the text is read from experiments/results/poster_numbers.json
(generated from the run folders by aggregate.py); a missing value appears as
[missing: key] rather than a typed-in number. Figures come from
experiments/analysis/plots/ (make_figures.py).

    python poster/build_poster.py
"""
import glob
import json
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

TEMPLATE = "poster/template.pptx"
OUT = "poster/poster_draft.pptx"
PLOTS = "experiments/analysis/plots"
NUMBERS = "experiments/results/poster_numbers.json"

# Styles taken from the Project Day example poster
FONT = "Arial"
HEAD_COLOR = RGBColor(0x41, 0x8F, 0xDE)
BODY_COLOR = RGBColor(0x0D, 0x22, 0x40)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD_PT, BODY_PT, SMALL_PT = 42, 22, 18

COL_X = [2.32, 10.30, 18.20, 26.15]   # template column positions (inches)
COL_W = 7.52
TOP = 5.15
BOTTOM = 23.45

NUMS = json.load(open(NUMBERS)) if os.path.exists(NUMBERS) else {}
MISSING = []


def num(key, fmt="{:.3f}", scale=1.0):
    v = NUMS.get(key, {}).get("value")
    if v is None:
        MISSING.append(key)
        return f"[missing: {key}]"
    return fmt.format(v * scale)


def runs_of(*keys):
    ids = set()
    for k in keys:
        ids.update(NUMS.get(k, {}).get("runs", []))
    return sorted(ids)


def figure(name):
    hits = sorted(glob.glob(os.path.join(PLOTS, f"{name}*.png")))
    hits = [h for h in hits if os.path.basename(h).split("__")[0].split(".")[0] == name]
    return hits[0] if hits else None


# ---------------- slide helpers ----------------

def style_run(run, size, bold=False, color=BODY_COLOR, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def textbox(slide, x, y, w, h, paragraphs, size=BODY_PT, color=BODY_COLOR, anchor=MSO_ANCHOR.TOP):
    """paragraphs: list of str or (str, {"bold": .., "size": ..}) or list of such segments."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(6)
        segments = para if isinstance(para, list) else [para]
        for seg in segments:
            text, opts = (seg, {}) if isinstance(seg, str) else seg
            r = p.add_run()
            r.text = text
            style_run(r, opts.get("size", size), opts.get("bold", False), opts.get("color", color),
                      opts.get("italic", False))
    return tb


def heading(slide, col, y, text):
    textbox(slide, COL_X[col], y, COL_W, 0.66, [(text, {"bold": True})], size=HEAD_PT, color=HEAD_COLOR)
    return y + 0.72


def picture(slide, col, y, name, width=COL_W):
    path = figure(name)
    if path is None:
        textbox(slide, COL_X[col], y, width, 0.5, [f"[missing figure: {name}]"])
        return y + 0.6
    pic = slide.shapes.add_picture(path, Inches(COL_X[col]), Inches(y), width=Inches(width))
    return y + pic.height / 914400 + 0.1


def fill_placeholder(ph, text, size, color, bold=True):
    tf = ph.text_frame
    tf.text = ""
    r = tf.paragraphs[0].add_run()
    r.text = text
    style_run(r, size, bold, color)


# ---------------- content ----------------

def text_blocks():
    t = {}
    t["abstract"] = [
        "Physics-informed neural networks (PINNs) train on the governing equations instead of "
        "simulation data. We compare a PINN with a data-driven network and a finite-difference (FD) "
        "solver on 2D lid-driven cavity flow at Re = 100, measuring accuracy, physical consistency "
        "and computational cost on CPU and GPU."
    ]
    t["intro"] = [
        "Does training with Navier–Stokes physics improve physical consistency enough to justify "
        "its extra cost?",
        [("RQ1 ", {"bold": True}), "Accuracy of u, v, p against FD"],
        [("RQ2 ", {"bold": True}), "Continuity and momentum residuals"],
        [("RQ3 ", {"bold": True}), "Data NN vs PINN: consistency and repeatability"],
        [("RQ4 ", {"bold": True}), "Training time, inference time, memory, storage"],
    ]
    t["methods"] = [
        "Unit cavity, lid u = tanh(10t). FD: 64×64 grid, Δt = 0.001, 51 snapshots t = 0–5. "
        "Both networks: (x, y, t) → (u, v, p), 5×64 tanh, 17,091 parameters, Adam 1e-3, "
        "20,000 steps, 5 seeds. Data NN: FD data only. PINN: PDE residuals + initial and "
        "boundary conditions only (no FD data).",
    ]
    t["validation"] = [
        "Steady FD solutions vs Marchi et al. (2009, 1024×1024): maximum centerline deviation of u "
        f"falls from {num('fd_val_n32_u_max_dev', '{:.4f}')} (32×32) to "
        f"{num('fd_val_n64_u_max_dev', '{:.4f}')} (64×64) and "
        f"{num('fd_val_n128_u_max_dev', '{:.4f}')} (128×128). At 64×64 the vortex center is "
        f"{num('fd_val_n64_vortex_dist_marchi', '{:.4f}')} from Marchi and "
        f"{num('fd_val_n64_vortex_dist_hou', '{:.4f}')} from Hou et al. (1995), within one grid "
        "spacing (0.0159).",
        "Transient check: halving Δt changes the 64×64 velocity at t = 5 by at most "
        f"{num('fd_dt_halving_max_diff_t5', '{:.1e}')}.",
    ]
    t["discussion"] = [
        "On test A the data NN's relative L2 error of u is "
        f"{num('dnn_gpu_test_a_relative_l2_u_mean')} ± {num('dnn_gpu_test_a_relative_l2_u_std')} and the "
        f"PINN's is {num('pinn_gpu_test_a_relative_l2_u_mean')} ± {num('pinn_gpu_test_a_relative_l2_u_std')} "
        "(mean ± std, 5 GPU seeds).",
        "RMS continuity residual: data NN "
        f"{num('dnn_gpu_continuity_residual_mean', '{:.2g}')}, PINN "
        f"{num('pinn_gpu_continuity_residual_mean', '{:.2g}')}; x-momentum: data NN "
        f"{num('dnn_gpu_x_momentum_residual_mean', '{:.2g}')}, PINN "
        f"{num('pinn_gpu_x_momentum_residual_mean', '{:.2g}')}.",
        "Training on the GPU took "
        f"{num('dnn_gpu_training_time_seconds_mean', '{:.0f}')} s (data NN) and "
        f"{num('pinn_gpu_training_time_seconds_mean', '{:.0f}')} s (PINN); one FD solve took "
        f"{num('fd_n64_solve_seconds_median', '{:.1f}')} s on the CPU.",
    ]
    t["conclusion"] = [
        "[Draft: to be written from the results above.]",
    ]
    t["references"] = [
        "[1] Hou, S., Zou, Q., Chen, S., Doolen, G., and Cogley, A. C. 1995. Simulation of cavity flow "
        "by the lattice Boltzmann method. J. Comput. Phys. 118, 2, 329–347.",
        "[2] Marchi, C. H., Suero, R., and Araki, L. K. 2009. The lid-driven square cavity flow: numerical "
        "solution with a 1024 x 1024 grid. J. Braz. Soc. Mech. Sci. Eng. 31, 3, 186–198.",
        "[3] Raissi, M., Perdikaris, P., and Karniadakis, G. E. 2019. Physics-informed neural networks. "
        "J. Comput. Phys. 378, 686–707.",
    ]
    return t


def cost_table(slide, x, y, w):
    rows = [
        ("", "Train (s)", "Full field (ms)", "Peak RAM (GB)", "Peak VRAM (MB)"),
        ("FD 64² (CPU)", num("fd_n64_solve_seconds_median", "{:.1f}") + "*", "–",
         num("fd_n64_peak_ram_bytes", "{:.2f}", 1 / 1e9), "–"),
        ("Data NN GPU", num("dnn_gpu_training_time_seconds_mean", "{:.0f}"),
         num("dnn_gpu_inference_time_seconds_mean", "{:.1f}", 1e3),
         num("dnn_gpu_peak_ram_bytes_mean", "{:.2f}", 1 / 1e9),
         num("dnn_gpu_peak_vram_bytes_mean", "{:.0f}", 1 / 2**20)),
        ("PINN GPU", num("pinn_gpu_training_time_seconds_mean", "{:.0f}"),
         num("pinn_gpu_inference_time_seconds_mean", "{:.1f}", 1e3),
         num("pinn_gpu_peak_ram_bytes_mean", "{:.2f}", 1 / 1e9),
         num("pinn_gpu_peak_vram_bytes_mean", "{:.0f}", 1 / 2**20)),
        ("Data NN CPU", num("dnn_cpu_training_time_seconds_mean", "{:.0f}"), "–", "–", "–"),
        ("PINN CPU †", num("pinn_cpu_time_per_step_seconds_mean", "{:.2f}") + " s/step", "–",
         num("pinn_cpu_peak_ram_bytes_mean", "{:.2f}", 1 / 1e9), "–"),
    ]
    h_row = 0.42
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                   Inches(h_row * len(rows)))
    table = shape.table
    widths = [1.9, 1.35, 1.45, 1.4, 1.42]
    for j, cw in enumerate(widths):
        table.columns[j].width = Inches(cw * w / sum(widths))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            r = cell.text_frame.paragraphs[0].add_run()
            r.text = val
            style_run(r, SMALL_PT, bold=(i == 0 or j == 0), color=WHITE if i == 0 else BODY_COLOR)
            cell.margin_left = cell.margin_right = Inches(0.04)
    return y + h_row * len(rows) + 0.05


def build():
    prs = Presentation(TEMPLATE)
    slide = prs.slides[0]

    # Header: title, author, adviser placeholders; remove every other placeholder
    keep = {}
    for ph in list(slide.placeholders):
        top = ph.top / 914400
        if 10 < ph.left / 914400 < 11 and top < 3.5:
            keep[round(top, 2)] = ph
        else:
            ph._element.getparent().remove(ph._element)
    title, names, advisers = (keep[k] for k in sorted(keep))
    fill_placeholder(title, "Evaluating Physics-Informed Learning for Fluid Dynamics", 66, HEAD_COLOR)
    fill_placeholder(names, "Roberto Pollicino (CS '26)", 40, HEAD_COLOR)
    fill_placeholder(advisers, "Adviser: Brian C. O'Neill, PhD (Computer Science)", 40, WHITE)

    t = text_blocks()

    # Column 1: Abstract, Introduction, Methods
    y = heading(slide, 0, TOP, "Abstract")
    textbox(slide, COL_X[0], y, COL_W, 2.6, t["abstract"])
    y = heading(slide, 0, y + 2.65, "Introduction")
    textbox(slide, COL_X[0], y, COL_W, 3.0, t["intro"])
    y = picture(slide, 0, y + 3.05, "schematic")
    y = heading(slide, 0, y, "Methods")
    textbox(slide, COL_X[0], y, COL_W, 2.2, t["methods"])
    picture(slide, 0, y + 2.25, "pipelines")

    # Column 2: FD validation
    y = heading(slide, 1, TOP, "FD Validation")
    textbox(slide, COL_X[1], y, COL_W, 3.3, t["validation"])
    y = picture(slide, 1, y + 3.35, "fd_centerlines")
    y = picture(slide, 1, y, "fd_vortex_center")
    picture(slide, 1, y, "fd_grid_convergence")

    # Column 3: accuracy and physics
    y = heading(slide, 2, TOP, "Accuracy & Physics")
    y = picture(slide, 2, y, "accuracy")
    y = picture(slide, 2, y, "error_vs_time")
    y = picture(slide, 2, y, "residuals")
    picture(slide, 2, y, "snapshots")

    # Column 4: cost, discussion, conclusion, references, contact
    y = heading(slide, 3, TOP, "Computational Cost")
    y = picture(slide, 3, y, "accuracy_vs_time")
    y = cost_table(slide, COL_X[3], y, COL_W)
    textbox(slide, COL_X[3], y, COL_W, 0.4,
            ["* FD solve time (CPU), median of 3. † 2,000-step timing run."], size=16)
    y = heading(slide, 3, y + 0.45, "Discussion")
    textbox(slide, COL_X[3], y, COL_W, 3.2, t["discussion"], size=20)
    y = heading(slide, 3, y + 3.25, "Conclusion")
    textbox(slide, COL_X[3], y, COL_W, 1.0, t["conclusion"], size=20)
    y = heading(slide, 3, y + 1.05, "References")
    textbox(slide, COL_X[3], y, COL_W, 2.2, t["references"], size=14)
    y = heading(slide, 3, y + 2.25, "Contact")
    textbox(slide, COL_X[3], y, COL_W, 0.5, ["CONTACT EMAIL"])

    prs.save(OUT)
    used = runs_of(*NUMS.keys())
    print(f"wrote {OUT}; numbers from {len(used)} runs; missing keys: {sorted(set(MISSING))}")
    return MISSING


if __name__ == "__main__":
    missing = build()
    sys.exit(0)
