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

SOURCES = "poster/poster_draft_sources.json"
NUMS = json.load(open(NUMBERS)) if os.path.exists(NUMBERS) else {}
MISSING = []
USED = {}          # every number placed on the poster: key -> printed text, value, run ids


def num(key, fmt="{:.3f}", scale=1.0):
    v = NUMS.get(key, {}).get("value")
    if v is None:
        MISSING.append(key)
        return f"[missing: {key}]"
    text = fmt.format(v * scale)
    entry = USED.setdefault(key, {"value": v, "description": NUMS[key].get("description"),
                                  "runs": NUMS[key].get("runs", []), "printed_as": []})
    if text not in entry["printed_as"]:
        entry["printed_as"].append(text)
    return text


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


def text_height(paragraphs, size, width):
    """Rough height (in) of wrapped Arial text: mean glyph width ~0.44 em, line height 1.2 em."""
    chars_per_line = max(1, int(width * 72 / (size * 0.44)))
    lines = 0
    for para in paragraphs:
        segments = para if isinstance(para, list) else [para]
        n = sum(len(seg if isinstance(seg, str) else seg[0]) for seg in segments)
        lines += max(1, -(-n // chars_per_line))
    return lines * size * 1.2 / 72 + len(paragraphs) * 6 / 72 + 0.1


def block(slide, col, y, paragraphs, size=BODY_PT):
    """Text block in a column; returns the y just below it."""
    h = text_height(paragraphs, size, COL_W - 0.1)
    textbox(slide, COL_X[col], y, COL_W, h, paragraphs, size=size)
    return y + h + 0.1


COLUMN_END = {}


def heading(slide, col, y, text):
    textbox(slide, COL_X[col], y, COL_W, 0.66, [(text, {"bold": True})], size=HEAD_PT, color=HEAD_COLOR)
    return y + 0.72


def picture(slide, col, y, name, width=COL_W):
    path = figure(name)
    if path is None:
        textbox(slide, COL_X[col], y, width, 0.5, [f"[missing figure: {name}]"])
        return y + 0.6
    x = COL_X[col] + (COL_W - width) / 2          # narrower figures are centred in the column
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(width))
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
        "boundary conditions only (no FD data). Test A: 15% random held-out points; "
        "test B: 10 whole snapshots (t = 0.5, 1.0, …, 5.0) never seen in training.",
    ]
    t["validation"] = [
        "Steady FD solutions vs Marchi et al. (2009, 1024×1024): maximum centerline deviation of u "
        f"falls from {num('fd_val_n32_u_max_dev', '{:.4f}')} (32×32) to "
        f"{num('fd_val_n64_u_max_dev', '{:.4f}')} (64×64) and "
        f"{num('fd_val_n128_u_max_dev', '{:.4f}')} (128×128). At 64×64 the vortex center is "
        f"{num('fd_val_n64_vortex_dist_marchi', '{:.4f}')} from Marchi and "
        f"{num('fd_val_n64_vortex_dist_hou', '{:.4f}')} from Hou et al. (1995), within one grid "
        "spacing (Δx = 1/63).",
        "Transient check: halving Δt changes the 64×64 velocity at t = 5 by at most "
        f"{num('fd_dt_halving_max_diff_t5', '{:.1e}')}.",
    ]
    t["discussion"] = [
        [("Accuracy. ", {"bold": True}),
         "The data NN is closer to FD: test-A relative L2 of u "
         f"{num('dnn_gpu_test_a_relative_l2_u_mean')} ± {num('dnn_gpu_test_a_relative_l2_u_std')} vs "
         f"{num('pinn_gpu_test_a_relative_l2_u_mean')} ± {num('pinn_gpu_test_a_relative_l2_u_std')} "
         "(PINN; mean ± std, 5 GPU seeds). Test B is similar: "
         f"{num('dnn_gpu_test_b_relative_l2_u_mean')} vs {num('pinn_gpu_test_b_relative_l2_u_mean')}."],
        [("Physics. ", {"bold": True}),
         "PINN fields obey the equations more closely: RMS continuity residual "
         f"{num('pinn_gpu_continuity_residual_mean', '{:.2g}')} vs "
         f"{num('dnn_gpu_continuity_residual_mean', '{:.2g}')}, x-momentum "
         f"{num('pinn_gpu_x_momentum_residual_mean', '{:.2g}')} vs "
         f"{num('dnn_gpu_x_momentum_residual_mean', '{:.2g}')} (data NN)."],
        [("Cost. ", {"bold": True}),
         f"GPU training: PINN {num('pinn_gpu_training_time_seconds_mean', '{:.0f}')} s, data NN "
         f"{num('dnn_gpu_training_time_seconds_mean', '{:.0f}')} s; one FD solve "
         f"{num('fd_n64_solve_seconds_median', '{:.1f}')} s (CPU). Training is repaid after about "
         f"{num('dnn_gpu_break_even_evaluations', '{:.0f}')} (data NN) or "
         f"{num('pinn_gpu_break_even_evaluations', '{:.0f}')} (PINN) full-field evaluations "
         "instead of FD runs."],
    ]
    t["conclusion"] = [
        "At Re = 100 the PINN learned the cavity flow without any simulation data, and its fields "
        "obey the Navier–Stokes equations more closely than the data-driven network's. The "
        "data-driven network is more accurate against FD and trains faster. For a single field, "
        "FD is cheapest. Next: Re 400 and PINN sensitivity to collocation and loss weighting.",
    ]
    t["references"] = [
        "[1] Hou, S., Zou, Q., Chen, S., Doolen, G., and Cogley, A. C. 1995. Simulation of cavity flow "
        "by the lattice Boltzmann method. J. Comput. Phys. 118, 2, 329–347.",
        "[2] Marchi, C. H., Suero, R., and Araki, L. K. 2009. The lid-driven square cavity flow: numerical "
        "solution with a 1024 x 1024 grid. J. Braz. Soc. Mech. Sci. Eng. 31, 3, 186–198.",
        "[3] Raissi, M., Perdikaris, P., and Karniadakis, G. E. 2019. Physics-informed neural networks: "
        "a deep learning framework for solving forward and inverse problems involving nonlinear partial "
        "differential equations. J. Comput. Phys. 378, 686–707.",
    ]
    return t


def cost_table(slide, x, y, w):
    rows = [
        ("", "Train (s)", "To field (s)", "Infer. (ms)", "RAM (GB)", "VRAM (MB)"),
        ("FD 64² CPU", "–", num("fd_n64_solve_seconds_median", "{:.1f}"), "–",
         num("fd_n64_peak_ram_bytes", "{:.2f}", 1 / 1e9), "–"),
        ("Data NN GPU", num("dnn_gpu_training_time_seconds_mean", "{:.0f}"),
         num("dnn_gpu_total_cost_to_field_seconds", "{:.0f}"),
         num("dnn_gpu_inference_time_seconds_mean", "{:.1f}", 1e3),
         num("dnn_gpu_peak_ram_bytes_mean", "{:.2f}", 1 / 1e9),
         num("dnn_gpu_peak_vram_bytes_mean", "{:.0f}", 1 / 2**20)),
        ("PINN GPU", num("pinn_gpu_training_time_seconds_mean", "{:.0f}"),
         num("pinn_gpu_total_cost_to_field_seconds", "{:.0f}"),
         num("pinn_gpu_inference_time_seconds_mean", "{:.1f}", 1e3),
         num("pinn_gpu_peak_ram_bytes_mean", "{:.2f}", 1 / 1e9),
         num("pinn_gpu_peak_vram_bytes_mean", "{:.0f}", 1 / 2**20)),
        ("Data NN CPU", num("dnn_cpu_training_time_seconds_mean", "{:.0f}"),
         num("dnn_cpu_total_cost_to_field_seconds", "{:.0f}"),
         num("dnn_cpu_inference_time_seconds_mean", "{:.0f}", 1e3),
         num("dnn_cpu_peak_ram_bytes_mean", "{:.2f}", 1 / 1e9), "–"),
        ("PINN CPU †", num("pinn_cpu_time_per_step_seconds_mean", "{:.2f}") + " s/step", "–", "–",
         num("pinn_cpu_peak_ram_bytes_mean", "{:.2f}", 1 / 1e9), "–"),
    ]
    h_head, h_row = 0.58, 0.33
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                   Inches(h_head + h_row * (len(rows) - 1)))
    table = shape.table
    for i in range(len(rows)):
        table.rows[i].height = Inches(h_head if i == 0 else h_row)
    widths = [1.7, 1.45, 1.15, 1.3, 0.95, 1.0]
    for j, cw in enumerate(widths):
        table.columns[j].width = Inches(cw * w / sum(widths))
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            r = cell.text_frame.paragraphs[0].add_run()
            r.text = val
            style_run(r, 16, bold=(i == 0 or j == 0), color=WHITE if i == 0 else BODY_COLOR)
            cell.margin_left = cell.margin_right = Inches(0.04)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
    return y + h_head + h_row * (len(rows) - 1) + 0.1


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
    y = block(slide, 0, y, t["abstract"])
    y = heading(slide, 0, y + 0.1, "Introduction")
    y = block(slide, 0, y, t["intro"])
    y = picture(slide, 0, y, "schematic")
    y = heading(slide, 0, y, "Methods")
    y = block(slide, 0, y, t["methods"])
    COLUMN_END[1] = picture(slide, 0, y, "pipelines")

    # Column 2: FD validation
    y = heading(slide, 1, TOP, "FD Validation")
    y = block(slide, 1, y, t["validation"])
    y = picture(slide, 1, y, "fd_centerlines")
    y = picture(slide, 1, y, "fd_vortex_center")
    COLUMN_END[2] = picture(slide, 1, y, "fd_grid_convergence")

    # Column 3: accuracy and physics
    y = heading(slide, 2, TOP, "Accuracy & Physics")
    y = picture(slide, 2, y, "accuracy")
    y = picture(slide, 2, y, "error_vs_time")
    y = picture(slide, 2, y, "residuals")
    COLUMN_END[3] = picture(slide, 2, y, "snapshots")

    # Column 4: cost, discussion, conclusion, references, contact
    y = heading(slide, 3, TOP, "Computational Cost")
    y = picture(slide, 3, y, "accuracy_vs_time")
    y = cost_table(slide, COL_X[3], y, COL_W)
    y = block(slide, 3, y, ["To field: FD = solve; data NN = FD solve + training; PINN = training. "
                            "Infer.: full 64×64×51 field, median of 20 calls. RAM, VRAM: peak. "
                            "† 2,000-step timing run."], size=14)
    y = heading(slide, 3, y, "Discussion")
    y = block(slide, 3, y, t["discussion"], size=19)
    y = heading(slide, 3, y, "Conclusion")
    y = block(slide, 3, y, t["conclusion"], size=19)
    y = heading(slide, 3, y, "References")
    y = block(slide, 3, y, t["references"], size=14)
    COLUMN_END[4] = block(slide, 3, y, [[("Contact: ", {"bold": True, "color": HEAD_COLOR}),
                                               "CONTACT EMAIL"]])

    for c, end_y in sorted(COLUMN_END.items()):
        status = "OVERFLOW" if end_y > BOTTOM else "ok"
        print(f"column {c}: content ends at {end_y:.2f} in (limit {BOTTOM}) {status}")

    prs.save(OUT)
    with open(SOURCES, "w") as f:
        json.dump(USED, f, indent=1)
    used = runs_of(*USED.keys())
    print(f"wrote {OUT} and {SOURCES}: {len(USED)} numbers from {len(used)} runs; "
          f"missing keys: {sorted(set(MISSING))}")
    return MISSING


if __name__ == "__main__":
    missing = build()
    sys.exit(0)
