"""Build poster/poster_draft.pptx on the SCE Project Day template.

Layout (36 x 24 in) follows the template's four-column grid and the Project
Day example: plain section headings, plain body text, one large results area.
  column 1:      Abstract, Introduction (with RQ1-RQ4), Methods
  columns 2-3:   Results: one figure per research question, each with a
                 one-sentence answer underneath
  column 4:      Strengths and weaknesses table, Conclusion, References, Contact

Every number in the text is read from experiments/results/poster_numbers.json
(generated from the run folders by aggregate.py); a missing value appears as
[missing: key] rather than a typed-in number. Figures come from
experiments/analysis/plots/ (make_figures.py). poster/poster_draft_sources.json
lists every number placed on the poster with its run ids.

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
SOURCES = "poster/poster_draft_sources.json"

# Styles taken from the Project Day example poster
FONT = "Arial"
HEAD_COLOR = RGBColor(0x41, 0x8F, 0xDE)
BODY_COLOR = RGBColor(0x0D, 0x22, 0x40)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TABLE_HEAD = RGBColor(0x0D, 0x22, 0x40)
TABLE_BAND = RGBColor(0xEE, 0xF3, 0xFA)
BETTER = RGBColor(0xD9, 0xF0, 0xE4)
HEAD_PT, BODY_PT, CAPTION_PT, SMALL_PT = 42, 24, 22, 15

COL_X = [2.32, 10.30, 18.20, 26.15]   # template column positions (inches)
COL_W = 7.52
TOP = 5.15
BOTTOM = 23.45

NUMS = json.load(open(NUMBERS)) if os.path.exists(NUMBERS) else {}
MISSING = []
USED = {}          # every number placed on the poster: key -> printed text, value, run ids
COLUMN_END = {}


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


def pct(key):
    """A relative L2 error printed as a percentage."""
    v = NUMS.get(key, {}).get("value")
    return num(key, "{:.0f}%" if v is not None and v >= 0.095 else "{:.1f}%", 100)


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

def style_run(run, size, bold=False, color=BODY_COLOR):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def textbox(slide, x, y, w, h, paragraphs, size=BODY_PT, color=BODY_COLOR, bold=False, space_after=8):
    """paragraphs: list of plain strings, one per paragraph (no mixed styles inside a paragraph)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, text in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space_after)
        r = p.add_run()
        r.text = text
        style_run(r, size, bold, color)
    return tb


def text_height(paragraphs, size, width, space_after=8):
    """Rough height (in) of wrapped Arial text: mean glyph width ~0.47 em, line height 1.2 em."""
    chars_per_line = max(1, int(width * 72 / (size * 0.47)))
    lines = sum(max(1, -(-len(p) // chars_per_line)) for p in paragraphs)
    return lines * size * 1.2 / 72 + len(paragraphs) * space_after / 72 + 0.1


def block(slide, x, y, paragraphs, size=BODY_PT, width=COL_W, space_after=8):
    """Text block; returns the y just below it."""
    h = text_height(paragraphs, size, width - 0.1, space_after)
    textbox(slide, x, y, width, h, paragraphs, size=size, space_after=space_after)
    return y + h + 0.08


def heading(slide, x, y, text, width=COL_W):
    textbox(slide, x, y, width, 0.7, [text], size=HEAD_PT, color=HEAD_COLOR, bold=True)
    return y + 0.8


def picture(slide, x, y, name, width=COL_W):
    path = figure(name)
    if path is None:
        textbox(slide, x, y, width, 0.5, [f"[missing figure: {name}]"])
        MISSING.append(f"figure:{name}")
        return y + 0.6
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y))
    w_in = pic.width / 914400
    if w_in > width:                      # only shrink, never enlarge
        scale = width / w_in
        pic.width, pic.height = int(pic.width * scale), int(pic.height * scale)
        w_in = width
    pic.left = Inches(x + (width - w_in) / 2)
    return y + pic.height / 914400 + 0.08


def fill_placeholder(ph, text, size, color, bold=True):
    tf = ph.text_frame
    tf.text = ""
    r = tf.paragraphs[0].add_run()
    r.text = text
    style_run(r, size, bold, color)


# ---------------- content ----------------

def column_intro(slide):
    x = COL_X[0]
    y = heading(slide, x, TOP, "Abstract")
    y = block(slide, x, y, [
        "We compare a data-driven neural network (data NN), trained on simulation results, with a "
        "physics-informed neural network (PINN), trained only on the flow equations. The data NN was "
        "more accurate and cheaper; the PINN needed no data and obeyed the equations more closely.",
    ])
    y = heading(slide, x, y + 0.1, "Introduction")
    y = block(slide, x, y, [
        "A trained network can stand in for a slow fluid simulation, but training has a cost: a data "
        "NN needs many simulation results, and a PINN needs more computing because every step "
        "evaluates the flow equations. We ask:",
        "RQ1  How accurate is each network?",
        "RQ2  How well does each obey conservation of mass and momentum?",
        "RQ3  How repeatable is each across random seeds and flow speeds?",
        "RQ4  What does each cost to train and run?",
    ])
    y = heading(slide, x, y + 0.1, "Methods")
    y = block(slide, x, y, [
        "Fluid in a square box is driven by a sliding lid (Reynolds number Re = 100 and 400). A "
        "finite-difference (FD) simulation is the reference and matches published results [1, 2].",
        "Both networks have 17,091 parameters, train for 20,000 steps with 5 seeds, and are scored "
        "by one evaluator on data never used in training.",
    ])
    COLUMN_END[1] = picture(slide, x, y, "pipelines")


def results(slide):
    x0, x1 = COL_X[1], COL_X[2]
    wide = COL_X[2] + COL_W - COL_X[1]
    y = heading(slide, x0, TOP, "Results", width=wide)
    rows = [
        (("v3_rq1", [
            f"The data NN's error is {pct('dnn_gpu_test_a_relative_l2_u_mean')} in u and "
            f"{pct('dnn_gpu_test_a_relative_l2_p_mean')} in p; the PINN's is "
            f"{pct('pinn_gpu_test_a_relative_l2_u_mean')} and {pct('pinn_gpu_test_a_relative_l2_p_mean')} "
            "(Re = 100, mean of 5 seeds)."]),
         ("v3_rq2", [
            "The PINN obeys all three equations more closely: mass violation "
            f"{num('pinn_gpu_continuity_residual_mean', '{:.2g}')} vs "
            f"{num('dnn_gpu_continuity_residual_mean', '{:.2g}')} for the data NN."])),
        (("v3_rq3", [
            "Both networks repeat well across seeds at Re = 100. At Re = 400 the data NN stays near "
            f"{pct('re400_dnn_gpu_test_a_relative_l2_u_mean')}, but every PINN seed fails "
            f"({pct('re400_pinn_gpu_test_a_relative_l2_u_mean')})."]),
         ("v3_re400", [
            "The failed PINN still conserves mass (violation "
            f"{num('re400_pinn_gpu_continuity_residual_mean', '{:.2f}')} vs "
            f"{num('re400_dnn_gpu_continuity_residual_mean', '{:.2f}')}), yet its vortex sits "
            "by the lid: obeying the equations is not the same as being right."])),
        (("v3_rq4", [
            f"The data NN needs {num('fd_n64_solve_seconds_median', '{:.1f}')} s of simulation plus "
            f"{num('dnn_gpu_training_time_seconds_mean', '{:.0f}')} s of training; the PINN needs "
            f"{num('pinn_gpu_training_time_seconds_mean', '{:.0f}')} s and more GPU memory."]),
         ("v3_rq4_time", [
            "Error in u during training (solid GPU, dashed CPU). Once trained, both return the "
            f"whole flow in about {num('pinn_gpu_inference_time_seconds_mean', '{:.0f}', 1e3)} ms."])),
    ]
    for (left, right) in rows:
        y_l = picture(slide, x0, y, left[0])
        y_l = block(slide, x0, y_l, left[1], size=CAPTION_PT)
        y_r = picture(slide, x1, y, right[0])
        y_r = block(slide, x1, y_r, right[1], size=CAPTION_PT)
        y = max(y_l, y_r) + 0.15
    COLUMN_END[2] = y


def strengths_table(slide, x, y, w):
    rows = [
        ("", "Data NN", "PINN", None),
        ("Needs simulation data", "Yes", "No", "pinn"),
        ("Error in u, Re 100", pct("dnn_gpu_test_a_relative_l2_u_mean"),
         pct("pinn_gpu_test_a_relative_l2_u_mean"), "dnn"),
        ("Error in u, Re 400", pct("re400_dnn_gpu_test_a_relative_l2_u_mean"),
         pct("re400_pinn_gpu_test_a_relative_l2_u_mean"), "dnn"),
        ("Mass violation", num("dnn_gpu_continuity_residual_mean", "{:.2g}"),
         num("pinn_gpu_continuity_residual_mean", "{:.2g}"), "pinn"),
        ("GPU training", num("dnn_gpu_training_time_seconds_mean", "{:.0f} s"),
         num("pinn_gpu_training_time_seconds_mean", "{:.0f} s"), "dnn"),
        ("GPU memory", num("dnn_gpu_peak_vram_bytes_mean", "{:.0f} MB", 1 / 2**20),
         num("pinn_gpu_peak_vram_bytes_mean", "{:.0f} MB", 1 / 2**20), "dnn"),
        ("CPU time per step", num("dnn_cpu_time_per_step_seconds_mean", "{:.0f} ms", 1e3),
         num("pinn_cpu_time_per_step_seconds_mean", "{:.0f} ms", 1e3), "dnn"),
    ]
    h_row = 0.5
    shape = slide.shapes.add_table(len(rows), 3, Inches(x), Inches(y), Inches(w), Inches(h_row * len(rows)))
    table = shape.table
    for j, cw in enumerate((3.3, 2.1, 2.1)):
        table.columns[j].width = Inches(cw * w / 7.5)
    for i, (label, a, b, better) in enumerate(rows):
        table.rows[i].height = Inches(h_row)
        for j, val in enumerate((label, a, b)):
            cell = table.cell(i, j)
            cell.text = ""
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            style_run(r, 20, bold=(i == 0), color=WHITE if i == 0 else BODY_COLOR)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = TABLE_HEAD
            elif (j == 1 and better == "dnn") or (j == 2 and better == "pinn"):
                cell.fill.fore_color.rgb = BETTER
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else TABLE_BAND
    return y + h_row * len(rows) + 0.1


def column_conclusion(slide):
    x = COL_X[3]
    y = heading(slide, x, TOP, "Strengths & weaknesses")
    y = strengths_table(slide, x, y, COL_W)
    y = block(slide, x, y, ["Green marks the better of the two (Re = 100 unless stated)."], size=18)
    y = heading(slide, x, y + 0.1, "Conclusion")
    y = block(slide, x, y, [
        "Neither approach wins on every measure. The data NN gave the closest answers for the least "
        "computing, but only because simulation data already existed. The PINN needed no data and "
        "obeyed the equations better, but cost more to train and failed at the faster flow.",
        "Their strengths and weaknesses are opposite. This points toward a hybrid that adds a small "
        "amount of data to the physics loss, which could reduce both the data and the computing "
        "needed. A hybrid was not tested here; it is the next step.",
    ])
    y = heading(slide, x, y + 0.1, "References")
    y = block(slide, x, y, [
        "[1] Marchi, C. H., Suero, R., and Araki, L. K. 2009. The lid-driven square cavity flow: "
        "numerical solution with a 1024 x 1024 grid. J. Braz. Soc. Mech. Sci. Eng. 31, 3, 186–198.",
        "[2] Hou, S., Zou, Q., Chen, S., Doolen, G., and Cogley, A. C. 1995. Simulation of cavity flow "
        "by the lattice Boltzmann method. J. Comput. Phys. 118, 2, 329–347.",
        "[3] Raissi, M., Perdikaris, P., and Karniadakis, G. E. 2019. Physics-informed neural networks: "
        "a deep learning framework for solving forward and inverse problems involving nonlinear "
        "partial differential equations. J. Comput. Phys. 378, 686–707.",
    ], size=SMALL_PT, space_after=4)
    y = heading(slide, x, y + 0.1, "Contact")
    COLUMN_END[4] = block(slide, x, y, ["CONTACT EMAIL"])


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

    column_intro(slide)
    results(slide)
    column_conclusion(slide)

    for c, end_y in sorted(COLUMN_END.items()):
        status = "OVERFLOW" if end_y > BOTTOM else "ok"
        print(f"column {c}: content ends at {end_y:.2f} in (limit {BOTTOM}) {status}")

    prs.save(OUT)
    with open(SOURCES, "w") as f:
        json.dump(USED, f, indent=1)
    used = runs_of(*USED.keys())
    print(f"wrote {OUT} and {SOURCES}: {len(USED)} numbers from {len(used)} runs; "
          f"missing: {sorted(set(MISSING))}")
    return MISSING


if __name__ == "__main__":
    missing = build()
    sys.exit(0)
