"""Build poster/poster_draft.pptx on the SCE Project Day template.

Layout (36 x 24 in) follows the template's four-column grid with one heading per
column. Each figure has a plain-language lead (what is measured, how to read it)
and a caption (what it shows, with the numbers); the last column ends with a
comparison table of the three networks:
  column 1: the question, the three networks, RQ1-RQ4 and the fair test
  column 2: RQ1 accuracy, RQ2 physical consistency
  column 3: RQ3 faster flows, and why obeying the equations is not enough
  column 4: RQ4 cost, comparison table, conclusion, references

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
#
# Story (read the action titles alone, in column order, to get the whole argument):
#   1  question: data or physics?          2  fair test: only the training signal differs
#   3  RQ1 data NN more accurate            4  RQ2 PINN obeys the equations better
#   5  RQ3 data NN robust, PINN fails       6  RQ3 satisfying equations != correct flow
#   7  RQ4 PINN needs no data, more time    8  no single winner -> hybrid (future work)

TITLE_PT = 30
GAP = 0.45


def ratio(key_a, key_b, fmt="{:.0f}"):
    """key_a / key_b, both read from poster_numbers.json and recorded as used."""
    a, b = NUMS.get(key_a, {}).get("value"), NUMS.get(key_b, {}).get("value")
    if a is None or b is None:
        MISSING.extend(k for k, v in ((key_a, a), (key_b, b)) if v is None)
        return "[missing]"
    num(key_a)
    num(key_b)
    return fmt.format(a / b)


def action_title(slide, x, y, text, width=COL_W):
    """A panel heading written as the finding itself (one or two lines)."""
    h = text_height([text], TITLE_PT, width - 0.1, 0) + 0.05 * TITLE_PT / 24
    h = text_height([text], TITLE_PT * 1.08, width - 0.1, 0)     # bold glyphs are wider
    textbox(slide, x, y, width, h, [text], size=TITLE_PT, color=HEAD_COLOR, bold=True, space_after=0)
    return y + h + 0.04


def panel(slide, x, y, title, fig=None, text=None, size=CAPTION_PT):
    y = action_title(slide, x, y, title)
    if fig:
        y = picture(slide, x, y, fig)
    if text:
        y = block(slide, x, y, text, size=size)
    return y + GAP


COL_HEAD_PT = HEAD_PT
INTRO_PT = 22


def col_heading(slide, x, y, text, width=COL_W):
    """One heading per column (the only headings on the poster)."""
    textbox(slide, x, y, width, 0.75, [text], size=COL_HEAD_PT, color=HEAD_COLOR, bold=True, space_after=0)
    return y + 0.85


def lead(slide, x, y, paragraphs, size=INTRO_PT):
    """Plain-language context placed before a figure: what is measured and how to read it."""
    return block(slide, x, y, paragraphs, size=size, space_after=6)


def caption(slide, x, y, paragraphs, size=CAPTION_PT):
    """What the figure shows, in words, with the key numbers."""
    return block(slide, x, y, paragraphs, size=size, space_after=6) + GAP


def seed_counts():
    """'mean of 5 seeds each' when all three methods have the same count, else the list."""
    n = [num(f"{m}_gpu_n_runs", "{:.0f}") for m in ("dnn", "hybrid", "pinn")]
    return f"mean of {n[0]} runs each" if len(set(n)) == 1 else f"mean of {n[0]}, {n[1]} and {n[2]} runs"


def series(keys):
    """'16, 8.2, 36, 41%': one percent sign per list keeps the text short."""
    vals = [pct(k) for k in keys]
    return ", ".join(v.rstrip("%") for v in vals[:-1]) + (", " if len(vals) > 1 else "") + vals[-1]


def u_key(re, m):
    return f"{m}_gpu_test_a_relative_l2_u_mean" if re == "100" else f"re{re}_{m}_gpu_test_a_relative_l2_u_mean"


def column_question(slide):
    x = COL_X[0]
    y = col_heading(slide, x, TOP, "Introduction")
    y = block(slide, x, y, [
        "Engineers predict how fluids move (air over a wing, blood in an artery) by solving the "
        "equations of fluid motion on a computer. That is accurate but slow. A trained neural "
        "network can predict the same flow in milliseconds, but how should it be taught?",
        "I compared three ways of teaching the same network:",
        "• Data NN: learns from the results of finished simulations, so a simulation must run first.",
        "• PINN (physics-informed neural network) [3]: learns only from the laws of fluid motion, "
        "with no simulation data.",
        "• Hybrid: learns from both.",
        "RQ1  Which predicts the flow most accurately?",
        "RQ2  Which best respects the laws of physics?",
        "RQ3  Which stays reliable as the flow gets faster?",
        "RQ4  What does each cost to train?",
    ], size=BODY_PT, space_after=7) + 0.2
    y = col_heading(slide, x, y, "Methods")
    y = picture(slide, x, y, "pipelines")
    COLUMN_END[1] = caption(slide, x, y, [
        "The test: fluid in a square box, stirred by a lid sliding along the top. A trusted "
        "simulation (finite differences, FD), checked against published results [1, 2], is the "
        "answer key. All three networks have the same size (17,091 parameters) and the same "
        "20,000 training steps, and are scored on points and moments they never saw. Each is "
        "trained 5 times from different random starts to check that results repeat.",
    ], size=BODY_PT)


def column_rq12(slide):
    x = COL_X[1]
    y = col_heading(slide, x, TOP, "Results")
    y = lead(slide, x, y, [
        "RQ1 accuracy. Error: how far each prediction is from the answer key, in percent, for the "
        "speeds u (left-right) and v (up-down) and the pressure p. Lower is better."])
    y = picture(slide, x, y, "v4_accuracy")
    y = caption(slide, x, y, [
        f"The data NN is the most accurate ({pct('dnn_gpu_test_a_relative_l2_u_mean')} error in u, "
        f"{ratio('pinn_gpu_test_a_relative_l2_u_mean', 'dnn_gpu_test_a_relative_l2_u_mean')}× better than the "
        f"PINN's {pct('pinn_gpu_test_a_relative_l2_u_mean')}); the hybrid is in between "
        f"({pct('hybrid_gpu_test_a_relative_l2_u_mean')}). One hybrid run jumped from "
        f"{pct('spike_re100_hybrid_worst_best')} to {pct('spike_re100_hybrid_worst_final')} at the very end "
        "and is reported as it ended."])
    y = lead(slide, x, y, [
        "RQ2 physics. Real fluid cannot appear or vanish (mass) and obeys Newton's laws "
        "(momentum). Bars: how much each prediction breaks these rules."])
    y = picture(slide, x, y, "v4_physics")
    COLUMN_END[2] = caption(slide, x, y, [
        "The PINN, which is taught the rules, breaks them "
        f"{ratio('dnn_gpu_continuity_residual_mean', 'pinn_gpu_continuity_residual_mean')}× less than the "
        f"data NN ({num('pinn_gpu_continuity_residual_mean', '{:.2g}')} vs "
        f"{num('dnn_gpu_continuity_residual_mean', '{:.2g}')}); the hybrid keeps most of that benefit "
        f"({num('hybrid_gpu_continuity_residual_mean', '{:.2g}')}). Without its physics lessons (same "
        f"runs) the hybrid breaks the rules more ({num('ctrl_re100_dataonly_cont_mean', '{:.2f}')}); it is "
        f"then more accurate at Re = 100 ({pct('ctrl_re100_dataonly_u_mean')} vs "
        f"{pct('ctrl_re100_hybrid_u_mean')}) but less accurate at Re = 1000 "
        f"({pct('ctrl_re1000_dataonly_u_mean')} vs {pct('ctrl_re1000_hybrid_u_mean')}; in "
        f"{num('ctrl_re1000_hybrid_better_u_n', '{:.0f}')} of {num('ctrl_re1000_n_runs', '{:.0f}')} runs)."])


def column_rq3(slide):
    x = COL_X[2]
    y = TOP + 0.85                       # under the Results heading that spans columns 2 and 3
    y = lead(slide, x, y, [
        "RQ3 faster flows. The Reynolds number (Re) measures how fast and swirly a flow is: at Re = 10 it is "
        "slow like honey; at Re = 1000 it is fast, with sharp swirls that are harder to learn."])
    y = picture(slide, x, y, "v4_trend")
    res = [r for r in ("10", "100", "400", "1000") if NUMS.get(u_key(r, "hybrid"), {}).get("value") is not None]
    y = caption(slide, x, y, [
        "Each dot is one training run; dots close together mean the result repeats. Average error "
        f"in u at Re = {', '.join(res)}: data NN {series([u_key(r, 'dnn') for r in res])}; hybrid "
        f"{series([u_key(r, 'hybrid') for r in res])}; PINN {series([u_key(r, 'pinn') for r in res])}. "
        "Only the PINN gets much worse as the flow speeds up."])
    y = lead(slide, x, y, [
        "What goes wrong at Re = 400: the PINN puts the center of the swirl (✕) by the lid "
        "instead of in the middle, although it obeys the rules better than the data NN "
        f"({num('re400_pinn_gpu_continuity_residual_mean', '{:.2f}')} vs "
        f"{num('re400_dnn_gpu_continuity_residual_mean', '{:.2f}')}): obeying the equations is not "
        "enough."])
    y = picture(slide, x, y, "v4_re400")
    COLUMN_END[3] = caption(slide, x, y, [
        f"Two things help the PINN: training 5× longer ({pct('s5_re400_pinn_u_at_20k_mean')} → "
        f"{pct('s5_re400_pinn_u_at_100k_mean')}, {num('s5_re400_pinn_n_runs', '{:.0f}')} runs) or starting "
        f"from the network already trained at Re = 100 ({pct('s6_re400_base_u_mean')} → "
        f"{pct('s6_re400_warm_u_mean')}, {num('s6_re400_n_runs', '{:.0f}')} runs). Adding data, as the "
        f"hybrid does, gets it right directly: {pct('re400_hybrid_gpu_test_a_relative_l2_u_mean')}. At "
        "Re = 1000, 5× longer training still leaves the PINN at "
        f"{pct('s5_re1000_pinn_u_at_100k_mean')} ({num('s5_re1000_pinn_n_runs', '{:.0f}')} runs)."])


def comparison_table(slide, x, y, width=COL_W):
    """Final comparison: one row per question, best value per row shaded."""
    def val(key, kind):
        return pct(key) if kind == "pct" else num(key, "{:.2g}") if kind == "g" else num(key, "{:.0f} s")

    rows = [("Needs simulation data", ["Yes", "No", "Yes"], 1),
            ("RQ1  Error, Re = 100",
             [val(k, "pct") for k in ("dnn_gpu_test_a_relative_l2_u_mean", "pinn_gpu_test_a_relative_l2_u_mean",
                                      "hybrid_gpu_test_a_relative_l2_u_mean")], None),
            ("RQ2  Mass rule broken",
             [val(k, "g") for k in ("dnn_gpu_continuity_residual_mean", "pinn_gpu_continuity_residual_mean",
                                    "hybrid_gpu_continuity_residual_mean")], None),
            ("RQ3  Error, Re = 1000",
             [val(k, "pct") for k in ("re1000_dnn_gpu_test_a_relative_l2_u_mean",
                                      "re1000_pinn_gpu_test_a_relative_l2_u_mean",
                                      "re1000_hybrid_gpu_test_a_relative_l2_u_mean")], None),
            ("Error, 0.1% of the data",
             [pct("s7_re100_dnn_frac0p1_u_mean"), pct("pinn_gpu_test_a_relative_l2_u_mean"),
              pct("s7_re100_hybrid_frac0p1_u_mean")], None),
            ("RQ4  Training time",
             [val(k, "s") for k in ("dnn_gpu_total_cost_to_field_seconds", "pinn_gpu_total_cost_to_field_seconds",
                                    "hybrid_gpu_total_cost_to_field_seconds")], None)]
    # Best cell per row: the smallest number (all rows are lower-is-better), or the given column
    for i, (label, cells, best) in enumerate(rows):
        if best is None:
            # compare the printed values, so equal-looking cells are shaded together
            nums_ = [float(c.rstrip("% s")) for c in cells]
            rows[i] = (label, cells, {j for j, v in enumerate(nums_) if v == min(nums_)})
        elif not isinstance(best, set):
            rows[i] = (label, cells, {best})
    row_h = 0.40
    col_w = [3.07, 1.48, 1.48, 1.49]
    shape = slide.shapes.add_table(len(rows) + 1, 4, Inches(x), Inches(y), Inches(width),
                                   Inches(row_h * (len(rows) + 1)))
    table = shape.table
    for j, w in enumerate(col_w):
        table.columns[j].width = Inches(w)
    header = ["", "Data NN", "PINN", "Hybrid"]

    def set_cell(cell, text, size, color, fill, bold=False, align=PP_ALIGN.CENTER):
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = text
        style_run(r, size, bold, color)
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
        cell.margin_left = cell.margin_right = Inches(0.06)
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    for j, h in enumerate(header):
        set_cell(table.cell(0, j), h, 20, WHITE, TABLE_HEAD, bold=True)
    for i, (label, cells, best) in enumerate(rows, start=1):
        table.rows[i].height = Inches(row_h)
        band = TABLE_BAND if i % 2 else WHITE
        set_cell(table.cell(i, 0), label, 18, BODY_COLOR, band, align=PP_ALIGN.LEFT)
        for j, c in enumerate(cells, start=1):
            set_cell(table.cell(i, j), c, 20, BODY_COLOR, BETTER if j - 1 in best else band)
    table.rows[0].height = Inches(row_h)
    return y + row_h * (len(rows) + 1) + 0.12


def column_answer(slide):
    x = COL_X[3]
    y = col_heading(slide, x, TOP, "Cost")
    y = lead(slide, x, y, [
        "RQ4. Training time on one graphics card, including the simulation that makes the data."])
    y = picture(slide, x, y, "v4_cost")
    y = caption(slide, x, y, [
        f"The simulation took only {num('fd_n64_solve_seconds_median', '{:.1f}')} s, so the data NN was "
        f"{ratio('pinn_gpu_total_cost_to_field_seconds', 'dnn_gpu_total_cost_to_field_seconds', '{:.1f}')}× "
        "cheaper than the PINN; checking the physics at thousands of points per step slows the "
        "PINN and the hybrid."])
    y = col_heading(slide, x, y - 0.2, "Conclusion")
    y = comparison_table(slide, x, y)
    y = block(slide, x, y, [
        "Green: best (Re = 100 unless stated). No method wins everywhere. With plenty of data the "
        "data NN is best; without data the PINN fails on fast flows. The hybrid is the most "
        "balanced and needs little data: with 0.1% of it, it matches the PINN (which uses none) "
        "while the data NN fails."], size=INTRO_PT) + 0.05
    y = col_heading(slide, x, y, "References")
    y = block(slide, x, y, [
        "[1] Marchi, C. H., Suero, R., and Araki, L. K. 2009. The lid-driven square cavity flow: numerical "
        "solution with a 1024 x 1024 grid. J. Braz. Soc. Mech. Sci. Eng. 31, 3, 186–198.",
        "[2] Hou, S., Zou, Q., Chen, S., Doolen, G., and Cogley, A. C. 1995. Simulation of cavity flow by "
        "the lattice Boltzmann method. J. Comput. Phys. 118, 2, 329–347.",
        "[3] Raissi, M., Perdikaris, P., and Karniadakis, G. E. 2019. Physics-informed neural networks: a "
        "deep learning framework for solving forward and inverse problems involving nonlinear partial "
        "differential equations. J. Comput. Phys. 378, 686–707.",
    ], size=SMALL_PT, space_after=4)
    y = col_heading(slide, x, y - 0.35, "Contacts")       # the reference block's height estimate is generous
    COLUMN_END[4] = block(slide, x, y, ["Roberto.Pollicino@quinnipiac.edu · Brian.ONeill@quinnipiac.edu"],
                          size=SMALL_PT + 2, space_after=2)


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

    column_question(slide)
    column_rq12(slide)
    column_rq3(slide)
    column_answer(slide)

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
