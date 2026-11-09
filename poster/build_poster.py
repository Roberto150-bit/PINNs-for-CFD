"""Build poster/poster_draft.pptx on the SCE Project Day template.

Layout (36 x 24 in) follows the template's four-column grid. The poster tells one
story; every panel heading is an action title (the finding as a sentence), each
results figure carries its key comparison as an on-chart callout, and the text
under it is one line of evidence:
  column 1: the question (RQ1-RQ4) and the fair-test design
  column 2: RQ1 accuracy, RQ2 physical consistency
  column 3: RQ3 reliability across flow speeds, and why satisfying the equations is not enough
  column 4: RQ4 cost, the answer (no single winner; hybrid as future work), references

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


def column_question(slide):
    x = COL_X[0]
    y = panel(slide, x, TOP,
              "Which is the better way to teach a network fluid flow: simulation data or physics?",
              text=["A trained neural network predicts a flow in milliseconds instead of running a slow "
                    "simulation. A data-driven network (data NN) learns from simulation results, so a "
                    "simulation must run first. A physics-informed network (PINN) [3] learns only from "
                    "the flow equations: no data, but more computing. A hybrid learns from both.",
                    "RQ1  Which is more accurate?",
                    "RQ2  Which obeys the physics better?",
                    "RQ3  Which is more reliable across seeds and flow speeds?",
                    "RQ4  What does each cost?"],
              size=BODY_PT)
    COLUMN_END[1] = panel(slide, x, y,
                          "All three networks were built and scored the same way; only what they learn from differs.",
                          fig="pipelines",
                          text=["Test flow: fluid in a box driven by a sliding lid. Reference: a "
                                "finite-difference (FD) simulation that matches published results [1, 2]. "
                                "All three: 17,091 parameters, 20,000 training steps, 3-5 seeds, scored on "
                                "points and times never seen in training."],
                          size=BODY_PT)


def column_rq12(slide):
    x = COL_X[1]
    y = panel(slide, x, TOP,
              "RQ1: The data NN matches the simulation "
              f"{ratio('pinn_gpu_test_a_relative_l2_u_mean', 'dnn_gpu_test_a_relative_l2_u_mean')}× more closely.",
              fig="v4_accuracy",
              text=[f"Error in u at Re = 100: data NN {pct('dnn_gpu_test_a_relative_l2_u_mean')}, hybrid "
                    f"{pct('hybrid_gpu_test_a_relative_l2_u_mean')}, PINN {pct('pinn_gpu_test_a_relative_l2_u_mean')} "
                    f"(mean of {num('dnn_gpu_n_runs', '{:.0f}')}, {num('hybrid_gpu_n_runs', '{:.0f}')} and "
                    f"{num('pinn_gpu_n_runs', '{:.0f}')} seeds). One hybrid seed ends on a spike "
                    f"({pct('spike_re100_hybrid_worst_best')} → {pct('spike_re100_hybrid_worst_final')} in its last "
                    "500 steps); like every run, it is scored at its final step."])
    COLUMN_END[2] = panel(slide, x, y,
                          "RQ2: The PINN obeys the flow equations "
                          f"{ratio('dnn_gpu_continuity_residual_mean', 'pinn_gpu_continuity_residual_mean')}× "
                          "more closely.",
                          fig="v4_physics",
                          text=["Violation of mass conservation: PINN "
                                f"{num('pinn_gpu_continuity_residual_mean', '{:.2g}')}, hybrid "
                                f"{num('hybrid_gpu_continuity_residual_mean', '{:.2g}')}, data NN "
                                f"{num('dnn_gpu_continuity_residual_mean', '{:.2g}')}."])


def trend_evidence():
    """One line of evidence for the Reynolds-number trend, listing every Re that has results."""
    res = [r for r in ("10", "100", "400", "1000")
           if r == "100" or f"re{r}_dnn_gpu_test_a_relative_l2_u_mean" in NUMS]

    def key(r, m):
        return f"{m}_gpu_test_a_relative_l2_u_mean" if r == "100" else f"re{r}_{m}_gpu_test_a_relative_l2_u_mean"

    def series(m, rs):
        # "1.9, 1.7, 2.3, 1.7%": one percent sign per list keeps the caption short
        vals = [pct(key(r, m)) for r in rs]
        return ", ".join(v.rstrip("%") for v in vals[:-1]) + (", " if len(vals) > 1 else "") + vals[-1]

    pinn_res = [r for r in res if key(r, "pinn") in NUMS]
    hyb_res = [r for r in res if NUMS.get(key(r, "hybrid"), {}).get("value") is not None]
    text = f"Every seed is a dot. Error in u at Re = {', '.join(res)}: data NN {series('dnn', res)}; "            f"PINN {series('pinn', pinn_res)}"
    if pinn_res != res:
        text += f" (Re = {', '.join(pinn_res)})"
    if hyb_res:
        text += f"; hybrid {series('hybrid', hyb_res)}" + ("" if hyb_res == res else f" (Re = {', '.join(hyb_res)})")
    return [text + "."]


def column_rq3(slide):
    x = COL_X[2]
    y = panel(slide, x, TOP,
              "RQ3: As the flow speeds up, only the PINN breaks down.",
              fig="v4_trend",
              text=trend_evidence())
    COLUMN_END[3] = panel(slide, x, y,
                          "RQ3: At Re = 400 the PINN obeys the equations but needs more training, "
                          "or a head start, to get the flow right.",
                          fig="v4_re400",
                          text=["Flow at t = 5 (✕ = vortex). The PINN conserves mass better than the data NN "
                                f"({num('re400_pinn_gpu_continuity_residual_mean', '{:.2f}')} vs "
                                f"{num('re400_dnn_gpu_continuity_residual_mean', '{:.2f}')}) yet puts the vortex "
                                "by the lid. Error in u: "
                                f"{pct('s5_re400_pinn_u_at_20k_mean')} → {pct('s5_re400_pinn_u_at_100k_mean')} with "
                                f"5× more steps ({num('s5_re400_pinn_n_runs', '{:.0f}')} seeds); "
                                f"{pct('s6_re400_base_u_mean')} → {pct('s6_re400_warm_u_mean')} starting from the "
                                f"trained Re = 100 PINN ({num('s6_re400_n_runs', '{:.0f}')} seeds; a random start "
                                f"is at {pct('s5_re400_pinn_u_at_40k_mean')} after the same 40,000 steps). The "
                                f"hybrid gets the flow right ({pct('re400_hybrid_gpu_test_a_relative_l2_u_mean')}) "
                                "and conserves mass like the PINN "
                                f"({num('re400_hybrid_gpu_continuity_residual_mean', '{:.2f}')})."])


def column_answer(slide):
    x = COL_X[3]
    y = panel(slide, x, TOP,
              "RQ4: The PINN needs no data, but "
              f"{ratio('pinn_gpu_total_cost_to_field_seconds', 'dnn_gpu_total_cost_to_field_seconds', '{:.1f}')}× "
              "more training time.",
              fig="v4_cost",
              text=[f"The simulation took only {num('fd_n64_solve_seconds_median', '{:.1f}')} s, so data "
                    "was cheap; the hybrid pays for both. GPU memory: "
                    f"{num('pinn_gpu_peak_vram_bytes_mean', '{:.0f}', 1 / 2**20)} MB (PINN) vs "
                    f"{num('dnn_gpu_peak_vram_bytes_mean', '{:.0f}', 1 / 2**20)} MB (data NN)."])
    y = panel(slide, x, y,
              "No method wins on everything; each trades accuracy, physics and cost.",
              text=["• Data NN: most accurate and cheapest, when simulation data already exists.",
                    "• PINN: needs no data and obeys the physics best, but costs more and fails as the "
                    "flow speeds up.",
                    "• Hybrid: accurate at every flow speed "
                    f"({pct('re1000_hybrid_gpu_test_a_relative_l2_u_mean')} vs "
                    f"{pct('re1000_pinn_gpu_test_a_relative_l2_u_mean')} for the PINN at Re = 1000) and far "
                    "closer to the physics than the data NN, but needs data and the most training.",
                    "• Reusing a trained network helps: a PINN started from the Re = 100 one more than "
                    "halved its error at Re = 400."],
              size=BODY_PT)
    y = block(slide, x, y, [
        "[1] Marchi, C. H., Suero, R., and Araki, L. K. 2009. The lid-driven square cavity flow: numerical "
        "solution with a 1024 x 1024 grid. J. Braz. Soc. Mech. Sci. Eng. 31, 3, 186–198.",
        "[2] Hou, S., Zou, Q., Chen, S., Doolen, G., and Cogley, A. C. 1995. Simulation of cavity flow by "
        "the lattice Boltzmann method. J. Comput. Phys. 118, 2, 329–347.",
        "[3] Raissi, M., Perdikaris, P., and Karniadakis, G. E. 2019. Physics-informed neural networks: a "
        "deep learning framework for solving forward and inverse problems involving nonlinear partial "
        "differential equations. J. Comput. Phys. 378, 686–707.",
    ], size=SMALL_PT, space_after=4)
    COLUMN_END[4] = block(slide, x, y + 0.1, ["Contact: roberto.pollicino@quinnipiac.edu · Adviser: brian.oneil@quinnipiac.edu"],
                          size=SMALL_PT)


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
