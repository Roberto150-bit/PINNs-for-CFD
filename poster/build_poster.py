"""Build poster/poster_draft.pptx on the SCE Project Day template.

Layout (36 x 24 in): one question across the top, then three columns:
  1. why it matters and how it was tested,
  2. what was found at Re = 100 (accuracy, physics, cost), one figure each,
  3. where it breaks (Re = 400), whether more training helps, takeaways, references.

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
from pptx.enum.shapes import MSO_SHAPE
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
METHOD_COLOR = {"fd": RGBColor(0x2A, 0x78, 0xD6), "dnn": RGBColor(0xEB, 0x68, 0x34),
                "pinn": RGBColor(0x1B, 0xAF, 0x7A)}
HEAD_PT, FIND_PT, BODY_PT, SMALL_PT = 40, 30, 25, 15

LEFT, RIGHT = 2.32, 33.67          # template content margins (inches)
GAP = 0.9
COL_W = (RIGHT - LEFT - 2 * GAP) / 3
COL_X = [LEFT + i * (COL_W + GAP) for i in range(3)]
QUESTION_Y = 4.95
TOP = 7.15
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

def style_run(run, size, bold=False, color=BODY_COLOR, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def textbox(slide, x, y, w, h, paragraphs, size=BODY_PT, color=BODY_COLOR, anchor=MSO_ANCHOR.TOP,
            align=PP_ALIGN.LEFT, space_after=6):
    """paragraphs: list of str or (str, {"bold": .., "size": ..}) or list of such segments."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        segments = para if isinstance(para, list) else [para]
        for seg in segments:
            text, opts = (seg, {}) if isinstance(seg, str) else seg
            r = p.add_run()
            r.text = text
            style_run(r, opts.get("size", size), opts.get("bold", False), opts.get("color", color),
                      opts.get("italic", False))
    return tb


def text_height(paragraphs, size, width, space_after=6, glyph=0.46):
    """Rough height (in) of wrapped Arial text: mean glyph width ~0.46 em, line height 1.2 em."""
    chars_per_line = max(1, int(width * 72 / (size * glyph)))
    lines = 0
    for para in paragraphs:
        segments = para if isinstance(para, list) else [para]
        n = sum(len(seg if isinstance(seg, str) else seg[0]) for seg in segments)
        lines += max(1, -(-n // chars_per_line))
    return lines * size * 1.2 / 72 + len(paragraphs) * space_after / 72 + 0.1


def block(slide, col, y, paragraphs, size=BODY_PT, space_after=6, glyph=0.46):
    """Text block in a column; returns the y just below it."""
    h = text_height(paragraphs, size, COL_W - 0.1, space_after, glyph)
    textbox(slide, COL_X[col], y, COL_W, h, paragraphs, size=size, space_after=space_after)
    return y + h + 0.08


def heading(slide, col, y, text):
    textbox(slide, COL_X[col], y, COL_W, 0.66, [(text, {"bold": True})], size=HEAD_PT, color=HEAD_COLOR)
    return y + 0.74


def finding(slide, col, y, number, text):
    """A numbered finding: the claim in one bold line."""
    para = [[(f"{number}  ", {"bold": True, "color": HEAD_COLOR}), (text, {"bold": True})]]
    return block(slide, col, y, para, size=FIND_PT, space_after=2, glyph=0.47)


def picture(slide, col, y, name, max_width=None):
    path = figure(name)
    if path is None:
        textbox(slide, COL_X[col], y, COL_W, 0.5, [f"[missing figure: {name}]"])
        MISSING.append(f"figure:{name}")
        return y + 0.6
    pic = slide.shapes.add_picture(path, Inches(COL_X[col]), Inches(y))
    w_in = pic.width / 914400
    limit = min(COL_W, max_width or COL_W)
    if w_in > limit:                      # never enlarge text; only shrink if wider than allowed
        scale = limit / w_in
        pic.width, pic.height = int(pic.width * scale), int(pic.height * scale)
        w_in = limit
    pic.left = Inches(COL_X[col] + (COL_W - w_in) / 2)
    return y + pic.height / 914400 + 0.12


def fill_placeholder(ph, text, size, color, bold=True):
    tf = ph.text_frame
    tf.text = ""
    r = tf.paragraphs[0].add_run()
    r.text = text
    style_run(r, size, bold, color)


# ---------------- content ----------------

def question(slide):
    textbox(slide, LEFT, QUESTION_Y, RIGHT - LEFT, 1.25,
            [[("Does training a neural network on the laws of physics make its answers more "
               "trustworthy, and is it worth the extra computing cost?", {"bold": True})]],
            size=46, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(LEFT), Inches(TOP - 0.42),
                                  Inches(RIGHT - LEFT), Inches(0.06))
    line.fill.solid()
    line.fill.fore_color.rgb = HEAD_COLOR
    line.line.fill.background()


def column_why_how(slide):
    c = 0
    y = heading(slide, c, TOP, "Why it matters")
    y = block(slide, c, y, [
        "Fluid simulations are accurate but slow. A trained neural network answers in milliseconds, "
        "but nothing forces its answer to obey physics.",
        [("A physics-informed neural network (PINN) ", {"bold": True}),
         "learns from the flow equations themselves instead of from simulation data."],
    ])
    y = heading(slide, c, y + 0.15, "How we tested it")
    y = block(slide, c, y, ["A lid slides across a box of fluid and sets it spinning "
                            "(lid-driven cavity, Reynolds number Re = 100)."])
    y = picture(slide, c, y, "schematic")
    y = block(slide, c, y, ["Three ways to get the same flow, scored by one evaluator:"])
    y = picture(slide, c, y, "pipelines")
    y = block(slide, c, y, [
        "• Same network size for both (17,091 parameters), 5 training seeds each.",
        "• Scored on points and whole time steps never used in training.",
        "• Simulation checked against two published studies: vortex center within "
        f"{num('fd_val_n64_vortex_dist_marchi', '{:.3f}')} of Marchi et al. (less than one grid cell).",
    ], size=23)
    COLUMN_END[1] = y


def column_findings(slide):
    c = 1
    y = heading(slide, c, TOP, "What we found (Re = 100)")
    y = finding(slide, c, y, "1", "Learning from data gives the closest answer.")
    y = block(slide, c, y, [f"Error in u: {pct('dnn_gpu_test_a_relative_l2_u_mean')} for the data NN, "
                            f"{pct('pinn_gpu_test_a_relative_l2_u_mean')} for the PINN (mean of 5 seeds)."],
              size=23)
    y = picture(slide, c, y, "v2_accuracy")
    y = finding(slide, c, y + 0.1, "2", "The PINN's answers obey physics better.")
    y = block(slide, c, y, ["How far each flow breaks the equations of motion (lower is better)."], size=23)
    y = picture(slide, c, y, "v2_physics")
    y = finding(slide, c, y + 0.1, "3", "Physics costs computing time.")
    y = block(slide, c, y, ["Time to a usable flow (data NN time includes making its simulation "
                            "data). Once trained, either network returns the whole flow in about "
                            f"{num('pinn_gpu_inference_time_seconds_mean', '{:.0f}', 1e3)} ms."], size=23)
    COLUMN_END[2] = picture(slide, c, y, "v2_cost")


def column_limits(slide):
    c = 2
    y = heading(slide, c, TOP, "Where it breaks")
    y = finding(slide, c, y, "4", "A small physics error is not a correct answer.")
    y = block(slide, c, y, [
        "At a faster flow (Re = 400) the PINN broke conservation of mass less than the data NN "
        f"({num('re400_pinn_gpu_continuity_residual_mean', '{:.2f}')} vs "
        f"{num('re400_dnn_gpu_continuity_residual_mean', '{:.2f}')}), yet its vortex (✕) sat by the lid. "
        f"Error in u: {pct('re400_pinn_gpu_test_a_relative_l2_u_mean')} (PINN, all 5 seeds) vs "
        f"{pct('re400_dnn_gpu_test_a_relative_l2_u_mean')} (data NN).",
    ], size=23)
    y = picture(slide, c, y, "v2_re400")
    y = finding(slide, c, y + 0.1, "5", "Does more training help?")
    if figure("v2_training_length") and NUMS.get("s5_re100_pinn_u_at_100k_mean"):
        y = block(slide, c, y, [
            "Same runs trained 5× longer (100,000 steps). Error in u at 20,000 → 100,000 steps: "
            f"PINN {pct('s5_re100_pinn_u_at_20k_mean')} → {pct('s5_re100_pinn_u_at_100k_mean')}, "
            f"data NN {pct('s5_re100_dnn_u_at_20k_mean')} → {pct('s5_re100_dnn_u_at_100k_mean')} (Re = 100)."
        ], size=23)
        y = picture(slide, c, y, "v2_training_length")
    else:
        y = block(slide, c, y, ["Runs trained 5× longer (100,000 steps) are in progress."], size=23)
    y = heading(slide, c, y + 0.1, "What the numbers show")
    y = block(slide, c, y, [
        "• Training on physics made the network obey physics, not match the simulation better.",
        "• It cost more: about "
        f"{num('pinn_gpu_training_time_seconds_mean', '{:.0f}')} s vs "
        f"{num('dnn_gpu_training_time_seconds_mean', '{:.0f}')} s of GPU training.",
        "• A network can satisfy the equations and still be wrong, so it must be checked "
        "against a trusted reference.",
    ], size=23)
    y = block(slide, c, y + 0.05, [
        "[1] Marchi, C. H., Suero, R., and Araki, L. K. 2009. The lid-driven square cavity flow: numerical "
        "solution with a 1024 x 1024 grid. J. Braz. Soc. Mech. Sci. Eng. 31, 3, 186–198.",
        "[2] Hou, S., Zou, Q., Chen, S., Doolen, G., and Cogley, A. C. 1995. Simulation of cavity flow "
        "by the lattice Boltzmann method. J. Comput. Phys. 118, 2, 329–347.",
        "[3] Raissi, M., Perdikaris, P., and Karniadakis, G. E. 2019. Physics-informed neural networks: "
        "a deep learning framework for solving forward and inverse problems involving nonlinear partial "
        "differential equations. J. Comput. Phys. 378, 686–707.",
    ], size=SMALL_PT, space_after=2)
    COLUMN_END[3] = block(slide, c, y, [[("Contact: ", {"bold": True, "color": HEAD_COLOR}),
                                           "CONTACT EMAIL"]], size=22)


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

    question(slide)
    column_why_how(slide)
    column_findings(slide)
    column_limits(slide)

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
