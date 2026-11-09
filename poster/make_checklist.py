"""Write poster/VERIFY_CHECKLIST.md: every number placed on the poster, where it comes from,
and how to check it by hand, plus the items to check by eye.

    python poster/make_checklist.py      (after build_poster.py)

Reads poster/poster_draft_sources.json (written by build_poster.py), so the list always
matches the poster that was last built.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SOURCES = os.path.join(HERE, "poster_draft_sources.json")
OUT = os.path.join(HERE, "VERIFY_CHECKLIST.md")

EXP = {"10": "experiment_003_re10", "100": "experiment_001_re100", "400": "experiment_002_re400",
       "1000": "experiment_004_re1000"}
METHOD = {"dnn": "data_driven_nn", "pinn": "pinn", "hybrid": "hybrid", "fd": "finite_difference"}


def results_file(key):
    m = re.match(r"(?:ctrl_)?re(\d+)_", key)
    if key.startswith("ctrl_re"):
        re_ = m.group(1)
        main = "experiment_results.csv" if re_ == "100" else f"experiment_results_re{re_}.csv"
        return f"experiments/results/control_re{re_}.csv and {main} (same seeds)"
    if key.startswith("s6_"):
        return "experiments/results/continuation_re400.csv vs experiment_results_re400.csv (same seed)"
    if key.startswith("s5_"):
        return "eval_history.csv of each listed run (rel_l2_u at step 20000 / 40000 / 100000)"
    if key.startswith("spike_"):
        return "eval_history.csv of each listed run (column rel_l2_u: final row vs smallest value)"
    if key.startswith("fd_"):
        return "experiments/results/poster_numbers.json (FD timing / validation; see description)"
    if m:
        return f"experiments/results/experiment_results_re{m.group(1)}.csv"
    return "experiments/results/experiment_results.csv (Re 100)"


def run_folder(run_id):
    m = re.match(r"(dnn|pinn|hybrid|fd)_re(\d+)_", run_id)
    if not m or m.group(2) not in EXP:
        return run_id
    sub = ("extended/" if "_steps100k" in run_id else "continuation/" if "_from_re" in run_id
           else "control/" if "_dataonly" in run_id else "")
    return f"experiments/{EXP[m.group(2)]}/{sub}{METHOD[m.group(1)]}/{run_id}/"


def compact(runs):
    """Group run ids like pinn_re100_gpu_s1..s5 into one line."""
    groups = {}
    for r in runs:
        m = re.match(r"(.*_s)(\d+)(.*)$", r)
        if m:
            groups.setdefault((m.group(1), m.group(3)), []).append(int(m.group(2)))
        else:
            groups.setdefault((r, ""), [])
    out = []
    for (head, tail), seeds in groups.items():
        if seeds:
            s = sorted(seeds)
            span = f"{s[0]}-{s[-1]}" if s == list(range(s[0], s[-1] + 1)) and len(s) > 1 else ",".join(map(str, s))
            folder = run_folder(f"{head}{s[0]}{tail}").replace(f"{head}{s[0]}{tail}", f"{head}{{{span}}}{tail}")
            out.append(f"`{folder}`")
        else:
            out.append(f"`{run_folder(head)}`")
    return "; ".join(out)


def main():
    src = json.load(open(SOURCES, encoding="utf-8"))
    lines = [
        "# Poster verification checklist",
        "",
        "Every number on `poster/poster_draft.pdf`, generated from `poster/poster_draft_sources.json`",
        "(written by `poster/build_poster.py`). Errors are test-A relative L2 at the final training",
        "step unless the description says otherwise; percentages are that value × 100. Ratios on the",
        "poster (e.g. \"5×\", \"5.5×\", \"8×\") are one row divided by another; those rows show the raw value.",
        "",
        "How to check one number: open the results file, find the listed run ids, and recompute the",
        "mean (or read the value). Each run folder holds `metrics.json` (the scored values),",
        "`eval_history.csv` (error every 500 steps) and `config.json` (settings).",
        "",
        "## 1. Check by eye",
        "",
        "- [ ] Title: \"Evaluating Physics-Informed Learning for Fluid Dynamics\"",
        "- [ ] Name and year: \"Roberto Pollicino (CS '26)\"",
        "- [ ] Adviser line: \"Adviser: Brian C. O'Neill, PhD (Computer Science)\"",
        "- [ ] Contact line: roberto.pollicino@quinnipiac.edu and the adviser's address. **The poster",
        "      spells the name O'Neill (two L's) but the email as brian.oneil@ (one L), as given.",
        "      Confirm which is right.**",
        "- [ ] References [1]-[3]: authors, year, journal, pages (Marchi 2009, Hou 1995, Raissi 2019)",
        "- [ ] Nothing cut off at the poster edges; every figure readable at arm's length",
        "",
        "## 2. Numbers on the poster",
        "",
        "| # | Printed as | Value | What it is | Results file | Run folders |",
        "|---|---|---|---|---|---|",
    ]
    for i, (key, e) in enumerate(src.items(), 1):
        printed = ", ".join(sorted(set(e.get("printed_as", []))))
        val = e["value"]
        val = f"{val:.6g}" if isinstance(val, (int, float)) else str(val)
        desc = e.get("description", "").replace("|", "/")
        lines.append(f"| {i} | {printed} | {val} | {desc} (`{key}`) | {results_file(key)} | "
                     f"{compact(e.get('runs', []))} |")
    lines += [
        "",
        "## 3. Figures",
        "",
        "Every poster figure has a `.values.json` next to its PNG in `experiments/analysis/plots/`",
        "(`v4_*`, `pipelines`). `python experiments/analysis/make_figures.py` re-checks each plotted",
        "value against the results files (gate G15) and prints PASS / FAIL per figure.",
        "",
    ]
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.relpath(OUT, REPO)}: {len(src)} numbers")


if __name__ == "__main__":
    main()
