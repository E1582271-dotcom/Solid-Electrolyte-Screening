"""
Project 1 -- visualize the adversarial audit + the family-coverage gap it exposed.

Two figures the pipeline never rendered:
  1. The manual audit of 04_screen_mp.py's top-15 MP hits (screen_audit.md) that
     found three failure modes and tightened the query filters (328 -> 184).
  2. How much of the final 184-candidate pool actually carries the model's
     single most important feature (Family, SHAP rank #1, fig 3a) -- versus
     falling back to 'unknown'.

Both data sources are already static/offline (a hand-audited historical record +
the shipped screen_mp_results.csv), so this script takes no arguments and never
hits the network. Run after 04_screen_mp.py:
    python 05_screen_audit_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SRC = os.path.join(HERE, "source_data")
sys.path.insert(0, HERE)
from src import plotstyle as ps  # noqa: E402

# kept identical to 04_screen_mp.py's BLUE_FAMILY_COLORS for the fig 5/7 pairing --
# update both if the ramp ever changes (not imported: '04_screen_mp' is not a
# valid Python module name, starting with a digit)
BLUE_FAMILY_COLORS = {
    "LGPS": "#0F4D92",
    "argyrodites": "#2E6DB4",
    "thio-LISICON": "#5B92CC",
    "sulfides": "#8FB8E0",
    "unknown": "#BCD4EA",
}

# Transcribed verbatim from screen_audit.md (2026-06-29 audit of the top-15 hits
# of an initial loose MP query: contains Li+S, e_above_hull<=0.05, band_gap>=0.5).
# Source of truth: screen_audit.md -- do NOT re-derive or re-query; this is a
# fixed historical human-audit record, not something the pipeline recomputes.
AUDIT_HITS = [
    # (material_id, formula, category, kept)
    ("mp-696128", "Li10Ge(PS6)2",      "true_positive_kept",       True),
    ("mp-696129", "Li10Si(PS6)2",      "true_positive_kept",       True),
    ("mp-696123", "Li10Sn(PS6)2",      "true_positive_kept",       True),
    ("mp-696138", "Li10Ge(PS6)2 (P1)", "true_positive_kept",       True),
    ("mp-720509", "Li10Si(PS6)2 (P1)", "true_positive_kept",       True),
    ("mp-1040451", "Li20Si3P3S23Cl",   "novel_lead_kept",          True),
    ("mp-753546", "Li8TiS6",           "novel_lead_kept",          True),
    ("mp-756490", "Li6MnS4",           "novel_redox_dropped",      False),
    ("mp-1153",   "Li2S",              "fp_binary_dropped",        False),
    ("mp-769032", "Li3NbS4",           "fp_redox_cathode_dropped", False),
    ("mp-753974", "Li8CrS6",           "fp_redox_cathode_dropped", False),
    ("mp-766506", "Li3CuS2",           "fp_redox_cathode_dropped", False),
    ("mp-777963", "Li3SbS4*9H2O",      "artifact_H_dropped",       False),
    ("mp-740717", "Li(NH4)SO4",        "artifact_H_dropped",       False),
    ("mp-760046", "LiPH21S3N7",        "artifact_H_dropped",       False),
]

# kept categories first (top of the horizontal bar, per the project's "best at
# top" convention in 04_screen_mp.py's _plot())
CATEGORY_ORDER = [
    "true_positive_kept", "novel_lead_kept",
    "novel_redox_dropped", "fp_redox_cathode_dropped",
    "fp_binary_dropped", "artifact_H_dropped",
]
CATEGORY_LABELS = {
    "true_positive_kept":       "True positive (kept)",
    "novel_lead_kept":          "Novel lead (kept)",
    "novel_redox_dropped":      "Novel, redox risk (dropped)",
    "fp_redox_cathode_dropped": "False pos.: redox-TM / cathode (dropped)",
    "fp_binary_dropped":        "False pos.: binary precursor (dropped)",
    "artifact_H_dropped":       "Artifact: H-hydrate/oxysalt (dropped)",
}
CATEGORY_KEPT = {c: c.endswith("_kept") for c in CATEGORY_ORDER}

# Funnel counts (screen_audit.md: "pool from 328 -> 184 clean candidates (64
# redox-TM sulfides removed)"). This 64 is the ONLY subtraction separately
# quantified anywhere in the repo -- no cached pre-filter data exists to split
# the remaining ~80 (dropped by H-exclusion + >=3-element filters combined),
# and re-querying Materials Project live would risk drifting from the frozen
# 2026-06-29 run this whole audit is based on. Do not fabricate that split.
FUNNEL_LOOSE_N = 328
FUNNEL_CLEAN_N = 184
FUNNEL_REDOX_TM_REMOVED = 64
FUNNEL_OTHER_REMOVED = FUNNEL_LOOSE_N - FUNNEL_CLEAN_N - FUNNEL_REDOX_TM_REMOVED  # 80, unsplit


def plot_audit_funnel():
    """Figure 6: (a) the 328->184 filter funnel, honest about what's NOT separately
    quantified; (b) the 15 manually-audited hits' verdict breakdown -- the exact,
    fully-traceable record behind the funnel."""
    ps.apply_publication_style()
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(ps.COL_DOUBLE_IN, 3.6),
        gridspec_kw={"width_ratios": [1, 1.5]})

    # --- panel a: loose query -> clean pool -----------------------------------
    stages = ["Loose query", "Audited filters"]
    counts = [FUNNEL_LOOSE_N, FUNNEL_CLEAN_N]
    axA.bar(stages, counts, color=[ps.PALETTE["neutral_light"], ps.PALETTE["blue_main"]],
            edgecolor=ps.PALETTE["neutral_black"], linewidth=0.6, width=0.55)
    for x, v in enumerate(counts):
        axA.annotate(str(v), (x, v), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=ps.FS_ANNOT, fontweight="bold")
    axA.set_ylabel("MP candidates")
    axA.set_ylim(0, 380)
    # place in the empty space ABOVE the short (184) bar, in DATA coords -- the
    # narrow gap between the two categorical bars is too tight for 3 lines of
    # text and an axes-fraction center previously overlapped the blue bar's fill
    axA.annotate(
        f"-{FUNNEL_REDOX_TM_REMOVED} redox-TM sulfides (quantified)\n"
        f"-{FUNNEL_OTHER_REMOVED} via H-exclusion + ≥3-element\n"
        "filters (combined; not separately\nlogged in this run)",
        xy=(1, 280), xycoords="data", fontsize=ps.FS_ANNOT,
        ha="center", va="center", color=ps.PALETTE["neutral_mid"])

    # --- panel b: the 15 audited hits, by verdict category --------------------
    order = CATEGORY_ORDER[::-1]
    vals = [sum(1 for _, _, c, _ in AUDIT_HITS if c == cat) for cat in order]
    colors = [ps.PALETTE["blue_main"] if CATEGORY_KEPT[cat] else ps.PALETTE["neutral_mid"]
             for cat in order]
    ypos = range(len(order))
    axB.barh(ypos, vals, color=colors, edgecolor=ps.PALETTE["neutral_black"], linewidth=0.5)
    axB.set_yticks(ypos)
    axB.set_yticklabels([CATEGORY_LABELS[c] for c in order])
    for y, v in zip(ypos, vals):
        axB.annotate(str(v), (v, y), textcoords="offset points", xytext=(3, 0),
                    va="center", fontsize=ps.FS_ANNOT)
    axB.set_xlabel("Audited top-15 MP hits")
    axB.set_xlim(0, 6.3)
    n_kept = sum(1 for *_, kept in AUDIT_HITS if kept)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=ps.PALETTE["blue_main"]),
              plt.Rectangle((0, 0), 1, 1, facecolor=ps.PALETTE["neutral_mid"])]
    axB.legend(handles, [f"kept ({n_kept}/15)", f"dropped ({15 - n_kept}/15)"],
              loc="lower right", fontsize=ps.FS_LEGEND)

    ps.add_panel_label(axA, "a")
    ps.add_panel_label(axB, "b", x=-0.55)
    os.makedirs(SRC, exist_ok=True)
    pd.DataFrame({"stage": stages, "n_candidates": counts}).to_csv(
        os.path.join(SRC, "fig06a_screen_audit_funnel.csv"), index=False)
    pd.DataFrame(AUDIT_HITS, columns=["material_id", "formula", "category", "kept"]).to_csv(
        os.path.join(SRC, "fig06b_screen_audit_verdicts.csv"), index=False)
    ps.finalize_figure(fig, os.path.join(FIG, "06_screen_audit.png"), w_pad=3.0)


def plot_family_coverage():
    """Figure 7: Family -- the #1 SHAP feature (fig 3a) -- is unassigned ('unknown')
    for most of the final screened pool. Reads screen_mp_results.csv (184 rows,
    already shipped by 04_screen_mp.py's live run); no live query here."""
    ps.apply_publication_style()
    results_csv = os.path.join(HERE, "screen_mp_results.csv")
    df = pd.read_csv(results_csv)
    counts = df["Family"].value_counts()
    order = [f for f in ["unknown", "thio-LISICON", "LGPS", "sulfides", "argyrodites"]
            if f in counts.index]
    labels = order[::-1]                       # largest bar at top
    vals = [int(counts[f]) for f in labels]
    colors = [BLUE_FAMILY_COLORS[f] for f in labels]
    pct = [100 * v / len(df) for v in vals]

    fig, ax = plt.subplots(figsize=(ps.COL_SINGLE_IN, 3.2))
    ypos = range(len(labels))
    ax.barh(ypos, vals, color=colors, edgecolor=ps.PALETTE["neutral_black"], linewidth=0.5)
    for y, v, p in zip(ypos, vals, pct):
        ax.annotate(f"{v} ({p:.1f}%)", (v, y), textcoords="offset points",
                   xytext=(3, 0), va="center", fontsize=ps.FS_ANNOT)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_xlabel(f"Screened MP candidates (n = {len(df)})")
    ax.annotate(
        "Family is the top SHAP feature (mean|SHAP| = 0.32, Fig. 3a)\n"
        "yet is unassigned for most screened candidates.",
        xy=(0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=ps.FS_ANNOT, color=ps.PALETTE["neutral_mid"])

    os.makedirs(SRC, exist_ok=True)
    pd.DataFrame({"Family": labels, "count": vals, "pct": pct}).to_csv(
        os.path.join(SRC, "fig07_family_coverage.csv"), index=False)
    ps.finalize_figure(fig, os.path.join(FIG, "07_family_coverage.png"))


def main():
    plot_audit_funnel()
    plot_family_coverage()
    print("Saved: figures/06_screen_audit.png, figures/07_family_coverage.png, "
          "source_data/fig06{a,b}_*.csv, source_data/fig07_family_coverage.csv")


if __name__ == "__main__":
    main()
