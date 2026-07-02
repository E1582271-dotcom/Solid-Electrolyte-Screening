"""
Project 1 -- apply the trained CatBoost model to candidate Li-sulfide solid
electrolytes and output a ranked top-list. This is the *screening-funnel* layer
of the portfolio: cheap composition+structure features -> a coarse conductivity
prior used to RANK candidates (not to predict absolute sigma; see README).

Two modes
---------
  --demo   No API key needed. Featurize a handful of well-characterised sulfide
           electrolytes (argyrodites, LGPS, thio-LISICONs, Li2S) using their
           approximate experimental cells and rank them. This proves the
           featurize -> model -> rank pipeline end-to-end offline, and doubles
           as a chemistry sanity check (argyrodite/LGPS should top Li2S).

  (live)   Query Materials Project for Li-S candidates via mp-api, featurize,
           predict, rank, and save a CSV + figure. Needs an API key, taken from
           --api-key, $MP_API_KEY, or a local `mp_api_key.txt` (keep it out of
           git). Filters to e_above_hull <= 0.05 eV/atom (near-stable) and a
           finite band gap (electronic insulator -- a basic SE prerequisite).

Honest caveat baked into the output: OBELiX's strongest feature is `Family`
(see 02_shap.py). For MP candidates we assign Family only by a TRANSPARENT
stoichiometry heuristic for the few classic classes (argyrodite Li6PS5X,
LGPS Li10MP2S12, thio-LISICON LixMS4, Li3PS4/Li7P3S11); everything else is
"unknown", an unseen CatBoost category, so its score leans on composition +
structure alone. The assigned family is written to the CSV so it is auditable.

Run:
    .venv\\Scripts\\python.exe 04_screen_mp.py --demo
    .venv\\Scripts\\python.exe 04_screen_mp.py --api-key <KEY> --max 400 --top 25
"""
from __future__ import annotations
import os, sys, re, math, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.featurize import (
    _composition_features, _cell_volume, _crystal_system, CAT_FEATURES,
    subscript_formula,
)
from src import plotstyle as ps
from catboost import CatBoostRegressor

FIG = os.path.join(HERE, "figures")
SRC = os.path.join(HERE, "source_data")
os.makedirs(FIG, exist_ok=True)

FAMILY_COLORS = ps.FAMILY_COLORS   # CVD-safe Okabe-Ito palette


# --------------------------------------------------------------------------- #
# Transparent family heuristic (only the classic, unambiguous sulfide classes)
# --------------------------------------------------------------------------- #
def guess_family(formula: str) -> str:
    """Map a composition to an OBELiX family by stoichiometry. Conservative:
    returns 'unknown' unless the pattern is a well-known sulfide class."""
    from pymatgen.core import Composition
    try:
        comp = Composition(formula)
    except Exception:
        return "unknown"
    el = {str(e) for e in comp.elements}
    d = comp.get_el_amt_dict()
    if "S" not in el:
        return "unknown"
    # argyrodite Li6PS5X (X = Cl/Br/I), allow Li-deficient/excess near 6
    if {"Li", "P", "S"} <= el and el & {"Cl", "Br", "I"}:
        x = sum(d.get(h, 0) for h in ("Cl", "Br", "I"))
        if 0.5 <= x and 4.0 <= d.get("S", 0) / max(x, 1e-9) <= 7.0:
            return "argyrodites"
    # LGPS-type Li10MP2S12 (M = Ge/Sn/Si)
    if {"Li", "P", "S"} <= el and el & {"Ge", "Sn", "Si"}:
        return "LGPS"
    # thio-LISICON LixMS4 (M = Ge/Sn/Si/Al/...), no P
    if "P" not in el and el & {"Ge", "Sn", "Si", "Al", "Ga"} and "Li" in el:
        return "thio-LISICON"
    # binary/ternary Li-P-S without halide: Li3PS4, Li7P3S11, ...
    if el <= {"Li", "P", "S"} and "Li" in el and "P" in el:
        return "sulfides"
    return "unknown"


# --------------------------------------------------------------------------- #
# Feature construction matching the trained model's columns
# --------------------------------------------------------------------------- #
def build_feature_row(formula, a, b, c, alpha, beta, gamma, sg_no, Z,
                      family=None) -> dict:
    feat = _composition_features(formula)
    feat.update({
        "a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma,
        "cell_volume": _cell_volume(a, b, c, alpha, beta, gamma),
        "spacegroup_no": sg_no, "Z": Z,
        "crystal_system": _crystal_system(sg_no),
        "Family": family if family is not None else guess_family(formula),
    })
    return feat


def predict(model: CatBoostRegressor, rows: list[dict]) -> pd.DataFrame:
    """rows -> DataFrame reindexed to the model's own feature order, then
    predicted. Returns rows + pred_log10_sigma + pred_sigma_S_cm, ranked."""
    names = list(model.feature_names_)
    X = pd.DataFrame(rows).reindex(columns=names)
    for col in names:
        if col in CAT_FEATURES:
            X[col] = X[col].astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    yhat = model.predict(X)
    out = pd.DataFrame(rows)
    out["pred_log10_sigma"] = yhat
    out["pred_sigma_S_cm"] = np.power(10.0, yhat)
    return out.sort_values("pred_log10_sigma", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Offline demo set: approximate EXPERIMENTAL cells of known sulfide SEs
# (a,b,c in A; angles in deg; sg = space-group number; Z = f.u. / cell)
# --------------------------------------------------------------------------- #
DEMO_SET = [
    # name, formula, a, b, c, al, be, ga, sg, Z, true note (log10 sigma RT)
    ("Li6PS5Cl",     "Li6PS5Cl",    9.86, 9.86, 9.86, 90, 90, 90, 216, 4, "~1e-3"),
    ("Li6PS5Br",     "Li6PS5Br",    9.98, 9.98, 9.98, 90, 90, 90, 216, 4, "~1e-3"),
    ("Li6PS5I",      "Li6PS5I",    10.14,10.14,10.14, 90, 90, 90, 216, 4, "~1e-6"),
    ("Li10GeP2S12",  "Li10GeP2S12", 8.69, 8.69,12.63, 90, 90, 90, 137, 2, "~1e-2"),
    ("Li10SnP2S12",  "Li10SnP2S12", 8.74, 8.74,12.76, 90, 90, 90, 137, 2, "~4e-3"),
    ("b-Li3PS4",     "Li3PS4",     12.82, 8.22, 6.12, 90, 90, 90,  62, 4, "~1e-4"),
    ("Li7P3S11",     "Li7P3S11",   12.50, 6.03,12.53,102,113, 74,   2, 2, "~1e-2"),
    ("Li4GeS4",      "Li4GeS4",    14.05, 7.76, 6.13, 90, 90, 90,  62, 4, "~1e-7"),
    ("Li2S",         "Li2S",        5.71, 5.71, 5.71, 90, 90, 90, 225, 4, "~1e-13"),
]


def run_demo(model):
    rows = []
    for name, formula, a, b, c, al, be, ga, sg, Z, note in DEMO_SET:
        row = build_feature_row(formula, a, b, c, al, be, ga, sg, Z)
        row = {"label": name, "formula": formula, "exp_log10_sigma_note": note,
               "spacegroup_no": sg, **row}
        rows.append(row)
    ranked = predict(model, rows)
    show = ranked[["label", "formula", "Family", "pred_log10_sigma",
                   "pred_sigma_S_cm", "exp_log10_sigma_note"]].copy()
    show["pred_log10_sigma"] = show["pred_log10_sigma"].round(2)
    show["pred_sigma_S_cm"] = show["pred_sigma_S_cm"].map(lambda v: f"{v:.1e}")
    print("\n=== DEMO: known sulfide electrolytes ranked by predicted log10 sigma ===")
    print(show.to_string(index=False))
    print("\nSanity check: argyrodites / LGPS should sit near the top, Li2S at "
          "the bottom -- matching the SHAP family ranking in 02_shap.py.")
    out_csv = os.path.join(HERE, "screen_demo.csv")
    ranked.to_csv(out_csv, index=False)
    _plot(ranked, "label", os.path.join(FIG, "04_screen_demo.png"))
    print(f"\nSaved: {os.path.relpath(out_csv, HERE)}, "
          "figures/04_screen_demo.png, source_data/fig04_screen_demo.csv")
    return ranked


# --------------------------------------------------------------------------- #
# Live Materials Project screen
# --------------------------------------------------------------------------- #
def resolve_key(cli_key):
    if cli_key:
        return cli_key
    if os.environ.get("MP_API_KEY"):
        return os.environ["MP_API_KEY"]
    keyfile = os.path.join(HERE, "mp_api_key.txt")
    if os.path.exists(keyfile):
        with open(keyfile) as f:
            return f.read().strip()
    return None


# Redox-active open-shell transition metals: their Li sulfides are mixed
# ionic-electronic conductors / cathodes (Li8CrS6, Li6MnS4, Li3CuS2 ...), not
# electrolytes. Demoted post-query -- see the top-hit audit (screen_audit.md).
REDOX_TM = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "W"}


def _has_redox_tm(formula: str) -> bool:
    from pymatgen.core import Composition
    try:
        return bool({str(e) for e in Composition(formula).elements} & REDOX_TM)
    except Exception:
        return False


def run_live(model, api_key, max_n, top_n):
    from mp_api.client import MPRester
    print(f"Querying Materials Project for Li-S candidates (cap {max_n}) ...")
    with MPRester(api_key) as mpr:
        # Filters distilled from the adversarial audit of an earlier looser run
        # (screen_audit.md): exclude H to drop hydrate / ammonium molecular
        # salts (Li3SbS4*9H2O, LiNH4SO4, ...); require >=3 elements to drop
        # binaries (Li2S antifluorite is a precursor, not an SE); raise the gap
        # floor to demote narrow-gap mixed conductors (Li3NbS4 etc.).
        docs = mpr.materials.summary.search(
            elements=["Li", "S"],
            exclude_elements=["H"],
            num_elements=(3, 5),
            energy_above_hull=(0, 0.05),       # near-stable / synthesizable
            band_gap=(1.5, None),              # electronic insulator (SE prereq)
            fields=["material_id", "formula_pretty", "structure",
                    "symmetry", "energy_above_hull", "band_gap"],
        )
    print(f"  retrieved {len(docs)} entries; featurizing ...")
    rows, skipped, miec = [], 0, 0
    for d in docs[:max_n]:
        try:
            formula = d.formula_pretty
            if _has_redox_tm(formula):         # MIEC / cathode, not an electrolyte
                miec += 1
                continue
            st = d.structure
            a, b, c = st.lattice.abc
            al, be, ga = st.lattice.angles
            sg = int(d.symmetry.number)
            _, z = st.composition.get_reduced_formula_and_factor()
            row = build_feature_row(formula, a, b, c, al, be, ga, sg, int(z))
            row = {"material_id": str(d.material_id), "formula": formula,
                   "e_above_hull": float(d.energy_above_hull),
                   "band_gap": float(d.band_gap) if d.band_gap is not None else np.nan,
                   "spacegroup_no": sg, **row}
            rows.append(row)
        except Exception as e:                 # one bad structure shouldn't kill the run
            skipped += 1
            continue
    if miec:
        print(f"  dropped {miec} redox-active TM sulfides (MIEC/cathode, not SE)")
    if skipped:
        print(f"  skipped {skipped} entries with incomplete data")
    if not rows:
        print("No usable candidates -- nothing to rank."); return None

    ranked = predict(model, rows)
    out_csv = os.path.join(HERE, "screen_mp_results.csv")
    keep = ["material_id", "formula", "Family", "spacegroup_no", "e_above_hull",
            "band_gap", "pred_log10_sigma", "pred_sigma_S_cm"]
    ranked[keep].to_csv(out_csv, index=False)

    head = ranked[keep].head(top_n).copy()
    head["pred_log10_sigma"] = head["pred_log10_sigma"].round(2)
    head["pred_sigma_S_cm"] = head["pred_sigma_S_cm"].map(lambda v: f"{v:.1e}")
    head["e_above_hull"] = head["e_above_hull"].round(3)
    print(f"\n=== Top {top_n} Li-S candidates by predicted log10 sigma ===")
    print(head.to_string(index=False))
    print("\nReminder: this is a RANK, not a quantitative sigma. Family is "
          "heuristic (see header); novel chemistries score on composition + "
          "structure only. Validate the head of this list with Project 2 MD.")
    _plot(ranked.head(min(top_n, 20)), "formula",
          os.path.join(FIG, "05_screen_mp.png"), highlight=AUDITED_LEADS)
    print(f"\nSaved: {os.path.relpath(out_csv, HERE)} ({len(ranked)} rows), "
          "figures/05_screen_mp.png, source_data/fig05_screen_mp.csv")
    return ranked


# the two literature-blank candidates that survived the screen_audit.md pass;
# they are the hand-off to Project 2 MD and get visual priority in the figure
AUDITED_LEADS = frozenset({"Li20Si3P3S23Cl", "Li8TiS6"})


# --------------------------------------------------------------------------- #
def _lighten(hex_color, amount=0.62):
    """Blend a hex colour toward white (0 = unchanged, 1 = white)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: int(round(c + (255 - c) * amount))
    return f"#{mix(r):02X}{mix(g):02X}{mix(b):02X}"


def _plot(ranked, label_col, path, highlight=frozenset()):
    # dedupe by composition (keep best-ranked polymorph) so distinct formulas
    # show; use numeric y-positions so repeated labels don't collapse onto one row
    uniq = ranked.drop_duplicates(subset="formula", keep="first")
    top = uniq.head(20).iloc[::-1].reset_index(drop=True)
    vals = top["pred_log10_sigma"].astype(float)
    # bars are anchored at a floor BELOW the worst value, not at 0: log sigma is
    # negative, so 0-anchored bars make the worst conductor the longest bar
    # (ink says the opposite of the ranking). With the floor, longer = better.
    floor = math.floor((vals.min() - 0.4) * 2) / 2
    is_lead = top["formula"].isin(highlight)
    fam_col = [FAMILY_COLORS.get(f, FAMILY_COLORS["unknown"]) for f in top["Family"]]
    if highlight:
        # visual hierarchy: audited leads keep the full family colour, everything
        # else fades to a light fill with a family-coloured edge
        fills = [c if lead else _lighten(c) for c, lead in zip(fam_col, is_lead)]
    else:
        fills = fam_col
    ypos = list(range(len(top)))
    fig, ax = plt.subplots(figsize=(ps.COL_DOUBLE_IN, max(3.0, 0.32 * len(top))))
    ax.barh(ypos, vals - floor, left=floor, color=fills, edgecolor=fam_col, linewidth=0.7)
    for y, v, lead in zip(ypos, vals, is_lead):
        ax.annotate(f"{v:.1f}" + ("  • lead" if lead else ""), (v, y),
                    textcoords="offset points", xytext=(3, 0), va="center", ha="left",
                    fontsize=ps.FS_ANNOT, fontweight="bold" if lead else "normal",
                    color=ps.PALETTE["neutral_black"] if lead else ps.PALETTE["neutral_mid"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([subscript_formula(s) for s in top[label_col].astype(str)])
    for tick, lead in zip(ax.get_yticklabels(), is_lead):
        if lead:
            tick.set_fontweight("bold")
    ax.set_ylim(-0.6, len(top) - 0.4)
    ax.set_xlim(floor, vals.max() + 1.1)          # headroom for the value labels
    ax.set_xlabel("Predicted log$_{10}$ σ  (S cm$^{-1}$)")
    # legend: only families present in this figure, placed to the right of the axes
    present = [f for f in FAMILY_COLORS if f in set(top["Family"])]
    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=(FAMILY_COLORS[f] if not highlight else _lighten(FAMILY_COLORS[f])),
                             edgecolor=FAMILY_COLORS[f], linewidth=0.7) for f in present]
    ax.legend(handles, present, title="Family (heuristic)", loc="upper left",
              bbox_to_anchor=(1.02, 1.0), fontsize=ps.FS_LEGEND, title_fontsize=ps.FS_LEGEND)
    # source data (rank order, best first)
    os.makedirs(SRC, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    uniq.head(20)[[label_col, "Family", "pred_log10_sigma"]].to_csv(
        os.path.join(SRC, f"fig{base}.csv"), index=False)
    ps.finalize_figure(fig, path)


def main():
    ps.apply_publication_style()
    ap = argparse.ArgumentParser(description="Screen Li-sulfide candidates with the OBELiX CatBoost model.")
    ap.add_argument("--demo", action="store_true", help="offline demo on known SEs (no API key)")
    ap.add_argument("--api-key", default=None, help="Materials Project API key")
    ap.add_argument("--max", type=int, default=400, help="max MP entries to featurize")
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    ap.add_argument("--replot", action="store_true", help="redraw figures/05_screen_mp.png "
                    "from the shipped screen_mp_results.csv (no MP query, no key)")
    args = ap.parse_args()

    if args.replot:
        csv = os.path.join(HERE, "screen_mp_results.csv")
        if not os.path.exists(csv):
            sys.exit("screen_mp_results.csv not found -- run a live screen first.")
        ranked = pd.read_csv(csv)
        _plot(ranked.head(min(args.top, 20)), "formula",
              os.path.join(FIG, "05_screen_mp.png"), highlight=AUDITED_LEADS)
        print("Replotted figures/05_screen_mp.png from screen_mp_results.csv "
              f"({len(ranked)} rows, data unchanged).")
        return

    model_path = os.path.join(HERE, "catboost_model.cbm")
    if not os.path.exists(model_path):
        sys.exit("catboost_model.cbm not found -- run 01_train_eval.py first.")
    model = CatBoostRegressor(); model.load_model(model_path)

    if args.demo:
        run_demo(model); return

    key = resolve_key(args.api_key)
    if not key:
        print("No MP API key (use --api-key, $MP_API_KEY, or mp_api_key.txt).")
        print("Running --demo instead so you can see the pipeline work:\n")
        run_demo(model); return
    run_live(model, key, args.max, args.top)


if __name__ == "__main__":
    main()
