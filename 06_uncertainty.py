"""
Project 1 -- W11 follow-up: does the screen KNOW when it is guessing?

The W11 MD campaign inverted the CatBoost prior's ranking: LiPS3 (gen021), ranked
LAST of the four leads (-7.12, Family=unknown), measured sigma300 ~ 10 mS/cm --
2nd-best of the campaign -- while Li8TiS6 (ranked 2nd) was falsified. Hypothesis:
generated/novel chemistries fall into the model's `Family=unknown` blind spot, and
an UNCERTAINTY-aware screen would have flagged them for MD instead of rank-filtering
them out.

This script tests that hypothesis without touching the shipped point model:
  1. Train a 10-seed CatBoost ensemble (same config as 01_train_eval.py:make_cat,
     only random_seed varies). The shipped catboost_model.cbm stays untouched.
  2. Calibration check on the held-out OBELiX TEST split: is the ensemble spread
     (std over seeds) actually informative about the real error? (binned RMSE +
     Spearman rank correlation; CatBoost virtual ensembles as a cross-check.)
  3. Retro-score the MP screen pool (same live query + filters as 04_screen_mp.py)
     AND the four W11 MD leads (CIFs from ../project2_mlip_md/data/leads/) with
     mean +/- std, then locate each lead's spread inside the pool's distribution.
  4. Figure 08: (a) calibration curve, (b) "risk map" (predicted sigma vs spread)
     with the four leads marked by their MD verdict.

If the hypothesis holds, the actionable rule is: route high-spread candidates to
MD (the funnel's layer 2) instead of trusting their point rank -- turning the W11
ranking inversion from an anecdote into a design principle.

Run:
    .venv/bin/python 06_uncertainty.py              # full: trains 10 models, queries MP
    .venv/bin/python 06_uncertainty.py --skip-mp    # offline: leads + calibration only
    .venv/bin/python 06_uncertainty.py --replot     # redraw fig 08 from source_data/
"""
from __future__ import annotations
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.featurize import load_obelix, feature_columns, CAT_FEATURES, subscript_formula
from src import plotstyle as ps
from catboost import CatBoostRegressor

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
SRC = os.path.join(HERE, "source_data")
PROJ2 = os.path.normpath(os.path.join(HERE, "..", "project2_mlip_md"))
N_SEEDS = 10
N_BINS = 5


def _load_p1_screen():
    """Import 04_screen_mp.py (name starts with a digit) for build_feature_row /
    guess_family / resolve_key / _has_redox_tm -- reuse, don't re-implement."""
    spec = importlib.util.spec_from_file_location(
        "p1_screen", os.path.join(HERE, "04_screen_mp.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Ensemble: same config as 01_train_eval.py:make_cat, only the seed varies
# --------------------------------------------------------------------------- #
def train_ensemble(X_tr, y_tr, n_seeds=N_SEEDS):
    models = []
    for seed in range(n_seeds):
        m = CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
                              loss_function="RMSE", random_seed=seed, verbose=False)
        m.fit(X_tr, y_tr, cat_features=CAT_FEATURES)
        models.append(m)
        print(f"  [ens] seed {seed} fitted")
    return models


def ens_predict(models, rows) -> pd.DataFrame:
    """rows (list[dict] or DataFrame) -> DataFrame with ens_mean / ens_std, using
    the same column alignment as 04_screen_mp.py:predict."""
    names = list(models[0].feature_names_)
    X = pd.DataFrame(rows).reindex(columns=names)
    for col in names:
        if col in CAT_FEATURES:
            X[col] = X[col].astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    preds = np.stack([m.predict(X) for m in models])          # (k, n)
    out = pd.DataFrame(rows)
    out["ens_mean"] = preds.mean(axis=0)
    out["ens_std"] = preds.std(axis=0)
    return out


# --------------------------------------------------------------------------- #
# Calibration on the held-out test split
# --------------------------------------------------------------------------- #
def calibration(models, X_te, y_te):
    from scipy.stats import spearmanr
    preds = np.stack([m.predict(X_te) for m in models])
    mu, sd = preds.mean(axis=0), preds.std(axis=0)
    err = np.abs(y_te.to_numpy() - mu)
    rho, pval = spearmanr(sd, err)

    # quantile-bin the spread; within each bin the empirical RMSE of the mean
    q = pd.qcut(sd, N_BINS, labels=False, duplicates="drop")
    bins = (pd.DataFrame({"bin": q, "sd": sd, "err": err})
            .groupby("bin")
            .agg(mean_std=("sd", "mean"),
                 rmse=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                 n=("err", "size"))
            .reset_index())

    # cross-check: virtual ensembles on the SHIPPED point model (tree-slice trick,
    # no retraining) -- do the two uncertainty notions rank the test set alike?
    ve_rho = None
    try:
        ship = CatBoostRegressor()
        ship.load_model(os.path.join(HERE, "catboost_model.cbm"))
        ve = ship.virtual_ensembles_predict(X_te, prediction_type="VirtEnsembles",
                                            virtual_ensembles_count=10)
        ve_sd = np.asarray(ve).std(axis=1)
        ve_rho = float(spearmanr(ve_sd, sd).statistic)
    except Exception as e:                                    # keep the analysis alive
        print(f"  [cal] virtual-ensemble cross-check skipped: {e}")

    return mu, sd, err, float(rho), float(pval), bins, ve_rho


# --------------------------------------------------------------------------- #
# Retro-scoring: MP pool (live re-query, 04's filters) + the four W11 MD leads
# --------------------------------------------------------------------------- #
def mp_pool_rows(p1, api_key, max_n=400):
    from mp_api.client import MPRester
    print(f"Querying Materials Project (04_screen_mp.py filters, cap {max_n}) ...")
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=["Li", "S"], exclude_elements=["H"], num_elements=(3, 5),
            energy_above_hull=(0, 0.05), band_gap=(1.5, None),
            fields=["material_id", "formula_pretty", "structure",
                    "symmetry", "energy_above_hull", "band_gap"],
        )
    rows = []
    for d in docs[:max_n]:
        try:
            formula = d.formula_pretty
            if p1._has_redox_tm(formula):
                continue
            st = d.structure
            a, b, c = st.lattice.abc
            al, be, ga = st.lattice.angles
            sg = int(d.symmetry.number)
            _, z = st.composition.get_reduced_formula_and_factor()
            row = p1.build_feature_row(formula, a, b, c, al, be, ga, sg, int(z))
            rows.append({"label": str(d.material_id), "formula": formula, **row})
        except Exception:
            continue
    print(f"  {len(rows)} clean candidates featurized")
    return rows


def lead_rows(p1):
    """Feature rows for the four W11 leads (project 2), spacegroup logic mirroring
    project3/src/score.py:_row_from_structure. The lead CIFs are MD SUPERCELLS
    (96-416 atoms), but the funnel scored the primitive/relaxed cells -- so reduce
    to primitive first, otherwise cell_volume/Z (SHAP top features) are inflated
    and the historical prior is NOT reproduced. Check: gen016 primitive-reduced
    gives -6.832, byte-identical to candidates_final.csv."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    rows = []
    for cif in sorted(glob.glob(os.path.join(PROJ2, "data", "leads", "lead_*.cif"))):
        stem = os.path.basename(cif)[len("lead_"):-len(".cif")]
        st = Structure.from_file(cif).get_primitive_structure(tolerance=0.25)
        formula = st.composition.reduced_formula
        a, b, c = st.lattice.abc
        al, be, ga = st.lattice.angles
        try:
            sg = SpacegroupAnalyzer(st, symprec=0.1).get_space_group_number()
        except Exception:
            sg = 1                                            # generated P1 cells
        _, z = st.composition.get_reduced_formula_and_factor()
        row = p1.build_feature_row(formula, a, b, c, al, be, ga, sg, int(z))
        rows.append({"label": stem, "formula": formula, **row})

        # MD verdict from project 2's metrics (sigma300 in mS/cm, MACE baseline)
        mpath = os.path.join(PROJ2, "data", f"metrics_lead_{stem}.json")
        try:
            with open(mpath, encoding="utf-8") as f:
                arr = json.load(f)["arrhenius"]["mace"]
            rows[-1]["sigma300_mS_cm"] = float(arr["sigma300_mS_cm"])
        except Exception:
            print(f"  [leads] no MD metrics for {stem} ({mpath})")
            rows[-1]["sigma300_mS_cm"] = np.nan
    return rows


# --------------------------------------------------------------------------- #
# Figure 08: (a) calibration, (b) risk map with the leads' MD verdicts
# --------------------------------------------------------------------------- #
def plot_fig08(bins, rho, mp_df, leads_df):
    fig, ax = plt.subplots(1, 2, figsize=(ps.COL_DOUBLE_IN, 2.9),
                           gridspec_kw={"width_ratios": [1, 1.5]})

    # (a) binned calibration: if spread is informative, RMSE rises with it
    ax[0].plot(bins["mean_std"], bins["rmse"], "o-", lw=1.0, ms=4,
               color=ps.PALETTE["blue_main"])
    ax[0].set_xlabel("Ensemble spread (log$_{10}$ σ)")
    ax[0].set_ylabel("Test RMSE in bin (log$_{10}$ σ)")
    ax[0].annotate(f"Spearman ρ = {rho:.2f}\n({N_SEEDS}-seed ensemble, "
                   f"{int(bins['n'].sum())} test rows)",
                   xy=(0.05, 0.95), xycoords="axes fraction", va="top", ha="left",
                   fontsize=ps.FS_ANNOT)

    # (b) risk map: point estimate (x) vs spread (y); Family=unknown pool candidates
    # are hollow -- the blind spot is literally visible as a separate cloud
    if mp_df is not None and len(mp_df):
        known = mp_df["Family"] != "unknown"
        ax[1].scatter(mp_df.loc[known, "ens_mean"], mp_df.loc[known, "ens_std"],
                      s=9, alpha=.65, linewidths=0, color=ps.PALETTE["blue_secondary"],
                      label="MP pool (known family)", zorder=2)
        ax[1].scatter(mp_df.loc[~known, "ens_mean"], mp_df.loc[~known, "ens_std"],
                      s=11, facecolors="none", linewidths=0.6,
                      edgecolors=ps.PALETTE["neutral_mid"],
                      label="MP pool (unknown)", zorder=2)
    verdict_col = {True: ps.PALETTE["teal"], False: ps.PALETTE["red_strong"]}
    # stagger the four labels above/below their stars so close-by leads (the two
    # generated ones sit within ~0.6 log units) never overprint each other
    offsets = [(5, 5), (5, -11), (5, 5), (-6, -11)]
    aligns = ["left", "left", "left", "right"]
    for i, (_, r) in enumerate(leads_df.iterrows()):
        good = bool(r["sigma300_mS_cm"] >= 1.0)               # same-order-as-SE threshold
        ax[1].scatter(r["ens_mean"], r["ens_std"], marker="*", s=110,
                      color=verdict_col[good], edgecolors=ps.PALETTE["neutral_black"],
                      linewidths=0.5, zorder=4)
        ax[1].annotate(subscript_formula(r["formula"]),
                       (r["ens_mean"], r["ens_std"]), textcoords="offset points",
                       xytext=offsets[i % 4], ha=aligns[i % 4], fontsize=ps.FS_ANNOT,
                       color=verdict_col[good], fontweight="bold", zorder=5)
    ax[1].set_xlabel("Ensemble mean log$_{10}$ σ  (S cm$^{-1}$)")
    ax[1].set_ylabel("Ensemble spread (log$_{10}$ σ)")
    handles, labels = ax[1].get_legend_handles_labels()
    star_ok = plt.Line2D([], [], marker="*", ls="", ms=9, color=ps.PALETTE["teal"],
                         markeredgecolor=ps.PALETTE["neutral_black"], markeredgewidth=0.5)
    star_bad = plt.Line2D([], [], marker="*", ls="", ms=9, color=ps.PALETTE["red_strong"],
                          markeredgecolor=ps.PALETTE["neutral_black"], markeredgewidth=0.5)
    ax[1].legend(handles + [star_ok, star_bad],
                 labels + ["lead: MD-confirmed", "lead: MD-falsified"],
                 loc="upper left", fontsize=ps.FS_LEGEND,
                 frameon=True, framealpha=0.9, edgecolor="none")
    ps.add_panel_label(ax[0], "a"); ps.add_panel_label(ax[1], "b", x=-0.05)
    ps.finalize_figure(fig, os.path.join(FIG, "08_uncertainty.png"), w_pad=2.0)


def replot_from_source_data():
    ps.apply_publication_style()
    bins = pd.read_csv(os.path.join(SRC, "fig08a_calibration.csv"))
    rho = float(json.load(open(os.path.join(DATA, "uncertainty.json")))["test_spearman_rho"])
    pool = pd.read_csv(os.path.join(SRC, "fig08b_risk_map.csv"))
    mp_df = pool[pool["kind"] == "mp"].copy()
    leads_df = pool[pool["kind"] == "lead"].copy()
    plot_fig08(bins, rho, mp_df if len(mp_df) else None, leads_df)
    print("Replotted figures/08_uncertainty.png from source_data/ (no retraining).")


# --------------------------------------------------------------------------- #
def main(args):
    ps.apply_publication_style()
    p1 = _load_p1_screen()

    train, test = load_obelix(DATA)
    cols = feature_columns(train)
    X_tr, y_tr = train[cols].copy(), train["y"]
    X_te, y_te = test[cols].copy(), test["y"]
    for c in CAT_FEATURES:
        X_tr[c] = X_tr[c].astype(str)
        X_te[c] = X_te[c].astype(str)

    print(f"Training {args.seeds}-seed ensemble (config = 01_train_eval.py:make_cat):")
    models = train_ensemble(X_tr, y_tr, args.seeds)

    mu, sd, err, rho, pval, bins, ve_rho = calibration(models, X_te, y_te)
    print(f"Calibration on held-out test: Spearman(spread, |error|) = {rho:.2f} "
          f"(p={pval:.1e})" + (f"; virtual-ensemble cross-check ρ = {ve_rho:.2f}"
                               if ve_rho is not None else ""))

    # sanity: the ensemble mean should agree with the shipped seed-42 model
    ship = CatBoostRegressor(); ship.load_model(os.path.join(HERE, "catboost_model.cbm"))
    dship = float(np.max(np.abs(ship.predict(X_te) - mu)))
    print(f"  max |shipped - ensemble mean| on test: {dship:.2f} log units")

    # ---- retro-score the pool + the four leads ----
    mp_df = None
    if not args.skip_mp:
        key = p1.resolve_key(None)
        if key:
            try:
                mp_df = ens_predict(models, mp_pool_rows(p1, key, args.max))
            except Exception as e:
                print(f"  [mp] live query failed ({e}) -- continuing with leads only")
        else:
            print("  [mp] no API key -- continuing with leads only")
    leads_df = ens_predict(models, lead_rows(p1))

    # each lead's spread as a percentile of the MP pool's spread distribution
    if mp_df is not None:
        pool_sd = mp_df["ens_std"].to_numpy()
        leads_df["std_percentile_in_pool"] = [
            float((pool_sd < s).mean() * 100.0) for s in leads_df["ens_std"]]
        fam_stats = (mp_df.assign(known=mp_df["Family"] != "unknown")
                     .groupby("known")["ens_std"].median())
        print("MP pool median spread: known-family "
              f"{fam_stats.get(True, float('nan')):.2f} vs unknown "
              f"{fam_stats.get(False, float('nan')):.2f} log units")
    print("\n=== W11 leads, retro-scored with uncertainty ===")
    show_cols = ["label", "formula", "Family", "ens_mean", "ens_std",
                 "sigma300_mS_cm"] + (["std_percentile_in_pool"] if mp_df is not None else [])
    print(leads_df[show_cols].round(3).to_string(index=False))

    # ---- artifacts: metrics JSON + figure + source data ----
    os.makedirs(SRC, exist_ok=True)
    summary = {
        "n_seeds": args.seeds,
        "test_spearman_rho": rho, "test_spearman_p": pval,
        "virtual_ensemble_cross_rho": ve_rho,
        "max_abs_diff_shipped_vs_ensmean": dship,
        "calibration_bins": bins.to_dict(orient="records"),
        "pool_median_spread": (
            {"known_family": float(fam_stats.get(True, np.nan)),
             "unknown": float(fam_stats.get(False, np.nan))} if mp_df is not None else None),
        "leads": leads_df[show_cols].to_dict(orient="records"),
        "verdict_threshold_mS_cm": 1.0,
    }
    with open(os.path.join(DATA, "uncertainty.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    bins.to_csv(os.path.join(SRC, "fig08a_calibration.csv"), index=False)
    keep = ["label", "formula", "Family", "ens_mean", "ens_std"]
    pool_rows = (mp_df[keep].assign(kind="mp", sigma300_mS_cm=np.nan)
                 if mp_df is not None else pd.DataFrame(columns=keep + ["kind", "sigma300_mS_cm"]))
    lead_keep = leads_df[keep + ["sigma300_mS_cm"]].assign(kind="lead")
    pd.concat([pool_rows, lead_keep], ignore_index=True).to_csv(
        os.path.join(SRC, "fig08b_risk_map.csv"), index=False)

    plot_fig08(bins, rho, mp_df, leads_df)
    print("\nSaved: data/uncertainty.json, figures/08_uncertainty.png, "
          "source_data/fig08a_calibration.csv + fig08b_risk_map.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="ensemble size")
    ap.add_argument("--max", type=int, default=400, help="max MP entries to featurize")
    ap.add_argument("--skip-mp", action="store_true",
                    help="skip the live MP re-query (offline: calibration + leads only)")
    ap.add_argument("--replot", action="store_true",
                    help="redraw figure 08 from source_data/ without retraining")
    a = ap.parse_args()
    if a.replot:
        replot_from_source_data()
    else:
        main(a)
