"""
Project 1 -- train & honestly evaluate ionic-conductivity regressors on OBELiX.

Pipeline:
  1. Featurize (src/featurize.py): log10(sigma) target, Magpie-lite composition
     descriptors + crystallography + 2 categoricals.
  2. 5-fold CV on the official TRAIN split for two models:
        - CatBoost (native categorical handling, NaN-safe)
        - RandomForest baseline (median impute + one-hot)  <- the OBELiX-style baseline
  3. Refit on full train, evaluate ONCE on the held-out official TEST split.
  4. Save data/metrics.json, a parity plot, and a target-distribution EDA figure.

Everything is in log10(S/cm) units. Run:
    .venv\\Scripts\\python.exe 01_train_eval.py
"""
from __future__ import annotations
import argparse
import os, json, math, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.featurize import load_obelix, feature_columns, CAT_FEATURES
from src import plotstyle as ps

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
SRC = os.path.join(HERE, "source_data")
os.makedirs(FIG, exist_ok=True)
RNG = 42


def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_rf(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("rf", RandomForestRegressor(n_estimators=500,
                                                  random_state=RNG, n_jobs=-1))])


def make_cat():
    return CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
                             loss_function="RMSE", random_seed=RNG, verbose=False)


def cv_eval(name, build_fn, X, y, cat_cols, is_catboost):
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    rows = []
    for tr_i, va_i in kf.split(X):
        Xtr, Xva = X.iloc[tr_i], X.iloc[va_i]
        ytr, yva = y.iloc[tr_i], y.iloc[va_i]
        m = build_fn()
        if is_catboost:
            m.fit(Xtr, ytr, cat_features=cat_cols)
        else:
            m.fit(Xtr, ytr)
        rows.append(metrics(yva, m.predict(Xva)))
    df = pd.DataFrame(rows)
    agg = {k: (float(df[k].mean()), float(df[k].std())) for k in df.columns}
    print(f"  [{name}] 5-fold CV  "
          + "  ".join(f"{k}={v[0]:.3f}±{v[1]:.3f}" for k, v in agg.items()))
    return agg


def plot_eda(ally, top_fam):
    """Figure 1: (a) target histogram, (b) median log10(sigma) by family. Panel b bars
    are anchored at a floor below the worst family, not at 0 -- the medians are negative,
    so 0-anchored bars would draw the WORST family as the longest bar."""
    fig, ax = plt.subplots(1, 2, figsize=(ps.COL_DOUBLE_IN, 2.8))
    ax[0].hist(ally, bins=30, color=ps.PALETTE["blue_main"], alpha=.9)
    ax[0].set_xlabel("log$_{10}$ σ  (S cm$^{-1}$)"); ax[0].set_ylabel("Count")
    floor = math.floor((top_fam.values.min() - 0.4) * 2) / 2
    ax[1].barh(top_fam.index, top_fam.values - floor, left=floor,
               color=ps.PALETTE["blue_main"])
    for y, v in enumerate(top_fam.values):
        ax[1].annotate(f"{v:.1f}", (v, y), textcoords="offset points", xytext=(3, 0),
                       va="center", ha="left", fontsize=ps.FS_ANNOT,
                       color=ps.PALETTE["neutral_mid"])
    ax[1].set_xlim(floor, top_fam.values.max() + 0.75)
    ax[1].set_xlabel("Median log$_{10}$ σ  (S cm$^{-1}$)")
    ps.add_panel_label(ax[0], "a"); ps.add_panel_label(ax[1], "b", x=-0.02)
    ps.finalize_figure(fig, os.path.join(FIG, "01_eda.png"))


def plot_parity(y_true, y_pred, cens):
    """Figure 2: held-out-test parity, censored labels drawn distinctly. MAE/R2 are
    recomputed from the arrays, so a replot can never disagree with its own points."""
    m = metrics(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(ps.COL_SINGLE_IN, 3.3))
    lo, hi = y_true.min() - .5, y_true.max() + .5
    ax.plot([lo, hi], [lo, hi], "--", lw=0.8, color=ps.PALETTE["neutral_dark"], zorder=1)
    ax.scatter(y_true[~cens], y_pred[~cens], s=14, alpha=.75, linewidths=0,
               color=ps.PALETTE["blue_main"], label="measured", zorder=3)
    ax.scatter(y_true[cens], y_pred[cens], s=18, facecolors="none", linewidths=0.7,
               edgecolors=ps.PALETTE["neutral_mid"], label="censored (< bound)", zorder=2)
    ax.set_xlabel("True log$_{10}$ σ  (S cm$^{-1}$)")
    ax.set_ylabel("Predicted log$_{10}$ σ  (S cm$^{-1}$)")
    ax.annotate(f"CatBoost\nMAE {m['MAE']:.2f},  R$^2$ {m['R2']:.2f}",
                xy=(0.04, 0.96), xycoords="axes fraction", va="top", ha="left",
                fontsize=ps.FS_ANNOT)
    ax.legend(loc="lower right", fontsize=ps.FS_LEGEND)
    ps.finalize_figure(fig, os.path.join(FIG, "02_parity_test.png"))


def replot_from_source_data():
    """Redraw figures 1 + 2 from the shipped source_data CSVs -- no retraining,
    so the model, metrics.json and every reported number stay byte-identical."""
    ps.apply_publication_style()
    ally = pd.read_csv(os.path.join(SRC, "fig01a_target_distribution.csv"))["log10_sigma"]
    top_fam = pd.read_csv(os.path.join(SRC, "fig01b_family_median.csv"),
                          index_col=0)["median_log10_sigma"]
    plot_eda(ally, top_fam)
    par = pd.read_csv(os.path.join(SRC, "fig02_parity.csv"))
    plot_parity(par["true_log10_sigma"].to_numpy(), par["pred_log10_sigma"].to_numpy(),
                par["censored"].to_numpy().astype(bool))
    print("Replotted figures/01_eda.png + 02_parity_test.png from source_data/ "
          "(no retraining).")


def main():
    ps.apply_publication_style()
    train, test = load_obelix(DATA)
    print(f"Featurized: train={len(train)}  test={len(test)}  "
          f"(censored in train: {int(train['censored'].sum())})")

    cols = feature_columns(train)
    num_cols = [c for c in cols if c not in CAT_FEATURES]
    X_tr, y_tr = train[cols].copy(), train["y"]
    X_te, y_te = test[cols].copy(), test["y"]
    for c in CAT_FEATURES:
        X_tr[c] = X_tr[c].astype(str)
        X_te[c] = X_te[c].astype(str)

    print("Cross-validation on official TRAIN split:")
    cv_cat = cv_eval("CatBoost", make_cat, X_tr, y_tr, CAT_FEATURES, True)
    cv_rf = cv_eval("RandomForest", lambda: make_rf(num_cols, CAT_FEATURES),
                    X_tr, y_tr, CAT_FEATURES, False)

    # Refit on full train, evaluate once on the held-out TEST split
    cat = make_cat(); cat.fit(X_tr, y_tr, cat_features=CAT_FEATURES)
    rf = make_rf(num_cols, CAT_FEATURES); rf.fit(X_tr, y_tr)
    pred_cat, pred_rf = cat.predict(X_te), rf.predict(X_te)
    test_cat, test_rf = metrics(y_te, pred_cat), metrics(y_te, pred_rf)
    print("Held-out TEST:")
    print("  [CatBoost]    ", {k: round(v, 3) for k, v in test_cat.items()})
    print("  [RandomForest]", {k: round(v, 3) for k, v in test_rf.items()})

    cat.save_model(os.path.join(HERE, "catboost_model.cbm"))
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"n_train": len(train), "n_test": len(test),
                   "units": "log10(S/cm)",
                   "cv": {"CatBoost": cv_cat, "RandomForest": cv_rf},
                   "test": {"CatBoost": test_cat, "RandomForest": test_rf}},
                  f, indent=2)

    os.makedirs(SRC, exist_ok=True)

    # --- Figure 1: target distribution (EDA), panels a (histogram) + b (by family) ---
    ally = pd.concat([train["y"], test["y"]])
    top_fam = train.groupby("Family")["y"].median().sort_values().iloc[-10:]
    plot_eda(ally, top_fam)
    ally.rename("log10_sigma").to_csv(
        os.path.join(SRC, "fig01a_target_distribution.csv"), index=False)
    top_fam.rename("median_log10_sigma").to_csv(
        os.path.join(SRC, "fig01b_family_median.csv"))

    # --- Figure 2: parity (held-out test); censored labels drawn distinctly ---
    plot_parity(y_te.to_numpy(), np.asarray(pred_cat),
                test["censored"].to_numpy().astype(bool))
    pd.DataFrame({"true_log10_sigma": y_te.to_numpy(), "pred_log10_sigma": pred_cat,
                  "censored": test["censored"].to_numpy()}).to_csv(
        os.path.join(SRC, "fig02_parity.csv"), index=False)
    print("Saved: data/metrics.json, catboost_model.cbm, "
          "figures/01_eda.png, figures/02_parity_test.png, source_data/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replot", action="store_true",
                    help="redraw figure 1 from source_data/ without retraining")
    if ap.parse_args().replot:
        replot_from_source_data()
    else:
        main()
