"""
Project 1 -- SHAP interpretation of the CatBoost ionic-conductivity model.

Uses CatBoost's native ShapValues (handles categorical features correctly), then
renders a single Nature-style figure with two panels:
  - a: mean(|SHAP|) ranking of the top-15 features (incl. categoricals)
  - b: SHAP beeswarm for the top-15 numeric features (colour = feature value)

The beeswarm is drawn directly with matplotlib (no `shap` package needed -- the SHAP
values come from CatBoost), so each panel owns its axes and there is no cross-bleed.

Run after 01_train_eval.py:
    python 02_shap.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src.featurize import load_obelix, feature_columns, CAT_FEATURES, pretty_feature
from src import plotstyle as ps
from catboost import CatBoostRegressor, Pool

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
SRC = os.path.join(HERE, "source_data")


def _beeswarm_offsets(x, width=0.42, nbins=100):
    """Violin-like vertical offsets: points in denser SHAP-value bins spread wider.
    Deterministic (no RNG), so the figure is reproducible."""
    x = np.asarray(x, float)
    off = np.zeros_like(x)
    if x.max() <= x.min():
        return off
    bins = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.digitize(x, bins) - 1, 0, nbins - 1)
    n = len(x)
    for b in np.unique(idx):
        pts = np.where(idx == b)[0]
        k = len(pts)
        if k > 1:
            spread = width * min(1.0, k / (0.03 * n + 1))
            off[pts] = np.linspace(-spread, spread, k)
    return off


def _beeswarm(ax, shap_sub, feat_vals, labels, cmap):
    """Manual SHAP beeswarm: one row per feature, colour = per-feature normalised value."""
    n = len(labels)
    sc = None
    for j in range(n):
        yv = n - 1 - j                       # largest-|SHAP| feature at the top row
        sv = shap_sub[:, j]
        fv = feat_vals[:, j].astype(float)
        lo, hi = np.nanpercentile(fv, 5), np.nanpercentile(fv, 95)
        norm = np.clip((fv - lo) / (hi - lo), 0, 1) if hi > lo else np.full_like(fv, 0.5)
        sc = ax.scatter(sv, yv + _beeswarm_offsets(sv), c=norm, cmap=cmap, vmin=0, vmax=1,
                        s=5, alpha=0.75, linewidths=0, rasterized=True)
    ax.axvline(0, color=ps.PALETTE["neutral_mid"], lw=0.6, zorder=0)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels[::-1])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("SHAP value  (impact on log$_{10}$ σ)")
    return sc


def main():
    ps.apply_publication_style()
    os.makedirs(SRC, exist_ok=True)
    train, _ = load_obelix(DATA)
    cols = feature_columns(train)
    X = train[cols].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    y = train["y"]

    model = CatBoostRegressor()
    model.load_model(os.path.join(HERE, "catboost_model.cbm"))

    pool = Pool(X, y, cat_features=CAT_FEATURES)
    shap_vals = model.get_feature_importance(pool, type="ShapValues")[:, :-1]  # drop base col
    mean_abs = np.abs(shap_vals).mean(axis=0)

    top_all = np.argsort(mean_abs)[::-1][:15]                     # bar: all features
    num_order = sorted((i for i in range(len(cols)) if cols[i] not in CAT_FEATURES),
                       key=lambda i: mean_abs[i], reverse=True)[:15]  # beeswarm: numeric

    cmap = plt.get_cmap("viridis")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(ps.COL_DOUBLE_IN, 4.4))

    # panel a: mean(|SHAP|) bar (top-15 features, categoricals included)
    axA.barh([pretty_feature(cols[i]) for i in top_all][::-1],
             mean_abs[top_all][::-1], color=ps.PALETTE["blue_main"])
    axA.set_xlabel("mean(|SHAP|)  (log$_{10}$ σ)")

    # panel b: beeswarm (top-15 numeric features; colour = feature value)
    sc = _beeswarm(axB, shap_vals[:, num_order],
                   X[[cols[i] for i in num_order]].to_numpy(dtype=float),
                   [pretty_feature(cols[i]) for i in num_order], cmap)
    cbar = fig.colorbar(sc, ax=axB, pad=0.02, fraction=0.045)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=ps.FS_ANNOT)
    cbar.ax.tick_params(labelsize=ps.FS_TICK, length=0)
    axB.annotate("numeric features only", xy=(0.5, 1.01), xycoords="axes fraction",
                 ha="center", va="bottom", fontsize=ps.FS_ANNOT, color=ps.PALETTE["neutral_mid"])

    ps.add_panel_label(axA, "a"); ps.add_panel_label(axB, "b", x=-0.02)
    ps.finalize_figure(fig, os.path.join(FIG, "03_shap.png"))

    rank = np.argsort(mean_abs)[::-1]
    pd.DataFrame({"feature": [cols[i] for i in rank],
                  "label": [pretty_feature(cols[i]) for i in rank],
                  "mean_abs_shap": mean_abs[rank]}).to_csv(
        os.path.join(SRC, "fig03_shap_mean_abs.csv"), index=False)
    print("Top features:", [cols[i] for i in top_all[:8]])
    print("Saved: figures/03_shap.png, source_data/fig03_shap_mean_abs.csv")


if __name__ == "__main__":
    main()
