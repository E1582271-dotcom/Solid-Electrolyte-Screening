# Figure captions & source data

Publication-style captions for the figures in [`figures/`](figures/). Each figure is
exported as a **600 dpi PNG** (Arial); its underlying numbers are in
[`source_data/`](source_data/). Styling is centralised in [`src/plotstyle.py`](src/plotstyle.py)
— add `"svg"`/`"pdf"` to `finalize_figure`'s `formats` for editable vector export.

**Model / evaluation (applies to Figs 2–3).** Model: CatBoost regressor (600 trees,
depth 6, learning rate 0.05). Features: hand-built *Magpie-lite* composition descriptors
(fraction-weighted mean/SD/min/max/range of element properties) + crystallography
(lattice, angles, cell volume, space group) + two categoricals (structural family, crystal
system). Target: `log₁₀(σ / S cm⁻¹)`. Split: OBELiX **official** 478 train / 121 test.
Cross-validation: 5-fold on train. Metrics: MAE and R² on `log₁₀ σ`. SHAP: CatBoost native
`ShapValues` (exact for tree ensembles, categorical-aware).

---

**Figure 1 | OBELiX conductivity distribution and family ranking.**
**a**, Histogram of `log₁₀ σ` for all n = 599 entries (478 train + 121 test); the long
low-σ tail and the pile-up at the `1×10⁻¹⁰` censoring bound are visible. **b**, Median
`log₁₀ σ` by structural family (top 10 by median), showing the superionic families (LGPS,
argyrodites) at the high-σ end. Source data: `source_data/fig01a_target_distribution.csv`,
`source_data/fig01b_family_median.csv`.

**Figure 2 | Held-out test parity.**
CatBoost predictions vs. measured `log₁₀ σ` on the official test split (n = 121); dashed
line is identity. Filled points are measured values; open gray circles are **censored**
labels (`<` bound, parsed to their numeric bound and flagged), which is why they line up at
the `1×10⁻¹⁰` bound. Test MAE = 1.30, R² = 0.44 (CV R² = 0.76 → test 0.44 gap is reported,
not hidden). Source data: `source_data/fig02_parity.csv`.

**Figure 3 | SHAP interpretation of the CatBoost model.**
**a**, Mean(|SHAP|) ranking of the top-15 features (train, n = 478; categoricals included) —
structural family and crystal system dominate, then mean electronegativity and cell
volume/lattice size. **b**, SHAP beeswarm for the top-15 **numeric** features (categoricals
omitted because colour encodes feature value); colour = per-feature value (viridis,
Low→High), x = SHAP value (impact on `log₁₀ σ`). Source data: `source_data/fig03_shap_mean_abs.csv`.

**Figure 4 | Offline demo screen of known sulfide electrolytes.**
Nine well-characterised sulfide solid electrolytes ranked by predicted `log₁₀ σ`, coloured
by heuristic structural family. Sanity check: LGPS and argyrodites rank highest, Li₂S
lowest — matching the SHAP family ranking. Bars are a **rank prior**, not a quantitative σ.
Source data: `source_data/fig04_screen_demo.csv`.

**Figure 5 | Live Materials Project screen of Li–S candidates.**
Top-ranked Li–S candidates (deduplicated by composition) from a blind Materials Project
query (n = 184 screened) after the audited filters (exclude H; band gap ≥ 1.5 eV;
≥ 3 elements; drop redox-active transition-metal sulfides), coloured by heuristic family.
The LGPS family (Li₁₀Ge/Si/Sn(PS₆)₂) tops the list — matching the SHAP ranking; only
families present are shown in the legend. Source data: `source_data/fig05_screen_mp.csv`.
