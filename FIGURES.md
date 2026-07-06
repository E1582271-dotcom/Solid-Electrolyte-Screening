# Figure captions & source data

Publication-style captions for the figures in [`figures/`](figures/). Each figure is
exported as a **600 dpi PNG** (Arial); its underlying numbers are in
[`source_data/`](source_data/). Styling is centralised in [`src/plotstyle.py`](src/plotstyle.py)
— add `"svg"`/`"pdf"` to `finalize_figure`'s `formats` for editable vector export.

## The story (why each figure exists)

What are we working with, and how good is a small-data tree model, honestly? (Fig. 1–2) →
what does the model actually key on? (Fig. 3, SHAP) → does that generalise as a chemical
sanity check on known materials? (Fig. 4) → apply it for real, to a live Materials Project
query (Fig. 5) → don't trust the raw screen — audit it against the literature (Fig. 6) →
what does the audited pool still not know about itself? (Fig. 7, the `Family=unknown`
transparency gap) → and the sharpest question of all: does the model know when it's
guessing? (Fig. 8) — the answer is a caution, not a reassurance: variance-based
uncertainty is confidently wrong on the one candidate (LiPS₃) that Project 2's MLIP-MD
later confirmed as a strong conductor, which is exactly why the funnel needs Project 2's
more expensive second layer rather than stopping here.

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

**Figure 6 | Manual adversarial audit of the top Materials Project hits.**
**a**, Funnel from the initial loose query (n = 328) to the audited, filter-tightened pool
(n = 184). Only the 64 redox-active transition-metal sulfides removed are separately
quantified (`screen_audit.md`); the remaining ~80 were dropped by the H-exclusion and
≥ 3-element filters combined, not logged separately in this run — stated on the panel, not
just in this caption. **b**, Verdict breakdown of the 15 hits manually audited against the
literature: 7 kept (5 known true positives + 2 literature-blank novel leads) and 8 dropped
across three failure modes (redox-active/cathode false positives, a binary-precursor false
positive, and H-containing hydrate/oxysalt artifacts). Source data:
`source_data/fig06a_screen_audit_funnel.csv`, `source_data/fig06b_screen_audit_verdicts.csv`.

**Figure 7 | Family coverage of the final MP candidate pool.**
`Family` — the single most important SHAP feature (Fig. 3a, mean|SHAP| = 0.32) — is
unassigned (`unknown`) for 144/184 (78.3%) of the screened candidates, a direct
consequence of the transparent stoichiometry heuristic used to label MP entries (which
have no native `Family` field). Note: one of the two Fig. 6b "novel leads",
Li₂₀Si₃P₃S₂₃Cl, is counted under **LGPS** here, not `unknown` — the heuristic correctly
recognises its Ge/Si/Sn-free LGPS-type stoichiometry; "novel" (no literature-reported
conductivity) and "family known" are independent axes, easy to conflate. Source data:
`source_data/fig07_family_coverage.csv`.

**Figure 8 | Does the screen know when it is guessing? Ensemble uncertainty vs the MD verdicts.**
**a**, Calibration of a 10-seed CatBoost ensemble (config identical to the shipped model,
only the seed varies) on the held-out OBELiX test split: within-bin RMSE of the ensemble
mean rises monotonically with the ensemble spread (Spearman ρ = 0.46, p ≈ 1e-7, n = 121)
— in-distribution, the spread is informative. **b**, Risk map (ensemble mean vs spread)
for the 184-candidate MP pool (dots: family assigned; hollow: `Family = unknown`) with the
four W11 MD-validated leads retro-scored on the same primitive cells the funnel originally
saw (stars; teal = MD-confirmed conductor, red = MD-falsified; label color matches).
The ranking-inverting lead LiPS₃ (measured ≈ 10 mS/cm after a prior of −7.1) sits in the
pool's bottom decile of spread (7th percentile) — the ensemble is confidently wrong, so
variance-based uncertainty is not a safe routing signal for out-of-distribution generated
candidates. Source data: `source_data/fig08a_calibration.csv`,
`source_data/fig08b_risk_map.csv`; metrics: `data/uncertainty.json`.
