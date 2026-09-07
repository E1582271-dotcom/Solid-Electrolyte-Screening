# Methods

## Scope and estimand

Project 1 estimates a composition-level ranking prior for ionic conductivity in
a frozen, explicitly filtered Li–S Materials Project population. The estimand is
relative ranking under the OBELiX evidence distribution. The model is not
intended to estimate intrinsic bulk conductivity with calibrated physical
uncertainty.

## OBELiX ingestion and target

The vendored OBELiX official splits contain 478 training and 121 test records.
The target is `log10(ionic conductivity / S cm-1)`. Strings beginning with `<`
or `>` are parsed into a positive numerical bound and a separate censor
direction. All 37 censored values are upper bounds (`<`); none is discarded.
The primary fit uses the numerical bound as its training label. Sensitivity
analyses both exclude censored training records and score upper-bound test
records with the one-sided error `max(prediction - bound, 0)`.

Formulas are parsed with pymatgen. Whitespace and hydrate separators are
normalized without changing stoichiometry. Model features and grouping use the
OBELiX `True Composition` (with `Reduced Composition` as an explicit fallback),
while a separate nominal key preserves the reported reduced formula for
catalogue matching. Both keys use sorted elemental atomic fractions rounded to
eight decimal places and are audited for disagreement.

DOIs are lower-cased and stripped of resolver prefixes. The official split has
zero DOI overlap but two normalized-composition overlaps. For the strict test,
the two overlapping training rows are removed; the complete 121-record test is
retained. A 119-record composition-novel test subset is reported separately.

## Representations

M0 contains 132 Magpie elemental-property statistics plus number of elements
and Li atomic fraction (134 numerical features). These descriptors require only
a formula and define the production experiment-to-MP contract.

M1 adds scale-invariant cell summaries: volume per atom
`V/atoms_in_true_unit_cell` when `True Composition` is present (otherwise
`V/(Z * atoms_in_reduced_formula)`), mean-atomic-mass/volume density proxy,
sorted axis ratios, sorted angle cosines, space-group number, and crystal system
(143 total features). M2 adds the legacy raw lattice lengths, angles, volume,
space-group number, Z, and crystal system (144 total features). M1 and M2 are
diagnostic ablations and never score MP formulas.

## Models and grouped evaluation

All outer CV comparisons use the same five stratified composition-grouped folds
within the official training split. Conductivity quantile bins provide the
stratification label; canonical composition keys are the grouping variable.

Baselines are a training-median regressor, ridge regression with inner
four-fold grouped selection of alpha, and a 500-tree random forest. The primary
model is a 15-member CatBoost perturbation ensemble:

| Configuration | Trees | Depth | Learning rate | Seeds |
|---|---:|---:|---:|---:|
| compact | 400 | 5 | 0.05 | 0–4 |
| base | 600 | 6 | 0.05 | 0–4 |
| deep_slow | 800 | 7 | 0.03 | 0–4 |

The prediction is the member mean; population standard deviation across members
is the ensemble-spread signal. Configuration and seed variation are both
included so spread is not merely a fixed-hyperparameter seed perturbation.

The strict test model is fitted on 476 decontaminated training records. Metrics
are MAE, RMSE, R², Spearman ρ, Kendall τ, and mean signed error. A 4,000-resample
cluster bootstrap samples test composition groups with replacement and reports
95% intervals. Matched strict-test predictions are generated for the median,
ridge, random forest, one base CatBoost, and the perturbation ensemble. Paired
cluster bootstraps report ensemble-minus-comparator MAE differences.

The predefined target domain contains records whose feature composition has Li
and S, no H or O, three to five elements, and none of the configured redox-risk
elements. Full and target-domain results are reported separately. Two training
sensitivities re-fit the perturbation ensemble with equal total weight per
composition and after excluding censored records; neither is selected using the
held-out test.

After evaluation is frozen, production ensemble members are fitted on all 599
records. Serialized model checksums and feature names are stored in
`data/model_manifest.json`.

### Supplementary training diagnostics

Two descriptive tables accompany the main evaluation and are rendered only in
Supplementary Figure S1. Boosting curves refit each of the three configurations
once per grouped fold (seed 42, `use_best_model=False` so the full trajectory is
kept) and record training and validation RMSE after every iteration. The
learning curve subsamples the 476-record strict training set by whole
composition groups at fractions 0.1–1.0 of its 388 compositions (five
subsampling seeds per fraction below 1.0), fits the single base CatBoost, and
scores the complete 121-record strict test. These 61 fits are diagnostics only:
they are not used for model selection, thresholds, or any candidate disposition.

## Applicability domain

M0 features are median-imputed and standardized using the relevant training
set. Full-SVD PCA retains 95% variance. The AD distance of a query is its mean
Euclidean distance to the five nearest training records in PCA space. Training
distances exclude self-neighbours. AD is fitted only on training evidence and
is not optimized against candidate rankings.

For risk analysis, ensemble spread and AD distance are compared with absolute
error using Spearman correlation in grouped out-of-fold predictions and the
strict test, both globally and in the predefined target domain. OOF percentile
ranks are normalized within fold before pooling. Their combined signal is the
product of the two percentile ranks; strict-test percentiles are normalized
within the evaluated sample. Uncertainty in each association is estimated with
a composition-cluster bootstrap (4,000 resamples; seed 20260716). Each replicate
samples the unique compositions with replacement, retains every record in each
sampled composition, recomputes the relevant ranks, and then recomputes Spearman
rho. The reported 95% interval is the 2.5th–97.5th percentile range. The global
OOF, strict-test, target-OOF, and target-test samples contain 478/390, 121/112,
117/97, and 24/23 records/compositions, respectively. These analyses are
associative, not error calibration; spread and AD are themselves strongly
correlated.

## Frozen Materials Project population

The reviewed query for Materials Project database version `2026.04.13` requires
Li and S, excludes H, contains 3–5 elements, limits energy above hull to
0–0.05 eV atom-1, and requires band gap >=1.5 eV. The query returns 248 unique
material IDs. Only minimal scalar metadata are retained; structures are not
distributed.

Each entry receives the first applicable disposition:

1. `oxygen_scope_exclusion` if O is present;
2. `redox_electronic_risk` if the oxygen-free formula contains V, Cr, Mn, Fe,
   Co, Ni, Cu, Mo, or W;
3. `obelix_composition_reference` if its canonical composition occurs in
   OBELiX;
4. `screen_candidate` otherwise.

This ordering is explicit and exhaustive. It produces 80, 19, 70, and 79
entries, respectively. Candidate entries are grouped by canonical composition.
The resulting multiplicity is 49 compositions with one MP structure, nine with
two, and four with three, giving 79 structure entries across 62 compositions.
The lowest-energy-above-hull entry (then lexicographically smallest material ID)
is the representative; entry multiplicity, minimum hull energy, maximum gap,
and whether any ICSD ID exists are aggregated. No ICSD field is interpreted as
a novelty label.

## Candidate evidence states

For the 62 unique candidate compositions, q75 thresholds are computed for
ranking score, ensemble spread, and AD distance. A composition is
`extrapolative` when spread or AD is at/above q75. Among the remainder, it is
`high_score_lower_risk` when score is at/above q75; all others are `unflagged`.
The same calculation is repeated at q70 and q80 for sensitivity, with all three
count partitions stored in `data/model_manifest.json`. The labels are descriptive
evidence states and do not define follow-up priority. Member-wise candidate ranks
are summarized by q25, median, q75, their interquartile range, and frequency of
appearing in the top quartile; these fields expose ranking stability without
converting it into a decision rule.

## Figures and reproducibility

All panels are rendered by Matplotlib from frozen source tables. A shared
Nature-informed style sets a 183 mm double-column width, a maximum 170 mm height,
Arial-first sans-serif typography, 8 pt bold panel letters, 7 pt panel titles,
5–7 pt supporting text, accessible colours, redundant marker shapes, frameless
legends, and no background grid. Files are saved at 900 dpi, converted through
Pillow to RGB on a pure-white canvas, and written only as PNG. This author-chosen
PNG contract provides visual consistency but does not replace the editable
vector source required for final Nature production. The release gate
independently validates the image inventory, pixel dimensions, DPI metadata,
colour mode, corner pixels, source tables, data counts, model checksums, and
tracked-file credential policy.

All seeds, model configurations, query filters, feature names, data versions,
and SHA-256 hashes are machine-readable in `src/p1screen/config.py` and `data/`.

## Excluded evidence

Project 1 includes no custom DFT, phonons, molecular dynamics, MLIP-MD,
synthesis, electrochemistry, or wet experiments. Such evidence must be reported
and evaluated independently and cannot be used to redefine this screen after
the fact.
