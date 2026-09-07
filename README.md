# Project 1 — Auditable formula-first screening of Li–S solid-electrolyte space

Project 1 turns 599 experimental ionic-conductivity records into a reproducible,
composition-first ranking of a frozen Materials Project (MP) search space. The
release is designed as an auditable statistical-screening study: it exposes split
overlap, censoring, representation transfer, applicability domain, empirical
risk heuristics, and the disposition of every queried MP entry.

The output is a **ranking and evidence-state map**, not a claim of ionic
conductivity, novelty, synthesizability, or mechanism. No named “hero” candidate
is promoted. MLIP-MD, new DFT calculations, synthesis, and wet experiments are
explicitly outside Project 1 and belong in separate projects.

![Study design](figures/01_study_design.png)

## What the rebuilt project establishes

- OBELiX contributes 478 official training and 121 held-out records. Canonical,
  scale-invariant composition keys reveal two compositions shared across the
  official split. Strict held-out evaluation therefore removes the corresponding
  two training rows and uses 476 train / 121 test records; DOI overlap is zero.
- All 37 one-sided conductivity upper bounds (29 train, 8 test) are retained.
  The held-out point-MAE and one-sided censor-aware MAE are both 1.353 log units
  for this release.
- The primary representation contains 132 Magpie descriptors, number of
  elements, and Li atomic fraction (134 formula-only features). In matched
  five-fold composition-grouped CV, the 15-member perturbation ensemble reaches
  MAE 0.843 ± 0.136, RMSE 1.381 ± 0.176, and Spearman ρ 0.882 ± 0.012.
- On the strict held-out test, MAE is 1.353, RMSE 1.947, R² 0.411, Spearman
  ρ 0.660, and mean bias −0.363 log units. These errors preclude interpreting
  predictions as quantitative conductivity measurements. The predefined
  Li–S target-domain subset is smaller and weaker (24 records; MAE 1.585,
  Spearman ρ 0.375; bootstrap ρ interval −0.255 to 0.881).
- Matched strict-test baselines are reported rather than inferred from CV. The
  random forest and single base CatBoost have MAE 1.293 and 1.291,
  respectively, versus 1.353 for the perturbation ensemble. The paired
  cluster-bootstrap ensemble-minus-single-model difference is 0.062
  (95% interval 0.020–0.106), so no strict-test ensemble advantage is claimed.
- Structural ablations are evaluated on identical folds. Raw cell conventions
  vary strongly across repeated compositions; even scale-invariant normalization
  leaves a residual outlier. Formula-only M0 is therefore the production
  representation because it is exactly computable for both experiments and MP.
- The frozen MP query uses database version `2026.04.13` and returns 248 unique
  entries. Every entry receives exactly one disposition: 80 oxygen-scope
  exclusions, 19 redox/electronic-risk entries, 70 OBELiX-composition references,
  and 79 candidate entries.
- The 79 candidate structure entries collapse to a complete queue of 62 unique
  normalized compositions: 49 compositions have one MP structure, nine have
  two, and four have three. At the predefined q75 descriptive thresholds, 27 are
  extrapolative, 12 are high-score/lower-risk, and 23 are unflagged. These states
  describe evidence, not experimental or computational priority.
- Ensemble spread and applicability-domain (AD) distance are related empirical
  risk heuristics, not calibrated uncertainty. Their associations with error are
  reported for both the full strict test and the smaller Li–S target-domain subset;
  in the latter, all three correlations are weak and statistically inconclusive.

## Five-figure evidence chain

Five main figures are exported to `figures/`; one supplementary figure
(boosting and learning curves) is exported to `figures/supplementary/` under
the same contract. Every file is a 900 dpi, RGB, pure-white,
grid-free PNG with a 183 mm double-column width and height no greater than
170 mm. A shared Nature-informed visual system controls Arial-first typography,
accessible colours, redundant marker shapes, evidence hierarchy, and panel
spacing. No PDF, SVG, TIFF, or alpha channel is produced under the selected
output contract; consequently, these files are visually prepared for manuscript
review but are not a substitute for Nature's final editable-vector source.
Stand-alone legends and panel-to-source mappings are in [FIGURES.md](FIGURES.md).

1. **Study design and accounting** — measured evidence, split correction, and
   exhaustive 248-entry disposition.

   ![Figure 1](figures/01_study_design.png)

2. **Model validation** — target coverage, grouped CV, strict test, censoring,
   and cluster-bootstrap uncertainty.

   ![Figure 2](figures/02_model_validation.png)

3. **Transfer and risk** — representation ablation, cell-convention audit,
   ensemble spread, and applicability domain.

   ![Figure 3](figures/03_transfer_and_risk.png)

4. **Global screening landscape** — all 248 MP entries in descriptor, score,
   rank, and risk views without silent removal.

   ![Figure 4](figures/04_screening_landscape.png)

5. **Complete candidate atlas** — all 62 compositions, their raw metrics,
   evidence states, and rank stability across all 15 ensemble members.

   ![Figure 5](figures/05_candidate_atlas.png)

## Scientific workflow

```text
OBELiX 599 records
  ├─ canonical composition + DOI + censor audit
  ├─ composition-grouped CV on official train
  └─ strict held-out test after two-row decontamination
                │
                ▼
formula-only 15-member perturbation ensemble
  ├─ seed + model-configuration variation
  └─ PCA / 5-nearest-neighbour applicability domain
                │
                ▼
frozen MP query: 248 entries
  ├─ 80 oxygen-scope exclusion
  ├─ 19 redox/electronic risk
  ├─ 70 OBELiX-composition reference
  └─ 79 candidate structure entries → 62 normalized compositions
       (49 × 1 structure, 9 × 2 structures, 4 × 3 structures)
```

The MP chemical and thermodynamic filters define the population being ranked;
they are not learned features. The production ensemble is trained on all 599
records only after evaluation is complete.

## Installation and commands

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

p1screen evaluate
p1screen training-curves
p1screen screen
p1screen figures
p1screen verify --frozen
```

The wheel contains the Python workflow, not the writable research payload. For
a non-editable installation, point it at an extracted Project 1 release before
running data commands:

```bash
export P1SCREEN_ROOT=/path/to/project1_screening
```

To rebuild every derived artifact from the vendored OBELiX splits and frozen MP
snapshot:

```bash
p1screen rebuild
```

Refreshing MP is an explicit, reviewed operation because database content can
change:

```bash
export MP_API_KEY='...'
p1screen refresh-mp
p1screen rebuild
```

The refresh command accepts the key only through `MP_API_KEY`, requires the
reviewed database version and 248-entry count, and stores scalar metadata only.
No crystal structures or credentials are distributed.

The numbered Python files remain thin compatibility wrappers around the same
CLI. `p1screen` is the authoritative interface.

## Repository contract

```text
src/p1screen/              package: data, features, models, screen, figures, gate
data/train.csv             vendored OBELiX official training split
data/test.csv              vendored OBELiX official test split
data/mp_snapshot.csv       frozen 248-entry minimal MP metadata snapshot
data/metrics.json          evaluation results and audit statistics
data/model_manifest.json   15 member definitions and SHA-256 checksums
data/release_manifest.json reviewed offline release checksums
artifacts/models/          15 serialized production ensemble members
mp_ledger.csv              exhaustive entry-level 248-row ledger
candidate_queue.csv        complete 62-composition candidate table
source_data/               panel-ready quantitative CSVs
figures/                   five main 900 dpi RGB PNG files
figures/supplementary/     Supplementary Fig. S1 (boosting and learning curves)
tests/                     unit and frozen-contract tests
```

The offline gate rejects stale figure/source inventories, missing candidates,
checksum mismatches, non-RGB images, non-white canvases, wrong DPI, database
drift, incomplete dispositions, and credential-like tracked files.

## Interpretation limits

- OBELiX is small and heterogeneous; measurement temperature, processing,
  microstructure, disorder, and interface effects are not uniformly represented.
- A formula-only model cannot resolve polymorphs, defects, phase mixtures,
  electronic leakage, or kinetic stability.
- The held-out MAE is approximately 1.35 orders of magnitude. The MP output must
  be used for population ranking, not conductivity prediction in physical units.
- In the 24-record target-domain subset, MAE rises to approximately 1.58 and R²
  is negative; its broad bootstrap intervals make transfer conclusions tentative.
- Ensemble spread and AD distance identify relative risk but are not calibrated
  confidence intervals and cannot guarantee error out of domain.
- The oxygen and redox/electronic rules are explicit scope filters, not universal
  chemical impossibility statements.
- An ICSD identifier is a provenance cue only. No novelty claim is made.
- No downstream DFT, molecular dynamics, synthesis, or wet-experiment outcome is
  used to evaluate or retroactively tune Project 1.

## Data, citation, and license

OBELiX data are credited to Therrien *et al.*, *Digital Discovery* (2026),
[doi:10.1039/D5DD00441A](https://doi.org/10.1039/D5DD00441A), and retain their
CC BY 4.0 terms. Materials Project metadata must be cited using the references
listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Repository code is
MIT licensed; data and third-party metadata are not relicensed by that grant.

See [METHODS.md](METHODS.md), [DATA_DICTIONARY.md](DATA_DICTIONARY.md),
[REPRODUCIBILITY.md](REPRODUCIBILITY.md), and [CITATION.cff](CITATION.cff).
