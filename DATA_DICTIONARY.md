# Data dictionary

## Frozen inputs

### `data/train.csv` and `data/test.csv`

Vendored OBELiX official splits. Key source columns are:

| Column | Meaning |
|---|---|
| `ID` | OBELiX record identifier |
| `Reduced Composition` | reported reduced or nominal formula |
| `True Composition` | expanded composition where supplied |
| `Ionic conductivity (S cm-1)` | numeric value or one-sided bound |
| `Space group #`, `a`–`gamma`, `Z` | reported crystallographic metadata |
| `DOI` | source publication identifier |

Derived targets, canonical keys, and censor fields are created in memory and in
panel source tables; the original CSV columns are preserved. `feature_formula`
and `feature_composition_key` define the model/grouping identity;
`nominal_composition_key` preserves the reduced-formula identity, and
`composition_key_mismatch` exposes disagreements between them.

### `data/mp_snapshot.csv`

Minimal scalar snapshot of the frozen 248-entry MP query.

| Column | Unit / definition |
|---|---|
| `database_version` | MP database version |
| `material_id` | unique MP identifier |
| `formula` | MP pretty formula |
| `composition_key` | scale/order-invariant atomic-fraction key |
| `elements` | semicolon-separated element symbols |
| `n_elements` | number of elements |
| `spacegroup_no`, `crystal_system` | MP symmetry summary |
| `e_above_hull` | eV atom-1 |
| `band_gap` | eV |
| `theoretical` | MP theoretical flag |
| `icsd_ids`, `pauling_ids` | catalogue identifiers, if present |
| `last_updated` | upstream timestamp |

`data/mp_query.json` records query filters, retrieval time, database version,
and returned count. No structure is distributed.

## Primary outputs

### `mp_ledger.csv`

One row for every MP entry. It contains the frozen metadata plus:

| Column | Definition |
|---|---|
| `disposition` | one of four exhaustive scope states |
| `disposition_reason` | human-readable rule result |
| `ranking_score` | mean prediction of 15 production members |
| `ensemble_spread` | population SD across members |
| `ad_distance` | five-neighbour distance in PCA descriptor space |
| `screen_rank` | rank among all 248 entries, score descending |
| `evidence_state` | candidate state or `not_applicable` |

### `candidate_queue.csv`

One row per normalized screen-candidate composition (62 rows).

| Column | Definition |
|---|---|
| `candidate_rank` | score-descending composition rank |
| `composition_key` | canonical composition identifier |
| `formula`, `material_id` | deterministic representative entry |
| `spacegroup_no`, `crystal_system` | representative symmetry |
| `ranking_score`, `ensemble_spread`, `ad_distance` | statistical outputs |
| `n_mp_entries` | MP entries with this composition |
| `min_e_above_hull`, `max_band_gap` | composition aggregate |
| `any_icsd_match` | catalogue-presence cue, not novelty |
| `any_non_theoretical_entry` | whether any grouped MP entry is not flagged theoretical |
| `evidence_state` | q75 descriptive state |
| `score_percentile`, `spread_percentile`, `ad_percentile` | within-queue ranks |
| `combined_risk_percentile` | max of spread and AD percentiles |
| `member_rank_q25`, `member_rank_median`, `member_rank_q75` | member-rank quartiles across the 15-member ensemble |
| `member_rank_iqr` | `member_rank_q75 − member_rank_q25` |
| `top_quartile_frequency` | fraction of members placing the composition in the top quartile |

## Evaluation outputs

- `data/metrics.json`: split audit; grouped-CV summaries; strict and
  composition-novel and target-domain test metrics; matched strict baselines;
  paired and target-domain bootstrap intervals; training/censor sensitivities;
  risk correlations and their composition-cluster bootstrap intervals;
  representation and cell-convention audits. The
  `risk_correlation_bootstrap` object records the 4,000-resample seed, grouping
  unit, percentile definition, and all 16 interval estimates used in Figure 3d.
- `data/model_manifest.json`: production representation, feature names, AD
  parameters, evidence-state thresholds, q70/q75/q80 route-sensitivity counts,
  and all 15 model checksums.
- `data/release_manifest.json`: reviewed environment, artifact contracts,
  sizes, and SHA-256 hashes written only by the explicit `p1screen manifest`
  release step.

## Panel source tables

`source_data/` contains only the CSVs listed in `FIGURES.md`: the main-figure
sources, the two Supplementary Figure S1 sources, and two tables released as
supplementary data that no figure reads (`fig02f_training_sensitivity.csv`,
`fig03d_oof_risk.csv`). Each is a tidy, headered table with no implicit index. `fig04_screen_ledger.csv` and
`fig05_candidate_queue.csv` intentionally duplicate primary outputs so a figure
can be regenerated from its declared source-data directory alone.
`fig01c_mp_query_contract.csv` is generated directly from `data/mp_query.json`
and supplies every query value displayed in Figure 1c.

`fig03c_risk_correlations.csv` has one row for each sample–signal combination
(four samples × four signals = 16 rows):

| Column | Definition |
|---|---|
| `sample` | `oof`, `test`, `oof_target`, or `test_target` |
| `signal` | spread–error, AD–error, combined-percentile–error, or spread–AD association |
| `spearman_rho` | unchanged point estimate from the frozen predictions |
| `ci95_low`, `ci95_high` | 2.5th and 97.5th percentiles of the composition-cluster bootstrap distribution |
| `n_records`, `n_compositions` | records and independent composition groups in the sample |
| `n_bootstrap` | 4,000 resamples |
| `rank_normalization` | `within_fold` for OOF samples or `within_sample` for strict-test samples |

The global OOF, strict-test, target-OOF, and target-test samples contain
478/390, 121/112, 117/97, and 24/23 records/compositions, respectively.

### Supplementary Figure S1 sources

`figS1a_boosting_curves.csv` has one row per configuration, grouped fold, and
boosting iteration (3 configurations × 5 folds × 400/600/800 iterations =
9,000 rows):

| Column | Definition |
|---|---|
| `config` | `compact`, `base`, or `deep_slow` |
| `fold` | grouped-CV fold, 1–5 |
| `iteration` | boosting iteration, 1 to the configuration's tree count |
| `train_rmse`, `validation_rmse` | RMSE on the fold's training and validation records after that iteration |
| `n_train`, `n_validation` | records in the fold split |

`figS1b_learning_curve.csv` has one row per composition-grouped subsample of
the strict training set (nine fractions × five seeds + the full set = 46 rows):

| Column | Definition |
|---|---|
| `fraction` | fraction of the 388 strict-training compositions sampled |
| `repeat_seed` | subsampling seed (0–4; the full set uses seed 0 only) |
| `n_train_records`, `n_train_compositions` | records and compositions in the subsample |
| `MAE`, `RMSE`, `R2`, `Spearman` | strict-test metrics of the single base CatBoost fitted on the subsample |
| `MAE_target_domain` | MAE on the 24 target-domain test records |

## Missing values and identifiers

Blank catalogue identifiers mean “not present in the retrieved MP metadata,”
not “material is novel.” Numerical missingness is median-imputed only inside a
fitted model or AD pipeline; raw frozen tables are not globally imputed.
