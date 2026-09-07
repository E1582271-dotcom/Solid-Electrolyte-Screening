# Reproducibility and release procedure

## Frozen offline rebuild

The normal release path does not require network access:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
p1screen rebuild
```

`rebuild` clears stale panel CSVs, recalculates evaluation, computes the
supplementary training diagnostics, fits the production ensemble, rebuilds the
ledger and candidate queue, renders the five main figures and Supplementary
Figure S1, and runs the read-only release-contract gate. The current workflow
performs 151 CatBoost fits across grouped CV, strict-test baselines and
sensitivities, representation ablations, and the production ensemble, plus 61
diagnostic fits for Supplementary Figure S1 (`p1screen training-curves`).

For a faster audit of committed artifacts:

```bash
pytest
p1screen verify --frozen
```

Frozen verification is read-only and compares the current payload with the
reviewed checksum manifest. Updating that manifest is an explicit release step:

```bash
p1screen manifest
```

## Reviewed live MP refresh

```bash
export MP_API_KEY='...'
p1screen refresh-mp
p1screen rebuild
```

The key is read only from the environment. Refresh aborts if the returned
database version is not `2026.04.13`, the count is not 248, or material IDs are
not unique. Any future database release requires intentional review of the
query, counts, dispositions, metrics, figures, documentation, and release
manifest together.

## Determinism

- grouped split seed: 42;
- CatBoost member seeds: 0–4 for each of three fixed configurations;
- random-forest seed: 42;
- composition-cluster bootstrap seed: 20260716;
- bootstrap resamples: 4,000;
- risk-correlation resampling unit: normalized composition;
- OOF risk percentiles: normalized within original fold;
- strict-test risk percentiles: normalized within evaluated sample;
- canonical composition rounding: eight decimals;
- AD: 95% PCA variance and five neighbours;
- candidate-state quantile: q75, with q70/q80 sensitivity;
- supplementary learning-curve subsampling seeds: 0–4 per fraction, by
  composition group; boosting curves use model seed 42.

Floating-point results can vary slightly across BLAS libraries and architectures.
The committed release artifacts and their SHA-256 hashes define the reviewed
reference realization.

## Figure contract

The renderer creates only:

```text
figures/01_study_design.png
figures/02_model_validation.png
figures/03_transfer_and_risk.png
figures/04_screening_landscape.png
figures/05_candidate_atlas.png
figures/supplementary/S1_training_curves.png
```

The five main figures live directly in `figures/`; the supplementary figure is
held separately under `figures/supplementary/` and obeys the same raster
contract. Each is 6480 pixels wide and no more than 6024 pixels high (183 mm × at most
170 mm), tagged at 900 dpi, RGB, alpha-free, grid-free, and pure white at all
four canvas corners. The shared renderer uses Arial first, 8 pt bold panel
letters, 7 pt titles, 5–7 pt supporting text, accessible colour encodings,
redundant shapes or line styles, and frameless legends. The release gate fails
on extra files or alternate formats. The PNG-only contract is an intentional
project constraint and is not claimed to satisfy Nature's final editable-vector
delivery requirement.

Object-level figure tests render the Matplotlib artists without writing files and
reject legends that intersect markers or data lines, free annotations that cover
data, and unclipped text that falls outside the canvas. Temporary 25% previews are
also reviewed in colour, greyscale, and simulated red–green colour-vision loss;
these previews are never added to `figures/`.

## Release checklist

1. Run `p1screen rebuild` from a clean environment.
2. Inspect all five figures at full size, thumbnail scale, and in greyscale.
3. Run `pytest` and `p1screen verify --frozen`.
4. Review `git diff`, especially frozen inputs, model manifest, metrics, and MP
   disposition counts.
5. Confirm no key, `.env`, notebook cache, or local path is tracked.
6. Run `p1screen manifest`, then rerun `p1screen verify --frozen` and confirm
   `git diff` contains only intentional manifest changes.
7. Tag the reviewed commit; do not refresh MP under an existing release tag.

## Known reproducibility boundary

The repository reconstructs statistical screening only. It cannot reproduce
DFT, molecular dynamics, MLIP-MD, synthesis, or wet-experiment results because
none is part of Project 1.
