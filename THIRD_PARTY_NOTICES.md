# Third-party data and software notices

## OBELiX

The files `data/train.csv` and `data/test.csv` originate from the OBELiX
dataset/repository and are redistributed under its CC BY 4.0 terms. Attribution:

> F. Therrien et al., “OBELiX: a curated dataset of experimentally measured
> ionic conductivities,” *Digital Discovery* (2026),
> doi:10.1039/D5DD00441A.

Upstream repository: <https://github.com/NRC-Mila/OBELiX>

Changes made here are limited to packaging the published train/test CSVs and
deriving normalized keys, targets, audits, and model outputs. The original data
are not covered by this repository's MIT code license.

## Materials Project

`data/mp_snapshot.csv` is a minimal scalar metadata snapshot retrieved through
the Materials Project API. When using these derived screening results, cite:

> M. K. Horton et al., “Accelerated data-driven materials science with the
> Materials Project,” *Nature Materials* **24**, 1522–1532 (2025),
> doi:10.1038/s41563-025-02272-0.

The frozen release stores no crystal structure files. Upstream terms and
citation guidance apply: <https://materialsproject.org/about>.

Some MP records may trace to third-party structure sources with additional
terms. Catalogue IDs are preserved only as provenance fields and do not grant
rights to redistribute upstream structures.

## Python software

This project depends on NumPy, pandas, SciPy, scikit-learn, CatBoost, pymatgen,
matminer, Matplotlib, Pillow, and mp-api. Each package retains its own license.
Exact reviewed versions are listed in `environment.lock.yml`; the repository's
MIT license does not replace dependency licenses.
