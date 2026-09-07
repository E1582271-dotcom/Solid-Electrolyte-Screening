"""Offline release gate for data, models, source tables, and PNG figures."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .composition import composition_key, element_symbols
from .config import (
    CANDIDATE_QUEUE,
    DATA,
    FIGURES,
    LEARNING_CURVE_FRACTIONS,
    LEARNING_CURVE_REPEATS,
    MAIN_FIGURES,
    METRICS,
    MODEL_MANIFEST,
    MODEL_SPECS,
    MODELS,
    MP_DATABASE_VERSION,
    MP_EXPECTED_QUERY_COUNT,
    MP_LEDGER,
    MP_QUERY,
    MP_SNAPSHOT,
    N_FOLDS,
    OBELIX_TEST,
    OBELIX_TRAIN,
    RELEASE_MANIFEST,
    ROOT,
    SOURCE_DATA,
    SUPPLEMENTARY,
    SUPPLEMENTARY_FIGURES,
)
from .data import load_obelix_raw, split_audit


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _check_csv(path: Path) -> pd.DataFrame:
    _require(path.exists(), f"missing CSV: {_relative(path)}")
    frame = pd.read_csv(path)
    _require(len(frame) > 0, f"empty CSV: {_relative(path)}")
    _require(not any(str(column).startswith("Unnamed:") for column in frame.columns),
             f"implicit index column in {_relative(path)}")
    return frame


def _check_raster(path: Path) -> dict:
    """Apply the shared PNG delivery contract to one figure file and describe it."""
    name = _relative(path)
    with Image.open(path) as image:
        dpi = image.info.get("dpi", (0, 0))
        _require(image.format == "PNG", f"not a PNG: {name}")
        _require(image.mode == "RGB", f"{name} must be RGB without alpha")
        _require(image.width == 6480, f"{name} width must be 6480 px at 900 dpi")
        _require(image.height <= 6024, f"{name} exceeds the 170-mm height limit")
        _require(path.stat().st_size <= 50 * 1024 * 1024,
                 f"{name} exceeds the 50-MB delivery limit")
        _require(
            all(abs(float(value) - 900.0) <= 1.0 for value in dpi),
            f"{name} does not carry 900-dpi metadata: {dpi}",
        )
        corners = [image.getpixel((0, 0)), image.getpixel((image.width-1, 0)),
                   image.getpixel((0, image.height-1)),
                   image.getpixel((image.width-1, image.height-1))]
        _require(all(pixel == (255, 255, 255) for pixel in corners),
                 f"{name} does not have a pure-white canvas")
        return {
            "path": name, "width_px": image.width,
            "height_px": image.height, "mode": image.mode,
            "dpi": [float(dpi[0]), float(dpi[1])], "sha256": _sha256(path),
        }


def _check_figures() -> list[dict]:
    files = sorted(path.name for path in FIGURES.iterdir() if path.is_file())
    _require(files == sorted(MAIN_FIGURES),
             f"figures/ must contain exactly the five declared PNGs; received {files}")
    return [_check_raster(FIGURES / name) for name in MAIN_FIGURES]


def _check_supplementary_figures() -> list[dict]:
    _require(SUPPLEMENTARY.is_dir(), "figures/supplementary/ is missing")
    files = sorted(path.name for path in SUPPLEMENTARY.iterdir() if path.is_file())
    _require(
        files == sorted(SUPPLEMENTARY_FIGURES),
        f"figures/supplementary/ must contain exactly the declared PNGs; received {files}",
    )
    return [_check_raster(SUPPLEMENTARY / name) for name in SUPPLEMENTARY_FIGURES]


def verify_release(*, frozen: bool = False, write_manifest: bool = False) -> dict:
    """Validate release contracts, optionally writing or comparing the manifest.

    Normal and frozen verification are read-only.  Updating the reviewed checksum
    manifest is an explicit operation so CI cannot silently bless changed files.
    """
    _require(not (frozen and write_manifest), "cannot compare and update the manifest together")
    required_root = [
        ROOT / "README.md", ROOT / "LICENSE", ROOT / "CITATION.cff",
        ROOT / "THIRD_PARTY_NOTICES.md", ROOT / "DATA_DICTIONARY.md",
        ROOT / "METHODS.md", ROOT / "FIGURES.md", ROOT / "REPRODUCIBILITY.md",
        ROOT / "pyproject.toml", ROOT / "environment.lock.yml",
    ]
    for path in required_root:
        _require(path.exists(), f"missing release document: {path.name}")

    train, test = load_obelix_raw()
    audit = split_audit(train, test)
    _require(len(train) == 478 and len(test) == 121, "OBELiX row-count contract failed")
    _require(audit["n_composition_overlap_groups"] == 2, "split-overlap contract failed")
    _require(audit["n_train_strict"] == 476, "strict-train contract failed")
    _require(
        audit["n_formula_identity_mismatch_train"] == 6
        and audit["n_formula_identity_mismatch_test"] == 6,
        "nominal/feature composition audit failed",
    )
    _require(audit["n_censored_train"] == 29 and audit["n_censored_test"] == 8,
             "censor-count contract failed")

    snapshot = _check_csv(MP_SNAPSHOT)
    ledger = _check_csv(MP_LEDGER)
    queue = _check_csv(CANDIDATE_QUEUE)
    _require(len(snapshot) == MP_EXPECTED_QUERY_COUNT == len(ledger),
             "MP 248-entry accounting contract failed")
    _require(snapshot.material_id.is_unique and ledger.material_id.is_unique,
             "MP material IDs must be unique")
    _require(set(snapshot.database_version.astype(str)) == {MP_DATABASE_VERSION},
             "MP database version drift")
    _require(
        snapshot.apply(
            lambda row: row.composition_key == composition_key(row.formula), axis=1
        ).all(),
        "MP snapshot contains stale composition keys",
    )
    for row in snapshot.itertuples(index=False):
        elements = set(element_symbols(row.formula))
        _require(
            {"Li", "S"}.issubset(elements)
            and "H" not in elements
            and 3 <= len(elements) <= 5
            and 0 <= float(row.e_above_hull) <= 0.05
            and float(row.band_gap) >= 1.5,
            f"MP query-filter violation: {row.material_id}",
        )
    pd.testing.assert_frame_equal(
        snapshot,
        ledger[snapshot.columns],
        check_dtype=False,
        obj="snapshot versus ledger metadata",
    )
    counts = ledger.disposition.value_counts().to_dict()
    _require(counts == {
        "oxygen_scope_exclusion": 80,
        "screen_candidate": 79,
        "obelix_composition_reference": 70,
        "redox_electronic_risk": 19,
    }, f"exhaustive disposition contract failed: {counts}")
    _require(len(queue) == 62 and queue.composition_key.is_unique,
             "candidate queue must contain 62 unique compositions")
    multiplicity = queue.n_mp_entries.value_counts().sort_index().to_dict()
    _require(
        multiplicity == {1: 49, 2: 9, 3: 4}
        and int(queue.n_mp_entries.sum()) == 79,
        f"79-structure/62-composition multiplicity contract failed: {multiplicity}",
    )
    _require(
        np.array_equal(queue.candidate_rank.to_numpy(), np.arange(1, len(queue) + 1))
        and queue.ranking_score.is_monotonic_decreasing,
        "candidate ranks are not consecutive and score-descending",
    )
    _require(queue.evidence_state.value_counts().to_dict() == {
        "extrapolative": 27, "unflagged": 23, "high_score_lower_risk": 12,
    }, "candidate evidence-state contract failed")
    _require(
        {
            "member_rank_q25", "member_rank_median", "member_rank_q75",
            "member_rank_iqr", "top_quartile_frequency",
        }
        <= set(queue.columns),
        "candidate queue lacks member-level rank-stability fields",
    )
    _require(
        (queue.member_rank_q25 <= queue.member_rank_median).all()
        and (queue.member_rank_median <= queue.member_rank_q75).all()
        and np.allclose(
            queue.member_rank_q75 - queue.member_rank_q25,
            queue.member_rank_iqr,
            rtol=0,
            atol=1e-12,
        ),
        "candidate member-rank quartiles or IQR are inconsistent",
    )

    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    _require(metrics["split_audit"]["n_train_strict"] == 476,
             "metrics.json split contract failed")
    _require(metrics["target_domain_test"]["n_records"] == 24,
             "target-domain test contract failed")
    _require("strict_test_baselines" in metrics and "training_sensitivity" in metrics,
             "strict baseline or training-sensitivity results are missing")
    _require("risk_correlation_bootstrap" in metrics,
             "risk-correlation bootstrap results are missing")
    query = json.loads(MP_QUERY.read_text(encoding="utf-8"))
    _require(query["database_version"] == MP_DATABASE_VERSION and
             query["n_entries"] == MP_EXPECTED_QUERY_COUNT,
             "MP query provenance contract failed")

    model_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    _require(model_manifest["n_members"] == 15 and len(model_manifest["members"]) == 15,
             "production ensemble must contain 15 members")
    route_sensitivity = pd.DataFrame(model_manifest.get("route_sensitivity", []))
    _require(
        {"quantile", "state", "count"} <= set(route_sensitivity.columns),
        "model manifest lacks route-sensitivity records",
    )
    _require(
        set(route_sensitivity["quantile"].unique()) == {0.70, 0.75, 0.80}
        and route_sensitivity.groupby("quantile")["count"].sum().eq(62).all(),
        "model manifest route-sensitivity accounting failed",
    )
    q75_states = (
        route_sensitivity.loc[route_sensitivity["quantile"].eq(0.75)]
        .set_index("state")["count"]
        .to_dict()
    )
    _require(
        q75_states == {
            "extrapolative": 27,
            "high_score_lower_risk": 12,
            "unflagged": 23,
        },
        f"q75 route-sensitivity state contract failed: {q75_states}",
    )
    expected_model_files = {member["path"] for member in model_manifest["members"]}
    actual_model_files = {path.name for path in MODELS.iterdir() if path.is_file()}
    _require(
        actual_model_files == expected_model_files,
        f"model directory contains stale or missing files: "
        f"{sorted(actual_model_files ^ expected_model_files)}",
    )
    for member in model_manifest["members"]:
        path = MODELS / member["path"]
        _require(path.exists(), f"missing model: {_relative(path)}")
        _require(_sha256(path) == member["sha256"], f"model checksum mismatch: {path.name}")

    expected_sources = {
        "fig01b_split_audit.csv", "fig01c_mp_disposition.csv",
        "fig01c_mp_query_contract.csv",
        "fig02a_targets.csv", "fig02b_grouped_cv.csv", "fig02c_test_predictions.csv",
        "fig02d_censor_sensitivity.csv", "fig03a_representation_ablation.csv",
        "fig02e_strict_baselines.csv", "fig02f_training_sensitivity.csv",
        "fig03b_cell_convention.csv", "fig03c_risk_correlations.csv",
        "fig03d_oof_risk.csv", "fig04a_descriptor_projection.csv",
        "fig04_screen_ledger.csv", "fig05_candidate_queue.csv",
        "figS1a_boosting_curves.csv", "figS1b_learning_curve.csv",
    }
    actual_sources = {path.name for path in SOURCE_DATA.glob("*.csv")}
    _require(actual_sources == expected_sources,
             f"source_data contains stale or missing tables: {sorted(actual_sources ^ expected_sources)}")
    for name in expected_sources:
        _check_csv(SOURCE_DATA / name)

    boosting = _check_csv(SOURCE_DATA / "figS1a_boosting_curves.csv")
    boosting_columns = {
        "config", "fold", "iteration", "train_rmse", "validation_rmse",
        "n_train", "n_validation",
    }
    _require(boosting_columns <= set(boosting.columns),
             "Supplementary boosting-curve source lacks required columns")
    _require(np.isfinite(boosting[["train_rmse", "validation_rmse"]]).all().all(),
             "Supplementary boosting-curve values are not finite")
    curve_lengths = boosting.groupby(["config", "fold"]).iteration.agg(["size", "max"])
    _require(set(boosting.config) == {spec.name for spec in MODEL_SPECS},
             "Supplementary boosting-curve source does not cover every configuration")
    for spec in MODEL_SPECS:
        lengths = curve_lengths.loc[spec.name]
        _require(
            len(lengths) == N_FOLDS
            and set(lengths["size"]) == {spec.iterations}
            and set(lengths["max"]) == {spec.iterations},
            f"Supplementary boosting curve for {spec.name} must span "
            f"{spec.iterations} iterations on {N_FOLDS} folds",
        )
    learning = _check_csv(SOURCE_DATA / "figS1b_learning_curve.csv")
    learning_columns = {
        "fraction", "repeat_seed", "n_train_records", "n_train_compositions",
        "MAE", "RMSE", "R2", "Spearman", "MAE_target_domain",
    }
    _require(learning_columns <= set(learning.columns),
             "Supplementary learning-curve source lacks required columns")
    _require(np.isfinite(learning[["MAE", "RMSE", "R2", "Spearman", "MAE_target_domain"]])
             .all().all(), "Supplementary learning-curve values are not finite")
    _require(np.allclose(sorted(learning.fraction.unique()), LEARNING_CURVE_FRACTIONS),
             "Supplementary learning-curve fractions do not match the declared grid")
    _require(len(learning) == (len(LEARNING_CURVE_FRACTIONS) - 1) * LEARNING_CURVE_REPEATS + 1,
             "Supplementary learning-curve row count does not match fractions × repeats")
    full_rows = learning.loc[np.isclose(learning.fraction, 1.0)]
    _require(
        len(full_rows) == 1
        and int(full_rows.n_train_records.iloc[0]) == 476
        and int(full_rows.n_train_compositions.iloc[0]) == 388,
        "Supplementary learning curve must end at the full 476-record / 388-composition strict train",
    )
    risk = _check_csv(SOURCE_DATA / "fig03c_risk_correlations.csv")
    risk_columns = {
        "sample", "signal", "spearman_rho", "ci95_low", "ci95_high",
        "n_records", "n_compositions", "n_bootstrap", "rank_normalization",
    }
    _require(risk_columns <= set(risk.columns),
             "Figure 3 risk source lacks bootstrap fields")
    _require(len(risk) == 16, "Figure 3 risk source must contain 16 rows")
    _require(np.isfinite(risk[["spearman_rho", "ci95_low", "ci95_high"]]).all().all()
             and (risk.ci95_low <= risk.ci95_high).all(),
             "Figure 3 risk intervals are invalid")
    expected_risk_samples = {
        "oof": (478, 390, "within_fold"),
        "test": (121, 112, "within_sample"),
        "oof_target_domain": (117, 97, "within_fold"),
        "test_target_domain": (24, 23, "within_sample"),
    }
    expected_signals = {"spread_error", "ad_error", "combined_error", "spread_ad"}
    for sample, (n_records, n_compositions, normalization) in expected_risk_samples.items():
        sample_rows = risk.loc[risk["sample"].eq(sample)]
        _require(set(sample_rows.signal) == expected_signals,
                 f"Figure 3 risk signals are incomplete for {sample}")
        _require(
            set(sample_rows.n_records) == {n_records}
            and set(sample_rows.n_compositions) == {n_compositions}
            and set(sample_rows.n_bootstrap) == {4000}
            and set(sample_rows.rank_normalization) == {normalization},
            f"Figure 3 risk sample accounting failed for {sample}",
        )
        bootstrap = metrics["risk_correlation_bootstrap"][sample]
        for row in sample_rows.itertuples(index=False):
            _require(
                np.allclose(
                    [row.ci95_low, row.ci95_high],
                    bootstrap["ci95"][row.signal],
                    rtol=0,
                    atol=1e-12,
                ),
                f"Figure 3 risk source/metrics mismatch for {sample}/{row.signal}",
            )
    query_contract = _check_csv(SOURCE_DATA / "fig01c_mp_query_contract.csv")
    _require(len(query_contract) == 1, "Figure 1 query contract must have exactly one row")
    query_row = query_contract.iloc[0]
    filters = query["filters"]
    _require(
        str(query_row.database_version) == str(query["database_version"])
        and int(query_row.n_entries) == int(query["n_entries"])
        and str(query_row.required_elements) == ";".join(filters["elements"])
        and str(query_row.excluded_elements) == ";".join(filters["exclude_elements"])
        and int(query_row.min_elements) == int(filters["num_elements"][0])
        and int(query_row.max_elements) == int(filters["num_elements"][1])
        and np.isclose(
            float(query_row.min_e_above_hull),
            float(filters["energy_above_hull_eV_atom"][0]),
        )
        and np.isclose(
            float(query_row.max_e_above_hull),
            float(filters["energy_above_hull_eV_atom"][1]),
        )
        and np.isclose(float(query_row.min_band_gap), float(filters["band_gap_eV"][0])),
        "Figure 1 query contract source does not match data/mp_query.json",
    )
    pd.testing.assert_frame_equal(
        ledger,
        pd.read_csv(SOURCE_DATA / "fig04_screen_ledger.csv"),
        check_dtype=False,
        obj="ledger versus figure source",
    )
    pd.testing.assert_frame_equal(
        queue,
        pd.read_csv(SOURCE_DATA / "fig05_candidate_queue.csv"),
        check_dtype=False,
        obj="queue versus figure source",
    )

    figures = _check_figures() + _check_supplementary_figures()
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    _require(not any(
        "api_key" in item.lower()
        or Path(item).name == ".env"
        or Path(item).name.startswith(".env.")
        or "credential" in Path(item).name.lower()
        for item in tracked
    ),
             "a credential-like file is tracked")

    implementation_paths = (
        sorted((ROOT / "src" / "p1screen").glob("*.py"))
        + sorted((ROOT / "tests").glob("*.py"))
        + sorted(ROOT.glob("[0-9][0-9]_*.py"))
        + [
            ROOT / ".gitignore",
            ROOT / "requirements.txt",
            ROOT / ".github" / "workflows" / "ci.yml",
        ]
    )
    payload_paths = required_root + implementation_paths + [
        OBELIX_TRAIN, OBELIX_TEST, MP_SNAPSHOT, MP_QUERY, METRICS,
        MODEL_MANIFEST, MP_LEDGER, CANDIDATE_QUEUE,
    ] + sorted(SOURCE_DATA.glob("*.csv")) + sorted(MODELS.glob("*.cbm"))
    manifest = {
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": "0.1.0",
        "python": platform.python_version(),
        "package_versions": {
            name: version(name) for name in
            ["numpy", "pandas", "scipy", "scikit-learn", "catboost",
             "pymatgen", "matminer", "matplotlib", "pillow", "mp-api"]
        },
        "contracts": {
            "obelix_records": 599, "strict_train_records": 476,
            "mp_entries": 248, "candidate_compositions": 62,
            "model_members": 15, "main_figures": 5,
            "supplementary_figures": len(SUPPLEMENTARY_FIGURES),
            "image_format": "PNG", "image_dpi": 900,
            "image_mode": "RGB", "background": "white",
            "image_width_px": 6480, "image_max_height_px": 6024,
        },
        "figures": figures,
        "files": [
            {"path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in payload_paths
        ],
    }
    if write_manifest:
        DATA.mkdir(parents=True, exist_ok=True)
        RELEASE_MANIFEST.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print("Release manifest updated after all contracts passed.")
    elif frozen:
        _require(RELEASE_MANIFEST.exists(), "missing frozen data/release_manifest.json")
        reviewed = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
        for key in ("contracts", "figures", "files"):
            _require(
                reviewed.get(key) == manifest[key],
                f"frozen release manifest differs in {key}; review changes and run `p1screen manifest`",
            )
        print("Frozen release gate passed without modifying the manifest.")
    else:
        print("Release contracts passed (manifest not modified).")
    return manifest
