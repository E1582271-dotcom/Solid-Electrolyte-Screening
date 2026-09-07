"""Frozen Materials Project query, exhaustive disposition, and candidate queue."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .composition import (
    composition_key,
    element_symbols,
    jsonable_database_ids,
)
from .config import (
    CANDIDATE_QUEUE,
    MODEL_MANIFEST,
    MODELS,
    MP_DATABASE_VERSION,
    MP_EXPECTED_QUERY_COUNT,
    MP_LEDGER,
    MP_QUERY,
    MP_SNAPSHOT,
    REDOX_RISK_ELEMENTS,
    ROUTE_QUANTILE,
    SOURCE_DATA,
)
from .data import load_obelix_raw
from .domain import ApplicabilityDomain
from .features import featurize_obelix, formula_matrix, m0_feature_names
from .modeling import ensemble_predictions, save_ensemble, train_ensemble, write_model_manifest


def refresh_mp_snapshot() -> pd.DataFrame:
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("MP_API_KEY is required; key files are intentionally unsupported")
    from mp_api.client import MPRester

    with MPRester(key) as mpr:
        database_version = str(mpr.db_version)
        docs = mpr.materials.summary.search(
            elements=["Li", "S"],
            exclude_elements=["H"],
            num_elements=(3, 5),
            energy_above_hull=(0, 0.05),
            band_gap=(1.5, None),
            fields=[
                "material_id", "formula_pretty", "elements", "nelements", "symmetry",
                "energy_above_hull", "band_gap", "database_IDs", "theoretical", "last_updated",
            ],
        )
    if database_version != MP_DATABASE_VERSION:
        raise RuntimeError(
            f"MP database drift: expected {MP_DATABASE_VERSION}, received {database_version}; "
            "review and update the release contract before refreshing"
        )
    rows = []
    for doc in docs:
        db_ids = jsonable_database_ids(doc.database_IDs)
        rows.append({
            "database_version": database_version,
            "material_id": str(doc.material_id),
            "formula": str(doc.formula_pretty),
            "composition_key": composition_key(doc.formula_pretty),
            "elements": ";".join(sorted(str(item) for item in doc.elements)),
            "n_elements": int(doc.nelements),
            "spacegroup_no": int(doc.symmetry.number),
            "crystal_system": str(doc.symmetry.crystal_system),
            "e_above_hull": float(doc.energy_above_hull),
            "band_gap": float(doc.band_gap),
            "theoretical": bool(doc.theoretical),
            "icsd_ids": ";".join(db_ids.get("icsd", [])),
            "pauling_ids": ";".join(db_ids.get("pauling", [])),
            "last_updated": str(doc.last_updated),
        })
    snapshot = pd.DataFrame(rows).sort_values("material_id", kind="stable").reset_index(drop=True)
    if len(snapshot) != MP_EXPECTED_QUERY_COUNT or not snapshot.material_id.is_unique:
        raise RuntimeError(
            f"MP snapshot contract expected {MP_EXPECTED_QUERY_COUNT} unique entries, "
            f"received {len(snapshot)}"
        )
    MP_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(MP_SNAPSHOT, index=False)
    query = {
        "database_version": database_version,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "elements": ["Li", "S"], "exclude_elements": ["H"],
            "num_elements": [3, 5], "energy_above_hull_eV_atom": [0, 0.05],
            "band_gap_eV": [1.5, None],
        },
        "n_entries": len(snapshot),
    }
    MP_QUERY.write_text(json.dumps(query, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot


def load_mp_snapshot() -> pd.DataFrame:
    if not MP_SNAPSHOT.exists():
        raise FileNotFoundError("data/mp_snapshot.csv is missing; run `p1screen refresh-mp`")
    snapshot = pd.read_csv(MP_SNAPSHOT)
    if len(snapshot) != MP_EXPECTED_QUERY_COUNT or not snapshot.material_id.is_unique:
        raise RuntimeError("frozen MP snapshot does not satisfy the 248-entry contract")
    return snapshot


def _disposition(formula: str, obelix_keys: set[str]) -> tuple[str, str]:
    elements = set(element_symbols(formula))
    redox = sorted(elements & REDOX_RISK_ELEMENTS)
    if "O" in elements:
        return "oxygen_scope_exclusion", "contains oxygen; outside the declared Li-S scope"
    if redox:
        return "redox_electronic_risk", ";".join(redox)
    if composition_key(formula) in obelix_keys:
        return "obelix_composition_reference", "measured composition in OBELiX"
    return "screen_candidate", "outside OBELiX composition set"


def build_screen() -> tuple[pd.DataFrame, pd.DataFrame]:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    raw_train, raw_test = load_obelix_raw()
    train, test = featurize_obelix(raw_train), featurize_obelix(raw_test)
    all_records = pd.concat([train, test], ignore_index=True)
    # Catalogue matching is deliberately conservative: either the nominal formula
    # or the measured feature composition is sufficient to mark an OBELiX reference.
    obelix_keys = set(all_records.nominal_composition_key) | set(
        all_records.feature_composition_key
    )
    snapshot = load_mp_snapshot().copy()
    snapshot["composition_key"] = snapshot.formula.map(composition_key)
    dispositions = snapshot.formula.map(lambda formula: _disposition(formula, obelix_keys))
    snapshot["disposition"] = dispositions.map(lambda item: item[0])
    snapshot["disposition_reason"] = dispositions.map(lambda item: item[1])

    columns = m0_feature_names()
    models = train_ensemble(all_records[columns], all_records.log10_sigma)
    member_manifest = save_ensemble(models, MODELS)
    X_mp = formula_matrix(snapshot.formula.tolist())
    mean, spread, predictions = ensemble_predictions(models, X_mp)
    snapshot["ranking_score"] = mean
    snapshot["ensemble_spread"] = spread
    domain = ApplicabilityDomain().fit(all_records[columns])
    snapshot["ad_distance"] = domain.distance(X_mp)
    snapshot["screen_rank"] = snapshot.ranking_score.rank(
        ascending=False, method="first"
    ).astype(int)

    # Two-dimensional view of the same standardized descriptor space used by AD.
    obelix_projection = domain.project(all_records[columns])[:, :2]
    mp_projection = domain.project(X_mp)[:, :2]
    projection_obelix = pd.DataFrame({
        "source": "OBELiX",
        "record_id": all_records.record_id.to_numpy(),
        "material_id": "",
        "disposition": "reference_measurement",
        "pc1": obelix_projection[:, 0],
        "pc2": obelix_projection[:, 1],
        "ranking_score": all_records.log10_sigma.to_numpy(),
        "ensemble_spread": np.nan,
        "ad_distance": domain.training_distance(),
    })
    projection_mp = pd.DataFrame({
        "source": "Materials Project",
        "record_id": "",
        "material_id": snapshot.material_id.to_numpy(),
        "disposition": snapshot.disposition.to_numpy(),
        "pc1": mp_projection[:, 0],
        "pc2": mp_projection[:, 1],
        "ranking_score": snapshot.ranking_score.to_numpy(),
        "ensemble_spread": snapshot.ensemble_spread.to_numpy(),
        "ad_distance": snapshot.ad_distance.to_numpy(),
    })

    candidates = snapshot[snapshot.disposition.eq("screen_candidate")].copy()
    representatives = (candidates.sort_values(
        ["composition_key", "e_above_hull", "material_id"], kind="stable"
    ).drop_duplicates("composition_key", keep="first"))
    aggregated = candidates.groupby("composition_key").agg(
        n_mp_entries=("material_id", "size"),
        min_e_above_hull=("e_above_hull", "min"),
        max_band_gap=("band_gap", "max"),
        any_icsd_match=("icsd_ids", lambda values: any(str(value).strip() not in {"", "nan"} for value in values)),
        any_non_theoretical_entry=("theoretical", lambda values: bool((~values.astype(bool)).any())),
    )
    queue = representatives[[
        "composition_key", "formula", "material_id", "spacegroup_no", "crystal_system",
        "ranking_score", "ensemble_spread", "ad_distance",
    ]].merge(aggregated, left_on="composition_key", right_index=True, validate="one_to_one")
    member_ranks = pd.DataFrame(
        predictions[:, representatives.index].T
    ).rank(axis=0, ascending=False, method="average")
    queue["member_rank_q25"] = member_ranks.quantile(0.25, axis=1).to_numpy()
    queue["member_rank_median"] = member_ranks.median(axis=1).to_numpy()
    queue["member_rank_q75"] = member_ranks.quantile(0.75, axis=1).to_numpy()
    queue["member_rank_iqr"] = queue.member_rank_q75 - queue.member_rank_q25
    top_quartile_cutoff = int(np.ceil(len(queue) * 0.25))
    queue["top_quartile_frequency"] = (
        member_ranks.le(top_quartile_cutoff).mean(axis=1).to_numpy()
    )
    score_q = float(queue.ranking_score.quantile(ROUTE_QUANTILE))
    spread_q = float(queue.ensemble_spread.quantile(ROUTE_QUANTILE))
    ad_q = float(queue.ad_distance.quantile(ROUTE_QUANTILE))
    queue["evidence_state"] = np.where(
        (queue.ensemble_spread >= spread_q) | (queue.ad_distance >= ad_q),
        "extrapolative",
        np.where(queue.ranking_score >= score_q, "high_score_lower_risk", "unflagged"),
    )
    queue["score_percentile"] = queue.ranking_score.rank(pct=True)
    queue["spread_percentile"] = queue.ensemble_spread.rank(pct=True)
    queue["ad_percentile"] = queue.ad_distance.rank(pct=True)
    queue["combined_risk_percentile"] = queue[[
        "spread_percentile", "ad_percentile"
    ]].max(axis=1)
    queue = queue.sort_values("ranking_score", ascending=False, kind="stable").reset_index(drop=True)
    queue.insert(0, "candidate_rank", np.arange(1, len(queue) + 1))
    state_map = queue.set_index("composition_key")["evidence_state"]
    snapshot["evidence_state"] = snapshot.composition_key.map(state_map).fillna("not_applicable")

    snapshot.to_csv(MP_LEDGER, index=False)
    queue.to_csv(CANDIDATE_QUEUE, index=False)
    snapshot.to_csv(SOURCE_DATA / "fig04_screen_ledger.csv", index=False)
    pd.concat([projection_obelix, projection_mp], ignore_index=True).to_csv(
        SOURCE_DATA / "fig04a_descriptor_projection.csv", index=False
    )
    queue.to_csv(SOURCE_DATA / "fig05_candidate_queue.csv", index=False)
    disposition = (snapshot.groupby("disposition", sort=False)
        .agg(n_entries=("material_id", "size"), n_compositions=("composition_key", "nunique"))
        .reset_index())
    disposition.to_csv(SOURCE_DATA / "fig01c_mp_disposition.csv", index=False)

    query = json.loads(MP_QUERY.read_text(encoding="utf-8"))
    filters = query["filters"]
    pd.DataFrame([{
        "database_version": query["database_version"],
        "n_entries": int(query["n_entries"]),
        "required_elements": ";".join(filters["elements"]),
        "excluded_elements": ";".join(filters["exclude_elements"]),
        "min_elements": int(filters["num_elements"][0]),
        "max_elements": int(filters["num_elements"][1]),
        "min_e_above_hull": float(filters["energy_above_hull_eV_atom"][0]),
        "max_e_above_hull": float(filters["energy_above_hull_eV_atom"][1]),
        "min_band_gap": float(filters["band_gap_eV"][0]),
    }]).to_csv(SOURCE_DATA / "fig01c_mp_query_contract.csv", index=False)

    route_sensitivity = []
    for quantile in (0.70, 0.75, 0.80):
        sq = queue.ranking_score.quantile(quantile)
        uq = queue.ensemble_spread.quantile(quantile)
        aq = queue.ad_distance.quantile(quantile)
        states = np.where(
            (queue.ensemble_spread >= uq) | (queue.ad_distance >= aq),
            "extrapolative",
            np.where(queue.ranking_score >= sq, "high_score_lower_risk", "unflagged"),
        )
        for state, count in pd.Series(states).value_counts().items():
            route_sensitivity.append({"quantile": quantile, "state": state, "count": int(count)})
    (SOURCE_DATA / "fig05d_route_sensitivity.csv").unlink(missing_ok=True)

    manifest = {
        "training_records": len(all_records),
        "training_compositions": int(all_records.composition_key.nunique()),
        "formula_identity": {
            "feature_source": "True Composition with Reduced Composition fallback",
            "grouping_key": "feature_composition_key",
            "catalogue_match_keys": ["nominal_composition_key", "feature_composition_key"],
            "n_nominal_feature_mismatches": int(all_records.composition_key_mismatch.sum()),
        },
        "training_weighting": "record_weighted",
        "representation": "M0_formula",
        "n_features": len(columns),
        "feature_names": columns,
        "n_members": len(models),
        "members": member_manifest,
        "ad": {
            "variance_retained": domain.variance,
            "n_components": domain.n_components,
            "neighbors": domain.neighbors,
        },
        "route_quantile": ROUTE_QUANTILE,
        "route_thresholds": {"score": score_q, "spread": spread_q, "ad": ad_q},
        "route_sensitivity": route_sensitivity,
        "variance_decomposition_note": "spread includes seed and model-configuration variation",
    }
    write_model_manifest(MODEL_MANIFEST, manifest)
    return snapshot, queue
