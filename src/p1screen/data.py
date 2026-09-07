"""OBELiX ingestion, target parsing, and split-integrity contracts."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .composition import as_composition, composition_key, normalized_doi
from .config import OBELIX_TEST, OBELIX_TRAIN

TARGET_COLUMN = "Ionic conductivity (S cm-1)"


@dataclass(frozen=True)
class ParsedConductivity:
    value: float
    censor_direction: str

    @property
    def censored(self) -> bool:
        return bool(self.censor_direction)


def parse_conductivity(value: object) -> ParsedConductivity:
    text = str(value).strip().replace(" ", "")
    direction = text[0] if text[:1] in {"<", ">"} else ""
    numeric = re.sub(r"^[<>~=]+", "", text).replace("E", "e")
    parsed = float(numeric)
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"conductivity must be finite and positive: {value!r}")
    return ParsedConductivity(parsed, direction)


def resolve_feature_formula(record: Mapping) -> tuple[str, str]:
    """Return the composition used for descriptors and its source column.

    ``True Composition`` is the measured unit-cell composition in OBELiX and is
    therefore preferred for model features.  The nominal/reduced formula remains
    available separately for provenance and catalogue matching.
    """
    for column in ("True Composition", "Reduced Composition"):
        value = record.get(column)
        try:
            as_composition(value)
        except (TypeError, ValueError):
            continue
        return str(value), column
    raise ValueError(f"record has no parseable formula: {record.get('ID', '<unknown>')}")


def _read_split(path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing vendored OBELiX split: {path}. The release requires train.csv and test.csv."
        )
    frame = pd.read_csv(path).copy()
    parsed = frame[TARGET_COLUMN].map(parse_conductivity)
    frame.insert(0, "source_split", split)
    resolved = frame.apply(resolve_feature_formula, axis=1)
    frame["feature_formula"] = resolved.map(lambda item: item[0])
    frame["feature_formula_source"] = resolved.map(lambda item: item[1])
    frame["nominal_composition_key"] = frame["Reduced Composition"].map(composition_key)
    frame["feature_composition_key"] = frame["feature_formula"].map(composition_key)
    # The grouping identity must be the same composition used to build model features.
    frame["composition_key"] = frame["feature_composition_key"]
    frame["composition_key_mismatch"] = frame["nominal_composition_key"].ne(
        frame["feature_composition_key"]
    )
    frame["doi_normalized"] = frame["DOI"].map(normalized_doi)
    frame["sigma_S_cm"] = parsed.map(lambda item: item.value)
    frame["log10_sigma"] = np.log10(frame["sigma_S_cm"])
    frame["censor_direction"] = parsed.map(lambda item: item.censor_direction)
    frame["censored"] = frame["censor_direction"].ne("")
    return frame


def load_obelix_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _read_split(OBELIX_TRAIN, "official_train")
    test = _read_split(OBELIX_TEST, "official_test")
    return train, test


def split_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    train_keys, test_keys = set(train.composition_key), set(test.composition_key)
    train_dois = set(train.doi_normalized) - {""}
    test_dois = set(test.doi_normalized) - {""}
    overlap = sorted(train_keys & test_keys)
    nominal_overlap = sorted(
        set(train.nominal_composition_key) & set(test.nominal_composition_key)
    )
    return {
        "n_train_original": int(len(train)),
        "n_test": int(len(test)),
        "n_train_composition_groups": int(train.composition_key.nunique()),
        "n_test_composition_groups": int(test.composition_key.nunique()),
        "n_composition_overlap_groups": int(len(overlap)),
        "composition_overlap_keys": overlap,
        "n_nominal_composition_overlap_groups": int(len(nominal_overlap)),
        "nominal_composition_overlap_keys": nominal_overlap,
        "n_formula_identity_mismatch_train": int(train.composition_key_mismatch.sum()),
        "n_formula_identity_mismatch_test": int(test.composition_key_mismatch.sum()),
        "n_train_rows_removed_for_strict_test": int(train.composition_key.isin(overlap).sum()),
        "n_train_strict": int((~train.composition_key.isin(overlap)).sum()),
        "n_test_composition_novel": int((~test.composition_key.isin(overlap)).sum()),
        "n_doi_overlap": int(len(train_dois & test_dois)),
        "n_censored_train": int(train.censored.sum()),
        "n_censored_test": int(test.censored.sum()),
        "censor_directions": sorted(set(train.censor_direction) | set(test.censor_direction)),
    }
