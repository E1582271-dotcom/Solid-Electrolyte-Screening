"""Formula-first features and structural ablations."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .composition import as_composition
from .data import resolve_feature_formula

_MAGPIE = None


def _magpie():
    global _MAGPIE
    if _MAGPIE is None:
        from matminer.featurizers.composition import ElementProperty

        _MAGPIE = ElementProperty.from_preset("magpie")
    return _MAGPIE


def m0_feature_names() -> list[str]:
    return list(_magpie().feature_labels()) + ["n_elements", "Li_frac"]


def composition_features(formula: object) -> dict[str, float]:
    comp = as_composition(formula)
    featurizer = _magpie()
    values = featurizer.featurize(comp)
    result = dict(zip(featurizer.feature_labels(), values, strict=True))
    fractions = comp.fractional_composition.get_el_amt_dict()
    result["n_elements"] = float(len(fractions))
    result["Li_frac"] = float(fractions.get("Li", 0.0))
    return result


def crystal_system(spacegroup_number: object) -> str:
    number = int(float(spacegroup_number))
    for high, name in (
        (2, "triclinic"),
        (15, "monoclinic"),
        (74, "orthorhombic"),
        (142, "tetragonal"),
        (167, "trigonal"),
        (194, "hexagonal"),
        (230, "cubic"),
    ):
        if number <= high:
            return name
    raise ValueError(f"invalid space-group number: {spacegroup_number!r}")


def cell_volume(a, b, c, alpha, beta, gamma) -> float:
    al, be, ga = np.radians([float(alpha), float(beta), float(gamma)])
    radicand = 1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
    radicand += 2 * np.cos(al) * np.cos(be) * np.cos(ga)
    return float(float(a) * float(b) * float(c) * math.sqrt(max(float(radicand), 0.0)))


def featurize_obelix(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, record in raw.iterrows():
        formula = str(record.get("feature_formula") or resolve_feature_formula(record)[0])
        formula_source = str(
            record.get("feature_formula_source") or resolve_feature_formula(record)[1]
        )
        base = composition_features(formula)
        volume = cell_volume(
            record["a"], record["b"], record["c"],
            record["alpha"], record["beta"], record["gamma"],
        )
        feature_comp = as_composition(formula)
        reduced_comp = as_composition(record["Reduced Composition"])
        mean_atomic_mass = float(feature_comp.weight / feature_comp.num_atoms)
        z_value = float(record["Z"])
        atoms_in_cell = (
            float(feature_comp.num_atoms)
            if formula_source == "True Composition"
            else z_value * float(reduced_comp.num_atoms)
        )
        volume_per_atom = volume / atoms_in_cell
        lengths = np.sort(np.asarray([record["a"], record["b"], record["c"]], dtype=float))
        angle_cosines = np.sort(np.cos(np.radians(
            np.asarray([record["alpha"], record["beta"], record["gamma"]], dtype=float)
        )))
        base.update({
            "record_id": str(record["ID"]),
            "source_split": str(record["source_split"]),
            "composition_key": str(record["composition_key"]),
            "doi_normalized": str(record["doi_normalized"]),
            "formula_raw": str(record["Reduced Composition"]),
            "feature_formula": formula,
            "feature_formula_source": formula_source,
            "nominal_composition_key": str(record["nominal_composition_key"]),
            "feature_composition_key": str(record["feature_composition_key"]),
            "composition_key_mismatch": bool(record["composition_key_mismatch"]),
            "log10_sigma": float(record["log10_sigma"]),
            "censored": bool(record["censored"]),
            "censor_direction": str(record["censor_direction"]),
            "a": float(record["a"]), "b": float(record["b"]), "c": float(record["c"]),
            "alpha": float(record["alpha"]), "beta": float(record["beta"]),
            "gamma": float(record["gamma"]), "cell_volume": volume,
            "spacegroup_no": int(record["Space group #"]), "Z": z_value,
            "crystal_system": crystal_system(record["Space group #"]),
            "volume_per_atom": volume_per_atom,
            "density_proxy": mean_atomic_mass / volume_per_atom,
            "axis_ratio_mid_min": float(lengths[1] / lengths[0]),
            "axis_ratio_max_min": float(lengths[2] / lengths[0]),
            "angle_cos_sorted_1": float(angle_cosines[0]),
            "angle_cos_sorted_2": float(angle_cosines[1]),
            "angle_cos_sorted_3": float(angle_cosines[2]),
        })
        rows.append(base)
    return pd.DataFrame(rows)


M1_EXTRA = [
    "volume_per_atom", "density_proxy", "axis_ratio_mid_min", "axis_ratio_max_min",
    "angle_cos_sorted_1", "angle_cos_sorted_2", "angle_cos_sorted_3",
    "spacegroup_no", "crystal_system",
]
M2_EXTRA = [
    "a", "b", "c", "alpha", "beta", "gamma", "cell_volume",
    "spacegroup_no", "Z", "crystal_system",
]


def representation_columns(name: str) -> tuple[list[str], list[str]]:
    base = m0_feature_names()
    if name == "M0_formula":
        return base, []
    if name == "M1_cell_normalized":
        return base + M1_EXTRA, ["crystal_system"]
    if name == "M2_legacy":
        return base + M2_EXTRA, ["crystal_system"]
    raise KeyError(f"unknown representation: {name}")


def formula_matrix(formulas: pd.Series | list[str]) -> pd.DataFrame:
    return pd.DataFrame([composition_features(formula) for formula in formulas]).reindex(
        columns=m0_feature_names()
    )
