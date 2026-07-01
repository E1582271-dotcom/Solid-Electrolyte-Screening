"""
OBELiX featurization for ionic-conductivity prediction (Project 1, AI4SSB).

Design goals (honest + explainable, for the W4 checkpoint):
- Target  : y = log10( ionic conductivity / S cm^-1 ), parsed from the raw
            string column. Censored values like "<1E-10" are parsed to their
            numeric bound (1e-10) and FLAGGED -- a documented limitation.
- Features: a compact, hand-built "Magpie-lite" composition descriptor set
            (fraction-weighted statistics of element properties via pymatgen)
            + crystallographic features (lattice, space group, cell volume)
            + two categoricals (Family, crystal system) handled natively by
            CatBoost. No matminer dependency -> every feature is explainable.

This module is import-only: load_obelix() returns ready-to-model frames.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

RAW_TARGET = "Ionic conductivity (S cm-1)"

# Element properties we average over a composition. All are reliably defined
# in pymatgen for the elements present in OBELiX; None -> NaN (CatBoost-safe).
_ELEM_PROPS = {
    "Z": lambda e: e.Z,
    "mass": lambda e: float(e.atomic_mass),
    "X": lambda e: e.X,                      # Pauling electronegativity
    "row": lambda e: e.row,
    "group": lambda e: e.group,
    "mendeleev": lambda e: e.mendeleev_no,
    "radius": lambda e: float(e.atomic_radius) if e.atomic_radius else np.nan,
}


def parse_conductivity(raw: str) -> tuple[float, int]:
    """Return (sigma in S/cm, is_censored). Handles '<', '>', '~', '=' prefixes."""
    s = str(raw).strip().replace(" ", "")
    censored = 1 if s[:1] in "<>" else 0
    s = re.sub(r"^[<>~=]+", "", s)
    s = s.replace("E", "e")
    try:
        return float(s), censored
    except ValueError:
        return np.nan, censored


def _crystal_system(sg_number: int) -> str:
    n = int(sg_number)
    bounds = [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
              (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"),
              (230, "cubic")]
    for hi, name in bounds:
        if n <= hi:
            return name
    return "unknown"


def _cell_volume(a, b, c, al, be, ga) -> float:
    al, be, ga = np.radians([al, be, ga])
    return float(a * b * c * np.sqrt(
        1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
        + 2 * np.cos(al) * np.cos(be) * np.cos(ga)))


def _composition_features(comp_str: str) -> dict:
    """Fraction-weighted mean/std/min/max + Li fraction + n_elements."""
    try:
        comp = Composition(comp_str)
    except Exception:
        return {}
    fracs = comp.fractional_composition.get_el_amt_dict()
    els = {Element(sym): w for sym, w in fracs.items()}
    feat = {
        "n_elements": len(els),
        "Li_frac": fracs.get("Li", 0.0),
    }
    for pname, getter in _ELEM_PROPS.items():
        vals = np.array([getter(e) for e in els], dtype=float)
        wts = np.array(list(els.values()), dtype=float)
        ok = ~np.isnan(vals)
        if ok.sum() == 0:
            feat[f"{pname}_mean"] = feat[f"{pname}_std"] = np.nan
            feat[f"{pname}_min"] = feat[f"{pname}_max"] = feat[f"{pname}_range"] = np.nan
            continue
        v, w = vals[ok], wts[ok]
        w = w / w.sum()
        mean = float((v * w).sum())
        feat[f"{pname}_mean"] = mean
        feat[f"{pname}_std"] = float(np.sqrt((w * (v - mean) ** 2).sum()))
        feat[f"{pname}_min"] = float(v.min())
        feat[f"{pname}_max"] = float(v.max())
        feat[f"{pname}_range"] = float(v.max() - v.min())
    return feat


CAT_FEATURES = ["Family", "crystal_system"]


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Take a raw OBELiX frame -> feature frame with target column 'y'."""
    rows = []
    for _, r in df.iterrows():
        sigma, censored = parse_conductivity(r[RAW_TARGET])
        comp_src = r.get("True Composition")
        if not isinstance(comp_src, str) or not comp_src.strip():
            comp_src = r["Reduced Composition"]
        feat = _composition_features(comp_src)
        feat.update({
            "a": r["a"], "b": r["b"], "c": r["c"],
            "alpha": r["alpha"], "beta": r["beta"], "gamma": r["gamma"],
            "cell_volume": _cell_volume(r["a"], r["b"], r["c"],
                                        r["alpha"], r["beta"], r["gamma"]),
            "spacegroup_no": r["Space group #"],
            "Z": r["Z"],
            "crystal_system": _crystal_system(r["Space group #"]),
            "Family": str(r["Family"]) if isinstance(r["Family"], str) else "unknown",
            "sigma": sigma,
            "censored": censored,
        })
        rows.append(feat)
    out = pd.DataFrame(rows)
    out = out[out["sigma"] > 0].copy()          # drop unparseable / zero
    out["y"] = np.log10(out["sigma"])
    return out


_OBELIX_RAW = "https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/"


def _ensure_data(data_dir: str):
    """Download OBELiX official train/test split if not already present."""
    import os, urllib.request
    os.makedirs(data_dir, exist_ok=True)
    for fname in ("train.csv", "test.csv"):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"Downloading {fname} from OBELiX ...")
            urllib.request.urlretrieve(_OBELIX_RAW + fname, path)


def load_obelix(data_dir: str):
    """Return (train_feat, test_feat) using OBELiX's official split.

    Data auto-downloads from the OBELiX repository on first run (not
    redistributed here, to respect the dataset's own license)."""
    import os
    _ensure_data(data_dir)
    tr = pd.read_csv(os.path.join(data_dir, "train.csv"))
    te = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return featurize(tr), featurize(te)


def feature_columns(feat_df: pd.DataFrame) -> list[str]:
    drop = {"sigma", "y", "censored"}
    return [c for c in feat_df.columns if c not in drop]


# ── Display helpers for publication figures ───────────────────────────────────
# Map the raw model feature names (code identifiers) to reader-friendly labels.
_PROP_LABELS = {
    "Z": "Atomic number", "mass": "Atomic mass", "X": "Electronegativity",
    "row": "Periodic row", "group": "Group number",
    "mendeleev": "Mendeleev number", "radius": "Atomic radius",
}
_STAT_LABELS = {"mean": "mean", "std": "SD", "min": "min", "max": "max", "range": "range"}
_SPECIAL_LABELS = {
    "n_elements": "No. of elements", "Li_frac": "Li fraction",
    "a": "Lattice a", "b": "Lattice b", "c": "Lattice c",
    "alpha": "Angle α", "beta": "Angle β", "gamma": "Angle γ",
    "cell_volume": "Cell volume", "spacegroup_no": "Space-group no.",
    "Z": "Formula units (Z)", "crystal_system": "Crystal system",
    "Family": "Structural family",
}


def pretty_feature(name: str) -> str:
    """Raw feature column name -> human-readable label for figures.

    e.g. 'X_mean' -> 'Electronegativity (mean)', 'spacegroup_no' -> 'Space-group no.'."""
    if name in _SPECIAL_LABELS:
        return _SPECIAL_LABELS[name]
    if "_" in name:
        prop, stat = name.rsplit("_", 1)
        if prop in _PROP_LABELS and stat in _STAT_LABELS:
            return f"{_PROP_LABELS[prop]} ({_STAT_LABELS[stat]})"
    return name


def subscript_formula(s: str) -> str:
    """Render digit runs in a chemical formula as matplotlib mathtext subscripts.

    Font-independent (uses mathtext, not Unicode subscript glyphs, which many fonts
    lack). e.g. 'Li10GeP2S12' -> 'Li$_{10}$GeP$_{2}$S$_{12}$'."""
    return re.sub(r"(\d+)", r"$_{\1}$", str(s))
