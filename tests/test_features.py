import numpy as np

from p1screen.data import load_obelix_raw
from p1screen.features import featurize_obelix, formula_matrix, representation_columns


def test_formula_representation_contract():
    columns, categorical = representation_columns("M0_formula")
    assert len(columns) == 134
    assert categorical == []
    matrix = formula_matrix(["Li6PS5Cl", "Li3PS4"])
    assert matrix.shape == (2, 134)
    assert np.isfinite(matrix.to_numpy(dtype=float)).all()


def test_scale_invariant_cell_summary_reduces_convention_variation():
    train, _ = load_obelix_raw()
    features = featurize_obelix(train)
    grouped = features.groupby("composition_key")
    repeated = [group for _, group in grouped if len(group) > 1]
    raw_ratios = [group.cell_volume.max() / group.cell_volume.min() for group in repeated]
    normalized = [group.volume_per_atom.max() / group.volume_per_atom.min() for group in repeated]
    assert max(raw_ratios) > 4.0
    assert max(normalized) < 1.52
    assert sum(value > 1.5 for value in normalized) == 1


def test_true_composition_controls_features_and_volume_normalization():
    train, _ = load_obelix_raw()
    row = train.loc[train.ID.eq("r5y")]
    features = featurize_obelix(row).iloc[0]
    assert features.feature_formula == "Li1.6Cd8.0Cl32.0"
    assert features.composition_key == features.feature_composition_key
    assert features.composition_key != features.nominal_composition_key
    assert np.isclose(features.volume_per_atom, features.cell_volume / 41.6)
