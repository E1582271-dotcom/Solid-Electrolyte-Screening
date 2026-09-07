from p1screen.data import load_obelix_raw, split_audit


def test_obelix_split_and_censor_contract():
    train, test = load_obelix_raw()
    audit = split_audit(train, test)
    assert (len(train), len(test)) == (478, 121)
    assert audit["n_train_composition_groups"] == 390
    assert audit["n_test_composition_groups"] == 112
    assert audit["n_composition_overlap_groups"] == 2
    assert audit["n_train_rows_removed_for_strict_test"] == 2
    assert audit["n_train_strict"] == 476
    assert audit["n_test_composition_novel"] == 119
    assert audit["n_doi_overlap"] == 0
    assert (audit["n_censored_train"], audit["n_censored_test"]) == (29, 8)
    assert audit["censor_directions"] == ["", "<"]
    assert audit["n_formula_identity_mismatch_train"] == 6
    assert audit["n_formula_identity_mismatch_test"] == 6
    assert train.composition_key.equals(train.feature_composition_key)
    assert test.composition_key.equals(test.feature_composition_key)
