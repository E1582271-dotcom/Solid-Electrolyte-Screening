from p1screen.composition import composition_key, normalized_doi


def test_composition_key_is_scale_and_order_invariant():
    expected = composition_key("Li3.35Ge0.35P0.65S4")
    assert composition_key("Li10.05Ge1.05P1.95S12") == expected
    assert composition_key("S4P0.65Ge0.35Li3.35") == expected


def test_doi_normalization():
    assert normalized_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalized_doi("DOI: 10.1000/ABC") == "10.1000/abc"
    assert normalized_doi(float("nan")) == ""
