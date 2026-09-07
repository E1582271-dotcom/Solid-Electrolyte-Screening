"""Canonical formula handling shared by data, screening, and tests."""
from __future__ import annotations

import re
from collections.abc import Mapping

from pymatgen.core import Composition


def normalize_formula_text(value: object) -> str:
    """Normalize dataset separators without changing stoichiometry.

    OBELiX contains one hydrate written with ``*`` between formula groups.
    Pymatgen parses the same expression once that separator is removed.
    """
    text = str(value).strip().replace(" ", "")
    text = text.replace("*", "").replace("·", "")
    return text


def as_composition(value: object) -> Composition:
    return Composition(normalize_formula_text(value))


def composition_key(value: object, decimals: int = 8) -> str:
    """Scale- and ordering-invariant elemental-fraction key."""
    fractional = as_composition(value).fractional_composition
    parts = [
        f"{element}:{float(amount):.{decimals}f}"
        for element, amount in sorted(fractional.items(), key=lambda item: str(item[0]))
        if float(amount) > 10 ** (-(decimals + 1))
    ]
    return "|".join(parts)


def reduced_formula(value: object) -> str:
    return as_composition(value).reduced_formula


def element_symbols(value: object) -> tuple[str, ...]:
    return tuple(sorted(str(element) for element in as_composition(value).elements))


def contains_element(value: object, symbol: str) -> bool:
    return symbol in element_symbols(value)


def normalized_doi(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"", "nan", "none"}:
        return ""
    text = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text


def jsonable_database_ids(value: object) -> dict[str, list[str]]:
    """Normalize MP database IDs to a deterministic JSON-compatible mapping."""
    if value is None:
        return {}
    mapping: Mapping = value if isinstance(value, Mapping) else dict(value)
    return {
        str(key).lower(): sorted(str(item) for item in (items if isinstance(items, list) else [items]))
        for key, items in sorted(mapping.items(), key=lambda item: str(item[0]).lower())
    }

