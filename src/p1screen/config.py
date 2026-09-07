"""Immutable release settings and repository paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_VERSION = "0.1.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("P1SCREEN_ROOT", DEFAULT_ROOT)).expanduser().resolve()
ROOT_SOURCE = "P1SCREEN_ROOT" if "P1SCREEN_ROOT" in os.environ else "package_location"
DATA = ROOT / "data"
SOURCE_DATA = ROOT / "source_data"
FIGURES = ROOT / "figures"
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"

OBELIX_TRAIN = DATA / "train.csv"
OBELIX_TEST = DATA / "test.csv"
MP_SNAPSHOT = DATA / "mp_snapshot.csv"
MP_QUERY = DATA / "mp_query.json"
METRICS = DATA / "metrics.json"
MODEL_MANIFEST = DATA / "model_manifest.json"
RELEASE_MANIFEST = DATA / "release_manifest.json"
MP_LEDGER = ROOT / "mp_ledger.csv"
CANDIDATE_QUEUE = ROOT / "candidate_queue.csv"

RANDOM_SEED = 42
BOOTSTRAP_SEED = 20260716
N_BOOTSTRAP = 4000
N_FOLDS = 5
N_SEEDS_PER_CONFIG = 5
AD_VARIANCE = 0.95
AD_NEIGHBORS = 5
ROUTE_QUANTILE = 0.75

MP_DATABASE_VERSION = "2026.04.13"
MP_EXPECTED_QUERY_COUNT = 248

REDOX_RISK_ELEMENTS = frozenset({"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "W"})


@dataclass(frozen=True)
class ModelSpec:
    name: str
    iterations: int
    depth: int
    learning_rate: float


MODEL_SPECS = (
    ModelSpec("compact", 400, 5, 0.05),
    ModelSpec("base", 600, 6, 0.05),
    ModelSpec("deep_slow", 800, 7, 0.03),
)

MAIN_FIGURES = (
    "01_study_design.png",
    "02_model_validation.png",
    "03_transfer_and_risk.png",
    "04_screening_landscape.png",
    "05_candidate_atlas.png",
)

SUPPLEMENTARY = FIGURES / "supplementary"
SUPPLEMENTARY_FIGURES = ("S1_training_curves.png",)

# Supplementary learning curve: composition-grouped subsampling of the strict training set.
LEARNING_CURVE_FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
LEARNING_CURVE_REPEATS = 5
