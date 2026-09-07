"""CatBoost perturbation ensemble used for evaluation and production ranking."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import MODEL_SPECS, N_SEEDS_PER_CONFIG


def ensemble_specs():
    for spec in MODEL_SPECS:
        for seed in range(N_SEEDS_PER_CONFIG):
            yield spec, seed


def make_model(spec, seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=spec.iterations,
        depth=spec.depth,
        learning_rate=spec.learning_rate,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def train_ensemble(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    categorical: list[str] | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> list[tuple[str, int, CatBoostRegressor]]:
    categorical = categorical or []
    models = []
    for spec, seed in ensemble_specs():
        model = make_model(spec, seed)
        model.fit(X, y, cat_features=categorical, sample_weight=sample_weight)
        models.append((spec.name, seed, model))
    return models


def ensemble_predictions(models, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.stack([model.predict(X) for _, _, model in models])
    return predictions.mean(axis=0), predictions.std(axis=0), predictions


def save_ensemble(models, directory: Path) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    spec_map = {spec.name: spec for spec in MODEL_SPECS}
    for name, seed, model in models:
        path = directory / f"{name}_seed{seed:02d}.cbm"
        model.save_model(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({
            "config": asdict(spec_map[name]),
            "seed": seed,
            "path": path.name,
            "sha256": digest,
            "feature_names": list(model.feature_names_),
        })
    return records


def load_ensemble(directory: Path, manifest: dict) -> list[tuple[str, int, CatBoostRegressor]]:
    models = []
    for member in manifest["members"]:
        path = directory / member["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
            raise RuntimeError(f"model checksum mismatch: {path}")
        model = CatBoostRegressor()
        model.load_model(path)
        models.append((member["config"]["name"], int(member["seed"]), model))
    return models


def write_model_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
