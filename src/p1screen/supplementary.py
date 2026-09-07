"""Supplementary training diagnostics: boosting curves and a learning curve.

Both tables are descriptive.  They do not feed the production ensemble, the
strict-test metrics, or any candidate disposition; they exist so a reader can
see (a) how the three CatBoost configurations converge on the grouped folds and
(b) how strict-test error responds to the amount of training data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    LEARNING_CURVE_FRACTIONS,
    LEARNING_CURVE_REPEATS,
    MODEL_SPECS,
    RANDOM_SEED,
    SOURCE_DATA,
)
from .data import load_obelix_raw, split_audit
from .evaluation import grouped_folds, regression_metrics, screen_target_mask
from .features import featurize_obelix, representation_columns
from .modeling import make_model

BOOSTING_CURVES_CSV = "figS1a_boosting_curves.csv"
LEARNING_CURVE_CSV = "figS1b_learning_curve.csv"


def _boosting_curves(train: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per-iteration train/validation RMSE for every configuration on every grouped fold."""
    y = train.log10_sigma.to_numpy()
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(grouped_folds(train), start=1):
        X_train, X_valid = train.iloc[train_idx][columns], train.iloc[valid_idx][columns]
        y_train, y_valid = y[train_idx], y[valid_idx]
        for spec in MODEL_SPECS:
            model = make_model(spec, RANDOM_SEED)
            # use_best_model=False keeps the full trajectory; CatBoost would otherwise
            # truncate the model at the best validation iteration.
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=False)
            history = model.get_evals_result()
            validation_key = next(key for key in history if key.startswith("validation"))
            learn = history["learn"]["RMSE"]
            validation = history[validation_key]["RMSE"]
            if len(learn) != spec.iterations or len(validation) != spec.iterations:
                raise RuntimeError(
                    f"{spec.name} fold {fold}: expected {spec.iterations} iterations, "
                    f"received {len(learn)}/{len(validation)}"
                )
            for iteration, (train_rmse, validation_rmse) in enumerate(
                zip(learn, validation), start=1
            ):
                rows.append({
                    "config": spec.name,
                    "fold": fold,
                    "iteration": iteration,
                    "train_rmse": float(train_rmse),
                    "validation_rmse": float(validation_rmse),
                    "n_train": int(len(train_idx)),
                    "n_validation": int(len(valid_idx)),
                })
    return pd.DataFrame(rows)


def _learning_curve(
    strict_train: pd.DataFrame, test: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Strict-test error of the single base CatBoost versus composition-grouped training size."""
    base_spec = next(spec for spec in MODEL_SPECS if spec.name == "base")
    compositions = np.array(sorted(strict_train.composition_key.unique()))
    y_test = test.log10_sigma.to_numpy()
    target = screen_target_mask(test)
    rows = []
    for fraction in LEARNING_CURVE_FRACTIONS:
        full = np.isclose(fraction, 1.0)
        seeds = (0,) if full else range(LEARNING_CURVE_REPEATS)
        n_compositions = len(compositions) if full else max(
            1, int(round(fraction * len(compositions)))
        )
        for seed in seeds:
            if full:
                chosen = set(compositions)
            else:
                rng = np.random.default_rng(seed)
                chosen = set(rng.choice(compositions, size=n_compositions, replace=False))
            subset = strict_train.loc[strict_train.composition_key.isin(chosen)]
            model = make_model(base_spec, RANDOM_SEED)
            model.fit(subset[columns], subset.log10_sigma.to_numpy())
            prediction = model.predict(test[columns])
            metrics = regression_metrics(y_test, prediction)
            rows.append({
                "fraction": float(fraction),
                "repeat_seed": int(seed),
                "n_train_records": int(len(subset)),
                "n_train_compositions": int(n_compositions),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Spearman": metrics["Spearman"],
                "MAE_target_domain": float(np.abs(prediction[target] - y_test[target]).mean()),
            })
    return pd.DataFrame(rows)


def run_training_curves() -> dict:
    """Write the two supplementary source tables and return a short summary."""
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    raw_train, raw_test = load_obelix_raw()
    train, test = featurize_obelix(raw_train), featurize_obelix(raw_test)
    audit = split_audit(raw_train, raw_test)
    columns, _ = representation_columns("M0_formula")

    overlap = set(audit["composition_overlap_keys"])
    strict_train = train.loc[~train.composition_key.isin(overlap)].copy()
    if len(strict_train) != audit["n_train_strict"]:
        raise RuntimeError("strict training set does not match the split audit")

    boosting = _boosting_curves(train, columns)
    learning = _learning_curve(strict_train, test, columns)
    boosting.to_csv(SOURCE_DATA / BOOSTING_CURVES_CSV, index=False)
    learning.to_csv(SOURCE_DATA / LEARNING_CURVE_CSV, index=False)

    summary = {"boosting": {}, "learning": {}}
    for spec in MODEL_SPECS:
        mean_curve = (
            boosting.loc[boosting.config.eq(spec.name)]
            .groupby("iteration").validation_rmse.mean()
        )
        summary["boosting"][spec.name] = {
            "final_validation_rmse": float(mean_curve.iloc[-1]),
            "min_validation_rmse": float(mean_curve.min()),
            "min_validation_iteration": int(mean_curve.idxmin()),
        }
    grouped = learning.groupby("fraction")
    summary["learning"] = {
        f"{fraction:.1f}": {
            "n_train_records": float(part.n_train_records.mean()),
            "MAE_mean": float(part.MAE.mean()),
            "Spearman_mean": float(part.Spearman.mean()),
        }
        for fraction, part in grouped
    }
    summary["n_fits"] = int(len(boosting.groupby(["config", "fold"])) + len(learning))
    return summary
