"""Strict grouped evaluation and publication source-data generation."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .composition import element_symbols
from .config import (
    BOOTSTRAP_SEED,
    METRICS,
    MODEL_SPECS,
    N_BOOTSTRAP,
    N_FOLDS,
    RANDOM_SEED,
    REDOX_RISK_ELEMENTS,
    SOURCE_DATA,
)
from .data import load_obelix_raw, split_audit
from .domain import ApplicabilityDomain
from .features import featurize_obelix, representation_columns
from .modeling import ensemble_predictions, make_model, train_ensemble


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rank_correlation = np.nan
    order_correlation = np.nan
    if np.ptp(np.asarray(y_pred, dtype=float)) > 1e-12 and np.ptp(
        np.asarray(y_true, dtype=float)
    ) > 1e-12:
        rank_correlation = spearmanr(y_true, y_pred).statistic
        order_correlation = kendalltau(y_true, y_pred).statistic
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(rank_correlation),
        "Kendall": float(order_correlation),
        "bias": float(np.mean(np.asarray(y_pred) - np.asarray(y_true))),
    }


def _json_number(value: float) -> float | None:
    """Return strict-JSON numbers while preserving undefined correlations as null."""
    number = float(value)
    return number if np.isfinite(number) else None


def _json_compatible(value):
    """Recursively replace non-finite scalars in metric payloads with JSON null."""
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return _json_number(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def grouped_folds(features: pd.DataFrame):
    y = features.log10_sigma.to_numpy()
    bins = pd.qcut(y, 5, labels=False, duplicates="drop")
    groups = features.composition_key.to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    return list(splitter.split(features, bins, groups))


def screen_target_mask(features: pd.DataFrame) -> np.ndarray:
    """Identify experimental records matching the chemistry of the MP candidate pool."""
    mask = []
    for formula in features.feature_formula:
        elements = set(element_symbols(formula))
        mask.append(
            {"Li", "S"}.issubset(elements)
            and "H" not in elements
            and "O" not in elements
            and 3 <= len(elements) <= 5
            and not bool(elements & REDOX_RISK_ELEMENTS)
        )
    return np.asarray(mask, dtype=bool)


def composition_level_metrics(y, prediction, groups) -> dict[str, float]:
    grouped = pd.DataFrame({
        "y": np.asarray(y, dtype=float),
        "prediction": np.asarray(prediction, dtype=float),
        "group": np.asarray(groups, dtype=object),
    }).groupby("group", as_index=False).agg({"y": "mean", "prediction": "mean"})
    return regression_metrics(grouped.y, grouped.prediction)


def equal_composition_weights(features: pd.DataFrame) -> np.ndarray:
    counts = features.groupby("composition_key")["composition_key"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    return weights / weights.mean()


def risk_correlations_for_frame(frame: pd.DataFrame, fold_normalized: bool = False) -> dict:
    ranked = frame.copy()
    if fold_normalized:
        ranked["spread_rank"] = ranked.groupby("fold").spread.rank(pct=True)
        ranked["ad_rank"] = ranked.groupby("fold").ad_distance.rank(pct=True)
    else:
        ranked["spread_rank"] = ranked.spread.rank(pct=True)
        ranked["ad_rank"] = ranked.ad_distance.rank(pct=True)
    ranked["combined_rank_product"] = ranked.spread_rank * ranked.ad_rank
    spread_result = spearmanr(ranked.spread_rank, ranked.absolute_error)
    ad_result = spearmanr(ranked.ad_rank, ranked.absolute_error)
    combined_result = spearmanr(ranked.combined_rank_product, ranked.absolute_error)
    return {
        "spread_error_rho": float(spread_result.statistic),
        "spread_error_p": float(spread_result.pvalue),
        "ad_error_rho": float(ad_result.statistic),
        "ad_error_p": float(ad_result.pvalue),
        "combined_error_rho": float(combined_result.statistic),
        "combined_error_p": float(combined_result.pvalue),
        "spread_ad_rho": float(spearmanr(ranked.spread_rank, ranked.ad_rank).statistic),
        "rank_normalization": "within_fold" if fold_normalized else "within_sample",
    }


def _cluster_bootstrap_risk_correlations(
    frame: pd.DataFrame,
    *,
    fold_normalized: bool = False,
) -> dict:
    """Composition-cluster bootstrap intervals for the four risk correlations."""
    groups = frame.composition_key.astype(str).to_numpy(dtype=object)
    unique = list(dict.fromkeys(groups.tolist()))
    indices = {group: np.where(groups == group)[0] for group in unique}
    signal_keys = (
        "spread_error_rho",
        "ad_error_rho",
        "combined_error_rho",
        "spread_ad_rho",
    )
    values = {key: [] for key in signal_keys}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(N_BOOTSTRAP):
        sampled = rng.integers(0, len(unique), size=len(unique))
        row_indices = np.concatenate([indices[unique[item]] for item in sampled])
        statistics = risk_correlations_for_frame(
            frame.iloc[row_indices],
            fold_normalized=fold_normalized,
        )
        for key in signal_keys:
            values[key].append(statistics[key])
    return {
        "n_bootstrap": N_BOOTSTRAP,
        "n_records": len(frame),
        "n_compositions": len(unique),
        "rank_normalization": "within_fold" if fold_normalized else "within_sample",
        "ci95": {
            key.removesuffix("_rho"): np.nanquantile(
                np.asarray(samples, dtype=float), [0.025, 0.975]
            ).tolist()
            for key, samples in values.items()
        },
    }


def _ridge_estimator(X, y, groups):
    estimator = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge()),
    ])
    search = GridSearchCV(
        estimator,
        {"model__alpha": np.logspace(-4, 4, 17)},
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(4),
        n_jobs=1,
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_, float(search.best_params_["model__alpha"])


def _cluster_bootstrap(y, prediction, groups) -> dict:
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    groups = np.asarray(groups, dtype=object)
    unique = list(dict.fromkeys(groups.tolist()))
    indices = {group: np.where(groups == group)[0] for group in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([indices[unique[item]] for item in sampled])
        values.append([
            mean_absolute_error(y[idx], prediction[idx]),
            np.sqrt(mean_squared_error(y[idx], prediction[idx])),
            spearmanr(y[idx], prediction[idx]).statistic,
        ])
    array = np.asarray(values, float)
    return {
        "n_bootstrap": N_BOOTSTRAP,
        "n_groups": len(unique),
        "MAE_ci95": np.quantile(array[:, 0], [0.025, 0.975]).tolist(),
        "RMSE_ci95": np.quantile(array[:, 1], [0.025, 0.975]).tolist(),
        "Spearman_ci95": np.nanquantile(array[:, 2], [0.025, 0.975]).tolist(),
    }


def _cluster_bootstrap_mae_delta(y, prediction_a, prediction_b, groups) -> dict:
    """Paired cluster-bootstrap MAE(A) minus MAE(B)."""
    y = np.asarray(y, float)
    prediction_a = np.asarray(prediction_a, float)
    prediction_b = np.asarray(prediction_b, float)
    groups = np.asarray(groups, dtype=object)
    unique = list(dict.fromkeys(groups.tolist()))
    indices = {group: np.where(groups == group)[0] for group in unique}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.integers(0, len(unique), size=len(unique))
        idx = np.concatenate([indices[unique[item]] for item in sampled])
        deltas.append(
            mean_absolute_error(y[idx], prediction_a[idx])
            - mean_absolute_error(y[idx], prediction_b[idx])
        )
    return {
        "definition": "MAE(perturbation ensemble) - MAE(comparator)",
        "observed": float(
            mean_absolute_error(y, prediction_a) - mean_absolute_error(y, prediction_b)
        ),
        "ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "n_bootstrap": N_BOOTSTRAP,
        "n_groups": len(unique),
    }


def run_evaluation() -> dict:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    raw_train, raw_test = load_obelix_raw()
    train, test = featurize_obelix(raw_train), featurize_obelix(raw_test)
    audit = split_audit(raw_train, raw_test)
    columns, _ = representation_columns("M0_formula")
    folds = grouped_folds(train)
    y = train.log10_sigma.to_numpy()
    cv_rows, oof_rows = [], []

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        X_train, X_valid = train.iloc[train_idx][columns], train.iloc[valid_idx][columns]
        y_train, y_valid = y[train_idx], y[valid_idx]
        groups_train = train.iloc[train_idx].composition_key.to_numpy()

        median = DummyRegressor(strategy="median").fit(X_train, y_train)
        ridge, ridge_alpha = _ridge_estimator(X_train, y_train, groups_train)
        forest = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=500, random_state=RANDOM_SEED, n_jobs=1
            )),
        ]).fit(X_train, y_train)
        ensemble = train_ensemble(X_train, y_train)
        ensemble_mean, ensemble_spread, _ = ensemble_predictions(ensemble, X_valid)

        predictions = {
            "Median": median.predict(X_valid),
            "Ridge": ridge.predict(X_valid),
            "RandomForest": forest.predict(X_valid),
            "Perturbation ensemble": ensemble_mean,
        }
        for model_name, prediction in predictions.items():
            cv_rows.append({
                "model": model_name,
                "fold": fold,
                "n_train": len(train_idx),
                "n_validation": len(valid_idx),
                "n_validation_groups": train.iloc[valid_idx].composition_key.nunique(),
                "ridge_alpha": ridge_alpha if model_name == "Ridge" else np.nan,
                **regression_metrics(y_valid, prediction),
            })

        domain = ApplicabilityDomain().fit(X_train)
        distances = domain.distance(X_valid)
        for local, row_index in enumerate(valid_idx):
            oof_rows.append({
                "record_id": train.iloc[row_index].record_id,
                "composition_key": train.iloc[row_index].composition_key,
                "fold": fold,
                "true_log10_sigma": y_valid[local],
                "prediction": ensemble_mean[local],
                "spread": ensemble_spread[local],
                "ad_distance": distances[local],
                "absolute_error": abs(y_valid[local] - ensemble_mean[local]),
                "censored": train.iloc[row_index].censored,
            })

    cv = pd.DataFrame(cv_rows)
    oof = pd.DataFrame(oof_rows).sort_values("record_id", kind="stable")
    target_by_record = pd.Series(
        screen_target_mask(train), index=train.record_id.astype(str)
    )
    oof["target_domain"] = oof.record_id.astype(str).map(target_by_record).astype(bool)
    cv.to_csv(SOURCE_DATA / "fig02b_grouped_cv.csv", index=False)
    oof.to_csv(SOURCE_DATA / "fig03d_oof_risk.csv", index=False)

    overlap = set(audit["composition_overlap_keys"])
    strict_mask = ~train.composition_key.isin(overlap)
    strict_train = train.loc[strict_mask].copy()
    strict_X = strict_train[columns]
    test_X = test[columns]
    strict_y = strict_train.log10_sigma.to_numpy()
    strict_groups = strict_train.composition_key.to_numpy()

    strict_median = DummyRegressor(strategy="median").fit(strict_X, strict_y)
    strict_ridge, strict_ridge_alpha = _ridge_estimator(
        strict_X, strict_y, strict_groups
    )
    strict_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_SEED, n_jobs=1
        )),
    ]).fit(strict_X, strict_y)
    base_spec = next(spec for spec in MODEL_SPECS if spec.name == "base")
    strict_base = make_model(base_spec, RANDOM_SEED).fit(strict_X, strict_y)
    test_ensemble = train_ensemble(strict_train[columns], strict_train.log10_sigma)
    test_mean, test_spread, _ = ensemble_predictions(test_ensemble, test[columns])
    strict_predictions = {
        "Median": strict_median.predict(test_X),
        "Ridge": strict_ridge.predict(test_X),
        "RandomForest": strict_forest.predict(test_X),
        "Single base CatBoost": strict_base.predict(test_X),
        "Perturbation ensemble": test_mean,
    }
    strict_domain = ApplicabilityDomain().fit(strict_train[columns])
    test_ad = strict_domain.distance(test[columns])
    target_domain = screen_target_mask(test)
    test_predictions = pd.DataFrame({
        "record_id": test.record_id,
        "composition_key": test.composition_key,
        "true_log10_sigma": test.log10_sigma,
        "prediction": test_mean,
        "spread": test_spread,
        "ad_distance": test_ad,
        "absolute_error": np.abs(test.log10_sigma.to_numpy() - test_mean),
        "censored": test.censored,
        "censor_direction": test.censor_direction,
        "composition_novel": ~test.composition_key.isin(overlap),
        "target_domain": target_domain,
    })
    test_predictions.to_csv(SOURCE_DATA / "fig02c_test_predictions.csv", index=False)

    test_metrics = regression_metrics(test.log10_sigma, test_mean)
    novel_mask = test_predictions.composition_novel.to_numpy(bool)
    novel_metrics = regression_metrics(
        test.log10_sigma.to_numpy()[novel_mask], test_mean[novel_mask]
    )
    target_metrics = regression_metrics(
        test.log10_sigma.to_numpy()[target_domain], test_mean[target_domain]
    )
    composition_metrics = composition_level_metrics(
        test.log10_sigma, test_mean, test.composition_key
    )
    strict_baseline_rows = []
    strict_baselines = {}
    for model_name, prediction in strict_predictions.items():
        samples = {
            "all_records": np.ones(len(test), dtype=bool),
            "target_domain": target_domain,
        }
        strict_baselines[model_name] = {}
        for sample_name, sample_mask in samples.items():
            sample_metrics = regression_metrics(
                test.log10_sigma.to_numpy()[sample_mask], prediction[sample_mask]
            )
            strict_baselines[model_name][sample_name] = sample_metrics
            strict_baseline_rows.append({
                "model": model_name,
                "sample": sample_name,
                "n_records": int(sample_mask.sum()),
                "ridge_alpha": strict_ridge_alpha if model_name == "Ridge" else np.nan,
                **sample_metrics,
            })
    pd.DataFrame(strict_baseline_rows).to_csv(
        SOURCE_DATA / "fig02e_strict_baselines.csv", index=False
    )

    # Sensitivity of the primary ensemble to repeated-composition weighting and
    # treating one-sided bounds as point labels during training.
    balanced_ensemble = train_ensemble(
        strict_X,
        strict_y,
        sample_weight=equal_composition_weights(strict_train),
    )
    balanced_mean, _, _ = ensemble_predictions(balanced_ensemble, test_X)
    uncensored_train = strict_train.loc[~strict_train.censored].copy()
    uncensored_ensemble = train_ensemble(
        uncensored_train[columns], uncensored_train.log10_sigma
    )
    uncensored_train_mean, _, _ = ensemble_predictions(uncensored_ensemble, test_X)
    training_sensitivity_predictions = {
        "record_weighted": test_mean,
        "composition_balanced": balanced_mean,
        "exclude_censored_training": uncensored_train_mean,
    }
    training_sensitivity = {}
    training_sensitivity_rows = []
    for definition, prediction in training_sensitivity_predictions.items():
        training_sensitivity[definition] = {}
        for sample_name, sample_mask in {
            "all_records": np.ones(len(test), dtype=bool),
            "target_domain": target_domain,
            "uncensored_records": ~test.censored.to_numpy(bool),
        }.items():
            sample_metrics = regression_metrics(
                test.log10_sigma.to_numpy()[sample_mask], prediction[sample_mask]
            )
            training_sensitivity[definition][sample_name] = sample_metrics
            training_sensitivity_rows.append({
                "training_definition": definition,
                "sample": sample_name,
                "n_records": int(sample_mask.sum()),
                **sample_metrics,
            })
    pd.DataFrame(training_sensitivity_rows).to_csv(
        SOURCE_DATA / "fig02f_training_sensitivity.csv", index=False
    )
    point_error = np.abs(test.log10_sigma.to_numpy() - test_mean)
    one_sided = point_error.copy()
    censored = test.censored.to_numpy(bool)
    one_sided[censored] = np.maximum(
        test_mean[censored] - test.log10_sigma.to_numpy()[censored], 0.0
    )
    censor_sensitivity = {
        "point_MAE": float(point_error.mean()),
        "one_sided_MAE": float(one_sided.mean()),
        "uncensored_MAE": float(point_error[~censored].mean()),
        "n_censored": int(censored.sum()),
    }
    pd.DataFrame([
        {"definition": "point_MAE", "value": censor_sensitivity["point_MAE"]},
        {"definition": "one_sided_MAE", "value": censor_sensitivity["one_sided_MAE"]},
        {"definition": "uncensored_MAE", "value": censor_sensitivity["uncensored_MAE"]},
    ]).to_csv(SOURCE_DATA / "fig02d_censor_sensitivity.csv", index=False)
    bootstrap = _cluster_bootstrap(
        test.log10_sigma, test_mean, test.composition_key.to_numpy(dtype=object)
    )
    target_bootstrap = _cluster_bootstrap(
        test.log10_sigma.to_numpy()[target_domain],
        test_mean[target_domain],
        test.loc[target_domain, "composition_key"].to_numpy(dtype=object),
    )
    strict_model_comparisons = {}
    for model_name, prediction in strict_predictions.items():
        if model_name == "Perturbation ensemble":
            continue
        strict_model_comparisons[model_name] = {
            "all_records": _cluster_bootstrap_mae_delta(
                test.log10_sigma, test_mean, prediction, test.composition_key
            ),
            "target_domain": _cluster_bootstrap_mae_delta(
                test.log10_sigma.to_numpy()[target_domain],
                test_mean[target_domain],
                prediction[target_domain],
                test.loc[target_domain, "composition_key"],
            ),
        }

    # Matched representation ablation on the same outer folds.
    ablation_rows = []
    for representation in ("M0_formula", "M1_cell_normalized", "M2_legacy"):
        rep_columns, categorical = representation_columns(representation)
        for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
            model = make_model(base_spec, RANDOM_SEED)
            model.fit(
                train.iloc[train_idx][rep_columns], y[train_idx], cat_features=categorical
            )
            prediction = model.predict(train.iloc[valid_idx][rep_columns])
            ablation_rows.append({
                "representation": representation,
                "fold": fold,
                "n_features": len(rep_columns),
                **regression_metrics(y[valid_idx], prediction),
            })
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(SOURCE_DATA / "fig03a_representation_ablation.csv", index=False)

    duplicate = train[["composition_key", "cell_volume", "volume_per_atom"]].copy()
    convention = (duplicate.groupby("composition_key")
        .agg(
            n_records=("composition_key", "size"),
            raw_volume_ratio=("cell_volume", lambda x: float(x.max() / x.min())),
            volume_per_atom_ratio=(
                "volume_per_atom", lambda x: float(x.max() / x.min())
            ),
        )
        .query("n_records > 1")
        .reset_index())
    convention.to_csv(SOURCE_DATA / "fig03b_cell_convention.csv", index=False)

    target_source = pd.concat([
        raw_train[["ID", "source_split", "composition_key", "log10_sigma", "censored", "censor_direction"]],
        raw_test[["ID", "source_split", "composition_key", "log10_sigma", "censored", "censor_direction"]],
    ], ignore_index=True).rename(columns={"ID": "record_id"})
    target_source.to_csv(SOURCE_DATA / "fig02a_targets.csv", index=False)
    pd.DataFrame([audit]).drop(columns=[
        "composition_overlap_keys", "nominal_composition_overlap_keys"
    ]).to_csv(
        SOURCE_DATA / "fig01b_split_audit.csv", index=False
    )

    risk_correlations = {
        "oof": risk_correlations_for_frame(oof, fold_normalized=True),
        "test": risk_correlations_for_frame(test_predictions),
        "oof_target_domain": risk_correlations_for_frame(
            oof.loc[oof.target_domain], fold_normalized=True
        ),
        "test_target_domain": risk_correlations_for_frame(
            test_predictions.loc[test_predictions.target_domain]
        ),
    }
    risk_frames = {
        "oof": (oof, True),
        "test": (test_predictions, False),
        "oof_target_domain": (oof.loc[oof.target_domain], True),
        "test_target_domain": (
            test_predictions.loc[test_predictions.target_domain],
            False,
        ),
    }
    risk_correlation_bootstrap = {
        sample: _cluster_bootstrap_risk_correlations(
            frame,
            fold_normalized=fold_normalized,
        )
        for sample, (frame, fold_normalized) in risk_frames.items()
    }
    risk_rows = []
    for sample, values in risk_correlations.items():
        bootstrap_result = risk_correlation_bootstrap[sample]
        for signal, value in values.items():
            if not signal.endswith("_rho"):
                continue
            signal_name = signal.removesuffix("_rho")
            ci95 = bootstrap_result["ci95"][signal_name]
            risk_rows.append({
                "sample": sample,
                "signal": signal_name,
                "spearman_rho": value,
                "ci95_low": ci95[0],
                "ci95_high": ci95[1],
                "n_records": bootstrap_result["n_records"],
                "n_compositions": bootstrap_result["n_compositions"],
                "n_bootstrap": bootstrap_result["n_bootstrap"],
                "rank_normalization": bootstrap_result["rank_normalization"],
            })
    pd.DataFrame(risk_rows).to_csv(
        SOURCE_DATA / "fig03c_risk_correlations.csv", index=False
    )

    summary = {
        "split_audit": audit,
        "grouped_cv": {
            model: {
                metric: [
                    _json_number(group[metric].mean()),
                    _json_number(group[metric].std(ddof=1)),
                ]
                for metric in ("MAE", "RMSE", "R2", "Spearman")
            }
            for model, group in cv.groupby("model")
        },
        "strict_test": test_metrics,
        "composition_novel_test": novel_metrics,
        "target_domain_test": {
            "definition": "Li+S; no H/O; 3-5 elements; no configured redox-risk element",
            "n_records": int(target_domain.sum()),
            "n_compositions": int(test.loc[target_domain, "composition_key"].nunique()),
            **target_metrics,
        },
        "composition_level_test": {
            "n_compositions": int(test.composition_key.nunique()),
            **composition_metrics,
        },
        "strict_test_baselines": strict_baselines,
        "strict_model_comparisons": strict_model_comparisons,
        "training_sensitivity": training_sensitivity,
        "cluster_bootstrap": bootstrap,
        "target_domain_cluster_bootstrap": target_bootstrap,
        "censor_sensitivity": censor_sensitivity,
        "risk_correlations": risk_correlations,
        "risk_correlation_bootstrap": risk_correlation_bootstrap,
        "representation_ablation": {
            name: {
                metric: [
                    _json_number(group[metric].mean()),
                    _json_number(group[metric].std(ddof=1)),
                ]
                for metric in ("MAE", "RMSE", "R2", "Spearman")
            }
            for name, group in ablation.groupby("representation")
        },
        "cell_convention": {
            "n_repeated_compositions": int(len(convention)),
            "raw_ratio_max": float(convention.raw_volume_ratio.max()),
            "raw_ratio_gt_1_5": int((convention.raw_volume_ratio > 1.5).sum()),
            "v_per_atom_ratio_max": float(convention.volume_per_atom_ratio.max()),
            "v_per_atom_ratio_gt_1_5": int((convention.volume_per_atom_ratio > 1.5).sum()),
        },
    }
    summary = _json_compatible(summary)
    METRICS.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
