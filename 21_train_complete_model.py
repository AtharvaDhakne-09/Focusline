"""Train and evaluate the complete leakage-aware HCSD recovery model.

This script trains two models for the existing target
``objective_recovery_time_seconds_target``:

* ``early``: features available before an interruption starts.
* ``duration``: the same features plus the completed interruption duration.

The repository does not contain a verified resumption timestamp or a
``recovery_time_remaining_seconds_target`` column. The duration model therefore
does not claim to predict a separately observed Stage 2 remaining-time target.
It predicts the existing target after duration becomes known.

The supplied train/test files are used unchanged. Model selection happens only
with grouped cross-validation inside the training participants; the test
participants are used once for final evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "HCSD_train_dataset.csv"
TEST_PATH = BASE_DIR / "HCSD_test_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "objective_recovery_time_seconds_target"
GROUP = "participant_id"
RANDOM_STATE = 42

NUMERIC_EARLY = [
    "sleep_hours",
    "sleep_quality_1_5",
    "pre_fatigue_1_5",
    "pre_stress_1_5",
    "pre_focus_1_5",
    "task_familiarity_1_5",
    "expected_difficulty_1_5",
    "baseline_typing_wpm",
    "baseline_error_rate",
    "baseline_pause_seconds",
    "interruption_urgency_1_5",
    "interruption_relevance_1_5",
    "interruption_cognitive_demand_1_5",
    "emotional_disturbance_1_5",
]
CATEGORICAL_EARLY = [
    "condition",
    "task_type",
    "environment",
    "deadline_pressure",
    "interruption_type",
    "interruption_expectedness",
]
DURATION = "interruption_duration_seconds"


def validate_data(train: pd.DataFrame, test: pd.DataFrame) -> None:
    required = set(NUMERIC_EARLY + CATEGORICAL_EARLY + [DURATION, TARGET, GROUP])
    for name, frame in (("training", train), ("test", test)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing columns in {name} data: {missing}")
        if frame[[TARGET, GROUP]].isna().any().any():
            raise ValueError(f"{name} data contains missing target or participant IDs.")

    overlap = set(train[GROUP]) & set(test[GROUP])
    if overlap:
        raise ValueError(
            f"Participant leakage detected: {len(overlap)} IDs occur in both splits."
        )
    if (train[TARGET] < 0).any() or (test[TARGET] < 0).any():
        raise ValueError("Recovery-time targets must be non-negative.")


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    estimator: Any,
) -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", estimator),
        ]
    )


def candidate_models(
    numeric_features: list[str], categorical_features: list[str]
) -> dict[str, Pipeline]:
    return {
        "ridge": build_pipeline(
            numeric_features,
            categorical_features,
            Ridge(alpha=1.0),
        ),
        "random_forest": build_pipeline(
            numeric_features,
            categorical_features,
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }


def select_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[str, Pipeline, pd.DataFrame]:
    n_splits = min(5, groups.nunique())
    if n_splits < 2:
        raise ValueError("At least two training participants are required for CV.")

    splitter = GroupKFold(n_splits=n_splits)
    rows: list[dict[str, float | str]] = []
    models = candidate_models(numeric_features, categorical_features)
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=splitter,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=1,
        )
        rows.append(
            {
                "model": name,
                "cv_mae_seconds": float(-scores["test_mae"].mean()),
                "cv_mae_std_seconds": float(scores["test_mae"].std()),
                "cv_r2": float(scores["test_r2"].mean()),
            }
        )

    cv_results = pd.DataFrame(rows).sort_values("cv_mae_seconds").reset_index(drop=True)
    selected_name = str(cv_results.iloc[0]["model"])
    selected = models[selected_name]
    selected.fit(X, y)
    return selected_name, selected, cv_results


def metrics(
    model_name: str, actual: pd.Series, predicted: np.ndarray, split: str
) -> dict[str, float | str]:
    return {
        "split": split,
        "model": model_name,
        "MAE_seconds": float(mean_absolute_error(actual, predicted)),
        "RMSE_seconds": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "median_absolute_error_seconds": float(
            np.median(np.abs(actual.to_numpy() - predicted))
        ),
        "p90_absolute_error_seconds": float(
            np.quantile(np.abs(actual.to_numpy() - predicted), 0.90)
        ),
        "rows": int(len(actual)),
    }


def forest_interval(model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return empirical 10th/90th percentile intervals for a fitted forest."""
    forest = model.named_steps["model"]
    if not isinstance(forest, RandomForestRegressor):
        raise TypeError("Prediction intervals require the selected model to be a forest.")
    transformed = model.named_steps["preprocessing"].transform(X)
    tree_predictions = np.vstack(
        [tree.predict(transformed) for tree in forest.estimators_]
    )
    return np.quantile(tree_predictions, 0.10, axis=0), np.quantile(
        tree_predictions, 0.90, axis=0
    )


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    validate_data(train, test)

    y_train = train[TARGET].astype(float)
    y_test = test[TARGET].astype(float)
    train_groups = train[GROUP]
    metrics_rows: list[dict[str, float | str]] = []
    prediction_output = test[[GROUP, "trial_id", TARGET]].copy()
    cv_output: list[pd.DataFrame] = []
    model_bundle: dict[str, Pipeline] = {}

    dummy = DummyRegressor(strategy="median")
    dummy.fit(np.zeros((len(train), 1)), y_train)
    dummy_pred = dummy.predict(np.zeros((len(test), 1)))
    metrics_rows.append(metrics("training_median_baseline", y_test, dummy_pred, "test"))
    prediction_output["baseline_prediction_seconds"] = dummy_pred

    modes = {
        "early": (NUMERIC_EARLY, CATEGORICAL_EARLY, NUMERIC_EARLY + CATEGORICAL_EARLY),
        "duration": (
            NUMERIC_EARLY + [DURATION],
            CATEGORICAL_EARLY,
            NUMERIC_EARLY + CATEGORICAL_EARLY + [DURATION],
        ),
    }

    for mode, (numeric, categorical, features) in modes.items():
        selected_name, model, cv_results = select_model(
            train[features], y_train, train_groups, numeric, categorical
        )
        test_pred = model.predict(test[features])
        train_pred = model.predict(train[features])
        model_key = f"{mode}_{selected_name}"
        model_bundle[mode] = model
        cv_results.insert(0, "mode", mode)
        cv_output.append(cv_results)
        metrics_rows.extend(
            [
                metrics(model_key, y_train, train_pred, "train"),
                metrics(model_key, y_test, test_pred, "test"),
            ]
        )
        prediction_output[f"{mode}_prediction_seconds"] = test_pred
        if selected_name == "random_forest":
            lower, upper = forest_interval(model, test[features])
            prediction_output[f"{mode}_interval_10th_seconds"] = lower
            prediction_output[f"{mode}_interval_90th_seconds"] = upper

    metrics_df = pd.DataFrame(metrics_rows)
    cv_df = pd.concat(cv_output, ignore_index=True)
    metrics_df.to_csv(OUTPUT_DIR / "production_model_metrics.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "production_cross_validation.csv", index=False)
    prediction_output.to_csv(OUTPUT_DIR / "production_test_predictions.csv", index=False)

    metadata = {
        "target": TARGET,
        "target_status": "existing synthetic objective target",
        "stage_2_status": "not trained: no independently observed remaining-time target",
        "group_column": GROUP,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_participants": int(train[GROUP].nunique()),
        "test_participants": int(test[GROUP].nunique()),
        "random_state": RANDOM_STATE,
        "feature_modes": {
            "early": NUMERIC_EARLY + CATEGORICAL_EARLY,
            "duration": NUMERIC_EARLY + CATEGORICAL_EARLY + [DURATION],
        },
        "selected_models": {
            mode: str(metrics_df.loc[
                (metrics_df["split"] == "test")
                & metrics_df["model"].str.startswith(f"{mode}_"),
                "model",
            ].iloc[0])
            for mode in modes
        },
    }
    (OUTPUT_DIR / "production_model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {"models": model_bundle, "metadata": metadata},
        OUTPUT_DIR / "production_models.joblib",
    )

    print("COMPLETE HCSD MODEL")
    print(metrics_df[metrics_df["split"] == "test"].round(4).to_string(index=False))
    print(f"\nSaved artifacts in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
