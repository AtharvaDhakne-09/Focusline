
"""
11_compare_condition_baseline.py

Compare:
1. Dummy baseline (training-target median)
2. Condition-average baseline
3. Model A: early-available features
4. Model B: early-available features + interruption duration

IMPORTANT:
The dataset is synthetic. Results are software/pipeline checks,
not evidence of real-world cognitive recovery performance.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "HCSD_train_dataset.csv"
TEST_PATH = BASE_DIR / "HCSD_test_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"

TARGET = "objective_recovery_time_seconds_target"
CONDITION = "condition"
PARTICIPANT = "participant_id"
DURATION = "interruption_duration_seconds"

RANDOM_STATE = 42
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------------

print("=" * 65)
print("CONDITION-AVERAGE BASELINE COMPARISON")
print("=" * 65)

for path in (TRAIN_PATH, TEST_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Check that both CSV files are in the project root."
        )

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

for name, df in (("training", train), ("testing", test)):
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing from {name} data."
        )
    if CONDITION not in df.columns:
        raise ValueError(
            f"Column '{CONDITION}' is missing from {name} data."
        )

if train[TARGET].isna().any() or test[TARGET].isna().any():
    raise ValueError("Target contains missing values.")

y_train = train[TARGET].astype(float)
y_test = test[TARGET].astype(float)

# Check that participants do not overlap between train and test.
if PARTICIPANT in train.columns and PARTICIPANT in test.columns:
    train_people = set(train[PARTICIPANT].dropna())
    test_people = set(test[PARTICIPANT].dropna())
    overlap = train_people.intersection(test_people)

    print(f"\nTraining rows: {len(train)}")
    print(f"Testing rows:  {len(test)}")
    print(f"Training participants: {len(train_people)}")
    print(f"Testing participants:  {len(test_people)}")
    print(f"Participant overlap:   {len(overlap)}")

    if overlap:
        raise ValueError(
            "Participant leakage detected: train and test "
            "contain overlapping participant IDs."
        )
else:
    warnings.warn(
        "Participant ID column not found in both files; "
        "participant separation could not be verified."
    )


# ---------------------------------------------------------
# 3. Identify usable feature columns
# ---------------------------------------------------------

# Exclude target and columns that directly identify a person
# or record. Also exclude obvious post-interruption outcomes
# and derived target/leakage fields from model inputs.
EXCLUDED_EXACT = {
    TARGET,
    PARTICIPANT,
    "participant",
    "participantid",
    "participant_id",
    "trial_id",
    "record_id",
    "row_id",
}

LEAKAGE_TERMS = (
    "recovery_time",
    "recovery_seconds",
    "resumption_lag",
    "post_interruption",
    "post-interruption",
    "after_interruption",
    "after-interruption",
    "target",
    "window",
    "post_",
    "recovery_observed",
    "reread_or_reconstruct",
    "forgot_next_action",
    "felt_slower",
    "felt_more_mistakes",
)

def normalize_name(name):
    return str(name).strip().lower().replace(" ", "_")


def is_excluded_feature(column):
    normalized = normalize_name(column)

    if normalized in EXCLUDED_EXACT:
        return True

    if any(term in normalized for term in LEAKAGE_TERMS):
        return True

    return False


candidate_columns = [
    column
    for column in train.columns
    if column in test.columns and not is_excluded_feature(column)
]

# Model A uses features available before/during the interruption,
# excluding interruption duration.
# Model B adds interruption duration, if present.
early_columns = [
    column for column in candidate_columns
    if column != DURATION
]

if not early_columns:
    raise ValueError(
        "No usable features found. Inspect the training CSV columns."
    )

if CONDITION not in early_columns:
    raise ValueError(
        f"'{CONDITION}' was excluded unexpectedly. "
        "It is required for the condition-average baseline."
    )

model_a_columns = early_columns.copy()
model_b_columns = early_columns.copy()

if DURATION in candidate_columns:
    model_b_columns.append(DURATION)

print("\nModel A feature count:", len(model_a_columns))
print("Model B feature count:", len(model_b_columns))
print("Duration available:", DURATION in candidate_columns)

print("\nModel A features:")
print(model_a_columns)

print("\nModel B features:")
print(model_b_columns)


# ---------------------------------------------------------
# 4. Metrics helper
# ---------------------------------------------------------

def calculate_metrics(model_name, actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "model": model_name,
        "MAE_seconds": mean_absolute_error(actual, predicted),
        "RMSE_seconds": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
        "test_rows": len(actual),
    }


# ---------------------------------------------------------
# 5. Dummy baseline
# ---------------------------------------------------------

dummy = DummyRegressor(strategy="median")
dummy.fit(np.zeros((len(y_train), 1)), y_train)

dummy_predictions = dummy.predict(
    np.zeros((len(y_test), 1))
)


# ---------------------------------------------------------
# 6. Condition-average baseline
# ---------------------------------------------------------

# Calculate each condition's average from TRAINING DATA ONLY.
condition_means = train.groupby(CONDITION)[TARGET].mean()
training_global_mean = float(y_train.mean())

# Conditions not seen in training fall back to the training
# global mean, never to a value calculated from test targets.
condition_predictions = (
    test[CONDITION]
    .map(condition_means)
    .fillna(training_global_mean)
    .to_numpy(dtype=float)
)


# ---------------------------------------------------------
# 7. Build leakage-conscious Random Forest pipelines
# ---------------------------------------------------------

def build_random_forest_pipeline(frame, feature_columns):
    numeric_columns = [
        column for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]

    categorical_columns = [
        column for column in feature_columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


# Model A
model_a = build_random_forest_pipeline(train, model_a_columns)
model_a.fit(train[model_a_columns], y_train)
model_a_predictions = model_a.predict(test[model_a_columns])

# Model B
model_b = build_random_forest_pipeline(train, model_b_columns)
model_b.fit(train[model_b_columns], y_train)
model_b_predictions = model_b.predict(test[model_b_columns])


# ---------------------------------------------------------
# 8. Compare all approaches on the same test observations
# ---------------------------------------------------------

predictions = {
    "Dummy baseline (training median)": dummy_predictions,
    "Condition-average baseline": condition_predictions,
    "Model A (early features)": model_a_predictions,
    "Model B (early features + duration)": model_b_predictions,
}

metrics_rows = [
    calculate_metrics(name, y_test, values)
    for name, values in predictions.items()
]

metrics_df = pd.DataFrame(metrics_rows)
metrics_df = metrics_df.sort_values(
    by="MAE_seconds",
    ascending=True,
).reset_index(drop=True)

# Improvement is measured relative to the condition-average baseline.
condition_mae = metrics_df.loc[
    metrics_df["model"] == "Condition-average baseline",
    "MAE_seconds",
].iloc[0]

metrics_df["MAE_improvement_vs_condition_baseline_percent"] = (
    (condition_mae - metrics_df["MAE_seconds"])
    / condition_mae
    * 100
    if condition_mae > 0
    else np.nan
)


# ---------------------------------------------------------
# 9. Save metrics and individual predictions
# ---------------------------------------------------------

metrics_path = OUTPUT_DIR / "condition_baseline_comparison.csv"
predictions_path = OUTPUT_DIR / "condition_baseline_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)

prediction_df = pd.DataFrame({
    "actual_recovery_time_seconds": y_test.to_numpy(),
    "condition": test[CONDITION].to_numpy(),
    "dummy_prediction": dummy_predictions,
    "condition_average_prediction": condition_predictions,
    "model_a_prediction": model_a_predictions,
    "model_b_prediction": model_b_predictions,
})

if PARTICIPANT in test.columns:
    prediction_df.insert(
        0,
        PARTICIPANT,
        test[PARTICIPANT].to_numpy(),
    )

prediction_df.to_csv(predictions_path, index=False)


# ---------------------------------------------------------
# 10. Print results
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("FINAL COMPARISON — LOWER MAE AND RMSE ARE BETTER")
print("=" * 65)

print(
    metrics_df.to_string(
        index=False,
        formatters={
            "MAE_seconds": "{:.3f}".format,
            "RMSE_seconds": "{:.3f}".format,
            "R2": "{:.3f}".format,
            "MAE_improvement_vs_condition_baseline_percent":
                "{:.2f}%".format,
        },
    )
)

print("\nSaved metrics to:")
print(metrics_path)

print("\nSaved predictions to:")
print(predictions_path)

print("\nIMPORTANT:")
print(
    "The dataset is synthetic. These metrics evaluate the "
    "pipeline on simulated data, not real human performance."
)
print("\nComparison completed.")
