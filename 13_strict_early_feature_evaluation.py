
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "HCSD_train_dataset.csv"
TEST_PATH = BASE_DIR / "HCSD_test_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "objective_recovery_time_seconds_target"
GROUP = "participant_id"
DURATION = "interruption_duration_seconds"
CONDITION = "condition"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD AND VALIDATE DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

for name, df in [("training", train), ("testing", test)]:
    for col in [TARGET, GROUP, CONDITION]:
        if col not in df.columns:
            raise ValueError(f"{col} is missing from {name} data.")

    if df[TARGET].isna().any():
        raise ValueError(f"Missing target values in {name} data.")

overlap = set(train[GROUP]) & set(test[GROUP])

print("=" * 65)
print("STRICT EARLY-FEATURE EVALUATION")
print("=" * 65)
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training participants:", train[GROUP].nunique())
print("Testing participants:", test[GROUP].nunique())
print("Participant overlap:", len(overlap))

if overlap:
    raise ValueError("Training and test participants overlap.")

y_train = train[TARGET].astype(float)
y_test = test[TARGET].astype(float)


# ============================================================
# 3. DEFINE FEATURE SETS
# ============================================================

# Conservative features intended to be available before
# the interruption occurs. Confirm the timing assumptions
# before interpreting this as a real-world early predictor.

numeric_early = [
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
]

categorical_early = [
    "task_type",
    "environment",
    "deadline_pressure",
]

# Model A: before interruption duration is known.
features_a = numeric_early + categorical_early

# Model B: duration is added once the interruption has ended.
features_b = features_a + [DURATION]

# The condition-average baseline uses only training target
# means grouped by condition. It does not fit on test targets.

required_columns = (
    features_b + [TARGET, GROUP, CONDITION]
)

for col in required_columns:
    if col not in train.columns or col not in test.columns:
        raise ValueError(f"Required column missing: {col}")

print("\nModel A features:")
print(features_a)

print("\nModel B features:")
print(features_b)


# ============================================================
# 4. BUILD MODEL PIPELINE
# ============================================================

def build_model(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    forest = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", forest),
    ])


# ============================================================
# 5. TRAIN MODELS USING TRAINING DATA ONLY
# ============================================================

model_a = build_model(numeric_early, categorical_early)
model_b = build_model(
    numeric_early + [DURATION],
    categorical_early,
)

print("\nTraining strict early-feature model...")
model_a.fit(train[features_a], y_train)
pred_a = model_a.predict(test[features_a])

print("Training duration-aware model...")
model_b.fit(train[features_b], y_train)
pred_b = model_b.predict(test[features_b])

# Dummy baseline: training-set median.
dummy = DummyRegressor(strategy="median")
dummy.fit(np.zeros((len(y_train), 1)), y_train)
pred_dummy = dummy.predict(np.zeros((len(y_test), 1)))

# Condition-average baseline: training-set mean per condition.
condition_means = train.groupby(CONDITION)[TARGET].mean()
global_mean = float(y_train.mean())

pred_condition = (
    test[CONDITION]
    .map(condition_means)
    .fillna(global_mean)
    .to_numpy(dtype=float)
)


# ============================================================
# 6. EVALUATE
# ============================================================

def metrics(name, actual, predicted):
    return {
        "model": name,
        "MAE_seconds": mean_absolute_error(actual, predicted),
        "RMSE_seconds": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
        "test_rows": len(actual),
    }


results = pd.DataFrame([
    metrics("Dummy baseline", y_test, pred_dummy),
    metrics("Condition-average baseline", y_test, pred_condition),
    metrics("Strict early-feature model", y_test, pred_a),
    metrics("Duration-aware model", y_test, pred_b),
])

condition_mae = results.loc[
    results["model"] == "Condition-average baseline",
    "MAE_seconds",
].iloc[0]

results["MAE_improvement_vs_condition_baseline_percent"] = (
    (condition_mae - results["MAE_seconds"]) / condition_mae * 100
)

results = results.sort_values(
    "MAE_seconds"
).reset_index(drop=True)

print("\n" + "=" * 65)
print("TEST RESULTS — LOWER MAE AND RMSE ARE BETTER")
print("=" * 65)
print(results.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}",
))


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_path = OUTPUT_DIR / "strict_early_feature_metrics.csv"
predictions_path = OUTPUT_DIR / "strict_early_feature_predictions.csv"

results.to_csv(results_path, index=False)

predictions = pd.DataFrame({
    GROUP: test[GROUP].to_numpy(),
    CONDITION: test[CONDITION].to_numpy(),
    "actual_recovery_time_seconds": y_test.to_numpy(),
    "dummy_prediction": pred_dummy,
    "condition_average_prediction": pred_condition,
    "strict_early_prediction": pred_a,
    "duration_aware_prediction": pred_b,
})

predictions.to_csv(predictions_path, index=False)

print("\nSaved metrics:", results_path)
print("Saved predictions:", predictions_path)

print(
    "\nLIMITATION: The dataset and target are synthetic. "
    "These results do not validate real human cognitive recovery."
)
print(
    "The early-feature timing assumptions must also be verified "
    "before any real-world use."
)
