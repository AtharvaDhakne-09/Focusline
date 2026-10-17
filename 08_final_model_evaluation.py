
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


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "HCSD_train_dataset.csv"
TEST_PATH = BASE_DIR / "HCSD_test_dataset.csv"

OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "objective_recovery_time_seconds_target"
GROUP = "participant_id"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

if train_df[TARGET].isna().any():
    raise ValueError("Training target contains missing values.")

if test_df[TARGET].isna().any():
    raise ValueError("Testing target contains missing values.")

# Check that the same participants are not in both sets.
overlap = set(train_df[GROUP]) & set(test_df[GROUP])

print("Training participants:", train_df[GROUP].nunique())
print("Testing participants:", test_df[GROUP].nunique())
print("Participant overlap:", len(overlap))

if overlap:
    raise ValueError("Participant overlap detected!")


# --------------------------------------------------
# 3. DEFINE FEATURES
# --------------------------------------------------

# These features are intended to be available before or
# at interruption onset, except duration which is only
# used in Model B after the interruption has ended.

numeric_early = [
    "sleep_hours",
    "sleep_quality_1_5",
    "pre_fatigue_1_5",
    "pre_stress_1_5",
    "pre_focus_1_5",
    "task_familiarity_1_5",
    "expected_difficulty_1_5",
    "interruption_urgency_1_5",
    "interruption_relevance_1_5",
    "interruption_cognitive_demand_1_5",
    "emotional_disturbance_1_5",
    "baseline_typing_wpm",
    "baseline_error_rate",
    "baseline_pause_seconds",
]

categorical_early = [
    "condition",
    "task_type",
    "environment",
    "deadline_pressure",
    "interruption_type",
    "interruption_expectedness",
]

features_a = numeric_early + categorical_early
features_b = features_a + ["interruption_duration_seconds"]

for feature in features_b + [TARGET, GROUP]:
    if feature not in train_df.columns or feature not in test_df.columns:
        raise ValueError(f"Required column not found: {feature}")


# --------------------------------------------------
# 4. BUILD MODEL
# --------------------------------------------------

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
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", forest),
    ])


# --------------------------------------------------
# 5. TRAIN USING ONLY THE TRAINING DATA
# --------------------------------------------------

X_train_a = train_df[features_a]
X_test_a = test_df[features_a]

X_train_b = train_df[features_b]
X_test_b = test_df[features_b]

y_train = train_df[TARGET]
y_test = test_df[TARGET]

model_a = build_model(numeric_early, categorical_early)

model_b = build_model(
    numeric_early + ["interruption_duration_seconds"],
    categorical_early,
)

print("\nTraining Model A: Early prediction...")
model_a.fit(X_train_a, y_train)

print("Training Model B: Duration available...")
model_b.fit(X_train_b, y_train)


# --------------------------------------------------
# 6. PREDICT ON UNSEEN TEST PARTICIPANTS
# --------------------------------------------------

pred_a = model_a.predict(X_test_a)
pred_b = model_b.predict(X_test_b)

dummy = DummyRegressor(strategy="median")
dummy.fit(np.zeros((len(y_train), 1)), y_train)
pred_dummy = dummy.predict(np.zeros((len(y_test), 1)))


# --------------------------------------------------
# 7. EVALUATE
# --------------------------------------------------

def get_metrics(name, actual, predicted):
    return {
        "model": name,
        "MAE_seconds": mean_absolute_error(actual, predicted),
        "RMSE_seconds": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
    }


results = pd.DataFrame([
    get_metrics("Dummy baseline", y_test, pred_dummy),
    get_metrics("Model A - Early prediction", y_test, pred_a),
    get_metrics("Model B - Duration available", y_test, pred_b),
])

print("\n=== FINAL TEST RESULTS ===")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

results.to_csv(
    OUTPUT_DIR / "final_test_metrics.csv",
    index=False,
)

predictions = pd.DataFrame({
    "participant_id": test_df[GROUP].values,
    "actual_recovery_time_seconds": y_test.values,
    "dummy_prediction_seconds": pred_dummy,
    "early_prediction_seconds": pred_a,
    "duration_prediction_seconds": pred_b,
})

predictions["early_absolute_error_seconds"] = np.abs(
    predictions["actual_recovery_time_seconds"]
    - predictions["early_prediction_seconds"]
)

predictions["duration_absolute_error_seconds"] = np.abs(
    predictions["actual_recovery_time_seconds"]
    - predictions["duration_prediction_seconds"]
)

predictions.to_csv(
    OUTPUT_DIR / "final_test_predictions.csv",
    index=False,
)

print("\nSaved files:")
print(OUTPUT_DIR / "final_test_metrics.csv")
print(OUTPUT_DIR / "final_test_predictions.csv")

print(
    "\nNOTE: The dataset is synthetic. These metrics evaluate "
    "performance on held-out simulated participants, not real people."
)
