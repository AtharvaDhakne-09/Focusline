
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HCSD_complete_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "objective_recovery_time_seconds_target"
GROUP = "participant_id"

RANDOM_STATE = 42


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

required_columns = [
    TARGET,
    GROUP,
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
    "condition",
    "task_type",
    "environment",
    "deadline_pressure",
    "interruption_type",
    "interruption_expectedness",
    "interruption_duration_seconds",
    "interruption_urgency_1_5",
    "interruption_relevance_1_5",
    "interruption_cognitive_demand_1_5",
    "emotional_disturbance_1_5",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET, GROUP]).reset_index(drop=True)

print("Dataset shape:", df.shape)
print("Unique participants:", df[GROUP].nunique())
print("Target:", TARGET)


# --------------------------------------------------
# 3. DEFINE FEATURES AVAILABLE AT PREDICTION TIME
# --------------------------------------------------

# Features available before or at interruption onset.
# Exclude duration, responses made after onset, and all
# post-interruption window measurements.

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
    "interruption_urgency_1_5",
    "interruption_relevance_1_5",
    "interruption_cognitive_demand_1_5",
    "emotional_disturbance_1_5",
]

categorical_early = [
    "condition",
    "task_type",
    "environment",
    "deadline_pressure",
    "interruption_type",
    "interruption_expectedness",
]

# Model A: early prediction.
features_a = numeric_early + categorical_early

# Model B: prediction after interruption ends.
# Adds completed duration, which is now known.
features_b = features_a + ["interruption_duration_seconds"]


# --------------------------------------------------
# 4. CREATE ONE PARTICIPANT-LEVEL TRAIN/TEST SPLIT
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(
    splitter.split(df, groups=df[GROUP])
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

train_participants = set(train_df[GROUP])
test_participants = set(test_df[GROUP])
overlap = train_participants.intersection(test_participants)

print("\n=== PARTICIPANT SPLIT ===")
print("Training participants:", len(train_participants))
print("Testing participants:", len(test_participants))
print("Participant overlap:", len(overlap))

if overlap:
    raise RuntimeError("Participant leakage detected!")


# --------------------------------------------------
# 5. BUILD THE MODEL PIPELINE
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

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


# --------------------------------------------------
# 6. EVALUATE EACH MODEL ON THE SAME TEST PARTICIPANTS
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

model_a.fit(X_train_a, y_train)
model_b.fit(X_train_b, y_train)

pred_a = model_a.predict(X_test_a)
pred_b = model_b.predict(X_test_b)

# Simple baseline: predict the training-set median.
dummy = DummyRegressor(strategy="median")
dummy.fit(np.zeros((len(y_train), 1)), y_train)
pred_dummy = dummy.predict(np.zeros((len(y_test), 1)))


# --------------------------------------------------
# 7. CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(name, actual, predicted):
    return {
        "model": name,
        "MAE_seconds": mean_absolute_error(actual, predicted),
        "RMSE_seconds": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2": r2_score(actual, predicted),
    }


results = pd.DataFrame([
    calculate_metrics("Dummy baseline", y_test, pred_dummy),
    calculate_metrics("Model A - Early prediction", y_test, pred_a),
    calculate_metrics("Model B - Duration available", y_test, pred_b),
])

print("\n=== MODEL COMPARISON ===")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

mae_a = mean_absolute_error(y_test, pred_a)
mae_b = mean_absolute_error(y_test, pred_b)

if mae_a > 0:
    improvement = (mae_a - mae_b) / mae_a * 100
else:
    improvement = np.nan

print("\n=== DURATION COMPARISON ===")
print(f"Model A MAE: {mae_a:.4f} seconds")
print(f"Model B MAE: {mae_b:.4f} seconds")
print(f"Relative MAE reduction from adding duration: {improvement:.2f}%")

if mae_b < mae_a:
    print("Adding duration reduced the test MAE.")
elif mae_b > mae_a:
    print("Adding duration increased the test MAE on this split.")
else:
    print("Both models have equal MAE on this split.")


# --------------------------------------------------
# 8. SAVE RESULTS AND PREDICTIONS
# --------------------------------------------------

results_path = OUTPUT_DIR / "leakage_aware_model_comparison.csv"
predictions_path = OUTPUT_DIR / "leakage_aware_predictions.csv"

results.to_csv(results_path, index=False)

predictions = pd.DataFrame({
    "participant_id": test_df[GROUP].values,
    "actual_target_seconds": y_test.values,
    "dummy_prediction_seconds": pred_dummy,
    "model_a_early_prediction_seconds": pred_a,
    "model_b_duration_prediction_seconds": pred_b,
})

predictions["model_a_residual_seconds"] = (
    predictions["actual_target_seconds"]
    - predictions["model_a_early_prediction_seconds"]
)

predictions["model_b_residual_seconds"] = (
    predictions["actual_target_seconds"]
    - predictions["model_b_duration_prediction_seconds"]
)

predictions.to_csv(predictions_path, index=False)

print("\n=== FILES SAVED ===")
print(results_path)
print(predictions_path)

print(
    "\nIMPORTANT: These results use synthetic data. "
    "They are pipeline checks, not empirical evidence "
    "about human cognitive recovery."
)
