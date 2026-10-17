
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "HCSD_train_dataset.csv"
TEST_PATH = BASE_DIR / "HCSD_test_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "objective_recovery_time_seconds_target"
GROUP = "participant_id"

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "task_type",
    "environment",
    "deadline_pressure",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Check required columns
required = FEATURES + [TARGET, GROUP]
for name, data in [("train", train), ("test", test)]:
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in {name}: {missing}")

# Preserve participant-level separation
overlap = set(train[GROUP]) & set(test[GROUP])
if overlap:
    raise ValueError(
        f"Participant leakage detected: {len(overlap)} participants overlap."
    )

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

# Build the same model used in Step 17
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=1.0)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# --------------------------------------------------
# A. Feature-target correlations: training data only
# --------------------------------------------------
numeric_train = train[NUMERIC_FEATURES + [TARGET]].copy()

correlations = (
    numeric_train.corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .sort_values(key=lambda s: s.abs(), ascending=False)
)

correlation_table = correlations.rename(
    "correlation_with_target"
).reset_index()

correlation_table.columns = [
    "feature",
    "correlation_with_target",
]

correlation_table.to_csv(
    OUTPUT_DIR / "stage1_feature_correlations.csv",
    index=False,
)

print("=" * 65)
print("STEP 18: STAGE 1 FEATURE DIAGNOSTICS")
print("=" * 65)

print("\nA. NUMERIC FEATURE-TARGET CORRELATIONS")
print(correlation_table.round(4).to_string(index=False))

# --------------------------------------------------
# B. Overall test-set errors
# --------------------------------------------------
errors = predictions - y_test.to_numpy()
absolute_errors = np.abs(errors)

print("\nB. TEST-SET PERFORMANCE")
print(f"MAE:  {mean_absolute_error(y_test, predictions):.4f} seconds")
print(
    "RMSE: "
    f"{np.sqrt(mean_squared_error(y_test, predictions)):.4f} seconds"
)
print(f"R2:   {r2_score(y_test, predictions):.4f}")
print(f"Median absolute error: {np.median(absolute_errors):.4f} seconds")
print(f"90th percentile absolute error: {np.quantile(absolute_errors, 0.90):.4f} seconds")

# --------------------------------------------------
# C. Save row-level test predictions and errors
# --------------------------------------------------
prediction_table = test[
    [GROUP, "trial_id", TARGET]
].copy()

prediction_table["predicted_target_seconds"] = predictions
prediction_table["signed_error_seconds"] = errors
prediction_table["absolute_error_seconds"] = absolute_errors

prediction_table.to_csv(
    OUTPUT_DIR / "stage1_diagnostic_predictions.csv",
    index=False,
)

# --------------------------------------------------
# D. Error distribution by participant
# --------------------------------------------------
participant_errors = (
    prediction_table.groupby(GROUP)
    .agg(
        test_trials=(TARGET, "size"),
        mean_absolute_error_seconds=("absolute_error_seconds", "mean"),
        mean_signed_error_seconds=("signed_error_seconds", "mean"),
    )
    .reset_index()
    .sort_values("mean_absolute_error_seconds", ascending=False)
)

participant_errors.to_csv(
    OUTPUT_DIR / "stage1_errors_by_participant.csv",
    index=False,
)

print("\nC. PARTICIPANT-LEVEL ERROR SUMMARY")
print(f"Participants evaluated: {len(participant_errors)}")
print(
    "Median participant MAE: "
    f"{participant_errors['mean_absolute_error_seconds'].median():.4f} seconds"
)
print(
    "Lowest participant MAE: "
    f"{participant_errors['mean_absolute_error_seconds'].min():.4f} seconds"
)
print(
    "Highest participant MAE: "
    f"{participant_errors['mean_absolute_error_seconds'].max():.4f} seconds"
)

print("\nFive participants with highest MAE:")
print(participant_errors.head(5).round(3).to_string(index=False))

print("\nD. FILES SAVED")
print(OUTPUT_DIR / "stage1_feature_correlations.csv")
print(OUTPUT_DIR / "stage1_diagnostic_predictions.csv")
print(OUTPUT_DIR / "stage1_errors_by_participant.csv")

print("\nInterpretation note:")
print("Correlations are associations, not causal effects.")
print("Participant-level errors describe this held-out synthetic dataset.")
print("The target-generation formula remains unverified.")
