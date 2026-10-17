
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
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

# Only features intended to be known before the interruption.
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

required = FEATURES + [TARGET, GROUP]
for name, data in [("training", train), ("test", test)]:
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in {name} data: {missing}")

# Verify that participants do not appear in both datasets.
overlap = set(train[GROUP].unique()) & set(test[GROUP].unique())
print("=" * 65)
print("STAGE 1: PRE-INTERRUPTION MODEL")
print("=" * 65)
print(f"Training rows: {len(train)}")
print(f"Test rows: {len(test)}")
print(f"Training participants: {train[GROUP].nunique()}")
print(f"Test participants: {test[GROUP].nunique()}")
print(f"Participant overlap: {len(overlap)}")

if overlap:
    raise ValueError(
        "Participant overlap detected. Stop and repair the split before evaluation."
    )

X_train = train[FEATURES]
y_train = train[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

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

baseline = DummyRegressor(strategy="median")
baseline.fit(X_train, y_train)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
baseline_predictions = baseline.predict(X_test)

def evaluate(name, actual, predicted):
    return {
        "model": name,
        "MAE_seconds": mean_absolute_error(actual, predicted),
        "RMSE_seconds": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }

results = pd.DataFrame([
    evaluate("Stage 1 Ridge", y_test, predictions),
    evaluate("Training median baseline", y_test, baseline_predictions),
])

print("\nRESULTS — SYNTHETIC DATA ONLY")
print(results.round(4).to_string(index=False))

results.to_csv(
    OUTPUT_DIR / "stage1_pre_interruption_results.csv",
    index=False,
)

prediction_output = test[[GROUP, "trial_id", TARGET]].copy()
prediction_output["stage1_prediction_seconds"] = predictions
prediction_output["median_baseline_prediction_seconds"] = baseline_predictions
prediction_output.to_csv(
    OUTPUT_DIR / "stage1_pre_interruption_predictions.csv",
    index=False,
)

print("\nSaved:")
print(OUTPUT_DIR / "stage1_pre_interruption_results.csv")
print(OUTPUT_DIR / "stage1_pre_interruption_predictions.csv")

print("\nIMPORTANT:")
print("- This is a provisional experiment on synthetic data.")
print("- The target-generation formula has not been verified.")
print("- Predictions are not validated estimates of real human recovery.")
print("- This model predicts the existing target, not a verified Stage 2 target.")
