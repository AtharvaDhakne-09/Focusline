
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_FILE = Path("HCSD_complete_dataset.csv")
OUTPUT_DIR = Path("model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {DATA_FILE}. Check your project folder."
    )

df = pd.read_csv(DATA_FILE)

target = "objective_recovery_time_seconds_target"

print("Dataset loaded:", df.shape)

# --------------------------------------------------
# 2. SELECT INPUT FEATURES
# Use only information available before the interruption
# or known interruption characteristics.
# Do NOT use post-interruption windows.
# --------------------------------------------------

numeric_features = [
    "sleep_hours",
    "sleep_quality_1_5",
    "pre_fatigue_1_5",
    "pre_stress_1_5",
    "pre_focus_1_5",
    "task_familiarity_1_5",
    "expected_difficulty_1_5",
    "interruption_duration_seconds",
    "interruption_urgency_1_5",
    "interruption_relevance_1_5",
    "interruption_cognitive_demand_1_5",
    "emotional_disturbance_1_5",
    "baseline_typing_wpm",
    "baseline_error_rate",
    "baseline_pause_seconds",
]

categorical_features = [
    "condition",
    "task_type",
    "environment",
    "deadline_pressure",
    "interruption_type",
    "interruption_expectedness",
    "responded_to_interruption",
]

features = numeric_features + categorical_features

required_columns = (
    features + [target, "participant_id"]
)

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Keep rows with a known target and participant ID
df = df.dropna(subset=[target, "participant_id"]).copy()

X = df[features]
y = df[target]
groups = df["participant_id"]

# --------------------------------------------------
# 3. SPLIT BY PARTICIPANT
# No participant should appear in both sets.
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

train_people = set(groups.iloc[train_idx])
test_people = set(groups.iloc[test_idx])

assert train_people.isdisjoint(test_people)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training participants:", len(train_people))
print("Testing participants:", len(test_people))
print("Participant overlap:", len(train_people & test_people))

# --------------------------------------------------
# 4. PREPROCESS FEATURES
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

models = {
    "Dummy baseline": Pipeline([
        ("preprocessor", preprocessor),
        ("model", DummyRegressor(strategy="median"))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# --------------------------------------------------
# 6. TRAIN AND EVALUATE
# --------------------------------------------------

results = []
predictions = pd.DataFrame({
    "participant_id": groups.iloc[test_idx].values,
    "actual_simulated_recovery_seconds": y_test.values
})

for name, model in models.items():
    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results.append({
        "model": name,
        "MAE_seconds": mae,
        "RMSE_seconds": rmse,
        "R2": r2
    })

    safe_name = name.lower().replace(" ", "_")
    predictions[f"{safe_name}_prediction_seconds"] = pred

    print(f"MAE:  {mae:.3f} seconds")
    print(f"RMSE: {rmse:.3f} seconds")
    print(f"R2:   {r2:.3f}")

# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "baseline_model_metrics.csv",
    index=False
)

predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv",
    index=False
)

print("\n=== MODEL COMPARISON ===")
print(results_df.round(3).to_string(index=False))

print("\nResults saved in:", OUTPUT_DIR.resolve())
print("Pipeline test completed.")
print(
    "Reminder: the dataset and target are simulated. "
    "These scores are not evidence of human prediction accuracy."
)
