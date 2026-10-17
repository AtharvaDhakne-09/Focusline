"""Train application-facing duration and typing-performance models.

Targets are observed columns in the supplied synthetic dataset:
* interruption_duration_seconds
* window1_post_typing_wpm
* window1_post_error_rate

No post-interruption feature is used as an input.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HCSD_complete_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

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
    "interruption_urgency_1_5",
    "interruption_relevance_1_5",
    "interruption_cognitive_demand_1_5",
    "emotional_disturbance_1_5",
]
CATEGORICAL_FEATURES = [
    "task_type",
    "environment",
    "deadline_pressure",
    "interruption_type",
    "interruption_expectedness",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGETS = {
    "duration": "interruption_duration_seconds",
    "typing_wpm": "window1_post_typing_wpm",
    "error_rate": "window1_post_error_rate",
}


def build_model() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + list(TARGETS.values())
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    models = {}
    for name, target in TARGETS.items():
        training = df[FEATURES + [target]].dropna(subset=[target])
        model = build_model()
        model.fit(training[FEATURES], training[target])
        models[name] = model

    bundle = {
        "models": models,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "targets": TARGETS,
        "dataset_note": "Synthetic HCSD data; predictions are not clinically validated.",
    }
    output = OUTPUT_DIR / "application_models.joblib"
    joblib.dump(bundle, output)
    print(f"Saved {output}")
    print("Models:", ", ".join(models))


if __name__ == "__main__":
    main()
