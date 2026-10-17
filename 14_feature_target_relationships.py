
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HCSD_train_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "objective_recovery_time_seconds_target"

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("FEATURE–TARGET RELATIONSHIP AUDIT")
print("=" * 65)

# 1. Check relationships between numeric features and target.
numeric = df.select_dtypes(include="number")

correlations = (
    numeric.corr()[TARGET]
    .drop(TARGET)
    .sort_values(key=lambda x: x.abs(), ascending=False)
)

correlations_df = correlations.rename(
    "correlation_with_target"
).reset_index()

correlations_df.columns = [
    "feature",
    "correlation_with_target",
]

print("\n1. STRONGEST NUMERIC RELATIONSHIPS")
print(correlations_df.head(20).to_string(index=False))

correlations_df.to_csv(
    OUTPUT_DIR / "feature_target_correlations.csv",
    index=False,
)

# 2. Examine duration and target by condition.
if "interruption_duration_seconds" in df.columns:
    print("\n2. DURATION BY CONDITION")

    duration_summary = (
        df.groupby("condition")
        .agg(
            trials=(TARGET, "size"),
            mean_duration=("interruption_duration_seconds", "mean"),
            mean_target=(TARGET, "mean"),
            duration_std=("interruption_duration_seconds", "std"),
            target_std=(TARGET, "std"),
        )
        .round(3)
    )

    print(duration_summary.to_string())

    duration_summary.to_csv(
        OUTPUT_DIR / "duration_target_by_condition.csv"
    )

# 3. Summarize target variability.
print("\n3. TARGET VARIABILITY")
print("Training target standard deviation:", round(df[TARGET].std(), 3))
print("Unique target values:", df[TARGET].nunique())

# 4. Save a reminder about the prediction timing.
print("\n4. TIMING REVIEW")
print(
    "Before-interruption candidates include sleep, pre-fatigue, "
    "pre-stress, pre-focus, task familiarity, expected difficulty, "
    "and baseline typing measures."
)
print(
    "Duration is only available after the interruption has ended."
)
print(
    "Post-interruption windows and post-interruption self-reports "
    "must not be used for a prediction made before the interruption."
)

print("\nSaved:")
print(OUTPUT_DIR / "feature_target_correlations.csv")
print(OUTPUT_DIR / "duration_target_by_condition.csv")

print(
    "\nReminder: correlation in this synthetic dataset does not "
    "establish causality or real-world predictive validity."
)
