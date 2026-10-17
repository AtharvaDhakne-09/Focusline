
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HCSD_complete_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"

TARGET = "objective_recovery_time_seconds_target"
DURATION = "interruption_duration_seconds"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("TARGET GENERATION INVESTIGATION")
print("=" * 70)

print("\n1. DATASET SHAPE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. TARGET DISTRIBUTION")
print(df[TARGET].describe().round(3))

print("\n3. TARGET BY CONDITION")
print(
    df.groupby("condition")[TARGET]
    .agg(["count", "mean", "std", "min", "max"])
    .round(3)
    .to_string()
)

print("\n4. TARGET BY INTERRUPTION DURATION")
if DURATION in df.columns:
    print(
        df.groupby(DURATION)[TARGET]
        .agg(["count", "mean", "std", "min", "max"])
        .round(3)
        .head(30)
        .to_string()
    )

    print(
        "\nOverall duration-target correlation:",
        round(df[TARGET].corr(df[DURATION]), 4),
    )

    print("\nCorrelation within each condition:")
    print(
        df.groupby("condition")
        .apply(
            lambda group: group[TARGET].corr(group[DURATION])
            if group[DURATION].nunique() > 1
            else np.nan,
            include_groups=False,
        )
        .round(4)
        .to_string()
    )

print("\n5. TARGET RELATIONSHIPS WITH NUMERIC COLUMNS")
numeric = df.select_dtypes(include="number")
correlations = numeric.corr()[TARGET].drop(TARGET)
correlations = correlations.reindex(
    correlations.abs().sort_values(ascending=False).index
)

print(correlations.head(20).round(4).to_string())

print("\n6. IDENTICAL TARGETS WITHIN PARTICIPANTS")
if "participant_id" in df.columns:
    participant_stats = df.groupby("participant_id")[TARGET].nunique()
    print(
        "Participants with one unique target across all trials:",
        int((participant_stats == 1).sum()),
    )
    print("Total participants:", len(participant_stats))

print("\n7. FIXED TARGET CHECK")
fixed_15 = np.isclose(df[TARGET], 1.5)
print("Rows with target exactly 1.5:", int(fixed_15.sum()))

if "condition" in df.columns:
    print("\nConditions associated with the 1.5-second target:")
    print(
        df.loc[fixed_15, "condition"]
        .value_counts()
        .to_string()
    )

print("\n8. INTERPRETATION")
print(
    "This script identifies statistical patterns and possible "
    "relationships. It cannot reconstruct the original target "
    "generation formula with certainty."
)
print(
    "Because this dataset is synthetic, observed correlations "
    "must not be interpreted as evidence of real human behavior."
)

print("\nInvestigation completed.")
