
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HCSD_complete_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "objective_recovery_time_seconds_target"

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("SYNTHETIC DATASET TARGET AUDIT")
print("=" * 65)

# 1. Basic checks
print("\n1. DATASET")
print("Rows:", len(df))
print("Participants:", df["participant_id"].nunique())
print("Target missing values:", df[TARGET].isna().sum())
print("Target minimum:", df[TARGET].min())
print("Target maximum:", df[TARGET].max())
print("Target mean:", round(df[TARGET].mean(), 3))
print("Target standard deviation:", round(df[TARGET].std(), 3))

# 2. Check target distribution by condition
print("\n2. TARGET BY CONDITION")

condition_summary = (
    df.groupby("condition", dropna=False)[TARGET]
    .agg(["count", "mean", "std", "min", "max"])
    .round(3)
)

print(condition_summary.to_string())

condition_summary.to_csv(
    OUTPUT_DIR / "target_audit_by_condition.csv"
)

# 3. Check for repeated or suspiciously fixed target values
print("\n3. TARGET VALUE CHECKS")

print("Unique target values:", df[TARGET].nunique())

print(
    "Rows with target exactly 1.5 seconds:",
    int(np.isclose(df[TARGET], 1.5).sum())
)

print("\nMost frequent target values:")
print(df[TARGET].value_counts().head(10).to_string())

# 4. Check relationship with interruption duration
if "interruption_duration_seconds" in df.columns:
    valid = df[
        [TARGET, "interruption_duration_seconds"]
    ].dropna()

    correlation = valid[TARGET].corr(
        valid["interruption_duration_seconds"]
    )

    print("\n4. DURATION RELATIONSHIP")
    print(
        "Pearson correlation across all rows:",
        round(correlation, 4)
    )

# 5. Save a row-level audit file
audit_columns = [
    "participant_id",
    "condition",
    "interruption_type",
    "interruption_duration_seconds",
    TARGET,
]

audit_columns = [
    col for col in audit_columns if col in df.columns
]

df[audit_columns].to_csv(
    OUTPUT_DIR / "target_audit_rows.csv",
    index=False
)

print("\n5. OUTPUT FILES")
print(OUTPUT_DIR / "target_audit_by_condition.csv")
print(OUTPUT_DIR / "target_audit_rows.csv")

print(
    "\nIMPORTANT: This audit describes the generated dataset. "
    "It does not validate real human recovery behavior."
)
