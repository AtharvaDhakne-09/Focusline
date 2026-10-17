
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HCSD_complete_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STEP 16: RECOVERY TARGET INVESTIGATION")
print("=" * 70)

# A. Inspect all recovery-related columns
recovery_cols = [
    col for col in df.columns
    if any(term in col.lower() for term in [
        "recovery", "window", "post_pause",
        "post_typing", "post_error", "interruption_duration"
    ])
]

print("\nA. RECOVERY-RELATED COLUMNS")
for col in recovery_cols:
    print(f"\n{col}")
    print("  Data type:", df[col].dtype)
    print("  Non-null:", df[col].notna().sum())
    print("  Unique values:", df[col].nunique(dropna=True))
    print("  Minimum:", df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else "N/A")
    print("  Maximum:", df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else "N/A")

# B. Compare target with recovery indicators
target = "objective_recovery_time_seconds_target"

comparison_cols = [
    col for col in [
        target,
        "recovery_observed_before_60s",
        "interruption_duration_seconds",
        "window1_post_pause_seconds",
        "window2_post_pause_seconds",
        "window3_post_pause_seconds",
        "window4_post_pause_seconds",
        "window5_post_pause_seconds",
        "window6_post_pause_seconds",
    ]
    if col in df.columns
]

print("\n\nB. TARGET AND RECOVERY INDICATORS")
print(df[comparison_cols].head(15).to_string(index=False))

# C. Save an inspectable summary
summary = df[comparison_cols].describe(include="all").transpose()
summary.to_csv(OUTPUT_DIR / "recovery_target_summary.csv")

print("\n\nC. SUMMARY SAVED")
print(OUTPUT_DIR / "recovery_target_summary.csv")

# D. Check whether recovery-observed flag is associated with target
flag = "recovery_observed_before_60s"

if flag in df.columns and target in df.columns:
    print("\nD. TARGET BY RECOVERY-OBSERVED FLAG")
    print(
        df.groupby(flag, dropna=False)[target]
        .agg(["count", "mean", "std", "min", "max"])
        .round(3)
        .to_string()
    )

print("\nInvestigation complete. No target values were changed.")
print("Reminder: this dataset is synthetic.")
