
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HCSD_train_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "objective_recovery_time_seconds_target"

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DUAL-STAGE TARGET AND TIMING AUDIT")
print("=" * 70)

# 1. Verify the existing target
print("\n1. EXISTING TARGET")
if TARGET in df.columns:
    print(df[TARGET].describe().round(3))
else:
    print(f"ERROR: Target column not found: {TARGET}")

# 2. Identify columns relevant to timing and recovery
keywords = [
    "recovery", "resum", "interrupt", "window",
    "post_", "baseline", "pre_", "pause",
    "target", "timestamp", "start", "end", "duration"
]

timing_columns = [
    col for col in df.columns
    if any(word in col.lower() for word in keywords)
]

print("\n2. TIMING-RELATED COLUMNS")
for col in timing_columns:
    print(f"- {col}")

# 3. Inspect target and interruption-related variables
inspect_columns = [
    col for col in [
        TARGET,
        "condition",
        "interruption_duration_seconds",
        "window1_post_pause_seconds",
        "window2_post_pause_seconds",
        "window1_speed_change_percent_vs_baseline",
        "window1_error_rate_change_vs_baseline",
        "post_focus_1_5",
        "post_stress_1_5",
    ]
    if col in df.columns
]

print("\n3. EXAMPLE ROWS")
print(df[inspect_columns].head(10).to_string(index=False))

# 4. Save the column inventory for review
inventory = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[col].dtype) for col in df.columns],
    "non_null_count": [df[col].notna().sum() for col in df.columns],
    "unique_values": [df[col].nunique(dropna=True) for col in df.columns],
})

inventory.to_csv(
    OUTPUT_DIR / "dual_stage_column_inventory.csv",
    index=False
)

print("\n4. SAVED INVENTORY")
print(OUTPUT_DIR / "dual_stage_column_inventory.csv")

print("\nAUDIT ONLY: No model was trained.")
print("Synthetic data must not be treated as real human observations.")
