
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HCSD_complete_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

TARGET = "objective_recovery_time_seconds_target"

metrics = [
    "post_typing_wpm",
    "post_error_rate",
    "post_pause_seconds",
    "correction_actions_count",
    "speed_change_percent_vs_baseline",
    "error_rate_change_vs_baseline",
]

print("=" * 70)
print("STEP 20: TARGET VS RECOVERY-WINDOW MEASUREMENTS")
print("=" * 70)

# A. Correlation of target with each window measurement
rows = []

for window in range(1, 7):
    for metric in metrics:
        col = f"window{window}_{metric}"

        if col not in df.columns:
            continue

        rows.append({
            "window": window,
            "metric": metric,
            "correlation_with_target": df[TARGET].corr(df[col]),
        })

correlations = pd.DataFrame(rows)
correlations.to_csv(
    OUTPUT_DIR / "target_window_correlations.csv",
    index=False,
)

print("\nA. CORRELATIONS WITH EXISTING TARGET")

for metric in metrics:
    subset = correlations[correlations["metric"] == metric]
    if not subset.empty:
        print(f"\n{metric}")
        print(subset[["window", "correlation_with_target"]]
              .round(4).to_string(index=False))

# B. Target summary by interruption condition
condition_summary = (
    df.groupby("condition")
    .agg(
        trials=(TARGET, "size"),
        target_mean=(TARGET, "mean"),
        target_std=(TARGET, "std"),
        duration_mean=("interruption_duration_seconds", "mean"),
    )
    .round(3)
)

condition_summary.to_csv(
    OUTPUT_DIR / "target_condition_summary.csv"
)

print("\nB. TARGET BY CONDITION")
print(condition_summary.to_string())

print("\nSaved:")
print(OUTPUT_DIR / "target_window_correlations.csv")
print(OUTPUT_DIR / "target_condition_summary.csv")

print("\nReminder:")
print("These are associations in synthetic data.")
print("Correlation cannot establish the target-generation formula.")
print("Post-interruption measurements remain invalid for a pre-interruption prediction.")
