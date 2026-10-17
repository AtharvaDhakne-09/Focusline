
import pandas as pd
from pathlib import Path

df = pd.read_csv("HCSD_complete_dataset.csv")

target = "objective_recovery_time_seconds_target"

print("\n=== RECOVERY TIME BY INTERRUPTION TYPE ===")
summary = (
    df.groupby("interruption_type")[target]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .round(2)
    .sort_values("median")
)
print(summary)

print("\n=== INTERRUPTION DURATION AND RECOVERY TIME ===")
interrupted = df[df["interruption_duration_seconds"] > 0]

correlation = interrupted[
    ["interruption_duration_seconds", target]
].corr().iloc[0, 1]

print(f"Pearson correlation: {correlation:.3f}")

print("\n=== FATIGUE AND RECOVERY TIME ===")
print(
    df.groupby("pre_fatigue_1_5")[target]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\n=== STRESS AND RECOVERY TIME ===")
print(
    df.groupby("pre_stress_1_5")[target]
    .agg(["count", "mean", "median"])
    .round(2)
)

print("\nStatistical summaries completed.")
