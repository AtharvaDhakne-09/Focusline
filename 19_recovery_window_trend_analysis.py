
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "HCSD_complete_dataset.csv"
OUTPUT_DIR = BASE_DIR / "model_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

metrics = [
    "post_typing_wpm",
    "post_error_rate",
    "post_pause_seconds",
    "correction_actions_count",
    "speed_change_percent_vs_baseline",
    "error_rate_change_vs_baseline",
]

rows = []

print("=" * 70)
print("STEP 19: RECOVERY-WINDOW TREND ANALYSIS")
print("=" * 70)

for window in range(1, 7):
    for metric in metrics:
        col = f"window{window}_{metric}"

        if col not in df.columns:
            continue

        rows.append({
            "window": window,
            "metric": metric,
            "mean": df[col].mean(),
            "std": df[col].std(),
            "median": df[col].median(),
            "min": df[col].min(),
            "max": df[col].max(),
        })

summary = pd.DataFrame(rows)
summary.to_csv(
    OUTPUT_DIR / "recovery_window_trends.csv",
    index=False,
)

for metric in metrics:
    subset = summary[summary["metric"] == metric]

    if subset.empty:
        continue

    print(f"\n{metric}")
    print(
        subset[["window", "mean", "std", "median"]]
        .round(4)
        .to_string(index=False)
    )

print("\nSaved:")
print(OUTPUT_DIR / "recovery_window_trends.csv")

print("\nInterpretation:")
print("These are descriptive trends in synthetic measurements.")
print("A trend across windows does not establish an exact resumption time.")

