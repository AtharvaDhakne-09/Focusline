
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load the same CSV used in the audit
file_path = Path("HCSD_complete_dataset.csv")
df = pd.read_csv(file_path)

target = "objective_recovery_time_seconds_target"

# Check the essential columns
required = [
    "interruption_type",
    "interruption_duration_seconds",
    target
]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

sns.set_theme(style="whitegrid")

# Plot 1: Distribution of recovery time
plt.figure(figsize=(9, 5))
sns.histplot(df[target], bins=20, kde=True)
plt.title("Distribution of Simulated Recovery Time")
plt.xlabel("Recovery time (seconds)")
plt.ylabel("Number of trials")
plt.tight_layout()
plt.savefig("eda_01_recovery_distribution.png", dpi=200)
plt.show()

# Plot 2: Recovery time by interruption type
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df,
    x="interruption_type",
    y=target
)
plt.title("Simulated Recovery Time by Interruption Type")
plt.xlabel("Interruption type")
plt.ylabel("Recovery time (seconds)")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig("eda_02_recovery_by_interruption.png", dpi=200)
plt.show()

# Plot 3: Interruption duration vs recovery time
interrupted = df[df["interruption_duration_seconds"] > 0]

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=interrupted,
    x="interruption_duration_seconds",
    y=target,
    hue="interruption_type",
    alpha=0.7
)
plt.title("Interruption Duration vs Simulated Recovery Time")
plt.xlabel("Interruption duration (seconds)")
plt.ylabel("Recovery time (seconds)")
plt.legend(
    title="Interruption type",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()
plt.savefig("eda_03_duration_vs_recovery.png", dpi=200)
plt.show()

# Descriptive summary
print("\nRecovery time by interruption type:")
print(
    df.groupby("interruption_type")[target]
    .agg(["count", "mean", "median", "std"])
    .round(2)
)

print("\nEDA completed. Three plots have been saved.")
