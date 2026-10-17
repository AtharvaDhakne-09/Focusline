
import pandas as pd
from pathlib import Path

# Find the Excel file in the current project folder
file_path = Path("HCSD_complete_dataset.csv")

if not file_path.exists():
    raise FileNotFoundError(
        "Excel file not found. Check the filename and folder."
    )

# Load the complete dataset
df = pd.read_csv(file_path)

print("\n========== 1. DATASET SIZE ==========")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n========== 2. COLUMN NAMES ==========")
for column in df.columns:
    print(column)

print("\n========== 3. DATA TYPES ==========")
print(df.dtypes.to_string())

print("\n========== 4. MISSING VALUES ==========")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))

if missing.sum() == 0:
    print("No missing values found.")

print("\n========== 5. DUPLICATE TRIALS ==========")
if "trial_id" in df.columns:
    print("Duplicate trial IDs:", df["trial_id"].duplicated().sum())
else:
    print("trial_id column is missing.")

print("\n========== 6. INTERRUPTION TYPES ==========")
if "interruption_type" in df.columns:
    print(df["interruption_type"].value_counts().to_string())

print("\n========== 7. RECOVERY TIME ==========")
target = "objective_recovery_time_seconds_target"

if target in df.columns:
    print(df[target].describe())
else:
    print(f"Target column not found: {target}")

print("\n========== 8. FIRST FIVE ROWS ==========")
print(df.head().to_string())

print("\n========== AUDIT FINISHED ==========")
