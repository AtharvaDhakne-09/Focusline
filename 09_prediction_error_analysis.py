
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "model_outputs"

PREDICTIONS_PATH = OUTPUT_DIR / "final_test_predictions.csv"
TEST_PATH = BASE_DIR / "HCSD_test_dataset.csv"

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. LOAD FILES
# --------------------------------------------------

pred = pd.read_csv(PREDICTIONS_PATH)
test = pd.read_csv(TEST_PATH)

required_predictions = [
    "participant_id",
    "actual_recovery_time_seconds",
    "early_prediction_seconds",
    "duration_prediction_seconds",
]

required_test = [
    "participant_id",
    "condition",
    "objective_recovery_time_seconds_target",
]

for col in required_predictions:
    if col not in pred.columns:
        raise ValueError(f"Missing prediction column: {col}")

for col in required_test:
    if col not in test.columns:
        raise ValueError(f"Missing test column: {col}")

# The evaluation script saved predictions in the same row order
# as HCSD_test_dataset.csv. Verify this before attaching metadata.
if len(pred) != len(test):
    raise ValueError("Prediction and test dataset row counts differ.")

if not np.array_equal(
    pred["participant_id"].to_numpy(),
    test["participant_id"].to_numpy()
):
    raise ValueError(
        "Participant row order differs. Regenerate predictions "
        "from the current test dataset before continuing."
    )

if not np.allclose(
    pred["actual_recovery_time_seconds"].to_numpy(),
    test["objective_recovery_time_seconds_target"].to_numpy()
):
    raise ValueError(
        "Actual targets do not match the test dataset row by row."
    )

# Attach test metadata only after validating the row order.
analysis = pred.copy()
analysis["condition"] = test["condition"].to_numpy()

# --------------------------------------------------
# 2. CALCULATE ERRORS
# --------------------------------------------------

analysis["early_error_seconds"] = (
    analysis["actual_recovery_time_seconds"]
    - analysis["early_prediction_seconds"]
)

analysis["duration_error_seconds"] = (
    analysis["actual_recovery_time_seconds"]
    - analysis["duration_prediction_seconds"]
)

analysis["early_absolute_error_seconds"] = (
    analysis["early_error_seconds"].abs()
)

analysis["duration_absolute_error_seconds"] = (
    analysis["duration_error_seconds"].abs()
)

analysis["early_squared_error"] = (
    analysis["early_error_seconds"] ** 2
)

analysis["duration_squared_error"] = (
    analysis["duration_error_seconds"] ** 2
)

# --------------------------------------------------
# 3. OVERALL ERROR SUMMARY
# --------------------------------------------------

summary_rows = []

for model_name, error_col, abs_col in [
    ("Model A - Early prediction",
     "early_error_seconds",
     "early_absolute_error_seconds"),
    ("Model B - Duration available",
     "duration_error_seconds",
     "duration_absolute_error_seconds"),
]:
    errors = analysis[error_col]

    summary_rows.append({
        "model": model_name,
        "MAE_seconds": analysis[abs_col].mean(),
        "RMSE_seconds": np.sqrt(
            analysis[f"{'early' if 'early' in error_col else 'duration'}_squared_error"].mean()
        ),
        "mean_signed_error_seconds": errors.mean(),
        "median_absolute_error_seconds": analysis[abs_col].median(),
        "max_absolute_error_seconds": analysis[abs_col].max(),
    })

summary = pd.DataFrame(summary_rows)

print("\n=== OVERALL ERROR SUMMARY ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

summary.to_csv(
    OUTPUT_DIR / "error_analysis_overall.csv",
    index=False,
)

# --------------------------------------------------
# 4. ERROR BY INTERRUPTION CONDITION
# --------------------------------------------------

condition_rows = []

for condition, group in analysis.groupby("condition"):
    for model_name, abs_col, error_col in [
        ("Model A - Early prediction",
         "early_absolute_error_seconds",
         "early_error_seconds"),
        ("Model B - Duration available",
         "duration_absolute_error_seconds",
         "duration_error_seconds"),
    ]:
        condition_rows.append({
            "condition": condition,
            "model": model_name,
            "trials": len(group),
            "MAE_seconds": group[abs_col].mean(),
            "RMSE_seconds": np.sqrt(
                np.mean(group[error_col] ** 2)
            ),
            "mean_signed_error_seconds": group[error_col].mean(),
        })

condition_summary = pd.DataFrame(condition_rows)

print("\n=== ERROR BY CONDITION ===")
print(
    condition_summary.sort_values(
        ["model", "MAE_seconds"],
        ascending=[True, False]
    ).to_string(index=False, float_format=lambda x: f"{x:.3f}")
)

condition_summary.to_csv(
    OUTPUT_DIR / "error_analysis_by_condition.csv",
    index=False,
)

# --------------------------------------------------
# 5. SAVE THE 20 LARGEST MODEL B ERRORS
# --------------------------------------------------

largest_errors = analysis.nlargest(
    20, "duration_absolute_error_seconds"
)

largest_errors.to_csv(
    OUTPUT_DIR / "largest_prediction_errors.csv",
    index=False,
)

print("\n=== 10 LARGEST MODEL B ERRORS ===")
print(
    largest_errors[
        [
            "participant_id",
            "condition",
            "actual_recovery_time_seconds",
            "duration_prediction_seconds",
            "duration_error_seconds",
            "duration_absolute_error_seconds",
        ]
    ].head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)

# --------------------------------------------------
# 6. PLOT ACTUAL VS PREDICTED
# --------------------------------------------------

actual = analysis["actual_recovery_time_seconds"]

min_value = min(
    actual.min(),
    analysis["early_prediction_seconds"].min(),
    analysis["duration_prediction_seconds"].min(),
)

max_value = max(
    actual.max(),
    analysis["early_prediction_seconds"].max(),
    analysis["duration_prediction_seconds"].max(),
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, pred_col, title in [
    (axes[0], "early_prediction_seconds", "Model A - Early prediction"),
    (axes[1], "duration_prediction_seconds", "Model B - Duration available"),
]:
    ax.scatter(actual, analysis[pred_col], alpha=0.65)
    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
    )
    ax.set_title(title)
    ax.set_xlabel("Actual simulated recovery time (seconds)")
    ax.set_ylabel("Predicted recovery time (seconds)")
    ax.grid(True, alpha=0.3)

fig.suptitle("Actual vs Predicted Recovery Time")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "error_actual_vs_predicted.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# --------------------------------------------------
# 7. PLOT MAE BY CONDITION
# --------------------------------------------------

pivot = condition_summary.pivot(
    index="condition",
    columns="model",
    values="MAE_seconds",
)

pivot = pivot.sort_values(
    "Model B - Duration available",
    ascending=True,
)

ax = pivot.plot(kind="barh", figsize=(11, 7))
ax.set_title("Prediction MAE by Interruption Condition")
ax.set_xlabel("Mean absolute error (seconds)")
ax.set_ylabel("Condition")
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "error_mae_by_condition.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

print("\n=== ANALYSIS COMPLETE ===")
print("Saved:")
print(OUTPUT_DIR / "error_analysis_overall.csv")
print(OUTPUT_DIR / "error_analysis_by_condition.csv")
print(OUTPUT_DIR / "largest_prediction_errors.csv")
print(OUTPUT_DIR / "error_actual_vs_predicted.png")
print(OUTPUT_DIR / "error_mae_by_condition.png")

print(
    "\nReminder: all results describe prediction errors on "
    "held-out synthetic data, not real human recovery."
)
