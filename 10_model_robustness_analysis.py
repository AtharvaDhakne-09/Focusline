
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "model_outputs"

PREDICTIONS_PATH = OUTPUT_DIR / "final_test_predictions.csv"
TEST_PATH = BASE_DIR / "HCSD_test_dataset.csv"

TARGET = "objective_recovery_time_seconds_target"

# --------------------------------------------------
# 1. LOAD AND VALIDATE DATA
# --------------------------------------------------

pred = pd.read_csv(PREDICTIONS_PATH)
test = pd.read_csv(TEST_PATH)

if len(pred) != len(test):
    raise ValueError("Prediction and test row counts do not match.")

if not np.allclose(
    pred["actual_recovery_time_seconds"].to_numpy(),
    test[TARGET].to_numpy(),
):
    raise ValueError(
        "Targets do not match. Regenerate predictions before analysis."
    )

if not np.array_equal(
    pred["participant_id"].to_numpy(),
    test["participant_id"].to_numpy(),
):
    raise ValueError(
        "Participant rows do not align with the test dataset."
    )

df = pred.copy()
df["condition"] = test["condition"].to_numpy()

# --------------------------------------------------
# 2. CALCULATE RESIDUALS
# --------------------------------------------------

for label, column in [
    ("early", "early_prediction_seconds"),
    ("duration", "duration_prediction_seconds"),
]:
    df[f"{label}_residual"] = (
        df["actual_recovery_time_seconds"] - df[column]
    )
    df[f"{label}_absolute_error"] = df[f"{label}_residual"].abs()

# Positive residual = underprediction.
# Negative residual = overprediction.

# --------------------------------------------------
# 3. GROUP BY ACTUAL RECOVERY-TIME RANGE
# --------------------------------------------------

bins = [0, 5, 10, 15, 20, 30, np.inf]
labels = ["0–5", "5–10", "10–15", "15–20", "20–30", "30+"]

df["actual_recovery_range"] = pd.cut(
    df["actual_recovery_time_seconds"],
    bins=bins,
    labels=labels,
    right=False,
    include_lowest=True,
)

summary_rows = []

for recovery_range, group in df.groupby(
    "actual_recovery_range", observed=False
):
    for model, residual_col, abs_col in [
        (
            "Model A - Early prediction",
            "early_residual",
            "early_absolute_error",
        ),
        (
            "Model B - Duration available",
            "duration_residual",
            "duration_absolute_error",
        ),
    ]:
        errors = group[residual_col].dropna()
        absolute_errors = group[abs_col].dropna()

        if len(errors) == 0:
            continue

        summary_rows.append({
            "actual_recovery_range_seconds": str(recovery_range),
            "model": model,
            "trials": len(group),
            "MAE_seconds": absolute_errors.mean(),
            "RMSE_seconds": np.sqrt(np.mean(errors ** 2)),
            "mean_signed_error_seconds": errors.mean(),
            "underprediction_count": int((errors > 0).sum()),
            "overprediction_count": int((errors < 0).sum()),
        })

range_summary = pd.DataFrame(summary_rows)

print("\n=== PERFORMANCE BY ACTUAL RECOVERY-TIME RANGE ===")
print(
    range_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

range_summary.to_csv(
    OUTPUT_DIR / "robustness_by_recovery_range.csv",
    index=False,
)

# --------------------------------------------------
# 4. PLOT ERROR BY RECOVERY-TIME RANGE
# --------------------------------------------------

mae_pivot = range_summary.pivot(
    index="actual_recovery_range_seconds",
    columns="model",
    values="MAE_seconds",
)

mae_pivot = mae_pivot.reindex(labels)

ax = mae_pivot.plot(kind="bar", figsize=(11, 6))
ax.set_title("Prediction Error by Actual Recovery-Time Range")
ax.set_xlabel("Actual simulated recovery-time range (seconds)")
ax.set_ylabel("Mean absolute error (seconds)")
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "robustness_error_by_range.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# --------------------------------------------------
# 5. PLOT RESIDUALS AGAINST ACTUAL TARGET
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

actual = df["actual_recovery_time_seconds"]

plots = [
    ("early_residual", "Model A - Early prediction"),
    ("duration_residual", "Model B - Duration available"),
]

for ax, (residual_col, title) in zip(axes, plots):
    ax.scatter(
        actual,
        df[residual_col],
        alpha=0.65,
    )
    ax.axhline(0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Actual simulated recovery time (seconds)")
    ax.set_ylabel("Residual: actual minus predicted (seconds)")
    ax.grid(True, alpha=0.3)

fig.suptitle("Residuals vs Actual Recovery Time")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "robustness_residuals_vs_actual.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# --------------------------------------------------
# 6. CONDITION-LEVEL MODEL COMPARISON
# --------------------------------------------------

condition_summary = (
    df.groupby("condition")
    .agg(
        trials=("participant_id", "size"),
        model_a_mae=("early_absolute_error", "mean"),
        model_b_mae=("duration_absolute_error", "mean"),
    )
    .reset_index()
)

condition_summary["MAE_change_B_minus_A"] = (
    condition_summary["model_b_mae"]
    - condition_summary["model_a_mae"]
)

print("\n=== CONDITION-LEVEL COMPARISON ===")
print(
    condition_summary.sort_values(
        "MAE_change_B_minus_A"
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)

condition_summary.to_csv(
    OUTPUT_DIR / "robustness_by_condition.csv",
    index=False,
)

# --------------------------------------------------
# 7. SAVE COMPLETED ANALYSIS
# --------------------------------------------------

print("\n=== SAVED FILES ===")
print(OUTPUT_DIR / "robustness_by_recovery_range.csv")
print(OUTPUT_DIR / "robustness_error_by_range.png")
print(OUTPUT_DIR / "robustness_residuals_vs_actual.png")
print(OUTPUT_DIR / "robustness_by_condition.csv")

print(
    "\nIMPORTANT: This analysis uses synthetic targets. "
    "It evaluates the model pipeline, not real human recovery."
)
