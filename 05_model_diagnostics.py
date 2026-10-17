
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data
df = pd.read_csv("HCSD_complete_dataset.csv")
output_dir = Path("model_outputs")
output_dir.mkdir(exist_ok=True)

target = "objective_recovery_time_seconds_target"

numeric_features = [
    "sleep_hours",
    "sleep_quality_1_5",
    "pre_fatigue_1_5",
    "pre_stress_1_5",
    "pre_focus_1_5",
    "task_familiarity_1_5",
    "expected_difficulty_1_5",
    "interruption_duration_seconds",
    "interruption_urgency_1_5",
    "interruption_relevance_1_5",
    "interruption_cognitive_demand_1_5",
    "emotional_disturbance_1_5",
    "baseline_typing_wpm",
    "baseline_error_rate",
    "baseline_pause_seconds",
]

categorical_features = [
    "condition",
    "task_type",
    "environment",
    "deadline_pressure",
    "interruption_type",
    "interruption_expectedness",
    "responded_to_interruption",
]

features = numeric_features + categorical_features

# 2. Prepare features and target
df = df.dropna(subset=[target, "participant_id"]).copy()
X = df[features]
y = df[target]
groups = df["participant_id"]

# 3. Reproduce participant-aware split from Step 3
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Training participants:", groups.iloc[train_idx].nunique())
print("Testing participants:", groups.iloc[test_idx].nunique())
print(
    "Participant overlap:",
    len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))
)

# 4. Preprocessing and model
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)
residuals = y_test.to_numpy() - pred

print("\n=== TEST METRICS ===")
print(f"MAE:  {mean_absolute_error(y_test, pred):.3f} seconds")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.3f} seconds")
print(f"R2:   {r2_score(y_test, pred):.3f}")

# 5. Save predictions and residuals
diagnostics = pd.DataFrame({
    "participant_id": groups.iloc[test_idx].to_numpy(),
    "actual_recovery_seconds": y_test.to_numpy(),
    "predicted_recovery_seconds": pred,
    "residual_seconds": residuals,
    "absolute_error_seconds": np.abs(residuals)
})
diagnostics.to_csv(
    output_dir / "diagnostic_predictions.csv",
    index=False
)

sns.set_theme(style="whitegrid")

# Plot A: Actual versus predicted
plt.figure(figsize=(7, 6))
sns.scatterplot(
    x=y_test,
    y=pred,
    alpha=0.7
)
low = min(y_test.min(), pred.min())
high = max(y_test.max(), pred.max())
plt.plot([low, high], [low, high], linestyle="--")
plt.xlabel("Actual simulated recovery time (seconds)")
plt.ylabel("Predicted recovery time (seconds)")
plt.title("Actual vs Predicted Recovery Time")
plt.tight_layout()
plt.savefig(output_dir / "diagnostic_actual_vs_predicted.png", dpi=200)
plt.show()

# Plot B: Residual distribution
plt.figure(figsize=(7, 5))
sns.histplot(residuals, bins=20, kde=True)
plt.axvline(0, linestyle="--")
plt.xlabel("Residual: actual minus predicted (seconds)")
plt.title("Prediction Residual Distribution")
plt.tight_layout()
plt.savefig(output_dir / "diagnostic_residuals.png", dpi=200)
plt.show()

# Plot C: Residuals against predictions
plt.figure(figsize=(7, 5))
sns.scatterplot(x=pred, y=residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted recovery time (seconds)")
plt.ylabel("Residual (seconds)")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig(output_dir / "diagnostic_residuals_vs_predicted.png", dpi=200)
plt.show()

# 6. Permutation importance on held-out test data
importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": features,
    "importance_mean": importance.importances_mean,
    "importance_std": importance.importances_std
}).sort_values("importance_mean", ascending=False)

importance_df.to_csv(
    output_dir / "permutation_feature_importance.csv",
    index=False
)

print("\n=== PERMUTATION FEATURE IMPORTANCE ===")
print(importance_df.head(10).round(4).to_string(index=False))

plt.figure(figsize=(9, 6))
top = importance_df.head(10).sort_values("importance_mean")
plt.barh(
    top["feature"],
    top["importance_mean"],
    xerr=top["importance_std"]
)
plt.xlabel("Decrease in model score when feature is shuffled")
plt.title("Top 10 Permutation Importances")
plt.tight_layout()
plt.savefig(output_dir / "diagnostic_feature_importance.png", dpi=200)
plt.show()

print("\nDiagnostics completed.")
print("Files saved in:", output_dir.resolve())
print(
    "All findings here describe simulated data, not real human observations."
)
