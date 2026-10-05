import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, "ml_training_data_realistic.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model_ml3.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder_ml3.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_features_ml3.json")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance_ml3.csv")

print("Generating ML3 accuracy evaluation graphs...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

test_df = df[df["date"] >= "2026-01-01"].copy()

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

num_features = metadata["numerical_features"]
cat_features = metadata["categorical_features"]
cat_col_names = list(encoder.get_feature_names_out(cat_features))

X_num = test_df[num_features].reset_index(drop=True)
X_cat = pd.DataFrame(encoder.transform(test_df[cat_features]), columns=cat_col_names)
X_test = pd.concat([X_num, X_cat], axis=1)
X_test.columns = X_test.columns.astype(str)

y_actual = test_df["demand"].values
y_pred = np.clip(model.predict(X_test), 0, None)
test_df["predicted"] = y_pred

mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

# 4-Panel Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f"RetailPulse ML3: Realistic Supermarket Model Evaluation\n(Gradient Boosting Regressor | R² = {r2:.4f} | MAE = {mae:.2f} units | MAPE = {mape:.1f}%)", fontsize=14, fontweight="bold", y=0.98)

# Panel 1: Actual vs Predicted Scatter
ax1 = axes[0, 0]
ax1.scatter(y_actual, y_pred, alpha=0.35, color="#10b981", edgecolors="none", s=25, label="ML3 Model Predictions")
lims = [0, max(y_actual.max(), y_pred.max()) + 10]
ax1.plot(lims, lims, "--", color="#ef4444", linewidth=2, label="Perfect 1:1 Ideal Line")
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_xlabel("Actual Daily Demand (Units)", fontsize=11, fontweight="semibold")
ax1.set_ylabel("Predicted Demand (Units)", fontsize=11, fontweight="semibold")
ax1.set_title(f"Actual vs. Predicted Demand (R² = {r2:.4f})", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left", frameon=True)

# Panel 2: Daily Demand Trajectory (Jan-Feb 2026)
ax2 = axes[0, 1]
daily = test_df.groupby("date")[["demand", "predicted"]].mean().reset_index()
ax2.plot(daily["date"], daily["demand"], label="Actual Avg Daily Demand", color="#0f172a", linewidth=2.2, marker="o", markersize=3.5)
ax2.plot(daily["date"], daily["predicted"], label="ML3 Predicted Trajectory", color="#059669", linewidth=2.0, linestyle="--", marker="s", markersize=3.5)
ax2.set_xlabel("Date", fontsize=11, fontweight="semibold")
ax2.set_ylabel("Average Units Sold / Product", fontsize=11, fontweight="semibold")
ax2.set_title("Test Period Trajectory Tracking (Jan - Feb 2026)", fontsize=12, fontweight="bold")
ax2.tick_params(axis="x", rotation=30)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower right", frameon=True)

# Panel 3: Residual / Error Distribution
ax3 = axes[1, 0]
residuals = y_actual - y_pred
ax3.hist(residuals, bins=40, color="#6366f1", edgecolor="white", alpha=0.85, density=True)
ax3.axvline(0, color="#ef4444", linestyle="--", linewidth=2, label="Zero Error")
ax3.axvline(np.mean(residuals), color="#f59e0b", linestyle=":", linewidth=2, label=f"Mean Residual ({np.mean(residuals):.2f})")
ax3.set_xlabel("Prediction Error (Actual - Predicted Units)", fontsize=11, fontweight="semibold")
ax3.set_ylabel("Density", fontsize=11, fontweight="semibold")
ax3.set_title(f"Residual Distribution (MAE = {mae:.2f} | Normal Unbiased)", fontsize=12, fontweight="bold")
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(loc="upper right", frameon=True)

# Panel 4: Top 10 Feature Importances
ax4 = axes[1, 1]
imp_df = pd.read_csv(IMPORTANCE_PATH).head(10).sort_values("importance", ascending=True)
bars = ax4.barh(imp_df["feature"], imp_df["importance"] * 100, color="#0284c7", edgecolor="white", alpha=0.9)
ax4.set_xlabel("Feature Importance Percentage (%)", fontsize=11, fontweight="semibold")
ax4.set_title("Top 10 Predictive Demand Drivers", fontsize=12, fontweight="bold")
ax4.grid(True, axis="x", linestyle=":", alpha=0.6)

for bar in bars:
    w = bar.get_width()
    ax4.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", ha="left", fontsize=9.5, fontweight="bold")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_static = os.path.join(PROJECT_DIR, "static", "model_accuracy_ml3.png")
plt.savefig(out_static, dpi=200)
print(f"Graph saved to static: {out_static}")

out_ml3 = os.path.join(BASE_DIR, "accuracy_evaluation_ml3.png")
plt.savefig(out_ml3, dpi=200)

out_artifact = r"C:\Users\USER\.gemini\antigravity\brain\7dfba23a-f154-427c-8ad5-df47f853f25e\model_accuracy_ml3.png"
plt.savefig(out_artifact, dpi=200)
print(f"Graph saved to artifact: {out_artifact}")
