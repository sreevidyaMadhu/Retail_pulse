import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ml_training_data_2year.csv")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
test_df = df[df["date"] >= "2026-01-01"].copy()

# 1. Load Original Random Forest
rf_model = joblib.load(os.path.join(BASE_DIR, "random_forest_model_2year.pkl"))
rf_encoder = joblib.load(os.path.join(BASE_DIR, "encoder_2year.pkl"))
with open(os.path.join(BASE_DIR, "model_features_2year.json")) as f:
    rf_meta = json.load(f)

rf_cat = rf_encoder.transform(test_df[rf_meta["categorical_features"]])
rf_cat_df = pd.DataFrame(rf_cat, columns=rf_encoder.get_feature_names_out(rf_meta["categorical_features"]), index=test_df.index)
X_rf = pd.concat([test_df[rf_meta["numerical_features"]], rf_cat_df], axis=1)
y_rf_pred = np.clip(rf_model.predict(X_rf), 0, None)

# 2. Load New Gradient Boosting with Price + Lag-7
gb_model = joblib.load(os.path.join(BASE_DIR, "gradient_boosting_model_2year.pkl"))
gb_encoder = joblib.load(os.path.join(BASE_DIR, "gradient_boosting_encoder_2year.pkl"))
with open(os.path.join(BASE_DIR, "gradient_boosting_features_2year.json")) as f:
    gb_meta = json.load(f)

gb_cat = gb_encoder.transform(test_df[gb_meta["categorical_features"]])
gb_cat_df = pd.DataFrame(gb_cat, columns=gb_encoder.get_feature_names_out(gb_meta["categorical_features"]), index=test_df.index)
X_gb = pd.concat([test_df[gb_meta["numerical_features"]], gb_cat_df], axis=1)
y_gb_pred = np.clip(gb_model.predict(X_gb), 0, None)

# 3. Baseline & Actuals
y_base = test_df["rolling_7day_sales"].values
y_actual = test_df["demand"].values

metrics = {
    "Baseline": {
        "MAE": float(mean_absolute_error(y_actual, y_base)),
        "RMSE": float(np.sqrt(mean_squared_error(y_actual, y_base))),
        "R2": float(r2_score(y_actual, y_base)),
        "Size_MB": 0.0
    },
    "Random Forest (Original)": {
        "MAE": float(mean_absolute_error(y_actual, y_rf_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_actual, y_rf_pred))),
        "R2": float(r2_score(y_actual, y_rf_pred)),
        "Size_MB": 102.9
    },
    "Gradient Boosting (New)": {
        "MAE": float(mean_absolute_error(y_actual, y_gb_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_actual, y_gb_pred))),
        "R2": float(r2_score(y_actual, y_gb_pred)),
        "Size_MB": 0.38
    }
}

print("Calculated comparison metrics:")
print(json.dumps(metrics, indent=2))

# Plot Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("RetailPulse: Model Upgrade Performance & Accuracy Comparison\n(Random Forest vs. Gradient Boosting with Pricing & Lag-7 Features)", fontsize=15, fontweight="bold", y=0.98)

# Panel 1: Metric Comparison Bar Chart
ax1 = axes[0, 0]
models = ["7-Day Baseline", "Random Forest\n(Original)", "Gradient Boosting\n(+Price & Lag-7)"]
mae_vals = [metrics["Baseline"]["MAE"], metrics["Random Forest (Original)"]["MAE"], metrics["Gradient Boosting (New)"]["MAE"]]
r2_vals = [metrics["Baseline"]["R2"], metrics["Random Forest (Original)"]["R2"], metrics["Gradient Boosting (New)"]["R2"]]

x = np.arange(len(models))
width = 0.35

rects1 = ax1.bar(x - width/2, mae_vals, width, label="MAE (Lower is Better)", color=["#94a3b8", "#3b82f6", "#10b981"])
ax1_twin = ax1.twinx()
rects2 = ax1_twin.bar(x + width/2, [r * 100 for r in r2_vals], width, label="R2 % (Higher is Better)", color=["#cbd5e1", "#60a5fa", "#34d399"], hatch="//")

ax1.set_ylabel("Mean Absolute Error (Units)", fontweight="bold")
ax1_twin.set_ylabel("R2 Score (% Variance Explained)", fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontweight="bold")
ax1.set_title("Overall Accuracy Gains on 2026 Test Set", fontsize=12, fontweight="bold")
ax1.set_ylim([0, 7.5])
ax1_twin.set_ylim([0, 35])
ax1.grid(True, linestyle=":", alpha=0.5)

# Value labels on bars
for bar in rects1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.1, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
for bar in rects2:
    h = bar.get_height()
    ax1_twin.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

# Panel 2: Actual vs Predicted Scatter (Gradient Boosting)
ax2 = axes[0, 1]
ax2.scatter(y_actual, y_gb_pred, alpha=0.35, color="#10b981", edgecolors="none", s=25, label="Gradient Boosting Predictions")
lims = [0, max(y_actual.max(), y_gb_pred.max()) + 5]
ax2.plot(lims, lims, "--", color="#ef4444", linewidth=2, label="Ideal Perfect Line (1:1)")
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_xlabel("Actual Demand (Units)", fontsize=11, fontweight="semibold")
ax2.set_ylabel("Predicted Demand (Units)", fontsize=11, fontweight="semibold")
ax2.set_title(f"New Model: Actual vs Predicted (R2 = {metrics['Gradient Boosting (New)']['R2']:.4f})", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left", frameon=True)

# Panel 3: Residual Comparison Density
ax3 = axes[1, 0]
rf_res = y_actual - y_rf_pred
gb_res = y_actual - y_gb_pred
ax3.hist(rf_res, bins=35, color="#3b82f6", alpha=0.45, density=True, label=f"Random Forest (MAE = {metrics['Random Forest (Original)']['MAE']:.2f})")
ax3.hist(gb_res, bins=35, color="#10b981", alpha=0.55, density=True, label=f"Gradient Boosting (MAE = {metrics['Gradient Boosting (New)']['MAE']:.2f})")
ax3.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5, label="Zero Error")
ax3.set_xlabel("Prediction Error (Actual - Predicted)", fontsize=11, fontweight="semibold")
ax3.set_ylabel("Probability Density", fontsize=11, fontweight="semibold")
ax3.set_title("Residual Distribution Comparison", fontsize=12, fontweight="bold")
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(loc="upper right", frameon=True)

# Panel 4: Model Footprint & Latency Efficiency
ax4 = axes[1, 1]
models_comp = ["Random Forest\n(Old)", "Gradient Boosting\n(New)"]
sizes = [102.9, 0.38]
colors = ["#f87171", "#34d399"]
bars_size = ax4.bar(models_comp, sizes, color=colors, width=0.45, edgecolor="black", alpha=0.85)
ax4.set_ylabel("Disk Storage Size (Megabytes)", fontweight="bold")
ax4.set_title("Model Footprint Reduction (99.6% Smaller)", fontsize=12, fontweight="bold")
ax4.set_yscale("log")
ax4.grid(True, which="both", linestyle=":", alpha=0.5)

for bar in bars_size:
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, h * 1.25, f"{h} MB", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_img = os.path.join(os.path.dirname(BASE_DIR), "static", "model_accuracy_comparison.png")
plt.savefig(out_img, dpi=200)
print("Comparison graph saved to:", out_img)

artifact_img = r"C:\Users\USER\.gemini\antigravity\brain\7dfba23a-f154-427c-8ad5-df47f853f25e\model_accuracy_comparison.png"
plt.savefig(artifact_img, dpi=200)
print("Saved to artifact:", artifact_img)
