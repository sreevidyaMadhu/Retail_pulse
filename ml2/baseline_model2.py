import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# RetailPulse - 7-Day Baseline Model
# ============================================================

print()
print("=" * 46)
print("RetailPulse - 7-Day Baseline Model")
print("=" * 46)


# ============================================================
# 1. Load ML dataset
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(
    base_dir,
    "ml_training_data_2year.csv"
)

print()
print("Loading 2-year ML dataset...")

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Dataset not found: {data_path}"
    )

df = pd.read_csv(data_path)

df["date"] = pd.to_datetime(df["date"])

print(f"Total records: {len(df)}")
print(f"First date: {df['date'].min()}")
print(f"Last date: {df['date'].max()}")


# ============================================================
# 2. Define the same final test period
# ============================================================

test_start = pd.Timestamp("2026-01-01")
test_end = pd.Timestamp("2026-02-07")

test_df = df[
    (df["date"] >= test_start) &
    (df["date"] <= test_end)
].copy()

print()
print("=" * 46)
print("TEST DATA")
print("=" * 46)

print(f"Test records: {len(test_df)}")
print(f"Test start:   {test_df['date'].min()}")
print(f"Test end:     {test_df['date'].max()}")


# ============================================================
# 3. Check required columns
# ============================================================

required_columns = [
    "demand",
    "rolling_7day_sales"
]

for column in required_columns:
    if column not in test_df.columns:
        raise ValueError(
            f"Required column missing: {column}"
        )

print()
print("Required columns available.")


# ============================================================
# 4. Baseline prediction
# ============================================================

print()
print("=" * 46)
print("CREATING BASELINE PREDICTIONS")
print("=" * 46)

# The baseline simply predicts today's demand
# using the average demand from the previous 7 days.

test_df["predicted_demand"] = (
    test_df["rolling_7day_sales"]
)


# ============================================================
# 5. Actual and predicted values
# ============================================================

y_actual = test_df["demand"]
y_predicted = test_df["predicted_demand"]


# ============================================================
# 6. Calculate metrics
# ============================================================

mae = mean_absolute_error(
    y_actual,
    y_predicted
)

rmse = np.sqrt(
    mean_squared_error(
        y_actual,
        y_predicted
    )
)

r2 = r2_score(
    y_actual,
    y_predicted
)


# ============================================================
# 7. Display results
# ============================================================

print()
print("=" * 46)
print("7-DAY BASELINE MODEL RESULTS")
print("=" * 46)

print()
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 8. Sample predictions
# ============================================================

print()
print("=" * 46)
print("SAMPLE PREDICTIONS")
print("=" * 46)

sample = test_df[
    [
        "date",
        "product_id",
        "product_name",
        "demand",
        "predicted_demand"
    ]
].head(20).copy()

sample["absolute_error"] = (
    abs(
        sample["demand"]
        - sample["predicted_demand"]
    )
)

print(sample.to_string(index=False))


# ============================================================
# 9. Compare with Random Forest
# ============================================================

print()
print("=" * 46)
print("COMPARISON WITH RANDOM FOREST")
print("=" * 46)

rf_mae = 5.6985
rf_rmse = 6.8243
rf_r2 = 0.2752

print()
print("                         MAE       RMSE       R²")
print("-" * 55)

print(
    f"7-Day Baseline       "
    f"{mae:.4f}    "
    f"{rmse:.4f}    "
    f"{r2:.4f}"
)

print(
    f"Random Forest        "
    f"{rf_mae:.4f}    "
    f"{rf_rmse:.4f}    "
    f"{rf_r2:.4f}"
)


# ============================================================
# 10. Difference
# ============================================================

print()
print("=" * 46)
print("BASELINE vs RANDOM FOREST")
print("=" * 46)

print()
print(
    f"MAE difference  : {rf_mae - mae:.4f}"
)

print(
    f"RMSE difference : {rf_rmse - rmse:.4f}"
)

print(
    f"R² difference   : {rf_r2 - r2:.4f}"
)


# ============================================================
# 11. Save baseline predictions
# ============================================================

output_path = os.path.join(
    base_dir,
    "baseline_predictions_2026.csv"
)

test_df.to_csv(
    output_path,
    index=False
)

print()
print(f"Predictions saved to:")
print(output_path)

print()
print("=" * 46)
print("BASELINE EVALUATION COMPLETE")
print("=" * 46)