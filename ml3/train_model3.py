import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ml_training_data_realistic.csv")

MODEL_PATH = os.path.join(BASE_DIR, "model_ml3.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder_ml3.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_features_ml3.json")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "feature_importance_ml3.csv")

print("=" * 60)
print("RetailPulse ML3: Model Training on Realistic Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

# Cyclical calendar features
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Chronological split: 2024-2025 Train, 2026 Test
TEST_START = "2026-01-01"
train_df = df[df["date"] < TEST_START].copy()
test_df = df[df["date"] >= TEST_START].copy()

print(f"Training records: {len(train_df)} (2024-02-08 to 2025-12-31)")
print(f"Test records:     {len(test_df)}  (2026-01-01 to 2026-02-07)")

numerical_features = [
    "temperature", "min_temperature", "max_temperature", "humidity", "rainfall",
    "wind_speed", "pressure", "day_of_week", "month", "day", "is_weekend",
    "sin_month", "cos_month", "sin_dow", "cos_dow",
    "previous_day_sales", "lag_7day_sales", "rolling_7day_sales", "price"
]
categorical_features = ["product_id", "category", "weather_dependency"]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(train_df[categorical_features])

cat_col_names = list(encoder.get_feature_names_out(categorical_features))
all_feature_names = numerical_features + cat_col_names

def prepare_X(data):
    num_part = data[numerical_features].reset_index(drop=True)
    cat_part = pd.DataFrame(encoder.transform(data[categorical_features]), columns=cat_col_names)
    X = pd.concat([num_part, cat_part], axis=1)
    X.columns = X.columns.astype(str)
    return X

X_train = prepare_X(train_df)
y_train = train_df["demand"].values

X_test = prepare_X(test_df)
y_test = test_df["demand"].values

print("\nTraining GradientBoostingRegressor...")
model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=5,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluation
y_pred = np.clip(model.predict(X_test), 0, None)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n" + "=" * 40)
print("FINAL ML3 ACCURACY METRICS (2026 Test Set)")
print("=" * 40)
print(f"Mean Absolute Error (MAE) : {mae:.4f} units/day")
print(f"Root Mean Squared (RMSE)  : {rmse:.4f} units")
print(f"R² Score                  : {r2:.4f} ({r2*100:.2f}% variance explained)")
print(f"Mean Percentage Err (MAPE): {mape:.2f}%")
print("=" * 40)

# Feature Importance
importances = pd.DataFrame({
    "feature": all_feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

importances.to_csv(IMPORTANCE_PATH, index=False)
print("\nTop 10 Most Important Features:")
print(importances.head(10).to_string(index=False))

# Save artifacts
joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

metadata = {
    "model_name": "GradientBoostingRegressor",
    "pipeline_version": "ML3-Realistic",
    "training_period": {"start": "2024-02-08", "end": "2025-12-31"},
    "test_period": {"start": "2026-01-01", "end": "2026-02-07"},
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "all_features": all_feature_names,
    "best_parameters": {
        "n_estimators": 150,
        "learning_rate": 0.08,
        "max_depth": 5,
        "min_samples_split": 3,
        "min_samples_leaf": 2
    },
    "metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape)
    }
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"\nArtifacts successfully saved to: {BASE_DIR}")
