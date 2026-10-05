import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SALES_PATH = os.path.join(BASE_DIR, "sales_history_2year.csv")
PRICES_PATH = os.path.join(BASE_DIR, "product_prices.json")
DATASET_OUT = os.path.join(BASE_DIR, "ml_training_data_2year.csv")

MODEL_PATH = os.path.join(BASE_DIR, "gradient_boosting_model_2year.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "gradient_boosting_encoder_2year.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "gradient_boosting_features_2year.json")

print("=" * 60)
print("Training Gradient Boosting Model with Pricing & Lag-7")
print("=" * 60)

with open(PRICES_PATH, "r") as f:
    prices = json.load(f)

df = pd.read_csv(SALES_PATH)
df["date"] = pd.to_datetime(df["date"])
df.sort_values(by=["product_id", "date"], inplace=True)
df.reset_index(drop=True, inplace=True)

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

df["previous_day_sales"] = df.groupby("product_id")["demand"].shift(1)
df["lag_7day_sales"] = df.groupby("product_id")["demand"].shift(7)
df["rolling_7day_sales"] = df.groupby("product_id")["demand"].transform(
    lambda x: x.shift(1).rolling(window=7, min_periods=7).mean()
).round(2)
df["price"] = df["product_name"].map(prices)

# Handle initial lag rows
df_clean = df.dropna(subset=["previous_day_sales", "lag_7day_sales", "rolling_7day_sales", "price"]).reset_index(drop=True)
df_clean.to_csv(DATASET_OUT, index=False)
print(f"Updated ML dataset saved: {len(df_clean)} rows.")

TEST_START = "2026-01-01"
train_df = df_clean[df_clean["date"] < TEST_START].copy()
test_df = df_clean[df_clean["date"] >= TEST_START].copy()

weather_features = ["temperature", "min_temperature", "max_temperature", "humidity", "rainfall", "wind_speed", "pressure"]
calendar_features = ["day_of_week", "month", "day", "is_weekend"]
history_features = ["previous_day_sales", "lag_7day_sales", "rolling_7day_sales", "price"]
categorical_features = ["product_id", "category", "weather_dependency"]
numerical_features = weather_features + calendar_features + history_features

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(train_df[categorical_features])

def prepare_features(data):
    num_data = data[numerical_features].reset_index(drop=True)
    cat_data = pd.DataFrame(
        encoder.transform(data[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features)
    )
    return pd.concat([num_data, cat_data], axis=1)

X_train = prepare_features(train_df)
y_train = train_df["demand"].values

X_test = prepare_features(test_df)
y_test = test_df["demand"].values

print("Training final GradientBoostingRegressor...")
model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = np.clip(model.predict(X_test), 0, None)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Final Model Metrics:")
print(f"  MAE : {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R2  : {r2:.4f}")
print(f"  MAPE: {mape:.2f}%")

# Save model, encoder, and metadata
joblib.dump(model, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

metadata = {
    "model": "GradientBoostingRegressor",
    "features": numerical_features + categorical_features,
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "best_parameters": {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 4
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

print(f"Model artifacts successfully saved to {BASE_DIR}")
