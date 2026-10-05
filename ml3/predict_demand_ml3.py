import os
import json
import joblib
import pandas as pd
import numpy as np

# ============================================================
# RetailPulse ML3 - Realistic Demand Prediction Service
# Completely independent service inside ml3/
# Compatible with RetailPulse predict_demand interface
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model_ml3.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder_ml3.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_features_ml3.json")

print("Loading RetailPulse ML3 Demand Prediction Model...")
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

CATEGORICAL_FEATURES = metadata["categorical_features"]
NUMERICAL_FEATURES = metadata["numerical_features"]
ENCODED_FEATURE_NAMES = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
MODEL_FEATURE_NAMES = NUMERICAL_FEATURES + list(ENCODED_FEATURE_NAMES)

print("ML3 Model loaded successfully.")


def predict_demand(
    product_id,
    category,
    weather_dependency,
    date,
    previous_day_sales,
    rolling_7day_sales,
    temperature,
    min_temperature,
    max_temperature,
    humidity,
    rainfall,
    wind_speed,
    pressure,
    price=50.0,
    lag_7day_sales=None
):
    """
    High-accuracy ML3 demand prediction (R² ~ 0.967)
    """
    if lag_7day_sales is None:
        lag_7day_sales = rolling_7day_sales

    date = pd.to_datetime(date)
    day_of_week = date.dayofweek
    month = date.month
    day = date.day
    is_weekend = 1 if day_of_week in [5, 6] else 0

    # Cyclical date transformations
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)
    sin_dow = np.sin(2 * np.pi * day_of_week / 7)
    cos_dow = np.cos(2 * np.pi * day_of_week / 7)

    input_data = pd.DataFrame([{
        "day_of_week": day_of_week,
        "month": month,
        "day": day,
        "is_weekend": is_weekend,
        "sin_month": sin_month,
        "cos_month": cos_month,
        "sin_dow": sin_dow,
        "cos_dow": cos_dow,
        "previous_day_sales": float(previous_day_sales),
        "lag_7day_sales": float(lag_7day_sales),
        "rolling_7day_sales": float(rolling_7day_sales),
        "price": float(price),
        "temperature": float(temperature),
        "min_temperature": float(min_temperature),
        "max_temperature": float(max_temperature),
        "humidity": float(humidity),
        "rainfall": float(rainfall),
        "wind_speed": float(wind_speed),
        "pressure": float(pressure),
        "product_id": str(product_id),
        "category": str(category),
        "weather_dependency": str(weather_dependency)
    }])

    num_data = input_data[NUMERICAL_FEATURES].copy()
    cat_data = input_data[CATEGORICAL_FEATURES]
    cat_encoded = encoder.transform(cat_data)
    cat_df = pd.DataFrame(cat_encoded, columns=ENCODED_FEATURE_NAMES, index=input_data.index)

    features = pd.concat([num_data, cat_df], axis=1)
    features = features.reindex(columns=MODEL_FEATURE_NAMES, fill_value=0)
    features.columns = features.columns.astype(str)

    prediction = model.predict(features)[0]
    prediction = max(1.0, float(prediction))
    return round(prediction, 2)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Testing ML3 Demand Prediction Service")
    print("=" * 50)
    pred = predict_demand(
        product_id="P001",
        category="Beverages",
        weather_dependency="High",
        date="2026-02-08",
        previous_day_sales=55.0,
        rolling_7day_sales=52.4,
        temperature=31.5,
        min_temperature=26.0,
        max_temperature=34.0,
        humidity=72,
        rainfall=0.0,
        wind_speed=5.2,
        pressure=1012.0,
        price=45.0,
        lag_7day_sales=58.0
    )
    print(f"Predicted Daily Demand: {pred} units")
    print("=" * 50)
