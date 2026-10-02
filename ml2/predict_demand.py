import os
import json
import joblib
import pandas as pd


# ============================================================
# RetailPulse - Demand Prediction Service
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(
    BASE_DIR,
    "random_forest_model_2year.pkl"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "encoder_2year.pkl"
)

METADATA_PATH = os.path.join(
    BASE_DIR,
    "model_features_2year.json"
)


# ============================================================
# Load trained model, encoder and metadata
# ============================================================

print("Loading RetailPulse demand prediction model...")

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

with open(METADATA_PATH, "r") as file:
    metadata = json.load(file)

print("Model loaded successfully.")


# ============================================================
# Features used during training
# ============================================================

CATEGORICAL_FEATURES = metadata["categorical_features"]

NUMERICAL_FEATURES = metadata["numerical_features"]


# Get the exact feature names used by the trained model

ENCODED_FEATURE_NAMES = encoder.get_feature_names_out(
    CATEGORICAL_FEATURES
)

MODEL_FEATURE_NAMES = (
    NUMERICAL_FEATURES
    + list(ENCODED_FEATURE_NAMES)
)


# ============================================================
# Prediction function
# ============================================================

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
    pressure
):

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    date = pd.to_datetime(date)

    day_of_week = date.dayofweek
    month = date.month
    day = date.day

    is_weekend = 1 if day_of_week in [5, 6] else 0


    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        # Numerical features
        "day_of_week": day_of_week,
        "month": month,
        "day": day,
        "is_weekend": is_weekend,
        "previous_day_sales": previous_day_sales,
        "rolling_7day_sales": rolling_7day_sales,

        "temperature": temperature,
        "min_temperature": min_temperature,
        "max_temperature": max_temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "pressure": pressure,

        # Categorical features
        "product_id": product_id,
        "category": category,
        "weather_dependency": weather_dependency

    }])


    # --------------------------------------------------------
    # Select numerical features
    # --------------------------------------------------------

    numerical_data = input_data[
        NUMERICAL_FEATURES
    ].copy()


    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------

    categorical_data = input_data[
        CATEGORICAL_FEATURES
    ]

    encoded_data = encoder.transform(
        categorical_data
    )


    encoded_df = pd.DataFrame(
        encoded_data,
        columns=ENCODED_FEATURE_NAMES,
        index=input_data.index
    )


    # --------------------------------------------------------
    # Combine numerical + encoded categorical features
    # --------------------------------------------------------

    final_features = pd.concat(
        [
            numerical_data,
            encoded_df
        ],
        axis=1
    )


    # --------------------------------------------------------
    # Force exact feature order used during training
    # --------------------------------------------------------

    final_features = final_features.reindex(
        columns=MODEL_FEATURE_NAMES
    )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(
        final_features
    )[0]


    # --------------------------------------------------------
    # Demand cannot be negative
    # --------------------------------------------------------

    prediction = max(0, prediction)


    return round(float(prediction), 2)


# ============================================================
# Test prediction
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("TESTING DEMAND PREDICTION")
    print("=" * 50)


    predicted = predict_demand(

        product_id="P001",

        category="Beverages",

        weather_dependency="High",

        date="2026-02-08",

        previous_day_sales=25,

        rolling_7day_sales=27.5,

        temperature=29.0,

        min_temperature=24.0,

        max_temperature=33.0,

        humidity=75,

        rainfall=2.0,

        wind_speed=6.0,

        pressure=1012.0
    )


    print("\nPrediction Result")
    print("------------------------------")

    print("Product ID       :", "P001")

    print("Prediction Date  :", "2026-02-08")

    print("Predicted Demand :", predicted, "units")

    print("------------------------------")