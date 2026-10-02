import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "ml_training_data_2year.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print("RetailPulse - Weather Ablation Experiment")
print("==============================================")

print("\nLoading 2-year ML dataset...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print(f"Total records: {len(df)}")
print(f"First date: {df['date'].min()}")
print(f"Last date:  {df['date'].max()}")


# ============================================================
# FEATURES
# ============================================================

# ------------------------------------------------------------
# Features WITHOUT actual weather measurements
# ------------------------------------------------------------

non_weather_features = [
    "day_of_week",
    "month",
    "day",
    "is_weekend",
    "previous_day_sales",
    "rolling_7day_sales",
    "product_id",
    "category",
    "weather_dependency"
]


# ------------------------------------------------------------
# Weather features
# ------------------------------------------------------------

weather_features = [
    "temperature",
    "min_temperature",
    "max_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure"
]


# ------------------------------------------------------------
# Complete feature set
# ------------------------------------------------------------

all_features = non_weather_features + weather_features


target = "demand"


# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print("\n==============================================")
print("CHRONOLOGICAL TRAIN / TEST SPLIT")
print("==============================================")

# Final test = 2026
train_df = df[df["date"] < "2026-01-01"].copy()

test_df = df[df["date"] >= "2026-01-01"].copy()

print(f"Training records : {len(train_df)}")
print(f"Testing records  : {len(test_df)}")

print(
    f"Training period  : "
    f"{train_df['date'].min()} to {train_df['date'].max()}"
)

print(
    f"Testing period   : "
    f"{test_df['date'].min()} to {test_df['date'].max()}"
)


# ============================================================
# FUNCTION TO TRAIN AND EVALUATE MODEL
# ============================================================

def train_and_evaluate(features, model_name):

    print("\n----------------------------------------------")
    print(model_name)
    print("----------------------------------------------")

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]


    # --------------------------------------------------------
    # Identify categorical and numerical columns
    # --------------------------------------------------------

    categorical_columns = [
        column
        for column in [
            "product_id",
            "category",
            "weather_dependency"
        ]
        if column in features
    ]

    numerical_columns = [
        column
        for column in features
        if column not in categorical_columns
    ]


    print("\nCategorical features:")
    print(categorical_columns)

    print("\nNumerical features:")
    print(numerical_columns)


    # --------------------------------------------------------
    # One-Hot Encoding
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            ),

            (
                "numerical",
                "passthrough",
                numerical_columns
            )
        ]
    )


    print("\nEncoding features...")

    X_train_encoded = preprocessor.fit_transform(X_train)

    X_test_encoded = preprocessor.transform(X_test)


    print(
        f"Encoded training shape: "
        f"{X_train_encoded.shape}"
    )


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_encoded,
        y_train
    )


    print("Training completed.")


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_encoded
    )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )


    print("\nRESULTS")

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


# ============================================================
# MODEL 1 - WITHOUT WEATHER
# ============================================================

result_without_weather = train_and_evaluate(
    non_weather_features,
    "MODEL 1 - WITHOUT WEATHER"
)


# ============================================================
# MODEL 2 - WITH WEATHER
# ============================================================

result_with_weather = train_and_evaluate(
    all_features,
    "MODEL 2 - WITH WEATHER"
)


# ============================================================
# COMPARISON
# ============================================================

print("\n==============================================")
print("WEATHER ABLATION COMPARISON")
print("==============================================")

comparison = pd.DataFrame([
    {
        "Model": result_without_weather["model"],
        "MAE": result_without_weather["mae"],
        "RMSE": result_without_weather["rmse"],
        "R2": result_without_weather["r2"]
    },
    {
        "Model": result_with_weather["model"],
        "MAE": result_with_weather["mae"],
        "RMSE": result_with_weather["rmse"],
        "R2": result_with_weather["r2"]
    }
])

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# IMPROVEMENT
# ============================================================

mae_without = result_without_weather["mae"]
mae_with = result_with_weather["mae"]

rmse_without = result_without_weather["rmse"]
rmse_with = result_with_weather["rmse"]

r2_without = result_without_weather["r2"]
r2_with = result_with_weather["r2"]


mae_improvement = mae_without - mae_with

rmse_improvement = rmse_without - rmse_with

r2_improvement = r2_with - r2_without


print("\n==============================================")
print("IMPACT OF WEATHER FEATURES")
print("==============================================")

print(
    f"MAE improvement  : "
    f"{mae_improvement:.4f}"
)

print(
    f"RMSE improvement : "
    f"{rmse_improvement:.4f}"
)

print(
    f"R² improvement   : "
    f"{r2_improvement:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(
    BASE_DIR,
    "weather_ablation_results.csv"
)

comparison.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")
print(output_path)


print("\n==============================================")
print("WEATHER ABLATION COMPLETE")
print("==============================================")
