import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# RETAILPULSE - 2 YEAR RANDOM FOREST TRAINING PIPELINE
# ============================================================

print("\n==============================================")
print("RetailPulse - 2-Year ML Training")
print("Walk-Forward Validation + Optimization")
print("==============================================\n")


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "ml_training_data_2year.csv"
)

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
# 2. LOAD DATA
# ============================================================

print("Loading 2-year ML dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

print("Total records:", len(df))
print("First date:", df["date"].min())
print("Last date:", df["date"].max())


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

weather_features = [
    "temperature",
    "min_temperature",
    "max_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure"
]

calendar_features = [
    "day_of_week",
    "month",
    "day",
    "is_weekend"
]

demand_history_features = [
    "previous_day_sales",
    "rolling_7day_sales"
]

categorical_features = [
    "product_id",
    "category",
    "weather_dependency"
]

numerical_features = (
    weather_features
    + calendar_features
    + demand_history_features
)

features = (
    numerical_features
    + categorical_features
)

target = "demand"


# ============================================================
# 4. CHECK FEATURES
# ============================================================

print("\nChecking required columns...")

missing_columns = [
    column
    for column in features + [target]
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

print("All required columns are available.")


# ============================================================
# 5. FINAL TEST PERIOD
# ============================================================
#
# IMPORTANT:
# 2026 data is NEVER used during optimization.
#
# This simulates predicting future demand.
# ============================================================

FINAL_TEST_START = "2026-01-01"

train_validation_df = df[
    df["date"] < FINAL_TEST_START
].copy()

test_df = df[
    df["date"] >= FINAL_TEST_START
].copy()

print("\n==============================================")
print("DATA SPLIT")
print("==============================================")

print(
    "Training + validation:",
    len(train_validation_df)
)

print(
    "Final test:",
    len(test_df)
)

print(
    "Training/validation dates:",
    train_validation_df["date"].min(),
    "to",
    train_validation_df["date"].max()
)

print(
    "Final test dates:",
    test_df["date"].min(),
    "to",
    test_df["date"].max()
)


# ============================================================
# 6. ENCODER
# ============================================================
#
# We create one encoder using the training/validation period.
#
# Unknown categories are safely ignored.
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoder.fit(
    train_validation_df[categorical_features]
)


# ============================================================
# 7. FUNCTION TO PREPARE FEATURES
# ============================================================

def prepare_features(data):

    numerical_data = data[
        numerical_features
    ].copy()

    categorical_data = encoder.transform(
        data[categorical_features]
    )

    categorical_data = pd.DataFrame(
        categorical_data,
        columns=encoder.get_feature_names_out(
            categorical_features
        ),
        index=data.index
    )

    final_data = pd.concat(
        [
            numerical_data,
            categorical_data
        ],
        axis=1
    )

    return final_data


# ============================================================
# 8. WALK-FORWARD VALIDATION
# ============================================================
#
# We don't randomly shuffle the time-series data.
#
# Example:
#
# Fold 1:
# Train → 2024 Feb to 2025 Jun
# Validate → 2025 Jul to Sep
#
# Fold 2:
# Train → 2024 Feb to 2025 Sep
# Validate → 2025 Oct to Dec
#
# ============================================================

print("\n==============================================")
print("WALK-FORWARD VALIDATION")
print("==============================================")


validation_periods = [
    (
        "2025-02-01",
        "2025-05-01",
        "2025-05-01",
        "2025-07-01"
    ),

    (
        "2025-02-01",
        "2025-07-01",
        "2025-07-01",
        "2025-09-01"
    ),

    (
        "2025-02-01",
        "2025-09-01",
        "2025-09-01",
        "2025-11-01"
    ),

    (
        "2025-02-01",
        "2025-11-01",
        "2025-11-01",
        "2026-01-01"
    )
]


# ============================================================
# 9. HYPERPARAMETER CANDIDATES
# ============================================================

parameter_sets = [

    {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 250,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 200,
        "max_depth": 25,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 250,
        "max_depth": 25,
        "min_samples_split": 3,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 250,
        "max_depth": 30,
        "min_samples_split": 3,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 300,
        "max_depth": 25,
        "min_samples_split": 4,
        "min_samples_leaf": 2
    }
]


# ============================================================
# 10. OPTIMIZATION
# ============================================================

best_parameters = None
best_validation_mae = float("inf")

optimization_results = []


for parameter_number, params in enumerate(
    parameter_sets,
    start=1
):

    print("\n----------------------------------------------")
    print(
        f"Testing parameter set {parameter_number}"
    )
    print("----------------------------------------------")

    print(params)

    fold_errors = []

    for fold_number, (
        train_start,
        train_end,
        validation_start,
        validation_end
    ) in enumerate(
        validation_periods,
        start=1
    ):

        train_fold = train_validation_df[
            (train_validation_df["date"] >= train_start)
            &
            (train_validation_df["date"] < train_end)
        ].copy()

        validation_fold = train_validation_df[
            (train_validation_df["date"] >= validation_start)
            &
            (train_validation_df["date"] < validation_end)
        ].copy()

        if len(train_fold) == 0:
            continue

        if len(validation_fold) == 0:
            continue

        X_train_fold = prepare_features(
            train_fold
        )

        X_validation_fold = prepare_features(
            validation_fold
        )

        y_train_fold = train_fold[target]

        y_validation_fold = validation_fold[target]

        model = RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train_fold,
            y_train_fold
        )

        validation_predictions = model.predict(
            X_validation_fold
        )

        fold_mae = mean_absolute_error(
            y_validation_fold,
            validation_predictions
        )

        fold_errors.append(fold_mae)

        print(
            f"Fold {fold_number} MAE: "
            f"{fold_mae:.4f}"
        )

    if not fold_errors:
        continue

    average_mae = np.mean(
        fold_errors
    )

    optimization_results.append(
        {
            "parameters": params,
            "fold_mae": fold_errors,
            "average_mae": float(
                average_mae
            )
        }
    )

    print(
        f"Average validation MAE: "
        f"{average_mae:.4f}"
    )

    if average_mae < best_validation_mae:

        best_validation_mae = average_mae

        best_parameters = params.copy()


# ============================================================
# 11. BEST PARAMETERS
# ============================================================

print("\n==============================================")
print("BEST PARAMETERS")
print("==============================================")

print(best_parameters)

print(
    "Best average validation MAE:",
    round(best_validation_mae, 4)
)


# ============================================================
# 12. RETRAIN FINAL MODEL
# ============================================================
#
# Now we use ALL 2024-2025 data.
#
# 2026 remains untouched until evaluation.
# ============================================================

print("\n==============================================")
print("TRAINING FINAL MODEL")
print("==============================================")

X_train_final = prepare_features(
    train_validation_df
)

y_train_final = train_validation_df[
    target
]

X_test_final = prepare_features(
    test_df
)

y_test_final = test_df[
    target
]


final_model = RandomForestRegressor(
    **best_parameters,
    random_state=42,
    n_jobs=-1
)

print("\nTraining optimized Random Forest...")

final_model.fit(
    X_train_final,
    y_train_final
)

print("Final model training completed.")


# ============================================================
# 13. FINAL 2026 PREDICTIONS
# ============================================================

print("\nMaking predictions on untouched 2026 data...")

y_pred = final_model.predict(
    X_test_final
)


# ============================================================
# 14. FINAL METRICS
# ============================================================

mae = mean_absolute_error(
    y_test_final,
    y_pred
)

rmse = mean_squared_error(
    y_test_final,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test_final,
    y_pred
)


print("\n==============================================")
print("FINAL MODEL EVALUATION")
print("==============================================")

print(
    "MAE:",
    round(mae, 4)
)

print(
    "RMSE:",
    round(rmse, 4)
)

print(
    "R² Score:",
    round(r2, 4)
)


# ============================================================
# 15. SAMPLE PREDICTIONS
# ============================================================

results = test_df[
    [
        "date",
        "product_id",
        "product_name",
        "demand"
    ]
].copy()

results["predicted_demand"] = y_pred

results["absolute_error"] = (
    results["demand"]
    - results["predicted_demand"]
).abs()


print("\n==============================================")
print("SAMPLE PREDICTIONS")
print("==============================================")

print(
    results.head(20).to_string(
        index=False
    )
)


# ============================================================
# 16. PRODUCT-WISE PERFORMANCE
# ============================================================

print("\n==============================================")
print("PRODUCT-WISE PERFORMANCE")
print("==============================================")


product_summary = (
    results
    .groupby(
        [
            "product_id",
            "product_name"
        ]
    )
    .agg(
        actual_average=(
            "demand",
            "mean"
        ),
        predicted_average=(
            "predicted_demand",
            "mean"
        ),
        MAE=(
            "absolute_error",
            "mean"
        )
    )
    .reset_index()
)

product_summary["difference"] = (
    product_summary["predicted_average"]
    - product_summary["actual_average"]
)

print(
    product_summary.to_string(
        index=False
    )
)


# ============================================================
# 17. FEATURE IMPORTANCE
# ============================================================

print("\n==============================================")
print("FEATURE IMPORTANCE")
print("==============================================")


feature_names = (
    numerical_features
    + list(
        encoder.get_feature_names_out(
            categorical_features
        )
    )
)

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": final_model.feature_importances_
    }
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 18. WEATHER FEATURE IMPORTANCE
# ============================================================

print("\n==============================================")
print("WEATHER FEATURE IMPORTANCE")
print("==============================================")


weather_importance = importance_df[
    importance_df["feature"].isin(
        weather_features
    )
]

print(
    weather_importance.to_string(
        index=False
    )
)


# ============================================================
# 19. SAVE MODEL
# ============================================================

print("\n==============================================")
print("SAVING MODEL ARTIFACTS")
print("==============================================")


joblib.dump(
    final_model,
    MODEL_PATH
)

joblib.dump(
    encoder,
    ENCODER_PATH
)


# ============================================================
# 20. SAVE METADATA
# ============================================================

metadata = {

    "model": "RandomForestRegressor",

    "training_period": {
        "start": str(
            train_validation_df["date"].min().date()
        ),
        "end": str(
            train_validation_df["date"].max().date()
        )
    },

    "test_period": {
        "start": str(
            test_df["date"].min().date()
        ),
        "end": str(
            test_df["date"].max().date()
        )
    },

    "features": features,

    "numerical_features": numerical_features,

    "categorical_features": categorical_features,

    "best_parameters": best_parameters,

    "validation_mae": float(
        best_validation_mae
    ),

    "test_mae": float(mae),

    "test_rmse": float(rmse),

    "test_r2": float(r2),

    "total_training_records": int(
        len(train_validation_df)
    ),

    "total_test_records": int(
        len(test_df)
    )
}


with open(
    METADATA_PATH,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("TRAINING PIPELINE COMPLETE")
print("==============================================")

print(
    "\nModel:",
    MODEL_PATH
)

print(
    "Encoder:",
    ENCODER_PATH
)

print(
    "Metadata:",
    METADATA_PATH
)

print("\nFinal metrics:")

print(
    f"MAE  = {mae:.4f}"
)

print(
    f"RMSE = {rmse:.4f}"
)

print(
    f"R²   = {r2:.4f}"
)

print("\nBest parameters:")

print(best_parameters)

print(
    "\n2026 test data was kept completely "
    "unseen during optimization."
)

print("\nDone!")