import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# RetailPulse - Gradient Boosting Model
# Walk-Forward Validation + Optimization
# ============================================================

print()
print("=" * 60)
print("RetailPulse - Gradient Boosting Training")
print("Walk-Forward Validation + Optimization")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

base_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(
    base_dir,
    "ml_training_data_2year.csv"
)

model_path = os.path.join(
    base_dir,
    "gradient_boosting_model_2year.pkl"
)

encoder_path = os.path.join(
    base_dir,
    "gradient_boosting_encoder_2year.pkl"
)

metadata_path = os.path.join(
    base_dir,
    "gradient_boosting_features_2year.json"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print()
print("Loading 2-year ML dataset...")

if not os.path.exists(data_path):
    raise FileNotFoundError(
        f"Dataset not found: {data_path}"
    )

df = pd.read_csv(data_path)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    by=["date", "product_id"]
).reset_index(drop=True)

print(f"Total records: {len(df)}")
print(f"First date: {df['date'].min()}")
print(f"Last date: {df['date'].max()}")


# ============================================================
# 3. FEATURES
# ============================================================

features = [
    "temperature",
    "min_temperature",
    "max_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure",
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

target = "demand"


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

print()
print("=" * 60)
print("CHECKING REQUIRED COLUMNS")
print("=" * 60)

required_columns = features + [target, "date"]

missing_columns = [
    column
    for column in required_columns
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

test_start = pd.Timestamp("2026-01-01")
test_end = pd.Timestamp("2026-02-07")

final_test = df[
    (df["date"] >= test_start) &
    (df["date"] <= test_end)
].copy()

train_validation = df[
    df["date"] < test_start
].copy()

print()
print("=" * 60)
print("DATA SPLIT")
print("=" * 60)

print(
    f"Training + validation records: "
    f"{len(train_validation)}"
)

print(
    f"Final test records: "
    f"{len(final_test)}"
)

print(
    f"Training/validation dates: "
    f"{train_validation['date'].min()} "
    f"to "
    f"{train_validation['date'].max()}"
)

print(
    f"Final test dates: "
    f"{final_test['date'].min()} "
    f"to "
    f"{final_test['date'].max()}"
)


# ============================================================
# 6. CATEGORICAL / NUMERICAL FEATURES
# ============================================================

categorical_columns = [
    "product_id",
    "category",
    "weather_dependency"
]

numerical_columns = [
    "temperature",
    "min_temperature",
    "max_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure",
    "day_of_week",
    "month",
    "day",
    "is_weekend",
    "previous_day_sales",
    "rolling_7day_sales"
]


# ============================================================
# 7. PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
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


# ============================================================
# 8. WALK-FORWARD VALIDATION
# ============================================================

print()
print("=" * 60)
print("WALK-FORWARD VALIDATION")
print("=" * 60)


# We use chronological validation.
#
# Fold 1:
# Train on early 2024 -> validate later 2024
#
# Fold 2:
# Train on 2024 -> validate early 2025
#
# Fold 3:
# Train on 2024 + early 2025 -> validate later 2025
#
# Fold 4:
# Train on most of 2025 -> validate final part of 2025


validation_periods = [
    (
        pd.Timestamp("2024-02-08"),
        pd.Timestamp("2024-07-31"),
        pd.Timestamp("2024-08-01"),
        pd.Timestamp("2024-12-31")
    ),
    (
        pd.Timestamp("2024-02-08"),
        pd.Timestamp("2024-12-31"),
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-04-30")
    ),
    (
        pd.Timestamp("2024-02-08"),
        pd.Timestamp("2025-04-30"),
        pd.Timestamp("2025-05-01"),
        pd.Timestamp("2025-08-31")
    ),
    (
        pd.Timestamp("2024-02-08"),
        pd.Timestamp("2025-08-31"),
        pd.Timestamp("2025-09-01"),
        pd.Timestamp("2025-12-31")
    )
]


# ============================================================
# 9. PARAMETER SETS
# ============================================================

parameter_sets = [

    {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 150,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_samples_split": 2,
        "min_samples_leaf": 1
    },

    {
        "n_estimators": 150,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_samples_split": 3,
        "min_samples_leaf": 2
    }
]


# ============================================================
# 10. OPTIMIZATION
# ============================================================

best_parameters = None
best_validation_mae = float("inf")


for parameter_number, params in enumerate(
    parameter_sets,
    start=1
):

    print()
    print("-" * 60)
    print(
        f"Testing parameter set {parameter_number}"
    )
    print("-" * 60)

    print(params)

    fold_scores = []

    for fold_number, (
        train_start,
        train_end,
        validation_start,
        validation_end
    ) in enumerate(
        validation_periods,
        start=1
    ):

        fold_train = train_validation[
            (train_validation["date"] >= train_start) &
            (train_validation["date"] <= train_end)
        ]

        fold_validation = train_validation[
            (train_validation["date"] >= validation_start) &
            (train_validation["date"] <= validation_end)
        ]

        X_train = fold_train[features]
        y_train = fold_train[target]

        X_validation = fold_validation[features]
        y_validation = fold_validation[target]

        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        **params,
                        random_state=42
                    )
                )
            ]
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_validation
        )

        fold_mae = mean_absolute_error(
            y_validation,
            predictions
        )

        fold_scores.append(fold_mae)

        print(
            f"Fold {fold_number} MAE: "
            f"{fold_mae:.4f}"
        )

    average_mae = np.mean(
        fold_scores
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

print()
print("=" * 60)
print("BEST PARAMETERS")
print("=" * 60)

print(best_parameters)

print(
    f"Best average validation MAE: "
    f"{best_validation_mae:.4f}"
)


# ============================================================
# 12. TRAIN FINAL MODEL
# ============================================================

print()
print("=" * 60)
print("TRAINING FINAL GRADIENT BOOSTING MODEL")
print("=" * 60)

X_train_final = train_validation[
    features
]

y_train_final = train_validation[
    target
]

X_test_final = final_test[
    features
]

y_test_final = final_test[
    target
]


final_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            GradientBoostingRegressor(
                **best_parameters,
                random_state=42
            )
        )
    ]
)

final_model.fit(
    X_train_final,
    y_train_final
)

print(
    "Final model training completed."
)


# ============================================================
# 13. FINAL TEST
# ============================================================

print()
print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

test_predictions = final_model.predict(
    X_test_final
)

test_predictions = np.maximum(
    test_predictions,
    0
)

mae = mean_absolute_error(
    y_test_final,
    test_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_final,
        test_predictions
    )
)

r2 = r2_score(
    y_test_final,
    test_predictions
)

print()
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ============================================================
# 14. SAMPLE PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

sample_predictions = final_test[
    [
        "date",
        "product_id",
        "product_name",
        "demand"
    ]
].copy()

sample_predictions[
    "predicted_demand"
] = test_predictions

sample_predictions[
    "absolute_error"
] = abs(
    sample_predictions["demand"]
    -
    sample_predictions["predicted_demand"]
)

print(
    sample_predictions
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 15. PRODUCT-WISE PERFORMANCE
# ============================================================

print()
print("=" * 60)
print("PRODUCT-WISE PERFORMANCE")
print("=" * 60)

evaluation_df = final_test[
    [
        "product_id",
        "product_name",
        "demand"
    ]
].copy()

evaluation_df[
    "predicted_demand"
] = test_predictions

evaluation_df[
    "absolute_error"
] = abs(
    evaluation_df["demand"]
    -
    evaluation_df["predicted_demand"]
)

product_performance = (
    evaluation_df
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

product_performance[
    "difference"
] = (
    product_performance[
        "predicted_average"
    ]
    -
    product_performance[
        "actual_average"
    ]
)

print(
    product_performance
    .to_string(index=False)
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

trained_preprocessor = (
    final_model
    .named_steps[
        "preprocessor"
    ]
)

regressor = (
    final_model
    .named_steps[
        "regressor"
    ]
)

feature_names = (
    trained_preprocessor
    .get_feature_names_out()
)

importance_values = (
    regressor
    .feature_importances_
)

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importance_values
    }
)

importance_df = (
    importance_df
    .sort_values(
        by="importance",
        ascending=False
    )
)

print(
    importance_df
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 17. WEATHER FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 60)
print("WEATHER FEATURE IMPORTANCE")
print("=" * 60)

weather_keywords = [
    "temperature",
    "min_temperature",
    "max_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure"
]

weather_importance = importance_df[
    importance_df["feature"].str.contains(
        "|".join(weather_keywords),
        case=False,
        regex=True
    )
].copy()

print(
    weather_importance
    .to_string(index=False)
)


# ============================================================
# 18. SAVE MODEL
# ============================================================

print()
print("=" * 60)
print("SAVING MODEL ARTIFACTS")
print("=" * 60)

joblib.dump(
    final_model,
    model_path
)

print(
    f"Model saved to:\n{model_path}"
)


# Save preprocessing information

metadata = {
    "model": "GradientBoostingRegressor",
    "features": features,
    "categorical_columns": categorical_columns,
    "numerical_columns": numerical_columns,
    "target": target,
    "best_parameters": best_parameters,
    "best_validation_mae": float(
        best_validation_mae
    ),
    "final_test_period": {
        "start": str(test_start.date()),
        "end": str(test_end.date())
    },
    "final_metrics": {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
}

with open(
    metadata_path,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )

print(
    f"Metadata saved to:\n{metadata_path}"
)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("GRADIENT BOOSTING PIPELINE COMPLETE")
print("=" * 60)

print()
print("Final metrics:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

print()
print("Best parameters:")
print(best_parameters)

print()
print(
    "2026 test data was kept completely "
    "unseen during optimization."
)

print()
print("Done!")