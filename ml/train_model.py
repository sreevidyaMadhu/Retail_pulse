import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# RANDOM FOREST DEMAND PREDICTION
# ============================================================

print("\nStarting Random Forest training...")


# ------------------------------------------------------------
# 1. Load ML dataset
# ------------------------------------------------------------

df = pd.read_csv(
    "ml/ml_training_data.csv"
)

df["date"] = pd.to_datetime(df["date"])

print("Total records:", len(df))


# ------------------------------------------------------------
# 2. Sort by date
# ------------------------------------------------------------

df.sort_values(
    by="date",
    inplace=True
)


# ------------------------------------------------------------
# 3. Define features
# ------------------------------------------------------------

features = [
    "temperature",
    "humidity",
    "rainfall",
    "day_of_week",
    "month",
    "previous_day_sales",
    "rolling_7day_sales",
    "product_id",
    "category",
    "weather_dependency"
]


# ------------------------------------------------------------
# 4. Create X and y
# ------------------------------------------------------------

X = df[features]

y = df["demand"]


# ------------------------------------------------------------
# 5. Define categorical columns
# ------------------------------------------------------------

categorical_columns = [
    "product_id",
    "category",
    "weather_dependency"
]


# ------------------------------------------------------------
# 6. Define numerical columns
# ------------------------------------------------------------

numerical_columns = [
    "temperature",
    "humidity",
    "rainfall",
    "day_of_week",
    "month",
    "previous_day_sales",
    "rolling_7day_sales"
]


# ------------------------------------------------------------
# 7. Train/Test split
# ------------------------------------------------------------

train_mask = df["date"] < "2025-11-01"

test_mask = df["date"] >= "2025-11-01"


X_train = X.loc[train_mask]

X_test = X.loc[test_mask]

y_train = y.loc[train_mask]

y_test = y.loc[test_mask]


# ------------------------------------------------------------
# 8. Create encoder
# ------------------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# ------------------------------------------------------------
# 9. Fit encoder ONLY on training data
# ------------------------------------------------------------

X_train_categorical = encoder.fit_transform(
    X_train[categorical_columns]
)


# ------------------------------------------------------------
# 10. Transform test data
# ------------------------------------------------------------

X_test_categorical = encoder.transform(
    X_test[categorical_columns]
)


# ------------------------------------------------------------
# 11. Get encoded column names
# ------------------------------------------------------------

encoded_columns = encoder.get_feature_names_out(
    categorical_columns
)


# ------------------------------------------------------------
# 12. Convert encoded data to DataFrames
# ------------------------------------------------------------

X_train_categorical = pd.DataFrame(
    X_train_categorical,
    columns=encoded_columns,
    index=X_train.index
)

X_test_categorical = pd.DataFrame(
    X_test_categorical,
    columns=encoded_columns,
    index=X_test.index
)


# ------------------------------------------------------------
# 13. Numerical features
# ------------------------------------------------------------

X_train_numerical = X_train[
    numerical_columns
]

X_test_numerical = X_test[
    numerical_columns
]


# ------------------------------------------------------------
# 14. Combine features
# ------------------------------------------------------------

X_train_final = pd.concat(
    [
        X_train_numerical,
        X_train_categorical
    ],
    axis=1
)

X_test_final = pd.concat(
    [
        X_test_numerical,
        X_test_categorical
    ],
    axis=1
)


# ------------------------------------------------------------
# 15. Create Random Forest model
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------------------------
# 16. TRAIN THE MODEL
# ------------------------------------------------------------

print("\nTraining Random Forest...")

model.fit(
    X_train_final,
    y_train
)

print("Training completed!")


# ------------------------------------------------------------
# 17. Make predictions
# ------------------------------------------------------------

print("\nMaking predictions...")

y_pred = model.predict(
    X_test_final
)

# ------------------------------------------------------------
# Product-wise prediction analysis
# ------------------------------------------------------------

test_info = df.loc[
    test_mask,
    ["product_id", "product_name"]
].copy().reset_index(drop=True)

test_info["actual_demand"] = y_test.reset_index(drop=True)
test_info["predicted_demand"] = y_pred

product_summary = (
    test_info
    .groupby(["product_id", "product_name"], as_index=False)
    .agg(
        actual_average=("actual_demand", "mean"),
        predicted_average=("predicted_demand", "mean")
    )
)

product_summary["difference"] = (
    product_summary["predicted_average"]
    - product_summary["actual_average"]
)

print("\nProduct-wise demand comparison:")
print(product_summary.to_string(index=False))


# ------------------------------------------------------------
# 18. Evaluate model
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# 19. Display results
# ------------------------------------------------------------

print("\n===================================")
print("MODEL EVALUATION")
print("===================================")

print(
    "MAE:",
    round(mae, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)

print(
    "R² Score:",
    round(r2, 4)
)


# ------------------------------------------------------------
# 20. Display sample predictions
# ------------------------------------------------------------

results = pd.DataFrame({
    "Actual Demand": y_test.values,
    "Predicted Demand": y_pred
})

print("\nSample Predictions:")

print(
    results.head(10)
)