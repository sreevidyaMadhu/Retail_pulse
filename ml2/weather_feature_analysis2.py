import os
import pandas as pd
import joblib

print("=" * 50)
print("RetailPulse - Weather Feature Analysis")
print("=" * 50)


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    base_dir,
    "random_forest_model_2year.pkl"
)

encoder_path = os.path.join(
    base_dir,
    "encoder_2year.pkl"
)


# ------------------------------------------------------------
# 2. Load model
# ------------------------------------------------------------

print("\nLoading trained Random Forest model...")

model = joblib.load(model_path)

print("Model loaded successfully.")


# ------------------------------------------------------------
# 3. Load encoder
# ------------------------------------------------------------

print("\nLoading encoder...")

encoder = joblib.load(encoder_path)

print("Encoder loaded successfully.")


# ------------------------------------------------------------
# 4. Get encoded categorical feature names
# ------------------------------------------------------------

print("\nExtracting encoded categorical features...")

encoded_features = list(
    encoder.get_feature_names_out()
)

print(
    f"Encoded categorical features: "
    f"{len(encoded_features)}"
)


# ------------------------------------------------------------
# 5. Numerical features
# ------------------------------------------------------------

numerical_features = [
    "day_of_week",
    "month",
    "day",
    "is_weekend",
    "previous_day_sales",
    "rolling_7day_sales",
    "temperature",
    "min_temperature",
    "max_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure"
]


print(
    f"Numerical features: "
    f"{len(numerical_features)}"
)


# ------------------------------------------------------------
# 6. Combine all model features
# ------------------------------------------------------------

all_features = (
    encoded_features +
    numerical_features
)


print(
    f"Total reconstructed features: "
    f"{len(all_features)}"
)

print(
    f"Random Forest importance values: "
    f"{len(model.feature_importances_)}"
)


# ------------------------------------------------------------
# 7. Safety check
# ------------------------------------------------------------

if len(all_features) != len(model.feature_importances_):

    raise RuntimeError(
        "\nFeature count mismatch!\n"
        f"Reconstructed features: {len(all_features)}\n"
        f"Model importances: {len(model.feature_importances_)}"
    )


print("\nFeature count verified successfully.")


# ------------------------------------------------------------
# 8. Create importance dataframe
# ------------------------------------------------------------

importance_df = pd.DataFrame({
    "feature": all_features,
    "importance": model.feature_importances_
})


importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)


# ------------------------------------------------------------
# 9. Display top 20
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("TOP 20 MODEL FEATURES")
print("=" * 50)

print(
    importance_df.head(20).to_string(index=False)
)


# ------------------------------------------------------------
# 10. Weather features
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


weather_df = importance_df[
    importance_df["feature"].isin(weather_features)
].copy()


# ------------------------------------------------------------
# 11. Display weather importance
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("WEATHER FEATURE IMPORTANCE")
print("=" * 50)

print(
    weather_df.to_string(index=False)
)


# ------------------------------------------------------------
# 12. Total weather importance
# ------------------------------------------------------------

total_weather_importance = (
    weather_df["importance"].sum()
)


print("\n" + "=" * 50)
print("TOTAL WEATHER FEATURE IMPORTANCE")
print("=" * 50)

print(
    f"Total weather feature importance: "
    f"{total_weather_importance:.4f}"
)


# ------------------------------------------------------------
# 13. Weather percentage
# ------------------------------------------------------------

weather_percentage = (
    total_weather_importance * 100
)


print(
    f"Weather features account for approximately "
    f"{weather_percentage:.2f}% of total Random Forest importance."
)


# ------------------------------------------------------------
# 14. Save results
# ------------------------------------------------------------

output_path = os.path.join(
    base_dir,
    "weather_feature_importance_2year.csv"
)

weather_df.to_csv(
    output_path,
    index=False
)


print("\nResults saved to:")
print(output_path)


# ------------------------------------------------------------
# 15. Complete
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("WEATHER FEATURE ANALYSIS COMPLETE")
print("=" * 50)