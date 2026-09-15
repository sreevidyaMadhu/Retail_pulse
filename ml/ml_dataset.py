import pandas as pd


# ============================================================
# PREPARE ML DATASET
# ============================================================

print("\nPreparing ML dataset...")


# ------------------------------------------------------------
# 1. Load sales history
# ------------------------------------------------------------

df = pd.read_csv(
    "ml/sales_history.csv"
)

print("Sales records:", len(df))


# ------------------------------------------------------------
# 2. Convert date
# ------------------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"]
)


# ------------------------------------------------------------
# 3. Sort data
# ------------------------------------------------------------

df.sort_values(
    by=[
        "product_id",
        "date"
    ],
    inplace=True
)


# ------------------------------------------------------------
# 4. Create time-based features
# ------------------------------------------------------------

df["day_of_week"] = df[
    "date"
].dt.dayofweek

df["month"] = df[
    "date"
].dt.month

df["day"] = df[
    "date"
].dt.day


# ------------------------------------------------------------
# 5. Create previous-day sales
# ------------------------------------------------------------

df["previous_day_sales"] = df.groupby(
    "product_id"
)["demand"].shift(1)


# ------------------------------------------------------------
# 6. Create 7-day rolling sales
# ------------------------------------------------------------

df["rolling_7day_sales"] = df.groupby(
    "product_id"
)["demand"].transform(
    lambda x: x.shift(1).rolling(7).mean()
)


# ------------------------------------------------------------
# 7. Remove rows without previous sales
# ------------------------------------------------------------

df.dropna(
    subset=[
        "previous_day_sales",
        "rolling_7day_sales"
    ],
    inplace=True
)


# ------------------------------------------------------------
# 8. Reset index
# ------------------------------------------------------------

df.reset_index(
    drop=True,
    inplace=True
)


# ------------------------------------------------------------
# 9. Display dataset
# ------------------------------------------------------------

print(
    "\nML dataset created!"
)

print(
    "Total records:",
    len(df)
)

print(
    "\nFirst 10 records:"
)

print(
    df.head(10)
)


# ------------------------------------------------------------
# 10. Display columns
# ------------------------------------------------------------

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


# ------------------------------------------------------------
# 11. Check missing values
# ------------------------------------------------------------

print(
    "\nMissing values:"
)

print(
    df.isnull().sum()
)


# ------------------------------------------------------------
# 12. Save ML dataset
# ------------------------------------------------------------

df.to_csv(
    "ml/ml_training_data.csv",
    index=False
)


print(
    "\nML training dataset saved successfully!"
)

print(
    "File: ml/ml_training_data.csv"
)