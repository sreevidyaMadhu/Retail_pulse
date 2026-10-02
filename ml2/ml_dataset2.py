import os
import pandas as pd


# ============================================================
# RetailPulse ML2
# Prepare 2-Year ML Training Dataset
# ============================================================

print("\n==============================================")
print("RetailPulse - 2-Year ML Dataset")
print("==============================================\n")


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

base_dir = os.path.dirname(
    os.path.abspath(__file__)
)

sales_path = os.path.join(
    base_dir,
    "sales_history_2year.csv"
)

output_path = os.path.join(
    base_dir,
    "ml_training_data_2year.csv"
)


# ------------------------------------------------------------
# 2. Check input file
# ------------------------------------------------------------

if not os.path.exists(sales_path):

    raise FileNotFoundError(
        f"Sales dataset not found:\n{sales_path}"
    )


# ------------------------------------------------------------
# 3. Load sales dataset
# ------------------------------------------------------------

print("Loading 2-year sales dataset...")

df = pd.read_csv(
    sales_path
)

print(
    "Sales records:",
    len(df)
)


# ------------------------------------------------------------
# 4. Convert date
# ------------------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"]
)


# ------------------------------------------------------------
# 5. Sort chronologically per product
# ------------------------------------------------------------

df.sort_values(
    by=[
        "product_id",
        "date"
    ],
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)


# ------------------------------------------------------------
# 6. Calendar features
# ------------------------------------------------------------

print(
    "\nCreating calendar features..."
)

df["day_of_week"] = (
    df["date"].dt.dayofweek
)

df["month"] = (
    df["date"].dt.month
)

df["day"] = (
    df["date"].dt.day
)

df["is_weekend"] = (
    df["day_of_week"]
    .isin([5, 6])
    .astype(int)
)


# ------------------------------------------------------------
# 7. Previous-day sales
# ------------------------------------------------------------

print(
    "Creating previous-day sales..."
)

df["previous_day_sales"] = (
    df.groupby("product_id")["demand"]
    .shift(1)
)


# ------------------------------------------------------------
# 8. Rolling 7-day average
# ------------------------------------------------------------

print(
    "Creating 7-day rolling sales..."
)

df["rolling_7day_sales"] = (
    df.groupby("product_id")["demand"]
    .transform(
        lambda x:
        x.shift(1)
        .rolling(
            window=7,
            min_periods=7
        )
        .mean()
    )
)


# ------------------------------------------------------------
# 9. Check missing values caused by lag features
# ------------------------------------------------------------

initial_count = len(df)

df.dropna(
    subset=[
        "previous_day_sales",
        "rolling_7day_sales"
    ],
    inplace=True
)

final_count = len(df)

dropped_count = (
    initial_count - final_count
)


print(
    f"\nRows removed because of "
    f"lag/rolling features: {dropped_count}"
)


# ------------------------------------------------------------
# 10. Round rolling average
# ------------------------------------------------------------

df["rolling_7day_sales"] = (
    df["rolling_7day_sales"]
    .round(2)
)


# ------------------------------------------------------------
# 11. Reset index
# ------------------------------------------------------------

df.reset_index(
    drop=True,
    inplace=True
)


# ------------------------------------------------------------
# 12. Display summary
# ------------------------------------------------------------

print(
    "\n=============================================="
)

print(
    "2-YEAR ML DATASET CREATED"
)

print(
    "=============================================="
)

print(
    "\nTotal records:",
    len(df)
)

print(
    "First date:",
    df["date"].min()
)

print(
    "Last date:",
    df["date"].max()
)


print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)


print(
    "\nMissing values:"
)

print(
    df.isnull().sum()
)


print(
    "\nFirst 10 records:"
)

print(
    df.head(10)
)


# ------------------------------------------------------------
# 13. Save ML dataset
# ------------------------------------------------------------

df.to_csv(
    output_path,
    index=False
)


print(
    "\n=============================================="
)

print(
    "ML TRAINING DATASET SAVED"
)

print(
    "=============================================="
)

print(
    f"File: {output_path}"
)