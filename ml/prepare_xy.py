import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PREPARE X, y AND TRAIN/TEST DATA
# ============================================================

print("\nPreparing X, y and train/test data...")


# ------------------------------------------------------------
# 1. Load dataset
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
# 5. Define train/test split
# ------------------------------------------------------------

train_mask = df["date"] < "2025-11-01"

test_mask = df["date"] >= "2025-11-01"


X_train = X.loc[train_mask]

X_test = X.loc[test_mask]

y_train = y.loc[train_mask]

y_test = y.loc[test_mask]


# ------------------------------------------------------------
# 6. Define columns
# ------------------------------------------------------------

categorical_columns = [
    "product_id",
    "category",
    "weather_dependency"
]

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
# 7. Create encoder
# ------------------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# ------------------------------------------------------------
# 8. FIT encoder ONLY on training data
# ------------------------------------------------------------

X_train_categorical = encoder.fit_transform(
    X_train[categorical_columns]
)


# ------------------------------------------------------------
# 9. TRANSFORM test data
# ------------------------------------------------------------

X_test_categorical = encoder.transform(
    X_test[categorical_columns]
)


# ------------------------------------------------------------
# 10. Get encoded column names
# ------------------------------------------------------------

encoded_columns = encoder.get_feature_names_out(
    categorical_columns
)


# ------------------------------------------------------------
# 11. Convert encoded data to DataFrames
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
# 12. Get numerical data
# ------------------------------------------------------------

X_train_numerical = X_train[
    numerical_columns
].copy()

X_test_numerical = X_test[
    numerical_columns
].copy()


# ------------------------------------------------------------
# 13. Combine numerical + categorical features
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
# 14. Display results
# ------------------------------------------------------------

print("\nTrain/Test Split:")

print(
    "Training period:",
    df.loc[train_mask, "date"].min().date(),
    "to",
    df.loc[train_mask, "date"].max().date()
)

print(
    "Testing period:",
    df.loc[test_mask, "date"].min().date(),
    "to",
    df.loc[test_mask, "date"].max().date()
)


print("\nTraining data:")

print(
    "X_train:",
    X_train_final.shape
)

print(
    "y_train:",
    y_train.shape
)


print("\nTesting data:")

print(
    "X_test:",
    X_test_final.shape
)

print(
    "y_test:",
    y_test.shape
)


# ------------------------------------------------------------
# 15. Check missing values
# ------------------------------------------------------------

print("\nMissing values:")

print(
    "X_train:",
    X_train_final.isnull().sum().sum()
)

print(
    "X_test:",
    X_test_final.isnull().sum().sum()
)

print(
    "y_train:",
    y_train.isnull().sum()
)

print(
    "y_test:",
    y_test.isnull().sum()
)