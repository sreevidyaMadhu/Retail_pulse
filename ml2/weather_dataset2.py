import os
import pandas as pd


# ============================================================
# RETAILPULSE
# DIRECT METEOSTAT DAILY WEATHER DATA
# THIRUVANANTHAPURAM - STATION 43372
#
# Target period:
# August 2024 -> August 2026
# ============================================================

STATION = "43372"

YEARS = [
    2024,
    2025,
    2026
]

START_DATE = "2024-02-01"
END_DATE = "2026-08-31"


# ------------------------------------------------------------
# Project directory
# ------------------------------------------------------------

base_dir = os.path.dirname(
    os.path.abspath(__file__)
)


print("\n==============================================")
print("RetailPulse - Direct Meteostat Weather Data")
print("==============================================\n")


all_data = []


# ============================================================
# 1. DOWNLOAD YEARLY WEATHER DATA
# ============================================================

for year in YEARS:

    url = (
        f"https://data.meteostat.net/daily/"
        f"{year}/{STATION}.csv.gz"
    )

    print(
        f"Downloading {year} weather data..."
    )

    try:

        df_year = pd.read_csv(
            url,
            compression="gzip"
        )

        print(
            f"{year}: {len(df_year)} records"
        )

        # ----------------------------------------------------
        # IMPORTANT
        # The Meteostat CSV may use the date as the first
        # unnamed/index column.
        # ----------------------------------------------------

        print(
            f"{year} columns:",
            df_year.columns.tolist()
        )

        all_data.append(
            df_year
        )

    except Exception as e:

        print(
            f"Could not download {year}: {e}"
        )


# ============================================================
# 2. CHECK DOWNLOAD
# ============================================================

if not all_data:

    raise RuntimeError(
        "No weather data could be downloaded."
    )


# ============================================================
# 3. COMBINE DATA
# ============================================================

df = pd.concat(
    all_data,
    ignore_index=True
)


print("\nCombined columns:")

print(
    df.columns.tolist()
)


# ============================================================
# 4. CREATE DATE FROM YEAR, MONTH AND DAY
# ============================================================

print("\nCreating date column from year, month and day...")


df["date"] = pd.to_datetime(
    {
        "year": df["year"],
        "month": df["month"],
        "day": df["day"]
    }
)


print(
    "Date column created successfully."
)
# ============================================================
# 6. FILTER REQUIRED PERIOD
# ============================================================

df = df[
    (df["date"] >= START_DATE)
    &
    (df["date"] <= END_DATE)
].copy()


# ============================================================
# 7. RENAME METEOSTAT COLUMNS
# ============================================================

rename_columns = {

    "temp": "temperature",

    "tmin": "min_temperature",

    "tmax": "max_temperature",

    "rhum": "humidity",

    "prcp": "rainfall",

    "wspd": "wind_speed",

    "pres": "pressure"
}


df.rename(
    columns=rename_columns,
    inplace=True
)


# ============================================================
# 8. SELECT USEFUL WEATHER FEATURES
# ============================================================

required_columns = [

    "date",

    "temperature",

    "min_temperature",

    "max_temperature",

    "humidity",

    "rainfall",

    "wind_speed",

    "pressure"
]


available_columns = [

    column
    for column in required_columns
    if column in df.columns

]


df = df[
    available_columns
]


# ============================================================
# 9. SORT BY DATE
# ============================================================

df.sort_values(
    by="date",
    inplace=True
)


df.reset_index(
    drop=True,
    inplace=True
)


# ============================================================
# 10. DISPLAY INFORMATION
# ============================================================

print("\n==============================================")
print("WEATHER DATASET SUMMARY")
print("==============================================")

print(
    "\nFirst date:",
    df["date"].min()
)

print(
    "Last date:",
    df["date"].max()
)

print(
    "Total records:",
    len(df)
)


print("\nMissing values:")

print(
    df.isnull().sum()
)


print("\nFirst 10 records:")

print(
    df.head(10)
)


print("\nLast 10 records:")

print(
    df.tail(10)
)


# ============================================================
# 11. CHECK EXPECTED DATE RANGE
# ============================================================

expected_start = pd.Timestamp(
    START_DATE
)

expected_end = pd.Timestamp(
    END_DATE
)


actual_start = df["date"].min()
actual_end = df["date"].max()


print("\n==============================================")
print("DATE RANGE CHECK")
print("==============================================")


if actual_start == expected_start:

    print(
        "✓ Start date is correct:",
        actual_start.date()
    )

else:

    print(
        "⚠ Actual start date:",
        actual_start.date()
    )


if actual_end == expected_end:

    print(
        "✓ End date is correct:",
        actual_end.date()
    )

else:

    print(
        "⚠ Actual end date:",
        actual_end.date()
    )


# ============================================================
# 12. SAVE DATASET
# ============================================================

output_path = os.path.join(
    base_dir,
    "weather_history_2year.csv"
)


df.to_csv(
    output_path,
    index=False
)


print("\n==============================================")
print("WEATHER DATASET CREATED")
print("==============================================")

print(
    "Saved to:",
    output_path
)

print(
    "Total records:",
    len(df)
)

print(
    "\nColumns:"
)

print(
    df.columns.tolist()
)