import os
import pandas as pd
import numpy as np


# ============================================================
# RetailPulse ML2
# 2-Year Historical Sales / Demand Dataset
# ============================================================

print("\n==============================================")
print("RetailPulse - 2-Year Sales Dataset")
print("==============================================\n")


# ------------------------------------------------------------
# 1. Find project paths
# ------------------------------------------------------------

base_dir = os.path.dirname(
    os.path.abspath(__file__)
)

project_dir = os.path.dirname(
    base_dir
)


product_path = os.path.join(
    project_dir,
    "ml",
    "product_master.csv"
)

weather_path = os.path.join(
    base_dir,
    "weather_history_2year.csv"
)

output_path = os.path.join(
    base_dir,
    "sales_history_2year.csv"
)


# ------------------------------------------------------------
# 2. Check files
# ------------------------------------------------------------

if not os.path.exists(product_path):

    raise FileNotFoundError(
        f"Product master not found:\n{product_path}"
    )


if not os.path.exists(weather_path):

    raise FileNotFoundError(
        f"Weather dataset not found:\n{weather_path}"
    )


# ------------------------------------------------------------
# 3. Load product master
# ------------------------------------------------------------

print("Loading product master...")

products = pd.read_csv(
    product_path
)

print(
    "Products loaded:",
    len(products)
)


# ------------------------------------------------------------
# 4. Load actual weather data
# ------------------------------------------------------------

print("\nLoading actual weather data...")

weather = pd.read_csv(
    weather_path
)

weather["date"] = pd.to_datetime(
    weather["date"]
)

weather.sort_values(
    "date",
    inplace=True
)

weather.reset_index(
    drop=True,
    inplace=True
)


print(
    "Weather records:",
    len(weather)
)

print(
    "Weather start:",
    weather["date"].min()
)

print(
    "Weather end:",
    weather["date"].max()
)


# ------------------------------------------------------------
# 5. Handle missing rainfall
# ------------------------------------------------------------

missing_rainfall = weather[
    "rainfall"
].isna().sum()

print(
    "\nMissing rainfall records:",
    missing_rainfall
)


# We use 0 only for missing rainfall values.
# This is documented so the preprocessing is reproducible.

weather["rainfall"] = weather[
    "rainfall"
].fillna(0)


# ------------------------------------------------------------
# 6. Set random seed
# ------------------------------------------------------------

np.random.seed(42)


# ------------------------------------------------------------
# 7. Create demand records
# ------------------------------------------------------------

print("\nGenerating demand records...")

sales_data = []


for _, product in products.iterrows():

    product_id = product["product_id"]

    product_name = product["product_name"]

    category = product["category"]

    dependency = product[
        "weather_dependency"
    ]

    shop_id = product["shop_id"]


    for _, day in weather.iterrows():

        date = day["date"]

        temperature = day["temperature"]

        min_temperature = day[
            "min_temperature"
        ]

        max_temperature = day[
            "max_temperature"
        ]

        humidity = day["humidity"]

        rainfall = day["rainfall"]

        wind_speed = day["wind_speed"]

        pressure = day["pressure"]


        # ----------------------------------------------------
        # Base demand
        # ----------------------------------------------------

        demand = np.random.randint(
            10,
            31
        )


        # ----------------------------------------------------
        # Calendar effects
        # ----------------------------------------------------

        day_of_week = date.dayofweek

        month = date.month

        is_weekend = (
            1
            if day_of_week in [5, 6]
            else 0
        )


        # Slight weekend demand increase
        if is_weekend:

            demand += np.random.randint(
                2,
                6
            )


        # ----------------------------------------------------
        # VERY HIGH weather dependency
        # ----------------------------------------------------

        if dependency == "Very High":

            # Hot-weather products
            if product_name in [

                "Vanilla Ice Cream",
                "Chocolate Ice Cream",
                "Strawberry Ice Cream",
                "Kulfi",
                "Ice Cream Cone",
                "Watermelon",
                "Cucumber"

            ]:

                if max_temperature >= 32:

                    demand += 15

                elif max_temperature >= 30:

                    demand += 9

                elif max_temperature >= 28:

                    demand += 4


            # Rain products
            elif product_name in [

                "Umbrella",
                "Raincoat",
                "Waterproof Shoe Cover"

            ]:

                if rainfall >= 20:

                    demand += 20

                elif rainfall >= 10:

                    demand += 14

                elif rainfall >= 5:

                    demand += 8


            # Other very weather-sensitive products
            else:

                if max_temperature >= 32:

                    demand += 10

                if rainfall >= 10:

                    demand += 8


        # ----------------------------------------------------
        # HIGH weather dependency
        # ----------------------------------------------------

        elif dependency == "High":

            if max_temperature >= 32:

                demand += 8

            elif max_temperature >= 30:

                demand += 5


            if rainfall >= 10:

                demand += 6

            elif rainfall >= 5:

                demand += 3


        # ----------------------------------------------------
        # MEDIUM weather dependency
        # ----------------------------------------------------

        elif dependency == "Medium":

            if max_temperature >= 32:

                demand += 4

            if rainfall >= 10:

                demand += 3


        # ----------------------------------------------------
        # LOW / VERY LOW dependency
        # ----------------------------------------------------

        else:

            if rainfall >= 20:

                demand -= 2


        # ----------------------------------------------------
        # Humidity effect
        # ----------------------------------------------------

        if dependency in [
            "Very High",
            "High"
        ]:

            if humidity >= 85:

                demand += 3


        # ----------------------------------------------------
        # Random variation
        # ----------------------------------------------------

        noise = np.random.randint(
            -5,
            6
        )

        demand += noise


        # ----------------------------------------------------
        # Prevent impossible demand
        # ----------------------------------------------------

        demand = max(
            1,
            demand
        )


        # ----------------------------------------------------
        # Store record
        # ----------------------------------------------------

        sales_data.append({

            "date": date,

            "shop_id": shop_id,

            "product_id": product_id,

            "product_name": product_name,

            "category": category,

            "weather_dependency": dependency,

            "temperature": temperature,

            "min_temperature": min_temperature,

            "max_temperature": max_temperature,

            "humidity": humidity,

            "rainfall": rainfall,

            "wind_speed": wind_speed,

            "pressure": pressure,

            "demand": demand

        })


# ------------------------------------------------------------
# 8. Convert to DataFrame
# ------------------------------------------------------------

sales_df = pd.DataFrame(
    sales_data
)


# ------------------------------------------------------------
# 9. Sort dataset
# ------------------------------------------------------------

sales_df.sort_values(
    by=[
        "product_id",
        "date"
    ],
    inplace=True
)

sales_df.reset_index(
    drop=True,
    inplace=True
)


# ------------------------------------------------------------
# 10. Display statistics
# ------------------------------------------------------------

print("\n==============================================")
print("SALES DATASET CREATED")
print("==============================================")

print(
    "\nTotal records:",
    len(sales_df)
)

print(
    "Expected records:",
    len(products) * len(weather)
)

print(
    "\nDate range:"
)

print(
    "First:",
    sales_df["date"].min()
)

print(
    "Last:",
    sales_df["date"].max()
)


print(
    "\nDemand statistics:"
)

print(
    sales_df["demand"].describe()
)


print(
    "\nMissing values:"
)

print(
    sales_df.isnull().sum()
)


print(
    "\nFirst 10 records:"
)

print(
    sales_df.head(10)
)


# ------------------------------------------------------------
# 11. Save dataset
# ------------------------------------------------------------

sales_df.to_csv(
    output_path,
    index=False
)


print(
    "\n=============================================="
)

print(
    "2-YEAR SALES DATASET SAVED SUCCESSFULLY"
)

print(
    "=============================================="
)

print(
    f"File: {output_path}"
)