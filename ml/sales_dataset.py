import pandas as pd
import numpy as np


# ============================================================
# RETAILPULSE - HISTORICAL SALES / DEMAND DATASET
# 80 PRODUCTS × HISTORICAL WEATHER DAYS
# ============================================================

print("\nCreating improved historical sales dataset...")


# ------------------------------------------------------------
# 1. Load product master
# ------------------------------------------------------------

products = pd.read_csv(
    "ml/product_master.csv"
)

print("Products loaded:", len(products))


# ------------------------------------------------------------
# 2. Load historical weather
# ------------------------------------------------------------

weather = pd.read_csv(
    "ml/weather_history.csv"
)

weather["date"] = pd.to_datetime(
    weather["date"]
)

print("Weather records loaded:", len(weather))


# ------------------------------------------------------------
# 3. Set random seed
# ------------------------------------------------------------

np.random.seed(42)


# ------------------------------------------------------------
# 4. Create sales records
# ------------------------------------------------------------

sales_data = []


for _, product in products.iterrows():

    product_id = product["product_id"]
    product_name = product["product_name"]
    category = product["category"]
    dependency = product["weather_dependency"]

    # Product-specific base demand
    #
    # This prevents every product from having the same
    # average demand.
    product_base_demand = np.random.randint(15, 45)

    # Product-specific sensitivity
    #
    # Slight variation between products belonging to the
    # same weather dependency level.
    temperature_sensitivity = np.random.uniform(0.7, 1.3)
    rainfall_sensitivity = np.random.uniform(0.7, 1.3)

    for _, day in weather.iterrows():

        date = day["date"]

        temperature = float(day["temperature"])
        rainfall = float(day["rainfall"])
        humidity = float(day["humidity"])


        # ====================================================
        # BASE DEMAND
        # ====================================================

        demand = float(product_base_demand)


        # ====================================================
        # WEEKEND EFFECT
        # ====================================================

        day_of_week = date.dayofweek

        # Saturday = 5
        # Sunday   = 6

        if day_of_week in [5, 6]:

            demand *= 1.12


        # ====================================================
        # MONTH / SEASON EFFECT
        # ====================================================

        month = date.month

        # Summer months
        if month in [3, 4, 5]:

            if dependency in ["Very High", "High"]:
                demand *= 1.10


        # Monsoon months
        elif month in [6, 7, 8, 9]:

            if dependency in ["Very High", "High"]:
                demand *= 1.08


        # ====================================================
        # TEMPERATURE EFFECT
        # ====================================================

        # ----------------------------------------------------
        # Very High dependency
        # ----------------------------------------------------

        if dependency == "Very High":

            if temperature >= 32:

                demand += (
                    15 *
                    temperature_sensitivity
                )

            elif temperature >= 30:

                demand += (
                    10 *
                    temperature_sensitivity
                )

            elif temperature >= 28:

                demand += (
                    5 *
                    temperature_sensitivity
                )


        # ----------------------------------------------------
        # High dependency
        # ----------------------------------------------------

        elif dependency == "High":

            if temperature >= 32:

                demand += (
                    9 *
                    temperature_sensitivity
                )

            elif temperature >= 30:

                demand += (
                    6 *
                    temperature_sensitivity
                )

            elif temperature >= 28:

                demand += (
                    3 *
                    temperature_sensitivity
                )


        # ----------------------------------------------------
        # Medium dependency
        # ----------------------------------------------------

        elif dependency == "Medium":

            if temperature >= 32:

                demand += (
                    5 *
                    temperature_sensitivity
                )

            elif temperature >= 30:

                demand += (
                    3 *
                    temperature_sensitivity
                )


        # ====================================================
        # RAINFALL EFFECT
        # ====================================================

        if rainfall >= 20:

            if dependency in ["Very High", "High"]:

                demand += (
                    18 *
                    rainfall_sensitivity
                )

            elif dependency == "Medium":

                demand += (
                    7 *
                    rainfall_sensitivity
                )

            else:

                demand += 1


        elif rainfall >= 10:

            if dependency in ["Very High", "High"]:

                demand += (
                    10 *
                    rainfall_sensitivity
                )

            elif dependency == "Medium":

                demand += (
                    4 *
                    rainfall_sensitivity
                )


        elif rainfall >= 5:

            if dependency in ["Very High", "High"]:

                demand += (
                    5 *
                    rainfall_sensitivity
                )


        # ====================================================
        # PRODUCT-SPECIFIC WEATHER BEHAVIOUR
        # ====================================================

        # ----------------------------------------------------
        # HOT WEATHER PRODUCTS
        # ----------------------------------------------------

        hot_products = [

            "Vanilla Ice Cream",
            "Chocolate Ice Cream",
            "Strawberry Ice Cream",
            "Kulfi",
            "Ice Cream Cone",
            "Watermelon",
            "Cucumber",
            "Coconut Water",
            "Lemon Juice",
            "Fruit Juice",
            "Soft Drink",
            "Coca Cola",
            "Pepsi",
            "Sprite",
            "Packaged Drinking Water"
        ]

        if product_name in hot_products:

            if temperature >= 32:

                demand += 10

            elif temperature >= 30:

                demand += 6

            elif temperature >= 28:

                demand += 3


        # ----------------------------------------------------
        # RAIN PRODUCTS
        # ----------------------------------------------------

        rain_products = [

            "Umbrella",
            "Raincoat"
        ]

        if product_name in rain_products:

            if rainfall >= 20:

                demand += 25

            elif rainfall >= 10:

                demand += 17

            elif rainfall >= 5:

                demand += 8


        # ----------------------------------------------------
        # BATTERY / EMERGENCY PRODUCTS
        # ----------------------------------------------------

        emergency_products = [

            "Battery",
            "Torch",
            "Candle"
        ]

        if product_name in emergency_products:

            if rainfall >= 20:

                demand += 8

            elif rainfall >= 10:

                demand += 4


        # ====================================================
        # HUMIDITY EFFECT
        # ====================================================

        if humidity >= 85:

            if dependency in ["Very High", "High"]:

                demand += 3

        elif humidity >= 75:

            if dependency == "Very High":

                demand += 1


        # ====================================================
        # RANDOM VARIATION
        # ====================================================

        # Smaller noise than the original dataset.
        #
        # This is important because excessive random noise
        # makes the target difficult for the ML model to learn.

        noise = np.random.normal(
            loc=0,
            scale=2.5
        )

        demand += noise


        # ====================================================
        # ROUND AND PREVENT NEGATIVE DEMAND
        # ====================================================

        demand = max(
            1,
            round(demand)
        )


        # ====================================================
        # SAVE RECORD
        # ====================================================

        sales_data.append({

            "date": date,

            "shop_id": product["shop_id"],

            "product_id": product_id,

            "product_name": product_name,

            "category": category,

            "weather_dependency": dependency,

            "temperature": round(
                temperature,
                2
            ),

            "humidity": round(
                humidity,
                2
            ),

            "rainfall": round(
                rainfall,
                2
            ),

            "demand": demand
        })


# ------------------------------------------------------------
# 5. Convert to DataFrame
# ------------------------------------------------------------

sales_df = pd.DataFrame(
    sales_data
)


# ------------------------------------------------------------
# 6. Sort dataset
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
# 7. Display information
# ------------------------------------------------------------

print(
    "\nSales dataset created successfully!"
)

print(
    "Total records:",
    len(sales_df)
)

print(
    "\nFirst 10 records:"
)

print(
    sales_df.head(10)
)

print(
    "\nColumns:"
)

print(
    sales_df.columns.tolist()
)


# ------------------------------------------------------------
# 8. Demand statistics
# ------------------------------------------------------------

print(
    "\nDemand statistics:"
)

print(
    sales_df["demand"].describe()
)


# ------------------------------------------------------------
# 9. Check missing values
# ------------------------------------------------------------

print(
    "\nMissing values:"
)

print(
    sales_df.isnull().sum()
)


# ------------------------------------------------------------
# 10. Save dataset
# ------------------------------------------------------------

sales_df.to_csv(
    "ml/sales_history.csv",
    index=False
)


print(
    "\nSales dataset saved successfully!"
)

print(
    "File: ml/sales_history.csv"
)