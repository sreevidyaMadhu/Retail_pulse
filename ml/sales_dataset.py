import pandas as pd
import numpy as np


# ============================================================
# HISTORICAL SALES / DEMAND DATASET
# 80 PRODUCTS × 365 DAYS
# ============================================================

print("\nCreating historical sales dataset...")


# ------------------------------------------------------------
# 1. Read product master
# ------------------------------------------------------------

products = pd.read_csv(
    "ml/product_master.csv"
)

print("Products loaded:", len(products))


# ------------------------------------------------------------
# 2. Read historical weather
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

    for _, day in weather.iterrows():

        date = day["date"]

        temperature = day["temperature"]
        rainfall = day["rainfall"]
        humidity = day["humidity"]


        # ----------------------------------------------------
        # Base demand
        # ----------------------------------------------------

        demand = np.random.randint(10, 31)


        # ----------------------------------------------------
        # Weather effect
        # ----------------------------------------------------

        # Very weather-dependent products
        if dependency == "Very High":

            # Hot weather products
            if product_name in [
                "Vanilla Ice Cream",
                "Chocolate Ice Cream",
                "Strawberry Ice Cream",
                "Kulfi",
                "Ice Cream Cone",
                "Watermelon",
                "Cucumber"
            ]:

                if temperature > 30:
                    demand += 15

                elif temperature > 28:
                    demand += 8


            # Rain-dependent products
            elif product_name in [
                "Umbrella",
                "Raincoat"
            ]:

                if rainfall > 10:
                    demand += 20

                elif rainfall > 5:
                    demand += 12


        # High weather dependency
        elif dependency == "High":

            if temperature > 30:
                demand += 8

            if rainfall > 10:
                demand += 5


        # Medium weather dependency
        elif dependency == "Medium":

            if temperature > 30:
                demand += 4

            if rainfall > 10:
                demand += 3


        # Low / Very Low dependency
        else:

            # Very little weather effect
            if rainfall > 20:
                demand -= 2


        # ----------------------------------------------------
        # Random variation
        # ----------------------------------------------------

        noise = np.random.randint(
            -5,
            6
        )

        demand += noise


        # ----------------------------------------------------
        # Prevent negative demand
        # ----------------------------------------------------

        demand = max(
            1,
            demand
        )


        # ----------------------------------------------------
        # Save record
        # ----------------------------------------------------

        sales_data.append({

            "date": date,

            "shop_id": product["shop_id"],

            "product_id": product_id,

            "product_name": product_name,

            "category": category,

            "weather_dependency": dependency,

            "temperature": temperature,

            "humidity": humidity,

            "rainfall": rainfall,

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

print("\nSales dataset created successfully!")

print(
    "Total records:",
    len(sales_df)
)

print("\nFirst 10 records:")

print(
    sales_df.head(10)
)

print("\nColumns:")

print(
    sales_df.columns.tolist()
)


# ------------------------------------------------------------
# 8. Save dataset
# ------------------------------------------------------------

sales_df.to_csv(
    "ml/sales_history.csv",
    index=False
)


print("\nSales dataset saved successfully!")

print(
    "File: ml/sales_history.csv"
)