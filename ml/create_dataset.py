import pandas as pd

# ---------------------------------------------------
# RETAILPULSE - PRODUCT MASTER DATA
# ---------------------------------------------------

products = [

    # Beverages
    ["P001", "Coca Cola", "Beverages", "High"],
    ["P002", "Pepsi", "Beverages", "High"],
    ["P003", "Sprite", "Beverages", "High"],
    ["P004", "Fruit Juice", "Beverages", "High"],
    ["P005", "Packaged Drinking Water", "Beverages", "High"],
    ["P006", "Energy Drink", "Beverages", "High"],
    ["P007", "Lemon Juice", "Beverages", "High"],
    ["P008", "Flavoured Milk", "Beverages", "Medium"],

    # Ice Cream & Frozen
    ["P009", "Vanilla Ice Cream", "Ice Cream & Frozen", "Very High"],
    ["P010", "Chocolate Ice Cream", "Ice Cream & Frozen", "Very High"],
    ["P011", "Strawberry Ice Cream", "Ice Cream & Frozen", "Very High"],
    ["P012", "Kulfi", "Ice Cream & Frozen", "Very High"],
    ["P013", "Ice Cream Cone", "Ice Cream & Frozen", "Very High"],
    ["P014", "Frozen Dessert", "Ice Cream & Frozen", "High"],

    # Fruits
    ["P015", "Watermelon", "Fruits", "Very High"],
    ["P016", "Mango", "Fruits", "High"],
    ["P017", "Orange", "Fruits", "Medium"],
    ["P018", "Grapes", "Fruits", "Medium"],
    ["P019", "Pineapple", "Fruits", "Medium"],
    ["P020", "Banana", "Fruits", "Low"],
    ["P021", "Papaya", "Fruits", "Low"],
    ["P022", "Apple", "Fruits", "Low"],

    # Vegetables
    ["P023", "Tomato", "Vegetables", "Medium"],
    ["P024", "Cucumber", "Vegetables", "High"],
    ["P025", "Carrot", "Vegetables", "Low"],
    ["P026", "Potato", "Vegetables", "Very Low"],
    ["P027", "Onion", "Vegetables", "Very Low"],
    ["P028", "Green Chilli", "Vegetables", "Low"],
    ["P029", "Spinach", "Vegetables", "Medium"],
    ["P030", "Capsicum", "Vegetables", "Low"],

    # Dairy
    ["P031", "Milk", "Dairy", "Low"],
    ["P032", "Curd", "Dairy", "Medium"],
    ["P033", "Butter", "Dairy", "Low"],
    ["P034", "Cheese", "Dairy", "Low"],
    ["P035", "Paneer", "Dairy", "Low"],
    ["P036", "Buttermilk", "Dairy", "High"],
    ["P037", "Flavoured Yogurt", "Dairy", "Medium"],
    ["P038", "Fresh Cream", "Dairy", "Low"],

    # Bakery
    ["P039", "Bread", "Bakery", "Low"],
    ["P040", "Bun", "Bakery", "Low"],
    ["P041", "Burger Bun", "Bakery", "Low"],
    ["P042", "Cake", "Bakery", "Medium"],
    ["P043", "Rusk", "Bakery", "Low"],
    ["P044", "Biscuits", "Bakery", "Low"],
    ["P045", "Croissant", "Bakery", "Low"],
    ["P046", "Bread Rolls", "Bakery", "Low"],

    # Snacks & Chocolates
    ["P047", "Potato Chips", "Snacks & Chocolates", "Medium"],
    ["P048", "Mixture", "Snacks & Chocolates", "Low"],
    ["P049", "Namkeen", "Snacks & Chocolates", "Low"],
    ["P050", "Popcorn", "Snacks & Chocolates", "Low"],
    ["P051", "Chocolate", "Snacks & Chocolates", "Medium"],
    ["P052", "Candy", "Snacks & Chocolates", "Low"],
    ["P053", "Nuts", "Snacks & Chocolates", "Low"],
    ["P054", "Instant Snack Mix", "Snacks & Chocolates", "Low"],

    # Ready-to-Eat & Convenience
    ["P055", "Instant Noodles", "Ready-to-Eat & Convenience", "Medium"],
    ["P056", "Soup Mix", "Ready-to-Eat & Convenience", "Medium"],
    ["P057", "Pasta", "Ready-to-Eat & Convenience", "Low"],
    ["P058", "Ready-to-Eat Meals", "Ready-to-Eat & Convenience", "Medium"],
    ["P059", "Canned Food", "Ready-to-Eat & Convenience", "Low"],
    ["P060", "Breakfast Cereal", "Ready-to-Eat & Convenience", "Low"],

    # Monsoon / Weather-related
    ["P061", "Umbrella", "Weather-related Household", "Very High"],
    ["P062", "Raincoat", "Weather-related Household", "Very High"],
    ["P063", "Waterproof Shoe Cover", "Weather-related Household", "High"],
    ["P064", "Candles", "Weather-related Household", "Medium"],
    ["P065", "Batteries", "Weather-related Household", "Medium"],
    ["P066", "Mosquito Repellent", "Weather-related Household", "High"],

    # Personal & Household Care
    ["P067", "Soap", "Personal & Household Care", "Low"],
    ["P068", "Shampoo", "Personal & Household Care", "Medium"],
    ["P069", "Conditioner", "Personal & Household Care", "Medium"],
    ["P070", "Toothpaste", "Personal & Household Care", "Very Low"],
    ["P071", "Detergent", "Personal & Household Care", "Low"],
    ["P072", "Dishwashing Liquid", "Personal & Household Care", "Low"],
    ["P073", "Floor Cleaner", "Personal & Household Care", "Low"],

    # Grocery Staples
    ["P074", "Rice", "Grocery Staples", "Very Low"],
    ["P075", "Wheat Flour", "Grocery Staples", "Very Low"],
    ["P076", "Sugar", "Grocery Staples", "Very Low"],
    ["P077", "Salt", "Grocery Staples", "Very Low"],
    ["P078", "Cooking Oil", "Grocery Staples", "Very Low"],
    ["P079", "Atta", "Grocery Staples", "Very Low"],
    ["P080", "Spices", "Grocery Staples", "Very Low"]
]


# ---------------------------------------------------
# CREATE PRODUCT DATAFRAME
# ---------------------------------------------------

columns = [
    "product_id",
    "product_name",
    "category",
    "weather_dependency"
]

df_products = pd.DataFrame(products, columns=columns)


# ---------------------------------------------------
# ADD SHOP INFORMATION
# ---------------------------------------------------

df_products["shop_id"] = "S001"
df_products["city"] = "Thiruvananthapuram"


# ---------------------------------------------------
# DISPLAY INFORMATION
# ---------------------------------------------------

print("Total products:", len(df_products))

print("\nFirst 10 products:")
print(df_products.head(10))


# ---------------------------------------------------
# SAVE PRODUCT MASTER
# ---------------------------------------------------

df_products.to_csv("product_master.csv", index=False)

print("\nProduct master saved successfully!")



import pandas as pd

from meteostat import Point, stations, daily
from datetime import datetime


# ============================================================
# HISTORICAL WEATHER DATASET
# Location: Thiruvananthapuram
# Year: 2025
# ============================================================


print("\nFetching historical weather data...")
print("Location: Thiruvananthapuram")
print("Period: 2025-01-01 to 2025-12-31")


# ------------------------------------------------------------
# 1. Thiruvananthapuram location
# ------------------------------------------------------------

latitude = 8.5241
longitude = 76.9366

location = Point(
    latitude,
    longitude
)


# ------------------------------------------------------------
# 2. Find nearby weather stations
# ------------------------------------------------------------

nearby_stations = stations.nearby(
    location,
    limit=2
)

print("\nNearby weather stations:")
print(nearby_stations)


# ------------------------------------------------------------
# 3. Set date range
# ------------------------------------------------------------

start = datetime(2025, 1, 1)
end = datetime(2025, 12, 31)


# ------------------------------------------------------------
# 4. Fetch daily historical weather
# ------------------------------------------------------------

weather = daily(
    nearby_stations,
    start,
    end
)


# ------------------------------------------------------------
# 5. Convert Meteostat TimeSeries to DataFrame
# ------------------------------------------------------------

weather_df = weather.fetch().reset_index()

print("\nRaw weather records:", len(weather_df))


# ------------------------------------------------------------
# 6. Select one Thiruvananthapuram weather station
# ------------------------------------------------------------

# Station ID is stored as a string in the current
# Meteostat package.

weather_df = weather_df[
    weather_df["station"] == "43371"
].copy()


# ------------------------------------------------------------
# 7. Select required weather columns
# ------------------------------------------------------------

weather_df = weather_df[
    [
        "time",
        "temp",
        "tmin",
        "tmax",
        "rhum",
        "prcp",
        "wspd",
        "pres"
    ]
].copy()


# ------------------------------------------------------------
# 8. Rename columns
# ------------------------------------------------------------

weather_df.rename(
    columns={
        "time": "date",
        "temp": "temperature",
        "tmin": "min_temperature",
        "tmax": "max_temperature",
        "rhum": "humidity",
        "prcp": "rainfall",
        "wspd": "wind_speed",
        "pres": "pressure"
    },
    inplace=True
)


# ------------------------------------------------------------
# 9. Add location/shop information
# ------------------------------------------------------------

weather_df["shop_id"] = "S001"
weather_df["city"] = "Thiruvananthapuram"


# ------------------------------------------------------------
# 10. Handle missing rainfall values
# ------------------------------------------------------------

weather_df["rainfall"] = weather_df[
    "rainfall"
].fillna(0)


# ------------------------------------------------------------
# 11. Handle missing weather values
# ------------------------------------------------------------

weather_columns = [
    "temperature",
    "min_temperature",
    "max_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure"
]

for column in weather_columns:
    weather_df[column] = pd.to_numeric(
        weather_df[column],
        errors="coerce"
    )


# Fill missing values using the previous available value
weather_df[weather_columns] = weather_df[
    weather_columns
].ffill()


# If the first value is missing, fill it using the next value
weather_df[weather_columns] = weather_df[
    weather_columns
].bfill()


# ------------------------------------------------------------
# 12. Sort by date
# ------------------------------------------------------------

weather_df.sort_values(
    by="date",
    inplace=True
)

weather_df.reset_index(
    drop=True,
    inplace=True
)


# ------------------------------------------------------------
# 13. Display final dataset information
# ------------------------------------------------------------

print("\nClean weather records:", len(weather_df))

print("\nFinal weather dataset:")
print(weather_df.head(10))

print("\nFinal columns:")
print(weather_df.columns.tolist())

print("\nMissing values:")
print(weather_df.isnull().sum())


# ------------------------------------------------------------
# 14. Save the weather dataset
# ------------------------------------------------------------

weather_df.to_csv(
    "weather_history.csv",
    index=False
)

print("\nWeather dataset saved successfully!")
print("File: weather_history.csv")