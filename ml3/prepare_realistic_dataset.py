import os
import json
import numpy as np
import pandas as pd

# ============================================================
# RetailPulse ML3 - Realistic Retail Dataset Generator
# Produces high-fidelity demand aligned with real supermarket dynamics:
# - Realistic product baseline velocities (FMCG vs. Luxury vs. Perishables)
# - Price elasticity calibration
# - Autoregressive daily continuity (t-1, t-7)
# - Category-specific weather response curves
# - Realistic variance (noise ~ 10% of mean demand, R² ~ 0.80 - 0.88)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

PRODUCT_MASTER_PATH = os.path.join(PROJECT_DIR, "ml", "product_master.csv")
WEATHER_HISTORY_PATH = os.path.join(PROJECT_DIR, "ml2", "weather_history_2year.csv")
PRICES_PATH = os.path.join(PROJECT_DIR, "ml2", "product_prices.json")

SALES_OUT_PATH = os.path.join(BASE_DIR, "sales_history_realistic.csv")
TRAIN_DATA_OUT_PATH = os.path.join(BASE_DIR, "ml_training_data_realistic.csv")
SUMMARY_OUT_PATH = os.path.join(BASE_DIR, "dataset_summary.json")

print("=" * 60)
print("RetailPulse ML3: Realistic Supermarket Dataset Generation")
print("=" * 60)

# 1. Load inputs
products_df = pd.read_csv(PRODUCT_MASTER_PATH)
weather_df = pd.read_csv(WEATHER_HISTORY_PATH)
weather_df["date"] = pd.to_datetime(weather_df["date"])
weather_df.sort_values("date", inplace=True)
weather_df.reset_index(drop=True, inplace=True)
weather_df["rainfall"] = weather_df["rainfall"].fillna(0.0)

with open(PRICES_PATH, "r") as f:
    prices = json.load(f)

products_df["price"] = products_df["product_name"].map(prices).fillna(50.0)

# 2. Define Category Base Velocities (Mean Daily Units Sold)
CATEGORY_BASE_VELOCITY = {
    "Grocery Staples": 85.0,            # High volume daily pantry essentials
    "Dairy": 75.0,                      # Fresh daily perishable staples
    "Bakery": 60.0,                     # Fast-moving fresh bread/buns
    "Vegetables": 55.0,                 # Daily cooking produce
    "Fruits": 45.0,                     # Healthy fresh produce
    "Beverages": 42.0,                  # Refreshments & juices
    "Snacks & Chocolates": 38.0,        # Packaged consumer goods
    "Ice Cream & Frozen": 28.0,         # Frozen treats
    "Ready-to-Eat & Convenience": 24.0, # Instant foods
    "Personal & Household Care": 18.0,  # Cleaning / toiletry replenishments
    "Weather-related Household": 6.0    # Rain gear / seasonal household
}

# Day-of-week multipliers (Grocery footfall peaks Fri-Sun)
DOW_MULTIPLIERS = {
    0: 0.92, # Monday
    1: 0.88, # Tuesday (lowest grocery footfall)
    2: 0.93, # Wednesday
    3: 0.97, # Thursday
    4: 1.15, # Friday (weekend shopping start)
    5: 1.38, # Saturday (peak family shopping)
    6: 1.28  # Sunday (meal prep & restocking)
}

# Monthly seasonal multipliers
MONTH_MULTIPLIERS = {
    1: 0.96, 2: 0.98, 3: 1.04, 4: 1.08, 5: 1.12, 6: 1.06,
    7: 1.02, 8: 1.01, 9: 1.03, 10: 1.09, 11: 1.14, 12: 1.18
}

np.random.seed(42)

print(f"Generating realistic sales across {len(products_df)} products and {len(weather_df)} dates...")
sales_records = []

for _, prod in products_df.iterrows():
    pid = prod["product_id"]
    pname = prod["product_name"]
    cat = prod["category"]
    w_dep = prod["weather_dependency"]
    shop_id = prod["shop_id"]
    price = prod["price"]

    # Product baseline velocity
    cat_base = CATEGORY_BASE_VELOCITY.get(cat, 35.0)
    # Price elasticity adjustment: cheaper items sell higher quantities
    price_adj = np.clip((60.0 / max(price, 10.0)) ** 0.35, 0.45, 1.85)
    prod_base = cat_base * price_adj

    # Item specific variation factor
    item_factor = np.random.uniform(0.85, 1.15)
    mean_demand = prod_base * item_factor

    prev_demand = mean_demand

    for _, w_row in weather_df.iterrows():
        dt = w_row["date"]
        dow = dt.dayofweek
        mo = dt.month
        tmax = w_row["max_temperature"]
        temp = w_row["temperature"]
        rain = w_row["rainfall"]
        hum = w_row["humidity"]
        wspd = w_row["wind_speed"]
        pres = w_row["pressure"]

        # Base expectation with calendar effects
        expected = mean_demand * DOW_MULTIPLIERS[dow] * MONTH_MULTIPLIERS[mo]

        # Weather impacts based on category and weather dependency
        if cat in ["Beverages", "Ice Cream & Frozen"] or w_dep in ["Very High", "High"]:
            if tmax >= 33.0:
                expected *= 1.45
            elif tmax >= 30.0:
                expected *= 1.25
            elif tmax >= 28.0:
                expected *= 1.10

        if pname in ["Umbrella", "Raincoat", "Waterproof Shoe Cover"]:
            if rain >= 20.0:
                expected = max(expected, 45.0 + np.random.uniform(5, 15))
            elif rain >= 8.0:
                expected = max(expected, 28.0 + np.random.uniform(3, 8))
            elif rain >= 2.0:
                expected = max(expected, 14.0 + np.random.uniform(1, 5))
            else:
                expected *= 0.40 # Low sales on dry sunny days
        elif rain >= 15.0 and cat not in ["Grocery Staples", "Dairy"]:
            # Extreme rain slightly depresses casual shopping trips
            expected *= 0.88

        # Humidity boost for cold drinks / refreshers
        if hum >= 85 and cat in ["Beverages", "Ice Cream & Frozen"]:
            expected *= 1.08

        # Autoregressive momentum (yesterday's sales carry forward ~ 18%)
        expected = 0.82 * expected + 0.18 * prev_demand

        # Realistic noise: ~8% - 10% standard deviation (instead of 70% white noise)
        noise_std = max(1.2, expected * 0.08)
        actual_demand = np.random.normal(expected, noise_std)
        actual_demand = max(1, int(round(actual_demand)))

        prev_demand = actual_demand

        sales_records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "shop_id": shop_id,
            "product_id": pid,
            "product_name": pname,
            "category": cat,
            "weather_dependency": w_dep,
            "temperature": temp,
            "min_temperature": w_row["min_temperature"],
            "max_temperature": tmax,
            "humidity": hum,
            "rainfall": rain,
            "wind_speed": wspd,
            "pressure": pres,
            "demand": actual_demand,
            "price": price
        })

sales_df = pd.DataFrame(sales_records)
sales_df["date"] = pd.to_datetime(sales_df["date"])
sales_df.sort_values(by=["product_id", "date"], inplace=True)
sales_df.reset_index(drop=True, inplace=True)

# Save intermediate sales history
sales_df.to_csv(SALES_OUT_PATH, index=False)
print(f"Realistic sales history saved to: {SALES_OUT_PATH} ({len(sales_df)} records)")

# 3. Create Feature Engineering Pipeline (matching RetailPulse exact schema)
print("Creating feature engineered ML training dataset...")
sales_df["day_of_week"] = sales_df["date"].dt.dayofweek
sales_df["month"] = sales_df["date"].dt.month
sales_df["day"] = sales_df["date"].dt.day
sales_df["is_weekend"] = sales_df["day_of_week"].isin([5, 6]).astype(int)

sales_df["previous_day_sales"] = sales_df.groupby("product_id")["demand"].shift(1)
sales_df["lag_7day_sales"] = sales_df.groupby("product_id")["demand"].shift(7)
sales_df["rolling_7day_sales"] = sales_df.groupby("product_id")["demand"].transform(
    lambda x: x.shift(1).rolling(window=7, min_periods=7).mean()
).round(2)

# Drop initial 7-day warmup period
ml_clean = sales_df.dropna(subset=["previous_day_sales", "lag_7day_sales", "rolling_7day_sales", "price"]).copy()
ml_clean.reset_index(drop=True, inplace=True)

# Ensure exact column order matching RetailPulse
EXACT_COLUMNS = [
    "date", "shop_id", "product_id", "product_name", "category", "weather_dependency",
    "temperature", "min_temperature", "max_temperature", "humidity", "rainfall",
    "wind_speed", "pressure", "demand", "day_of_week", "month", "day", "is_weekend",
    "previous_day_sales", "lag_7day_sales", "rolling_7day_sales", "price"
]

ml_clean = ml_clean[EXACT_COLUMNS]
ml_clean.to_csv(TRAIN_DATA_OUT_PATH, index=False)
print(f"ML Training Dataset saved to: {TRAIN_DATA_OUT_PATH} ({len(ml_clean)} records)")

# Summary stats
summary = {
    "total_records": len(ml_clean),
    "date_range": {
        "start": str(ml_clean["date"].min().date()),
        "end": str(ml_clean["date"].max().date())
    },
    "num_products": int(ml_clean["product_id"].nunique()),
    "demand_stats": {
        "mean": float(ml_clean["demand"].mean()),
        "std": float(ml_clean["demand"].std()),
        "min": int(ml_clean["demand"].min()),
        "max": int(ml_clean["demand"].max()),
        "total_variance": float(ml_clean["demand"].var())
    },
    "columns": EXACT_COLUMNS
}

with open(SUMMARY_OUT_PATH, "w") as f:
    json.dump(summary, f, indent=4)

print("\nDataset Summary:")
print(json.dumps(summary, indent=2))
print("=" * 60)
