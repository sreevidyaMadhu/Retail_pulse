import sys
import os
from datetime import date

# Allow Python to find files from the main project folder
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from app import app, get_sales_features
from models import Product
from weather_service import fetch_weather_by_city
from predict_demand import predict_demand


with app.app_context():

    print("=" * 60)
    print("RetailPulse End-to-End Demand Prediction Test")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Select product
    # --------------------------------------------------

    product_id = 1
    prediction_date = date(2026, 2, 8)

    product = Product.query.get(product_id)

    if not product:
        print("Product not found!")
        exit()

    print("\nProduct")
    print("-" * 40)
    print("Product ID       :", product.id)
    print("Product Name     :", product.name)
    print("Category         :", product.category)
    print("Weather Dependency:", product.weather_dependency)

    # --------------------------------------------------
    # 2. Get real sales features from PostgreSQL
    # --------------------------------------------------

    sales_features = get_sales_features(
        product.id,
        prediction_date
    )

    previous_day_sales = sales_features[
        "previous_day_sales"
    ]

    rolling_7day_sales = sales_features[
        "rolling_7day_sales"
    ]

    print("\nSales Features")
    print("-" * 40)
    print(
        "Previous Day Sales :",
        previous_day_sales
    )
    print(
        "Rolling 7-Day Sales:",
        rolling_7day_sales
    )

    # --------------------------------------------------
    # 3. Get live weather
    # --------------------------------------------------

    weather = fetch_weather_by_city(
        "Thiruvananthapuram"
    )

    if not weather:
        print("\nWeather data unavailable!")
        exit()

    print("\nLive Weather")
    print("-" * 40)
    print("Temperature :", weather["temp"])
    print("Min Temp    :", weather["min_temperature"])
    print("Max Temp    :", weather["max_temperature"])
    print("Humidity    :", weather["humidity"])
    print("Rainfall    :", weather["rainfall"])
    print("Wind Speed  :", weather["wind_speed"])
    print("Pressure    :", weather["pressure"])
    print("Condition   :", weather["condition"])

    # --------------------------------------------------
    # 4. Predict demand
    # --------------------------------------------------

    predicted_demand = predict_demand(

        product_id=f"P{product.id:03d}",

        category=product.category,

        weather_dependency=product.weather_dependency,

        date=prediction_date,

        previous_day_sales=previous_day_sales,

        rolling_7day_sales=rolling_7day_sales,

        temperature=weather["temp"],

        min_temperature=weather["min_temperature"],

        max_temperature=weather["max_temperature"],

        humidity=weather["humidity"],

        rainfall=weather["rainfall"],

        wind_speed=weather["wind_speed"],

        pressure=weather["pressure"]
    )

    # --------------------------------------------------
    # 5. Display result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL PREDICTION")
    print("=" * 60)

    print("Product          :", product.name)
    print("Prediction Date  :", prediction_date)
    print("Predicted Demand :", predicted_demand, "units")

    print("=" * 60)