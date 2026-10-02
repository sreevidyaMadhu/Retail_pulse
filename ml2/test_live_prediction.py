import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from weather_service import fetch_weather_by_city
from predict_demand import predict_demand


print("=" * 60)
print("RetailPulse Live Weather + ML Prediction Test")
print("=" * 60)


# -----------------------------------------
# 1. Get live weather
# -----------------------------------------

weather = fetch_weather_by_city(
    "Thiruvananthapuram"
)


if weather is None:

    print("Weather data could not be fetched.")
    exit()


print("\nLive Weather")
print("-" * 40)

print("Temperature     :", weather["temp"])
print("Min Temperature :", weather["min_temperature"])
print("Max Temperature :", weather["max_temperature"])
print("Humidity        :", weather["humidity"])
print("Rainfall        :", weather["rainfall"])
print("Wind Speed      :", weather["wind_speed"])
print("Pressure        :", weather["pressure"])
print("Condition       :", weather["condition"])


# -----------------------------------------
# 2. Test ML prediction
# -----------------------------------------

predicted_demand = predict_demand(

    product_id="P001",

    category="Beverages",

    weather_dependency="High",

    date="2026-02-08",

    # Temporary values for testing
    previous_day_sales=25,

    rolling_7day_sales=27.5,

    # Live weather
    temperature=weather["temp"],

    min_temperature=weather["min_temperature"],

    max_temperature=weather["max_temperature"],

    humidity=weather["humidity"],

    rainfall=weather["rainfall"],

    wind_speed=weather["wind_speed"],

    pressure=weather["pressure"]
)


# -----------------------------------------
# 3. Display prediction
# -----------------------------------------

print("\nPrediction")
print("-" * 40)

print(
    "Predicted Demand:",
    predicted_demand,
    "units"
)

print("=" * 60)