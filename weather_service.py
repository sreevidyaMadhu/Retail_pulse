import os
import requests
from dotenv import load_dotenv

load_dotenv()

# OpenWeatherMap API key
API_KEY = os.getenv("OPENWEATHER_API_KEY")


def fetch_weather_by_city(city_name):
    """
    Fetch current weather information for a city.

    Returns the weather features required by the
    RetailPulse demand prediction model.
    """

    # Temporary mock mode if API key is not configured
    if not API_KEY:
        return {
            "city": city_name,
            "temp": 32.5,
            "min_temperature": 27.0,
            "max_temperature": 34.0,
            "humidity": 75,
            "rainfall": 0.0,
            "wind_speed": 6.0,
            "pressure": 1012.0,
            "condition": "Hot",
            "description": "clear sky"
        }

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city_name}"
        f"&appid={API_KEY}"
        "&units=metric"
    )

    try:

        response = requests.get(
            url,
            timeout=5
        )

        if response.status_code != 200:

            print(
                f"Weather API Error: "
                f"Status Code {response.status_code}"
            )

            return None

        data = response.json()

        # -------------------------
        # WEATHER VALUES
        # -------------------------

        temp = data["main"]["temp"]

        min_temperature = data["main"]["temp_min"]

        max_temperature = data["main"]["temp_max"]

        humidity = data["main"]["humidity"]

        pressure = data["main"]["pressure"]

        wind_speed = data.get(
            "wind",
            {}
        ).get(
            "speed",
            0.0
        )

        # -------------------------
        # RAINFALL
        # -------------------------

        rainfall = data.get(
            "rain",
            {}
        ).get(
            "1h",
            0.0
        )

        # -------------------------
        # WEATHER CONDITION
        # -------------------------

        main_condition = data["weather"][0]["main"]

        description = data["weather"][0]["description"]

        if temp >= 30:

            simplified_condition = "Hot"

        elif (
            rainfall > 0
            or main_condition in [
                "Rain",
                "Drizzle",
                "Thunderstorm"
            ]
        ):

            simplified_condition = "Rainy"

        elif temp <= 20:

            simplified_condition = "Cold"

        else:

            simplified_condition = "Normal"

        # -------------------------
        # RETURN DATA
        # -------------------------

        return {

            "city": data["name"],

            "temp": temp,

            "min_temperature": min_temperature,

            "max_temperature": max_temperature,

            "humidity": humidity,

            "rainfall": rainfall,

            "wind_speed": wind_speed,

            "pressure": pressure,

            "condition": simplified_condition,

            "description": description
        }

    except Exception as e:

        print(
            f"Failed to connect to Weather API: {e}"
        )

        return None


# # ---------------------------------
# # TEST WEATHER SERVICE
# # ---------------------------------

# if __name__ == "__main__":

#     print("=" * 50)

#     print("Testing RetailPulse Weather Service")

#     print("=" * 50)

#     weather = fetch_weather_by_city(
#         "Thiruvananthapuram"
#     )

#     if weather:

#         print("\nWeather Data")
#         print("------------------------------")

#         for key, value in weather.items():

#             print(
#                 f"{key:20}: {value}"
#             )

#     else:

#         print("\nWeather data could not be fetched.")
def fetch_weather_forecast(city_name):
    """
    Fetch 5-day weather forecast for a city.

    OpenWeather provides forecast data at 3-hour intervals.
    This function converts that data into daily weather
    features required by the RetailPulse demand model.
    """

    # Temporary mock mode if API key is not configured
    if not API_KEY:
        print("Weather API key not configured.")
        return []

    url = (
        "https://api.openweathermap.org/data/2.5/forecast"
        f"?q={city_name}"
        f"&appid={API_KEY}"
        "&units=metric"
    )

    try:

        response = requests.get(
            url,
            timeout=10
        )

        if response.status_code != 200:

            print(
                f"Weather Forecast API Error: "
                f"Status Code {response.status_code}"
            )

            return []

        data = response.json()

        # ---------------------------------
        # GROUP 3-HOUR FORECASTS BY DATE
        # ---------------------------------

        daily_data = {}

        for item in data["list"]:

            date = item["dt_txt"].split(" ")[0]

            if date not in daily_data:
                daily_data[date] = []

            daily_data[date].append(item)

        # ---------------------------------
        # CONVERT TO DAILY WEATHER
        # ---------------------------------

        forecast = []

        for date, items in daily_data.items():

            temperatures = []
            humidity_values = []
            wind_values = []
            pressure_values = []
            rainfall_values = []

            for item in items:

                temperatures.append(
                    item["main"]["temp"]
                )

                humidity_values.append(
                    item["main"]["humidity"]
                )

                wind_values.append(
                    item.get(
                        "wind",
                        {}
                    ).get(
                        "speed",
                        0.0
                    )
                )

                pressure_values.append(
                    item["main"]["pressure"]
                )

                rainfall_values.append(
                    item.get(
                        "rain",
                        {}
                    ).get(
                        "3h",
                        0.0
                    )
                )

            # ---------------------------------
            # DAILY VALUES
            # ---------------------------------

            avg_temperature = (
                sum(temperatures)
                / len(temperatures)
            )

            min_temperature = min(
                temperatures
            )

            max_temperature = max(
                temperatures
            )

            avg_humidity = (
                sum(humidity_values)
                / len(humidity_values)
            )

            avg_wind_speed = (
                sum(wind_values)
                / len(wind_values)
            )

            avg_pressure = (
                sum(pressure_values)
                / len(pressure_values)
            )

            total_rainfall = sum(
                rainfall_values
            )

            # ---------------------------------
            # WEATHER CONDITION
            # ---------------------------------

            if total_rainfall > 0:
                condition = "Rainy"

            elif avg_temperature >= 30:
                condition = "Hot"

            elif avg_temperature <= 20:
                condition = "Cold"

            else:
                condition = "Normal"

            forecast.append({

                "date": date,

                "temp": round(
                    avg_temperature,
                    2
                ),

                "min_temperature": round(
                    min_temperature,
                    2
                ),

                "max_temperature": round(
                    max_temperature,
                    2
                ),

                "humidity": round(
                    avg_humidity,
                    2
                ),

                "rainfall": round(
                    total_rainfall,
                    2
                ),

                "wind_speed": round(
                    avg_wind_speed,
                    2
                ),

                "pressure": round(
                    avg_pressure,
                    2
                ),

                "condition": condition
            })

        # ---------------------------------
        # RETURN FIRST 5 DAYS
        # ---------------------------------

        return forecast[:5]

    except Exception as e:

        print(
            f"Failed to fetch weather forecast: {e}"
        )

        return []
if __name__ == "__main__":

    print("=" * 60)
    print("Testing RetailPulse Weather Forecast")
    print("=" * 60)

    forecast = fetch_weather_forecast(
        "Thiruvananthapuram"
    )

    if forecast:

        print("\n5-Day Weather Forecast")
        print("-" * 60)

        for day in forecast:

            print(
                f"\nDate        : {day['date']}"
            )

            print(
                f"Temperature : {day['temp']} °C"
            )

            print(
                f"Min Temp    : {day['min_temperature']} °C"
            )

            print(
                f"Max Temp    : {day['max_temperature']} °C"
            )

            print(
                f"Humidity    : {day['humidity']} %"
            )

            print(
                f"Rainfall    : {day['rainfall']} mm"
            )

            print(
                f"Wind Speed  : {day['wind_speed']} m/s"
            )

            print(
                f"Pressure    : {day['pressure']} hPa"
            )

            print(
                f"Condition   : {day['condition']}"
            )

    else:

        print(
            "\nWeather forecast could not be fetched."
        )