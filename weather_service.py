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


# ---------------------------------
# TEST WEATHER SERVICE
# ---------------------------------

if __name__ == "__main__":

    print("=" * 50)

    print("Testing RetailPulse Weather Service")

    print("=" * 50)

    weather = fetch_weather_by_city(
        "Thiruvananthapuram"
    )

    if weather:

        print("\nWeather Data")
        print("------------------------------")

        for key, value in weather.items():

            print(
                f"{key:20}: {value}"
            )

    else:

        print("\nWeather data could not be fetched.")