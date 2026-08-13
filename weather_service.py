import requests

# OpenWeatherMap API Key
# (Using fallback mock mode if key is default)
API_KEY = "c2bd5cb2c0ab9c420634537aa5016d5f"

def fetch_weather_by_city(city_name):
    """
    Fetch live weather metrics for a given city.
    Returns temperature, humidity, rainfall, and a simplified condition.
    """
    if API_KEY == "c2bd5cb2c0ab9c420634537aa5016d5f":
        # Fallback mock data so you can test without an API key right away
        return {
            "city": city_name,
            "temp": 32.5,
            "humidity": 75,
            "rainfall": 0.0,
            "condition": "Hot",
            "description": "clear sky"
        }

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            rain_mm = data.get("rain", {}).get("1h", 0.0)
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            main_condition = data["weather"][0]["main"]
            
            # Categorize condition for inventory logic
            if temp >= 30:
                simplified_condition = "Hot"
            elif rain_mm > 0 or main_condition in ["Rain", "Drizzle", "Thunderstorm"]:
                simplified_condition = "Rainy"
            elif temp <= 20:
                simplified_condition = "Cold"
            else:
                simplified_condition = "Normal"

            return {
                "city": data["name"],
                "temp": temp,
                "humidity": humidity,
                "rainfall": rain_mm,
                "condition": simplified_condition,
                "description": data["weather"][0]["description"]
            }
        else:
            print(f"Weather API Error: Status Code {response.status_code}")
            return None
    except Exception as e:
        print(f"Failed to connect to Weather API: {e}")
        return None

# Test script locally
if __name__ == "__main__":
    test_weather = fetch_weather_by_city("Thiruvananthapuram")
    print("Fetched Weather:", test_weather)