import httpx
import os

API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "your_default_api_key_here")

async def get_weather_data(location: str):
    """
    Pings the OpenWeatherMap API using the extracted location.
    """
    if not location or location == "Unknown":
        return None
        
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    "temperature_celsius": data["main"]["temp"],
                    "humidity_percent": data["main"]["humidity"],
                    "conditions": data["weather"][0]["description"],
                    "wind_speed_ms": data["wind"]["speed"]
                }
            else:
                return {"error": f"Failed to fetch weather. Status: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
