from tabulate import tabulate  # Optional: if you want to print tabular data in logs
from HVAC_RL_Module import SensorBasedThermalEnv  
from ComparisonModule import ComparisonModule
from stable_baselines3 import SAC
import requests

@app.post("/get_forecast_and_predict")
async def get_forecast_and_predict(request: Request):
    data = await request.json()
    lat = data["latitude"]
    lon = data["longitude"]

    # Weather API call
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    weather_data = response.json()

    forecast = weather_data["list"][0]
    temp = forecast["main"]["temp"]
    wind = forecast["wind"]["speed"]
    humidity = forecast["main"]["humidity"]

    # create dataset from the api call
    # df = ....

    # Setup your environment with weather data
    env = SensorBasedThermalEnv(df)  # assume you have a method to do this
    comparer = ComparisonModule(env)
    model = SAC.load('HVAC_RL_Model.zip')  # Load your trained RL model

    # Run Baselines and RL
    avg_energy_baseline, avg_comfort_baseline = comparer.traditional_hvac_baseline(episodes=1)
    avg_energy_advanced, avg_comfort_advanced = comparer.advanced_hvac_baseline(episodes=1)
    avg_energy_rl, avg_comfort_rl = comparer.evaluate_rl_agent(model, episodes=1)

    # Energy savings %
    savings_vs_traditional = ((avg_energy_baseline - avg_energy_rl) / avg_energy_baseline) * 100
    savings_vs_advanced = ((avg_energy_advanced - avg_energy_rl) / avg_energy_advanced) * 100

    # Optional Logging
    print(tabulate([
        ["Avg Energy Used", avg_energy_baseline, avg_energy_advanced, avg_energy_rl],
        ["Avg Comfort Penalty", avg_comfort_baseline, avg_comfort_advanced, avg_comfort_rl]
    ], headers=["Metric", "Traditional", "Advanced", "RL"], tablefmt="grid"))

    return {
        "temp": temp,
        "wind": wind,
        "humidity": humidity,
        "results": {
            "traditional": {"energy": avg_energy_baseline, "comfort": avg_comfort_baseline},
            "advanced": {"energy": avg_energy_advanced, "comfort": avg_comfort_advanced},
            "rl": {"energy": avg_energy_rl, "comfort": avg_comfort_rl},
            "savings_vs_traditional": savings_vs_traditional,
            "savings_vs_advanced": savings_vs_advanced
        }
    }
