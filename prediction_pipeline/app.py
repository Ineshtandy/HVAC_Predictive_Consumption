from tabulate import tabulate  # Optional: if you want to print tabular data in logs
from HVAC_RL_Module import SensorBasedThermalEnv  
from ComparisonModule import ComparisonModule
from stable_baselines3 import SAC
import requests
import pandas as pd

# lat = data["latitude"]
lat = 33.44
# lon = data["longitude"]
lon = -94.04
OPENWEATHER_API_KEY = "dc5d0c3993fc54fd5e9669c076a608cb"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any frontend (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/get_forecast_and_predict")
async def get_forecast_and_predict(request: Request):
    data = await request.json()
    lat = data["latitude"]
    lon = data["longitude"]
    OPENWEATHER_API_KEY = "dc5d0c3993fc54fd5e9669c076a608cb"

    # Weather API call
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&exclude=daily&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    weather_data = response.json()

    current = weather_data["list"][0]
    next = weather_data["list"][1]

    current["main"]["temp"] = (current["main"]["temp"] - 273.15) * 1.8 + 32
    current = {
        "main": {"temp": current["main"]["temp"], "humidity": current["main"]["humidity"]},
        "wind": {"speed": current["wind"]["speed"]}
    }

    next["main"]["temp"] = (next["main"]["temp"] - 273.15) * 1.8 + 32
    next_list = [
        {"main": {"temp": next["main"]["temp"], "humidity": next["main"]["humidity"]}, "wind": {"speed": next["wind"]["speed"]}},
        {"main": {"temp": next["main"]["temp"], "humidity": next["main"]["humidity"]}, "wind": {"speed": next["wind"]["speed"]}},
        {"main": {"temp": next["main"]["temp"], "humidity": next["main"]["humidity"]}, "wind": {"speed": next["wind"]["speed"]}},
        {"main": {"temp": next["main"]["temp"], "humidity": next["main"]["humidity"]}, "wind": {"speed": next["wind"]["speed"]}},
        {"main": {"temp": next["main"]["temp"], "humidity": next["main"]["humidity"]}, "wind": {"speed": next["wind"]["speed"]}},
        {"main": {"temp": next["main"]["temp"], "humidity": next["main"]["humidity"]}, "wind": {"speed": next["wind"]["speed"]}},
        {"main": {"temp": next["main"]["temp"], "humidity": next["main"]["humidity"]}, "wind": {"speed": next["wind"]["speed"]}},
        {"main": {"temp": next["main"]["temp"], "humidity": next["main"]["humidity"]}, "wind": {"speed": next["wind"]["speed"]}}
    ]

    # Start by creating the first row from current
    rows = [[
        current["main"]["temp"],
        current["main"]["humidity"],
        current["wind"]["speed"]
    ]]

    # Now add the next 8 rows
    for entry in next_list:
        row = [
            entry["main"]["temp"],
            entry["main"]["humidity"],
            entry["wind"]["speed"]
        ]
        rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(rows, columns=["temp", "humidity", "wind_speed"])

    # Setup your environment with weather data
    env = SensorBasedThermalEnv(df)  # assume you have a method to do this
    comparer = ComparisonModule(df, env)
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
        "temp": current["main"]["temp"],
        "wind": current["wind"]["speed"],
        "humidity": current["main"]["humidity"],
        "results": {
            "traditional": {"energy": avg_energy_baseline, "comfort": avg_comfort_baseline},
            "advanced": {"energy": avg_energy_advanced, "comfort": avg_comfort_advanced},
            "rl": {"energy": avg_energy_rl, "comfort": avg_comfort_rl},
            "savings_vs_traditional": savings_vs_traditional,
            "savings_vs_advanced": savings_vs_advanced
        }
    }
