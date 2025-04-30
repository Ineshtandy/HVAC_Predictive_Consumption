# 🔥 ClimaSense: Predictive HVAC Control using Reinforcement Learning
ClimaSense is a reinforcement learning-based HVAC controller that anticipates future weather conditions to optimize indoor comfort and reduce energy consumption. Instead of reacting to real-time temperatures, it learns to act in advance using 2-hour-ahead weather forecasts.

## 🧠 Project Motivation
Traditional HVAC systems are reactive: they wait for discomfort to occur before acting. This leads to inefficient energy use and delayed response in dynamic environments.

ClimaSense solves this by:

🔄 Using reinforcement learning (Soft Actor-Critic) to learn efficient control strategies

⏱️ Incorporating future outdoor temperature forecasts to plan actions in advance

💡 Achieving up to 15% energy savings and 27% improvement in comfort

## 🏗️ Architecture Overview
Environment: SensorBasedThermalEnv built with Gymnasium

Observations: [T_in, T_out, T_out_future, humidity, wind speed, timestep]

Action: Continuous HVAC power level (0.0 to 1.0)

Reward: - comfort_penalty - 0.1 * energy_used

RL Agent: Soft Actor-Critic (SAC) from Stable-Baselines3

Data: Real hourly weather data (~45,000 rows) from multiple cities (Vancouver, Chicago, Miami)
