import gymnasium as gym  
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env  

class SensorBasedThermalEnv(gym.Env):
    """
    Updated HVAC environment:
    - All temperatures in Fahrenheit
    - T_in between 65°F and 75°F
    - Simulated sensor-like indoor temp readings
    """
    def __init__(self, df):
        super(SensorBasedThermalEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.max_time_index = len(self.df) - 9  # 8 steps ahead needed for future

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32) # defines the valid actions the agent can take in the environment
        # Action: [HVAC_power] -> 0.0 (off) to 1.0 (max power) : this is irrespective of cooling/heating

        # Observation: [T_in, T_out, T_out_future, Humid, Wind_sp, Time_step]
        low_obs = np.array([60.0, -10.0, -10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([90.0, 120.0, 120.0, 100.0, 50.0, 7.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.energy_rate = 1.0
        self.episode_length = 8  # 2 hours (assuming 15-min steps)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)
        """
        Reset environment
        """
        self.start_idx = 0 # picking up the first index
        self.current_idx = self.start_idx

        self.T_in = 72.0  # Indoor temp in F
        self.T_target = self.T_in
        self.time_step = 0

        print(f"Step {self.time_step}: T_in={self.T_in:.2f}°F, T_out={self.df.loc[self.current_idx, 'temp']:.1f}°F, Humid={self.df.loc[self.current_idx, 'humid']:.1f}%, Wind={self.df.loc[self.current_idx, 'wind_sp']:.1f} m/s")
        state = self._get_obs()
        return state, {}

    def step(self, action):
        """
        Step the environment
        """
        HVAC_power = np.clip(action[0], 0.0, 1.0)
        energy_used = abs(HVAC_power) * self.energy_rate

        # Current weather readings
        T_out = self.df.loc[self.current_idx, 'temp']  # Already in Fahrenheit
        humid = self.df.loc[self.current_idx, 'humid']
        wind_sp = self.df.loc[self.current_idx, 'wind_sp']

        # Sensor-based T_in_next simulation:
        # Random small noise, outdoor influence, HVAC effect independently applied
        temp_drift = np.random.uniform(0.05, 0.2) * (T_out - self.T_in) / 10  # Outdoor drift
        wind_cooling = np.random.uniform(0.01, 0.05) * (-wind_sp) / 10        # Wind cooling

        hvac_effect = HVAC_power * np.random.uniform(4.0, 6.0) # incorporating cooling/heating effect
        #if outside temp < inside temp: heating ie tin_next increase
        #if outside temp > inside temp: cooling ie tin_next decrease
        if T_out > self.T_in:
            hvac_effect = -hvac_effect

        sensor_noise = np.random.normal(0, 0.2)  # Small random sensor noise

        T_in_next = self.T_in + temp_drift + wind_cooling + hvac_effect + sensor_noise
        T_in_next = np.clip(T_in_next, 60.0, 90.0)  # Reasonable indoor temp range

        # Calculate reward
        comfort_penalty = abs(T_in_next - self.T_target)
        reward = -comfort_penalty - 0.1 * energy_used

        # Update for next step
        self.T_in = T_in_next
        self.current_idx += 1
        self.time_step += 1
        done = (self.time_step >= self.episode_length)

        next_state = self._get_obs()
        return next_state, reward, done, False, {}

    def _get_obs(self):
        """
        Current observation
        """
        T_out = self.df.loc[self.current_idx, 'temp']
        humid = self.df.loc[self.current_idx, 'humid']
        wind_sp = self.df.loc[self.current_idx, 'wind_sp']

        T_out_future = self.df.loc[self.current_idx + 2, 'temp']  # 2 hours ahead

        state = np.array([self.T_in, T_out, T_out_future, humid, wind_sp, self.time_step], dtype=np.float32)
        return state

    def render(self, mode='human'):
        print(f"Step {self.time_step}: T_in={self.T_in:.2f}°F, T_out={self.df.loc[self.current_idx, 'temp']:.1f}°F, Humid={self.df.loc[self.current_idx, 'humid']:.1f}%, Wind={self.df.loc[self.current_idx, 'wind_sp']:.1f} m/s")

    def close(self):
        pass