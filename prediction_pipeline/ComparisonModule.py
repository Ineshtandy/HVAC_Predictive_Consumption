import numpy as np

class ComparisonModule():
    def __init__(self, df, env):
        self.df = df
        self.env = env

    
    def traditional_hvac_baseline(self, threshold=1.0, hvac_power_on=1.0, episodes=1):
        total_energy = 0
        total_comfort_penalty = 0

        # print("---- Traditional HVAC Baseline ----") 
        for ep in range(episodes):
            obs, _ = self.env.reset(seed = 42) # same seed for reproducibility of data point
            T_target = self.env.T_target
            episode_energy = 0
            episode_comfort = 0

            for step in range(self.env.episode_length):
                T_in, T_out, T_out_future, humid, wind_sp, time_step = obs

                # Rule-based control logic:
                if T_in > T_target + threshold:
                    hvac_action = np.array([-hvac_power_on])  # Full cooling
                elif T_in < T_target - threshold:
                    hvac_action = np.array([hvac_power_on])   # Full heating
                else:
                    hvac_action = np.array([0.0])  # HVAC OFF

                obs, reward, done, _, _ = self.env.step(hvac_action)

                # Extract energy and comfort penalties for comparison
                energy_used = abs(hvac_action[0]) * self.env.energy_rate
                comfort_penalty = abs(self.env.T_in - T_target)

                episode_energy += energy_used
                episode_comfort += comfort_penalty

                self.env.render()
                # print(f"Action: {hvac_action[0]:.2f}, T_in: {self.env.T_in:.2f}°F, Energy: {energy_used:.2f}, Comfort Penalty: {comfort_penalty:.2f}")

                if done:
                    break

            total_energy += episode_energy
            total_comfort_penalty += episode_comfort

            # print(f"Episode {ep+1}: Energy = {episode_energy:.2f}, Comfort Penalty = {episode_comfort:.2f}")
            # print("--------------------------------------------------")

        
        avg_energy = total_energy / episodes
        avg_comfort = total_comfort_penalty / episodes
        # print("\n--- Traditional HVAC Summary ---")
        # print(f"Avg Energy Used: {avg_energy:.2f}")
        # print(f"Avg Comfort Penalty: {avg_comfort:.2f}")
        return avg_energy, avg_comfort
    
    def advanced_hvac_baseline(self, threshold=1.0, episodes=1):
        total_energy = 0
        total_comfort_penalty = 0

        # print("---- Advanced HVAC Baseline ----")
        for ep in range(episodes):
            obs, _ = self.env.reset(seed = 42)
            T_target = self.env.T_target
            episode_energy = 0
            episode_comfort = 0

            for step in range(self.env.episode_length):
                T_in, T_out, T_out_future, humid, wind_sp, time_step = obs

                # Optimized rule-based control logic:
                temp_diff = T_in - T_target
                if abs(temp_diff) > threshold:
                    hvac_action = np.array([-temp_diff / (2 * threshold)])  # Scale power between -1 and 1
                    hvac_action = np.clip(hvac_action, -1.0, 1.0)  # Ensure action stays within bounds
                else:
                    hvac_action = np.array([0.0])  # HVAC OFF if within threshold

                obs, reward, done, _, _ = self.env.step(hvac_action)

                # Extract energy and comfort penalties for comparison
                energy_used = abs(hvac_action[0]) * self.env.energy_rate
                comfort_penalty = abs(self.env.T_in - T_target)

                episode_energy += energy_used
                episode_comfort += comfort_penalty

                self.env.render()
                # print(f"Action: {hvac_action[0]:.2f}, T_in: {self.env.T_in:.2f}°F, Energy: {energy_used:.2f}, Comfort Penalty: {comfort_penalty:.2f}")

                if done:
                    break

            total_energy += episode_energy
            total_comfort_penalty += episode_comfort

            # print(f"Episode {ep+1}: Energy = {episode_energy:.2f}, Comfort Penalty = {episode_comfort:.2f}")
            # print("--------------------------------------------------")

        avg_energy = total_energy / episodes
        avg_comfort = total_comfort_penalty / episodes
        # print("\n--- Advanced HVAC Summary ---")
        # print(f"Avg Energy Used: {avg_energy:.2f}")
        # print(f"Avg Comfort Penalty: {avg_comfort:.2f}")
        return avg_energy, avg_comfort
    
    def evaluate_rl_agent(self, model, episodes=1): # num of times prediction is being done
        total_energy = 0
        total_comfort_penalty = 0

        # print("---- RL Agent Evaluation ----")
        for ep in range(episodes):
            obs, _ = self.env.reset(seed = 42)
            T_target = self.env.T_target
            episode_energy = 0
            episode_comfort = 0

            for step in range(self.env.episode_length): # single prediction time steps 
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = self.env.step(action)

                energy_used = abs(action[0]) * self.env.energy_rate
                comfort_penalty = abs(self.env.T_in - T_target)

                episode_energy += energy_used
                episode_comfort += comfort_penalty

                self.env.render()
                print(f"Action: {action[0]:.2f}, T_in: {self.env.T_in:.2f}°F, Energy: {energy_used:.2f}, Comfort Penalty: {comfort_penalty:.2f}")

                if done:
                    break

            total_energy += episode_energy
            total_comfort_penalty += episode_comfort

            print(f"Episode {ep+1}: Energy = {episode_energy:.2f}, Comfort Penalty = {episode_comfort:.2f}")
            # print("--------------------------------------------------")

        avg_energy = total_energy / episodes
        avg_comfort = total_comfort_penalty / episodes
        # print("\n--- RL Agent Summary ---")
        # print(f"Avg Energy Used: {avg_energy:.2f}")
        # print(f"Avg Comfort Penalty: {avg_comfort:.2f}")
        return avg_energy, avg_comfort