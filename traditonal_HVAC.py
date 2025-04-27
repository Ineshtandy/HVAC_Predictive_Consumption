import numpy as np

def traditional_hvac_baseline(env, threshold=0.5, hvac_power_on=1.0):
    obs = env.reset()
    T_target = env.T_target
    total_energy_used = 0
    total_comfort_penalty = 0

    print("---- Traditional HVAC Baseline ----")
    for _ in range(env.episode_length):
        T_in, T_out, T_out_future, Solar_rad, time_step = obs

        # Rule-based control:
        if T_in > T_target + threshold:
            hvac_action = np.array([hvac_power_on])  # Full power ON
        else:
            hvac_action = np.array([0.0])  # HVAC OFF

        obs, reward, done, _ = env.step(hvac_action)

        # Extract energy and comfort penalties for comparison
        energy_used = hvac_action[0] * env.energy_rate
        comfort_penalty = abs(env.T_in - T_target)

        total_energy_used += energy_used
        total_comfort_penalty += comfort_penalty

        env.render()
        print(f"Action: {hvac_action[0]:.2f}, T_in: {env.T_in:.2f}, Energy: {energy_used:.2f}, Comfort Penalty: {comfort_penalty:.2f}")

        if done:
            break

    print("------------------------------------")
    print(f"Total Energy Used: {total_energy_used:.2f}")
    print(f"Total Comfort Penalty: {total_comfort_penalty:.2f}")
    return total_energy_used, total_comfort_penalty
