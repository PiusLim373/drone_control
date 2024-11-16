import os
from drone_control_test_gym import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Load the saved model
model_path = "./20241116_190314_drone_ppo/ppo_model_5600000.zip"  # Replace with your actual model filename
model = PPO.load(model_path)

# Create the environment (use the same environment as training)
env = DroneControlGym(render=True)
vec_env = DummyVecEnv([lambda: env])  # Use DummyVecEnv for vectorized environments

# Evaluate the model
num_episodes = 10  # Number of episodes to evaluate
total_rewards = []

for episode in range(num_episodes):
    obs = vec_env.reset()  # Reset the environment for a new episode
    done = False
    episode_reward = 0

    while not done:
        action, _states = model.predict(obs)  # Get the action from the model
        obs, reward, done, info = vec_env.step(action)  # Step the environment
        episode_reward += reward  # Accumulate the reward for this episode

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

# Calculate average reward
average_reward = np.mean(total_rewards)
print(f"Average Reward over {num_episodes} episodes: {average_reward}")

# Close the environment
env.close()
