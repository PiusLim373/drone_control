import os
from drone_control_test_gym import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Configuration
RENDER = True
CHECKPOINT_DIR = "saves/"
MODEL_NAME = "sample_sb_model.zip"  # Change this if you need to load different model
MODEL_PATH = os.path.join(CHECKPOINT_DIR, MODEL_NAME)

# Load the saved model
model = PPO.load(MODEL_PATH)

# Create the environment (use the same environment as training)
env = DroneControlGym(render=RENDER)
gym_env = DummyVecEnv([lambda: env])  # Use DummyVecEnv for vectorized environments

# Testing the agent
n_episodes = 10  # Number of episodes to run
scores = []  # Initialize a list to store episode scores

for episode in range(n_episodes):
    observation = gym_env.reset()  # Reset the environment for a new episode
    done = False
    score = 0

    while not done:
        action, _states = model.predict(observation)  # Get the action from the model
        observation, reward, done, info = gym_env.step(action)  # Step the environment
        score += reward  # Accumulate score

    scores.append(score)  # Add episode score to the scores list
    print(f"Episode {episode + 1}: Reward= {score}")

# Calculate and print average reward
average_reward = np.mean(scores)
print(f"Average Reward over {n_episodes} episodes: {average_reward}")

# End of the testing
gym_env.close()  # Close the environment window if applicable
