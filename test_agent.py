#!/usr/bin/python3
import os
import numpy as np  
from drone_control_gym import *
from training_agent import Agent
import torch

# Configuration
RENDER = True
CHECKPOINT_DIR = "saves/"
USE_AUTOSAVE = False  # Set to True if you want to load autosave

# Determine which checkpoint files to load
if USE_AUTOSAVE:
    ACTOR_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "autosave_actor_torch_ppo.pth")
    CRITIC_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "autosave_critic_torch_ppo.pth")
else:
    ACTOR_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "actor_torch_ppo.pth")
    CRITIC_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "critic_torch_ppo.pth")

# Initialize environment and agent
gym_env = DroneControlGym(render=RENDER)
agent = Agent(input_dims=11, action_dims=16, learning_rate=0.0003, 
              discount=0.99, gae_lambda=0.95, ppo_clip=0.2, 
              batch_size=64, n_epoch=15, checkpoint_dir=CHECKPOINT_DIR)

if os.path.isfile(ACTOR_CHECKPOINT) and os.path.isfile(CRITIC_CHECKPOINT):
    agent.actor.checkpoint_file = ACTOR_CHECKPOINT
    agent.critic.checkpoint_file = CRITIC_CHECKPOINT
    agent.load_models()  # Use the load_models method
else:
    print("Starting fresh, no models loaded.")

# Testing the agent
n_episodes = 5  # Number of episodes to run
scores = []  # Initialize a list to store episode scores

for episode in range(n_episodes):
    observation = gym_env.reset()  # Reset the environment for a new episode
    done = False
    score = 0
    
    while not done:
        action, prob, val = agent.choose_action(observation)  # Choose action
        reward, done, observation_new = gym_env.step(ACTIONS[action])  # Take a step in the environment
        score += reward  # Accumulate score
        observation = observation_new  # Update observation for the next step

    scores.append(score)  # Add episode score to the scores list
    print(f"Episode {episode + 1}: Reward= {score}")

# Calculate and print average reward 
average_reward = np.mean(scores)
print(f"Average Reward over {n_episodes} episodes: {average_reward}")

# End of the testing
gym_env.close()  # Close the environment window if applicable