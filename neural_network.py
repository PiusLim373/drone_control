#!/usr/bin/python3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# Memory class to store all rollouts information
class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

    def generate_batches(self):
        # randomly shuffle the memory and generate mini batches
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] -> [[5, 2, 7, 1, 9], [3, 6, 4, 8, 10]]
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )


# The Actor (policy)network
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, learning_rate, fc1_dims=256, fc2_dims=256, chkpt_dir="saves"):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "actor_torch_ppo.pth")

        # a fully connected neural network with two hidden layer (256nodes each) with ReLU activation function
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1),
        )

        # adam optimizer with learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, state):
        # Get the action from the actor nn
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_checkpoint(self):
        # save the model state and optimizer state, for saving model
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_file)

    def load_checkpoint(self):
        # load the model state and optimizer state, for loading model / resuming training
        checkpoint = torch.load(self.checkpoint_file)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# The Critic (value) network
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, learning_rate, fc1_dims=256, fc2_dims=256, chkpt_dir="saves"):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo.pth")

        # a fully connected neural network with two hidden layer (256nodes each) with ReLU activation function
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )

        # adam optimizer with learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, state):
        # Get the value from the critic nn
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        # save the model state and optimizer state, for saving model
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_file)

    def load_checkpoint(self):
        # load the model state and optimizer state, for loading model / resuming training
        checkpoint = torch.load(self.checkpoint_file)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
