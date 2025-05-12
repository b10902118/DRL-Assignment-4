import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import numpy as np
import torch


# Do not modify the input of the 'act' function and the '__init__' function.
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256]):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_size),
            nn.Tanh(),  # to ensure output is in [-1, 1]
        )

    def forward(self, state):
        return 2 * self.net(state)


actor = Actor(3, 1)
actor.load_state_dict(torch.load("actor_learner.pth", map_location=device))
actor.to(device)


class Agent(object):
    """Agent that acts randomly."""

    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

    def act(self, observation):
        state = torch.from_numpy(observation).to(device)
        with torch.no_grad():
            return actor(state).cpu().numpy()
