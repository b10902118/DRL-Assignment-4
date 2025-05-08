import torch
import torch.nn as nn
import gymnasium
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return self.net(state)


def get_action(actor, state):
    state = torch.from_numpy(state).float().to(device)
    with torch.no_grad():
        action = actor(state).cpu().numpy()
    return np.clip(action, -1, 1)


actor = Actor(5, 1)
actor.load_state_dict(torch.load("./actor_learner_250_993.pth"))
actor.to(device)


# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""

    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)

    def act(self, observation):
        return get_action(actor, observation)
