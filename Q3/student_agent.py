import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


state_size = 67
action_size = 21

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACActor(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        hidden_sizes=[512, 512, 512],
        log_std_bounds=[-20, 2],
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_sizes[2], action_size)
        self.log_std_layer = nn.Linear(hidden_sizes[2], action_size)
        self.log_std_bounds = log_std_bounds

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        std = torch.exp(log_std)
        return mean, std


def get_action(actor, state):
    # batch size 1
    state = torch.from_numpy(state).float().to(device).unsqueeze(0)
    with torch.no_grad():
        mean, _ = actor(state)
        action = torch.tanh(mean)
    return action[0].cpu().numpy()


actor = SACActor(state_size, action_size).to(device)
actor.load_state_dict(torch.load("./sac_actor_1667_717.pth"))


# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""

    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

    def act(self, observation):
        return get_action(actor, observation)
