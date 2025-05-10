from dmc import make_dmc_env
import numpy as np

def make_env():
    # Create environment with state observations
    env_name = "humanoid-walk"
    env = make_dmc_env(
        env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False
    )
    return env


env = make_env()

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tqdm import tqdm

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


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[256, 256]):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


#class OUNoise:
#    """Ornstein-Uhlenbeck process for temporally correlated noise."""
#
#    def __init__(self, size, mu=0.0, sigma=0.2, theta=0.15, seed=None):
#        self.mu = mu
#        self.sigma = sigma
#        self.theta = theta
#        self.size = size
#        self.rng = np.random.default_rng(seed)
#        self.reset()
#
#    def reset(self):
#        self.state = np.ones(self.size) * self.mu
#
#    def sample(self):
#        x = self.state
#        dx = self.theta * (self.mu - x) + self.sigma * self.rng.standard_normal(
#            self.size
#        )
#        self.state = x + dx
#        return self.state
#
#
#noise = OUNoise(env.action_space.shape[0], mu=0.0, sigma=0.2, theta=0.15, seed=42)


def get_action(actor, state, add_noise=True, noise_decay=None):
    state = torch.from_numpy(state).float().to(device)
    with torch.no_grad():
        action = actor(state).cpu().numpy()
    if add_noise:
        #action += noise_decay * noise.sample()
        action += np.random.normal(0, 0.2, action.shape)
    return np.clip(action, -1, 1)



def train(
    actor_learner,
    critic_learner,
    actor_target,
    critic_target,
    replay_buffer,
    actor_optimizer,
    critic_optimizer,
    tau = 0.005
):
    batch = replay_buffer.sample()
    state, action, reward, next_state, done = (
        batch["state"],
        batch["action"],
        batch["reward"],
        batch["next_state"],
        batch["done"],
    )
    y = reward + 0.99 * critic_target(next_state, actor_target(next_state)).squeeze(
        -1
    ) * (1 - done)
    critic_loss = F.mse_loss(critic_learner(state, action).squeeze(-1), y)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    for param in critic_learner.parameters():
        param.requires_grad = False
    actor_loss = -critic_learner(state, actor_learner(state)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    for param in critic_learner.parameters():
        param.requires_grad = True
    for target_param, param in zip(
        critic_target.parameters(), critic_learner.parameters()
    ):
        target_param.data.copy_(target_param.data * (1-tau) + param.data * tau)
    for target_param, param in zip(
        actor_target.parameters(), actor_learner.parameters()
    ):
        target_param.data.copy_(target_param.data * (1-tau) + param.data * tau)


NUM_EPISODES = 6000
TARGET_SCORE = 500
BATCH_SIZE = 512
UPDATE_INTERVAL = 25
PRINT_INTERVAL = 100
MAX_FALL_DURATION = 100

actor_learner = Actor(67, 21)
actor_target = Actor(67, 21)
actor_target.load_state_dict(actor_learner.state_dict())
actor_optimizer = optim.Adam(actor_learner.parameters(), lr=0.00025)

critic_learner = Critic(67, 21)
critic_target = Critic(67, 21)
critic_target.load_state_dict(critic_learner.state_dict())
critic_optimizer = optim.Adam(critic_learner.parameters(), lr=0.00025)

actor_learner.to(device)
actor_target.to(device)
critic_learner.to(device)
critic_target.to(device)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(1000000, device=device), batch_size=BATCH_SIZE
)

below_target_score = True
score_deque = deque(maxlen=200)
for t in tqdm(range(NUM_EPISODES)):
    t += 1
    state, _ = env.reset()
    score = 0
    done = False
    step = 0
    fall_duration = 0
    while not done:  # 1000 steps
        # print(state)
        if t < 100:  # warmup
            action = np.random.uniform(-1.0, 1.0, size=env.action_space.shape)
        else:
            action = get_action(
                actor_learner, state, add_noise=True, noise_decay=0.999 ** (t - 1)
            )
        next_state, reward, done, truncated, _ = env.step(action)
        # print(f"{reward=}") # often < 10**-10
        replay_buffer.add(
            {
                "state": torch.from_numpy(state).float(),
                "action": torch.from_numpy(action).long(),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_state": torch.from_numpy(next_state).float(),
                "done": torch.tensor(done, dtype=torch.int32),
            }
        )
        if len(replay_buffer) > BATCH_SIZE:
            train(
                actor_learner,
                critic_learner,
                actor_target,
                critic_target,
                replay_buffer,
                actor_optimizer,
                critic_optimizer,
            )
        state = next_state
        score += reward
        step += 1
        if reward < 1e-14:
            fall_duration +=1
        done = done or truncated or fall_duration > MAX_FALL_DURATION
        #if step % UPDATE_INTERVAL == 0:
        #    actor_target.load_state_dict(actor_learner.state_dict())
        #    critic_target.load_state_dict(critic_learner.state_dict())

    # std = np.std(scores_deque)
    # score = mean - std
    if t % PRINT_INTERVAL == 0:
        print(f"Mean: {np.mean(score_deque)}")
        if t > 1000:
            scores = []
            for i in range(50):  # 1 episode/s
                state, _ = env.reset()
                score = 0
                done = False
                while not done:
                    action = get_action(actor_learner, state, add_noise=False)
                    next_state, reward, done, truncated, _ = env.step(action)
                    state = next_state
                    score += reward
                    done = done or truncated
                scores.append(score)

            mean = np.mean(scores)
            std = np.std(scores)
            score = mean - std
            print(f"Episode {t}:\tMean: {mean:.2f}\tStd: {std:.2f}\tScore: {score:.2f}")

            if score > TARGET_SCORE:
                print(f"Saving Weights")
                torch.save(
                    actor_learner.state_dict(), f"actor_learner_{t}_{int(score)}.pth"
                )
                torch.save(
                    critic_learner.state_dict(), f"critic_learner_{t}_{int(score)}.pth"
                )

torch.save(actor_learner.state_dict(), "actor_learner_last.pth")
torch.save(critic_learner.state_dict(), "critic_learner_last.pth")
