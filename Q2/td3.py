import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from torchrl.data import ReplayBuffer, LazyTensorStorage
from dmc import make_dmc_env
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env():
    # Create environment with state observations
    env_name = "cartpole-balance"
    env = make_dmc_env(
        env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False
    )
    return env


env = make_env()
#print(type(env))
#print(env))
# print(env.action_space) #Box(-1.0, 1.0, (1,), float64)
# print(env.observation_space) #Box(-inf, inf, (5,), float64)

hidden_sizes=[128, 128]

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=hidden_sizes):
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
    def __init__(self, state_size, action_size, hidden_sizes=hidden_sizes):
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


def get_action(actor, state, add_noise=True, noise_decay=None):
    state = torch.from_numpy(state).float().to(device)
    with torch.no_grad():
        action = actor(state).cpu().numpy()
    if add_noise:
        action += np.random.normal(0, 0.1, size=action.shape) * noise_decay
    return np.clip(action, -1, 1)


def train(
    actor_learner,
    critic1_learner,
    critic2_learner,
    actor_target,
    critic1_target,
    critic2_target,
    replay_buffer,
    actor_optimizer,
    critic1_optimizer,
    critic2_optimizer,
    update_actor_and_target=False,
):
    batch = replay_buffer.sample()
    state, action, reward, next_state, done = (
        batch["state"],
        batch["action"],
        batch["reward"],
        batch["next_state"],
        batch["done"],
    )
    noise = torch.normal(0, 0.2, size=action.shape).clamp(-0.5, 0.5).to(device)
    target_action = (actor_target(next_state) + noise).clamp(-1, 1)

    q1 = critic1_target(next_state, target_action).squeeze(-1)
    q2 = critic2_target(next_state, target_action).squeeze(-1)
    min_q = torch.min(q1, q2)

    y = reward + 0.99 * min_q * (1 - done)
    for critic_learner, critic_optimizer in zip(
        [critic1_learner, critic2_learner], [critic1_optimizer, critic2_optimizer]
    ):
        critic_loss = F.mse_loss(critic_learner(state, action).squeeze(-1), y.detach())
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

    if not update_actor_and_target:
        return

    # update actor
    for critic_learner in [critic1_learner, critic2_learner]:
        for param in critic_learner.parameters():
            param.requires_grad = False

    actor_loss = -critic1_learner(state, actor_learner(state)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    for critic_learner in [critic1_learner, critic2_learner]:
        for param in critic_learner.parameters():
            param.requires_grad = True

    # Soft update of target networks
    for target, learner in zip(
        [actor_target, critic1_target, critic2_target],
        [actor_learner, critic1_learner, critic2_learner],
    ):
        for target_param, param in zip(target.parameters(), learner.parameters()):
            target_param.data.copy_(
                target_param.data * 0.995 + param.data * (1.0 - 0.995)
            )


TARGET_SCORE = 1100
BATCH_SIZE = 256
UPDATE_INTERVAL = 2
PRINT_INTERVAL = 50

scores_deque = deque(maxlen=UPDATE_INTERVAL)

actor_learner = Actor(5, 1)
actor_target = Actor(5, 1)
actor_target.load_state_dict(actor_learner.state_dict())
actor_optimizer = optim.AdamW(actor_learner.parameters(), lr=0.0001, weight_decay=0.01)

critic1_learner = Critic(5, 1)
critic2_learner = Critic(5, 1)
critic1_target = Critic(5, 1)
critic2_target = Critic(5, 1)
critic1_target.load_state_dict(critic1_learner.state_dict())
critic2_target.load_state_dict(critic2_learner.state_dict())
# weight decay (l2 regularization) can be bad for noisy target
critic1_optimizer = optim.Adam(critic1_learner.parameters(), lr=0.0001)
critic2_optimizer = optim.Adam(critic2_learner.parameters(), lr=0.0001)

actor_learner.to(device)
actor_target.to(device)
critic1_learner.to(device)
critic2_learner.to(device)
critic1_target.to(device)
critic2_target.to(device)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(10000, device=device), batch_size=BATCH_SIZE
)

t = 0
below_target_score = True
while below_target_score and t < 2000:
    t += 1
    state, _ = env.reset()
    score = 0
    done = False
    step = 0
    while not done:  # 1000 steps
        if t < 10:
            action = np.random.uniform(-1, 1, size=(1,))
        else:
            action = get_action(
                actor_learner, state, add_noise=True, noise_decay=0.999**t
            )
        ret = env.step(action)
        next_state, reward, done, truncated, _ = ret
        replay_buffer.add(
            {
                "state": torch.from_numpy(state).float(),
                "action": torch.from_numpy(action).long(),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_state": torch.from_numpy(next_state).float(),
                "done": torch.tensor(done, dtype=torch.int32),
            }
        )
        done = done or truncated
        if len(replay_buffer) > BATCH_SIZE:
            train(
                actor_learner,
                critic1_learner,
                critic2_learner,
                actor_target,
                critic1_target,
                critic2_target,
                replay_buffer,
                actor_optimizer,
                critic1_optimizer,
                critic2_optimizer,
                update_actor_and_target=step % UPDATE_INTERVAL == 0,
            )
        state = next_state
        score += reward
        step += 1

    if t % PRINT_INTERVAL == 0:
        scores = []
        for i in range(25):
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
            below_target_score = False

print(f"Environment solved in {t} episodes! Average score: {np.mean(scores_deque):.2f}")
torch.save(actor_learner.state_dict(), f"actor_learner{hidden_sizes[0]}.pth")
torch.save(critic1_learner.state_dict(), f"critic1_learner_{hidden_sizes[0]}.pth")
torch.save(critic2_learner.state_dict(), f"critic2_learner{hidden_sizes[0]}.pth")
