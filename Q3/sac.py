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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- SAC Actor ---
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

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


# --- SAC Critic (Twin Q) ---
class SACCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[512, 512, 512]):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)


# --- Training function ---
def sac_train(
    actor,
    critic,
    target_critic,
    replay_buffer,
    actor_optimizer,
    critic_optimizer,
    alpha,
    gamma=0.99,
    tau=0.005,
):
    batch = replay_buffer.sample()
    state, action, reward, next_state, done = (
        batch["state"],
        batch["action"],
        batch["reward"],
        batch["next_state"],
        batch["done"],
    )

    # Compute target Q value
    with torch.no_grad():
        next_action, next_log_prob = actor.sample(next_state)
        target_q1, target_q2 = target_critic(next_state, next_action)
        target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
        q_target = reward + (1 - done) * gamma * target_q.squeeze(-1)

    # Current Q estimates
    current_q1, current_q2 = critic(state, action)
    current_q1 = current_q1.squeeze(-1)
    current_q2 = current_q2.squeeze(-1)
    critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor update
    new_action, log_prob = actor.sample(state)
    q1_new, q2_new = critic(state, new_action)
    q_new = torch.min(q1_new, q2_new)
    actor_loss = (alpha * log_prob - q_new).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update of target critic
    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# --- Action selection ---
def get_action(actor, state, deterministic=False):
    # batch size 1
    state = torch.from_numpy(state).float().to(device).unsqueeze(0)
    with torch.no_grad():
        if deterministic:
            mean, _ = actor(state)
            action = torch.tanh(mean)
        else:
            action, _ = actor.sample(state)
    return action[0].cpu().numpy()


NUM_EPISODES = 5000
TARGET_SCORE = 500
BATCH_SIZE = 256
UPDATE_INTERVAL = 25
PRINT_INTERVAL = 200
# MAX_FALL_DURATION = 200
UPDATE_PER_STEP = 1
WARMUP_EPISODES = 25

# --- Initialize everything ---
state_size = 67
action_size = 21
alpha = 0.2  # fixed entropy coefficient
actor = SACActor(state_size, action_size).to(device)
critic = SACCritic(state_size, action_size).to(device)
target_critic = SACCritic(state_size, action_size).to(device)
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(1_000_000, device=device), batch_size=BATCH_SIZE
)

# --- Training loop ---
score_deque = deque(maxlen=200)
step_deque = deque(maxlen=200)
best_eval_score = -float("inf")

for t in tqdm(range(1, NUM_EPISODES + 1)):
    state, _ = env.reset()
    score = 0
    step = 0
    done = False
    fall_duration = 0
    while not done:
        if t <= WARMUP_EPISODES:
            action = np.random.uniform(-1.0, 1.0, size=action_size)
        else:
            action = get_action(actor, state)

        next_state, reward, done, truncated, _ = env.step(action)
        replay_buffer.add(
            {
                "state": torch.from_numpy(state).float(),
                "action": torch.from_numpy(action).float(),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "next_state": torch.from_numpy(next_state).float(),
                "done": torch.tensor(done, dtype=torch.int32),
            }
        )

        if len(replay_buffer) > BATCH_SIZE and t > WARMUP_EPISODES:
            for i in range(UPDATE_PER_STEP):
                sac_train(
                    actor,
                    critic,
                    target_critic,
                    replay_buffer,
                    actor_optimizer,
                    critic_optimizer,
                    alpha,
                )

        state = next_state
        score += reward
        # if reward < 1e-10:
        #    fall_duration += 1
        # else:
        #    fall_duration = 0

        done = done or truncated  # or fall_duration > MAX_FALL_DURATION
        step += 1

    score_deque.append(score)
    step_deque.append(step)
    if t % PRINT_INTERVAL == 0:
        print(
            f"Episode {t} | Mean Score: {np.mean(score_deque):.2f} Mean Step: {np.mean(step_deque):.2f}"
        )

        if t > 500:
            eval_scores = []
            for _ in range(20):
                s, _ = env.reset()
                done = False
                ep_score = 0
                while not done:
                    a = get_action(actor, s, deterministic=True)
                    s, r, done, truncated, _ = env.step(a)
                    ep_score += r
                    done = done or truncated
                eval_scores.append(ep_score)

            mean = np.mean(eval_scores)
            std = np.std(eval_scores)
            final_score = mean - std
            print(
                f"Eval | Mean: {mean:.2f} | Std: {std:.2f} | Score: {final_score:.2f}"
            )

            if final_score > best_eval_score:
                print("Saving models...")
                torch.save(actor.state_dict(), f"sac_actor_{t}_{int(final_score)}.pth")
                torch.save(
                    critic.state_dict(), f"sac_critic_{t}_{int(final_score)}.pth"
                )
                best_eval_score = final_score
                if best_eval_score > 5:
                    UPDATE_PER_STEP = 2
                elif best_eval_score > 50:
                    UPDATE_PER_STEP = 4


# Final save
torch.save(actor.state_dict(), "sac_actor_final.pth")
torch.save(critic.state_dict(), "sac_critic_final.pth")
