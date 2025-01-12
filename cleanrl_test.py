"""Basic code which shows what it's like to run PPO on the Pistonball env using the parallel API, this code is inspired by CleanRL.

This code is exceedingly basic, with no logging or weights saving.
The intention was for users to have a (relatively clean) ~200 line file to refer to when they want to design their own learning algorithm.

Author: Jet (https://github.com/jjshoots)
"""

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical

from pettingzoo.butterfly import pistonball_v6
from torch.utils.tensorboard import SummaryWriter



class Agent(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x


def add_noise(observation, noise_level, device):
    """为状态添加随机噪声"""
    noise = torch.normal(0, noise_level, size=observation.shape).to(device)
    return torch.clamp(observation + noise, 0.0, 255.0)


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    stack_size = 4
    frame_size = (64, 64)
    max_cycles = 125
    total_episodes = 200

    """ RENDER THE POLICY """
    env = pistonball_v6.parallel_env(render_mode="human", continuous=False)
    env = color_reduction_v0(env)
    env = resize_v1(env, 64, 64)
    env = frame_stack_v1(env, stack_size=4)

    num_actions = env.action_space(env.possible_agents[0]).n
    agent = Agent(num_actions=num_actions).to(device)
    writer = SummaryWriter(log_dir=f'logs3/log_noise_test_e_{total_episodes}')

    rewards_list = []
    noise_levels = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    eval_episodes = 20
    for noise_level in noise_levels:
        try:
            agent.load_state_dict(torch.load(f'./pth/agent_noise_{noise_level}_e_{total_episodes}.pth'))
        except FileNotFoundError:
            print(f"File './pth/agent_noise_{noise_level}_e_{total_episodes}.pth' not found, skipping...")
            continue

        agent.eval()

        total_reward = 0
        with torch.no_grad():
            for episode in range(eval_episodes):
                obs, infos = env.reset(seed=None)
                obs = batchify_obs(obs, device)
                terms = [False]
                truncs = [False]
                while not any(terms) and not any(truncs):
                    obs = add_noise(obs, noise_level, device)
                    actions, logprobs, _, values = agent.get_action_and_value(obs)
                    obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                    obs = batchify_obs(obs, device)
                    terms = [terms[a] for a in terms]
                    truncs = [truncs[a] for a in truncs]
                    # print(type(obs), obs.shape)
                    total_reward += sum(rewards.values())

            print(f'Average reward of noise level {noise_level}: {total_reward / eval_episodes}')
            rewards_list.append(total_reward / eval_episodes)
            writer.add_scalar("Average_Reward", total_reward / eval_episodes, noise_level)

    # 绘制鲁棒性曲线
    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, rewards_list, marker='o', label="Baseline Policy")
    plt.xlabel("Noise Level")
    plt.ylabel("Average Reward")
    plt.title("Robustness Evaluation under Noise Perturbations")
    plt.legend()
    plt.grid()
    plt.savefig('log_noise_test_e_{total_episodes}.png', dpi=300, bbox_inches='tight')
    plt.show()

    writer.close()