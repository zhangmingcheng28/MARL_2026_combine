import numpy as np


class SharedMultiAgentEnv:
    def __init__(self, n_agents: int, obs_dim: int, action_dim: int, max_steps: int = 50):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.step_count = 0
        self.max_steps = max_steps

    def reset(self):
        self.step_count = 0
        return [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.n_agents)]

    def step(self, actions):
        self.step_count += 1
        next_obs = [np.random.randn(self.obs_dim).astype(np.float32) for _ in range(self.n_agents)]
        rewards = [float(np.random.randn() * 0.1) for _ in range(self.n_agents)]
        dones = [self.step_count >= self.max_steps for _ in range(self.n_agents)]
        info = {}
        return next_obs, rewards, dones, info
