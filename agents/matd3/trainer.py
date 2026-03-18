import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.base_trainer import BaseTrainer
from agents.common.buffer import ReplayBuffer
from agents.common.utils import soft_update
from agents.matd3.networks import Actor, Critic


class MATD3Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.device = torch.device(config["device"])
        self.n_agents = config["env"]["n_agents"]
        self.obs_dim = config["env"]["obs_dim"]
        self.action_dim = config["env"]["action_dim"]
        self.hidden_dim = config["train"]["hidden_dim"]
        self.gamma = config["train"]["gamma"]
        self.tau = config["train"]["tau"]
        self.batch_size = config["train"]["batch_size"]

        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.update_step = 0

        total_obs_dim = self.n_agents * self.obs_dim
        total_action_dim = self.n_agents * self.action_dim

        self.actors = []
        self.target_actors = []
        self.critic1 = []
        self.critic2 = []
        self.target_critic1 = []
        self.target_critic2 = []
        self.actor_optimizers = []
        self.critic1_optimizers = []
        self.critic2_optimizers = []

        for _ in range(self.n_agents):
            actor = Actor(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
            target_actor = Actor(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
            target_actor.load_state_dict(actor.state_dict())

            c1 = Critic(total_obs_dim, total_action_dim, self.hidden_dim).to(self.device)
            tc1 = Critic(total_obs_dim, total_action_dim, self.hidden_dim).to(self.device)
            tc1.load_state_dict(c1.state_dict())

            c2 = Critic(total_obs_dim, total_action_dim, self.hidden_dim).to(self.device)
            tc2 = Critic(total_obs_dim, total_action_dim, self.hidden_dim).to(self.device)
            tc2.load_state_dict(c2.state_dict())

            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.critic1.append(c1)
            self.critic2.append(c2)
            self.target_critic1.append(tc1)
            self.target_critic2.append(tc2)

            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=config["train"]["actor_lr"]))
            self.critic1_optimizers.append(optim.Adam(c1.parameters(), lr=config["train"]["critic_lr"]))
            self.critic2_optimizers.append(optim.Adam(c2.parameters(), lr=config["train"]["critic_lr"]))

        self.buffer = ReplayBuffer(config["train"]["buffer_size"])

    def select_action(self, obs, evaluate=False):
        actions = []
        for i in range(self.n_agents):
            obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action = self.actors[i](obs_tensor).cpu().numpy()[0]
            if not evaluate:
                action += np.random.normal(0, 0.1, size=self.action_dim)
            actions.append(np.clip(action, -1.0, 1.0))
        return actions

    def store_transition(self, obs, actions, rewards, next_obs, dones):
        self.buffer.push(obs, actions, rewards, next_obs, dones)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        self.update_step += 1

        obs_b, actions_b, rewards_b, next_obs_b, dones_b = self.buffer.sample(self.batch_size)

        obs_all = torch.tensor(np.array(obs_b), dtype=torch.float32, device=self.device)
        actions_all = torch.tensor(np.array(actions_b), dtype=torch.float32, device=self.device)
        next_obs_all = torch.tensor(np.array(next_obs_b), dtype=torch.float32, device=self.device)

        obs_all_flat = obs_all.reshape(self.batch_size, -1)
        actions_all_flat = actions_all.reshape(self.batch_size, -1)
        next_obs_all_flat = next_obs_all.reshape(self.batch_size, -1)

        for agent_i in range(self.n_agents):
            rewards = torch.tensor(np.array([r[agent_i] for r in rewards_b]), dtype=torch.float32, device=self.device).unsqueeze(-1)
            dones = torch.tensor(np.array([d[agent_i] for d in dones_b]), dtype=torch.float32, device=self.device).unsqueeze(-1)

            with torch.no_grad():
                next_actions = []
                for j in range(self.n_agents):
                    a = self.target_actors[j](next_obs_all[:, j, :])
                    noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                    a = (a + noise).clamp(-1.0, 1.0)
                    next_actions.append(a)

                next_actions_all = torch.stack(next_actions, dim=1).reshape(self.batch_size, -1)
                q1_target = self.target_critic1[agent_i](next_obs_all_flat, next_actions_all)
                q2_target = self.target_critic2[agent_i](next_obs_all_flat, next_actions_all)
                q_target = torch.min(q1_target, q2_target)
                y = rewards + self.gamma * (1 - dones) * q_target

            q1 = self.critic1[agent_i](obs_all_flat, actions_all_flat)
            q2 = self.critic2[agent_i](obs_all_flat, actions_all_flat)

            critic1_loss = F.mse_loss(q1, y)
            critic2_loss = F.mse_loss(q2, y)

            self.critic1_optimizers[agent_i].zero_grad()
            critic1_loss.backward()
            self.critic1_optimizers[agent_i].step()

            self.critic2_optimizers[agent_i].zero_grad()
            critic2_loss.backward()
            self.critic2_optimizers[agent_i].step()

            if self.update_step % self.policy_delay == 0:
                current_actions = []
                for j in range(self.n_agents):
                    if j == agent_i:
                        current_actions.append(self.actors[j](obs_all[:, j, :]))
                    else:
                        current_actions.append(actions_all[:, j, :].detach())

                current_actions_all = torch.stack(current_actions, dim=1).reshape(self.batch_size, -1)
                actor_loss = -self.critic1[agent_i](obs_all_flat, current_actions_all).mean()

                self.actor_optimizers[agent_i].zero_grad()
                actor_loss.backward()
                self.actor_optimizers[agent_i].step()

                soft_update(self.target_actors[agent_i], self.actors[agent_i], self.tau)
                soft_update(self.target_critic1[agent_i], self.critic1[agent_i], self.tau)
                soft_update(self.target_critic2[agent_i], self.critic2[agent_i], self.tau)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), os.path.join(path, f"matd3_actor_{i}.pt"))
            torch.save(self.critic1[i].state_dict(), os.path.join(path, f"matd3_critic1_{i}.pt"))
            torch.save(self.critic2[i].state_dict(), os.path.join(path, f"matd3_critic2_{i}.pt"))

    def load(self, path):
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(torch.load(os.path.join(path, f"matd3_actor_{i}.pt"), map_location=self.device))
            self.critic1[i].load_state_dict(torch.load(os.path.join(path, f"matd3_critic1_{i}.pt"), map_location=self.device))
            self.critic2[i].load_state_dict(torch.load(os.path.join(path, f"matd3_critic2_{i}.pt"), map_location=self.device))
