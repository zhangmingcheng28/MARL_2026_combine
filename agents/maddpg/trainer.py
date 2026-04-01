import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.base_trainer import BaseTrainer
from agents.common.buffer import ReplayBuffer
from agents.common.utils import soft_update
from agents.maddpg.networks import Actor, CentralCritic


class MADDPGTrainer(BaseTrainer):
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

        total_obs_dim = self.n_agents * self.obs_dim
        total_action_dim = self.n_agents * self.action_dim

        self.actors = []
        self.target_actors = []
        self.critics = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []

        for _ in range(self.n_agents):
            actor = Actor(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
            target_actor = Actor(self.obs_dim, self.action_dim, self.hidden_dim).to(self.device)
            target_actor.load_state_dict(actor.state_dict())

            critic = CentralCritic(total_obs_dim, total_action_dim, self.hidden_dim).to(self.device)
            target_critic = CentralCritic(total_obs_dim, total_action_dim, self.hidden_dim).to(self.device)
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.critics.append(critic)
            self.target_critics.append(target_critic)

            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=config["train"]["actor_lr"]))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=config["train"]["critic_lr"]))

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
                next_actions_list = []
                for j in range(self.n_agents):
                    next_actions_list.append(self.target_actors[j](next_obs_all[:, j, :]))
                next_actions_all = torch.stack(next_actions_list, dim=1).reshape(self.batch_size, -1)

                target_q = self.target_critics[agent_i](next_obs_all_flat, next_actions_all)
                y = rewards + self.gamma * (1 - dones) * target_q

            current_q = self.critics[agent_i](obs_all_flat, actions_all_flat)
            critic_loss = F.mse_loss(current_q, y)

            self.critic_optimizers[agent_i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_i].step()

            current_actions = []
            for j in range(self.n_agents):
                if j == agent_i:
                    current_actions.append(self.actors[j](obs_all[:, j, :]))
                else:
                    current_actions.append(actions_all[:, j, :].detach())

            current_actions_all = torch.stack(current_actions, dim=1).reshape(self.batch_size, -1)
            actor_loss = -self.critics[agent_i](obs_all_flat, current_actions_all).mean()

            self.actor_optimizers[agent_i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_i].step()

            soft_update(self.target_actors[agent_i], self.actors[agent_i], self.tau)
            soft_update(self.target_critics[agent_i], self.critics[agent_i], self.tau)

    def save(self, path, episode=None, step=None):
        os.makedirs(path, exist_ok=True)
        for i in range(self.n_agents):
            actor_path = os.path.join(path, f"maddpg_actor_{i}.pt")
            critic_path = os.path.join(path, f"maddpg_critic_{i}.pt")
            torch.save(self.actors[i].state_dict(), actor_path)
            torch.save(self.critics[i].state_dict(), critic_path)
            if episode is not None:
                torch.save(self.actors[i].state_dict(), os.path.join(path, f"maddpg_actor_{i}_ep{int(episode)}.pt"))
                torch.save(self.critics[i].state_dict(), os.path.join(path, f"maddpg_critic_{i}_ep{int(episode)}.pt"))
            if step is not None:
                torch.save(self.actors[i].state_dict(), os.path.join(path, f"maddpg_actor_{i}_step{int(step)}.pt"))
                torch.save(self.critics[i].state_dict(), os.path.join(path, f"maddpg_critic_{i}_step{int(step)}.pt"))

    def load(self, path, checkpoint_tag=None):
        for i in range(self.n_agents):
            actor_name = f"maddpg_actor_{i}.pt" if checkpoint_tag is None else f"maddpg_actor_{i}_{checkpoint_tag}.pt"
            critic_name = f"maddpg_critic_{i}.pt" if checkpoint_tag is None else f"maddpg_critic_{i}_{checkpoint_tag}.pt"
            self.actors[i].load_state_dict(torch.load(os.path.join(path, actor_name), map_location=self.device))
            self.critics[i].load_state_dict(torch.load(os.path.join(path, critic_name), map_location=self.device))
