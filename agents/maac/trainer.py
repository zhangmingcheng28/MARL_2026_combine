import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.base_trainer import BaseTrainer
from agents.common.buffer import ReplayBuffer
from agents.common.utils import soft_update
from agents.maac.networks import Actor, MAACAttentionCritic


class MAACTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.device = torch.device(config["device"])
        self.dtype_name = config.get("dtype", "float32")
        self.torch_dtype = torch.float64 if self.dtype_name == "float64" else torch.float32
        self.numpy_dtype = np.float64 if self.dtype_name == "float64" else np.float32
        self.n_agents = config["env"]["n_agents"]
        self.action_dim = config["env"]["action_dim"]
        self.hidden_dim = config["train"]["hidden_dim"]
        self.gamma = config["train"]["gamma"]
        self.tau = config["train"]["tau"]
        self.batch_size = config["train"]["batch_size"]
        self.update_every = config["train"]["update_every"]
        self.max_grad_norm = float(config["train"].get("max_grad_norm", 0.0))
        self.attention_heads = int(config["train"].get("maac_attend_heads", 4))
        self.exploration = config.get("exploration", {})

        self.obs_dim = None
        self.obs_split_dims = None

        self.actor = None
        self.target_actor = None
        self.critic = None
        self.target_critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.current_episode = 1
        self.action_step = 0
        self.pending_load_path = None

        self.buffer = ReplayBuffer(config["train"]["buffer_size"])
        self.last_action_info = {}
        self.last_update_info = {
            "update_performed": False,
            "actor_updated": False,
            "update_step": 0,
            "buffer_size": 0,
            "learning_starts": self.batch_size,
            "batch_size": self.batch_size,
            "policy_delay": 1,
            "l2_reg": 0.0,
            "non_stationary_adam": False,
            "policy_noise": 0.0,
            "noise_clip": 0.0,
            "max_grad_norm": self.max_grad_norm,
        }

    @staticmethod
    def _mean_tensor_value(values):
        if not values:
            return None
        return float(torch.stack(values).mean().detach().cpu().item())

    @staticmethod
    def _grad_norm(parameters):
        squared_norm = 0.0
        has_grad = False
        for parameter in parameters:
            if parameter.grad is None:
                continue
            has_grad = True
            grad_norm = parameter.grad.detach().data.norm(2).item()
            squared_norm += grad_norm ** 2
        if not has_grad:
            return None
        return squared_norm ** 0.5

    def _flatten_agent_obs(self, obs, agent_idx):
        flattened_parts = []
        for portion in obs[:3]:
            flattened_parts.append(np.asarray(portion[agent_idx], dtype=self.numpy_dtype).reshape(-1))
        return np.concatenate(flattened_parts, axis=0)

    def _extract_obs_split_dims(self, obs):
        if len(obs) < 3:
            raise ValueError("Expected observation with three portions: own obs, neighbors, radar.")
        return tuple(int(np.asarray(portion[0]).reshape(-1).shape[0]) for portion in obs[:3])

    def _flatten_obs_batch(self, obs):
        return np.asarray(
            [self._flatten_agent_obs(obs, agent_idx) for agent_idx in range(self.n_agents)],
            dtype=self.numpy_dtype,
        )

    def _split_actor_obs_tensor(self, obs_tensor):
        start = 0
        split_tensors = []
        for dim in self.obs_split_dims:
            end = start + dim
            split_tensors.append(obs_tensor[:, start:end])
            start = end
        return split_tensors

    def _split_obs_batch(self, obs_batch, agent_idx):
        return self._split_actor_obs_tensor(obs_batch[:, agent_idx, :])

    def _split_all_agent_obs(self, obs_batch):
        own_dim, neigh_dim, radar_dim = self.obs_split_dims
        own_obs = obs_batch[:, :, :own_dim]
        neigh_obs = obs_batch[:, :, own_dim:own_dim + neigh_dim]
        radar_obs = obs_batch[:, :, own_dim + neigh_dim:own_dim + neigh_dim + radar_dim]
        return [own_obs, neigh_obs, radar_obs]

    def _ensure_models_initialized(self, obs):
        flattened_obs = self._flatten_obs_batch(obs)
        inferred_obs_dim = int(flattened_obs.shape[1])
        inferred_split_dims = self._extract_obs_split_dims(obs)

        if self.obs_dim is not None:
            if inferred_obs_dim != self.obs_dim:
                raise ValueError(
                    "MAAC observation dimension changed from {} to {}.".format(
                        self.obs_dim,
                        inferred_obs_dim,
                    )
                )
            if inferred_split_dims != self.obs_split_dims:
                raise ValueError(
                    "MAAC observation split changed from {} to {}.".format(
                        self.obs_split_dims,
                        inferred_split_dims,
                    )
                )
            return flattened_obs

        self.obs_dim = inferred_obs_dim
        self.obs_split_dims = inferred_split_dims
        self.actor = Actor(self.obs_split_dims, self.action_dim, self.hidden_dim).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        self.target_actor = deepcopy(self.actor).to(device=self.device, dtype=self.torch_dtype)
        self.critic = MAACAttentionCritic(
            self.obs_split_dims,
            self.n_agents,
            self.action_dim,
            self.hidden_dim,
            attention_heads=self.attention_heads,
        ).to(device=self.device, dtype=self.torch_dtype)
        self.target_critic = deepcopy(self.critic).to(device=self.device, dtype=self.torch_dtype)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config["train"]["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config["train"]["critic_lr"])

        if self.pending_load_path is not None:
            self._load_state(*self.pending_load_path)
            self.pending_load_path = None

        return flattened_obs

    def begin_episode(self, episode):
        self.current_episode = max(1, int(episode))

    def _noise_scale(self):
        mode = self.config.get("train", {}).get("stop_mode", "episode")
        start = float(self.exploration.get("eps_start", 1.0))
        end = float(self.exploration.get("eps_end", 0.03))
        period = max(1, int(self.exploration.get("eps_period", 1)))
        progress = self.action_step if mode == "step" else self.current_episode

        if progress > period:
            return end
        if period == 1:
            return end

        slope = (end - start) / float(period - 1)
        return start + slope * float(progress - 1)

    def select_action(self, obs, evaluate=False):
        self._ensure_models_initialized(obs)
        with torch.no_grad():
            obs_tensor = torch.tensor(self._flatten_obs_batch(obs), dtype=self.torch_dtype, device=self.device)
            raw_actions = self.actor(self._split_actor_obs_tensor(obs_tensor)).cpu().numpy()
        noise_scale = 0.0 if evaluate else self._noise_scale()
        action_noise = np.zeros_like(raw_actions, dtype=self.numpy_dtype)
        if not evaluate:
            action_noise = np.random.normal(0, noise_scale, size=raw_actions.shape).astype(self.numpy_dtype)
        actions = np.clip(raw_actions + action_noise, -1.0, 1.0).astype(self.numpy_dtype)
        self.last_action_info = {
            "raw_mean": float(np.mean(raw_actions)),
            "raw_std": float(np.std(raw_actions)),
            "raw_min": float(np.min(raw_actions)),
            "raw_max": float(np.max(raw_actions)),
            "noise_scale": float(noise_scale),
            "sampled_noise_mean": float(np.mean(action_noise)),
            "sampled_noise_std": float(np.std(action_noise)),
            "final_mean": float(np.mean(actions)),
            "final_std": float(np.std(actions)),
            "final_min": float(np.min(actions)),
            "final_max": float(np.max(actions)),
            "final_abs_mean": float(np.mean(np.abs(actions))),
            "clip_rate": float(np.mean(np.isclose(np.abs(actions), 1.0, atol=1e-6))),
        }
        self.action_step += 1
        return [actions[i] for i in range(self.n_agents)]

    def store_transition(self, obs, actions, rewards, next_obs, dones):
        flat_obs = self._ensure_models_initialized(obs)
        flat_next_obs = self._ensure_models_initialized(next_obs)
        flat_actions = np.asarray(actions, dtype=self.numpy_dtype)
        flat_rewards = np.asarray(rewards, dtype=self.numpy_dtype)
        flat_dones = np.asarray(dones, dtype=np.float32)
        self.buffer.push(flat_obs, flat_actions, flat_rewards, flat_next_obs, flat_dones)

    def _clip_grad_norm(self, parameters):
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

    def update(self, i_episode=None, total_step_count=None, single_eps_critic_cal_record=None):
        if single_eps_critic_cal_record is None:
            single_eps_critic_cal_record = []
        self.last_update_info = {
            "update_performed": False,
            "actor_updated": False,
            "update_step": int(i_episode if i_episode is not None else self.current_episode),
            "buffer_size": len(self.buffer),
            "learning_starts": self.batch_size,
            "batch_size": self.batch_size,
            "policy_delay": 1,
            "l2_reg": 0.0,
            "non_stationary_adam": False,
            "policy_noise": 0.0,
            "noise_clip": 0.0,
            "max_grad_norm": self.max_grad_norm,
        }
        if self.obs_dim is None:
            return None, None, single_eps_critic_cal_record
        if len(self.buffer) < self.batch_size:
            return None, None, single_eps_critic_cal_record

        obs_b, actions_b, rewards_b, next_obs_b, dones_b = self.buffer.sample(self.batch_size)

        obs_all = torch.tensor(np.asarray(obs_b), dtype=self.torch_dtype, device=self.device)
        actions_all = torch.tensor(np.asarray(actions_b), dtype=self.torch_dtype, device=self.device)
        rewards_all = torch.tensor(np.asarray(rewards_b), dtype=self.torch_dtype, device=self.device)
        next_obs_all = torch.tensor(np.asarray(next_obs_b), dtype=self.torch_dtype, device=self.device)
        dones_all = torch.tensor(np.asarray(dones_b), dtype=self.torch_dtype, device=self.device)
        split_obs_all = self._split_all_agent_obs(obs_all)
        split_next_obs_all = self._split_all_agent_obs(next_obs_all)

        critic_losses = []
        actor_losses = []
        q_means = []
        target_q_means = []
        y_means = []
        reward_means = []
        done_means = []

        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            next_actions_all = torch.stack(
                [self.target_actor(self._split_obs_batch(next_obs_all, other_i)) for other_i in range(self.n_agents)],
                dim=1,
            )
        for agent_i in range(self.n_agents):
            rewards = rewards_all[:, agent_i : agent_i + 1]
            dones = dones_all[:, agent_i : agent_i + 1]
            agent_index = torch.full((self.batch_size,), agent_i, dtype=torch.long, device=self.device)

            with torch.no_grad():
                target_q = self.target_critic(split_next_obs_all, next_actions_all, agent_index)
                y = rewards + self.gamma * (1 - dones) * target_q
                target_q_means.append(target_q.detach().mean())
                y_means.append(y.detach().mean())
                reward_means.append(rewards.detach().mean())
                done_means.append(dones.detach().mean())

            current_q, regs = self.critic(
                split_obs_all,
                actions_all,
                agent_index,
                regularize=True,
            )
            q_means.append(current_q.detach().mean())
            critic_loss_i = F.mse_loss(current_q, y)
            if regs:
                critic_loss_i = critic_loss_i + sum(regs)
            critic_losses.append(critic_loss_i.detach())
            critic_loss_i.backward(retain_graph=agent_i < self.n_agents - 1)
        critic_grad_norm = self._grad_norm(self.critic.parameters())
        self._clip_grad_norm(self.critic.parameters())
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        for agent_i in range(self.n_agents):
            policy_actions = []
            for other_i in range(self.n_agents):
                sampled_action = self.actor(self._split_obs_batch(obs_all, other_i))
                if other_i != agent_i:
                    sampled_action = sampled_action.detach()
                policy_actions.append(sampled_action)
            policy_actions_all = torch.stack(policy_actions, dim=1)
            agent_index = torch.full((self.batch_size,), agent_i, dtype=torch.long, device=self.device)
            actor_loss_i = -self.critic(
                split_obs_all,
                policy_actions_all,
                agent_index,
            ).mean()
            actor_losses.append(actor_loss_i.detach())
            actor_loss_i.backward(retain_graph=agent_i < self.n_agents - 1)
        actor_grad_norm = self._grad_norm(self.actor.parameters())
        self._clip_grad_norm(self.actor.parameters())
        self.actor_optimizer.step()

        update_index = self.current_episode if i_episode is None else i_episode
        if update_index % self.update_every == 0:
            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic, self.critic, self.tau)

        avg_critic_loss = torch.stack(critic_losses).mean()
        avg_actor_loss = torch.stack(actor_losses).mean()
        self.last_update_info = {
            "update_performed": True,
            "actor_updated": True,
            "update_step": int(i_episode if i_episode is not None else self.current_episode),
            "buffer_size": len(self.buffer),
            "learning_starts": self.batch_size,
            "batch_size": self.batch_size,
            "policy_delay": 1,
            "l2_reg": 0.0,
            "non_stationary_adam": False,
            "policy_noise": 0.0,
            "noise_clip": 0.0,
            "max_grad_norm": self.max_grad_norm,
            "critic_loss": float(avg_critic_loss.detach().cpu().item()),
            "actor_loss": float(avg_actor_loss.detach().cpu().item()),
            "q1_mean": self._mean_tensor_value(q_means),
            "q2_mean": None,
            "target_q1_mean": self._mean_tensor_value(target_q_means),
            "target_q2_mean": None,
            "target_min_q_mean": self._mean_tensor_value(target_q_means),
            "target_y_mean": self._mean_tensor_value(y_means),
            "reward_batch_mean": self._mean_tensor_value(reward_means),
            "done_batch_mean": self._mean_tensor_value(done_means),
            "target_twin_gap_abs_mean": None,
            "critic1_grad_norm": critic_grad_norm,
            "critic2_grad_norm": None,
            "actor_grad_norm": actor_grad_norm,
        }
        return [avg_critic_loss], [avg_actor_loss], single_eps_critic_cal_record

    def save(self, path, episode=None, step=None, stop_mode=None):
        if self.actor is None or self.critic is None:
            return
        if stop_mode == "step":
            episode = None
        elif stop_mode == "episode":
            step = None
        elif stop_mode is not None:
            raise ValueError("Unsupported stop_mode: {}. Expected 'step' or 'episode'.".format(stop_mode))

        os.makedirs(path, exist_ok=True)
        actor_path = os.path.join(path, "maac_actor.pt")
        critic_path = os.path.join(path, "maac_critic.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        if episode is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, f"maac_actor_ep{int(episode)}.pt"))
            torch.save(self.critic.state_dict(), os.path.join(path, f"maac_critic_ep{int(episode)}.pt"))
        if step is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, f"maac_actor_step{int(step)}.pt"))
            torch.save(self.critic.state_dict(), os.path.join(path, f"maac_critic_step{int(step)}.pt"))

    def _load_state(self, path, checkpoint_tag=None):
        actor_name = "maac_actor.pt" if checkpoint_tag is None else f"maac_actor_{checkpoint_tag}.pt"
        critic_name = "maac_critic.pt" if checkpoint_tag is None else f"maac_critic_{checkpoint_tag}.pt"
        self.actor.load_state_dict(torch.load(os.path.join(path, actor_name), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, critic_name), map_location=self.device))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def load(self, path, checkpoint_tag=None):
        if self.actor is None or self.critic is None:
            self.pending_load_path = (path, checkpoint_tag)
            return
        self._load_state(path, checkpoint_tag)
