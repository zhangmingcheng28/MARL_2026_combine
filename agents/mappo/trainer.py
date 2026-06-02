import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from agents.base_trainer import BaseTrainer
from agents.mappo.networks import Actor, CentralValueCritic


class MAPPORolloutBuffer:
    def __init__(self):
        self.buffer = []

    def push(self, obs, actions, rewards, next_obs, dones, log_probs, values):
        self.buffer.append((obs, actions, rewards, next_obs, dones, log_probs, values))

    def clear(self):
        self.buffer.clear()

    def unpack(self):
        return map(list, zip(*self.buffer))

    def __len__(self):
        return len(self.buffer)


class MAPPOTrainer(BaseTrainer):
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
        self.batch_size = config["train"]["batch_size"]
        self.rollout_size = int(config["train"].get("mappo_rollout_size", self.batch_size))
        self.ppo_epochs = int(config["train"].get("mappo_ppo_epochs", 4))
        self.clip_ratio = float(config["train"].get("mappo_clip_ratio", 0.2))
        self.value_coef = float(config["train"].get("mappo_value_coef", 0.5))
        self.entropy_coef = float(config["train"].get("mappo_entropy_coef", 0.01))
        self.update_every = int(config["train"].get("update_every", 1))
        self.max_grad_norm = float(config["train"].get("max_grad_norm", 0.0))

        self.obs_dim = None
        self.obs_split_dims = None

        self.actor = None
        self.critic = None
        self.log_std = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.current_episode = 1
        self.action_step = 0
        self.pending_load_path = None
        self.pending_action_meta = deque()

        self.buffer = MAPPORolloutBuffer()
        self.last_action_info = {}
        self.last_update_info = {
            "update_performed": False,
            "actor_updated": False,
            "update_step": 0,
            "buffer_size": 0,
            "learning_starts": self.rollout_size,
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

    def _clip_grad_norm(self, parameters):
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

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

    def _critic_values(self, obs_all):
        split_obs_all = self._split_all_agent_obs(obs_all)
        values = []
        batch_size = obs_all.size(0)
        for agent_i in range(self.n_agents):
            agent_index = torch.full((batch_size,), agent_i, dtype=torch.long, device=self.device)
            values.append(self.critic(split_obs_all, agent_index))
        return torch.cat(values, dim=1)

    def _policy_std(self):
        return self.log_std.exp().view(1, 1, -1)

    def _ensure_models_initialized(self, obs):
        flattened_obs = self._flatten_obs_batch(obs)
        inferred_obs_dim = int(flattened_obs.shape[1])
        inferred_split_dims = self._extract_obs_split_dims(obs)

        if self.obs_dim is not None:
            if inferred_obs_dim != self.obs_dim:
                raise ValueError(
                    "MAPPO observation dimension changed from {} to {}.".format(
                        self.obs_dim,
                        inferred_obs_dim,
                    )
                )
            if inferred_split_dims != self.obs_split_dims:
                raise ValueError(
                    "MAPPO observation split changed from {} to {}.".format(
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
        self.critic = CentralValueCritic(self.obs_split_dims, self.n_agents, self.hidden_dim).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        initial_std = max(1e-3, float(self.config.get("exploration", {}).get("initial_noise_sigma", 0.2)))
        self.log_std = torch.nn.Parameter(
            torch.full(
                (self.action_dim,),
                float(np.log(initial_std)),
                dtype=self.torch_dtype,
                device=self.device,
            )
        )

        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + [self.log_std],
            lr=self.config["train"]["actor_lr"],
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config["train"]["critic_lr"])

        if self.pending_load_path is not None:
            self._load_state(*self.pending_load_path)
            self.pending_load_path = None

        return flattened_obs

    def begin_episode(self, episode):
        self.current_episode = max(1, int(episode))

    def _noise_scale(self):
        if self.log_std is None:
            return None
        return float(self.log_std.exp().mean().detach().cpu().item())

    def select_action(self, obs, evaluate=False):
        self._ensure_models_initialized(obs)
        with torch.no_grad():
            obs_tensor = torch.tensor(self._flatten_obs_batch(obs), dtype=self.torch_dtype, device=self.device)
            action_mean = self.actor(self._split_actor_obs_tensor(obs_tensor))
            action_std = self.log_std.exp().view(1, -1).expand_as(action_mean)
            if evaluate:
                sampled_actions = action_mean
                log_probs = None
                values = None
            else:
                dist = Normal(action_mean, action_std)
                sampled_actions = dist.sample()
                clipped_actions = torch.clamp(sampled_actions, -1.0, 1.0)
                log_probs = dist.log_prob(clipped_actions).sum(dim=-1)
                values = self._critic_values(obs_tensor.unsqueeze(0)).squeeze(0)
                sampled_actions = clipped_actions

        actions = torch.clamp(sampled_actions, -1.0, 1.0).cpu().numpy().astype(self.numpy_dtype)
        if not evaluate:
            self.pending_action_meta.append(
                {
                    "log_probs": log_probs.cpu().numpy().astype(self.numpy_dtype),
                    "values": values.cpu().numpy().astype(self.numpy_dtype),
                }
            )
        self.last_action_info = {
            "raw_mean": float(np.mean(action_mean.detach().cpu().numpy())),
            "raw_std": float(np.std(action_mean.detach().cpu().numpy())),
            "raw_min": float(np.min(action_mean.detach().cpu().numpy())),
            "raw_max": float(np.max(action_mean.detach().cpu().numpy())),
            "noise_scale": self._noise_scale(),
            "sampled_noise_mean": float(np.mean(actions - action_mean.detach().cpu().numpy())),
            "sampled_noise_std": float(np.std(actions - action_mean.detach().cpu().numpy())),
            "final_mean": float(np.mean(actions)),
            "final_std": float(np.std(actions)),
            "final_min": float(np.min(actions)),
            "final_max": float(np.max(actions)),
            "final_abs_mean": float(np.mean(np.abs(actions))),
            "clip_rate": float(np.mean(np.isclose(np.abs(actions), 1.0, atol=1e-6))),
            "policy_std_mean": self._noise_scale(),
        }
        self.action_step += 1
        return [actions[i] for i in range(self.n_agents)]

    def store_transition(self, obs, actions, rewards, next_obs, dones):
        flat_obs = self._ensure_models_initialized(obs)
        flat_next_obs = self._ensure_models_initialized(next_obs)
        if not self.pending_action_meta:
            raise ValueError("MAPPO action metadata queue is empty during store_transition.")
        action_meta = self.pending_action_meta.popleft()
        flat_actions = np.asarray(actions, dtype=self.numpy_dtype)
        flat_rewards = np.asarray(rewards, dtype=self.numpy_dtype)
        flat_dones = np.asarray(dones, dtype=np.float32)
        self.buffer.push(
            flat_obs,
            flat_actions,
            flat_rewards,
            flat_next_obs,
            flat_dones,
            action_meta["log_probs"],
            action_meta["values"],
        )

    def update(self, i_episode=None, total_step_count=None, single_eps_critic_cal_record=None):
        if single_eps_critic_cal_record is None:
            single_eps_critic_cal_record = []

        update_index = int(i_episode if i_episode is not None else self.current_episode)
        self.last_update_info = {
            "update_performed": False,
            "actor_updated": False,
            "update_step": update_index,
            "buffer_size": len(self.buffer),
            "learning_starts": self.rollout_size,
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
        if len(self.buffer) < self.rollout_size:
            return None, None, single_eps_critic_cal_record
        if update_index % self.update_every != 0:
            return None, None, single_eps_critic_cal_record

        obs_b, actions_b, rewards_b, next_obs_b, dones_b, old_log_probs_b, old_values_b = self.buffer.unpack()
        self.buffer.clear()

        obs_all = torch.tensor(np.asarray(obs_b), dtype=self.torch_dtype, device=self.device)
        actions_all = torch.tensor(np.asarray(actions_b), dtype=self.torch_dtype, device=self.device)
        rewards_all = torch.tensor(np.asarray(rewards_b), dtype=self.torch_dtype, device=self.device)
        next_obs_all = torch.tensor(np.asarray(next_obs_b), dtype=self.torch_dtype, device=self.device)
        dones_all = torch.tensor(np.asarray(dones_b), dtype=self.torch_dtype, device=self.device)
        old_log_probs_all = torch.tensor(np.asarray(old_log_probs_b), dtype=self.torch_dtype, device=self.device)
        old_values_all = torch.tensor(np.asarray(old_values_b), dtype=self.torch_dtype, device=self.device)

        with torch.no_grad():
            next_values_all = self._critic_values(next_obs_all)
            returns_all = rewards_all + self.gamma * (1 - dones_all) * next_values_all
            advantages_all = returns_all - old_values_all
            advantage_mean = advantages_all.mean()
            advantage_std = advantages_all.std(unbiased=False)
            if float(advantage_std.detach().cpu().item()) > 1e-8:
                advantages_all = (advantages_all - advantage_mean) / (advantage_std + 1e-8)

        actor_losses = []
        critic_losses = []
        entropy_values = []
        ratio_values = []
        value_means = []
        return_means = []
        advantage_means = []
        actor_grad_norm = None
        critic_grad_norm = None

        batch_count = obs_all.size(0)
        minibatch_size = min(self.batch_size, batch_count)

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(batch_count, device=self.device)
            for start in range(0, batch_count, minibatch_size):
                batch_index = permutation[start:start + minibatch_size]
                batch_obs = obs_all[batch_index]
                batch_actions = torch.clamp(actions_all[batch_index], -1.0, 1.0)
                batch_old_log_probs = old_log_probs_all[batch_index]
                batch_returns = returns_all[batch_index]
                batch_advantages = advantages_all[batch_index]

                policy_means = []
                for agent_i in range(self.n_agents):
                    policy_means.append(self.actor(self._split_obs_batch(batch_obs, agent_i)))
                policy_means = torch.stack(policy_means, dim=1)
                policy_std = self._policy_std().expand_as(policy_means)
                dist = Normal(policy_means, policy_std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped_objective = ratios * batch_advantages
                clipped_objective = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                actor_loss = -(torch.min(unclipped_objective, clipped_objective).mean() + self.entropy_coef * entropy.mean())

                current_values = self._critic_values(batch_obs)
                critic_loss = self.value_coef * F.mse_loss(current_values, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = self._grad_norm(list(self.actor.parameters()) + [self.log_std])
                self._clip_grad_norm(list(self.actor.parameters()) + [self.log_std])
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = self._grad_norm(self.critic.parameters())
                self._clip_grad_norm(self.critic.parameters())
                self.critic_optimizer.step()

                actor_losses.append(actor_loss.detach())
                critic_losses.append(critic_loss.detach())
                entropy_values.append(entropy.mean().detach())
                ratio_values.append(ratios.mean().detach())
                value_means.append(current_values.mean().detach())
                return_means.append(batch_returns.mean().detach())
                advantage_means.append(batch_advantages.mean().detach())

        avg_actor_loss = torch.stack(actor_losses).mean()
        avg_critic_loss = torch.stack(critic_losses).mean()
        self.last_update_info = {
            "update_performed": True,
            "actor_updated": True,
            "update_step": update_index,
            "buffer_size": 0,
            "learning_starts": self.rollout_size,
            "batch_size": self.batch_size,
            "policy_delay": 1,
            "l2_reg": 0.0,
            "non_stationary_adam": False,
            "policy_noise": 0.0,
            "noise_clip": 0.0,
            "max_grad_norm": self.max_grad_norm,
            "critic_loss": float(avg_critic_loss.detach().cpu().item()),
            "actor_loss": float(avg_actor_loss.detach().cpu().item()),
            "q1_mean": self._mean_tensor_value(value_means),
            "q2_mean": None,
            "target_q1_mean": None,
            "target_q2_mean": None,
            "target_min_q_mean": None,
            "target_y_mean": self._mean_tensor_value(return_means),
            "reward_batch_mean": float(rewards_all.mean().detach().cpu().item()),
            "done_batch_mean": float(dones_all.mean().detach().cpu().item()),
            "target_twin_gap_abs_mean": None,
            "critic1_grad_norm": critic_grad_norm,
            "critic2_grad_norm": None,
            "actor_grad_norm": actor_grad_norm,
            "entropy_mean": self._mean_tensor_value(entropy_values),
            "ratio_mean": self._mean_tensor_value(ratio_values),
            "advantage_mean": self._mean_tensor_value(advantage_means),
            "value_mean": self._mean_tensor_value(value_means),
        }
        return [avg_critic_loss], [avg_actor_loss], single_eps_critic_cal_record

    def save(self, path, episode=None, step=None, stop_mode=None):
        if self.actor is None or self.critic is None or self.log_std is None:
            return
        if stop_mode == "step":
            episode = None
        elif stop_mode == "episode":
            step = None
        elif stop_mode is not None:
            raise ValueError("Unsupported stop_mode: {}. Expected 'step' or 'episode'.".format(stop_mode))

        os.makedirs(path, exist_ok=True)
        actor_payload = {
            "actor_state_dict": self.actor.state_dict(),
            "log_std": self.log_std.detach().cpu(),
        }
        critic_payload = {
            "critic_state_dict": self.critic.state_dict(),
        }
        torch.save(actor_payload, os.path.join(path, "mappo_actor.pt"))
        torch.save(critic_payload, os.path.join(path, "mappo_critic.pt"))
        if episode is not None:
            torch.save(actor_payload, os.path.join(path, f"mappo_actor_ep{int(episode)}.pt"))
            torch.save(critic_payload, os.path.join(path, f"mappo_critic_ep{int(episode)}.pt"))
        if step is not None:
            torch.save(actor_payload, os.path.join(path, f"mappo_actor_step{int(step)}.pt"))
            torch.save(critic_payload, os.path.join(path, f"mappo_critic_step{int(step)}.pt"))

    def _load_state(self, path, checkpoint_tag=None):
        actor_name = "mappo_actor.pt" if checkpoint_tag is None else f"mappo_actor_{checkpoint_tag}.pt"
        critic_name = "mappo_critic.pt" if checkpoint_tag is None else f"mappo_critic_{checkpoint_tag}.pt"
        actor_payload = torch.load(os.path.join(path, actor_name), map_location=self.device)
        critic_payload = torch.load(os.path.join(path, critic_name), map_location=self.device)

        actor_state_dict = actor_payload.get("actor_state_dict", actor_payload)
        critic_state_dict = critic_payload.get("critic_state_dict", critic_payload)

        self.actor.load_state_dict(actor_state_dict)
        self.critic.load_state_dict(critic_state_dict)
        if "log_std" in actor_payload:
            self.log_std.data.copy_(actor_payload["log_std"].to(device=self.device, dtype=self.torch_dtype))

    def load(self, path, checkpoint_tag=None):
        if self.actor is None or self.critic is None or self.log_std is None:
            self.pending_load_path = (path, checkpoint_tag)
            return
        self._load_state(path, checkpoint_tag)
