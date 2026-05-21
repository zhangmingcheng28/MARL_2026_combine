import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agents.base_trainer import BaseTrainer
from agents.common.buffer import ReplayBuffer
from agents.common.utils import soft_update
from agents.matd3.networks import Actor, AttentionCentralCritic, EmbeddedCentralCritic


class MATD3Trainer(BaseTrainer):
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
        self.exploration = config.get("exploration", {})
        self.flags = config.get("flags", {})
        self.use_critic_attention = (
            self.flags.get("use_critic_attention", False)
            or config.get("algorithm", "").lower() == "matd3-critic-attention"
        )

        self.policy_noise = float(config["train"].get("policy_noise", 0.2))
        self.noise_clip = float(config["train"].get("noise_clip", 0.5))
        self.policy_delay = max(1, int(config["train"].get("policy_delay", 2)))
        self.l2_reg = float(config["train"].get("matd3_l2_reg", 0.0))
        self.non_stationary_adam = bool(config["train"].get("matd3_non_stationary_adam", False))
        self.learning_starts = max(
            self.batch_size,
            int(config["train"].get("learning_starts", self.batch_size)),
        )
        self.max_grad_norm = float(config["train"].get("max_grad_norm", 10.0))
        self.update_step = 0

        self.obs_dim = None
        self.obs_split_dims = None

        self.actor = None
        self.target_actor = None
        self.critic1 = None
        self.critic2 = None
        self.target_critic1 = None
        self.target_critic2 = None
        self.actor_optimizer = None
        self.critic1_optimizer = None
        self.critic2_optimizer = None
        self.current_episode = 1
        self.action_step = 0
        self.pending_load_path = None

        self.buffer = ReplayBuffer(config["train"]["buffer_size"])
        self.last_action_info = {}
        self.last_update_info = {
            "update_performed": False,
            "update_step": self.update_step,
            "buffer_size": 0,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
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
                    "MATD3 observation dimension changed from {} to {}.".format(
                        self.obs_dim,
                        inferred_obs_dim,
                    )
                )
            if inferred_split_dims != self.obs_split_dims:
                raise ValueError(
                    "MATD3 observation split changed from {} to {}.".format(
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

        critic_cls = AttentionCentralCritic if self.use_critic_attention else EmbeddedCentralCritic
        self.critic1 = critic_cls(self.obs_split_dims, self.n_agents, self.action_dim, self.hidden_dim).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        self.critic2 = critic_cls(self.obs_split_dims, self.n_agents, self.action_dim, self.hidden_dim).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        self.target_critic1 = deepcopy(self.critic1).to(device=self.device, dtype=self.torch_dtype)
        self.target_critic2 = deepcopy(self.critic2).to(device=self.device, dtype=self.torch_dtype)

        optimizer_kwargs = {
            "weight_decay": self.l2_reg,
            "amsgrad": self.non_stationary_adam,
        }
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.config["train"]["actor_lr"],
            **optimizer_kwargs,
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=self.config["train"]["critic_lr"],
            **optimizer_kwargs,
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=self.config["train"]["critic_lr"],
            **optimizer_kwargs,
        )

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

    def update(self, i_episode=None, total_step_count=None, single_eps_critic_cal_record=None):
        if single_eps_critic_cal_record is None:
            single_eps_critic_cal_record = []
        self.last_update_info = {
            "update_performed": False,
            "update_step": self.update_step,
            "buffer_size": len(self.buffer),
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
        }
        if self.obs_dim is None:
            return None, None, single_eps_critic_cal_record
        if len(self.buffer) < self.learning_starts:
            return None, None, single_eps_critic_cal_record

        self.update_step += 1

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
        q1_means = []
        q2_means = []
        target_q1_means = []
        target_q2_means = []
        target_min_q_means = []
        y_means = []
        reward_means = []
        done_means = []
        twin_gap_means = []

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        for agent_i in range(self.n_agents):
            rewards = rewards_all[:, agent_i : agent_i + 1]
            dones = dones_all[:, agent_i : agent_i + 1]
            agent_index = torch.full(
                (self.batch_size,),
                agent_i,
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                next_actions_list = []
                for other_i in range(self.n_agents):
                    target_action = self.target_actor(self._split_obs_batch(next_obs_all, other_i))
                    noise = (torch.randn_like(target_action) * self.policy_noise).clamp(
                        -self.noise_clip,
                        self.noise_clip,
                    )
                    next_actions_list.append((target_action + noise).clamp(-1.0, 1.0))
                next_actions_all = torch.stack(next_actions_list, dim=1)
                target_q1 = self.target_critic1(split_next_obs_all, next_actions_all, agent_index)
                target_q2 = self.target_critic2(split_next_obs_all, next_actions_all, agent_index)
                target_q1 = target_q1.clamp(-40, 40)
                target_q2 = target_q2.clamp(-40, 40)
                # Target-Q ablation: keep exactly one target_q line active.
                # A: standard TD3 clipped double-Q target.
                target_q = torch.min(target_q1, target_q2)
                # B: single-critic target, closer to DDPG/MADDPG.
                # target_q = target_q1
                # C: averaged twin-critic target, less pessimistic than min-Q.
                # target_q = 0.5 * (target_q1 + target_q2)

                y = rewards + self.gamma * (1 - dones) * target_q
                target_q1_means.append(target_q1.detach().mean())
                target_q2_means.append(target_q2.detach().mean())
                target_min_q_means.append(target_q.detach().mean())
                y_means.append(y.detach().mean())
                reward_means.append(rewards.detach().mean())
                done_means.append(dones.detach().mean())
                twin_gap_means.append((target_q1 - target_q2).detach().abs().mean())

            current_q1 = self.critic1(split_obs_all, actions_all, agent_index)
            current_q2 = self.critic2(split_obs_all, actions_all, agent_index)
            q1_means.append(current_q1.detach().mean())
            q2_means.append(current_q2.detach().mean())
            critic1_loss_i = F.mse_loss(current_q1, y)
            critic2_loss_i = F.mse_loss(current_q2, y)
            critic_losses.extend([critic1_loss_i.detach(), critic2_loss_i.detach()])
            critic1_loss_i.backward()
            critic2_loss_i.backward()
        critic1_grad_norm = self._grad_norm(self.critic1.parameters())
        critic2_grad_norm = self._grad_norm(self.critic2.parameters())
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        actor_grad_norm = None
        if self.update_step % self.policy_delay == 0:
            for parameter in self.critic1.parameters():
                parameter.requires_grad = False
            try:
                self.actor_optimizer.zero_grad()
                for agent_i in range(self.n_agents):
                    current_obs_i = self._split_obs_batch(obs_all, agent_i)
                    mixed_actions = actions_all.detach().clone()
                    agent_actions = self.actor(current_obs_i)
                    mixed_actions[:, agent_i, :] = agent_actions
                    agent_index = torch.full(
                        (self.batch_size,),
                        agent_i,
                        dtype=torch.long,
                        device=self.device,
                    )
                    actor_loss_i = -self.critic1(
                        split_obs_all,
                        mixed_actions,
                        agent_index,
                    ).mean()
                    actor_losses.append(actor_loss_i.detach())
                    actor_loss_i.backward()
                actor_grad_norm = self._grad_norm(self.actor.parameters())
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
            finally:
                for parameter in self.critic1.parameters():
                    parameter.requires_grad = True

            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic1, self.critic1, self.tau)
            soft_update(self.target_critic2, self.critic2, self.tau)

        avg_critic_loss = torch.stack(critic_losses).mean()
        avg_actor_loss = torch.stack(actor_losses).mean() if actor_losses else None
        self.last_update_info = {
            "update_performed": True,
            "actor_updated": bool(actor_losses),
            "update_step": self.update_step,
            "buffer_size": len(self.buffer),
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "policy_delay": self.policy_delay,
            "l2_reg": self.l2_reg,
            "non_stationary_adam": self.non_stationary_adam,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
            "max_grad_norm": self.max_grad_norm,
            "critic_loss": float(avg_critic_loss.detach().cpu().item()),
            "actor_loss": float(avg_actor_loss.detach().cpu().item()) if avg_actor_loss is not None else None,
            "q1_mean": self._mean_tensor_value(q1_means),
            "q2_mean": self._mean_tensor_value(q2_means),
            "target_q1_mean": self._mean_tensor_value(target_q1_means),
            "target_q2_mean": self._mean_tensor_value(target_q2_means),
            "target_min_q_mean": self._mean_tensor_value(target_min_q_means),
            "target_y_mean": self._mean_tensor_value(y_means),
            "reward_batch_mean": self._mean_tensor_value(reward_means),
            "done_batch_mean": self._mean_tensor_value(done_means),
            "target_twin_gap_abs_mean": self._mean_tensor_value(twin_gap_means),
            "critic1_grad_norm": critic1_grad_norm,
            "critic2_grad_norm": critic2_grad_norm,
            "actor_grad_norm": actor_grad_norm,
        }
        if actor_losses:
            return [avg_critic_loss], [avg_actor_loss], single_eps_critic_cal_record
        return [avg_critic_loss], [], single_eps_critic_cal_record

    def save(self, path, episode=None, step=None, stop_mode=None):
        if self.actor is None or self.critic1 is None or self.critic2 is None:
            return

        if stop_mode == "step":
            episode = None
        elif stop_mode == "episode":
            step = None
        elif stop_mode is not None:
            raise ValueError("Unsupported stop_mode: {}. Expected 'step' or 'episode'.".format(stop_mode))

        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "matd3_actor.pt"))
        torch.save(self.critic1.state_dict(), os.path.join(path, "matd3_critic1.pt"))
        torch.save(self.critic2.state_dict(), os.path.join(path, "matd3_critic2.pt"))

        if episode is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, f"matd3_actor_ep{int(episode)}.pt"))
            torch.save(self.critic1.state_dict(), os.path.join(path, f"matd3_critic1_ep{int(episode)}.pt"))
            torch.save(self.critic2.state_dict(), os.path.join(path, f"matd3_critic2_ep{int(episode)}.pt"))
        if step is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, f"matd3_actor_step{int(step)}.pt"))
            torch.save(self.critic1.state_dict(), os.path.join(path, f"matd3_critic1_step{int(step)}.pt"))
            torch.save(self.critic2.state_dict(), os.path.join(path, f"matd3_critic2_step{int(step)}.pt"))

    def _load_state(self, path, checkpoint_tag=None):
        actor_name = "matd3_actor.pt" if checkpoint_tag is None else f"matd3_actor_{checkpoint_tag}.pt"
        critic1_name = "matd3_critic1.pt" if checkpoint_tag is None else f"matd3_critic1_{checkpoint_tag}.pt"
        critic2_name = "matd3_critic2.pt" if checkpoint_tag is None else f"matd3_critic2_{checkpoint_tag}.pt"
        self.actor.load_state_dict(torch.load(os.path.join(path, actor_name), map_location=self.device))
        self.critic1.load_state_dict(torch.load(os.path.join(path, critic1_name), map_location=self.device))
        self.critic2.load_state_dict(torch.load(os.path.join(path, critic2_name), map_location=self.device))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def load(self, path, checkpoint_tag=None):
        if self.actor is None or self.critic1 is None or self.critic2 is None:
            self.pending_load_path = (path, checkpoint_tag)
            return
        self._load_state(path, checkpoint_tag)
