import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_trainer import BaseTrainer
from agents.common.utils import soft_update
from agents.learned_pcb_biased_att_iddpg.buffer import Experience, ReplayMemory
from agents.learned_pcb_biased_att_iddpg.networks import (
    ActorNetworkVariableNeiWRadar,
    CriticNetworkVariableNeiWRadar,
)


class LearnedPCBBiasedATTIDDPGTrainer(BaseTrainer):
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
        self.beta = float(config["train"].get("beta", 1.0))
        self.flags = config.get("flags", {})
        self.exploration = config.get("exploration", {})

        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self.buffer = ReplayMemory(config["train"]["buffer_size"])
        self.update_step = 0
        self.action_step = 0
        self.current_episode = 1
        self.pending_load_path = None
        self.pending_eval_load_path = None
        self.max_grad_norm = float(config["train"].get("max_grad_norm", 0.0))
        self.own_dim = None
        self.radar_dim = None
        self.neighbor_token_dim = None
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

    def _require_supported_configuration(self):
        if self.flags.get("full_observable_critic"):
            raise NotImplementedError("learned_pcb_biased_att_iddpg only supports non-full-observable critic mode.")
        if self.flags.get("use_gru"):
            raise NotImplementedError("learned_pcb_biased_att_iddpg does not support GRU mode yet.")
        if self.flags.get("own_obs_only"):
            raise NotImplementedError("learned_pcb_biased_att_iddpg does not support own_obs_only mode.")

    def _extract_dims(self, obs):
        if len(obs) < 4:
            raise ValueError("learned_pcb_biased_att_iddpg expects four observation portions: own obs, fixed neighbors, radar, variable neighbors.")
        own_dim = len(obs[0][0])
        radar_dim = len(obs[2][0])
        sample_neighbor = np.asarray(obs[3][0], dtype=self.numpy_dtype)
        if sample_neighbor.ndim == 3 and sample_neighbor.shape[1] == 1:
            sample_neighbor = np.squeeze(sample_neighbor, axis=1)
        if sample_neighbor.ndim == 1:
            sample_neighbor = sample_neighbor.reshape(1, -1)
        neighbor_token_dim = int(sample_neighbor.shape[-1])
        return own_dim, radar_dim, neighbor_token_dim

    def _ensure_models_initialized(self, obs):
        if self.actor is not None:
            return

        self._require_supported_configuration()
        self.own_dim, self.radar_dim, self.neighbor_token_dim = self._extract_dims(obs)

        self.actor = ActorNetworkVariableNeiWRadar(
            self.own_dim,
            self.radar_dim,
            self.neighbor_token_dim,
            self.action_dim,
            beta=self.beta,
        ).to(device=self.device, dtype=self.torch_dtype)
        self.actor_target = deepcopy(self.actor).to(device=self.device, dtype=self.torch_dtype)
        self.critic = CriticNetworkVariableNeiWRadar(
            self.own_dim,
            self.radar_dim,
            self.neighbor_token_dim,
            self.action_dim,
            beta=self.beta,
        ).to(device=self.device, dtype=self.torch_dtype)
        self.critic_target = deepcopy(self.critic).to(device=self.device, dtype=self.torch_dtype)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config["train"]["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config["train"]["critic_lr"])

        if self.pending_load_path is not None:
            pending_path, pending_checkpoint_tag = self.pending_load_path
            self._load_state(pending_path, pending_checkpoint_tag)
            self.pending_load_path = None
        if self.pending_eval_load_path is not None:
            pending_path, pending_checkpoint_tag = self.pending_eval_load_path
            self._load_actor_state(pending_path, pending_checkpoint_tag)
            self.pending_eval_load_path = None

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

    def _normalize_neighbor_array(self, neighbor_obs):
        neighbor_array = np.asarray(neighbor_obs, dtype=self.numpy_dtype)
        if neighbor_array.ndim == 3 and neighbor_array.shape[1] == 1:
            neighbor_array = np.squeeze(neighbor_array, axis=1)
        if neighbor_array.ndim == 1:
            neighbor_array = neighbor_array.reshape(1, -1)
        return neighbor_array.astype(self.numpy_dtype)

    def _pad_neighbor_batch(self, neighbor_batch):
        tensors = []
        valid_masks = []
        max_neighbors = 1
        for neighbor_array in neighbor_batch:
            tensor = torch.tensor(self._normalize_neighbor_array(neighbor_array), dtype=self.torch_dtype, device=self.device)
            valid_mask = tensor.abs().sum(dim=-1) > 0
            tensors.append(tensor)
            valid_masks.append(valid_mask)
            max_neighbors = max(max_neighbors, int(tensor.size(0)))

        padded = torch.zeros(
            (len(tensors), max_neighbors, self.neighbor_token_dim),
            dtype=self.torch_dtype,
            device=self.device,
        )
        padded_mask = torch.zeros((len(tensors), max_neighbors), dtype=torch.bool, device=self.device)
        for batch_idx, tensor in enumerate(tensors):
            length = int(tensor.size(0))
            padded[batch_idx, :length] = tensor
            padded_mask[batch_idx, :length] = valid_masks[batch_idx]
        return padded, padded_mask

    def select_action(self, obs, evaluate=False):
        self._ensure_models_initialized(obs)

        own_tensor = torch.tensor(np.stack(obs[0]), dtype=self.torch_dtype, device=self.device)
        radar_tensor = torch.tensor(np.stack(obs[2]), dtype=self.torch_dtype, device=self.device)

        actions = []
        raw_actions = []
        noise_samples = []
        with torch.no_grad():
            for i in range(self.n_agents):
                neighbor_tokens, neighbor_mask = self._pad_neighbor_batch([obs[3][i]])
                action = self.actor(
                    [
                        own_tensor[i].unsqueeze(0),
                        radar_tensor[i].unsqueeze(0),
                        neighbor_tokens,
                        neighbor_mask,
                    ]
                ).cpu().numpy()[0]
                raw_action = action.copy()
                noise = np.zeros(self.action_dim, dtype=self.numpy_dtype)
                if not evaluate:
                    noise = np.random.normal(0, self._noise_scale(), size=self.action_dim).astype(self.numpy_dtype)
                    action += noise
                clipped_action = np.clip(action, -1.0, 1.0).astype(self.numpy_dtype)
                raw_actions.append(raw_action.astype(self.numpy_dtype))
                noise_samples.append(noise)
                actions.append(clipped_action)

        raw_actions_arr = np.asarray(raw_actions, dtype=self.numpy_dtype)
        noise_arr = np.asarray(noise_samples, dtype=self.numpy_dtype)
        actions_arr = np.asarray(actions, dtype=self.numpy_dtype)
        self.last_action_info = {
            "raw_mean": float(np.mean(raw_actions_arr)),
            "raw_std": float(np.std(raw_actions_arr)),
            "raw_min": float(np.min(raw_actions_arr)),
            "raw_max": float(np.max(raw_actions_arr)),
            "noise_scale": float(0.0 if evaluate else self._noise_scale()),
            "sampled_noise_mean": float(np.mean(noise_arr)),
            "sampled_noise_std": float(np.std(noise_arr)),
            "final_mean": float(np.mean(actions_arr)),
            "final_std": float(np.std(actions_arr)),
            "final_min": float(np.min(actions_arr)),
            "final_max": float(np.max(actions_arr)),
            "final_abs_mean": float(np.mean(np.abs(actions_arr))),
            "clip_rate": float(np.mean(np.isclose(np.abs(actions_arr), 1.0, atol=1e-6))),
        }
        self.action_step += 1
        return actions

    def store_transition(self, obs, actions, rewards, next_obs, dones, history=None, cur_hidden=None, next_hidden=None):
        self._ensure_models_initialized(obs)

        reward_array = np.asarray(rewards, dtype=self.numpy_dtype)
        done_array = np.asarray([int(value) for value in dones], dtype=self.numpy_dtype)
        action_array = np.asarray(actions, dtype=self.numpy_dtype)

        for i in range(self.n_agents):
            self.buffer.push(
                np.asarray(obs[0][i], dtype=self.numpy_dtype),
                np.asarray(obs[2][i], dtype=self.numpy_dtype),
                self._normalize_neighbor_array(obs[3][i]),
                np.asarray(action_array[i], dtype=self.numpy_dtype),
                np.asarray(next_obs[0][i], dtype=self.numpy_dtype),
                np.asarray(next_obs[2][i], dtype=self.numpy_dtype),
                self._normalize_neighbor_array(next_obs[3][i]),
                float(reward_array[i]),
                float(done_array[i]),
            )

    def update(self, i_episode=None, total_step_count=None, single_eps_critic_cal_record=None):
        if self.actor is None:
            return None
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
        if len(self.buffer) <= self.batch_size:
            return None, None, single_eps_critic_cal_record

        self._require_supported_configuration()

        if i_episode is None:
            i_episode = self.current_episode

        c_loss = []
        a_loss = []

        transitions = self.buffer.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        stacked_obs = torch.tensor(np.array(batch.states_obs), dtype=self.torch_dtype, device=self.device)
        stacked_grid = torch.tensor(np.array(batch.states_grid), dtype=self.torch_dtype, device=self.device)
        stacked_actions = torch.tensor(np.array(batch.actions), dtype=self.torch_dtype, device=self.device)
        stacked_rewards = torch.tensor(np.array(batch.rewards), dtype=self.torch_dtype, device=self.device)
        stacked_dones = torch.tensor(np.array(batch.dones), dtype=self.torch_dtype, device=self.device)

        next_stacked_obs = torch.tensor(np.array(batch.next_states_obs), dtype=self.torch_dtype, device=self.device)
        next_stacked_grid = torch.tensor(np.array(batch.next_states_grid), dtype=self.torch_dtype, device=self.device)

        neigh_tokens, neigh_mask = self._pad_neighbor_batch(batch.states_var_nei)
        next_neigh_tokens, next_neigh_mask = self._pad_neighbor_batch(batch.next_states_var_nei)

        non_final_next_actions = self.actor_target(
            [next_stacked_obs, next_stacked_grid, next_neigh_tokens, next_neigh_mask]
        )
        current_Q = self.critic(
            [stacked_obs, stacked_grid, neigh_tokens, neigh_mask],
            stacked_actions,
        )

        with torch.no_grad():
            next_target_critic_value = self.critic_target(
                [next_stacked_obs, next_stacked_grid, next_neigh_tokens, next_neigh_mask],
                non_final_next_actions,
            ).squeeze()

            reward_cal = stacked_rewards.clone()
            tar_Q_before_rew = self.gamma * next_target_critic_value * (1 - stacked_dones)
            target_Q = stacked_rewards + (self.gamma * next_target_critic_value * (1 - stacked_dones))
            target_Q = target_Q.unsqueeze(1)
            tar_Q_after_rew = target_Q.clone()

        loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
        cal_loss_Q = loss_Q.clone()
        single_eps_critic_cal_record.append([
            tar_Q_before_rew.detach().cpu().numpy(),
            reward_cal.detach().cpu().numpy(),
            tar_Q_after_rew.detach().cpu().numpy(),
            cal_loss_Q.detach().cpu().numpy(),
            (tar_Q_before_rew.detach().cpu().numpy().min(), tar_Q_before_rew.detach().cpu().numpy().max()),
            (reward_cal.detach().cpu().numpy().min(), reward_cal.detach().cpu().numpy().max()),
            (tar_Q_after_rew.detach().cpu().numpy().min(), tar_Q_after_rew.detach().cpu().numpy().max()),
            (cal_loss_Q.detach().cpu().numpy().min(), cal_loss_Q.detach().cpu().numpy().max()),
        ])
        self.critic_optimizer.zero_grad()
        loss_Q.backward()
        critic_grad_norm = self._grad_norm(self.critic.parameters())
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        action_i = self.actor([stacked_obs, stacked_grid, neigh_tokens, neigh_mask])
        actor_loss = -self.critic([stacked_obs, stacked_grid, neigh_tokens, neigh_mask], action_i).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._grad_norm(self.actor.parameters())
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        actor_updated = True

        c_loss.append(loss_Q)
        a_loss.append(actor_loss)

        if i_episode % self.update_every == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        self.last_update_info = {
            "update_performed": True,
            "actor_updated": actor_updated,
            "update_step": int(i_episode),
            "buffer_size": len(self.buffer),
            "learning_starts": self.batch_size,
            "batch_size": self.batch_size,
            "policy_delay": 1,
            "l2_reg": 0.0,
            "non_stationary_adam": False,
            "policy_noise": 0.0,
            "noise_clip": 0.0,
            "max_grad_norm": self.max_grad_norm,
            "critic_loss": float(loss_Q.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "q1_mean": float(current_Q.detach().mean().cpu().item()),
            "q2_mean": None,
            "target_q1_mean": float(next_target_critic_value.detach().mean().cpu().item()),
            "target_q2_mean": None,
            "target_min_q_mean": float(next_target_critic_value.detach().mean().cpu().item()),
            "target_y_mean": float(target_Q.detach().mean().cpu().item()),
            "reward_batch_mean": float(stacked_rewards.detach().mean().cpu().item()),
            "done_batch_mean": float(stacked_dones.detach().mean().cpu().item()),
            "target_twin_gap_abs_mean": None,
            "critic1_grad_norm": critic_grad_norm,
            "critic2_grad_norm": None,
            "actor_grad_norm": actor_grad_norm,
        }
        return c_loss, a_loss, single_eps_critic_cal_record

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
        actor_path = os.path.join(path, "learned_pcb_biased_att_iddpg_actor.pt")
        critic_path = os.path.join(path, "learned_pcb_biased_att_iddpg_critic.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

        if episode is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, "learned_pcb_biased_att_iddpg_actor_ep{}.pt".format(int(episode))))
            torch.save(self.critic.state_dict(), os.path.join(path, "learned_pcb_biased_att_iddpg_critic_ep{}.pt".format(int(episode))))
        if step is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, "learned_pcb_biased_att_iddpg_actor_step{}.pt".format(int(step))))
            torch.save(self.critic.state_dict(), os.path.join(path, "learned_pcb_biased_att_iddpg_critic_step{}.pt".format(int(step))))

    def _load_state(self, path, checkpoint_tag=None):
        actor_name = "learned_pcb_biased_att_iddpg_actor.pt"
        critic_name = "learned_pcb_biased_att_iddpg_critic.pt"
        if checkpoint_tag:
            actor_name = f"learned_pcb_biased_att_iddpg_actor_{checkpoint_tag}.pt"
            critic_name = f"learned_pcb_biased_att_iddpg_critic_{checkpoint_tag}.pt"

        actor_path = os.path.join(path, actor_name)
        critic_path = os.path.join(path, critic_name)
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def _load_actor_state(self, path, checkpoint_tag=None):
        actor_name = "learned_pcb_biased_att_iddpg_actor.pt"
        if checkpoint_tag:
            actor_name = f"learned_pcb_biased_att_iddpg_actor_{checkpoint_tag}.pt"

        actor_path = os.path.join(path, actor_name)
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.actor_target = deepcopy(self.actor)

    def load(self, path, checkpoint_tag=None):
        if self.actor is None or self.critic is None:
            self.pending_load_path = (path, checkpoint_tag)
            return
        self._load_state(path, checkpoint_tag)

    def load_for_eval(self, path, checkpoint_tag=None):
        if self.actor is None:
            self.pending_eval_load_path = (path, checkpoint_tag)
            return
        self._load_actor_state(path, checkpoint_tag)
