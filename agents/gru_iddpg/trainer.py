import os
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_trainer import BaseTrainer
from agents.common.utils import soft_update
from agents.gru_iddpg.buffer import Experience, ReplayMemory
from agents.gru_iddpg.networks import GRUActorNetwork, GRUCriticNetwork


class GRUIDDPGTrainer(BaseTrainer):
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
        self.history_length = int(config["train"].get("gru_history_length", 4))
        self.exploration = config.get("exploration", {})

        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.obs_dim = None

        self.buffer = ReplayMemory(config["train"]["buffer_size"])
        self.action_step = 0
        self.current_episode = 1
        self.pending_load_path = None
        self.pending_eval_load_path = None
        self.max_grad_norm = float(config["train"].get("max_grad_norm", 0.0))
        self.context_histories = {}
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

    def _flatten_obs_batch(self, obs):
        if len(obs) < 3:
            raise ValueError("gru-iddpg expects three observation portions: own obs, neighbors, radar.")
        flattened = []
        for agent_idx in range(self.n_agents):
            flattened.append(
                np.concatenate(
                    [
                        np.asarray(obs[0][agent_idx], dtype=self.numpy_dtype).reshape(-1),
                        np.asarray(obs[1][agent_idx], dtype=self.numpy_dtype).reshape(-1),
                        np.asarray(obs[2][agent_idx], dtype=self.numpy_dtype).reshape(-1),
                    ],
                    axis=0,
                )
            )
        return np.asarray(flattened, dtype=self.numpy_dtype)

    def _ensure_models_initialized(self, obs):
        flat_obs = self._flatten_obs_batch(obs)
        obs_dim = int(flat_obs.shape[1])
        if self.actor is not None:
            if obs_dim != self.obs_dim:
                raise ValueError("gru-iddpg observation dimension changed from {} to {}.".format(self.obs_dim, obs_dim))
            return

        self.obs_dim = obs_dim
        self.actor = GRUActorNetwork(obs_dim, self.action_dim, hidden_dim=self.hidden_dim).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
        self.actor_target = deepcopy(self.actor).to(device=self.device, dtype=self.torch_dtype)
        self.critic = GRUCriticNetwork(obs_dim, self.action_dim, hidden_dim=self.hidden_dim).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
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

    def reset_context(self, context_id):
        self.context_histories[context_id] = deque(maxlen=self.history_length)

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

    def _ensure_context_current(self, context_id, obs):
        flat_obs = self._flatten_obs_batch(obs)
        history = self.context_histories.setdefault(context_id, deque(maxlen=self.history_length))
        if len(history) == 0:
            history.append(flat_obs.copy())
        elif history[-1].shape != flat_obs.shape or not np.allclose(history[-1], flat_obs):
            history.append(flat_obs.copy())
        return history

    def _build_sequence_batch(self, history):
        seq_len = len(history)
        sequence = np.stack(list(history), axis=0).astype(self.numpy_dtype)
        padded = np.zeros((self.history_length, self.n_agents, self.obs_dim), dtype=self.numpy_dtype)
        padded[-seq_len:] = sequence
        return np.transpose(padded, (1, 0, 2))

    def select_action_with_context(self, obs, context_id, evaluate=False):
        self._ensure_models_initialized(obs)
        history = self._ensure_context_current(context_id, obs)
        sequence_batch = self._build_sequence_batch(history)
        sequence_tensor = torch.tensor(sequence_batch, dtype=self.torch_dtype, device=self.device)

        with torch.no_grad():
            action_tensor = self.actor(sequence_tensor).cpu().numpy()

        raw_actions = action_tensor.copy()
        noise_arr = np.zeros_like(raw_actions, dtype=self.numpy_dtype)
        if not evaluate:
            noise_arr = np.random.normal(0, self._noise_scale(), size=raw_actions.shape).astype(self.numpy_dtype)
            action_tensor = action_tensor + noise_arr
        actions_arr = np.clip(action_tensor, -1.0, 1.0).astype(self.numpy_dtype)
        self.last_action_info = {
            "raw_mean": float(np.mean(raw_actions)),
            "raw_std": float(np.std(raw_actions)),
            "raw_min": float(np.min(raw_actions)),
            "raw_max": float(np.max(raw_actions)),
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
        return [actions_arr[i] for i in range(self.n_agents)]

    def select_action(self, obs, evaluate=False):
        return self.select_action_with_context(obs, context_id="default", evaluate=evaluate)

    def store_transition_with_context(self, obs, actions, rewards, next_obs, dones, context_id, history=None, cur_hidden=None, next_hidden=None):
        self._ensure_models_initialized(obs)
        history = self._ensure_context_current(context_id, obs)
        current_seq_batch = self._build_sequence_batch(history)

        next_flat_obs = self._flatten_obs_batch(next_obs)
        next_history = deque(history, maxlen=self.history_length)
        next_history.append(next_flat_obs.copy())
        next_seq_batch = self._build_sequence_batch(next_history)

        actions_arr = np.asarray(actions, dtype=self.numpy_dtype)
        rewards_arr = np.asarray(rewards, dtype=self.numpy_dtype)
        dones_arr = np.asarray([int(value) for value in dones], dtype=self.numpy_dtype)
        for agent_idx in range(self.n_agents):
            self.buffer.push(
                current_seq_batch[agent_idx].copy(),
                actions_arr[agent_idx].copy(),
                next_seq_batch[agent_idx].copy(),
                float(rewards_arr[agent_idx]),
                float(dones_arr[agent_idx]),
            )

        if any(dones):
            self.reset_context(context_id)
        else:
            self.context_histories[context_id] = next_history

    def store_transition(self, obs, actions, rewards, next_obs, dones, history=None, cur_hidden=None, next_hidden=None):
        self.store_transition_with_context(obs, actions, rewards, next_obs, dones, context_id="default")

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

        if i_episode is None:
            i_episode = self.current_episode

        transitions = self.buffer.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        state_seq = torch.tensor(np.array(batch.state_seq), dtype=self.torch_dtype, device=self.device)
        action_batch = torch.tensor(np.array(batch.action), dtype=self.torch_dtype, device=self.device)
        next_state_seq = torch.tensor(np.array(batch.next_state_seq), dtype=self.torch_dtype, device=self.device)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=self.torch_dtype, device=self.device)
        done_batch = torch.tensor(np.array(batch.done), dtype=self.torch_dtype, device=self.device)

        current_Q = self.critic(state_seq, action_batch)
        with torch.no_grad():
            next_action = self.actor_target(next_state_seq)
            next_target_critic_value = self.critic_target(next_state_seq, next_action).squeeze()
            reward_cal = reward_batch.clone()
            tar_Q_before_rew = self.gamma * next_target_critic_value * (1 - done_batch)
            target_Q = reward_batch + (self.gamma * next_target_critic_value * (1 - done_batch))
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

        action_i = self.actor(state_seq)
        actor_loss = -self.critic(state_seq, action_i).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._grad_norm(self.actor.parameters())
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        if i_episode % self.update_every == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        self.last_update_info = {
            "update_performed": True,
            "actor_updated": True,
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
            "reward_batch_mean": float(reward_batch.detach().mean().cpu().item()),
            "done_batch_mean": float(done_batch.detach().mean().cpu().item()),
            "target_twin_gap_abs_mean": None,
            "critic1_grad_norm": critic_grad_norm,
            "critic2_grad_norm": None,
            "actor_grad_norm": actor_grad_norm,
        }
        return [loss_Q], [actor_loss], single_eps_critic_cal_record

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
        actor_path = os.path.join(path, "gru_iddpg_actor.pt")
        critic_path = os.path.join(path, "gru_iddpg_critic.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        if episode is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, "gru_iddpg_actor_ep{}.pt".format(int(episode))))
            torch.save(self.critic.state_dict(), os.path.join(path, "gru_iddpg_critic_ep{}.pt".format(int(episode))))
        if step is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, "gru_iddpg_actor_step{}.pt".format(int(step))))
            torch.save(self.critic.state_dict(), os.path.join(path, "gru_iddpg_critic_step{}.pt".format(int(step))))

    def _load_state(self, path, checkpoint_tag=None):
        actor_name = "gru_iddpg_actor.pt"
        critic_name = "gru_iddpg_critic.pt"
        if checkpoint_tag:
            actor_name = f"gru_iddpg_actor_{checkpoint_tag}.pt"
            critic_name = f"gru_iddpg_critic_{checkpoint_tag}.pt"
        actor_path = os.path.join(path, actor_name)
        critic_path = os.path.join(path, critic_name)
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def _load_actor_state(self, path, checkpoint_tag=None):
        actor_name = "gru_iddpg_actor.pt"
        if checkpoint_tag:
            actor_name = f"gru_iddpg_actor_{checkpoint_tag}.pt"
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
