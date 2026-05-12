import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from agents.base_trainer import BaseTrainer
from agents.common.buffer import ReplayBuffer
from agents.common.utils import soft_update
from agents.maddpg.networks import Actor, AttentionCentralCritic, EmbeddedCentralCritic


class MADDPGTrainer(BaseTrainer):
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
            or config.get("algorithm", "").lower() == "maddpg-critic-attention"
        )

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

    def _build_actor_input_from_raw_obs(self, obs, agent_idx):
        return [
            torch.tensor(np.asarray(obs[0][agent_idx]), dtype=self.torch_dtype, device=self.device).reshape(1, -1),
            torch.tensor(np.asarray(obs[1][agent_idx]), dtype=self.torch_dtype, device=self.device).reshape(1, -1),
            torch.tensor(np.asarray(obs[2][agent_idx]), dtype=self.torch_dtype, device=self.device).reshape(1, -1),
        ]

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
                    "MADDPG observation dimension changed from {} to {}.".format(
                        self.obs_dim,
                        inferred_obs_dim,
                    )
                )
            if inferred_split_dims != self.obs_split_dims:
                raise ValueError(
                    "MADDPG observation split changed from {} to {}.".format(
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
        self.critic = critic_cls(self.obs_split_dims, self.n_agents, self.action_dim, self.hidden_dim).to(
            device=self.device,
            dtype=self.torch_dtype,
        )
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
            actions = self.actor(self._split_actor_obs_tensor(obs_tensor)).cpu().numpy()
        if not evaluate:
            actions = actions + np.random.normal(0, self._noise_scale(), size=actions.shape).astype(self.numpy_dtype)
        actions = np.clip(actions, -1.0, 1.0).astype(self.numpy_dtype)
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

        self.critic_optimizer.zero_grad()
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
                    next_actions_list.append(self.target_actor(self._split_obs_batch(next_obs_all, other_i)))
                next_actions_all = torch.stack(next_actions_list, dim=1)
                target_q = self.target_critic(
                    split_next_obs_all,
                    next_actions_all,
                    agent_index,
                )
                y = rewards + self.gamma * (1 - dones) * target_q
            current_q = self.critic(
                split_obs_all,
                actions_all,
                agent_index,
            )
            critic_loss_i = F.mse_loss(current_q, y)
            critic_losses.append(critic_loss_i.detach())
            critic_loss_i.backward(retain_graph=agent_i < self.n_agents - 1)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        for agent_i in range(self.n_agents):
            current_obs_i = self._split_obs_batch(obs_all, agent_i)
            # Original MADDPG-style actor update:
            # replace only the current agent's action with the actor output,
            # while keeping the other agents' actions fixed from replay.
            mixed_actions = actions_all.detach().clone()
            agent_actions = self.actor(current_obs_i)
            mixed_actions[:, agent_i, :] = agent_actions
            agent_index = torch.full(
                (self.batch_size,),
                agent_i,
                dtype=torch.long,
                device=self.device,
            )
            actor_loss_i = -self.critic(
                split_obs_all,
                mixed_actions,
                agent_index,
            ).mean()
            actor_losses.append(actor_loss_i.detach())
            actor_loss_i.backward(retain_graph=agent_i < self.n_agents - 1)

            # Previous local-critic version kept for reference only.
            # It reduced MADDPG to a more IDDPG-like update because the critic only
            # saw one agent's local observation and action.
            # actor_loss_i = -self.critic(current_obs_i[0], current_obs_i[1], current_obs_i[2], agent_actions).mean()
            # actor_loss_i.backward(retain_graph=agent_i < self.n_agents - 1)
        self.actor_optimizer.step()

        update_index = self.current_episode if i_episode is None else i_episode
        if update_index % self.update_every == 0:
            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic, self.critic, self.tau)

        avg_critic_loss = torch.stack(critic_losses).mean()
        avg_actor_loss = torch.stack(actor_losses).mean()
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
        actor_path = os.path.join(path, "maddpg_actor.pt")
        critic_path = os.path.join(path, "maddpg_critic.pt")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        if episode is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, f"maddpg_actor_ep{int(episode)}.pt"))
            torch.save(self.critic.state_dict(), os.path.join(path, f"maddpg_critic_ep{int(episode)}.pt"))
        if step is not None:
            torch.save(self.actor.state_dict(), os.path.join(path, f"maddpg_actor_step{int(step)}.pt"))
            torch.save(self.critic.state_dict(), os.path.join(path, f"maddpg_critic_step{int(step)}.pt"))

    def _load_state(self, path, checkpoint_tag=None):
        actor_name = "maddpg_actor.pt" if checkpoint_tag is None else f"maddpg_actor_{checkpoint_tag}.pt"
        critic_name = "maddpg_critic.pt" if checkpoint_tag is None else f"maddpg_critic_{checkpoint_tag}.pt"
        self.actor.load_state_dict(torch.load(os.path.join(path, actor_name), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, critic_name), map_location=self.device))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def load(self, path, checkpoint_tag=None):
        if self.actor is None or self.critic is None:
            self.pending_load_path = (path, checkpoint_tag)
            return
        self._load_state(path, checkpoint_tag)
