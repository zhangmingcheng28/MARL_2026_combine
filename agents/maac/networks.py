import math

import torch
import torch.nn as nn

from agents.maddpg.networks import Actor


class MAACAttentionCritic(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, hidden_dim=128, attention_heads=4):
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError(
                "hidden_dim ({}) must be divisible by attention_heads ({}).".format(
                    hidden_dim,
                    attention_heads,
                )
            )

        own_dim, neigh_dim, radar_dim = critic_obs
        branch_dim = max(32, hidden_dim // 2)

        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.attend_dim = hidden_dim // attention_heads

        self.own_encoder = nn.Sequential(
            nn.Linear(own_dim, branch_dim),
            nn.ReLU(),
        )
        self.neigh_encoder = nn.Sequential(
            nn.Linear(neigh_dim, branch_dim),
            nn.ReLU(),
        )
        self.radar_encoder = nn.Sequential(
            nn.Linear(radar_dim, branch_dim),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(branch_dim * 3, hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(n_actions, branch_dim),
            nn.ReLU(),
        )
        self.state_action_encoder = nn.Sequential(
            nn.Linear(hidden_dim + branch_dim, hidden_dim),
            nn.ReLU(),
        )

        self.key_extractors = nn.ModuleList(
            [nn.Linear(hidden_dim, self.attend_dim, bias=False) for _ in range(attention_heads)]
        )
        self.selector_extractors = nn.ModuleList(
            [nn.Linear(hidden_dim, self.attend_dim, bias=False) for _ in range(attention_heads)]
        )
        self.value_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, self.attend_dim),
                    nn.ReLU(),
                )
                for _ in range(attention_heads)
            ]
        )

        self.target_id_embedding = nn.Embedding(n_agents, 32)
        merge_hidden = max(256, hidden_dim * 2)
        self.merge_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 32, merge_hidden),
            nn.ReLU(),
            nn.Linear(merge_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_states(self, obs_all):
        own_obs, neigh_obs, radar_obs = obs_all
        batch_size, n_agents, _ = own_obs.shape

        own_features = self.own_encoder(own_obs.reshape(batch_size * n_agents, -1))
        neigh_features = self.neigh_encoder(neigh_obs.reshape(batch_size * n_agents, -1))
        radar_features = self.radar_encoder(radar_obs.reshape(batch_size * n_agents, -1))
        state_features = self.state_encoder(
            torch.cat([own_features, neigh_features, radar_features], dim=-1)
        )
        return state_features.view(batch_size, n_agents, self.hidden_dim)

    def _encode_state_actions(self, state_features, action_all):
        batch_size, n_agents, _ = state_features.shape
        action_features = self.action_encoder(action_all.reshape(batch_size * n_agents, -1))
        sa_features = self.state_action_encoder(
            torch.cat([state_features.reshape(batch_size * n_agents, -1), action_features], dim=-1)
        )
        return sa_features.view(batch_size, n_agents, self.hidden_dim)

    def forward(self, obs_all, action_all, target_agent_idx, regularize=False, return_attend=False):
        state_features = self._encode_states(obs_all)
        sa_features = self._encode_state_actions(state_features, action_all)

        batch_size = action_all.size(0)
        batch_index = torch.arange(batch_size, device=action_all.device)
        target_state = state_features[batch_index, target_agent_idx, :]
        target_sa = sa_features[batch_index, target_agent_idx, :]
        target_id_features = self.target_id_embedding(target_agent_idx)

        attended_values = []
        attention_regs = []
        attention_probs = []
        agent_positions = torch.arange(self.n_agents, device=action_all.device).view(1, -1)
        other_agent_mask = agent_positions != target_agent_idx.unsqueeze(1)

        for key_extractor, selector_extractor, value_extractor in zip(
            self.key_extractors,
            self.selector_extractors,
            self.value_extractors,
        ):
            selectors = selector_extractor(target_state).unsqueeze(1)
            keys = key_extractor(sa_features)
            values = value_extractor(sa_features)

            logits = torch.matmul(selectors, keys.transpose(1, 2)) / math.sqrt(self.attend_dim)
            masked_logits = logits.masked_fill(
                ~other_agent_mask.unsqueeze(1),
                torch.finfo(logits.dtype).min,
            )
            weights = torch.softmax(masked_logits, dim=-1)
            weights = weights * other_agent_mask.unsqueeze(1).to(dtype=weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            attended = torch.matmul(weights, values).squeeze(1)

            if self.n_agents == 1:
                attended = torch.zeros_like(attended)
                weights = torch.zeros_like(weights)

            attended_values.append(attended)
            attention_probs.append(weights.squeeze(1))

            if regularize:
                valid_logits = logits.masked_fill(~other_agent_mask.unsqueeze(1), 0.0)
                attention_regs.append((valid_logits ** 2).mean())

        critic_input = torch.cat(
            [
                target_state,
                target_sa,
                torch.cat(attended_values, dim=-1),
                target_id_features,
            ],
            dim=-1,
        )
        q_value = self.merge_fc(critic_input)

        outputs = [q_value]
        if regularize:
            outputs.append(tuple(1e-3 * reg for reg in attention_regs))
        if return_attend:
            outputs.append(attention_probs)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
