import torch
import torch.nn as nn

from agents.maddpg.networks import Actor


class CentralValueCritic(nn.Module):
    def __init__(self, critic_obs, n_agents, hidden_dim=128):
        super().__init__()
        own_dim, neigh_dim, radar_dim = critic_obs
        self.n_agents = n_agents
        self.per_agent_encoder = nn.Sequential(
            nn.Linear(own_dim + neigh_dim + radar_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.agent_id_embedding = nn.Embedding(n_agents, 32)
        self.merge_fc = nn.Sequential(
            nn.Linear(128 * n_agents + 128 + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs_all, target_agent_idx):
        per_agent_input = torch.cat(obs_all, dim=-1)
        per_agent_features = self.per_agent_encoder(per_agent_input)
        flat_features = per_agent_features.reshape(per_agent_features.size(0), -1)

        batch_index = torch.arange(per_agent_features.size(0), device=per_agent_features.device)
        target_features = per_agent_features[batch_index, target_agent_idx, :]
        target_id_features = self.agent_id_embedding(target_agent_idx)

        merged = torch.cat([flat_features, target_features, target_id_features], dim=-1)
        return self.merge_fc(merged)
