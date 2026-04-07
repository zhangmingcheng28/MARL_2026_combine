import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, actor_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.own_nei = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        self.own_radar = nn.Sequential(nn.Linear(actor_dim[2], 64), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64 + 64 + 64, 256), nn.ReLU())
        self.act_out = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        own_obs = self.own_fc(obs[0])
        own_nei = self.own_nei(obs[1])
        own_radar = self.own_radar(obs[2])
        merged = torch.cat((own_obs, own_nei, own_radar), dim=1)
        features = self.merge_feature(merged)
        return self.act_out(features)


class EmbeddedCentralCritic(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, hidden_dim=128):
        super().__init__()
        own_dim, neigh_dim, radar_dim = critic_obs
        self.n_agents = n_agents
        self.per_agent_encoder = nn.Sequential(
            nn.Linear(own_dim + neigh_dim + radar_dim + n_actions, 256),
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

    def forward(self, obs_all, action_all, target_agent_idx):
        per_agent_input = torch.cat(
            [obs_all[0], obs_all[1], obs_all[2], action_all],
            dim=-1,
        )
        per_agent_features = self.per_agent_encoder(per_agent_input)
        flat_features = per_agent_features.reshape(per_agent_features.size(0), -1)

        batch_index = torch.arange(per_agent_features.size(0), device=per_agent_features.device)
        target_features = per_agent_features[batch_index, target_agent_idx, :]
        target_id_features = self.agent_id_embedding(target_agent_idx)

        merged = torch.cat([flat_features, target_features, target_id_features], dim=-1)
        return self.merge_fc(merged)
