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


class AttentionCentralCritic(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, hidden_dim=128, attention_heads=4):
        super().__init__()
        own_dim, neigh_dim, radar_dim = critic_obs
        self.n_agents = n_agents
        self.feature_dim = 128
        self.neighbor_token_dim = 5 if neigh_dim % 5 == 0 else neigh_dim
        self.neighbor_count = max(1, neigh_dim // self.neighbor_token_dim)
        self.own_action_encoder = nn.Sequential(
            nn.Linear(own_dim + n_actions, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU(),
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(self.neighbor_token_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU(),
        )
        self.radar_encoder = nn.Sequential(
            nn.Linear(radar_dim, self.feature_dim),
            nn.ReLU(),
        )
        self.neighbor_attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.target_id_embedding = nn.Embedding(n_agents, 32)
        merge_hidden = max(256, hidden_dim)
        self.merge_fc = nn.Sequential(
            nn.Linear(self.feature_dim * 3 + 32, merge_hidden),
            nn.ReLU(),
            nn.Linear(merge_hidden, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs_all, action_all, target_agent_idx):
        batch_index = torch.arange(action_all.size(0), device=action_all.device)
        target_own_obs = obs_all[0][batch_index, target_agent_idx, :]
        target_neigh_obs = obs_all[1][batch_index, target_agent_idx, :]
        target_radar_obs = obs_all[2][batch_index, target_agent_idx, :]
        target_action = action_all[batch_index, target_agent_idx, :]

        target_own_action = self.own_action_encoder(
            torch.cat([target_own_obs, target_action], dim=-1)
        )

        neighbor_tokens = target_neigh_obs.reshape(
            target_neigh_obs.size(0),
            self.neighbor_count,
            self.neighbor_token_dim,
        )
        valid_neighbor_mask = neighbor_tokens.abs().sum(dim=-1) > 0
        key_padding_mask = ~valid_neighbor_mask
        no_valid_neighbors = ~valid_neighbor_mask.any(dim=1)
        if no_valid_neighbors.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[no_valid_neighbors, 0] = False

        neighbor_features = self.neighbor_encoder(neighbor_tokens)
        attended_neighbors, _ = self.neighbor_attention(
            target_own_action.unsqueeze(1),
            neighbor_features,
            neighbor_features,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended_neighbors = attended_neighbors.squeeze(1)
        attended_neighbors = attended_neighbors.masked_fill(no_valid_neighbors.unsqueeze(1), 0.0)

        target_radar_features = self.radar_encoder(target_radar_obs)
        target_id_features = self.target_id_embedding(target_agent_idx)

        merged = torch.cat(
            [target_own_action, attended_neighbors, target_radar_features, target_id_features],
            dim=-1,
        )
        return self.merge_fc(merged)
