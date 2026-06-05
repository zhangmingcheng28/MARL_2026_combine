import math

import torch
import torch.nn as nn


class ConflictAwareNeighborAttention(nn.Module):
    def __init__(self, query_dim, token_dim, hidden_dim=128, tcpa_bias_scale=4.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tcpa_bias_scale = float(tcpa_bias_scale)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.token_encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _build_tcpa_bias(neighbor_tokens, valid_mask, eps=1e-6):
        rel_pos = neighbor_tokens[..., 0:2]
        host_vel = neighbor_tokens[..., 2:4]
        neigh_vel = neighbor_tokens[..., 4:6]
        host_bound = neighbor_tokens[..., 6]
        neigh_bound = neighbor_tokens[..., 7]
        rel_vel = neigh_vel - host_vel
        rel_speed_sq = (rel_vel ** 2).sum(dim=-1).clamp_min(eps)
        tcpa = -(rel_pos * rel_vel).sum(dim=-1) / rel_speed_sq
        closest_rel_pos = rel_pos + tcpa.unsqueeze(-1) * rel_vel
        dcpa = torch.linalg.norm(closest_rel_pos, dim=-1)
        potential_mask = valid_mask & (tcpa > 0.0) & (dcpa <= (host_bound + neigh_bound))

        tcpa_bias = torch.full_like(tcpa, -2.0)
        tcpa_bias = torch.where(
            potential_mask,
            1.0 / (1.0 + tcpa),
            tcpa_bias,
        )
        return tcpa_bias

    def forward(self, query_source, neighbor_tokens, valid_mask=None):
        if valid_mask is None:
            valid_mask = neighbor_tokens.abs().sum(dim=-1) > 0

        no_valid_neighbors = ~valid_mask.any(dim=1)
        safe_mask = valid_mask.clone()
        if no_valid_neighbors.any():
            safe_mask[no_valid_neighbors, 0] = True

        query = self.query_proj(query_source).unsqueeze(1)
        encoded_tokens = self.token_encoder(neighbor_tokens)
        keys = self.key_proj(encoded_tokens)
        values = self.value_proj(encoded_tokens)

        logits = (query * keys).sum(dim=-1) / math.sqrt(float(self.hidden_dim))
        logits = logits + self.tcpa_bias_scale * self._build_tcpa_bias(neighbor_tokens, safe_mask)
        logits = logits.masked_fill(~safe_mask, -1e9)

        weights = torch.softmax(logits, dim=-1)
        context = torch.sum(weights.unsqueeze(-1) * values, dim=1)
        context = context.masked_fill(no_valid_neighbors.unsqueeze(1), 0.0)
        return context


class ActorNetworkVariableNeiWRadar(nn.Module):
    def __init__(self, own_dim, radar_dim, neighbor_token_dim, action_dim, tcpa_bias_scale=4.0):
        super().__init__()
        self.own_fc = nn.Sequential(nn.Linear(own_dim, 64), nn.ReLU())
        self.radar_fc = nn.Sequential(nn.Linear(radar_dim, 64), nn.ReLU())
        self.neighbor_attention = ConflictAwareNeighborAttention(
            64 + 64,
            neighbor_token_dim,
            hidden_dim=128,
            tcpa_bias_scale=tcpa_bias_scale,
        )
        self.merge_feature = nn.Sequential(nn.Linear(64 + 64 + 128, 256), nn.ReLU())
        self.act_out = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def forward(self, current_state):
        own_obs, radar_obs, neighbor_tokens, valid_mask = current_state
        own_features = self.own_fc(own_obs)
        radar_features = self.radar_fc(radar_obs)
        query_source = torch.cat((own_features, radar_features), dim=1)
        neighbor_context = self.neighbor_attention(query_source, neighbor_tokens, valid_mask=valid_mask)
        merged = torch.cat((own_features, radar_features, neighbor_context), dim=1)
        features = self.merge_feature(merged)
        return self.act_out(features)


class CriticNetworkVariableNeiWRadar(nn.Module):
    def __init__(self, own_dim, radar_dim, neighbor_token_dim, action_dim, tcpa_bias_scale=4.0):
        super().__init__()
        self.own_action_fc = nn.Sequential(nn.Linear(own_dim + action_dim, 64), nn.ReLU())
        self.radar_fc = nn.Sequential(nn.Linear(radar_dim, 64), nn.ReLU())
        self.neighbor_attention = ConflictAwareNeighborAttention(
            64 + 64,
            neighbor_token_dim,
            hidden_dim=128,
            tcpa_bias_scale=tcpa_bias_scale,
        )
        self.merge_feature = nn.Sequential(nn.Linear(64 + 64 + 128, 512), nn.ReLU())
        self.out_feature_q = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, single_state, single_action):
        own_obs, radar_obs, neighbor_tokens, valid_mask = single_state
        obs_w_action = torch.cat((own_obs, single_action), dim=1)
        own_features = self.own_action_fc(obs_w_action)
        radar_features = self.radar_fc(radar_obs)
        query_source = torch.cat((own_features, radar_features), dim=1)
        neighbor_context = self.neighbor_attention(query_source, neighbor_tokens, valid_mask=valid_mask)
        merged = torch.cat((own_features, radar_features, neighbor_context), dim=1)
        features = self.merge_feature(merged)
        return self.out_feature_q(features)
