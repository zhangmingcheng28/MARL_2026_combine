import math

import torch
import torch.nn as nn


class ConflictAwareNeighborAttention(nn.Module):
    def __init__(self, query_dim, token_dim, hidden_dim=128, beta=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = float(beta)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.token_encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
        )

    @staticmethod
    def build_conflict_risk(
        neighbor_tokens,
        valid_mask,
        tcpa_threshold=2.0,
        tcpa_smoothness=0.5,
        dcpa_smoothness_ratio=0.1,
        eps=1e-6,
    ):
        rel_pos = neighbor_tokens[..., 0:2]
        host_vel = neighbor_tokens[..., 2:4]
        neigh_vel = neighbor_tokens[..., 4:6]
        host_radius = neighbor_tokens[..., 6]
        neigh_radius = neighbor_tokens[..., 7]
        rel_vel = neigh_vel - host_vel

        rel_speed_sq = rel_vel.square().sum(dim=-1).clamp_min(eps)

        tcpa = -(rel_pos * rel_vel).sum(dim=-1) / rel_speed_sq
        closest_rel_pos = rel_pos + tcpa.unsqueeze(-1) * rel_vel
        dcpa = torch.linalg.norm(closest_rel_pos, dim=-1)

        protective_distance = host_radius + neigh_radius

        time_risk = torch.sigmoid((tcpa_threshold - tcpa) / tcpa_smoothness)

        dcpa_smoothness = (dcpa_smoothness_ratio * protective_distance).clamp_min(eps)

        distance_risk = torch.sigmoid((protective_distance - dcpa) / dcpa_smoothness)

        future_cpa = tcpa > 0.0

        conflict_risk = (
            time_risk
            * distance_risk
            * future_cpa.float()
            * valid_mask.float()
        )

        return conflict_risk, tcpa, dcpa

    def forward(self, query_source, neighbor_tokens, valid_mask=None):
        if valid_mask is None:
            valid_mask = neighbor_tokens.abs().sum(dim=-1) > 0
        # return FALSE for at least one false value in 1st dimension (due to flip sign).
        no_valid_neighbors = ~valid_mask.any(dim=1)  # for each sample are there at least one true value in valid_mask, if yes, return false, but because of the flip "~", we return false for at least one true value in 1st dimension.
        safe_mask = valid_mask.clone()
        if no_valid_neighbors.any():  # any true value in the tensor, if yes, goes into the loop
            safe_mask[no_valid_neighbors, 0] = True  # set the first column to True for all samples with no valid neighbors. So that softmax will not break.

        query = self.query_proj(query_source).unsqueeze(1)
        encoded_tokens = self.token_encoder(neighbor_tokens)
        keys = self.key_proj(encoded_tokens)
        values = self.value_proj(encoded_tokens)

        learned_logits = (query * keys).sum(dim=-1) / math.sqrt(float(self.hidden_dim))  # original style: query @ keys.transpose(-2, -1) / sqrt(self.hidden_dim)
        conflict_risk, tcpa, dcpa = self.build_conflict_risk(neighbor_tokens, valid_mask)
        logits = learned_logits + self.beta * conflict_risk
        logits = logits.masked_fill(~safe_mask, -1e9)  # safe mask: valid_mask is for sample that is having valid neighbour, is True, else is False. While safe_mask, is a copy of it but samples that all neighbours are False, we set the 1st value to True.

        weights = torch.softmax(logits, dim=-1)
        context = torch.sum(weights.unsqueeze(-1) * values, dim=1)  # or in the form of torch.bmm(weights.unsqueeze(-1), values).squeeze(1), they are the same.
        context = context.masked_fill(no_valid_neighbors.unsqueeze(1), 0.0)  # The safe_mask trick only makes softmax numerically workable, meaning prevent the division by 0 case, when I have the situation where no neighbours, logits=[-1e9, -1e9, -1e9], softmax will have a division by zero, so we use the trick of safe_mask[no_valid_neighbors, 0] = True. But that will only makes softmax will not have division by 0. And final context have some "fake values". But eventually at this step, the entire row is zeroed, or the row with no neighbours detected.
        return context


class ActorNetworkVariableNeiWRadar(nn.Module):
    def __init__(self, own_dim, radar_dim, neighbor_token_dim, action_dim, beta=1.0):
        super().__init__()
        self.own_fc = nn.Sequential(nn.Linear(own_dim, 64), nn.ReLU())
        self.radar_fc = nn.Sequential(nn.Linear(radar_dim, 64), nn.ReLU())
        self.neighbor_attention = ConflictAwareNeighborAttention(
            64 + 64,
            neighbor_token_dim,
            hidden_dim=128,
            beta=beta,
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
    def __init__(self, own_dim, radar_dim, neighbor_token_dim, action_dim, beta=1.0):
        super().__init__()
        self.own_action_fc = nn.Sequential(nn.Linear(own_dim + action_dim, 64), nn.ReLU())
        self.radar_fc = nn.Sequential(nn.Linear(radar_dim, 64), nn.ReLU())
        self.neighbor_attention = ConflictAwareNeighborAttention(
            64 + 64,
            neighbor_token_dim,
            hidden_dim=128,
            beta=beta,
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
