import math

import torch
import torch.nn as nn


class RandomFeatureProjector(nn.Module):
    def __init__(self, dims, dim_index, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.fc = nn.Linear(dims[dim_index], dims[dim_index])
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            out_features, in_features = self.fc.weight.shape
            identity = torch.eye(out_features, in_features, dtype=self.fc.weight.dtype, device=self.fc.weight.device)
            std = math.sqrt(2.0 / float(in_features + out_features))
            normal = torch.randn_like(self.fc.weight) * std
            self.fc.weight.copy_(self.alpha * identity + self.alpha * normal)
            self.fc.bias.zero_()

    def forward(self, x):
        return self.fc(x)


class FMActorNetworkAllNeiWRadar(nn.Module):
    def __init__(self, actor_dim, action_dim):
        super().__init__()
        self.random_own = RandomFeatureProjector(actor_dim, dim_index=0)
        self.random_nei = RandomFeatureProjector(actor_dim, dim_index=1)
        self.random_radar = RandomFeatureProjector(actor_dim, dim_index=2)

        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.own_full_nei = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[2], 64), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64 + 64 + 64, 256), nn.ReLU())
        self.act_out = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

    def _encode(self, current_state, use_random=False):
        own_input = self.random_own(current_state[0]) if use_random else current_state[0]
        nei_input = self.random_nei(current_state[1]) if use_random else current_state[1]
        radar_input = self.random_radar(current_state[2]) if use_random else current_state[2]

        own_obs = self.own_fc(own_input)
        own_nei = self.own_full_nei(nei_input)
        own_radar = self.own_grid(radar_input)
        merged = torch.cat((own_obs, own_nei, own_radar), dim=1)
        return self.merge_feature(merged)

    def forward(self, current_state, use_random=False):
        features = self._encode(current_state, use_random=use_random)
        return self.act_out(features), features


class FMCriticSingleTwoPortionWRadar(nn.Module):
    def __init__(self, critic_dim, action_dim):
        super().__init__()
        self.random_own = RandomFeatureProjector(critic_dim, dim_index=0)
        self.random_nei = RandomFeatureProjector(critic_dim, dim_index=1)
        self.random_radar = RandomFeatureProjector(critic_dim, dim_index=2)

        self.sa_fc = nn.Sequential(nn.Linear(critic_dim[0] + action_dim, 64), nn.ReLU())
        self.s_all_nei = nn.Sequential(nn.Linear(critic_dim[1], 128), nn.ReLU())
        self.s_radar = nn.Sequential(nn.Linear(critic_dim[2], 128), nn.ReLU())
        self.merge_fc_grid = nn.Sequential(nn.Linear(64 + 128 + 128, 512), nn.ReLU())
        self.out_feature_q = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _encode(self, single_state, single_action, use_random=False):
        own_input = self.random_own(single_state[0]) if use_random else single_state[0]
        nei_input = self.random_nei(single_state[1]) if use_random else single_state[1]
        radar_input = self.random_radar(single_state[2]) if use_random else single_state[2]

        obs_w_action = torch.cat((own_input, single_action), dim=1)
        own_obs_w_action = self.sa_fc(obs_w_action)
        own_full_neigh = self.s_all_nei(nei_input)
        own_radar = self.s_radar(radar_input)
        merged = torch.cat((own_obs_w_action, own_full_neigh, own_radar), dim=1)
        return self.merge_fc_grid(merged)

    def forward(self, single_state, single_action, use_random=False):
        features = self._encode(single_state, single_action, use_random=use_random)
        return self.out_feature_q(features), features
