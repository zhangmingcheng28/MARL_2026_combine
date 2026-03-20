import torch
import torch.nn as nn


class ActorNetworkAllNeiWRadar(nn.Module):
    def __init__(self, actor_dim, action_dim):
        super().__init__()
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

    def forward(self, current_state):
        own_obs = self.own_fc(current_state[0])
        own_nei = self.own_full_nei(current_state[1])
        own_radar = self.own_grid(current_state[2])
        merged = torch.cat((own_obs, own_nei, own_radar), dim=1)
        features = self.merge_feature(merged)
        return self.act_out(features)


class CriticSingleTwoPortionWRadar(nn.Module):
    def __init__(self, critic_dim, action_dim):
        super().__init__()
        self.sa_fc = nn.Sequential(nn.Linear(critic_dim[0] + action_dim, 64), nn.ReLU())
        self.s_all_nei = nn.Sequential(nn.Linear(critic_dim[1], 128), nn.ReLU())
        self.s_radar = nn.Sequential(nn.Linear(critic_dim[2], 128), nn.ReLU())
        self.merge_fc_grid = nn.Sequential(nn.Linear(64 + 128 + 128, 512), nn.ReLU())
        self.out_feature_q = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, single_state, single_action):
        obs_w_action = torch.cat((single_state[0], single_action), dim=1)
        own_obs_w_action = self.sa_fc(obs_w_action)
        own_full_neigh = self.s_all_nei(single_state[1])
        own_radar = self.s_radar(single_state[2])
        merged = torch.cat((own_obs_w_action, own_full_neigh, own_radar), dim=1)
        features = self.merge_fc_grid(merged)
        return self.out_feature_q(features)
