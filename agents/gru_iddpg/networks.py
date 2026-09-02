import torch
import torch.nn as nn

from agents.common.blocks import GRUActorBackbone


class GRUActorNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.backbone = GRUActorBackbone(input_dim, hidden_dim=hidden_dim)
        self.act_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state_seq):
        features = self.backbone(state_seq)
        return self.act_out(features)


class GRUCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.backbone = GRUActorBackbone(input_dim, hidden_dim=hidden_dim)
        self.out_feature_q = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_seq, action):
        features = self.backbone(state_seq)
        merged = torch.cat((features, action), dim=1)
        return self.out_feature_q(merged)
