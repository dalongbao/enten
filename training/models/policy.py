"""MLP policy network for fish control."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class FishPolicy(nn.Module):
    def __init__(
        self,
        num_rays: int = 16,
        num_lateral: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.num_rays = num_rays
        self.num_lateral = num_lateral
        self.hidden_dim = hidden_dim

        self.visual_dim = num_rays * 2
        self.lateral_dim = num_lateral * 2
        self.proprio_dim = 4
        self.internal_dim = 4
        self.social_dim = 4
        self.obs_dim = (
            self.visual_dim + self.lateral_dim + self.proprio_dim +
            self.internal_dim + self.social_dim
        )

        self.visual_encoder = nn.Sequential(
            nn.Linear(self.visual_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.lateral_encoder = nn.Sequential(
            nn.Linear(self.lateral_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        combined_dim = 16 + 8 + self.proprio_dim + self.internal_dim + self.social_dim
        self.combined = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 6 actions: body_freq, body_amp, left_pec_freq, left_pec_amp, right_pec_freq, right_pec_amp
        self.action_mean = nn.Linear(hidden_dim, 6)
        self.action_log_std = nn.Parameter(torch.zeros(6))
        self.value_head = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor):
        visual = obs[:, : self.visual_dim]
        lateral = obs[:, self.visual_dim : self.visual_dim + self.lateral_dim]
        proprio_start = self.visual_dim + self.lateral_dim
        proprio = obs[:, proprio_start : proprio_start + self.proprio_dim]
        internal_start = proprio_start + self.proprio_dim
        internal = obs[:, internal_start : internal_start + self.internal_dim]
        social_start = internal_start + self.internal_dim
        social = obs[:, social_start : social_start + self.social_dim]

        visual_feat = self.visual_encoder(visual)
        lateral_feat = self.lateral_encoder(lateral)

        combined = torch.cat([visual_feat, lateral_feat, proprio, internal, social], dim=1)
        hidden = self.combined(combined)

        action_mean = self.action_mean(hidden)
        value = self.value_head(hidden)
        clamped_log_std = torch.clamp(self.action_log_std, -2.0, 2.0)

        return action_mean, clamped_log_std, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_log_std, value = self.forward(obs)
        std = action_log_std.exp()

        if deterministic:
            unbounded_action = action_mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
            entropy = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(action_mean, std)
            unbounded_action = dist.rsample()
            log_prob = dist.log_prob(unbounded_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        # All 6 actions use sigmoid (all in [0, 1] range)
        # body_freq, body_amp, left_pec_freq, left_pec_amp, right_pec_freq, right_pec_amp
        action = torch.sigmoid(unbounded_action)

        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, action_log_std, value = self.forward(obs)
        std = action_log_std.exp()

        # All 6 actions are in [0, 1] range - use logit to convert back to unbounded
        actions_clamped = actions.clamp(1e-6, 1 - 1e-6)
        unbounded = torch.logit(actions_clamped)

        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(unbounded).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, _, value = self.forward(obs)
        return value
