"""
MLP Policy network for fish control.

Architecture:
- Visual encoder: raycast(32) -> 32 -> 16
- Lateral encoder: lateral(16) -> 16 -> 8
- Proprioception: 4 features passed through directly
- Internal state: 4 features passed through directly
- Social: 4 features passed through directly (nearest_dist, nearest_angle, num_nearby, heading_diff)
- Combined: 36 -> 64 -> 64 -> action(3) + value(1)

Action space (3 controls - hybrid system):
- speed [0, 1]: Desired forward speed (sigmoid)
- direction [-1, 1]: Turn rate (tanh)
- urgency [0, 1]: Movement intensity (sigmoid)

Fin animation is computed automatically by the renderer.

Observation breakdown (60 features total):
- Raycasts: 32 (16 rays x 2 values)
- Lateral line: 16 (8 sensors x 2 values)
- Proprioception: 4 (vel_forward, vel_lateral, angular_vel, speed)
- Internal: 4 (hunger, stress, social_comfort, energy)
- Social: 4 (nearest_dist, nearest_angle, num_nearby, heading_diff)

Total parameters: ~7,000
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class FishPolicy(nn.Module):
    """MLP policy with separate encoders for visual and lateral inputs."""

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

        # Input dimensions
        self.visual_dim = num_rays * 2  # 32
        self.lateral_dim = num_lateral * 2  # 16
        self.proprio_dim = 4  # forward vel, lateral vel, angular vel, speed
        self.internal_dim = 4  # hunger, stress, social_comfort, energy
        self.social_dim = 4  # nearest_dist, nearest_angle, num_nearby, heading_diff
        self.obs_dim = (
            self.visual_dim + self.lateral_dim + self.proprio_dim +
            self.internal_dim + self.social_dim
        )  # 60

        # Visual encoder (raycasts)
        self.visual_encoder = nn.Sequential(
            nn.Linear(self.visual_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Lateral line encoder
        self.lateral_encoder = nn.Sequential(
            nn.Linear(self.lateral_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

        # Combined processing (16 + 8 + 4 + 4 + 4 = 36)
        combined_dim = 16 + 8 + self.proprio_dim + self.internal_dim + self.social_dim
        self.combined = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action head (3 outputs: speed, direction, urgency)
        self.action_mean = nn.Linear(hidden_dim, 3)

        # Learnable log std for action distribution
        self.action_log_std = nn.Parameter(torch.zeros(3))

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

        # Smaller initialization for action head (more stable learning)
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
        nn.init.zeros_(self.action_mean.bias)

        # Initialize value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs: torch.Tensor):
        """
        Forward pass.

        Args:
            obs: (batch, 60) observation tensor

        Returns:
            action_mean: (batch, 3) action means
            action_log_std: (3,) log standard deviations
            value: (batch, 1) state value estimate
        """
        # Split observation into components
        visual = obs[:, : self.visual_dim]
        lateral = obs[:, self.visual_dim : self.visual_dim + self.lateral_dim]
        proprio_start = self.visual_dim + self.lateral_dim
        proprio = obs[:, proprio_start : proprio_start + self.proprio_dim]
        internal_start = proprio_start + self.proprio_dim
        internal = obs[:, internal_start : internal_start + self.internal_dim]
        social_start = internal_start + self.internal_dim
        social = obs[:, social_start : social_start + self.social_dim]

        # Encode each modality
        visual_feat = self.visual_encoder(visual)  # (batch, 16)
        lateral_feat = self.lateral_encoder(lateral)  # (batch, 8)

        # Combine features
        combined = torch.cat([visual_feat, lateral_feat, proprio, internal, social], dim=1)  # (batch, 36)
        hidden = self.combined(combined)  # (batch, hidden_dim)

        # Outputs
        action_mean = self.action_mean(hidden)  # (batch, 3)
        value = self.value_head(hidden)  # (batch, 1)

        return action_mean, self.action_log_std, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            obs: (batch, 60) observation tensor
            deterministic: if True, return mean action without sampling

        Returns:
            action: (batch, 3) actions with bounds applied
            log_prob: (batch,) log probabilities
            entropy: (batch,) entropy of distribution
            value: (batch, 1) state value
        """
        action_mean, action_log_std, value = self.forward(obs)
        std = action_log_std.exp()

        if deterministic:
            unbounded_action = action_mean
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
            entropy = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(action_mean, std)
            unbounded_action = dist.rsample()  # Reparameterized sample
            log_prob = dist.log_prob(unbounded_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

        # Apply bounds:
        # speed [0, 1] - sigmoid
        # direction [-1, 1] - tanh
        # urgency [0, 1] - sigmoid
        action = torch.stack(
            [
                torch.sigmoid(unbounded_action[:, 0]),  # speed: [0, 1]
                torch.tanh(unbounded_action[:, 1]),     # direction: [-1, 1]
                torch.sigmoid(unbounded_action[:, 2]),  # urgency: [0, 1]
            ],
            dim=1,
        )

        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability of actions (for PPO update).

        Args:
            obs: (batch, 60) observations
            actions: (batch, 3) actions (already bounded)

        Returns:
            log_prob: (batch,) log probabilities
            entropy: (batch,) entropy
            value: (batch, 1) value estimate
        """
        action_mean, action_log_std, value = self.forward(obs)
        std = action_log_std.exp()

        # Inverse transform to get unbounded actions
        # Clamp to avoid numerical issues at boundaries
        speed_clamped = actions[:, 0].clamp(1e-6, 1 - 1e-6)
        direction_clamped = actions[:, 1].clamp(-1 + 1e-6, 1 - 1e-6)
        urgency_clamped = actions[:, 2].clamp(1e-6, 1 - 1e-6)

        unbounded = torch.stack(
            [
                torch.logit(speed_clamped),        # Inverse sigmoid
                torch.atanh(direction_clamped),    # Inverse tanh
                torch.logit(urgency_clamped),      # Inverse sigmoid
            ],
            dim=1,
        )

        dist = Normal(action_mean, std)
        log_prob = dist.log_prob(unbounded).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only (for GAE computation)."""
        _, _, value = self.forward(obs)
        return value
