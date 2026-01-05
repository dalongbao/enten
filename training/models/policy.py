"""
MLP Policy network for fish control.

Architecture:
- Visual encoder: raycast(32) -> 32 -> 16
- Lateral encoder: lateral(16) -> 16 -> 8
- Proprioception: 3 features passed through directly
- Hunger: 1 feature passed through directly
- Combined: 28 -> 64 -> 64 -> action(2) + value(1)

Total parameters: ~6,500
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
        self.proprio_dim = 3
        self.hunger_dim = 1
        self.obs_dim = self.visual_dim + self.lateral_dim + self.proprio_dim + self.hunger_dim  # 52

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

        # Combined processing (16 + 8 + 3 + 1 = 28)
        combined_dim = 16 + 8 + self.proprio_dim + self.hunger_dim
        self.combined = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action head (2 outputs: thrust mean, turn mean)
        self.action_mean = nn.Linear(hidden_dim, 2)

        # Learnable log std for action distribution
        self.action_log_std = nn.Parameter(torch.zeros(2))

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
            obs: (batch, 52) observation tensor

        Returns:
            action_mean: (batch, 2) action means
            action_log_std: (2,) log standard deviations
            value: (batch, 1) state value estimate
        """
        # Split observation into components
        visual = obs[:, : self.visual_dim]
        lateral = obs[:, self.visual_dim : self.visual_dim + self.lateral_dim]
        proprio_start = self.visual_dim + self.lateral_dim
        proprio = obs[:, proprio_start : proprio_start + self.proprio_dim]
        hunger = obs[:, -self.hunger_dim :]

        # Encode each modality
        visual_feat = self.visual_encoder(visual)  # (batch, 16)
        lateral_feat = self.lateral_encoder(lateral)  # (batch, 8)

        # Combine features
        combined = torch.cat([visual_feat, lateral_feat, proprio, hunger], dim=1)  # (batch, 28)
        hidden = self.combined(combined)  # (batch, hidden_dim)

        # Outputs
        action_mean = self.action_mean(hidden)  # (batch, 2)
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
            obs: (batch, 52) observation tensor
            deterministic: if True, return mean action without sampling

        Returns:
            action: (batch, 2) actions with bounds applied [thrust, turn]
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

        # Apply bounds: thrust [0, 1], turn [-1, 1]
        action = torch.stack(
            [
                torch.sigmoid(unbounded_action[:, 0]),  # thrust: [0, 1]
                torch.tanh(unbounded_action[:, 1]),  # turn: [-1, 1]
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
            obs: (batch, 52) observations
            actions: (batch, 2) actions (already bounded)

        Returns:
            log_prob: (batch,) log probabilities
            entropy: (batch,) entropy
            value: (batch, 1) value estimate
        """
        action_mean, action_log_std, value = self.forward(obs)
        std = action_log_std.exp()

        # Inverse transform to get unbounded actions
        # Clamp to avoid numerical issues at boundaries
        thrust_clamped = actions[:, 0].clamp(1e-6, 1 - 1e-6)
        turn_clamped = actions[:, 1].clamp(-1 + 1e-6, 1 - 1e-6)

        unbounded = torch.stack(
            [
                torch.logit(thrust_clamped),  # Inverse sigmoid
                torch.atanh(turn_clamped),  # Inverse tanh
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
