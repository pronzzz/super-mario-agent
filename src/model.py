"""
Rainbow DQN model with STN and multi-branch architecture.
Combines visual features (frames) with action history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialTransformerNetwork(nn.Module):
    """
    Spatial Transformer Network to learn attention and focus on relevant regions.
    """
    def __init__(self, in_channels):
        super().__init__()
        
        # Localization network - learns the transformation parameters
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        
        # Calculate the size after conv layers (84x84 -> 42x42 -> 21x21 -> 10x10)
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 10 * 10, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)  # 2x3 affine matrix
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        # Create sampling grid
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        # Sample the input using the grid
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x


class NoisyLinear(nn.Module):
    """
    Noisy linear layer for parameter space noise exploration.
    Replaces epsilon-greedy exploration.
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise buffers (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        # Initialize parameters
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """Sample new noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product for weight noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Factorized Gaussian noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """
    Rainbow DQN with:
    - Spatial Transformer Network for attention
    - Multi-branch architecture (visual + action history)
    - Dueling architecture (value + advantage)
    - Distributional RL (C51)
    - Noisy layers for exploration
    """
    def __init__(self, num_actions, num_atoms=51, v_min=-10, v_max=10, 
                 frame_stack=4, action_history_len=8):
        super().__init__()
        
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.action_history_len = action_history_len
        
        # Support for distributional RL
        self.register_buffer('atoms', torch.linspace(v_min, v_max, num_atoms))
        
        # === Visual Branch ===
        # Spatial Transformer Network
        self.stn = SpatialTransformerNetwork(frame_stack)
        
        # Convolutional backbone (Nature DQN architecture)
        self.conv = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size: 84 -> 20 -> 9 -> 7
        conv_out_size = 64 * 7 * 7
        
        # === Action History Branch ===
        # Embed actions and process through MLP
        self.action_embedding = nn.Embedding(num_actions, 32)
        self.action_mlp = nn.Sequential(
            nn.Linear(32 * action_history_len, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # === Fusion Layer ===
        fusion_size = conv_out_size + 128
        
        # === Dueling Architecture with Noisy Layers ===
        # Value stream
        self.value_hidden = NoisyLinear(fusion_size, 512)
        self.value_out = NoisyLinear(512, num_atoms)  # Distributional
        
        # Advantage stream
        self.advantage_hidden = NoisyLinear(fusion_size, 512)
        self.advantage_out = NoisyLinear(512, num_actions * num_atoms)  # Distributional

    def forward(self, frames, action_history):
        """
        Args:
            frames: (batch, 4, 84, 84) - stacked frames
            action_history: (batch, 8) - last 8 actions
        
        Returns:
            q_values: (batch, num_actions) - expected Q values
            q_dist: (batch, num_actions, num_atoms) - Q value distributions
        """
        batch_size = frames.size(0)
        
        # Visual branch with STN
        x_visual = self.stn(frames)
        x_visual = self.conv(x_visual)
        x_visual = x_visual.view(batch_size, -1)
        
        # Action history branch
        x_action = self.action_embedding(action_history)  # (batch, 8, 32)
        x_action = x_action.view(batch_size, -1)  # (batch, 256)
        x_action = self.action_mlp(x_action)  # (batch, 128)
        
        # Fusion
        x_fused = torch.cat([x_visual, x_action], dim=1)
        
        # Dueling streams
        value = F.relu(self.value_hidden(x_fused))
        value = self.value_out(value).view(batch_size, 1, self.num_atoms)
        
        advantage = F.relu(self.advantage_hidden(x_fused))
        advantage = self.advantage_out(advantage).view(batch_size, self.num_actions, self.num_atoms)
        
        # Combine value and advantage (dueling formula)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probability distribution over atoms
        q_dist = F.softmax(q_atoms, dim=2)
        
        # Calculate expected Q values
        q_values = torch.sum(q_dist * self.atoms.view(1, 1, -1), dim=2)
        
        return q_values, q_dist

    def reset_noise(self):
        """Reset noise in all noisy layers"""
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()
