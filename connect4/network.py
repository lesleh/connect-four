"""ResNet neural network with policy and value heads (AlphaZero architecture)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import ROWS, COLS


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class Connect4Net(nn.Module):
    """AlphaZero-style network.

    Input:  (batch, 2, 6, 7) — current player's pieces + opponent's pieces
    Output: policy logits (batch, 7), value (batch, 1) in [-1, 1]
    """

    def __init__(self, num_res_blocks: int = 5, channels: int = 128) -> None:
        super().__init__()
        # Initial convolution
        self.conv_in = nn.Conv2d(2, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *(ResBlock(channels) for _ in range(num_res_blocks))
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * ROWS * COLS, COLS)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(ROWS * COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shared trunk
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # raw logits — softmax applied externally

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, encoded_state: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Single-state inference. Returns (policy probs, value scalar)."""
        self.eval()
        with torch.no_grad():
            if encoded_state.dim() == 3:
                encoded_state = encoded_state.unsqueeze(0)
            device = next(self.parameters()).device
            encoded_state = encoded_state.to(device)
            p_logits, v = self(encoded_state)
            policy = F.softmax(p_logits, dim=1).squeeze(0)
            value = v.item()
        return policy, value

    def predict_batch(self, encoded_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched inference. Returns (policies [N, 7], values [N])."""
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            encoded_states = encoded_states.to(device)
            p_logits, v = self(encoded_states)
            policies = F.softmax(p_logits, dim=1)
            values = v.squeeze(1)
        return policies, values
