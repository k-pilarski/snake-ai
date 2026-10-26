"""Neural network that approximates Q-values for the DQN agent."""

from __future__ import annotations

import torch
from torch import nn

from snake_ai.env import NUM_ACTIONS
from snake_ai.state import STATE_SIZE


class DQNModel(nn.Module):
    """Multi-layer perceptron: STATE_SIZE -> hidden_size -> NUM_ACTIONS.

    Input has shape (batch, STATE_SIZE) and output has shape (batch, NUM_ACTIONS),
    one Q-value per action. The output layer is linear because Q-values can be negative.
    """

    def __init__(
        self,
        input_size: int = STATE_SIZE,
        hidden_size: int = 256,
        output_size: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)
