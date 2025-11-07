from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, input_dim: int, n_actions: int = 38, hidden_scale: float = 1.0):
        super().__init__()
        width1 = int(512 * hidden_scale)
        width2 = int(512 * hidden_scale)
        width3 = int(256 * hidden_scale)
        self.fc1 = nn.Linear(input_dim, width1)
        self.fc2 = nn.Linear(width1, width2)
        self.fc3 = nn.Linear(width2, width3)
        self.out = nn.Linear(width3, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)
