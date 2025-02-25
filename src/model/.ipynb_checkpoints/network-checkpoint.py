# src/model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions, num_layers=10):
        super(AlphaZeroNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers (residual blocks)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Policy head
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        
        return policy_logits, value.squeeze(-1)