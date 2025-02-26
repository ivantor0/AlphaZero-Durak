# src/model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class AlphaZeroNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_actions, num_layers=10, use_history=False, history_dim=128):
        super(AlphaZeroNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.use_history = use_history
        self.history_dim = history_dim
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # History encoder (LSTM) - optional
        if use_history:
            self.history_input_dim = num_actions + 1  # action + player flag
            self.lstm = nn.LSTM(
                input_size=self.history_input_dim, 
                hidden_size=history_dim, 
                batch_first=True
            )
            # Combined layer for merging state and history features
            self.combine_layer = nn.Linear(hidden_dim + history_dim, hidden_dim)
        
        # Hidden layers (residual blocks)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Policy head
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, history_actions=None, history_actors=None, history_lengths=None):
        """
        Forward pass supporting optional history input.
        
        Args:
            x: State tensor [batch_size, input_dim]
            history_actions: Tensor of action indices [batch_size, max_seq_len]
            history_actors: Tensor indicating if action was by current player (1) or opponent (0) [batch_size, max_seq_len]
            history_lengths: Lengths of each history sequence [batch_size]
        """
        # Process state features
        x = F.relu(self.input_layer(x))
        
        # Process history if enabled and provided
        if self.use_history and history_actions is not None:
            batch_size = x.size(0)
            device = x.device
            
            # Handle empty history case (when all lengths are 0 or None)
            if history_lengths is None or torch.all(history_lengths <= 0):
                # Just use zeros for the history embedding
                hist_embed = torch.zeros(batch_size, self.history_dim, device=device)
            else:
                # Filter out zero-length sequences
                non_zero_mask = history_lengths > 0
                if torch.any(non_zero_mask):
                    # Get indices of non-zero sequences
                    non_zero_indices = torch.nonzero(non_zero_mask).squeeze(1)
                    
                    # Get the non-zero lengths
                    valid_lengths = history_lengths[non_zero_indices]
                    
                    # Sort by length for packing
                    sorted_lens, sort_idx = torch.sort(valid_lengths, descending=True)
                    sorted_lens = sorted_lens.cpu()  # Ensure on CPU for packing
                    
                    # Get corresponding actions and actors
                    valid_actions = history_actions[non_zero_indices]
                    valid_actors = history_actors[non_zero_indices]
                    
                    sorted_actions = valid_actions[sort_idx]
                    sorted_actors = valid_actors[sort_idx]
                    
                    # Prepare sequences
                    seq_inputs = []
                    for i, seq_len in enumerate(sorted_lens):
                        L = int(seq_len.item())
                        # Get actions and actors for this sequence
                        acts = sorted_actions[i, :L]
                        actors = sorted_actors[i, :L].float()
                        
                        # One-hot encode actions
                        acts_onehot = F.one_hot(acts, num_classes=self.num_actions).float()
                        
                        # Combine actor flag and action encoding
                        step_feats = torch.cat([actors.unsqueeze(1), acts_onehot], dim=1)
                        seq_inputs.append(step_feats)
                    
                    # Pad and pack sequences (only if we have valid sequences)
                    if seq_inputs:
                        padded = pad_sequence(seq_inputs, batch_first=True)
                        packed = pack_padded_sequence(padded, sorted_lens, batch_first=True)
                        
                        # Run LSTM
                        _, (h_n, _) = self.lstm(packed)
                        valid_hist_embed = h_n.squeeze(0)  # [valid_batch_size, history_dim]
                        
                        # Restore original batch order among valid sequences
                        _, inv_idx = torch.sort(sort_idx)
                        valid_hist_embed = valid_hist_embed[inv_idx]
                        
                        # Create full embedding with zeros for invalid sequences
                        hist_embed = torch.zeros(batch_size, self.history_dim, device=device)
                        hist_embed[non_zero_indices] = valid_hist_embed
                    else:
                        # Fallback if somehow we have no valid sequences
                        hist_embed = torch.zeros(batch_size, self.history_dim, device=device)
                else:
                    # If all lengths are zero (but not caught by earlier check)
                    hist_embed = torch.zeros(batch_size, self.history_dim, device=device)
            
            # Combine state features with history features
            x = torch.cat([x, hist_embed], dim=1)
            x = F.relu(self.combine_layer(x))
        
        # Run through residual blocks
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output heads
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        
        return policy_logits, value.squeeze(-1)