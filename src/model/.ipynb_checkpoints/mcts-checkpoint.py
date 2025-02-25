# src/model/mcts.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyspiel
from src.durak.durak_game import DurakObserver

class Node:
    def __init__(self, state, prior_p, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # mapping from action to Node
        self.visit_count = 0
        self.value_sum = 0
        self.prior_p = prior_p
        self.expanded = False
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, action_probs):
        self.expanded = True
        for action, prob in action_probs:
            if action not in self.children:
                next_state = self.state.clone()
                next_state.apply_action(action)
                self.children[action] = Node(next_state, prob, self)
    
    def select(self, c_puct):
        # Select action among children that maximizes UCB score
        best_score = -float("inf")
        best_action = -1
        best_child = None
        
        for action, child in self.children.items():
            # UCB score calculation
            score = child.value() + c_puct * child.prior_p * math.sqrt(self.visit_count) / (1 + child.visit_count)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def update(self, value):
        self.visit_count += 1
        self.value_sum += value
    
    def is_leaf(self):
        return len(self.children) == 0

class MCTS:
    def __init__(self, network, n_simulations=800, c_puct=1.0, temperature=1.0, use_argmax=False, device="cpu"):
        self.network = network
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.use_argmax = use_argmax
        self.device = device
    
    def run(self, root_state):
        root = Node(root_state, 1.0)
        root_player = root_state.current_player()
        
        # Expand the root node first
        legal_actions = root_state.legal_actions()
        if not legal_actions:
            return {}  # No legal actions, return empty policy
            
        action_probs = [(action, 1.0 / len(legal_actions)) for action in legal_actions]
        root.expand(action_probs)
        
        # Run simulations
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]
            current_state = root_state.clone()
            
            # Selection: traverse until we reach a leaf node
            while node.expanded and not node.is_leaf():
                action, node = node.select(self.c_puct)
                current_state.apply_action(action)
                search_path.append(node)
            
            # Check if the game is over
            if current_state.is_terminal():
                value = current_state.returns()[root_player]
            else:
                # Expansion + Evaluation
                legal_actions = current_state.legal_actions()
                observer = DurakObserver(
                    pyspiel.IIGObservationType(
                        perfect_recall=False, public_info=True,
                        private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER),
                    params=None
                )
                observer.set_from(current_state, current_state.current_player())
                
                with torch.no_grad():
                    obs_tensor = torch.tensor(observer.tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
                    policy_logits, value = self.network(obs_tensor)
                    # Convert policy logits to probabilities
                    policy = F.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]
                
                # Only consider legal actions
                policy_legal = [(action, policy[action]) for action in legal_actions]
                # Normalize probabilities
                policy_sum = sum(p for _, p in policy_legal)
                policy_normalized = [(a, p / policy_sum) for a, p in policy_legal]
                
                # Expand the node
                node.expand(policy_normalized)
                
                # Use network's value prediction
                value = value.item()
                
                # If not the root player's turn, negate the value
                if current_state.current_player() != root_player:
                    value = -value
            
            # Backpropagation
            for node in search_path:
                node.update(value)
                value = -value  # Value flips as we go up the tree
        
        # Return the action probabilities
        visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
        sum_visits = sum(visits for _, visits in visit_counts)
        
        if self.use_argmax:
            # Choose the most visited action deterministically
            action_probs = {action: 0.0 for action, _ in visit_counts}
            best_action = max(visit_counts, key=lambda x: x[1])[0]
            action_probs[best_action] = 1.0
        else:
            # Convert visit counts to probabilities
            if self.temperature == 0:  # In the limit of temperature -> 0, equivalent to argmax
                action_probs = {action: 0.0 for action, _ in visit_counts}
                best_action = max(visit_counts, key=lambda x: x[1])[0]
                action_probs[best_action] = 1.0
            else:
                # Apply temperature
                visit_counts_temp = [(action, visits ** (1.0 / self.temperature)) for action, visits in visit_counts]
                sum_visits_temp = sum(visits for _, visits in visit_counts_temp)
                action_probs = {action: visits / sum_visits_temp for action, visits in visit_counts_temp}
        
        return action_probs
        
    def evaluate(self, state, root_player):
        observer = DurakObserver(
            pyspiel.IIGObservationType(
                perfect_recall=False, public_info=True,
                private_info=pyspiel.PrivateInfoType.SINGLE_PLAYER),
            params=None
        )
        observer.set_from(state, state.current_player())
        obs = torch.tensor(observer.tensor, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.network(obs)
            value = value.item()
        return value