from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from src.agents.policy import epsilon_greedy
from src.durak.durak_game import (
    apply_action,
    current_player,
    initial_state,
    is_terminal,
    legal_actions,
    winner_id,
)
from src.durak.encoding import encode_state


def play_one_game(
    qnet: torch.nn.Module,
    device: torch.device,
    eps: float,
    truesight: bool,
    rng: np.random.Generator,
) -> Tuple[List[Tuple[np.ndarray, int, int]], int]:
    state = initial_state(rng)
    trajectory: List[Tuple[np.ndarray, int, int]] = []
    while not is_terminal(state):
        pid = current_player(state)
        obs = encode_state(state, perspective_player=pid, truesight=truesight)
        mask = legal_actions(state)
        obs_tensor = torch.from_numpy(obs).to(device).unsqueeze(0)
        with torch.no_grad():
            q_values = qnet(obs_tensor).squeeze(0)
        action = epsilon_greedy(mask, eps, q_values)
        state, _, _ = apply_action(state, action)
        trajectory.append((obs, action, pid))
    winner = winner_id(state)
    if winner is None:
        raise RuntimeError("Game ended without a winner")
    labeled = [
        (obs, action, 1.0 if pid == winner else -1.0)
        for (obs, action, pid) in trajectory
    ]
    return labeled, winner
