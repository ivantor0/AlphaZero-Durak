from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.qnet import QNet
from src.rl.replay import ReplayBuffer
from src.rl.selfplay import play_one_game


@dataclass
class TrainerConfig:
    lr: float = 1e-4
    replay_size: int = 500_000
    batch_size: int = 1024
    grad_clip: Optional[float] = 1.0
    games_per_iter: int = 500
    train_steps_per_iter: int = 1000


class DMCTrainer:
    def __init__(self, qnet: QNet, device: torch.device, cfg: TrainerConfig):
        self.qnet = qnet.to(device)
        self.device = device
        self.cfg = cfg
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size)
        self._loss_fn = nn.MSELoss()

    def train_step(self) -> Optional[float]:
        if len(self.replay) < self.cfg.batch_size:
            return None
        obs, actions, targets = self.replay.sample(self.cfg.batch_size)
        obs_t = torch.from_numpy(obs).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        targets_t = torch.from_numpy(targets).to(self.device)

        q_values = self.qnet(obs_t)
        q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss = self._loss_fn(q_taken, targets_t)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    def selfplay_and_fill(
        self, episodes: int, eps: float, truesight: bool, rng: np.random.Generator
    ) -> tuple[int, int]:
        wins = 0
        self.qnet.eval()
        with torch.no_grad():
            for _ in range(episodes):
                labeled, winner = play_one_game(self.qnet, self.device, eps, truesight, rng)
                wins += int(winner == 0)
                for obs, action, outcome in labeled:
                    self.replay.add(obs, action, outcome)
        self.qnet.train()
        return wins, episodes
