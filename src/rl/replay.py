from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, maxlen: int = 500_000):
        self._storage: Deque[Tuple[np.ndarray, int, float]] = deque(maxlen=maxlen)

    def add(self, obs: np.ndarray, action: int, outcome: float) -> None:
        self._storage.append((obs.astype(np.float32), int(action), float(outcome)))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self._storage, batch_size)
        obs, act, tgt = zip(*batch)
        return (
            np.stack(obs, axis=0),
            np.asarray(act, dtype=np.int64),
            np.asarray(tgt, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._storage)
