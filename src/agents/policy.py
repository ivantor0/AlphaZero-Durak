from __future__ import annotations

import numpy as np
import torch


def masked_argmax(q: torch.Tensor, legal_mask: np.ndarray) -> int:
    if q.dim() != 1:
        raise ValueError("q must be 1D tensor")
    mask = torch.as_tensor(legal_mask, device=q.device, dtype=torch.bool)
    neg_inf = torch.finfo(q.dtype).min
    q_masked = torch.where(mask, q, torch.tensor(neg_inf, device=q.device))
    return int(torch.argmax(q_masked).item())


def epsilon_greedy(legal_mask: np.ndarray, eps: float, q: torch.Tensor) -> int:
    legal_indices = np.flatnonzero(legal_mask)
    if legal_indices.size == 0:
        raise ValueError("No legal actions available")
    if np.random.rand() < eps:
        return int(np.random.choice(legal_indices))
    return masked_argmax(q, legal_mask)
