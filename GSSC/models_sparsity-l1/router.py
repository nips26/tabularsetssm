"""Permutation-invariant tree-style router for tabular MoE."""

from __future__ import annotations

import torch
import torch.nn as nn


class TreeRouter(nn.Module):
    """
    Pooling-based routing logits over experts.

    Uses only symmetric statistics over feature / set axes (no positional
    feature indices).
    """

    def __init__(self, num_experts: int, hidden_dim: int = 64, pool_dim: int = 4) -> None:
        super().__init__()
        if pool_dim != 4:
            raise ValueError("TreeRouter currently expects pool_dim=4 (mean, std, min, max).")
        self.num_experts = num_experts
        self.net = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    @staticmethod
    def _symmetric_pool(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        xf = x.float()
        m = xf.mean(dim=(1, 2))
        s = xf.std(dim=(1, 2), unbiased=False).clamp_min(1e-6)
        mn = xf.amin(dim=(1, 2))
        mx = xf.amax(dim=(1, 2))
        return torch.stack([m, s, mn, mx], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self._symmetric_pool(x))
