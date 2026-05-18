"""Sample-axis set SSM for permutation-equivariant sample mixing."""

from __future__ import annotations

import torch
import torch.nn as nn


class SampleSetSSM(nn.Module):
    """
    Sample-axis factorized global convolution.

    Treats samples as a set and applies permutation-equivariant global mixing
    without constructing an explicit N x N matrix.
    """

    def __init__(self, hidden_dim: int, rank_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.out_proj = nn.Linear(rank_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F, H]
        Returns:
            [B, N, F, H]
        """
        if x.dim() != 4:
            raise ValueError("x must have shape [B, N, F, H].")

        sample_desc = x.mean(dim=2)  # [B, N, H]
        q = self.q_proj(sample_desc)  # [B, N, R]
        k = self.k_proj(sample_desc)  # [B, N, R]
        v = self.v_proj(x)  # [B, N, F, R]

        pooled = torch.einsum("bnr,bnfr->bfr", k, v)  # [B, F, R]
        h_rank = q.unsqueeze(2) * pooled.unsqueeze(1)  # [B, N, F, R]
        h = self.out_proj(h_rank)
        return self.dropout(h)
