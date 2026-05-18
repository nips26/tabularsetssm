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

    def forward(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        if context_mask is not None:
            if context_mask.dim() == 1:
                context_mask = context_mask.unsqueeze(0).expand(x.shape[0], -1)
            if context_mask.shape != (x.shape[0], x.shape[1]):
                raise ValueError("context_mask must have shape [N] or [B, N].")
            cm = context_mask.to(dtype=k.dtype, device=x.device).unsqueeze(-1)
            k_ctx = k * cm
            v_ctx = v * cm.unsqueeze(-1)
            pooled = torch.einsum("bnr,bnfr->bfr", k_ctx, v_ctx)  # [B, F, R]
        else:
            pooled = torch.einsum("bnr,bnfr->bfr", k, v)  # [B, F, R]
        h_rank = q.unsqueeze(2) * pooled.unsqueeze(1)  # [B, N, F, R]
        if query_mask is not None:
            if query_mask.dim() == 1:
                query_mask = query_mask.unsqueeze(0).expand(x.shape[0], -1)
            if query_mask.shape != (x.shape[0], x.shape[1]):
                raise ValueError("query_mask must have shape [N] or [B, N].")
            qmask = query_mask.to(dtype=h_rank.dtype, device=x.device).unsqueeze(-1).unsqueeze(-1)
            h_rank = h_rank * qmask
        h = self.out_proj(h_rank)
        return self.dropout(h)
