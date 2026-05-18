"""Global permutation-equivariant feature SSM with factorized kernels."""

from __future__ import annotations

import torch
import torch.nn as nn


class GlobalFeatureSSM(nn.Module):
    """
    Feature-axis global SSM convolution:
        h_f = <q_f, sum_{f'} k_{f'} * v_{f'}>
    where q_f, k_f are from positional encodings and v_f is from content.

    This implementation avoids explicit F x F kernel construction.
    """

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        rank_dim: int,
        symmetric_kernel: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.symmetric_kernel = symmetric_kernel

        self.q_proj = nn.Linear(z_dim, rank_dim, bias=False)
        self.k_proj = self.q_proj if symmetric_kernel else nn.Linear(z_dim, rank_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.out_proj = nn.Linear(rank_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, F, H]
            z: [F, Z] (static) or [B, N, F, Z] (content-updated)
        Returns:
            [B, N, F, H]
        """
        if x.dim() != 4:
            raise ValueError("x must have shape [B, N, F, H].")

        v = self.v_proj(x)  # [B, N, F, R]
        if z.dim() == 2:
            q = self.q_proj(z)  # [F, R]
            k = self.k_proj(z)  # [F, R]
            pooled = torch.einsum("fr,bnfr->bnr", k, v)  # [B, N, R]
            h_rank = torch.einsum("fr,bnr->bnfr", q, pooled)  # [B, N, F, R]
        elif z.dim() == 4:
            q = self.q_proj(z)  # [B, N, F, R]
            k = self.k_proj(z)  # [B, N, F, R]
            pooled = torch.einsum("bnfr,bnfr->bnr", k, v)  # [B, N, R]
            h_rank = q * pooled.unsqueeze(2)  # [B, N, F, R]
        else:
            raise ValueError("z must have shape [F, Z] or [B, N, F, Z].")

        h = self.out_proj(h_rank)
        return self.dropout(h)
