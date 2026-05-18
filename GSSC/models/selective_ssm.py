"""Content-dependent selective positional update for feature kernels."""

from __future__ import annotations

import torch
import torch.nn as nn


class SelectivePositionUpdate(nn.Module):
    """
    Data-dependent update for positional encodings.

    Produces a per-sample, per-feature z_tilde while preserving the same
    factorized global aggregation pattern as GSSC Eq. (6)-style updates.
    """

    def __init__(self, z_dim: int, hidden_dim: int, rank_dim: int) -> None:
        super().__init__()
        self.x_to_z = nn.Linear(hidden_dim, z_dim, bias=False)
        self.dq_proj = nn.Linear(z_dim, rank_dim, bias=False)
        self.dk_proj = nn.Linear(z_dim, rank_dim, bias=False)
        self.dv_proj = nn.Linear(z_dim, z_dim, bias=False)
        self.out_norm = nn.LayerNorm(z_dim)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [F, Z]
            x: [B, N, F, H]
        Returns:
            z_tilde: [B, N, F, Z]
        """
        if z.dim() != 2:
            raise ValueError("z must have shape [F, Z].")
        if x.dim() != 4:
            raise ValueError("x must have shape [B, N, F, H].")

        q = self.dq_proj(z)  # [F, R]
        k = self.dk_proj(z)  # [F, R]
        global_k = k.sum(dim=0)  # [R]
        alpha = (q * global_k.unsqueeze(0)).sum(dim=-1)  # [F]

        x_gate = self.x_to_z(x)  # [B, N, F, Z]
        base = z.unsqueeze(0).unsqueeze(0) * x_gate  # [B, N, F, Z]
        updated = self.dv_proj(base) * alpha.view(1, 1, -1, 1)
        z_tilde = z.unsqueeze(0).unsqueeze(0) + torch.tanh(updated)
        return self.out_norm(z_tilde)
