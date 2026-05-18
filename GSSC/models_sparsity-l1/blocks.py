"""Permutation-invariant SSM block over feature and sample axes."""

from __future__ import annotations

import torch
import torch.nn as nn

from .global_ssm import GlobalFeatureSSM
from .sample_ssm import SampleSetSSM
from .selective_ssm import SelectivePositionUpdate


class PermInvariantSSMBlock(nn.Module):
    """Feature-SSM + Sample-SSM block with residuals and layer normalization."""

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        rank_dim: int,
        use_selective: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_selective = use_selective

        self.selective = (
            SelectivePositionUpdate(z_dim=z_dim, hidden_dim=hidden_dim, rank_dim=rank_dim)
            if use_selective
            else None
        )
        self.feature_ssm = GlobalFeatureSSM(
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            rank_dim=rank_dim,
            symmetric_kernel=True,
            dropout=dropout,
        )
        self.sample_ssm = SampleSetSSM(hidden_dim=hidden_dim, rank_dim=rank_dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("x must have shape [B, N, F, H].")

        z_use = self.selective(z, x) if self.use_selective and self.selective is not None else z
        h_feat = self.feature_ssm(x, z_use)
        x = self.norm1(x + self.resid_dropout(h_feat))

        h_samp = self.sample_ssm(x)
        x = self.norm2(x + self.resid_dropout(h_samp))
        return x
