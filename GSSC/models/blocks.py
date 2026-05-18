"""Permutation-invariant SSM block over feature and sample axes."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        use_glu_ffn: bool = True,
        ffn_mult: float = 2.0,
        use_feature_cross: bool = False,
        cross_rank: int = 16,
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
        self.use_glu_ffn = use_glu_ffn
        self.use_feature_cross = use_feature_cross
        ffn_hidden = max(1, int(hidden_dim * ffn_mult))
        if use_glu_ffn:
            self.norm3 = nn.LayerNorm(hidden_dim)
            self.ffn_in = nn.Linear(hidden_dim, 2 * ffn_hidden)
            self.ffn_out = nn.Linear(ffn_hidden, hidden_dim)
        else:
            self.norm3 = None
            self.ffn_in = None
            self.ffn_out = None

        if use_feature_cross:
            self.cross_u = nn.Linear(hidden_dim, cross_rank, bias=False)
            self.cross_v = nn.Linear(hidden_dim, cross_rank, bias=False)
            self.cross_out = nn.Linear(cross_rank, hidden_dim, bias=False)
            self.norm_cross = nn.LayerNorm(hidden_dim)
        else:
            self.cross_u = None
            self.cross_v = None
            self.cross_out = None
            self.norm_cross = None

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("x must have shape [B, N, F, H].")

        z_use = self.selective(z, x) if self.use_selective and self.selective is not None else z
        h_feat = self.feature_ssm(x, z_use)
        x = self.norm1(x + self.resid_dropout(h_feat))

        h_samp = self.sample_ssm(x, context_mask=context_mask, query_mask=query_mask)
        x = self.norm2(x + self.resid_dropout(h_samp))
        if self.use_glu_ffn and self.norm3 is not None and self.ffn_in is not None and self.ffn_out is not None:
            t = self.norm3(x)
            t = F.glu(self.ffn_in(t), dim=-1)
            t = self.ffn_out(t)
            x = x + self.resid_dropout(t)

        if self.use_feature_cross and self.cross_u is not None and self.cross_v is not None and self.cross_out is not None:
            u = self.cross_u(x)
            v = self.cross_v(x)
            g = u.mean(dim=2, keepdim=True)
            cross = self.cross_out(v * g)
            if self.norm_cross is not None:
                x = self.norm_cross(x + self.resid_dropout(cross))
            else:
                x = x + self.resid_dropout(cross)
        return x
