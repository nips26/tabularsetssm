"""Lightweight SSM expert: only value embed, blocks, and head.

FeatureGraphBuilder and EigenAugmentedPE live **once** on TabularSetSSM; each
forward passes the shared ``z`` tensor from the parent's graph + PE. Experts do
not own or duplicate those modules.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from models.blocks import PermInvariantSSMBlock


TaskType = Literal["classification", "regression"]


class TabularSetSSMExpert(nn.Module):
    """SSM stack + head; ``z`` must come from the parent's shared PE."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        z_dim: int,
        rank_dim: int,
        num_blocks: int,
        use_selective: bool = True,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_glu_ffn: bool = True,
        ffn_mult: float = 2.0,
        use_feature_cross: bool = False,
        cross_rank: int = 16,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.value_embed = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                PermInvariantSSMBlock(
                    hidden_dim=hidden_dim,
                    z_dim=z_dim,
                    rank_dim=rank_dim,
                    use_selective=use_selective,
                    dropout=dropout,
                    use_glu_ffn=use_glu_ffn,
                    ffn_mult=ffn_mult,
                    use_feature_cross=use_feature_cross,
                    cross_rank=cross_rank,
                )
                for _ in range(num_blocks)
            ]
        )
        self.activation = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.pre_head_norm = nn.LayerNorm(hidden_dim)
        self.head: Optional[nn.Linear] = None

    def build_head(self, task_type: TaskType, n_classes: Optional[int], device: torch.device) -> None:
        if task_type == "classification":
            if n_classes is None:
                raise ValueError("n_classes required for classification expert head.")
            self.head = nn.Linear(self.hidden_dim, int(n_classes))
        else:
            # Regression predicts heteroscedastic outputs: [mu, log_var].
            self.head = nn.Linear(self.hidden_dim, 2)
        self.head.to(device)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        row_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("Expert head is not built. Call build_head() after inferring task.")
        was_2d = x.dim() == 2
        if was_2d:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("Expert input must be [N, F] or [B, N, F].")
        if x.shape[-1] != self.num_features:
            raise ValueError("Feature dimension mismatch for expert.")

        h = self.value_embed(x.unsqueeze(-1))
        if row_context is not None:
            if row_context.dim() == 2:
                row_context = row_context.unsqueeze(0)
            if row_context.dim() != 3:
                raise ValueError("row_context must have shape [N, H] or [B, N, H].")
            h = h + row_context.unsqueeze(2)
        for block in self.blocks:
            h = block(h, z, context_mask=context_mask, query_mask=query_mask)

        h = self.activation(self.pre_head_norm(h))
        sample_repr = h.mean(dim=2)
        out = self.head(sample_repr)
        if was_2d:
            out = out.squeeze(0)
        return out
