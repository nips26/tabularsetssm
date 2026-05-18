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
            self.head = nn.Linear(self.hidden_dim, 1)
        self.head.to(device)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
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
        for block in self.blocks:
            h = block(h, z)

        h = self.activation(self.pre_head_norm(h))
        sample_repr = h.mean(dim=2)
        out = self.head(sample_repr)
        if was_2d:
            out = out.squeeze(0)
        return out
