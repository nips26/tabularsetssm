"""Feature-graph construction and cached Laplacian eigendecomposition."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


GraphMethod = Literal["correlation", "mutual_information", "learnable"]


class FeatureGraphBuilder(nn.Module):
    """
    Builds a feature graph and caches Laplacian eigenpairs.

    The cached eigenvectors/eigenvalues are non-trainable buffers and are meant
    to be computed once during dataset initialization.
    """

    def __init__(
        self,
        num_features: int,
        pe_dim: int,
        method: GraphMethod = "correlation",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_features < 2:
            raise ValueError("num_features must be >= 2.")
        if pe_dim < 1:
            raise ValueError("pe_dim must be >= 1.")

        self.num_features = num_features
        self.pe_dim = pe_dim
        self.method = method
        self.eps = eps

        if method == "learnable":
            init = torch.randn(num_features, num_features) * 0.02
            self.learnable_logits = nn.Parameter(init)
        else:
            self.register_parameter("learnable_logits", None)

        # Pre-size buffers so inference can load_state_dict without calling fit() first
        # (torch.empty(0) cannot receive [F,F] weights from a checkpoint).
        zf = torch.zeros(num_features, num_features, dtype=torch.float32)
        self.register_buffer("adjacency", zf)
        self.register_buffer("laplacian", zf.clone())
        self.register_buffer("eigvals", torch.zeros(pe_dim, dtype=torch.float32))
        self.register_buffer("eigvecs", torch.zeros(num_features, pe_dim, dtype=torch.float32))

    def _build_adjacency(self, x: torch.Tensor) -> torch.Tensor:
        """Build a symmetric feature adjacency matrix."""
        if self.method == "correlation":
            centered = x - x.mean(dim=0, keepdim=True)
            std = centered.std(dim=0, keepdim=True).clamp_min(self.eps)
            z = centered / std
            corr = (z.transpose(0, 1) @ z) / max(1, z.shape[0] - 1)
            adj = corr.abs()
        elif self.method == "mutual_information":
            # Lightweight MI-like proxy using non-linear correlation magnitude.
            # This avoids heavyweight dependencies and remains deterministic.
            centered = x - x.mean(dim=0, keepdim=True)
            std = centered.std(dim=0, keepdim=True).clamp_min(self.eps)
            z = centered / std
            corr = (z.transpose(0, 1) @ z) / max(1, z.shape[0] - 1)
            adj = torch.sqrt(corr.abs().clamp_min(0.0))
        elif self.method == "learnable":
            logits = 0.5 * (self.learnable_logits + self.learnable_logits.transpose(0, 1))
            adj = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported graph construction method: {self.method}")

        adj = adj - torch.diag_embed(torch.diagonal(adj))
        adj = 0.5 * (adj + adj.transpose(0, 1))
        return adj

    def fit(self, x: torch.Tensor) -> "FeatureGraphBuilder":
        """
        Fit feature graph from data.

        Args:
            x: Tensor of shape [N, F] or [B, N, F].
        """
        if x.dim() == 3:
            x_2d = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            x_2d = x
        else:
            raise ValueError("x must have shape [N, F] or [B, N, F].")

        if x_2d.shape[1] != self.num_features:
            raise ValueError("x feature dimension does not match num_features.")

        with torch.no_grad():
            adj = self._build_adjacency(x_2d)
            deg = adj.sum(dim=1)
            inv_sqrt_deg = (deg + self.eps).rsqrt()
            norm_adj = inv_sqrt_deg.unsqueeze(1) * adj * inv_sqrt_deg.unsqueeze(0)
            lap = torch.eye(self.num_features, device=adj.device, dtype=adj.dtype) - norm_adj

            evals, evecs = torch.linalg.eigh(lap)
            evals = evals.clamp_min(0.0)

            start = 1 if evals.shape[0] > 1 else 0
            end = min(self.num_features, start + self.pe_dim)
            evals = evals[start:end]
            evecs = evecs[:, start:end]

            if evals.numel() < self.pe_dim:
                pad = self.pe_dim - evals.numel()
                evals = torch.nn.functional.pad(evals, (0, pad), value=0.0)
                evecs = torch.nn.functional.pad(evecs, (0, pad), value=0.0)

            self.adjacency.detach().copy_(adj)
            self.laplacian.detach().copy_(lap)
            self.eigvals.detach().copy_(evals.to(dtype=self.eigvals.dtype))
            self.eigvecs.detach().copy_(evecs.to(dtype=self.eigvecs.dtype))
        return self

    def is_fitted(self) -> bool:
        """Return whether graph eigenpairs look initialized (fit or loaded from checkpoint)."""
        if self.eigvecs.shape != (self.num_features, self.pe_dim):
            return False
        return bool(self.eigvecs.detach().norm().item() > 1e-12)

    def get_eigenpairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached (eigenvalues, eigenvectors)."""
        if not self.is_fitted():
            raise RuntimeError("FeatureGraphBuilder is not fitted. Call fit() first.")
        return self.eigvals, self.eigvecs
