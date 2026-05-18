"""Utilities for tabular permutation-invariant SSM experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch


TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class EarlyStoppingState:
    """Tracks validation score and epoch for early stopping."""

    best_score: float
    best_epoch: int
    patience_counter: int


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float_tensor(x: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert an array-like object to float32 tensor on a device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def infer_task_type(y: np.ndarray | torch.Tensor, threshold: int = 30) -> TaskType:
    """
    Infer task type from target values.

    Heuristic:
    - Integer / boolean labels with limited unique values -> classification
    - Otherwise -> regression
    """
    if isinstance(y, torch.Tensor):
        y_cpu = y.detach().cpu()
        if y_cpu.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
            return "classification"
        values = y_cpu.numpy()
    else:
        values = np.asarray(y)
        if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
            return "classification"

    unique_count = int(np.unique(values).shape[0])
    if unique_count <= threshold and np.allclose(values, np.round(values)):
        return "classification"
    return "regression"


def train_val_split(
    x: torch.Tensor,
    y: torch.Tensor,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Split tensors into train/validation subsets.

    If stratify is True (typically for classification labels), uses sklearn's
    stratified split; falls back to a random permutation if stratify fails
    (e.g. too few samples per class).
    """
    if val_ratio <= 0.0:
        return x, y, None, None

    if stratify:
        try:
            from sklearn.model_selection import train_test_split as sk_split

            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            x_tr, x_va, y_tr, y_va = sk_split(
                x_np, y_np, test_size=val_ratio, random_state=seed, stratify=y_np
            )
            return (
                torch.as_tensor(x_tr, dtype=x.dtype, device=x.device),
                torch.as_tensor(y_tr, dtype=y.dtype, device=y.device),
                torch.as_tensor(x_va, dtype=x.dtype, device=x.device),
                torch.as_tensor(y_va, dtype=y.dtype, device=y.device),
            )
        except ValueError:
            pass

    generator = torch.Generator(device=x.device if x.is_cuda else "cpu")
    generator.manual_seed(seed)
    n_samples = x.shape[0]
    perm = torch.randperm(n_samples, generator=generator, device=x.device)
    n_val = max(1, int(n_samples * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]
