"""Simple training entrypoint for TabularSetSSM."""

from __future__ import annotations

from typing import Optional

import numpy as np

from models.tabular_set_ssm import TabularSetSSM, TabularSetSSMConfig


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[TabularSetSSMConfig] = None,
    task_type: str = "auto",
    num_classes: Optional[int] = None,
) -> TabularSetSSM:
    """Construct and train a TabularSetSSM model."""
    if config is None:
        config = TabularSetSSMConfig(num_features=x_train.shape[1])
    model = TabularSetSSM(config=config, task_type=task_type, num_classes=num_classes)
    model.fit(x_train, y_train)
    return model
