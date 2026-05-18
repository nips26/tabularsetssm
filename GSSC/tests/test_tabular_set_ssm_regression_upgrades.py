import pathlib
import sys

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.tabular_set_ssm import TabularSetSSM, TabularSetSSMConfig


def _make_reg_model(val_ratio=None, max_epochs=2) -> TabularSetSSM:
    cfg = TabularSetSSMConfig(
        num_features=6,
        hidden_dim=32,
        rank_dim=16,
        num_blocks=2,
        max_experts=4,
        router_top_k=2,
        router_top_k_train_mode="soft_topk",
        val_ratio=val_ratio,
        max_epochs=max_epochs,
        batch_size=32,
        patience=2,
        lr_verbose=False,
        epoch_metrics_verbose=False,
        device="cpu",
        reg_use_smooth_l1=True,
        reg_use_hetero_nll=True,
        reg_calibrate_on_val=True,
        reg_calibrate_variance=True,
    )
    torch.manual_seed(0)
    np.random.seed(0)
    return TabularSetSSM(cfg, task_type="regression")


def _make_reg_data(n=160, f=6):
    x = np.random.randn(n, f).astype(np.float32)
    y = (2.0 * x[:, 0] - 0.7 * x[:, 1] + 0.5 * np.sin(x[:, 2]) + 0.3 * np.random.randn(n)).astype(np.float32)
    return x, y


def test_regression_raw_output_has_mu_logvar_channels():
    m = _make_reg_model(val_ratio=None, max_epochs=1)
    x, y = _make_reg_data(n=64)
    m.fit(x, y)
    xb = torch.tensor(x[:8], dtype=torch.float32)
    with torch.no_grad():
        raw, _ = m._forward_moe_combined(xb, tau=1.0)
    assert raw.shape == (8, 2)


def test_predict_regression_shape_unchanged():
    m = _make_reg_model(val_ratio=None, max_epochs=1)
    x, y = _make_reg_data(n=64)
    m.fit(x, y)
    p = m.predict(x[:10])
    assert p.shape == (10,)


def test_regression_calibration_disabled_without_val_split():
    m = _make_reg_model(val_ratio=0.0, max_epochs=1)
    x, y = _make_reg_data(n=96)
    m.fit(x, y)
    assert float(m.reg_calib_a.item()) == 1.0
    assert float(m.reg_calib_b.item()) == 0.0
    assert float(m.reg_calib_c.item()) == 0.0


def test_regression_calibration_runs_with_explicit_val():
    m = _make_reg_model(val_ratio=0.0, max_epochs=2)
    x, y = _make_reg_data(n=120)
    x_tr, y_tr = x[:80], y[:80]
    x_val, y_val = x[80:], y[80:]
    m.fit(x_tr, y_tr, x_val=x_val, y_val=y_val)
    changed = (
        abs(float(m.reg_calib_a.item()) - 1.0) > 1e-6
        or abs(float(m.reg_calib_b.item())) > 1e-6
        or abs(float(m.reg_calib_c.item())) > 1e-6
    )
    assert changed


def test_regression_calibration_runs_with_internal_val_ratio():
    m = _make_reg_model(val_ratio=0.2, max_epochs=2)
    x, y = _make_reg_data(n=140)
    m.fit(x, y)
    changed = (
        abs(float(m.reg_calib_a.item()) - 1.0) > 1e-6
        or abs(float(m.reg_calib_b.item())) > 1e-6
        or abs(float(m.reg_calib_c.item())) > 1e-6
    )
    assert changed

