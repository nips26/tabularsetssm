import pathlib
import sys

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.tabular_set_ssm import TabularSetSSM, TabularSetSSMConfig


def _make_model(max_experts: int = 4, top_k: int = 2) -> TabularSetSSM:
    cfg = TabularSetSSMConfig(
        num_features=6,
        hidden_dim=32,
        rank_dim=16,
        num_blocks=2,
        max_experts=max_experts,
        router_top_k=top_k,
        router_top_k_train_mode="soft_topk",
        expert_use_glu_ffn=True,
        expert_use_feature_cross=False,
        device="cpu",
        lr_verbose=False,
        epoch_metrics_verbose=False,
    )
    torch.manual_seed(0)
    np.random.seed(0)
    return TabularSetSSM(cfg, task_type="classification", num_classes=2)


def test_topk_weights_are_renormalized():
    m = _make_model(max_experts=5, top_k=2)
    x = torch.randn(11, 6)
    with torch.no_grad():
        p = m._router_routing_probs(m.router(x), tau=1.0)
        w, idx = m._topk_routes(p)
    assert w.shape == (11, 2)
    assert idx.shape == (11, 2)
    assert torch.allclose(w.sum(dim=-1), torch.ones(11), atol=1e-6)


def test_topk_executes_only_selected_experts():
    m = _make_model(max_experts=6, top_k=2)
    x = torch.randn(13, 6)
    calls = [0 for _ in range(len(m.experts))]
    orig = [e.forward for e in m.experts]

    for i, e in enumerate(m.experts):
        def _wrap(*args, _i=i, _orig=orig[i], **kwargs):
            calls[_i] += 1
            return _orig(*args, **kwargs)
        e.forward = _wrap

    m.train()
    with torch.no_grad():
        _ = m.forward(x)
        p = m._router_routing_probs(m.router(x), tau=m._router_tau())
        _, idx = m._topk_routes(p)
        selected = set(idx.reshape(-1).tolist())
    called = {i for i, c in enumerate(calls) if c > 0}
    assert called.issubset(selected)
    assert len(called) <= len(selected)


def test_topk_k_equals_experts_matches_dense_mix():
    m = _make_model(max_experts=4, top_k=4)
    x = torch.randn(9, 6)
    with torch.no_grad():
        x_work = x.unsqueeze(0)
        m._ensure_graph_cache(x_work)
        eigvals, eigvecs = m.feature_graph.get_eigenpairs()
        z = m.pe_encoder(eigvals, eigvecs)
        p = m._router_routing_probs(m.router(x), tau=1.0)
        dense = torch.stack([e(x, z) for e in m.experts], dim=1)
        dense_logits = (dense * p.unsqueeze(-1)).sum(dim=1)
        topk_logits, _ = m._forward_moe_combined(x, tau=1.0)
    assert torch.allclose(topk_logits, dense_logits, atol=1e-5)


def test_context_query_labels_are_masked_and_train_labels_influence():
    m = _make_model(max_experts=4, top_k=2)
    x_tr = torch.randn(8, 6)
    y_tr = torch.randint(0, 2, (8,))
    x_q = torch.randn(5, 6)
    x_all = torch.cat([x_tr, x_q], dim=0)
    n_tr = x_tr.shape[0]
    y_all_a = torch.cat([y_tr, torch.zeros(5, dtype=torch.long)], dim=0)
    y_all_b = torch.cat([y_tr, torch.ones(5, dtype=torch.long)], dim=0)
    with torch.no_grad():
        qa = m.forward_with_context(x_all, y_all_a, single_eval_pos=n_tr)
        qb = m.forward_with_context(x_all, y_all_b, single_eval_pos=n_tr)
    assert torch.allclose(qa, qb, atol=1e-6)

    y_tr_flip = 1 - y_tr
    y_all_c = torch.cat([y_tr_flip, torch.zeros(5, dtype=torch.long)], dim=0)
    with torch.no_grad():
        qc = m.forward_with_context(x_all, y_all_c, single_eval_pos=n_tr)
    assert not torch.allclose(qa, qc, atol=1e-6)


def test_context_query_permutation_and_chunking_invariance():
    m = _make_model(max_experts=4, top_k=2)
    train_x = np.random.randn(10, 6).astype(np.float32)
    train_y = np.random.randint(0, 2, size=(10,), dtype=np.int64)
    query_x = np.random.randn(12, 6).astype(np.float32)

    p_full = m.predict_proba_with_context(train_x, train_y, query_x)

    perm = np.random.permutation(query_x.shape[0])
    p_perm = m.predict_proba_with_context(train_x, train_y, query_x[perm])
    inv = np.argsort(perm)
    assert np.allclose(p_full, p_perm[inv], atol=1e-6)

    p_chunk1 = m.predict_proba_with_context(train_x, train_y, query_x[:6])
    p_chunk2 = m.predict_proba_with_context(train_x, train_y, query_x[6:])
    p_chunks = np.concatenate([p_chunk1, p_chunk2], axis=0)
    assert np.allclose(p_full, p_chunks, atol=1e-6)
