"""End-to-end permutation-invariant global SSM model for tabular data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import inspect
from pathlib import Path
import re
from typing import Any, Literal, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.feature_graph import FeatureGraphBuilder, GraphMethod
from models.moe import TabularSetSSMExpert
from models.positional_encoding import EigenAugmentedPE
from models.router import TreeRouter
from utils import infer_task_type, set_seed, to_float_tensor, train_val_split


TaskType = Literal["classification", "regression"]


@dataclass
class TabularSetSSMConfig:
    """Configuration for TabularSetSSM."""

    num_features: int
    hidden_dim: int = 128
    pe_dim: int = 16
    pe_num_filters: int = 4
    rank_dim: int = 64
    num_blocks: int = 4
    use_selective: bool = True
    dropout: float = 0.1
    graph_method: GraphMethod = "correlation"
    activation: str = "gelu"
    learning_rate: Optional[float] = None
    default_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 15
    # Holdout fraction in (0, 1] when splitting inside fit(). When x_val/y_val are passed explicitly,
    # they always create a validation loader (temperature T tuning, val metrics) even if val_ratio is
    # None or 0. If val_ratio > 0 and no external val is passed, a holdout is split from (x, y).
    val_ratio: Optional[float] = None
    # If True (classification only), use stratified train/val split when val_ratio > 0.
    val_stratify: bool = False
    # Optional per-class weights for CrossEntropyLoss during fit (length = num_classes).
    classification_ce_weight: Optional[tuple[float, ...]] = None
    seed: int = 42
    device: str = "cpu"
    save_best_checkpoint: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-6
    scheduler_monitor: Literal["metric", "train_loss"] = "train_loss"
    lr_finder_end_lr: float = 10.0
    lr_finder_num_iter: int = 100
    lr_finder_step_mode: str = "exp"
    lr_finder_start_lr: float = 1e-6
    lr_finder_max_samples: int = 50000
    lr_finder_diverge_th: float = 8.0
    lr_finder_retry_end_lrs: tuple[float, ...] = (10.0, 1.0, 0.3, 0.1)
    auto_lr: bool = False
    lr_verbose: bool = True
    epoch_metrics_verbose: bool = True
    # Hybrid loss: CE + soft_f1_lambda * (1 - soft_F1) on training (auxiliary)
    soft_f1_lambda: float = 0.5
    # Temperature scaling (logits / T); T tuned on val NLL only when val exists
    temperature_lr: float = 0.05
    temperature_tune_steps: int = 1
    # Router + MoE (tau is router annealing, unrelated to calibration T)
    max_experts: int = 32
    tau_start: float = 1.0
    tau_end: float = 0.01
    anneal_epochs: int = 20
    lambda_balance: float = 0.01
    lambda_sparse: float = 0.001
    expert_usage_threshold: float = 0.01
    hard_routing_after_anneal: bool = True
    router_hidden_dim: int = 64
    router_top_k: int = 2
    router_top_k_train_mode: Literal["soft_topk", "hard_topk_ste"] = "soft_topk"
    expert_use_glu_ffn: bool = True
    expert_ffn_mult: float = 2.0
    expert_use_feature_cross: bool = False
    expert_cross_rank: int = 16
    # Regression upgrades
    reg_use_smooth_l1: bool = True
    reg_smooth_l1_beta: float = 1.0
    reg_alpha_smooth_l1: float = 1.0
    reg_use_hetero_nll: bool = True
    reg_beta_hetero_nll: float = 1.0
    reg_rare_weight_lambda: float = 1.0
    reg_rare_weight_bins: int = 20
    reg_rare_weight_clip_min: float = 0.25
    reg_rare_weight_clip_max: float = 4.0
    reg_rare_weight_eps: float = 1e-6
    reg_calibrate_on_val: bool = True
    reg_calibrate_variance: bool = True

    def __post_init__(self) -> None:
        """
        Keep external callers safe when they instantiate optimizers directly.

        If learning_rate is None, mark auto_lr=True and materialize
        learning_rate with default_learning_rate so code paths that read
        cfg.learning_rate directly do not fail.
        """
        if self.learning_rate is None:
            self.auto_lr = True
            self.learning_rate = float(self.default_learning_rate)
        if self.router_top_k < 1:
            raise ValueError("router_top_k must be >= 1.")
        if self.router_top_k > self.max_experts:
            self.router_top_k = int(self.max_experts)


_EXPERT_HEAD_KEY_PATTERN = re.compile(r"^experts\.\d+\.head\.(weight|bias)$")


def normalize_state_dict_key(key: str) -> str:
    """Strip ``module.`` / ``_orig_mod.`` prefixes from a ``state_dict`` key."""
    nk = key
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
                changed = True
    return nk


def strip_checkpoint_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``state_dict`` with wrapper prefixes removed from keys."""
    return {normalize_state_dict_key(k): v for k, v in state_dict.items()}


def infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> Optional[int]:
    """
    Infer ``C`` from expert ``Linear`` head weights ``[C, H]`` or bias ``[C]``.

    Tolerates ``module.`` / ``_orig_mod.`` key prefixes and array-like values.
    """
    for key, param in state_dict.items():
        nk = normalize_state_dict_key(key)
        if not _EXPERT_HEAD_KEY_PATTERN.match(nk):
            continue
        if isinstance(param, torch.Tensor):
            if param.dim() == 2:
                return int(param.shape[0])
            if param.dim() == 1:
                return int(param.shape[0])
        shape = getattr(param, "shape", None)
        if shape is not None:
            if len(shape) == 2:
                return int(shape[0])
            if len(shape) == 1:
                return int(shape[0])
    return None


class TabularSetSSM(nn.Module):
    """
    Global factorized SSM for tabular data with tree-routed MoE.

    A single shared ``FeatureGraphBuilder`` and ``EigenAugmentedPE`` supply ``z``
    to every expert; experts only duplicate SSM blocks and output heads.

    Classification supports any ``C >= 2`` classes: each expert ends with
    ``Linear(H, C)`` and the loss is multi-class softmax + CE (plus soft-F1).
    If ``num_classes`` is omitted, ``fit(x, y)`` sets it from the number of
    distinct labels in ``y``.

    Input shape:
        - [N, F] (single dataset)
        - [B, N, F] (batched meta-datasets)

    Internal shape:
        [B, N, F, H] where each feature value is treated as a node signal.
    """

    def __init__(
        self,
        config: TabularSetSSMConfig,
        task_type: Literal["auto", "classification", "regression"] = "auto",
        num_classes: Optional[int] = None,
    ) -> None:
        """
        Args:
            num_classes: Output dimension for classification heads. If ``None``,
                it is set on the first ``fit(x, y)`` from ``torch.unique(y)`` (any
                ``C >= 2``). Passing an integer builds heads immediately so
                ``forward`` works before ``fit`` (optional).
        """
        super().__init__()
        self.config = config
        self.task_type_mode = task_type
        self.task_type_: Optional[TaskType] = None if task_type == "auto" else task_type
        self.num_classes = num_classes

        self.feature_graph = FeatureGraphBuilder(
            num_features=config.num_features,
            pe_dim=config.pe_dim,
            method=config.graph_method,
        )
        self.pe_encoder = EigenAugmentedPE(
            pe_dim=config.pe_dim,
            num_filters=config.pe_num_filters,
            dropout=config.dropout,
        )

        eh = max(1, config.hidden_dim // 2)
        eb = max(1, config.num_blocks // 2)
        er = max(1, config.rank_dim // 2)
        z_dim_shared = int(self.pe_encoder.output_dim)
        self.experts = nn.ModuleList(
            [
                TabularSetSSMExpert(
                    num_features=config.num_features,
                    hidden_dim=eh,
                    z_dim=z_dim_shared,
                    rank_dim=er,
                    num_blocks=eb,
                    use_selective=config.use_selective,
                    dropout=config.dropout,
                    activation=config.activation,
                    use_glu_ffn=config.expert_use_glu_ffn,
                    ffn_mult=config.expert_ffn_mult,
                    use_feature_cross=config.expert_use_feature_cross,
                    cross_rank=config.expert_cross_rank,
                )
                for _ in range(config.max_experts)
            ]
        )
        self.router = TreeRouter(
            num_experts=config.max_experts,
            hidden_dim=config.router_hidden_dim,
        )

        self.head: Optional[nn.Linear] = None
        self._expert_hidden_dim: int = eh
        self.class_context_embed: Optional[nn.Embedding] = None
        self.reg_context_proj: Optional[nn.Linear] = None
        # Positive temperature via exp (init T=1); only used for classification
        self.log_temperature: Optional[nn.Parameter] = None
        self.best_checkpoint_path: Optional[str] = None
        self.current_learning_rate: Optional[float] = None

        self._fit_epoch: int = 0
        self._tau_value: float = float(config.tau_start)
        self._usage_ema: Optional[torch.Tensor] = None
        self._annealing_complete: bool = False
        self._active_expert_mask: Optional[torch.Tensor] = None
        self._k_active: int = config.max_experts
        self._printed_annealing_footer: bool = False
        self._printed_inference_experts_msg: bool = False
        self._reg_min_log_var: float = -8.0
        self._reg_max_log_var: float = 8.0
        self._reg_bin_edges: Optional[torch.Tensor] = None
        self._reg_bin_weights: Optional[torch.Tensor] = None
        self.register_buffer("reg_calib_a", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("reg_calib_b", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("reg_calib_c", torch.tensor(0.0, dtype=torch.float32))

        self.to(torch.device(config.device))
        dev = torch.device(config.device)
        if self.task_type_ == "regression":
            self._materialize_expert_heads(n_classes=None, device=dev)
        elif self.task_type_ == "classification" and self.num_classes is not None:
            self._materialize_expert_heads(n_classes=int(self.num_classes), device=dev)

    strip_checkpoint_state_dict_keys = staticmethod(strip_checkpoint_state_dict_keys)
    infer_num_classes_from_state_dict = staticmethod(infer_num_classes_from_state_dict)

    @staticmethod
    def _resolve_training_artifact_dir() -> Path:
        """
        Resolve a directory near the training call site.

        - For regular Python files: use the caller file's parent directory.
        - For notebooks / interactive sessions: fallback to current working dir.
        """
        this_file = Path(__file__).resolve()
        for frame_info in inspect.stack()[2:]:
            frame_path = Path(frame_info.filename).resolve()
            if frame_path == this_file:
                continue
            if frame_path.exists() and frame_path.suffix in {".py", ".ipynb"}:
                return frame_path.parent
        return Path.cwd()

    @staticmethod
    def _build_checkpoint_payload(
        state_dict: dict[str, torch.Tensor],
        config: TabularSetSSMConfig,
        task_type: Optional[TaskType],
        num_classes: Optional[int],
        best_epoch: int,
        best_metric: float,
        metric_name: str,
    ) -> dict[str, object]:
        """Build standardized checkpoint payload."""
        return {
            "model_state_dict": state_dict,
            "config": config.__dict__.copy(),
            "task_type": task_type,
            "num_classes": num_classes,
            "best_epoch": best_epoch,
            "best_metric": float(best_metric),
            "best_metric_name": metric_name,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _build_head(self, y: torch.Tensor) -> None:
        if self.task_type_ is None:
            self.task_type_ = infer_task_type(y)
        dev = next(self.parameters()).device
        n_classes: Optional[int] = None
        if self.task_type_ == "classification":
            n_classes = self.num_classes or int(torch.unique(y).numel())
            self.num_classes = n_classes
        self._materialize_expert_heads(n_classes=n_classes, device=dev)

    def _materialize_expert_heads(self, *, n_classes: Optional[int], device: torch.device) -> None:
        """Create expert output heads (+ calibration log_temperature for classification)."""
        if self.task_type_ is None:
            raise RuntimeError("Internal: task_type_ must be set before materializing expert heads.")
        for ex in self.experts:
            ex.build_head(self.task_type_, n_classes, device)
        if self.task_type_ == "classification":
            if self.log_temperature is None:
                self.log_temperature = nn.Parameter(torch.zeros(1, device=device))
            self.log_temperature.requires_grad = False
            if n_classes is None:
                raise RuntimeError("n_classes is required for classification context embedding.")
            self.class_context_embed = nn.Embedding(int(n_classes), self._expert_hidden_dim).to(device)
            self.reg_context_proj = None
        else:
            self.class_context_embed = None
            self.reg_context_proj = nn.Linear(1, self._expert_hidden_dim).to(device)
        self.head = None

    def _ensure_expert_heads_ready(self, x: torch.Tensor) -> None:
        """
        Expert heads are normally created in fit() or at __init__ when task_type/num_classes are known.
        Manual training loops can call forward first if task_type is explicit (and num_classes for classification).
        """
        if not self.experts:
            return
        if self.experts[0].head is not None:
            return
        if self.task_type_ is None:
            raise RuntimeError(
                "TabularSetSSM: expert heads are not initialized. Call fit(x, y) first "
                "(classification num_classes is inferred from y), or pass task_type and optional num_classes."
            )
        if self.task_type_ == "classification" and self.num_classes is None:
            raise RuntimeError(
                "TabularSetSSM: call fit(x, y) first so num_classes can be inferred from labels, "
                "or pass num_classes=... to the constructor for forward-only use before fit."
            )
        self._materialize_expert_heads(
            n_classes=int(self.num_classes) if self.task_type_ == "classification" else None,
            device=x.device,
        )

    def _tau_schedule(self, epoch: int) -> float:
        t0 = float(self.config.tau_start)
        t1 = float(self.config.tau_end)
        te = max(1e-8, float(self.config.anneal_epochs))
        e = (epoch - 1) / te
        return float(max(t1, t0 * ((t1 / t0) ** e)))

    def _training_uses_ste(self) -> bool:
        return bool(
            self.config.hard_routing_after_anneal
            and self._fit_epoch > int(self.config.anneal_epochs)
        )

    def _eval_use_hard_gather(self) -> bool:
        if self.training:
            return False
        if self._annealing_complete:
            return True
        return self._fit_epoch > int(self.config.anneal_epochs)

    def _output_dim_predict(self) -> int:
        if self.task_type_ == "classification":
            if self.num_classes is None:
                raise RuntimeError("num_classes unknown before fit.")
            return int(self.num_classes)
        return 2

    def _apply_active_expert_mask_to_logits(self, router_logits: torch.Tensor) -> torch.Tensor:
        if self._active_expert_mask is None:
            return router_logits
        m = self._active_expert_mask.to(router_logits.device)
        return router_logits.masked_fill(~m.unsqueeze(0), float("-inf"))

    @staticmethod
    def _router_scaled_stable(router_logits: torch.Tensor, tau: float, clamp_abs: float = 20.0) -> torch.Tensor:
        """
        Scale router logits by tau, clamp finite values to [-clamp_abs, clamp_abs] to avoid
        NaNs when tau is tiny; leave -inf (masked experts) untouched for softmax.
        """
        tau_safe = max(float(tau), 1e-6)
        s = router_logits / tau_safe
        return torch.where(torch.isfinite(s), s.clamp(-clamp_abs, clamp_abs), s)

    def _router_routing_probs(self, router_logits: torch.Tensor, tau: float) -> torch.Tensor:
        """Stable routing distribution (softmax on scaled, clamped logits)."""
        s = self._router_scaled_stable(router_logits, tau)
        return F.softmax(s, dim=-1)

    def _topk_routes(self, p_soft: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = max(1, min(int(self.config.router_top_k), p_soft.shape[-1]))
        vals, idx = torch.topk(p_soft, k=k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return vals, idx

    def _gather_expert_outputs_topk(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        row_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        b, k = int(indices.size(0)), int(indices.size(1))
        od = self._output_dim_predict()
        out = torch.zeros(b, od, device=device, dtype=dtype)
        row_ids = torch.arange(b, device=device).unsqueeze(-1).expand(-1, k).reshape(-1)
        exp_ids = indices.reshape(-1)
        ws = weights.reshape(-1)
        for kid_t in torch.unique(exp_ids, sorted=True):
            kid = int(kid_t.item())
            sel = torch.nonzero(exp_ids == kid_t, as_tuple=False).view(-1)
            if sel.numel() == 0:
                continue
            rows = row_ids.index_select(0, sel)
            if context_mask is not None or query_mask is not None:
                y_full = self.experts[kid](
                    x,
                    z,
                    context_mask=context_mask,
                    query_mask=query_mask,
                    row_context=row_context,
                ).to(dtype=out.dtype)
                y_grp = y_full.index_select(0, rows)
            else:
                x_grp = x.index_select(0, rows)
                ctx_grp = None if row_context is None else row_context.index_select(0, rows)
                y_grp = self.experts[kid](
                    x_grp,
                    z,
                    context_mask=None,
                    query_mask=None,
                    row_context=ctx_grp,
                ).to(dtype=out.dtype)
            out.index_add_(0, rows, y_grp * ws.index_select(0, sel).unsqueeze(-1).to(dtype=out.dtype))
        return out

    def _forward_moe_combined(
        self,
        x: torch.Tensor,
        tau: float,
        context_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        row_context: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_expert_heads_ready(x)
        x_work = x.unsqueeze(0) if x.dim() == 2 else x
        self._ensure_graph_cache(x_work)
        eigvals, eigvecs = self.feature_graph.get_eigenpairs()
        z = self.pe_encoder(eigvals.to(x.device), eigvecs.to(x.device))

        router_logits = self.router(x)
        router_logits = self._apply_active_expert_mask_to_logits(router_logits)
        p_soft = self._router_routing_probs(router_logits, tau)

        eval_hard = self._eval_use_hard_gather()
        topk_w_soft, topk_idx = self._topk_routes(p_soft)
        if eval_hard:
            hard_idx = topk_idx[:, :1]
            hard_w = torch.ones_like(hard_idx, dtype=p_soft.dtype, device=p_soft.device)
            logits = self._gather_expert_outputs_topk(
                x,
                z,
                hard_w,
                hard_idx,
                context_mask=context_mask,
                query_mask=query_mask,
                row_context=row_context,
            )
            return logits, p_soft

        if self._training_uses_ste() or self.config.router_top_k_train_mode == "hard_topk_ste":
            hard_w = torch.ones_like(topk_w_soft)
            p_route = hard_w - topk_w_soft.detach() + topk_w_soft
        else:
            p_route = topk_w_soft
        logits = self._gather_expert_outputs_topk(
            x,
            z,
            p_route,
            topk_idx,
            context_mask=context_mask,
            query_mask=query_mask,
            row_context=row_context,
        )
        return logits, p_soft

    def _router_tau(self) -> float:
        return float(self._tau_value)

    def get_temperature(self) -> torch.Tensor:
        """Scalar T>0 (init 1.0). Only for classification; otherwise 1.0."""
        if self.log_temperature is None:
            z = next(self.parameters())
            return torch.ones((), device=z.device, dtype=z.dtype)
        return torch.exp(self.log_temperature).squeeze().clamp(min=1e-6)

    def _raw_logits(
        self,
        x: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        row_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self._forward_moe_combined(
            x,
            self._router_tau(),
            context_mask=context_mask,
            query_mask=query_mask,
            row_context=row_context,
        )
        return logits

    @staticmethod
    def _soft_f1_loss(scaled_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """1 - macro soft-F1 from softmax probabilities (matches CE on same scaled logits)."""
        p = F.softmax(scaled_logits, dim=-1)
        c = p.shape[-1]
        y_oh = F.one_hot(y, num_classes=c).float()
        tp = (p * y_oh).sum(0)
        fp = p.sum(0) - tp
        fn = y_oh.sum(0) - tp
        eps = 1e-7
        f1 = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        return 1.0 - f1.mean()

    def _split_regression_raw(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if raw.dim() < 2 or raw.shape[-1] != 2:
            raise ValueError("Regression raw output must have shape [B, 2].")
        mu = raw[..., 0]
        log_var = raw[..., 1].clamp(min=self._reg_min_log_var, max=self._reg_max_log_var)
        return mu, log_var

    def _apply_regression_calibration(
        self,
        mu: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        a = self.reg_calib_a.to(device=mu.device, dtype=mu.dtype)
        b = self.reg_calib_b.to(device=mu.device, dtype=mu.dtype)
        mu_cal = a * mu + b
        if log_var is None:
            return mu_cal, None
        c = self.reg_calib_c.to(device=log_var.device, dtype=log_var.dtype)
        return mu_cal, log_var + c

    def _gaussian_nll_from_mu_logvar(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        inv_var = torch.exp(-log_var)
        sq_err = (target - mu) ** 2
        return 0.5 * (log_var + sq_err * inv_var)

    def _build_regression_rare_weights(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bins = max(2, int(self.config.reg_rare_weight_bins))
        eps = float(self.config.reg_rare_weight_eps)
        q = torch.linspace(0.0, 1.0, steps=bins + 1, device=y.device, dtype=y.dtype)
        edges = torch.quantile(y, q)
        edges = torch.cummax(edges, dim=0).values
        edges[0] = torch.min(y) - eps
        for i in range(1, edges.numel()):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + eps
        edges[-1] = torch.maximum(edges[-1], torch.max(y) + eps)
        bin_idx = torch.bucketize(y, edges[1:-1], right=False)
        counts = torch.bincount(bin_idx, minlength=bins).to(dtype=y.dtype)
        inv = 1.0 / (counts + eps)
        w = inv[bin_idx]
        w = w / w.mean().clamp_min(eps)
        w = w.clamp(min=float(self.config.reg_rare_weight_clip_min), max=float(self.config.reg_rare_weight_clip_max))
        return edges.detach(), w.detach()

    def _map_regression_rare_weights(self, y: torch.Tensor) -> torch.Tensor:
        if self._reg_bin_edges is None or self._reg_bin_weights is None:
            return torch.ones_like(y, dtype=y.dtype, device=y.device)
        edges = self._reg_bin_edges.to(device=y.device, dtype=y.dtype)
        bw = self._reg_bin_weights.to(device=y.device, dtype=y.dtype)
        idx = torch.bucketize(y, edges[1:-1], right=False)
        idx = idx.clamp(max=bw.shape[0] - 1)
        return bw[idx]

    @staticmethod
    def _expected_calibration_error_toplabel(
        y_true: np.ndarray,
        prob: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        y_true = np.asarray(y_true).astype(int)
        prob = np.asarray(prob, dtype=float)
        conf = prob.max(axis=1)
        pred = prob.argmax(axis=1)
        correct = (pred == y_true).astype(float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(conf, bins[1:-1], right=False)
        ece = 0.0
        n = len(y_true)
        if n == 0:
            return 0.0
        for b in range(n_bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
        return float(ece)

    def _compute_classification_metrics(self, y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
        y_true = np.asarray(y_true).astype(int)
        prob = np.asarray(prob, dtype=float)
        pred = prob.argmax(axis=1)
        metrics = {
            "accuracy": float(accuracy_score(y_true, pred)),
            "precision": float(
                precision_score(
                    y_true,
                    pred,
                    average="binary" if prob.shape[1] == 2 else "macro",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    pred,
                    average="binary" if prob.shape[1] == 2 else "macro",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    pred,
                    average="binary" if prob.shape[1] == 2 else "macro",
                    zero_division=0,
                )
            ),
            "ece": self._expected_calibration_error_toplabel(y_true, prob, n_bins=15),
        }
        try:
            if prob.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob[:, 1]))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        return metrics

    @staticmethod
    def _rankdata_average(a: np.ndarray) -> np.ndarray:
        """Average-rank transform (1..N), tie-aware, NumPy-only."""
        x = np.asarray(a, dtype=float).ravel()
        n = x.size
        if n == 0:
            return np.asarray([], dtype=float)
        order = np.argsort(x, kind="mergesort")
        xs = x[order]
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i + 1
            while j < n and xs[j] == xs[i]:
                j += 1
            avg_rank = 0.5 * ((i + 1) + j)
            ranks[order[i:j]] = avg_rank
            i = j
        return ranks

    @classmethod
    def _spearman_rank_corr(cls, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
            return float("nan")
        rt = cls._rankdata_average(yt)
        rp = cls._rankdata_average(yp)
        rt_c = rt - rt.mean()
        rp_c = rp - rp.mean()
        den = float(np.sqrt(np.sum(rt_c**2) * np.sum(rp_c**2)))
        if den <= 0.0:
            return float("nan")
        return float(np.sum(rt_c * rp_c) / den)

    @classmethod
    def _compute_regression_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        if yt.size == 0 or yp.size == 0 or yt.size != yp.size:
            return {"rmse": float("nan"), "spearman": float("nan"), "r2": float("nan"), "mae": float("nan")}
        err = yp - yt
        mse = float(np.mean(err**2))
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(mse))
        var = float(np.var(yt))
        r2 = float(1.0 - mse / var) if var > 0.0 else float("nan")
        spearman = cls._spearman_rank_corr(yt, yp)
        return {"rmse": rmse, "spearman": spearman, "r2": r2, "mae": mae}

    def _resolve_initial_lr(
        self,
        x_tr: torch.Tensor,
        y_tr: torch.Tensor,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        """
        Resolve initial LR for the current dataset.

        If config.learning_rate is provided, use it directly.
        Otherwise, try torch-lr-finder and fallback to default_learning_rate.
        """
        base_lr = float(
            self.config.learning_rate
            if self.config.learning_rate is not None
            else self.config.default_learning_rate
        )
        if not self.config.auto_lr:
            lr = base_lr
            if self.config.lr_verbose:
                print(f"[TabularSetSSM] Using user-specified learning rate: {lr:.8f}")
            return lr

        try:
            from torch_lr_finder import LRFinder  # type: ignore
        except Exception:
            fallback_lr = base_lr
            if self.config.lr_verbose:
                print(
                    "[TabularSetSSM] torch-lr-finder unavailable; "
                    f"falling back to configured learning_rate={fallback_lr:.8f}"
                )
            return fallback_lr

        # Use a sampled subset for LR search on large datasets.
        x_lr = x_tr
        y_lr = y_tr
        if x_tr.shape[0] > self.config.lr_finder_max_samples:
            perm = torch.randperm(x_tr.shape[0], device=x_tr.device)[: self.config.lr_finder_max_samples]
            x_lr = x_tr[perm]
            y_lr = y_tr[perm]
            if self.config.lr_verbose:
                print(
                    "[TabularSetSSM] LR finder sampling subset: "
                    f"{x_lr.shape[0]}/{x_tr.shape[0]} rows"
                )

        # Snapshot model state to safely restore after LR search.
        model_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
        lr_loader = DataLoader(
            TensorDataset(x_lr, y_lr),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        suggested_lr = base_lr
        used_finder = False
        retry_end_lrs = tuple(self.config.lr_finder_retry_end_lrs) or (self.config.lr_finder_end_lr,)

        try:
            for end_lr in retry_end_lrs:
                lr_param_groups = list(self.parameters())
                if self.log_temperature is not None:
                    lr_param_groups = [p for n, p in self.named_parameters() if n != "log_temperature"]
                temp_optimizer = torch.optim.Adam(
                    lr_param_groups,
                    lr=float(self.config.lr_finder_start_lr),
                    weight_decay=self.config.weight_decay,
                )
                lr_finder = LRFinder(self, temp_optimizer, criterion, device=str(device))
                if self.config.lr_verbose:
                    print(
                        "[TabularSetSSM] LR finder attempt "
                        f"(start_lr={self.config.lr_finder_start_lr:.2e}, end_lr={float(end_lr):.4f}, "
                        f"num_iter={int(self.config.lr_finder_num_iter)})"
                    )
                try:
                    lr_finder.range_test(
                        lr_loader,
                        end_lr=float(end_lr),
                        num_iter=int(self.config.lr_finder_num_iter),
                        step_mode=self.config.lr_finder_step_mode,
                        diverge_th=float(self.config.lr_finder_diverge_th),
                    )
                    suggestion = lr_finder.suggestion()
                    losses = np.asarray(lr_finder.history.get("loss", []), dtype=float)
                    lrs = np.asarray(lr_finder.history.get("lr", []), dtype=float)
                    valid = np.isfinite(losses) & np.isfinite(lrs) & (lrs > 0)

                    if suggestion is not None and np.isfinite(suggestion) and suggestion > 0:
                        suggested_lr = float(max(suggestion, self.config.lr_finder_start_lr))
                        used_finder = True
                    elif valid.any():
                        losses_v = losses[valid]
                        lrs_v = lrs[valid]
                        # Trim noisy boundaries if we have enough points.
                        if len(losses_v) >= 10:
                            lo = int(0.1 * len(losses_v))
                            hi = max(lo + 1, int(0.9 * len(losses_v)))
                            losses_core = losses_v[lo:hi]
                            lrs_core = lrs_v[lo:hi]
                        else:
                            losses_core = losses_v
                            lrs_core = lrs_v
                        best_idx = int(np.argmin(losses_core))
                        # Conservative pick below minimum-loss LR for stability.
                        suggested_lr = float(max(lrs_core[best_idx] / 5.0, self.config.lr_finder_start_lr))
                        used_finder = True
                except Exception:
                    used_finder = False
                finally:
                    try:
                        lr_finder.reset()
                    except Exception:
                        pass
                    # restore pre-search state before potential retry
                    self.load_state_dict(model_state)
                    self.to(device)
                    self.train()

                if used_finder:
                    break
        except Exception:
            suggested_lr = base_lr
            used_finder = False

        # Final restore guard.
        self.load_state_dict(model_state)
        self.to(device)
        self.train()

        if self.config.lr_verbose:
            if used_finder:
                print(f"[TabularSetSSM] LR finder selected start learning rate: {suggested_lr:.8f}")
            else:
                print(
                    "[TabularSetSSM] LR finder did not produce a stable suggestion; "
                    f"using configured learning_rate={suggested_lr:.8f}"
                )
        return float(suggested_lr)

    def _ensure_graph_cache(self, x: torch.Tensor) -> None:
        if not self.feature_graph.is_fitted():
            self.feature_graph.fit(x.detach())
        self.feature_graph.to(x.device)

    def _build_row_context(
        self,
        y_all: Optional[torch.Tensor],
        single_eval_pos: int,
        n_rows: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if y_all is None:
            return None
        if y_all.dim() != 1 or int(y_all.shape[0]) != n_rows:
            raise ValueError("y_all must have shape [N] and match x_all rows.")
        if self.task_type_ == "classification":
            if self.class_context_embed is None:
                return None
            y_ctx = y_all.clone().to(device=device, dtype=torch.long)
            y_ctx[single_eval_pos:] = 0
            emb = self.class_context_embed(y_ctx)
            mask = torch.zeros(n_rows, device=device, dtype=emb.dtype)
            mask[:single_eval_pos] = 1.0
            return emb * mask.unsqueeze(-1)
        if self.task_type_ == "regression":
            if self.reg_context_proj is None:
                return None
            y_ctx = y_all.to(device=device, dtype=torch.float32)
            y_ctx[single_eval_pos:] = 0.0
            emb = self.reg_context_proj(y_ctx.unsqueeze(-1))
            mask = torch.zeros(n_rows, device=device, dtype=emb.dtype)
            mask[:single_eval_pos] = 1.0
            return emb * mask.unsqueeze(-1)
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._raw_logits(x)
        if self.task_type_ == "classification":
            if self.log_temperature is None:
                return raw
            return raw / self.get_temperature()
        mu, _ = self._split_regression_raw(raw)
        mu, _ = self._apply_regression_calibration(mu, None)
        return mu

    def forward_with_context(
        self,
        x_all: torch.Tensor,
        y_all_optional: Optional[torch.Tensor],
        single_eval_pos: int,
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        self._ensure_expert_heads_ready(x_all)
        if x_all.dim() != 2:
            raise ValueError("x_all must have shape [N, F].")
        n_rows = int(x_all.shape[0])
        if not (0 < int(single_eval_pos) < n_rows):
            raise ValueError("single_eval_pos must be in (0, N).")
        device = x_all.device
        context_mask = torch.zeros(n_rows, device=device, dtype=torch.bool)
        context_mask[:single_eval_pos] = True
        query_mask = ~context_mask
        row_context = self._build_row_context(y_all_optional, single_eval_pos, n_rows, device)
        logits = self._raw_logits(
            x_all,
            context_mask=context_mask,
            query_mask=query_mask,
            row_context=row_context,
        )
        if self.task_type_ == "classification":
            if self.log_temperature is not None:
                logits = logits / self.get_temperature()
            return logits if return_all_logits else logits[single_eval_pos:]
        mu, _ = self._split_regression_raw(logits)
        mu, _ = self._apply_regression_calibration(mu, None)
        return mu if return_all_logits else mu[single_eval_pos:]

    def predict_proba_with_context(
        self,
        train_rows: np.ndarray | torch.Tensor,
        train_labels: np.ndarray | torch.Tensor,
        query_rows: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        if self.task_type_ != "classification":
            raise RuntimeError("predict_proba_with_context() is only available for classification.")
        device = torch.device(self.config.device)
        x_tr = to_float_tensor(train_rows, device=device)
        x_q = to_float_tensor(query_rows, device=device)
        y_tr = torch.as_tensor(train_labels, device=device)
        if y_tr.dim() > 1 and y_tr.shape[-1] == 1:
            y_tr = y_tr.squeeze(-1)
        n_tr = int(x_tr.shape[0])
        x_all = torch.cat([x_tr, x_q], dim=0)
        y_all = torch.zeros(int(x_all.shape[0]), device=device, dtype=y_tr.dtype)
        y_all[:n_tr] = y_tr
        logits_q = self.forward_with_context(x_all, y_all, single_eval_pos=n_tr, return_all_logits=False)
        return torch.softmax(logits_q, dim=-1).detach().cpu().numpy()

    def _tune_temperature_on_val(
        self,
        val_loader: DataLoader,
        device: torch.device,
        ce_weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Minimize val NLL w.r.t. log_temperature only; backbone frozen."""
        if self.log_temperature is None:
            return
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.log_temperature.requires_grad = True
        opt_t = torch.optim.Adam([self.log_temperature], lr=float(self.config.temperature_lr))
        ce_kw: dict = {}
        if ce_weight is not None:
            ce_kw["weight"] = ce_weight
        for _ in range(max(1, int(self.config.temperature_tune_steps))):
            for xb, yb in val_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.long)
                opt_t.zero_grad(set_to_none=True)
                raw = self._raw_logits(xb)
                t = self.get_temperature()
                scaled = raw / t
                loss = F.cross_entropy(scaled, yb, **ce_kw)
                loss.backward()
                opt_t.step()
        for p in self.parameters():
            p.requires_grad = True
        self.log_temperature.requires_grad = False

    def _set_identity_regression_calibration(self) -> None:
        self.reg_calib_a.fill_(1.0)
        self.reg_calib_b.fill_(0.0)
        self.reg_calib_c.fill_(0.0)

    @torch.no_grad()
    def _tune_regression_calibration_on_val(
        self,
        val_loader: DataLoader,
        device: torch.device,
    ) -> None:
        if self.task_type_ != "regression" or not bool(self.config.reg_calibrate_on_val):
            return
        mu_parts: list[torch.Tensor] = []
        lv_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []
        self.eval()
        for xb, yb in val_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            raw = self._raw_logits(xb)
            mu, log_var = self._split_regression_raw(raw)
            mu_parts.append(mu.detach().cpu())
            lv_parts.append(log_var.detach().cpu())
            y_parts.append(yb.detach().cpu())
        if not mu_parts:
            self._set_identity_regression_calibration()
            return
        mu = torch.cat(mu_parts, dim=0).float()
        yv = torch.cat(y_parts, dim=0).float()
        lv = torch.cat(lv_parts, dim=0).float()

        x_mean = mu.mean()
        y_mean = yv.mean()
        var_x = ((mu - x_mean) ** 2).mean().clamp_min(1e-12)
        cov_xy = ((mu - x_mean) * (yv - y_mean)).mean()
        a = cov_xy / var_x
        b = y_mean - a * x_mean
        mu_cal = a * mu + b

        c = torch.tensor(0.0)
        if bool(self.config.reg_calibrate_variance):
            mse = (yv - mu_cal) ** 2
            base = mse * torch.exp(-lv)
            c = torch.log(base.mean().clamp_min(1e-12))
            c = c.clamp(min=-5.0, max=5.0)

        self.reg_calib_a.copy_(a.to(dtype=self.reg_calib_a.dtype))
        self.reg_calib_b.copy_(b.to(dtype=self.reg_calib_b.dtype))
        self.reg_calib_c.copy_(c.to(dtype=self.reg_calib_c.dtype))

    def fit(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        x_val: Optional[np.ndarray | torch.Tensor] = None,
        y_val: Optional[np.ndarray | torch.Tensor] = None,
    ) -> "TabularSetSSM":
        """Train the model with early stopping."""
        set_seed(self.config.seed)
        device = torch.device(self.config.device)
        self.train()

        # Keep full datasets on CPU to avoid large-device allocations for big tabular sets.
        x_t = to_float_tensor(x, device=torch.device("cpu"))
        y_t = torch.as_tensor(y, device=torch.device("cpu"))
        if y_t.dim() > 1 and y_t.shape[-1] == 1:
            y_t = y_t.squeeze(-1)

        self._build_head(y_t)
        if self.task_type_ == "classification":
            y_t = y_t.long()
        else:
            y_t = y_t.float()

        external_val_given = x_val is not None and y_val is not None
        split_nontrivial = self.config.val_ratio is not None and float(self.config.val_ratio) > 0.0

        used_external_val = False
        if external_val_given:
            x_val_t = to_float_tensor(x_val, device=torch.device("cpu"))
            y_val_t = torch.as_tensor(y_val, device=torch.device("cpu"))
            if y_val_t.dim() > 1 and y_val_t.shape[-1] == 1:
                y_val_t = y_val_t.squeeze(-1)
            if self.task_type_ == "classification":
                y_val_t = y_val_t.long()
            else:
                y_val_t = y_val_t.float()
            x_tr, y_tr = x_t, y_t
            x_val, y_val = x_val_t, y_val_t
            used_external_val = True
            split_ratio = float(self.config.val_ratio) if split_nontrivial else 0.0
        elif split_nontrivial:
            split_ratio = float(self.config.val_ratio)
            strat = bool(self.config.val_stratify) and self.task_type_ == "classification"
            x_tr, y_tr, x_val, y_val = train_val_split(
                x_t, y_t, val_ratio=split_ratio, seed=self.config.seed, stratify=strat
            )
        else:
            x_tr, y_tr = x_t, y_t
            x_val, y_val = None, None
            split_ratio = 0.0

        train_loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader: Optional[DataLoader] = None
        if x_val is not None and y_val is not None:
            val_loader = DataLoader(
                TensorDataset(x_val, y_val),
                batch_size=self.config.batch_size,
                shuffle=False,
            )

        if self.config.lr_verbose or self.config.epoch_metrics_verbose:
            if val_loader is not None:
                if used_external_val:
                    print(
                        f"[TabularSetSSM] External validation: n_train={x_tr.shape[0]} n_val={x_val.shape[0]} — "
                        "temperature T will be tuned on this val split."
                    )
                else:
                    strat_note = bool(self.config.val_stratify) and self.task_type_ == "classification"
                    print(
                        f"[TabularSetSSM] Random holdout validation: n_train={x_tr.shape[0]} n_val={x_val.shape[0]} "
                        f"(val_ratio={split_ratio:.4f}, stratify={strat_note}) — temperature T will be tuned on val."
                    )
            else:
                print(
                    "[TabularSetSSM] No validation split (val_ratio is 0 or None, and no x_val/y_val). "
                    "Calibration temperature T stays at 1.0 (learnable T updates disabled)."
                )

        # Refit feature graph for every fit() call to adapt to each dataset.
        self.feature_graph.fit(x_tr.detach())
        self.feature_graph.to(device)

        ce_weight_t: Optional[torch.Tensor] = None
        if self.task_type_ == "classification" and self.config.classification_ce_weight is not None:
            ce_weight_t = torch.tensor(
                list(self.config.classification_ce_weight), dtype=torch.float32, device=device
            )

        criterion = (
            nn.CrossEntropyLoss(weight=ce_weight_t)
            if self.task_type_ == "classification"
            else nn.SmoothL1Loss(beta=float(self.config.reg_smooth_l1_beta))
        )
        initial_lr = self._resolve_initial_lr(x_tr=x_tr, y_tr=y_tr, criterion=criterion, device=device)
        self.current_learning_rate = float(initial_lr)
        if self.config.lr_verbose:
            print(f"[TabularSetSSM] Starting optimizer LR: {self.current_learning_rate:.8f}")

        main_params = list(self.parameters())
        if self.log_temperature is not None:
            main_params = [p for n, p in self.named_parameters() if n != "log_temperature"]
        optimizer = torch.optim.Adam(
            main_params,
            lr=initial_lr,
            weight_decay=self.config.weight_decay,
        )

        maximize_metric = self.task_type_ == "classification"
        best_metric = float("-inf") if maximize_metric else float("inf")
        best_state = None
        best_epoch = -1
        patience_left = self.config.patience
        if self.task_type_ == "classification" and val_loader is not None:
            metric_name = "val_roc_auc"
        elif self.task_type_ == "classification":
            metric_name = "train_roc_auc"
        elif x_val is not None and y_val is not None:
            metric_name = "val_nll"
        else:
            metric_name = "train_loss"
        scheduler_mode = "max" if metric_name in {"train_roc_auc", "val_roc_auc"} else "min"
        if self.config.scheduler_monitor == "train_loss":
            scheduler_mode = "min"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.scheduler_min_lr,
        )

        checkpoint_dir: Optional[Path] = None
        if self.config.save_best_checkpoint:
            train_artifact_root = self._resolve_training_artifact_dir()
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = train_artifact_root / f"ssm_training_best_epochs_{run_stamp}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._annealing_complete = False
        self._active_expert_mask = None
        self._usage_ema = torch.ones(self.config.max_experts, dtype=torch.float32) / float(self.config.max_experts)
        best_usage_ema: Optional[torch.Tensor] = None
        self._set_identity_regression_calibration()
        ce_kw: dict = {}
        if ce_weight_t is not None:
            ce_kw["weight"] = ce_weight_t
        if self.task_type_ == "regression":
            edges_cpu, w_train = self._build_regression_rare_weights(y_tr.to(dtype=torch.float32))
            bin_idx_train = torch.bucketize(y_tr.to(dtype=torch.float32), edges_cpu[1:-1], right=False)
            bins = max(2, int(self.config.reg_rare_weight_bins))
            bw = torch.ones(bins, dtype=torch.float32)
            for bi in range(bins):
                m = bin_idx_train == bi
                if bool(m.any()):
                    bw[bi] = float(w_train[m].mean().item())
            self._reg_bin_edges = edges_cpu.detach().cpu()
            self._reg_bin_weights = bw.detach().cpu()

        for epoch in range(1, self.config.max_epochs + 1):
            self._fit_epoch = epoch
            self._tau_value = self._tau_schedule(epoch)
            self.train()
            running_loss = 0.0
            running_ce = 0.0
            running_sf1 = 0.0
            running_base_s1 = 0.0
            running_hetero = 0.0
            running_bal = 0.0
            running_spa = 0.0
            sample_count = 0
            usage_accum = torch.zeros(self.config.max_experts, device=device, dtype=torch.float32)
            epoch_targets_parts: list[np.ndarray] = []
            epoch_probs_parts: list[np.ndarray] = []
            epoch_reg_targets_parts: list[np.ndarray] = []
            epoch_reg_pred_parts: list[np.ndarray] = []
            lb_m = float(self.config.lambda_balance)
            ls_m = float(self.config.lambda_sparse)
            k_exp = int(self.config.max_experts)

            for xb, yb in train_loader:
                xb = xb.to(device=device, dtype=torch.float32)
                if self.task_type_ == "classification":
                    yb = yb.to(device=device, dtype=torch.long)
                else:
                    yb = yb.to(device=device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                raw, p_soft = self._forward_moe_combined(xb, self._tau_value)
                usage = p_soft.mean(0)
                usage_accum += usage.detach() * float(xb.shape[0])
                l_bal = lb_m * float(k_exp) * (usage**2).sum()
                l_sp = ls_m * (-(usage * (usage + 1e-8).log()).sum())

                if self.task_type_ == "classification" and self.log_temperature is not None:
                    t_det = self.get_temperature().detach()
                    scaled = raw / t_det
                    ce_loss = F.cross_entropy(scaled, yb, **ce_kw)
                    sf1_loss = self._soft_f1_loss(scaled, yb)
                    lam = float(self.config.soft_f1_lambda)
                    loss = ce_loss + lam * sf1_loss + l_bal + l_sp
                    running_ce += float(ce_loss.item()) * xb.shape[0]
                    running_sf1 += float(sf1_loss.item()) * xb.shape[0]
                    epoch_probs_parts.append(torch.softmax(scaled, dim=-1).detach().cpu().numpy())
                    epoch_targets_parts.append(yb.detach().cpu().numpy())
                else:
                    mu_raw, log_var_raw = self._split_regression_raw(raw)
                    w = self._map_regression_rare_weights(yb).to(device=device, dtype=mu_raw.dtype)
                    w = w * float(self.config.reg_rare_weight_lambda)
                    w_sum = w.sum().clamp_min(1e-8)
                    smooth = F.smooth_l1_loss(
                        mu_raw,
                        yb,
                        reduction="none",
                        beta=float(self.config.reg_smooth_l1_beta),
                    )
                    smooth_loss = (smooth * w).sum() / w_sum
                    hetero = self._gaussian_nll_from_mu_logvar(mu_raw, log_var_raw, yb)
                    hetero_loss = (hetero * w).sum() / w_sum
                    alpha = float(self.config.reg_alpha_smooth_l1) if self.config.reg_use_smooth_l1 else 0.0
                    beta = float(self.config.reg_beta_hetero_nll) if self.config.reg_use_hetero_nll else 0.0
                    loss = alpha * smooth_loss + beta * hetero_loss + l_bal + l_sp
                    running_base_s1 += float(smooth_loss.item()) * xb.shape[0]
                    running_hetero += float(hetero_loss.item()) * xb.shape[0]
                    epoch_reg_targets_parts.append(yb.detach().cpu().numpy())
                    epoch_reg_pred_parts.append(mu_raw.detach().cpu().numpy())

                running_bal += float(l_bal.detach().item()) * xb.shape[0]
                running_spa += float(l_sp.detach().item()) * xb.shape[0]
                running_loss += float(loss.item()) * xb.shape[0]
                sample_count += int(xb.shape[0])
                loss.backward()
                optimizer.step()

            usage_epoch_mean = (usage_accum / max(1, sample_count)).detach().cpu()
            self._usage_ema = 0.9 * self._usage_ema + 0.1 * usage_epoch_mean

            if int(epoch) == int(self.config.anneal_epochs) and not self._printed_annealing_footer:
                self._printed_annealing_footer = True
                act_ids = (self._usage_ema >= float(self.config.expert_usage_threshold)).nonzero(as_tuple=True)[
                    0
                ].tolist()
                print("ANNEALING COMPLETE")
                print(f"  Optimal K_active (by usage threshold) = {len(act_ids)}")
                print(f"  Active Expert IDs = {act_ids}")
                print("  Hard Routing Enabled (post-anneal, train STE / eval hard) = True")

            if self.task_type_ == "classification" and val_loader is not None:
                self._tune_temperature_on_val(val_loader, device, ce_weight_t)
            elif self.task_type_ == "regression" and val_loader is not None:
                self._tune_regression_calibration_on_val(val_loader, device)

            train_loss = running_loss / max(1, sample_count)
            train_ce_epoch = running_ce / max(1, sample_count) if self.task_type_ == "classification" else float("nan")
            train_sf1_epoch = running_sf1 / max(1, sample_count) if self.task_type_ == "classification" else float("nan")
            train_s1_epoch = running_base_s1 / max(1, sample_count) if self.task_type_ == "regression" else float("nan")
            train_het_epoch = running_hetero / max(1, sample_count) if self.task_type_ == "regression" else float("nan")
            train_bal_epoch = running_bal / max(1, sample_count)
            train_spa_epoch = running_spa / max(1, sample_count)
            current_metric = train_loss
            current_lr = optimizer.param_groups[0]["lr"]
            usage_list = [float(f"{u:.4f}") for u in self._usage_ema.tolist()]
            n_active_cur = int((self._usage_ema >= float(self.config.expert_usage_threshold)).sum().item())

            if self.task_type_ == "classification" and epoch_probs_parts:
                train_probs_epoch = np.concatenate(epoch_probs_parts, axis=0)
                train_targets_epoch = np.concatenate(epoch_targets_parts, axis=0)
                train_m = self._compute_classification_metrics(train_targets_epoch, train_probs_epoch)
            else:
                train_m = {
                    "accuracy": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1": float("nan"),
                    "roc_auc": float("nan"),
                    "ece": float("nan"),
                }
            if self.task_type_ == "regression" and epoch_reg_pred_parts:
                tr_targets_epoch = np.concatenate(epoch_reg_targets_parts, axis=0)
                tr_pred_epoch = np.concatenate(epoch_reg_pred_parts, axis=0)
                # Reflect forward() behavior by applying current regression calibration.
                tr_pred_epoch = (
                    float(self.reg_calib_a.item()) * tr_pred_epoch + float(self.reg_calib_b.item())
                )
                train_reg_m = self._compute_regression_metrics(tr_targets_epoch, tr_pred_epoch)
            else:
                train_reg_m = {"rmse": float("nan"), "spearman": float("nan"), "r2": float("nan"), "mae": float("nan")}

            if self.task_type_ == "classification":
                current_metric = train_m["roc_auc"]
                val_m = None
                val_loss_epoch = float("nan")
                val_nll_epoch = float("nan")
                val_reg_m = None
                if val_loader is not None:
                    self.eval()
                    val_targets_parts: list[np.ndarray] = []
                    val_probs_parts: list[np.ndarray] = []
                    val_ce_sum = 0.0
                    val_n = 0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb = xb.to(device=device, dtype=torch.float32)
                            yb = yb.to(device=device, dtype=torch.long)
                            val_logits = self.forward(xb)
                            val_ce_sum += F.cross_entropy(val_logits, yb, reduction="sum", **ce_kw).item()
                            val_n += int(yb.shape[0])
                            val_probs_parts.append(torch.softmax(val_logits, dim=-1).detach().cpu().numpy())
                            val_targets_parts.append(yb.detach().cpu().numpy())
                    val_probs_epoch = np.concatenate(val_probs_parts, axis=0)
                    val_targets_epoch = np.concatenate(val_targets_parts, axis=0)
                    val_m = self._compute_classification_metrics(val_targets_epoch, val_probs_epoch)
                    val_loss_epoch = float(val_ce_sum / max(1, val_n))
                    current_metric = val_m["roc_auc"]
            elif val_loader is not None:
                self.eval()
                val_running_loss = 0.0
                val_running_nll = 0.0
                val_count = 0
                val_reg_targets_parts: list[np.ndarray] = []
                val_reg_pred_parts: list[np.ndarray] = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device=device, dtype=torch.float32)
                        yb = yb.to(device=device, dtype=torch.float32)
                        raw_v = self._raw_logits(xb)
                        mu_v, lv_v = self._split_regression_raw(raw_v)
                        mu_v, lv_v = self._apply_regression_calibration(mu_v, lv_v)
                        s1_v = (
                            F.smooth_l1_loss(
                                mu_v,
                                yb,
                                reduction="mean",
                                beta=float(self.config.reg_smooth_l1_beta),
                            )
                            if self.config.reg_use_smooth_l1
                            else torch.zeros((), device=yb.device)
                        )
                        het_v = (
                            self._gaussian_nll_from_mu_logvar(mu_v, lv_v, yb).mean()
                            if self.config.reg_use_hetero_nll
                            else torch.zeros((), device=yb.device)
                        )
                        val_nll = self._gaussian_nll_from_mu_logvar(mu_v, lv_v, yb).mean()
                        val_loss = float(self.config.reg_alpha_smooth_l1) * s1_v + float(self.config.reg_beta_hetero_nll) * het_v
                        bs = xb.shape[0]
                        val_running_loss += float(val_loss.item()) * bs
                        val_running_nll += float(val_nll.item()) * bs
                        val_count += bs
                        val_reg_targets_parts.append(yb.detach().cpu().numpy())
                        val_reg_pred_parts.append(mu_v.detach().cpu().numpy())
                val_nll_epoch = val_running_nll / max(1, val_count)
                current_metric = val_nll_epoch
                val_m = None
                val_loss_epoch = current_metric
                val_loss_epoch = val_running_loss / max(1, val_count)
                if val_reg_pred_parts:
                    vr_t = np.concatenate(val_reg_targets_parts, axis=0)
                    vr_p = np.concatenate(val_reg_pred_parts, axis=0)
                    val_reg_m = self._compute_regression_metrics(vr_t, vr_p)
                else:
                    val_reg_m = {"rmse": float("nan"), "spearman": float("nan"), "r2": float("nan"), "mae": float("nan")}
            else:
                val_m = None
                val_loss_epoch = float("nan")
                val_nll_epoch = float("nan")
                val_reg_m = None

            if self.config.epoch_metrics_verbose:
                t_log = f"{self.get_temperature().item():.5f}" if self.log_temperature is not None else "n/a"
                print(f"Epoch {epoch:03d}")
                print(f"  tau = {self._tau_value:.5f} | T (calib) = {t_log}")
                if self.task_type_ == "classification":
                    print(
                        f"  Train Loss = {train_loss:.5f} | Base (CE) = {train_ce_epoch:.5f} | "
                        f"SoftF1 aux = {train_sf1_epoch:.5f} | Balance = {train_bal_epoch:.5f} | "
                        f"Sparse = {train_spa_epoch:.5f}"
                    )
                    if val_loader is not None and val_m is not None:
                        print(
                            f"  Val Metric (ROC-AUC) = {val_m['roc_auc']:.5f} | Val NLL (CE) = {val_loss_epoch:.5f} | "
                            f"Val ECE = {val_m['ece']:.5f}"
                        )
                    print(f"  Expert Usage (EMA) = {usage_list}")
                    print(f"  Active Experts (current threshold) = {n_active_cur}")
                    print(
                        f"  Train Acc/Prec/Rec/F1/ROC = {train_m['accuracy']:.4f}/{train_m['precision']:.4f}/"
                        f"{train_m['recall']:.4f}/{train_m['f1']:.4f}/{train_m['roc_auc']:.4f} | LR = {current_lr:.6f}"
                    )
                else:
                    print(
                        f"  Train Loss = {train_loss:.6f} | Base (SmoothL1) = {train_s1_epoch:.6f} | "
                        f"HeteroNLL = {train_het_epoch:.6f} | "
                        f"Balance = {train_bal_epoch:.6f} | Sparse = {train_spa_epoch:.6f}"
                    )
                    print(
                        f"  Train RMSE/Spearman/R2/MAE = {train_reg_m['rmse']:.5f}/"
                        f"{train_reg_m['spearman']:.5f}/{train_reg_m['r2']:.5f}/{train_reg_m['mae']:.5f}"
                    )
                    if val_loader is not None:
                        a = float(self.reg_calib_a.item())
                        b = float(self.reg_calib_b.item())
                        c = float(self.reg_calib_c.item())
                        print(
                            f"  Val NLL (calib) = {val_nll_epoch:.6f} | Val Loss (composite) = {val_loss_epoch:.6f} | "
                            f"Calib(a,b,c)=({a:.4f},{b:.4f},{c:.4f})"
                        )
                        if val_reg_m is not None:
                            print(
                                f"  Val RMSE/Spearman/R2/MAE = {val_reg_m['rmse']:.5f}/"
                                f"{val_reg_m['spearman']:.5f}/{val_reg_m['r2']:.5f}/{val_reg_m['mae']:.5f}"
                            )
                    print(f"  Expert Usage (EMA) = {usage_list}")
                    print(f"  Active Experts (current threshold) = {n_active_cur} | LR = {current_lr:.6f}")

            is_improvement = False
            if not np.isnan(current_metric):
                if maximize_metric:
                    is_improvement = current_metric > best_metric
                else:
                    is_improvement = current_metric < best_metric

            if is_improvement:
                best_metric = current_metric
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                best_usage_ema = self._usage_ema.clone()
                best_epoch = epoch
                patience_left = self.config.patience

                if checkpoint_dir is not None:
                    checkpoint_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    checkpoint_path = checkpoint_dir / f"best_epoch_{epoch:03d}_{checkpoint_stamp}.pth"
                    torch.save(
                        self._build_checkpoint_payload(
                            state_dict=best_state,
                            config=self.config,
                            task_type=self.task_type_,
                            num_classes=self.num_classes,
                            best_epoch=best_epoch,
                            best_metric=best_metric,
                            metric_name=metric_name,
                        ),
                        checkpoint_path,
                    )
                    self.best_checkpoint_path = str(checkpoint_path)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

            scheduler_value = train_loss if self.config.scheduler_monitor == "train_loss" else current_metric
            if not np.isnan(scheduler_value):
                prev_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(scheduler_value)
                new_lr = optimizer.param_groups[0]["lr"]
                if self.config.lr_verbose and new_lr != prev_lr:
                    print(
                        "[TabularSetSSM] ReduceLROnPlateau adjusted learning rate: "
                        f"{prev_lr:.8f} -> {new_lr:.8f} (epoch={epoch}, monitor={self.config.scheduler_monitor})"
                    )

        if best_state is not None:
            self.load_state_dict(best_state)
            if best_usage_ema is not None:
                self._usage_ema = best_usage_ema.clone()
            self.to(device)
        self._tau_value = float(self.config.tau_end)
        with torch.no_grad():
            thr = float(self.config.expert_usage_threshold)
            active = self._usage_ema >= thr
            if not bool(active.any()):
                active = torch.zeros(self.config.max_experts, dtype=torch.bool)
                active[int(self._usage_ema.argmax())] = True
            self._active_expert_mask = active.to(device)
            self._k_active = int(active.sum().item())
        self._annealing_complete = True
        self._fit_epoch = max(1, int(self.config.max_epochs))
        if self.config.epoch_metrics_verbose:
            ids_t = torch.nonzero(self._active_expert_mask.cpu(), as_tuple=False).view(-1)
            ids = ids_t.tolist()
            print(
                "[TabularSetSSM] Inference: hard routing with masked inactive experts | "
                f"K_active = {self._k_active} | active ids = {ids}"
            )
        self.eval()
        self.best_training_epoch_ = int(best_epoch)
        self.best_training_metric_ = float(best_metric)
        return self

    @torch.no_grad()
    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict class labels or regression values."""
        if self._active_expert_mask is not None and not self._printed_inference_experts_msg:
            print(f"[TabularSetSSM] Using {self._k_active} active experts only (hard routing).")
            self._printed_inference_experts_msg = True
        device = torch.device(self.config.device)
        x_t = to_float_tensor(x, device=device)
        preds = self.forward(x_t)
        if self.task_type_ == "classification":
            out = preds.argmax(dim=-1)
        else:
            out = preds
        return out.detach().cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.task_type_ != "classification":
            raise RuntimeError("predict_proba() is only available for classification.")
        if self._active_expert_mask is not None and not self._printed_inference_experts_msg:
            print(f"[TabularSetSSM] Using {self._k_active} active experts only (hard routing).")
            self._printed_inference_experts_msg = True
        device = torch.device(self.config.device)
        x_t = to_float_tensor(x, device=device)
        logits = self.forward(x_t)
        probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()
