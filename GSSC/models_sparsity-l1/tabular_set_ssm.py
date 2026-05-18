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
    # Holdout fraction in (0, 1] when splitting inside fit(); None or 0 disables val, T tuning, and
    # ignores x_val/y_val arguments to fit() (use a notebook holdout only for post-fit steps if needed).
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
    # Sparsity: lambda_sparse * mean(|router_logits|) on pre-softmax outputs (batch- and K-invariant scale)
    lambda_sparse: float = 0.001
    expert_usage_threshold: float = 0.01
    hard_routing_after_anneal: bool = True
    router_hidden_dim: int = 64

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
                )
                for _ in range(config.max_experts)
            ]
        )
        self.router = TreeRouter(
            num_experts=config.max_experts,
            hidden_dim=config.router_hidden_dim,
        )

        self.head: Optional[nn.Linear] = None
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
        return 1

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

    def _gather_expert_outputs_hard(
        self, x: torch.Tensor, z: torch.Tensor, idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Grouped dispatch: for each expert k, run one forward on all batch rows assigned
        to k, then scatter outputs back to original row order.
        """
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        b = int(idx.size(0))
        od = self._output_dim_predict()
        out = torch.zeros(b, od, device=device, dtype=dtype)
        row = torch.arange(b, device=device)
        for kid_t in torch.unique(idx, sorted=True):
            kid = int(kid_t.item())
            sel = row[idx == kid_t]
            if sel.numel() == 0:
                continue
            x_grp = x.index_select(0, sel)
            y_grp = self.experts[kid](x_grp, z)
            out.index_copy_(0, sel, y_grp.to(dtype=out.dtype))
        return out

    def _forward_moe_combined(
        self, x: torch.Tensor, tau: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_expert_heads_ready(x)
        x_work = x.unsqueeze(0) if x.dim() == 2 else x
        self._ensure_graph_cache(x_work)
        eigvals, eigvecs = self.feature_graph.get_eigenpairs()
        z = self.pe_encoder(eigvals.to(x.device), eigvecs.to(x.device))

        router_logits = self.router(x)
        router_logits = self._apply_active_expert_mask_to_logits(router_logits)
        p_soft = self._router_routing_probs(router_logits, tau)

        eval_hard = self._eval_use_hard_gather()
        if eval_hard:
            s_argmax = self._router_scaled_stable(router_logits, tau)
            idx = s_argmax.argmax(dim=-1)
            logits = self._gather_expert_outputs_hard(x, z, idx)
            return logits, p_soft, router_logits

        expert_outputs = torch.stack([e(x, z) for e in self.experts], dim=1)
        if self._training_uses_ste():
            p_route = torch.zeros_like(p_soft).scatter_(1, p_soft.argmax(-1, keepdim=True), 1.0)
            p_route = p_route - p_soft.detach() + p_soft
        else:
            p_route = p_soft
        logits = (expert_outputs * p_route.unsqueeze(-1)).sum(dim=1)
        return logits, p_soft, router_logits

    def _router_tau(self) -> float:
        return float(self._tau_value)

    def get_temperature(self) -> torch.Tensor:
        """Scalar T>0 (init 1.0). Only for classification; otherwise 1.0."""
        if self.log_temperature is None:
            z = next(self.parameters())
            return torch.ones((), device=z.device, dtype=z.dtype)
        return torch.exp(self.log_temperature).squeeze().clamp(min=1e-6)

    def _raw_logits(self, x: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self._forward_moe_combined(x, self._router_tau())
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._raw_logits(x)
        if self.task_type_ != "classification" or self.log_temperature is None:
            return logits
        return logits / self.get_temperature()

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
        if split_nontrivial and external_val_given:
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
            split_ratio = float(self.config.val_ratio)
            used_external_val = True
        elif split_nontrivial:
            split_ratio = float(self.config.val_ratio)
            strat = bool(self.config.val_stratify) and self.task_type_ == "classification"
            x_tr, y_tr, x_val, y_val = train_val_split(
                x_t, y_t, val_ratio=split_ratio, seed=self.config.seed, stratify=strat
            )
        else:
            if external_val_given and (self.config.lr_verbose or self.config.epoch_metrics_verbose):
                print(
                    "[TabularSetSSM] val_ratio is 0 (or None): ignoring x_val/y_val inside fit() — "
                    "no validation metrics or per-epoch T tuning (T stays 1.0)."
                )
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
            else nn.MSELoss()
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
            metric_name = "val_loss"
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
        ce_kw: dict = {}
        if ce_weight_t is not None:
            ce_kw["weight"] = ce_weight_t

        for epoch in range(1, self.config.max_epochs + 1):
            self._fit_epoch = epoch
            self._tau_value = self._tau_schedule(epoch)
            self.train()
            running_loss = 0.0
            running_ce = 0.0
            running_sf1 = 0.0
            running_base_mse = 0.0
            running_bal = 0.0
            running_spa = 0.0
            sample_count = 0
            usage_accum = torch.zeros(self.config.max_experts, device=device, dtype=torch.float32)
            epoch_targets_parts: list[np.ndarray] = []
            epoch_probs_parts: list[np.ndarray] = []
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
                raw, p_soft, router_logits = self._forward_moe_combined(xb, self._tau_value)
                usage = p_soft.mean(0)
                usage_accum += usage.detach() * float(xb.shape[0])
                l_bal = lb_m * float(k_exp) * (usage**2).sum()
                # L1 on pre-softmax router logits (standard MoE sparsity); mean over batch×experts for stable scale.
                l_sp = ls_m * router_logits.abs().mean()

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
                    base_loss = F.mse_loss(raw.squeeze(-1), yb)
                    loss = base_loss + l_bal + l_sp
                    running_base_mse += float(base_loss.item()) * xb.shape[0]

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

            train_loss = running_loss / max(1, sample_count)
            train_ce_epoch = running_ce / max(1, sample_count) if self.task_type_ == "classification" else float("nan")
            train_sf1_epoch = running_sf1 / max(1, sample_count) if self.task_type_ == "classification" else float("nan")
            train_mse_epoch = running_base_mse / max(1, sample_count) if self.task_type_ == "regression" else float("nan")
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

            if self.task_type_ == "classification":
                current_metric = train_m["roc_auc"]
                val_m = None
                val_loss_epoch = float("nan")
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
                val_count = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device=device, dtype=torch.float32)
                        yb = yb.to(device=device, dtype=torch.float32)
                        val_preds = self.forward(xb)
                        val_loss = F.mse_loss(val_preds.squeeze(-1), yb)
                        bs = xb.shape[0]
                        val_running_loss += val_loss.item() * bs
                        val_count += bs
                current_metric = val_running_loss / max(1, val_count)
                val_m = None
                val_loss_epoch = current_metric
            else:
                val_m = None
                val_loss_epoch = float("nan")

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
                        f"  Train Loss = {train_loss:.6f} | Base (MSE) = {train_mse_epoch:.6f} | "
                        f"Balance = {train_bal_epoch:.6f} | Sparse = {train_spa_epoch:.6f}"
                    )
                    if val_loader is not None:
                        print(f"  Val MSE = {val_loss_epoch:.6f}")
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
            out = preds.squeeze(-1)
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
