"""Classification metrics computed on every test fold and on the hold-out set."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# name -> (pretty label, higher_is_better)
METRICS: dict[str, tuple[str, bool]] = {
    "f1_macro": ("Macro F1", True),
    "accuracy": ("Accuracy", True),
    "precision_macro": ("Macro Precision", True),
    "recall_macro": ("Macro Recall", True),
    "f1_weighted": ("Weighted F1", True),
    "precision_weighted": ("Weighted Precision", True),
    "recall_weighted": ("Weighted Recall", True),
    "hamming_loss": ("Hamming Loss", False),
    "mcc": ("MCC", True),
    "cohen_kappa": ("Cohen's κ", True),
    "log_loss": ("Log Loss", False),
}


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> dict[str, float]:
    kw = dict(labels=labels, zero_division=0)
    out = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", **kw),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", **kw),
        "recall_macro": recall_score(y_true, y_pred, average="macro", **kw),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", **kw),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", **kw),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", **kw),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_proba is not None:
        out["log_loss"] = log_loss(y_true, y_proba, labels=labels)
    return {k: float(v) for k, v in out.items()}


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Vectorised macro-F1 (identical to sklearn's) used inside the bootstrap loop."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes**2).reshape(
        n_classes, n_classes
    )
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0)
    true_pos = cm.sum(axis=1)
    denom = pred_pos + true_pos
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())
