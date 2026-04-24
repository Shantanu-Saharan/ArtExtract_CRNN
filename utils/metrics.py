from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(y_true, y_pred, logits: Optional[torch.Tensor] = None, 
                                  return_per_class: bool = False) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        logits: Optional logits for top-k accuracy
        return_per_class: If True, include per-class F1 scores
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    # Add per-class F1 scores if requested
    if return_per_class:
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        metrics["per_class_f1"] = per_class_f1.tolist()
    
    # Add top-k metrics if logits are provided
    if logits is not None:
        labels_tensor = torch.tensor(y_true)
        metrics["top1_accuracy"] = top_k_accuracy(logits, labels_tensor, k=1)
        metrics["top5_accuracy"] = top_k_accuracy(logits, labels_tensor, k=5)
    
    return metrics


def top_k_accuracy(logits, labels, k: int = 5) -> float:
    """Compute top-k accuracy. k is capped at num_classes. Safe against size mismatches."""
    if logits.size(0) != labels.size(0):
        # Size mismatch can happen if logit gathering was uneven across ranks; skip gracefully.
        return 0.0
    num_classes = logits.size(1)
    k = min(k, num_classes)
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).sum().item()
    return correct / max(labels.size(0), 1)


def numpy_mean(values):
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))
