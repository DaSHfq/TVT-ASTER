# -*- coding: utf-8 -*-
from typing import Dict, Tuple
import numpy as np
import torch

@torch.no_grad()
def compute_binary_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = preds.to(torch.int64); targets = targets.to(torch.int64)
    N = targets.numel(); eps = 1e-12
    TP = ((preds == 1) & (targets == 1)).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    FN = ((preds == 0) & (targets == 1)).sum().item()
    TN = ((preds == 0) & (targets == 0)).sum().item()
    acc = (TP + TN) / max(N, 1)
    prec = TP / max(TP + FP, 1) if (TP + FP) > 0 else 0.0
    rec  = TP / max(TP + FN, 1) if (TP + FN) > 0 else 0.0
    f1 = 2 * prec * rec / max(prec + rec, eps) if (prec + rec) > 0 else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": TP, "fp": FP, "fn": FN, "tn": TN, "n": N}

def metrics_from_probs(probs: np.ndarray, targets: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64); N = targets.size
    TP = int(((preds == 1) & (targets == 1)).sum())
    FP = int(((preds == 1) & (targets == 0)).sum())
    FN = int(((preds == 0) & (targets == 1)).sum())
    TN = int(((preds == 0) & (targets == 0)).sum())
    acc = (TP + TN) / max(N, 1)
    prec = TP / max(TP + FP, 1) if (TP + FP) > 0 else 0.0
    rec  = TP / max(TP + FN, 1) if (TP + FN) > 0 else 0.0
    f1 = 2*prec*rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": TP, "fp": FP, "fn": FN, "tn": TN, "n": N}

def find_best_threshold(probs: np.ndarray, targets: np.ndarray, t_min=0.30, t_max=0.70, num=41) -> Tuple[float, float]:
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(t_min, t_max, num):
        m = metrics_from_probs(probs, targets, float(t))
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t, best_f1
