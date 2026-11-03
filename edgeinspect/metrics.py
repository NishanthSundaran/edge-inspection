"""Anomaly detection metrics, computed from raw scores so every runtime is scored by the same code.

image AUROC  : per-image score vs label.
pixel AUROC  : every test pixel, anomaly map vs ground-truth mask.
AUPRO        : MVTec's per-region overlap, integrated up to 30% pixel FPR and normalised by 0.3.
               Computed exactly by sorting pixels (the trick in MVTec's official evaluation code):
               each defect pixel carries weight 1 / (region size x number of regions), each normal pixel
               1 / (number of normal pixels), so cumulative sums give the PRO and FPR curves.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    return float(roc_auc_score(masks.reshape(-1).astype(bool), maps.reshape(-1)))


def aupro(maps: np.ndarray, masks: np.ndarray, fpr_limit: float = 0.3) -> float:
    """maps, masks: (N, H, W). Regions are 8-connected components of each mask."""
    maps = maps.astype(np.float64)
    weights = np.zeros(maps.shape, np.float64)
    is_defect = np.zeros(maps.shape, bool)
    regions = []
    for i, m in enumerate(masks):
        lab, n = ndimage.label(m > 0, structure=np.ones((3, 3)))
        for r in range(1, n + 1):
            regions.append((i, lab == r))
    if not regions:
        return float("nan")
    for i, reg in regions:
        weights[i][reg] = 1.0 / (reg.sum() * len(regions))
        is_defect[i] |= reg
    normal = ~is_defect
    weights[normal] = 1.0 / normal.sum()

    s, w, d = maps.reshape(-1), weights.reshape(-1), is_defect.reshape(-1)
    order = np.argsort(-s, kind="stable")
    s, w, d = s[order], w[order], d[order]
    pro = np.cumsum(np.where(d, w, 0.0))
    fpr = np.cumsum(np.where(d, 0.0, w))
    # keep the last point of each run of equal scores (one threshold per distinct value)
    last = np.r_[s[1:] != s[:-1], True]
    pro, fpr = np.r_[0.0, pro[last]], np.r_[0.0, fpr[last]]
    # integrate up to fpr_limit, interpolating the crossing point
    k = np.searchsorted(fpr, fpr_limit, side="right")
    if k < len(fpr):
        p_lim = np.interp(fpr_limit, fpr[k - 1:k + 1], pro[k - 1:k + 1])
        fpr, pro = np.r_[fpr[:k], fpr_limit], np.r_[pro[:k], p_lim]
    return float(np.trapezoid(pro, fpr) / fpr_limit)


def operating_point(test_scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = test_scores > threshold
    good, bad = labels == 0, labels == 1
    return {
        "threshold": float(threshold),
        "false_alarm_rate": float(pred[good].mean()),   # good parts rejected
        "miss_rate": float((~pred[bad]).mean()),         # defective parts shipped
    }


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Oracle threshold picked on the test set itself (what anomalib's default val split does)."""
    best, thr = -1.0, 0.0
    for t in np.unique(scores):
        pred = scores >= t
        tp = (pred & (labels == 1)).sum()
        f1 = 2 * tp / max(1, pred.sum() + (labels == 1).sum())
        if f1 > best:
            best, thr = f1, t
    return float(np.nextafter(thr, -np.inf))  # so that scores >= t are flagged under `>`
