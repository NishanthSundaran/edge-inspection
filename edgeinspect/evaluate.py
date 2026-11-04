"""Score one predictor on one category: test set metrics, plus the calibrated deployment operating point."""

from __future__ import annotations

import numpy as np
import torch

from . import data, metrics
from .predict import resize


def _run(pred, paths, size: int, bicubic: bool = False, batch: int = 16):
    scores, maps = [], []
    batch = getattr(pred, "eval_batch", batch)  # CPU runtimes: the batch x 1024 x bank distance matrix lives in RAM
    for i in range(0, len(paths), batch):
        if getattr(pred, "takes_paths", False):  # the deployed path does its own (PIL) preprocessing
            s, m = pred.from_paths(paths[i:i + batch])
            scores.append(s)
            maps.append(m.astype(np.float32))
            continue
        x = torch.stack([resize(data.load_rgb(p), size, bicubic) for p in paths[i:i + batch]]).numpy()
        s, m = _call_split(pred, x)
        scores.append(s)
        maps.append(m.astype(np.float32))
    return np.concatenate(scores), np.concatenate(maps)


def _call_split(pred, x):
    """Run a batch; on a CUDA out-of-memory error (large full-train memory banks: the distance matrix is
    batch x 1024 x bank), split the batch in half and retry. Results are per image, so they are unchanged."""
    try:
        return pred(x)
    except Exception as e:  # torch.OutOfMemoryError, or ONNX Runtime's "Failed to allocate memory"
        if len(x) == 1 or not (isinstance(e, torch.OutOfMemoryError) or "allocate" in str(e)):
            raise
        torch.cuda.empty_cache()
        h = len(x) // 2
        (s1, m1), (s2, m2) = _call_split(pred, x[:h]), _call_split(pred, x[h:])
        return np.concatenate([s1, s2]), np.concatenate([m1, m2])


def evaluate(pred, category: str, size: int, bicubic: bool = False, keep_maps: bool = False) -> dict:
    items = data.test_items(category)
    paths = [p for p, _, _ in items]
    labels = np.array([l for _, l, _ in items])
    masks = np.stack([data.load_mask(m, (size, size)) for _, _, m in items])
    s_test, m_test = _run(pred, paths, size, bicubic)
    s_cal, _ = _run(pred, data.calib_images(category), size, bicubic)

    thr_cal = float(s_cal.max())                    # reject nothing a known-good part scored
    thr_f1 = metrics.best_f1_threshold(s_test, labels)
    out = {
        "category": category,
        "n_test": len(items), "n_defect": int(labels.sum()), "n_calib": len(s_cal),
        "image_auroc": metrics.image_auroc(s_test, labels),
        "pixel_auroc": metrics.pixel_auroc(m_test, masks),
        "aupro": metrics.aupro(m_test, masks),
        "calibrated": metrics.operating_point(s_test, labels, thr_cal),
        "oracle_f1": metrics.operating_point(s_test, labels, thr_f1),
    }
    arrays = {"test_scores": s_test, "labels": labels, "calib_scores": s_cal}
    if keep_maps:
        arrays["test_maps"] = m_test.astype(np.float16)
    return out, arrays
