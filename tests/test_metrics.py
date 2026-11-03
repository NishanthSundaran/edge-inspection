import numpy as np
import pytest
from edgeinspect import metrics


def _synthetic(seed=0, n=8, s=64):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, s, s), np.uint8)
    maps = rng.random((n, s, s)).astype(np.float32)
    for i in range(n // 2):  # half the images get one or two square defects that score higher
        for _ in range(rng.integers(1, 3)):
            y, x, h = rng.integers(0, s - 16), rng.integers(0, s - 16), rng.integers(4, 16)
            masks[i, y:y + h, x:x + h] = 1
            maps[i, y:y + h, x:x + h] += rng.random() * 1.5
    return maps, masks


def test_aupro_perfect_and_random():
    maps, masks = _synthetic()
    assert metrics.aupro(masks.astype(np.float32) + 1e-3 * maps, masks) == pytest.approx(1.0, abs=1e-6)
    rng = np.random.default_rng(1)
    r = metrics.aupro(rng.random(masks.shape), masks)
    assert 0.1 < r < 0.2  # random scores: PRO ~ FPR, so the normalised area up to 0.3 is ~0.15


def test_aupro_matches_anomalib():
    pytest.importorskip("anomalib")
    from anomalib.metrics.aupro import _AUPRO
    maps, masks = _synthetic(seed=3)
    ref = _AUPRO(fpr_limit=0.3)
    import torch
    ref.update(torch.from_numpy(maps), torch.from_numpy(masks).int())
    assert metrics.aupro(maps, masks) == pytest.approx(float(ref.compute()), abs=2e-3)


def test_operating_point():
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.25])
    y = np.array([0, 0, 0, 1, 1, 1])
    op = metrics.operating_point(s, y, threshold=0.3)
    assert op["false_alarm_rate"] == 0.0 and op["miss_rate"] == pytest.approx(1 / 3)
    t = metrics.best_f1_threshold(s, y)
    assert metrics.operating_point(s, y, t)["miss_rate"] == 0.0
