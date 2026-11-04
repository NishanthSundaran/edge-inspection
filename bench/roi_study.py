"""Score with and without the inspection region (edgeinspect/roi.py), on the full-train PatchCore models.

For each category that gets a ROI: image AUROC with the max over the whole map vs the max inside the ROI,
pixel AUROC / AUPRO with the map zeroed outside the ROI, and the share of ground-truth defect pixels the
ROI drops (must be ~0). Scores go to results/<tag>_roi/<category>.npz for bench/thresholds.py.

  python bench/roi_study.py --tag patchcore_fulltrain --device cpu
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from edgeinspect import data, metrics, roi  # noqa: E402
from edgeinspect.predict import TorchPredictor, resize  # noqa: E402

RUNS = pathlib.Path.home() / "ml_data/inspect/runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="patchcore_fulltrain")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--categories", nargs="+", default=["all"])
    a = ap.parse_args()
    torch.set_num_threads(6)
    cats = data.CATEGORIES if a.categories == ["all"] else a.categories
    out = REPO / "results" / f"{a.tag}_roi"
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for c in cats:
        ck = torch.load(RUNS / a.tag / c / "core.pt", weights_only=False)
        size = ck["size"]
        train = sorted((data.DATA / c / "train" / "good").glob("*.png"))
        tmpl = roi.template([resize(data.load_rgb(p), size, ck["bicubic"]).numpy() for p in train])
        if not roi.uses_roi(c, tmpl):
            print(c, "full frame", flush=True)
            continue
        pred = TorchPredictor(ck["core"], a.device)
        items = data.test_items(c)
        y = np.array([l for _, l, _ in items])
        s_full, s_roi, maps, maps_roi, masks, lost, tot = [], [], [], [], [], 0, 0
        for p, _, mp in items:
            x = resize(data.load_rgb(p), size, ck["bicubic"]).numpy()
            _, m = pred(x[None])
            m = m.reshape(size, size)
            r = roi.roi(x, tmpl)
            gt = data.load_mask(mp, (size, size))
            lost += int((gt.astype(bool) & ~r).sum())
            tot += int(gt.astype(bool).sum())
            s_full.append(m.max())
            s_roi.append(m[r].max())
            maps.append(m)
            maps_roi.append(np.where(r, m, m.min()))
            masks.append(gt)
        s_full, s_roi, maps, maps_roi, masks = map(np.array, (s_full, s_roi, maps, maps_roi, masks))
        rows[c] = {"roi_area": float(tmpl.mean()), "defect_px_outside_roi": lost / max(tot, 1)}
        for k, s, mm in (("full", s_full, maps), ("roi", s_roi, maps_roi)):
            rows[c][k] = {"image_auroc": metrics.image_auroc(s, y), "pixel_auroc": metrics.pixel_auroc(mm, masks),
                          "aupro": metrics.aupro(mm, masks)}
        np.savez_compressed(out / f"{c}.npz", test_scores=s_roi, labels=y, full_scores=s_full)
        print(c, json.dumps(rows[c]), flush=True)
    (REPO / "results" / f"{a.tag}_roi.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
