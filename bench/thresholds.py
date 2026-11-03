"""Where should the threshold come from? Compares calibration protocols from stored scores (no retraining).

  A  held-out training good images (10% split, ~25 per category, same capture session as training);
     model: <tag> (trained on the other 90%)
  B  cross-fitted: out-of-fold scores of every good training image (~250 per category, same session);
     model: <tag>_fulltrain (trained on all good images)
  C  field calibration: a random half of the *test* good parts plays "fresh parts from the line"; the
     operating point is measured on the other half plus every defect, over 50 random splits, so no image
     is ever both calibration and evaluation; model: <tag>_fulltrain

Rules on the calibration scores: `max` (no calibration part would be rejected) and `gauss` (mean +
2.33 SD, a 1% false-alarm target under a normal fit).

  python bench/thresholds.py --tag patchcore
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import norm

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
CATS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"]
RULES = {"max": lambda c: c.max(), "gauss": lambda c: c.mean() + norm.ppf(0.99) * c.std(ddof=1)}


def op(s, y, t):
    return float((s[y == 0] > t).mean()), float((s[y == 1] <= t).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="patchcore")
    ap.add_argument("--splits", type=int, default=50)
    a = ap.parse_args()
    out = {}
    for c in CATS:
        r = {}
        fa = RES / a.tag / f"{c}.npz"
        if fa.exists():
            d = np.load(fa)
            for k, f in RULES.items():
                r[f"A_{k}"] = op(d["test_scores"], d["labels"], f(d["calib_scores"]))
        fb, ff = RES / f"{a.tag}_oof" / f"{c}.npz", RES / f"{a.tag}_fulltrain" / f"{c}.npz"
        if ff.exists():
            d = np.load(ff)
            s, y = d["test_scores"], d["labels"]
            if fb.exists():
                oof = np.load(fb)["oof_scores"]
                for k, f in RULES.items():
                    r[f"B_{k}"] = op(s, y, f(oof))
            rng = np.random.default_rng(0)
            good = np.flatnonzero(y == 0)
            for k, f in RULES.items():
                ops = []
                for _ in range(a.splits):
                    cal = rng.choice(good, len(good) // 2, replace=False)
                    keep = np.setdiff1d(np.arange(len(y)), cal)
                    ops.append(op(s[keep], y[keep], f(s[cal])))
                r[f"C_{k}"] = tuple(np.mean(ops, 0).tolist())
        if r:
            out[c] = r
    keys = sorted({k for r in out.values() for k in r})
    print("| category | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for c, r in out.items():
        print(f"| {c} | " + " | ".join(f"{100 * r[k][0]:.0f} / {100 * r[k][1]:.0f}" if k in r else "" for k in keys) + " |")
    full = [c for c in out if all(k in out[c] for k in keys)]
    print(f"| **mean ({len(full)} cat.)** | " + " | ".join(
        f"**{100 * np.mean([out[c][k][0] for c in full]):.1f} / {100 * np.mean([out[c][k][1] for c in full]):.1f}**"
        for k in keys) + " |")
    print("\ncells: false-alarm rate / miss rate, in %")
    (RES / f"{a.tag}_thresholds.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
