"""Tables for the README from results/<tag>/*.json and results/<tag>_runtimes/*.json.

  python bench/summarize.py                 # model comparison + per-category for each tag
  python bench/summarize.py --runtimes patchcore
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
CATS = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"]


def load(tag: str) -> dict[str, dict]:
    return {c: json.loads((RES / tag / f"{c}.json").read_text()) for c in CATS if (RES / tag / f"{c}.json").exists()}


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def per_category(tag: str) -> None:
    rs = load(tag)
    print(f"\n### {tag} ({len(rs)}/15 categories)\n")
    print("| category | image AUROC | pixel AUROC | AUPRO | calibrated: miss | calibrated: false alarm | oracle F1: miss | oracle F1: false alarm |")
    print("|---|---|---|---|---|---|---|---|")
    for c, r in rs.items():
        print(f"| {c} | {pct(r['image_auroc'])} | {pct(r['pixel_auroc'])} | {pct(r['aupro'])} | "
              f"{pct(r['calibrated']['miss_rate'])} | {pct(r['calibrated']['false_alarm_rate'])} | "
              f"{pct(r['oracle_f1']['miss_rate'])} | {pct(r['oracle_f1']['false_alarm_rate'])} |")
    if rs:
        m = lambda f: np.mean([f(r) for r in rs.values()])
        print(f"| **mean** | **{pct(m(lambda r: r['image_auroc']))}** | **{pct(m(lambda r: r['pixel_auroc']))}** | "
              f"**{pct(m(lambda r: r['aupro']))}** | {pct(m(lambda r: r['calibrated']['miss_rate']))} | "
              f"{pct(m(lambda r: r['calibrated']['false_alarm_rate']))} | {pct(m(lambda r: r['oracle_f1']['miss_rate']))} | "
              f"{pct(m(lambda r: r['oracle_f1']['false_alarm_rate']))} |")


def models(tags: list[str]) -> None:
    print("\n| model | categories | image AUROC | pixel AUROC | AUPRO | calibrated miss | calibrated false alarm | GPU ms | fit s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in tags:
        rs = load(t)
        if not rs:
            continue
        m = lambda f: np.mean([f(r) for r in rs.values()])
        print(f"| {t} | {len(rs)} | {pct(m(lambda r: r['image_auroc']))} | {pct(m(lambda r: r['pixel_auroc']))} | "
              f"{pct(m(lambda r: r['aupro']))} | {pct(m(lambda r: r['calibrated']['miss_rate']))} | "
              f"{pct(m(lambda r: r['calibrated']['false_alarm_rate']))} | {m(lambda r: r['latency']['median_ms']):.1f} | "
              f"{m(lambda r: r['fit_s']):.0f} |")


def runtimes(tag: str) -> None:
    d = RES / f"{tag}_runtimes"
    rs = {c: json.loads((d / f"{c}.json").read_text()) for c in CATS if (d / f"{c}.json").exists()}
    names = [n for n in next(iter(rs.values())) if not n.startswith("_")] if rs else []
    ref = "pytorch-cuda-fp32"
    print(f"\n### {tag} runtimes ({len(rs)}/15 categories)\n")
    print("| runtime | image AUROC | pixel AUROC | AUPRO | max abs. AUPRO change | median ms | p95 ms | errors |")
    print("|---|---|---|---|---|---|---|---|")
    for n in names:
        ok = [r[n] for r in rs.values() if n in r and "error" not in r[n]]
        err = sum(1 for r in rs.values() if n in r and "error" in r[n])
        if not ok:
            print(f"| {n} | | | | | | | {err} |")
            continue
        dpro = max(abs(r[n]["aupro"] - r[ref]["aupro"]) for r in rs.values() if n in r and "error" not in r[n] and ref in r)
        m = lambda k: np.mean([o[k] for o in ok])
        print(f"| {n} | {pct(m('image_auroc'))} | {pct(m('pixel_auroc'))} | {pct(m('aupro'))} | {pct(dpro)} | "
              f"{np.mean([o['latency']['median_ms'] for o in ok]):.1f} | {np.mean([o['latency']['p95_ms'] for o in ok]):.1f} | {err} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtimes", nargs="*", default=None)
    a = ap.parse_args()
    if a.runtimes:
        for t in a.runtimes:
            runtimes(t)
        return
    tags = sorted(p.name for p in RES.iterdir() if p.is_dir() and not p.name.endswith("_runtimes"))
    models(tags)
    for t in tags:
        per_category(t)


if __name__ == "__main__":
    main()
