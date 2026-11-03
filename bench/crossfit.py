"""Cross-fitted threshold calibration.

A deployment threshold needs good-part scores from parts the model never trained on. Holding out 10% of
the good training images gives only 21 to 39 scores per category, too few to place a threshold in the
1% tail (a single unusual good part, e.g. a capsule turned so its print is hidden, sets the maximum).

Here every good training image gets an out-of-fold score: K models, each trained on (K-1)/K of the good
images, score the fold they did not see. The deployed model is trained on all good images (the
published setting); its threshold comes from the out-of-fold scores of all of them.

  python bench/crossfit.py --model patchcore --folds 5
Writes results/<model>_oof/<category>.npz with `oof_scores`; the full-train model is bench/train.py
--full-train (tag <model>_fulltrain).
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench.train import build  # noqa: E402
from edgeinspect import data, patches  # noqa: E402
from edgeinspect.evaluate import _run  # noqa: E402
from edgeinspect.predict import Core, TorchPredictor  # noqa: E402

RUNS = pathlib.Path.home() / "ml_data/inspect/runs"


def fold_root(category: str, held: set[str], k: int) -> pathlib.Path:
    root = data.DATA.parent / f"MVTecAD_fold{k}"
    d = root / category / "train" / "good"
    if d.exists():
        for p in d.iterdir():
            p.unlink()
    d.mkdir(parents=True, exist_ok=True)
    for p in (data.DATA / category / "train" / "good").glob("*.png"):
        if p.name not in held:
            (d / p.name).symlink_to(p)
    # the datamodule expects test and masks; validation is off, so they are never read. Per-file links:
    # Path.glob("**") in Python 3.10 does not descend into symlinked directories.
    for sub in ("test", "ground_truth"):
        for src in (data.DATA / category / sub).rglob("*.png"):
            dst = root / src.relative_to(data.DATA)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src)
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="patchcore")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--categories", nargs="+", default=["all"])
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--size", default=None)
    ap.add_argument("--k-shot", type=int, default=0)
    args = ap.parse_args()

    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    cats = data.CATEGORIES if args.categories == ["all"] else args.categories
    out = REPO / "results" / f"{args.model}_oof"
    out.mkdir(parents=True, exist_ok=True)
    for c in cats:
        if (out / f"{c}.npz").exists():
            print("skip", c)
            continue
        names = sorted(p.name for p in (data.DATA / c / "train" / "good").glob("*.png"))
        rng = random.Random(f"oof-{c}")
        order = names[:]
        rng.shuffle(order)
        folds = [sorted(order[k::args.folds]) for k in range(args.folds)]
        oof, oof_names, t0 = [], [], time.time()
        for k, held in enumerate(folds):
            torch.manual_seed(k)
            model = build(args.model, c, args)
            dm = MVTecAD(root=fold_root(c, set(held), k), category=c, train_batch_size=32, eval_batch_size=32,
                         num_workers=8)
            Engine(accelerator="gpu", devices=1, logger=False, limit_val_batches=0, num_sanity_val_steps=0,
                   default_root_dir=RUNS / f"{args.model}_oof" / c, max_epochs=1).fit(model=model, datamodule=dm)
            core, size, bicubic = Core.from_anomalib(model)
            s, _ = _run(TorchPredictor(core, "cuda"), [data.DATA / c / "train" / "good" / n for n in held], size, bicubic)
            oof.append(s)
            oof_names += held
            del model, core
            torch.cuda.empty_cache()
        np.savez_compressed(out / f"{c}.npz", oof_scores=np.concatenate(oof), names=np.array(oof_names))
        print(f"{c}: {len(oof_names)} out-of-fold scores, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    patches.patchcore_cpu_bank()
    main()
