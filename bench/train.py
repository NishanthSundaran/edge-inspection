"""Fit an anomalib model per MVTec category on the training images minus the calibration split, save the
bare `Core` graph, and score it in PyTorch on the GPU.

  python bench/train.py --model patchcore --categories all
  python bench/train.py --model patchcore --categories bottle --full-train   # reproduction check
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from edgeinspect import data, patches  # noqa: E402
from edgeinspect.evaluate import evaluate  # noqa: E402
from edgeinspect.predict import Core, TorchPredictor, latency_ms  # noqa: E402

RUNS = pathlib.Path.home() / "ml_data/inspect/runs"
MODELS = ["patchcore", "padim", "efficientad", "winclip"]


def build(name: str, category: str, args):
    from anomalib import models
    if name == "patchcore":
        patches.patchcore_cpu_bank()
        return models.Patchcore(backbone=args.backbone or "wide_resnet50_2", coreset_sampling_ratio=args.coreset,
                                evaluator=False, visualizer=False, post_processor=False)
    if name == "padim":
        return models.Padim(backbone=args.backbone or "resnet18", evaluator=False, visualizer=False,
                            post_processor=False)
    if name == "efficientad":
        return models.EfficientAd(imagenet_dir=pathlib.Path.home() / "ml_data/inspect/data/imagenette",
                                  model_size=args.size or "small", evaluator=False, visualizer=False,
                                  post_processor=False)
    if name == "winclip":
        return models.WinClip(class_name=category.replace("_", " "), k_shot=args.k_shot,
                              evaluator=False, visualizer=False, post_processor=False)
    raise ValueError(name)


@torch.no_grad()
def efficientad_quantiles_from_calib(model, category: str) -> None:
    """EfficientAD normalises its two maps with 90% / 99.5% quantiles of good-image maps. anomalib computes
    them in on_validation_start from the validation split, which is the test set by default; here they
    come from the calibration split (held-out good training images), as the paper intends."""
    from edgeinspect.predict import resize
    m = model.model.eval().cuda()
    x = torch.stack([resize(data.load_rgb(p), 256) for p in data.calib_images(category)]).cuda()
    st, ae = zip(*(m.get_maps(x[i:i + 8], normalize=False) for i in range(0, len(x), 8)))
    q = {}
    for k, maps in (("st", st), ("ae", ae)):
        flat = torch.cat([t.flatten() for t in maps])
        q[f"qa_{k}"], q[f"qb_{k}"] = torch.quantile(flat, 0.9), torch.quantile(flat, 0.995)
    m.quantiles.update(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--categories", nargs="+", default=["all"])
    ap.add_argument("--tag", default=None, help="run name, default = model")
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--size", default=None, help="EfficientAD model size")
    ap.add_argument("--k-shot", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--full-train", action="store_true", help="train on all good images (published setting)")
    args = ap.parse_args()

    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    cats = data.CATEGORIES if args.categories == ["all"] else args.categories
    tag = args.tag or args.model + ("_fulltrain" if args.full_train else "")
    root = data.DATA if args.full_train else data.build_fit_root()
    out_dir = REPO / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in cats:
        if (out_dir / f"{c}.json").exists():
            print("skip", c)
            continue
        torch.manual_seed(0)
        np.random.seed(0)
        model = build(args.model, c, args)
        t0 = time.time()
        if args.model != "winclip":
            bs = 1 if args.model == "efficientad" else 32
            dm = MVTecAD(root=root, category=c, train_batch_size=bs, eval_batch_size=32, num_workers=8,
                         )  # validation is disabled below, so the default val split (the test set) is never read
            kw = {"max_steps": args.max_steps} if args.max_steps > 0 else {}
            engine = Engine(accelerator="gpu", devices=1, logger=False, limit_val_batches=0, num_sanity_val_steps=0, default_root_dir=RUNS / tag / c,
                            **({"max_epochs": 1} if args.model in ("patchcore", "padim") else {}), **kw)
            engine.fit(model=model, datamodule=dm)
        else:
            model.model.setup(model.class_name, None)
        if args.model == "efficientad":
            efficientad_quantiles_from_calib(model, c)
        fit_s = time.time() - t0

        core, size, bicubic = Core.from_anomalib(model)
        ck = RUNS / tag / c / "core.pt"
        ck.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"core": core.cpu(), "size": size, "bicubic": bicubic}, ck)
        pred = TorchPredictor(core, "cuda")
        res, arrays = evaluate(pred, c, size, bicubic)
        x1 = np.random.rand(1, 3, size, size).astype(np.float32)
        res.update(model=args.model, size=size, bicubic=bicubic, tag=tag, fit_s=round(fit_s, 1), runtime="pytorch-cuda-fp32",
                   latency=latency_ms(pred, x1), params=sum(p.numel() for p in core.parameters()),
                   buffers_mb=round(sum(b.numel() * b.element_size() for b in core.buffers()) / 1e6, 1),
                   ckpt_mb=round(ck.stat().st_size / 1e6, 1))
        np.savez_compressed(out_dir / f"{c}.npz", **arrays)
        (out_dir / f"{c}.json").write_text(json.dumps(res, indent=1))
        print(f"{tag} {c}: img AUROC {res['image_auroc']:.4f}  pix AUROC {res['pixel_auroc']:.4f}  "
              f"AUPRO {res['aupro']:.4f}  calibrated miss {res['calibrated']['miss_rate']:.3f} "
              f"FAR {res['calibrated']['false_alarm_rate']:.3f}  fit {fit_s:.0f}s  "
              f"{res['latency']['median_ms']:.1f} ms", flush=True)
        del model, core, pred
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
