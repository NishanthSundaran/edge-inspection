"""Build the service's model folder: deploy/<category>/{model.xml, model.bin, meta.json}.

The model is the OpenVINO INT8 graph with the anomaly head kept in FP32 (see bench/runtimes.py). The
threshold (a starting point; POST /calibrate on the line replaces it) is the highest score the *deployed path itself* (PIL resize + this INT8 graph) gives a
calibration image, since quantisation shifts the score scale; calibration images are good training
images the model never trained on.

  python bench/deploy.py --tag patchcore --categories all --out deploy
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import torch
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench.runtimes import RUNS, quantize_int8  # noqa: E402
from edgeinspect import data, roi  # noqa: E402
from edgeinspect.predict import export_onnx  # noqa: E402
from edgeinspect.serving import InspectionModel, preprocess  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="patchcore")
    ap.add_argument("--categories", nargs="+", default=["all"])
    ap.add_argument("--out", default=str(REPO / "deploy"))
    a = ap.parse_args()
    cats = data.CATEGORIES if a.categories == ["all"] else a.categories
    for c in cats:
        ck = torch.load(RUNS / a.tag / c / "core.pt", weights_only=False)
        exp = RUNS / a.tag / c / "export"
        exp.mkdir(exist_ok=True)
        onnx_path = exp / "model.onnx"
        onnx_path.exists() or export_onnx(ck["core"], onnx_path, ck["size"])
        src = exp / "ov_int8_head-fp32.xml"
        src.exists() or quantize_int8(onnx_path, src, c, ck["size"], ck["bicubic"], "head-fp32")
        d = pathlib.Path(a.out) / c
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d / "model.xml")
        shutil.copy(src.with_suffix(".bin"), d / "model.bin")
        train = sorted((data.DATA / c / "train" / "good").glob("*.png"))
        tmpl = roi.template([preprocess(Image.open(p), ck["size"])[0] for p in train])
        (d / "roi_template.png").unlink(missing_ok=True)
        if roi.uses_roi(c, tmpl):
            Image.fromarray(tmpl.astype(np.uint8) * 255).save(d / "roi_template.png")
        (d / "meta.json").write_text(json.dumps({"category": c, "size": ck["size"], "threshold": 0.0}))
        m = InspectionModel(d)
        s = np.array([m(Image.open(p)).score for p in data.calib_images(c)])
        meta = {"category": c, "size": ck["size"], "threshold": float(s.max()),
                "calib_median": float(np.median(s)), "calib_std": float(s.std()), "n_calib": len(s),
                "runtime": "openvino-cpu-int8 (head fp32)", "model": a.tag,
                "roi": m.roi_template is not None, "threshold_source": "max over held-out good training images"}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(c, meta, flush=True)


if __name__ == "__main__":
    main()
