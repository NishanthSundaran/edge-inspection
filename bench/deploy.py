"""Build the service's model folder: deploy/<category>/{model.xml, model.bin, meta.json}.

The model is the OpenVINO INT8 graph with the anomaly head kept in FP32 (see bench/runtimes.py) and the
memory bank rebuilt from the INT8 backbone's own features (bench/int8_bank.py), if it passes a
label-free gate: on the calibration images (good parts the model never trained on), INT8 scores must be
within GATE (5%) of FP32 on average. A category that fails ships the FP32 graph (screw: 10% error; its
test AUROC would drop 6 points in INT8, the other categories <1). The gate never looks at test images.

The threshold (a starting point; POST /calibrate on the line replaces it) is the highest score the
*deployed path itself* (PIL resize, the shipped graph, the ROI) gives a calibration image.

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

from bench.runtimes import RUNS  # noqa: E402
from edgeinspect import data, roi  # noqa: E402
from edgeinspect.predict import OpenVINOPredictor, export_onnx  # noqa: E402
from edgeinspect.serving import InspectionModel, preprocess  # noqa: E402


GATE = 0.05


def int8_gate(fp32: pathlib.Path, int8: pathlib.Path, category: str, size: int) -> float:
    """Mean relative score difference INT8 vs FP32 on the calibration images (same preprocessing)."""
    a, b = OpenVINOPredictor(fp32), OpenVINOPredictor(int8)
    err = []
    for p in data.calib_images(category):
        x = preprocess(Image.open(p), size)
        sa, sb = float(a(x)[0][0]), float(b(x)[0][0])
        err.append(abs(sb - sa) / abs(sa))
    return float(np.mean(err))


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
        int8 = exp / "ov_int8_head-fp32_rebank.xml"  # bench/int8_bank.py: bank rebuilt from INT8 features
        fp32 = exp / "ov_fp32.xml"
        for f in (int8, fp32):
            if not f.exists():
                raise SystemExit(f"{f} missing: run bench/runtimes.py and bench/int8_bank.py for {c} first")
        gate = int8_gate(fp32, int8, c, ck["size"])
        src, runtime = (int8, "openvino-cpu-int8 (head fp32, int8 bank)") if gate <= GATE else \
                       (fp32, "openvino-cpu-fp32 (int8 gate failed)")
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
                "runtime": runtime, "int8_score_error": round(gate, 4), "model": a.tag,
                "roi": m.roi_template is not None, "threshold_source": "max over held-out good training images"}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))
        print(c, meta, flush=True)


if __name__ == "__main__":
    main()
