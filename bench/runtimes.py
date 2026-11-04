"""Export a trained Core to the edge runtimes and score each one with the same evaluation code.

For every category: ONNX (fp32) on ONNX Runtime CPU / CUDA / TensorRT-fp16, OpenVINO CPU (fp32, fp16
weights), and OpenVINO INT8 (NNCF post-training quantisation calibrated on the *calibration split*, i.e.
good training images only, never test images). Reports accuracy deltas against PyTorch and batch-1 latency.

  python bench/runtimes.py --tag patchcore --categories all
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from edgeinspect import data  # noqa: E402
from edgeinspect.evaluate import evaluate  # noqa: E402
from edgeinspect.predict import (OnnxPredictor, OpenVINOPredictor, TorchPredictor, export_onnx,  # noqa: E402
                                 latency_ms, resize)

RUNS = pathlib.Path.home() / "ml_data/inspect/runs"
CPU_THREADS = int(os.environ.get("CPU_THREADS", "8"))


def export_openvino(onnx_path: pathlib.Path, out: pathlib.Path, fp16: bool) -> pathlib.Path:
    import openvino as ov
    m = ov.convert_model(str(onnx_path))
    ov.save_model(m, str(out), compress_to_fp16=fp16)
    return out


def quantize_int8(onnx_path: pathlib.Path, out: pathlib.Path, category: str, size: int, bicubic: bool,
                  mode: str) -> pathlib.Path:
    """mode: naive        NNCF post-training quantisation of the whole graph;
             weights      weight-only INT8 (activations stay FP32);
             head-fp32    quantise the feature extractor and MatMuls, keep the anomaly head FP32
                          (distances, upsampling, blur). Its scores leave the calibrated range of good
                          images exactly where a defect is, so quantising them clips the signal.
             backbone     quantise the feature extractor only; the nearest-neighbour search stays FP32.
                          The search computes |x|^2 + |y|^2 - 2 x.y, a small difference of large terms,
                          so INT8 error in the x.y MatMul swamps the gap between close neighbours.
    Calibration images are the calibration split: good training images only."""
    import nncf
    import openvino as ov
    m = ov.convert_model(str(onnx_path))
    if mode == "weights":
        q = nncf.compress_weights(m, mode=nncf.CompressWeightsMode.INT8_ASYM)
    else:
        calib = [resize(data.load_rgb(p), size, bicubic)[None].numpy() for p in data.calib_images(category)]
        kw = {}
        if mode == "backbone":
            head = [op.get_friendly_name() for op in m.get_ops()
                    if "feature_extractor" not in op.get_friendly_name()
                    and op.get_type_name() not in ("Constant", "Parameter", "Result")]
            kw["ignored_scope"] = nncf.IgnoredScope(names=head, validate=False)
        if mode == "head-fp32":
            head = [op.get_friendly_name() for op in m.get_ops()
                    if "feature_extractor" not in op.get_friendly_name()
                    and op.get_type_name() not in ("Constant", "Parameter", "Result", "MatMul")]
            kw["ignored_scope"] = nncf.IgnoredScope(names=head, validate=False)
        q = nncf.quantize(m, nncf.Dataset(calib), subset_size=len(calib), **kw)
    ov.save_model(q, str(out))
    return out


class ServicePredictor:
    """What ships: edgeinspect.serving.InspectionModel on deploy/<category> (PIL resize, the gated INT8 or
    FP32 graph, the ROI). Build the folder with bench/deploy.py first."""
    takes_paths = True
    name = "service"

    def __init__(self, model_dir):
        from edgeinspect.serving import InspectionModel
        self.m = InspectionModel(model_dir)

    def from_paths(self, paths):
        from PIL import Image
        vs = [self.m(Image.open(p)) for p in paths]
        return np.array([v.score for v in vs], np.float32), np.stack([v.anomaly_map for v in vs])

    def __call__(self, x):  # latency of the graph plus the ROI step on an already-resized image
        from PIL import Image
        img = Image.fromarray((x[0].transpose(1, 2, 0) * 255).astype(np.uint8))
        v = self.m(img)
        return np.array([v.score]), v.anomaly_map[None]

    def sync(self):
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--categories", nargs="+", default=["all"])
    ap.add_argument("--only", nargs="*", default=None, help="subset of runtime names")
    args = ap.parse_args()
    cats = data.CATEGORIES if args.categories == ["all"] else args.categories
    out_dir = REPO / "results" / f"{args.tag}_runtimes"
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in cats:
        res_file = out_dir / f"{c}.json"
        done = json.loads(res_file.read_text()) if res_file.exists() else {}
        ck = torch.load(RUNS / args.tag / c / "core.pt", weights_only=False)
        core, size, bicubic = ck["core"], ck["size"], ck["bicubic"]
        exp = RUNS / args.tag / c / "export"
        exp.mkdir(exist_ok=True)
        onnx_path = exp / "model.onnx"
        if not onnx_path.exists():
            export_onnx(core, onnx_path, size)

        def make(name):
            if name == "pytorch-cuda-fp32":
                return TorchPredictor(core, "cuda")
            if name == "pytorch-cuda-fp16":
                return TorchPredictor(core, "cuda", half=True)
            if name == "pytorch-cpu-fp32":
                torch.set_num_threads(CPU_THREADS)
                return TorchPredictor(core, "cpu")
            if name == "ort-cpu-fp32":
                return OnnxPredictor(onnx_path, "CPUExecutionProvider", CPU_THREADS)
            if name == "ort-cuda-fp32":
                return OnnxPredictor(onnx_path, "CUDAExecutionProvider")
            if name == "ort-tensorrt-fp16":
                return OnnxPredictor(onnx_path, "TensorrtExecutionProvider")
            if name == "openvino-cpu-fp32":
                p = exp / "ov_fp32.xml"
                p.exists() or export_openvino(onnx_path, p, fp16=False)
                return OpenVINOPredictor(p, "CPU")
            if name.startswith("openvino-cpu-int8-"):
                mode = name.removeprefix("openvino-cpu-int8-")
                p = exp / f"ov_int8_{mode}.xml"
                p.exists() or quantize_int8(onnx_path, p, c, size, bicubic, mode)
                return OpenVINOPredictor(p, "CPU")
            if name == "service":
                d = REPO / "deploy" / c
                if not (d / "meta.json").exists():
                    raise FileNotFoundError(f"{d}: run bench/deploy.py first")
                return ServicePredictor(d)
            raise ValueError(name)

        names = args.only or ["pytorch-cuda-fp32", "pytorch-cuda-fp16", "ort-cuda-fp32", "ort-tensorrt-fp16",
                              "pytorch-cpu-fp32", "ort-cpu-fp32", "openvino-cpu-fp32", "openvino-cpu-int8-naive",
                              "openvino-cpu-int8-weights", "openvino-cpu-int8-head-fp32", "openvino-cpu-int8-backbone", "service"]
        x1 = np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)
        for name in names:
            if name in done:
                continue
            try:
                pred = make(name)
                if "cpu" in name or name == "service":
                    pred.eval_batch = 2
                r, arrays = evaluate(pred, c, size, bicubic)
                r["latency"] = latency_ms(pred, x1, warmup=5, reps=30)
            except Exception as e:  # a runtime that fails to export or run is a result too
                r = {"error": f"{type(e).__name__}: {e}"[:500]}
                print(c, name, "ERROR", r["error"], flush=True)
            done[name] = r
            res_file.write_text(json.dumps(done, indent=1))
            if "error" not in r:
                print(f"{c} {name:20s} img {r['image_auroc']:.4f} pix {r['pixel_auroc']:.4f} "
                      f"aupro {r['aupro']:.4f} {r['latency']['median_ms']:7.1f} ms", flush=True)
            del pred
            torch.cuda.empty_cache()
        sizes = {f.name: round(f.stat().st_size / 1e6, 1) for f in exp.glob("*") if f.is_file()}
        done["_files_mb"] = sizes
        res_file.write_text(json.dumps(done, indent=1))


if __name__ == "__main__":
    main()
