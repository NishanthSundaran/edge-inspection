"""Rebuild the PatchCore memory bank with the INT8 backbone.

INT8 activations move every patch embedding a little. The FP32 memory bank was built from FP32
embeddings, so after quantisation a good test patch sits slightly off its FP32 neighbours, and that
offset is the size of the subtle defects (capsule image AUROC 0.97 -> 0.64). The fix tested here: run
the training images through the quantised backbone, select the coreset from those INT8 embeddings, and
put that bank in the deployed graph, so the bank and the queries come from the same feature extractor.

  python bench/int8_bank.py --tag patchcore --categories capsule screw
Adds runtimes "openvino-cpu-int8-<mode>-rebank" to results/<tag>_runtimes/<category>.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench.runtimes import RUNS, quantize_int8  # noqa: E402
from edgeinspect import data  # noqa: E402
from edgeinspect.evaluate import evaluate  # noqa: E402
from edgeinspect.predict import OpenVINOPredictor, export_onnx, latency_ms, resize  # noqa: E402


def int8_embeddings(xml: pathlib.Path, paths, size: int, bicubic: bool) -> np.ndarray:
    """Patch embeddings (n*h*w, d) from the quantised graph: the input of the nearest-neighbour MatMul."""
    import openvino as ov
    core = ov.Core()
    m = core.read_model(str(xml))
    mm = next(op for op in m.get_ops() if op.get_type_name() == "MatMul")
    m.add_outputs([mm.input(0).get_source_output()])
    c = core.compile_model(m, "CPU")
    out = c.outputs[-1]
    return np.concatenate([c(resize(data.load_rgb(p), size, bicubic)[None].numpy())[out] for p in paths])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="patchcore")
    ap.add_argument("--categories", nargs="+", default=["all"])
    ap.add_argument("--modes", nargs="+", default=["head-fp32"])
    ap.add_argument("--coreset", type=float, default=0.1)
    a = ap.parse_args()
    from anomalib.models.components.sampling import KCenterGreedy
    cats = data.CATEGORIES if a.categories == ["all"] else a.categories
    for c in cats:
        ck = torch.load(RUNS / a.tag / c / "core.pt", weights_only=False)
        size, bicubic = ck["size"], ck["bicubic"]
        exp = RUNS / a.tag / c / "export"
        calib = {p.name for p in data.calib_images(c)}
        train = sorted(p for p in (data.DATA / c / "train" / "good").glob("*.png") if p.name not in calib)
        res_file = REPO / "results" / f"{a.tag}_runtimes" / f"{c}.json"
        done = json.loads(res_file.read_text())
        for mode in a.modes:
            xml = exp / f"ov_int8_{mode}.xml"
            xml.exists() or quantize_int8(exp / "model.onnx", xml, c, size, bicubic, mode)
            emb = torch.from_numpy(int8_embeddings(xml, train, size, bicubic)).cuda()
            torch.manual_seed(0)
            bank = KCenterGreedy(emb, a.coreset).sample_coreset().cpu()
            del emb
            torch.cuda.empty_cache()
            core = copy.deepcopy(ck["core"])
            core.model.memory_bank = bank
            onnx2 = exp / f"model_rebank_{mode}.onnx"
            export_onnx(core, onnx2, size)
            xml2 = exp / f"ov_int8_{mode}_rebank.xml"
            quantize_int8(onnx2, xml2, c, size, bicubic, mode)
            pred = OpenVINOPredictor(xml2, "CPU")
            pred.eval_batch = 2
            r, _ = evaluate(pred, c, size, bicubic)
            r["latency"] = latency_ms(pred, np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32),
                                      warmup=5, reps=30)
            r["bank"] = list(bank.shape)
            name = f"openvino-cpu-int8-{mode}-rebank"
            done[name] = r
            res_file.write_text(json.dumps(done, indent=1))
            print(f"{c} {name} img {r['image_auroc']:.4f} pix {r['pixel_auroc']:.4f} aupro {r['aupro']:.4f} "
                  f"{r['latency']['median_ms']:.1f} ms (fp32 img {done['openvino-cpu-fp32']['image_auroc']:.4f})",
                  flush=True)


if __name__ == "__main__":
    main()
