"""The deployed inference path, with no torch dependency: PIL resize -> OpenVINO (or ONNX Runtime) -> score,
map, verdict against the calibrated threshold.

PIL's bilinear filter is antialiased on downscale, like torchvision's `antialias=True` resize the models
were trained with; the two agree within 1/255 per pixel (uint8 rounding). bench/runtimes.py scores this
exact path on the full test set ("service"), so the published numbers are for what ships.
"""

from __future__ import annotations

import io
import json
import pathlib
from dataclasses import dataclass

import numpy as np
from PIL import Image


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    """PIL image -> float32 (1, 3, S, S) in [0, 1]."""
    rgb = img.convert("RGB").resize((size, size), Image.BILINEAR)
    return (np.asarray(rgb, np.float32) / 255.0).transpose(2, 0, 1)[None]


@dataclass
class Verdict:
    score: float
    threshold: float
    defect: bool
    anomaly_map: np.ndarray  # (S, S) float32, raw


class InspectionModel:
    """One category's deployed model: `<dir>/model.xml` (OpenVINO IR) or `model.onnx`, plus `meta.json`
    with {"category", "size", "threshold", "runtime", ...}."""

    def __init__(self, model_dir: pathlib.Path, device: str = "CPU"):
        model_dir = pathlib.Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.size = int(self.meta["size"])
        self.threshold = float(self.meta["threshold"])
        xml, onnx = model_dir / "model.xml", model_dir / "model.onnx"
        if xml.exists():
            import openvino as ov
            core = ov.Core()
            self._compiled = core.compile_model(core.read_model(str(xml)), device)
            self._run = lambda x: [np.asarray(v) for v in self._compiled(x).values()]
        elif onnx.exists():
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            self._run = lambda x: sess.run(None, {"image": x})
        else:
            raise FileNotFoundError(f"no model.xml or model.onnx in {model_dir}")

    def __call__(self, img: Image.Image) -> Verdict:
        score, amap = self._run(preprocess(img, self.size))
        s = float(np.asarray(score).reshape(-1)[0])
        return Verdict(s, self.threshold, s > self.threshold, np.asarray(amap).reshape(self.size, self.size))


def heatmap_png(img: Image.Image, amap: np.ndarray, threshold: float, size: int) -> bytes:
    """Overlay: map scaled so that the decision threshold sits at the middle of the colour range."""
    base = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), np.float32)
    a = np.clip(amap / (2 * max(threshold, 1e-6)), 0, 1)
    heat = np.stack([a, np.clip(1.5 - np.abs(4 * a - 3), 0, 1) * 0.6, 1 - a], -1) * 255  # blue -> red
    alpha = (0.15 + 0.55 * a)[..., None]
    out = (base * (1 - alpha) + heat * alpha).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue()
