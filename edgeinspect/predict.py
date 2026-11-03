"""One predictor interface for every runtime: float32 (N, 3, S, S) in [0, 1] -> raw scores (N,), maps (N, S, S).

The graph each runtime runs is `Core`: ImageNet normalisation + anomalib's bare torch model, no
post-processor. anomalib's own export also bakes in min-max normalisation with a threshold learned on its
validation split (the test set by default), which this study does not want in a deployed model.
Resizing happens on the host with `resize`, identically for every runtime.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

def resize(rgb: np.ndarray, size: int, bicubic: bool = False) -> torch.Tensor:
    """uint8 (H, W, 3) -> float (3, S, S) in [0, 1], antialiased like anomalib's training transform."""
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
    mode = TF.InterpolationMode.BICUBIC if bicubic else TF.InterpolationMode.BILINEAR
    return TF.resize(t, [size, size], interpolation=mode, antialias=True).clamp(0, 1)


class Core(torch.nn.Module):
    """Normalisation + bare torch model. mean/std come from the model's own anomalib pre-processor
    (ImageNet for PatchCore and PaDiM, CLIP for WinCLIP, none for EfficientAD, which normalises inside)."""

    def __init__(self, model: torch.nn.Module, mean=None, std=None):
        super().__init__()
        self.model = model
        mean = torch.zeros(3) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        std = torch.ones(3) if std is None else torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 3, 1, 1))
        self.register_buffer("std", std.view(1, 3, 1, 1))

    @classmethod
    def from_anomalib(cls, module) -> tuple["Core", int, bool]:
        """-> (core, input size, bicubic) read from the module's pre-processor transform."""
        from torchvision.transforms.v2 import Normalize, Resize
        mean = std = None
        size, bicubic = 256, False
        for t in module.pre_processor.transform.transforms:
            if isinstance(t, Normalize):
                mean, std = t.mean, t.std
            if isinstance(t, Resize):
                size = t.size[0]
                bicubic = t.interpolation == TF.InterpolationMode.BICUBIC
        return cls(module.model, mean, std), size, bicubic

    def forward(self, x):
        out = self.model((x - self.mean) / self.std)
        amap = out.anomaly_map
        if amap.dim() == 4:
            amap = amap[:, 0]
        if amap.shape[-1] != x.shape[-1]:
            amap = F.interpolate(amap[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return out.pred_score.reshape(-1), amap


class TorchPredictor:
    name = "pytorch"

    def __init__(self, core: Core, device: str = "cuda", half: bool = False):
        self.core = core.eval().to(device)
        self.device, self.half = device, half
        if half:
            self.core.half()

    @torch.inference_mode()
    def __call__(self, x: np.ndarray):
        t = torch.from_numpy(x).to(self.device)
        if self.half:
            t = t.half()
        s, m = self.core(t)
        return s.float().cpu().numpy(), m.float().cpu().numpy()

    def sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()


def export_onnx(core: Core, path: pathlib.Path, size: int, opset: int = 17) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    core = core.eval().cpu()
    torch.onnx.export(core, (torch.zeros(1, 3, size, size),), str(path), opset_version=opset,
                      input_names=["image"], output_names=["score", "anomaly_map"],
                      dynamic_axes={"image": {0: "n"}, "score": {0: "n"}, "anomaly_map": {0: "n"}},
                      dynamo=False)
    return path


class OnnxPredictor:
    def __init__(self, path: pathlib.Path, provider: str = "CPUExecutionProvider", threads: int | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        provs = [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
        if provider == "TensorrtExecutionProvider":
            cache = path.parent / "trt_cache"
            provs = [("TensorrtExecutionProvider", {"trt_engine_cache_enable": True,
                                                    "trt_engine_cache_path": str(cache),
                                                    "trt_fp16_enable": True}),
                     "CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(path), so, providers=provs)
        self.name = f"onnxruntime-{provider.replace('ExecutionProvider', '').lower()}"

    def __call__(self, x: np.ndarray):
        s, m = self.sess.run(None, {"image": x.astype(np.float32)})
        return s.reshape(-1), m

    def sync(self):
        pass


class OpenVINOPredictor:
    def __init__(self, path: pathlib.Path, device: str = "CPU"):
        import openvino as ov
        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(str(path)), device)
        self.name = f"openvino-{device.lower()}"

    def __call__(self, x: np.ndarray):
        r = self.compiled(x.astype(np.float32))
        outs = list(r.values())
        return np.asarray(outs[0]).reshape(-1), np.asarray(outs[1])

    def sync(self):
        pass


def latency_ms(pred, x1: np.ndarray, warmup: int = 10, reps: int = 50) -> dict:
    """Batch-1 latency of the model graph alone (host resize excluded), median and p95."""
    for _ in range(warmup):
        pred(x1)
    pred.sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        pred(x1)
        pred.sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}
