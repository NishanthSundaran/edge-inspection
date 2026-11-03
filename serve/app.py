"""Inspection service: one deployed model per MVTec category, CPU only, no torch.

  MODELS_DIR=deploy uvicorn serve.app:app --port 8000
  curl -F file=@part.png "localhost:8000/inspect/bottle?heatmap=true"

Endpoints
  POST /inspect/{category}   image upload -> score, threshold, verdict, latency (and a heatmap PNG, base64)
  GET  /health               models loaded
  GET  /stats                per-category rolling window: flag rate, score drift, latency percentiles
  GET  /metrics              the same in Prometheus text format
  GET  /                     a small live dashboard over /stats

Drift: the threshold was set from calibration scores of known-good parts. If the line changes (lighting,
camera, a new supplier) good parts drift up before anything is flagged. `drift_z` is the rolling median
score of the last N parts, in calibration standard deviations from the calibration median.
"""

from __future__ import annotations

import base64
import collections
import io
import os
import pathlib
import threading
import time

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from PIL import Image, UnidentifiedImageError

from edgeinspect.serving import InspectionModel, heatmap_png

MODELS_DIR = pathlib.Path(os.environ.get("MODELS_DIR", "deploy"))
WINDOW = int(os.environ.get("DRIFT_WINDOW", "200"))
MAX_BYTES = 20 * 1024 * 1024

app = FastAPI(title="edge-inspection", version="0.1.0")
models: dict[str, InspectionModel] = {}
locks: dict[str, threading.Lock] = {}
stats: dict[str, dict] = {}


def _load() -> None:
    for d in sorted(p for p in MODELS_DIR.iterdir() if (p / "meta.json").exists()) if MODELS_DIR.exists() else []:
        m = InspectionModel(d)
        models[m.meta["category"]] = m
        locks[m.meta["category"]] = threading.Lock()
        stats[m.meta["category"]] = {"n": 0, "flagged": 0, "scores": collections.deque(maxlen=WINDOW),
                                     "flags": collections.deque(maxlen=WINDOW),
                                     "latency": collections.deque(maxlen=WINDOW)}


_load()


@app.get("/health")
def health():
    return {"status": "ok" if models else "no models", "categories": sorted(models)}


@app.post("/inspect/{category}")
async def inspect_part(category: str, file: UploadFile = File(...), heatmap: bool = False):
    if category not in models:
        raise HTTPException(404, f"no model for '{category}', have {sorted(models)}")
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image larger than 20 MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "not a readable image") from None
    m = models[category]
    t0 = time.perf_counter()
    with locks[category]:  # one compiled model per category; serialise calls on it
        v = m(img)
    ms = (time.perf_counter() - t0) * 1e3
    st = stats[category]
    st["n"] += 1
    st["flagged"] += int(v.defect)
    st["scores"].append(v.score)
    st["flags"].append(int(v.defect))
    st["latency"].append(ms)
    out = {"category": category, "score": round(v.score, 5), "threshold": round(v.threshold, 5),
           "defect": v.defect, "latency_ms": round(ms, 1), "model": m.meta.get("runtime", "")}
    if heatmap:
        out["heatmap_png"] = base64.b64encode(heatmap_png(img, v.anomaly_map, v.threshold, m.size)).decode()
    return out


def _summary(category: str) -> dict:
    st, meta = stats[category], models[category].meta
    s = np.array(st["scores"], float)
    lat = np.array(st["latency"], float)
    cal_med, cal_sd = meta.get("calib_median"), meta.get("calib_std")
    drift = float((np.median(s) - cal_med) / cal_sd) if len(s) and cal_sd else None
    return {"inspected": st["n"], "flagged": st["flagged"], "window": len(s),
            "window_flag_rate": float(np.mean(st["flags"])) if len(s) else None,
            "drift_z": None if drift is None else round(drift, 2),
            "latency_ms_p50": float(np.percentile(lat, 50)) if len(lat) else None,
            "latency_ms_p95": float(np.percentile(lat, 95)) if len(lat) else None,
            "threshold": meta["threshold"], "runtime": meta.get("runtime", "")}


@app.get("/stats")
def all_stats():
    return {c: _summary(c) for c in models}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = []
    for name, key, help_ in [("inspect_parts_total", "inspected", "parts inspected"),
                             ("inspect_flagged_total", "flagged", "parts flagged as defective"),
                             ("inspect_window_flag_rate", "window_flag_rate", "flag rate over the rolling window"),
                             ("inspect_drift_z", "drift_z", "rolling median score vs calibration, in calibration SDs"),
                             ("inspect_latency_ms_p50", "latency_ms_p50", "median model latency"),
                             ("inspect_latency_ms_p95", "latency_ms_p95", "p95 model latency")]:
        kind = "counter" if name.endswith("_total") else "gauge"
        lines += [f"# HELP {name} {help_}", f"# TYPE {name} {kind}"]
        for c in models:
            v = _summary(c)[key]
            if v is not None:
                lines.append(f'{name}{{category="{c}"}} {v}')
    return "\n".join(lines) + "\n"


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (pathlib.Path(__file__).parent / "dashboard.html").read_text()
