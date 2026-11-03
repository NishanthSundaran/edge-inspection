"""Service tests with a tiny OpenVINO model built in code (score = mean brightness, map = channel mean),
so CI needs neither MVTec nor torch."""

import base64
import importlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from tests.tiny_model import S, write_tiny_model as _tiny_model


@pytest.fixture()
def client(tmp_path, monkeypatch):
    d = tmp_path / "tiny"
    d.mkdir()
    _tiny_model(d)
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    from fastapi.testclient import TestClient
    import serve.app as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


def _png(value: int, size=64) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), value, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "categories": ["tiny"]}


def test_verdicts_and_stats(client):
    dark = client.post("/inspect/tiny", files={"file": ("a.png", _png(20), "image/png")}).json()
    bright = client.post("/inspect/tiny?heatmap=true", files={"file": ("b.png", _png(240), "image/png")}).json()
    assert dark["defect"] is False and bright["defect"] is True
    assert dark["score"] == pytest.approx(20 / 255, abs=1e-3)
    png = base64.b64decode(bright["heatmap_png"])
    assert Image.open(io.BytesIO(png)).size == (S, S)
    st = client.get("/stats").json()["tiny"]
    assert st["inspected"] == 2 and st["flagged"] == 1 and st["window_flag_rate"] == 0.5
    m = client.get("/metrics").text
    assert 'inspect_parts_total{category="tiny"} 2' in m


def test_bad_requests(client):
    assert client.post("/inspect/nope", files={"file": ("a.png", _png(1), "image/png")}).status_code == 404
    assert client.post("/inspect/tiny", files={"file": ("a.txt", b"not an image", "text/plain")}).status_code == 415


def test_preprocess_matches_training_resize():
    torch = pytest.importorskip("torch")
    from edgeinspect.predict import resize
    from edgeinspect.serving import preprocess
    rgb = (np.random.default_rng(0).random((300, 300, 3)) * 255).astype(np.uint8)
    a = preprocess(Image.fromarray(rgb), 64)[0]
    b = resize(rgb, 64).numpy()
    assert np.abs(a - b).max() < 2 / 255
