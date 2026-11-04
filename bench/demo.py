"""Record the service demo: real HTTP calls to a running service, one frame per inspected part.

  MODELS_DIR=deploy uvicorn serve.app:app --port 8000 &
  python bench/demo.py --url http://localhost:8000 --out demo/service_demo.gif

Each frame: the part as sent, the heatmap the service returned, and its verdict, score / threshold and
latency, exactly as in the JSON response.
"""

from __future__ import annotations

import argparse
import base64
import io
import pathlib
import sys

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from edgeinspect import data  # noqa: E402

# (category, test image): a good part and a defect per category, plus the good toothbrush with a loose
# bristle on the background that the inspection region keeps from being flagged
PARTS = [("bottle", "good/000"), ("bottle", "broken_large/002"),
         ("capsule", "good/003"), ("capsule", "crack/005"),
         ("toothbrush", "good/006"), ("toothbrush", "defective/012"),
         ("hazelnut", "good/010"), ("hazelnut", "print/003"),
         ("screw", "good/012"), ("screw", "scratch_neck/009"),
         ("leather", "good/004"), ("leather", "cut/007"),
         ("metal_nut", "good/005"), ("metal_nut", "flip/004"),
         ("transistor", "good/011"), ("transistor", "misplaced/003")]
S = 360


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if pathlib.Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def frame(img, heat, r, truth):
    W, H = 2 * S + 30, S + 96
    f = Image.new("RGB", (W, H), (18, 20, 24))
    f.paste(img.convert("RGB").resize((S, S)), (10, 10))
    f.paste(heat.resize((S, S)), (S + 20, 10))
    d = ImageDraw.Draw(f)
    ok = not r["defect"]
    colour = (90, 200, 120) if ok else (235, 80, 70)
    d.text((14, S + 20), f"{r['category']}: {'OK' if ok else 'DEFECT'}", fill=colour, font=font(26))
    d.text((14, S + 58), f"score {r['score']:.1f} / threshold {r['threshold']:.1f}    {r['latency_ms']:.0f} ms on CPU",
           fill=(200, 200, 205), font=font(17))
    d.text((W - 14, S + 26), f"truth: {truth}", fill=(150, 150, 160), font=font(17), anchor="ra")
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--out", default=str(REPO / "demo" / "service_demo.gif"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for cat in dict(PARTS):  # first call per model compiles kernels; not part of the steady-state latency
        p = data.DATA / cat / "test" / "good" / "000.png"
        requests.post(f"{a.url}/inspect/{cat}", files={"file": (p.name, p.read_bytes(), "image/png")}, timeout=120)
    frames = []
    for cat, name in PARTS:
        p = data.DATA / cat / "test" / f"{name}.png"
        r = requests.post(f"{a.url}/inspect/{cat}?heatmap=true", files={"file": (p.name, p.read_bytes(), "image/png")},
                          timeout=60).json()
        heat = Image.open(io.BytesIO(base64.b64decode(r.pop("heatmap_png"))))
        truth = "good" if name.startswith("good") else name.split("/")[0].replace("_", " ")
        print(cat, name, r, flush=True)
        frames.append(frame(Image.open(p), heat, r, truth))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=1600, loop=0, optimize=True)
    for i, fr in enumerate(frames):
        fr.save(out.parent / f"frame_{i:02d}.png")
    print("wrote", out, len(frames), "frames")


if __name__ == "__main__":
    main()
