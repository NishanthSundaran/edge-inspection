"""MVTec AD on disk, plus the calibration split this study adds.

anomalib's default validation split is the test set, so thresholds end up tuned on test images. Here a
seeded 10% (at least 10) of each category's *good training* images is held out as a calibration set: the
model never trains on them, and the deployment threshold is set from their scores only. A factory has
exactly this: known-good parts, and no catalogue of future defects.

`build_fit_root` mirrors the dataset with per-file symlinks (train minus calibration, test and masks
unchanged), so anomalib's MVTecAD datamodule trains on the reduced set without knowing about the split.
"""

from __future__ import annotations

import json
import os
import pathlib
import random

import numpy as np
from PIL import Image

DATA = pathlib.Path(os.environ.get("MVTEC_ROOT", pathlib.Path.home() / "ml_data/inspect/data/MVTecAD"))
FIT_ROOT = DATA.parent / "MVTecAD_fit"
SPLIT_FILE = pathlib.Path(__file__).resolve().parents[1] / "results" / "calib_split.json"

CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
              "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
SEED = 0


def _pngs(d: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in d.glob("*.png"))


def make_split() -> dict[str, list[str]]:
    """Deterministic: sorted file names, one seeded shuffle per category."""
    if SPLIT_FILE.exists():
        return json.loads(SPLIT_FILE.read_text())
    split = {}
    for c in CATEGORIES:
        names = [p.name for p in _pngs(DATA / c / "train" / "good")]
        rng = random.Random(f"{SEED}-{c}")
        rng.shuffle(names)
        split[c] = sorted(names[: max(10, round(0.1 * len(names)))])
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text(json.dumps(split, indent=1))
    return split


def build_fit_root() -> pathlib.Path:
    split = make_split()
    for c in CATEGORIES:
        held = set(split[c])
        for src in (DATA / c).rglob("*.png"):
            rel = src.relative_to(DATA)
            if rel.parts[1] == "train" and src.name in held:
                continue
            dst = FIT_ROOT / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src)
    return FIT_ROOT


def calib_images(category: str) -> list[pathlib.Path]:
    return [DATA / category / "train" / "good" / n for n in make_split()[category]]


def test_items(category: str) -> list[tuple[pathlib.Path, int, pathlib.Path | None]]:
    """(image, label, mask path or None for good parts), sorted by defect type then file."""
    items = []
    for d in sorted((DATA / category / "test").iterdir()):
        for p in _pngs(d):
            if d.name == "good":
                items.append((p, 0, None))
            else:
                items.append((p, 1, DATA / category / "ground_truth" / d.name / f"{p.stem}_mask.png"))
    return items


def load_rgb(path: pathlib.Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: pathlib.Path | None, size: tuple[int, int]) -> np.ndarray:
    """Binary mask at `size` (H, W), nearest-neighbour; zeros for good parts."""
    if path is None:
        return np.zeros(size, np.uint8)
    m = Image.open(path).convert("L").resize((size[1], size[0]), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)
