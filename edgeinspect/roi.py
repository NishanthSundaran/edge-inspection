"""Inspection region (ROI): score only the part, not the background around it.

MVTec labels an image by the part. A loose bristle lying on the black background next to a good
toothbrush is still a good toothbrush, but PatchCore (rightly) finds the bristle unusual, and those three
images cost toothbrush 8 points of image AUROC. A production cell has the same problem with dust on the
fixture, so the usual fix is an inspection region.

ROI = (this image's part mask) OR (the category template), both dilated:
  part mask   pixels that differ from the background colour, estimated from the image border; the
              largest connected blob, holes filled. Follows a part that is shifted or rotated.
  template    union of the part masks of all good *training* images. Keeps the place where a part
              should be inside the ROI even when it is missing (transistor "misplaced").
Textures, and objects whose template covers most of the frame (bottle, cable, metal_nut, screw,
transistor, zipper), use the full image.
No test image is used to build anything here.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

FULL_FRAME = 0.6  # template covering more than this fraction of the image -> no ROI
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}  # the whole frame is the part


def part_mask(x: np.ndarray, thr: float = 0.12, border: int = 8) -> np.ndarray:
    """x: (3, H, W) float image in [0, 1]. Returns a boolean (H, W) mask of the part."""
    edge = np.concatenate([x[:, :border].reshape(3, -1), x[:, -border:].reshape(3, -1),
                           x[:, :, :border].reshape(3, -1), x[:, :, -border:].reshape(3, -1)], 1)
    bg = np.median(edge, 1)[:, None, None]
    fg = np.abs(x - bg).max(0) > thr
    fg = ndimage.binary_opening(fg, iterations=2)
    lab, n = ndimage.label(fg)
    if n == 0:
        return np.ones(fg.shape, bool)
    fg = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndimage.binary_fill_holes(fg)


def template(images: list[np.ndarray]) -> np.ndarray:
    t = np.zeros(images[0].shape[1:], bool)
    for x in images:
        t |= part_mask(x)
    return t


def uses_roi(category: str, tmpl: np.ndarray) -> bool:
    return category not in TEXTURES and tmpl.mean() <= FULL_FRAME


def roi(x: np.ndarray, tmpl: np.ndarray, dilate: float = 1 / 32) -> np.ndarray:
    """Dilated by `dilate` of the image side (8 px at 256), so edge defects stay inside."""
    if tmpl.mean() > FULL_FRAME:
        return np.ones(tmpl.shape, bool)
    return ndimage.binary_dilation(part_mask(x) | tmpl, iterations=max(1, round(dilate * tmpl.shape[0])))
