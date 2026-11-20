"""Colour-agnostic poster sighting and matching on host-supplied PIL frames.

Only numpy, scipy and Pillow are used. Nothing here is keyed to a colour or to
a coordinate. Generation 3 adds two detector paths, both structural:
  * sight(img, scene_chroma): a near-view fallback. When the parent rule finds
    nothing, the chroma threshold is re-anchored to the scene baseline (median
    chroma of earlier non-poster maze frames) instead of the current frame's
    median, which a close poster view contaminates. Only near components are
    accepted from this path, so distant-poster behaviour is unchanged.
  * sight_question(img): question photos sit on a neutral backdrop (chroma 0),
    so an Otsu split of chroma separates poster from backdrop without a median.
Identity is a normalised chroma/luminance grid compared under the eight
dihedral transforms (unchanged from generation 2).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

GRID = 8  # descriptor grid side (cells); structural, not episode-fitted
NEAR_COVER = 0.5   # a component covering more than half the frame is a near view (question-scale)
MEDIAN_FACTOR = 2.0  # chroma above twice the reference median counts as poster material
MIN_AREA_FRACTION = 0.005  # HUD text and specks are far below this share of the frame
FILL_STD = 2.0     # chroma std below this across the whole frame = flat fill (no separable structure)
CLOSE_ITER = 2     # binary closing passes for question masks (heals low-chroma poster cells)


def _largest_component(mask, chroma, thr):
    """Return the largest connected component of a boolean mask, or None if it is a speck."""
    h, w = mask.shape
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    if sizes[k - 1] < MIN_AREA_FRACTION * h * w:
        return None
    comp = labels == k
    ys, xs = np.where(comp)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    cover = (y1 - y0) * (x1 - x0) / float(h * w)
    inside = chroma[y0:y1, x0:x1]
    gy, gx = np.gradient(inside)
    texture = float(np.mean(np.hypot(gx, gy)) / (inside.mean() + 1e-6))
    return {"kind": "poster", "box": [x0, y0, x1, y1], "cover": round(cover, 3),
            "near": bool(cover > NEAR_COVER), "threshold": round(thr, 1), "texture": round(texture, 3),
            "fg_chroma": round(float(chroma[comp].mean()), 1),
            "bg_chroma": round(float(chroma[~comp].mean()), 1)}


def sight(img: Image.Image, scene_chroma=None) -> dict:
    """Return a JSON-safe record describing the poster (if any) in one maze frame."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    if float(chroma.std()) < FILL_STD:
        return {"kind": "fill", "chroma_mean": round(float(chroma.mean()), 1)}
    # Parent rule: threshold relative to this frame's own median (structural prior).
    thr = MEDIAN_FACTOR * float(np.median(chroma))
    rec = _largest_component(chroma > thr, chroma, thr)
    if rec is None and scene_chroma is not None:
        # Near-view fallback: a close poster view makes the frame's own median poster-coloured,
        # so re-anchor to the scene baseline and accept only near components.
        thr_scene = MEDIAN_FACTOR * float(scene_chroma)
        cand = _largest_component(chroma > thr_scene, chroma, thr_scene)
        if cand is not None and cand["near"]:
            cand["via"] = "scene_baseline"
            rec = cand
    if rec is None:
        return {"kind": "none", "threshold": round(thr, 1)}
    return rec


def chroma_median(img: Image.Image) -> float:
    """Median chroma of a whole frame: the scene sample the agent keeps for the near-view baseline."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return float(np.median(arr.max(axis=2) - arr.min(axis=2)))


def _otsu_threshold(ci: np.ndarray) -> int:
    hist = np.bincount(ci.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    levels = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - mu) ** 2 / (w0 * (1.0 - w0))
    between = np.nan_to_num(between)
    return int(np.argmax(between))


def sight_question(img: Image.Image) -> dict:
    """Localise the poster in a question photo: Otsu split of chroma, then closing and fill holes."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    if float(chroma.max()) <= 0.0:
        return {"kind": "none", "method": "otsu_question"}
    t = _otsu_threshold(np.rint(chroma).astype(np.int64))
    hi = chroma > t
    lo = ~hi
    if not hi.any() or not lo.any():
        return {"kind": "none", "method": "otsu_question", "threshold": t}
    hi_mean, lo_mean = float(chroma[hi].mean()), float(chroma[lo].mean())
    if hi_mean < MEDIAN_FACTOR * lo_mean:
        return {"kind": "none", "method": "otsu_question", "threshold": t}
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(hi, iterations=CLOSE_ITER))
    rec = _largest_component(mask, chroma, float(t))
    if rec is None:
        return {"kind": "none", "method": "otsu_question", "threshold": t}
    rec["method"] = "otsu_question"
    return rec


def describe(img: Image.Image, box) -> list:
    """Normalised GRID x GRID descriptor of a box, as a flat list of floats."""
    crop = img.convert("RGB").crop(tuple(box)).resize((GRID * 4, GRID * 4), Image.Resampling.BILINEAR)
    a = np.asarray(crop, dtype=np.float32)
    cells = a.reshape(GRID, 4, GRID, 4, 3).mean(axis=(1, 3))
    lum = cells.mean(axis=2)
    chroma = cells.max(axis=2) - cells.min(axis=2)
    feat = np.stack([lum - lum.mean(), chroma - chroma.mean()], axis=0)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return [round(float(v), 5) for v in feat.reshape(-1)]


def _transforms(grid_feat: np.ndarray):
    """Eight dihedral transforms of a (2, GRID, GRID) feature."""
    out = []
    for flip in (False, True):
        base = grid_feat[:, :, ::-1] if flip else grid_feat
        for k in range(4):
            out.append(np.rot90(base, k, axes=(1, 2)))
    return out


def similarity(desc_a: list, desc_b: list) -> dict:
    """Best dihedral normalised correlation between two descriptors."""
    a = np.asarray(desc_a, dtype=np.float64).reshape(2, GRID, GRID)
    b = np.asarray(desc_b, dtype=np.float64).reshape(2, GRID, GRID)
    scores = [float(np.sum(t * b)) for t in _transforms(a)]
    best = int(np.argmax(scores))
    return {"score": round(scores[best], 4), "transform": best}
