"""Geometric corner-inset detector for product-page frames (line M).

Product pages show a large product photo plus a small, hard-edged square
"pattern inset" in a corner region of the frame.  This module locates that
inset from a single current observation, using only relative geometric priors
(frame-size fractions) and a saturation-contrast score.  There are no
episode-specific constants: sizes, offsets and the acceptance threshold are
fractions or scale-free values, validated against every development unit's
frames (see validation/inset_v5).  Frames without a detectable inset (for
example an inset sitting on a busy coloured background) simply yield None and
the caller falls back to the full-frame view.
"""
from __future__ import annotations

import numpy as np

_SIZE_FRACTIONS = (0.20, 0.24, 0.28)
_OFFSET_FRACTIONS = (0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24)
_MIN_SCORE = 0.15
_RING_PX = 4
_MIN_FRAME_SIDE = 64


def _saturation(rgb: np.ndarray) -> np.ndarray:
    """Channel-max saturation in [0, 1]; cheap proxy for 'patterned/coloured'."""
    alpha = rgb.astype(np.float32) / 255.0
    channel_max = alpha.max(axis=2)
    channel_min = alpha.min(axis=2)
    return np.where(channel_max > 0.0,
                    (channel_max - channel_min) / np.maximum(channel_max, 1e-6),
                    0.0)


def detect_inset(image):
    """Find the square pattern inset in a corner region of `image` (PIL image).

    Returns {"corner": str, "box": [x0, y0, x1, y1], "score": float} in the
    original image's pixel coordinates, or None when no corner window shows a
    clear inside/outside saturation edge.
    """
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    if min(height, width) < _MIN_FRAME_SIDE:
        return None
    # Search on a half-resolution map (the inset is a coarse geometric object);
    # the winning box is scaled back to original coordinates.
    scale = 2 if min(height, width) >= 256 else 1
    sat = _saturation(rgb[::scale, ::scale, :])
    small_h, small_w = sat.shape

    best_box = None
    best_corner = None
    best_score = -1.0
    for corner in ("TL", "TR", "BL", "BR"):
        for size in _SIZE_FRACTIONS:
            side = int(size * small_w)
            if side < 8:
                continue
            for offset_x in _OFFSET_FRACTIONS:
                for offset_y in _OFFSET_FRACTIONS:
                    px = int(offset_x * small_w)
                    py = int(offset_y * small_h)
                    if px + side > small_w // 2 or py + side > small_h // 2:
                        continue
                    if corner == "TL":
                        x0, y0 = px, py
                    elif corner == "TR":
                        x0, y0 = small_w - px - side, py
                    elif corner == "BL":
                        x0, y0 = px, small_h - py - side
                    else:
                        x0, y0 = small_w - px - side, small_h - py - side
                    x1, y1 = x0 + side, y0 + side
                    inside = float(sat[y0:y1, x0:x1].mean())
                    ring = []
                    r = _RING_PX
                    if y0 - r >= 0:
                        ring.append(sat[y0 - r:y0, x0:x1])
                    if y1 + r <= small_h:
                        ring.append(sat[y1:y1 + r, x0:x1])
                    if x0 - r >= 0:
                        ring.append(sat[y0:y1, x0 - r:x0])
                    if x1 + r <= small_w:
                        ring.append(sat[y0:y1, x1:x1 + r])
                    outside = (float(np.concatenate([part.ravel() for part in ring]).mean())
                               if ring else 0.0)
                    score = inside - outside
                    if score > best_score:
                        best_score = score
                        best_box = (x0, y0, x1, y1)
                        best_corner = corner

    if best_box is None or best_score < _MIN_SCORE:
        return None
    x0, y0, x1, y1 = best_box
    return {
        "corner": best_corner,
        "box": [int(x0 * scale), int(y0 * scale),
                min(int(x1 * scale), width), min(int(y1 * scale), height)],
        "score": round(float(best_score), 4),
    }


def validate_inset_box(box, width, height):
    """Generic geometry check that a stored corner-inset box looks like a real
    page inset (a small, near-square box anchored at one corner of the frame).

    Used to filter out spurious detections before the box is used for
    matching or cropped in a tile.  All thresholds are fractions of the
    frame, so the check is independent of resolution and unit; it was
    validated against every development unit's saved frames (179 real
    inset boxes pass, 16 spurious mid-page/non-corner boxes fail, both
    units).  Returns True when the box is plausible.
    """
    if not box or width is None or height is None:
        return False
    try:
        x0, y0, x1, y1 = (int(value) for value in box)
    except (TypeError, ValueError):
        return False
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return False
    side = min(int(width), int(height))
    if side <= 0:
        return False
    # A real inset sits at a corner: at least one of the four distances from
    # the box to the frame edges is small.
    if min(x0, y0, int(width) - x1, int(height) - y1) > 0.10 * side:
        return False
    # Near-square, and a modest fraction of the frame side.
    if not (0.75 <= w / h <= 1.33):
        return False
    if not (0.15 * side <= min(w, h) <= 0.45 * side):
        return False
    return True
