"""Whole-view matching for line M.

A stored walk view is a frame kept at a reduced resolution. The question painting is
extracted at full resolution (painting_match.painting_patch), rescaled by the view's
factor, and searched over VIEW_SCALES with the same NCC used for panel crops. The panel
side (L or R) is read from where the best placement sits relative to the doorway column.
Stdlib, numpy, scipy and Pillow only.
"""
from __future__ import annotations

import numpy as np

import painting_match as pm

# Template scales relative to the query painting at view resolution. Dev matches sat at 0.5-1.6;
# the upper end is open enough that a best match is not pinned at the boundary.
VIEW_SCALES = np.geomspace(0.4, 2.2, 10)
# The coarse grid steps by a factor of about 1.2, which can cost ~0.15 NCC when the true scale falls
# between grid points. Views that reach REFINE_FLOOR get two refinement passes at half the previous step.
REFINE_FLOOR = 0.30
COARSE_RATIO = float(VIEW_SCALES[1] / VIEW_SCALES[0])


def query_patch(image):
    """Full-resolution painting patch and its de-rotation angle for a question image."""
    patch, angle = pm.painting_patch(pm.to_array(image))
    return patch, float(angle)


def scaled_patch(patch, factor):
    if abs(factor - 1.0) < 1e-9:
        return patch
    height, width = patch.shape[:2]
    return pm._resize_channels(patch, max(1, int(round(width * factor))), max(1, int(round(height * factor))))


def _scale_hit(view, base, scale):
    width = int(round(base.shape[1] * scale))
    height = int(round(base.shape[0] * scale))
    if width < pm.MIN_TEMPLATE_SIDE or height < pm.MIN_TEMPLATE_SIDE:
        return None
    surface = pm.ncc_map(view, pm._resize_channels(base, width, height))
    if surface is None:
        return None
    y, x = np.unravel_index(int(np.argmax(surface)), surface.shape)
    return {"score": float(surface[y, x]), "scale": float(scale), "x": int(x), "y": int(y),
            "w": width, "h": height, "centre": (x + width / 2.0) / float(view.shape[1])}


def best_in_view(view, patch, factor):
    """Best NCC for one stored view (float RGB array at view resolution).

    Coarse search over VIEW_SCALES; then, for a view at or above REFINE_FLOOR, two refinement
    passes around the best scale (step ratio COARSE_RATIO**0.5, then **0.25). Never lowers a score.
    """
    base = scaled_patch(patch, factor)
    best = None
    for scale in VIEW_SCALES:
        hit = _scale_hit(view, base, scale)
        if hit is not None and (best is None or hit["score"] > best["score"]):
            best = hit
    if best is None:
        return {"score": -1.0, "scale": None, "x": None, "y": None, "w": None, "h": None, "centre": None}
    if best["score"] >= REFINE_FLOOR:
        ratio = COARSE_RATIO ** 0.5
        for _pass in range(2):
            centre_scale = best["scale"]
            for scale in (centre_scale / ratio, centre_scale * ratio):
                hit = _scale_hit(view, base, scale)
                if hit is not None and hit["score"] > best["score"]:
                    best = hit
            ratio = ratio ** 0.5
    return best


def side_of(centre, door_x):
    """L when the matched panel centre lies left of the doorway column, R otherwise."""
    return "L" if centre < door_x else "R"


def box_full(best, factor, full_size):
    """Map the best placement back to full-resolution frame pixels [x0, y0, x1, y1]."""
    width, height = full_size
    x0 = int(np.floor(best["x"] / factor))
    y0 = int(np.floor(best["y"] / factor))
    x1 = int(np.ceil((best["x"] + best["w"]) / factor))
    y1 = int(np.ceil((best["y"] + best["h"]) / factor))
    return [max(0, x0), max(0, y0), min(width, x1), min(height, y1)]

