"""Full-page slide matching for dmvhard2 recall probes (line M).

A recall probe is a re-photograph of a product (or of a page's small pattern
inset): it may be rotated, rescaled, partly visible and differently lit.
Sliding the probe -- resized to several generic scales and rotated in 90-degree
steps -- over each candidate page and taking the maximum Normalised
Cross-Correlation is invariant to the probe's position on the page and
robust to scale and lighting.  No unit-specific constants appear here: the
scales, rotations and survivor count are generic, and the candidate set is
supplied by the caller from this episode's own saved frames.

The module keeps a process-local grayscale cache keyed by the frame file
path.  It is deliberately NOT part of the agent's JSON state: it is
recomputed from the saved PNGs on demand, so checkpoints stay serialisable.
"""
from __future__ import annotations

import time

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

# Generic matching parameters (not fitted to any development unit).
_COARSE_SIZES = (128,)
_FINE_SIZES = (96, 128, 160, 192, 224)
_ROTATIONS = (0, 1, 2, 3)
_TOP_N = 12  # coarse survivors that receive the expensive fine pass

_GRAY_CACHE = {}


def to_gray(image):
    """PIL image or 2-D array -> 2-D float32 grayscale array."""
    if isinstance(image, np.ndarray):
        return image.astype(np.float32, copy=False)
    return np.asarray(image.convert("L"), dtype=np.float32)


def gray_from_path(path):
    """Load a saved frame PNG and return its grayscale array (cached)."""
    key = str(path)
    gray = _GRAY_CACHE.get(key)
    if gray is None:
        with Image.open(path) as stored:
            gray = to_gray(stored)
        _GRAY_CACHE[key] = gray
    return gray


def ncc_max(template, image):
    """Maximum NCC of `template` slid over `image` (2-D float arrays).

    Returns a value in [-1, 1]; 0.0 when either is (near) constant.
    """
    th, tw = template.shape
    ih, iw = image.shape
    if th > ih or tw > iw:
        template, image = image, template
        th, tw = template.shape
    n = th * tw
    t = template - template.mean()
    t_norm = float(np.sqrt((t * t).sum()))
    if t_norm < 1e-8:
        return 0.0
    ones = np.ones((th, tw), dtype=template.dtype)
    i_sum = fftconvolve(image, ones[::-1, ::-1], mode="valid")
    i2_sum = fftconvolve(image * image, ones[::-1, ::-1], mode="valid")
    i_mean = i_sum / n
    i_var = np.maximum(i2_sum / n - i_mean * i_mean, 0.0)
    cc = fftconvolve(image, t[::-1, ::-1], mode="valid")
    with np.errstate(divide="ignore", invalid="ignore"):
        ncc = cc / (t_norm * np.sqrt(i_var * n))
    ncc[~np.isfinite(ncc)] = 0.0
    return float(ncc.max())


def _slide_scores(probe, page, sizes, rots):
    """Best NCC of `probe` (PIL) at each size/rotation over `page` (2-D)."""
    best = -2.0
    for size in sizes:
        template = to_gray(probe.resize((size, size), Image.BILINEAR))
        for k in rots:
            template = np.rot90(template, k)
            score = ncc_max(template, page)
            if score > best:
                best = score
    return best


def rank_probe(probe, pages, top_n=_TOP_N):
    """Two-stage NCC ranking of `pages` against the current `probe`.

    probe: PIL image (the current probe observation).
    pages: list of (key, page) where page is a 2-D grayscale array or a PIL
    image, for every candidate page.

    Stage 1 (cheap) scores every page at one scale; stage 2 (expensive)
    re-scores the `top_n` survivors at all scales.  Returns a list of
    (key, score) for every page, sorted by score descending.
    """
    if not pages:
        return []
    gray_pages = [(key, to_gray(page)) for key, page in pages]
    coarse = {key: _slide_scores(probe, gray, _COARSE_SIZES, _ROTATIONS)
              for key, gray in gray_pages}
    survivors = set(sorted(coarse, key=lambda k: -coarse[k])[:top_n])
    out = []
    for key, gray in gray_pages:
        if key in survivors:
            out.append((key, _slide_scores(probe, gray, _FINE_SIZES, _ROTATIONS)))
        else:
            out.append((key, coarse[key]))
    out.sort(key=lambda item: -item[1])
    return out

# ---------------------------------------------------------------------------
# g3: wider-grid full-page matching, pattern-inset matching and central
# product-region matching.  Together they back the MATCH_PROBE decision
# engine in visual_tools.py.  All parameters are generic fractions of the
# frame or fixed NCC grids: nothing is fitted to a development unit.
# ---------------------------------------------------------------------------

_EXT_COARSE_SIZES = (96, 128)
_EXT_FINE_SIZES = (48, 64, 80, 96, 128, 160, 192, 224, 256, 320)
_EXT_TIME_BUDGET = 6.0  # seconds; when the ranking runs over, the coarse
                        # scores stand (graceful degradation, no stall)
_INSET_SCALES = (0.75, 1.0, 1.25)  # probe sizes as a fraction of the inset width
_PRODUCT_SIZES = (96, 160, 224)
_PRODUCT_CENTER = 0.60  # the product photo occupies the central 60% of a page
_PRODUCT_MAX_SIDE = 512


def rank_ext(probe, pages, top_n=_TOP_N, time_budget=_EXT_TIME_BUDGET):
    """Two-stage NCC ranking with a wider scale grid than rank_probe.

    Stage 1 scores every page at two mid scales (rotation 0 only); stage 2
    re-scores the `top_n` survivors across ten scales and all 90-degree
    rotations.  The wider grid catches strongly rescaled re-photographs that
    the base grid misses.  If the ranking runs past `time_budget` seconds the
    remaining survivors keep their coarse scores, so a crowded pool degrades
    gracefully instead of stalling the phase.  Returns (key, score) for every
    page, sorted by score descending.
    """
    if not pages:
        return []
    started = time.monotonic()
    gray_pages = [(key, to_gray(page)) for key, page in pages]
    coarse = {key: _slide_scores(probe, gray, _EXT_COARSE_SIZES, (0,))
              for key, gray in gray_pages}
    survivors = set(sorted(coarse, key=lambda k: -coarse[k])[:top_n])
    out = []
    for key, gray in gray_pages:
        if key in survivors and time.monotonic() - started <= time_budget:
            out.append((key, _slide_scores(probe, gray, _EXT_FINE_SIZES, _ROTATIONS)))
        else:
            out.append((key, coarse[key]))
    out.sort(key=lambda item: -item[1])
    return out


def rank_inset(probe, pages, boxes):
    """Rank only the pages that carry a pattern inset, matching the probe
    against each page's inset crop.

    The probe is slid over the inset region at probe sizes 0.75/1.0/1.25 of
    the inset width and all 90-degree rotations.  A pattern re-photograph
    matches its source inset far better than any other page's inset, so for
    pattern questions this ranking is usually more decisive than the
    full-page one.

    pages: list of (key, 2-D grayscale page array).
    boxes: dict key -> [x0, y0, x1, y1] (only keys present here are ranked).
    Returns a list of (key, score), sorted by score descending.
    """
    out = []
    for key, gray in pages:
        box = boxes.get(key)
        if not box:
            continue
        x0, y0, x1, y1 = (int(value) for value in box)
        region = gray[y0:y1, x0:x1]
        if region.size == 0:
            continue
        width = int(region.shape[1])
        sizes = tuple(int(max(8, width * scale)) for scale in _INSET_SCALES)
        best = -2.0
        for size in sizes:
            template = to_gray(probe.resize((size, size), Image.BILINEAR))
            for rotation in _ROTATIONS:
                template = np.rot90(template, rotation)
                score = ncc_max(template, region)
                if score > best:
                    best = score
        out.append((key, best))
    out.sort(key=lambda item: -item[1])
    return out


def _product_region(gray):
    """Central 60% crop of a page (where the product photo sits), downscaled
    to at most 512px on a side for speed.  Returns None for an empty page."""
    height, width = gray.shape
    crop_h = int(height * _PRODUCT_CENTER)
    crop_w = int(width * _PRODUCT_CENTER)
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    region = gray[top:top + crop_h, left:left + crop_w]
    if region.size == 0:
        return None
    if max(region.shape) > _PRODUCT_MAX_SIDE:
        factor = _PRODUCT_MAX_SIDE / float(max(region.shape))
        region = np.asarray(
            Image.fromarray(region.astype(np.uint8)).resize(
                (max(8, int(width * factor)), max(8, int(height * factor)))),
            dtype=np.float32)
    return region


def rank_product(probe, pages):
    """Rank pages by matching the probe against each page's central product
    region, at three generic probe sizes and all 90-degree rotations.
    Returns a list of (key, score) for every page, sorted descending."""
    out = []
    for key, gray in pages:
        region = _product_region(gray)
        if region is None:
            continue
        out.append((key, _slide_scores(probe, region, _PRODUCT_SIZES, _ROTATIONS)))
    out.sort(key=lambda item: -item[1])
    return out
