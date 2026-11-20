"""Question-painting extraction and multi-scale template matching.

Stdlib, numpy, scipy and Pillow only.  The query is the painting in a question
image; the references are saved panel crops from walk frames.  Scores are
normalised cross-correlation (NCC) in [-1, 1]: invariant to brightness and
contrast, tolerant of small rotation (the query is de-rotated first) and of
scale (the template is searched over SCALES).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

STRONG_SCORE = 0.45  # provisional: dev units gave lures <= 0.32 and seen panels >= 0.51 (2 units only)
SCALES = np.linspace(0.15, 2.0, 24)
ANGLES = np.arange(-12.0, 12.5, 1.0)
MIN_TEMPLATE_SIDE = 12
VARIANCE_FLOOR = 25.0  # added to local variance so flat windows cannot inflate NCC
BACKGROUND_TOL = 24.0
INSET = 4
MAX_REFERENCE_SIDE = 640


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def painting_patch(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (patch, angle) for the largest non-background component, de-rotated and inset."""
    border = np.concatenate([img[:4].reshape(-1, 3), img[-4:].reshape(-1, 3),
                             img[:, :4].reshape(-1, 3), img[:, -4:].reshape(-1, 3)])
    background = np.median(border, axis=0)
    mask = np.abs(img - background).max(axis=2) > BACKGROUND_TOL
    mask = ndimage.binary_opening(mask, iterations=2)
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("no painting found in the query image")
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    keep = labels == (int(np.argmax(sizes)) + 1)
    best = None
    for angle in ANGLES:
        rotated_mask = ndimage.rotate(keep.astype(np.uint8), angle, reshape=False, order=0)
        ys, xs = np.nonzero(rotated_mask)
        if ys.size == 0:
            continue
        area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
        if best is None or area < best[0]:
            best = (area, float(angle))
    if best is None:
        raise ValueError("query painting could not be aligned")
    angle = best[1]
    rotated = np.stack([ndimage.rotate(img[..., c], angle, reshape=False, order=1) for c in range(3)], axis=2)
    rmask = ndimage.rotate(keep.astype(np.uint8), angle, reshape=False, order=0) > 0
    ys, xs = np.nonzero(rmask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = rotated[y0 + INSET:y1 - INSET, x0 + INSET:x1 - INSET]
    if patch.shape[0] < 16 or patch.shape[1] < 16:
        raise ValueError("query painting is too small")
    return patch, angle


def _normalize(template: np.ndarray) -> np.ndarray:
    template = template - template.mean(axis=(0, 1), keepdims=True)
    return template / (np.sqrt((template ** 2).mean(axis=(0, 1), keepdims=True)) + 1e-6)


def ncc_map(frame: np.ndarray, template: np.ndarray) -> np.ndarray | None:
    """NCC of a 3-channel template over every valid placement in a 3-channel frame."""
    th, tw, _ = template.shape
    fh, fw, _ = frame.shape
    if th >= fh or tw >= fw:
        return None
    t = _normalize(template)
    score = np.zeros((fh - th + 1, fw - tw + 1), dtype=np.float64)
    for c in range(3):
        score += fftconvolve(frame[..., c].astype(np.float64), t[::-1, ::-1, c], mode="valid")
    score /= 3.0
    ones = np.ones((th, tw), dtype=np.float64)
    spread = np.zeros(score.shape, dtype=np.float64)
    for c in range(3):
        f = frame[..., c].astype(np.float64)
        mean = fftconvolve(f, ones, mode="valid") / (th * tw)
        mean2 = fftconvolve(f * f, ones, mode="valid") / (th * tw)
        variance = np.maximum(mean2 - mean * mean, 0.0)
        spread += np.sqrt(variance + VARIANCE_FLOOR)
    spread /= 3.0
    return score / ((th * tw) * spread + 1e-6)


def _resize_channels(patch: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.stack([np.asarray(Image.fromarray(patch[..., c].astype(np.uint8)).resize(
        (width, height), Image.BILINEAR), dtype=np.float64) for c in range(3)], axis=2)


def best_match(reference: np.ndarray, patch: np.ndarray) -> tuple[float, float | None, tuple[int, int] | None]:
    best = (-1.0, None, None)
    for scale in SCALES:
        width = int(round(patch.shape[1] * scale))
        height = int(round(patch.shape[0] * scale))
        if width < MIN_TEMPLATE_SIDE or height < MIN_TEMPLATE_SIDE:
            continue
        template = _resize_channels(patch, width, height)
        surface = ncc_map(reference, template)
        if surface is None:
            continue
        index = np.unravel_index(int(np.argmax(surface)), surface.shape)
        value = float(surface[index])
        if value > best[0]:
            best = (value, float(scale), (int(index[0]), int(index[1])))
    return best


def reference_array(image: Image.Image) -> np.ndarray:
    """Reference crops are capped in size so the scale search stays bounded."""
    image = image.convert("RGB")
    if max(image.size) > MAX_REFERENCE_SIDE:
        image = image.copy()
        image.thumbnail((MAX_REFERENCE_SIDE, MAX_REFERENCE_SIDE), Image.Resampling.BILINEAR)
    return to_array(image)


def score_references(query: Image.Image, references: list[Image.Image]) -> dict:
    """Score one question image against many reference crops.

    Returns the de-rotation angle, the query patch shape and one (score, scale, loc)
    tuple per reference, in input order.  Raises ValueError if the query is unreadable.
    """
    patch, angle = painting_patch(to_array(query))
    scores = []
    for reference in references:
        scores.append(best_match(reference_array(reference), patch))
    return {"angle": angle, "patch_shape": [int(patch.shape[0]), int(patch.shape[1])], "scores": scores}
