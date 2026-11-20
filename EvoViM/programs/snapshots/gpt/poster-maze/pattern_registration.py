"""Optional, model-managed visual records and numerical pattern retrieval.

Only public pixels and model-selected records are used. No recognition model,
game palette, renderer calibration, episode constants, or hidden map is needed.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, signal


def content_regions(image):
    """Propose colored content bounds using an image-adaptive Otsu split.

    These are proposals, not object identities. The model may override the crop.
    Saturation is independent of which hue a poster happens to have.
    """
    thumb = image.copy().convert('RGB')
    thumb.thumbnail((320, 320))
    pixels = np.asarray(thumb, dtype=float)
    saturation = pixels.max(2) - pixels.min(2)
    hist, _ = np.histogram(saturation, bins=256, range=(0, 256))
    prob = hist / max(hist.sum(), 1)
    weight = np.cumsum(prob)
    mean = np.cumsum(prob * np.arange(256))
    variance = (mean[-1] * weight - mean) ** 2 / np.maximum(weight * (1-weight), 1e-12)
    threshold = int(np.argmax(variance))
    mask = saturation > threshold
    # Closing radius scales with resolution, not a measured maze geometry.
    radius = max(1, min(mask.shape)//100)
    mask = ndimage.binary_closing(mask, iterations=radius, border_value=1)
    labels, count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    areas = np.bincount(labels.ravel())
    regions = []
    for label in sorted(range(1, count+1), key=lambda i: int(areas[i]), reverse=True)[:4]:
        sy, sx = objects[label-1]
        if sx.stop-sx.start < 4 or sy.stop-sy.start < 4:
            continue
        region = [sx.start/thumb.width, sy.start/thumb.height,
                  sx.stop/thumb.width, sy.stop/thumb.height]
        regions.append(region)
    return regions or [[0.0, 0.0, 1.0, 1.0]]


def pixel_box(image, region):
    if len(region) != 4 or not all(math.isfinite(float(x)) for x in region):
        raise ValueError('region must be [left, top, right, bottom], fractions of image size')
    x0, y0, x1, y1 = [float(x) for x in region]
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('crop fractions must be ordered within [0,1]')
    box = [int(x0*image.width), int(y0*image.height),
           int(math.ceil(x1*image.width)), int(math.ceil(y1*image.height))]
    if box[2]-box[0] < 2 or box[3]-box[1] < 2:
        raise ValueError('crop is too small')
    return box


def descriptor(image, region, size=40):
    crop = image.crop(pixel_box(image, region)).convert('RGB')
    values = np.asarray(crop.resize((size, size), Image.Resampling.BILINEAR), dtype=float)/255
    # Independent channel centering removes brightness and white balance offsets.
    mean = values.mean((0, 1), keepdims=True)
    std = values.std((0, 1), keepdims=True)
    return (values-mean)/np.maximum(std, 1/255)


def _correlate(a, b):
    return signal.fftconvolve(a, b[::-1, ::-1], mode='full')


def _overlap_score(source, template, mask):
    """Masked local Pearson correlation with partial-overlap support."""
    ones = np.ones(source.shape[:2])
    count = _correlate(ones, mask)
    safe = np.maximum(count, 1)
    cross = np.zeros_like(count)
    var_a = np.zeros_like(count)
    var_b = np.zeros_like(count)
    for c in range(3):
        a, b = source[:, :, c], template[:, :, c]*mask
        sum_a = _correlate(a, mask)
        sum_b = _correlate(ones, b)
        cross += _correlate(a, b) - sum_a*sum_b/safe
        var_a += _correlate(a*a, mask) - sum_a*sum_a/safe
        var_b += _correlate(ones, b*b*mask) - sum_b*sum_b/safe
    scores = cross / np.sqrt(np.maximum(var_a*var_b, 1e-12))
    # Require most of the smaller view and substantial evidence in the larger.
    required = max(0.70*min(mask.sum(), ones.sum()), 0.25*max(mask.sum(), ones.sum()))
    valid = (count >= required) & (var_a > 1e-5) & (var_b > 1e-5)
    scores[~valid] = -1
    index = np.unravel_index(np.argmax(scores), scores.shape)
    return float(np.clip(scores[index], -1, 1)), float(count[index]/min(mask.sum(), ones.sum()))


def compare(source, query):
    """Search rotation, scale and translation; return similarity, never an action.

    The fixed search grid is a generic image-registration grid. It is neither
    fitted to evidence nor dependent on a maze's dimensions or palette.
    """
    best = {'similarity': -1.0, 'rotation': 0, 'scale_x': 1.0, 'scale_y':1.0, 'overlap': 0.0}
    seeds = []
    for quarter in range(4):
        upright = np.rot90(query, quarter)
        for angle in (-30, -15, 0, 15, 30, 45):
            rotated = ndimage.rotate(upright, angle, axes=(1, 0), reshape=True, order=1)
            mask = ndimage.rotate(np.ones(upright.shape[:2]), angle, reshape=True, order=1)
            mask = (mask > 0.99).astype(float)
            for scale in (0.5, 2**-0.5, 1.0, 2**0.5, 2.0):
                template = ndimage.zoom(rotated, (scale, scale, 1), order=1)
                support = ndimage.zoom(mask, scale, order=0)
                score, overlap = _overlap_score(source, template, support)
                seeds.append((score,quarter*90+angle,scale))
                if score > best['similarity']:
                    best = {'similarity': score, 'rotation': quarter*90+angle,
                            'scale_x': scale, 'scale_y':scale, 'overlap': overlap}
    # A close wall may be clipped differently on each axis. Refine the best
    # coarse registrations with independent axis scales and finer rotations.
    for _, angle, scale in sorted(seeds, reverse=True)[:3]:
        for offset in (-5, 0, 5):
            rotated = ndimage.rotate(query, angle+offset, axes=(1,0), reshape=True, order=1)
            mask = (ndimage.rotate(np.ones(query.shape[:2]), angle+offset,
                                   reshape=True, order=1)>0.99).astype(float)
            for sx in (scale*2**-0.5, scale, scale*2**0.5):
                for sy in (scale*2**-0.5, scale, scale*2**0.5):
                    template = ndimage.zoom(rotated,(sy,sx,1),order=1)
                    support = ndimage.zoom(mask,(sy,sx),order=0)
                    score, overlap = _overlap_score(source,template,support)
                    if score > best['similarity']:
                        best = {'similarity':score,'rotation':angle+offset,
                                'scale_x':sx,'scale_y':sy,'overlap':overlap}
    return best

