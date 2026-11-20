"""Painting and probe generators for the Minecraft gallery (Pillow only; the GameWorld venv has no numpy).

Same design as postermaze: every painting of an episode is a GxG arrangement over one small palette of
similar colours, generated per seed, so colour names separate nothing and no pool can be memorised."""
from __future__ import annotations

import colorsys
import hashlib
import random

from PIL import Image, ImageEnhance, ImageFilter

PROBE_SIZE = 360


def rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def palette(seed: int, p: int) -> list[tuple[int, int, int]]:
    r = rng("palette", seed)
    h0 = r.random()
    out = []
    for i in range(p):
        h = (h0 + (i - (p - 1) / 2) * 0.035) % 1.0
        l = 0.42 + 0.12 * (i % 2) + r.uniform(-0.03, 0.03)
        s = 0.55 + r.uniform(-0.08, 0.08)
        out.append(tuple(int(255 * x) for x in colorsys.hls_to_rgb(h, l, s)))
    return out


def painting(seed: int, idx: int, g: int, pal: list[tuple[int, int, int]]) -> Image.Image:
    r = rng("poster", seed, idx)
    cells = [i % len(pal) for i in range(g * g)]
    r.shuffle(cells)
    img = Image.new("RGB", (g, g))
    img.putdata([pal[c] for c in cells])
    return img.resize((64, 64), Image.NEAREST)


def probe(img: Image.Image, salt: str) -> Image.Image:
    r = rng("probe", salt)
    im = img.resize((240, 240), Image.NEAREST)
    im = im.rotate(r.uniform(-4, 4), resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
    crop = r.uniform(0.92, 1.0)
    w = int(240 * crop)
    x0, y0 = r.randint(0, 240 - w), r.randint(0, 240 - w)
    im = im.crop((x0, y0, x0 + w, y0 + w)).resize((r.randint(200, 280),) * 2, Image.BILINEAR)
    im = ImageEnhance.Brightness(im).enhance(r.uniform(0.7, 1.0)).filter(ImageFilter.GaussianBlur(r.uniform(0.3, 0.7)))
    canvas = Image.new("RGB", (PROBE_SIZE, PROBE_SIZE), (90, 90, 90))
    canvas.paste(im, ((PROBE_SIZE - im.width) // 2, (PROBE_SIZE - im.height) // 2))
    return canvas
