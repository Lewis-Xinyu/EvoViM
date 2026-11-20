"""DMV-Hard image recall using product photos from DMV-Bench (Apache-2.0).

The agent browses products, then identifies transformed detail or pattern crops.
The primary score is the fraction of probes answered correctly.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get("EVOVIM_DMV_ROOT", HERE.parent / "upstream" / "DMV-Bench")).expanduser().resolve()
REGISTRY = UPSTREAM / "data" / "vismem_diag_v2" / "cue_registry.json"
BASE_DIR = Path(os.environ.get("DMVHARD_BASE_DIR", UPSTREAM / "data" / "vismem_diag_v2" / "images" / "base"))

TASK_RE = re.compile(r"^dmv_hard2/j(\d+)_v(\d+)_p(\d+)$")
ANSWER_RE = re.compile(r"^\s*answer\s+([0-9a-f]{8})\s*$", re.I)
VIEW_SIZE = 512           # product photo as shown to the agent
PATCH_FRACTION = 0.26     # patch side relative to the shown photo
DETAIL_FRACTION = 0.40    # side of the photo region a detail probe is cut from
PROBE_SIZE = 256          # probe image side
SALT = "dmvhard.v2"
# Probe appearance change ("v2 mild" in the offline calibration).
ROT_DEG, SCALE_JITTER, VISIBLE, LIGHT, BLUR = 15.0, 0.15, 0.80, 0.20, 0.5
PALETTE = np.array([[24, 40, 80], [40, 120, 130], [220, 200, 150], [170, 80, 50]], dtype=np.float32)
N_BLOBS, SIGMA, HIGHPASS = 60, (0.025, 0.05), 0.12

RULES = """You are shopping in an online home-furnishing store over a chain of sessions. Each session you browse one product category and compare its products: every step shows one product page (its URL and its photo). Reply "continue" to move to the next page.

Between sessions the user may ask you a question instead of showing a product page. Follow the instruction given in that question."""


def _rng(*parts: object) -> random.Random:
    return random.Random(hashlib.sha256("|".join(map(str, (SALT, *parts))).encode()).hexdigest())


def _blur(f: np.ndarray, sigma: float) -> np.ndarray:
    k = np.fft.fftfreq(f.shape[0])
    g = np.exp(-2 * (np.pi * sigma) ** 2 * (k[:, None] ** 2 + k[None, :] ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(f) * g))


def _field(url_hash: str, size: int = 64) -> np.ndarray:
    r = _rng("field", url_hash)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    f = np.zeros((size, size), np.float32)
    for _ in range(N_BLOBS):
        cx, cy, s = r.random(), r.random(), r.uniform(*SIGMA)
        f += r.uniform(-1.0, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    f = f - _blur(f, HIGHPASS * size)
    lo, hi = np.percentile(f, 2), np.percentile(f, 98)
    return np.clip((f - lo) / max(float(hi - lo), 1e-6), 0, 1)


def _colorize(f: np.ndarray) -> np.ndarray:
    pos = f * (len(PALETTE) - 1)
    lo = np.clip(np.floor(pos).astype(int), 0, len(PALETTE) - 2)
    t = (pos - lo)[..., None]
    return PALETTE[lo] * (1 - t) + PALETTE[lo + 1] * t


def patch_image(url_hash: str, side: int) -> Image.Image:
    rgb = _colorize(_field(url_hash)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").resize((side, side), Image.Resampling.BICUBIC)


def patch_corner(url_hash: str) -> tuple[int, int]:
    side, margin = int(VIEW_SIZE * PATCH_FRACTION), int(VIEW_SIZE * 0.04)
    corner = _rng("corner", url_hash).randrange(4)
    x = margin if corner in (0, 2) else VIEW_SIZE - side - margin
    y = margin if corner in (0, 1) else VIEW_SIZE - side - margin
    return x, y


def base_photo(row: dict) -> Image.Image:
    path = BASE_DIR / row["cat"] / row["style"] / f"{row['prod_idx']:02d}.png"
    with Image.open(path) as image:
        return image.convert("RGB").resize((VIEW_SIZE, VIEW_SIZE), Image.Resampling.LANCZOS)


def product_photo(row: dict) -> Image.Image:
    photo = base_photo(row)
    photo.paste(patch_image(row["url_hash"], int(VIEW_SIZE * PATCH_FRACTION)), patch_corner(row["url_hash"]))
    return photo


def detail_region(row: dict) -> tuple[int, int, int, int]:
    """A product-centred square of the photo that does not overlap the patch corner."""
    r = _rng("detail", row["url_hash"])
    side = int(VIEW_SIZE * DETAIL_FRACTION)
    px, py = patch_corner(row["url_hash"])
    psz = int(VIEW_SIZE * PATCH_FRACTION)
    for _ in range(50):
        x = r.randint(int(VIEW_SIZE * 0.18), int(VIEW_SIZE * 0.82) - side)
        y = r.randint(int(VIEW_SIZE * 0.25), int(VIEW_SIZE * 0.85) - side)
        if x + side <= px or px + psz <= x or y + side <= py or py + psz <= y:
            return x, y, x + side, y + side
    return x, y, x + side, y + side


def _rephotograph(src: Image.Image, r: random.Random) -> Image.Image:
    """Rotate, crop to ~80 %, rescale, relight and blur; place on a grey card."""
    w = src.width
    im = src.rotate(r.uniform(-ROT_DEG, ROT_DEG), resample=Image.Resampling.BICUBIC, fillcolor=(128, 128, 128))
    cw = int(w * VISIBLE)
    x0, y0 = r.randint(0, w - cw), r.randint(0, w - cw)
    inner = int(PROBE_SIZE * 0.86 * (1 + r.uniform(-SCALE_JITTER, SCALE_JITTER)))
    inner = min(inner, PROBE_SIZE - 4)
    im = im.crop((x0, y0, x0 + cw, y0 + cw)).resize((inner, inner), Image.Resampling.BICUBIC)
    im = ImageEnhance.Brightness(im).enhance(r.uniform(1 - LIGHT, 1 + LIGHT))
    im = ImageEnhance.Color(im).enhance(r.uniform(1 - LIGHT, 1 + LIGHT))
    im = im.filter(ImageFilter.GaussianBlur(BLUR))
    card = Image.new("RGB", (PROBE_SIZE, PROBE_SIZE), (128, 128, 128))
    off = (PROBE_SIZE - inner) // 2
    card.paste(im, (off, off))
    return card


def probe_image(row: dict, kind: str, probe_key: str) -> Image.Image:
    r = _rng("probe", row["url_hash"], kind, probe_key)
    if kind == "pattern":
        src = patch_image(row["url_hash"], 256)
    else:
        src = base_photo(row).crop(detail_region(row)).resize((256, 256), Image.Resampling.LANCZOS)
    return _rephotograph(src, r)


class DMVHard2Backend:
    protocol_version = 2

    def __init__(self, game: str, task_id: str, observation_profile: str, horizon: int, variant: str = ""):
        m = TASK_RE.match(task_id)
        if not m:
            raise ValueError(f"task id must look like dmv_hard2/j<J>_v<views>_p<probes>, got {task_id!r}")
        self.sessions, self.views, self.probes_per = (int(x) for x in m.groups())
        if self.sessions < 2 or not 1 <= self.views <= 40 or self.probes_per < 1:
            raise ValueError("need J >= 2, 1 <= views <= 40, probes >= 1")
        self.game, self.task_id, self.observation_profile = game, task_id, observation_profile
        self.horizon = int(horizon)
        if not REGISTRY.is_file():
            raise FileNotFoundError(f"Missing {REGISTRY}. Set EVOVIM_DMV_ROOT to the DMV-Bench checkout.")
        rows = json.loads(REGISTRY.read_text(encoding="utf-8"))["rows"]
        self.by_cat: dict[str, dict[str, list[dict]]] = {}
        for row in rows:
            self.by_cat.setdefault(row["cat"], {}).setdefault(row["style"], []).append(row)
        self._reset_state()

    def _chain(self, seed: int) -> list[dict]:
        cats = sorted(self.by_cat)
        _rng("cats", seed).shuffle(cats)
        steps, seen_by_session, probed = [], [], set()
        for j in range(self.sessions):
            r = _rng("session", seed, j)
            cat = cats[j % len(cats)]
            styles = sorted(self.by_cat[cat])
            r.shuffle(styles)
            styles = styles[:r.randint(3, 4)]
            pool = [row for s in styles for row in sorted(self.by_cat[cat][s], key=lambda x: x["prod_idx"])]
            visited = r.sample(pool, min(self.views, len(pool)))
            for i, row in enumerate(visited):
                steps.append({"kind": "view", "session": j, "index": i, "row": row})
            seen_by_session.append(visited)
            if j >= 1:
                for n in range(self.probes_per):
                    reaches = [d for d in range(1, j + 1)
                               if any(x["url_hash"] not in probed for x in seen_by_session[j - d])]
                    if not reaches:
                        break
                    reach = r.choice(reaches)
                    row = r.choice([x for x in seen_by_session[j - reach] if x["url_hash"] not in probed])
                    probed.add(row["url_hash"])
                    steps.append({"kind": "probe", "session": j, "index": n, "row": row, "reach": reach,
                                  "probe_kind": r.choice(("pattern", "detail"))})
        return steps

    def _reset_state(self) -> None:
        self.seed = None
        self.steps: list[dict] = []
        self.cursor = 0
        self.step_count = 0
        self.results: list[dict] = []
        self.invalid = 0
        self.terminated = False
        self.truncated = False
        self._last_public: dict[str, Any] | None = None

    def _current(self) -> dict | None:
        return self.steps[self.cursor] if self.cursor < len(self.steps) else None

    def _public(self, status: str) -> dict[str, Any]:
        cur = self._current()
        long_term = f"Chain of {self.sessions} shopping sessions; {self.views} product pages per session."
        if cur is None:
            text, image = f"{status} The chain is over.", Image.new("RGB", (PROBE_SIZE, PROBE_SIZE), (128, 128, 128))
        elif cur["kind"] == "view":
            row = cur["row"]
            text = (f"{status} Session {cur['session'] + 1}/{self.sessions}, browsing category '{row['cat']}'. "
                    f"Product page {cur['index'] + 1}/{self.views}: URL /product/{row['url_hash']}. "
                    "Reply continue for the next page.")
            image = product_photo(row)
        else:
            # The text never says which kind of detail the image shows.
            text = (f"{status} Question {cur['index'] + 1} after session {cur['session'] + 1}: the image shows "
                    "something from a product page you saw in an earlier session, re-photographed (it may be rotated, "
                    "rescaled, partly visible and differently lit). Take me back to that exact product: reply "
                    "answer <id>, where <id> is the 8-character hexadecimal id from that product's URL /product/<id>. "
                    "One answer per question; a wrong or malformed answer scores nothing.")
            image = probe_image(cur["row"], cur["probe_kind"], f"{self.seed}:{cur['session']}:{cur['index']}")
        self._last_public = {"text": {"long_term_context": long_term, "short_term_context": text},
                             "frame_id": f"f{self.step_count:05d}", "image": image}
        return dict(self._last_public)

    def action_spec(self) -> dict[str, Any]:
        return {"kind": "free_text", "max_length": 32, "examples": ["continue"],
                "description": 'Reply "continue" on a product page; when the user asks a question, reply as that question instructs.'}

    def metric_description(self) -> dict[str, Any]:
        return {"name": "DMV-Hard v2 recall accuracy: correct probes / probes", "aggregation": "final", "range": [0.0, 1.0]}

    def instruction(self) -> str:
        return RULES

    def describe(self) -> dict[str, Any]:
        return {"protocol_version": self.protocol_version, "game": self.game, "task_id": self.task_id,
                "actual_native_env_id": f"dmvhard2/{self.task_id}",
                "observation_profile": self.observation_profile,
                "horizon": self.horizon, "action_spec": self.action_spec(), "metric": self.metric_description()}

    def reset(self, seed: int) -> dict[str, Any]:
        self._reset_state()
        self.seed = int(seed)
        self.steps = self._chain(self.seed)
        if self.horizon < len(self.steps):
            raise ValueError(f"horizon {self.horizon} < chain length {len(self.steps)}")
        return self._public("Session 1 starts.")

    def observe(self) -> dict[str, Any]:
        if self._last_public is None:
            raise RuntimeError("backend has not been reset")
        return dict(self._last_public)

    def step(self, action: str) -> dict[str, Any]:
        if not self.steps:
            raise RuntimeError("backend has not been reset")
        if self.terminated or self.truncated:
            raise RuntimeError("episode is over")
        cur = self._current()
        self.step_count += 1
        text = str(action).strip()
        if cur["kind"] == "view":
            if text.lower() != "continue":
                self.invalid += 1
            status = "Next page." if text.lower() == "continue" else f"'{text}' is not an action on a product page; moved to the next page."
        else:
            m = ANSWER_RE.match(text)
            given = m.group(1).lower() if m else None
            if given is None:
                self.invalid += 1
            self.results.append({"session": cur["session"], "reach": cur["reach"], "kind": cur["probe_kind"],
                                 "correct": given == cur["row"]["url_hash"], "answered": given is not None})
            status = "Answer recorded." if given is not None else f"'{text}' is not a valid answer; recorded as wrong."
        self.cursor += 1
        nxt = self._current()
        if nxt is None:
            self.terminated = True
        elif self.step_count >= self.horizon:
            self.truncated = True
        elif nxt["kind"] == "view" and nxt["index"] == 0:
            status += f" Session {nxt['session'] + 1} starts."
        observation = self._public(status)
        return {"observation": observation, "reward": 0.0, "terminated": self.terminated,
                "truncated": self.truncated, "metrics": self.score(),
                "action_mapping": {"requested": text, "native": text, "fallback": False}}

    def primary_score(self) -> float:
        n = sum(s["kind"] == "probe" for s in self.steps)
        return sum(r["correct"] for r in self.results) / n if n else 0.0

    def score(self) -> dict[str, Any]:
        def acc(key):
            out: dict[str, list[bool]] = {}
            for r in self.results:
                out.setdefault(str(r[key]), []).append(r["correct"])
            return {k: sum(v) / len(v) for k, v in sorted(out.items())}
        return {"primary": self.primary_score(),
                "success": (self.primary_score() == 1.0) if (self.terminated or self.truncated) else None,
                "seed": self.seed, "probes_total": sum(s["kind"] == "probe" for s in self.steps),
                "probes_answered": len(self.results), "probes_correct": sum(r["correct"] for r in self.results),
                "accuracy_by_reach": acc("reach"), "accuracy_by_kind": acc("kind"),
                "invalid_actions": self.invalid, "steps": self.step_count,
                "terminated": self.terminated, "truncated": self.truncated}

    def close(self) -> None:
        self.steps = []
