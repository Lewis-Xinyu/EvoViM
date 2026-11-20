"""Code-maintained product-page catalog and pixel matcher for visual line M.

The catalog is derived only from the public observation text plus the current
image the host already hands the agent.  It records, for every product page,
the 8-hex product id from the page URL, the session, page number and category,
the host frame id of that page's photo, and a small grayscale thumbnail.

What the code does automatically: parse and index what was already shown,
compute image-similarity scores on request, and validate ids.
What the LLM keeps doing: deciding when to match, which candidates to inspect
visually, what appearance notes to write, and which id to answer with.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import fft as sfft

PRODUCT_RE = re.compile(r"/product/([0-9a-fA-F]{8})\b")
SESSION_RE = re.compile(r"Session\s+(\d+)\s*/\s*(\d+)")
PAGE_RE = re.compile(r"Product page\s+(\d+)\s*/\s*(\d+)")
CATEGORY_RE = re.compile(r"category\s+'([^']+)'")
QUESTION_RE = re.compile(r"Question\s+(\d+)\s+after session\s+(\d+)")

THUMB_SIDE = 128
FILL = 128.0
# Geometric scale ladder over the probe's content box.  Kept generic: it spans
# small partly-visible crops to nearly full-page views at the thumbnail size.
SCALES = tuple(round(0.07 * (1.15 ** i), 4) for i in range(18))
MIN_STD = 6.0
ROTATIONS = (90, 180, 270, 15, -15, 30, -30, 45, -45)
ROTATION_PENALTY = 0.06
_CACHE: dict = {}


def _load_thumb(path):
    """Small in-process cache of decoded thumbnails; never part of agent state."""
    key = str(path)
    try:
        stat = os.stat(key)
    except OSError:
        return None
    stamp = (stat.st_mtime_ns, stat.st_size)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        with Image.open(key) as image:
            array = np.asarray(image.convert("L"), dtype=np.float64)
    except Exception:
        return None
    if len(_CACHE) > 512:
        _CACHE.clear()
    _CACHE[key] = (stamp, array)
    return array


def _resize(array, width, height):
    if width < 1 or height < 1:
        return None
    return np.asarray(
        Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).resize((width, height), Image.BILINEAR),
        dtype=np.float64,
    )


def thumbnail(image, side=THUMB_SIDE):
    """Downscale to a bounded grayscale thumbnail, preserving aspect ratio."""
    gray = image.convert("L")
    width, height = gray.size
    scale = side / max(width, height)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return gray.resize(size, Image.BILINEAR)


def content_crop(probe, fill=FILL, threshold=5.0):
    """Trim the uniform grey border the probe canvas uses; no-op if there is none."""
    mask = np.abs(probe - fill) > threshold
    if not mask.any():
        return probe
    ys, xs = np.where(mask)
    if ys.max() - ys.min() < 8 or xs.max() - xs.min() < 8:
        return probe
    return probe[max(0, ys.min() - 1):ys.max() + 2, max(0, xs.min() - 1):xs.max() + 2]


def content_box(probe, fill=FILL, threshold=5.0):
    """Box (x0, y0, x1, y1) in probe pixels of exactly the region content_crop returns."""
    height, width = probe.shape
    mask = np.abs(probe - fill) > threshold
    if not mask.any():
        return (0, 0, width, height)
    ys, xs = np.where(mask)
    if ys.max() - ys.min() < 8 or xs.max() - xs.min() < 8:
        return (0, 0, width, height)
    return (int(max(0, xs.min() - 1)), int(max(0, ys.min() - 1)),
            int(min(width, xs.max() + 2)), int(min(height, ys.max() + 2)))


def _rotate(array, angle):
    if angle % 360 == 0:
        return array
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    return np.asarray(
        image.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=int(FILL)), dtype=np.float64)


class _Page:
    """FFT and integral images for one catalog thumbnail."""
    __slots__ = ("page", "fft", "ii", "ii2", "shape")

    def __init__(self, array):
        self.page = array
        self.shape = array.shape
        self.fft = sfft.rfft2(array)
        height, width = array.shape
        self.ii = np.zeros((height + 1, width + 1), np.float64)
        self.ii[1:, 1:] = array.cumsum(0).cumsum(1)
        self.ii2 = np.zeros((height + 1, width + 1), np.float64)
        self.ii2[1:, 1:] = (array * array).cumsum(0).cumsum(1)

    def _values(self, template, min_std=MIN_STD):
        """NCC map of `template` over every placement on this page, or None if unusable."""
        height, width = self.shape
        th, tw = template.shape
        if th > height or tw > width or th < 5 or tw < 5:
            return None
        centered = template - template.mean()
        norm = float(np.sqrt((centered * centered).sum()))
        if norm < 1e-9:
            return None
        padded = np.zeros((height, width), np.float64)
        padded[:th, :tw] = centered
        corr = sfft.irfft2(self.fft * np.conj(sfft.rfft2(padded)), s=(height, width))
        corr = corr[:height - th + 1, :width - tw + 1]
        s1 = self.ii[th:, tw:] - self.ii[:-th, tw:] - self.ii[th:, :-tw] + self.ii[:-th, :-tw]
        s2 = self.ii2[th:, tw:] - self.ii2[:-th, tw:] - self.ii2[th:, :-tw] + self.ii2[:-th, :-tw]
        count = th * tw
        var = s2 - s1 * s1 / count
        good = var / count > min_std * min_std
        if not good.any():
            return None
        safe = np.maximum(var, 1e-6)
        values = np.full(var.shape, -2.0)
        values[good] = (corr / np.sqrt(safe * norm * norm))[good]
        return values

    def score(self, template, min_std=MIN_STD):
        values = self._values(template, min_std)
        return -2.0 if values is None else float(values.max())

    def locate(self, template, min_std=MIN_STD):
        """(score, y, x) of the best placement of `template` on this page, or None."""
        values = self._values(template, min_std)
        if values is None:
            return None
        y, x = np.unravel_index(int(np.argmax(values)), values.shape)
        return float(values[y, x]), int(y), int(x)


def _best_scales(prepared, content, scales=SCALES):
    best = -2.0
    ch, cw = content.shape
    for scale in scales:
        th, tw = int(round(ch * scale)), int(round(cw * scale))
        if th < 5 or tw < 5 or th > prepared.shape[0] or tw > prepared.shape[1]:
            continue
        if (tw, th) == (cw, ch):
            template = content
        else:
            template = _resize(content, tw, th)
            if template is None:
                continue
        best = max(best, prepared.score(template))
    return best


def best_window(prepared, content, scales=SCALES):
    """Where the best-scoring placement sits on the page: (score, y, x, th, tw) or None.

    The score equals _best_scales(prepared, content); (y, x) is the top-left corner
    and (th, tw) the size of that placement, in thumbnail pixels.
    """
    best = None
    ch, cw = content.shape
    for scale in scales:
        th, tw = int(round(ch * scale)), int(round(cw * scale))
        if th < 5 or tw < 5 or th > prepared.shape[0] or tw > prepared.shape[1]:
            continue
        if (tw, th) == (cw, ch):
            template = content
        else:
            template = _resize(content, tw, th)
            if template is None:
                continue
        found = prepared.locate(template)
        if found is not None and (best is None or found[0] > best[0]):
            best = (found[0], found[1], found[2], th, tw)
    return best


def thumb_box_to_frame(box, thumb_shape, frame_size):
    """Map a thumbnail-pixel box (x0, y0, x1, y1) onto the full-resolution frame, clipped."""
    thumb_h, thumb_w = thumb_shape
    frame_w, frame_h = frame_size
    x0, y0, x1, y1 = box
    fx0 = int(np.floor(x0 * frame_w / thumb_w))
    fy0 = int(np.floor(y0 * frame_h / thumb_h))
    fx1 = int(np.ceil(x1 * frame_w / thumb_w))
    fy1 = int(np.ceil(y1 * frame_h / thumb_h))
    fx0, fy0 = max(0, fx0), max(0, fy0)
    fx1, fy1 = min(frame_w, fx1), min(frame_h, fy1)
    if fx1 <= fx0 or fy1 <= fy0:
        return None
    return (fx0, fy0, fx1, fy1)


class PageCatalog:
    """Persistent index of product pages and their thumbnails."""

    def __init__(self, root):
        self.root = Path(root) / "catalog"
        self.thumbs = self.root / "thumbs"
        self.thumbs.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "pages.json"
        self.data = {"schema_version": 1, "pages": {}, "questions": {}, "answers": []}
        self.last_step = -1
        self.reload()

    # ---------- persistence ----------
    def reload(self):
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("pages"), dict):
                    loaded.setdefault("questions", {})
                    loaded.setdefault("answers", [])
                    loaded.setdefault("schema_version", 1)
                    self.data = loaded
            except Exception:
                pass

    def _persist(self):
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def reset(self):
        """Drop everything derived from the previous episode.

        Called when the agent's step counter is reset: page step numbers from
        an old episode would no longer refer to the same observations.
        """
        self.data = {"schema_version": 1, "pages": {}, "questions": {}, "answers": []}
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass
        try:
            for thumb in self.thumbs.glob("*.png"):
                thumb.unlink()
        except OSError:
            pass
        self.last_step = -1

    def _thumb_path(self, product_id):
        return self.thumbs / "{}.png".format(product_id)

    # ---------- observation ingestion ----------
    def window(self, product_id, content):
        """Matched window of one stored page for the probe `content`, in frame pixels.

        Returns a dict with score, frame_id and box (x0, y0, x1, y1) on the stored
        full-resolution frame, or a dict with a reason and no box when the page cannot
        be cut out (no stored frame or thumbnail, or no usable placement).
        """
        entry = self.get(product_id)
        if entry is None:
            return {"product_id": product_id, "reason": "not indexed"}
        if not entry.get("frame_id") or not entry.get("frame_size"):
            return {"product_id": product_id, "reason": "no stored frame size"}
        array = _load_thumb(self._thumb_path(product_id))
        if array is None:
            return {"product_id": product_id, "reason": "no stored thumbnail"}
        found = best_window(_Page(array), content)
        if found is None:
            return {"product_id": product_id, "reason": "no usable placement"}
        score, y, x, th, tw = found
        box = thumb_box_to_frame((x, y, x + tw, y + th), array.shape, entry["frame_size"])
        if box is None:
            return {"product_id": product_id, "reason": "window outside the frame"}
        return {"product_id": product_id, "score": score, "frame_id": entry["frame_id"], "box": box}

    def observe(self, step, text, image=None, frame_id=None):
        """Index what the current observation shows. Returns a short status line."""
        text = str(text or "")
        question = QUESTION_RE.search(text)
        if question:
            number, after = int(question.group(1)), int(question.group(2))
            self.data["questions"][str(step)] = {"number": number, "after_session": after}
            self._persist()
            return "recall question {} (after session {}) at step {}; eligible pool: sessions 1-{} ({} pages)".format(
                number, after, step, max(0, after - 1), self.pool(after_session=after, before_step=step).__len__())

        product = PRODUCT_RE.search(text)
        if not product:
            return "catalog holds {} pages".format(len(self.data["pages"]))

        product_id = product.group(1).lower()
        session_match = SESSION_RE.search(text)
        page_match = PAGE_RE.search(text)
        category_match = CATEGORY_RE.search(text)
        session = int(session_match.group(1)) if session_match else None
        page_no = int(page_match.group(1)) if page_match else None
        page_total = int(page_match.group(2)) if page_match else None
        category = category_match.group(1) if category_match else None

        entry = self.data["pages"].get(product_id)
        if entry is None:
            entry = {"product_id": product_id, "first_step": int(step), "last_step": int(step),
                     "session": session, "page_no": page_no, "page_total": page_total,
                     "category": category, "frame_id": None,
                     "note": "", "note_step": None, "confidence": None}
            self.data["pages"][product_id] = entry
        else:
            entry["last_step"] = int(step)
            if entry.get("session") is None:
                entry["session"] = session
            if entry.get("page_no") is None:
                entry["page_no"] = page_no
            if entry.get("category") is None:
                entry["category"] = category
        if frame_id and not entry.get("frame_id"):
            entry["frame_id"] = str(frame_id)
        if image is not None and not entry.get("frame_size"):
            entry["frame_size"] = [int(image.size[0]), int(image.size[1])]
        if image is not None and not self._thumb_path(product_id).exists():
            try:
                store = thumbnail(image)
                tmp = self._thumb_path(product_id).with_suffix(".png.tmp")
                store.save(tmp, format="PNG")
                tmp.replace(self._thumb_path(product_id))
            except Exception:
                pass
        self._persist()
        return "page {} ({}) indexed at step {}: session {}, page {}/{}".format(
            product_id, category or "?", step, session or "?", page_no or "?", page_total or "?")

    # ---------- queries ----------
    def entries(self, session=None, category=None, product_id=None, earlier_sessions_only=False,
                before_step=None, after_session=None):
        rows = []
        for entry in self.data["pages"].values():
            if product_id and entry["product_id"] != product_id.lower():
                continue
            if session is not None and entry.get("session") != session:
                continue
            if category and (entry.get("category") or "").casefold() != str(category).casefold():
                continue
            if before_step is not None and entry.get("first_step", 0) >= before_step:
                continue
            if earlier_sessions_only and entry.get("session") is None:
                continue
            rows.append(entry)
        rows.sort(key=lambda item: (item.get("first_step") or 0))
        return rows

    def pool(self, after_session=None, before_step=None):
        """Pages from sessions strictly earlier than the question's session."""
        limit = after_session - 1 if after_session else None
        rows = []
        for entry in self.data["pages"].values():
            if before_step is not None and entry.get("first_step", 0) >= before_step:
                continue
            if limit is not None and (entry.get("session") is None or entry["session"] > limit):
                continue
            rows.append(entry)
        rows.sort(key=lambda item: (item.get("first_step") or 0))
        return rows

    def question_for(self, step):
        return self.data["questions"].get(str(step))

    def get(self, product_id):
        return self.data["pages"].get(str(product_id or "").lower())

    def last_page(self):
        rows = self.entries()
        return rows[-1] if rows else None

    # ---------- notes ----------
    def note(self, product_id, note, step, confidence=None):
        entry = self.get(product_id)
        if entry is None:
            return None
        entry["note"] = str(note)[:600]
        entry["note_step"] = int(step)
        if confidence:
            entry["confidence"] = str(confidence)
        self._persist()
        return entry

    # ---------- answer ledger ----------
    # The ledger only records what the agent itself committed: which id it answered
    # at which step, and for which recall question. It never judges correctness; the
    # environment does not report that in the observation text.
    def record_answer(self, step, product_id):
        """Remember a committed answer. Idempotent per step."""
        answers = self.data.setdefault("answers", [])
        if any(int(a.get("step", -1)) == int(step) for a in answers):
            return None
        question = self.question_for(int(step))
        record = {"step": int(step), "product_id": str(product_id or "").lower(),
                  "question": question["number"] if question else None}
        answers.append(record)
        self._persist()
        return record

    def answered_index(self):
        """Map product id -> first committed answer that named it."""
        index = {}
        for record in self.data.get("answers", []):
            index.setdefault(record.get("product_id"), record)
        return index

    @staticmethod
    def _answer_tag(record):
        if record is None:
            return ""
        return " [already answered{} at step {}]".format(
            "" if record.get("question") is None else " for question {}".format(record["question"]),
            record.get("step"))

    # ---------- rendering ----------
    @staticmethod
    def _line(entry, with_note=True):
        pieces = ["step {}".format(entry.get("first_step")),
                  entry["product_id"],
                  "s{}".format(entry.get("session") if entry.get("session") is not None else "?"),
                  "p{}/{}".format(entry.get("page_no") or "?", entry.get("page_total") or "?")]
        if entry.get("category"):
            pieces.append(entry["category"])
        line = " | ".join(pieces)
        note = (entry.get("note") or "").strip()
        if with_note and note:
            line += " | note: " + note.replace("\n", " ")
        return line

    def render(self, session=None, category=None, product_id=None, limit=40):
        rows = self.entries(session=session, category=category, product_id=product_id)
        if not rows:
            return "PAGE CATALOG: no matching pages yet ({} indexed).".format(len(self.data["pages"]))
        shown = rows[-limit:]
        header = "PAGE CATALOG: {} pages indexed ({} session(s)); showing {}.".format(
            len(self.data["pages"]), len({r.get("session") for r in rows if r.get("session")}), len(shown))
        return "\n".join([header] + [self._line(r) for r in shown])

    def context(self, step, limit=5):
        """Bounded per-call context: what the catalog knows about the current step."""
        question = self.question_for(step)
        total = len(self.data["pages"])
        ledger = self._ledger_line()
        if question:
            pool = self.pool(after_session=question["after_session"], before_step=step)
            sessions = sorted({r.get("session") for r in pool if r.get("session")})
            text = ("RECALL QUESTION {} after session {}: the attached image is a re-photograph of a product "
                    "page from an earlier session. Eligible pool: {} pages from sessions {} (catalog has {} "
                    "pages in total). Use MATCH_CURRENT_IMAGE, then VIEW_PAGES on the top candidates.").format(
                question["number"], question["after_session"], len(pool),
                "-".join(str(s) for s in sessions) or "none", total)
            return text + ("\n" + ledger if ledger else "")
        recent = self.entries()[-limit:]
        if not recent:
            return "PAGE CATALOG: empty so far." + ("\n" + ledger if ledger else "")
        lines = ["PAGE CATALOG: {} pages indexed; most recent:".format(total)]
        lines += ["  " + self._line(r) for r in recent]
        if ledger:
            lines.append(ledger)
        return "\n".join(lines)

    def _ledger_line(self, limit=8):
        """Committed answers so far, newest last; shown to the model, never acted on by code."""
        answers = self.data.get("answers", [])
        if not answers:
            return ""
        parts = []
        for record in answers[-limit:]:
            parts.append("{} (question {}, step {})".format(
                record.get("product_id"), record.get("question") or "?", record.get("step")))
        return "ANSWERS ALREADY GIVEN in this episode: " + "; ".join(parts) + "."

    # ---------- matching ----------
    def match(self, probe_image, after_session=None, before_step=None, sessions=None,
              rotation=False, top_k=10, min_score=None):
        """Rank catalog pages by normalized cross-correlation with `probe_image`."""
        if probe_image is None:
            raise ValueError("no current image is available to match")
        pool = self.pool(after_session=after_session, before_step=before_step)
        if sessions:
            wanted = set(sessions)
            pool = [r for r in pool if r.get("session") in wanted]
        if not pool:
            raise ValueError("the eligible page pool is empty; check the session filter")

        probe = np.asarray(probe_image.convert("L"), dtype=np.float64)
        content = content_crop(probe)
        if content.shape[0] < 12 or content.shape[1] < 12:
            raise ValueError("the current image has no usable content to match")

        results = []
        prepared = {}
        for entry in pool:
            path = self._thumb_path(entry["product_id"])
            array = _load_thumb(path)
            if array is None:
                continue
            if entry["product_id"] not in prepared:
                prepared[entry["product_id"]] = _Page(array)
            score = _best_scales(prepared[entry["product_id"]], content)
            results.append({"entry": entry, "score": score, "angle": 0})
        if not results:
            raise ValueError("catalog pages have no stored thumbnails yet")

        results.sort(key=lambda item: -item["score"])

        if rotation:
            for item in results[:max(1, int(top_k))]:
                base = item["score"]
                prepared_page = prepared[item["entry"]["product_id"]]
                best, best_angle = base, 0
                for angle in ROTATIONS:
                    rotated = content_crop(_rotate(probe, angle))
                    if rotated.shape[0] < 12 or rotated.shape[1] < 12:
                        continue
                    candidate = _best_scales(prepared_page, rotated) - ROTATION_PENALTY
                    if candidate > best:
                        best, best_angle = candidate, angle
                item["score"], item["angle"] = best, best_angle
            results.sort(key=lambda item: -item["score"])

        if min_score is not None:
            results = [item for item in results if item["score"] >= float(min_score)]

        per_session = {}
        for item in results:
            key = item["entry"].get("session")
            if key is not None and key not in per_session:
                per_session[key] = item
        return results, per_session

    def render_matches(self, results, per_session, total_pool, question=None, limit=10):
        if not results:
            return "IMAGE MATCH: no candidate scored above the requested minimum."
        premise = ""
        if question:
            premise = "Question {} (probe from session {}); ".format(question["number"], question["after_session"])
        header = ("{}matched {} eligible page(s) by normalized cross-correlation of the current image against "
                  "each stored page (higher is more similar; 1.0 means near-identical content after scaling). "
                  "Top {}:").format(premise, total_pool, min(limit, len(results)))
        lines = [header]
        answered = self.answered_index()
        for rank, item in enumerate(results[:limit], 1):
            entry = item["entry"]
            line = "{:2d}. {} score={:.3f} | {} | {:.0f}deg{}".format(
                rank, entry["product_id"], item["score"], self._line(entry), item.get("angle") or 0,
                self._answer_tag(answered.get(entry["product_id"])))
            lines.append(line)
        if per_session:
            lines.append("Best per session: " + "; ".join(
                "s{}->{} ({:.3f})".format(session, item["entry"]["product_id"], item["score"])
                for session, item in sorted(per_session.items())))
        lines.append("If several products share a category (they often look alike), compare the exact page photos "
                     "from that category with VIEW_PAGES; the score gap between rank 1 and the rest is a useful cue.")
        return "\n".join(lines)
