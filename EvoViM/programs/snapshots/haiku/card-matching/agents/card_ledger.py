"""Pixel-derived card ledger for grid memory-match boards.

The ledger turns each public frame into records: the grid geometry, one state
per cell (face-down, face-up, removed), a picture identity for every face-up
cell (exact pixel comparison), and a check of each announced verdict against
the pixels. It never chooses an action and never decides what the LLM should
remember. The LLM sees a bounded summary on every call and can query details
with the CARD_LEDGER tool; its own notes and action choices are unchanged.

All state is plain JSON-serialisable data (pixel arrays are base64 of zlib),
so a checkpoint round-trips.
"""
import base64
import re
import zlib

import numpy as np

BG_TOL = 30          # summed channel difference above which a pixel is "not background"
RUN_FRACTION = 0.5   # projection threshold, as a fraction of the maximum projection
INSET_FRACTION = 0.09  # crop margin, as a fraction of the detected cell width
FLAT_STD = 1.0       # a crop with std below this is flat: a removed (blank) cell
BACK_TOL = 1.0       # mean abs difference to the learned card back below this = face-down
PICTURE_TOL = 2.0    # mean abs difference below this = same picture (rendering is exact)
MAX_LIST = 24        # bound on how many items a summary line may list
MAX_UNSEEN = 36      # bound on the listed never-seen face-down cells in the summary
MAX_MISSES = 3       # bound on the remembered missed known-partner choices
MAX_FLAGS = 40
MAX_VERDICTS_SHOWN = 20

VERDICT_RE = re.compile(r"You flipped ([a-z][A-Z]) and ([a-z][A-Z])\. (No match|Match)")
FIRST_RE = re.compile(r"You flipped ([a-z][A-Z])\. Flip the second card")
ROUND_RE = re.compile(r"Round \d+ (start|next)")


def cell_name(row, col):
    return chr(ord("a") + row) + chr(ord("A") + col)


def _runs(profile):
    if profile.size == 0 or float(profile.max()) <= 0:
        return []
    keep = profile > RUN_FRACTION * float(profile.max())
    runs, start = [], None
    for i, flag in enumerate(keep):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, int(keep.size)))
    return runs


def detect_grid(arr):
    """Return (x_runs, y_runs) of card extents, or None when the board is not recognised.

    arr is an HxWx3 uint8 array. The background is taken from a pixel inside the
    top-left margin; card runs are contiguous projection spans above half the peak.
    """
    if arr.ndim != 3 or arr.shape[0] < 8 or arr.shape[1] < 8:
        return None
    bg = arr[2, 2].astype(np.int16)
    mask = np.abs(arr.astype(np.int16) - bg).sum(axis=2) > BG_TOL
    xs = _runs(mask.sum(axis=0))
    ys = _runs(mask.sum(axis=1))
    if len(xs) < 2 or len(ys) < 2:
        return None
    widths = [b - a for a, b in xs] + [b - a for a, b in ys]
    if max(widths) > 1.25 * min(widths):
        return None
    return xs, ys


def _enc(arr):
    arr = np.ascontiguousarray(arr)
    return {"dtype": str(arr.dtype), "shape": list(arr.shape),
            "data": base64.b64encode(zlib.compress(arr.tobytes())).decode("ascii")}


def _dec(obj):
    raw = zlib.decompress(base64.b64decode(obj["data"]))
    return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()


class CardLedger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = None            # {"rows", "cols", "crop_h", "crop_w", "inset", "xs", "ys"}
        self.back = None            # uint8 crop of the learned card back
        self.reps = []              # uint8 crop per picture index
        self.pictures = []          # {"id", "cells": set(name), "first_step"}
        self.cell_pic = {}          # name -> picture index (last face-up identity)
        self.state = {}             # name -> "D" | "U" | "R" in the latest frame
        self.removed = set()
        self.verdicts = []          # one record per announced verdict
        self.pending_first = None   # first card of a round awaiting its partner
        self.pending_partners = None  # face-down partners of pending_first, read when it was flipped
        self.partner_stats = {"known": 0, "hit": 0}  # second flips made while a partner was known
        self.misses = []            # latest missed known-partner choices (text)
        self.last_step = None
        self.last_verdict_step = None
        self.last_match_step = None
        self.last_good_step = None  # step of the last frame that was recognised
        self.frames = 0
        self.flags = []

    # ------------------------------------------------------------------ perception
    def _flag(self, step, text):
        self.flags.append("step %s: %s" % (step, text))
        if len(self.flags) > MAX_FLAGS:
            self.flags = self.flags[-MAX_FLAGS:]

    def _crops(self, arr, xs, ys):
        if self.grid is None:
            w = min(b - a for a, b in xs)
            h = min(b - a for a, b in ys)
            inset = max(1, int(round(INSET_FRACTION * w)))
            self.grid = {"rows": len(ys), "cols": len(xs), "crop_h": h - 2 * inset,
                         "crop_w": w - 2 * inset, "inset": inset,
                         "xs": [list(p) for p in xs], "ys": [list(p) for p in ys]}
        g = self.grid
        out = {}
        for r, (y0, _y1) in enumerate(ys):
            for c, (x0, _x1) in enumerate(xs):
                if r >= g["rows"] or c >= g["cols"]:
                    continue
                y = y0 + g["inset"]
                x = x0 + g["inset"]
                out[cell_name(r, c)] = arr[y:y + g["crop_h"], x:x + g["crop_w"]].copy()
        return out

    def _picture_for(self, crop, step):
        best, best_d = None, None
        for i, rep in enumerate(self.reps):
            d = float(np.mean(np.abs(rep.astype(np.float32) - crop.astype(np.float32))))
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is not None and best_d < PICTURE_TOL:
            return best
        self.reps.append(crop)
        self.pictures.append({"id": "p%02d" % (len(self.reps)), "cells": set(), "first_step": step})
        if best_d is not None and best_d < 4 * PICTURE_TOL:
            self._flag(step, "new picture is close to an existing one (mad %.2f)" % best_d)
        return len(self.reps) - 1

    def _classify(self, crops, step):
        flat = {n: float(c.astype(np.float32).std()) < FLAT_STD for n, c in crops.items()}
        nonflat = [n for n in crops if not flat[n]]
        if self.back is None:
            if len(nonflat) < 2:
                self._flag(step, "no face-down cards in first frame; back template not learned")
                return None
            stack = np.stack([crops[n] for n in nonflat]).astype(np.float32)
            self.back = np.round(np.median(stack, axis=0)).astype(np.uint8)
        back = self.back.astype(np.float32)
        states = {}
        for n, crop in crops.items():
            if flat[n]:
                states[n] = "R"
            elif float(np.mean(np.abs(crop.astype(np.float32) - back))) < BACK_TOL:
                states[n] = "D"
            else:
                states[n] = "U"
        return states

    def observe(self, image, short_text, step):
        """Record one public frame. Returns the derived summary text for this call."""
        try:
            self._observe(image, short_text or "", step)
        except Exception as exc:  # the ledger must never stop the agent; it reports itself
            self._flag(step, "ledger error: %r" % (exc,))
        try:
            return self.summary_text()
        except Exception as exc:  # the summary is derived text; a failure here is flagged, not raised
            self._flag(step, "summary error: %r" % (exc,))
            return "CARD LEDGER: unavailable this step (internal error flagged); rely on your notes."

    def _observe(self, image, short_text, step):
        if step == self.last_step:
            return
        self.last_step = step
        if image is None:
            self._flag(step, "no image; ledger not updated")
            return
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        found = detect_grid(arr)
        if found is None:
            self._flag(step, "board not recognised in frame")
            return
        xs, ys = found
        crops = self._crops(arr, xs, ys)
        states = self._classify(crops, step)
        if states is None:
            return
        self.frames += 1
        self.last_good_step = step
        removed_before = set(self.removed)
        for name, st in states.items():
            if st == "U":
                pic = self._picture_for(crops[name], step)
                self.cell_pic[name] = pic
                self.pictures[pic]["cells"].add(name)
            if st == "R":
                self.removed.add(name)
            elif name in self.removed:
                self._flag(step, "cell %s was removed but now shows a card" % name)
        self.state = states
        n_up = sum(1 for s in states.values() if s == "U")
        if n_up > 2:
            self._flag(step, "%d face-up cells in one frame" % n_up)
        for p in self.pictures:
            if len(p["cells"]) > 2:
                self._flag(step, "%s seen at %d cells" % (p["id"], len(p["cells"])))
                break
        self._read_text(short_text, step)
        # Cells may only disappear after an announced Match (shown at this step or the previous one).
        new_removed = self.removed - removed_before
        if new_removed and (self.last_match_step is None or step - self.last_match_step > 1):
            self._flag(step, "%d cell(s) became blank without a recent Match: %s"
                       % (len(new_removed), ",".join(sorted(new_removed))))

    def _read_text(self, short_text, step):
        m_verdict = VERDICT_RE.search(short_text)
        if m_verdict and step != self.last_verdict_step:
            self._record_verdict(m_verdict, step)
        m_first = FIRST_RE.search(short_text)
        if m_first:
            # The partner is read now, while the first card is face-up and its partner face-down.
            self.pending_first = m_first.group(1)
            self.pending_partners = self._partners_of(self.pending_first)
        if ROUND_RE.search(short_text) or m_verdict:
            self.pending_first = None
            self.pending_partners = None

    def _partners_of(self, cell):
        """Face-down cells whose picture is the same as the picture of cell (derived, pixel identity)."""
        idx = self.cell_pic.get(cell)
        if idx is None:
            return []
        return sorted(n for n in self._cells_of(idx) if n != cell and self.state.get(n) == "D")

    def _record_verdict(self, m, step):
        a, b, verdict = m.group(1), m.group(2), m.group(3)
        pa, pb = self.cell_pic.get(a), self.cell_pic.get(b)
        consistent = None
        if pa is not None and pb is not None:
            consistent = (pa == pb) == (verdict == "Match")
            if not consistent:
                self._flag(step, "verdict '%s' for %s,%s disagrees with pixel pictures" % (verdict, a, b))
        # Second-card audit: when the first card had a known face-down partner, was it chosen?
        if self.pending_first == a and self.pending_partners:
            self.partner_stats["known"] += 1
            if b in self.pending_partners:
                self.partner_stats["hit"] += 1
            else:
                self.misses.append("step %d: first %s, second %s, known partner %s" % (
                    step, a, b, ",".join(self.pending_partners)))
                self.misses = self.misses[-MAX_MISSES:]
        self.verdicts.append({"step": step, "a": a, "b": b, "verdict": verdict,
                              "pic_a": self._pid(pa), "pic_b": self._pid(pb),
                              "consistent": consistent})
        self.last_verdict_step = step
        if verdict == "Match":
            self.last_match_step = step

    def _pid(self, idx):
        return None if idx is None else self.pictures[idx]["id"]

    # ------------------------------------------------------------------ derived views
    def _all_names(self):
        if self.grid is None:
            return []
        return [cell_name(r, c) for r in range(self.grid["rows"]) for c in range(self.grid["cols"])]

    def _face_down(self):
        return [n for n in self._all_names() if self.state.get(n) == "D"]

    def ready_pairs(self):
        groups = {}
        for n in self._face_down():
            idx = self.cell_pic.get(n)
            if idx is not None:
                groups.setdefault(idx, []).append(n)
        return [(self.pictures[i]["id"], cells) for i, cells in sorted(groups.items()) if len(cells) >= 2]

    def known_down(self):
        groups = {}
        for n in self._face_down():
            idx = self.cell_pic.get(n)
            if idx is not None and len(self._cells_of(idx)) < 2:
                groups.setdefault(idx, []).append(n)
        return [(self.pictures[i]["id"], cells) for i, cells in sorted(groups.items())]

    def _cells_of(self, idx):
        return self.pictures[idx]["cells"]

    def _pending_line(self):
        x = self.pending_first
        if self.state.get(x) != "U" or x not in self.cell_pic:
            return ("- PENDING FIRST CARD %s: not shown face-up with a recognised picture in this frame, so the "
                    "partner check is unavailable here; choose the second card." % x)
        pid = self.pictures[self.cell_pic[x]]["id"]
        partners = self._partners_of(x)
        if partners:
            return ("- PENDING FIRST CARD %s (picture %s): face-down partner with the same picture: %s. "
                    "Flipping %s as the second card makes a match."
                    % (x, pid, ", ".join(partners), partners[0]))
        return ("- PENDING FIRST CARD %s (picture %s): no face-down partner is known for it; "
                "choose the second card." % (x, pid))

    def pending_query(self):
        x = self.pending_first
        if not x:
            return "No first card is pending: the next flip is a first card."
        if self.state.get(x) != "U" or x not in self.cell_pic:
            return "First card %s is pending but its picture is not known in this frame." % x
        pid = self.pictures[self.cell_pic[x]]["id"]
        partners = self._partners_of(x)
        if partners:
            return ("First card %s (picture %s). Known face-down partner(s): %s. "
                    "Choosing %s as the second card makes a match." % (x, pid, ", ".join(partners), partners[0]))
        return "First card %s (picture %s). No face-down partner is known for it." % (x, pid)

    def summary_text(self):
        if self.grid is None or not self.state:
            return ("CARD LEDGER: board not recognised in this frame; rely on the observation "
                    "text and your notes.")
        names = self._all_names()
        n_total = len(names)
        down = self._face_down()
        up = [n for n in names if self.state.get(n) == "U"]
        unseen = [n for n in down if n not in self.cell_pic]
        lines = ["CARD LEDGER (derived from this frame's pixels; your own notes stay authoritative):"]
        if self.last_good_step != self.last_step:
            lines.append("- WARNING: the current frame was not recognised; the figures below are from "
                         "step %s and may be stale." % self.last_good_step)
        lines.append("- removed %d of %d cells; face-down %d (picture known %d, not yet seen %d); face-up now %d"
                     % (len(self.removed), n_total, len(down), len(down) - len(unseen), len(unseen), len(up)))
        if up:
            parts = ["%s=%s" % (n, self._pid(self.cell_pic.get(n))) for n in up]
            lines.append("- face-up now: " + ", ".join(parts)
                         + " (non-matching face-up cards turn face-down when your next first card is flipped)")
        if self.pending_first:
            lines.append(self._pending_line())
        ready = self.ready_pairs()
        if ready:
            shown = ["%s at %s" % (pid, " and ".join(cells)) for pid, cells in ready[:MAX_LIST]]
            more = "" if len(ready) <= MAX_LIST else " (+%d more)" % (len(ready) - MAX_LIST)
            lines.append("- READY PAIRS (both face-down, same picture): " + "; ".join(shown) + more)
        else:
            lines.append("- READY PAIRS: none")
        known = self.known_down()
        if known:
            shown = ["%s at %s" % (pid, ",".join(cells)) for pid, cells in known[:MAX_LIST]]
            more = "" if len(known) <= MAX_LIST else " (+%d more)" % (len(known) - MAX_LIST)
            lines.append("- face-down, picture seen before, partner not yet seen: " + "; ".join(shown) + more)
        if unseen:
            shown = ", ".join(unseen[:MAX_UNSEEN])
            more = "" if len(unseen) <= MAX_UNSEEN else " (+%d more)" % (len(unseen) - MAX_UNSEEN)
            lines.append("- face-down, no picture seen yet (%d): %s%s" % (len(unseen), shown, more))
        if self.partner_stats["known"]:
            kn, hit = self.partner_stats["known"], self.partner_stats["hit"]
            tail = ("; latest misses: " + " | ".join(self.misses)) if self.misses else ""
            lines.append("- second flips made while a partner was known: %d; the known partner was chosen %d "
                         "time(s), missed %d%s" % (kn, hit, kn - hit, tail))
        agree = sum(1 for v in self.verdicts if v["consistent"] is True)
        disagree = sum(1 for v in self.verdicts if v["consistent"] is False)
        unknown = sum(1 for v in self.verdicts if v["consistent"] is None)
        if self.verdicts:
            lines.append("- announced verdicts checked against pixels: %d agree, %d disagree, %d unknown"
                         % (agree, disagree, unknown))
        if self.flags:
            lines.append("- ledger flags (latest): " + " | ".join(self.flags[-3:]))
        lines.append("Use CARD_LEDGER (query: pending | cell | picture | ready | unseen | verdicts | flags) for detail.")
        return "\n".join(lines)

    def query(self, query, cell="", picture=""):
        q = (query or "").strip().lower()
        if q == "summary":
            return self.summary_text()
        if q == "pending":
            return self.pending_query()
        if q == "ready":
            ready = self.ready_pairs()
            if not ready:
                return "No ready pairs: no two face-down cells share a seen picture."
            return "\n".join("%s: %s" % (pid, ", ".join(cells)) for pid, cells in ready)
        if q == "unseen":
            names = [n for n in self._face_down() if n not in self.cell_pic]
            return "Face-down cells with no picture seen yet (%d): %s" % (len(names), ", ".join(names) or "none")
        if q == "verdicts":
            if not self.verdicts:
                return "No verdicts announced yet."
            lines = []
            for v in self.verdicts[-MAX_VERDICTS_SHOWN:]:
                lines.append("step %d: %s and %s -> %s; pictures %s/%s; pixel check %s" % (
                    v["step"], v["a"], v["b"], v["verdict"], v["pic_a"], v["pic_b"],
                    {True: "agrees", False: "DISAGREES", None: "unknown"}[v["consistent"]]))
            return "\n".join(lines)
        if q == "flags":
            return "\n".join(self.flags) if self.flags else "No ledger flags."
        if q == "cell":
            name = (cell or "").strip()
            if name not in self._all_names():
                return "Unknown cell '%s'. Cells are a-l (row) followed by A-L (column), e.g. aA." % name
            st = self.state.get(name, "unknown")
            label = {"D": "face-down", "U": "face-up", "R": "removed", "unknown": "not yet observed"}[st]
            idx = self.cell_pic.get(name)
            if idx is None:
                return "%s: %s; picture not yet seen." % (name, label)
            pid = self.pictures[idx]["id"]
            partners = sorted(self._cells_of(idx) - {name})
            return "%s: %s; picture %s; other cell(s) seen with this picture: %s" % (
                name, label, pid, ", ".join(partners) or "none yet")
        if q == "picture":
            want = (picture or "").strip().lower()
            for idx, p in enumerate(self.pictures):
                if p["id"].lower() == want:
                    cells = sorted(p["cells"])
                    desc = ["%s=%s" % (n, {"D": "face-down", "U": "face-up", "R": "removed"}.get(
                        self.state.get(n), "unknown")) for n in cells]
                    return "%s: seen at %s; first seen step %s" % (p["id"], ", ".join(desc), p["first_step"])
            return "Unknown picture '%s'." % picture
        return ("Unknown query '%s'. Use summary, pending, cell, picture, ready, unseen, verdicts or flags."
                % query)

    # ------------------------------------------------------------------ checkpoint
    def to_state(self):
        return {
            "grid": self.grid,
            "back": _enc(self.back) if self.back is not None else None,
            "reps": [_enc(r) for r in self.reps],
            "pictures": [{"id": p["id"], "cells": sorted(p["cells"]), "first_step": p["first_step"]}
                         for p in self.pictures],
            "cell_pic": self.cell_pic,
            "state": self.state,
            "removed": sorted(self.removed),
            "verdicts": self.verdicts,
            "pending_first": self.pending_first,
            "pending_partners": self.pending_partners,
            "partner_stats": self.partner_stats,
            "misses": self.misses,
            "last_step": self.last_step,
            "last_verdict_step": self.last_verdict_step,
            "last_match_step": self.last_match_step,
            "last_good_step": self.last_good_step,
            "frames": self.frames,
            "flags": self.flags,
        }

    def from_state(self, data):
        self.reset()
        if not data:
            return
        self.grid = data.get("grid")
        self.back = _dec(data["back"]) if data.get("back") else None
        self.reps = [_dec(r) for r in data.get("reps", [])]
        self.pictures = [{"id": p["id"], "cells": set(p["cells"]), "first_step": p["first_step"]}
                         for p in data.get("pictures", [])]
        self.cell_pic = {k: int(v) for k, v in data.get("cell_pic", {}).items()}
        self.state = dict(data.get("state", {}))
        self.removed = set(data.get("removed", []))
        self.verdicts = list(data.get("verdicts", []))
        self.pending_first = data.get("pending_first")
        pp = data.get("pending_partners")
        self.pending_partners = list(pp) if pp is not None else None
        ps = data.get("partner_stats") or {}
        self.partner_stats = {"known": int(ps.get("known", 0)), "hit": int(ps.get("hit", 0))}
        self.misses = list(data.get("misses", []))
        self.last_step = data.get("last_step")
        self.last_verdict_step = data.get("last_verdict_step")
        self.last_match_step = data.get("last_match_step")
        self.last_good_step = data.get("last_good_step")
        self.frames = int(data.get("frames", 0))
        self.flags = list(data.get("flags", []))

