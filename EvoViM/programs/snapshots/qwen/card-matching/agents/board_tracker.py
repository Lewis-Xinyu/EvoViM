"""Code-side board tracker for the Memory Match task (line M).

Derives a per-cell board model from the episode's own public observations
(the text observation plus the current frame).  This is automatic
COMPUTATION over public observations: parsing, normalisation, indexing and
cross-checking.  It never decides the agent's action; the inner LLM keeps
full authority over its memory files and can correct the tracker through
the SET_CELL_STATE / CLEAR_FINGERPRINT tools.

Everything the tracker learns comes from this episode's observations:
grid size from the task text, board geometry from the first frame, the card
back template from the first frame, and card identities from frames in
which the text says a card was flipped.  No constant is taken from any
particular development episode.
"""
import hashlib
import json
import os
import re

import numpy as np

MAX_SHIFT = 8        # sprite jitter search radius in pixels (minimum)
# Margins below are measured on ALL development units (42_0, 46_0 a004) and
# two synthetic boards with unrelated sprites/seeds (seeds 7, 1234) through
# the candidate crop pipeline (validation/threshold_margins.py):
#   same-picture crop pairs:  max MAD 7.93   (dev 4.83 / 4.81, synth 7.40/7.93)
#   different-picture pairs:  min MAD 31.37  (dev 33.39 / 31.37, synth ~75)
#   re-flip own-crop pairs:   max MAD 10.04  (dev 0 - no jitter, synth 7.15/10.04)
#   back cells vs template:   worst L1 1745, worst LINF 28
#   face crops vs template:   best L1 4109,  best LINF 75
IDENTICAL_MAD = 20.0 # aligned mean-abs-diff below this => same picture
                    # (2.52x above worst same-picture MAD, 1.57x below worst
                    # different-picture MAD; fragments self-heal on re-flip,
                    # false merges do not, so the lower side is kept tighter)
REFLIP_MAD = 25.0    # a re-flipped cell matches its own saved crop below this
                    # (2.49x above the worst measured re-flip MAD)
BACK_SIG_L1 = 3000   # block-signature L1 distance vs back template: face-down
                    # (1.72x above worst back, 1.37x below best face)
BACK_SIG_LINF = 50   # max single-block deviation vs back template: face-down
                    # (1.79x above worst back LINF, 1.50x below best face LINF)
REMOVED_WHITE = 0.5  # fraction of near-white pixels => removed slot
WHITE_LEVEL = 230
BG_DIST = 40         # L1 distance from corner colour for board mask
BAND_FRAC = 0.5      # row/col fraction of board pixels to count as board

COORD_RE = re.compile(r"(?<![A-Za-z0-9])([a-z]{1,2})([A-Z]{1,2})(?![A-Za-z0-9])")
FLIPS_RE = re.compile(r"Flips used:\s*(\d+)\s*/\s*(\d+)")
SCORE_RE = re.compile(r"Score:\s*(\d+)")
PAIRS_RE = re.compile(r"Remaining pairs:\s*(\d+)")
TOTAL_RE = re.compile(r"(\d+)\s+pairs?\s+on\s+a\s+(\d+)\s*[xX]\s*(\d+)")
ROWS_RE = re.compile(r"rows\s+([a-z]+)\s*,\s*columns\s+([A-Z]+)")
FIRST_FLIP_RE = re.compile(r"You flipped (\S+?)\. Flip the second")
ROUND_END_RE = re.compile(r"You flipped (\S+?) and (\S+?)\.\s*(Match!|No match)")
ROUND_START_RE = re.compile(r"Round\s+\d+\s+start")
ROUND_NUM_RE = re.compile(r"Round\s+(\d+)\s+(?:start|next)")
INVALID_RE = re.compile(r"Invalid flip (\S+?):\s*(it is the card you already flipped this round|that slot is already empty)")

# A round ends when its second card is flipped; with a 2-flip-per-round game
# an episode of N rounds fits easily in this cap, so the log stays bounded.
ROUND_LOG_CAP = 400


def _letter_index(s):
    """a=0, b=1, ..., z=25, aa=26, ..."""
    value = 0
    for ch in s:
        value = value * 26 + (ord(ch.lower()) - ord("a") + 1)
    return value - 1


def _index_letter(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(ord("a") + r) + s
    return s


def _pad(x, p):
    return np.pad(x, ((p, p), (p, p), (0, 0)))


def aligned_mad(a, b, max_shift=MAX_SHIFT):
    """Minimum mean absolute difference between two same-size RGB crops over
    integer shifts of b within +/-max_shift pixels (two-stage: coarse search
    at half resolution, then refinement at full resolution)."""
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    if a.size == 0:
        return 1e9
    # coarse at half resolution
    ac = a[::2, ::2].astype(np.int32)
    bc = b[::2, ::2].astype(np.int32)
    h, w, _ = ac.shape
    ph = max_shift // 2
    pcb = _pad(bc, ph).astype(np.int32)
    best_d, best_dr, best_dc = 1e9, 0, 0
    for dr in range(-ph, ph + 1):
        base = pcb[ph + dr:ph + dr + h, :, :]
        if base.shape[0] != h:
            continue
        for dc in range(2 * ph + 1):
            d = float(np.abs(base[:, dc:dc + w, :] - ac).mean())
            if d < best_d:
                best_d, best_dr, best_dc = d, dr, dc - ph
    # fine at full resolution around the coarse best
    pbf = _pad(b.astype(np.int32), max_shift)
    lo_r = max(-max_shift, 2 * best_dr - 2)
    hi_r = min(max_shift, 2 * best_dr + 2)
    lo_c = max(-max_shift, 2 * best_dc - 2)
    hi_c = min(max_shift, 2 * best_dc + 2)
    for dr in range(lo_r, hi_r + 1):
        base = pbf[max_shift + dr:max_shift + dr + a.shape[0], :, :]
        if base.shape[0] != a.shape[0]:
            continue
        for dc in range(lo_c, hi_c + 1):
            d = float(np.abs(base[:, max_shift + dc:max_shift + dc + a.shape[1], :] - a.astype(np.int32)).mean())
            if d < best_d:
                best_d = d
    return best_d


def white_ratio(crop):
    return float((crop > WHITE_LEVEL).all(axis=2).mean())


def block_sig(crop, blocks=8):
    """Centrally-anchored square of the crop reduced to blocks x blocks x 3
    block means.  The square is centred (not top-left anchored) so that on
    rectangular cell windows the same relative region of the card is sampled
    for every cell; on square windows this is identical to the full crop."""
    h, w, _ = crop.shape
    s = min(h, w)
    crop = crop[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    m = (s - s // blocks * blocks) // 2
    crop = crop[m:m + s // blocks * blocks, m:m + s // blocks * blocks]
    size = s // blocks * blocks
    return (crop.reshape(blocks, size // blocks, blocks, size // blocks, 3)
            .mean(axis=(1, 3)).round().astype(np.int32).flatten().tolist())


class BoardTracker:
    """Per-cell board model derived from this episode's public observations."""

    def __init__(self, board_dir, memory_root):
        self.board_dir = board_dir
        self.memory_root = memory_root
        self.rows = 12
        self.cols = 12
        self.bg_color = None       # frame corner colour (board background)
        self.interior_shape = None   # canonical interior-crop (h, w)
        self.total_pairs = 72
        self.geom = None            # [r0, c0, r1, c1] in pixels
        self.back_crop = None       # cell-sized int16 mean card back
        self.back_sig = None        # 8x8x3 block means of the back template
        self.crop_h = 68            # per-cell crop size, derived from geometry
        self.crop_w = 68
        self.max_shift = MAX_SHIFT  # jitter search radius, derived from geometry
        self.cells = {}             # coord -> {"state","token","sig","note","seen"}
        self.tokens = {}            # token -> {"rep": coord, "cells": [coord]}
        self.round_first = None
        self.flips_used = 0
        self.flip_budget = 360
        self.score = 0
        self.remaining_pairs = None
        self.warnings = []
        self._warn_total = 0        # monotonic count of warnings ever issued
        self._warns_reviewed = 0    # warnings already shown to the LLM in a LOG
        self.step = 0
        self.initialized = False
        self.pixel_enabled = False
        self._crops = {}            # coord -> cell crop int16 (lazy from disk)
        self.round_log = []         # structured per-round outcome lines
        self.round_no = 0           # number of completed rounds seen so far
        os.makedirs(self.board_dir, exist_ok=True)
        state_path = os.path.join(self.board_dir, "board_state.json")
        if os.path.isfile(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    self.import_state(json.load(f))
            except Exception:
                pass

    # ------------------------------------------------------------------ coords
    def all_coords(self):
        return ["%s%s" % (_index_letter(r), _index_letter(c).upper())
                for r in range(self.rows) for c in range(self.cols)]

    def parse_coord(self, text):
        """Return the coordinate string from text if it is on this grid."""
        for m in COORD_RE.finditer(text):
            r, c = _letter_index(m.group(1)), _letter_index(m.group(2))
            if 0 <= r < self.rows and 0 <= c < self.cols:
                return m.group(0)
        return None

    def _coord_index(self, coord):
        return _letter_index(coord[0].lower()), _letter_index(coord[1].upper())

    def _valid(self, coord):
        if not coord or len(coord) != 2:
            return False
        r, c = _letter_index(coord[0].lower()), _letter_index(coord[1].upper())
        return 0 <= r < self.rows and 0 <= c < self.cols

    # ------------------------------------------------------------------ state
    def _cell(self, coord):
        cell = self.cells.get(coord)
        if cell is None:
            cell = {"state": "unknown", "token": None, "sig": None, "note": "", "seen": 0}
            self.cells[coord] = cell
        return cell

    def _remove_token(self, coord):
        cell = self.cells.get(coord)
        if not cell or not cell.get("token"):
            return
        tok = cell["token"]
        entry = self.tokens.get(tok)
        if entry:
            if coord in entry["cells"]:
                entry["cells"].remove(coord)
            if not entry["cells"]:
                del self.tokens[tok]
            elif entry["rep"] == coord and entry["cells"]:
                entry["rep"] = entry["cells"][0]
        cell["token"] = None

    def _add_token(self, coord, token):
        self.cells[coord]["token"] = token
        entry = self.tokens.setdefault(token, {"rep": coord, "cells": []})
        if coord not in entry["cells"]:
            entry["cells"].append(coord)

    # ------------------------------------------------------------------ pixels
    def _cell_crop(self, image, coord):
        r, c = self._coord_index(coord)
        r0, c0, r1, c1 = self.geom
        ch = (r1 - r0 + 1) / float(self.rows)
        cw = (c1 - c0 + 1) / float(self.cols)
        rr0 = int(round(r0 + r * ch))
        cc0 = int(round(c0 + c * cw))
        H, W = image.shape[:2]
        rr0 = min(max(rr0, 0), H - 1)
        cc0 = min(max(cc0, 0), W - 1)
        hh = min(self.crop_h, H - rr0)
        ww = min(self.crop_w, W - cc0)
        crop = image[rr0:rr0 + hh, cc0:cc0 + ww].astype(np.int16)
        if crop.shape[0] < self.crop_h or crop.shape[1] < self.crop_w:
            crop = np.pad(crop, ((0, self.crop_h - crop.shape[0]),
                                 (0, self.crop_w - crop.shape[1]), (0, 0)),
                          mode="edge")
        return self._card_interior(crop)

    def _body_mask(self, crop):
        """Boolean mask of the card body inside a cell window, or None when
        no body can be isolated (degenerate frame, or the whole window is
        one colour such as a removed all-white slot).  Background is the
        frame corner colour, learned from this episode's own frames."""
        if self.bg_color is None:
            return None
        mask = (np.abs(crop.astype(np.int32) - self.bg_color).sum(axis=2)
                > BG_DIST)
        if mask.mean() < 0.25 or mask.mean() > 0.995:
            return None
        return mask

    def _card_bbox(self, crop):
        """Bounding box of the card body inside a cell window, or None."""
        mask = self._body_mask(crop)
        if mask is None:
            return None
        rs = np.where(mask.any(axis=1))[0]
        cs = np.where(mask.any(axis=0))[0]
        r0, r1 = int(rs[0]), int(rs[-1])
        c0, c1 = int(cs[0]), int(cs[-1])
        if r1 - r0 + 1 < 12 or c1 - c0 + 1 < 12:
            return None
        return (r0, r1, c0, c1)

    def _card_interior(self, crop):
        """Return the canonical interior crop of the card in a cell window.

        The cell window is a slice of the board's bounding box, not of the
        card pitch: the card therefore occupies a cell-dependent, jitter-
        shifted portion of the window, and outer columns/rows can clip it.
        Comparing raw windows would mix background bands into identity
        crops and block signatures, which breaks _is_back and token
        merging on boards whose cells are not (near-)square.

        The interior crop is anchored on the card body's own mask
        centroid (the card's centre), which is independent of where the
        window places the card and immune to edge clipping: a bbox-edge
        anchor drifts with the cell position and loses 1-3px to clipping,
        while the centroid stays at the card centre in every cell.  The
        crop is the central ~80% of the full card body, normalised to one
        canonical even shape per episode (sized in _init_back from the
        least-clipped card bodies) so that all cell crops come out
        exactly that shape and are stackable and comparable; same-sprite
        crops from different cells then differ only by the ~1px sprite
        jitter.  Falls back to a centred whole-window crop if no body can
        be isolated (degenerate frame)."""
        if self.bg_color is None or self.interior_shape is None:
            return crop
        ih, iw = self.interior_shape
        h, w = crop.shape[:2]
        if h < ih or w < iw:
            ph = max(0, ih - h)
            pw = max(0, iw - w)
            crop = np.pad(crop,
                          ((ph // 2, ph - ph // 2),
                           (pw // 2, pw - pw // 2), (0, 0)), mode="edge")
            h, w = crop.shape[:2]
        mask = self._body_mask(crop)
        if mask is None:
            # no isolated body (e.g. a removed all-white slot): centre
            return crop[(h - ih) // 2:(h - ih) // 2 + ih,
                        (w - iw) // 2:(w - iw) // 2 + iw]
        ys, xs = np.nonzero(mask)
        r_a = int(round(float(ys.mean()))) - ih // 2
        c_a = int(round(float(xs.mean()))) - iw // 2
        r_a = min(max(r_a, 0), h - ih)
        c_a = min(max(c_a, 0), w - iw)
        return crop[r_a:r_a + ih, c_a:c_a + iw]

    def _init_geometry(self, image):
        arr = image.astype(np.int16)
        corner = arr[0, 0].astype(np.int32)
        self.bg_color = corner
        dist = np.abs(arr.astype(np.int32) - corner).sum(axis=2)
        mask = dist > BG_DIST
        row_frac = mask.mean(axis=1)
        col_frac = mask.mean(axis=0)
        rs = np.where(row_frac > BAND_FRAC)[0]
        cs = np.where(col_frac > BAND_FRAC)[0]
        if len(rs) < 4 or len(cs) < 4:
            return False
        self.geom = [int(rs[0]), int(cs[0]), int(rs[-1]), int(cs[-1])]
        ch = (self.geom[2] - self.geom[0] + 1) / float(self.rows)
        cw = (self.geom[3] - self.geom[1] + 1) / float(self.cols)
        self.crop_h = int(round(ch))
        self.crop_w = int(round(cw))
        self.max_shift = max(MAX_SHIFT, int(0.15 * min(self.crop_h, self.crop_w)))
        return True

    def _init_back(self, image):
        # pass 1: size the canonical interior from the card bodies of the
        # first frame.  The max over cells is used so a clipped edge cell
        # cannot shrink the canonical size below the full card interior;
        # sizes are kept even for stable stacking.
        windows = [self._cell_crop(image, c) for c in self.all_coords()]
        boxes = [self._card_bbox(cr) for cr in windows]
        boxes = [b for b in boxes if b is not None]
        if boxes:
            ih = max(12, ((max(b[1] - b[0] + 1 for b in boxes)) * 4) // 5)
            iw = max(12, ((max(b[3] - b[2] + 1 for b in boxes)) * 4) // 5)
            self.interior_shape = (ih - ih % 2, iw - iw % 2)
        # pass 2: centroid-anchored interior crops (uniform shape)
        crops = [self._cell_crop(image, c) for c in self.all_coords()]
        mean = np.mean(np.stack(crops), axis=0).astype(np.int16)
        self.back_sig = block_sig(mean)
        back_like = [cr for cr in crops if self._is_back(cr)]
        if len(back_like) * 10 < len(crops) * 9:
            # board is not fully face-down at initialisation; use the
            # back-like majority as the template
            if len(back_like) * 2 < len(crops):
                self.back_sig = None
                return False
            mean = np.mean(np.stack(back_like), axis=0).astype(np.int16)
            self.back_sig = block_sig(mean)
        self.back_crop = mean
        path = os.path.join(self.board_dir, "_back.npy")
        np.save(path, mean)
        return True

    def _load_back(self):
        path = os.path.join(self.board_dir, "_back.npy")
        if os.path.isfile(path):
            try:
                self.back_crop = np.load(path).astype(np.int16)
                self.back_sig = block_sig(self.back_crop)
                return True
            except Exception:
                return False
        return False

    def _save_crop(self, coord, crop):
        path = os.path.join(self.board_dir, "%s.npy" % coord)
        np.save(path, crop.astype(np.int16))

    def _get_crop(self, coord):
        if coord in self._crops:
            return self._crops[coord]
        path = os.path.join(self.board_dir, "%s.npy" % coord)
        if os.path.isfile(path):
            try:
                crop = np.load(path).astype(np.int16)
                self._crops[coord] = crop
                return crop
            except Exception:
                return None
        return None

    def _is_back(self, crop):
        """Shift-robust face-down test: block-mean signature distance to the
        back template.  Raw aligned MAD against the (sub-pixel blurred) mean
        template is ~15-19 for true backs, so it cannot separate backs from
        faces; block means are stable under the 1px sprite jitter."""
        if self.back_sig is None:
            return None
        a = np.asarray(block_sig(crop), dtype=np.int32)
        b = np.asarray(self.back_sig, dtype=np.int32)
        d = np.abs(a - b)
        return d.sum() < BACK_SIG_L1 and d.max() < BACK_SIG_LINF

    def _identify(self, coord, crop):
        """Assign an identity token to a face-up crop by aligned comparison
        against the partner of the round (first) and all known tokens."""
        # a re-flip of a known cell: its own saved crop is the strongest
        # identity signal (a card's picture never changes in this game);
        # this keeps borderline edge-strip jitter from minting new tokens
        cell = self._cell(coord)
        old_token = cell.get("token")
        if old_token:
            own = self._get_crop(coord)
            if own is not None and \
                    aligned_mad(crop, own, self.max_shift) < REFLIP_MAD:
                return old_token
        # partner of the current round first: most likely match
        if self.round_first and self.round_first != coord:
            partner_crop = self._get_crop(self.round_first)
            if partner_crop is not None and \
                    aligned_mad(crop, partner_crop, self.max_shift) < IDENTICAL_MAD:
                return self.cells[self.round_first]["token"]
        for tok, entry in list(self.tokens.items()):
            rep_crop = self._get_crop(entry["rep"])
            if rep_crop is not None and \
                    aligned_mad(crop, rep_crop, self.max_shift) < IDENTICAL_MAD:
                return tok
        token = hashlib.md5(np.ascontiguousarray(crop).tobytes()).hexdigest()[:8]
        return token

    def _record_reveal(self, coord, image):
        """Record the identity of a card the text says is face-up now."""
        cell = self._cell(coord)
        cell["state"] = "face_up"
        cell["seen"] += 1
        if image is None or self.geom is None:
            return
        crop = self._cell_crop(image, coord)
        if self._is_back(crop):
            self._warn("%s: text says flipped but pixels show a card back" % coord)
            cell["state"] = "known" if cell["token"] else "unknown"
            return
        self._save_crop(coord, crop)
        self._crops[coord] = crop
        cell["sig"] = block_sig(crop)
        old_token = cell.get("token")
        token = self._identify(coord, crop)
        if old_token and old_token != token:
            self._warn("%s: re-flip shows a different picture than recorded (%s vs %s)"
                       % (coord, old_token, token))
        self._remove_token(coord)
        self._add_token(coord, token)

    def _check_removed(self, coord, image):
        if image is None or self.geom is None:
            return
        crop = self._cell_crop(image, coord)
        if white_ratio(crop) > REMOVED_WHITE:
            self._cell(coord)["state"] = "removed"

    def _check_face_down(self, coord, image):
        if image is None or self.geom is None:
            return
        crop = self._cell_crop(image, coord)
        if not self._is_back(crop):
            self._warn("%s: expected face-down but pixels show a face or a removed slot" % coord)

    def _warn(self, message):
        self._warn_total += 1
        self.warnings.append(message)
        self.warnings = self.warnings[-20:]

    # ------------------------------------------------------------- round log
    def _log_round(self, x, y, outcome, obs_text):
        """Append one structured line for the round that just ended.

        Called only when the current observation (the environment's response
        to the previous action) reports a round outcome, so this records
        step t-1's result at step t and never invents an outcome."""
        m = ROUND_NUM_RE.search(obs_text)
        if m:
            number = max(1, int(m.group(1)) - 1)  # text announces the NEXT round
        else:
            number = max(1, self.round_no)
        self.round_no = max(self.round_no, number)
        self.round_log.append("R%d: %s+%s %s" % (number, x, y, outcome))
        self.round_log = self.round_log[-ROUND_LOG_CAP:]

    def _log_invalid(self, coord, reason, obs_text):
        m = ROUND_NUM_RE.search(obs_text)
        number = max(1, int(m.group(1))) if m else max(1, self.round_no + 1)
        short = ("slot already empty" if "empty" in reason
                 else "re-flipped this round's first card")
        self.round_log.append("R%d: INVALID %s (%s)" % (number, coord, short))
        self.round_log = self.round_log[-ROUND_LOG_CAP:]

    def round_log_tail(self, n=8):
        """Bounded view of the recent round outcomes (derived context shown
        to the LLM; the full log stays in round_log.txt for TAIL/SEARCH)."""
        if not self.round_log:
            return "(no completed rounds yet)"
        return "\n".join(self.round_log[-n:])

    # ------------------------------------------------------------- exceptions
    def log_exceptions(self, step, obs_text, prev_action=None,
                       prev_validated_action=None):
        """Derive the contradictions, if any, between this episode's own
        observation at step t (the environment's response to step t-1's
        action) and the tracker's model.  Returns a list of short strings;
        empty means the tracker and the observation agree and no LLM review
        of the log is needed.  Every check compares values that both come
        from this episode's public observations or its step index; no
        development-episode constant is used.  This is derived context: it
        only decides whether the LLM gets a LOG decision point, and the LLM
        still decides what to store or correct."""
        exceptions = []
        new_warnings = self._warn_total - self._warns_reviewed
        if new_warnings > 0:
            shown = self.warnings[-min(new_warnings, 3):]
            note = (" (showing last %d of %d)" % (len(shown), new_warnings)
                    if new_warnings > len(shown) else "")
            exceptions.append("tracker warnings since the last review: "
                              + " | ".join(shown) + note)
        m = FLIPS_RE.search(obs_text)
        if m and int(m.group(1)) != step:
            exceptions.append("observation says flips used %s but the step "
                              "index is %d" % (m.group(1), step))
        if self.remaining_pairs is not None and \
                self.score + self.remaining_pairs != self.total_pairs:
            exceptions.append("score %d + remaining pairs %d != total pairs %d"
                              % (self.score, self.remaining_pairs,
                                 self.total_pairs))
        if INVALID_RE.search(obs_text):
            exceptions.append("the observation reports an INVALID flip")
        if prev_action is not None and prev_validated_action is not None \
                and prev_action != prev_validated_action:
            exceptions.append("you committed %r but the host executed %r"
                              % (prev_action, prev_validated_action))
        return exceptions

    def mark_warnings_reviewed(self):
        """Call after the LLM has had its LOG decision point (or after a
        step where none was needed) so the same warnings are not re-shown
        on the next exception review."""
        self._warns_reviewed = self._warn_total

    # ------------------------------------------------------------------ text
    def _parse_grid(self, obs_text):
        m = TOTAL_RE.search(obs_text)
        if m:
            self.total_pairs = int(m.group(1))
            self.rows, self.cols = int(m.group(2)), int(m.group(3))
            return
        m = ROWS_RE.search(obs_text)
        if m:
            self.rows = _letter_index(m.group(1)[-1]) + 1
            self.cols = _letter_index(m.group(2)[-1]) + 1

    def update(self, obs_text, image, step):
        """Incorporate one observation (the environment's response to the
        previous action) into the board model.  At step 0 there is no previous
        action: only geometry and the back template are initialised."""
        self.step = step
        self._parse_grid(obs_text)
        if image is not None:
            image = np.asarray(image)
            if image.ndim == 3 and image.shape[2] == 3:
                image = image.astype(np.int16)
            else:
                image = None
        else:
            image = None

        if not self.initialized:
            if image is not None and self._init_geometry(image):
                if self._init_back(image):
                    self.pixel_enabled = True
            self.initialized = True

        m = FLIPS_RE.search(obs_text)
        if m:
            self.flips_used, self.flip_budget = int(m.group(1)), int(m.group(2))
        m = SCORE_RE.search(obs_text)
        if m:
            self.score = int(m.group(1))
        m = PAIRS_RE.search(obs_text)
        if m:
            self.remaining_pairs = int(m.group(1))

        # remember the cards that were face-up in the previous frame so we can
        # verify they are face-down again after a no-match
        prev_round = self.round_first
        prev_second = getattr(self, "_prev_second", None)
        self._prev_second = None

        matched_pair = None
        m = ROUND_END_RE.search(obs_text)
        if m:
            x = self.parse_coord(m.group(1))
            y = self.parse_coord(m.group(2))
            matched = m.group(3) == "Match!"
            if x and y:
                self._record_reveal(x, image)
                self._record_reveal(y, image)
                if matched:
                    matched_pair = (x, y)
                    for coord in (x, y):
                        self._cell(coord)["state"] = "removed"
                        self._remove_token(coord)
                    self._log_round(x, y, "MATCH", obs_text)
                else:
                    for coord in (x, y):
                        self._cell(coord)["state"] = "known"
                    self._log_round(x, y, "NOMATCH", obs_text)
                self.round_first = None
            else:
                self._warn("could not parse coordinates in round-end text")
        else:
            m = FIRST_FLIP_RE.search(obs_text)
            if m:
                x = self.parse_coord(m.group(1))
                if x:
                    # the previous round's cards should be face-down again
                    if prev_round and prev_round != x:
                        self._check_face_down(prev_round, image)
                    if prev_second and prev_second != x:
                        self._check_face_down(prev_second, image)
                    self.round_first = x
                    self._record_reveal(x, image)
                else:
                    self._warn("could not parse coordinate in first-flip text")
            elif ROUND_START_RE.search(obs_text):
                if prev_round:
                    self._check_face_down(prev_round, image)
                if prev_second:
                    self._check_face_down(prev_second, image)
                self.round_first = None
            else:
                m = INVALID_RE.search(obs_text)
                if m:
                    x = self.parse_coord(m.group(1))
                    reason = m.group(2)
                    if x:
                        if "empty" in reason:
                            if self._cell(x)["state"] != "removed":
                                self._warn("%s: invalid flip says slot empty; marking removed" % x)
                            self._cell(x)["state"] = "removed"
                            self._remove_token(x)
                            self._check_removed(x, image)
                            self._log_invalid(x, reason, obs_text)
                        else:
                            if self.round_first != x:
                                self._warn("invalid flip says '%s' already flipped this round" % x)
                            self._log_invalid(x, reason, obs_text)
                else:
                    self._warn("unrecognised observation template: %r" % obs_text[:80])

        # keep the round's face-up pair for next-step verification
        if matched_pair is None:
            m2 = ROUND_END_RE.search(obs_text)
            if m2:
                x = self.parse_coord(m2.group(1))
                y = self.parse_coord(m2.group(2))
                if x and y:
                    self._prev_second = y
        self._persist()
        self._write_files()

    # ------------------------------------------------------------------ output
    def _state_of(self, coord):
        return self._cell(coord)["state"]

    def known_pairs(self):
        """Tokens with at least two face-down (known) cells, best first."""
        pairs = []
        for tok, entry in self.tokens.items():
            face_down = [c for c in entry["cells"] if self._state_of(c) == "known"]
            if len(face_down) >= 2:
                pairs.append((tok, face_down))
        pairs.sort(key=lambda item: item[1][0])
        return pairs

    def partner_of(self, coord):
        cell = self.cells.get(coord)
        if not cell or not cell.get("token"):
            return None
        for other in self.tokens.get(cell["token"], {}).get("cells", []):
            if other != coord and self._state_of(other) == "known":
                return other
        return None

    def suggest(self):
        """Always-legal next flip: (coordinate, reason) or (None, reason)."""
        if self.remaining_pairs == 0:
            return None, "board cleared"
        legal = [c for c in self.all_coords() if self._state_of(c) != "removed"]
        if not legal:
            return None, "no legal flip"
        if self.round_first is None:
            for tok, face_down in self.known_pairs():
                return face_down[0], "known pair with %s (guaranteed match)" % face_down[1]
            for c in legal:
                if self._state_of(c) == "unknown":
                    return c, "explore unknown card"
            return legal[0], "no unknowns left; flip any face-down card"
        first = self.round_first
        partner = self.partner_of(first)
        if partner:
            return partner, "partner of %s (guaranteed match)" % first
        for c in legal:
            if c != first and self._state_of(c) == "unknown":
                return c, "explore unknown card"
        for c in legal:
            if c != first:
                return c, "no unknowns left; flip any face-down card"
        return None, "no legal second card"

    def status_block(self):
        lines = []
        pairs_left = self.remaining_pairs if self.remaining_pairs is not None else (
            self.total_pairs - self.score)
        if self.round_first is None:
            round_desc = "awaiting first card of a new round"
        else:
            tok = self.cells.get(self.round_first, {}).get("token")
            round_desc = "awaiting second card (first=%s%s)" % (
                self.round_first, (" tok=" + tok) if tok else "")
        lines.append("flips %d/%d | score %d | pairs left %d | %s"
                     % (self.flips_used, self.flip_budget, self.score, pairs_left, round_desc))
        face_up = [c for c in self.cells if self._state_of(c) == "face_up"]
        if face_up:
            lines.append("face-up now: " + " ".join(
                "%s(%s)" % (c, self.cells[c]["token"] or "?") for c in face_up))
        pairs = self.known_pairs()
        if pairs:
            shown = ["%s+%s" % (fd[0], fd[1]) for _, fd in pairs[:8]]
            extra = len(pairs) - len(shown)
            lines.append("ready pairs (flip one, then its partner): " + "  ".join(shown)
                         + (" (+%d more)" % extra if extra > 0 else ""))
        singles = [c for c in self.cells
                   if self._state_of(c) == "known" and not self.partner_of(c)]
        if singles:
            shown = ["%s(%s)" % (c, self.cells[c]["token"] or "?") for c in singles[:10]]
            extra = len(singles) - len(shown)
            lines.append("known face-down singles: " + " ".join(shown)
                         + (" (+%d more)" % extra if extra > 0 else ""))
        unknown = sum(1 for c in self.all_coords() if self._state_of(c) == "unknown")
        removed = sum(1 for c in self.all_coords() if self._state_of(c) == "removed")
        lines.append("unknown: %d | removed: %d | pixel tracking: %s"
                     % (unknown, removed, "on" if self.pixel_enabled else "off"))
        if self.warnings:
            lines.append("warnings: " + " | ".join(self.warnings[-3:]))
        coord, reason = self.suggest()
        if coord:
            lines.append("SUGGESTED NEXT FLIP: %s (%s)" % (coord, reason))
        else:
            lines.append("SUGGESTED NEXT FLIP: none (%s)" % reason)
        return "\n".join(lines)

    # ------------------------------------------------------------------ tools
    def find_match(self, coordinate):
        if not self._valid(coordinate):
            return "coordinate %r is not on the board (rows %d, cols %d)" % (
                coordinate, self.rows, self.cols)
        cell = self.cells.get(coordinate)
        if not cell or not cell.get("token"):
            return "%s has no recorded picture yet" % coordinate
        others = [c for c in self.tokens.get(cell["token"], {}).get("cells", [])
                  if c != coordinate]
        if not others:
            return "%s (tok %s): no other cell with this picture is known yet" % (
                coordinate, cell["token"])
        desc = ", ".join("%s [%s]" % (c, self._state_of(c)) for c in others)
        return "%s (tok %s) matches: %s" % (coordinate, cell["token"], desc)

    def compare(self, a, b):
        for coord in (a, b):
            if not self._valid(coord):
                return "coordinate %r is not on the board" % coord
        if a == b:
            return "same coordinate"
        ca, cb = self._get_crop(a), self._get_crop(b)
        if ca is None or cb is None:
            missing = [c for c in (a, b) if self._get_crop(c) is None]
            return "no pixel data for %s (flip it first)" % ", ".join(missing)
        d = aligned_mad(ca, cb, self.max_shift)
        verdict = "IDENTICAL (same picture)" if d < IDENTICAL_MAD else "DIFFERENT"
        return "%s vs %s: aligned pixel distance %.1f -> %s" % (a, b, d, verdict)

    def set_cell_state(self, coordinate, state, note=None):
        if not self._valid(coordinate):
            return "coordinate %r is not on the board" % coordinate
        if state not in ("unknown", "known", "face_up", "removed"):
            return "state must be one of unknown, known, face_up, removed"
        cell = self._cell(coordinate)
        old = cell["state"]
        if state == "unknown":
            self._remove_token(coordinate)
            cell["state"] = "unknown"
            cell["sig"] = None
        else:
            cell["state"] = state
        if note:
            cell["note"] = str(note)[:200]
        self._persist()
        self._write_files()
        return "corrected %s: %s -> %s%s" % (
            coordinate, old, state, (" note=%r" % cell["note"]) if cell["note"] else "")

    def clear_fingerprint(self, coordinate):
        if not self._valid(coordinate):
            return "coordinate %r is not on the board" % coordinate
        cell = self._cell(coordinate)
        old_token = cell.get("token")
        self._remove_token(coordinate)
        cell["sig"] = None
        self._persist()
        self._write_files()
        return "cleared fingerprint of %s (was %s)" % (coordinate, old_token or "none")

    # ------------------------------------------------------------------ files
    def _write_files(self):
        try:
            lines = ["# auto-tracked board state (one line per cell; LLM-readable)"]
            for c in self.all_coords():
                cell = self._cell(c)
                line = "%s %s" % (c, cell["state"])
                if cell.get("token"):
                    line += " tok=%s" % cell["token"]
                if cell.get("note"):
                    line += " note=%s" % cell["note"]
                lines.append(line)
            with open(os.path.join(self.memory_root, "board_state.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            idx = ["# fingerprint index: token -> cells"]
            for tok in sorted(self.tokens):
                idx.append("%s: %s" % (tok, " ".join(self.tokens[tok]["cells"])))
            with open(os.path.join(self.memory_root, "match_index.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(idx) + "\n")
            with open(os.path.join(self.memory_root, "round_log.txt"), "w", encoding="utf-8") as f:
                f.write("# auto-recorded round outcomes: R<number>: <first>+<second> MATCH|NOMATCH|INVALID\n"
                        + "".join(line + "\n" for line in self.round_log))
        except Exception:
            pass

    def _persist(self):
        try:
            with open(os.path.join(self.board_dir, "board_state.json"), "w", encoding="utf-8") as f:
                json.dump(self.export_state(), f, ensure_ascii=False)
        except Exception:
            pass

    # ------------------------------------------------------------------ state
    def export_state(self):
        return {
            "rows": self.rows, "cols": self.cols, "total_pairs": self.total_pairs,
            "geom": self.geom,
            "bg_color": (self.bg_color.tolist()
                         if self.bg_color is not None else None),
            "interior_shape": (list(self.interior_shape)
                               if self.interior_shape is not None else None),
            "back_sig": self.back_sig,
            "crop_h": self.crop_h, "crop_w": self.crop_w,
            "max_shift": self.max_shift,
            "cells": {c: {"state": cell["state"], "token": cell["token"],
                          "sig": cell["sig"], "note": cell["note"], "seen": cell["seen"]}
                      for c, cell in self.cells.items()},
            "tokens": {t: {"rep": e["rep"], "cells": list(e["cells"])}
                       for t, e in self.tokens.items()},
            "round_first": self.round_first,
            "flips_used": self.flips_used, "flip_budget": self.flip_budget,
            "score": self.score, "remaining_pairs": self.remaining_pairs,
            "warnings": list(self.warnings), "step": self.step,
            "warn_total": self._warn_total,
            "warns_reviewed": self._warns_reviewed,
            "initialized": self.initialized, "pixel_enabled": self.pixel_enabled,
            "round_log": list(self.round_log), "round_no": self.round_no,
        }

    def import_state(self, state):
        if not isinstance(state, dict):
            return
        self.rows = int(state.get("rows", self.rows))
        self.cols = int(state.get("cols", self.cols))
        self.total_pairs = int(state.get("total_pairs", self.total_pairs))
        self.geom = state.get("geom")
        ish = state.get("interior_shape")
        self.interior_shape = (tuple(int(v) for v in ish)
                               if ish is not None else None)
        bg = state.get("bg_color")
        self.bg_color = (np.asarray(bg, dtype=np.int32)
                         if bg is not None else None)
        self.back_sig = state.get("back_sig")
        self.crop_h = int(state.get("crop_h", self.crop_h))
        self.crop_w = int(state.get("crop_w", self.crop_w))
        self.max_shift = int(state.get("max_shift", self.max_shift))
        self.cells = {}
        for c, cell in (state.get("cells") or {}).items():
            self.cells[c] = {"state": cell.get("state", "unknown"),
                             "token": cell.get("token"), "sig": cell.get("sig"),
                             "note": cell.get("note", ""), "seen": int(cell.get("seen", 0))}
        self.tokens = {}
        for t, e in (state.get("tokens") or {}).items():
            self.tokens[t] = {"rep": e.get("rep"), "cells": list(e.get("cells", []))}
        self.round_first = state.get("round_first")
        self.flips_used = int(state.get("flips_used", 0))
        self.flip_budget = int(state.get("flip_budget", 360))
        self.score = int(state.get("score", 0))
        self.remaining_pairs = state.get("remaining_pairs")
        self.warnings = list(state.get("warnings", []))[-20:]
        self._warn_total = int(state.get("warn_total", len(self.warnings)))
        self._warns_reviewed = int(state.get("warns_reviewed", 0))
        self.step = int(state.get("step", 0))
        self.initialized = bool(state.get("initialized", False))
        self.pixel_enabled = bool(state.get("pixel_enabled", False))
        self.round_log = [str(x) for x in (state.get("round_log") or [])][-ROUND_LOG_CAP:]
        self.round_no = int(state.get("round_no", 0))
        self._crops = {}
        if self.pixel_enabled and self.back_crop is None:
            self._load_back()

    def reset(self):
        self.rows, self.cols, self.total_pairs = 12, 12, 72
        self.geom = None
        self.bg_color = None
        self.interior_shape = None
        self.back_crop = None
        self.back_sig = None
        self.crop_h, self.crop_w, self.max_shift = 68, 68, MAX_SHIFT
        self.cells = {}
        self.tokens = {}
        self.round_first = None
        self.flips_used = 0
        self.flip_budget = 360
        self.score = 0
        self.remaining_pairs = None
        self.warnings = []
        self._warn_total = 0
        self._warns_reviewed = 0
        self.step = 0
        self.initialized = False
        self.pixel_enabled = False
        self._crops = {}
        self._prev_second = None
        self.round_log = []
        self.round_no = 0
