"""Painting pixel memory for corridor-gallery episodes.

The inner LLM decides when to store and which doorway number each painting
belongs to (it reads the central doorway sign from the image).  This module
only performs deterministic computation on public observations — detection,
frame-trimmed reference crops, colour and structure (grayscale) features,
rotation search and group ranking — and keeps a plain-dict, JSON-serialisable
store on ``agent.paintings``.  No decision is made automatically: every stored
claim carries its label, provenance and similarity evidence, and every match
attaches a same-scale comparison image, so the LLM can inspect, correct or
override it.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

# Feature: resize to a small square, flatten RGB, mean-subtract, L2-normalise.
# The size tracks the intrinsic pixel-art block grid of the level format, not
# any particular episode.
FEATURE_SIZE = 30
ROTATION_ANGLES = (-9, -6, -3, 0, 3, 6, 9)
SATURATION_THRESHOLD = 40

# Walk-frame pair extraction (first-person corridor view).
WALK_MIN_SIDE = 90
WALK_MIN_AREA = 2000
CLEAN_ASPECT = 0.93
CLEAN_FILL = 0.88
RECOVER_FILL = 0.85

# Question-frame single-panel detection (re-photographed painting on gray).
QUESTION_MIN_SIDE = 100
QUESTION_MIN_AREA = 5000
QUESTION_ASPECT = 0.9
# The question panel is centered on its background; flanking paintings in a
# corridor view hang off-center (measured center offsets >= 0.13 of the half
# width, panels <= 0.01).  This is the size-independent discriminator.
QUESTION_MAX_CENTER_OFFSET = 0.10

# Decision guidance shown to the LLM (advisory; the LLM commits the answer).
# Calibrated on both development units: best scores of true matches
# 0.959-0.997 with gaps 0.021-0.075; best scores of unseen (lure) paintings
# 0.912-0.942 with gaps <= 0.001.
SEEN_MIN_SCORE = 0.95
SEEN_MIN_GAP = 0.01
# A new reference this similar to one in a DIFFERENT group is almost certainly
# the same physical painting mislabelled (cross-group similarity of different
# paintings measured <= 0.948 on both development units).
SAME_PAINTING_SIM = 0.97

# Border trim used to remove the thin frame around a detected painting so the
# stored reference is the canvas itself (what question panels show).  A fixed
# small morphological cleanup, not fitted to any episode.
INNER_TRIM_ITERS = 3

# Attached comparison crops for the LLM's visual check: the question panel and
# the top reference panels, all resized to this side and tiled at the same
# scale so the LLM compares painting to painting directly.
TILE_MAX_SIDE = 360
TILE_COLS = 2
OUTPUT_SIDE_LIMIT = 1536


def _saturation_mask(image):
    arr = np.asarray(image, dtype=np.int16)
    return (arr.max(axis=2) - arr.min(axis=2)) > SATURATION_THRESHOLD


def _blobs(mask, min_area):
    labels, count = ndimage.label(mask)
    out = []
    for index in range(1, count + 1):
        ys, xs = np.where(labels == index)
        if len(xs) < min_area:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1
        out.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "w": w, "h": h, "area": int(len(xs)),
            "aspect": min(w, h) / max(w, h),
            "fill": len(xs) / (w * h),
            "cx": (x0 + x1) / 2.0,
        })
    return out


def _box_of(blob):
    return (blob["x0"], blob["y0"], blob["x1"], blob["y1"])


def _clamp_box(box, width, height):
    x0 = max(0, min(box[0], width - 2))
    y0 = max(0, min(box[1], height - 2))
    x1 = max(x0 + 1, min(box[2], width - 1))
    y1 = max(y0 + 1, min(box[3], height - 1))
    return (x0, y0, x1, y1)


def _box_fill(mask, box):
    sub = mask[box[1]:box[3] + 1, box[0]:box[2] + 1]
    return float(sub.mean()) if sub.size else 0.0


def inner_box(image, box, iters=INNER_TRIM_ITERS):
    """Trim the thin frame border around a detected painting.

    The detected box is the bounding box of the saturated blob, which includes
    a thin frame border a few pixels wide.  Eroding the saturation mask and
    taking the largest connected component returns the painting canvas itself,
    which is what the question panels show (canvas only, no frame).  If the
    erosion leaves nothing, the original box is returned unchanged.
    """
    sub = image.crop((box[0], box[1], box[2] + 1, box[3] + 1))
    mask = _saturation_mask(sub)
    eroded = ndimage.binary_erosion(mask, iterations=iters)
    labels, count = ndimage.label(eroded)
    if count == 0:
        return box
    sizes = ndimage.sum(eroded, labels, range(1, count + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labels == biggest)
    return (box[0] + int(xs.min()), box[1] + int(ys.min()),
            box[0] + int(xs.max()), box[1] + int(ys.max()))


def _is_clean(blob):
    return (blob["aspect"] > CLEAN_ASPECT and blob["fill"] > CLEAN_FILL
            and min(blob["w"], blob["h"]) >= WALK_MIN_SIDE)


def _recover_partner(mask, junk, clean, side):
    """One side's painting is merged with foreground (e.g. the held block).

    Derive its box from the clean partner's size and the junk blob's far
    edge; accept only if the boxes overlap substantially and the derived box
    is mostly saturated pixels.
    """
    if side == "right":
        cand = (junk["x0"], clean["y0"], junk["x0"] + clean["w"] - 1,
                clean["y0"] + clean["h"] - 1)
    else:
        cand = (junk["x1"] - clean["w"] + 1, clean["y0"], junk["x1"],
                clean["y0"] + clean["h"] - 1)
    ox = min(cand[2], junk["x1"]) - max(cand[0], junk["x0"]) + 1
    oy = min(cand[3], junk["y1"]) - max(cand[1], junk["y0"]) + 1
    if ox > 0.5 * clean["w"] and oy > 0.5 * clean["h"]:
        if _box_fill(mask, cand) > RECOVER_FILL:
            return cand
    return None


def extract_walk_pair(image):
    """Detect the paintings flanking the central doorway of a corridor view.

    Returns (left_box, right_box, notes); either box may be None.  Edge-touching
    and center-spanning blobs are excluded: cut-off near-wall paintings and the
    central doorway sign/foreground are not flanking paintings.
    """
    width, height = image.size
    center = width / 2.0
    mask = _saturation_mask(image)
    blobs = [b for b in _blobs(mask, WALK_MIN_AREA)
             if b["x0"] > 5 and b["x1"] < width - 6
             and not (b["x0"] < center - 20 < b["x1"])
             and min(b["w"], b["h"]) >= WALK_MIN_SIDE]
    left = [b for b in blobs if b["cx"] < center]
    right = [b for b in blobs if b["cx"] > center]

    def _largest_clean(side_blobs):
        best = None
        for b in side_blobs:
            if _is_clean(b) and (best is None or b["area"] > best["area"]):
                best = b
        return best

    notes = []
    left_clean = _largest_clean(left)
    right_clean = _largest_clean(right)
    left_box = _box_of(left_clean) if left_clean else None
    right_box = _box_of(right_clean) if right_clean else None
    if left_box and right_box:
        return left_box, right_box, notes
    if left_box:
        junk = max(right, key=lambda b: b["area"], default=None)
        if junk is not None:
            cand = _recover_partner(mask, junk, left_clean, "right")
            if cand is not None:
                right_box = cand
                notes.append("right painting merged with foreground; box recovered from left partner")
        return left_box, right_box, notes
    if right_box:
        junk = max(left, key=lambda b: b["area"], default=None)
        if junk is not None:
            cand = _recover_partner(mask, junk, right_clean, "left")
            if cand is not None:
                left_box = cand
                notes.append("left painting merged with foreground; box recovered from right partner")
        return left_box, right_box, notes
    return None, None, notes


def extract_question(image):
    """Detect the single re-photographed painting panel of a question frame.

    A question frame shows exactly one centered, near-square, well-filled
    panel; corridor views have their paintings off to the sides, so the
    center-offset requirement separates the two regardless of frame size.
    """
    width, height = image.size
    center = width / 2.0
    mask = _saturation_mask(image)
    blobs = [b for b in _blobs(mask, QUESTION_MIN_AREA)
             if b["aspect"] > QUESTION_ASPECT and min(b["w"], b["h"]) >= QUESTION_MIN_SIDE
             and abs(b["cx"] - center) <= QUESTION_MAX_CENTER_OFFSET * center]
    if not blobs:
        return None
    best = max(blobs, key=lambda b: b["area"])
    return _box_of(best)


def _feature_from_crop(crop, mode="rgb"):
    """Resize to a small square, flatten, mean-subtract, L2-normalise.

    mode="rgb" keeps colour (palette + pattern); mode="gray" keeps only the
    luminance pattern (structure without palette).  Both are computed for
    every reference and query so the LLM gets two independent similarity
    signals.
    """
    if mode == "rgb":
        small = crop.resize((FEATURE_SIZE, FEATURE_SIZE), Image.BILINEAR)
        vec = np.asarray(small, dtype=np.float64).reshape(-1)
    else:
        small = crop.convert("L").resize((FEATURE_SIZE, FEATURE_SIZE), Image.BILINEAR)
        vec = np.asarray(small, dtype=np.float64).reshape(-1)
    vec = vec - vec.mean()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def feature(image, box, mode="rgb"):
    crop = image.crop((box[0], box[1], box[2] + 1, box[3] + 1))
    return _feature_from_crop(crop, mode)


def features_rotated(image, box, mode="rgb"):
    """The question painting may be slightly rotated; score all candidate
    orientations against each stored reference."""
    crop = image.crop((box[0], box[1], box[2] + 1, box[3] + 1))
    fill = (90, 90, 90) if mode == "rgb" else 90
    return [_feature_from_crop(crop.rotate(angle, resample=Image.BILINEAR,
                                           fillcolor=fill), mode)
            for angle in ROTATION_ANGLES]


def ensure_store(agent):
    store = getattr(agent, "paintings", None)
    if not isinstance(store, dict) or not isinstance(store.get("refs"), dict):
        store = {"next_id": 1, "refs": {}}
        agent.paintings = store
    if "next_id" not in store:
        store["next_id"] = 1
    return store


def index_text(store):
    """Compact one-block summary shown in both phase prompts."""
    refs = store.get("refs", {})
    if not refs:
        return ""
    groups = {}
    for ref in refs.values():
        groups.setdefault((ref["number"], ref["side"]), []).append(ref)
    parts = []
    for number, side in sorted(groups):
        steps = sorted({r["step"] for r in groups[(number, side)]})
        if len(steps) == 1:
            parts.append("{}{} (step {})".format(number, side, steps[0]))
        else:
            parts.append("{}{} (steps {}-{})".format(number, side, steps[0], steps[-1]))
    doorways = sorted({r["number"] for r in refs.values()})
    return ("PAINTING MEMORY (pixel references stored via LOG_PAINTINGS):\n"
            "  " + ", ".join(parts) + "\n"
            "  {} reference(s) across doorways {}. For recall questions call "
            "MATCH_PAINTING and use its ranking plus the attached images.".format(
                len(refs), doorways))


def save_flanking(agent, number):
    """LOG_PAINTINGS handler: store both flanking paintings of doorway `number`."""
    image = getattr(agent, "current_image", None)
    if image is None:
        raise ValueError("no current observation image")
    store = ensure_store(agent)
    width, height = image.size
    left_box, right_box, notes = extract_walk_pair(image)
    if left_box is None and right_box is None:
        if extract_question(image) is not None:
            return ("current frame looks like a recall-question frame (a single centered "
                    "painting panel), not a corridor view; nothing saved.")
        extra = (" Notes: " + "; ".join(notes) + ".") if notes else ""
        return ("no painting pair detected in this frame (paintings may be out of "
                "view or cut off at the frame edges); nothing saved." + extra)
    step = int(getattr(agent, "step_count", 0) or 0)
    frame_id = getattr(agent, "current_frame_id", None)
    saved, kept, superseded, conflicts = [], [], [], []
    for side, box in (("L", left_box), ("R", right_box)):
        if box is None:
            continue
        box = _clamp_box(box, width, height)
        inner = inner_box(image, box)
        vec = feature(image, inner)
        gray_vec = feature(image, inner, mode="gray")
        best_ref, best_ref_id, best_sim = None, None, -2.0
        for ref_id, ref in store["refs"].items():
            sim = float(np.dot(vec, np.asarray(ref["feature"], dtype=np.float64)))
            if sim > best_sim:
                best_sim, best_ref, best_ref_id = sim, ref, ref_id
        if best_ref is not None and best_sim >= SAME_PAINTING_SIM:
            if (best_ref["number"], best_ref["side"]) == (number, side):
                new_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
                if new_area >= best_ref["w"] * best_ref["h"]:
                    del store["refs"][best_ref_id]
                    superseded.append(
                        "{}{} replaced the smaller reference from step {}".format(
                            number, side, best_ref["step"]))
                else:
                    kept.append(
                        "{}{} already stored (best reference from step {}); kept it".format(
                            number, side, best_ref["step"]))
                    continue
            else:
                conflicts.append(
                    "the {} painting matches the one stored as {}{} (similarity {:.2f}, "
                    "saved at step {}) — that is the same physical painting under a "
                    "different label. Re-read the number on the central sign and, if "
                    "you misread it, call LOG_PAINTINGS again with the corrected "
                    "number; nothing was saved for this label.".format(
                        side, best_ref["number"], best_ref["side"], best_sim, best_ref["step"]))
                continue
        ref_id = "p_{:04d}".format(store["next_id"])
        store["refs"][ref_id] = {
            "step": step,
            "frame_id": frame_id,
            "number": int(number),
            "side": side,
            "box": [int(v) for v in inner],
            "full_box": [int(v) for v in box],
            "w": int(inner[2] - inner[0] + 1),
            "h": int(inner[3] - inner[1] + 1),
            # rounded: keeps the JSON state small; cosine error <= ~3e-3,
            # far below the 0.021 smallest observed seen/lure margin
            "feature": [round(float(v), 5) for v in vec],
            "gray_feature": [round(float(v), 5) for v in gray_vec],
        }
        store["next_id"] += 1
        saved.append("{}{} (step {}, {}x{} px)".format(
            number, side, step, store["refs"][ref_id]["w"], store["refs"][ref_id]["h"]))
    lines = ["LOG_PAINTINGS(number={}):".format(number)]
    for item in saved:
        lines.append("  saved " + item)
    for item in kept:
        lines.append("  " + item)
    for item in superseded:
        lines.append("  " + item)
    for item in conflicts:
        lines.append("  CONFLICT: " + item)
    for note in notes:
        lines.append("  note: " + note)
    lines.append(index_text(store))
    return "\n".join(lines)


def _tile_program(agent, question_box, top):
    """Host visual program: the question panel first, then the top reference
    panels, all resized to the same side and tiled so the LLM compares
    painting to painting at one scale."""
    ops, srcs, labels = [], [], []
    frame_id = getattr(agent, "current_frame_id", None)
    if (frame_id and question_box and question_box[2] > question_box[0]
            and question_box[3] > question_box[1]):
        # stored boxes are inclusive; the host crop op is right/bottom exclusive
        box = [int(question_box[0]), int(question_box[1]),
               int(question_box[2]) + 1, int(question_box[3]) + 1]
        ops.append({"op": "frame", "id": frame_id, "as": "fq"})
        ops.append({"op": "crop", "src": "fq", "box": box, "as": "cq"})
        ops.append({"op": "resize", "src": "cq", "max_side": TILE_MAX_SIDE, "as": "rq"})
        srcs.append("rq")
        labels.append("QUESTION (current image)")
    for rank, (key, entry) in enumerate(top, start=1):
        ref = entry["ref"]
        ref_frame = ref.get("frame_id")
        if not ref_frame:
            continue
        box = [int(ref["box"][0]), int(ref["box"][1]),
               int(ref["box"][2]) + 1, int(ref["box"][3]) + 1]
        if (box[2] <= box[0] or box[3] <= box[1]):
            continue
        number, side = key
        fa, ca, ra = "f{}".format(rank), "c{}".format(rank), "r{}".format(rank)
        ops.append({"op": "frame", "id": ref_frame, "as": fa})
        ops.append({"op": "crop", "src": fa, "box": box, "as": ca})
        ops.append({"op": "resize", "src": ca, "max_side": TILE_MAX_SIDE, "as": ra})
        srcs.append(ra)
        labels.append(("{}: {}{} {:.3f} step {}".format(
            rank, number, side, entry["score"], ref["step"]))[:40])
    if not srcs:
        return None
    cols = min(TILE_COLS, len(srcs))
    while cols > 1 and TILE_MAX_SIDE * cols > OUTPUT_SIDE_LIMIT:
        cols -= 1
    ops.append({"op": "tile", "srcs": srcs, "cols": cols, "labels": labels, "as": "out"})
    return {"ops": ops, "output": "out"}


def match_question(agent, top_k=3):
    """MATCH_PAINTING handler: rank stored (doorway, side) groups against the
    question painting using colour and structure (grayscale) similarity, and
    attach a same-scale comparison image (question panel plus top reference
    panels) for the LLM's visual check."""
    image = getattr(agent, "current_image", None)
    if image is None:
        raise ValueError("no current observation image")
    box = extract_question(image)
    if box is None:
        return ("no single painted panel detected in the current frame, so there is no "
                "question painting to match. If a question is displayed, answer from the "
                "current image; otherwise continue the task.")
    store = ensure_store(agent)
    refs = store.get("refs", {})
    if not refs:
        return ("question painting detected, but no corridor painting references are "
                "stored yet. Answer from your current image and notes; if you saw "
                "paintings during the walk, they were never saved.")
    query_vecs = features_rotated(image, box)
    query_gray_vecs = features_rotated(image, box, mode="gray")
    groups = {}
    for ref_id, ref in refs.items():
        ref_vec = np.asarray(ref["feature"], dtype=np.float64)
        best = max(float(np.dot(q, ref_vec)) for q in query_vecs)
        gray_best = None
        gray_ref = ref.get("gray_feature")
        if gray_ref is not None:
            gray_ref_vec = np.asarray(gray_ref, dtype=np.float64)
            gray_best = max(float(np.dot(q, gray_ref_vec)) for q in query_gray_vecs)
        key = (ref["number"], ref["side"])
        entry = groups.get(key)
        if entry is None or best > entry["score"]:
            groups[key] = {"score": best, "gray_score": gray_best,
                           "ref_id": ref_id, "ref": ref}
    ranked = sorted(groups.items(), key=lambda item: (-item[1]["score"], item[0][0], item[0][1]))
    count = max(1, min(int(top_k), len(ranked)))
    top = ranked[:count]
    lines = ["MATCH_PAINTING:"]
    lines.append("  question painting detected ({}x{} px crop)".format(
        box[2] - box[0] + 1, box[3] - box[1] + 1))
    lines.append("  ranked {} stored group(s) by colour similarity:".format(len(groups)))
    for rank, (key, entry) in enumerate(top, start=1):
        number, side = key
        ref = entry["ref"]
        line = "  {}. doorway {} side {}: colour {:.3f}".format(
            rank, number, side, entry["score"])
        if entry["gray_score"] is not None:
            line += ", structure {:.3f}".format(entry["gray_score"])
        line += " (best reference step {}, {}x{} px)".format(ref["step"], ref["w"], ref["h"])
        if rank == 1 and len(ranked) > 1:
            line += ", gap {:.3f}".format(entry["score"] - ranked[1][1]["score"])
        lines.append(line)
    best_key, best_entry = top[0]
    best_score = best_entry["score"]
    gap = best_score - top[1][1]["score"] if len(top) > 1 else best_score
    if best_score >= SEEN_MIN_SCORE and gap >= SEEN_MIN_GAP:
        lines.append("  Recommendation: SEEN — best match is doorway {} side {} "
                     "(colour {:.3f}, gap {:.3f}); answer {},{}, unless the attached "
                     "comparison shows the pattern is different.".format(
                         best_key[0], best_key[1], best_score, gap,
                         best_key[0], best_key[1]))
        if (best_entry["gray_score"] is not None
                and best_entry["gray_score"] >= 0.5 * best_score):
            lines.append("  The structure score ({:.3f}) corroborates the colour "
                         "match: same palette and same block pattern.".format(
                             best_entry["gray_score"]))
    else:
        lines.append("  Recommendation: not_seen — the best match is weak or ambiguous "
                     "(colour {:.3f}, gap {:.3f}). Colour scores below {:.2f} are too "
                     "close to decide by colour alone, because unseen paintings can "
                     "share a similar palette: compare the block PATTERN of the "
                     "question with every reference in the attached comparison "
                     "(orientation of strips, where blobs sit); if the question is "
                     "not the same painting as any of them, answer not_seen.".format(
                         best_score, gap, SEEN_MIN_SCORE))
        if (best_entry["gray_score"] is not None
                and best_entry["gray_score"] < 0.5 * best_score):
            lines.append("  The top-1 structure score ({:.3f}) is far below its colour "
                         "score ({:.3f}): similar palette but different block pattern "
                         "— the signature of an unseen painting. Do not answer a "
                         "doorway unless the comparison image shows an exact pattern "
                         "match.".format(best_entry["gray_score"], best_score))
    lines.append("  The attached image shows the question panel next to the top "
                 "reference panels, all at the same scale; your own inspection of "
                 "it has the final say.")
    program = _tile_program(agent, box, top)
    if program is not None:
        agent.visual_request = program
        lines.append("  (comparison image is attached to your next message)")
    return "\n".join(lines)


def gallery_cue(agent, phase):
    """Advisory line computed from the CURRENT frame and the painting store.

    Structural change: the same detectors the painting tools use classify the
    current frame (both flanking paintings fully visible / one side / none /
    single centered question panel) BEFORE the LLM spends a request, so the
    LLM is told what a LOG_PAINTINGS call on this frame would do and whether
    the frame is a recall-question frame.  This is derived context only: the
    cue never reads the doorway sign number, never writes to memory, and
    never decides — the LLM still issues every tool call and commits every
    action and answer.  Returns "" on any failure (advisory must not break
    the phase).  No development-episode constants: only detector outputs on
    this episode's own frame plus this episode's own store state.
    """
    try:
        image = getattr(agent, "current_image", None)
        if image is None or not hasattr(image, "size"):
            return ""
        store = getattr(agent, "paintings", None)
        refs = store.get("refs") if isinstance(store, dict) else None
        if not isinstance(refs, dict):
            refs = {}
        sides_by_number = {}
        for ref in refs.values():
            sides_by_number.setdefault(int(ref["number"]), set()).add(ref["side"])
        if sides_by_number:
            bits = []
            for number in sorted(sides_by_number):
                if sides_by_number[number] == {"L", "R"}:
                    bits.append("{} (both)".format(number))
                else:
                    bits.append("{} ({} only)".format(
                        number, next(iter(sides_by_number[number]))))
            stored = ", ".join(bits)
        else:
            stored = "none yet"
        left_box, right_box, _notes = extract_walk_pair(image)
        if left_box is None and right_box is None and extract_question(image) is not None:
            if phase == "LOG":
                return ("GALLERY CUE: this frame shows a single centered painting "
                        "panel — a recall-question frame, not a corridor view. "
                        "There is nothing to log; output <|DONE|> without any other "
                        "command.")
            return ("GALLERY CUE: this frame is a recall-question frame (a single "
                    "centered painting panel). Call <|MATCH_PAINTING|>{}<|END|>, "
                    "inspect the ranked result and the attached comparison, and "
                    "commit the answer exactly as the question instructs.")
        if left_box is not None and right_box is not None:
            lw, lh = left_box[2] - left_box[0] + 1, left_box[3] - left_box[1] + 1
            rw, rh = right_box[2] - right_box[0] + 1, right_box[3] - right_box[1] + 1
            if phase == "LOG":
                return ("GALLERY CUE: both flanking paintings are fully visible in "
                        "this frame (left {}x{} px, right {}x{} px). Doorways "
                        "already stored: {}. Read the number on the central sign: "
                        "if it is listed as (both), that doorway was already "
                        "stored — output <|DONE|> alone; otherwise output the "
                        "<|LOG_PAINTINGS|> call with that number together with "
                        "<|DONE|> in this same response.".format(
                            lw, lh, rw, rh, stored))
            return ("GALLERY CUE: both flanking paintings are fully visible in this "
                    "frame (left {}x{} px, right {}x{} px). Doorways already stored: "
                    "{}. If the central sign's number is listed as (both), commit "
                    "move_forward now (that doorway was already stored). If it "
                    "shows one side only or is not in the list (at step 0 there is "
                    "no LOG phase, or a save was missed), call LOG_PAINTINGS once "
                    "with that number, then commit move_forward.".format(
                        lw, lh, rw, rh, stored))
        if left_box is not None or right_box is not None:
            side = "left" if left_box is not None else "right"
            box = left_box if left_box is not None else right_box
            w, h = box[2] - box[0] + 1, box[3] - box[1] + 1
            if phase == "LOG":
                return ("GALLERY CUE: only the {} flanking painting is fully "
                        "visible in this frame ({}x{} px); the other side is cut "
                        "off or out of view. Doorways already stored: {}. If the "
                        "PAINTING MEMORY index does not yet list {} for the "
                        "central sign's number, call <|LOG_PAINTINGS|> with that "
                        "number and then output <|DONE|>; otherwise just output "
                        "<|DONE|>.".format(
                            side, w, h, stored, "L" if side == "left" else "R"))
            return ("GALLERY CUE: only the {} flanking painting is fully visible "
                    "in this frame ({}x{} px). Doorways already stored: {}. If the "
                    "central sign's number is listed as (both), commit "
                    "move_forward; otherwise call LOG_PAINTINGS once with that "
                    "number, then commit move_forward.".format(side, w, h, stored))
        if phase == "LOG":
            return ("GALLERY CUE: no flanking painting is fully visible in this "
                    "frame (cut off at the frame edges, out of view, or the "
                    "corridor may have ended). LOG_PAINTINGS would save nothing — "
                    "output <|DONE|> without calling it.")
        return ("GALLERY CUE: no flanking painting is fully visible in this frame "
                "(cut off, out of view, or the corridor may have ended). Do not "
                "call LOG_PAINTINGS; keep walking with move_forward until a solid "
                "wall ends the corridor.")
    except Exception:
        return ""


def install_paintings(agent, registry):
    """Register LOG_PAINTINGS and MATCH_PAINTING; keep agent state plain-dict."""
    from registry import Tool
    if not isinstance(getattr(agent, "paintings", None), dict):
        agent.paintings = {"next_id": 1, "refs": {}}

    def log_paintings(number):
        return save_flanking(agent, number)

    def match_painting(top_k=3):
        return match_question(agent, top_k)

    obj = {"type": "object", "additionalProperties": False}
    registry.register(Tool(
        "LOG_PAINTINGS",
        "store pixel references of the two paintings flanking the doorway whose number is shown on the sign in the CENTER of your view; pass that number; call once per doorway while walking (it detects both flanking paintings itself and reports label conflicts to re-check)",
        {**obj, "properties": {"number": {"type": "integer", "minimum": 1,
                                          "maximum": 999}},
         "required": ["number"]},
        log_paintings, ("LOG", "PLAN"), "memory_write", "text"))
    registry.register(Tool(
        "MATCH_PAINTING",
        "compare the painting in the current question image against all stored corridor painting references; returns a ranked list of (doorway, side) candidates with colour and structure (grayscale) match scores, plus a recommendation, and attaches one comparison image showing the question panel next to the top reference panels at the same scale; use it to answer recall questions",
        {**obj, "properties": {"top_k": {"type": "integer", "minimum": 1,
                                         "maximum": 9}},
         "required": []},
        match_painting, ("PLAN",), "memory_read", "text"))
