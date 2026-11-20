"""M-line poster tools: derived map, sighting index, numeric matching.

Pixels reach the model only through agent.visual_request, which the host renders
from its own frame copies, and only inside the handlers below (model-invoked).
Tools return metadata; the LLM decides what to name, note, show and answer.
Sightings are indexed by the host from maze frames only (see poster_memory_agent).
"""
from __future__ import annotations

from registry import Tool
from agents import poster_map as pm
from agents import poster_vision as pv

MAX_NOTE = 300
MAX_LIST = 30
MAX_RANK = 10
DUPLICATE_SCORE = 0.995  # descriptors this alike are the same recorded view (float noise)


def _clamp(value, low, high, default):
    if value is None:
        return default
    return max(low, min(high, int(value)))


def _pad(box, size, frac=0.04):
    x0, y0, x1, y1 = box
    w, h = size
    px = int((x1 - x0) * frac) + 2
    py = int((y1 - y0) * frac) + 2
    return [max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)]


def _answer_hint(agent, sighting: dict):
    """Derived 'r,c,D' from the sighting's pose and the current wall map; a checkable hint."""
    pos = sighting.get("pos")
    if not pos:
        return None
    return pm.answer_cell(agent.wall_state, pos[0], pos[1], pos[2])


def _public(agent, sighting: dict) -> dict:
    out = {k: v for k, v in sighting.items() if k != "desc"}
    out["answer_hint"] = _answer_hint(agent, sighting)
    return out


def install_poster_tools(agent, registry):
    obj = {"type": "object", "additionalProperties": False}

    def map_tool():
        return pm.summary(agent.wall_state)

    def sightings(limit=None):
        n = _clamp(limit, 1, MAX_LIST, 10)
        recent = agent.sightings[-n:]
        return {"total": len(agent.sightings), "shown": len(recent),
                "entries": [_public(agent, s) for s in recent]}

    def show_sighting(sighting_id, pad=None):
        match = [s for s in agent.sightings if s["id"] == sighting_id]
        if not match:
            raise ValueError("unknown sighting_id")
        s = match[0]
        if not s.get("frame_id"):
            raise ValueError("sighting has no host frame id")
        box = _pad(s["box"], s["size"], frac=_clamp(pad, 0, 40, 4) / 100.0)
        agent.visual_request = {"ops": [
            {"op": "frame", "id": s["frame_id"], "as": "f"},
            {"op": "crop", "src": "f", "box": box, "as": "c"},
            {"op": "resize", "src": "c", "max_side": 512, "as": "out"},
        ], "output": "out"}
        return {"sighting": _public(agent, s), "shown_box": box}

    def match_poster(top=None):
        if not getattr(agent, "question_phase", False):
            raise ValueError("MATCH_POSTER is for question photos; no question has been asked yet")
        image = getattr(agent, "current_image", None)
        frame_id = getattr(agent, "current_frame_id", None)
        if image is None or not frame_id:
            raise ValueError("no host frame is available at this step")
        rec = pv.sight_question(image)
        if rec["kind"] != "poster":
            return {"now": rec, "ranking": [],
                    "note": "question photo not localised by the host detector; no ranking. "
                            "Compare against SIGHTINGS and SHOW_SIGHTING by eye."}
        desc = pv.describe(image, rec["box"])
        ranked = [(pv.similarity(desc, s["desc"])["score"], s) for s in agent.sightings]
        ranked.sort(key=lambda item: -item[0])
        distinct, duplicates = [], 0
        for score, s in ranked:
            if any(pv.similarity(s["desc"], other["desc"])["score"] > DUPLICATE_SCORE for _, other in distinct):
                duplicates += 1
                continue
            distinct.append((score, s))
        k = _clamp(top, 1, 4, 3)
        chosen = distinct[:k]
        size = list(image.size)
        ops = [{"op": "frame", "id": frame_id, "as": "q"},
               {"op": "crop", "src": "q", "box": _pad(rec["box"], size), "as": "qc"},
               {"op": "resize", "src": "qc", "max_side": 320, "as": "qr"}]
        srcs, labels = ["qr"], ["question"]
        for i, (score, s) in enumerate(chosen):
            if not s.get("frame_id"):
                continue
            a = "s%d" % i
            ops += [{"op": "frame", "id": s["frame_id"], "as": a + "f"},
                    {"op": "crop", "src": a + "f", "box": _pad(s["box"], s["size"]), "as": a + "c"},
                    {"op": "resize", "src": a + "c", "max_side": 320, "as": a + "r"}]
            srcs.append(a + "r")
            labels.append("%s %.2f" % (s["id"], score))
        ops.append({"op": "tile", "srcs": srcs, "cols": len(srcs), "labels": labels, "as": "out"})
        agent.visual_request = {"ops": ops, "output": "out"}
        # margin: gap between the best and second-best distinct sighting; a small gap means the
        # shortlist does not single out one view, so the answer should rest on the tile, not the rank
        margin = round(distinct[0][0] - distinct[1][0], 4) if len(distinct) > 1 else None
        return {"now": {"box": rec["box"], "near": rec["near"], "method": rec.get("method")},
                "ranking": [{"id": s["id"], "score": round(score, 4), "step": s["step"],
                             "near": s["near"], "via": s.get("via", "frame"), "pos": s["pos"],
                             "answer_hint": _answer_hint(agent, s), "note": s.get("note", "")}
                            for score, s in distinct[:MAX_RANK]],
                "margin": margin,
                "duplicate_views_skipped": duplicates,
                "shown_tile": labels,
                "caution": "scores are a shortlist from a coarse grid, not an answer; "
                           "compare the attached tile before answering"}

    def note_sighting(sighting_id, note):
        if len(note) > MAX_NOTE:
            raise ValueError("note exceeds %d characters" % MAX_NOTE)
        match = [s for s in agent.sightings if s["id"] == sighting_id]
        if not match:
            raise ValueError("unknown sighting_id")
        match[0]["note"] = note
        return {"sighting_id": sighting_id, "note": note}

    registry.register(Tool(
        "MAP",
        "derived maze map from observation text: ASCII view, visited cells, known walls, "
        "contradictions, and the nearest unexplored edges with paths; no arguments",
        {**obj, "properties": {}, "required": []}, map_tool, ("LOG", "PLAN"), "memory_read", "metadata"))
    registry.register(Tool(
        "SIGHTINGS",
        "list the most recent maze sightings (id, step, box, near, pos, answer_hint, note); "
        "answer_hint is the cell and wall side reached by walking forward from pos over known-open edges, "
        "or null if unknown; it is a hint to verify, not an answer",
        {**obj, "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIST}}, "required": []},
        sightings, ("LOG", "PLAN"), "memory_read", "metadata"))
    registry.register(Tool(
        "SHOW_SIGHTING",
        "show the pixels of one sighting crop in the next model request; pad is percent of box size (0-40)",
        {**obj, "properties": {"sighting_id": {"type": "string"},
                               "pad": {"type": "integer", "minimum": 0, "maximum": 40}},
         "required": ["sighting_id"]}, show_sighting, ("LOG", "PLAN"), "memory_read", "image"))
    registry.register(Tool(
        "MATCH_POSTER",
        "during questions only: rank maze sightings against the current question photo by a coarse "
        "chroma/luminance grid under rotations and flips; attaches a tile of the question and the top "
        "sightings; each ranked entry carries its answer_hint",
        {**obj, "properties": {"top": {"type": "integer", "minimum": 1, "maximum": 4}}, "required": []},
        match_poster, ("LOG", "PLAN"), "memory_read", "image"))
    registry.register(Tool(
        "NOTE_SIGHTING",
        "attach your own verified note (max 300 characters) to a sighting after inspecting its pixels",
        {**obj, "properties": {"sighting_id": {"type": "string"}, "note": {"type": "string"}},
         "required": ["sighting_id", "note"]}, note_sighting, ("LOG", "PLAN"), "memory_write", "metadata"))
