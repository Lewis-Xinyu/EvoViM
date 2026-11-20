"""Model-invoked tools for whole-view memory (line M).

Handlers return JSON-able metadata. The only pixels the model sees are host renders named by
a visual request (frame ids plus crop/resize/tile ops over frames already observed).
"""
from __future__ import annotations

from registry import Tool

from painting_match import STRONG_SCORE
from view_store import DEFAULT_DOOR_X, DOOR_MAX

TILE_SIDE = 360
TOP_TILES = 3
CROP_MARGIN = 0.15


def _crop_box(entry):
    width, height = entry["full_size"]
    x0, y0, x1, y1 = entry["box"]
    mx = int(round((x1 - x0) * CROP_MARGIN))
    my = int(round((y1 - y0) * CROP_MARGIN))
    return [max(0, x0 - mx), max(0, y0 - my), min(width, x1 + mx), min(height, y1 + my)]


def _tile_program(query_frame, entries):
    ops = [{"op": "frame", "id": query_frame, "as": "q"}]
    sources, labels = ["q"], ["question painting"]
    for index, entry in enumerate(entries):
        ops.append({"op": "frame", "id": entry["frame_id"], "as": "r%d" % index})
        ops.append({"op": "crop", "src": "r%d" % index, "box": _crop_box(entry), "as": "c%d" % index})
        ops.append({"op": "resize", "src": "c%d" % index, "max_side": TILE_SIDE, "as": "s%d" % index})
        sources.append("s%d" % index)
        labels.append("%s score %.2f" % (entry["label"], entry["score"]))
    ops.append({"op": "tile", "srcs": sources, "cols": len(sources), "labels": labels, "as": "out"})
    return {"ops": ops, "output": "out"}


def _door(value):
    if type(value) is not int or not (1 <= value <= DOOR_MAX):
        raise ValueError("door must be the doorway number k, an integer from 1 to %d" % DOOR_MAX)
    return value


def _door_x(value):
    if value is None:
        return DEFAULT_DOOR_X
    if type(value) not in (int, float) or not (0.0 <= float(value) <= 1.0):
        raise ValueError("door_x must be a number from 0 (left edge) to 1 (right edge)")
    return float(value)


def install(agent, registry):
    meta = {"type": "object", "additionalProperties": False}
    door_schema = {"type": "integer", "minimum": 1, "maximum": DOOR_MAX}
    door_x_schema = {"type": "number", "minimum": 0, "maximum": 1}
    view_id_schema = {"type": "string", "maxLength": 16}

    def save_view(door, door_x=None):
        image = agent.current_image
        if image is None or not agent.current_frame_id:
            raise ValueError("no current image is available")
        if agent.question_mode:
            raise ValueError("question images are not saved; call MATCH_QUERY_IMAGE to compare them")
        door = _door(door)
        door_x = _door_x(door_x)
        existing = agent.views.find_frame(agent.current_frame_id)
        if existing is not None:
            raise ValueError("this frame is already saved as %s" % existing["id"])
        item = agent.views.add(image, step=agent.step_count, frame_id=agent.current_frame_id,
                               door=door, door_x=door_x)
        return {"tool": "SAVE_VIEW", "saved": item["id"], "doorway": item["door"], "door_x": item["door_x"],
                "step": item["step"], "views_for_doorway": agent.views.coverage()[door]}

    def list_views():
        items = [{"id": i["id"], "step": i["step"], "doorway": i["door"], "door_x": i["door_x"]}
                 for i in agent.views.items]
        return {"tool": "LIST_VIEWS", "count": len(items), "views": items,
                "coverage": {str(k): v for k, v in agent.views.coverage().items()}}

    def relabel_view(view_id, door, door_x=None):
        item = agent.views.get(view_id)
        door = _door(door)
        door_x = _door_x(door_x) if door_x is not None else item["door_x"]
        item = agent.views.relabel(view_id, door, door_x)
        return {"tool": "RELABEL_VIEW", "view_id": item["id"], "doorway": item["door"], "door_x": item["door_x"]}

    def delete_view(view_id):
        item = agent.views.delete(view_id)
        return {"tool": "DELETE_VIEW", "deleted": item["id"]}

    def match_query_image():
        record = agent.run_match()
        if "error" in record:
            return {"tool": "MATCH_QUERY_IMAGE", "error": record["error"], "strong_labels": []}
        ranking = record["ranking"]
        top = [e for e in ranking if e.get("box")][:TOP_TILES]
        if top and agent.current_frame_id:
            agent.visual_request = _tile_program(agent.current_frame_id, top)
        result = {"tool": "MATCH_QUERY_IMAGE", "query_rotation_deg": record.get("angle"),
                  "strong_threshold": STRONG_SCORE, "strong_labels": record["strong_labels"],
                  "matches": [{k: e[k] for k in ("view_id", "label", "score", "scale", "centre", "strong")}
                              for e in ranking[:5]],
                  "note": ("Only strong matches can support an answer. Answer not_seen only when strong_labels "
                           "is empty. The best-matching views, if any, are attached to your next message.")}
        if "note" in record:
            result["note"] = record["note"]
        return result

    registry.register(Tool(
        "SAVE_VIEW", "save the whole current image as a view of doorway k (the number on the doorway you walk toward); "
        "door_x optionally gives that doorway's horizontal position, 0 = left edge to 1 = right edge",
        {**meta, "properties": {"door": door_schema, "door_x": door_x_schema}, "required": ["door"]},
        save_view, ("LOG", "PLAN"), "memory_write", "metadata"))
    registry.register(Tool(
        "LIST_VIEWS", "list saved views with ids, steps, doorway numbers and coverage (no pixels)",
        {**meta, "properties": {}, "required": []}, list_views, ("LOG", "PLAN"), "memory_read", "metadata"))
    registry.register(Tool(
        "RELABEL_VIEW", "correct the doorway number (and door_x) of a saved view",
        {**meta, "properties": {"view_id": view_id_schema, "door": door_schema, "door_x": door_x_schema},
         "required": ["view_id", "door"]}, relabel_view, ("LOG",), "memory_write", "metadata"))
    registry.register(Tool(
        "DELETE_VIEW", "delete a saved view",
        {**meta, "properties": {"view_id": view_id_schema}, "required": ["view_id"]},
        delete_view, ("LOG",), "memory_write", "metadata"))
    registry.register(Tool(
        "MATCH_QUERY_IMAGE", "compare the painting in the current question image with every saved view; "
        "reports the doorway and side (k,S) of strong matches",
        {**meta, "properties": {}, "required": []}, match_query_image, ("PLAN",), "memory_read", "image"))

