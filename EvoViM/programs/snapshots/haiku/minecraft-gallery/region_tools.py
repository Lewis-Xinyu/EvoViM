"""Model-invoked tools for region memory (line M).

Every handler returns JSON-able metadata.  The only pixels the model can see are
host renders named by a visual request: a frame id plus host-side ops (annotate,
crop, resize, tile).  The candidate never sends pixel arrays.
"""
from __future__ import annotations

from registry import Tool

from region_store import MAX_LABEL, normalize_label
from painting_match import STRONG_SCORE

GRID_COLS, GRID_ROWS = 6, 4
TILE_SIDE = 360
TOP_TILES = 3


def _grid_program(frame_id, width, height):
    marks = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x0, y0 = col * width // GRID_COLS, row * height // GRID_ROWS
            x1, y1 = (col + 1) * width // GRID_COLS - 1, (row + 1) * height // GRID_ROWS - 1
            marks.append({"type": "box", "box": [x0, y0, x1, y1], "text": "%d,%d" % (x0, y0)})
    return {"ops": [{"op": "frame", "id": frame_id, "as": "f"},
                    {"op": "annotate", "src": "f", "marks": marks, "as": "out"}],
            "output": "out"}


def _tile_program(query_frame, entries):
    ops = [{"op": "frame", "id": query_frame, "as": "q"}]
    sources, labels = ["q"], ["question painting"]
    for index, entry in enumerate(entries):
        ops.append({"op": "frame", "id": entry["frame_id"], "as": "r%d" % index})
        ops.append({"op": "crop", "src": "r%d" % index, "box": entry["box"], "as": "c%d" % index})
        ops.append({"op": "resize", "src": "c%d" % index, "max_side": TILE_SIDE, "as": "s%d" % index})
        sources.append("s%d" % index)
        labels.append("%s score %.2f" % (entry["label"], entry["score"]))
    ops.append({"op": "tile", "srcs": sources, "cols": len(sources), "labels": labels, "as": "out"})
    return {"ops": ops, "output": "out"}


def install(agent, registry):
    obj = {"type": "object", "additionalProperties": False}
    box_schema = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 100000}, "maxItems": 4}
    label_schema = {"type": "string", "maxLength": MAX_LABEL}
    region_id_schema = {"type": "string", "maxLength": 16}

    def show_grid():
        if agent.current_image is None or not agent.current_frame_id:
            raise ValueError("no current image is available")
        width, height = agent.current_image.size
        agent.visual_request = _grid_program(agent.current_frame_id, width, height)
        return {"tool": "SHOW_GRID", "image_size": [width, height], "grid": [GRID_COLS, GRID_ROWS],
                "note": "The current image with a labelled grid is attached to your next message. "
                        "Each cell label is the pixel corner (x,y) of that cell."}

    def save_region(box, label):
        image = agent.current_image
        if image is None or not agent.current_frame_id:
            raise ValueError("no current image is available")
        if agent.question_mode:
            raise ValueError("question images are not saved; call MATCH_QUERY_IMAGE to compare them")
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("box must be [x0,y0,x1,y1] integers")
        x0, y0, x1, y1 = box
        width, height = image.size
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("box is outside the %dx%d image" % (width, height))
        if x1 - x0 < 16 or y1 - y0 < 16:
            raise ValueError("box must be at least 16 px on each side")
        normalized = normalize_label(label)
        text = normalized or label.strip()[:MAX_LABEL]
        if not text:
            raise ValueError("label is empty")
        item = agent.regions.add(image.crop((x0, y0, x1, y1)), step=agent.step_count,
                                 frame_id=agent.current_frame_id, box=box, label=text,
                                 label_valid=normalized is not None, question_frame=False)
        result = {"tool": "SAVE_REGION", "saved": item["id"], "label": text, "label_valid": item["label_valid"],
                  "box": item["box"], "step": item["step"]}
        if normalized is None:
            result["note"] = "label is not in k,S form, so this region cannot answer a question until relabelled"
        return result

    def list_regions():
        items = [{"id": i["id"], "step": i["step"], "label": i["label"], "label_valid": i["label_valid"],
                  "box": i["box"]} for i in agent.regions.items]
        return {"tool": "LIST_REGIONS", "count": len(items), "regions": items}

    def relabel_region(region_id, label):
        normalized = normalize_label(label)
        if normalized is None:
            raise ValueError("label must be k,S with k a doorway number and S = L or R")
        item = agent.regions.relabel(region_id, normalized, True)
        return {"tool": "RELABEL_REGION", "region_id": item["id"], "label": item["label"], "label_valid": True}

    def delete_region(region_id):
        item = agent.regions.delete(region_id)
        return {"tool": "DELETE_REGION", "deleted": item["id"]}

    def match_query_image():
        record = agent.run_match()
        if "error" in record:
            return {"tool": "MATCH_QUERY_IMAGE", "error": record["error"], "strong_labels": []}
        ranking = record["ranking"]
        top = [e for e in ranking if e.get("frame_id") and e.get("box")][:TOP_TILES]
        if top and agent.current_frame_id:
            agent.visual_request = _tile_program(agent.current_frame_id, top)
        return {"tool": "MATCH_QUERY_IMAGE", "query_rotation_deg": record.get("angle"),
                "strong_threshold": STRONG_SCORE, "strong_labels": record["strong_labels"],
                "matches": ranking[:5],
                "note": ("Only strong matches with a valid k,S label can support an answer. "
                         "Answer not_seen only when strong_labels is empty. The comparison image, if any, "
                         "is attached to your next message.")}

    meta = {"type": "object", "additionalProperties": False}
    registry.register(Tool(
        "SHOW_GRID", "show the current image with a labelled pixel grid, to read box coordinates",
        {**meta, "properties": {}, "required": []}, show_grid, ("LOG", "PLAN"), "memory_read", "image"))
    registry.register(Tool(
        "SAVE_REGION", "save a crop of the current image; box is [x0,y0,x1,y1] pixels around one panel; label is k,S",
        {**meta, "properties": {"box": box_schema, "label": label_schema}, "required": ["box", "label"]},
        save_region, ("LOG", "PLAN"), "memory_write", "metadata"))
    registry.register(Tool(
        "LIST_REGIONS", "list saved regions with ids, steps, labels and boxes (no pixels)",
        {**meta, "properties": {}, "required": []}, list_regions, ("LOG", "PLAN"), "memory_read", "metadata"))
    registry.register(Tool(
        "RELABEL_REGION", "correct the label of a saved region",
        {**meta, "properties": {"region_id": region_id_schema, "label": label_schema},
         "required": ["region_id", "label"]}, relabel_region, ("LOG",), "memory_write", "metadata"))
    registry.register(Tool(
        "DELETE_REGION", "delete a saved region",
        {**meta, "properties": {"region_id": region_id_schema}, "required": ["region_id"]},
        delete_region, ("LOG",), "memory_write", "metadata"))
    registry.register(Tool(
        "MATCH_QUERY_IMAGE", "compare the painting in the current question image with saved regions; reports strong matches",
        {**meta, "properties": {}, "required": []}, match_query_image, ("PLAN",), "memory_read", "metadata"))
