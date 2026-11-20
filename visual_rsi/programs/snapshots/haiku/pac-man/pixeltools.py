"""Memory tools that measure the episode's own pixels (line M).

Each tool is a plain function over the public observation the host already
supplied (``agent.current_image``).  The handles are deterministic measurements
of the current frame plus a per-episode store the LLM decides whether to use.

No tool writes a memory file on its own, no tool chooses an action, and no
pixels are sent to the model except through ``agent.visual_request``, which is
only assigned inside the handler of a tool the model invoked.
"""
from __future__ import annotations

from registry import Tool

READ = "memory_read"
WRITE = "memory_write"

SCENE_DESC = ("measure the current game frame and return exact numbers: your "
              "cell, whether you moved since the previous frame, open "
              "directions, ghosts with path distance, pellets left, power state")


def _scene(agent):
    sync = getattr(agent, "_sync_scene", None)
    if callable(sync):
        sync()
    return getattr(agent, "scene", None)


def install_pixel_tools(agent, registry):
    """Register measurement and structured-map tools on the host registry."""

    def sense(detail="line", view=False):
        scene = _scene(agent)
        if scene is None or not scene.seen:
            return ("[no frame measured yet: the host has not supplied an image "
                    "for this step]")
        out = scene.scene_line()
        if detail == "map":
            out += "\n" + scene.ascii_map(radius=8)
        elif detail == "full":
            out += "\n" + scene.ascii_map(radius=9)
            near_pellet = scene.nearest_target("pellet")
            near_power = scene.nearest_target("power")
            if near_pellet:
                out += ("\nnearest pellet: %s, %d cells, target cell %s"
                        % (near_pellet["direction"], near_pellet["cells"],
                           near_pellet["target"]))
            if near_power:
                out += ("\nnearest power pellet: %s, %d cells, target cell %s"
                        % (near_power["direction"], near_power["cells"],
                           near_power["target"]))
            if scene.marks:
                out += "\nmarks you saved: " + ", ".join(
                    "%r@(%d,%d)" % (m.get("tag", ""), m.get("r"), m.get("c"))
                    for m in scene.marks[:20])
        if view:
            agent.visual_request = scene.canvas(radius=9, cell_px=18)
            out += ("\n[semantic map requested: it will be attached to your "
                    "next message]")
        return out

    def map_tool(radius=8):
        scene = _scene(agent)
        if scene is None or not scene.seen:
            return "[no frame measured yet]"
        radius = max(4, min(12, int(radius)))
        return scene.ascii_map(radius=radius)

    def check_move(direction):
        scene = _scene(agent)
        if scene is None or not scene.seen:
            return "[no frame measured yet]"
        if direction not in ("up", "down", "left", "right"):
            return "[direction must be up, down, left or right]"
        return scene.check("move_" + direction)

    def mark_cell(row, col, tag=""):
        scene = _scene(agent)
        if scene is None or not scene.seen:
            return "[no frame measured yet]"
        try:
            row, col = int(row), int(col)
        except (TypeError, ValueError):
            return "[row and col must be integers]"
        if not scene.in_field(row, col):
            return "[cell (%d,%d) is outside the measured maze]" % (row, col)
        if not str(tag).strip():
            return scene.unmark(row, col)
        return scene.mark(row, col, tag)

    def list_marks():
        scene = _scene(agent)
        if scene is None or not scene.seen:
            return "[no frame measured yet]"
        if not scene.marks:
            return "[no marks saved yet]"
        return "\n".join("%r at (row=%d,col=%d)" % (m.get("tag", ""), m.get("r"), m.get("c"))
                         for m in scene.marks[:60])

    def view_map(radius=9):
        scene = _scene(agent)
        if scene is None or not scene.seen:
            return "[no frame measured yet]"
        radius = max(4, min(12, int(radius)))
        cell_px = max(6, min(24, 300 // (radius * 2 + 3)))
        agent.visual_request = scene.canvas(radius=radius, cell_px=cell_px)
        return ("[semantic map of a %dx%d cell window requested; it is attached "
                "to your next message. @=you .=pellet O=power g=ghost m=mark]"
                % (radius * 2 + 1, radius * 2 + 1))

    registry.register(Tool(
        "SENSE", SCENE_DESC,
        {"type": "object", "properties": {
            "detail": {"type": "string", "enum": ["line", "map", "full"]},
            "view": {"type": "boolean"}},
         "required": [], "additionalProperties": False},
        sense, ("LOG", "PLAN"), READ, "text"))
    registry.register(Tool(
        "MAP", "ASCII map of measured cells around you: walls, pellets, "
               "ghosts and your own saved marks",
        {"type": "object", "properties": {
            "radius": {"type": "integer", "minimum": 4, "maximum": 12}},
         "required": [], "additionalProperties": False},
        map_tool, ("LOG", "PLAN"), READ, "text"))
    registry.register(Tool(
        "CHECK_MOVE", "check one direction against the measured wall map before "
                      "committing to it",
        {"type": "object", "properties": {
            "direction": {"type": "string", "enum": ["up", "down", "left", "right"]}},
         "required": ["direction"], "additionalProperties": False},
        check_move, ("LOG", "PLAN"), READ, "text"))
    registry.register(Tool(
        "MARK_CELL", "save or remove a note on a map cell you choose (empty tag "
                     "removes it)",
        {"type": "object", "properties": {
            "row": {"type": "integer", "minimum": 0, "maximum": 200},
            "col": {"type": "integer", "minimum": 0, "maximum": 200},
            "tag": {"type": "string", "maxLength": 24}},
         "required": ["row", "col"], "additionalProperties": False},
        mark_cell, ("LOG", "PLAN"), WRITE, "text"))
    registry.register(Tool(
        "LIST_MARKS", "list the map cells and notes you have saved",
        {"type": "object", "properties": {}, "required": [],
         "additionalProperties": False},
        list_marks, ("PLAN",), READ, "text"))
    registry.register(Tool(
        "VIEW_MAP", "render the measured map around you as an image through the "
                    "host visual channel",
        {"type": "object", "properties": {
            "radius": {"type": "integer", "minimum": 4, "maximum": 12}},
         "required": [], "additionalProperties": False},
        view_map, ("LOG", "PLAN"), READ, "image"))
    return registry
