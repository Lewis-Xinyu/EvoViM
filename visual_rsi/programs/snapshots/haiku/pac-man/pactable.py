"""Per-direction route table and previous-step route check (line M, g2 revision).

Every number here is measured from the current frame (pacvision.Scene) or from
the previous frame's table.  Nothing in this module calls a tool, writes a
memory file, or chooses an action.  The text is appended to every LOG and PLAN
observation; the model decides what to record and what to press.
"""
from __future__ import annotations

from collections import deque

from pacvision import ACTION_TO_DIR, DELTA, DIRS

HORIZON = 12        # corridor cells searched from Pac-Man
RUN_MAX = 12        # straight cells counted for the run length
LAST_SEEN_MAX = 8   # steps a ghost stays in the table after it leaves the frame


def _label_bfs(scene, start):
    """Each reachable cell within HORIZON, labelled by the first step of its shortest path."""
    label = {start: None}
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        if dist[cur] >= HORIZON:
            continue
        for d in DIRS:
            dr, dc = DELTA[d]
            nb = (cur[0] + dr, cur[1] + dc)
            if nb in label or not scene.in_field(*nb):
                continue
            if scene.edge_open(cur[0], cur[1], d) is not True:
                continue
            label[nb] = d if cur == start else label[cur]
            dist[nb] = dist[cur] + 1
            queue.append(nb)
    return label, dist


def update_sightings(sightings, scene, step):
    """Last measured cell of each ghost, kept for LAST_SEEN_MAX steps after it leaves the frame."""
    visible = []
    seen_colours = set()
    for g in scene.ghosts or []:
        if g.get("r") is None:
            continue
        seen_colours.add(g.get("colour"))
        visible.append({"colour": g.get("colour", "?"), "r": int(g["r"]),
                        "c": int(g["c"]), "step": int(step)})
    kept = [s for s in (sightings or [])
            if s.get("colour") not in seen_colours and step - s.get("step", step) <= LAST_SEEN_MAX]
    return visible + kept


def route_table(scene, sightings, step):
    """Facts per first step (up/down/left/right), or None when Pac-Man is not measured."""
    if not scene.pac or scene.pac.get("r") is None:
        return None
    r, c = int(scene.pac["r"]), int(scene.pac["c"])
    start = (r, c)
    label, dist = _label_bfs(scene, start)
    table = {}
    for d in DIRS:
        opened = scene.edge_open(r, c, d) is True
        table[d] = {"open": opened,
                    "run": int(scene.probe_direction(d, RUN_MAX)) if opened else 0,
                    "pellets": 0, "nearest": None, "power": 0,
                    "ghosts": []}
    for cell in scene.pellets:
        d = label.get(cell)
        if d is None:
            continue
        entry = table[d]
        entry["pellets"] += 1
        steps = dist[cell]
        entry["nearest"] = steps if entry["nearest"] is None else min(entry["nearest"], steps)
    for cell in scene.power:
        d = label.get(cell)
        if d is not None:
            table[d]["power"] += 1
    here, far = [], []
    for g in sightings or []:
        cell = (int(g["r"]), int(g["c"]))
        age = int(step) - int(g.get("step", step))
        kind = "edible" if scene.powered else "threat"
        if cell == start:
            # A remembered ghost (age > 0) is labelled with its age, as the other
            # entries are; an unlabelled entry read as a current sighting.
            text = "%s %s, same cell" % (g["colour"], kind)
            if age:
                text += " (last seen %d step%s ago)" % (age, "" if age == 1 else "s")
            here.append(text)
            continue
        d = label.get(cell)
        if d is None:
            text = "%s beyond %d" % (g["colour"], HORIZON)
            if age:
                text += " (seen %d ago)" % age
            far.append(text)
            continue
        text = "%s %s %d" % (g["colour"], kind, dist[cell])
        if age:
            text += " (last seen %d step%s ago)" % (age, "" if age == 1 else "s")
        table[d]["ghosts"].append(text)
    return {"start": [r, c], "dirs": table, "here": here, "far": far}


def render(table):
    lines = ["ROUTE TABLE (measured facts, no action chosen; run = open cells straight ahead, "
             "you coast in your heading until a wall):"]
    for d in DIRS:
        e = table["dirs"][d]
        if not e["open"]:
            lines.append("  %-5s wall" % d)
            continue
        text = "  %-5s open, run %d, pellets %d" % (d, e["run"], e["pellets"])
        if e["nearest"] is not None:
            text += " (nearest %d)" % e["nearest"]
        if e["power"]:
            text += ", power pellets %d" % e["power"]
        text += ", ghosts: " + (", ".join(e["ghosts"]) if e["ghosts"] else "none")
        lines.append(text)
    if table["here"]:
        lines.append("  here: " + ", ".join(table["here"]))
    if table["far"]:
        lines.append("  far: " + ", ".join(table["far"]))
    return "\n".join(lines)


def _moved_direction(scene):
    if scene.moved is None:
        return "unknown (no earlier frame)"
    if scene.moved is False or not scene.move:
        return "did not move"
    dx, dy = scene.move
    if abs(dx) >= abs(dy):
        return "moved right" if dx > 0 else "moved left"
    return "moved down" if dy > 0 else "moved up"


def route_check(prev, pressed, scene):
    """What the previous table said about the pressed direction, against this frame."""
    d = ACTION_TO_DIR.get(pressed) if pressed else None
    if d is None or not prev or not prev.get("dirs"):
        return None
    e = prev["dirs"][d]
    said = ("open, run %d, pellets %d" % (e["run"], e["pellets"])) if e["open"] else "wall"
    text = ("ROUTE CHECK (previous step): you pressed %s; the table said %s. Measured now: %s; "
            "pellets eaten %d" % (d, said, _moved_direction(scene), scene.eaten_delta or 0))
    before = prev.get("lives")
    if before is not None and scene.lives is not None:
        text += "; lives %d -> %d" % (before, scene.lives)
    return text + "."
