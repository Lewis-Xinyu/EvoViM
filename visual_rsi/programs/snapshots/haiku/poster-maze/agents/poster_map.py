"""Host-derived wall map built from observation text only (no pixels).

State is a plain JSON-safe dict so that checkpoints round-trip:
    size    [rows, cols] parsed from the long-term context, or None
    goal    [r, c] parsed from the short-term context, or None
    pos     [r, c, F] of the most recent navigation observation, or None
    edges   "r,c,D" -> {"open": bool, "step": int}; mirrored onto the neighbour
    visited "r,c" -> first step the cell was seen
    conflicts  list of strings recording contradictory evidence about one edge
Walls are inferred from the action list: 'forward' absent (or the explicit
'Cannot move forward' text) means a wall in front; 'backward' absent means a
wall behind. The LLM decides what to write down; this map is a derived view.
"""
from __future__ import annotations

import re
from collections import deque

DIRS = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
POS_RE = re.compile(r"You are at cell \(\s*(\d+)\s*,\s*(\d+)\s*\) facing ([NESW])")
SIZE_RE = re.compile(r"(\d+)x(\d+) cells")
GOAL_RE = re.compile(r"goal is cell \(\s*(\d+)\s*,\s*(\d+)\s*\)")
ACTIONS_RE = re.compile(r"Available actions: ([^.]*)\.")
BLOCKED_RE = re.compile(r"Cannot move forward")
NOW_RE = re.compile(r"Now at cell \(\s*(\d+)\s*,\s*(\d+)\s*\)")
CONFLICT_CAP = 50
PATH_CAP = 60


def new_state() -> dict:
    return {"size": None, "goal": None, "pos": None, "edges": {}, "visited": {}, "conflicts": []}


def _inside(state: dict, r: int, c: int) -> bool:
    if state["size"] is None:
        return True
    return 0 <= r < state["size"][0] and 0 <= c < state["size"][1]


def _set_edge(state: dict, r: int, c: int, direction: str, is_open: bool, step: int) -> None:
    key = "%d,%d,%s" % (r, c, direction)
    old = state["edges"].get(key)
    if old is not None and old["open"] != is_open and len(state["conflicts"]) < CONFLICT_CAP:
        state["conflicts"].append("step %d: %s open=%s contradicts step %d open=%s"
                                  % (step, key, is_open, old["step"], old["open"]))
    state["edges"][key] = {"open": is_open, "step": step}
    dr, dc = DIRS[direction]
    nr, nc = r + dr, c + dc
    if _inside(state, nr, nc):
        state["edges"]["%d,%d,%s" % (nr, nc, OPP[direction])] = {"open": is_open, "step": step}


def observe(state: dict, text: dict, step: int) -> None:
    """Update the map from one observation's text fields."""
    long_ctx = text.get("long_term_context", "") or ""
    short_ctx = text.get("short_term_context", "") or ""
    size = SIZE_RE.search(long_ctx)
    if size and state["size"] is None:
        state["size"] = [int(size.group(1)), int(size.group(2))]
    goal = GOAL_RE.search(short_ctx)
    if goal:
        state["goal"] = [int(goal.group(1)), int(goal.group(2))]
    pos = POS_RE.search(short_ctx)
    if not pos:
        # End-of-navigation text states the cell but not the facing; keep the
        # last known facing and do not infer any wall from the action list.
        now = NOW_RE.search(short_ctx)
        if now and state["pos"] is not None:
            r, c = int(now.group(1)), int(now.group(2))
            state["pos"] = [r, c, state["pos"][2]]
            state["visited"].setdefault("%d,%d" % (r, c), step)
        return
    r, c, facing = int(pos.group(1)), int(pos.group(2)), pos.group(3)
    state["pos"] = [r, c, facing]
    state["visited"].setdefault("%d,%d" % (r, c), step)
    actions = ACTIONS_RE.search(short_ctx)
    if actions is None:
        return
    listed = {a.strip() for a in actions.group(1).split(",")}
    front_open = "forward" in listed and not BLOCKED_RE.search(short_ctx)
    _set_edge(state, r, c, facing, front_open, step)
    _set_edge(state, r, c, OPP[facing], "backward" in listed, step)


def _neighbour(state: dict, r: int, c: int, direction: str):
    dr, dc = DIRS[direction]
    nr, nc = r + dr, c + dc
    return (nr, nc) if _inside(state, nr, nc) else None


def frontier(state: dict, limit: int = 5) -> list:
    """Nearest unexplored edges reachable through known-open edges, with paths."""
    if state["pos"] is None:
        return []
    start = (state["pos"][0], state["pos"][1])
    parent = {start: None}
    queue = deque([start])
    found = []
    while queue and len(found) < limit:
        cell = queue.popleft()
        r, c = cell
        for direction in DIRS:
            key = "%d,%d,%s" % (r, c, direction)
            if key not in state["edges"] and _inside(state, *_step(cell, direction)):
                path = []
                node = cell
                while node is not None and len(path) < PATH_CAP:
                    path.append(list(node))
                    node = parent[node]
                found.append({"cell": [r, c], "unknown_side": direction, "path": list(reversed(path))})
                if len(found) >= limit:
                    break
        for direction in DIRS:
            edge = state["edges"].get("%d,%d,%s" % (r, c, direction))
            nxt = _neighbour(state, r, c, direction)
            if edge and edge["open"] and nxt is not None and nxt not in parent:
                parent[nxt] = cell
                queue.append(nxt)
    return found


def _step(cell, direction):
    dr, dc = DIRS[direction]
    return cell[0] + dr, cell[1] + dc


def answer_cell(state: dict, r: int, c: int, facing: str):
    """Derived answer for a view taken at (r, c) facing `facing`.

    A poster ahead sits on the first closed wall reached by walking forward
    through known-open edges; the answer is that last cell and its wall side,
    'r,c,D'. Returns None when an edge on the walk is unknown or the walk cycles.
    This is a hint for the LLM to check, not an answer.
    """
    cur = (int(r), int(c))
    seen = {cur}
    while True:
        nxt = _step(cur, facing)
        if not _inside(state, *nxt):
            return "%d,%d,%s" % (cur[0], cur[1], facing)
        edge = state["edges"].get("%d,%d,%s" % (cur[0], cur[1], facing))
        if edge is None:
            return None
        if not edge["open"]:
            return "%d,%d,%s" % (cur[0], cur[1], facing)
        if nxt in seen:
            return None
        seen.add(nxt)
        cur = nxt


def frontier_hint(state: dict, limit: int = 3) -> list:
    """Short form of frontier() for the observation text: target cell and step count."""
    out = []
    for item in frontier(state, limit):
        out.append({"cell": item["cell"], "unknown_side": item["unknown_side"],
                    "steps": max(0, len(item["path"]) - 1)})
    return out


def render(state: dict) -> str:
    """ASCII view: '@' agent, 'G' goal, '.' visited, '?' unvisited cell;
    '#' or '-'/'|' wall, ' ' open, '~' unknown edge."""
    if state["size"] is None:
        return "(maze size unknown)"
    rows, cols = state["size"]
    grid = [[" "] * (2 * cols + 1) for _ in range(2 * rows + 1)]
    for r in range(rows):
        for c in range(cols):
            if state["pos"] is not None and (r, c) == (state["pos"][0], state["pos"][1]):
                ch = "@"
            elif state["goal"] == [r, c]:
                ch = "G"
            elif "%d,%d" % (r, c) in state["visited"]:
                ch = "."
            else:
                ch = "?"
            grid[2 * r + 1][2 * c + 1] = ch
    for r in range(rows):
        for c in range(cols):
            for direction, (gy, gx) in (("S", (2 * r + 2, 2 * c + 1)), ("E", (2 * r + 1, 2 * c + 2))):
                edge = state["edges"].get("%d,%d,%s" % (r, c, direction))
                if edge is None:
                    mark = "~"
                elif edge["open"]:
                    mark = " "
                else:
                    mark = "-" if direction == "S" else "|"
                grid[gy][gx] = mark
    for r in range(rows + 1):
        for c in range(cols + 1):
            grid[2 * r][2 * c] = "+"
    border_h, border_v = "-", "|"
    for c in range(cols):
        grid[0][2 * c + 1] = border_h
        grid[2 * rows][2 * c + 1] = border_h
    for r in range(rows):
        grid[2 * r + 1][0] = border_v
        grid[2 * r + 1][2 * cols] = border_v
    return "\n".join("".join(row).rstrip() for row in grid)


def summary(state: dict) -> dict:
    known = sum(1 for e in state["edges"].values() if e["open"] is False)
    opened = sum(1 for e in state["edges"].values() if e["open"] is True)
    return {"size": state["size"], "goal": state["goal"], "pos": state["pos"],
            "visited_cells": len(state["visited"]), "known_walls": known, "known_open": opened,
            "conflicts": state["conflicts"][-10:], "frontier": frontier(state, 5),
            "ascii": render(state)}
