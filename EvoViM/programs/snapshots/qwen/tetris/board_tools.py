"""Board-aware memory tools for the Tetris line (line M).

BOARD_STATE re-checks the board against the previous step (delta view: events,
max height, headroom, score estimate; the full board block is already inlined
in every prompt) and optionally requests a crop of the side panels (score /
next) through the host visual channel.  LOG_NOTE appends one ephemeral line to
a memory file; APPEND_NOTE records one tagged strategic note in strategy.txt.
All tools are invoked by the LLM; candidate code only computes derived context
from public observations and never sends pixels itself.
"""
from __future__ import annotations

import os


def _panel_program(state, size, frame_id, panels):
    """Build a visual-channel program cropping the panel band next to the board.

    Geometry is derived from the board border detected in this episode's own
    frame (one board-width to the left/right of the playfield), so it adapts
    to different layouts.  Returns None when the crop is not buildable.
    """
    try:
        width, height = int(size[0]), int(size[1])
        frame = state["frame"]
        board_left = int(frame["left"][0])
        board_right_outer = int(frame["right"][1])
        board_inner_left = int(frame["left"][1])
        board_inner_right = int(frame["right"][0])
        top = max(0, int(frame["top"][0]))
        bottom = min(height, int(frame["bottom"][1]) + 1)
        board_width = max(1, board_inner_right - board_inner_left)
    except Exception:
        return None
    if panels == "left":
        x0, x1 = max(0, board_left - board_width), min(width, board_left + 2)
    elif panels == "right":
        x0, x1 = max(0, board_right_outer), min(width, board_right_outer + board_width)
    else:
        x0, x1 = max(0, board_left - board_width), min(width, board_right_outer + board_width)
    x0 = int(max(0, min(x0, width - 1)))
    x1 = int(max(1, min(x1, width)))
    y0, y1 = int(top), int(max(top + 1, bottom))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    ops = [
        {"op": "frame", "id": frame_id, "as": "f"},
        {"op": "crop", "src": "f", "box": [x0, y0, x1, y1], "as": "c"},
    ]
    output = "c"
    if max(x1 - x0, y1 - y0) > 1536:
        ops.append({"op": "resize", "src": "c", "max_side": 1536, "as": "r"})
        output = "r"
    return {"ops": ops, "output": output}


def install_board_tools(agent, registry):
    from registry import Tool

    obj = {"type": "object", "additionalProperties": False}

    def board_state(panels=None):
        agent._decode_current_frame()
        state = agent._board_state or {}
        events = getattr(agent, "_board_events", None) or []
        if state.get("detected"):
            text = ("Board re-verified from this same frame - the full board block "
                    "is already in your prompt, do not call BOARD_STATE to "
                    "re-read it.\nchanges since the previous step: {}"
                    ).format("; ".join(events) if events else "none detected")
            text += ("\nstack: max height {}/{}; headroom {}"
                     ).format(state["max_height"], state["n_rows"],
                              state["headroom"])
        else:
            text = ("Board not detected in this frame - rely on the screenshot "
                    "and your memory; no board block was added to your prompt.")
        estimate = getattr(agent, "_score_estimate", None)
        if estimate is not None:
            text += ("\nrunning score estimate: {} (derived from drop distances; "
                     "the exact value is on the score panel)").format(estimate)
        if panels in ("left", "right", "both"):
            frame_id = getattr(agent, "current_frame_id", None)
            image = getattr(agent, "current_image", None)
            program = None
            if frame_id and image is not None and state.get("detected"):
                program = _panel_program(state, image.size, frame_id, panels)
            if program:
                agent.visual_request = program
                text += ("\n[The {} panel crop will be attached to your next message "
                         "as a labelled historical image; read the score from there.]"
                         ).format(panels)
            else:
                text += "\n[panel crop unavailable right now - use the screenshot instead]"
        return text

    registry.register(Tool(
        "BOARD_STATE",
        "re-check the board against the previous step (events, max height, headroom, score estimate); the full board block is already in your prompt; panels=left/right/both crops the score/next panels into the next message",
        {**obj, "properties": {"panels": {"type": "string", "enum": ["left", "right", "both"]}},
         "required": []},
        board_state, ("LOG", "PLAN"), "memory_read", "text"))

    def log_note(file, note):
        name = os.path.basename(str(file).replace("\\", "/").replace("..", ""))
        if not name:
            name = "notes.txt"
        if not name.endswith(".txt"):
            name += ".txt"
        line = " ".join(str(note).split())[:300]
        if not line:
            return "[LOG_NOTE: empty note - nothing written]"
        agent.memory.append(name, line)
        return "[LOG_NOTE: appended 1 line to {}]".format(name)

    registry.register(Tool(
        "LOG_NOTE",
        "append exactly one short ephemeral line (single line, max 300 chars) to a memory file such as notes.txt or plan.txt; for anything worth keeping use APPEND_NOTE instead",
        {**obj, "properties": {"file": {"type": "string"},
                               "note": {"type": "string", "maxLength": 300}},
         "required": ["file", "note"]},
        log_note, ("LOG", "PLAN"), "memory_write", "text"))

    def append_note(tag, note):
        line = " ".join(str(note).split())[:200]
        if not line:
            return "[APPEND_NOTE: empty note - nothing written]"
        step = getattr(agent, "step_count", 0)
        agent.memory.append("strategy.txt",
                            "step {} [{}] {}".format(step, tag, line))
        try:
            n = len([l for l in (agent.memory.read("strategy.txt") or "")
                     .splitlines() if l.strip()])
        except Exception:
            n = 0
        return ("[APPEND_NOTE: tagged strategic note stored in strategy.txt "
                "({} lines total); the last lines are auto-shown in every prompt]"
                ).format(n)

    registry.register(Tool(
        "APPEND_NOTE",
        "record ONE strategic note in strategy.txt (tags: plan, threat, mistake, progress, reset); use for resets, line clears, mistakes, plan changes and score progress - NOT for per-step board status, which is auto-logged",
        {**obj, "properties": {"tag": {"type": "string",
                                       "enum": ["plan", "threat", "mistake", "progress", "reset"]},
                              "note": {"type": "string", "maxLength": 200}},
         "required": ["tag", "note"]},
        append_note, ("LOG", "PLAN"), "memory_write", "text"))
