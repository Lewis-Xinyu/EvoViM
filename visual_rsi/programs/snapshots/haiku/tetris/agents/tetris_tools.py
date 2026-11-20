"""Model-invoked tools that make the derived board record checkable and usable.

Each tool is a deliberate step the model takes: it can ask for the board again,
ask for one cell, ask what a specific placement would do, or promote a measured
board into a memory file with a name of its own choosing.  None of them write
memory on the model's behalf, and none of them choose an action.
"""
from __future__ import annotations

from registry import Tool

from agents import placement
from agents import tetris_vision as vision


def install_tetris(agent, registry):
    """Register the board tools.  They are additive; legacy memory tools remain."""

    def board(refresh=None):
        if refresh or agent.board_record is None:
            record = agent._perceive()
            agent.board_record = record
        if agent.board_record is None:
            raise ValueError("No playfield could be measured in the current frame")
        return agent._describe(agent.board_record)

    def cell(row, col):
        record = agent.board_record or agent._perceive()
        if record is None:
            raise ValueError("No playfield could be measured in the current frame")
        rows, cols = record["rows"], record["cols"]
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError("cell is outside the measured grid {}x{}".format(cols, rows))
        value = record["cells"][row][col]
        letter = vision.cell_letter(tuple(value) if value else None)
        if value is None:
            return {"row": row, "col": col, "occupied": False, "letter": "."}
        return {"row": row, "col": col, "occupied": True, "rgb": list(value),
                "family": vision.colour_family(tuple(value)),
                "shadow": vision.is_shadow(tuple(value)), "letter": letter}

    def simulate(placement):
        record = agent.board_record or agent._perceive()
        if record is None:
            raise ValueError("No playfield could be measured in the current frame")
        cells = agent._decode_record(record)
        rows, cols = record["rows"], record["cols"]
        stack = vision.fill(cells, rows, cols)
        loose = agent._loose_piece(record, cells, stack, rows, cols) if record.get("piece") else None
        if loose is None:
            raise ValueError("The falling piece is not separable from the stack in this frame")
        shape = vision.normalise_shape(loose["points"])
        options = vision.drop_outcomes(stack, rows, cols, shape)
        if placement == "all":
            chosen = options
        else:
            try:
                target = int(placement)
            except (TypeError, ValueError):
                chosen = options
            else:
                chosen = [item for item in options if item[1] == target]
                if not chosen:
                    raise ValueError("no legal landing in column {}".format(target))
        lines = []
        for r0, c0, rot, _land in chosen[:12]:
            board = vision.apply_piece(stack, rot, r0, c0, "X")
            full = vision.cleared_rows(board)
            kept = [row for r, row in enumerate(board) if r not in full]
            while len(kept) < rows:
                kept.insert(0, [None] * cols)
            heights, total_h, hole_n, bump, wells = vision.board_features(kept, rows, cols)
            value = vision.score_placement(kept, rows, cols, total_h, hole_n,
                                           len(full), bump, wells)
            lines.append("col {} rotation {} clears {} covered holes after (total) {} "
                         "height sum {} bumpiness {} wells {} score {:.1f}".format(
                             c0, _fmt(rot), len(full), hole_n, total_h, bump, wells, value))
        top = sorted(lines, key=lambda line: -float(line.rsplit(" ", 1)[1]))[:4]
        return "shape {}\n{}".format(vision.shape_name(loose["points"]), "\n".join(top))

    def _fmt(rot):
        return "/".join("{},{}".format(r, c) for r, c in rot)

    def placements(limit=10, col=None, hole_free_only=False):
        """Ranked table of every legal drop of the falling piece (derived, not chosen)."""
        record = agent.board_record or agent._perceive()
        if record is None:
            raise ValueError("No playfield could be measured in the current frame")
        loose_points = record.get("piece")
        if not loose_points:
            raise ValueError("The falling piece is not separable from the stack in this frame")
        stack, _clean = agent._stack_of(record)
        entries = placement.placement_table(stack, record["rows"], record["cols"], loose_points)
        if col is not None:
            entries = [e for e in entries if e["col"] == int(col)]
            if not entries:
                raise ValueError("no legal landing in column {}".format(col))
        if hole_free_only:
            entries = [e for e in entries if e["new_holes"] <= 0 and not e["danger"]]
        count = max(1, min(40, int(limit)))
        lines = [placement.format_row(i, e) for i, e in enumerate(entries[:count], 1)]
        return placement.summary(entries) + ("\n" + "\n".join(lines) if lines else "")

    def ledger(last=10):
        """The lock ledger: one row per hard drop, from the two frames around it."""
        return agent._ledger_text(last=max(1, min(200, int(last))))

    def commit(filename, label=None):
        """Write the current measured board into a memory file under a chosen name."""
        text = board()
        header = "# board snapshot step {} ({})".format(
            agent.step_count, label or "unlabelled")
        agent.memory.append(filename, header)
        for line in text.splitlines():
            agent.memory.append(filename, line)
        return "[board snapshot written to {} ({} lines)]".format(
            filename, len(text.splitlines()) + 1)

    def diff():
        if agent.board_record is None:
            raise ValueError("No board has been measured yet")
        return agent._change_context()

    obj = {"type": "object", "additionalProperties": False}
    registry.register(Tool(
        "BOARD_STATE",
        "Return the derived board record for the current frame: grid, stack, falling piece, "
        "landing shadow and the top simulated landings. Use it to check or refresh what you "
        "believe about the board.",
        {**obj, "properties": {"refresh": {"type": "boolean"}}, "required": []},
        board, ("LOG", "PLAN"), "memory_read", "text"))
    registry.register(Tool(
        "CELL_AT",
        "Return the measured content of one grid cell (row 0 is the top). Use it when a single "
        "cell is in doubt rather than re-reading the whole board.",
        {**obj, "properties": {"row": {"type": "integer", "minimum": 0, "maximum": 200},
                               "col": {"type": "integer", "minimum": 0, "maximum": 200}},
         "required": ["row", "col"]},
        cell, ("LOG", "PLAN"), "memory_read", "text"))
    registry.register(Tool(
        "SIMULATE_DROP",
        "Simulate landing the detected falling piece. placement='all' ranks every legal "
        "column/rotation; an integer restricts the simulation to that column.",
        {**obj, "properties": {"placement": {"type": "string"}}, "required": ["placement"]},
        simulate, ("PLAN",), "memory_read", "text"))
    registry.register(Tool(
        "COMMIT_BOARD",
        "Append the current derived board record to a memory file you name, with an optional "
        "label. Use it to keep a snapshot when a position matters, or to correct the written "
        "record.",
        {**obj, "properties": {"filename": {"type": "string"}, "label": {"type": "string"}},
         "required": ["filename"]},
        commit, ("LOG",), "memory_write", "text"))
    registry.register(Tool(
        "PLACEMENTS",
        "List the legal drops of the falling piece, ranked: column, rotation, keys needed, fall, "
        "points, new covered holes, clears, max height and points per key. Use col to inspect one "
        "column; hole_free_only drops options that create covered holes or reach the spawn region.",
        {**obj, "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 40},
                               "col": {"type": "integer", "minimum": 0, "maximum": 200},
                               "hole_free_only": {"type": "boolean"}},
         "required": []},
        placements, ("LOG", "PLAN"), "memory_read", "text"))
    registry.register(Tool(
        "LEDGER",
        "Return the lock ledger: the measured outcome of each hard drop (fall in rows, points by "
        "the verified rule, new covered holes, clears, and whether the drop-in-place prediction "
        "was exact), plus a calibration summary. Rows are written from frames only after the drop.",
        {**obj, "properties": {"last": {"type": "integer", "minimum": 1, "maximum": 200}},
         "required": []},
        ledger, ("LOG", "PLAN"), "memory_read", "text"))
    registry.register(Tool(
        "FRAME_DIFF",
        "Report which grid cells changed between the frame your previous action produced and "
        "the current frame. This is the two-phase outcome of your last action.",
        {**obj, "properties": {}, "required": []},
        diff, ("LOG", "PLAN"), "memory_read", "text"))
