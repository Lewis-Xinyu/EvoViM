"""A memory agent for a pixel-only falling-block game.

The parent agent already implements the two-phase loop and a file-backed
memory that the model edits through tool calls.  This subclass keeps all of
that and adds two things:

1. A derived board record.  Every LOG/PLAN call sees a compact measurement of
   the current frame (board grid, resting stack, falling piece, landing
   shadow, candidate drops) computed by ``tetris_vision``.  The record is a
   supplement, not a substitute: the model still decides through explicit
   memory tool calls what to keep, correct or discard, and the file system is
   unchanged.
2. A per-frame observation ledger.  Two-phase logging means the step-t prompt
   only ever contains the outcome of step t-1's action.  The ledger stores the
   derived record keyed by the frame it was measured from, so the LOG prompt
   can show "what changed between the frame you have now and the frame your
   previous action produced" without ever reading a future frame, and the PLAN
   prompt can show the same comparison.

Everything the subclass stores is JSON-serialisable so the host checkpoint
round trip keeps working.
"""
from __future__ import annotations

import copy

from agents import placement
from agents import tetris_vision as vision
from agents.memory_agent import MemoryAgent


BOARD_SECTION = """\
DERIVED BOARD RECORD — {source}
This is a measurement of pixels the host supplied for this step. It is derived
context, not a substitute for your memory: verify an entry before you rely on
it, and record corrections in your files if the measurement looks wrong.
{board}
"""

LOG_EXTRA = """\
CHANGE SINCE YOUR PREVIOUS ACTION (derived by comparing the two frames):
{change}

Log the outcome of that action, then note anything about this board you will
need when choosing the next move. Write only what this pair of frames
supports; do not restate the whole board in prose when the record above
already shows it. Prefer one compact line per step in `board_log.txt`, and keep
`strategy.txt` for durable intentions rather than per-frame transcripts.
"""

PLAN_EXTRA = """\
WHAT THE LAST ACTION CHANGED (derived):
{change}

Use this when deciding. A piece that did not move when you asked it to is a
fact about the interface, not about your strategy.
"""

LOCK_EXTRA = """\
LAST LOCK (derived from the two frames around your previous hard drop; the
outcome of your previous action only, nothing earlier and nothing later):
{lock}
"""

REPEAT_DROP_NOTE = """\
DERIVED CHECK before you commit: your previous action (step {step}) was a hard
drop and it did not lock the piece ({reason}). Across this episode so far: {stats}.
This is a measurement, not an instruction. If you still want a hard drop now,
commit it again and it will be used; otherwise commit the action you judge best."""

LOG_SYSTEM_TETRIS = """\
You are playing a falling-block game and maintaining your external memory.

{action_description}

ENVIRONMENT RESPONSE TO YOUR PREVIOUS ACTION (text):
{observation}

{board_section}

{change_section}

{lock_section}

YOUR MEMORY FILES:
{file_index}

You may use the memory commands and any tool listed in the system message.
Write entries that will still be useful in ten steps: rules you have verified,
mistakes you have made, and corrections to the derived record. Finish with
<|DONE|> when you have nothing further worth recording."""

PLAN_SYSTEM_TETRIS = """\
{instruction_prompt}

You have an external memory system. Use it when the current board leaves a
choice you cannot settle from the record alone. The placement table in the
board record ranks every legal drop of the falling piece; PLACEMENTS and LEDGER
return more of the same derived data.

{board_section}

{change_section}

{lock_section}

YOUR MEMORY FILES:
{file_index}

CURRENT OBSERVATION (text):
{observation}

Decide the single next action. Commit with:
   <|ACTION|>your_chosen_action<|END|>

You may use any registered tool before committing. A committed action is
final for this step."""

INITIAL_FILES_TETRIS = {
    "board_log.txt": "",
    "strategy.txt": "",
    "interface_notes.txt": "",
}


class TetrisMemoryAgent(MemoryAgent):
    """MemoryAgent with a derived board record and a two-frame change ledger."""

    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder, config)
        self._ledger = {}
        self._prev_record = None
        self._last_frame_id = None
        self.board_record = None
        # Lock ledger: one row per hard drop, written at the next step from the
        # two frames around the drop.  `_pending` is the drop-in-place prediction
        # made from the frame the drop was pressed on.
        self._locks = []
        self._pending = None
        self._last_lock = None
        # Executed actions of earlier steps (most recent last, at most four).  A
        # lock row needs the action before its drop: a hard drop pressed straight
        # after another hard drop is a different measurement from one pressed after
        # a sideways move or a rotation.
        self._exec_hist = []

    def reset(self):
        # Base reset clears the memory files and counters only.  Lock history and
        # the frame record belong to one episode, so they are cleared here too;
        # otherwise an agent reused for the next episode inherits this episode's
        # ledger and its derived statistics.
        super().reset()
        self._ledger = {}
        self._prev_record = None
        self._last_frame_id = None
        self.board_record = None
        self._locks = []
        self._pending = None
        self._last_lock = None
        self._exec_hist = []

    # ------------------------------------------------------------ memory init

    def _get_initial_files(self):
        if self._env_name and self._env_name in INITIAL_FILES_BY_ENV_LOCAL:
            return copy.deepcopy(INITIAL_FILES_BY_ENV_LOCAL[self._env_name])
        return copy.deepcopy(INITIAL_FILES_TETRIS)

    # ------------------------------------------------------------- perception

    def _perceive(self):
        image = getattr(self, "current_image", None)
        if image is None:
            return None
        arr = vision.as_array(image)
        field = vision.find_playfield(arr)
        if field is None:
            return None
        cols, rows, cw, ch = vision.infer_grid(arr, field)
        cells = vision.read_cells(arr, field, cols, rows, cw, ch)
        record = {
            "frame_id": getattr(self, "current_frame_id", None),
            "field": field,
            "cols": cols,
            "rows": rows,
            "cell_w": round(float(cw), 2),
            "cell_h": round(float(ch), 2),
            "cells": [[None if v is None else list(map(int, v)) for v in row] for row in cells],
        }
        record["shadow"] = [[r, c] for r in range(rows) for c in range(cols)
                            if vision.is_shadow(cells[r][c])]
        # Store the piece with the record: the next step uses it to follow a
        # piece that has come to rest on the stack, where neither connectivity
        # nor a landing ghost can separate it.
        try:
            decoded = self._decode_record(record)
            stack = vision.fill(decoded, rows, cols)
            piece = self._loose_piece(record, decoded, stack, rows, cols)
            # A falling piece is drawn in bright colours inside the well.  When
            # none of the reported cells is bright (the piece is above the well,
            # so only its landing ghost is visible) the reported cells are the
            # ghost, not the piece: report no piece instead.
            bright = [p for p in piece["points"]
                      if 0 <= p[0] < rows and 0 <= p[1] < cols
                      and decoded[p[0]][p[1]] is not None
                      and not vision.is_shadow(decoded[p[0]][p[1]])] if piece else []
            record["piece"] = [list(p) for p in piece["points"]] if (piece and bright) else None
        except Exception:
            # A measurement failure must never prevent the agent from acting;
            # the model sees that no piece was reported instead.
            record["piece"] = None
        return record

    @staticmethod
    def _decode_record(record):
        cells = []
        for row in record.get("cells") or []:
            cells.append([None if v is None else tuple(v) for v in row])
        return cells

    def _describe(self, record):
        if not record:
            return "No board could be measured from the current frame."
        cells = self._decode_record(record)
        rows, cols = record["rows"], record["cols"]
        measured = vision.fill(cells, rows, cols)
        # Use the piece the frame measurement accepted (see _perceive), so the
        # display, the table and the lock ledger agree on what the piece is.
        loose = self._loose_piece(record, cells, measured, rows, cols) if record.get("piece") else None
        # The falling piece is not part of the stack: leaving it in the board
        # would make every cell beneath it look like a covered hole.
        stack = [list(row) for row in measured]
        if loose:
            for r, c in loose["points"]:
                if 0 <= r < rows:
                    stack[r][c] = None
        heights = vision.column_heights(cols, rows, stack)
        holes = vision.holes(cols, rows, stack)
        parts = [
            "well {x0},{y0} to {x1},{y1}; {cols}x{rows} cells".format(
                cols=cols, rows=rows, **record["field"]),
            vision.render_field(cols, rows, cells),
            "stack column heights, left to right (piece excluded): "
            + ",".join(str(h) for h in heights),
            "covered holes in the stack: {}".format(holes),
        ]
        shadow = record.get("shadow") or []
        if shadow:
            parts.append("landing shadow cells (row,col): " + " ".join(
                "({},{})".format(r, c) for r, c in shadow[:12]))
        if loose:
            parts.append("falling piece: shape {} at rows {}-{}, cols {}-{} "
                         "(source: {}{})".format(
                             loose["shape"], loose["bbox"][0], loose["bbox"][1],
                             loose["bbox"][2], loose["bbox"][3], loose.get("source", "measured"),
                             "; {} other shape(s) fit the same ghost".format(
                                 loose["alternatives"]) if loose.get("alternatives") else ""))
            entries = placement.placement_table(stack, rows, cols, loose["points"])
            parts.append("PLACEMENT TABLE for the falling piece (derived; ranked by danger, "
                         "new covered holes, then points per key press):\n"
                         + placement.format_table(entries, limit=5))
        else:
            parts.append("falling piece: not separable from the stack in this frame")
        return "\n".join(parts)

    def _loose_piece(self, record, cells, stack, rows, cols):
        """The current piece, found by connectivity and completed by its ghost.

        A locked stack is connected to the floor; the falling piece is not.
        That makes the piece detectable without history and without a palette.
        When the piece touches the stack (the instant before it locks) the
        cells that changed since the previous frame decide instead.  The
        landing ghost then recovers the piece's full shape when part of it is
        clipped at the top or hidden behind a tall column.

        Returns a dict with ``points`` (may extend above row 0), ``bbox``,
        ``shape``, ``source`` and ``alternatives`` (0 when only one shape was
        possible, otherwise how many distinct tetrominoes remained).
        """
        prev_cells = None
        prev_piece = None
        if self._prev_record is not None:
            prev_cells = self._decode_record(self._prev_record)
            prev_piece = self._prev_record.get("piece")
        board, piece, source, note = vision.detect_piece(
            cells, rows, cols, prev_cells, prev_piece)
        if not piece:
            return None
        result = self._piece_from_points(piece)
        result["source"] = source or "measured"
        if note:
            result["source"] += "; " + note
        result["alternatives"] = 0
        if result["shape"] == "irregular":
            # Report how many tetrominoes could fit the same ghost, so the
            # model knows the measurement is incomplete rather than wrong.
            ghost = [tuple(p) for p in (record.get("shadow") or [])]
            bright = [(r, c) for r in range(rows) for c in range(cols)
                      if cells[r][c] is not None and not vision.is_shadow(cells[r][c])]
            fits = vision.fit_ghost(board, rows, cols, ghost, bright=bright) if ghost else []
            result["alternatives"] = len({fit["shape"] for fit in fits}) or 1
        return result

    @staticmethod
    def _piece_from_points(points):
        rows_ = [p[0] for p in points]
        cols_ = [p[1] for p in points]
        return {"points": points, "bbox": (min(rows_), max(rows_), min(cols_), max(cols_)),
                "shape": vision.shape_name(points)}

    @staticmethod
    def _complete_piece(points, cells, rows, cols):
        """Add bright cells adjacent to an incomplete ghost shape.

        The ghost is drawn under the piece, so up to a few of its cells can be
        hidden by the piece itself.  A cell can only be added when it is
        bright, touches the current set, and the result is a shape the
        simulator already knows; otherwise the original set is kept.
        """
        points = list(points)
        if vision.shape_name(points) != "irregular":
            return points
        for _ in range(2):
            candidates = set()
            for r, c in points:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in points:
                        if cells[nr][nc] is not None and not vision.is_shadow(cells[nr][nc]):
                            candidates.add((nr, nc))
            for candidate in sorted(candidates):
                trial = points + [candidate]
                if vision.shape_name(trial) != "irregular":
                    points = trial
                    break
            else:
                break
        return points


    def _stack_of(self, record, remove_piece=True):
        """Resting stack of a record, optionally with the falling piece taken out."""
        rows, cols = record["rows"], record["cols"]
        stack = vision.fill(self._decode_record(record), rows, cols)
        if remove_piece:
            for r, c in record.get("piece") or []:
                if 0 <= r < rows and 0 <= c < cols:
                    stack[r][c] = None
        return stack, record.get("piece") is not None

    def _predict(self, record):
        """Drop-in-place prediction for the piece in this frame (no keys pressed)."""
        if record is None or not record.get("piece"):
            return None
        rows, cols = record["rows"], record["cols"]
        stack, _clean = self._stack_of(record)
        points = [tuple(p) for p in record["piece"]]
        shape = vision.normalise_shape(points)
        top = min(r for r, _c in points)
        left = min(c for _r, c in points)
        land = vision.landing_row(stack, rows, cols, shape, left)
        if land is None:
            return None
        return {"step": self.step_count, "points": [list(p) for p in points],
                "shape": vision.shape_name(points), "predicted_fall": land - top}

    def _record_lock(self, executed, record, prev_action=None):
        """Write the outcome of the previous action, if it was a hard drop.

        Called at step t with the frame of step t, so the row describes the
        action of step t-1 and nothing else.  `prev_action` is the action of
        step t-2 (None on the first action of the episode).
        """
        pending, self._pending = self._pending, None
        if executed != "hard_drop":
            self._last_lock = None
            return
        prev = self._prev_record
        if pending is None:
            row = {"step": self.step_count - 1, "action": executed, "shape": None, "fall": None,
                   "points": None, "predicted_fall": None, "match": False, "clears": None,
                   "new_holes": None, "max_height": None,
                   "reason": "the falling piece was not measured in the frame the drop was pressed on"}
        elif prev is None or record is None:
            row = {"step": pending["step"], "action": executed, "shape": pending["shape"],
                   "fall": None, "points": None, "predicted_fall": pending["predicted_fall"],
                   "match": False, "clears": None, "new_holes": None, "max_height": None,
                   "reason": "no frame was measured after the drop"}
        else:
            before, _ = self._stack_of(prev)
            after_full, _ = self._stack_of(record, remove_piece=False)
            next_piece = [tuple(p) for p in record.get("piece") or []]
            row = placement.lock_outcome(
                pending["step"], executed, [tuple(p) for p in pending["points"]], pending["shape"],
                before, after_full, record["rows"], record["cols"], pending["predicted_fall"],
                next_piece=next_piece, prev_action=prev_action)
        row["prev_action"] = prev_action
        # A row is measured only when both frames and the piece were available; an
        # unmeasured row says nothing about whether the piece locked.
        row["measured"] = pending is not None and prev is not None and record is not None
        self._locks.append(row)
        self._last_lock = row

    def _lock_text(self):
        row = self._last_lock
        if row is None or row["step"] != self.step_count - 1:
            return "your previous action was not a hard drop, so no piece locked."
        text = placement.format_lock(row)
        if row.get("prev_action") == "hard_drop":
            stats = placement.repeat_text(self._locks)
            if stats:
                text += " Derived for this episode: " + stats + "."
        return text

    def _repeat_drop_note(self, committed):
        """Derived check shown once when a hard drop is committed right after a measured unlocked one.

        Returns None when the commitment is not a repeat of an unlocked, measured hard drop, so
        the common case never costs an extra model call.  The model may commit the same
        action again; the note is a measurement, not a block.
        """
        if committed != "hard_drop":
            return None
        row = self._last_lock
        if (row is None or row["step"] != self.step_count - 1 or row["fall"] is not None
                or not row.get("measured", False)):
            return None
        return REPEAT_DROP_NOTE.format(
            step=row["step"],
            reason=row["reason"] or "no landing was confirmed by the frame after the drop",
            stats=placement.repeat_text(self._locks))

    def _ledger_text(self, last=10):
        rows = self._locks
        head = placement.calibration(rows)
        if not rows:
            return head
        lines = []
        for row in rows[-max(1, int(last)):]:
            if row["fall"] is None:
                body = "unverified ({})".format(row["reason"])
            else:
                body = "fall {} rows = {} points; new covered holes {:+d}; clears {}; max height {}".format(
                    row["fall"], row["points"], row["new_holes"],
                    "unknown" if row["clears"] is None else row["clears"], row["max_height"])
            pred = "" if row["predicted_fall"] is None else "; predicted fall {}{}".format(
                row["predicted_fall"], " (exact)" if row["match"] else "")
            lines.append("step {} hard drop of {}: {}{}".format(
                row["step"], row["shape"] or "unknown piece", body, pred))
        return head + "\n" + "\n".join(lines)

    @staticmethod
    def _compare(prev, cur):
        """Compact, honest description of what changed between two records."""
        if not prev or not cur:
            return "no previous frame is available for comparison"
        if prev["cols"] != cur["cols"] or prev["rows"] != cur["rows"]:
            return "the grid dimensions changed; comparison skipped"
        rows, cols = cur["rows"], cur["cols"]
        a, b = TetrisMemoryAgent._decode_record(prev), TetrisMemoryAgent._decode_record(cur)
        changed = vision.diff_cells(a, b, rows, cols)
        if not changed:
            return "no cell changed colour or occupancy"
        appeared = [p for p in changed if b[p[0]][p[1]] is not None and a[p[0]][p[1]] is None]
        vanished = [p for p in changed if b[p[0]][p[1]] is None and a[p[0]][p[1]] is not None]
        recoloured = [p for p in changed if a[p[0]][p[1]] is not None and b[p[0]][p[1]] is not None
                      and tuple(a[p[0]][p[1]]) != tuple(b[p[0]][p[1]])]

        def fmt(points):
            return " ".join("({},{})".format(r, c) for r, c in sorted(points)[:16]) or "none"

        return ("cells newly filled: {}\ncells emptied: {}\ncells recoloured: {}\n"
                "total changed cells: {}".format(fmt(appeared), fmt(vanished),
                                                 fmt(recoloured), len(changed)))

    # ------------------------------------------------------------- phase hooks

    def _board_context(self, record):
        source = "frame {}".format(record.get("frame_id")) if record else "unavailable"
        board = self._describe(record) + "\nLOCK LEDGER: " + placement.calibration(self._locks)
        return BOARD_SECTION.format(source=source, board=board)

    def _change_context(self):
        return self._compare(self._prev_record, self.board_record)

    def act(self, obs, prev_action=None, prev_validated_action=None):
        # Step t sees only the frame of step t.  The record the model is shown
        # at step t-1 is the "previous" record for this comparison: it was
        # measured from the frame the previous action produced, so nothing here
        # invents an outcome before the host has supplied one.
        # Set the previous frame first: piece detection compares this frame with
        # it.  (The parent assigned it after measuring, so the comparison used
        # the frame two steps back.)
        executed = prev_validated_action if prev_validated_action is not None else prev_action
        self._prev_record = self.board_record
        record = self._perceive()
        self.board_record = record
        self._last_frame_id = getattr(self, "current_frame_id", None)
        if record is not None:
            key = record.get("frame_id") or "_"
            self._ledger[key] = record
            if len(self._ledger) > 4:
                for old in list(self._ledger)[:-4]:
                    self._ledger.pop(old, None)
        if self.step_count > 0:
            # _exec_hist[-1] is the action of step t-2: it was appended on the
            # previous call, when executed was the action of step t-2.
            before = self._exec_hist[-1] if self._exec_hist else None
            self._record_lock(executed, record, before)
        self._exec_hist = (self._exec_hist + [executed])[-4:]
        self._pending = self._predict(record)
        return super().act(obs, prev_action, prev_validated_action)

    # The parent builds its prompts from module constants.  Replacing the two
    # phase methods keeps the tool dispatch (`_execute_*`) and the model-call
    # protocol exactly as the host expects, while changing what the model is
    # asked to do and what derived context it sees.
    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, spatial_guidance, map_analysis):
        import agents.memory_agent as base
        from prompt_builder.history import Message

        prev_step = self.step_count - 1
        if prev_validated_action is not None and prev_validated_action != prev_action:
            action_description = (
                "At the PREVIOUS step (step {}) your LLM output was \"{}\"; it was "
                "not a valid action and the environment ran \"{}\" instead.").format(
                    prev_step, prev_action, prev_validated_action)
        else:
            action_description = "At the PREVIOUS step (step {}) you took action: \"{}\"".format(
                prev_step, prev_validated_action or prev_action)

        initial_prompt = LOG_SYSTEM_TETRIS.format(
            action_description=action_description,
            observation=obs_text,
            board_section=self._board_context(self.board_record),
            change_section=LOG_EXTRA.format(change=self._change_context()),
            lock_section=LOCK_EXTRA.format(lock=self._lock_text()),
            file_index=self.memory.file_index())

        messages = [Message(role="user", content=initial_prompt)]
        total_in = total_out = 0
        turns = []

        for _iteration in range(self.max_log_iterations):
            try:
                response = self.client.generate(messages)
            except Exception as exc:
                turns.append({"prompt": messages[-1].content,
                              "raw_response": "[ERROR: {}]".format(exc),
                              "tool_results": ""})
                break
            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, has_done = base.parse_log_calls(assistant_text)
            result_parts = []
            for op, args in calls:
                result = self._execute_write_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)
            results_text = "\n\n".join(result_parts) if result_parts else ""
            turns.append({"prompt": messages[-1].content,
                          "raw_response": assistant_text,
                          "tool_results": results_text})
            messages.append(Message(role="assistant", content=assistant_text))
            if has_done or not calls:
                break
            messages.append(Message(role="user", content=base.LOG_CONTINUE.format(
                results=results_text, file_index=self.memory.file_index())))

        return {"initial_prompt": initial_prompt, "turns": turns,
                "input_tokens": total_in, "output_tokens": total_out}

    def _plan_validate_phase(self, obs_text, instruction_prompt,
                             diff_notice, spatial_guidance, map_analysis,
                             prev_validated_action=None):
        import agents.memory_agent as base
        from prompt_builder.history import Message

        initial_prompt = PLAN_SYSTEM_TETRIS.format(
            instruction_prompt=instruction_prompt,
            board_section=self._board_context(self.board_record),
            change_section=PLAN_EXTRA.format(change=self._change_context()),
            lock_section=LOCK_EXTRA.format(lock=self._lock_text()),
            file_index=self.memory.file_index(),
            observation=obs_text)

        messages = [Message(role="user", content=initial_prompt)]
        total_in = total_out = 0
        turns = []
        final_action = None
        model_id = "unknown"
        checked = False

        for _iteration in range(self.max_validate_iterations):
            try:
                response = self.client.generate(messages)
            except Exception as exc:
                turns.append({"prompt": messages[-1].content,
                              "raw_response": "[ERROR: {}]".format(exc),
                              "tool_results": "", "committed_action": None})
                break
            model_id = response.model_id
            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, committed_action = base.parse_plan_calls(assistant_text)
            result_parts = []
            for op, args in calls:
                result = self._execute_read_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)
            results_text = "\n\n".join(result_parts) if result_parts else ""
            turn = {"prompt": messages[-1].content,
                    "raw_response": assistant_text,
                    "tool_results": results_text,
                    "committed_action": committed_action}
            turns.append(turn)
            messages.append(Message(role="assistant", content=assistant_text))
            if committed_action is not None:
                note = None if checked else self._repeat_drop_note(committed_action)
                if note is not None:
                    # Shown once per step.  The model's next commitment is final, so
                    # the check informs the choice without blocking it.
                    checked = True
                    turn["derived_check"] = note
                    messages.append(Message(role="user", content=note))
                    continue
                final_action = committed_action
                break
            if not calls:
                final_action = self._get_fallback_action()
                break
            messages.append(Message(role="user", content=base.PLAN_CONTINUE.format(
                results=results_text)))

        if final_action is None:
            final_action = self._get_fallback_action()
        return {"initial_prompt": initial_prompt, "turns": turns,
                "final_action": final_action, "input_tokens": total_in,
                "output_tokens": total_out, "model_id": model_id}

    # ------------------------------------------------------------ persistence

    def export_state(self):
        return {"step_count": int(self.step_count),
                "unchanged_count": int(getattr(self, "_unchanged_count", 0)),
                "ledger": {str(k): v for k, v in self._ledger.items()},
                "last_frame_id": self._last_frame_id,
                "board_record": self.board_record,
                "prev_record": self._prev_record,
                "locks": self._locks,
                "pending": self._pending,
                "last_lock": self._last_lock,
                "exec_hist": list(self._exec_hist)}

    def import_state(self, state):
        self.step_count = int(state.get("step_count", 0))
        self._unchanged_count = int(state.get("unchanged_count", 0))
        self._ledger = dict(state.get("ledger") or {})
        self._last_frame_id = state.get("last_frame_id")
        self.board_record = state.get("board_record")
        self._prev_record = state.get("prev_record")
        self._locks = list(state.get("locks") or [])
        self._pending = state.get("pending")
        self._last_lock = state.get("last_lock")
        self._exec_hist = list(state.get("exec_hist") or [])


INITIAL_FILES_BY_ENV_LOCAL = {}
