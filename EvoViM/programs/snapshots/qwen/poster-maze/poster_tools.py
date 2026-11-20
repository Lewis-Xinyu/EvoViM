"""Postermaze poster memory tools (line M, generation 2).

Two tools extend the LLM's action space.  The LLM decides when to call them;
the handlers only compute from public observations and return numbers/text.

RECORD_POSTER  Store one visible poster from the current navigation frame.
               A poster sits on the first wall in its sighting direction, so
               the (cell, side) answer is computed by walking the current
               maze graph from the agent's cell toward that wall (corridor
               sightings are placed on the wall actually in front, not on
               the agent's own cell).  The record keeps the sighting
               geometry so the label is re-derived with the latest map at
               question time.  The LLM picks which visible poster to store
               and may override the guessed side or add a note.  Re-recording
               the same wall with a similar poster keeps the more prominent
               sighting instead of duplicating the record.

MATCH_POSTER   Compare the poster in the current question frame against all
               records stored during navigation.  Records are grouped by
               (cell, side) so duplicate sightings of one poster cannot
               masquerade as a close rival; the answer is a GROUP when the
               best group clears both the similarity threshold and the
               margin over the next-best group, otherwise not_seen.  The
               LLM still commits the final answer with <|ACTION|>.

Both handlers accept all-optional JSON parameters and return a short text
result (never a crash) when the arguments are empty or the situation does not
apply.  No layout, route, colour or timing constant from any one episode is
baked in: detection is hue-agnostic and matching is lighting/rotation
invariant.
"""
from __future__ import annotations

import numpy as np

from registry import Tool

from agents.postermaze_agent import (
    ALREADY_REC_SIM,
    GRID,
    PosterMazeAgent,
    detect_posters,
    match_groups,
    sim,
)

MAX_RECORDS = 40
REFRESH_SIM = ALREADY_REC_SIM  # single source of truth (postermaze_agent)


def install_poster(agent, registry):
    def record(which=None, side=None, note=None):
        if getattr(agent, "_pm_question", False):
            return ("The current frame is a question, not a navigation view; "
                    "recording it would attach the wrong cell. Use MATCH_POSTER instead.")
        if agent.current_image is None:
            return "No current image is available; nothing recorded."
        posters = detect_posters(agent.current_image)
        if not posters:
            return "No poster detected in the current view; nothing recorded."
        index = 0
        if which is not None and str(which).strip():
            token = str(which).strip().lower()
            if token in ("front", "left", "right"):
                for i, p in enumerate(posters):
                    if p["wall"] == token:
                        index = i
                        break
            else:
                try:
                    index = int(token)
                except ValueError:
                    return "'which' must be an index (0, 1, ...) or front/left/right."
            if not 0 <= index < len(posters):
                return "'which' out of range; {} poster(s) visible.".format(len(posters))
        poster = posters[index]
        cell = agent.pm_cell
        facing = agent.pm_facing
        if not cell or facing not in ("N", "E", "S", "W"):
            return "Cell/facing not known yet; nothing recorded."
        r, c = int(cell[0]), int(cell[1])
        if side is not None:
            dir_name = str(side)
        else:
            dir_name = PosterMazeAgent._wall_to_side(poster["wall"], facing)
        # The poster sits on the first wall in that direction; place it with
        # the current maze graph (corridor sightings point past this cell);
        # dist = open cells between here and that wall.
        cell_lbl, side_lbl, status, dist = agent._label_for_sighting((r, c), dir_name)
        new_lm = np.asarray(poster["lm"]).reshape(GRID, GRID)

        updated = None
        for record_ in agent.pm_posters:
            if record_.get("sight_cell") and record_.get("sight_facing") in ("N", "E", "S", "W"):
                dir0 = PosterMazeAgent._wall_to_side(record_.get("sight_wall", "front"),
                                                     record_["sight_facing"])
                cell0, side0, st0, dist0 = agent._label_for_sighting(record_["sight_cell"], dir0)
            else:
                cell0, side0 = record_.get("cell"), record_.get("side")
                st0, dist0 = None, None
            if cell0 == cell_lbl and side0 == side_lbl:
                stored = np.asarray(record_["lm"]).reshape(GRID, GRID)
                best = sim(new_lm, stored)
                if best >= REFRESH_SIM:
                    if int(poster["npix"]) > int(record_.get("npix", 0)):
                        record_.update(
                            lm=list(poster["lm"]), channel=poster["channel"],
                            npix=int(poster["npix"]), step=int(agent.step_count),
                            frame_id=agent.current_frame_id)
                    # A closer CONFIRMED front sighting of the same poster
                    # upgrades the record's sighting geometry: the label is
                    # re-derived from this geometry at question time, and a
                    # wall distance <= QUOTA_DIST with a certain stop is
                    # what the well-seen quota counts.  Walking the wall
                    # cell later therefore upgrades an early corridor
                    # sighting in place, without a duplicate record.
                    if (side is None and poster["wall"] == "front"
                            and status in ("wall", "boundary")
                            and (st0 not in ("wall", "boundary")
                                 or dist < (dist0 or 0))):
                        record_.update(sight_cell=[r, c], sight_facing=facing,
                                       sight_wall="front")
                    if note:
                        record_["note"] = str(note)[:200]
                    updated = (record_["id"], best)
                    break
        if updated is not None:
            return ("Already have {} (sim {:.2f}; same wall, same poster; kept the "
                    "more prominent sighting). Answer: {},{},{}").format(
                        updated[0], updated[1], cell_lbl[0], cell_lbl[1], side_lbl)

        if len(agent.pm_posters) >= MAX_RECORDS:
            agent.pm_posters.pop(0)
        agent.pm_poster_seq = int(agent.pm_poster_seq) + 1
        record_id = "P{:03d}".format(agent.pm_poster_seq)
        record_ = {
            "id": record_id, "cell": cell_lbl, "side": side_lbl,
            "sight_cell": [r, c], "sight_facing": facing, "sight_wall": poster["wall"],
            "lm": list(poster["lm"]), "channel": poster["channel"],
            "step": int(agent.step_count), "frame_id": agent.current_frame_id,
            "npix": int(poster["npix"]), "wall": poster["wall"],
        }
        if note:
            record_["note"] = str(note)[:200]
        agent.pm_posters.append(record_)
        image_id = None
        visual = getattr(agent, "visual", None)
        if visual is not None and agent.current_frame_id:
            try:
                saved = visual.save(agent.current_image, agent.step_count,
                                    "poster {} at ({},{}) {}".format(record_id, cell_lbl[0], cell_lbl[1], side_lbl),
                                    "")
                image_id = saved.get("id")
                if image_id:
                    record_["image_id"] = image_id
            except Exception:  # provenance copy is best-effort, never fatal
                image_id = None
        extra = " (frame saved as {})".format(image_id) if image_id else ""
        tag = "certain" if status == "wall" else "estimated from map"
        return ("Recorded {} -> answer {},{},{} ({} wall; location {}) {}"
                " (step {}, prominence {} px)").format(
                    record_id, cell_lbl[0], cell_lbl[1], side_lbl, poster["wall"], tag,
                    extra, agent.step_count, poster["npix"])

    def match(top=None):
        if not agent.pm_posters:
            return ("No posters were recorded during navigation; nothing to match. "
                    "Recommendation: answer not_seen.")
        if agent.current_image is None:
            return "No current image is available; cannot match."
        posters = detect_posters(agent.current_image)
        if not posters:
            return ("No poster detected in the current view; cannot match. "
                    "Recommendation: answer not_seen.")
        limit = 6
        if top is not None:
            try:
                limit = max(1, min(10, int(top)))
            except (TypeError, ValueError):
                limit = 6
        groups, rec, reason = match_groups(posters[0]["lm"], agent._poster_answers())
        lines = ["QUESTION POSTER GROUPS ({} stored posters; groups by (cell, side), "
                 "best first; [est] = location estimated from maze map):".format(
                     len(agent.pm_posters))]
        for g in groups[:limit]:
            est = " [est]" if g.get("label_status") != "wall" else ""
            ids = g["best_id"] if len(g["ids"]) == 1 else "+".join(g["ids"])
            lines.append("  ({},{}) side {} sim={} margin={} [{}{}]".format(
                g["cell"][0], g["cell"][1], g["side"], g["sim"],
                g["margin"], ids, est))
        lines.append("RECOMMENDATION: {} ({})".format(rec, reason))
        return "\n".join(lines)

    obj = {"type": "object", "additionalProperties": False}
    registry.register(Tool(
        "RECORD_POSTER",
        "Store a poster visible in the current navigation frame with its answer "
        "location (cell, side): the cell directly in front of the wall that "
        "carries the poster, computed by walking the maze map from your cell "
        "toward that wall. Call it whenever a poster is visible and not yet "
        "recorded: questions later show re-photographed posters and you can "
        "only answer about posters you stored. 'which' picks among visible "
        "posters (\"0\" = most prominent, or front/left/right); 'side' "
        "overrides the guessed wall side; 'note' is an optional short "
        "description.",
        {**obj, "properties": {
            "which": {"type": "string"},
            "side": {"type": "string", "enum": ["N", "E", "S", "W"]},
            "note": {"type": "string", "maxLength": 200}}, "required": []},
        record, ("LOG", "PLAN"), "memory_write", "text"))
    registry.register(Tool(
        "MATCH_POSTER",
        "Compare the poster in the current question frame against every poster "
        "stored during navigation. Records are grouped by (cell, side): a "
        "group is recommended only when it clears the similarity threshold "
        "AND leads the next-best group by a safe margin; otherwise answer "
        "not_seen. Use it on a question step before committing your answer.",
        {**obj, "properties": {
            "top": {"type": "integer", "minimum": 1, "maximum": 10}}, "required": []},
        match, ("PLAN",), "memory_read", "text"))
