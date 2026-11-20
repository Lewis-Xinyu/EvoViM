"""MemoryAgent with a derived wall map, a host sighting index and numeric matching.

Parent behaviour is kept: LOG records only the previous step's outcome, the
LLM still chooses every write, read and answer, and all parent phases run.
Generation 2 changes, all structural:
  (a) question phase is sticky: once a 'Question k/n' observation appears, no
      later frame is indexed as a maze sighting (the parent indexed question
      photos with a stale pose, so MATCH_POSTER could match a photo to itself);
  (b) question photos are kept in their own list, separate from sightings;
  (c) the maze observation gains a bounded frontier hint (text only, no moves).
Generation 3 changes, both in the host-side detector path (no new tools, no LLM decisions moved):
  (d) question photos are localised by pv.sight_question (Otsu split of chroma), not the
      median rule that failed when the poster covered more than half of the photo;
  (e) maze frames keep a scene-chroma sample from frames the detector called none; the
      near-view fallback in pv.sight uses the median of these samples as its baseline, so
      a close poster view that fills the frame is still a sighting.
"""
from __future__ import annotations

import re

import numpy as np

from agents import poster_map as pm
from agents import poster_vision as pv
from agents.memory_agent import MemoryAgent

SIGHTING_CAP = 400
QUESTION_CAP = 16
SCENE_SAMPLE_CAP = 200  # most recent non-poster frames kept for the scene baseline
FRONTIER_HINT_LIMIT = 3
QUESTION_RE = re.compile(r"Question \d+/\d+")
ANSWER_HINT = ("Answer format: reply `answer r,c,D` (row,column of the cell in front of the poster; "
               "wall side N, E, S or W) or `answer not_seen`. Write the word answer once.")


class PosterMemoryAgent(MemoryAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder, config)
        self._init_poster_state()

    def _init_poster_state(self):
        self.wall_state = pm.new_state()
        self.sightings = []
        self.sighting_kinds = {}
        self.question_phase = False
        self.question_photos = []
        self.scene_samples = []  # chroma medians of maze frames the detector called none

    def _scene_baseline(self):
        if not self.scene_samples:
            return None
        return float(np.median(self.scene_samples[-SCENE_SAMPLE_CAP:]))

    def reset(self):
        super().reset()
        self._init_poster_state()

    def act(self, obs, prev_action=None, prev_validated_action=None):
        text = (obs or {}).get("text", {}) or {}
        short = text.get("short_term_context", "") or ""
        if QUESTION_RE.search(short):
            self.question_phase = True
        pm.observe(self.wall_state, text, self.step_count)
        if self.question_phase:
            self._record_question_photo()
        else:
            self._index_sighting()
        return super().act(obs, prev_action, prev_validated_action)

    def _index_sighting(self):
        image = getattr(self, "current_image", None)
        if image is None:
            return
        rec = pv.sight(image, scene_chroma=self._scene_baseline())
        self.sighting_kinds[rec["kind"]] = self.sighting_kinds.get(rec["kind"], 0) + 1
        if rec["kind"] == "none":
            self.scene_samples.append(pv.chroma_median(image))
            del self.scene_samples[:-SCENE_SAMPLE_CAP]
        if rec["kind"] != "poster" or len(self.sightings) >= SIGHTING_CAP:
            return
        if self.sightings and self.sightings[-1]["step"] == self.step_count:
            return
        self.sightings.append({
            "id": "s%d" % len(self.sightings),
            "step": self.step_count,
            "frame_id": getattr(self, "current_frame_id", None),
            "box": rec["box"],
            "near": rec["near"],
            "cover": rec["cover"],
            "via": rec.get("via", "frame"),
            "size": list(image.size),
            "pos": list(self.wall_state["pos"]) if self.wall_state["pos"] else None,
            "desc": pv.describe(image, rec["box"]),
            "note": "",
        })

    def _record_question_photo(self):
        """Keep the question photo apart from the maze index; never pooled as a sighting."""
        image = getattr(self, "current_image", None)
        if image is None or len(self.question_photos) >= QUESTION_CAP:
            return
        rec = pv.sight_question(image)
        entry = {"step": self.step_count, "frame_id": getattr(self, "current_frame_id", None),
                 "kind": rec["kind"], "size": list(image.size), "method": rec.get("method")}
        if rec["kind"] == "poster":
            entry["box"] = rec["box"]
        self.question_photos.append(entry)

    def _format_obs(self, obs):
        base = super()._format_obs(obs)
        if QUESTION_RE.search(base):
            return base + "\n" + ANSWER_HINT
        if self.wall_state["pos"] is not None and self.wall_state["size"] is not None:
            hints = pm.frontier_hint(self.wall_state, FRONTIER_HINT_LIMIT)
            if hints:
                parts = ["cell (%d,%d) via %d steps (unknown side %s)" % (h["cell"][0], h["cell"][1],
                                                                       h["steps"], h["unknown_side"])
                         for h in hints]
                base += "\nDerived map hint (not an action): nearest unexplored edges: " + "; ".join(parts) + "."
        return base
