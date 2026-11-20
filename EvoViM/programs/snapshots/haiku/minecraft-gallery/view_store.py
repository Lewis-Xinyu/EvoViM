"""Episode-local store of whole walk views for line M.

A view is a walk frame saved with the doorway number k that the model read from it and
door_x, the doorway's horizontal position across the image (0 = left edge, 1 = right edge).
Views are kept at reduced resolution for matching; the full-frame size and the resolution
factor are recorded so matched boxes map back to host frames.
"""
from __future__ import annotations

import copy
import os
import re

import numpy as np
from PIL import Image

VIEW_MAX_SIDE = 640
MAX_VIEWS = 24
DOOR_MAX = 999
DEFAULT_DOOR_X = 0.5
LABEL_RE = re.compile(r"^\s*(\d{1,3})\s*,\s*([LR])\s*$", re.IGNORECASE)
ANSWER_RE = re.compile(r"^\s*answer\s+(not_seen|\d{1,3}\s*,\s*[LR])\s*$", re.IGNORECASE)


def normalize_label(label: str) -> str | None:
    """'3, l' -> '3,L'; anything that is not a doorway label -> None."""
    match = LABEL_RE.match(str(label))
    if not match:
        return None
    return "%d,%s" % (int(match.group(1)), match.group(2).upper())


def view_factor(size) -> float:
    return min(1.0, VIEW_MAX_SIDE / float(max(size[0], size[1])))


class ViewStore:
    def __init__(self):
        self.directory: str | None = None
        self.items: list[dict] = []
        self.seq = 0

    def bind(self, directory: str) -> None:
        self.directory = directory

    def _path(self, view_id: str) -> str:
        if self.directory is None:
            raise RuntimeError("view store is not bound to an episode directory")
        return os.path.join(self.directory, view_id + ".png")

    def get(self, view_id: str) -> dict:
        for item in self.items:
            if item["id"] == view_id:
                return item
        raise ValueError("unknown view id: %s" % view_id)

    def find_frame(self, frame_id):
        for item in self.items:
            if item["frame_id"] == frame_id:
                return item
        return None

    def add(self, image: Image.Image, *, step: int, frame_id: str, door: int, door_x: float) -> dict:
        if len(self.items) >= MAX_VIEWS:
            raise ValueError("view limit reached (%d); delete a view first" % MAX_VIEWS)
        width, height = image.size
        factor = view_factor((width, height))
        small_size = (max(1, int(round(width * factor))), max(1, int(round(height * factor))))
        small = image.convert("RGB").resize(small_size, Image.Resampling.BILINEAR)
        os.makedirs(self.directory, exist_ok=True)
        self.seq += 1
        view_id = "view_%04d" % self.seq
        small.save(self._path(view_id), format="PNG")
        item = {"id": view_id, "step": int(step), "frame_id": frame_id, "door": int(door),
                "door_x": float(door_x), "full_size": [int(width), int(height)],
                "view_size": [small_size[0], small_size[1]], "factor": small_size[0] / float(width)}
        self.items.append(item)
        return item

    def load_view(self, view_id: str):
        """Return (float RGB array at view resolution, resolution factor)."""
        item = self.get(view_id)
        array = np.asarray(Image.open(self._path(view_id)).convert("RGB"), dtype=np.float32)
        return array, float(item["factor"])

    def relabel(self, view_id: str, door: int, door_x: float) -> dict:
        item = self.get(view_id)
        item["door"] = int(door)
        item["door_x"] = float(door_x)
        return item

    def delete(self, view_id: str) -> dict:
        item = self.get(view_id)
        self.items.remove(item)
        try:
            os.remove(self._path(view_id))
        except (FileNotFoundError, RuntimeError):
            pass
        return item

    def coverage(self) -> dict:
        counts: dict = {}
        for item in self.items:
            counts[item["door"]] = counts.get(item["door"], 0) + 1
        return dict(sorted(counts.items()))

    def clear(self) -> None:
        for item in list(self.items):
            try:
                os.remove(self._path(item["id"]))
            except (FileNotFoundError, RuntimeError):
                pass
        self.items = []

    def export_state(self) -> dict:
        return {"seq": self.seq, "items": copy.deepcopy(self.items)}

    def import_state(self, state: dict) -> None:
        items = state.get("items", [])
        if not isinstance(items, list) or not all(isinstance(i, dict) and "id" in i for i in items):
            raise ValueError("invalid view index in checkpoint")
        self.seq = int(state.get("seq", 0))
        self.items = copy.deepcopy(items)

