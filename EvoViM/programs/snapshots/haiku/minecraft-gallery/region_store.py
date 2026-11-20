"""Episode-local store of saved image regions (PNG crops plus a JSON index).

Pixels never leave the candidate process except through host visual requests
that name a host frame id and a crop box; the crop files stay in the scratch
directory so checkpoints carry them along with the text memory.
"""
from __future__ import annotations

import copy
import os
import re

from PIL import Image

LABEL_RE = re.compile(r"^\s*(\d{1,3})\s*,\s*([LR])\s*$", re.IGNORECASE)
ANSWER_RE = re.compile(r"^\s*answer\s+(not_seen|\d{1,3}\s*,\s*[LR])\s*$", re.IGNORECASE)
MAX_REGIONS = 80
MAX_LABEL = 40


def normalize_label(label: str) -> str | None:
    """'3, l' -> '3,L'; anything that is not a doorway label -> None."""
    match = LABEL_RE.match(str(label))
    if not match:
        return None
    return "%d,%s" % (int(match.group(1)), match.group(2).upper())


class RegionStore:
    def __init__(self):
        self.directory: str | None = None
        self.items: list[dict] = []
        self.seq = 0

    def bind(self, directory: str) -> None:
        self.directory = directory

    def _path(self, region_id: str) -> str:
        if self.directory is None:
            raise RuntimeError("region store is not bound to an episode directory")
        return os.path.join(self.directory, region_id + ".png")

    def get(self, region_id: str) -> dict:
        for item in self.items:
            if item["id"] == region_id:
                return item
        raise ValueError("unknown region id: %s" % region_id)

    def add(self, crop: Image.Image, *, step: int, frame_id: str | None, box: list[int],
            label: str, label_valid: bool, question_frame: bool) -> dict:
        if len(self.items) >= MAX_REGIONS:
            raise ValueError("region limit reached (%d); delete a region first" % MAX_REGIONS)
        os.makedirs(self.directory, exist_ok=True)
        self.seq += 1
        region_id = "reg_%04d" % self.seq
        crop.save(self._path(region_id), format="PNG")
        item = {"id": region_id, "step": int(step), "frame_id": frame_id, "box": [int(v) for v in box],
                "label": label, "label_valid": bool(label_valid), "question_frame": bool(question_frame)}
        self.items.append(item)
        return item

    def load(self, region_id: str) -> Image.Image:
        return Image.open(self._path(region_id)).convert("RGB")

    def relabel(self, region_id: str, label: str, label_valid: bool) -> dict:
        item = self.get(region_id)
        item["label"] = label
        item["label_valid"] = bool(label_valid)
        return item

    def delete(self, region_id: str) -> dict:
        item = self.get(region_id)
        self.items.remove(item)
        try:
            os.remove(self._path(region_id))
        except FileNotFoundError:
            pass
        return item

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
            raise ValueError("invalid region index in checkpoint")
        self.seq = int(state.get("seq", 0))
        self.items = copy.deepcopy(items)
