"""M0 PNG storage. Only explicit recall returns pixels to the agent.

Extension over the parent: entries may carry a JSON-safe `meta` dict (used by
the dmvhard2 product index to link an image to its product id, session, page
and detected inset box).  All other behaviour is unchanged.
"""
import json
from pathlib import Path
import re

from PIL import Image


class VisualMemoryStore:
    def __init__(self, episode_dir, event=None):
        self.root = Path(episode_dir) / "visual"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.event = event or (lambda *args, **kwargs: None)
        self.data = json.loads(self.index_path.read_text()) if self.index_path.exists() else {
            "schema_version": 1, "next_id": 1, "images": {}}
        if not self.index_path.exists():
            self._persist()

    def _persist(self):
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(self.index_path)

    def reload(self):
        """Re-read the index after the host restores a checkpoint."""
        if self.index_path.exists():
            self.data = json.loads(self.index_path.read_text())

    def _entry(self, image_id):
        if not re.fullmatch(r"img_\d{4,}", image_id) or image_id not in self.data["images"]:
            known = ", ".join(sorted(self.data["images"])) or "none saved yet"
            raise ValueError(f"Unknown visual memory ID {image_id!r}; saved IDs: {known}")
        return self.data["images"][image_id]

    @staticmethod
    def _public(image_id, entry):
        return {"id": image_id, "step": entry["step"], "label": entry["label"]}

    def index(self):
        rows = [self._public(i, e) for i, e in self.data["images"].items()]
        return "VISUAL MEMORY INDEX\n" + ("\n".join(
            f'{r["id"]} | step {r["step"]} | {r["label"]}' for r in rows) or "(empty)")

    def save(self, image, step, label, note, debug=None, frame_id=None, meta=None):
        if image is None:
            raise ValueError("No current observation image")
        label, note = label.strip(), note.strip()
        if not label or len(label) > 120 or len(note) > 1000 or "\n" in label:
            raise ValueError("Label must be one line, 1\u2013120 characters; note at most 1000")
        image_id = f'img_{self.data["next_id"]:04d}'
        path = self.root / f"{image_id}.png"
        # PNG is lossless; do not render, crop, resize, or caption the observation.
        with path.open("xb") as stream:
            image.save(stream, format="PNG")
        entry = {"step": int(step), "label": label, "note": note,
                 "type": "frame", "path": f"visual/{path.name}"}
        if frame_id is not None:
            entry["frame_id"] = frame_id
        if meta is not None:
            entry["meta"] = meta
        if debug:
            entry["offline_debug"] = debug
        self.data["images"][image_id] = entry
        self.data["next_id"] += 1
        self._persist()
        self.event("visual_operation", operation="SAVE_FRAME", image_id=image_id, step=step, label=label)
        return self._public(image_id, entry)

    def update_meta(self, image_id, meta):
        """Replace the entry's meta dict (derived record; LLM metadata untouched)."""
        entry = self._entry(image_id)
        entry["meta"] = meta
        self._persist()
        return self._public(image_id, entry)

    def search(self, keyword):
        if not keyword.strip():
            raise ValueError("Search keyword must not be empty")
        rows = [self._public(i, e) for i, e in self.data["images"].items()
                if keyword.casefold() in (e["label"] + " " + e["note"]).casefold()]
        self.event("visual_operation", operation="SEARCH_IMAGE", keyword=keyword, matches=[r["id"] for r in rows])
        return rows

    def recall(self, image_id):
        entry = self._entry(image_id)
        with Image.open(self.root / f"{image_id}.png") as stored:
            image = stored.convert("RGB").copy()
        self.event("visual_operation", operation="RECALL_IMAGE", image_id=image_id,
                   attached_to_next_request=False)
        return self._public(image_id, entry), image

    def delete(self, image_id):
        self._entry(image_id)
        (self.root / f"{image_id}.png").unlink()
        del self.data["images"][image_id]
        self._persist()
        self.event("visual_operation", operation="DELETE_IMAGE", image_id=image_id)
        return {"deleted": image_id}

    def annotate(self, image_id, label=None, note=None):
        entry = self._entry(image_id)
        if label is not None:
            label = label.strip()
            if not label or len(label) > 120 or "\n" in label:
                raise ValueError("Label must be one line, 1\u2013120 characters")
            entry["label"] = label
        if note is not None:
            note = note.strip()
            if len(note) > 1000:
                raise ValueError("Note must be at most 1000 characters")
            entry["note"] = note
        self._persist()
        self.event("visual_operation", operation="ANNOTATE_IMAGE", image_id=image_id)
        return self._public(image_id, entry)
