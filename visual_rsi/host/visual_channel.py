"""Deterministic, host-owned derived-image channel for the M line."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageOps

from .dto import ContainmentError

PALETTE = {
    "wall": "#374151", "floor": "#f3f4f6", "visited": "#bfdbfe",
    "unknown": "#d1d5db", "door": "#92400e", "key": "#facc15",
    "goal": "#22c55e", "danger": "#ef4444", "agent": "#2563eb",
    "object": "#a855f7", "water": "#06b6d4", "other": "#6b7280",
}


@dataclass(frozen=True)
class VisualLimits:
    max_program_ops: int = 32
    # BALROG's native NLE tile observation is 1264x336.  The previous 768
    # bound made an exact saved frame impossible to recall even though the
    # current-frame V/M path already delivered it.  1536 admits that audited
    # native frame while retaining a finite host-enforced bound.
    max_output_side: int = 1536
    max_tile_frames: int = 16
    # There is no per-phase or per-episode cap on historical images: each
    # delivery is counted (phase_deliveries / episode_deliveries) and reported,
    # never refused.  The earlier 1-per-phase and 64-per-episode caps made 14%
    # of Crafter recalls in pilot r2 fail with an error.
    max_mark_text: int = 40
    # Canvas: no grid, primitive, label or path-length caps (a full 64x64 map
    # alone needs 4096 cells).  Colours stay semantic-palette-only, which keeps
    # a candidate from encoding pixels cell by cell; size is bounded only by
    # max_output_side.


class FrameStore:
    def __init__(self, limits: VisualLimits | None = None):
        self.limits = limits or VisualLimits()
        self.frames: dict[str, tuple[int, Image.Image]] = {}
        self.phase_deliveries = 0
        self.episode_deliveries = 0
        self.canvas_action_labels = 0
        self.last_description = ""

    def add(self, frame_id: str, step: int, image: Image.Image) -> None:
        if frame_id in self.frames:
            raise ContainmentError("duplicate frame id")
        if not isinstance(image, Image.Image):
            raise ContainmentError("frame is not a PIL image")
        self.frames[frame_id] = (step, image.convert("RGB").copy())

    def begin_phase(self) -> None:
        self.phase_deliveries = 0

    def _frame(self, frame_id: str, current_step: int) -> Image.Image:
        if frame_id not in self.frames:
            raise ValueError("unknown frame id")
        step, image = self.frames[frame_id]
        if step > current_step:
            raise ValueError("cannot reference a future frame")
        return image.copy()

    def _check_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if max(image.size) > self.limits.max_output_side:
            raise ValueError("derived image exceeds output side limit")
        return image

    def render(self, program: dict[str, Any], *, current_step: int, has_next_request: bool,
               legal_actions: list[str] | None = None) -> Image.Image:
        if not isinstance(program, dict) or set(program) - {"ops", "output"} or not isinstance(program.get("ops"), list):
            raise ContainmentError("visual request must be a JSON program")
        ops = program["ops"]
        if not ops or len(ops) > self.limits.max_program_ops:
            raise ValueError("visual program length limit")
        if not has_next_request:
            raise ValueError("no next request remains in phase")
        values: dict[str, Image.Image] = {}
        sources: list[tuple[str, int]] = []
        grids: dict[str, tuple[int, int, int]] = {}
        action_labels = 0
        for item in ops:
            if not isinstance(item, dict) or set(item) - {"op", "id", "src", "srcs", "as", "box", "max_side", "cols", "labels", "marks", "grid", "cell_px", "prims"}:
                raise ContainmentError("unknown visual operation field")
            op, alias = item.get("op"), item.get("as")
            if not isinstance(alias, str) or alias in values:
                raise ValueError("visual operation requires a unique alias")
            if op == "frame":
                values[alias] = self._frame(item.get("id"), current_step)
                sources.append((str(item.get("id")), self.frames[item.get("id")][0]))
            elif op == "canvas":
                if set(item) != {"op", "grid", "cell_px", "as"}:
                    raise ValueError("canvas requires grid, cell_px and alias")
                grid, cell_px = item["grid"], item["cell_px"]
                if not isinstance(grid, list) or len(grid) != 2 or any(type(v) is not int or v < 1 for v in grid):
                    raise ValueError("canvas grid must be two positive integers")
                if type(cell_px) is not int or cell_px < 1:
                    raise ValueError("invalid canvas cell_px")
                width, height = grid[0] * cell_px, grid[1] * cell_px
                if max(width, height) > self.limits.max_output_side:
                    raise ValueError("derived image exceeds output side limit")
                values[alias] = Image.new("RGB", (width, height), PALETTE["unknown"])
                grids[alias] = (grid[0], grid[1], cell_px)
            elif op == "draw":
                if set(item) != {"op", "src", "prims", "as"}:
                    raise ValueError("draw requires src, prims and alias")
                src = item["src"]
                if src not in values:
                    raise ValueError("unknown visual source alias")
                image = values[src].copy()
                cols, rows, cell_px = grids.get(src, ((image.width + 31) // 32, (image.height + 31) // 32, 32))
                prims = item["prims"]
                if not isinstance(prims, list):
                    raise ValueError("draw prims must be a list")
                painter = ImageDraw.Draw(image)
                def point(value):
                    if (not isinstance(value, list) or len(value) != 2 or
                            any(type(v) is not int for v in value) or
                            not (0 <= value[0] < cols and 0 <= value[1] < rows)):
                        raise ValueError("canvas coordinate out of bounds")
                    return value[0] * cell_px, value[1] * cell_px
                for prim in prims:
                    if not isinstance(prim, dict):
                        raise ValueError("invalid canvas primitive")
                    kind = prim.get("type")
                    if kind == "cell":
                        if set(prim) != {"type", "at", "color"} or prim["color"] not in PALETTE:
                            raise ValueError("canvas cell needs a semantic palette color")
                        x, y = point(prim["at"])
                        painter.rectangle((x, y, min(x + cell_px - 1, image.width - 1),
                                           min(y + cell_px - 1, image.height - 1)), fill=PALETTE[prim["color"]])
                    elif kind == "agent":
                        if set(prim) - {"type", "at", "facing"} or prim.get("facing", "N") not in {"N", "E", "S", "W"}:
                            raise ValueError("invalid canvas agent")
                        x, y = point(prim["at"])
                        radius = max(2, cell_px // 4)
                        cx, cy = x + cell_px // 2, y + cell_px // 2
                        painter.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=PALETTE["agent"])
                        dx, dy = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}[prim.get("facing", "N")]
                        painter.line((cx, cy, cx + dx * radius * 2, cy + dy * radius * 2),
                                     fill=PALETTE["wall"], width=max(1, cell_px // 12))
                    elif kind == "label":
                        if set(prim) != {"type", "at", "text"} or not isinstance(prim["text"], str):
                            raise ValueError("canvas label too long or invalid")
                        x, y = point(prim["at"])
                        painter.text((x, y), prim["text"], fill=PALETTE["wall"])
                        action_labels += sum(action.casefold() in prim["text"].casefold()
                                             for action in (legal_actions or []) if action)
                    elif kind == "path":
                        pts = prim.get("pts")
                        if set(prim) != {"type", "pts"} or not isinstance(pts, list) or len(pts) < 2:
                            raise ValueError("invalid canvas path")
                        pixels = [point(p) for p in pts]
                        painter.line([(x + cell_px // 2, y + cell_px // 2) for x, y in pixels],
                                     fill=PALETTE["visited"], width=max(1, cell_px // 8))
                    else:
                        raise ValueError("unknown canvas primitive")
                values[alias] = image
                grids[alias] = (cols, rows, cell_px)
            elif op in {"crop", "resize", "grayscale", "quantize", "annotate"}:
                src = item.get("src")
                if src not in values:
                    raise ValueError("unknown visual source alias")
                image = values[src].copy()
                if op == "crop":
                    box = item.get("box")
                    if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box):
                        raise ValueError("crop box must contain four integers")
                    if box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1] or box[2] > image.width or box[3] > image.height:
                        raise ValueError("crop box out of bounds")
                    image = image.crop(tuple(box))
                elif op == "resize":
                    max_side = item.get("max_side")
                    if type(max_side) is not int or max_side <= 0 or max_side > self.limits.max_output_side:
                        raise ValueError("invalid resize max_side")
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                elif op == "grayscale":
                    image = ImageOps.grayscale(image).convert("RGB")
                elif op == "quantize":
                    colors = item.get("max_side", 16)
                    if type(colors) is not int or colors < 2 or colors > 256:
                        raise ValueError("invalid quantize colors")
                    image = image.quantize(colors=colors).convert("RGB")
                else:
                    image = self._annotate(image, item.get("marks", []))
                values[alias] = image
                if op in {"grayscale", "quantize", "annotate"} and src in grids:
                    grids[alias] = grids[src]
            elif op == "diff":
                srcs = item.get("srcs")
                if not isinstance(srcs, list) or len(srcs) != 2 or any(src not in values for src in srcs):
                    raise ValueError("diff requires two known sources")
                left, right = values[srcs[0]], values[srcs[1]]
                if left.size != right.size:
                    right = right.resize(left.size, Image.Resampling.NEAREST)
                values[alias] = ImageChops.difference(left, right)
            elif op == "tile":
                srcs = item.get("srcs")
                cols = item.get("cols")
                if not isinstance(srcs, list) or not 1 <= len(srcs) <= self.limits.max_tile_frames or not isinstance(cols, int) or not 1 <= cols <= len(srcs):
                    raise ValueError("invalid tile limits")
                if any(src not in values for src in srcs):
                    raise ValueError("unknown tile source alias")
                labels = item.get("labels", [])
                if labels and (not isinstance(labels, list) or len(labels) != len(srcs)):
                    raise ValueError("tile labels must match sources")
                width = max(values[src].width for src in srcs)
                height = max(values[src].height for src in srcs)
                rows = (len(srcs) + cols - 1) // cols
                # Check the existing output contract before allocating the tile.
                # A large composition must not exhaust the trusted host first.
                if max(width * cols, height * rows) > self.limits.max_output_side:
                    raise ValueError("derived image exceeds output side limit")
                image = Image.new("RGB", (width * cols, height * rows), "white")
                draw = ImageDraw.Draw(image)
                for index, src in enumerate(srcs):
                    x, y = (index % cols) * width, (index // cols) * height
                    image.paste(values[src], (x, y))
                    if labels:
                        draw.rectangle((x, y, x + min(width, 240), y + 18), fill="white")
                        draw.text((x + 2, y + 2), str(labels[index])[: self.limits.max_mark_text], fill="black")
                values[alias] = image
            else:
                raise ValueError("unknown visual operation")
            values[alias] = self._check_image(values[alias])
        output = program.get("output")
        if output not in values:
            raise ValueError("visual output alias is missing")
        self.phase_deliveries += 1
        self.episode_deliveries += 1
        self.canvas_action_labels += action_labels
        kinds = sorted({str(item.get("op")) for item in ops} - {"frame"})
        # Steps only: host frame IDs are not valid RECALL_IMAGE arguments.
        origin = ", ".join(f"step {step}" for _frame, step in sources) or "no stored frame"
        self.last_description = ("Historical image rendered by the host for your last tool call, from "
                                 f"{origin}" + (f"; operations: {', '.join(kinds)}" if kinds else "") + ".")
        return values[output].copy()

    def _annotate(self, image: Image.Image, marks: Any) -> Image.Image:
        if not isinstance(marks, list) or len(marks) > 32:
            raise ValueError("invalid annotation list")
        image = image.copy()
        draw = ImageDraw.Draw(image)
        for mark in marks:
            if not isinstance(mark, dict) or mark.get("type") not in {"point", "box", "arrow"}:
                raise ValueError("invalid annotation type")
            text = str(mark.get("text", ""))
            if len(text) > self.limits.max_mark_text:
                raise ValueError("annotation text too long")
            box = mark.get("box")
            if not isinstance(box, list) or len(box) not in {2, 4} or any(type(x) is not int for x in box):
                raise ValueError("annotation coordinates must be integer")
            if any(x < 0 for x in box) or box[-2] > image.width or box[-1] > image.height:
                raise ValueError("annotation out of bounds")
            if mark["type"] == "point":
                x, y = box
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
            elif mark["type"] == "box":
                draw.rectangle(tuple(box), outline="red", width=2)
            else:
                draw.line(tuple(box), fill="red", width=2)
            if text:
                draw.text((box[0], box[1]), text, fill="red")
        return image
