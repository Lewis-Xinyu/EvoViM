"""Episode-local scene measurement for Pac-Man screenshots (line M).

Every quantity here is re-measured from public frames of the *current* episode.
Nothing from a recorded episode is baked in: the pellet lattice, playfield
bounds, sprite positions, pellet inventory and passable directions all come from
the pixels of the frame at hand.

The module only describes the scene.  It never chooses an action, and it does
not write any memory file: the agent decides what to store.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

DIRS = ("up", "down", "left", "right")
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_TO_DIR = {"move_up": "up", "move_down": "down",
                 "move_left": "left", "move_right": "right"}


# --------------------------------------------------------------------- colour
def _hsv(rgb):
    r, g, b = (float(v) / 255.0 for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    d = mx - mn
    if d <= 1e-6:
        return 0.0, 0.0, mx
    if mx == r:
        h = ((g - b) / d) % 6.0
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return h * 60.0, (d / mx if mx else 0.0), mx


def colour_name(rgb):
    """Hue-based name; a descriptor for the model, not a game rule."""
    h, s, v = _hsv(rgb)
    if v < 0.30 or s < 0.25:
        return "grey"
    if 48.0 <= h <= 78.0 and s >= 0.45:
        return "yellow"
    if v >= 0.35 and s >= 0.25 and 200.0 <= h < 285.0:
        return "blue"
    if 160.0 <= h < 200.0:
        return "cyan"
    if 20.0 <= h < 48.0:
        return "orange"
    if 285.0 <= h <= 348.0:
        return "pink"
    if h < 20.0 or h > 348.0:
        return "red"
    return "other"


def _masks(a):
    f = a.astype(np.int16)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(2)
    mn = f.min(2)
    return {
        # pellets are neutral near-white: blue and green channels agree
        "white": (mn > 232) & ((mx - mn) < 18) & (np.abs(b - g) < 10),
        # the whole pink/red wall family: bright red, pale fill, and the
        # anti-aliased edge between them (blue never below green)
        "fill": ((r > 232) & (b > 150) & (b >= g) & (r >= g + 18)),
        "outline": (r > 190) & (g < 115) & (b < 135),
        # candidate sprite pixels: bright and coloured
        "sprite": (mx > 150) & ((mx - mn) > 38),
    }


def _components(mask, lo, hi, max_side=None, colour_of=None):
    lab, n = ndimage.label(mask)
    if n == 0:
        return []
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        sz = int(m.sum())
        if sz < lo or sz > hi:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if max_side is not None and (w > max_side or h > max_side):
            continue
        ys, xs = np.nonzero(m)
        item = {"x": float(xs.mean() + sl[1].start),
                "y": float(ys.mean() + sl[0].start),
                "n": sz, "w": int(w), "h": int(h),
                "fill": float(sz) / float(max(w * h, 1))}
        if colour_of is not None:
            gy, gx = ys + sl[0].start, xs + sl[1].start
            pix = colour_of[gy, gx]
            vals, counts = np.unique(pix, axis=0, return_counts=True)
            item["rgb"] = tuple(int(v) for v in vals[int(np.argmax(counts))])
        out.append(item)
    return out


def _drop_small_components(mask, min_side, min_area):
    """Remove compact blobs (sprites, text glyphs) from a wall mask."""
    keep = np.zeros_like(mask)
    lab, n = ndimage.label(mask)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if w >= min_side or h >= min_side:
            keep[sl][lab[sl] == i] = True
        else:
            m = lab[sl] == i
            if int(m.sum()) >= min_area:
                keep[sl][m] = True
    return keep


def _turn(dr, dc):
    if dr == 0 and dc == 0:
        return "here"
    out = ""
    if dr < 0:
        out += "up"
    elif dr > 0:
        out += "down"
    if dc > 0:
        out += "right"
    elif dc < 0:
        out += "left"
    return out


def _circular_origin(values, step):
    vals = np.asarray(values, dtype=float)
    ang = 2.0 * np.pi * vals / step
    c = float(np.cos(ang).mean())
    s = float(np.sin(ang).mean())
    return float((np.arctan2(s, c) % (2.0 * np.pi)) * step / (2.0 * np.pi))


class Scene:
    """Measured model of one episode, updated once per frame."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.seen = False
        self.sx = 0.0
        self.sy = 0.0
        self.ox = 0.0
        self.oy = 0.0
        self.left = 0
        self.right = 0
        self.top = 0
        self.bottom = 0
        self.pac = None
        self.prev_pac = None
        self.ghosts = []
        self.ghost_prev = []
        self.pellets = set()
        self.power = set()
        self.pellet_count = 0
        self.power_count = 0
        self.eaten_total = 0
        self.eaten_delta = 0
        self.deaths = 0
        self.lives = None
        self.powered = False
        self.blue_px = 0
        self.marks = []
        self.notes = []
        self.trail = []
        self.visited = set()
        self.move = None
        self.moved = None
        self.heading_change = None
        self.frames = 0
        self.last_image_id = None
        self.frames_static = 0
        self._wall = None
        self._miss = {}
        self.gone = set()
        self._sig = None
        self.edge_cache = {}
        self._dist = None

    # --------------------------------------------------------------- storage
    def export(self):
        return {
            "seen": self.seen, "sx": self.sx, "sy": self.sy,
            "ox": self.ox, "oy": self.oy,
            "left": self.left, "right": self.right,
            "top": self.top, "bottom": self.bottom,
            "pac": self.pac, "prev_pac": self.prev_pac,
            "ghosts": self.ghosts, "ghost_prev": self.ghost_prev,
            "pellets": [list(p) for p in sorted(self.pellets)],
            "power": [list(p) for p in sorted(self.power)],
            "pellet_count": self.pellet_count,
            "power_count": self.power_count,
            "eaten_total": self.eaten_total, "eaten_delta": self.eaten_delta,
            "deaths": self.deaths, "lives": self.lives,
            "powered": self.powered, "blue_px": self.blue_px,
            "marks": list(self.marks), "notes": list(self.notes),
            "trail": list(self.trail),
            "visited": [list(p) for p in sorted(self.visited)],
            "move": list(self.move) if self.move else None,
            "moved": self.moved, "heading_change": self.heading_change,
            "frames": self.frames, "frames_static": self.frames_static,
            "last_image_id": self.last_image_id,
            "miss": {"%d,%d" % k: v for k, v in self._miss.items()},
            "gone": [list(p) for p in sorted(self.gone)],
        }

    def load(self, data):
        if not isinstance(data, dict) or not data:
            return
        for key in ("sx", "sy", "ox", "oy"):
            if isinstance(data.get(key), (int, float)):
                setattr(self, key, float(data[key]))
        for key in ("left", "right", "top", "bottom", "pellet_count",
                    "power_count", "eaten_total", "eaten_delta", "deaths",
                    "frames", "frames_static", "blue_px"):
            if isinstance(data.get(key), int):
                setattr(self, key, int(data[key]))
        for key in ("pac", "prev_pac"):
            v = data.get(key)
            setattr(self, key, dict(v) if isinstance(v, dict) else None)
        for key in ("ghosts", "ghost_prev", "marks", "notes"):
            v = data.get(key)
            if isinstance(v, list):
                setattr(self, key, [dict(x) for x in v if isinstance(x, dict)])
        for key, attr in (("pellets", "pellets"), ("power", "power")):
            v = data.get(key)
            if isinstance(v, list):
                setattr(self, attr, {tuple(int(i) for i in p) for p in v})
        if isinstance(data.get("trail"), list):
            self.trail = [tuple(p) for p in data["trail"] if isinstance(p, list)]
        if isinstance(data.get("visited"), list):
            self.visited = {tuple(p) for p in data["visited"] if isinstance(p, list)}
        if isinstance(data.get("move"), list):
            self.move = tuple(data["move"])
        if isinstance(data.get("gone"), list):
            self.gone = {tuple(int(i) for i in p) for p in data["gone"]
                         if isinstance(p, list) and len(p) == 2}
        if isinstance(data.get("miss"), dict):
            self._miss = {tuple(int(v) for v in k.strip("()").split(",")): int(n)
                          for k, n in data["miss"].items()}
        for key in ("moved", "powered"):
            if isinstance(data.get(key), bool):
                setattr(self, key, data[key])
        if isinstance(data.get("lives"), int):
            self.lives = data["lives"]
        if isinstance(data.get("last_image_id"), str):
            self.last_image_id = data["last_image_id"]
        if isinstance(data.get("heading_change"), str):
            self.heading_change = data["heading_change"]
        self.seen = bool(data.get("seen"))
        self._wall = None
        self.edge_cache = {}
        self._dist = None

    # ---------------------------------------------------------------- update
    def update(self, image, frame_id=None):
        """Measure one frame; repeated calls on the same frame are no-ops."""
        if image is None:
            return self
        try:
            a = np.asarray(image.convert("RGB"))
            sig = (frame_id, a.shape,
                   hash(a[::29, ::29].tobytes()), int(a.sum()))
            if sig == self._sig:
                return self
            self._sig = sig
            self._update(a, frame_id)
        except Exception:                    # a malformed frame must not kill a step
            self.frames += 1
        return self

    def _update(self, a, frame_id):
        if a.ndim != 3 or a.shape[2] < 3:
            return
        h, w = a.shape[:2]
        m = _masks(a)

        # ---- playfield bounds from the pale wall fill --------------------
        ys_f, xs_f = np.nonzero(m["fill"])
        if len(xs_f) >= 500:
            lo_x = int(np.percentile(xs_f, 0.1))
            hi_x = int(np.percentile(xs_f, 99.9))
            lo_y = int(np.percentile(ys_f, 0.1))
            hi_y = int(np.percentile(ys_f, 99.9))
            if not self.seen:
                self.left, self.right = lo_x, hi_x
                self.top, self.bottom = lo_y, hi_y
        if self.right <= self.left or self.bottom <= self.top:
            return

        # ---- pellet lattice ---------------------------------------------
        white = m["white"].copy()
        white[: max(0, self.top - int(0.04 * h))] = False
        white[self.bottom + 4:] = False
        white[:, : max(0, self.left - int(0.04 * w))] = False
        white[:, self.right + 4:] = False
        # Pellet size is learned from this frame: the most common small white
        # blob size is the dot, and only blobs near that size count.  Fragments
        # are anti-aliased sprite edges or ghost eyes; a partly covered pellet
        # is handled by the occlusion rule below, not counted as still visible.
        small = [d for d in _components(white, 3, 60, max_side=9)
                 if d["w"] <= 8 and d["h"] <= 8]
        dot_area = self._modal_area(small)
        dots = [d for d in small
                if dot_area * 0.9 <= d["n"] <= dot_area * 1.1] if dot_area else []
        # A pellet sits on dark floor.  A white blob of the same size that is
        # surrounded by saturated sprite colour is a ghost's eye, whatever the
        # ghost's current colour is (frightened blue included).
        if dots:
            # Ghosts are body-coloured and draw pale eyes; a pellet on floor has
            # no saturated pixel within two pixels of its blob.
            sat = m["sprite"] & ~m["white"]
            near_body = ndimage.binary_dilation(sat, iterations=2)
            dots = [d for d in dots
                    if not near_body[max(0, int(d["y"]) - 1):int(d["y"]) + 2,
                                     max(0, int(d["x"]) - 2):int(d["x"]) + 3].any()]
        if len(dots) >= 6:
            xs = np.array([d["x"] for d in dots])
            ys = np.array([d["y"] for d in dots])
            sx = self._spacing(xs)
            sy = self._spacing(ys)
            if sx and sy:
                ox = _circular_origin(xs, sx)
                oy = _circular_origin(ys, sy)
                if not self.seen:
                    self.sx, self.sy, self.ox, self.oy = sx, sy, ox, oy
                else:
                    if abs(sx - self.sx) <= 1.5:
                        self.sx = 0.75 * self.sx + 0.25 * sx
                    if abs(sy - self.sy) <= 1.5:
                        self.sy = 0.75 * self.sy + 0.25 * sy
                    self.ox = self._blend(ox, self.ox, self.sx)
                    self.oy = self._blend(oy, self.oy, self.sy)
        if not (self.sx and self.sy):
            self.frames += 1
            return

        # ---- wall mask (fill + long outline segments) -------------------
        outline = _drop_small_components(m["outline"],
                                         int(3.0 * self.sx),
                                         int(2.5 * self.sx * self.sy))
        wall = m["fill"] | outline
        self._wall = wall
        self.edge_cache = {}
        self._dist = None

        # ---- sprites: compact coloured blobs inside the playfield -------
        pac, ghosts = self._sprites(a, m, wall)

        # ---- pellets and power pellets ----------------------------------
        # Ghost eyes are small white blobs too, so any white blob inside a
        # sprite's own bounding box is the sprite, not a pellet.  The box is
        # what the sprite physically covers; a pellet under it is invisible
        # for that frame and must not be counted as eaten.
        boxes = []
        for sp in list(ghosts) + ([pac] if pac else []):
            hw = max(2.0, float(sp.get("w", self.sx)) / 2.0 + 1.0)
            hh = max(2.0, float(sp.get("h", self.sy)) / 2.0 + 1.0)
            boxes.append((sp["x"] - hw, sp["x"] + hw, sp["y"] - hh, sp["y"] + hh))

        def occluded(x, y):
            return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in boxes)

        seen_pellets = set()
        for d in dots:
            rc = self.cell_of(d["x"], d["y"])
            if self.in_field(*rc) and not occluded(d["x"], d["y"]):
                seen_pellets.add(rc)
        seen_power = set()
        for d in _components(white, 70, 900, max_side=26):
            if d["w"] < 9:
                continue
            rc = self.cell_of(d["x"], d["y"])
            if self.in_field(*rc) and rc not in seen_pellets and not occluded(d["x"], d["y"]):
                seen_power.add(rc)
        # A sprite hides any dot inside its own footprint (ghosts are wider
        # than one cell and often straddle two lattice cells).  Cells under
        # Pac-Man are handled by _track_trail, so only ghosts are expanded here.
        covered = set()
        for sp in list(ghosts) + ([pac] if pac else []):
            rc = self.cell_of(sp["x"], sp["y"])
            covered.add(rc)
            half_w = max(self.sx, float(sp.get("w", self.sx))) * 0.5 + 2.0
            half_h = max(self.sy, float(sp.get("h", self.sy))) * 0.5 + 2.0
            for r in range(rc[0] - 2, rc[0] + 3):
                for c in range(rc[1] - 2, rc[1] + 3):
                    x, y = self.xy_of(r, c)
                    if abs(x - sp["x"]) <= half_w and abs(y - sp["y"]) <= half_h:
                        covered.add((r, c))

        prev_count = self.pellet_count
        prev_power = self.power_count
        if not self.seen or not self.pellets:
            self.power = set(seen_power)
            self.pellets = set(seen_pellets) - self.power
            self._miss = {}
            self.gone = set()
        else:
            # A new level start shows many dots at once.  Pellets never
            # respawn within a level, so a large jump means the game reset.
            unseen_gone = [c for c in self.gone if c in seen_pellets]
            if len(unseen_gone) > 40:
                self.gone = set()
                self._miss = {}
            # new dots are accepted; a vanished dot is only accepted as eaten
            # after it stays missing while uncovered (sprites hide dots) and
            # it is never re-added afterwards: pellets do not respawn.
            self.pellets |= ((seen_pellets - covered) - self.gone)
            self.power |= ((seen_power - covered) - self.gone)
            # a dot inside a power pellet is the same object: never count both
            self.pellets -= self.power
            for cell in list(self.pellets):
                if cell in seen_pellets or cell in covered:
                    self._miss.pop(cell, None)
                    continue
                n = self._miss.get(cell, 0) + 1
                self._miss[cell] = n
                if n >= 3:
                    self.pellets.discard(cell)
                    self.gone.add(cell)
                    self._miss.pop(cell, None)
            for cell in list(self.power):
                if cell in seen_power or cell in covered:
                    self._miss.pop(cell, None)
                    continue
                n = self._miss.get(cell, 0) + 1
                self._miss[cell] = n
                if n >= 3:
                    self.power.discard(cell)
                    self.gone.add(cell)
                    self._miss.pop(cell, None)
        gained = ((prev_count - len(self.pellets)) +
                  (prev_power - len(self.power))) if self.seen else 0
        self.eaten_delta = gained if 0 < gained < 30 else 0
        if self.eaten_delta:
            self.eaten_total += self.eaten_delta

        # ---- entities ----------------------------------------------------
        if pac is not None:
            self.prev_pac = dict(self.pac) if self.pac else None
            self._set_pac(pac)
        self.ghosts = [self._ghost_record(g) for g in ghosts]

        f = a.astype(np.int16)
        blue = (f[..., 2] > 140) & (f[..., 0] < 150) & (f[..., 1] < 170)
        self.blue_px = int(blue.sum())
        self.powered = self.blue_px > 600
        self._count_lives(a, m)
        self._track_trail()
        self.pellet_count = len(self.pellets)
        self.power_count = len(self.power)
        if self.moved is False:
            self.frames_static += 1
        elif self.moved:
            self.frames_static = 0
        self.frames += 1
        self.seen = True
        self.last_image_id = frame_id

    # ------------------------------------------------------------ internals
    @staticmethod
    def _modal_area(small):
        """Most frequent component area among small white blobs (the pellet)."""
        if not small:
            return None
        areas = [d["n"] for d in small]
        counts = {}
        for a in areas:
            counts[a] = counts.get(a, 0) + 1
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        return best[0] if best[1] >= 8 else None

    @staticmethod
    def _spacing(vals):
        u = np.unique(np.round(vals).astype(int))
        d = np.diff(u)
        d = d[(d >= 8) & (d <= 60)]
        if len(d) < 3:
            return None
        step = float(np.median(d))
        return step if step >= 8 else None

    @staticmethod
    def _blend(new, cur, step):
        diff = (new - cur + step / 2.0) % step - step / 2.0
        if abs(diff) > step * 0.3:
            return float(cur)
        return float((cur + 0.3 * diff) % step)

    def _sprites(self, a, m, wall):
        lo_y = max(0, self.top - self.sy)
        hi_y = min(a.shape[0], self.bottom + 0.5 * self.sy)
        lo_x = max(0, self.left - self.sx)
        hi_x = min(a.shape[1], self.right + 0.5 * self.sx)

        cand = m["sprite"] & ~wall & ~m["white"]
        blobs = []
        for blob in _components(cand, 140, 2600, max_side=46, colour_of=a):
            if not (lo_x <= blob["x"] <= hi_x and lo_y <= blob["y"] <= hi_y):
                continue
            if blob["fill"] < 0.34:
                continue
            blob = dict(blob)
            blob["colour"] = colour_name(blob.get("rgb", (0, 0, 0)))
            if blob["colour"] in ("grey", "other"):
                continue
            blobs.append(blob)

        pacs = [b for b in blobs if b["colour"] == "yellow"]
        pac = None
        if pacs:
            if self.pac is not None and self.pac.get("x") is not None:
                pac = min(pacs, key=lambda b: (b["x"] - self.pac["x"]) ** 2 +
                                             (b["y"] - self.pac["y"]) ** 2)
            else:
                pac = max(pacs, key=lambda b: b["n"])
        ghosts = [b for b in blobs if b is not pac and b["colour"] != "yellow"]
        by_colour = {}
        for g in ghosts:
            by_colour.setdefault(g["colour"], []).append(g)
        ghosts = []
        for name, group in by_colour.items():
            if name == "blue":
                ghosts.extend(sorted(group, key=lambda z: -z["n"])[:4])
            else:
                ghosts.append(max(group, key=lambda z: z["n"]))
        return pac, ghosts

    def _ghost_record(self, c):
        rc = self.cell_of(c["x"], c["y"])
        return {"colour": c["colour"], "x": round(c["x"], 1),
                "y": round(c["y"], 1), "r": rc[0], "c": rc[1],
                "rgb": list(c.get("rgb", ()))}

    def _set_pac(self, c):
        rc = self.cell_of(c["x"], c["y"])
        heading = None
        if self.prev_pac and self.prev_pac.get("x") is not None:
            dx = c["x"] - self.prev_pac["x"]
            dy = c["y"] - self.prev_pac["y"]
            self.move = (round(dx, 1), round(dy, 1))
            self.moved = (abs(dx) + abs(dy)) >= 3.0
            if abs(dx) >= abs(dy) and abs(dx) > 2.0:
                heading = "right" if dx > 0 else "left"
            elif abs(dy) > 2.0:
                heading = "down" if dy > 0 else "up"
            if heading and self.pac and self.pac.get("heading") and heading != self.pac["heading"]:
                self.heading_change = self.pac["heading"] + "->" + heading
        self.pac = {"x": round(c["x"], 1), "y": round(c["y"], 1),
                    "r": rc[0], "c": rc[1],
                    "heading": heading or (self.pac or {}).get("heading")}

    def _count_lives(self, a, m):
        """Life icons sit below the maze and are Pac-Man shaped (yellow).

        The bonus fruit is drawn in the same strip but is red/other, so the
        colour test is what separates the two; it does not assume a count.
        """
        icons = _components(m["sprite"] & ~m["white"], 100, 900, max_side=40,
                            colour_of=a)
        icons = [c for c in icons if c["y"] > self.bottom + self.sy]
        icons = [c for c in icons if colour_name(c.get("rgb", (0, 0, 0))) == "yellow"]
        self.lives = len(icons)

    def _track_trail(self):
        if not self.pac or self.pac.get("r") is None:
            return
        cell = (self.pac["r"], self.pac["c"])
        last = self.trail[-1] if self.trail else None
        if last == cell:
            return
        if last is not None:
            dr, dc = cell[0] - last[0], cell[1] - last[1]
            if ((dr == 0) != (dc == 0)) and max(abs(dr), abs(dc)) <= 6:
                n = max(abs(dr), abs(dc))
                for k in range(1, n):
                    mid = (last[0] + int(round(dr * k / n)),
                           last[1] + int(round(dc * k / n)))
                    self.pellets.discard(mid)
                    self.power.discard(mid)
                    self.gone.add(mid)
                    self.trail.append(mid)
                    self.visited.add(mid)
        self.pellets.discard(cell)
        self.power.discard(cell)
        self.gone.add(cell)
        self.trail.append(cell)
        self.visited.add(cell)
        if len(self.trail) > 140:
            del self.trail[:-140]

    # ------------------------------------------------------------- geometry
    def cell_of(self, x, y):
        return (int(round((float(y) - self.oy) / self.sy)),
                int(round((float(x) - self.ox) / self.sx)))

    def xy_of(self, r, c):
        return (self.ox + c * self.sx, self.oy + r * self.sy)

    def in_field(self, r, c):
        if not (self.sx and self.sy):
            return False
        # stay strictly inside the measured maze box: half a cell inside the
        # pale-wall bounding box, which is exactly the corridor lattice
        x, y = self.xy_of(r, c)
        return (self.left + 0.5 * self.sx <= x <= self.right - 0.5 * self.sx and
                self.top + 0.5 * self.sy <= y <= self.bottom - 0.5 * self.sy)

    # -------------------------------------------------------------- probing
    def _on_wall_band(self, x, y, rad=2):
        """Any wall pixel within a small square around an image point?"""
        wall = self._wall
        if wall is None:
            return None
        h, w = wall.shape
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return True
        patch = wall[max(0, yi - rad):yi + rad + 1,
                     max(0, xi - rad):xi + rad + 1]
        return bool(patch.size and patch.any())

    def wall_at(self, r, c):
        """Does a wall touch this lattice point (a wall band or the border)?"""
        if self._wall is None:
            return None
        x, y = self.xy_of(r, c)
        return self._on_wall_band(x, y, rad=1)

    def edge_open(self, r, c, d):
        """Can Pac-Man move from (r,c) in direction d without crossing a wall?

        Wall bands are roughly one half-cell thick and sit between corridor
        cells, so a straight tube between two adjacent lattice points crosses
        one exactly when the move is blocked.  A small disc at the destination
        catches the outer border, which lies just beyond the outermost point.
        """
        key = (int(r), int(c), d)
        if key in self.edge_cache:
            return self.edge_cache[key]
        dr, dc = DELTA[d]
        nr, nc = int(r) + dr, int(c) + dc
        wall = self._wall
        if wall is None:
            return None
        if not self.in_field(nr, nc):
            self.edge_cache[key] = False
            return False
        x1, y1 = self.xy_of(r, c)
        x2, y2 = self.xy_of(nr, nc)
        h, w = wall.shape
        ok = True
        # A wall band crossed by the tube gives a run of consecutive hits.  A
        # single stray hit is not a wall: a sprite's red body pixels can enter
        # the outline mask for one sample (seen at 42_0 frame 59, a ghost on
        # Pac-Man's own cell closed an open corridor).  Require a run of two.
        run = 0
        for t in np.linspace(0.12, 0.88, 15):
            xi = int(round(x1 + (x2 - x1) * t))
            yi = int(round(y1 + (y2 - y1) * t))
            if not (0 <= xi < w and 0 <= yi < h):
                ok = False
                break
            run = run + 1 if wall[yi, xi] else 0
            if run >= 2:
                ok = False
                break
        if ok:
            # the border band may start just past the destination point
            ok = not self._on_wall_band(x2, y2, rad=1)
        self.edge_cache[key] = ok
        return ok

    def _near_field(self, r, c):
        """Is this lattice point within one cell of the measured maze box?"""
        if not (self.sx and self.sy):
            return False
        x, y = self.xy_of(r, c)
        return (self.left - self.sx <= x <= self.right + self.sx and
                self.top - self.sy <= y <= self.bottom + self.sy)

    def probe_direction(self, d, max_cells=4):
        if not self.pac or self.pac.get("r") is None:
            return None
        r, c = self.pac["r"], self.pac["c"]
        dr, dc = DELTA[d]
        for k in range(1, int(max_cells) + 1):
            if not self.edge_open(r + (k - 1) * dr, c + (k - 1) * dc, d):
                return k - 1
        return int(max_cells)

    def open_dirs(self):
        if not self.pac or self.pac.get("r") is None:
            return []
        return [d for d in DIRS
                if self.edge_open(self.pac["r"], self.pac["c"], d) is True]

    def next_open_ahead(self, heading, max_cells=8):
        if heading not in DELTA:
            return None
        return self.probe_direction(heading, max_cells)

    # --------------------------------------------------------------- search
    def dist_map(self, limit=1200):
        """Steps from Pac-Man to every reachable measured cell."""
        from collections import deque
        if self._dist is not None:
            return self._dist
        if not self.pac or self.pac.get("r") is None:
            return {}
        start = (self.pac["r"], self.pac["c"])
        dist = {start: 0}
        q = deque([start])
        while q and len(dist) < limit:
            cur = q.popleft()
            for d in DIRS:
                dr, dc = DELTA[d]
                nb = (cur[0] + dr, cur[1] + dc)
                if nb in dist or not self.in_field(*nb):
                    continue
                if self.edge_open(cur[0], cur[1], d) is not True:
                    continue
                dist[nb] = dist[cur] + 1
                q.append(nb)
        self._dist = dist
        return dist

    def path_to(self, cell, limit=1200):
        """BFS path from Pac-Man to a cell (or None)."""
        from collections import deque
        if not self.pac or self.pac.get("r") is None:
            return None
        goal = (int(cell[0]), int(cell[1]))
        start = (self.pac["r"], self.pac["c"])
        if goal == start:
            return []
        seen = {start}
        q = deque([(start, [])])
        expanded = 0
        while q and expanded < limit:
            cur, path = q.popleft()
            expanded += 1
            for d in DIRS:
                dr, dc = DELTA[d]
                nb = (cur[0] + dr, cur[1] + dc)
                if nb in seen or not self.in_field(*nb):
                    continue
                if self.edge_open(cur[0], cur[1], d) is not True:
                    continue
                if nb == goal:
                    return path + [nb]
                seen.add(nb)
                q.append((nb, path + [nb]))
        return None

    def nearest_target(self, kind):
        if not self.pac or self.pac.get("r") is None:
            return None
        if kind == "pellet":
            goals = self.pellets
        elif kind == "power":
            goals = self.power
        elif kind == "ghost":
            goals = {(g["r"], g["c"]) for g in self.ghosts if g.get("r") is not None}
        elif kind == "visited":
            goals = self.visited
        else:
            goals = set()
        if not goals:
            return None
        dist = self.dist_map()
        best = min((dist[g], g) for g in goals if g in dist) if any(g in dist for g in goals) else None
        if best is None:
            return None
        steps, cell = best
        start = (self.pac["r"], self.pac["c"])
        p = self.path_to(cell)
        if p is None:
            return None
        first = p[0] if p else start
        return {"direction": _turn(first[0] - start[0], first[1] - start[1]),
                "cells": len(p), "target": [cell[0], cell[1]]}

    # --------------------------------------------------------------- output
    def ascii_map(self, radius=8):
        if not self.pac or self.pac.get("r") is None or not self.sx:
            return "(no measurement yet)"
        pr, pc = self.pac["r"], self.pac["c"]
        marks = {}
        for mk in self.marks:
            try:
                marks[(int(mk["r"]), int(mk["c"]))] = str(mk.get("tag", "?"))[:1]
            except (KeyError, TypeError, ValueError):
                continue
        ghosts = {}
        for i, g in enumerate(self.ghosts):
            if g.get("r") is not None:
                ghosts.setdefault((g["r"], g["c"]), str((i + 1) % 10))
        dist = self.dist_map()
        lines = ["cell map around you (row, col); @ you, # wall, . pellet, "
                 "O power, 1-9 ghosts, letters your marks, blank open floor"]
        lines.append("     " + "".join(str(abs(c) % 10)
                                       for c in range(pc - radius, pc + radius + 1)))
        for r in range(pr - radius, pr + radius + 1):
            row = []
            for c in range(pc - radius, pc + radius + 1):
                if (r, c) == (pr, pc):
                    ch = "@"
                elif (r, c) in ghosts:
                    ch = ghosts[(r, c)]
                elif (r, c) in self.power:
                    ch = "O"
                elif (r, c) in self.pellets:
                    ch = "."
                elif (r, c) in marks:
                    ch = marks[(r, c)]
                elif (r, c) in dist:
                    ch = " "
                elif not self.in_field(r, c):
                    ch = "#" if self._near_field(r, c) else " "
                else:
                    ch = "?"
                row.append(ch)
            lines.append("%4d %s" % (r, "".join(row)))
        return "\n".join(lines)

    def scene_line(self):
        if not self.seen:
            return "MEASURED SCENE: not available yet"
        if not self.pac or self.pac.get("r") is None:
            return "MEASURED SCENE: player not visible in this frame"
        p = self.pac
        parts = ["you (row=%d,col=%d)" % (p["r"], p["c"])]
        if self.heading_change:
            parts.append("turned " + self.heading_change)
        if self.moved is False:
            parts.append("YOU DID NOT MOVE since the previous frame")
        elif self.move:
            parts.append("moved (dx=%+.0f,dy=%+.0f)px" % self.move)
        if p.get("heading"):
            ahead = self.next_open_ahead(p["heading"])
            parts.append("heading %s (coasting), wall %s cells ahead"
                         % (p["heading"], ahead))
        opened = self.open_dirs()
        parts.append("open: " + (",".join(opened) if opened else "none measured"))
        if self.ghosts:
            dist = self.dist_map()
            gs = []
            for g in self.ghosts:
                if g.get("r") is None:
                    continue
                d = dist.get((g["r"], g["c"]))
                gs.append("%s@(%d,%d)%s" % (g["colour"], g["r"], g["c"],
                                            "" if d is None else "/%d" % d))
            parts.append("ghosts " + ", ".join(gs))
        else:
            parts.append("no ghosts visible")
        parts.append("pellets left %d" % self.pellet_count)
        if self.eaten_delta:
            parts.append("last frame ate %d" % self.eaten_delta)
        if self.powered:
            parts.append("POWERED")
        if self.lives is not None:
            parts.append("lives %d" % self.lives)
        return "MEASURED SCENE: " + " | ".join(parts)

    def check(self, action):
        """Report the measured wall facts for one direction.

        This deliberately does not promise that the next frame will move: the
        game keeps Pac-Man coasting in its current heading and only accepts a
        turn once the requested direction is possible, so an open edge can
        still produce no displacement this frame.
        """
        d = ACTION_TO_DIR.get(action)
        if d is None:
            return "action '%s' is not a movement; there is no wall to check" % action
        if not self.pac or self.pac.get("r") is None:
            return "no measured position yet"
        opened = self.edge_open(self.pac["r"], self.pac["c"], d)
        if opened is None:
            return "wall map not measured yet"
        heading = self.pac.get("heading") or "unknown"
        if not opened:
            return ("%s: a wall lies between (%d,%d) and its %s neighbour. "
                    "Moving %s cannot start from here; if you are travelling %s "
                    "you will keep coasting until the next junction."
                    % (action, self.pac["r"], self.pac["c"], d, d, heading))
        cells = self.probe_direction(d, 8)
        return ("%s: no wall between (%d,%d) and its %s neighbour; the corridor "
                "stays open %s cell(s) %s. Heading is %s: the turn only takes "
                "effect where the game allows it, so movement this frame is "
                "not guaranteed until the next junction."
                % (action, self.pac["r"], self.pac["c"], d, max(1, cells), d,
                   heading))

    def mark(self, r, c, tag):
        tag = str(tag)[:24]
        self.marks = [m for m in self.marks
                      if not (m.get("r") == r and m.get("c") == c)]
        self.marks.append({"r": int(r), "c": int(c), "tag": tag})
        return "mark %r at (%d,%d); %d saved" % (tag, r, c, len(self.marks))

    def unmark(self, r, c):
        before = len(self.marks)
        self.marks = [m for m in self.marks
                      if not (m.get("r") == r and m.get("c") == c)]
        return "removed %d mark(s) at (%d,%d)" % (before - len(self.marks), r, c)

    def note(self, text):
        text = str(text)[:200]
        self.notes.append(text)
        if len(self.notes) > 40:
            del self.notes[:-40]
        return "noted: " + text

    def canvas(self, radius=9, cell_px=20):
        """Semantic map around the player, for the host visual channel."""
        grid = radius * 2 + 1
        prims = [{"type": "label", "at": [0, 0],
                  "text": "@=you .=pellet O=power g=ghost m=mark #=wall"}]
        base = 1
        if not (self.sx and self.pac and self.pac.get("r") is not None):
            return {"ops": [{"op": "canvas", "grid": [2, 14],
                             "cell_px": 24, "as": "m"},
                            {"op": "draw", "src": "m",
                             "prims": [{"type": "label", "at": [0, 0],
                                        "text": "no scene measurement"}],
                             "as": "out"}],
                    "output": "out"}
        pr, pc = self.pac["r"], self.pac["c"]
        ghosts = {(g["r"], g["c"]) for g in self.ghosts if g.get("r") is not None}
        marks = {(int(mk["r"]), int(mk["c"])) for mk in self.marks
                 if isinstance(mk.get("r"), int) and isinstance(mk.get("c"), int)}
        dist = self.dist_map()
        for i in range(grid):
            for j in range(grid):
                r, c = pr - radius + i, pc - radius + j
                if (r, c) == (pr, pc):
                    kind = "agent"
                elif (r, c) in ghosts:
                    kind = "danger"
                elif (r, c) in self.power:
                    kind = "goal"
                elif (r, c) in self.pellets:
                    kind = "object"
                elif (r, c) in marks:
                    kind = "key"
                elif (r, c) in dist:
                    kind = "floor"
                elif not self.in_field(r, c):
                    kind = "other"
                elif self.wall_at(r, c):
                    kind = "wall"
                else:
                    kind = "unknown"
                prims.append({"type": "cell", "at": [base + i, j], "color": kind})
        prims.append({"type": "label", "at": [base + grid, 0],
                      "text": "you row %d col %d" % (pr, pc)})
        return {"ops": [{"op": "canvas", "grid": [base + grid + 1, grid],
                         "cell_px": int(cell_px), "as": "m"},
                        {"op": "draw", "src": "m", "prims": prims, "as": "out"}],
                "output": "out"}
