"""Render recall probes on the RNGBench 3D Maze (MIT).

After navigation the agent locates previously visible posters from transformed
images. Unseen posters are lures. The task ID fixes the score denominator.
"""
from __future__ import annotations

import colorsys
import hashlib
import os
import math
import random
import re
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

RNG_ROOT = Path(os.environ.get("EVOVIM_RNGBENCH_ROOT", Path(__file__).resolve().parents[1] / "upstream" / "RNGBench")).expanduser().resolve()
UPSTREAM = RNG_ROOT / "2_3d_maze"
if not UPSTREAM.is_dir():
    raise FileNotFoundError(f"Missing {UPSTREAM}. Set EVOVIM_RNGBENCH_ROOT to the RNGBench checkout.")
sys.path.insert(0, str(UPSTREAM))

from game import FACE_DELTA_CELL, FOV_DEG, OPPOSITE, RENDER_H, RENDER_W, TURN_LEFT, MazeGame3D  # noqa: E402

TASK_RE = re.compile(r"^poster_maze/n(\d+)_k(\d+)_q(\d+)_g(\d+)p(\d+)(_aware)?$")
ANSWER_RE = re.compile(r"^\s*answer\s+(?:(\d+)\s*,\s*(\d+)\s*,\s*([NESW])|(not_seen))\s*$", re.I)
NAV_ACTIONS = ("forward", "backward", "turn_left", "turn_right")
MIN_COLS, MIN_FOG = 40, 0.4  # a sighting also needs: frontal, >= 1 cell away, the whole poster in view
N_LURES = 2  # fixed, so answering not_seen to everything earns 2 / Q however many posters were seen
PROBE_SIZE = 360


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def make_palette(seed: int, p: int) -> np.ndarray:
    """p colours of similar hue and lightness (hard to tell apart by name)."""
    r = _rng("palette", seed)
    h0 = r.random()
    cols = []
    for i in range(p):
        h = (h0 + (i - (p - 1) / 2) * 0.035) % 1.0
        l = 0.42 + 0.12 * (i % 2) + r.uniform(-0.03, 0.03)
        s = 0.55 + r.uniform(-0.08, 0.08)
        cols.append([int(255 * x) for x in colorsys.hls_to_rgb(h, l, s)])
    return np.array(cols, dtype=np.float32)


def make_poster(seed: int, idx: int, g: int, palette: np.ndarray) -> np.ndarray:
    """GxG arrangement of palette colours, 64x64 RGB float; balanced colour counts."""
    r = _rng("poster", seed, idx)
    p = len(palette)
    cells = [i % p for i in range(g * g)]
    r.shuffle(cells)
    grid = palette[np.array(cells).reshape(g, g)]
    return np.asarray(Image.fromarray(grid.astype(np.uint8)).resize((64, 64), Image.NEAREST), dtype=np.float32)


def probe_image(tex: np.ndarray, salt: str) -> Image.Image:
    r = _rng("probe", salt)
    img = Image.fromarray(tex.astype(np.uint8)).resize((240, 240), Image.NEAREST)
    img = img.rotate(r.uniform(-4, 4), resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
    crop = r.uniform(0.92, 1.0)
    w = int(240 * crop)
    x0, y0 = r.randint(0, 240 - w), r.randint(0, 240 - w)
    img = img.crop((x0, y0, x0 + w, y0 + w)).resize((r.randint(200, 280),) * 2, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) * r.uniform(0.7, 1.0)
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r.uniform(0.3, 0.7)))
    canvas = Image.new("RGB", (PROBE_SIZE, PROBE_SIZE), (90, 90, 90))
    canvas.paste(img, ((PROBE_SIZE - img.width) // 2, (PROBE_SIZE - img.height) // 2))
    return canvas


class PosterMaze(MazeGame3D):
    """MazeGame3D with texture posters on wall faces and a sighting log of what the agent saw."""

    def __init__(self, *a, **k):
        self.posters: dict[tuple, np.ndarray] = {}
        self.sightings: dict[tuple, tuple[int, float]] = {}  # face -> best (columns, fog) in one frame
        super().__init__(*a, **k)

    def wall_faces(self) -> list[tuple]:
        out = []
        for r in range(self.maze_size):
            for c in range(self.maze_size):
                cr, cc = 2 * r + 1, 2 * c + 1
                for d, (dr, dc) in FACE_DELTA_CELL.items():
                    wr, wc = cr + dr, cc + dc
                    if self.grid[wr][wc] == self.WALL:
                        out.append(((wr, wc, d), (r, c)))
        return out

    def _build_3d_scene(self):
        pixels = MazeGame3D._build_3d_scene(self)
        if not self.posters:
            return pixels
        W, H = RENDER_W, RENDER_H
        ang = math.radians(self.angle)
        dx, dy = math.cos(ang), math.sin(ang)
        t = math.tan(math.radians(FOV_DEG / 2))
        plx, ply = -dy * t, dx * t
        fog_dist = float(self.vision_range * 2)
        hits: dict[tuple, list[float]] = {}
        for x in range(W):
            cam = 2.0 * x / W - 1.0
            rdx, rdy = dx + plx * cam, dy + ply * cam
            mx, my = int(self.px), int(self.py)
            ddx = abs(1 / rdx) if abs(rdx) > 1e-12 else 1e30
            ddy = abs(1 / rdy) if abs(rdy) > 1e-12 else 1e30
            sx, sdx = (-1, (self.px - mx) * ddx) if rdx < 0 else (1, (mx + 1 - self.px) * ddx)
            sy, sdy = (-1, (self.py - my) * ddy) if rdy < 0 else (1, (my + 1 - self.py) * ddy)
            hit, side = False, 0
            while True:
                if sdx < sdy:
                    sdx += ddx; mx += sx; side = 0
                else:
                    sdy += ddy; my += sy; side = 1
                perp = (sdx - ddx) if side == 0 else (sdy - ddy)
                if perp > fog_dist + 0.5 or not (0 <= my < self.grid_h and 0 <= mx < self.grid_w):
                    break
                if self.grid[my][mx] == self.WALL:
                    hit = True
                    break
            if not hit:
                continue
            perp = max(0.08, (sdx - ddx) if side == 0 else (sdy - ddy))
            if side == 0:
                face_dir, wall_u = ("E" if sx > 0 else "W"), self.py + perp * rdy
            else:
                face_dir, wall_u = ("S" if sy > 0 else "N"), self.px + perp * rdx
            wall_u -= math.floor(wall_u)
            if (side == 0 and rdx < 0) or (side == 1 and rdy > 0):
                wall_u = 1 - wall_u
            tex = self.posters.get((my, mx, face_dir))
            if tex is None:
                continue
            fog_f = max(0.0, 1.0 - perp / fog_dist)
            line_h = int(H / perp)
            top = H // 2 - line_h // 2
            y0, y1 = max(0, top), min(H - 1, H // 2 + line_h // 2)
            m = 0.1
            if not (m <= wall_u <= 1 - m):
                continue
            u = (wall_u - m) / (1 - 2 * m)
            ys = np.arange(y0, y1 + 1)
            v = (ys - top) / max(line_h, 1)
            ok = (v >= m) & (v <= 1 - m)
            th, tw = tex.shape[:2]
            vv = ((v[ok] - m) / (1 - 2 * m) * th).astype(int).clip(0, th - 1)
            pixels[ys[ok], x] = (tex[vv, min(tw - 1, int(u * tw))] * fog_f).astype(np.uint8)
            hits.setdefault((my, mx, face_dir), []).append((fog_f, perp))
        for key, vals in hits.items():
            fog = float(np.median([f for f, _ in vals]))
            dist = float(np.median([d for _, d in vals]))
            expected = 0.8 * W / (2 * t * dist)          # poster width in columns if fully in view
            frontal = key[2] == self.facing               # face perpendicular to the view: an undistorted rectangle
            good = frontal and dist >= 1.5 and fog >= MIN_FOG and len(vals) >= 0.95 * expected
            best = self.sightings.get(key)
            if good and (best is None or len(vals) > best[0]):
                self.sightings[key] = (len(vals), fog)
        return pixels


def bfs_path(game: MazeGame3D) -> list[tuple[int, int]]:
    start, goal = game.agent_cell_pos(), game.goal_cell_pos()
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for d in FACE_DELTA_CELL:
            if game._can_move_cell(cur[0], cur[1], d):
                nr, nc = cur[0] + FACE_DELTA_CELL[d][0], cur[1] + FACE_DELTA_CELL[d][1]
                if (nr, nc) not in prev:
                    prev[(nr, nc)] = cur
                    q.append((nr, nc))
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    return path[::-1]


class PosterMazeBackend:
    protocol_version = 2

    def __init__(self, game: str, task_id: str, observation_profile: str, horizon: int, variant: str = ""):
        m = TASK_RE.match(task_id)
        if not m:
            raise ValueError(f"task id must look like poster_maze/n<N>_k<K>_q<Q>_g<G>p<P>[_aware], got {task_id!r}")
        self.n, self.k, self.q, self.g, self.p = (int(x) for x in m.groups()[:5])
        self.aware = bool(m.group(6))
        if self.n % 2 == 0 or self.n < 5 or self.k < 4 or self.q < 2 or self.p < 2:
            raise ValueError("need odd N >= 5, K >= 4, Q >= 2, P >= 2")
        self.game, self.task_id, self.observation_profile = game, task_id, observation_profile
        self.horizon = int(horizon)
        self._reset_state()

    def _reset_state(self) -> None:
        self.seed = None
        self.maze: PosterMaze | None = None
        self.phase = "navigate"
        self.nav_cap = 0
        self.step_count = 0
        self.invalid = 0
        self.reached_goal = False
        self.questions: list[dict] = []
        self.results: list[dict] = []
        self.qi = 0
        self.terminated = False
        self.truncated = False
        self.optimal = 0
        self._last_public: dict[str, Any] | None = None

    def _place_posters(self) -> None:
        """Prefer faces seen head-on along the shortest route (walls ahead at turns and corridor ends), so any
        agent that reaches the goal passes most posters; fill up with faces next to the route."""
        g = self.maze
        path = bfs_path(g)
        self.optimal = max(1, len(path) - 1)
        probe = PosterMaze(maze_size=self.n, seed=self.seed, vision_range=4, action_space="v4",
                           maze_type="v5", wall_style="plain")
        probe.posters = {key: np.zeros((8, 8, 3), np.float32) for key, _ in probe.wall_faces()}
        face = probe.facing
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            d = next(k for k, (dr, dc) in FACE_DELTA_CELL.items() if (r0 + dr, c0 + dc) == (r1, c1))
            while face != d:
                probe.move("turn_left" if TURN_LEFT[face] == d or OPPOSITE[face] == d else "turn_right")
                face = probe.facing
                probe._build_3d_scene()
            probe.move("forward")
            probe._build_3d_scene()
        frontal = sorted((k for k in probe.sightings),
                         key=lambda k: -probe.sightings[k][0])
        near = set(path)
        for r, c in path:
            for d in FACE_DELTA_CELL:
                if g._can_move_cell(r, c, d):
                    near.add((r + FACE_DELTA_CELL[d][0], c + FACE_DELTA_CELL[d][1]))
        cell_of = {key: cell for key, cell in g.wall_faces()}
        rest = [key for key, cell in g.wall_faces() if cell in near and key not in frontal]
        r = _rng("place", self.seed)
        r.shuffle(frontal)
        r.shuffle(rest)
        chosen = (frontal[: (2 * self.k) // 3] + rest)[: self.k]
        self.palette = make_palette(self.seed, self.p)
        self.face_cell = {}
        for i, key in enumerate(chosen):
            g.posters[key] = make_poster(self.seed, i, self.g, self.palette)
            cell = cell_of[key]
            self.face_cell[key] = (cell[0], cell[1], key[2], i)
        self.n_frontal = min(len(frontal), (2 * self.k) // 3)

    def _make_questions(self) -> None:
        g = self.maze
        well = [k for k, (cols, fog) in g.sightings.items() if cols >= MIN_COLS and fog >= MIN_FOG and k in self.face_cell]
        r = _rng("questions", self.seed, len(well))
        well.sort()
        r.shuffle(well)
        seen = well[: min(self.q - N_LURES, len(well))]
        qs = []
        for key in seen:
            row, col, d, idx = self.face_cell[key]
            qs.append({"kind": "seen", "tex": g.posters[key], "answer": (row, col, d)})
        for j in range(N_LURES):
            qs.append({"kind": "lure", "tex": make_poster(self.seed, 1000 + j, self.g, self.palette), "answer": None})
        r.shuffle(qs)
        self.questions = qs
        self.n_well_seen = len(well)

    def _status_text(self, status: str) -> str:
        g = self.maze
        if self.phase == "navigate":
            (r, c), (gr, gc) = g.agent_cell_pos(), g.goal_cell_pos()
            avail = g.get_available_actions()
            return (f"{status} You are at cell ({r},{c}) facing {g.facing}. The goal is cell ({gr},{gc}). "
                    f"Available actions: {', '.join(avail)}. Step {self.step_count}/{self.nav_cap}.")
        if self.phase == "questions":
            return (f"{status} Question {self.qi + 1}/{len(self.questions)}: the image shows a poster, re-photographed (it may be "
                    "rescaled, rotated, partly cropped, differently lit and blurred). If you saw this exact poster on a "
                    "wall of the maze, reply answer r,c,D where (r,c) is the cell directly in front of it and D is the "
                    "side of that cell whose wall carries it (N, E, S or W). If you never saw it, reply answer not_seen. "
                    "One answer per question.")
        return f"{status} The episode is over."

    def _public(self, status: str) -> dict[str, Any]:
        g = self.maze
        if self.phase == "navigate":
            image = g.render_frame(show_minimap=False, show_hud=True)
        elif self.phase == "questions":
            image = probe_image(self.questions[self.qi]["tex"], f"{self.seed}:{self.qi}")
        else:
            image = Image.new("RGB", (PROBE_SIZE, PROBE_SIZE), (90, 90, 90))
        long_term = (f"First-person 3D maze, {self.n}x{self.n} cells; rows and columns are numbered from 0. "
                     "Reach the goal cell.")
        self._last_public = {"text": {"long_term_context": long_term, "short_term_context": self._status_text(status)},
                             "frame_id": f"f{self.step_count:05d}", "image": image}
        return dict(self._last_public)

    def action_spec(self) -> dict[str, Any]:
        return {"kind": "free_text", "max_length": 32, "examples": list(NAV_ACTIONS) + ["answer 3,4,E", "answer not_seen"],
                "description": "While navigating reply one of forward, backward, turn_left, turn_right. When asked a "
                               "question, reply as the question instructs."}

    def metric_description(self) -> dict[str, Any]:
        return {"name": "Poster Maze recall accuracy: correct answers / questions", "aggregation": "final", "range": [0.0, 1.0]}

    def instruction(self) -> str:
        text = ("You are in a first-person 3D maze rendered with fog: walls further than a few cells fade out. "
                "Each step you see the view ahead and a text line with your cell (row, column), facing (N/E/S/W), "
                "the goal cell and the available actions. Actions: forward and backward move one cell, turn_left "
                "and turn_right rotate 90 degrees. Rows grow southward and columns grow eastward. Reach the goal "
                f"cell; you have {self.nav_cap if self.nav_cap else 'a limited number of'} steps.")
        if self.aware:
            text += (" Some walls carry pictures. After the maze you will be shown pictures and asked where you "
                     "saw them.")
        return text

    def describe(self) -> dict[str, Any]:
        return {"protocol_version": self.protocol_version, "game": self.game, "task_id": self.task_id,
                "actual_native_env_id": f"rngbench/2_3d_maze+posters/{self.task_id}",
                "observation_profile": self.observation_profile, "horizon": self.horizon,
                "action_spec": self.action_spec(), "metric": self.metric_description()}

    def reset(self, seed: int) -> dict[str, Any]:
        self._reset_state()
        self.seed = int(seed)
        self.maze = PosterMaze(maze_size=self.n, seed=self.seed, vision_range=4, action_space="v4",
                               maze_type="v5", wall_style="plain")
        self._place_posters()
        self.nav_cap = max(60, 4 * self.optimal)
        if self.horizon < self.nav_cap + self.q:
            raise ValueError(f"horizon {self.horizon} < navigation cap {self.nav_cap} + {self.q} questions")
        return self._public("The maze starts.")

    def observe(self) -> dict[str, Any]:
        if self._last_public is None:
            raise RuntimeError("backend has not been reset")
        return dict(self._last_public)

    def step(self, action: str) -> dict[str, Any]:
        if self.maze is None:
            raise RuntimeError("backend has not been reset")
        if self.terminated or self.truncated:
            raise RuntimeError("episode is over")
        self.step_count += 1
        text = str(action).strip()
        if self.phase == "navigate":
            act = text.lower()
            if act not in NAV_ACTIONS:
                self.invalid += 1
                status = f"'{text}' is not a movement action; nothing happened."
            else:
                ok, msg = self.maze.move(act)
                status = msg
            if self.maze.is_game_over():
                self.reached_goal = True
            if self.reached_goal or self.step_count >= self.nav_cap:
                self.phase = "questions"
                self._make_questions()
                status += (" You reached the goal." if self.reached_goal else " The navigation steps are used up.")
                status += " Now answer some questions about the maze."
        else:
            cur = self.questions[self.qi]
            m = ANSWER_RE.match(text)
            given = None
            if m is None:
                self.invalid += 1
            elif m.group(4):
                given = "not_seen"
            else:
                given = (int(m.group(1)), int(m.group(2)), m.group(3).upper())
            correct = (given == "not_seen") if cur["answer"] is None else (given == cur["answer"])
            self.results.append({"kind": cur["kind"], "correct": bool(correct), "answered": given is not None})
            status = "Answer recorded." if given is not None else f"'{text}' is not a valid answer; recorded as wrong."
            self.qi += 1
            if self.qi >= len(self.questions):
                self.phase = "done"
                self.terminated = True
        if not self.terminated and self.step_count >= self.horizon:
            self.truncated = True
        observation = self._public(status)
        return {"observation": observation, "reward": 0.0, "terminated": self.terminated,
                "truncated": self.truncated, "metrics": self.score(),
                "action_mapping": {"requested": text, "native": text, "fallback": False}}

    def primary_score(self) -> float:
        # Divide by Q from the task id, not by the questions actually asked: fewer posters seen well means
        # fewer seen questions, and dividing by those would raise the always-not_seen floor (2/4 with two seen).
        return sum(r["correct"] for r in self.results) / self.q

    def score(self) -> dict[str, Any]:
        def acc(kind):
            rs = [r["correct"] for r in self.results if r["kind"] == kind]
            return (sum(rs) / len(rs)) if rs else None
        done = self.terminated or self.truncated
        return {"primary": self.primary_score(), "success": (self.primary_score() == 1.0) if done else None,
                "seed": self.seed, "reached_goal": self.reached_goal, "optimal_steps": self.optimal,
                "nav_steps": min(self.step_count, self.nav_cap) if self.phase != "navigate" else self.step_count,
                "posters_placed": len(self.maze.posters) if self.maze else 0,
                "posters_well_seen": getattr(self, "n_well_seen", None),
                "questions_total": len(self.questions) if self.questions else self.q, "score_denominator": self.q, "questions_answered": len(self.results),
                "questions_correct": sum(r["correct"] for r in self.results),
                "accuracy_seen": acc("seen"), "accuracy_lure": acc("lure"),
                "invalid_actions": self.invalid, "steps": self.step_count,
                "terminated": self.terminated, "truncated": self.truncated}

    def close(self) -> None:
        self.maze = None
