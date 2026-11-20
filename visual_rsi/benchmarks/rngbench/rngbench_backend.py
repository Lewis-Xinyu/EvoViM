"""Run the RNGBench Matching Pairs board and card renderer (MIT).

One step flips one card. The second flip reveals both faces until the next step.
An invalid flip consumes a step. Score is the fraction of pairs cleared.
"""
from __future__ import annotations

import os
import re
import string
import sys
from pathlib import Path
from typing import Any

RNG_ROOT = Path(os.environ.get("EVOVIM_RNGBENCH_ROOT", Path(__file__).resolve().parents[1] / "upstream" / "RNGBench")).expanduser().resolve()
UPSTREAM = RNG_ROOT / "1_matching_pairs_new"
if not UPSTREAM.is_dir():
    raise FileNotFoundError(f"Missing {UPSTREAM}. Set EVOVIM_RNGBENCH_ROOT to the RNGBench checkout.")
sys.path.insert(0, str(UPSTREAM))

from env.board import coord_to_str  # noqa: E402
from env.matching_env import InvalidActionError, MatchingEnv  # noqa: E402

TASK_RE = re.compile(r"^matching_pairs/([a-z_]+)_(\d+)x(\d+)$")
CELL = 64  # upstream default --cell-size

RULES = """You are playing a Memory Match card game. Your goal is to clear all card pairs from the board in as few flips as possible.

Rules:
- Cards are placed face-down on a grid. Each round you flip two cards, one per action, and you see the board after each flip.
- If the two cards show the same picture they are removed and you score 1 point. Otherwise both are flipped back face-down before your next flip.
- Use your memory of previously revealed cards to find matches. A card's picture is shown only in the image; it is never described in text.

Board legend (image): card backs are face-down cards and are the only valid choices; blank green slots are already matched/removed.

Coordinate system: rows are lowercase letters (a, b, c, ...), columns are uppercase letters (A, B, C, ...); an action is a coordinate such as "cD" (row c, column D).

An invalid flip (an empty slot, or the same position as your first flip of the round) does not change the board but still uses one action."""


def _str_to_coord(s: str) -> tuple[int, int]:
    return string.ascii_lowercase.index(s[0]), string.ascii_uppercase.index(s[1])


class MatchingPairsBackend:
    protocol_version = 2

    def __init__(self, game: str, task_id: str, observation_profile: str, horizon: int, variant: str = ""):
        m = TASK_RE.match(task_id)
        if not m:
            raise ValueError(f"task id must look like matching_pairs/<theme>_<R>x<C>, got {task_id!r}")
        self.theme, self.rows, self.cols = m.group(1), int(m.group(2)), int(m.group(3))
        if not (UPSTREAM / "assets" / self.theme).is_dir():
            raise ValueError(f"unknown theme {self.theme!r}")
        if (self.rows * self.cols) % 2:
            raise ValueError("board needs an even number of cells")
        self.game, self.task_id, self.observation_profile = game, task_id, observation_profile
        self.horizon = int(horizon)
        self.actions = [coord_to_str((r, c)) for r in range(self.rows) for c in range(self.cols)]
        self.env: MatchingEnv | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.seed = None
        self.step_count = 0
        self.invalid_flips = 0
        self.rounds = 0
        self.terminated = False
        self.truncated = False
        self._last_public: dict[str, Any] | None = None

    def _public(self, image, status: str) -> dict[str, Any]:
        obs = self.env.get_observation()
        self._last_public = {
            "text": {"long_term_context": (f"Task: clear all {self.env.total_pairs} pairs on a "
                                           f"{self.rows}x{self.cols} board (rows a-{string.ascii_lowercase[self.rows - 1]}, "
                                           f"columns A-{string.ascii_uppercase[self.cols - 1]})."),
                     "short_term_context": f"{status} Score: {obs.score}. Remaining pairs: {obs.remaining_pairs}. "
                                           f"Flips used: {self.step_count}/{self.horizon}."},
            "frame_id": f"f{self.step_count:05d}",
            "image": image,
        }
        return dict(self._last_public)

    def action_spec(self) -> dict[str, Any]:
        return {"kind": "finite", "actions": list(self.actions), "dynamic": False}

    def metric_description(self) -> dict[str, Any]:
        return {"name": "Matching Pairs score: matched pairs / total pairs", "aggregation": "final",
                "range": [0.0, 1.0]}

    def instruction(self) -> str:
        return RULES + f"\n\nAction budget: {self.horizon} flips."

    def describe(self) -> dict[str, Any]:
        return {"protocol_version": self.protocol_version, "game": self.game, "task_id": self.task_id,
                "actual_native_env_id": f"rngbench/matching_pairs/{self.theme}/{self.rows}x{self.cols}",
                "observation_profile": self.observation_profile, "horizon": self.horizon,
                "action_spec": self.action_spec(), "metric": self.metric_description()}

    def reset(self, seed: int) -> dict[str, Any]:
        self._reset_state()
        self.seed = int(seed)
        self.env = MatchingEnv(rows=self.rows, cols=self.cols, seed=self.seed, theme=self.theme,
                               assets_dir=str(UPSTREAM / "assets"), cell_size=(CELL, CELL))
        self.env.reset()
        self.rounds = 1
        return self._public(self.env.render_board(mode="image"),
                            "Round 1 start. Flip the first card of this round.")

    def observe(self) -> dict[str, Any]:
        if self._last_public is None:
            raise RuntimeError("backend has not been reset")
        return dict(self._last_public)

    def step(self, action: str) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("backend has not been reset")
        if self.terminated or self.truncated:
            raise RuntimeError("episode is over")
        if action not in self.actions:
            raise ValueError(f"illegal action: {action}")
        coord = _str_to_coord(action)
        phase = self.env.phase
        first = self.env.first_flip_coord
        self.step_count += 1
        try:
            self.env.step(coord)
        except InvalidActionError as exc:
            self.invalid_flips += 1
            why = {"already_removed": "that slot is already empty",
                   "same_as_first": "it is the card you already flipped this round"}.get(exc.code, exc.reason)
            image = self.env.render_board(mode="image")
            status = f"Invalid flip {action}: {why}. The board did not change."
            status += " Flip the second card of this round." if phase == "flip_second" else " Flip the first card of this round."
        else:
            if phase == "flip_first":
                image = self.env.render_board(mode="image")
                status = f"You flipped {action}. Flip the second card of this round."
            else:
                result = self.env.last_result or {}
                image = self.env.render_both_flips(first, coord, mode="image")
                verdict = "Match! Both cards are removed." if result.get("matched") else \
                    "No match. Both cards will be flipped back face-down before your next flip."
                status = f"You flipped {coord_to_str(first)} and {action}. {verdict}"
                if not self.env.get_observation().done:
                    self.rounds += 1
                    status += f" Round {self.rounds} next: flip the first card."
        obs = self.env.get_observation()
        if obs.done:
            self.terminated = True
            status += " All pairs cleared."
        elif self.step_count >= self.horizon:
            self.truncated = True
        observation = self._public(image, status)
        return {"observation": observation, "reward": 0.0, "terminated": self.terminated,
                "truncated": self.truncated, "metrics": self.score(),
                "action_mapping": {"requested": action, "native": action, "fallback": False}}

    def primary_score(self) -> float:
        if self.env is None or self.env.board is None:
            return 0.0
        return self.env.get_observation().score / self.env.total_pairs

    def score(self) -> dict[str, Any]:
        obs = self.env.get_observation() if self.env is not None and self.env.board is not None else None
        return {"primary": self.primary_score(), "success": bool(obs and obs.done) if self.terminated or self.truncated else None,
                "seed": self.seed, "theme": self.theme, "board": f"{self.rows}x{self.cols}",
                "pairs_matched": obs.score if obs else 0, "total_pairs": self.env.total_pairs if obs else None,
                "terminated": self.terminated, "truncated": self.truncated, "steps": self.step_count,
                "rounds": self.rounds, "invalid_flips": self.invalid_flips}

    def close(self) -> None:
        self.env = None
