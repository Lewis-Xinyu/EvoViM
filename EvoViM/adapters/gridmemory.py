"""A seeded grid environment for running and evolving visual memory programs."""
from __future__ import annotations

import random
from typing import Any

from PIL import Image, ImageDraw

from .base import PublicObs, StepResult


class GridMemoryAdapter:
    name = "gridmemory"
    has_image = True
    public_text_fields = ("long_term_context", "short_term_context")
    fallback_action = "Noop"

    def __init__(self, size: int = 7, max_steps: int = 32):
        if size < 5 or size % 2 == 0:
            raise ValueError("size must be an odd integer >= 5")
        self.size = size
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = max_steps
        self._step = 0
        self._player = (size // 2, size // 2)
        self._target = (size // 2, size // 2 - 2)
        self._visited: set[tuple[int, int]] = set()

    def reset(self, seed: int) -> None:
        self._step = 0
        self._player = (self.size // 2, self.size // 2)
        choices = [(r, c) for r in range(1, self.size - 1)
                   for c in range(1, self.size - 1)
                   if (r, c) != self._player]
        self._target = random.Random(seed).choice(choices)
        self._visited = {self._player}

    def legal_actions(self) -> list[str]:
        return ["Move North", "Move South", "Move East", "Move West", "Do", "Noop", "Task Done"]

    def action_spec(self) -> dict[str, Any]:
        return {"kind": "finite", "actions": self.legal_actions(), "dynamic": False}

    def task_instruction(self) -> str:
        return "Explore the grid, remember where the blue relic was seen, and return to it."

    def metric_description(self) -> str:
        return "fraction of episodes in which the agent returns to the remembered relic"

    def _visible(self) -> list[str]:
        rows = []
        pr, pc = self._player
        for r in range(max(0, pr - 1), min(self.size, pr + 2)):
            cells = []
            for c in range(max(0, pc - 1), min(self.size, pc + 2)):
                if (r, c) == self._player:
                    cells.append("you")
                elif (r, c) == self._target:
                    cells.append("blue relic")
                else:
                    cells.append("empty")
            rows.append(" / ".join(cells))
        return rows

    def _image(self) -> Image.Image:
        cell = 32
        image = Image.new("RGB", (self.size * cell, self.size * cell), "white")
        draw = ImageDraw.Draw(image)
        for r in range(self.size):
            for c in range(self.size):
                fill = "#e9eef5" if (r, c) in self._visited else "#ffffff"
                if (r, c) == self._player:
                    fill = "#ef8354"
                elif (r, c) == self._target:
                    fill = "#356ae6"
                box = (c * cell, r * cell, (c + 1) * cell - 1, (r + 1) * cell - 1)
                draw.rectangle(box, fill=fill, outline="#8593a6")
        return image

    def observe(self) -> PublicObs:
        nearby = ", ".join(self._visible())
        target_visible = all(abs(a - b) <= 1 for a, b in zip(self._target, self._player))
        text = {
            "long_term_context": "A deterministic grid has walls at its boundary.",
            "short_term_context": (
                f"Step {self._step}; nearby cells: {nearby}. "
                + ("The blue relic is visible nearby." if target_visible else "No relic is visible nearby.")
            ),
        }
        return PublicObs(text=text, image=self._image(), frame_id=f"f{self._step:05d}")

    def step(self, action: str) -> StepResult:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action: {action}")
        old = self._player
        moves = {"Move North": (-1, 0), "Move South": (1, 0), "Move East": (0, 1), "Move West": (0, -1)}
        if action in moves:
            dr, dc = moves[action]
            candidate = (old[0] + dr, old[1] + dc)
            if 0 < candidate[0] < self.size - 1 and 0 < candidate[1] < self.size - 1:
                self._player = candidate
        self._step += 1
        self._visited.add(self._player)
        done = action == "Task Done" or self._step >= self.max_steps
        success = self._player == self._target
        private = {"target": self._target, "player": self._player, "success": success}
        reward = 1.0 if success else 0.0
        return StepResult(self.observe(), done, reward, private,
                          terminated=action == "Task Done", truncated=action != "Task Done" and self._step >= self.max_steps,
                          metrics={"success": success})

    def score(self, private_record: Any) -> float:
        return 1.0 if isinstance(private_record, dict) and private_record.get("success") else 0.0

    def close(self) -> None:
        """GridMemory holds no external resources."""
