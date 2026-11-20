"""Test incidental visual recall inside the GameWorld Minecraft Clone.

A seeded corridor contains a painting on each side of each doorway. After
walking, the agent identifies transformed views by doorway and side or reports
an unseen painting. The task ID fixes the score denominator.
"""
from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gameworld"))
sys.path.insert(0, str(HERE))

from PIL import Image  # noqa: E402

import paint  # noqa: E402
from gameworld_backend import GameWorldBackend  # noqa: E402

TASK_RE = re.compile(r"^mc_gallery/g(\d+)_q(\d+)_c(\d+)p(\d+)(_aware)?$")
ANSWER_RE = re.compile(r"^\s*answer\s+(?:(\d+)\s*,\s*([LR])|(not_seen))\s*$", re.I)
BASE_TASK = "18_minecraft-clone-glm/18_01"
GAP, FIRST_GAP = 8, 6
N_LURES = 2
GALLERY_JS = (HERE / "gallery.js").read_text()
SIGHTING_JS = (HERE / "sighting.js").read_text()


class MCGalleryBackend(GameWorldBackend):
    def __init__(self, game: str, task_id: str, observation_profile: str, horizon: int, variant: str = ""):
        m = TASK_RE.match(task_id)
        if not m:
            raise ValueError(f"task id must look like mc_gallery/g<gates>_q<Q>_c<G>p<P>[_aware], got {task_id!r}")
        self.n_gates, self.q, self.g, self.p = (int(x) for x in m.groups()[:4])
        self.aware = bool(m.group(5))
        if not (2 <= self.n_gates <= 12 and self.q > N_LURES and self.p >= 2):
            raise ValueError("need 2 <= gates <= 12, Q > 2, P >= 2")
        super().__init__(game, BASE_TASK, observation_profile, horizon, variant=variant)
        self.gallery_task_id = task_id
        self.walk_cap = 3 * self.n_gates + 16
        if self.horizon < self.walk_cap + self.q:
            raise ValueError(f"horizon {self.horizon} < walking cap {self.walk_cap} + {self.q} questions")

    def _reset_state(self) -> None:
        super()._reset_state()
        self.phase = "walk"
        self.paintings: list[dict] = []
        self.sight: dict[str, Any] = {}
        self.questions: list[dict] = []
        self.results: list[dict] = []
        self.qi = 0
        self.invalid = 0
        self.reached_end = False
        self.n_well_seen = None

    def _page(self):
        return self.env.game_manager.page

    def _sighting(self) -> dict[str, Any]:
        self.sight = self._run(self._page().evaluate(SIGHTING_JS)) or {}
        return self.sight

    def _text(self, status: str) -> str:
        if self.phase == "walk":
            return (f"{status} Walk through the numbered doorways to the gold wall at the end of the corridor. "
                    f"Step {self.step_count}/{self.walk_cap}.")
        if self.phase == "questions":
            return (f"{status} Question {self.qi + 1}/{len(self.questions)}: the image shows a painting, re-photographed "
                    "(it may be rescaled, slightly rotated or cropped, differently lit and blurred). If you saw this exact "
                    "painting in the corridor, reply answer k,S where k is the number on the doorway of the wall it hangs "
                    "on and S is L if it hangs left of that doorway or R if right (as seen while walking toward it). If you "
                    "never saw it, reply answer not_seen. One answer per question.")
        return f"{status} The episode is over."

    def _publish(self, status: str, image: Image.Image | None = None) -> dict[str, Any]:
        if image is None:
            obs = self._screenshot()
            obs["text"] = {"long_term_context": "Minecraft-style first-person world: a walled corridor with numbered doorways.",
                           "short_term_context": self._text(status)}
            self._last_public = obs
            return dict(obs)
        self._last_public = {"text": {"long_term_context": "Minecraft-style first-person world: a walled corridor with numbered doorways.",
                                      "short_term_context": self._text(status)},
                             "frame_id": f"f{self.step_count:05d}", "image": image}
        return dict(self._last_public)

    def action_spec(self) -> dict[str, Any]:
        return {"kind": "free_text", "max_length": 32, "examples": list(self.actions) + ["answer 3,L", "answer not_seen"],
                "description": "While walking reply one registered game action (" + ", ".join(self.actions) +
                               "). When asked a question, reply as the question instructs."}

    def metric_description(self) -> dict[str, Any]:
        return {"name": "Minecraft Gallery recall accuracy: correct answers / questions", "aggregation": "final",
                "range": [0.0, 1.0]}

    def instruction(self) -> str:
        from agents.harness.prompting import render_semantic_action_space, render_system_prompt
        task = ("This episode uses a custom level: you start in a straight walled corridor. Walk forward through the "
                "numbered doorways until you reach the gold wall at the end of the corridor. You do not need to mine, "
                "place or collect anything.")
        if self.aware:
            task += (" Some walls carry paintings. After the corridor you will be shown paintings and asked where you "
                     "saw them.")
        prompt = render_system_prompt(
            template_name="game_agent.j2", game_rules=self.game_spec.game_rules, task_prompt=task,
            role_section=self.role.prompt.role_section, computer_use_controls_section=None,
            semantic_action_space=render_semantic_action_space(self.role.semantic_controls), output_format=None,
        ).strip()
        return (prompt + "\n\nThe game is paused while you decide. Each registered action runs for its own duration and "
                f"the game pauses again. Walking budget: {self.walk_cap} actions. After the walk, questions may follow; "
                "answer them as they instruct.")

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"task_id": self.gallery_task_id, "actual_native_env_id": f"18_minecraft-clone-glm+mc_gallery/{self.gallery_task_id}",
                  "action_spec": self.action_spec(), "metric": self.metric_description()})
        return d

    def reset(self, seed: int) -> dict[str, Any]:
        super().reset(seed)
        pal = paint.palette(self.seed, self.p)
        self.palette = pal
        self.paintings = []
        payload = []
        for k in range(self.n_gates):
            for side in "LR":
                idx = len(self.paintings)
                img = paint.painting(self.seed, idx, self.g, pal)
                self.paintings.append({"gate": k, "side": side, "img": img})
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                payload.append({"gate": k, "side": side, "png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})
        self.geometry = self._run(self._page().evaluate(GALLERY_JS, {"paintings": payload, "nGates": self.n_gates,
                                                                        "gap": GAP, "firstGap": FIRST_GAP}))
        # one short real-time tick so the rebuilt chunks and the texture images are drawn
        self._run(self.env.resume_game())
        self._run(self._page().wait_for_timeout(400))
        self._run(self.env.pause_game())
        self._sighting()
        return self._publish("The level starts.")

    def _make_questions(self) -> None:
        seen = sorted(int(i) for i in (self.sight.get("seen") or []))
        self.n_well_seen = len(seen)
        r = paint.rng("questions", self.seed, len(seen))
        r.shuffle(seen)
        qs = [{"kind": "seen", "img": self.paintings[i]["img"],
               "answer": (self.paintings[i]["gate"] + 1, self.paintings[i]["side"])} for i in seen[: self.q - N_LURES]]
        for j in range(N_LURES):
            qs.append({"kind": "lure", "img": paint.painting(self.seed, 1000 + j, self.g, self.palette), "answer": None})
        r.shuffle(qs)
        self.questions = qs

    def step(self, action: str) -> dict[str, Any]:
        if self.env is None or self.agent is None:
            raise RuntimeError("backend has not been reset")
        if self.terminated or self.truncated:
            raise RuntimeError("episode is over")
        self.step_count += 1
        text = str(action).strip()
        image = None
        if self.phase == "walk":
            if text in self.semantic_map:
                mapped = dict(self.semantic_map[text])
                mapped["semantic_controls"] = text
                executor = self.env._get_executor(self.agent)
                inspection = executor.inspect_action(mapped)
                if not inspection.get("is_valid"):
                    raise RuntimeError(f"upstream rejected semantic action {text}: {inspection.get('reason')}")
                self._run(self.env.resume_game())
                try:
                    self._run(self.env.execute_action(self.agent, inspection["normalized_action"], already_normalized=True))
                finally:
                    self._run(self.env.pause_game())
                status = f"Did {text}."
            else:
                self.invalid += 1
                status = f"'{text}' is not a registered game action; nothing happened."
            sight = self._sighting()
            if sight.get("atEnd"):
                self.reached_end = True
            if self.reached_end or self.step_count >= self.walk_cap:
                self.phase = "questions"
                self._make_questions()
                status += (" You reached the end of the corridor." if self.reached_end else " The walking budget is used up.")
                status += " Now answer some questions about the corridor."
                image = paint.probe(self.questions[0]["img"], f"{self.seed}:0")
        else:
            cur = self.questions[self.qi]
            m = ANSWER_RE.match(text)
            given = None
            if m is None:
                self.invalid += 1
            elif m.group(3):
                given = "not_seen"
            else:
                given = (int(m.group(1)), m.group(2).upper())
            correct = (given == "not_seen") if cur["answer"] is None else (given == cur["answer"])
            self.results.append({"kind": cur["kind"], "correct": bool(correct), "answered": given is not None})
            status = "Answer recorded." if given is not None else f"'{text}' is not a valid answer; recorded as wrong."
            self.qi += 1
            if self.qi >= len(self.questions):
                self.phase = "done"
                self.terminated = True
                image = Image.new("RGB", (paint.PROBE_SIZE, paint.PROBE_SIZE), (90, 90, 90))
            else:
                image = paint.probe(self.questions[self.qi]["img"], f"{self.seed}:{self.qi}")
        if not self.terminated and self.step_count >= self.horizon:
            self.truncated = True
        observation = self._publish(status, image)
        return {"observation": observation, "reward": 0.0, "terminated": self.terminated, "truncated": self.truncated,
                "metrics": self.score(), "action_mapping": {"requested": text, "native": text, "fallback": False}}

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
                "seed": self.seed, "reached_end": self.reached_end, "gates_passed": self.sight.get("passed"),
                "walk_steps": self.step_count if self.phase == "walk" else min(self.step_count, self.walk_cap),
                "paintings_placed": len(self.paintings), "paintings_well_seen": self.n_well_seen,
                "questions_total": len(self.questions) if self.questions else self.q, "score_denominator": self.q,
                "questions_answered": len(self.results), "questions_correct": sum(r["correct"] for r in self.results),
                "accuracy_seen": acc("seen"), "accuracy_lure": acc("lure"), "invalid_actions": self.invalid,
                "steps": self.step_count, "terminated": self.terminated, "truncated": self.truncated}
