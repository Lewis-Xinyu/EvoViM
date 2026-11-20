"""Run GameWorld semantic actions in a seeded, paused browser game.

Screenshots and task instructions are public. Native game state drives the evaluator.
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path(os.environ.get("EVOVIM_GAMEWORLD_ROOT", ROOT.parent / "upstream" / "GameWorld")).expanduser().resolve()
if not (UPSTREAM / "catalog").is_dir():
    raise FileNotFoundError(f"Missing GameWorld catalog in {UPSTREAM}. Set EVOVIM_GAMEWORLD_ROOT to the checkout.")
sys.path.insert(0, str(UPSTREAM))

from PIL import Image  # noqa: E402

from agents.harness.prompting import (  # noqa: E402
    build_semantic_controls_map,
    render_semantic_action_space,
    render_system_prompt,
)
from catalog.games import load_game  # noqa: E402
from catalog.tasks import load_task  # noqa: E402
from env import build_task_evaluator, reset_task_evaluator_episode_metrics  # noqa: E402
from runtime.env import GameEnv  # noqa: E402
from runtime.game_agent import Agent  # noqa: E402
from runtime.runtime_config import RuntimeConfig  # noqa: E402

PROTOCOL_VERSION = 2

# Upstream modules call LOGGER.model/.game/.task/.env, which upstream utils.py
# installs; utils.py itself imports the provider SDKs we do not need, so the
# same four level methods are installed here (identical names and levels).
_CUSTOM_LOG_LEVELS = {"model": 25, "game": 26, "task": 27, "env": 28}


def _install_custom_logger_methods() -> None:
    import logging

    for name, level in _CUSTOM_LOG_LEVELS.items():
        logging.addLevelName(level, name.upper())
        if not hasattr(logging.Logger, name):
            def _log(self, message, *args, _level=level, **kwargs):
                if self.isEnabledFor(_level):
                    self._log(_level, message, args, **kwargs)
            setattr(logging.Logger, name, _log)


_install_custom_logger_methods()


PINNED_SEED_LITERAL = "(42 >>> 0)"


class SeededGameServer:
    """Serve one GameWorld game directory over HTTP with the page's pinned seed replaced.

    Most GameWorld-Games pages carry an inline [DETERMINISTIC_RANDOM] block whose
    seed is the literal ``(42 >>> 0)``; it runs after (and overrides) the
    runtime's injected Math.random seeding, so upstream ``random_seed`` has no
    effect on those games.  Replacing that single literal in served HTML files
    is the only change: for seed 42 the bytes served are identical to upstream.
    """

    def __init__(self, game_dir: Path, seed: int):
        import http.server
        import threading

        game_dir = Path(game_dir).resolve()
        literal = f"({int(seed) & 0xFFFFFFFF} >>> 0)".encode()
        self.replacements: dict[str, int] = {}
        owner = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(game_dir), **kwargs)

            def log_message(self, *args):  # keep the RPC stderr log small
                return

            def send_head(self):
                path = self.translate_path(self.path)
                if os.path.isdir(path):
                    path = os.path.join(path, "index.html")
                if not path.endswith(".html") or not os.path.isfile(path):
                    return super().send_head()
                body = Path(path).read_bytes()
                count = body.count(PINNED_SEED_LITERAL.encode())
                owner.replacements[os.path.relpath(path, game_dir)] = count
                body = body.replace(PINNED_SEED_LITERAL.encode(), literal)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return io.BytesIO(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self.httpd.server_address[1])
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/index.html"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def parse_task_id(task_id: str) -> tuple[str, str]:
    """``05_breakout/05_01`` -> (game_id, task_id)."""
    if "/" not in task_id:
        raise ValueError("GameWorld task ids look like '<game_id>/<task_id>', e.g. 05_breakout/05_01")
    game_id, task = task_id.split("/", 1)
    return game_id, task


class GameWorldBackend:
    protocol_version = PROTOCOL_VERSION

    def __init__(self, game: str, task_id: str, observation_profile: str, horizon: int, variant: str = ""):
        self.game = game
        self.task_id = task_id
        self.observation_profile = observation_profile or "gameworld_screenshot_v1"
        self.horizon = int(horizon)
        # Optional variant "W<width>" downsizes the screenshot sent to the model
        # (the browser always renders at the native GameWorld resolution).
        if variant and not (variant.startswith("W") and variant[1:].isdigit()):
            raise ValueError("GameWorld variant must be empty or W<width>")
        # Default: the native GameWorld viewport width.  On a HiDPI Mac the
        # browser screenshot is captured at 2x, so it is resized back.
        self.obs_width = int(variant[1:]) if variant else None
        self.game_id, self.native_task_id = parse_task_id(task_id)
        self.game_spec = load_game(self.game_id)
        # "<game_id>/levels": the reset seed selects the level, so that development
        # and held-out units can be different levels without changing the host.
        # seed = 1000 * level + rng_seed; level L uses the upstream task file
        # "<prefix>_0L" (its ?level=L page and its own objective).
        self.level_mode = self.native_task_id == "levels"
        self.level: int | None = None
        self.task_spec = load_task(self.game_id, self._level_task_id(1) if self.level_mode else self.native_task_id)
        if self.game_spec.role_count != 1:
            raise ValueError(f"{self.game_id} has {self.game_spec.role_count} roles; only single-role games are supported")
        self.role = self.game_spec.game_roles[0]
        if self.obs_width is None:
            self.obs_width = int(self.game_spec.width)
        self.semantic_map = build_semantic_controls_map(self.role.semantic_controls)
        self.actions = [str(action.action_id) for action in self.role.semantic_controls]
        self.loop = asyncio.new_event_loop()
        self.server: SeededGameServer | None = None
        self.seed: int | None = None
        self.env: GameEnv | None = None
        self.agent: Agent | None = None
        self.evaluator = None
        self._reset_state()

    def _level_task_id(self, level: int) -> str:
        return f"{self.game_id.split('_', 1)[0]}_{int(level):02d}"

    # In multi-level tasks each standard evaluation seed selects a fixed level.
    LEVEL_OF_SEED = {42: 1, 46: 2, 43: 3, 44: 4, 45: 5}

    def _split_seed(self, seed: int) -> tuple[int, int]:
        if not self.level_mode:
            return int(seed), None
        if int(seed) in self.LEVEL_OF_SEED:
            return int(seed), self.LEVEL_OF_SEED[int(seed)]
        level, rng_seed = divmod(int(seed), 1000)
        if level < 1:
            raise ValueError("levels tasks need seed = 1000 * level + rng_seed with level >= 1")
        return rng_seed, level

    def _reset_state(self) -> None:
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.success: bool | None = None
        self.metrics: dict[str, Any] = {}
        self.fail_resets = 0
        self.startup_resets = 0
        self._last_public: dict[str, Any] | None = None
        self._last_status = "unknown"

    def _run(self, coro):
        return self.loop.run_until_complete(coro)

    def _config(self, seed: int, url: str) -> RuntimeConfig:
        return RuntimeConfig(
            url=url, speed_multiplier=self.game_spec.speed_multiplier,
            pause_during_inference=True, random_seed=int(seed),
            width=self.game_spec.width, height=self.game_spec.height,
            task_id=self.task_spec.task_id, game_id=self.task_spec.game_id,
            game_url_suffix=self.task_spec.game_url_suffix,
            evaluator_config=self.task_spec.evaluator_config, max_steps=self.horizon,
        )

    def _screenshot(self) -> dict[str, Any]:
        assert self.env is not None
        path = self._run(self.env.capture_screenshot("player"))
        image = Image.open(path).convert("RGB")
        if self.obs_width and image.width > self.obs_width:
            height = round(image.height * self.obs_width / image.width)
            image = image.resize((self.obs_width, height), Image.LANCZOS)
        try:
            Path(path).unlink()
        except OSError:
            pass
        self._last_public = {
            "text": {"long_term_context": "Task: " + self.task_spec.task_prompt.strip(),
                     "short_term_context": "Current observation is the game screenshot (paused)."},
            "frame_id": f"f{self.step_count:05d}",
            "image": image,
        }
        return dict(self._last_public)

    def action_spec(self) -> dict[str, Any]:
        return {"kind": "finite", "actions": list(self.actions), "dynamic": False}

    def metric_description(self) -> dict[str, Any]:
        return {"name": "GameWorld task progress (best within the step budget)", "aggregation": "max",
                "range": [0.0, 1.0]}

    def instruction(self) -> str:
        prompt = render_system_prompt(
            template_name="game_agent.j2", game_rules=self.game_spec.game_rules,
            task_prompt=self.task_spec.task_prompt, role_section=self.role.prompt.role_section,
            computer_use_controls_section=None,
            semantic_action_space=render_semantic_action_space(self.role.semantic_controls),
            output_format=None,
        ).strip()
        return (prompt + "\n\nThe game is paused while you decide. Each registered action runs for its own "
                "duration and the game pauses again. Only the screenshot is observable. "
                f"Action budget: {self.horizon}. If the game ends in failure before the task is complete, "
                "it restarts automatically and your best progress so far still counts.")

    def describe(self) -> dict[str, Any]:
        return {"protocol_version": self.protocol_version, "game": self.game, "task_id": self.task_id,
                "actual_native_env_id": f"{self.game_id}+{self.native_task_id}",
                "observation_profile": self.observation_profile, "horizon": self.horizon,
                "action_spec": self.action_spec(), "metric": self.metric_description()}

    def reset(self, seed: int) -> dict[str, Any]:
        self.close()
        self._reset_state()
        rng_seed, level = self._split_seed(seed)
        if level is not None:
            self.task_spec = load_task(self.game_id, self._level_task_id(level))
            suffix = str(self.task_spec.game_url_suffix or "")
            if f"level={level}" not in suffix:
                raise ValueError(f"{self.game_id}: task {self._level_task_id(level)} is not level {level} ({suffix!r})")
            self.level = level
        seed = rng_seed
        from env.game_launcher import GameLauncher
        if self.game_spec.url:
            raise ValueError(f"{self.game_id} uses a remote url; only local GameWorld-Games pages are supported")
        self.server = SeededGameServer(GameLauncher.resolve_game_directory(self.game_id), seed)
        self.seed = int(seed)
        self.env = GameEnv(self._config(seed, self.server.url), headless=True)
        try:
            self._run(self.env.start())
        except RuntimeError as exc:
            # Under heavy host load a real-time game (e.g. Flappy Bird) can reach its
            # terminal state before the startup readiness gate sees it; the upstream
            # gameAPI reset (the same one used for reset-on-fail) restarts it cleanly.
            if "readiness" not in str(exc) or self.env.game_manager is None:
                raise
            self.startup_resets += 1
            if not self._run(self.env.reset_game()):
                raise RuntimeError(f"{self.game_id}: startup readiness failed and the game reset did not recover") from exc
        self._run(self.env.pause_game())
        self.agent = Agent(agent_id="player", agent_type="general", client=None,
                           controls=self.role.controls, semantic_controls_map=self.semantic_map)
        self.evaluator = build_task_evaluator(evaluator_config=self.task_spec.evaluator_config,
                                              max_steps=self.horizon)
        return self._screenshot()

    def observe(self) -> dict[str, Any]:
        if self._last_public is None:
            raise RuntimeError("backend has not been reset")
        return dict(self._last_public)

    def step(self, action: str) -> dict[str, Any]:
        if self.env is None or self.agent is None:
            raise RuntimeError("backend has not been reset")
        if self.terminated or self.truncated:
            raise RuntimeError("episode is over")
        if action not in self.semantic_map:
            raise ValueError(f"illegal action: {action}")
        mapped = dict(self.semantic_map[action])
        mapped["semantic_controls"] = action
        executor = self.env._get_executor(self.agent)
        inspection = executor.inspect_action(mapped)
        if not inspection.get("is_valid"):
            raise RuntimeError(f"upstream rejected semantic action {action}: {inspection.get('reason')}")
        normalized = inspection["normalized_action"]
        self._run(self.env.resume_game())
        try:
            self._run(self.env.execute_action(self.agent, normalized, already_normalized=True))
        finally:
            self._run(self.env.pause_game())
        snapshot = self._run(self.env.capture_state())
        state = snapshot.state if snapshot else None
        self.step_count += 1
        self.agent.step_index = self.step_count
        result = self._run(self.evaluator(state=state, step_index=self.step_count, metrics=self.agent.eval_metrics))
        self.agent.eval_metrics = dict(result.metrics)
        self.metrics = dict(result.metrics)
        self._last_status = str(result.status)
        if result.should_reset:
            self.fail_resets += 1
            if not self._run(self.env.reset_game()):
                raise RuntimeError("GameWorld reset after a terminal failure did not succeed")
            self.agent.eval_metrics = reset_task_evaluator_episode_metrics(self.agent.eval_metrics)
        if result.status == "success":
            self.success = True
        if result.should_stop and not result.should_reset:
            if result.stop_reason == "max_steps_exhausted":
                self.truncated = True
            else:
                self.terminated = True
            if self.success is None:
                self.success = False
        observation = self._screenshot()
        return {"observation": observation, "reward": 0.0, "terminated": self.terminated,
                "truncated": self.truncated, "metrics": self.score(),
                "action_mapping": {"requested": action, "native": normalized, "fallback": False}}

    def primary_score(self) -> float:
        value = self.metrics.get("progress")
        return float(value) if isinstance(value, (int, float)) else 0.0

    def score(self) -> dict[str, Any]:
        return {"primary": self.primary_score(), "success": self.success, "seed": self.seed, "level": self.level,
                "native_task": self.task_spec.task_id,
                "seed_literal_replacements": dict(self.server.replacements) if self.server else {}, "terminated": self.terminated,
                "truncated": self.truncated, "steps": self.step_count, "fail_resets": self.fail_resets,
                "startup_resets": self.startup_resets,
                "task_status": self._last_status,
                "progress": self.metrics.get("progress"), "target_reached": self.metrics.get("target_reached")}

    def close(self) -> None:
        try:
            if self.env is not None:
                self._run(self.env.close_game())
        finally:
            self.env = None
            self.agent = None
            if self.server is not None:
                self.server.close()
                self.server = None
