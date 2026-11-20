"""Line M candidate: the parent memory agent with measured pixel state.

Structural changes relative to the parent (see method_description.md):

1. A per-episode ``Scene`` measurement layer (pacvision.Scene) turns the
   public screenshot into numbers every step: maze lattice, cell of Pac-Man,
   heading and displacement, ghosts, remaining pellets, power state, lives,
   and a wall graph fitted from the pale-wall pixels of that episode.
2. The measured scene line is appended to the observation text, so the model
   is no longer blind to a constant text observation.  Only a bounded derived
   summary is shown; the full map, wall checks and images are obtained through
   explicit tool calls the model chooses.
3. Six read-only measurement tools and one mark tool are registered next to
   the legacy file tools and the host's visual tools.
4. (g2 revision) A ROUTE TABLE is appended to every observation (pactable.py):
   per first step, the open run, the pellets reached through it, and the ghosts
   on its shortest path, with last-seen ages for ghosts that left the frame.
   Each LOG also sees a ROUTE CHECK of the previous step's press against the
   measured result. Both are measurements; no tool, memory write or action
   choice is made in code.

5. (g3) A pinned working note: the PIN command (LOG only) stores one short note
   the LLM wrote, as the memory file pinned_note.txt.  Each observation shows an
   index line only (exists, step written, age, lives changed?).  The note text is
   never put in the prompt; the LLM decides whether to READ it.

The LLM still decides every write to memory: no tool writes a memory file on
its own, and no tool deletes one.  Memory files persist across steps and are
read through the same READ/SEARCH/TAIL/COUNT commands as the parent.
"""
from __future__ import annotations

import json

from agents.memory_agent import MemoryAgent
from pacvision import Scene
import pactable
from pin_tools import install_pin, pin_index
from pixeltools import install_pixel_tools
from visual_tools import install_visual

# Mirrors the worker's checkpoint exclusion list: these hold live objects.
_LIVE = {"client", "config", "prompt_builder", "memory", "visual",
         "current_image", "current_frame_id", "visual_request",
         "visual_observation_available", "_memory_dir", "_memory_base"}


class Agent(MemoryAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder, config)
        self.scene = Scene()
        self._reset_route()

    def _reset_route(self):
        # Route state is plain JSON, so export_state carries it across checkpoints.
        self._route_prev = None        # table measured on the previous frame
        self._route_sight = []         # last measured ghost cells, with their step
        self._route_action = None      # action pressed on the previous step
        self._pin_meta = None          # {step, lives} of the pinned note, or None

    def reset(self):
        super().reset()
        self._reset_route()

    def act(self, obs, prev_action=None, prev_validated_action=None):
        self._route_action = prev_validated_action or prev_action
        return super().act(obs, prev_action=prev_action,
                           prev_validated_action=prev_validated_action)

    # ------------------------------------------------------------ pixel state
    def _sync_scene(self):
        scene = getattr(self, "scene", None)
        if scene is None:
            scene = self.scene = Scene()
        scene.update(getattr(self, "current_image", None),
                     getattr(self, "current_frame_id", None))

    def _format_obs(self, obs):
        base = super()._format_obs(obs)
        try:
            self._sync_scene()
            line = self.scene.scene_line()
        except Exception as exc:               # never lose a step to perception
            line = "MEASURED SCENE: unavailable (%s)" % type(exc).__name__
        try:
            block = self._route_block()
        except Exception as exc:               # the table is optional; the step is not lost
            self._route_prev = None
            block = "ROUTE TABLE: unavailable (%s)" % type(exc).__name__
        pin = pin_index(self)
        return base + "\n\n" + pin + "\n\n" + line + ("\n\n" + block if block else "")

    def _route_block(self):
        """Per-direction facts for this frame, and the check of the previous step.

        Measurement only: nothing here writes memory, calls a tool or picks an action.
        """
        step = self.step_count
        scene = self.scene
        prev = self._route_prev
        self._route_sight = pactable.update_sightings(self._route_sight, scene, step)
        table = pactable.route_table(scene, self._route_sight, step)
        parts = []
        if prev is not None and self._route_action is not None and step > 0:
            check = pactable.route_check(prev, self._route_action, scene)
            if check:
                parts.append(check)
        if table is None:
            self._route_prev = None
            if not parts:
                return ""
            parts.append("ROUTE TABLE: Pac-Man not measured in this frame")
            return "\n".join(parts)
        table["lives"] = scene.lives
        self._route_prev = table
        parts.append(pactable.render(table))
        return "\n".join(parts)

    # ----------------------------------------------------------- checkpointing
    def export_state(self):
        value = {}
        for key, item in self.__dict__.items():
            if key in _LIVE or callable(item):
                continue
            try:
                json.dumps(item, allow_nan=False)
            except (TypeError, ValueError):
                continue
            value[key] = item
        value["scene"] = self.scene.export()
        return value

    def import_state(self, data):
        data = dict(data or {})
        scene = data.pop("scene", None)
        for key, item in data.items():
            if key not in _LIVE:
                setattr(self, key, item)
        self.scene = Scene()
        if scene:
            self.scene.load(scene)


def install(agent, registry):
    """Install the preserved visual tools, then the pixel-measurement tools."""
    install_visual(agent, registry)
    install_pixel_tools(agent, registry)
    return install_pin(agent, registry)
