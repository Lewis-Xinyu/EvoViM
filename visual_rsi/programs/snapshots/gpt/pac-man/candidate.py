"""M2: grounded transition inspection and a usable LOG-to-PLAN place handoff."""
import math
from pathlib import Path

import numpy as np
from PIL import Image

from agents.memory_agent import MemoryAgent
from place_memory import PlaceMemory
from registry import Tool
from visual_tools import install_visual


GUIDANCE = """MEMORY WORKFLOW:
Keep a short route target and a next visible checkpoint with ROUTE_MEMORY.
Choose these from this episode's observations, and retain the target across
routine actions. Change it when reached, contradicted, unsafe, or unproductive.
Use PLACE_UPDATE to keep a stable name for a junction/corridor, verified exits,
and remaining targets. Name a place by visible landmarks and distinguish being
between junctions from being aligned at a turn. Do not invent connectivity.
PLACE_UPDATE replaces fields in place; [] clears an obsolete list. Omitted
fields retain their dated evidence. Use PLACE_RECALL for relevant exits and
previous outcomes; correct mistaken claims rather than append contradictory
copies. A blocked turn is an observed attempt, not automatically a wall.
In LOG, outcome describes ONLY the previous action, using this new screenshot.
Code supplies its action and step. 'progress=none' requires observed evidence;
no score gain does not mean no movement. Use unknown when uncertain. Record
important blocked turns, repeated cleared transit, and corrections. Supply
from_place to link places only when the preceding location was established.
In PLAN,
store observations or intentions only; do not report an unobserved outcome.
Save an informative layout/junction with SAVE_FRAME when it may help later.
COMPARE_FRAME compares an explicitly saved image with the current image. A
box [left,top,right,bottom] uses fractions of the image; choose a small region
for the player/corridor if needed. It returns exact change measurements and a
historical/current pair in the next request. Pixel change is not movement or
progress; animations and moving threats also change pixels. Inspect both.
Historical threats are stale. Verify threats and alignment in current pixels.
Use ROUTE_MEMORY new_scene only if this observation shows a changed layout or
restart needing re-localization; it clears place claims and the route. A lost
life may preserve cleared targets; inspect rather than assume replenishment.
Issue exactly one tool command per reply, await its result, then continue.
These tools are optional; the model selects every memory operation and action.
"""

HANDOFF_GUIDANCE = """
OBSERVATION-GROUNDED HANDOFF:
PLACE_UPDATE with an outcome also retrieves the actual preceding observation
by default (inspect=true). The NEXT request shows that historical observation
alongside the host's current original image. Locate the controlled character
in BOTH images, not a life/status icon or another moving object. Compare actual
alignment, movement, and remaining targets before saying DONE. If the initial
claim was wrong, call PLACE_UPDATE again to replace the same action's outcome
and facts. inspect=false avoids repeating inspection after checking the pair.
Optional focus=[left,top,right,bottom] in image fractions gives a magnified
same-region pair and exact local change numbers. Choose enough space to contain
both positions; pixel change also includes animation/threats and is NOT progress.
No score gain can still be useful movement through a cleared passage.
Use a stable place name when revisiting. supersedes="old mistaken name" removes
that explicitly named incorrect place record; it does not invent connections.
Clear remaining=[] when its listed targets are visibly exhausted. Remaining
targets refer to concrete observed objects/areas, rather than repeating the
whole mission. The handoff shows dated current facts and earlier target claims;
they may be stale. Check those before choosing another tour of a cleared loop.
Repeated-attempt counts are reports of your stored claims, never wall or safety
classifications. Historical threats are stale; act using the current image.
"""


class Agent(MemoryAgent):
    def configure_memory(self, env_name, task, episode_idx=0):
        super().configure_memory(env_name, task, episode_idx)
        PlaceMemory(self._memory_dir)
        self._memory_phase = 'PLAN'
        self._observed_previous_action = None
        self._last_observation = None
        self._transition_source = None
        self._inspected_action_step = None

    def reset(self):
        super().reset()
        if self.memory is not None:
            PlaceMemory(self._memory_dir)
        self._memory_phase = 'PLAN'
        self._observed_previous_action = None
        self._last_observation = None
        self._transition_source = None
        self._inspected_action_step = None

    def act(self, obs, prev_action=None, prev_validated_action=None):
        self._observed_previous_action = prev_validated_action or prev_action
        self._transition_source = self._last_observation
        self._inspected_action_step = None
        observed_step = self.step_count
        result = super().act(obs, prev_action, prev_validated_action)
        # One replaceable predecessor raster for optional computation, with no
        # semantic claim or automatic outcome. The model chooses retrieval.
        # A subdirectory stays outside the legacy text file index.
        if self.current_image is not None and self.current_frame_id:
            cache = Path(self._memory_dir) / '_transition_cache'
            cache.mkdir(exist_ok=True)
            self.current_image.save(cache / 'previous.png')
            self._last_observation = {'frame_id': self.current_frame_id,
                                      'observed_step': observed_step}
        else:
            self._last_observation = None
        return result

    def _format_obs(self, obs):
        return super()._format_obs(obs)

    def _memory_context(self):
        return GUIDANCE + HANDOFF_GUIDANCE + '\n' + PlaceMemory(self._memory_dir).cue(self.step_count)

    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, spatial_guidance, map_analysis):
        self._memory_phase = 'LOG'
        return super()._log_phase(obs_text, prev_action, prev_validated_action,
                                 diff_notice, self._memory_context(), map_analysis)

    def _plan_validate_phase(self, obs_text, instruction_prompt, diff_notice,
                             spatial_guidance, map_analysis, prev_validated_action=None):
        self._memory_phase = 'PLAN'
        return super()._plan_validate_phase(obs_text, instruction_prompt, diff_notice,
                                          self._memory_context(), map_analysis,
                                          prev_validated_action=prev_validated_action)


def validate_box(box):
    box = [0.0, 0.0, 1.0, 1.0] if box is None else box
    if (not isinstance(box, list) or len(box) != 4 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        or not 0 <= v <= 1 for v in box) or box[0] >= box[2] or box[1] >= box[3]):
        raise ValueError('box must be four ordered finite fractions within [0,1]')
    return box


def compare_images(before, current, box=None):
    """Exact image differences only; deliberately no progress classification."""
    box = validate_box(box)
    if before.size != current.size:
        return {'status': 'different_dimensions', 'before_size': list(before.size),
                'current_size': list(current.size)}, None
    width, height = current.size
    bounds = [int(math.floor(box[0]*width)), int(math.floor(box[1]*height)),
              int(math.ceil(box[2]*width)), int(math.ceil(box[3]*height))]
    a = np.asarray(before.convert('RGB').crop(tuple(bounds)), dtype=np.int16)
    b = np.asarray(current.convert('RGB').crop(tuple(bounds)), dtype=np.int16)
    delta = np.abs(a-b)
    changed = np.any(delta != 0, axis=2)
    ys, xs = np.where(changed)
    bbox = [int(xs.min())+bounds[0], int(ys.min())+bounds[1],
            int(xs.max())+bounds[0]+1, int(ys.max())+bounds[1]+1] if len(xs) else None
    return {'status': 'compared', 'crop_pixels': bounds,
            'identical_pixels': not bool(changed.any()),
            'changed_fraction': float(changed.mean()),
            'mean_absolute_channel_difference': float(delta.mean()),
            'changed_bbox_pixels': bbox,
            'interpretation': 'Exact pixel change only. Does not establish movement, safety, score gain, or progress.'}, bounds


def install(agent, registry):
    install_visual(agent, registry)
    obj = {'type': 'object', 'additionalProperties': False}
    string = {'type': 'string', 'maxLength': 600}
    name = {'type': 'string', 'maxLength': 100}
    strings = {'type': 'array', 'maxItems': 12, 'items': {'type': 'string', 'maxLength': 200}}

    def place_update(inspect=True, focus=None, **kwargs):
        if not isinstance(inspect, bool):
            raise ValueError('inspect must be a boolean')
        if focus is not None:
            validate_box(focus)
        inspection = None
        program = None
        # Only this model-invoked handler can request historical pixels.
        outcome = kwargs.get('outcome')
        if inspect and outcome is not None and (
                agent._inspected_action_step != agent.step_count-1 or focus is not None):
            source = agent._transition_source
            if (agent._memory_phase == 'LOG' and source and
                    source['observed_step'] == agent.step_count-1 and
                    agent.current_image is not None and agent.current_frame_id):
                cache_path = Path(agent._memory_dir) / '_transition_cache' / 'previous.png'
                with Image.open(cache_path) as saved:
                    measured, bounds = compare_images(saved, agent.current_image, focus)
                inspection = {'source_frame_id': source['frame_id'],
                              'source_step': source['observed_step'],
                              'current_frame_id': agent.current_frame_id,
                              'current_step': agent.step_count,
                              'measurement': measured,
                              'notice': 'Inspect the next historical/current images; correct this claim if contradicted. Numbers do not identify progress.'}
                ops = [{'op': 'frame', 'id': source['frame_id'], 'as': 'old'}]
                if focus is not None and bounds is not None:
                    ops += [{'op': 'frame', 'id': agent.current_frame_id, 'as': 'now'}]
                    for alias in ('old', 'now'):
                        ops += [{'op': 'crop', 'src': alias, 'box': bounds, 'as': alias+'_crop'},
                                {'op': 'resize', 'src': alias+'_crop', 'max_side': 768, 'as': alias+'_view'}]
                    ops += [{'op': 'tile', 'srcs': ['old_view', 'now_view'], 'cols': 2,
                             'labels': ['before previous action', 'observed after action'], 'as': 'pair'}]
                    program = {'ops': ops, 'output': 'pair'}
                else:
                    ops += [{'op': 'resize', 'src': 'old', 'max_side': 1536, 'as': 'before'}]
                    program = {'ops': ops, 'output': 'before'}
        result = PlaceMemory(agent._memory_dir).update_place(
            agent.step_count, agent.current_frame_id, agent._memory_phase,
            agent._observed_previous_action, inspection=inspection, **kwargs)
        if program is not None:
            agent.visual_request = program
            agent._inspected_action_step = agent.step_count-1
            result['inspection'] = inspection
        return result

    def place_recall(query='', limit=6):
        return PlaceMemory(agent._memory_dir).recall(query, limit)

    def route_memory(operation='read', **kwargs):
        return PlaceMemory(agent._memory_dir).update_route(
            agent.step_count, agent.current_frame_id, operation, **kwargs)

    def compare_frame(image_id=None, box=None):
        if agent.current_image is None or not agent.current_frame_id:
            return {'status': 'no_current_image'}
        if image_id is None:
            entries = [(i, e) for i, e in agent.visual.data['images'].items()
                       if e['step'] < agent.step_count]
            if not entries:
                return {'status': 'no_saved_earlier_frame',
                        'hint': 'Use SAVE_FRAME at a useful place, then compare it on a later observation.'}
            image_id = max(entries, key=lambda row: row[1]['step'])[0]
        metadata, before = agent.visual.recall(image_id)
        entry = agent.visual.data['images'][image_id]
        frame_id = entry.get('frame_id')
        if not frame_id:
            raise ValueError('Saved entry has no host frame id')
        result, bounds = compare_images(before, agent.current_image, box)
        result.update(before=metadata, before_frame_id=frame_id,
                      current_frame_id=agent.current_frame_id, current_step=agent.step_count)
        if bounds is not None:
            ops = [{'op': 'frame', 'id': frame_id, 'as': 'old'},
                   {'op': 'frame', 'id': agent.current_frame_id, 'as': 'now'}]
            for source in ('old', 'now'):
                ops += [{'op': 'crop', 'src': source, 'box': bounds, 'as': source+'_crop'},
                        {'op': 'resize', 'src': source+'_crop', 'max_side': 700, 'as': source+'_view'}]
            ops.append({'op': 'tile', 'srcs': ['old_view', 'now_view'], 'cols': 2,
                        'labels': ['saved observation', 'current observation'], 'as': 'pair'})
            agent.visual_request = {'ops': ops, 'output': 'pair'}
        return result

    registry.register(Tool('PLACE_UPDATE',
        'Store/correct current observed place facts. [] clears a list. In LOG only, outcome reports the PREVIOUS action. With an outcome, inspect=true (default) retrieves its actual predecessor image on the NEXT request once per action; compare with current image, then correct mistaken claims before DONE. Optional focus selects a fractional region for numerical change and a paired crop. inspect=false skips recall. supersedes removes an explicitly named incorrect old place without inferring connectivity.',
        {**obj, 'properties': {'place': name, 'exits': strings, 'remaining': strings,
            'note': string, 'confidence': {'type': 'string', 'enum': ['confirmed', 'tentative']},
            'outcome': {'type': 'string', 'enum': ['moved', 'blocked', 'unchanged', 'unknown', 'reset']},
            'from_place': name, 'progress': {'type': 'string', 'enum': ['gain', 'none', 'unknown']},
            'supersedes': name, 'inspect': {'type': 'boolean'},
            'focus': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                      'items': {'type': 'number', 'minimum': 0, 'maximum': 1}}},
         'required': ['place']}, place_update, ('LOG', 'PLAN'), 'memory_write', 'metadata'))
    registry.register(Tool('PLACE_RECALL',
        'Retrieve dated active place facts, observed action outcomes, the route, and explicitly retired identity labels. Query a landmark or exit; empty query retrieves recent/current places. Claims can be tentative or stale; use current pixels and PLACE_UPDATE to correct them.',
        {**obj, 'properties': {'query': {'type': 'string', 'maxLength': 200},
                              'limit': {'type': 'integer', 'minimum': 1, 'maximum': 12}}, 'required': []},
        place_recall, ('LOG', 'PLAN'), 'memory_read', 'metadata'))
    registry.register(Tool('ROUTE_MEMORY',
        'Read/update/clear the persistent route intention: target, next checkpoint, and reason. Update only provided fields. new_scene explicitly invalidates all place claims for a changed layout or restart. Never treats a plan as an observed outcome.',
        {**obj, 'properties': {'operation': {'type': 'string', 'enum': ['read', 'update', 'clear', 'new_scene']},
                              'target': string, 'checkpoint': string, 'reason': string}, 'required': []},
        route_memory, ('LOG', 'PLAN'), 'memory_write', 'metadata'))
    registry.register(Tool('COMPARE_FRAME',
        'Measure exact pixel differences between a saved image and current image, then show a paired view on the next request. image_id defaults to newest explicitly saved earlier frame; optional box is [left,top,right,bottom] in image fractions. Changes are not proof of progress. Use for uncertain movement, alignment, or topology; await the next request to inspect pixels.',
        {**obj, 'properties': {'image_id': name, 'box': {'type': 'array', 'maxItems': 4,
                                                       'items': {'type': 'number', 'minimum': 0, 'maximum': 1}}},
         'required': []}, compare_frame, ('LOG', 'PLAN'), 'memory_read', 'image'))
