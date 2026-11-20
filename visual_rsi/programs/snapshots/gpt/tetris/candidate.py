"""M1: explicit replaceable state, intent persistence and measured grid recall."""
from agents import memory_agent as upstream
from agents.memory_agent import MemoryAgent
from agents import state_memory as state
from client import LLMResponse
from prompt_builder.history import Message
from registry import Tool
from visual_tools import install_visual


MEMORY_GUIDANCE = """Your external memory supports replaceable state rather than only a growing diary.
SAVE_STATE stores an observed fact, a lesson, or an INTENTION; NOTE_RESULT in LOG
records the observed result of the previous action. Reuse the same key to replace
a stale claim. Corrections supersede the old revision. FORGET_STATE deletes a
claim when it no longer applies, including a placement intention after a piece
locks or the board resets. Only explicitly pinned notes are previewed by default;
RECALL_STATE retrieves full entries and, optionally, evidence images/comparisons.
You choose every save and recall; skip operations that will not help.

For a grid visible in the screenshot, GRID_CONFIG can record geometry and an
empty cell selected from THIS image (pixel bbox; cells are [column,row], zero
based). READ_GRID then measures cell colors and optional changes against a
captured state. It does not recognize objects, active pieces, progress or score.
If there is a ghost/landing preview, select a ghost_cell in GRID_CONFIG so it
is excluded from solid cells. Verify calibration against the current screenshot.
Use active_cells in READ_GRID to exclude the moving piece before interpreting
column heights, holes and bottom gaps. Pixel measurements are optional and
unavailable until you configure the grid; normal text/image memory still works.

For falling-block tasks, distinguish the moving piece, landing preview and
settled stack. Keep a concrete placement intention across movement/rotation
steps: desired landing cells, orientation, next subgoal and a condition that
invalidates it. Save this with kind=intent, pin=true. After each action, check
the successor observation before continuing. A piece moving down does not prove
it locked; a changed color count does not prove a line clear. Confirm line/score
claims from the visible labelled counter; use unknown if unreadable. Use the
settled grid and gaps to evaluate the intended landing before committing a drop.
Repeated commands can have different effects; record observed responsiveness as
a revisable lesson rather than assuming a timing. Infer all locations and control
effects from the current episode. Current pixels override stale text.
When updating a captured board, compare against it before replacing it. Older
revisions retain text and frame provenance, but only the latest measured grid.
"""


class Agent(MemoryAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder, config)
        self._memory_phase = 'PLAN'
        self._observed_previous_action = None

    def _format_obs(self, obs):
        return super()._format_obs(obs)

    def act(self, obs, prev_action=None, prev_validated_action=None):
        # No automatically written history or predicted outcomes. The host
        # supplies a current successor before we open LOG.
        obs_text = self._format_obs(obs)
        self._observed_previous_action = prev_validated_action or prev_action
        log = None
        total_in = total_out = 0
        if self.step_count > 0 and prev_action is not None:
            log = self._log_phase(obs_text, prev_action, prev_validated_action, '', '', '')
            total_in += log['input_tokens']
            total_out += log['output_tokens']
        plan = self._plan_validate_phase(obs_text, self.prompt_builder.system_prompt or '', '', '', '',
                                         prev_validated_action=prev_validated_action)
        total_in += plan['input_tokens']
        total_out += plan['output_tokens']
        self.last_step_debug = {'step': self.step_count, 'log_phase': log, 'plan_phase': plan,
                                'memory_snapshot': self.memory.snapshot()}
        self.step_count += 1
        return LLMResponse(model_id=plan['model_id'], completion=plan['final_action'], stop_reason='stop',
                           input_tokens=total_in, output_tokens=total_out, reasoning=None)

    def _run_phase(self, initial_prompt, phase, limit):
        messages = [Message(role='user', content=initial_prompt)]
        self._memory_phase = phase
        turns, total_in, total_out = [], 0, 0
        final_action, model_id = None, 'unknown'
        for _ in range(min(20, max(1, int(limit)))):
            try:
                response = self.client.generate(messages)
            except Exception as exc:
                # Preserve upstream's handling of broker/client failures;
                # HostStop is a BaseException and still reaches the host.
                turns.append({'prompt': messages[-1].content, 'raw_response': '[ERROR: {}]'.format(exc),
                              'tool_results': '', 'committed_action': None})
                break
            model_id = response.model_id
            total_in += response.input_tokens
            total_out += response.output_tokens
            text = response.completion
            if phase == 'LOG':
                calls, done = upstream.parse_log_calls(text)
                action = None
            else:
                calls, action = upstream.parse_plan_calls(text)
                done = False
            outputs = []
            for op,args in calls:
                execute = self._execute_write_op if phase == 'LOG' else self._execute_read_op
                result = execute(op,args)
                _,formatted = self._format_tool_result(op,args,result)
                outputs.append(formatted)
            results = '\n\n'.join(outputs)
            turns.append({'prompt': messages[-1].content, 'raw_response': text, 'tool_results': results,
                          'committed_action': action})
            messages.append(Message(role='assistant', content=text))
            if done:
                break
            if action is not None:
                final_action = action
                break
            if not calls:
                break
            messages.append(Message(role='user', content=results+'\n\n'+state.brief(self)+
                                    '\nContinue with exactly one command allowed in this phase.'))
        if phase == 'PLAN' and final_action is None:
            final_action = self._get_fallback_action()
        return {'initial_prompt': initial_prompt, 'turns': turns, 'input_tokens': total_in,
                'output_tokens': total_out, 'model_id': model_id, 'final_action': final_action}

    def _log_phase(self, obs_text, prev_action, prev_validated_action, diff_notice,
                   spatial_guidance, map_analysis):
        self._observed_previous_action = prev_validated_action or prev_action
        prompt = ('LOG at observation step {}. Only now can you record the result of action '
                  '{} executed at step {}. You have NOT yet acted at the current step.\n'
                  'Current successor observation:\n{}\n\n{}\n\n{}\n\nMemory files:\n{}\n'
                  'Inspect differences when uncertain. Update a state or result only when useful. '
                  'Do not call an intended outcome an observed result. Issue one memory command '
                  'per reply; finish with <|DONE|>.').format(self.step_count,
                    self._observed_previous_action, self.step_count-1, obs_text, MEMORY_GUIDANCE,
                    state.brief(self), self.memory.file_index())
        return self._run_phase(prompt, 'LOG', self.max_log_iterations)

    def _plan_validate_phase(self, obs_text, instruction_prompt, diff_notice,
                             spatial_guidance, map_analysis, prev_validated_action=None):
        prompt = ('{}\n\nPLAN at step {}.\nCurrent observation:\n{}\n\n{}\n\n{}\n\n'
                  'Memory files:\n{}\n\n'
                  'Choose one action using the current observation and relevant memory. '
                  'If an intention remains valid, continue its subgoal; revise it when current '
                  'evidence contradicts it. You may SAVE_STATE a proposed intention or current '
                  'observation here, but its future result is unknown. RECALL_STATE or READ_GRID '
                  'can resolve uncertainty. Use exactly one command per reply. Only ACTION '
                  'commits a world action; eventually commit exactly one legal action.').format(
                    instruction_prompt, self.step_count, obs_text, MEMORY_GUIDANCE,
                    state.brief(self), self.memory.file_index())
        return self._run_phase(prompt, 'PLAN', self.max_validate_iterations)


def install(agent, registry):
    install_visual(agent, registry)
    obj = {'type': 'object', 'additionalProperties': False}
    cell = {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'maxItems': 2}
    key = {'type': 'string', 'maxLength': 48}
    text = {'type': 'string', 'maxLength': 1200}
    boolean = {'type': 'boolean'}

    registry.register(Tool('GRID_CONFIG', 'Optional: inspect or explicitly configure a grid from this episode screenshot. With no arguments returns configuration/help. bbox=[left,top,right,bottom] in image pixels; empty_cell and optional ghost_cell=[column,row], zero based. Verify READ_GRID afterwards. Does not save a board snapshot.',
        {**obj, 'properties': {'bbox': {'type':'array','items':{'type':'number','minimum':0},'maxItems':4},
         'rows': {'type':'integer','minimum':1,'maximum':64}, 'columns':{'type':'integer','minimum':1,'maximum':64},
         'empty_cell':cell, 'ghost_cell':cell}, 'required': []},
        lambda **kw: state.configure_grid(agent, **kw), ('LOG','PLAN'), 'memory_write', 'metadata'))
    registry.register(Tool('READ_GRID', 'Measure current grid color symbols/components. No storage. Optional active_cells excludes the moving piece for settled-board heights/holes/gaps. compare_key compares against an explicitly captured grid; changes are not progress. First configure with GRID_CONFIG.',
        {**obj, 'properties': {'active_cells': {'type':'array','items':cell,'maxItems':128}, 'compare_key':key}, 'required': []},
        lambda **kw: state.read_grid(agent, **kw), ('LOG','PLAN'), 'memory_read', 'metadata'))
    registry.register(Tool('SAVE_STATE', 'Explicitly save/replace a keyed current observation, revisable lesson or future intention. Reusing key supersedes old text; pin=true previews this text in future prompts (max 4). capture_grid=true saves current measured grid if configured, plus the host frame reference. Never record an unobserved outcome here.',
        {**obj, 'properties': {'key':key, 'text':text, 'kind':{'type':'string','enum':['observation','intent','lesson']},
         'capture_grid':boolean, 'pin':boolean}, 'required':['key','text','kind']},
        lambda **kw: state.put(agent, **kw), ('LOG','PLAN'), 'memory_write', 'metadata'))
    registry.register(Tool('NOTE_RESULT', 'LOG only: save/replace a keyed observed result of the preceding executed action. Automatically stamps action step t-1, observed step t and frame provenance. pin=true previews it; capture_grid=true captures current measurement. Describe uncertainty honestly.',
        {**obj,'properties':{'key':key,'text':text,'capture_grid':boolean,'pin':boolean},'required':['key','text']},
        lambda **kw: state.put(agent, kind='result', **kw), ('LOG',), 'memory_write', 'metadata'))
    registry.register(Tool('RECALL_STATE', 'Retrieve a keyed state with evidence step/age and superseded revisions; no key lists entries. compare=true numerically compares a captured grid with current pixels. show_image=true requests that entry historical frame through the host for the NEXT model call. Current observation overrides stale claims.',
        {**obj,'properties':{'key':key,'compare':boolean,'show_image':boolean},'required':[]},
        lambda **kw: state.recall(agent, **kw), ('LOG','PLAN'), 'memory_read', 'metadata'))
    registry.register(Tool('FORGET_STATE', 'Explicitly delete a keyed obsolete state/intention, for example after a confirmed reset or completed placement. Does not delete unrelated lessons or legacy files.',
        {**obj,'properties':{'key':key},'required':['key']},
        lambda **kw: state.forget(agent, **kw), ('LOG','PLAN'), 'memory_write', 'metadata'))
