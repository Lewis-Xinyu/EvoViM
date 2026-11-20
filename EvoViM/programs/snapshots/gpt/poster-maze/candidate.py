"""M revision: model-selected observation bundles and evidence-bearing recall."""
import json
import re

import agents.memory_agent as upstream
from agents.memory_agent import MemoryAgent
from prompt_builder.history import Message
from visual_tools import install_visual
from query_review import review_status


def parse_public(text):
    question = bool(re.search(r'Question\s+\d+\s*/\s*\d+', text, re.I))
    pose = re.search(r'You are at cell\s*\((\d+),\s*(\d+)\)\s*facing\s*([NESW])', text)
    actions = re.search(r'Available actions:\s*([^\.\n]+)', text)
    goal = re.search(r'[Gg]oal is cell\s*\((\d+),\s*(\d+)\)', text)
    square = re.search(r'\b(\d+)\s*[x×]\s*(\d+)\s+cells\b', text, re.I)
    numbered = bool(re.search(r'rows and columns are numbered from 0\b', text, re.I))
    # Unlabelled non-square dimensions have an ambiguous row/column order.
    # Accept only explicit zero-based square grids; otherwise keep unknown.
    grid = ({'rows': int(square[1]), 'cols': int(square[2]), 'zero_based': True}
            if square and numbered and int(square[1]) == int(square[2]) and int(square[1]) > 0 else None)
    return {
        'pose': {'row': int(pose[1]), 'col': int(pose[2]), 'facing': pose[3]} if pose and not question else None,
        'actions': [x.strip() for x in actions[1].split(',')] if actions and not question else [],
        'actions_known': bool(actions) and not question,
        'goal': [int(goal[1]), int(goal[2])] if goal else None,
        'grid': grid,
        'question': question,
    }


class Agent(MemoryAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder, config)
        self.public_observation = {}
        self.previous_pose = None
        self.observed_transition = None

    def configure_memory(self, env_name, task, episode_idx=0):
        super().configure_memory(env_name, task, episode_idx)
        self.public_observation = {}
        self.previous_pose = None
        self.observed_transition = None

    def reset(self):
        super().reset()
        self.public_observation = {}
        self.previous_pose = None
        self.observed_transition = None

    def _format_obs(self, obs):
        text = super()._format_obs(obs)
        self.public_observation = parse_public(text)
        return text

    def act(self, obs, prev_action=None, prev_validated_action=None):
        current = parse_public(super()._format_obs(obs))
        self.public_observation = current
        self.observed_transition = None
        if self.step_count > 0 and prev_action is not None and self.previous_pose and current['pose']:
            self.observed_transition = {
                'action_step': self.step_count - 1, 'observed_step': self.step_count,
                'action': prev_validated_action or prev_action,
                'before': self.previous_pose, 'after': current['pose'],
            }
        response = super().act(obs, prev_action, prev_validated_action)
        # Only the immediately preceding public pose is carried implicitly.
        # No frame, route, exit history or outcome is persisted without a tool.
        self.previous_pose = current['pose']
        return response

    def _context(self):
        return json.dumps({'current_public': self.public_observation,
                           'observed_previous_transition': self.observed_transition,
                           'requested_query_review': review_status(self)}, ensure_ascii=False)

    def _phase(self, prompt, phase):
        """Use the host-patched decoders/dispatchers and unchanged phase limits."""
        messages = [Message(role='user', content=prompt)]
        turns = []
        total_in = total_out = 0
        final_action = None
        model_id = 'unknown'
        limit = self.max_log_iterations if phase == 'LOG' else self.max_validate_iterations
        for request_index in range(limit):
            remaining = limit - request_index - 1
            budget = f'\n{phase} request {request_index + 1}/{limit}; {remaining} further requests remain in this phase.'
            if phase == 'PLAN' and remaining == 0:
                budget += ' Commit your chosen ACTION now. A memory command here leaves no request to inspect its result or choose an action.'
            elif phase == 'PLAN' and remaining <= 3:
                budget += ' Use the evidence already retrieved and commit your answer/action before this phase ends. Repeating unchanged images adds no evidence.'
            elif phase == 'LOG' and remaining == 0:
                budget += ' Finish LOG with <|DONE|>; PLAN is where you commit the answer/action.'
            messages[-1].content += budget
            current_prompt = messages[-1].content
            try:
                response = self.client.generate(messages)
            except Exception as exc:
                turns.append({'prompt': current_prompt, 'raw_response': '[ERROR: %s]' % exc,
                              'tool_results': '', 'committed_action': None})
                break
            model_id = response.model_id
            total_in += response.input_tokens
            total_out += response.output_tokens
            reply = response.completion
            if phase == 'LOG':
                calls, done = upstream.parse_log_calls(reply)
                action = None
            else:
                calls, action = upstream.parse_plan_calls(reply)
                done = False
            results = []
            for op, args in calls:
                dispatch = self._execute_write_op if phase == 'LOG' else self._execute_read_op
                result = dispatch(op, args)
                _, formatted = self._format_tool_result(op, args, result)
                results.append(formatted)
            result_text = '\n\n'.join(results)
            turns.append({'prompt': current_prompt, 'raw_response': reply,
                          'tool_results': result_text, 'committed_action': action})
            messages.append(Message(role='assistant', content=reply))
            if action is not None:
                final_action = action
                break
            if done or not calls:
                break
            continuation = ('Continue with one memory command, or finish LOG with <|DONE|>.' if phase == 'LOG'
                            else 'Inspect the retrieved evidence, then choose another tool or commit one ACTION. You choose the answer/action.')
            if self.public_observation.get('question'):
                status = review_status(self)
                if status:
                    continuation += '\nQUERY WORKSPACE (from requested retrieval): ' + json.dumps(status, ensure_ascii=False)
                continuation += '\nJudge exact pattern identity. If no saved view matches, answer not_seen is valid. A nearest-ranked candidate is not evidence that it was seen. Record an assessment if useful; do not alternate the same unchanged frames without a new test.'
            messages.append(Message(role='user', content=result_text + '\n\n' + continuation))
        result = {'initial_prompt': prompt, 'turns': turns, 'input_tokens': total_in,
                  'output_tokens': total_out, 'model_id': model_id}
        if phase == 'PLAN':
            result['final_action'] = final_action if final_action is not None else self._get_fallback_action()
        return result

    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, spatial_guidance, map_analysis):
        prompt = f'''LOG, observation step {self.step_count}.
The PREVIOUS action (step {self.step_count-1}) was {prev_validated_action or prev_action!r}.
Only its now-observed result may be logged. A planned next action has no outcome yet.
{obs_text}
PUBLIC DERIVATION (not saved): {self._context()}

You control persistent memory. Prefer SAVE_FRAME {{}} to retain a useful navigation view or new cell/facing: one call saves the original frame AND the public pose, legal exits and observed previous transition. This includes clear approach views of patterns, before a close view clips them. The same call provides place memory even for a view without a panel. You may skip redundant views. Generic color words do not preserve a pattern.
No need to transcribe positions or estimate a distant wall to save evidence. RECALL_IMAGE can later derive a frontal wall candidate from saved public exits. Optional region selects a crop; ANNOTATE_IMAGE can correct a crop, wall or note. A side panel needs a model-verified wall; observer pose alone is not its location.
Use APPEND/WRITE/CREATE for additional text you want to keep; READ/SEARCH/TAIL/COUNT retrieve it. Original visual tools remain available. A question image is a query, not a navigation sighting. Finish with <|DONE|> when enough is recorded.
Memory files:\n{self.memory.file_index()}'''
        return self._phase(prompt, 'LOG')

    def _plan_validate_phase(self, obs_text, instruction_prompt, diff_notice,
                             spatial_guidance, map_analysis, prev_validated_action=None):
        if self.public_observation.get('question'):
            guidance = '''For this question, use <|RECALL_IMAGE|>{}<|END|> if you need a comparison. This opens a persistent query workspace: comparisons and your assessments survive LOG to PLAN, with request counts so you can avoid repeating unchanged views. The tile shows the query and historical alternatives together; its text legend gives row-major order. image_id shows that full frame alongside the query and comparison. Optional assessments stores YOUR verdict: use image_id from the retrieved results, verdict different_pattern/same_pattern/uncertain/unreviewed, and an optional note with a specific mismatching arrangement or unresolved feature. You can revise it later. Treat each question separately. If LOG already retrieved sufficient evidence, use it to answer without repeating retrieval.
Judge the exact pattern, not just shared colors or a nearest rank: even an unseen pattern has a top candidate. If none is the same pattern, commit answer not_seen. A high partial correlation alone can match a lure. Identity evidence is separate from wall-location evidence: a frontal ray endpoint is conditional on a frontal panel, and observer pose is not a distant wall cell. Unknown location alone does not establish not_seen. Choose a specific additional comparison only if it tests a missing feature; repeated unchanged full frames add no evidence. Commit your own ACTION before the request budget ends; tools never commit it.'''
        else:
            guidance = '''Choose a legal action using the current public pose and action list. To retrieve saved exits, checked facings, and unexplored neighboring places use <|RECALL_IMAGE|>{"mode":"places"}<|END|>. Use place memory when you revisit a cell or become unsure: avoid repeating turns/backtracks without checking which exits remain unknown. Unknown edges are not walls. You select the route and every world action. At step zero or before leaving an informative view you may explicitly SAVE_FRAME {} in PLAN; no previous outcome exists at step zero.'''
        prompt = f'''{instruction_prompt}

PLAN, step {self.step_count}.
{obs_text}
PUBLIC DERIVATION (not saved): {self._context()}
{guidance}

You decide what to store and retrieve through optional memory commands. SAVE_FRAME {{}} bundles a selected frame with its public place evidence. Text READ/SEARCH/TAIL/COUNT and LOG writes remain available. Use one command per reply. Only <|ACTION|>your_action<|END|> commits a world action; eventually commit one.
Memory files:\n{self.memory.file_index()}'''
        return self._phase(prompt, 'PLAN')


def install(agent, registry):
    install_visual(agent, registry)
