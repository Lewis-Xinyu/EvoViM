"""MemoryAgent with a pixel-derived card ledger (revision of the M line, generation 3).

The parent's LLM memory loop (LOG and PLAN phases, text files, visual tools) is kept.
Changes in this revision:
  R2  PLAN: a first-card commit onto a known-unpaired cell gets one ledger warning inside the
      same plan loop, and the model may commit again. Logic lives in agents/ledger_gate.py.
  R1  LOG: a reply that writes a file (APPEND, WRITE or CREATE) ends the LOG phase at once, so
      no DONE request follows a write. The LOG wording says so.
The ledger itself (perception, summary, query tool) is unchanged from the parent.
"""
import json
import logging

import agents.memory_agent as memory_agent
from agents.card_ledger import CardLedger
from agents.ledger_gate import is_lure, lure_message
from agents.memory_agent import MemoryAgent
from prompt_builder.history import Message

logger = logging.getLogger(__name__)

# Attributes that never enter the checkpoint (mirrors the worker's exclusion list).
_EXCLUDED = {"client", "config", "prompt_builder", "memory", "visual", "current_image",
             "current_frame_id", "visual_request", "visual_observation_available",
             "_memory_dir", "_memory_base", "ledger", "_ledger_text", "_second_flip"}

_WRITE_OPS = frozenset(["APPEND", "WRITE", "CREATE"])

_LOG_RULE_OLD = "You may issue multiple commands per turn. Finish with <|DONE|>."
_LOG_RULE_NEW = ("Send one command per reply. A reply that writes a file (APPEND, WRITE or CREATE) ends "
                 "logging at once, so write only the entry that matters most and do not send DONE after "
                 "a write. Send <|DONE|> alone only when nothing needs writing.")
assert _LOG_RULE_OLD in memory_agent.LOG_SYSTEM, "parent LOG_SYSTEM changed; update the R1 wording"
LOG_SYSTEM_R1 = memory_agent.LOG_SYSTEM.replace(_LOG_RULE_OLD, _LOG_RULE_NEW)

LOG_CONTINUE_R1 = """\
Command results:
{results}

Updated files:
{file_index}

Continue logging: SEARCH before you write. Then send one write, or <|DONE|> if nothing needs writing."""


def _action_description(prev_step, prev_action, prev_validated_action):
    if prev_validated_action is not None and prev_validated_action != prev_action:
        return (
            "At the PREVIOUS step (step {prev_step}) your LLM output was: "
            "\"{prev_action}\"\n"
            "This was NOT a valid action. The environment executed the "
            "default action \"{prev_validated}\" instead.\n"
            "THIS IS AN IMPORTANT FAILURE TO LOG — your output did not "
            "match any valid action."
        ).format(prev_step=prev_step, prev_action=prev_action, prev_validated=prev_validated_action)
    effective = prev_validated_action or prev_action
    return ("At the PREVIOUS step (step {prev_step}) you took action: \"{effective}\"").format(
        prev_step=prev_step, effective=effective)


class LedgerMemoryAgent(MemoryAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder, config)
        self.ledger = CardLedger()
        self._ledger_text = ""
        self._second_flip = False

    def configure_memory(self, env_name, task, episode_idx=0):
        super().configure_memory(env_name, task, episode_idx)
        self.ledger.reset()
        self._ledger_text = ""
        self._second_flip = False

    def reset(self):
        super().reset()
        self.ledger.reset()
        self._ledger_text = ""
        self._second_flip = False

    def act(self, obs, prev_action=None, prev_validated_action=None):
        short_text = (obs.get("text") or {}).get("short_term_context", "")
        self._ledger_text = self.ledger.observe(getattr(self, "current_image", None),
                                                short_text, self.step_count)
        self._second_flip = "Flip the second card" in short_text
        return super().act(obs, prev_action, prev_validated_action)

    def _format_obs(self, obs):
        base = super()._format_obs(obs)
        if self._ledger_text:
            return base + "\n\n" + self._ledger_text
        return base

    # ------------------------------------------------------------------ R1: LOG ends at a write
    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, spatial_guidance, map_analysis):
        prev_step = self.step_count - 1
        file_index = self.memory.file_index()
        initial_prompt = LOG_SYSTEM_R1.format(
            action_description=_action_description(prev_step, prev_action, prev_validated_action),
            diff_notice=diff_notice,
            spatial_guidance=spatial_guidance,
            map_analysis=map_analysis,
            observation=obs_text,
            file_index=file_index,
        )
        messages = [Message(role="user", content=initial_prompt)]
        total_in = 0
        total_out = 0
        turns = []

        for _iteration in range(self.max_log_iterations):
            current_prompt = messages[-1].content
            try:
                response = self.client.generate(messages)
            except Exception as e:
                logger.error("Log phase LLM call failed: %s", e)
                turns.append({"prompt": current_prompt,
                              "raw_response": "[ERROR: {}]".format(e),
                              "tool_results": ""})
                break

            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, has_done = memory_agent.parse_log_calls(assistant_text)

            result_parts = []
            for op, args in calls:
                result = self._execute_write_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)
            results_text = "\n\n".join(result_parts) if result_parts else ""

            turns.append({"prompt": current_prompt,
                          "raw_response": assistant_text,
                          "tool_results": results_text})
            messages.append(Message(role="assistant", content=assistant_text))

            if has_done:
                break
            if not calls:
                break
            if any(op in _WRITE_OPS for op, _args in calls):
                break  # R1: a write is the last LOG request of the step

            continue_msg = LOG_CONTINUE_R1.format(results=results_text,
                                                  file_index=self.memory.file_index())
            messages.append(Message(role="user", content=continue_msg))

        return {
            "initial_prompt": initial_prompt,
            "turns": turns,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

    # ------------------------------------------------------------------ R2: first-card gate in PLAN
    def _plan_validate_phase(self, obs_text, instruction_prompt,
                             diff_notice, spatial_guidance, map_analysis,
                             prev_validated_action=None):
        file_index = self.memory.file_index()

        stuckness_warning = ""
        if diff_notice:
            stuckness_warning = diff_notice
            if prev_validated_action:
                stuckness_warning += (
                    "\nYour last executed action was '{}'. "
                    "Do NOT repeat it — choose something different."
                ).format(prev_validated_action)

        initial_prompt = memory_agent.PLAN_SYSTEM.format(
            instruction_prompt=instruction_prompt,
            spatial_guidance=spatial_guidance,
            stuckness_warning=stuckness_warning,
            map_analysis=map_analysis,
            file_index=file_index,
            observation=obs_text,
        )
        messages = [Message(role="user", content=initial_prompt)]
        total_in = 0
        total_out = 0
        turns = []
        final_action = None
        model_id = "unknown"
        gate_warned = False

        for iteration in range(self.max_validate_iterations):
            current_prompt = messages[-1].content
            try:
                response = self.client.generate(messages)
            except Exception as e:
                logger.error("Plan phase LLM call failed: %s", e)
                turns.append({"prompt": current_prompt,
                              "raw_response": "[ERROR: {}]".format(e),
                              "tool_results": "",
                              "committed_action": None})
                break

            model_id = response.model_id
            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, committed_action = memory_agent.parse_plan_calls(assistant_text)

            result_parts = []
            for op, args in calls:
                result = self._execute_read_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)
            results_text = "\n\n".join(result_parts) if result_parts else ""

            # The gate fires once per step and never on the last allowed iteration, so a
            # warned step always ends with a committed action from the parent's own budget.
            gated = (committed_action is not None and not gate_warned
                     and iteration < self.max_validate_iterations - 1
                     and is_lure(self.ledger, committed_action, self._second_flip))
            if gated:
                gate_warned = True

            turns.append({"prompt": current_prompt,
                          "raw_response": assistant_text,
                          "tool_results": results_text,
                          "committed_action": committed_action,
                          "gate_warning": gated})
            messages.append(Message(role="assistant", content=assistant_text))

            if gated:
                head = ("Search results:\n" + results_text + "\n\n") if results_text else ""
                messages.append(Message(role="user",
                                        content=head + lure_message(self.ledger, committed_action)))
                continue
            if committed_action is not None:
                final_action = committed_action
                break
            if not calls:
                logger.warning("Plan phase: no SEARCH/READ/TAIL/COUNT/ACTION found. Raw: %s",
                               assistant_text[:200])
                final_action = self._get_fallback_action()
                break
            messages.append(Message(role="user",
                                    content=memory_agent.PLAN_CONTINUE.format(results=results_text)))

        if final_action is None:
            fallback = self._get_fallback_action()
            final_action = fallback
            logger.warning("Plan phase exhausted %d iterations without <|ACTION|>. Defaulting to '%s'.",
                           self.max_validate_iterations, fallback)

        return {
            "initial_prompt": initial_prompt,
            "turns": turns,
            "final_action": final_action,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "model_id": model_id,
        }

    # ------------------------------------------------------------------ checkpoint
    def export_state(self):
        base = {k: v for k, v in self.__dict__.items()
                if k not in _EXCLUDED and not callable(v) and _jsonable(v)}
        return {"base": base, "ledger": self.ledger.to_state()}

    def import_state(self, data):
        for key, value in (data.get("base") or {}).items():
            if key not in _EXCLUDED:
                setattr(self, key, value)
        self.ledger.from_state(data.get("ledger"))
        self._ledger_text = ""
        self._second_flip = False


def _jsonable(value):
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False
