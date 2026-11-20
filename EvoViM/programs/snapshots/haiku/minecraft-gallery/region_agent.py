"""Region-memory agent for line M.

LOG (walk steps): the model saves crops of wall panels with a k,S label and can
inspect a labelled grid.  PLAN (question steps): the model must run
MATCH_QUERY_IMAGE before committing an answer, and the commit is checked against
the match results.  Walk steps keep the parent's text-file tools.

Command form (ASCII; matches the child decoder COMMAND_RE):
    <|NAME|>payload<|END|>
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re

import agents.memory_agent as parent_module
from agents.memory_agent import MemoryAgent
from prompt_builder.history import Message

from painting_match import STRONG_SCORE, score_references
from region_store import ANSWER_RE, RegionStore, normalize_label

logger = logging.getLogger(__name__)

LOG_MAX_ITERATIONS = 4
PLAN_MAX_ITERATIONS = 6
QUESTION_RE = re.compile(r"Question\s+\d+\s*/\s*\d+", re.IGNORECASE)
EXCLUDED_STATE = {"client", "config", "prompt_builder", "memory", "visual", "regions",
                  "current_image", "current_frame_id", "visual_request", "visual_observation_available",
                  "_memory_dir", "_memory_base"}

WALK_LOG = """You are maintaining your memory while walking through a corridor.

{action_description}

{diff_notice}

ENVIRONMENT RESPONSE TO YOUR PREVIOUS ACTION:
{observation}

SAVED REGIONS (id, step, label, box):
{region_index}

TEXT MEMORY FILES:
{file_index}

WHAT TO SAVE
Paintings hang on the walls beside numbered doorways. When one painting panel is clearly visible in the current image, save a crop of that panel alone:
  <|SAVE_REGION|>{"box":[x0,y0,x1,y1],"label":"k,S"}<|END|>
- box: pixel coordinates [left, top, right, bottom] around that one panel in the current image. Use <|SHOW_GRID|>{}<|END|> first if you need coordinates; the grid image is attached to your next message.
- label: k is the number on the doorway the panel hangs beside. S is L if the panel is left of that doorway, R if it is right, as seen while walking toward it. If you cannot read k or S, use label "unknown".
- Do not save a panel that is already listed under SAVED REGIONS. Save nothing when no new panel is visible.
- Text notes are optional; keep them to one short line.

COMMANDS (one per reply):
  <|SAVE_REGION|>{"box":[x0,y0,x1,y1],"label":"k,S"}<|END|>
  <|SHOW_GRID|>{}<|END|>
  <|LIST_REGIONS|>{}<|END|>
  <|RELABEL_REGION|>{"region_id":"reg_0001","label":"k,S"}<|END|>
  <|DELETE_REGION|>{"region_id":"reg_0001"}<|END|>
  the text-file commands listed by the system message
  DONE alone (use the exact DONE form given in the system message) when you have nothing more to record.
"""

LOG_CONTINUE = """Command result:
{results}

SAVED REGIONS:
{region_index}

TEXT MEMORY FILES:
{file_index}

Give one more command, or DONE alone when nothing else is worth recording."""

WALK_PLAN = """{instruction}

{stuckness}

SAVED REGIONS (id, step, label, box):
{region_index}

TEXT MEMORY FILES:
{file_index}

CURRENT OBSERVATION:
{observation}

Choose the next action. Check saved regions with <|LIST_REGIONS|>{}<|END|> when it helps.
Commit exactly one action with <|ACTION|>your_action<|END|>."""

QUESTION_PLAN = """QUESTION STEP. The current image shows a painting that the question asks about. It may be re-photographed: rescaled, slightly rotated, cropped, or lit differently.

CURRENT OBSERVATION (contains the question and its answer format):
{observation}

Saved panel regions: {count}

PROCEDURE
1. First compare this painting with your saved regions: <|MATCH_QUERY_IMAGE|>{}<|END|>
2. Read the result. strong_labels lists the doorway labels whose saved panels match this painting strongly. Weaker matches are not evidence.
3. Commit one answer, in the question's format:
   - <|ACTION|>answer <doorway number>,<L or R><|END|> using a label from strong_labels, or
   - <|ACTION|>answer not_seen<|END|> only when strong_labels is empty.
A commit that the match results do not support is rejected.

Commands: <|MATCH_QUERY_IMAGE|>{}<|END|>, <|LIST_REGIONS|>{}<|END|>, then one <|ACTION|> commit."""

PLAN_CONTINUE = """Result:
{results}

Issue one more command, or commit exactly one <|ACTION|> when ready."""

GATE_TEXT = """[GATE] Your ACTION was not accepted: {reason}.
Reply with one command: <|MATCH_QUERY_IMAGE|>{}<|END|> if you have not compared the painting yet, or a valid <|ACTION|> commit."""


def is_question_text(text: str) -> bool:
    """Question steps carry a 'Question k/n' line and the answer format; walk steps do not."""
    return bool(QUESTION_RE.search(str(text))) and "not_seen" in str(text).lower()


def _fill(template: str, values: dict) -> str:
    """Single-pass placeholder fill: inserted values are never scanned for placeholders."""
    return re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), template)


class RegionMemoryAgent(MemoryAgent):
    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder, config)
        self.regions = RegionStore()
        self.question_mode = False
        self.last_match = None

    # ----- episode lifecycle -------------------------------------------------
    def configure_memory(self, env_name, task, episode_idx=0):
        super().configure_memory(env_name, task, episode_idx)
        self.regions.bind(os.path.join(self._memory_dir, "regions"))

    def reset(self):
        super().reset()
        self.regions.clear()
        self.question_mode = False
        self.last_match = None

    def export_state(self):
        state = {}
        for key, value in self.__dict__.items():
            if key in EXCLUDED_STATE or callable(value):
                continue
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError):
                continue
            state[key] = copy.deepcopy(value)
        state["_regions"] = self.regions.export_state()
        return state

    def import_state(self, state):
        for key, value in state.items():
            if key == "_regions":
                self.regions.import_state(value)
            elif key not in EXCLUDED_STATE and not key.startswith("__"):
                setattr(self, key, value)

    # ----- helpers -------------------------------------------------------------
    def _region_index_text(self) -> str:
        if not self.regions.items:
            return "  (none)"
        rows = []
        for item in self.regions.items[-40:]:
            suffix = "" if item["label_valid"] else " (unlabelled)"
            rows.append("  %s step %d label %s box %s%s" % (
                item["id"], item["step"], item["label"], item["box"], suffix))
        return "\n".join(rows)

    def run_match(self) -> dict:
        """Compare the current question image with saved walk-panel regions; sets last_match."""
        record = {"step": self.step_count, "angle": None, "ranking": [], "strong_labels": []}
        try:
            if self.current_image is None:
                raise ValueError("no current image is available")
            references = [item for item in self.regions.items if not item["question_frame"]]
            if not references:
                record["note"] = "no saved regions yet"
            else:
                crops = [self.regions.load(item["id"]) for item in references]
                scored = score_references(self.current_image, crops)
                record["angle"] = round(scored["angle"], 1)
                ranking = []
                for item, (score, scale, _loc) in zip(references, scored["scores"]):
                    strong = bool(score >= STRONG_SCORE and item["label_valid"])
                    ranking.append({"region_id": item["id"], "label": item["label"],
                                    "label_valid": item["label_valid"], "step": item["step"],
                                    "score": round(float(score), 3),
                                    "scale": None if scale is None else round(float(scale), 2),
                                    "strong": strong, "frame_id": item["frame_id"], "box": item["box"]})
                ranking.sort(key=lambda entry: entry["score"], reverse=True)
                record["ranking"] = ranking
                for entry in ranking:
                    if entry["strong"] and entry["label"] not in record["strong_labels"]:
                        record["strong_labels"].append(entry["label"])
        except Exception as exc:  # reported to the model; never crashes the step
            record["error"] = str(exc)
        self.last_match = record
        return record

    def _current_match(self):
        record = self.last_match
        if record is not None and record.get("step") == self.step_count:
            return record
        return None

    def _gate(self, action: str):
        """None if the commit is allowed; otherwise the reason it is refused (question steps only)."""
        if not self.question_mode:
            return None
        match = ANSWER_RE.match(action)
        if not match:
            return "this is a question step, so commit 'answer <k,S>' or 'answer not_seen'"
        answer = match.group(1)
        record = self._current_match()
        if record is None:
            return "call MATCH_QUERY_IMAGE in this step before answering"
        strong = record.get("strong_labels", [])
        if answer.lower() == "not_seen":
            if strong:
                return "strong matches exist (%s), so answer one of those labels" % ", ".join(strong)
            return None
        label = normalize_label(answer)
        if label in strong:
            return None
        if strong:
            return "%s is not a strong match; strong matches are %s" % (label, ", ".join(strong))
        return "there is no strong match in this step, so answer not_seen"

    def _fallback_action(self) -> str:
        if not self.question_mode:
            return self._get_fallback_action()
        record = self._current_match()
        if record is None:
            record = self.run_match()
        strong = record.get("strong_labels", [])
        return "answer %s" % strong[0] if strong else "answer not_seen"

    # ----- phases ----------------------------------------------------------------
    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, spatial_guidance, map_analysis):
        self.question_mode = is_question_text(obs_text)
        if self.question_mode:
            return {"initial_prompt": "", "turns": [], "input_tokens": 0, "output_tokens": 0,
                    "skipped": "question_step"}
        prev_step = self.step_count - 1
        effective = prev_validated_action or prev_action
        action_description = 'At the PREVIOUS step (step %d) you took action: "%s".' % (prev_step, effective)
        if prev_validated_action is not None and prev_validated_action != prev_action:
            action_description += ' Your output "%s" was not valid; the environment ran "%s" instead.' % (
                prev_action, prev_validated_action)
        prompt = _fill(WALK_LOG, {
            "action_description": action_description, "diff_notice": diff_notice,
            "observation": obs_text, "region_index": self._region_index_text(),
            "file_index": self.memory.file_index()})
        messages = [Message(role="user", content=prompt)]
        total_in = total_out = 0
        turns = []
        max_iterations = max(1, min(int(self.max_log_iterations), LOG_MAX_ITERATIONS))
        for _iteration in range(max_iterations):
            current_prompt = messages[-1].content
            try:
                response = self.client.generate(messages)
            except Exception as exc:
                logger.error("Log phase LLM call failed: %s", exc)
                turns.append({"prompt": current_prompt, "raw_response": "[ERROR: %s]" % exc, "tool_results": ""})
                break
            total_in += response.input_tokens
            total_out += response.output_tokens
            text = response.completion
            calls, has_done = parent_module.parse_log_calls(text)
            result_parts = []
            for op, args in calls:
                result = self._execute_write_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)
            results_text = "\n\n".join(result_parts)
            turns.append({"prompt": current_prompt, "raw_response": text, "tool_results": results_text})
            messages.append(Message(role="assistant", content=text))
            if has_done or not calls:
                break
            messages.append(Message(role="user", content=_fill(LOG_CONTINUE, {
                "results": results_text, "region_index": self._region_index_text(),
                "file_index": self.memory.file_index()})))
        return {"initial_prompt": prompt, "turns": turns, "input_tokens": total_in, "output_tokens": total_out}

    def _plan_validate_phase(self, obs_text, instruction_prompt, diff_notice, spatial_guidance,
                             map_analysis, prev_validated_action=None):
        self.question_mode = is_question_text(obs_text)
        if self.question_mode:
            prompt = _fill(QUESTION_PLAN, {"observation": obs_text, "count": len(self.regions.items)})
        else:
            stuckness = ""
            if diff_notice:
                stuckness = diff_notice
                if prev_validated_action:
                    stuckness += "\nYour last executed action was '%s'. Do NOT repeat it." % prev_validated_action
            prompt = _fill(WALK_PLAN, {
                "instruction": instruction_prompt, "stuckness": stuckness,
                "region_index": self._region_index_text(), "file_index": self.memory.file_index(),
                "observation": obs_text})
        messages = [Message(role="user", content=prompt)]
        total_in = total_out = 0
        turns = []
        final_action = None
        model_id = "unknown"
        max_iterations = max(1, min(int(self.max_validate_iterations), PLAN_MAX_ITERATIONS))
        for _iteration in range(max_iterations):
            current_prompt = messages[-1].content
            try:
                response = self.client.generate(messages)
            except Exception as exc:
                logger.error("Plan phase LLM call failed: %s", exc)
                turns.append({"prompt": current_prompt, "raw_response": "[ERROR: %s]" % exc,
                              "tool_results": "", "committed_action": None})
                break
            model_id = response.model_id
            total_in += response.input_tokens
            total_out += response.output_tokens
            text = response.completion
            calls, committed = parent_module.parse_plan_calls(text)
            messages.append(Message(role="assistant", content=text))
            if isinstance(committed, str):
                reason = self._gate(committed)
                turns.append({"prompt": current_prompt, "raw_response": text, "tool_results": "",
                              "committed_action": None if reason else committed, "gate": reason})
                if reason is None:
                    final_action = committed
                    break
                messages.append(Message(role="user", content=GATE_TEXT.replace("{reason}", reason)))
                continue
            result_parts = []
            for op, args in calls:
                result = self._execute_read_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)
            results_text = "\n\n".join(result_parts)
            turns.append({"prompt": current_prompt, "raw_response": text, "tool_results": results_text,
                          "committed_action": None})
            if not calls:
                logger.warning("Plan phase: no command or ACTION. Raw: %s", text[:200])
                break
            messages.append(Message(role="user", content=PLAN_CONTINUE.replace("{results}", results_text)))
        if final_action is None:
            final_action = self._fallback_action()
            logger.warning("Plan phase ended without a valid ACTION; using '%s'.", final_action)
        return {"initial_prompt": prompt, "turns": turns, "final_action": final_action,
                "input_tokens": total_in, "output_tokens": total_out, "model_id": model_id,
                "question_mode": self.question_mode}
