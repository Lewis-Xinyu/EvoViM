import copy
import logging
import os
import re

from omegaconf import OmegaConf

from agents.base import BaseAgent
from agents.board_tracker import BoardTracker
from client import LLMResponse
from prompt_builder.history import Message

logger = logging.getLogger(__name__)

MAX_LOG_ITERATIONS = 5
MAX_VALIDATE_ITERATIONS = 5
MAX_SEARCH_RESULTS = 100

# The LOG phase makes at most one model request per step.  The tracker has
# already recorded the previous step's outcome automatically (board_state,
# match_index, round_log), so one request is enough to decide whether a
# correction is needed; follow-up requests only ever produced redundant
# re-logging in the development episodes.
LOG_PHASE_REQUESTS = 1

# LOG is skipped entirely unless an exception fires (tracker/observation
# contradiction, invalid flip, host-executed-action mismatch) or a periodic
# checkpoint is due.  Development evidence: in 458 LOG requests across both
# units the model issued 457 DONE replies and one SET_CELL_STATE that the
# tracker had already applied, i.e. always-on LOG never changed a decision.
# The checkpoint cadence keeps a bounded number of live LLM decision points
# where the LLM may still write to its memory files: APPEND/WRITE/CREATE are
# LOG-only tools, so a pure 0-LOG design would leave the LLM no storage
# authority at all (every write would be tracker-automated).  15 checkpoints
# per ~230-step episode cost ~3% of the episode's input tokens.
LOG_CHECKPOINT_EVERY = 16

FALLBACK_ACTION_DEFAULT = "wait"

_TWO_STEP_COMMANDS = frozenset([
    "open", "close", "kick", "fight", "fire", "zap",
    "eat", "quaff", "wear", "wield", "read", "apply", "puton", "remove",
    "takeoff", "dip", "rub", "invoke", "quiver", "throw", "offer", "drop",
    "force", "loot", "cast", "pickup",
])
_ALL_DIRECTIONS = frozenset([
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
])


def repair_combined_action(raw_action):
    stripped = raw_action.strip()
    parts = stripped.split(None, 1)
    if len(parts) < 2:
        return stripped, False
    cmd = parts[0].lower()
    if cmd == "far" and parts[1].strip().lower() in _ALL_DIRECTIONS:
        return stripped, False
    if cmd in _TWO_STEP_COMMANDS:
        return cmd, True
    return stripped, False


STRATEGY_INIT = """\
Memory Match strategy (guidance, not state):
- The board tracker is the source of truth for card positions and pictures. It updates automatically from every observation and is mirrored in board_state.txt (per cell), match_index.txt (picture token -> cells) and round_log.txt (one line per completed round: the two flips and MATCH / NOMATCH / INVALID). Do not duplicate that information in other files.
- Play order: (1) while a "ready pair" (two known face-down cards with the same picture) exists, flip them; (2) otherwise flip two unknown cards to learn pictures; (3) use the picture tokens in match_index.txt to find partners, never guess.
- A wrong-guess round costs 2 flips and scores nothing; a known pair costs 2 flips and scores 1. Always take a known pair first.
- Never flip a blank (removed) slot and never flip the round's first card twice: both waste a flip without changing the board.
- If the tracker ever contradicts the observation (text or image), correct it with SET_CELL_STATE or CLEAR_FINGERPRINT; optionally record one line in corrections.txt.
"""

LOG_SYSTEM = """\
You are playing a Memory Match card game and maintaining your external memory.

{action_description}
{mismatch_note}

ENVIRONMENT RESPONSE TO THAT ACTION (current observation):
{observation}

BOARD STATE (auto-tracked from this episode's images and text):
{board_block}

RECENT ROUND OUTCOMES (auto-recorded in round_log.txt):
{round_log}

YOUR MEMORY FILES:
{file_index}

The tracker has ALREADY recorded this outcome: board_state.txt, match_index.txt and round_log.txt are updated automatically. Do NOT re-log coordinates, matches, scores or flip counts, and do NOT APPEND to notes.txt.

LOG gets exactly ONE model request this step: your single command is executed and no follow-up request is made. If the tracker agrees with the observation (the usual case), reply with <|DONE|> immediately. If the tracker visibly contradicts the observation text or the image, issue exactly one correction command (SET_CELL_STATE or CLEAR_FINGERPRINT, with JSON arguments) and nothing else.

Available commands:
  <|READ|>filename.txt<|END|>
  <|SEARCH|>filename.txt|keyword<|END|>
  <|TAIL|>filename.txt|N<|END|>
  <|COUNT|>filename.txt|keyword<|END|>
  <|APPEND|>corrections.txt|one short line<|END|>
  plus the correction tools listed for this phase (JSON arguments),
  or <|DONE|> when nothing needs correcting."""

LOG_EXCEPTION_SYSTEM = """\
You are playing a Memory Match card game and maintaining your external memory.

{action_description}
{mismatch_note}

CONTRADICTION DETECTED (the tracker's model and this episode's observation do not agree):
{exceptions}

ENVIRONMENT RESPONSE TO THAT ACTION (current observation):
{observation}

BOARD STATE (auto-tracked from this episode's images and text):
{board_block}

RECENT ROUND OUTCOMES (auto-recorded in round_log.txt):
{round_log}

YOUR MEMORY FILES:
{file_index}

The tracker has ALREADY recorded this outcome: board_state.txt, match_index.txt and round_log.txt are updated automatically. Do NOT re-log coordinates, matches, scores or flip counts.

LOG gets exactly ONE model request this step: your single command is executed and no follow-up request is made, and file reads (READ/SEARCH/TAIL/COUNT) will not show you their result before this step ends. If the board block above already matches the observation (the tracker's own update resolved the contradiction), reply with <|DONE|> immediately. If the observation proves a tracker cell wrong, issue exactly one correction command (SET_CELL_STATE or CLEAR_FINGERPRINT, with a JSON object argument) and nothing else.

Available commands:
  <|APPEND|>corrections.txt|one short line<|END|>
  plus the correction tools listed for this phase (JSON arguments),
  or <|DONE|> when nothing needs correcting."""

LOG_CHECKPOINT_SYSTEM = """\
You are playing a Memory Match card game and maintaining your external memory.

PERIODIC MEMORY REVIEW (checkpoint {checkpoint_no}): the tracker auto-records every flip, match and score, so do NOT re-log that information.

{action_description}
{mismatch_note}

ENVIRONMENT RESPONSE TO THAT ACTION (current observation):
{observation}

BOARD STATE (auto-tracked from this episode's images and text):
{board_block}

RECENT ROUND OUTCOMES (auto-recorded in round_log.txt):
{round_log}

YOUR MEMORY FILES:
{file_index}

LOG gets exactly ONE model request this step: your single command is executed and no follow-up request is made, and file reads (READ/SEARCH/TAIL/COUNT) will not show you their result before this step ends. Decide exactly one thing:
- if the board block above visibly contradicts the observation text or the image, issue exactly one correction command (SET_CELL_STATE or CLEAR_FINGERPRINT, with a JSON object argument) and nothing else;
- otherwise, if there is a durable fact the tracker does NOT capture, APPEND at most one short line to notes.txt (never tracker data, never more than one line);
- otherwise reply with <|DONE|> immediately.

Available commands:
  <|APPEND|>notes.txt|one short line<|END|>
  <|APPEND|>corrections.txt|one short line<|END|>
  plus the correction tools listed for this phase (JSON arguments),
  or <|DONE|> when nothing needs correcting."""

PLAN_SYSTEM = """\
{instruction_prompt}

You have an external memory system. A code-side tracker watches this episode's board image and text and maintains an up-to-date model of every cell; it is shown below and mirrored in your memory files board_state.txt (per cell), match_index.txt (picture token -> cells) and round_log.txt (recent round outcomes).

{board_block}

{stuckness_warning}

YOUR MEMORY FILES:
{file_index}

CURRENT OBSERVATION:
{observation}

DECISION RULE - follow it exactly:
1. The line "SUGGESTED NEXT FLIP" in the BOARD STATE block is always a legal move: a guaranteed match when the tracker knows a pair, otherwise the best exploration flip.
2. Commit it unchanged with <|ACTION|>coordinate<|END|>.
3. Override it ONLY if your memory files prove the suggested target is wrong (e.g. the slot is removed, or the round's first card has a known partner elsewhere). Before overriding, verify with SEARCH on match_index.txt (token -> cells) or TAIL on round_log.txt. Then flip the partner from the "ready pairs" line, or an unknown slot.
4. NEVER flip a blank (removed) slot or the card already flipped this round: both waste a flip. The tracker never suggests them.

If you must fix the tracker before committing, issue SET_CELL_STATE or CLEAR_FINGERPRINT (JSON arguments) first; the correction applies immediately and is visible in the next board state.
Available commands:
  <|READ|>filename.txt<|END|>
  <|SEARCH|>filename.txt|keyword<|END|>
  <|TAIL|>filename.txt|N<|END|>
  <|COUNT|>filename.txt|keyword<|END|>
  plus the tools listed for this phase (e.g. SET_CELL_STATE, CLEAR_FINGERPRINT and the visual-memory tools),
  and <|ACTION|>coordinate<|END|> to commit.
Issue exactly one command per reply. Commit your flip with one ACTION command when ready."""

PLAN_CONTINUE = """\
Command result:
{results}

Now commit your flip with exactly one ACTION command. Prefer the SUGGESTED NEXT FLIP from the board state unless the result above proves it wrong."""

_OPEN = {
    "READ":   r"<\|?READ\|?>",
    "SEARCH": r"<\|?SEARCH\|?>",
    "TAIL":   r"<\|?TAIL\|?>",
    "COUNT":  r"<\|?COUNT\|?>",
    "APPEND": r"<\|?APPEND\|?>",
    "WRITE":  r"<\|?WRITE\|?>",
    "CREATE": r"<\|?CREATE\|?>",
    "ACTION": r"<\|?ACTION\|?>",
    "DONE":   r"<\|?DONE\|?>",
}
_END = r"<\|?END\|?[>}]"

_ACTION_PATTERNS = [
    re.compile(r'<\|?ACTION\|?>(.*?)<\|?END\|?[>}]', re.DOTALL | re.IGNORECASE),
    re.compile(r'<ACTION>(.*?)</ACTION>', re.DOTALL | re.IGNORECASE),
    re.compile(r'(?:^|\n)\s*ACTION:\s*([^\n]{1,40})\s*(?:\n|$)', re.IGNORECASE),
]


class MemoryFileSystem:
    def __init__(self, root_dir, initial_files=None, max_search_results=MAX_SEARCH_RESULTS):
        self.root_dir = root_dir
        self.max_search_results = max_search_results
        os.makedirs(self.root_dir, exist_ok=True)
        if initial_files:
            for name, content in initial_files.items():
                self._write_raw(self._safe_name(name), str(content))

    def _safe_name(self, filename):
        name = os.path.basename(filename.replace("\\", "/"))
        name = name.replace("..", "")
        if not name:
            name = "unnamed.txt"
        return name

    def _path(self, filename):
        return os.path.join(self.root_dir, filename)

    def _exists(self, filename):
        return os.path.isfile(self._path(filename))

    def _read_raw(self, filename):
        path = self._path(filename)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _write_raw(self, filename, content):
        with open(self._path(filename), "w", encoding="utf-8") as f:
            f.write(content)

    def _list_files(self):
        if not os.path.isdir(self.root_dir):
            return []
        return sorted(
            f for f in os.listdir(self.root_dir)
            if os.path.isfile(os.path.join(self.root_dir, f))
        )

    def read(self, filename):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[FILE IS EMPTY]"
        return content

    def search(self, filename, keyword):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[FILE IS EMPTY — no matches]"
        matches = [
            line for line in content.splitlines()
            if keyword.lower() in line.lower()
        ]
        if not matches:
            return "[No lines matching '{}']".format(keyword)

        total = len(matches)
        if total > self.max_search_results:
            matches = matches[-self.max_search_results:]
            return "[Showing last {} of {} matches for '{}']\n{}".format(
                self.max_search_results, total, keyword, "\n".join(matches),
            )
        return "\n".join(matches)

    def tail(self, filename, n=10):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[FILE IS EMPTY]"
        lines = content.splitlines()
        tail_lines = lines[-n:]
        return "\n".join(tail_lines)

    def count(self, filename, keyword):
        filename = self._safe_name(filename)
        if not self._exists(filename):
            return "[ERROR: file '{}' does not exist]".format(filename)
        content = self._read_raw(filename)
        if not content.strip():
            return "[0 matches for '{}' — file is empty]".format(keyword)
        matches = [
            line for line in content.splitlines()
            if keyword.lower() in line.lower()
        ]
        return "[{} line(s) matching '{}']".format(len(matches), keyword)

    def append(self, filename, content):
        filename = self._safe_name(filename)
        path = self._path(filename)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content.rstrip("\n") + "\n")
        return "[Appended to {}]".format(filename)

    def write(self, filename, content):
        filename = self._safe_name(filename)
        if content.strip():
            self._write_raw(filename, content.rstrip("\n") + "\n")
        else:
            self._write_raw(filename, "")
        return "[Wrote to {}]".format(filename)

    def create(self, filename):
        filename = self._safe_name(filename)
        if self._exists(filename):
            return "[File '{}' already exists]".format(filename)
        self._write_raw(filename, "")
        return "[Created {}]".format(filename)

    def file_index(self):
        files = self._list_files()
        if not files:
            return "  (no files yet)"
        lines = []
        for name in files:
            content = self._read_raw(name) or ""
            n_lines = len(content.splitlines()) if content.strip() else 0
            lines.append("  - {} ({} lines)".format(name, n_lines))
        return "\n".join(lines)

    def snapshot(self):
        result = {}
        for name in self._list_files():
            result[name] = self._read_raw(name) or ""
        return result

    def reset(self, initial_files=None):
        import shutil
        if os.path.isdir(self.root_dir):
            shutil.rmtree(self.root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        if initial_files:
            for name, content in initial_files.items():
                self._write_raw(self._safe_name(name), str(content))


def parse_log_calls(text):
    calls = []
    has_done = bool(re.search(_OPEN["DONE"], text))

    for m in re.finditer(_OPEN["READ"] + r"(.*?)" + _END, text, re.DOTALL):
        fn = m.group(1).strip()
        if fn:
            calls.append(("READ", {"filename": fn}))

    for m in re.finditer(_OPEN["SEARCH"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("SEARCH", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["TAIL"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            try:
                n = int(parts[1].strip())
                calls.append(("TAIL", {"filename": parts[0].strip(), "n": n}))
            except ValueError:
                pass

    for m in re.finditer(_OPEN["COUNT"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("COUNT", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["APPEND"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("APPEND", {
                "filename": parts[0].strip(),
                "content": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["WRITE"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip():
            calls.append(("WRITE", {
                "filename": parts[0].strip(),
                "content": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["CREATE"] + r"(.*?)" + _END, text, re.DOTALL):
        fn = m.group(1).strip()
        if fn:
            calls.append(("CREATE", {"filename": fn}))

    return calls, has_done


def parse_plan_calls(text):
    calls = []
    committed_action = None

    for m in re.finditer(_OPEN["READ"] + r"(.*?)" + _END, text, re.DOTALL):
        fn = m.group(1).strip()
        if fn:
            calls.append(("READ", {"filename": fn}))

    for m in re.finditer(_OPEN["SEARCH"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("SEARCH", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for m in re.finditer(_OPEN["TAIL"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            try:
                n = int(parts[1].strip())
                calls.append(("TAIL", {"filename": parts[0].strip(), "n": n}))
            except ValueError:
                pass

    for m in re.finditer(_OPEN["COUNT"] + r"(.*?)" + _END, text, re.DOTALL):
        parts = m.group(1).split("|", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            calls.append(("COUNT", {
                "filename": parts[0].strip(),
                "query": parts[1].strip(),
            }))

    for pattern in _ACTION_PATTERNS:
        m = pattern.search(text)
        if m:
            candidate = m.group(1).strip()
            if candidate and len(candidate) < 60:
                committed_action = candidate
                break

    return calls, committed_action


class MemoryAgent(BaseAgent):
    """Memory Match agent with a code-side board tracker.

    The tracker is automatic computation over this episode's public
    observations (text + current frame): it parses, indexes, cross-checks and
    persists a per-cell board model.  It never chooses the action.  The inner
    LLM keeps explicit memory tool calls for reading, writing, searching,
    querying and correcting, and commits every action through the ACTION
    command.  A legality check at the commit boundary replaces only commits
    the tracker can prove waste a flip (removed slot, round's first card,
    or not a board coordinate at all).
    """

    def __init__(self, client_factory, prompt_builder, config):
        super().__init__(client_factory, prompt_builder)
        self.config = config
        self.max_log_iterations = OmegaConf.select(
            config, "agent.max_log_iterations", default=MAX_LOG_ITERATIONS,
        )
        self.max_validate_iterations = OmegaConf.select(
            config, "agent.max_validate_iterations", default=MAX_VALIDATE_ITERATIONS,
        )
        self.max_search_results = OmegaConf.select(
            config, "agent.max_search_results", default=MAX_SEARCH_RESULTS,
        )
        self.fallback_action_override = OmegaConf.select(
            config, "agent.fallback_action", default=None,
        )
        base_dir = OmegaConf.select(config, "agent.memory_dir", default=None)
        self._memory_base = base_dir or os.path.join(os.getcwd(), "balrog_memory")
        self._memory_dir = self._memory_base
        self._env_name = None
        self.memory = None
        self.board = None
        self.current_image = None
        self.current_frame_id = None
        self.visual_request = None
        self.visual_observation_available = False
        self.step_count = 0
        self.last_step_debug = {}
        self._prev_obs_text = None
        self._unchanged_count = 0
        logger.info("Memory agent files at: %s", self._memory_dir)

    def _get_initial_files(self):
        return {
            "strategy.txt": STRATEGY_INIT,
            "notes.txt": "",
            "corrections.txt": "",
        }

    def _get_spatial_guidance(self):
        return ""

    def _get_instruction_prompt_override(self):
        return None

    def _get_fallback_action(self):
        if self.fallback_action_override is not None:
            return self.fallback_action_override
        return FALLBACK_ACTION_DEFAULT

    def configure_memory(self, env_name, task, episode_idx=0):
        self._env_name = env_name
        self._memory_dir = os.path.join(
            self._memory_base, env_name, task, "ep_{:02d}".format(episode_idx),
        )
        self.memory = MemoryFileSystem(
            self._memory_dir, initial_files=self._get_initial_files(),
            max_search_results=self.max_search_results,
        )
        self.board = BoardTracker(
            os.path.join(self._memory_dir, "board"), self._memory_dir,
        )

    def reset(self):
        self.prompt_builder.reset()
        self.step_count = 0
        self.last_step_debug = {}
        self._prev_obs_text = None
        self._unchanged_count = 0
        if self.memory is not None:
            self.memory.reset(initial_files=self._get_initial_files())
        if self.board is not None:
            self.board.reset()

    # ------------------------------------------------------------------ board
    def _board_block(self):
        if self.board is None:
            return "(board tracker not initialised yet)"
        try:
            return self.board.status_block()
        except Exception as exc:
            logger.warning("board status_block failed: %s", exc)
            return "(board tracker error: %s)" % exc

    def _update_board(self, obs_text):
        if self.board is None:
            return
        try:
            # Two-phase invariant: the observation arriving at step t is the
            # environment's response to step t-1's action, so incorporating it
            # here records only already-observed outcomes.  At step 0 there is
            # no previous action: only geometry and the card-back template are
            # initialised, no outcome is invented.
            self.board.update(obs_text, self.current_image, self.step_count)
        except Exception as exc:
            logger.warning("board tracker update failed: %s", exc)

    def _sanity_check_action(self, action):
        """Validate the model-committed action against the proven board state.

        Returns (action_to_use, was_replaced, reason).  Replaces only commits
        that would provably waste a flip or fail: not a board coordinate, the
        round's first card, or a removed slot.  Everything else passes through
        untouched — the LLM's choice is final.
        """
        if self.board is None:
            return action, False, ""
        coord = str(action).strip()
        try:
            if not self.board._valid(coord):
                return self._safe_alternative(coord, "not a board coordinate")
            if self.board.round_first is not None and coord == self.board.round_first:
                return self._safe_alternative(coord, "already flipped this round")
            if self.board._state_of(coord) == "removed":
                return self._safe_alternative(coord, "slot is removed")
        except Exception as exc:
            logger.warning("sanity check failed: %s", exc)
        return action, False, ""

    def _safe_alternative(self, bad_coord, reason):
        try:
            coord, why = self.board.suggest()
        except Exception as exc:
            logger.warning("suggest failed: %s", exc)
            return self._get_fallback_action(), True, reason + "; suggest failed"
        if coord:
            return coord, True, "%s; switched to %s (%s)" % (reason, coord, why)
        return self._get_fallback_action(), True, reason + "; no alternative"

    # ------------------------------------------------------------------ act
    def act(self, obs, prev_action=None, prev_validated_action=None):
        obs_text = self._format_obs(obs)
        self._update_board(obs_text)
        instruction_prompt = self.prompt_builder.system_prompt or ""
        board_block = self._board_block()

        diff_notice = ""
        if self._prev_obs_text is not None:
            if obs_text.strip() == self._prev_obs_text.strip():
                self._unchanged_count += 1
                if self._unchanged_count >= 3:
                    diff_notice = ("WARNING: the text observation has been UNCHANGED for {} "
                                   "consecutive actions; the tracker shows the current state "
                                   "below — rely on it, not on the repeated text.").format(
                                       self._unchanged_count)
            else:
                self._unchanged_count = 0

        total_in = 0
        total_out = 0

        log_debug = None
        if prev_action is not None and self.step_count > 0:
            log_debug = self._log_phase(
                obs_text, prev_action, prev_validated_action, diff_notice, board_block,
            )
            total_in += log_debug["input_tokens"]
            total_out += log_debug["output_tokens"]

        plan_debug = self._plan_validate_phase(
            obs_text, instruction_prompt, diff_notice, board_block,
            prev_validated_action=prev_validated_action,
        )
        total_in += plan_debug["input_tokens"]
        total_out += plan_debug["output_tokens"]

        final_action = plan_debug["final_action"]
        repaired_action, was_repaired = repair_combined_action(final_action)
        if was_repaired:
            final_action = repaired_action

        final_action, replaced, replace_reason = self._sanity_check_action(final_action)

        self.last_step_debug = {
            "step": self.step_count,
            "log_phase": log_debug,
            "plan_phase": plan_debug,
            "safety_net": {"replaced": replaced, "reason": replace_reason},
            "memory_snapshot": self.memory.snapshot(),
            "diff_notice": diff_notice,
            "unchanged_count": self._unchanged_count,
        }

        final = LLMResponse(
            model_id=plan_debug.get("model_id", "unknown"),
            completion=final_action,
            stop_reason="stop",
            input_tokens=total_in,
            output_tokens=total_out,
            reasoning=None,
        )

        self._prev_obs_text = obs_text
        self.step_count += 1
        return final

    def _format_obs(self, obs):
        parts = []
        long_ctx = obs.get("text", {}).get("long_term_context", "")
        short_ctx = obs.get("text", {}).get("short_term_context", "")
        if long_ctx:
            parts.append(long_ctx)
        if short_ctx:
            parts.append(short_ctx)
        return "\n".join(parts) if parts else "(no observation)"

    # ------------------------------------------------------------------ exec
    def _execute_read_op(self, op, args):
        if op == "READ":
            return self.memory.read(args["filename"])
        elif op == "SEARCH":
            return self.memory.search(args["filename"], args["query"])
        elif op == "TAIL":
            return self.memory.tail(args["filename"], args.get("n", 10))
        elif op == "COUNT":
            return self.memory.count(args["filename"], args["query"])
        return "[Unknown read op: {}]".format(op)

    def _execute_write_op(self, op, args):
        if op == "READ":
            return self.memory.read(args["filename"])
        elif op == "SEARCH":
            return self.memory.search(args["filename"], args["query"])
        elif op == "TAIL":
            return self.memory.tail(args["filename"], args.get("n", 10))
        elif op == "COUNT":
            return self.memory.count(args["filename"], args["query"])
        elif op == "APPEND":
            return self.memory.append(args["filename"], args["content"])
        elif op == "WRITE":
            return self.memory.write(args["filename"], args["content"])
        elif op == "CREATE":
            return self.memory.create(args["filename"])
        return "[Unknown op: {}]".format(op)

    def _format_tool_result(self, op, args, result):
        if op == "READ":
            header = "READ {}".format(args["filename"])
        elif op == "SEARCH":
            header = "SEARCH {} for '{}'".format(args["filename"], args["query"])
        elif op == "TAIL":
            header = "TAIL {} (last {} lines)".format(args["filename"], args.get("n", 10))
        elif op == "COUNT":
            header = "COUNT {} for '{}'".format(args["filename"], args["query"])
        elif op == "APPEND":
            header = "APPEND to {}".format(args["filename"])
            return header, "── {} ──\n{}".format(header, result)
        elif op == "WRITE":
            header = "WRITE to {}".format(args["filename"])
            return header, "── {} ──\n{}".format(header, result)
        elif op == "CREATE":
            header = "CREATE {}".format(args["filename"])
            return header, "── {} ──\n{}".format(header, result)
        else:
            return result, result
        return header, "── {} ──\n{}".format(header, result)

    # ------------------------------------------------------------------ LOG
    def _log_phase(self, obs_text, prev_action, prev_validated_action,
                   diff_notice, board_block):
        prev_step = self.step_count - 1
        file_index = self.memory.file_index()

        mismatch_note = ""
        if prev_validated_action is not None and prev_validated_action != prev_action:
            mismatch_note = (
                "\nNOTE: your raw output was {prev!r}; the host executed the "
                "validated action {validated!r} instead."
            ).format(prev=prev_action, validated=prev_validated_action)
        effective = prev_validated_action or prev_action
        action_description = (
            "At the PREVIOUS step (step {prev_step}) you took action: \"{effective}\""
        ).format(prev_step=prev_step, effective=effective)

        # LOG by exception + checkpoint cadence: the tracker auto-records
        # every outcome, so the model gets a LOG decision point only when
        # something actually needs LLM judgment (a contradiction between the
        # tracker's model and this episode's observation) or at a periodic
        # checkpoint, which is also the only place the LLM can still write to
        # its memory files (APPEND/WRITE/CREATE are LOG-only tools).
        exceptions = []
        if self.board is not None:
            try:
                exceptions = self.board.log_exceptions(
                    self.step_count, obs_text,
                    prev_action, prev_validated_action,
                )
            except Exception as exc:
                logger.warning("log_exceptions failed: %s", exc)
        checkpoint_due = (self.step_count >= 1 and
                          (self.step_count - 1) % LOG_CHECKPOINT_EVERY == 0)
        if not exceptions and not checkpoint_due:
            if self.board is not None:
                try:
                    self.board.mark_warnings_reviewed()
                except Exception:
                    pass
            return {
                "skipped": "no exception, no checkpoint due (step %d)"
                           % self.step_count,
                "fired": None,
                "exceptions": [],
                "initial_prompt": None,
                "turns": [],
                "input_tokens": 0,
                "output_tokens": 0,
            }

        round_log = (self.board.round_log_tail(8) if self.board is not None
                     else "(board tracker not initialised yet)")
        if exceptions:
            template = LOG_EXCEPTION_SYSTEM
            extra = {"exceptions": "\n".join("- " + e for e in exceptions)}
            fired = "exception"
        else:
            template = LOG_CHECKPOINT_SYSTEM
            extra = {"checkpoint_no": 1 + (self.step_count - 1) // LOG_CHECKPOINT_EVERY}
            fired = "checkpoint"

        initial_prompt = template.format(
            action_description=action_description,
            mismatch_note=mismatch_note,
            observation=obs_text,
            board_block=board_block,
            round_log=round_log,
            file_index=file_index,
            **extra,
        )
        messages = [Message(role="user", content=initial_prompt)]

        total_in = 0
        total_out = 0
        turns = []

        for iteration in range(LOG_PHASE_REQUESTS):
            current_prompt = messages[-1].content

            try:
                response = self.client.generate(messages)
            except Exception as e:
                logger.error("Log phase LLM call failed: %s", e)
                turns.append({
                    "prompt": current_prompt,
                    "raw_response": "[ERROR: {}]".format(e),
                    "tool_results": "",
                })
                break

            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, has_done = parse_log_calls(assistant_text)

            result_parts = []
            for op, args in calls:
                result = self._execute_write_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)

            results_text = "\n\n".join(result_parts) if result_parts else ""

            turns.append({
                "prompt": current_prompt,
                "raw_response": assistant_text,
                "tool_results": results_text,
            })

            messages.append(Message(role="assistant", content=assistant_text))

            if has_done:
                break
            if not calls:
                break

            # Single-shot LOG: the command above has already been executed;
            # its effect is visible in the next step's board state.  No
            # follow-up request is made (LOG_PHASE_REQUESTS == 1 makes the
            # loop end here anyway).

        if self.board is not None:
            try:
                self.board.mark_warnings_reviewed()
            except Exception:
                pass
        return {
            "skipped": None,
            "fired": fired,
            "exceptions": exceptions,
            "initial_prompt": initial_prompt,
            "turns": turns,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

    # ------------------------------------------------------------------ PLAN
    def _plan_validate_phase(self, obs_text, instruction_prompt,
                             diff_notice, board_block,
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

        initial_prompt = PLAN_SYSTEM.format(
            instruction_prompt=instruction_prompt,
            board_block=board_block,
            stuckness_warning=stuckness_warning,
            file_index=file_index,
            observation=obs_text,
        )

        messages = [Message(role="user", content=initial_prompt)]

        total_in = 0
        total_out = 0
        turns = []
        final_action = None
        model_id = "unknown"

        for iteration in range(self.max_validate_iterations):
            current_prompt = messages[-1].content

            try:
                response = self.client.generate(messages)
            except Exception as e:
                logger.error("Plan phase LLM call failed: %s", e)
                turns.append({
                    "prompt": current_prompt,
                    "raw_response": "[ERROR: {}]".format(e),
                    "tool_results": "",
                    "committed_action": None,
                })
                break

            model_id = response.model_id
            total_in += response.input_tokens
            total_out += response.output_tokens
            assistant_text = response.completion

            calls, committed_action = parse_plan_calls(assistant_text)

            result_parts = []
            for op, args in calls:
                result = self._execute_read_op(op, args)
                _header, formatted = self._format_tool_result(op, args, result)
                result_parts.append(formatted)

            results_text = "\n\n".join(result_parts) if result_parts else ""

            turns.append({
                "prompt": current_prompt,
                "raw_response": assistant_text,
                "tool_results": results_text,
                "committed_action": committed_action,
            })

            messages.append(Message(role="assistant", content=assistant_text))

            if committed_action is not None:
                final_action = committed_action
                break

            if not calls:
                logger.warning(
                    "Plan phase: no SEARCH/READ/TAIL/COUNT/ACTION found. Raw: %s",
                    assistant_text[:200],
                )
                final_action = self._get_fallback_action()
                break

            continue_msg = PLAN_CONTINUE.format(results=results_text)
            messages.append(Message(role="user", content=continue_msg))

        if final_action is None:
            fallback = self._get_fallback_action()
            final_action = fallback
            logger.warning(
                "Plan phase exhausted %d iterations without <|ACTION|>. Defaulting to '%s'.",
                self.max_validate_iterations, fallback,
            )

        return {
            "initial_prompt": initial_prompt,
            "turns": turns,
            "final_action": final_action,
            "input_tokens": total_in,
            "output_tokens": total_out,
            "model_id": model_id,
        }

    # ------------------------------------------------------------------ state
    def export_state(self):
        return {
            "board": self.board.export_state() if self.board is not None else None,
            "step_count": self.step_count,
            "_prev_obs_text": self._prev_obs_text,
            "_unchanged_count": self._unchanged_count,
        }

    def import_state(self, state):
        if not isinstance(state, dict):
            return
        if self.board is not None and state.get("board") is not None:
            self.board.import_state(state["board"])
        self.step_count = int(state.get("step_count", self.step_count))
        self._prev_obs_text = state.get("_prev_obs_text")
        self._unchanged_count = int(state.get("_unchanged_count", self._unchanged_count))
