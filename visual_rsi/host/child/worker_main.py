"""Unprivileged candidate worker. Only JSON DTOs cross stdio; it never sees the world or a key.

Ported from the r2 C01/C05 workers. The candidate's own ``agent.act``,
``_log_phase`` and ``_plan_validate_phase`` run unchanged. The runtime only
replaces the module-level command parsers, the per-instance tool dispatchers
and the result formatter with the host registry, and wraps (never replaces)
the two phase methods to know which phase is active. The host enforces every
limit; nothing here is a security boundary.

This file is copied next to ``registry.py`` into a per-worker runtime
directory and must not import anything from the repository.
"""
import base64
import importlib
import importlib.util
import io
import json
from collections import namedtuple
from pathlib import Path
import resource
import sys
import traceback
import types

# Resolve host dependencies while only the copied host runtime and the venv
# are on sys.path.  In particular, image decoding must never pick up a PIL or
# OmegaConf lookalike from the candidate bundle.
import numpy  # noqa: F401
import PIL  # noqa: F401
from PIL import Image, PngImagePlugin  # noqa: F401
from omegaconf import OmegaConf

MAX_MESSAGE = 64 * 1024 ** 2  # must equal host/worker_process.py
WIRE_OUT = sys.stdout
WIRE_IN = sys.stdin
EXCLUDED_STATE = {"client", "config", "prompt_builder", "memory", "visual", "current_image",
                  "current_frame_id", "visual_request", "visual_observation_available",
                  "_memory_dir", "_memory_base"}


def _visual_prompt_compat(content, line):
    """Align frozen upstream text with the visual host protocol before model delivery."""
    if line not in {"T", "V", "M"}:
        return content
    replacements = (
        ("Continue logging (remember: SEARCH before writing to avoid duplicates)",
         "Continue logging (check existing notes when needed to avoid duplicates)"),
        ("Think step by step: propose -> search -> validate -> commit or revise.",
         "Choose the next action; consult memory when it helps resolve uncertainty."),
        ("revise your plan, search for the new action, then commit.",
         "revise your plan, consult memory if useful, then commit."),
        ("REVISE your plan and search again for the new candidate.",
         "REVISE your plan; search again only if it would resolve remaining uncertainty."),
        ("Before committing to an action, you should search your memory for relevant information.",
         "Use relevant memory when it can inform the next action."),
        ("4. BEFORE committing, SEARCH your memory for information related to your planned action, current location, or situation. For example:",
         "4. Search relevant memory when it can resolve uncertainty about your next action. For example:"),
        ("1. BEFORE appending or writing to ANY file, first SEARCH or READ it to check whether the information is already recorded. Do NOT duplicate entries.",
         "1. Check a file before writing when you are unsure whether the information is already recorded. Avoid duplicate entries."),
        ("You may issue multiple commands per turn. Finish with <|DONE|>.",
         "Issue one command per reply. You may use several commands across replies; finish with <|DONE|>."),
        ("UNCHANGED observations (the action had NO EFFECT — this is important!)",
         "Unchanged observations (the action may have had no visible effect)"),
        ("SPATIAL AWARENESS: Pay careful attention to spatial descriptions and any map or grid in the observation. Log positions and landmarks in a structured format so you can navigate effectively.",
         "SPATIAL AWARENESS: Record visible landmarks and positions only when the observation supports them."),
    )
    for old, new in replacements:
        content = content.replace(old, new)
    if line == "M":
        content = content.replace(
            "AVAILABLE COMMANDS (read-only — no writing in this phase):",
            "TEXT MEMORY COMMANDS (visual tools are listed in the system message):")
        content = content.replace(
            "You can issue more SEARCH/READ/TAIL/COUNT commands, or commit with <|ACTION|>.",
            "You can use any tool listed in the system message, or commit with <|ACTION|>.")
    return content


class HostStop(BaseException):
    """Raised when the host ends the decision; BaseException so upstream `except Exception` cannot swallow it."""


# Upstream's inner scaffold keys hand-written per-game content (initial knowledge
# files, spatial guidance, system prompts, NLE map analysis) on these names.  The
# study's G0 is the generic scaffold: game knowledge is never supplied by the
# researchers, so a colliding adapter name is passed to the agent neutralised.
UPSTREAM_KNOWLEDGE_KEYS = frozenset({"crafter", "nle", "minihack"})


def knowledge_neutral_env_name(adapter_name: str) -> str:
    return adapter_name + "_generic" if adapter_name in UPSTREAM_KNOWLEDGE_KEYS else adapter_name


def send(value):
    WIRE_OUT.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    WIRE_OUT.flush()


def receive():
    line = WIRE_IN.readline(MAX_MESSAGE + 1)
    if not line or len(line) > MAX_MESSAGE:
        raise SystemExit("invalid host message")
    return json.loads(line)


def _jsonable(value):
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (TypeError, ValueError):
        return False


def _decode_png(value):
    if not value:
        return None
    return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")


def preload_prompt_builder(bundle):
    """Load the evolving scaffold's real prompt_builder from its exact path."""
    package = bundle / "prompt_builder"
    spec = importlib.util.spec_from_file_location(
        "prompt_builder", package / "__init__.py", submodule_search_locations=[str(package)])
    if spec is None or spec.loader is None:
        raise ImportError("candidate prompt_builder package is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules["prompt_builder"] = module
    spec.loader.exec_module(module)
    return sys.modules["prompt_builder.history"].HistoryPromptBuilder


def _exit_with_parent() -> None:
    """Exit if the host dies, even while candidate code is stuck in a loop.

    The worker runs in its own session, so killing the host's process group
    does not reach it; without this an infinite loop would outlive the run.
    """
    import os
    import threading
    import time
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=watch, daemon=True).start()


def main():
    bundle = Path(sys.argv[1]).resolve()
    scratch = Path(sys.argv[2]).resolve()
    runtime = Path(__file__).resolve().parent
    # No CPU-time rlimit (removed 2026-09-26; it capped a whole episode at 600 s
    # of candidate CPU).  The host's per-wait backstop ends a hung candidate.
    _exit_with_parent()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 ** 2, 256 * 1024 ** 2))  # 16 MB -> 256 MB, 2026-09-26
    sys.stdout = sys.stderr  # candidate print() cannot corrupt the JSON channel
    init = receive()
    try:
        sys.path.insert(0, str(runtime))
        from registry import ProtocolError, Registry, decode, register_legacy
        HistoryPromptBuilder = preload_prompt_builder(bundle)
        sys.path.insert(0, str(bundle))
        client = types.ModuleType("client")
        client.LLMResponse = namedtuple("LLMResponse", "model_id completion stop_reason input_tokens output_tokens reasoning")
        sys.modules["client"] = client
        agents = types.ModuleType("agents")
        agents.__path__ = [str(bundle / "agents")]
        sys.modules["agents"] = agents
        import agents.memory_agent as original

        executed = {}
        if init.get("profile"):
            prefix = str(bundle) + "/"

            def profile_call(frame, kind, arg):
                if kind == "call":
                    filename = frame.f_code.co_filename
                    if filename.startswith(prefix):
                        key = filename[len(prefix):] + ":" + frame.f_code.co_name
                        executed[key] = executed.get(key, 0) + 1
            sys.setprofile(profile_call)

        candidate = importlib.import_module("candidate")
        actions = list(init["actions"])
        action_spec = init.get("action_spec") or {"kind": "finite", "actions": actions}

        # The frozen upstream agent contains an NLE-only compatibility repair
        # that turns commands such as ``open east`` into the first half of a
        # multi-step NetHack command.  A free-text adapter (TextWorld) has a
        # different contract: the model-committed command is already complete
        # and must reach the environment unchanged.  Disable that legacy repair
        # at this protocol boundary without changing the candidate bundle.
        if action_spec.get("kind") == "free_text" and callable(
                getattr(original, "repair_combined_action", None)):
            original.repair_combined_action = lambda raw_action: (str(raw_action).strip(), False)

        state = {"phase": "PLAN"}
        holder = {}

        def event(kind, **fields):
            send(dict(kind="event", event=kind, **fields))

        def memory_files():
            lister = getattr(holder["agent"].memory, "_list_files", None)
            try:
                return list(lister()) if callable(lister) else []
            except Exception:
                return []

        def forward_visual(in_tool):
            """Returns (sent, error): whether a request reached the host, and its rejection reason."""
            agent = holder["agent"]
            program = getattr(agent, "visual_request", None)
            if program is None:
                return False, None
            agent.visual_request = None
            if not _jsonable(program):
                return False, "visual request is not a JSON program"
            send(dict(kind="visual_request", phase=state["phase"], program=program, in_tool=in_tool))
            reply = receive()
            if reply.get("kind") == "stop":
                raise HostStop(reply.get("reason"))
            if reply.get("kind") != "visual_response":
                raise HostStop("invalid visual reply")
            return True, reply.get("error")

        class Client:
            def generate(self, messages):
                forward_visual(False)
                agent = holder["agent"]
                wire = [dict(role=m.role, content=str(m.content)) for m in messages]
                for item in wire:
                    item["content"] = _visual_prompt_compat(item["content"], init.get("line"))
                visual = getattr(agent, "visual", None)
                if wire and callable(getattr(visual, "index", None)):
                    wire[0]["content"] += "\n\n" + visual.index()
                send(dict(kind="model", phase=state["phase"], messages=wire,
                          tools=holder["registry"].specs(), files=memory_files()))
                reply = receive()
                if reply.get("kind") == "stop":
                    raise HostStop(reply.get("reason"))
                if reply.get("kind") != "response":
                    raise HostStop("invalid broker reply")
                value = reply["value"]
                return client.LLMResponse(value.get("model_id", "unknown"), value.get("completion", ""),
                                          value.get("stop_reason", "stop"), int(value.get("input_tokens", 0)),
                                          int(value.get("output_tokens", 0)), None)

        config = OmegaConf.create({"agent": {
            "max_log_iterations": int(init.get("max_log_iterations", 10)),
            "max_validate_iterations": int(init.get("max_validate_iterations", 10)),
            "max_search_results": 1000, "fallback_action": init["fallback_action"],
            "memory_dir": str(scratch / "memory")}})
        # Matches upstream config.yaml (max_text_history: 10).  Upstream MemoryAgent
        # never calls get_prompt(): every LOG/PLAN prompt is built from the current
        # observation and memory files, so this window only matters to a candidate
        # that chooses to read the builder's history.
        builder = HistoryPromptBuilder(system_prompt=init["instruction"], max_image_history=0,
                                       max_text_history=10)
        agent = candidate.Agent(lambda: Client(), builder, config)
        agent.configure_memory(knowledge_neutral_env_name(init["adapter_name"]), "default", 0)
        agent.current_image = None
        agent.current_frame_id = None
        agent.visual_request = None
        registry = Registry(event, file_index=memory_files)
        holder.update(agent=agent, registry=registry)
        if init.get("no_automem"):
            # N0 is a host-defined ablation of LLM-written memory, not a
            # candidate rewrite.  The LOG phase -- the only phase in which the
            # LLM writes -- is suppressed, every write is refused, and the
            # visual recall channel is removed.  The PLAN prompt and its read
            # tools stay identical to G0's and read the same initial files, so
            # N0 differs from G0 only in what the LLM records during play.
            # (Until 2026-09-25 N0 replaced memory with a stub whose reads were
            # unregistered commands, so the unchanged PLAN prompt produced
            # protocol errors on every attempted SEARCH/READ.)
            def refuse(*args, **kwargs):
                return "[N0: memory is read-only in this condition]"
            for name in ("append", "write", "create"):
                if callable(getattr(agent.memory, name, None)):
                    setattr(agent.memory, name, refuse)
            register_legacy(registry, agent.memory, read_only=True)
            agent.visual = None
            agent._log_phase = lambda *args, **kwargs: {"input_tokens": 0, "output_tokens": 0,
                                                         "turns": [], "disabled": True}
        else:
            register_legacy(registry, agent.memory)
        if not init.get("no_automem") and hasattr(candidate, "install"):
            candidate.install(agent, registry)
        if init.get("disable_visual_recall"):
            registry.tools.pop("RECALL_IMAGE", None)

        def correct_example(phase):
            if phase == "PLAN":
                example = actions[0] if actions else init.get("fallback_action", "look")
                return "<|ACTION|>%s<|END|>" % example
            files = memory_files()
            return "<|READ|>%s<|END|>" % files[0] if files else "<|DONE|>"

        def parsed(text, phase):
            try:
                op, args = decode(text, phase, registry, action_spec)
                if op == "DONE":
                    return [], True
                if op == "ACTION":
                    return [], args["action"]
                return [(op, args)], (False if phase == "LOG" else None)
            except ProtocolError as exc:
                return [("ERROR", {"message": str(exc)})], (False if phase == "LOG" else None)

        original.parse_log_calls = lambda text: parsed(text, "LOG")
        original.parse_plan_calls = lambda text: parsed(text, "PLAN")

        def dispatch(self, op, args):
            if op == "ERROR":
                return "[PROTOCOL ERROR: %s]\nCorrect example: %s" % (args["message"], correct_example(state["phase"]))
            result = registry.execute(op, args, state["phase"])
            sent, error = forward_visual(True)
            if error:
                result += "\n[TOOL ERROR: invalid visual request: %s]" % error
                event("visual_tool_error", name=op, error=error)
            elif sent:
                # The pixels travel with the next model request; say so, because
                # the model otherwise describes the image from this metadata alone.
                result += ("\n[The requested image is attached to your next message, labelled as a historical "
                           "image. Do not describe it before you see it.]")
            return result

        agent._execute_read_op = types.MethodType(dispatch, agent)
        agent._execute_write_op = types.MethodType(dispatch, agent)
        formatter = agent._format_tool_result

        def format_result(self, op, args, result):
            tool = registry.tools.get(op)
            if tool is not None and tool.legacy and not str(result).startswith("[TOOL ERROR:"):
                return formatter(op, args, result)
            return op, str(result)

        agent._format_tool_result = types.MethodType(format_result, agent)
        log_phase, plan_phase = agent._log_phase, agent._plan_validate_phase

        def wrapped_log(*args, **kwargs):
            state["phase"] = "LOG"
            return log_phase(*args, **kwargs)

        def wrapped_plan(*args, **kwargs):
            state["phase"] = "PLAN"
            return plan_phase(*args, **kwargs)

        agent._log_phase, agent._plan_validate_phase = wrapped_log, wrapped_plan

        def checkpoint():
            if hasattr(agent, "export_state"):
                value = agent.export_state()
            else:
                value = {k: v for k, v in agent.__dict__.items()
                         if k not in EXCLUDED_STATE and not callable(v) and _jsonable(v)}
            files = {str(p.relative_to(scratch)): base64.b64encode(p.read_bytes()).decode()
                     for p in sorted(scratch.rglob("*")) if p.is_file() and not p.is_symlink()}
            return dict(schema=1, state=value, files=files)

        def restore(data):
            for name, encoded in data.get("files", {}).items():
                path = scratch / name
                if not path.resolve().is_relative_to(scratch):
                    raise ValueError("checkpoint traversal")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(encoded))
            if hasattr(agent, "import_state"):
                agent.import_state(data.get("state", {}))
            else:
                for key, value in data.get("state", {}).items():
                    if key not in EXCLUDED_STATE:
                        setattr(agent, key, value)
            reload_visual = getattr(getattr(agent, "visual", None), "reload", None)
            if callable(reload_visual):
                reload_visual()

        if init.get("checkpoint"):
            restore(init["checkpoint"])
    except BaseException as exc:
        send(dict(kind="crash", error=repr(exc), traceback=traceback.format_exc()))
        return
    send(dict(kind="ready", tools=registry.specs()))
    while True:
        request = receive()
        op = request.get("op")
        if op == "close":
            return
        if op == "checkpoint":
            send(dict(kind="checkpoint", value=checkpoint()))
            continue
        if op != "act":
            send(dict(kind="crash", error="unknown worker operation", traceback=""))
            return
        try:
            agent.step_count = int(request["step"])
            agent.visual_observation_available = bool(request.get("visual_observation_available"))
            # The frozen upstream agent infers "stuck" from text equality.
            # Visual tasks can have constant text and changing pixels; even
            # identical pixels do not prove the previous action had no hidden
            # effect. Disable that text-only inference for visual observations.
            if agent.visual_observation_available:
                if hasattr(agent, "_prev_obs_text"):
                    agent._prev_obs_text = None
                if hasattr(agent, "_unchanged_count"):
                    agent._unchanged_count = 0
            if init.get("line") == "M":
                agent.current_image = _decode_png(request.get("image"))
                agent.current_frame_id = request.get("frame_id")
            else:
                agent.current_image = None
                agent.current_frame_id = None
            response = agent.act(request["observation"], request.get("previous_action"),
                                 request.get("previous_validated_action"))
            forward_visual(False)
            send(dict(kind="decision", action=getattr(response, "completion", str(response)),
                      memory=agent.memory.snapshot(), source_calls=dict(executed)))
        except HostStop:
            return
        except BaseException as exc:
            send(dict(kind="crash", error=repr(exc), traceback=traceback.format_exc()))
            return


if __name__ == "__main__":
    main()
