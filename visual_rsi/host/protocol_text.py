"""Host-owned command protocol: the system prefix shown to the model and an independent decoder."""
from __future__ import annotations

import json
import re
from typing import Any

from .child.registry import COMMAND_RE, LEGACY_KEYS, ProtocolError, sample_arguments, validate

OVERRIDE = "These formatting rules override any permission to issue multiple commands in the user prompt."


def _legacy_example(name: str, files: list[str]) -> str:
    filename = files[0] if files else "notes.txt"
    payload = {"READ": filename, "SEARCH": f"{filename}|keyword", "TAIL": f"{filename}|5",
               "COUNT": f"{filename}|keyword", "APPEND": f"{filename}|note text",
               "WRITE": f"{filename}|full new content", "CREATE": "new_notes.txt"}[name]
    return f"<|{name}|>{payload}<|END|>"


def _example(tool: dict[str, Any], files: list[str]) -> str:
    if tool.get("legacy"):
        return _legacy_example(tool["name"], files)
    args = sample_arguments(tool.get("parameters", {}), files)
    return "<|%s|>%s<|END|>" % (tool["name"], json.dumps(args, ensure_ascii=False, separators=(",", ":")))


def _usage(tool: dict[str, Any]) -> str:
    if tool.get("legacy"):
        return "<|%s|>%s<|END|>" % (tool["name"], "|".join(LEGACY_KEYS[tool["name"]]))
    # A concrete JSON object, never a placeholder: in pilot r2 the model copied
    # the old "{JSON arguments}" text literally in 19% of Crafter RECALL_IMAGE calls.
    example = json.dumps(sample_arguments(tool.get("parameters", {})), ensure_ascii=False, separators=(",", ":"))
    return "<|%s|>%s<|END|> (a JSON object like this one, with your own values) parameters=%s" % (
        tool["name"], example, json.dumps(tool.get("parameters", {}), ensure_ascii=False, sort_keys=True))


def _normalize_action_spec(value: list[str] | dict[str, Any]) -> dict[str, Any]:
    return {"kind": "finite", "actions": value, "dynamic": False} if isinstance(value, list) else value


def _action_allowed(payload: str, action_spec: list[str] | dict[str, Any]) -> bool:
    spec = _normalize_action_spec(action_spec)
    if spec.get("kind") == "finite":
        return payload in spec.get("actions", [])
    if spec.get("kind") == "free_text":
        return bool(payload) and len(payload) <= int(spec.get("max_length", 512)) and not any(
            char in payload for char in ("\x00", "\n", "\r"))
    return False


def system_prefix(phase: str, tools: list[dict[str, Any]], actions: list[str] | dict[str, Any], files: list[str]) -> str:
    if phase not in {"LOG", "PLAN"}:
        raise ValueError("unknown phase")
    action_spec = _normalize_action_spec(actions)
    finite = [str(a) for a in action_spec.get("actions", [])]
    free_text_examples = [str(value) for value in action_spec.get("examples", [])]
    action_example = (free_text_examples[0] if action_spec.get("kind") == "free_text" and free_text_examples
                      else finite[0] if finite else "look")
    available = [t for t in tools if phase in t.get("phases", [])]
    legacy_reads = [t for t in available if t.get("legacy") and t["name"] == "READ"]
    examples = [_example(t, files) for t in legacy_reads]
    if phase == "LOG":
        examples += [_example(t, files) for t in available if t.get("legacy") and t["name"] == "APPEND"]
        extra = [t for t in available if not t.get("legacy")]
        if extra:
            examples.append(_example(extra[0], files))
    else:
        if action_spec.get("kind") == "free_text" and free_text_examples:
            examples.extend(f"<|ACTION|>{value}<|END|>" for value in free_text_examples)
        else:
            examples.append(f"<|ACTION|>{action_example}<|END|>")
    lines = [
        f"PHASE: {phase}",
        "You are using a plain-text command interface, not native function calling.",
        "Reply with exactly ONE command, then stop. No analysis, Markdown or extra text.",
        "For a command, use exactly one opening command marker such as <|APPEND|> and exactly one <|END|> closing marker.",
        "A reply containing two commands is rejected even if each command is individually valid; never batch APPEND entries.",
        "Write a command as <|NAME|>arguments<|END|>, where NAME is one of the command names listed for this phase. In LOG only, <|DONE|> alone is the non-command completion form and has no <|END|>.",
        "NAME is a placeholder: never write <|NAME|> or <|COMMAND|> literally.",
        "Examples valid in this phase:",
        *examples,
    ]
    if phase == "LOG":
        lines.append("When logging is finished, reply with <|DONE|> alone.")
    else:
        if action_spec.get("kind") == "free_text":
            lines.append("Commit exactly one parser command with <|ACTION|>command text<|END|>.")
        else:
            lines.append("Commit exactly one action with <|ACTION|>action<|END|>, using a canonical legal action.")
    lines += [
        "After a command, wait for its result before choosing another. Never invent tool results.",
        OVERRIDE,
        "Commands available in this phase:",
        *(f"- {_usage(t)}: {t.get('description', '')}" for t in available),
    ]
    if phase == "PLAN":
        lines.append("Memory search is optional; use it when it helps decide the next action.")
        if any(t.get("permission") == "memory_write" for t in available):
            lines.append("The tool list above defines what is available in PLAN, including visual memory writes. Saved labels and notes are unverified; use recalled pixels to check them.")
    if action_spec.get("kind") == "free_text":
        lines.append("Action contract: free text, UTF-8, one line, maximum %d characters. %s" % (
            int(action_spec.get("max_length", 512)), action_spec.get("description", "")))
    else:
        lines.append("Canonical legal actions: " + ", ".join(finite))
    return "\n".join(lines)


def host_decode(text: str, phase: str, tools: list[dict[str, Any]], actions: list[str] | dict[str, Any]):
    """Return (name, payload) for a well-formed single command in this phase, else None."""
    if text.strip() == "<|DONE|>":
        return ("DONE", "") if phase == "LOG" else None
    matches = list(COMMAND_RE.finditer(text))
    if len(matches) != 1:
        return None
    name, payload = matches[0].group(1), matches[0].group(2).strip()
    if "<|" in payload:
        return None
    if name == "ACTION":
        return (name, payload) if phase == "PLAN" and _action_allowed(payload, actions) else None
    tool = next((t for t in tools if t.get("name") == name and phase in t.get("phases", [])), None)
    if tool is None:
        return None
    try:
        if tool.get("legacy"):
            keys = LEGACY_KEYS[name]
            parts = payload.split("|", len(keys) - 1)
            if len(parts) != len(keys):
                return None
            args = dict(zip(keys, parts))
            if name == "TAIL":
                args["n"] = int(args["n"])
        else:
            args = json.loads(payload)
            if not isinstance(args, dict):
                return None
        validate(args, tool.get("parameters", {}))
        return name, payload
    except (ValueError, TypeError, ProtocolError):
        return None


def memory_filenames(files: Any) -> list[str]:
    if isinstance(files, list):
        return [str(f) for f in files]
    return re.findall(r"-\s+(\S+)\s+\(", str(files or ""))
