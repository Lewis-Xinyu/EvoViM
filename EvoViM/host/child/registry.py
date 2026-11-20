"""Dynamic tool registry and strict command decoder.

This module is part of the self-contained child runtime: it is copied next to
the worker and imported by candidates as the plain ``registry`` module. It must
not import anything from the repository.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable

LEGACY_KEYS = {"READ": ["filename"], "SEARCH": ["filename", "query"], "TAIL": ["filename", "n"],
               "COUNT": ["filename", "query"], "WRITE": ["filename", "content"],
               "APPEND": ["filename", "content"], "CREATE": ["filename"]}
COMMAND_RE = re.compile(r"<\|([A-Z][A-Z0-9_]*)\|>(.*?)<\|END\|>", re.S)


class ProtocolError(ValueError):
    pass


def validate(value: Any, schema: dict[str, Any]) -> None:
    kind = schema.get("type")
    types = {"object": dict, "string": str, "integer": int, "number": (int, float),
             "array": list, "boolean": bool, "null": type(None)}
    if kind and (kind not in types or not isinstance(value, types[kind])
                 or kind in ("integer", "number") and isinstance(value, bool)):
        raise ProtocolError(f"parameter type: {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ProtocolError("parameter enum")
    if kind == "object":
        props = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise ProtocolError("missing parameter: " + ",".join(sorted(missing)))
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise ProtocolError("unknown parameter")
        for key, item in value.items():
            if key in props:
                validate(item, props[key])
    if kind == "array":
        for item in value:
            validate(item, schema.get("items", {}))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ProtocolError("below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ProtocolError("above maximum")
    if isinstance(value, (str, list)) and len(value) > schema.get("maxLength", schema.get("maxItems", 1000000)):
        raise ProtocolError("parameter too long")


def sample_arguments(schema: dict[str, Any], files: list[str] | None = None) -> dict[str, Any]:
    """Schema-valid example arguments; used for protocol examples and replay stubs."""
    sample: dict[str, Any] = {}
    for name, item in schema.get("properties", {}).items():
        if name not in schema.get("required", []):
            continue
        kind = item.get("type")
        if "enum" in item:
            sample[name] = item["enum"][0]
        elif name == "filename" and kind == "string":
            sample[name] = (files or ["memory.txt"])[0]
        elif kind == "integer":
            sample[name] = int(item.get("minimum", 1))
        elif kind == "number":
            sample[name] = float(item.get("minimum", 1.0))
        elif kind == "boolean":
            sample[name] = False
        elif kind == "object":
            sample[name] = {}
        elif kind == "array":
            sample[name] = []
        elif kind == "null":
            sample[name] = None
        else:
            sample[name] = "example"
    return sample


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    phases: tuple[str, ...]
    permission: str = "memory"
    returns: str = "text"
    legacy: bool = False


class Registry:
    RESERVED = {"ACTION", "DONE", "ERROR"}

    def __init__(self, event: Callable[..., None] | None = None, file_index: Callable[[], Any] | None = None):
        self.tools: dict[str, Tool] = {}
        self.event = event or (lambda *args, **kwargs: None)
        self.file_index = file_index or (lambda: [])

    def register(self, tool: Tool) -> None:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", tool.name) or tool.name in self.tools or tool.name in self.RESERVED:
            raise ValueError("invalid, duplicate or reserved tool name")
        if not tool.phases or not set(tool.phases) <= {"LOG", "PLAN"}:
            raise ValueError("invalid tool phases")
        if tool.permission not in {"memory", "memory_read", "memory_write"}:
            raise ValueError("invalid tool permission")
        self.tools[tool.name] = tool

    def describe(self, phase: str) -> list[Tool]:
        return [t for t in self.tools.values() if phase in t.phases]

    def specs(self) -> list[dict[str, Any]]:
        """JSON-safe description of every registered tool for the host protocol text."""
        return [{"name": t.name, "description": t.description, "parameters": t.parameters,
                 "phases": list(t.phases), "permission": t.permission, "returns": t.returns,
                 "legacy": t.legacy} for t in self.tools.values()]

    def execute(self, name: str, args: dict[str, Any], phase: str) -> str:
        try:
            if name not in self.tools:
                raise ProtocolError("unknown tool: " + name)
            tool = self.tools[name]
            if phase not in tool.phases:
                raise ProtocolError("tool unavailable in " + phase)
            validate(args, tool.parameters)
            result = tool.handler(**args)
            output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, allow_nan=False)
        except Exception as exc:  # every failure is charged to the model as a tool error
            output = "[TOOL ERROR: " + str(exc) + "]"
        self.event("tool_result", name=name, phase=phase, error=output.startswith("[TOOL ERROR:"))
        return output


READ_TOOLS = ("READ", "SEARCH", "TAIL", "COUNT")


def register_legacy(registry: Registry, memory: Any, *, read_only: bool = False) -> None:
    specs = [("READ", memory.read, ("LOG", "PLAN"), "read a memory file"),
             ("SEARCH", memory.search, ("LOG", "PLAN"), "search a memory file for a keyword"),
             ("TAIL", memory.tail, ("LOG", "PLAN"), "read the last n lines of a memory file"),
             ("COUNT", memory.count, ("LOG", "PLAN"), "count lines containing a keyword"),
             ("APPEND", memory.append, ("LOG",), "append a note to a memory file"),
             ("WRITE", memory.write, ("LOG",), "overwrite a memory file"),
             ("CREATE", memory.create, ("LOG",), "create an empty memory file")]
    if read_only:
        specs = [spec for spec in specs if spec[0] in READ_TOOLS]
    for name, fn, phases, description in specs:
        keys = LEGACY_KEYS[name]
        props = {key: {"type": "integer", "minimum": 0, "maximum": 10000} if key == "n" else {"type": "string"}
                 for key in keys}

        def handler(_fn=fn, _keys=keys, **kwargs):
            return _fn(*(kwargs[key] for key in _keys))
        registry.register(Tool(name, description, {"type": "object", "properties": props,
                                                   "required": list(keys), "additionalProperties": False},
                               handler, phases,
                               "memory_read" if name in READ_TOOLS else "memory_write",
                               "text", True))


def _action_allowed(payload: str, action_spec: list[str] | dict[str, Any]) -> bool:
    if isinstance(action_spec, list):
        return payload in action_spec
    if action_spec.get("kind") == "finite":
        return payload in action_spec.get("actions", [])
    if action_spec.get("kind") == "free_text":
        maximum = int(action_spec.get("max_length", 512))
        return bool(payload) and len(payload) <= maximum and not any(c in payload for c in ("\x00", "\n", "\r"))
    return False


def parse_command(text: str, phase: str, actions: list[str] | dict[str, Any]) -> tuple[str, str]:
    """Read one command and check the phase and adapter action contract."""
    if text.strip() == "<|DONE|>":
        if phase != "LOG":
            raise ProtocolError("DONE only available in LOG")
        return "DONE", ""
    matches = list(COMMAND_RE.finditer(text))
    if len(matches) != 1:
        raise ProtocolError("exactly one complete command required")
    name, payload = matches[0].group(1), matches[0].group(2).strip()
    if "<|" in payload:
        raise ProtocolError("nested command")
    if name == "DONE":
        raise ProtocolError("DONE must appear alone")
    if name == "ACTION" and (phase != "PLAN" or not _action_allowed(payload, actions)):
        raise ProtocolError("invalid/phase-forbidden ACTION")
    return name, payload


def parse_arguments(name: str, payload: str, legacy: bool) -> dict[str, Any]:
    if legacy:
        keys = LEGACY_KEYS[name]
        parts = payload.split("|", len(keys) - 1)
        if len(parts) != len(keys):
            raise ProtocolError("invalid arguments for " + name)
        args = dict(zip(keys, parts))
        if name == "TAIL":
            try:
                args["n"] = int(args["n"])
            except ValueError as exc:
                raise ProtocolError("TAIL n must be integer") from exc
        return args
    try:
        args = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProtocolError("new tool arguments must be a JSON object") from exc
    if not isinstance(args, dict):
        raise ProtocolError("new tool arguments must be a JSON object")
    return args


def decode(text: str, phase: str, registry: Registry,
           actions: list[str] | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name, payload = parse_command(text, phase, actions)
    if name == "DONE":
        return name, {}
    if name == "ACTION":
        return name, {"action": payload}
    tool = registry.tools.get(name)
    if tool is None or phase not in tool.phases:
        raise ProtocolError("unknown/phase-forbidden tool")
    return name, parse_arguments(name, payload, tool.legacy)
