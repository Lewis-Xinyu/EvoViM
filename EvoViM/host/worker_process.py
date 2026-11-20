"""Run candidate decisions in a child process and validate model requests and actions.

In process mode the caller trusts the candidate. Seatbelt mode also restricts
its filesystem and network access on macOS.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from PIL import Image

from .bundle import profile, verify_bundle
from .dto import CandidateHung, ContainmentError, Violation
from .model import BudgetExhausted, ModelCall
from .protocol_text import host_decode

MODES = ("seatbelt", "process")
MAX_MESSAGE = 64 * 1024 ** 2
WATCHDOG = {"disk_bytes": 1024 ** 3, "files": 32768, "rss_kb": 4 * 1024 ** 2}
CHILD_FILES = ("worker_main.py", "registry.py")
ROLES = {"system", "user", "assistant"}


def _png_b64(image) -> str | None:
    if image is None:
        return None
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def archive_images(images, archive: Path) -> list[dict]:
    """Store exactly the pixels supplied to ModelCall, including derived images."""
    directory = archive / "model_images"
    directory.mkdir(exist_ok=True)
    records = []
    for image in images:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        path = directory / f"{digest}.png"
        if not path.exists():
            path.write_bytes(data)
        records.append({"path": str(path.relative_to(archive)), "sha256": digest,
                        "size": list(image.size)})
    return records


def child_env(scratch: Path) -> dict[str, str]:
    scratch = str(Path(scratch).resolve())
    return {"PATH": "/usr/bin:/bin", "HOME": scratch, "TMPDIR": scratch, "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}


BACKSTOP_SECONDS = 30 * 60.0


class CandidateStartFailure(ContainmentError):
    """Startup traceback points into the candidate bundle."""


class Worker:
    def __init__(self, bundle: Path, root: Path, *, line: str, instruction: str, actions: list[str],
                 adapter_name: str, fallback_action: str, action_spec: dict[str, Any] | None = None,
                 mode: str = "process",
                 compute_timeout: float | None = BACKSTOP_SECONDS, startup_timeout: float | None = BACKSTOP_SECONDS,
                 phase_caps: dict[str, int] | None = None, checkpoint: dict | None = None,
                 profile_calls: bool = False, watchdog: dict | None = None,
                 no_automem: bool = False, disable_visual_recall: bool = False):
        if mode not in MODES:
            raise ValueError("worker mode must be one of " + ", ".join(MODES))
        if line not in {"T", "V", "M"}:
            raise ValueError("line must be T, V or M")
        self.proc = None
        self.selector = None
        self.monitor = None
        self.stderr = None
        self.bundle = Path(bundle).resolve()
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        for name in CHILD_FILES:
            shutil.copyfile(Path(__file__).with_name("child") / name, self.runtime / name)
        verify_bundle(self.bundle)
        self.line, self.mode, self.actions = line, mode, list(actions)
        self.action_spec = action_spec or {"kind": "finite", "actions": self.actions, "dynamic": False}
        self.fallback_action = fallback_action
        self.compute_timeout = None if compute_timeout is None else float(compute_timeout)
        self.compute_seconds = 0.0
        self.caps = dict(phase_caps or {"LOG": 10, "PLAN": 10})
        self.limits = dict(watchdog or WATCHDOG)
        self.visual_counts = {"valid": 0, "invalid": 0, "canvas_action_labels": 0}
        self.closed = False
        self.buffer = b""
        self.resource_error = None
        argv = [sys.executable, "-B", str(self.runtime / "worker_main.py"), str(self.bundle), str(self.scratch)]
        if mode == "seatbelt":
            policy = self.root / "worker.sb"
            policy.write_text(profile(self.bundle, self.scratch, self.runtime), encoding="utf-8")
            argv = ["/usr/bin/sandbox-exec", "-f", str(policy), *argv]
        self.stderr = (self.root / "stderr.log").open("wb")
        try:
            self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                         cwd=self.scratch, env=child_env(self.scratch), start_new_session=True)
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.proc.stdout, selectors.EVENT_READ)
            self.monitor = threading.Thread(target=self._watch, daemon=True)
            self.monitor.start()
            self._send({"line": line, "instruction": instruction, "actions": self.actions,
                        "action_spec": self.action_spec,
                        "adapter_name": adapter_name, "fallback_action": fallback_action,
                        "max_log_iterations": self.caps["LOG"], "max_validate_iterations": self.caps["PLAN"],
                        "checkpoint": checkpoint, "profile": bool(profile_calls),
                        "no_automem": bool(no_automem),
                        "disable_visual_recall": bool(disable_visual_recall)})
            ready = self._read(startup_timeout)
            if ready.get("kind") == "crash":
                trace = str(ready.get("traceback", ""))
                failure = f"candidate failed to start: {ready.get('error')}\n{trace}"
                if str(self.bundle) in trace:
                    raise CandidateStartFailure(failure)
                raise ContainmentError(failure)
            if ready.get("kind") != "ready":
                raise ContainmentError("worker initialization failed")
        except BaseException:
            self.close()
            raise
        self.tools = ready.get("tools", [])

    def _watch(self):
        tick, files, disk = 0, [], 0
        while not self.closed and self.proc.poll() is None:
            try:
                if tick % 5 == 0:  # Scan scratch once per second.
                    files = [p for p in self.scratch.rglob("*") if p.is_file()]
                    disk = sum(p.stat().st_size for p in files)
                rss = subprocess.check_output(["/bin/ps", "-o", "rss=", "-p", str(self.proc.pid)],
                                              text=True, stderr=subprocess.DEVNULL).strip()
                rss_kb = int(rss) if rss else 0
                if disk > self.limits["disk_bytes"] or len(files) > self.limits["files"] or rss_kb > self.limits["rss_kb"]:
                    self.resource_error = (f"resource watchdog: scratch {disk} B / {self.limits['disk_bytes']}, "
                                           f"{len(files)} files / {self.limits['files']}, "
                                           f"RSS {rss_kb} KB / {self.limits['rss_kb']}")
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    return
            except (OSError, ValueError, subprocess.CalledProcessError):
                pass
            tick += 1
            time.sleep(0.2)

    def _send(self, value: dict) -> None:
        raw = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        if len(raw) > MAX_MESSAGE:
            raise ContainmentError("IPC message too large")
        try:
            self.proc.stdin.write(raw)
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ContainmentError("worker pipe closed: " + self._stderr_tail()) from exc

    def _stderr_tail(self) -> str:
        self.stderr.flush()
        return (self.root / "stderr.log").read_text(errors="replace")[-2000:]

    def _read(self, timeout: float | None) -> dict:
        started = time.monotonic()
        deadline = None if timeout is None else started + timeout
        try:
            while b"\n" not in self.buffer:
                if self.resource_error:
                    raise ContainmentError(self.resource_error)
                if len(self.buffer) > MAX_MESSAGE:
                    raise ContainmentError("oversized worker output")
                # Without a deadline, poll each second so resource watchdog errors are still seen.
                wait = 1.0 if deadline is None else max(0.0, deadline - time.monotonic())
                if not self.selector.select(wait):
                    if deadline is None:
                        continue
                    raise CandidateHung(f"candidate computed for {timeout:.0f} s without answering the host "
                                        "(backstop against infinite loops)")
                chunk = os.read(self.proc.stdout.fileno(), 65536)
                if not chunk:
                    raise ContainmentError("worker exited: %s; %s" % (self.proc.poll(), self._stderr_tail()))
                self.buffer += chunk
        finally:
            self.compute_seconds += time.monotonic() - started
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def act(self, *, step: int, observation: dict, previous_action: str | None,
            previous_validated_action: str | None, frame_id: str | None, image, current_image,
            model, system_prefix: Callable[[str, list, list], str], frame_store=None) -> dict[str, Any]:
        """Run one candidate decision under host rules. Never raises for candidate faults."""
        record: dict[str, Any] = {"calls": [], "exchanges": [], "protocol_errors": 0, "tool_events": [], "visual": [],
                                  "memory": {}, "source_calls": {}}
        seen_messages = {"LOG": 0, "PLAN": 0}
        counts = {"LOG": 0, "PLAN": 0}
        seen_plan = False
        current_phase = None
        committed = None
        # Rendered historical images wait here until the next model request of
        # this step, whichever phase it belongs to (a LOG recall reaches PLAN).
        pending: list[tuple[Image.Image, str]] = []
        tools = self.tools
        try:
            self._send({"op": "act", "observation": observation, "step": step,
                        "previous_action": previous_action, "previous_validated_action": previous_validated_action,
                        "visual_observation_available": image is not None,
                        "frame_id": frame_id if self.line == "M" else None,
                        "image": _png_b64(image) if self.line == "M" else None})
            while True:
                message = self._read(self.compute_timeout)
                kind = message.get("kind")
                if kind == "event":
                    event = {k: v for k, v in message.items() if k != "kind"}
                    record["tool_events"].append(event)
                    if record["exchanges"]:
                        record["exchanges"][-1]["tool_events"].append(event)
                    continue
                if kind == "model":
                    phase = message.get("phase")
                    if phase not in counts:
                        raise Violation("unknown phase")
                    if phase == "LOG" and (previous_action is None or seen_plan):
                        raise Violation("LOG requires a real preceding action and must precede PLAN")
                    seen_plan = seen_plan or phase == "PLAN"
                    if phase != current_phase:
                        current_phase = phase
                        if frame_store is not None:
                            frame_store.begin_phase()
                    counts[phase] += 1
                    if counts[phase] > self.caps[phase]:
                        raise Violation(f"{phase} request cap of {self.caps[phase]} exceeded")
                    messages = message.get("messages")
                    if not isinstance(messages, list) or any(
                            not isinstance(m, dict) or set(m) != {"role", "content"} or m["role"] not in ROLES
                            or not isinstance(m["content"], str) for m in messages):
                        raise Violation("invalid model request DTO")
                    tools = message.get("tools") or tools
                    files = message.get("files") or []
                    images, labels = [], []
                    if self.line in {"V", "M"} and current_image is not None:
                        images.append(current_image)
                        labels.append(f"Current observation image (step {step}).")
                    if self.line == "M":
                        for rendered_image, label in pending:
                            images.append(rendered_image)
                            labels.append(label)
                    pending = []
                    wire = [{"role": "system", "content": system_prefix(phase, tools, files)}] + messages
                    call = ModelCall(phase, step, counts[phase], wire, images, labels)
                    image_records = archive_images(images, self.root.parent)
                    # Persist the request before calling the provider, including
                    # attempts interrupted by budgets, provider faults or host exit.
                    with (self.root.parent / "model_requests.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"schema_version": 3, "step": step, "phase": phase,
                                                 "call": counts[phase], "messages": wire,
                                                 "image_records": image_records, "image_labels": labels},
                                                ensure_ascii=False, allow_nan=False) + "\n")
                    reply = model.generate(call)
                    record["calls"].append(call)
                    completion = reply.completion
                    if reply.stop_reason == "length":
                        completion = "[TRUNCATED COMMAND; no action]"
                    decoded = host_decode(completion, phase, tools, self.action_spec)
                    if decoded is None:
                        record["protocol_errors"] += 1
                    elif phase == "PLAN" and decoded[0] == "ACTION":
                        committed = decoded[1]
                    start = seen_messages[phase] if counts[phase] > 1 else 0
                    seen_messages[phase] = len(messages)
                    record["exchanges"].append({
                        "phase": phase, "call": counts[phase], "new_messages": messages[start:],
                        "schema_version": 3, "messages": wire, "image_records": image_records,
                        "message_count": len(messages), "images": len(images), "image_labels": labels,
                        "completion": completion, "raw_completion": reply.completion,
                        "model_id": reply.model_id, "input_tokens": int(reply.input_tokens),
                        "output_tokens": int(reply.output_tokens), "stop_reason": reply.stop_reason,
                        "valid_command": decoded is not None, "tool_events": [], "visual_events": []})
                    self._send({"kind": "response", "value": {
                        "model_id": reply.model_id, "completion": completion, "stop_reason": reply.stop_reason,
                        "input_tokens": int(reply.input_tokens), "output_tokens": int(reply.output_tokens)}})
                elif kind == "visual_request":
                    visual_event = {"program": message.get("program")}
                    if record["exchanges"]:
                        record["exchanges"][-1]["visual_events"].append(visual_event)
                    if self.line != "M":
                        raise Violation(f"line {self.line} cannot request historical images")
                    if not message.get("in_tool"):
                        raise Violation("visual requests are only allowed inside a model-invoked tool")
                    phase = message.get("phase") if message.get("phase") in counts else current_phase
                    try:
                        if frame_store is None:
                            raise ValueError("no host frame store")
                        # A LOG-phase request is always followed by at least one PLAN
                        # request; a PLAN request needs a PLAN call left in this step.
                        has_next = phase == "LOG" or counts.get(phase, 0) < self.caps.get(phase, 0)
                        rendered = frame_store.render(message.get("program"), current_step=step,
                                                      has_next_request=has_next, legal_actions=self.actions)
                    except (ValueError, KeyError, TypeError, ContainmentError) as exc:
                        visual_event.update(valid=False, error=str(exc))
                        self.visual_counts["invalid"] += 1
                        record["visual"].append({"valid": False, "error": str(exc)})
                        self._send({"kind": "visual_response", "error": str(exc)})
                    else:
                        visual_event.update(valid=True, size=list(rendered.size),
                                            label=frame_store.last_description)
                        self.visual_counts["valid"] += 1
                        self.visual_counts["canvas_action_labels"] = frame_store.canvas_action_labels
                        record["visual"].append({"valid": True, "size": list(rendered.size),
                                                 "label": frame_store.last_description})
                        pending.append((rendered, frame_store.last_description))
                        self._send({"kind": "visual_response", "error": None})
                elif kind == "decision":
                    action = message.get("action")
                    record.update(memory=message.get("memory", {}), source_calls=message.get("source_calls", {}))
                    # Only a host-validated model action can reach the adapter.
                    override = committed is not None and action != committed
                    executed = committed if committed is not None else self.fallback_action
                    record.update(status="ok", action=executed, proposed_action=action,
                                  committed_action=committed, executed_action=executed,
                                  fallback=committed is None, action_override=override, counts=counts)
                    return record
                elif kind == "crash":
                    self.close()
                    record.update(status="candidate_crashed", error=message.get("error"),
                                  traceback=message.get("traceback"), failure_origin="candidate", counts=counts)
                    return record
                else:
                    raise ContainmentError("unexpected worker message")
        except Violation as exc:
            self.close()
            record.update(status="containment_violation", error=str(exc), counts=counts)
            return record
        except CandidateHung as exc:
            self.close()
            record.update(status="candidate_hung", error=str(exc), failure_origin="candidate", counts=counts)
            return record
        except ContainmentError as exc:
            self.close()
            record.update(status="candidate_crashed", error=str(exc), counts=counts)
            return record
        except BudgetExhausted as exc:
            self.close()
            record.update(status="budget_exhausted", error=str(exc), counts=counts)
            return record
        except Exception as exc:
            self.close()
            record.update(status="model_error", error=str(exc), counts=counts)
            return record

    def checkpoint(self) -> dict:
        self._send({"op": "checkpoint"})
        message = self._read(self.compute_timeout)
        if message.get("kind") != "checkpoint":
            raise ContainmentError("checkpoint failed")
        return message["value"]

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b'{"op": "close"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=5)
        if self.selector is not None:
            self.selector.close()
        if self.stderr is not None:
            self.stderr.close()
        for stream in (self.proc.stdin, self.proc.stdout) if self.proc is not None else ():
            try:
                stream.close()
            except OSError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
