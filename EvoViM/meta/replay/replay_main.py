"""Offline, open-loop replay of a candidate over recorded public evidence.

Runs inside a Meta workspace with no repository access. The candidate is the
parent bundle (input/source) overlaid with output/revised_code, executed by the
same host Worker as a real evaluation: its own agent.act in a separate process.
A stub model walks the tool list the candidate actually registered and calls
every non-legacy tool through a model reply. Nothing here is paid or networked.

    python input/replay/replay_main.py --candidate output/revised_code --parent input/source \
        --evidence input/evidence --line T --output output/replay_report.json
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import tempfile
import traceback

from PIL import Image

from replay_host.bundle import install_bundle, overlay_files
from replay_host.child.registry import sample_arguments
from replay_host.model import ModelReply
from replay_host.protocol_text import memory_filenames, system_prefix
from replay_host.visual_channel import FrameStore
from replay_host.worker_process import Worker


def definitions(source: str) -> dict[str, str]:
    result = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.setdefault(node.name, "")
            result[node.name] += ast.dump(node, include_attributes=False)
    return result


def changed_definitions(parent: Path, overlay: dict[str, str]) -> set[str]:
    changed = set()
    for name, source in overlay.items():
        old_path = parent / name
        old = definitions(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
        for definition, dump in definitions(source).items():
            if old.get(definition) != dump:
                changed.add(f"{name}:{definition}")
    return changed


def replay_action(meta: dict, actions: list[str]) -> str:
    """Choose a valid deterministic action for finite and free-text evidence."""
    action_spec = meta.get("action_spec") or {"kind": "finite", "actions": actions}
    if action_spec.get("kind") == "free_text":
        action = str(meta.get("fallback_action") or "look")
        if not action or len(action) > int(action_spec.get("max_length", 512)) or any(
                char in action for char in ("\x00", "\n", "\r")):
            raise ValueError("evidence contains no valid free-text replay action")
        return action
    if not actions:
        raise ValueError("finite-action evidence contains no legal replay action")
    return actions[0]


class StubModel:
    """Calls every registered non-legacy tool once through the model channel, then finishes the phase."""

    def __init__(self, tools: list[dict], action: str):
        self.log_queue = [t for t in tools if not t.get("legacy") and "LOG" in t.get("phases", [])]
        self.plan_queue = [t for t in tools if not t.get("legacy") and "LOG" not in t.get("phases", [])]
        # Save before recall or deletion so the model can exercise the visual channel.
        visual_order = {"SAVE_FRAME": 0, "RECALL_IMAGE": 1, "ANNOTATE_IMAGE": 2, "DELETE_IMAGE": 3}
        self.log_queue.sort(key=lambda tool: (visual_order.get(tool.get("name"), 10), tool.get("name", "")))
        self.plan_queue.sort(key=lambda tool: (tool.get("name") != "RECALL_IMAGE",
                                               tool.get("name") == "DELETE_IMAGE", tool.get("name", "")))
        self.action = action
        self.emitted: list[str] = []

    def _command(self, tool: dict, files: list[str]) -> str:
        self.emitted.append(tool["name"])
        args = sample_arguments(tool.get("parameters", {}), files)
        if "image_id" in args:
            args["image_id"] = "img_0001"
        return "<|%s|>%s<|END|>" % (tool["name"], json.dumps(args, ensure_ascii=False))

    def generate(self, call):
        system = call.messages[0]["content"]
        files = re.findall(r"<\|READ\|>(.+?)<\|END\|>", system)
        if call.phase == "LOG":
            if self.log_queue and call.call_index < 9:
                return ModelReply(self._command(self.log_queue.pop(0), files), 1, 1)
            return ModelReply("<|DONE|>", 1, 1)
        if call.step == 0 and self.plan_queue and self.plan_queue[0].get("name") == "RECALL_IMAGE":
            return ModelReply(f"<|ACTION|>{self.action}<|END|>", 1, 1)
        if self.plan_queue and call.call_index < 9:
            return ModelReply(self._command(self.plan_queue.pop(0), files), 1, 1)
        if call.call_index == 1 and files:
            return ModelReply(f"<|READ|>{files[0]}<|END|>", 1, 1)
        return ModelReply(f"<|ACTION|>{self.action}<|END|>", 1, 1)


def load_episode(evidence: Path) -> tuple[Path, dict, list[dict]]:
    episodes = sorted(p.parent for p in evidence.rglob("trajectory.jsonl") if (p.parent / "episode.json").exists())
    if not episodes:
        raise FileNotFoundError("no evidence episode with trajectory.jsonl and episode.json")
    episode = episodes[0]
    meta = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (episode / "trajectory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return episode, meta, rows


def replay(candidate: Path, parent: Path, evidence: Path, line: str, steps: int | None, mode: str = "process") -> dict:
    report = {"crashed": False, "status": "ok", "traceback": "", "tools_called_via_model": [], "tool_errors": {},
              "changed_functions_executed": {}, "visual_requests": {"valid": 0, "invalid": 0},
              "state_round_trip": False, "steps": 0, "final_memory": {}, "evidence_episode": None}
    overlay = overlay_files(candidate)
    changed = changed_definitions(parent, overlay)
    with tempfile.TemporaryDirectory(prefix="evovim-replay-") as tmp:
        tmp = Path(tmp)
        worker = second = None
        try:
            episode, meta, rows = load_episode(evidence)
            report["evidence_episode"] = episode.name
            actions = list(meta["legal_actions"])
            action_spec = meta.get("action_spec") or {"kind": "finite", "actions": actions, "dynamic": False}
            bundle = install_bundle(tmp / "bundle", parent, overlay)
            frames = FrameStore()
            images = {}
            for index, row in enumerate(rows):
                frame_id = row.get("observation", {}).get("frame_id") or f"f{index:05d}"
                path = episode / "frames" / f"{frame_id}.png"
                if path.exists():
                    images[frame_id] = Image.open(path).convert("RGB")
                    frames.add(frame_id, index, images[frame_id])
            options = dict(line=line, instruction=meta["instruction"], actions=actions,
                           action_spec=action_spec, adapter_name=meta["adapter_name"],
                           fallback_action=meta["fallback_action"], mode=mode,
                           phase_caps=meta.get("phase_caps"))  # the caps the evidence was recorded under
            worker = Worker(bundle, tmp / "worker", profile_calls=True, **options)
            report["phase_caps"] = dict(worker.caps)
            model = StubModel(worker.tools, replay_action(meta, actions))
            prefix = lambda phase, tools, files: system_prefix(phase, tools, action_spec, memory_filenames(files))
            executed: dict[str, int] = {}
            previous = None
            for index, row in enumerate(rows[: steps or len(rows)]):
                frame_id = row.get("observation", {}).get("frame_id") or f"f{index:05d}"
                image = images.get(frame_id)
                result = worker.act(step=index, observation={"text": row["observation"]["text"]},
                                    previous_action=previous, previous_validated_action=previous, frame_id=frame_id,
                                    image=image if line == "M" else None,
                                    current_image=image if line in {"V", "M"} else None,
                                    model=model, system_prefix=prefix, frame_store=frames)
                for event in result.get("tool_events", []):
                    if event.get("event") == "tool_result":
                        executed[event["name"]] = executed.get(event["name"], 0) + 1
                        if event.get("error"):
                            report["tool_errors"][event["name"]] = report["tool_errors"].get(event["name"], 0) + 1
                    elif event.get("event") == "visual_tool_error":
                        report["tool_errors"][event["name"]] = report["tool_errors"].get(event["name"], 0) + 1
                if result.get("status") != "ok":
                    report.update(crashed=True, status=result.get("status"),
                                  traceback=str(result.get("traceback") or result.get("error")))
                    break
                report["steps"] += 1
                report["final_memory"] = result.get("memory", {})
                report["changed_functions_executed"] = {k: v for k, v in result.get("source_calls", {}).items() if k in changed}
                previous = row.get("action")
            report["tools_called_via_model"] = sorted(set(model.emitted) & set(executed))
            report["visual_requests"] = dict(worker.visual_counts)
            if not report["crashed"]:
                state = worker.checkpoint()
                second = Worker(bundle, tmp / "worker_restored", checkpoint=state, **options)
                report["state_round_trip"] = second.checkpoint() == state
        except Exception as exc:
            report.update(crashed=True, status="candidate_crashed",
                          traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        finally:
            for item in (worker, second):
                if item is not None:
                    item.close()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--line", choices=["T", "V", "M"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("process", "seatbelt"), default="process")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args(argv)
    report = replay(args.candidate.resolve(), args.parent.resolve(), args.evidence.resolve(), args.line, args.steps, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"crashed": report["crashed"], "steps": report["steps"], "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
