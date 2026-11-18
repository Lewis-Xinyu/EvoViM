"""Run a visual memory program in a small grid using an offline scripted model."""

import json
import sys
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

from visual_rsi.adapters.gridmemory import GridMemoryAdapter
from visual_rsi.host.episode import run_episode
from visual_rsi.host.model import ModelReply


class ScriptedModel:
    def generate(self, call):
        if call.phase == "LOG":
            return ModelReply("<|DONE|>")
        tools = call.messages[0]["content"]
        if call.step == 0 and call.call_index == 1 and "<|SAVE_FRAME|>" in tools:
            return ModelReply('<|SAVE_FRAME|>{"label":"starting view"}<|END|>')
        if call.step == 1 and call.call_index == 1 and "<|RECALL_IMAGE|>" in tools:
            return ModelReply('<|RECALL_IMAGE|>{"image_id":"img_0001"}<|END|>')
        action = "Move East" if call.step == 0 else "Task Done"
        return ModelReply(f"<|ACTION|>{action}<|END|>")


def add_arguments(parser):
    parser.add_argument("--bundle", type=Path, help="Directory containing a trusted memory program and candidate.py")
    parser.add_argument("--output", type=Path, help="Save the episode trace to a new directory (temporary by default)")


def run(args) -> int:
    bundle = args.bundle or files("visual_rsi").joinpath("inner", "visual_memory")
    if not (bundle / "candidate.py").is_file():
        print(f"Missing candidate.py in {bundle}", file=sys.stderr)
        return 2

    with TemporaryDirectory(prefix="evovim-demo-") as temp:
        archive = args.output or Path(temp)
        if args.output is not None:
            try:
                archive.mkdir(parents=True, exist_ok=False)
            except OSError as error:
                print(f"Cannot create a new output directory: {error}", file=sys.stderr)
                return 2
        result = run_episode(GridMemoryAdapter(max_steps=2), bundle, line="M", seed=0,
                             model=ScriptedModel(), archive=archive)
        summary = {name: result[name] for name in ("status", "steps", "actions", "score")}
        summary["visual_requests"] = result["visual_requests"]
        summary["trace"] = str(archive.resolve()) if args.output else "temporary"
        if result.get("error"):
            summary["error"] = result["error"]
        print(json.dumps(summary, indent=2))
        return 0 if result["status"] in {"completed", "terminated", "truncated"} else 1
