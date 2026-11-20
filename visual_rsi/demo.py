"""Run a visual memory program in a small grid using an offline scripted model."""

import json
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


def run(args) -> int:
    with TemporaryDirectory(prefix="evovim-demo-") as temp:
        archive = args.output or Path(temp)
        if args.output:
            archive.mkdir(parents=True, exist_ok=False)
        if args.cards:
            from .card_demo import run_card_demo
            result = run_card_demo(archive)
        else:
            bundle = args.bundle or files("visual_rsi").joinpath("inner", "visual_memory")
            if not (bundle / "candidate.py").is_file():
                raise ValueError(f"Missing candidate.py in {bundle}")
            result = run_episode(GridMemoryAdapter(max_steps=2), bundle, line="M", seed=0,
                                 model=ScriptedModel(), archive=archive)
        fields = ("status", "steps", "actions", "score", "visual_requests", "recalled_pair", "error")
        summary = {key: result[key] for key in fields if key in result}
        summary["trace"] = str(archive.resolve()) if args.output else "temporary"
        print(json.dumps(summary, indent=2))
        return 0 if result["status"] in {"completed", "terminated", "truncated"} else 1
