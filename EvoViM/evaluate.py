"""Evaluate a released memory program with its benchmark's task ID and horizon."""

import json
import os

from .adapters import get_adapter
from .adapters.benchmark import BENCHMARKS
from .host.episode import run_episode
from .host.model import ModelReply
from .host.paid import Ledger, PaidModel
from .host.transport import ProviderError
from .programs import open_program


class FirstAction:
    def __init__(self, adapter):
        self.adapter = adapter

    def generate(self, call):
        if call.phase == "LOG":
            return ModelReply("<|DONE|>")
        spec = self.adapter.action_spec()
        action = (spec.get("actions") or spec["examples"])[0]
        return ModelReply(f"<|ACTION|>{action}<|END|>")


def model_settings(lineage, task, name):
    model = {"provider": "openai", "name": name, "max_completion_tokens": 4096,
             "timeout": 180, "max_attempts": 3, "base_url": os.environ.get("EVOVIM_EPISODE_BASE_URL")}
    if lineage == "gpt":
        model["reasoning_effort"] = "medium" if task == "poster-maze" else "none"
    elif lineage == "haiku":
        model["provider"] = "anthropic"
    elif lineage == "qwen":
        model.update(timeout=300, extra_body={
            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.7,
            "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        })
    return model


def run(args):
    lineage, task = args.program.split("/")
    settings = json.loads((BENCHMARKS / "tasks.json").read_text(encoding="utf-8"))[task]
    seed = args.seed if args.seed is not None else settings["holdout_seeds"][0]
    adapter = get_adapter(settings["adapter"], {**settings, "python": args.benchmark_python})
    description = {"program": args.program, "task_id": adapter.task_id, "seed": seed,
                   "mode": "one_step_smoke" if args.smoke else "evaluation"}
    with open_program(args.program) as bundle:
        if args.smoke:
            model = FirstAction(adapter)
        else:
            config = model_settings(lineage, task, args.model)
            description["model"] = config
            # The single-episode runner uses the task horizon and the paper's per-phase
            # call limits. Multi-episode campaigns supply their own token budgets.
            auth = {"batch_id": "evaluation", "max_input_tokens": None, "max_output_tokens": None}
            try:
                model = PaidModel({"model": config}, auth, Ledger(args.output / "ledger.json", {}),
                                  args.output / "requests.jsonl")
            except ProviderError as error:
                raise ValueError(str(error)) from error
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "evaluation.json").write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
        result = run_episode(adapter, bundle, line="M", seed=seed, model=model, archive=args.output,
                             max_steps=1 if args.smoke else adapter.max_steps, phase_caps={"LOG": 20, "PLAN": 20})
    fields = ("status", "steps", "actions", "error") if args.smoke else ("status", "steps", "score", "error")
    print(json.dumps({**description, **{k: result[k] for k in fields if k in result},
                      "trace": str(args.output.resolve())}, indent=2))
    return 0 if result["status"] in {"completed", "terminated", "truncated"} else 1
