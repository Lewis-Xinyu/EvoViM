"""Offline campaign using the real grid, candidate worker, selection and storage."""

import json
import shlex
import subprocess
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

from EvoViM.evolution import codex_runner
from EvoViM.evolution.quick_evo import Run
from EvoViM.host import transport
from EvoViM.host.repair import Promotion


def manifest(workspaces):
    return {
        "experiment_mode": "exploratory",
        "environments": [{"family": "grid", "adapter": "gridmemory", "task_id": "gridmemory",
                          "line": "M", "max_steps": 1}],
        "seeds": {"dev": [0], "holdout": [25]},
        "generations": {"max": 1},
        "runtime": {"mode": "process"},
        "gate": {"rule": "strict_mean", "alpha": 0.05},
        "noise": {"parent_repetitions": 1},
        "model": {"name": "offline-model", "provider": "openai",
                  "base_url": "https://offline.invalid/v1", "max_completion_tokens": 64},
        "meta": {"provider": "codex", "codex": {"binary": "offline-codex-stub",
                                                 "workspace_root": str(workspaces)}},
        "review": {"require_human": False},
        "budget": {"input_tokens": 1_000_000, "output_tokens": 100_000, "amount": None},
        "authorization": {"max_units": 100, "max_input_tokens": 1_000_000,
                          "max_output_tokens": 100_000},
        "holdout_after_lineage": True,
        "holdout_versions": ["baseline", "parent"],
    }


def reply(payload):
    text = json.dumps(payload["messages"])
    action = "Move East" if "Try the relic to your east." in text else "Noop"
    return {"model": payload["model"],
            "choices": [{"message": {"content": f"<|ACTION|>{action}<|END|>"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6}}


def revise(argv, *, cwd, **kwargs):
    """Simulate one local Codex CLI response using a revision of the public prompt."""
    output = Path(cwd)
    relative = Path("agents/memory_agent.py")
    source = (output.parent / "input" / "source" / relative).read_text()
    line = "        obs_text = self._format_obs(obs)"
    if source.count(line) != 1:
        raise AssertionError("Expected one observation formatter in the starting agent")
    addition = '\n        if "you / blue relic" in obs_text:\n            obs_text += " Try the relic to your east."'
    changed = output / "revised_code" / relative
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text(source.replace(line, line + addition), encoding="utf-8")
    command = (output.parent / "replay_command.txt").read_text()
    subprocess.run(shlex.split(command), cwd=output, check=True, capture_output=True, timeout=60)
    report = json.loads((output / "replay_report.json").read_text())
    if report["crashed"] or report["steps"] != 1:
        raise AssertionError(report)
    transcript = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100,
                                                                  "output_tokens": 30}})
    return subprocess.CompletedProcess(argv, returncode=0, stdout=transcript, stderr="")


class CampaignTest(unittest.TestCase):
    def test_development_holdout_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            config = base / "manifest.json"
            config.write_text(json.dumps(manifest(base / "revision workspaces")))
            root = base / "run"
            meta = partial(codex_runner.run_meta_slot, runner=revise)
            with patch.object(codex_runner, "run_preflight", return_value={"passed": True}), \
                 patch.object(codex_runner, "run_meta_slot", side_effect=meta) as revision, \
                 patch.object(transport, "make_sender", return_value=reply) as sender:
                # The first supervisor stops just after publishing the accepted parent.
                commit = Promotion.commit

                def interrupt_after_commit(promotion, comparison):
                    commit(promotion, comparison)
                    raise InterruptedError("test interruption")

                with patch.object(Promotion, "commit", interrupt_after_commit):
                    with self.assertRaises(InterruptedError):
                        Run(root, config).run()
                result = Run(root, config).run()
                self.assertEqual(result["status"], "COMPLETED", result)
                revision.assert_called_once()
                self.assertTrue(sender.called)
                line = root / "environments" / "grid" / "lines" / "M"
                generation = line / "generations" / "g1_fresh"
                gate = json.loads((generation / "gate.json").read_text())
                self.assertTrue(gate["accepted"], gate)
                self.assertEqual(json.loads((generation / "scores.json").read_text())["scores"], {"0": [1.0]})
                for name, expected in [("baseline", 0.0), ("parent", 1.0)]:
                    scores = json.loads((line / f"holdout_{name}" / "scores.json").read_text())
                    self.assertTrue(scores["complete"])
                    self.assertEqual(scores["scores"], {"25": [expected]})

                sender.reset_mock()
                revision.reset_mock()
                resumed = Run(root, config).run()
                self.assertEqual(resumed["status"], "COMPLETED", resumed)
                sender.assert_not_called()
                revision.assert_not_called()


if __name__ == "__main__":
    unittest.main()
