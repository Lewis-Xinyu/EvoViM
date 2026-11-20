"""Check Codex preflight retry and recovery without a model or network."""

import json
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
import unittest

from visual_rsi.evolution.codex_runner import run_preflight


class PreflightTest(unittest.TestCase):
    def test_retry_cannot_reuse_old_probe_results(self):
        config = {"meta": {"provider": "codex", "codex": {}}}

        def passed(argv, *, cwd, **kwargs):
            output = Path(cwd)
            checks = {"input_read": True, "input_write_denied": True, "network_denied": True}
            (output / "checks.json").write_text(json.dumps(checks))
            (output / "output_probe.txt").write_text("CAN_WRITE_OUTPUT")
            return CompletedProcess(argv, 0, '{"type":"turn.completed"}', "")

        def incomplete(argv, **kwargs):
            return CompletedProcess(argv, 0, '{"type":"turn.completed"}', "")

        with TemporaryDirectory() as temp:
            directory = Path(temp)
            self.assertTrue(run_preflight(config, directory, runner=passed)["passed"])
            self.assertFalse(run_preflight(config, directory, runner=incomplete)["passed"])
            self.assertTrue(run_preflight(config, directory, runner=passed)["passed"])


if __name__ == "__main__":
    unittest.main()
