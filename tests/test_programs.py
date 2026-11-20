import tempfile
import unittest
from pathlib import Path

from EvoViM.card_demo import run_card_demo
from EvoViM.host.bundle import verify_bundle
from EvoViM.programs import open_program, program_names


class ReleasedProgramsTest(unittest.TestCase):
    def test_all_paper_programs_materialize(self):
        tasks = ("card-matching", "dmv-hard", "minecraft-gallery", "poster-maze", "tetris", "pac-man")
        names = {f"{model}/{task}" for model in ("gpt", "haiku", "qwen") for task in tasks}
        self.assertEqual(set(program_names()), names)
        for name in sorted(names):
            with self.subTest(program=name), open_program(name) as bundle:
                self.assertIn("candidate.py", verify_bundle(bundle))

    def test_evolved_card_retrieval(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_card_demo(Path(temp))
        self.assertEqual(result["status"], "terminated", result.get("error"))
        self.assertEqual(result["recalled_pair"], ["bA", "aA"])
        self.assertEqual(result["actions"][-1], "aA")
        self.assertEqual(result["visual_requests"]["valid"], 1)
        self.assertEqual(result["protocol_errors"], 0)
        self.assertEqual(result["score"], 1)
