"""Exercise both evaluation modes with a tiny local DMV image registry."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from visual_rsi.cli import main
from visual_rsi.host import transport


class EvaluationTest(unittest.TestCase):
    def test_dmv_evaluation_and_one_step_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data" / "vismem_diag_v2"
            photos = data / "images" / "base"
            folder = photos / "chair" / "plain"
            folder.mkdir(parents=True)
            Image.new("RGB", (24, 24), "cyan").save(folder / "00.png")
            row = {"cat": "chair", "style": "plain", "prod_idx": 0, "url_hash": "000000aa"}
            (data / "cue_registry.json").write_text(json.dumps({"rows": [row]}))

            def scripted_reply(payload):
                self.assertEqual(payload["model"], "fixture")
                phase = payload["messages"][0]["content"].splitlines()[0]
                reply = "<|DONE|>" if phase == "PHASE: LOG" else "<|ACTION|>continue<|END|>"
                return {"choices": [{"message": {"content": reply}}],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 5}}

            with patch.dict(os.environ, {"EVOVIM_DMV_ROOT": temp, "DMVHARD_BASE_DIR": str(photos)}), \
                 patch.object(transport, "make_sender", return_value=scripted_reply) as connect:
                for mode in (["--smoke"], ["--model", "fixture"]):
                    out = root / mode[0][2:]
                    with self.subTest(mode=mode), redirect_stdout(io.StringIO()):
                        code = main(["evaluate", "gpt/dmv-hard", *mode, "--output", str(out)])
                        result = json.loads((out / "summary.json").read_text())
                        self.assertEqual(code, 0, result.get("error"))
                        self.assertTrue((out / "frames" / "f00000.png").is_file())
                        self.assertEqual(result["actions"][0], "continue")
                        self.assertEqual(result["protocol_errors"], 0)
                        if mode == ["--smoke"]:
                            connect.assert_not_called()
                            self.assertEqual(result["steps"], 1)
                        else:
                            connect.assert_called_once()
                            self.assertTrue(result["terminated"])
                            self.assertGreater(result["steps"], 1)
                            self.assertTrue((out / "ledger.json").is_file())
