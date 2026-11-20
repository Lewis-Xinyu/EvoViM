"""The native Haiku episode transport can be exercised without credentials or network."""

import base64
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from EvoViM.host import transport
from EvoViM.host.model import ModelCall
from EvoViM.host.paid import Ledger, PaidModel


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/event-stream"}


class AnthropicTest(unittest.TestCase):
    def test_native_cached_multimodal_round_trip(self):
        events = [
            {"type": "message_start", "message": {"id": "example", "model": "haiku",
                "usage": {"input_tokens": 11, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 3}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "⟦ACT"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ION⟧go⟦END⟧"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 6}},
            {"type": "message_stop"},
        ]
        stream = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
        with TemporaryDirectory() as temp, patch.dict(os.environ, {"ANTHROPIC_API_KEY": "offline"}), \
                patch.object(transport.urllib.request, "urlopen", return_value=Response(stream)) as post:
            path = Path(temp)
            ledger = Ledger(path / "ledger.json", {})
            model = PaidModel({"model": {"name": "haiku", "provider": "anthropic"}},
                              {"max_input_tokens": None, "max_output_tokens": None}, ledger, path / "requests.jsonl")
            call = ModelCall("PLAN", 1, 2, [
                {"role": "system", "content": "Use <|ACTION|>"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<|READ|>notes<|END|>"},
                {"role": "user", "content": "Choose."},
            ], [Image.new("RGB", (2, 3), "red")], ["current frame"])
            reply = model.generate(call)
            self.assertEqual(reply.completion, "<|ACTION|>go<|END|>")
            self.assertEqual((reply.input_tokens, reply.cached_input_tokens, reply.cache_write_tokens, reply.output_tokens),
                             (34, 20, 3, 6))
            self.assertEqual(ledger.snapshot()["input_tokens"], 34)
            request = post.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.anthropic.com/v1/messages")
            self.assertEqual(request.get_header("X-api-key"), "offline")
            self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
            body = json.loads(request.data)
            self.assertEqual(set(body), {"model", "system", "messages", "stream", "max_tokens"})
            cache = {"type": "ephemeral"}
            self.assertEqual(body["system"], [{"type": "text", "text": "Use ⟦ACTION⟧", "cache_control": cache}])
            previous = body["messages"][-2]["content"][-1]
            self.assertEqual(previous, {"type": "text", "text": "⟦READ⟧notes⟦END⟧", "cache_control": cache})
            frame = body["messages"][-1]["content"]
            self.assertEqual([part.get("text") for part in frame[:-1]], ["Choose.", "current frame"])
            self.assertEqual(frame[-1]["source"]["media_type"], "image/png")
            with Image.open(io.BytesIO(base64.b64decode(frame[-1]["source"]["data"]))) as image:
                self.assertEqual(image.size, (2, 3))

    def test_truncated_stream_obeys_attempt_limit(self):
        config = {"model": {"provider": "anthropic", "max_attempts": 1, "base_url": "https://model.invalid/v1"}}
        with patch.object(transport, "load_key", return_value="offline"), \
                patch.object(transport.urllib.request, "urlopen", return_value=Response(b"")) as post:
            sender = transport.make_sender(config)
            self.assertEqual(sender.url, "https://model.invalid/v1/messages")
            with self.assertRaises(transport.TransientError):
                sender({"model": "haiku", "max_completion_tokens": 12, "messages": []})
            post.assert_called_once()
