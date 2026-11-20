"""Regression checks for candidate installation, promotion and execution."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from EvoViM.host import transport
from EvoViM.host.bundle import install_bundle, verify_bundle
from EvoViM.host.child.registry import ProtocolError, Registry, Tool, decode
from EvoViM.host.dto import ContainmentError
from EvoViM.host.protocol_text import host_decode
from EvoViM.host.repair import Promotion


class RuntimeTest(unittest.TestCase):
    def test_failed_install_can_be_retried_and_cannot_rehash_a_changed_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "candidate.py").write_text("value = 1\n")
            (source / ".DS_Store").write_text("desktop metadata")
            parent = install_bundle(root / "parent", source)
            self.assertFalse((parent / ".DS_Store").exists())
            child = root / "child"
            with self.assertRaises(SyntaxError):
                install_bundle(child, parent, {"candidate.py": "value ="})
            self.assertFalse(child.exists())
            install_bundle(child, parent, {"candidate.py": "value = 2"})
            self.assertNotEqual(verify_bundle(parent), verify_bundle(child))
            (parent / "candidate.py").write_text("value = 3")
            with self.assertRaises(ContainmentError):
                install_bundle(root / "corrupt", parent)

    def test_can_resume_promotion_after_move_or_pointer_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "candidate.py").write_text("value = 1")
            promotion = Promotion(root)
            record = promotion.stage("g1", {"version_hash": "g0"}, source, root / "evidence", {}, {})
            final = promotion.versions / record["candidate_hash"]
            Path(record["staging"]).rename(final)
            committed = promotion.commit("g1")
            self.assertEqual(promotion.resolve_parent(), final / "bundle")
            (promotion.journal / "g1.json").write_text(json.dumps(record))
            self.assertEqual(promotion.commit("g1"), committed)

    def test_host_and_candidate_parse_the_same_command_syntax(self):
        registry = Registry()
        schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
        registry.register(Tool("COUNTER", "", schema, lambda n: n, ("PLAN",)))
        actions = {"kind": "free_text", "max_length": 24}
        for phase, command in [
            ("LOG", "<|DONE|>"),
            ("PLAN", "<|ACTION|>open east<|END|>"),
            ("PLAN", '<|COUNTER|>{"n":3}<|END|>'),
        ]:
            self.assertEqual(decode(command, phase, registry, actions)[0],
                             host_decode(command, phase, registry.specs(), actions)[0])
        for phase, command in [
            ("PLAN", "<|DONE|><|END|>"),
            ("PLAN", "<|DONE|>"),
            ("PLAN", "<|ACTION|>two\nlines<|END|>"),
            ("LOG", "<|COUNTER|>{}<|END|>"),
            ("PLAN", "<|ACTION|>look<|END|><|ACTION|>go<|END|>"),
        ]:
            self.assertIsNone(host_decode(command, phase, registry.specs(), actions))
            with self.assertRaises(ProtocolError):
                decode(command, phase, registry, actions)

    def test_transient_retries_record_each_attempt(self):
        responses = [(503, b"", {}), (200, b"{}", {}), (200, b'{"usage":{"prompt_tokens":2}}', {})]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline"}), \
             patch.object(transport, "http_post", side_effect=responses), \
             patch.object(transport.time, "sleep"):
            sender = transport.make_sender({"model": {"provider": "openai"}})
            events = []
            sender.attempt_hook = lambda event, attempt, detail: events.append((event, attempt))
            self.assertEqual(sender({})["usage"], {"prompt_tokens": 2})
            self.assertEqual([n for event, n in events if event == "before"], [1, 2, 3])
            self.assertEqual([n for event, n in events if event == "unknown"], [1, 2])

    def test_attempt_limit_also_covers_routing_and_missing_usage(self):
        recoverable = (400, b"upstream_400 does not support reasoning effort", {})
        response_without_usage = (200, b"{}", {})
        for last in [recoverable, response_without_usage, (429, b"limited", {}), (503, b"busy", {})]:
            responses = [recoverable, response_without_usage, last]
            with self.subTest(last=last), patch.object(transport, "load_key", return_value="offline"), \
                 patch.object(transport, "http_post", side_effect=responses) as post, \
                 patch.object(transport.time, "sleep"):
                sender = transport.make_sender({"model": {"provider": "openai", "max_attempts": 3}})
                if last == response_without_usage:
                    self.assertNotIn("usage", sender({}))
                else:
                    with self.assertRaises(transport.ProviderError):
                        sender({})
                self.assertEqual(post.call_count, 3)
