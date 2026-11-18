"""Codex subscription Meta launcher. No Meta call is made by importing this module."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping
from urllib.parse import urlsplit


def _route(config: Mapping[str, Any]) -> str:
    return str(config.get("meta", {}).get("codex", {}).get("route", "subscription"))


def _api_key_env(meta: Mapping[str, Any]) -> str:
    name = str(meta.get("api_key_env", "OPENAI_API_KEY"))
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("meta.codex.api_key_env must name an environment variable")
    return name


def command(config: Mapping[str, Any], workspace: Path) -> list[str]:
    meta = config.get("meta", {}).get("codex", {})
    route = _route(config)
    provider = []
    if route == "api":
        base_url = str(meta.get("base_url", "https://api.openai.com/v1"))
        url = urlsplit(base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("meta.codex.base_url must be an HTTP(S) endpoint without credentials or a query")
        if not meta.get("model"):
            raise ValueError("meta.codex.model is required for an API route")
        name = _api_key_env(meta)
        provider = ["--config", 'model_provider="configured"',
                    "--config", 'model_providers.configured.name="Configured endpoint"',
                    "--config", f'model_providers.configured.base_url={json.dumps(base_url)}',
                    "--config", f'model_providers.configured.env_key={json.dumps(name)}',
                    "--config", 'model_providers.configured.wire_api="responses"',
                    "--config", f'shell_environment_policy.exclude={json.dumps([name])}']
    elif route != "subscription":
        raise ValueError("meta.codex.route must be subscription or api")
    model_args = ["--model", str(meta["model"])] if meta.get("model") else []
    effort = json.dumps(str(meta.get("effort", "high")))
    output = Path(workspace).resolve() / "output"
    if not output.is_dir() or not (output.parent / "input").is_dir():
        raise ValueError("Meta workspace needs input/ and output/")
    return [str(meta.get("binary") or "codex"), "exec", "--json", "--ephemeral",
            "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
            *model_args, *provider, "--sandbox", "workspace-write",
            "--cd", str(output), "--config", f"model_reasoning_effort={effort}",
            "--config", "sandbox_workspace_write.network_access=false", "-"]


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Meta input may not contain symlinks")
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def parse_usage(transcript: str) -> dict[str, int]:
    """Sum Codex turn usage, preserving missing fields as zero for stub validation."""
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
              "reasoning_tokens": 0}
    for line in transcript.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "turn.completed":
            continue
        usage = row.get("usage") or {}
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["cached_input_tokens"] += int(usage.get("cached_input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["reasoning_tokens"] += int(usage.get("reasoning_output_tokens") or
                                         usage.get("reasoning_tokens") or 0)
    return totals


def _safe_env(config: Mapping[str, Any]) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(Path.home()),
           "LANG": os.environ.get("LANG", "C")}
    if _route(config) == "api":
        name = _api_key_env(config["meta"]["codex"])
        if not os.environ.get(name):
            raise PermissionError(f"Set {name} for the configured Codex endpoint")
        env[name] = os.environ[name]
    return env


def usage_cost_usd(usage: Mapping[str, int], prices: Mapping[str, float]) -> float | None:
    if not prices:
        return None
    cached = int(usage.get("cached_input_tokens") or 0)
    ordinary = max(int(usage.get("input_tokens") or 0) - cached, 0)
    return round((ordinary * prices["input"] + cached * prices.get("cached_input", prices["input"])
                  + int(usage.get("output_tokens") or 0) * prices["output"]) / 1e6, 6)


def run_meta_slot(config: Mapping[str, Any], run_dir: Path, workspace: Path, out_dir: Path, *,
                  slot_id: str, preflight_marker: Path | None = None,
                  runner=subprocess.run) -> dict[str, Any]:
    workspace, out_dir = Path(workspace).resolve(), Path(out_dir)
    if (out_dir / "meta.launch.json").exists():
        raise PermissionError("Meta slot has already been launched")
    if runner is subprocess.run:
        marker = Path(preflight_marker or Path(run_dir) / "codex_preflight_passed.json")
        if not marker.is_file() or not json.loads(marker.read_text()).get("passed"):
            raise PermissionError("real Codex Meta requires a passed isolation preflight")
    argv = command(config, workspace)
    route = _route(config)
    before = _tree_hash(workspace / "input")
    prompt = (workspace / "prompt.md").read_text(encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    (out_dir / "meta.launch.json").write_text(json.dumps({"argv": argv, "slot": slot_id,
        "input_sha256": before,
        "auth": "Configured environment variable" if route == "api" else "ChatGPT subscription",
        "network": "disabled"}, indent=2) + "\n")
    try:
        process = runner(argv, input=prompt, capture_output=True, text=True, cwd=str(workspace / "output"),
                         env=_safe_env(config), timeout=float(config.get("meta", {}).get("timeout_minutes", 60)) * 60)
        transcript, stderr = process.stdout, process.stderr
        status = "completed" if process.returncode == 0 else "generation_failed"
    except subprocess.TimeoutExpired as exc:
        transcript = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        status = "timed_out"
    elapsed = time.monotonic() - started
    (out_dir / "meta.stream.jsonl").write_text(transcript)
    (out_dir / "meta.stderr.txt").write_text(stderr)
    if _tree_hash(workspace / "input") != before:
        status = "isolation_failed"
    usage = parse_usage(transcript)
    result = {"status": status, "slot": slot_id, "usage": usage,
              "wall_seconds": round(elapsed, 3),
              "cost_usd": usage_cost_usd(usage, config["meta"]["codex"].get("price_per_million", {})),
              "billing_basis": "Configured API prices" if route == "api" else "Codex subscription",
              "input_sha256": before}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_preflight(config: Mapping[str, Any], run_dir: Path, runner=subprocess.run) -> dict[str, Any]:
    """Real preflight is paid through the Codex subscription; never called in stage 0."""
    workspace = Path(run_dir) / "codex_preflight" / "workspace"
    inputs, output = workspace / "input", workspace / "output"
    inputs.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    probe = inputs / "read_only_probe.txt"
    probe.write_text("CAN_READ_INPUT\n")
    probe.chmod(0o444)
    script = output / "preflight.sh"
    script.write_text("""#!/bin/sh
set -u
if [ \"$(cat ../input/read_only_probe.txt 2>/dev/null)\" = CAN_READ_INPUT ]; then
  read_ok=true
else
  read_ok=false
fi
printf CAN_WRITE_OUTPUT > output_probe.txt
if printf SHOULD_NOT_WRITE >> ../input/read_only_probe.txt 2>/dev/null; then
  input_denied=false
else
  input_denied=true
fi
if /usr/bin/curl -fsSI --max-time 3 https://example.com >/dev/null 2>&1; then
  network_denied=false
else
  network_denied=true
fi
printf '{\"input_read\":%s,\"input_write_denied\":%s,\"network_denied\":%s}\\n' \\
  \"$read_ok\" \"$input_denied\" \"$network_denied\" > checks.json
""")
    prompt = ("Run /bin/sh ./preflight.sh in the current directory. Report its result. "
              "Do not create checks.json yourself; the script must write it.")
    (workspace / "prompt.md").write_text(prompt)
    argv = command(config, workspace)
    before = _tree_hash(inputs)
    result = runner(argv, input=prompt, capture_output=True, text=True, cwd=str(output),
                    env=_safe_env(config), timeout=300)
    transcript = result.stdout
    (workspace / "preflight.stream.jsonl").write_text(transcript)
    script_checks = json.loads((output / "checks.json").read_text()) if (output / "checks.json").exists() else {}
    checks = {"output_write": (output / "output_probe.txt").read_text().strip() == "CAN_WRITE_OUTPUT"
              if (output / "output_probe.txt").exists() else False,
              "input_read": script_checks.get("input_read") is True,
              "input_write_denied": script_checks.get("input_write_denied") is True,
              "network_denied": script_checks.get("network_denied") is True,
              "input_unchanged": _tree_hash(inputs) == before,
              "sandbox_command": "sandbox_workspace_write.network_access=false" in argv,
              "json_trace": any('"type":"turn.completed"' in line or '"type": "turn.completed"' in line
                                for line in transcript.splitlines())}
    passed = result.returncode == 0 and all(checks.values())
    record = {"passed": passed, "checks": checks, "usage": parse_usage(transcript),
              "cost_usd": usage_cost_usd(parse_usage(transcript), config["meta"]["codex"].get("price_per_million", {})),
              "billing_basis": "Configured API prices" if _route(config) == "api" else "Codex subscription"}
    (Path(run_dir) / "codex_preflight_passed.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
