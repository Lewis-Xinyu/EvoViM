"""Episode model transport, retries and per-attempt accounting."""
from __future__ import annotations

from functools import partial

import base64
import io
import http.client
import json
import ssl
import os
import random
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.minimax.cn/v1"
# Transient errors retry until recovery or the evaluation budget is exhausted.
MAX_TRANSIENT_RETRIES: int | None = None
BACKOFF_CAP_SECONDS = 60
_cooldown_lock = threading.Lock()
_cooldown_until: dict[str, float] = {}


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter so parallel workers do not retry in lockstep."""
    return min(2 ** min(attempt, 10), BACKOFF_CAP_SECONDS) * random.uniform(0.5, 1.0)


def _wait_for_provider(url: str) -> None:
    """Share a 429 cooldown across episode threads using the same endpoint."""
    while True:
        with _cooldown_lock:
            deadline = _cooldown_until.get(url, 0.0)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(remaining)
        with _cooldown_lock:
            if _cooldown_until.get(url, 0.0) <= deadline:
                _cooldown_until.pop(url, None)
                return


def _delay_provider(url: str, seconds: float) -> None:
    with _cooldown_lock:
        _cooldown_until[url] = max(_cooldown_until.get(url, 0.0), time.monotonic() + seconds)


class ProviderError(RuntimeError):
    pass


class TransientError(ProviderError):
    pass


def load_key(config: dict) -> str:
    model = config.get("model", {})
    default = {"minimax": "MINIMAX_API_KEY", "openai": "OPENAI_API_KEY",
               "anthropic": "ANTHROPIC_API_KEY"}[model.get("provider", "minimax")]
    name = model.get("api_key_env", default)
    if not isinstance(name, str) or not name.isidentifier() or not name.isascii():
        raise ProviderError("model.api_key_env must name an environment variable")
    key = os.environ.get(name)
    if not key:
        raise ProviderError(f"Set {name} for the configured episode model")
    return key


def stream_data(response, timeout):
    """Yield SSE data records until the request deadline, including through heartbeat traffic."""
    deadline = time.monotonic() + 2 * timeout
    for raw_line in response:
        if time.monotonic() > deadline:
            raise TimeoutError(f"stream exceeded {2 * timeout:.0f} s without finishing")
        line = raw_line.decode("utf-8").strip()
        if line.startswith("data:"):
            yield line[5:].strip()


def http_post(url: str, headers: dict[str, str], body: bytes, timeout: float, *,
              read_response=None) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = read_response(response, timeout) if read_response else response.read()
            return response.status, data, dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)
    except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException,
            UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransientError(f"{type(exc).__name__}: {exc}") from exc


def _read_chat_stream(response, timeout):
    if "text/event-stream" not in (response.headers.get("Content-Type") or ""):
        return response.read()
    content = []
    answer = {"object": "chat.completion"}
    finish = None
    for data in stream_data(response, timeout):
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        for key in ("id", "model", "created", "system_fingerprint"):
            if chunk.get(key) is not None:
                answer.setdefault(key, chunk[key])
        if chunk.get("usage"):
            answer["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    else:
        raise http.client.IncompleteRead("".join(content).encode("utf-8"))
    answer["choices"] = [{"index": 0, "finish_reason": finish,
                          "message": {"role": "assistant", "content": "".join(content)}}]
    return json.dumps(answer).encode("utf-8")


def http_post_stream(url, headers, body, timeout):
    return http_post(url, headers, body, timeout, read_response=_read_chat_stream)


MAX_USAGE_RETRIES = 4

# Some upstream relays occasionally reject an otherwise supported request.
UPSTREAM_ROUTING_400 = ("does not support reasoning effort",)
MAX_ROUTING_RETRIES = 40


def is_upstream_routing_fault(status: int, raw: bytes) -> bool:
    if status != 400:
        return False
    text = raw[:2000].decode("utf-8", errors="replace")
    unsupported_none = "unsupported_value" in text and "'none' is not supported" in text
    codex_account = "not supported when using Codex with a ChatGPT account" in text
    model_not_found = "model_not_found" in text and "unknown provider for model" in text
    return (unsupported_none or model_not_found or codex_account
            or ("upstream_400" in text and any(marker in text for marker in UPSTREAM_ROUTING_400)))


def make_sender(config: dict):
    model = config.get("model", {})
    provider = model.get("provider", "minimax")
    if provider not in {"minimax", "openai", "anthropic"}:
        raise ProviderError("unsupported model provider")
    timeout = float(model.get("timeout", 90))
    max_attempts = model.get("max_attempts")
    if max_attempts is not None and (type(max_attempts) is not int or max_attempts < 1):
        raise ValueError("model.max_attempts must be a positive integer")
    key = load_key(config)
    stream = bool(model.get("stream", False))
    if provider == "anthropic":
        from . import anthropic
        base = str(model.get("base_url") or "https://api.anthropic.com").rstrip("/")
        url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        headers = {"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"}
        post = partial(http_post, read_response=anthropic.read_response)
    else:
        base = "https://api.openai.com/v1" if provider == "openai" else DEFAULT_BASE_URL
        url = str(model.get("base_url") or base).rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        post = http_post_stream if stream else http_post

    def notify(event, attempt, **details):
        if sender.attempt_hook is not None:
            sender.attempt_hook(event, attempt, details)

    def wait(attempt, reason, delay):
        notify("waiting_for_provider", attempt, reason=reason, wait_seconds=delay,
               next_attempt_utc=time.time() + delay)
        time.sleep(delay)
        notify("provider_resumed", attempt)

    def sender(payload: dict) -> dict:
        if provider == "anthropic":
            payload = anthropic.messages_request(payload, cloudrouter="cloudrouter" in url)
        elif stream:
            payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        attempt = 0
        usage_retries = routing_retries = 0
        while True:
            attempt += 1
            at_limit = max_attempts is not None and attempt >= max_attempts
            last = at_limit or (MAX_TRANSIENT_RETRIES is not None and attempt >= MAX_TRANSIENT_RETRIES)
            with _cooldown_lock:
                remaining = max(0.0, _cooldown_until.get(url, 0.0) - time.monotonic())
            if remaining:
                notify("waiting_for_provider", attempt, reason="shared HTTP 429 cooldown",
                       wait_seconds=remaining, next_attempt_utc=time.time() + remaining)
            _wait_for_provider(url)
            if remaining:
                notify("provider_resumed", attempt)
            notify("before", attempt)
            try:
                posted = post(url, headers, body, timeout)
            except TransientError as exc:
                sender.failed_attempts += 1
                notify("unknown", attempt, reason=f"transport error: {exc}")
                if last:
                    raise
                wait(attempt, "transport error", backoff_seconds(attempt - 1))
                continue

            status, raw = posted[:2]
            response_headers = posted[2] if len(posted) > 2 else {}
            if status == 429 or status >= 500:
                sender.failed_attempts += 1
                reason = f"HTTP {status}"
                notify("unknown", attempt, reason=reason)
                if last:
                    raise ProviderError(reason)
                delay = backoff_seconds(attempt - 1)
                if status == 429:
                    _delay_provider(url, delay)
                else:
                    wait(attempt, reason, delay)
                continue
            if not at_limit and is_upstream_routing_fault(status, raw) and routing_retries < MAX_ROUTING_RETRIES:
                routing_retries += 1
                sender.failed_attempts += 1
                reason = f"HTTP {status} upstream routing fault"
                notify("unknown", attempt, reason=reason)
                wait(attempt, reason, backoff_seconds(routing_retries + 1))
                continue
            if status >= 400:
                notify("unknown", attempt, reason=f"HTTP {status}")
                raise ProviderError(f"HTTP {status}: {raw[:300].decode('utf-8', errors='replace')}")
            answer = json.loads(raw.decode("utf-8"))
            if not at_limit and not answer.get("usage") and usage_retries < MAX_USAGE_RETRIES:
                usage_retries += 1
                sender.failed_attempts += 1
                reason = "response without usage"
                notify("unknown", attempt, reason=reason)
                wait(attempt, reason, backoff_seconds(attempt - 1))
                continue
            answer["_rate_limits"] = {k.lower(): v for k, v in response_headers.items()
                                      if k.lower().startswith("x-ratelimit-")}
            return answer

    sender.failed_attempts = 0
    sender.attempt_hook = None
    sender.url = url
    return sender


def image_data_url(image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
