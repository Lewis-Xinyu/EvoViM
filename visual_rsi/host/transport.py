"""OpenAI-compatible chat-completions transport for episode model requests."""
from __future__ import annotations

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
# Connection errors, 429 and 5xx are retried with exponential backoff, matching
# upstream AutoMem's client (``delay * 2 ** (retries - 1)``).  A fixed one-second
# retry could not absorb a provider outage: it burned every bounded physical
# attempt within seconds and left whole families unmeasurable.
# None: keep retrying connection errors, 429 and 5xx until the provider answers.
# Many lineages share one token plan, so a rate limit means "wait", not "fail";
# a finite six-try budget (about 30 s) turned hour-long 429 spells into lost units.
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
    default = "MINIMAX_API_KEY" if model.get("provider", "minimax") == "minimax" else "OPENAI_API_KEY"
    name = model.get("api_key_env", default)
    if not isinstance(name, str) or not name.isidentifier() or not name.isascii():
        raise ProviderError("model.api_key_env must name an environment variable")
    key = os.environ.get(name)
    if not key:
        raise ProviderError(f"Set {name} for the configured episode model")
    return key


def http_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)
    except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException) as exc:
        # ssl.SSLError ("[SSL] record layer failure") and IncompleteRead surface while
        # reading the body, outside URLError; on 2026-09-30/10-01 they voided whole VSI
        # evaluations instead of being retried like any other transport failure.
        raise TransientError(f"{type(exc).__name__}: {exc}") from exc


def http_post_stream(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes, dict[str, str]]:
    """POST a ``stream: true`` request and reassemble the SSE chunks into one chat completion.

    Streaming keeps connections active while the provider computes a response.
    """
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if "text/event-stream" not in (response.headers.get("Content-Type") or ""):
                return response.status, response.read(), dict(response.headers)
            content: list[str] = []
            answer: dict = {"object": "chat.completion"}
            finish = None
            done = False
            # 2026-10-10: a relay can keep a dead stream alive with non-data lines, so the socket
            # timeout never fires (two probe episodes hung 15+ min).  Bound the whole stream.
            deadline = time.monotonic() + 2 * timeout
            for raw_line in response:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"stream exceeded {2 * timeout:.0f} s without finishing")
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
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
            if not done:
                # A stream cut before [DONE] is a transport failure, never a short answer.
                raise http.client.IncompleteRead("".join(content).encode("utf-8"))
            answer["choices"] = [{"index": 0, "finish_reason": finish,
                                  "message": {"role": "assistant", "content": "".join(content)}}]
            return response.status, json.dumps(answer).encode("utf-8"), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)
    except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException,
            UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TransientError(f"{type(exc).__name__}: {exc}") from exc


MAX_USAGE_RETRIES = 4

# Some upstream relays occasionally reject an otherwise supported request.
UPSTREAM_ROUTING_400 = ("does not support reasoning effort",)
MAX_ROUTING_RETRIES = 40


def is_upstream_routing_fault(status: int, raw: bytes) -> bool:
    if status != 400:
        return False
    text = raw[:2000].decode("utf-8", errors="replace")
    # 2026-10-10: the same fault also arrives without "upstream_400", as
    # {"code":"unsupported_value","message":"Unsupported value: 'none' is not supported with the ..."}.
    unsupported_none = "unsupported_value" in text and "'none' is not supported" in text
    codex_account = "not supported when using Codex with a ChatGPT account" in text
    model_not_found = "model_not_found" in text and "unknown provider for model" in text
    return unsupported_none or model_not_found or codex_account or ("upstream_400" in text and any(marker in text for marker in UPSTREAM_ROUTING_400))

def make_sender(config: dict):
    model = config.get("model", {})
    provider = model.get("provider", "minimax")
    if provider not in {"minimax", "openai"}:
        raise ProviderError("unsupported model provider")
    base_url = "https://api.openai.com/v1" if provider == "openai" else DEFAULT_BASE_URL
    url = str(model.get("base_url") or base_url).rstrip("/") + "/chat/completions"
    timeout = float(model.get("timeout", 90))
    key = load_key(config)
    stream = bool(model.get("stream", False))
    post = http_post_stream if stream else http_post

    def sender(payload: dict) -> dict:
        text = json.dumps(payload, ensure_ascii=False)
        if stream:
            # Transport-only (2026-10-09): the same request, answered as SSE; see http_post_stream.
            streamed = json.loads(text)
            streamed.update(stream=True, stream_options={"include_usage": True})
            text = json.dumps(streamed, ensure_ascii=False)
        body = text.encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        attempt = -1
        usage_retries = 0
        routing_retries = 0
        while True:
            attempt += 1
            last = MAX_TRANSIENT_RETRIES is not None and attempt >= MAX_TRANSIENT_RETRIES - 1
            with _cooldown_lock:
                remaining = max(0.0, _cooldown_until.get(url, 0.0) - time.monotonic())
            if remaining and sender.attempt_hook is not None:
                sender.attempt_hook("waiting_for_provider", attempt + 1,
                                    {"reason": "shared HTTP 429 cooldown", "wait_seconds": remaining,
                                     "next_attempt_utc": time.time() + remaining})
            _wait_for_provider(url)
            if remaining and sender.attempt_hook is not None:
                sender.attempt_hook("provider_resumed", attempt + 1, {})
            if sender.attempt_hook is not None:
                sender.attempt_hook("before", attempt + 1, {})
            try:
                posted = post(url, headers, body, timeout)
                status, raw = posted[:2]
                response_headers = posted[2] if len(posted) > 2 else {}
            except TransientError as exc:
                sender.failed_attempts += 1
                if sender.attempt_hook is not None:
                    sender.attempt_hook("unknown", attempt + 1, {"reason": f"transport error: {exc}"})
                if last:
                    raise
                delay = backoff_seconds(attempt)
                if sender.attempt_hook is not None:
                    sender.attempt_hook("waiting_for_provider", attempt + 1,
                                        {"reason": "transport error", "wait_seconds": delay,
                                         "next_attempt_utc": time.time() + delay})
                time.sleep(delay)
                if sender.attempt_hook is not None:
                    sender.attempt_hook("provider_resumed", attempt + 1, {})
                continue
            if status == 429:
                sender.failed_attempts += 1
                if sender.attempt_hook is not None:
                    sender.attempt_hook("unknown", attempt + 1, {"reason": f"HTTP {status}"})
                if last:
                    raise ProviderError(f"HTTP {status}")
                delay = backoff_seconds(attempt)
                _delay_provider(url, delay)
                continue
            if status >= 500:
                sender.failed_attempts += 1
                if sender.attempt_hook is not None:
                    sender.attempt_hook("unknown", attempt + 1, {"reason": f"HTTP {status}"})
                if last:
                    raise ProviderError(f"HTTP {status}")
                delay = backoff_seconds(attempt)
                if sender.attempt_hook is not None:
                    sender.attempt_hook("waiting_for_provider", attempt + 1,
                                        {"reason": f"HTTP {status}", "wait_seconds": delay,
                                         "next_attempt_utc": time.time() + delay})
                time.sleep(delay)
                if sender.attempt_hook is not None:
                    sender.attempt_hook("provider_resumed", attempt + 1, {})
                continue
            if is_upstream_routing_fault(status, raw) and routing_retries < MAX_ROUTING_RETRIES:
                routing_retries += 1
                sender.failed_attempts += 1
                if sender.attempt_hook is not None:
                    sender.attempt_hook("unknown", attempt + 1, {"reason": f"HTTP {status} upstream routing fault"})
                delay = backoff_seconds(min(routing_retries + 1, 10))
                if sender.attempt_hook is not None:
                    sender.attempt_hook("waiting_for_provider", attempt + 1,
                                        {"reason": f"HTTP {status} upstream routing fault", "wait_seconds": delay,
                                         "next_attempt_utc": time.time() + delay})
                time.sleep(delay)
                if sender.attempt_hook is not None:
                    sender.attempt_hook("provider_resumed", attempt + 1, {})
                continue
            if status >= 400:
                if sender.attempt_hook is not None:
                    sender.attempt_hook("unknown", attempt + 1, {"reason": f"HTTP {status}"})
                raise ProviderError(f"HTTP {status}: {raw[:300].decode('utf-8', errors='replace')}")
            answer = json.loads(raw.decode("utf-8"))
            if not answer.get("usage") and usage_retries < MAX_USAGE_RETRIES:
                # Retry responses missing the provider's token accounting and record each attempt.
                usage_retries += 1
                sender.failed_attempts += 1
                if sender.attempt_hook is not None:
                    sender.attempt_hook("unknown", attempt + 1, {"reason": "response without usage"})
                delay = backoff_seconds(attempt)
                if sender.attempt_hook is not None:
                    sender.attempt_hook("waiting_for_provider", attempt + 1,
                                        {"reason": "response without usage", "wait_seconds": delay,
                                         "next_attempt_utc": time.time() + delay})
                time.sleep(delay)
                if sender.attempt_hook is not None:
                    sender.attempt_hook("provider_resumed", attempt + 1, {})
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
