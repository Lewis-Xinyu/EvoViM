"""Translate episode messages and Claude command markers at the provider boundary."""

import http.client
import json
import re

from .transport import ProviderError, stream_data


_COMMAND = re.compile(r"<\|([A-Za-z_]*)\|>")
_CLAUDE_COMMAND = re.compile(r"⟦([A-Za-z_]*)⟧")
_ESCAPED_COMMAND = re.compile(r"\\u27[eE]6([A-Za-z_]*)\\u27[eE]7")


def _parts(content, cloudrouter):
    parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
    for part in parts:
        if part["type"] == "text":
            text = part.get("text") or ""
            if text.strip():
                if cloudrouter:
                    text = text.replace("Overwrite a file entirely", "Replace the whole file")
                yield {"type": "text", "text": _COMMAND.sub(r"⟦\1⟧", text)}
        elif part["type"] == "image_url":
            url = part["image_url"]["url"]
            match = re.fullmatch(r"data:(image/[A-Za-z0-9.+-]+);base64,(.*)", url, re.S)
            if not match:
                raise ProviderError("Anthropic image content requires a base64 data URL")
            yield {"type": "image", "source": {"type": "base64", "media_type": match[1], "data": match[2]}}
        else:
            raise ProviderError(f"Unsupported Anthropic message part: {part['type']}")


def messages_request(payload, *, cloudrouter=False):
    system, turns = [], []
    for message in payload["messages"]:
        role = message["role"]
        parts = list(_parts(message.get("content"), cloudrouter))
        if role == "system":
            system.extend(parts)
            continue
        if role not in {"user", "assistant"}:
            raise ProviderError(f"Unsupported Anthropic message role: {role}")
        parts = parts or [{"type": "text", "text": "(empty)"}]
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(parts)
        else:
            turns.append({"role": role, "content": parts})
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": [{"type": "text", "text": "(empty)"}]})

    # The final user turn contains the current frame. Cache only the shared prefix.
    maximum = int(payload.get("max_completion_tokens") or payload.get("max_tokens") or 4096)
    body = {"model": payload["model"], "max_tokens": maximum,
            "messages": turns, "stream": True}
    if system:
        system[-1]["cache_control"] = {"type": "ephemeral"}
        body["system"] = system
    for turn in reversed(turns[:-1]):
        if turn["role"] == "assistant":
            turn["content"][-1]["cache_control"] = {"type": "ephemeral"}
            break
    for key in ("temperature", "top_p", "stop_sequences", "thinking"):
        if key in payload:
            body[key] = payload[key]
    return body


def _message_stream(response, timeout):
    message, blocks = {}, {}
    for data in stream_data(response, timeout):
        event = json.loads(data)
        kind = event["type"]
        if kind == "message_start":
            message = event["message"]
        elif kind == "content_block_start":
            blocks[event["index"]] = event["content_block"]
        elif kind == "content_block_delta" and event["delta"]["type"] == "text_delta":
            block = blocks.setdefault(event["index"], {"type": "text", "text": ""})
            block["text"] += event["delta"]["text"]
        elif kind == "message_delta":
            message.update({k: v for k, v in event["delta"].items() if v is not None})
            message.setdefault("usage", {}).update({k: v for k, v in event.get("usage", {}).items() if v is not None})
        elif kind == "message_stop":
            message["content"] = [blocks[index] for index in sorted(blocks)]
            return message
        elif kind == "error":
            raise http.client.HTTPException(f"Anthropic stream error: {event.get('error')}")
    raise http.client.IncompleteRead(b"")


def read_response(response, timeout):
    if "text/event-stream" in (response.headers.get("Content-Type") or ""):
        message = _message_stream(response, timeout)
    else:
        message = json.loads(response.read())
    text = "".join(b["text"] for b in message.get("content", []) if b["type"] == "text")
    text = _ESCAPED_COMMAND.sub(r"<|\1|>", _CLAUDE_COMMAND.sub(r"<|\1|>", text))
    stop = message.get("stop_reason")
    finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}.get(stop, stop or "stop")
    answer = {"id": message.get("id"), "model": message.get("model"), "object": "chat.completion",
              "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}]}
    if usage := message.get("usage"):
        read = int(usage.get("cache_read_input_tokens") or 0)
        written = int(usage.get("cache_creation_input_tokens") or 0)
        inputs = int(usage.get("input_tokens") or 0) + read + written
        outputs = int(usage.get("output_tokens") or 0)
        thinking = int((usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
        answer["usage"] = {"prompt_tokens": inputs, "completion_tokens": outputs, "total_tokens": inputs + outputs,
                           "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": written},
                           "completion_tokens_details": {"reasoning_tokens": thinking}}
    return json.dumps(answer).encode("utf-8")
