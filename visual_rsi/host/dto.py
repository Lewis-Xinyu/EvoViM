"""Positive public-observation allow-list and containment errors (host side)."""
from __future__ import annotations

import re
from typing import Any, Iterable


class ContainmentError(RuntimeError):
    """The candidate process failed or broke a host boundary; the decision is void."""


class CandidateHung(ContainmentError):
    """The candidate computed past the backstop without answering the host (an infinite loop, in practice)."""


class Violation(ContainmentError):
    """The candidate broke a host rule (phase order, caps, action commitment, modality)."""


def validate_frame_id(frame_id: str) -> None:
    if not isinstance(frame_id, str) or not re.fullmatch(r"f\d{5}", frame_id):
        raise ContainmentError("invalid frame id")


def validate_public_obs(raw: dict[str, Any], allowed_fields: Iterable[str], *, line: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) - {"text", "image", "frame_id"}:
        raise ContainmentError("observation contains non-public fields")
    text = raw.get("text")
    allowed = set(allowed_fields)
    if not isinstance(text, dict) or set(text) - allowed:
        raise ContainmentError("observation text field is not in adapter allow-list")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in text.items()):
        raise ContainmentError("public text fields must be strings")
    image = raw.get("image")
    frame_id = raw.get("frame_id")
    if line == "T" and (image is not None or frame_id is not None):
        raise ContainmentError("T line cannot receive images or frame ids")
    if line == "V" and image is not None:
        raise ContainmentError("V candidate cannot receive pixels")
    if line == "M" and image is not None and not hasattr(image, "convert"):
        raise ContainmentError("M image must be a PIL-compatible image")
    if frame_id is not None:
        validate_frame_id(frame_id)
    return {"text": dict(text), **({"image": image} if image is not None else {}),
            **({"frame_id": frame_id} if frame_id is not None else {})}
