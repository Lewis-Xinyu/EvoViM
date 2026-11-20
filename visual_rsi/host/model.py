"""Model-call DTOs shared by the trusted host and offline model stubs."""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


class BudgetExhausted(RuntimeError):
    """A model request would exceed the authorized evaluation budget."""


@dataclass(frozen=True)
class ModelCall:
    phase: str
    step: int
    call_index: int
    messages: list[dict[str, str]]
    images: list[Image.Image] = field(default_factory=list)
    # One text label per image, same order; the paid transport places each
    # label immediately before its image so the model can tell them apart.
    image_labels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelReply:
    completion: str
    input_tokens: int = 0
    output_tokens: int = 0
    model_id: str = "stub"
    stop_reason: str = "stop"
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    wall_seconds: float = 0.0
