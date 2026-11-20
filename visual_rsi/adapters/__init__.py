"""Trusted adapter registry."""
from __future__ import annotations

from functools import partial
from typing import Callable

from .base import EnvAdapter
from .benchmark import PROFILES, BenchmarkAdapter
from .gridmemory import GridMemoryAdapter

_REGISTRY: dict[str, Callable[[dict], EnvAdapter]] = {}


def register_adapter(name: str, factory: Callable[[dict], EnvAdapter]) -> None:
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError("invalid adapter registration")
    _REGISTRY[name] = factory


def get_adapter(name: str, config: dict | None = None) -> EnvAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"unknown adapter: {name}")
    return _REGISTRY[name](config or {})


def registered_adapters() -> tuple[str, ...]:
    return tuple(_REGISTRY)


register_adapter("gridmemory", lambda cfg: GridMemoryAdapter(size=int(cfg.get("size", 7)),
                                                             max_steps=int(cfg.get("max_steps", 32))))
for name in PROFILES:
    register_adapter(name, partial(BenchmarkAdapter, name))
