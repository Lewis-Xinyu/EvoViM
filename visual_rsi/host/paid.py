"""Authorized, budget-capped model for paid evaluation (ported from r2 reserve/settle accounting)."""
from __future__ import annotations

import copy
from functools import wraps
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid

from .episode import BudgetExhausted
from .model import ModelCall, ModelReply

AUTH_KEYS = {"batch_id", "phase", "max_units", "max_input_tokens", "max_output_tokens", "max_amount"}
IMAGE_RESERVATION = 8192


def _synchronized(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapper


def load_authorization(run_dir: Path, phase: str) -> dict:
    delegated = Path(run_dir) / "delegated_authorization.json"
    if delegated.exists():
        data = json.loads(delegated.read_text(encoding="utf-8"))
        if data.get("authorization_mode") != "delegated_workload" or data.get("scope") != "this_run_only":
            raise PermissionError("invalid delegated authorization")
        limits = data.get("limits") or {}
        required = ("max_units", "max_input_tokens", "max_output_tokens")
        if any(type(limits.get(k)) is not int or limits[k] <= 0 for k in required):
            raise PermissionError("delegated authorization has invalid workload caps")
        cap = limits.get("monetary_cap")
        if cap is not None and (type(cap) not in (int, float) or cap <= 0):
            raise PermissionError("delegated monetary cap must be null or positive")
        enforcement = data.get("enforcement", "hard")
        if enforcement not in {"hard", "record_only"}:
            raise PermissionError("invalid delegated enforcement mode")
        return {"batch_id": "delegated-" + str(data.get("manifest_sha256", "unknown"))[:16], "phase": phase,
                "max_units": limits["max_units"], "max_input_tokens": limits["max_input_tokens"],
                "max_output_tokens": limits["max_output_tokens"], "max_amount": cap,
                "issued_by": "scheduler", "basis": "explicit_user_instruction", "enforcement": enforcement}
    path = Path(run_dir) / "authorization.json"
    if not path.exists():
        raise PermissionError("paid phase requires a user-written authorization.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data) != AUTH_KEYS or data["phase"] != phase:
        raise PermissionError("authorization.json does not match the requested phase")
    for key in ("max_units", "max_input_tokens", "max_output_tokens"):
        if type(data[key]) is not int or data[key] <= 0:
            raise PermissionError("authorization caps must be positive integers")
    if type(data["max_amount"]) not in (int, float) or data["max_amount"] <= 0:
        raise PermissionError("authorization amount cap must be positive")
    return data


class Ledger:
    """Cumulative run ledger plus the current authorized batch; written after every request."""

    FIELDS = ("episodes_completed", "requests", "failed_attempts", "input_tokens", "output_tokens", "cost_yuan",
              "cached_input_tokens", "cache_write_tokens", "reasoning_tokens", "wall_seconds", "cost_usd",
              "long_context_surcharge_usd")
    ATTEMPT_FIELDS = ("reserved_attempts", "settled_attempts", "unknown_attempts",
                      "reserved_input_tokens", "reserved_output_tokens", "estimated_cost_yuan")

    # One object per ledger file in this process.  Evaluations of the same
    # environment run in parallel threads (paired remeasure, holdout seeds and
    # versions, final remeasure); separate objects each held their own counts
    # and overwrote each other's file, and shared one temp name, so a
    # concurrent rename failed the episode.  Every caller now shares one
    # object, one lock and one in-memory state.
    _instances: dict[Path, "Ledger"] = {}
    _instances_lock = threading.Lock()

    def __new__(cls, path: Path, price: dict[str, float]):
        key = Path(path).resolve()
        with cls._instances_lock:
            instance = cls._instances.get(key)
            if instance is None:
                # Keep the in-process ledger identity stable even if a concurrent
                # filesystem check briefly observes no destination during a write.
                # A new run must use its own run directory and ledger path.
                instance = super().__new__(cls)
                instance._initialised = False
                cls._instances[key] = instance
            return instance

    def __init__(self, path: Path, price: dict[str, float]):
        # __new__ publishes the shared object before Python calls __init__.
        # Initialise it under the registry lock so a second evaluation thread
        # cannot receive a half-built ledger.
        with self._instances_lock:
            if self._initialised:
                return
            self.path = Path(path)
            self.lock = threading.RLock()
            if price and (not {"input", "output"} <= price.keys() or
                          any(type(v) not in {int, float} or not math.isfinite(v) or v < 0 for v in price.values())):
                raise ValueError("price_per_million requires non-negative input and output rates")
            self.price = dict(price)
            self.data = json.loads(self.path.read_text()) if self.path.exists() else {k: 0 for k in self.FIELDS}
            for key in self.FIELDS:
                self.data.setdefault(key, 0)
            for key in self.ATTEMPT_FIELDS:
                self.data.setdefault(key, 0)
            self.data.setdefault("batches", {})
            self._initialised = True

    def cost(self, input_tokens: float, output_tokens: float, cached_input_tokens: float = 0,
             cache_write_tokens: float = 0) -> float:
        ordinary = input_tokens - cached_input_tokens - cache_write_tokens
        if ordinary < 0:
            raise ValueError("cache token counts exceed input tokens")
        if not self.price:
            raise ValueError("Monetary accounting requires budget.price_per_million")
        return (ordinary * self.price["input"] + cached_input_tokens * self.price.get("cached_input", self.price["input"])
                + cache_write_tokens * self.price.get("cache_write", self.price["input"])
                + output_tokens * self.price["output"]) / 1e6

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(self.data, indent=2) + "\n")
        tmp.replace(self.path)

    def _batch(self, batch_id: str) -> dict[str, Any]:
        value = self.data["batches"].setdefault(batch_id, {k: 0 for k in self.FIELDS})
        for key in self.FIELDS:
            value.setdefault(key, 0)
        for key in self.ATTEMPT_FIELDS:
            value.setdefault(key, 0)
        value.setdefault("units_reserved", 0)
        value.setdefault("unit_attempts", [])
        value.setdefault("request_attempts", {})
        return value

    def _recalculate(self, value: dict[str, Any]) -> None:
        if not self.price:
            for field in ("cost_usd", "cost_yuan", "estimated_cost_yuan"):
                value[field] = None
            return
        actual = round(self.cost(value["input_tokens"], value["output_tokens"],
                                 value.get("cached_input_tokens", 0), value.get("cache_write_tokens", 0)), 8)
        actual = round(actual + float(value.get("long_context_surcharge_usd", 0.0)), 8)
        value["cost_usd"] = actual
        value["cost_yuan"] = actual  # legacy field retained for old controller readers
        value["estimated_cost_yuan"] = round(
            self.cost(value["input_tokens"] + value["reserved_input_tokens"],
                      value["output_tokens"] + value["reserved_output_tokens"],
                      value.get("cached_input_tokens", 0), value.get("cache_write_tokens", 0)), 6)

    @_synchronized
    def record_overage(self, batch_id: str, dimension: str, projected: float, estimate: float) -> None:
        """Keep the first crossing and latest projection, without growing per request."""
        batch = self._batch(batch_id)
        for value in (self.data, batch):
            overages = value.setdefault("overages", {})
            record = overages.setdefault(dimension, {"estimate": estimate, "first_projected": projected})
            record["latest_projected"] = projected
        self._persist()

    @_synchronized
    def reserve_unit(self, batch_id: str, unit_id: str, attempt_id: str, max_units: int,
                     *, record_only: bool = False) -> dict[str, Any]:
        """Persist a paid-unit reservation before constructing its first request."""
        batch = self._batch(batch_id)
        identity = {"unit_id": unit_id, "attempt_id": attempt_id}
        if identity in batch["unit_attempts"]:
            return dict(batch)
        if int(batch["units_reserved"]) >= int(max_units):
            if not record_only:
                raise BudgetExhausted("authorized unit cap would be exceeded")
            self.record_overage(batch_id, "authorized_units", int(batch["units_reserved"]) + 1, max_units)
        batch["units_reserved"] += 1
        batch["unit_attempts"].append(identity)
        self._persist()
        return dict(batch)

    @_synchronized
    def add(self, *, batch_id: str | None = None, **delta) -> None:
        for key, value in delta.items():
            self.data[key] = self.data.get(key, 0) + value
        self._recalculate(self.data)
        if batch_id is not None:
            batch = self._batch(batch_id)
            for key, value in delta.items():
                batch[key] = batch.get(key, 0) + value
            self._recalculate(batch)
        self._persist()

    @_synchronized
    def reserve_request_attempt(self, batch_id: str, request_attempt_id: str, *, unit_id: str | None,
                                unit_attempt_id: str | None, reserve_input_tokens: int,
                                reserve_output_tokens: int) -> dict[str, Any]:
        """Persist an external-attempt reservation before any bytes are sent."""
        batch = self._batch(batch_id)
        attempts = batch["request_attempts"]
        if request_attempt_id in attempts:
            return copy.deepcopy(attempts[request_attempt_id])
        record = {"request_attempt_id": request_attempt_id, "unit_id": unit_id,
                  "unit_attempt_id": unit_attempt_id, "status": "reserved",
                  "reserved_input_tokens": int(reserve_input_tokens),
                  "reserved_output_tokens": int(reserve_output_tokens)}
        attempts[request_attempt_id] = record
        for value in (self.data, batch):
            value["reserved_attempts"] += 1
            value["reserved_input_tokens"] += int(reserve_input_tokens)
            value["reserved_output_tokens"] += int(reserve_output_tokens)
            self._recalculate(value)
        self._persist()
        return copy.deepcopy(record)

    @_synchronized
    def mark_request_attempt_unknown(self, batch_id: str, request_attempt_id: str, reason: str) -> None:
        batch = self._batch(batch_id)
        record = batch["request_attempts"].get(request_attempt_id)
        if not record or record.get("status") in {"unknown", "settled"}:
            return
        record.update(status="unknown", reason=str(reason))
        for value in (self.data, batch):
            value["unknown_attempts"] += 1
            value["failed_attempts"] += 1
            self._recalculate(value)
        self._persist()

    @_synchronized
    def settle_request_attempt(self, batch_id: str, request_attempt_id: str, *, input_tokens: int,
                               output_tokens: int, request_id: str, model: str,
                               cached_input_tokens: int = 0, cache_write_tokens: int = 0,
                               reasoning_tokens: int = 0, wall_seconds: float = 0.0) -> None:
        batch = self._batch(batch_id)
        record = batch["request_attempts"].get(request_attempt_id)
        if not record:
            raise RuntimeError("cannot settle an unreserved external attempt")
        if record.get("status") == "settled":
            return
        if record.get("status") == "unknown":
            raise RuntimeError("cannot settle an attempt already marked unknown")
        reserve_in = int(record["reserved_input_tokens"])
        reserve_out = int(record["reserved_output_tokens"])
        record.update(status="settled", input_tokens=int(input_tokens), output_tokens=int(output_tokens),
                      cached_input_tokens=int(cached_input_tokens), cache_write_tokens=int(cache_write_tokens),
                      reasoning_tokens=int(reasoning_tokens), wall_seconds=float(wall_seconds),
                      request_id=str(request_id), model=str(model))
        for value in (self.data, batch):
            value["reserved_input_tokens"] -= reserve_in
            value["reserved_output_tokens"] -= reserve_out
            value["settled_attempts"] += 1
            value["requests"] += 1
            value["input_tokens"] += int(input_tokens)
            value["output_tokens"] += int(output_tokens)
            value["cached_input_tokens"] += int(cached_input_tokens)
            value["cache_write_tokens"] += int(cache_write_tokens)
            value["reasoning_tokens"] += int(reasoning_tokens)
            value["wall_seconds"] += float(wall_seconds)
            self._recalculate(value)
        self._persist()

    @_synchronized
    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    @_synchronized
    def batch_snapshot(self, batch_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._batch(batch_id))


class PaidModel:
    def __init__(self, config: dict, authorization: dict, ledger: Ledger, requests_log: Path,
                 exchange_log: Path | None = None):
        import importlib
        transport = importlib.import_module(__package__ + ".transport")
        self.transport = transport
        self.sender = transport.make_sender(config)  # attribute lookup at call time: tests can substitute it
        self.config, self.auth, self.ledger = config, authorization, ledger
        # Direct offline tests historically constructed PaidModel with only
        # token/amount caps. Production authorization still requires batch_id
        # through load_authorization().
        self.batch_id = str(authorization.get("batch_id", "legacy-offline-test"))
        self.requests_log = Path(requests_log)
        self.exchange_log = Path(exchange_log) if exchange_log is not None else None
        budget = config.get("budget", {})
        amount = budget.get("amount")
        self.run_caps = {"input": int(budget.get("input_tokens", 0)) or None,
                         "output": int(budget.get("output_tokens", 0)) or None,
                         # Exploratory runs intentionally use a null monetary
                         # cap while retaining hard token caps.  Do not coerce
                         # that explicit null through float().
                         "amount": None if amount is None else (float(amount) or None)}
        self.active_unit_id: str | None = None
        self.active_attempt_id: str | None = None
        self._request_sequence = 0
        self._session_id = uuid.uuid4().hex

    def set_context(self, unit_id: str, attempt_id: str) -> None:
        self.active_unit_id = str(unit_id)
        self.active_attempt_id = str(attempt_id)

    def _check(self, reserve_in: int, reserve_out: int) -> None:
        used = self.ledger.snapshot()
        batch = self.ledger.batch_snapshot(self.batch_id)
        batch_in = batch["input_tokens"] + batch["reserved_input_tokens"] + reserve_in
        batch_out = batch["output_tokens"] + batch["reserved_output_tokens"] + reserve_out
        record_only = self.auth.get("enforcement") == "record_only"
        def exceeded(dimension: str, projected: float, estimate: float | None, reason: str) -> None:
            if estimate is None or projected <= estimate:
                return
            if record_only:
                self.ledger.record_overage(self.batch_id, dimension, projected, estimate)
            else:
                raise BudgetExhausted(reason)
        exceeded("authorized_input_tokens", batch_in, self.auth["max_input_tokens"],
                 "authorized token cap would be exceeded")
        exceeded("authorized_output_tokens", batch_out, self.auth["max_output_tokens"],
                 "authorized token cap would be exceeded")
        if self.auth.get("max_amount") is not None:
            exceeded("authorized_amount", self.ledger.cost(batch_in, batch_out), self.auth["max_amount"],
                     "authorized amount cap would be exceeded")
        run_in = used["input_tokens"] + used["reserved_input_tokens"] + reserve_in
        run_out = used["output_tokens"] + used["reserved_output_tokens"] + reserve_out
        exceeded("run_input_tokens", run_in, self.run_caps["input"], "run input-token cap would be exceeded")
        exceeded("run_output_tokens", run_out, self.run_caps["output"], "run output-token cap would be exceeded")
        if self.run_caps["amount"] is not None:
            exceeded("run_amount", self.ledger.cost(run_in, run_out), self.run_caps["amount"],
                     "run amount cap would be exceeded")

    def payload(self, call: ModelCall) -> dict[str, Any]:
        model = self.config["model"]
        messages = copy.deepcopy(call.messages)
        if call.images:
            labels = list(call.image_labels) + [""] * (len(call.images) - len(call.image_labels))
            parts = []
            for image, label in zip(call.images, labels):
                if label:
                    parts.append({"type": "text", "text": label})
                parts.append({"type": "image_url", "image_url": {"url": self.transport.image_data_url(image)}})
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] = [{"type": "text", "text": messages[-1]["content"]}, *parts]
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": "Attached images."}, *parts]})
        payload = {"model": model["name"], "messages": messages,
                   "max_completion_tokens": int(model.get("max_completion_tokens", 4096))}
        if model.get("provider") == "openai":
            if model.get("reasoning_effort") is not None:
                payload["reasoning_effort"] = model["reasoning_effort"]
            # Provider-specific switches for OpenAI-compatible endpoints, e.g. Volcengine
            # Ark needs {"thinking": {"type": "disabled"}} to match the no-reasoning inner setting.
            payload.update(dict(model.get("extra_body") or {}))
        else:
            payload.update(temperature=float(model.get("temperature", 0.1)),
                           reasoning_split=True, thinking={"type": "disabled"})
        return payload

    def generate(self, call: ModelCall) -> ModelReply:
        payload = self.payload(call)
        if self.exchange_log is not None:
            self.exchange_log.parent.mkdir(parents=True, exist_ok=True)
            self.exchange_log.write_text(json.dumps({"request": payload}, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
        reserve_in = sum(len(str(m["content"]).encode("utf-8")) + 16 for m in call.messages) + IMAGE_RESERVATION * len(call.images)
        reserve_out = int(payload["max_completion_tokens"])
        self._request_sequence += 1
        logical_id = (f"{self._session_id}:{self.active_attempt_id or 'no-unit'}:{call.phase}:{call.step}:"
                      f"{call.call_index}:{self._request_sequence}")
        current_attempt: str | None = None
        settled = False
        provider_wait_seconds = 0.0

        def log(row: dict[str, Any]) -> None:
            self.requests_log.parent.mkdir(parents=True, exist_ok=True)
            with self.requests_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"batch_id": self.batch_id, "unit_id": self.active_unit_id,
                                         "attempt_id": self.active_attempt_id, "phase": call.phase,
                                         "step": call.step, "call": call.call_index, **row}) + "\n")

        def reserve(network_attempt: int) -> str:
            nonlocal current_attempt
            with self.ledger.lock:
                self._check(reserve_in, reserve_out)
                current_attempt = f"{logical_id}:network-{network_attempt}"
                self.ledger.reserve_request_attempt(
                    self.batch_id, current_attempt, unit_id=self.active_unit_id,
                    unit_attempt_id=self.active_attempt_id, reserve_input_tokens=reserve_in,
                    reserve_output_tokens=reserve_out)
            log({"status": "reserved", "request_attempt_id": current_attempt,
                 "network_attempt": network_attempt, "reserved_input_tokens": reserve_in,
                 "reserved_output_tokens": reserve_out})
            return current_attempt

        def mark_unknown(reason: str) -> None:
            if current_attempt is None:
                return
            self.ledger.mark_request_attempt_unknown(self.batch_id, current_attempt, reason)
            log({"status": "transport_attempt_unknown", "request_attempt_id": current_attempt,
                 "reason": reason, "reserved_input_tokens": reserve_in,
                 "reserved_output_tokens": reserve_out})

        supports_hooks = hasattr(self.sender, "attempt_hook")
        if supports_hooks:
            def hook(event: str, network_attempt: int, detail: dict[str, Any]) -> None:
                nonlocal provider_wait_seconds
                if event == "before":
                    reserve(network_attempt)
                elif event == "unknown":
                    mark_unknown(str(detail.get("reason", "transport outcome unknown")))
                elif event == "waiting_for_provider":
                    provider_wait_seconds += float(detail.get("wait_seconds", 0.0))
                    log({"status": event, "network_attempt": network_attempt, **detail,
                         "cumulative_wait_seconds": round(provider_wait_seconds, 3)})
                elif event == "provider_resumed":
                    log({"status": event, "network_attempt": network_attempt,
                         "cumulative_wait_seconds": round(provider_wait_seconds, 3)})
                elif event == "provider_text_compat":
                    log({"status": event, **detail})
            self.sender.attempt_hook = hook
        else:
            reserve(1)
        started = time.monotonic()
        try:
            response = self.sender(payload)
            if self.exchange_log is not None:
                self.exchange_log.write_text(json.dumps({"request": payload, "response": response},
                                                        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            usage = response.get("usage")
            request_id = response.get("id") or response.get("request_id")
            server_model = response.get("model")
            if not usage:
                mark_unknown("provider response omitted usage")
                log({"status": "usage_unknown", "request_attempt_id": current_attempt,
                     "request_id": request_id, "model": server_model,
                     "reserved_input_tokens": reserve_in, "reserved_output_tokens": reserve_out})
                raise RuntimeError("provider response without usage; reservation retained, stopping")
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
            prompt_details = usage.get("prompt_tokens_details") or {}
            completion_details = usage.get("completion_tokens_details") or {}
            cached_tokens = int(prompt_details.get("cached_tokens") or 0)
            cache_writes = int(prompt_details.get("cache_write_tokens") or 0)
            cache_write_reported = "cache_write_tokens" in prompt_details
            reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)
            elapsed = time.monotonic() - started
            if current_attempt is None:
                raise RuntimeError("transport returned without reserving its external attempt")
            self.ledger.settle_request_attempt(
                self.batch_id, current_attempt, input_tokens=input_tokens, output_tokens=output_tokens,
                request_id=str(request_id or ""), model=str(server_model or ""),
                cached_input_tokens=cached_tokens, cache_write_tokens=cache_writes,
                reasoning_tokens=reasoning_tokens, wall_seconds=elapsed)
            settled = True
            choice = response["choices"][0]
            log({"status": "settled", "request_attempt_id": current_attempt,
                 "request_id": request_id, "model": server_model, "input_tokens": input_tokens,
                 "price_per_million": self.ledger.price,
                 "cached_input_tokens": cached_tokens, "cache_write_tokens": cache_writes,
                 "cache_write_reported": cache_write_reported,
                 "output_tokens": output_tokens, "reasoning_tokens": reasoning_tokens,
                 "wall_seconds": round(elapsed, 3), "rate_limits": response.get("_rate_limits", {}),
                 "finish_reason": choice.get("finish_reason")})
            return ModelReply(str(choice.get("message", {}).get("content") or ""), input_tokens, output_tokens,
                              str(server_model or ""), str(choice.get("finish_reason") or "stop"),
                              cached_tokens, cache_writes, reasoning_tokens, elapsed)
        except Exception:
            if current_attempt is not None and not settled:
                record = self.ledger.batch_snapshot(self.batch_id).get("request_attempts", {}).get(current_attempt, {})
                if record.get("status") == "reserved":
                    mark_unknown("request raised before usage settlement")
            raise
        finally:
            if supports_hooks:
                self.sender.attempt_hook = None
