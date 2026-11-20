"""Stable, host-owned evaluation outcome contract.

An attempt completing its control flow is not sufficient for selection.  This
module is intentionally dependency-free so controller, reports, recovery and
offline audit use exactly the same classification.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any


VALID_EXECUTION = {"completed", "terminated", "truncated"}
METHOD_STATUSES = {"candidate_crashed", "candidate_hung", "containment_violation", "method_failed"}
INFRA_STATUSES = {"model_error", "provider_error", "environment_error", "host_error", "infrastructure_failed"}
BLOCKED_STATUSES = {"budget_exhausted", "budget_blocked", "authorization_blocked", "blocked_unknown"}


def finite_score(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


@dataclass(frozen=True)
class Outcome:
    schema_version: int
    logical_round_id: str
    logical_unit_id: str
    attempt_id: str
    candidate_hash: str | None
    parent_before_hash: str | None
    eval_fingerprint: str | None
    execution_status: str
    failure_origin: str | None
    error_code: str | None
    retryable: bool
    score_valid: bool
    selection_eligible: bool
    score: float | None
    diagnostic_score: float | None
    evidence_refs: list[str]
    missing_evidence: list[str]

    def json(self) -> dict[str, Any]:
        return asdict(self)


def classify_summary(summary: dict[str, Any], *, logical_round_id: str, logical_unit_id: str,
                     attempt_id: str, candidate_hash: str | None = None,
                     parent_before_hash: str | None = None, eval_fingerprint: str | None = None,
                     candidate_failure_score: float | None = None) -> Outcome:
    status = str(summary.get("status", "blocked_unknown"))
    raw_score = summary.get("score")
    if status in VALID_EXECUTION and finite_score(raw_score):
        return Outcome(1, logical_round_id, logical_unit_id, attempt_id, candidate_hash, parent_before_hash,
                       eval_fingerprint, status, None, None, False, True, True, float(raw_score), None,
                       list(summary.get("evidence_refs", [])), list(summary.get("missing_evidence", [])))
    if status in METHOD_STATUSES:
        origin = ("candidate" if status not in {"candidate_crashed", "candidate_hung"}
                  or summary.get("failure_origin") == "candidate" else "unknown")
        code = "METHOD_VIOLATION_ACTION_OVERRIDE" if status == "containment_violation" and "action" in str(summary.get("error", "")).lower() else status.upper()
        if (status in {"candidate_crashed", "candidate_hung"} and origin == "candidate"
                and finite_score(candidate_failure_score)):
            return Outcome(1, logical_round_id, logical_unit_id, attempt_id, candidate_hash, parent_before_hash,
                           eval_fingerprint, status, origin, code, False, True, True,
                           float(candidate_failure_score), float(raw_score) if finite_score(raw_score) else None,
                           list(summary.get("evidence_refs", [])), [])
        return Outcome(1, logical_round_id, logical_unit_id, attempt_id, candidate_hash, parent_before_hash,
                       eval_fingerprint, "method_failed", origin, code, False, False, False, None,
                       float(raw_score) if finite_score(raw_score) else None, list(summary.get("evidence_refs", [])), [])
    if status in INFRA_STATUSES:
        origin = "provider" if status in {"model_error", "provider_error"} else ("environment" if status == "environment_error" else "host")
        return Outcome(1, logical_round_id, logical_unit_id, attempt_id, candidate_hash, parent_before_hash,
                       eval_fingerprint, "infrastructure_failed", origin, status.upper(), True, False, False,
                       None, float(raw_score) if finite_score(raw_score) else None, list(summary.get("evidence_refs", [])), [])
    origin = "authorization" if status in {"budget_exhausted", "authorization_blocked"} else "unknown"
    normalized = "blocked_budget" if status in {"budget_exhausted", "budget_blocked"} else ("blocked_auth" if status == "authorization_blocked" else "blocked_unknown")
    return Outcome(1, logical_round_id, logical_unit_id, attempt_id, candidate_hash, parent_before_hash,
                   eval_fingerprint, normalized, origin, status.upper(), False, False, False, None,
                   float(raw_score) if finite_score(raw_score) else None, list(summary.get("evidence_refs", [])), [])


def aggregate_outcomes(outcomes: list[Outcome], expected_units: set[str]) -> dict[str, Any]:
    by_unit: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        by_unit.setdefault(outcome.logical_unit_id, []).append(outcome)
    duplicate = sorted(unit for unit, values in by_unit.items() if len(values) != 1)
    missing = sorted(expected_units - set(by_unit))
    selected = {unit: values[0] for unit, values in by_unit.items() if len(values) == 1}
    score_valid = not missing and not duplicate and all(item.score_valid for item in selected.values())
    scores = {unit: item.score for unit, item in selected.items() if item.score_valid}
    return {"schema_version": 1, "outcomes": [item.json() for item in outcomes], "scores": scores,
            "diagnostics": {unit: item.diagnostic_score for unit, item in selected.items() if item.diagnostic_score is not None},
            "expected_units": sorted(expected_units), "missing_units": missing, "duplicate_units": duplicate,
            "execution_complete": not missing and not duplicate,
            "scores_complete": score_valid,
            "selection_eligible": score_valid and all(item.selection_eligible for item in selected.values()),
            "complete": score_valid}
