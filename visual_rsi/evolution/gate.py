"""User-selected gates with tie-retain-parent semantics and safety guards."""
from __future__ import annotations

import itertools
import math
import statistics


def _samples(values) -> list[float] | None:
    values = values if isinstance(values, list) else [values]
    if not values or not all(type(item) in (int, float) and math.isfinite(float(item)) for item in values):
        return None
    return [float(item) for item in values]


def _paired(parent: dict, candidate: dict) -> list[tuple[float, float]]:
    """Return a full paired measurement only; intersections are never selectable."""
    if ("selection_eligible" in parent or "complete" in parent) and not parent.get("selection_eligible", parent.get("complete", False)):
        raise ValueError("invalid_parent")
    if ("selection_eligible" in candidate or "complete" in candidate) and not candidate.get("selection_eligible", candidate.get("complete", False)):
        raise ValueError("invalid_candidate")
    parent_scores = parent.get("selection_scores", parent.get("scores", {}))
    candidate_scores = candidate.get("selection_scores", candidate.get("scores", {}))
    if not isinstance(parent_scores, dict) or not isinstance(candidate_scores, dict):
        raise ValueError("scores must be mappings")
    expected_parent = set(map(str, parent.get("expected_units", parent_scores)))
    expected_candidate = set(map(str, candidate.get("expected_units", candidate_scores)))
    if not expected_parent or expected_parent != expected_candidate:
        raise ValueError("incomplete_or_mismatched_units")
    if set(parent_scores) != expected_parent or set(candidate_scores) != expected_parent:
        raise ValueError("incomplete_or_mismatched_units")
    if parent.get("eval_fingerprint") is not None and candidate.get("eval_fingerprint") is not None and parent["eval_fingerprint"] != candidate["eval_fingerprint"]:
        raise ValueError("eval_fingerprint_mismatch")
    pairs = []
    for unit in sorted(expected_parent, key=str):
        left, right = _samples(parent_scores[unit]), _samples(candidate_scores[unit])
        if left is None or right is None:
            raise ValueError("invalid_score")
        if len(left) != len(right):
            raise ValueError("measurement_count_mismatch")
        pairs.append((statistics.mean(left), statistics.mean(right)))
    return pairs


def _permutation_pvalue(deltas: list[float]) -> float:
    observed = sum(deltas) / len(deltas)
    count = total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        value = sum(delta * sign for delta, sign in zip(deltas, signs)) / len(deltas)
        count += value >= observed - 1e-12
        total += 1
    return count / total


def compare_successful_efficiency(parent: dict, candidate: dict) -> dict:
    """Compare complete costs on the same logical units, using successful runs only."""
    units = sorted(set(map(str, parent.get("expected_units", []))))
    detail = {"common_success_units": 0, "total_units": len(units), "parent_steps": None,
              "candidate_steps": None, "parent_tokens": None, "candidate_tokens": None,
              "improved": False, "reason": "efficiency_data_unavailable"}
    if not units or set(units) != set(map(str, candidate.get("expected_units", []))):
        return detail
    def rows(result, unit):
        mapping = result.get("efficiency_by_unit", result.get("episode_efficiency", {}))
        value = mapping.get(unit) if isinstance(mapping, dict) else None
        return value if isinstance(value, list) else [value]
    per_unit = []
    for unit in units:
        left, right = rows(parent, unit), rows(candidate, unit)
        if not left or not right or any(not isinstance(row, dict) or type(row.get("success")) is not bool
                                      for row in left + right):
            return detail
        if statistics.mean(row["success"] for row in right) < statistics.mean(row["success"] for row in left):
            detail["reason"] = "success_regressed"
            return detail
        if not any(row["success"] for row in left) or not any(row["success"] for row in right):
            continue
        successful = ([row for row in left if row["success"]], [row for row in right if row["success"]])
        values = []
        for side in successful:
            if any(not all(type(row.get(field)) in (int, float) and math.isfinite(row[field])
                           for field in ("steps", "input_tokens", "output_tokens")) for row in side):
                return detail
            values.append((statistics.mean(row["steps"] for row in side),
                           statistics.mean(row["input_tokens"] + row["output_tokens"] for row in side)))
        per_unit.append((values[0], values[1]))
    detail["common_success_units"] = len(per_unit)
    if not per_unit:
        detail["reason"] = "no_common_success"
        return detail
    detail.update(parent_steps=statistics.mean(v[0][0] for v in per_unit),
                  candidate_steps=statistics.mean(v[1][0] for v in per_unit),
                  parent_tokens=statistics.mean(v[0][1] for v in per_unit),
                  candidate_tokens=statistics.mean(v[1][1] for v in per_unit))
    def compare(a, b):
        return 0 if math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9) else (-1 if b < a else 1)
    steps = compare(detail["parent_steps"], detail["candidate_steps"])
    tokens = compare(detail["parent_tokens"], detail["candidate_tokens"])
    detail["improved"] = steps <= 0 and tokens <= 0 and (steps < 0 or tokens < 0)
    detail["reason"] = "same_score_efficiency_improved" if detail["improved"] else "no_efficiency_improvement"
    return detail


def gate_candidate(parent: dict, candidate: dict, *, rule: str | None, delta_min: float = 0.0,
                   z: float = 1.96, alpha: float = 0.05, guards: dict | None = None,
                   unit_regression_max: float | None = None) -> dict:
    if not rule:
        raise ValueError("gate.rule is required; controller provides no default")
    if rule == "score_then_efficiency" and (not parent.get("eval_fingerprint") or
                                            parent.get("eval_fingerprint") != candidate.get("eval_fingerprint")):
        return {"accepted": False, "status": "not_comparable", "reason": "eval_fingerprint_mismatch_or_missing",
                "rule": rule, "detail": {}, "tie_rule": "efficiency then retain parent"}
    try:
        pairs = _paired(parent, candidate)
    except ValueError as exc:
        return {"accepted": False, "status": "not_comparable", "reason": str(exc), "rule": rule,
                "detail": {}, "tie_rule": "retain parent"}
    deltas = [c - p for p, c in pairs]
    difference = statistics.mean(deltas)
    # Per-unit regression guard (gen2, 2026-10-08): with several development units
    # (seeds or levels) a candidate may not buy a gain on one unit with a clear loss
    # on another.  None disables the guard (the pre-gen2 behaviour).
    worst_unit_delta = min(deltas)
    unit_guard_ok = unit_regression_max is None or worst_unit_delta >= -float(unit_regression_max) - 1e-12
    passed_rule = False
    detail = {}
    if rule == "strict_mean":
        passed_rule = difference > 0
        detail["warning"] = "under noisy episodes, strict_mean can be close to a coin flip"
    elif rule == "score_then_efficiency":
        # A tie means no unit moved: an efficiency-only acceptance must not hide a
        # loss on one unit behind a gain on another.
        score_tie = (math.isclose(difference, 0.0, rel_tol=0.0, abs_tol=1e-9)
                     and (unit_regression_max is None or all(abs(delta) <= 1e-9 for delta in deltas)))
        efficiency_detail = compare_successful_efficiency(parent, candidate)
        passed_rule = difference > 1e-9 or bool(score_tie and efficiency_detail["improved"])
        detail["efficiency_comparison"] = efficiency_detail
        detail["acceptance_reason"] = ("score_improved" if difference > 1e-9 else
                                       (efficiency_detail["reason"] if score_tie else "score_declined"))
    elif rule == "delta_threshold":
        se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
        passed_rule = difference >= delta_min and difference >= z * se
        detail.update(standard_error=se, threshold=delta_min)
    elif rule == "paired_permutation":
        minimum = math.ceil(math.log2(1 / alpha)) if alpha > 0 else math.inf
        if len(deltas) < minimum:
            raise ValueError(f"paired_permutation needs at least {minimum} seeds for alpha={alpha}")
        pvalue = _permutation_pvalue(deltas)
        passed_rule = difference > 0 and pvalue <= alpha
        detail.update(pvalue=pvalue, alpha=alpha)
    else:
        raise ValueError("unknown gate rule: " + rule)
    parent_guard = parent.get("guards", {})
    candidate_guard = candidate.get("guards", {})
    guards = guards or {}
    guard_ok = True if rule == "score_then_efficiency" else candidate_guard.get("crashes", 0) == 0
    if rule != "score_then_efficiency":
        guard_ok = guard_ok and candidate_guard.get("protocol_error_rate", 0) - parent_guard.get("protocol_error_rate", 0) <= guards.get("protocol_error_delta", 0.05)
        guard_ok = guard_ok and candidate_guard.get("fallback_rate", 0) - parent_guard.get("fallback_rate", 0) <= guards.get("fallback_delta", 0.05)
    guard_ok = guard_ok and candidate_guard.get("pseudo_strategy_rate", 0) <= guards.get("pseudo_strategy_max", 1.0)
    detail["unit_deltas"] = deltas
    detail["unit_regression_max"] = unit_regression_max
    detail["unit_guard_passed"] = unit_guard_ok
    accepted = bool(passed_rule and guard_ok and unit_guard_ok and (rule == "score_then_efficiency" or difference > 0))
    efficiency = candidate.get("efficiency") or {}
    efficiency_improved = bool(efficiency.get("complete") and efficiency.get("reduction_at_least_threshold"))
    detail["efficiency_improvement_ignored"] = bool(rule != "score_then_efficiency" and difference == 0 and efficiency_improved)
    detail["efficiency_regressions"] = efficiency.get("regressions", [])
    reason = ("unit_regression" if passed_rule and guard_ok and not unit_guard_ok else
              "guard_rejected" if passed_rule and not guard_ok else
              detail.get("acceptance_reason", "score_improved" if accepted else "no_improvement"))
    return {"accepted": accepted, "status": "accepted" if accepted else "parent_retained", "rule": rule,
            "difference": difference, "parent_mean": statistics.mean([p for p, _ in pairs]),
            "candidate_mean": statistics.mean([c for _, c in pairs]), "rule_passed": passed_rule,
            "guard_passed": guard_ok, "detail": detail,
            "reason": reason,
            "tie_rule": "efficiency then retain parent" if rule == "score_then_efficiency" else "retain parent"}
