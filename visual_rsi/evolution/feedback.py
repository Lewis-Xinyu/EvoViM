"""Host-owned feedback from one Meta generation to the next.

Three projections, all written by the host and never by the Meta:

* episode digests of an evaluated candidate (stop reason, score, tool use and
  a condensed step table), so a rejected candidate's failure is visible to the
  next generation instead of only its per-seed score;
* the noise band of the paired score difference, attached to every gate line;
* the prediction ledger: the Meta's ``predictions.json`` is locked when the
  host collects the output (before any evaluation), and the gate later records
  the measured parent -> candidate change next to each prediction.

Every result here is an annotation.  Nothing in this module rejects a
candidate or changes a selection input.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

from .noise import estimate_parent_noise

FEEDBACK_MODES = ("basic", "enhanced")

# Per-episode quantities the host measures for parent and candidate alike.
EFFICIENCY_METRICS = ("steps", "requests", "input_tokens", "output_tokens", "tool_calls",
                      "legacy_memory_calls", "action_overrides")
GUARD_METRICS = ("protocol_error_rate", "fallback_rate")
# Tool-call families, summed per episode from tool_calls_by_name.
MEMORY_READ_TOOLS = frozenset({"READ", "SEARCH", "TAIL", "COUNT"})
MEMORY_WRITE_TOOLS = frozenset({"APPEND", "WRITE", "CREATE"})
FAMILY_METRICS = ("memory_reads", "memory_writes", "visual_tool_calls", "other_tool_calls")
PREDICTION_METRICS = ("score",) + EFFICIENCY_METRICS + FAMILY_METRICS + GUARD_METRICS
TOOL_METRIC = re.compile(r"^tool_calls:([A-Z][A-Z0-9_]{0,63})$")


def tool_family(name: str) -> str:
    if name in MEMORY_READ_TOOLS:
        return "memory_reads"
    if name in MEMORY_WRITE_TOOLS:
        return "memory_writes"
    if "IMAGE" in name or "FRAME" in name or "CANVAS" in name or "VISUAL" in name:
        return "visual_tool_calls"
    return "other_tool_calls"

PREDICTION_SCHEMA = f"""# predictions.json

Write `output/predictions.json` before your session ends.  The host copies and
locks it when it collects your output, before the candidate is evaluated, so it
cannot be edited after the results exist.  After the gate, the host records the
measured parent -> candidate change next to each prediction in
`lineage/predictions.md`.

```json
{{"predictions": [
  {{"metric": "tool_calls:RECALL_IMAGE", "expected_delta": 12,
   "mechanism": "why the change should move this metric",
   "falsified_if": "what measured value would show the mechanism is wrong"}}
]}}
```

- `metric`: one of {", ".join(PREDICTION_METRICS)}, or `tool_calls:<TOOL_NAME>`
  for the calls of one registered tool.  Every value is a per-episode mean over
  the development episodes, except the two rates, which are pooled.
- `expected_delta`: the signed change, candidate minus parent, in the metric's
  own unit.  0 predicts no change.
- `memory_reads`, `memory_writes`, `visual_tool_calls` and `other_tool_calls`
  are per-episode sums over tool families (READ/SEARCH/TAIL/COUNT;
  APPEND/WRITE/CREATE; tools whose name contains IMAGE, FRAME, CANVAS or VISUAL;
  everything else).
- Include at least one metric other than `score`: counts of tool calls, steps and
  requests vary far less between episodes than the score does, so they test your
  mechanism more sharply.

A missing or malformed file is recorded as such; it never rejects a candidate.
"""


# ---------------------------------------------------------------------------- noise band
def difference_band(config: dict, parent: dict) -> dict:
    """Half-width of the paired parent/candidate mean difference expected from noise alone.

    A configured ``noise.difference_band`` (measured by an A/A run) takes
    precedence.  Otherwise, with at least two parent repetitions per seed, the
    band is the 95% half-width implied by the within-seed variance.
    """
    configured = (config.get("noise") or {}).get("difference_band")
    if isinstance(configured, (int, float)) and not isinstance(configured, bool):
        return {"band": float(configured), "source": "configured (A/A)"}
    scores = parent.get("scores") or {}
    if scores and all(isinstance(v, list) and len(v) >= 2 for v in scores.values()):
        noise = estimate_parent_noise(scores)
        band = 1.96 * math.sqrt(2 * noise["mean_variance"] / (noise["repetitions"] * noise["seed_count"]))
        return {"band": band, "source": "parent repetitions"}
    return {"band": None, "source": "not measured"}


def band_text(difference: Any, band: dict) -> str:
    if band.get("band") is None:
        return "noise band not measured"
    if type(difference) not in (int, float):
        return "noise band ±%.3f" % band["band"]
    verdict = "beyond noise" if abs(difference) > band["band"] else "within noise"
    return "Δ %+.3f vs noise band ±%.3f (%s): %s" % (difference, band["band"], band["source"], verdict)


# ---------------------------------------------------------------------------- episode digests
def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _selected_attempt(unit: Path) -> Path | None:
    chosen = _read_json(unit / "unit.json").get("selected_attempt")
    if chosen and (unit / "attempts" / chosen).is_dir():
        return unit / "attempts" / chosen
    attempts = sorted((unit / "attempts").glob("a[0-9][0-9][0-9]")) if (unit / "attempts").is_dir() else []
    return attempts[-1] if attempts else None


def _observation_head(observation: dict, limit: int = 100) -> str:
    """The start of the first non-empty text field, lines joined: a bare header such as Crafter's 'You see:' says nothing."""
    text = observation.get("text") if isinstance(observation, dict) else None
    if isinstance(text, dict):
        for value in text.values():
            if isinstance(value, str) and value.strip():
                joined = " / ".join(line.strip() for line in value.strip().splitlines() if line.strip())
                return joined[:limit]
    return ""


def _step_rows(trajectory: Path) -> tuple[list[str], int]:
    rows = []
    if trajectory.is_file():
        for line in trajectory.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    out = []
    for row in rows:  # every step; nothing is elided
        flags = ",".join(name for name in ("fallback", "action_override") if row.get(name))
        observation = _observation_head(row.get("observation", {})).replace("|", "/")
        out.append(f"| {row.get('step')} | {row.get('action')} | {row.get('score')} | {flags} | {observation} |")
    return out, len(rows)


def _tool_counts(calls: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if calls.is_file():
        for line in calls.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for event in row.get("tool_events") or []:
                if isinstance(event, dict) and event.get("event") == "tool_result":
                    counts[str(event.get("name"))] = counts.get(str(event.get("name")), 0) + 1
    return counts


def write_episode_digest(evidence: Path, destination: Path) -> Path | None:
    """Condense each evaluated unit into one Markdown section the next Meta can read quickly."""
    units = sorted(p for p in (Path(evidence) / "units").glob("*") if p.is_dir()) if Path(evidence).is_dir() else []
    if not units:
        return None
    lines = ["# Episode digest", "",
             "Host-written summary of these development episodes.  Scores are single samples.", ""]
    for unit in units:
        attempt = _selected_attempt(unit)
        if attempt is None:
            lines += [f"## unit {unit.name}: no attempt recorded", ""]
            continue
        summary = _read_json(attempt / "summary.json")
        lines.append(f"## unit {unit.name} (attempt {attempt.name})")
        lines.append(f"- status {summary.get('status')}; stop_reason {summary.get('stop_reason')}; "
                     f"score {summary.get('score')}; steps {summary.get('steps')}")
        lines.append(f"- requests {summary.get('requests')}; protocol_errors {summary.get('protocol_errors')}; "
                     f"fallbacks {summary.get('fallbacks')}; action_overrides {summary.get('action_overrides')}; "
                     f"visual_requests {summary.get('visual_requests')}")
        lines.append(f"- end_type {summary.get('end_type')}; final_message {summary.get('final_message')}; "
                     f"aux_metrics {summary.get('aux_metrics')}; score_status {summary.get('score_status')}")
        if "native_result" in summary:
            lines.append(f"- native result {summary.get('native_result')}; final answer {summary.get('prediction')}; "
                         f"reference {summary.get('reference')}; native success {summary.get('native_success')}")
            lines.append(f"- observation: {summary.get('observation_frames')} frames; "
                         f"{summary.get('scene_observation_requests')} requests; "
                         f"{summary.get('scene_observation_input_tokens')} input / "
                         f"{summary.get('scene_observation_output_tokens')} output tokens; "
                         f"{summary.get('observation_wall_seconds')} seconds")
            lines.append(f"- answer: {summary.get('answer_input_tokens')} input / "
                         f"{summary.get('answer_output_tokens')} output tokens; "
                         f"{summary.get('answer_wall_seconds')} seconds; "
                         f"initial images {summary.get('answer_initial_image_count')}; "
                         f"successful history image recalls {summary.get('successful_history_image_recalls')}")
            lines.append(f"- observation tools (success/error): {summary.get('observation_tool_results')}; "
                         f"answer tools: {summary.get('answer_tool_results')}")
            lines.append(f"- raw evidence: observation {summary.get('scene_observation')}; "
                         f"snapshot {summary.get('scene_snapshot')}; answer {summary.get('answer_archive')}; "
                         f"trajectory scope {summary.get('trace_scope')}")
        tools = _tool_counts(attempt / "calls.jsonl")
        lines.append("- tool calls: " + (", ".join(f"{k} {v}" for k, v in sorted(tools.items(), key=lambda kv: -kv[1]))
                                         or "none"))
        if summary.get("error"):
            lines.append(f"- error: {summary['error']}")
        for name in ("candidate_traceback.txt", "model_traceback.txt", "environment_traceback.txt"):
            if (attempt / name).is_file():
                text = (attempt / name).read_text(encoding="utf-8", errors="replace")
                lines += [f"- {name}:", "```", text, "```"]
        steps, total = _step_rows(attempt / "trajectory.jsonl")
        lines += ["", f"Steps ({total} total):", "",
                  "| step | action | score | flags | observation (start) |", "|---:|---|---:|---|---|", *steps, ""]
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "episodes.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_gate_comparison(parent: dict, candidate: dict, parent_evidence: Path,
                          candidate_evidence: Path, destination: Path, line_dir: Path,
                          measurements: list[dict] | None = None) -> Path:
    """Short, host-only dev comparison for the next enhanced Meta workspace."""
    def fmt(value):
        return "unknown" if value is None else str(round(value, 3) if isinstance(value, float) else value)

    def unit_row(result, unit):
        rows = (result.get("efficiency_by_unit") or {}).get(unit)
        if rows is None:
            rows = [result.get("episode_efficiency", {}).get(unit)]
        rows = [row for row in rows if isinstance(row, dict)]
        success = None if not rows or any(type(row.get("success")) is not bool for row in rows) else (
            f"{sum(row['success'] for row in rows)}/{len(rows)}")
        def mean(field):
            values = [row.get(field) for row in rows]
            return statistics.mean(values) if values and all(type(v) in (int, float) for v in values) else None
        tokens = [row.get("input_tokens") + row.get("output_tokens") for row in rows
                  if type(row.get("input_tokens")) in (int, float)
                  and type(row.get("output_tokens")) in (int, float)]
        return (success, mean("steps"), mean("requests"),
                statistics.mean(tokens) if len(tokens) == len(rows) and tokens else None)

    def summary(evidence, unit):
        attempt = _selected_attempt(evidence / "units" / unit.replace(":", "_"))
        return (_read_json(attempt / "summary.json"), attempt) if attempt else ({}, None)

    def divergence(left, right):
        if not left or not right:
            return "trajectory unavailable"
        def actions(path):
            if not path.is_file():
                return None
            return [row.get("executed_action", row.get("action")) for row in
                    (_read_json_line(raw) for raw in path.read_text(encoding="utf-8").splitlines()) if row]
        a, b = actions(left / "trajectory.jsonl"), actions(right / "trajectory.jsonl")
        if a is None or b is None:
            return "trajectory unavailable"
        for step, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return f"step {step}: {x} / {y} ({left / 'trajectory.jsonl'}; {right / 'trajectory.jsonl'})"
        return (f"one trace ended after {min(len(a), len(b))} steps" if len(a) != len(b)
                else "no executed-action divergence")

    units = sorted(set(parent.get("expected_units", [])))
    lines = ["# Development comparison", "", "Only measurements available at this gate are shown.", "",
             "| task/seed | parent score | candidate score | success parent/candidate | steps parent/candidate | requests parent/candidate | total tokens parent/candidate | protocol errors parent/candidate | fallbacks parent/candidate | first action difference |",
             "|---|---:|---:|---|---|---|---|---|---|---|"]
    for unit in units:
        p_success, p_steps, p_requests, p_tokens = unit_row(parent, unit)
        c_success, c_steps, c_requests, c_tokens = unit_row(candidate, unit)
        p_summary, p_attempt = summary(parent_evidence, unit)
        c_summary, c_attempt = summary(candidate_evidence, unit)
        p_score = parent.get("selection_scores", {}).get(unit)
        c_score = candidate.get("selection_scores", {}).get(unit)
        if isinstance(p_score, list): p_score = statistics.mean(p_score)
        if isinstance(c_score, list): c_score = statistics.mean(c_score)
        lines.append(f"| {unit} | {fmt(p_score)} | {fmt(c_score)} | {fmt(p_success)} / {fmt(c_success)} | "
                     f"{fmt(p_steps)} / {fmt(c_steps)} | {fmt(p_requests)} / {fmt(c_requests)} | "
                     f"{fmt(p_tokens)} / {fmt(c_tokens)} | "
                     f"{fmt(p_summary.get('protocol_errors'))} / {fmt(c_summary.get('protocol_errors'))} | "
                     f"{fmt(p_summary.get('fallbacks'))} / {fmt(c_summary.get('fallbacks'))} | "
                     f"{divergence(p_attempt, c_attempt)} |")
    lines += ["", "## Repeat observations of this parent version", ""]
    records = measurements or []
    for unit in units:
        values, outcomes = [], []
        for item in records:
            result = item.get("result", {})
            for key, score in result.get("selection_scores", {}).items():
                if key == unit or key.startswith(unit + ":r"):
                    values.extend(score if isinstance(score, list) else [score])
            for item_outcome in result.get("outcomes", []):
                key = item_outcome.get("logical_unit_id", "")
                if key == unit or key.startswith(unit + ":r"):
                    outcomes.append(item_outcome)
        if not values:
            continue
        known = [row for row in outcomes if row.get("score_valid")]
        success_rows = (parent.get("efficiency_by_unit") or {}).get(unit)
        if success_rows is None:
            one = parent.get("episode_efficiency", {}).get(unit)
            success_rows = [one] if isinstance(one, dict) else []
        success = [row.get("success") for row in success_rows if type(row.get("success")) is bool]
        codes = {}
        for row in outcomes:
            if row.get("error_code"):
                code = row["error_code"]
                codes[code] = codes.get(code, 0) + 1
        lines.append(f"- {unit}: success {sum(success)}/{len(success)} known; score range "
                     f"{fmt(min(values))}–{fmt(max(values))}; measurements {len(values)}; "
                     f"explicit errors {codes or 'none'}; valid outcomes {len(known)}.")
    if not records:
        lines.append("- No parent measurement history was archived.")
    lines += ["", "## Accepted-version retention", ""]
    found = False
    for record_path in sorted((line_dir / "versions").glob("*/measurements.json")):
        items = _read_json(record_path).get("measurements", [])
        if not items or items[0].get("id") == "G0":
            continue
        found = True
        first = [v for vs in items[0].get("result", {}).get("scores", {}).values() for v in vs]
        later = [v for item in items[1:] for vs in item.get("result", {}).get("scores", {}).values() for v in vs]
        lines.append(f"- {record_path.parent.name[:12]} accepted at {items[0]['id']}: "
                     f"acceptance mean {fmt(statistics.mean(first) if first else None)}; "
                     f"later mean {fmt(statistics.mean(later) if later else None)} over {len(later)} samples; "
                     f"change {fmt(statistics.mean(later) - statistics.mean(first) if first and later else None)}. "
                     "Prediction comparison: see existing lineage/predictions.md when available.")
    if not found:
        lines.append("- No accepted version with subsequent measurements yet.")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "comparison.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_json_line(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------- predictions
def lock_predictions(source: Path, *destinations: Path) -> dict:
    """Validate and copy the Meta's predictions; the returned record is an annotation only."""
    if not Path(source).is_file():
        return {"status": "missing", "predictions": [], "problems": ["output/predictions.json not written"]}
    raw = Path(source).read_bytes()
    record: dict[str, Any] = {"sha256": hashlib.sha256(raw).hexdigest(), "problems": []}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {**record, "status": "invalid", "predictions": [], "problems": [f"not valid JSON: {exc}"]}
    items = data.get("predictions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {**record, "status": "invalid", "predictions": [], "problems": ["top-level 'predictions' list missing"]}
    kept = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            record["problems"].append(f"#{index}: not an object"); continue
        metric, delta = item.get("metric"), item.get("expected_delta")
        if not isinstance(metric, str) or not (metric in PREDICTION_METRICS or TOOL_METRIC.match(metric)):
            record["problems"].append(f"#{index}: unknown metric {metric!r}"); continue
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
            record["problems"].append(f"#{index}: expected_delta must be a finite number"); continue
        kept.append({"metric": metric, "expected_delta": float(delta),
                     "mechanism": str(item.get("mechanism", "")),
                     "falsified_if": str(item.get("falsified_if", ""))})
    if kept and all(p["metric"] == "score" for p in kept):
        record["problems"].append("no mechanism metric: every prediction is on score")
    record["status"] = "valid" if kept and not record["problems"] else ("partial" if kept else "invalid")
    record["predictions"] = kept
    for destination in destinations:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(raw)
    return record


def _metric_value(result: dict, metric: str) -> float | None:
    if metric == "score":
        values = [float(v) for vs in (result.get("scores") or {}).values()
                  for v in (vs if isinstance(vs, list) else [vs]) if type(v) in (int, float)]
        return statistics.mean(values) if values else None
    if metric in GUARD_METRICS:
        value = (result.get("guards") or {}).get(metric)
        return float(value) if type(value) in (int, float) else None
    rows = [r for r in (result.get("episode_efficiency") or {}).values() if isinstance(r, dict)]
    match = TOOL_METRIC.match(metric)
    if match:
        values = [(r.get("tool_calls_by_name") or {}).get(match.group(1), 0) for r in rows
                  if isinstance(r.get("tool_calls_by_name"), dict)]
    elif metric in FAMILY_METRICS:
        values = [sum(count for name, count in r["tool_calls_by_name"].items() if tool_family(name) == metric)
                  for r in rows if isinstance(r.get("tool_calls_by_name"), dict)]
    else:
        values = [r.get(metric) for r in rows]
    values = [float(v) for v in values if type(v) in (int, float)]
    return statistics.mean(values) if values and len(values) == len(rows) else None


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def score_predictions(locked: dict, parent: dict, candidate: dict, band: dict) -> dict:
    rows = []
    for item in locked.get("predictions", []):
        before, after = _metric_value(parent, item["metric"]), _metric_value(candidate, item["metric"])
        row = {**item, "parent": before, "candidate": after}
        if before is None or after is None:
            row.update(observed_delta=None, status="unmeasured")
        else:
            observed = after - before
            row.update(observed_delta=observed, error=observed - item["expected_delta"],
                       direction_correct=_sign(observed) == _sign(item["expected_delta"]), status="measured")
            if item["metric"] == "score" and band.get("band") is not None:
                row["within_score_noise"] = abs(observed) <= band["band"]
        rows.append(row)
    return {"schema_version": 1, "prediction_status": locked.get("status", "missing"),
            "problems": locked.get("problems", []), "score_band": band, "rows": rows}


def _fmt(value: Any) -> str:
    return "-" if value is None else ("%.3f" % value if isinstance(value, float) else str(value))


def append_prediction_table(lineage: Path, comparison_id: str, outcome: dict, *, verdict: str) -> None:
    path = Path(lineage) / "predictions.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Predictions and measured outcomes\n\n"
                        "Written by the host.  Predictions were locked before evaluation.  Δ = candidate − parent, "
                        "per-episode means on the development seeds; scores are single samples.\n\n", encoding="utf-8")
    heading = f"## {comparison_id} ({verdict})"
    if heading in path.read_text(encoding="utf-8"):
        return
    lines = [heading, "",
             f"- prediction file: {outcome['prediction_status']}"
             + (f"; problems: {'; '.join(outcome['problems'])}" if outcome["problems"] else ""),
             f"- score {band_text(None, outcome['score_band'])}", ""]
    if outcome.get("parent_n") is not None:
        lines.insert(-1, f"- parent measurements n={outcome['parent_n']}")
    if outcome["rows"]:
        lines += ["| metric | predicted Δ | observed Δ | parent → candidate | direction | note |",
                  "|---|---:|---:|---|---|---|"]
        for row in outcome["rows"]:
            direction = {True: "right", False: "wrong"}.get(row.get("direction_correct"), "-")
            note = row["status"]
            if "within_score_noise" in row:
                note += "; within score noise" if row["within_score_noise"] else "; beyond score noise"
            lines.append(f"| {row['metric']} | {_fmt(row['expected_delta'])} | {_fmt(row.get('observed_delta'))} | "
                         f"{_fmt(row['parent'])} → {_fmt(row['candidate'])} | {direction} | {note} |")
        lines.append("")
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- mechanism comparison
_PER_STEP = (("requests per step", "requests"), ("memory reads per step", "memory_reads"),
             ("memory writes per step", "memory_writes"), ("visual tool calls per step", "visual_tool_calls"),
             ("other tool calls per step", "other_tool_calls"))


def _episode_rows(result: dict) -> list[dict]:
    return [r for r in (result.get("episode_efficiency") or {}).values()
            if isinstance(r, dict) and type(r.get("steps")) in (int, float) and r["steps"]]


def _per_step(result: dict, quantity: str) -> float | None:
    values = []
    for row in _episode_rows(result):
        by_name = row.get("tool_calls_by_name")
        if quantity in FAMILY_METRICS:
            if not isinstance(by_name, dict):
                continue
            count = sum(c for name, c in by_name.items() if tool_family(name) == quantity)
        else:
            count = row.get(quantity)
            if type(count) not in (int, float):
                continue
        values.append(count / row["steps"])
    return statistics.mean(values) if values else None


def _per_tool(result: dict) -> dict[str, float]:
    rows = [r for r in _episode_rows(result) if isinstance(r.get("tool_calls_by_name"), dict)]
    names = sorted({name for r in rows for name in r["tool_calls_by_name"]})
    return {name: statistics.mean(r["tool_calls_by_name"].get(name, 0) / r["steps"] for r in rows) for name in names}


def mechanism_comparison(parent: dict, candidate: dict, *, parent_n: int = 1) -> tuple[str, str]:
    """Host-measured behaviour of the parent (all its measurements) next to the candidate.

    Returns (markdown, one-line summary).  Rates are per step, averaged over episodes.
    """
    rows = [("score (per episode)", _metric_value(parent, "score"), _metric_value(candidate, "score")),
            ("steps (per episode)", _metric_value(parent, "steps"), _metric_value(candidate, "steps"))]
    rows += [(label, _per_step(parent, key), _per_step(candidate, key)) for label, key in _PER_STEP]
    rows += [(name.replace("_", " "), _metric_value(parent, name), _metric_value(candidate, name))
             for name in GUARD_METRICS]

    def change(before, after):
        if before is None or after is None:
            return "-"
        if before == 0:
            return "new" if after else "0%"
        return "%+.0f%%" % (100 * (after - before) / before)

    lines = ["# Mechanism comparison", "",
             f"Host-measured behaviour: the parent this candidate was derived from ({parent_n} measurement(s)) "
             "versus the candidate.  Per-step values are averaged over episodes.", "",
             "| quantity | parent | candidate | change |", "|---|---:|---:|---:|"]
    lines += [f"| {label} | {_fmt(a)} | {_fmt(b)} | {change(a, b)} |" for label, a, b in rows]
    parent_tools, candidate_tools = _per_tool(parent), _per_tool(candidate)
    new_tools = sorted(set(candidate_tools) - set(parent_tools))
    if new_tools:
        lines += ["", "New tool use by paired unit (descriptive association, not causal attribution):", "",
                  "| unit | tool | calls | parent score | candidate score | Δ |",
                  "|---|---|---:|---:|---:|---:|"]
        parent_scores = parent.get("selection_scores") or parent.get("scores") or {}
        candidate_scores = candidate.get("selection_scores") or candidate.get("scores") or {}
        for unit, raw in sorted(candidate_scores.items()):
            rows_for_unit = (candidate.get("efficiency_by_unit") or {}).get(unit, [])
            rows_for_unit = rows_for_unit if isinstance(rows_for_unit, list) else [rows_for_unit]
            for tool in new_tools:
                calls = sum(int((row.get("tool_calls_by_name") or {}).get(tool, 0))
                            for row in rows_for_unit if isinstance(row, dict))
                if calls <= 0:
                    continue
                previous = parent_scores.get(unit, [])
                previous = previous if isinstance(previous, list) else [previous]
                current = raw if isinstance(raw, list) else [raw]
                if not previous or not current:
                    continue
                before, after = statistics.mean(previous), statistics.mean(current)
                lines.append(f"| {unit} | {tool} | {calls} | {before:.3f} | {after:.3f} | {after-before:+.3f} |")
    names = sorted(set(parent_tools) | set(candidate_tools),
                   key=lambda n: -(candidate_tools.get(n, 0) + parent_tools.get(n, 0)))
    if names:
        lines += ["", "Calls per step, by tool:", "", "| tool | family | parent | candidate |", "|---|---|---:|---:|"]
        lines += [f"| {n} | {tool_family(n)} | {_fmt(parent_tools.get(n))} | {_fmt(candidate_tools.get(n))} |"
                  for n in names]
    keyed = {label: (a, b) for label, a, b in rows}
    summary = "; ".join(f"{label.replace(' per step', '/step')} {_fmt(keyed[label][0])}→{_fmt(keyed[label][1])}"
                        for label in ("memory reads per step", "memory writes per step", "visual tool calls per step")
                        if label in keyed)
    return "\n".join(lines) + "\n", summary
