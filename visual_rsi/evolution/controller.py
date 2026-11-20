"""EvoViM evolution controller: frozen config, authorized paid evaluation, Meta workspaces, lineage and gates.

Every command prints one single-line JSON object as its last stdout line.

Run directory layout:
    config.json, ledger.json, requests.jsonl, authorization.json (user-written)
    lines/<L>/parent/              current lineage parent bundle
    lines/<L>/parent_scores.json   scores of the current parent (parent-eval or accepted candidate)
    lines/<L>/parent_evidence/     public episode archives of the current parent
    lines/<L>/meta_state/          sanitized Meta notebook, toolkit and playbook
    lines/<L>/lineage/             lineage.md plus each collected candidate's Meta documents
    lines/<L>/generations/g<G>_<slot>/  workspace.json, review.json, replay_report.json, status.json,
                                        bundle/, evidence/, scores.json, gate.json, claude/
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
import statistics
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from omegaconf import OmegaConf

from ..adapters import get_adapter, registered_adapters
from ..host.bundle import install_bundle, overlay_files
from ..host.child.registry import LEGACY_KEYS
from ..host.episode import run_episode
from ..host.worker_process import BACKSTOP_SECONDS
from ..host.outcome import aggregate_outcomes, classify_summary
from ..host.repair import AttemptStore, Promotion, atomic_json, efficiency, finalize_feedback, tree_hash
from ..host.paid import Ledger, PaidModel, load_authorization
from ..host.provenance import startup_runtime, bind_runtime
from ..host.visual_channel import VisualLimits
from ..meta.state import transfer_state
from ..meta.workspace import build_workspace, install_replay, make_contact_sheet, readonly, writable
from .feedback import (FEEDBACK_MODES, PREDICTION_SCHEMA, append_prediction_table, band_text, difference_band,
                       lock_predictions, mechanism_comparison, score_predictions, write_episode_digest,
                       write_gate_comparison)
from .gate import compare_successful_efficiency, gate_candidate
from .noise import estimate_parent_noise
from .review import review_candidate

ROOT = Path(__file__).resolve().parents[1]
LEGACY_MEMORY_TOOLS = frozenset(LEGACY_KEYS)
REQUIRED = ("benchmark", "seeds", "gate", "noise", "generations", "meta", "budget", "model", "runtime")
META_DOCUMENTS = ("analysis.md", "method_description.md", "tool_contracts.json", "validation_report.md",
                  "interface_requests.md")


def load_config(path: Path) -> dict:
    data = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    return load_config_value(data)


def load_config_value(data: dict) -> dict:
    """Validate a loaded config, including supervisor-derived per-family configs."""
    missing = [key for key in REQUIRED if key not in data]
    if missing:
        raise ValueError("missing required config fields: " + ",".join(missing))
    adapter = data["benchmark"]["adapter"]
    if adapter not in registered_adapters():
        raise ValueError(f"Unknown adapter {adapter!r}. Registered adapters: {', '.join(registered_adapters())}")
    if not data["gate"].get("rule"):
        raise ValueError("gate.rule is required")
    if data["gate"]["rule"] not in {"strict_mean", "score_then_efficiency", "delta_threshold", "paired_permutation"}:
        raise ValueError("unknown gate.rule")
    feedback_units = set(data.get("meta", {}).get("feedback_units") or [])
    selection_units = set(data.get("gate", {}).get("selection_units") or [])
    if feedback_units and selection_units and feedback_units & selection_units:
        raise ValueError("Meta feedback and gate selection units must be disjoint")
    exploratory = data.get("experiment_mode") == "exploratory"
    repetitions = int(data["noise"].get("parent_repetitions", 0))
    if repetitions < (1 if exploratory else 2):
        raise ValueError("exploratory mode needs parent_repetitions >= 1; formal mode needs >= 2")
    if not data["model"].get("name"):
        raise ValueError("model.name is required")
    if not isinstance(data["runtime"].get("infra_retry", False), bool):
        raise ValueError("runtime.infra_retry must be true or false")
    if data["runtime"].get("mode") not in {"seatbelt", "process"}:
        raise ValueError("runtime.mode must be seatbelt or process")
    seeds = data["seeds"].get("dev") or []
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds.dev must be a non-empty list of unique integers")
    if data.get("lines", ["M"]) != ["M"]:
        raise ValueError("The visual memory method requires lines: [M]")
    _start_bundle(data, "M")
    numbers = [("gate", "alpha"), ("budget", "input_tokens"), ("budget", "output_tokens")]
    if not (exploratory and data["budget"].get("amount") is None):
        numbers.append(("budget", "amount"))
    if data["gate"]["rule"] == "delta_threshold":
        numbers += [("gate", "delta_min"), ("gate", "z")]
    for section, key in numbers:
        value = data[section].get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{section}.{key} must be a number")
    if data["meta"].get("feedback", "basic") not in FEEDBACK_MODES:
        raise ValueError("meta.feedback must be one of " + ", ".join(FEEDBACK_MODES))
    if not isinstance(data["meta"].get("revision_slot", False), bool):
        raise ValueError("meta.revision_slot must be true or false")
    if not isinstance(data["meta"].get("evidence_images", True), bool):
        raise ValueError("meta.evidence_images must be true or false")
    unit_max = data["gate"].get("unit_regression_max")
    if unit_max is not None and (type(unit_max) not in (int, float) or not math.isfinite(float(unit_max)) or unit_max < 0):
        raise ValueError("gate.unit_regression_max must be a non-negative number when set")
    if not isinstance(data["gate"].get("parent_remeasure", False), bool):
        raise ValueError("gate.parent_remeasure must be true or false")
    if data["gate"].get("reference", "parent") not in {"parent", "best"}:
        raise ValueError("gate.reference must be parent or best")
    if data["gate"].get("reference", "parent") == "best" and not data["gate"].get("parent_remeasure", False):
        raise ValueError("gate.reference: best compares measured means and needs gate.parent_remeasure: true")
    caps = data["runtime"].get("phase_caps")
    if caps is not None and (not isinstance(caps, dict) or set(caps) != {"LOG", "PLAN"}
                             or any(type(v) is not int or v < 1 for v in caps.values())):
        raise ValueError("runtime.phase_caps must be {LOG: positive int, PLAN: positive int}")
    band = data["noise"].get("difference_band")
    if band is not None and (isinstance(band, bool) or not isinstance(band, (int, float)) or band < 0):
        raise ValueError("noise.difference_band must be a non-negative number")
    if (data["runtime"]["mode"] == "seatbelt" and data["meta"].get("provider", "claude") != "codex"
            and not isinstance(data["meta"].get("claude", {}).get("base_url"), str)):
        raise ValueError("meta.claude.base_url is required")
    if data["gate"]["rule"] == "paired_permutation":
        minimum = math.ceil(math.log2(1 / float(data["gate"].get("alpha", 0.05))))
        if len(seeds) < minimum:
            raise ValueError(f"paired_permutation needs at least {minimum} dev seeds at alpha={data['gate'].get('alpha', 0.05)}")
    if exploratory:
        if repetitions != 1:
            raise ValueError("exploratory mode requires exactly one measurement per seed")
        if data["gate"]["rule"] not in {"strict_mean", "score_then_efficiency"}:
            raise ValueError("exploratory mode requires strict_mean or score_then_efficiency")
    holdout = list(data["seeds"].get("holdout") or [])
    if (any(type(seed) is not int for seed in holdout) or len(set(holdout)) != len(holdout)
            or set(holdout) & set(seeds)):
        raise ValueError("seeds.holdout must be unique integers disjoint from seeds.dev")
    return data


def _frozen(run_dir: Path) -> dict:
    return json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))


def _mde(config: dict, scores: dict | None) -> dict:
    if config.get("experiment_mode") == "exploratory":
        return {"mde_80pct": None, "mean_variance": None, "status": "not_estimated"}
    alpha = float(config["gate"].get("alpha", 0.05))
    if scores and all(len(v) >= 2 for v in scores.values()):
        return estimate_parent_noise(scores, alpha=alpha)
    sigma = config["noise"].get("sigma")
    if isinstance(sigma, (int, float)):
        seeds = config["seeds"]["dev"]
        synthetic = {str(s): [0.0, math.sqrt(2) * float(sigma)] for s in seeds}
        return {**estimate_parent_noise(synthetic, alpha=alpha), "source": "configured sigma"}
    return {"mde_80pct": None, "status": "measured after parent-eval"}


def _line(run_dir: Path, line: str) -> Path:
    if line != "M":
        raise ValueError("The visual memory method uses line M")
    return Path(run_dir) / "lines" / line


def _slot(run_dir: Path, line: str, generation: int, slot: str) -> Path:
    if slot not in {"fresh", "revision"}:
        raise ValueError("--slot must be fresh or revision")
    return _line(run_dir, line) / "generations" / f"g{int(generation)}_{slot}"


def _parent_bundle(line_dir: Path) -> Path:
    """Resolve the parent through the promotion pointer once it exists.

    The legacy ``parent/`` directory remains a bootstrap/import compatibility
    copy only; controller consumers must never silently mix it with a promoted
    score/evidence version.
    """
    resolved = Promotion(line_dir).resolve_parent()
    return resolved if resolved is not None else line_dir / "parent"


def _parent_scores(line_dir: Path) -> dict:
    pointer = Promotion(line_dir).current()
    if pointer and pointer.get("scores"):
        score_path = line_dir / str(pointer["scores"])
        if score_path.is_file():
            return _read(score_path)
    return _read(line_dir / "parent_scores.json")


def _parent_evidence(line_dir: Path) -> Path:
    pointer = Promotion(line_dir).current()
    if pointer and pointer.get("evidence"):
        candidate = line_dir / str(pointer["evidence"])
        if candidate.is_dir():
            return candidate
    return line_dir / "parent_evidence"


def _measurement_path(line_dir: Path, bundle: Path) -> Path:
    return line_dir / "versions" / tree_hash(bundle) / "measurements.json"


def _record_measurement(line_dir: Path, bundle: Path, measurement_id: str, result: dict) -> None:
    path = _measurement_path(line_dir, bundle)
    record = _read(path) if path.exists() else {"version_hash": tree_hash(bundle), "measurements": []}
    if not any(item["id"] == measurement_id for item in record["measurements"]):
        record["measurements"].append({"id": measurement_id, "result": result})
        atomic_json(path, record)


def _measured_parent(line_dir: Path, bundle: Path, initial: dict) -> tuple[dict, int]:
    path = _measurement_path(line_dir, bundle)
    records = (_read(path).get("measurements") if path.exists() else None) or [{"result": initial}]
    results = [item["result"] for item in records
               if item["result"].get("selection_eligible", item["result"].get("complete", True))
               and item["result"].get("eval_fingerprint") == initial.get("eval_fingerprint")]
    if not results:
        return {**initial, "selection_eligible": False}, 0
    units = list(initial.get("expected_units", []))
    selection = {}
    for unit in units:
        values = []
        for result in results:
            for key, sample in result.get("selection_scores", {}).items():
                if key == unit or key.startswith(unit + ":r"):
                    values.extend(sample if isinstance(sample, list) else [sample])
        if values:
            selection[unit] = [statistics.mean(values)]
    scores = {}
    for seed in initial.get("scores", {}):
        values = [float(value) for result in results for value in result.get("scores", {}).get(seed, [])]
        if values:
            scores[seed] = [statistics.mean(values)]
    guards = {}
    for name in initial.get("guards", {}):
        values = [float(result.get("guards", {}).get(name, 0)) for result in results]
        guards[name] = statistics.mean(values)
    efficiency_rows = {f"{item.get('id', 'initial')}:{unit}": row for item in records
                       if item["result"] in results
                       for unit, row in item["result"].get("episode_efficiency", {}).items()}
    efficiency_by_unit = {unit: [row for result in results
                                 for key, row in result.get("episode_efficiency", {}).items()
                                 if key == unit or key.startswith(unit + ":r")]
                          for unit in units}
    return {**initial, "selection_scores": selection, "scores": scores,
            "guards": guards, "episode_efficiency": efficiency_rows,
            "efficiency_by_unit": efficiency_by_unit}, len(results)


def _reference_is_best(config: dict) -> bool:
    gate = config.get("gate", {})
    return gate.get("reference", "parent") == "best" and bool(gate.get("parent_remeasure", False))


def _store_version(line_dir: Path, bundle: Path, evidence: Path, scores: dict) -> None:
    """Keep a restorable copy of a version (G0 is never staged through Promotion)."""
    version = line_dir / "versions" / tree_hash(bundle)
    if not (version / "bundle").is_dir():
        shutil.copytree(bundle, version / "bundle")  # byte-identical, so its tree hash matches
    if evidence.is_dir() and not (version / "evidence").is_dir():
        shutil.copytree(evidence, version / "evidence")
    if not (version / "scores.json").exists():
        atomic_json(version / "scores.json", scores)


def _measured_versions(line_dir: Path) -> list[dict]:
    """Every restorable version with its mean over all of its measurements."""
    rows = []
    for record_path in sorted((line_dir / "versions").glob("*/measurements.json")):
        version = record_path.parent
        if not (version / "bundle").is_dir():
            continue
        record = _read(record_path)
        values = [float(v) for item in record.get("measurements", [])
                  for vs in item["result"].get("scores", {}).values() for v in vs]
        if not values:
            continue
        rows.append({"hash": version.name, "mean": statistics.mean(values), "n": len(record["measurements"]),
                     "first_id": record["measurements"][0]["id"], "path": version,
                     "result": _measured_parent(line_dir, version / "bundle", record["measurements"][0]["result"])[0]})
    return rows


def _best_version(line_dir: Path, *, rule: str = "strict_mean", current_hash: str | None = None) -> dict | None:
    """Highest measured mean; the new rule resolves a tie only with clear efficiency gain."""
    rows = _measured_versions(line_dir)
    if not rows:
        return None
    order = {row["hash"]: index for index, row in enumerate(
        sorted(rows, key=lambda r: (r["path"] / "measurements.json").stat().st_mtime))}
    if rule != "score_then_efficiency":
        return max(rows, key=lambda r: (r["mean"], -order[r["hash"]]))
    rows.sort(key=lambda r: (0 if r["hash"] == current_hash else 1, order[r["hash"]]))
    best = rows[0]
    for row in rows[1:]:
        if row["mean"] > best["mean"] + 1e-9:
            best = row
        elif abs(row["mean"] - best["mean"]) <= 1e-9 and (
                row["result"].get("eval_fingerprint") == best["result"].get("eval_fingerprint")
                and compare_successful_efficiency(best["result"], row["result"])["improved"]):
            best = row
    return best


def finalize_best_parent(run_dir: Path, line: str, *, episode_semaphore=None) -> dict:
    """After the last generation: give a once-measured parent one more measurement, then pick the best.

    The last accepted version has only its own lucky gate measurement; without
    this it would reach the holdout on the winner's-curse sample alone.
    """
    config = _frozen(run_dir)
    line_dir = _line(run_dir, line)
    if not _reference_is_best(config):
        return {"changed": False, "reason": "gate.reference is not best"}
    parent = _parent_bundle(line_dir)
    path = _measurement_path(line_dir, parent)
    measured = _read(path).get("measurements", []) if path.exists() else []
    remeasured = None
    if len(measured) < 2 and not any(item["id"] == "final" for item in measured):
        out = line_dir / "final_remeasure"
        result, code = _evaluate(run_dir, config, line, parent, "parent-remeasure", out,
                                 unit_suffix=":rfinal", episode_semaphore=episode_semaphore)
        result["version_hash"] = tree_hash(parent)
        atomic_json(out / "scores.json", result)
        if code:
            return {"changed": False, "status": result.get("status"), "reason": "final remeasure incomplete"}
        _record_measurement(line_dir, parent, "final", result)
        remeasured = result.get("scores")
    return {**select_best_parent(run_dir, line, when="final"), "final_remeasure": remeasured}


def select_best_parent(run_dir: Path, line: str, *, when: str) -> dict:
    """With gate.reference: best, make the best measured version the lineage parent.

    A parent whose remeasured mean has fallen below an earlier accepted
    version is replaced by that version (rolled back).  Idempotent.
    """
    config = _frozen(run_dir)
    line_dir = _line(run_dir, line)
    if not _reference_is_best(config):
        return {"changed": False, "reason": "gate.reference is not best"}
    current_hash = tree_hash(_parent_bundle(line_dir))
    best = _best_version(line_dir, rule=config["gate"]["rule"], current_hash=current_hash)
    if best is None or best["hash"] == current_hash:
        return {"changed": False, "parent": current_hash}
    current = next((row for row in _measured_versions(line_dir) if row["hash"] == current_hash), None)
    version = best["path"]
    pointer = {"version_hash": best["hash"], "comparison_id": f"rollback_{when}",
               "bundle": str((version / "bundle").relative_to(line_dir)),
               "scores": str((version / "scores.json").relative_to(line_dir)),
               "evidence": str((version / "evidence").relative_to(line_dir)) if (version / "evidence").is_dir() else None}
    Promotion(line_dir)  # ensures versions/ and promotion_journal/ exist
    atomic_json(line_dir / "promotion_journal" / f"rollback_{when}.json",
                       {"schema_version": 1, "kind": "rollback", "from": current_hash, "to": best["hash"],
                        "from_mean": current and current["mean"], "from_n": current and current["n"],
                        "to_mean": best["mean"], "to_n": best["n"], "state": "committed"})
    atomic_json(line_dir / "parent.current.json", pointer)
    _replace_tree(version / "bundle", line_dir / "parent")
    if (version / "evidence").is_dir():
        _replace_tree(version / "evidence", line_dir / "parent_evidence")
    atomic_json(line_dir / "parent_scores.json", _read(version / "scores.json"))
    lineage = line_dir / "lineage" / "lineage.md"
    lineage.parent.mkdir(parents=True, exist_ok=True)
    if not lineage.exists():
        lineage.write_text("# Lineage\n\n")
    note = (f"- {when}: parent rolled back to the best measured version (accepted at {best['first_id']}, "
            f"mean {best['mean']:.3f} over {best['n']} measurement(s)); the previous parent's mean was "
            + (f"{current['mean']:.3f} over {current['n']}" if current else "unmeasured") + ".\n")
    if note not in lineage.read_text(encoding="utf-8"):
        with lineage.open("a", encoding="utf-8") as stream:
            stream.write(note)
    return {"changed": True, "from": current_hash, "to": best["hash"], "to_mean": best["mean"], "to_n": best["n"]}


def _sync_feedback_to_lineage(line_dir: Path) -> None:
    """Project host-owned facts into the only directory copied to workspaces."""
    source, destination = line_dir / "host_feedback", line_dir / "lineage" / "host_feedback"
    if not source.is_dir():
        return
    if destination.exists():
        writable(destination)
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _shadowed_review(slot_dir: Path) -> list[str]:
    path = Path(slot_dir) / "review.json"
    return _read(path).get("shadowed_modules", []) if path.is_file() else []


def _replace_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        writable(destination)
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _eval_fingerprint(config: dict, line: str, *, no_automem: bool = False) -> str:
    public = {"benchmark": config.get("benchmark", {}), "model": config.get("model", {}),
              "runtime": config.get("runtime", {}), "line": line, "no_automem": bool(no_automem),
              "score_version": 2, "runtime_sha256": startup_runtime()["sha256"]}
    return hashlib.sha256(json.dumps(public, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _cached_evaluation(prior: dict, config: dict, line: str, bundle: Path, *,
                       seeds=None, no_automem=False, unit_suffix="") -> bool:
    if not prior.get("complete"):
        return False
    wanted = config["seeds"]["dev"] if seeds is None else seeds
    expected = {f"{int(seed)}:{rep}{unit_suffix}" for seed in wanted
                for rep in range(int(config["noise"]["parent_repetitions"]))}
    if (prior.get("candidate_hash") != tree_hash(bundle)
            or prior.get("eval_fingerprint") != _eval_fingerprint(config, line, no_automem=no_automem)
            or set(prior.get("expected_units", [])) != expected):
        raise ValueError("completed evaluation identity changed; preserve old evidence and use a new run directory")
    return True


def _episode_efficiency(summary: dict, attempt_dir: Path) -> dict:
    """Extract measurement costs from the same selected episode attempt.

    Missing token records stay ``None``.  This is intentionally a diagnostic
    projection and never changes the strict_mean inputs used by the gate.
    """
    input_tokens = output_tokens = tool_calls = legacy_memory_calls = 0
    by_name: dict[str, int] = {}
    usage_rows = 0
    calls = Path(attempt_dir) / "calls.jsonl"
    if calls.is_file():
        for raw in calls.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(raw)
                value_in, value_out = row.get("input_tokens"), row.get("output_tokens")
                if type(value_in) in (int, float) and type(value_out) in (int, float):
                    input_tokens += int(value_in)
                    output_tokens += int(value_out)
                    usage_rows += 1
                for event in row.get("tool_events") or []:
                    if isinstance(event, dict) and event.get("event") == "tool_result":
                        tool_calls += 1
                        legacy_memory_calls += int(event.get("name") in LEGACY_MEMORY_TOOLS)
                        by_name[str(event.get("name"))] = by_name.get(str(event.get("name")), 0) + 1
            except json.JSONDecodeError:
                continue
    # Record whether the model used any tool during the episode.
    requests = summary.get("requests")
    have_usage = type(requests) is int and requests > 0 and usage_rows == requests
    success = (False if summary.get("status") in {"candidate_crashed", "candidate_hung"}
               else summary.get("native_success"))
    return {"steps": summary.get("steps"), "requests": requests, "success": success,
            "input_tokens": input_tokens if have_usage else None,
            "output_tokens": output_tokens if have_usage else None,
            "tool_calls": tool_calls if calls.is_file() else None,
            "legacy_memory_calls": legacy_memory_calls if calls.is_file() else None,
            "tool_calls_by_name": by_name if calls.is_file() else None,
            "memory_bypass": (tool_calls == 0) if calls.is_file() else None,
            "action_overrides": summary.get("action_overrides"),
            "source": "selected_attempt_calls_and_summary", "token_scope": "provider_reported"}


INFRA_RETRY_BACKOFF = (60, 300, 900)  # seconds before re-running an infrastructure-failed unit


def _evaluate(run_dir: Path, config: dict, line: str, bundle: Path, phase: str, out_dir: Path,
              seeds: list | None = None, no_automem: bool = False,
              unit_suffix: str = "", shared_ledger: Ledger | None = None,
              episode_semaphore=None) -> tuple[dict, int]:
    bind_runtime(run_dir)
    # Each parallel seed owns a model context and shares the evaluation ledger.
    wanted_parallel = seeds if seeds is not None else config["seeds"]["dev"]
    if config.get("runtime", {}).get("parallel_units") and len(wanted_parallel) > 1:
        shared = shared_ledger or Ledger(run_dir / "ledger.json", config["budget"].get("price_per_million", {}))
        with ThreadPoolExecutor(max_workers=len(wanted_parallel), thread_name_prefix="unit") as pool:
            futures = [pool.submit(_evaluate, run_dir, config, line, bundle, phase, out_dir, seeds=[seed],
                                   no_automem=no_automem, unit_suffix=unit_suffix, shared_ledger=shared,
                                   episode_semaphore=episode_semaphore) for seed in wanted_parallel]
            parts = [future.result()[0] for future in futures]
        merged = _merge_results(parts)
        return merged, 0 if merged["status"] == "completed" else 1
    authorization = load_authorization(run_dir, "candidate-eval" if phase == "parent-remeasure" else phase)
    ledger = shared_ledger or Ledger(run_dir / "ledger.json", config["budget"].get("price_per_million", {}))
    model = PaidModel(config, authorization, ledger, run_dir / "requests.jsonl")
    repetitions = int(config["noise"]["parent_repetitions"])
    scores, efficiency_rows, guards = {}, {}, {"crashes": 0, "requests": 0, "protocol_errors": 0, "fallbacks": 0, "steps": 0}
    def add_guards(summary: dict) -> None:
        guards["crashes"] += int(summary["status"] in {"candidate_crashed", "candidate_hung", "containment_violation"})
        for key in ("requests", "protocol_errors", "fallbacks", "steps"):
            guards[key] += int(summary.get(key, 0))
    outcomes, status = [], "completed"
    # Keep interrupted attempts so resume only runs missing units.
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_seeds = seeds if seeds is not None else config["seeds"]["dev"]
    expected_units = {f"{int(seed)}:{repetition}{unit_suffix}" for seed in wanted_seeds for repetition in range(repetitions)}
    bundle_digest, fingerprint = tree_hash(bundle), _eval_fingerprint(config, line, no_automem=no_automem)
    failure_score = (0.0 if config["benchmark"]["adapter"] in {"minigrid_memory", "minigrid_keycorridor"}
                     and config["gate"]["rule"] == "score_then_efficiency" else None)
    for seed in wanted_seeds:
        for repetition in range(repetitions):
            unit_id = f"{int(seed)}:{repetition}{unit_suffix}"
            unit_dir = out_dir / "units" / unit_id.replace(":", "_")
            attempts_dir = unit_dir / "attempts"
            store = AttemptStore(out_dir, unit_id)
            summary = store.completed(bundle_digest, fingerprint)
            if summary is not None:
                outcome = classify_summary(summary, logical_round_id=f"{line}:{phase}", logical_unit_id=unit_id,
                                           attempt_id=str(summary.get("outcome", {}).get("attempt_id", "reused")), candidate_hash=bundle_digest,
                                           eval_fingerprint=fingerprint, candidate_failure_score=failure_score)
                outcomes.append(outcome)
                if outcome.score_valid:
                    scores.setdefault(str(seed), []).append(float(outcome.score))
                    efficiency_rows[unit_id] = _episode_efficiency(summary, summary.get("_attempt_dir", unit_dir / "attempts" / str(summary.get("outcome", {}).get("attempt_id", ""))))
                    add_guards(summary)
                continue
            existing = sorted(attempts_dir.glob("a*/summary.json")) if attempts_dir.exists() else []
            if existing:
                previous = _read(existing[-1])
                prior = classify_summary(previous, logical_round_id=f"{line}:{phase}", logical_unit_id=unit_id,
                                         attempt_id=existing[-1].parent.name, candidate_hash=bundle_digest,
                                         eval_fingerprint=fingerprint, candidate_failure_score=failure_score)
                if prior.execution_status in {"method_failed", "blocked_budget", "blocked_auth", "blocked_unknown"}:
                    outcomes.append(prior)
                    continue
            if (authorization.get("enforcement") != "record_only" and
                    len(outcomes) >= int(authorization["max_units"])):
                status = "budget_exhausted"
                break
            backoff = INFRA_RETRY_BACKOFF if config.get("runtime", {}).get("infra_retry") is True else ()
            for infra_try in range(len(backoff) + 1):
                attempt_id, attempt_dir = store.begin(bundle_digest, fingerprint)
                adapter = get_adapter(config["benchmark"]["adapter"], config["benchmark"])
                ledger.reserve_unit(str(authorization.get("batch_id", "offline")), unit_id, attempt_id,
                                    int(authorization["max_units"]),
                                    record_only=authorization.get("enforcement") == "record_only")
                model.set_context(unit_id, attempt_id)
                with episode_semaphore if episode_semaphore is not None else nullcontext():
                    summary = run_episode(adapter, bundle, line=line, seed=int(seed), model=model,
                                          archive=attempt_dir,
                                          max_steps=config["benchmark"].get("max_steps"), mode=config["runtime"]["mode"],
                                          compute_timeout=_compute_timeout(config),
                                          no_automem=no_automem, phase_caps=_phase_caps(config))
                outcome = classify_summary(summary, logical_round_id=f"{line}:{phase}", logical_unit_id=unit_id,
                                           attempt_id=attempt_dir.name, candidate_hash=bundle_digest,
                                           eval_fingerprint=fingerprint, candidate_failure_score=failure_score)
                if outcome.score_valid and summary.get("status") in {"candidate_crashed", "candidate_hung"}:
                    summary["native_success"] = False
                summary["outcome"] = outcome.json()
                store.finalize(attempt_id, summary, select=outcome.score_valid)
                if outcome.execution_status != "infrastructure_failed" or infra_try == len(backoff):
                    break
                time.sleep(backoff[infra_try])
            outcomes.append(outcome)
            if outcome.score_valid:
                ledger.add(episodes_completed=1)
                scores.setdefault(str(seed), []).append(float(outcome.score))
                efficiency_rows[unit_id] = _episode_efficiency(summary, attempt_dir)
            if outcome.execution_status == "infrastructure_failed":
                status = "infrastructure_failed"
                break
            if outcome.execution_status.startswith("blocked_"):
                status = str(summary.get("status", outcome.execution_status))
                break
            add_guards(summary)
        if status != "completed":
            break
    rates = {"crashes": guards["crashes"],
             "protocol_error_rate": guards["protocol_errors"] / max(1, guards["requests"]),
             "fallback_rate": guards["fallbacks"] / max(1, guards["steps"]), "pseudo_strategy_rate": 0.0}
    aggregate = aggregate_outcomes(outcomes, expected_units)
    aggregate["selection_scores"] = aggregate.pop("scores")
    result = {"status": status if status != "completed" else ("completed" if aggregate["scores_complete"] else "incomplete"),
              "episodes": len(outcomes), "scores": scores, "guards": rates, "candidate_hash": bundle_digest,
              "guard_counts": guards,
              "eval_fingerprint": fingerprint, "episode_efficiency": efficiency_rows,
              "no_automem": bool(no_automem), **aggregate}
    return result, 0 if result["status"] == "completed" else 1


def _noise_summary(config: dict, scores: dict) -> str:
    flat = [s for values in scores.values() for s in (values if isinstance(values, list) else [values])]
    if not flat:
        return "no parent scores recorded"
    mde = _mde(config, scores)
    band = difference_band(config, {"scores": scores})
    band_note = ("" if band["band"] is None else
                 " A paired mean score difference within ±%.3f (%s) is expected from noise alone."
                 % (band["band"], band["source"]))
    if config.get("experiment_mode") == "exploratory":
        return ("%d fixed development seed(s), one episode each; variance, standard error, and minimum "
                "detectable effect are not estimated. Scores are for exploratory paired-total selection only."
                % len(scores) + band_note)
    return ("parent mean %.3f over %d episodes on %d dev seeds; mean within-seed variance %.3f; "
            "minimum detectable effect at 80%% power and alpha=%s: %.3f"
            % (sum(flat) / len(flat), len(flat), len(scores), mde.get("mean_variance", 0.0),
               config["gate"].get("alpha", 0.05), mde.get("mde_80pct") or 0.0) + "." + band_note)


def _compute_timeout(config: dict) -> float | None:
    """Backstop per wait on the candidate: 30 minutes unless the config sets another value (null = none)."""
    runtime = config.get("runtime") or {}
    if "compute_timeout_seconds" not in runtime:
        return BACKSTOP_SECONDS
    value = runtime["compute_timeout_seconds"]
    return None if value is None else float(value)


def _phase_caps(config: dict) -> dict[str, int]:
    """Per-step model-request caps; upstream AutoMem and the default use 10 each."""
    caps = (config.get("runtime") or {}).get("phase_caps") or {}
    return {"LOG": int(caps.get("LOG", 10)), "PLAN": int(caps.get("PLAN", 10))}


def _feedback_mode(config: dict) -> str:
    return str(config.get("meta", {}).get("feedback", "basic"))


def _contracts(feedback: str = "basic", caps: dict[str, int] | None = None) -> dict[str, str]:
    host = ROOT / "host"
    caps = caps or {"LOG": 10, "PLAN": 10}
    limits = VisualLimits()
    extra = {"predictions_schema.md": PREDICTION_SCHEMA} if feedback == "enhanced" else {}
    return {**extra,
        "registry.py": (host / "child" / "registry.py").read_text(encoding="utf-8"),
        "worker_contract.md": (
            "# Candidate runtime contract\n\n"
            "- `candidate.py` exports `Agent` (MemoryAgent-compatible) and optionally `install(agent, registry)`.\n"
            "- Candidates import `from registry import Tool`; new tools take JSON arguments.\n"
            "- The candidate's own `act`, `_log_phase` and `_plan_validate_phase` run unchanged; the runtime\n"
            "  replaces `agents.memory_agent.parse_log_calls` / `parse_plan_calls`, the instance\n"
            "  `_execute_read_op` / `_execute_write_op`, and `_format_tool_result` for non-legacy tools.\n"
            f"- Host limits per step: at most {caps['LOG']} LOG and {caps['PLAN']} PLAN model requests, LOG only after a real action and\n"
            "  before PLAN; the executed action must equal the model's last committed ACTION (or the fallback).\n"
            "- Line M: `agent.current_image`, `agent.current_frame_id`; set `agent.visual_request` only inside a\n"
            "  model-invoked tool. Lines T and V: no pixels and no visual requests.\n"
            "- State must be JSON-serialisable or provide `export_state` / `import_state`.\n"),
        "visual_channel.md": (ROOT / "docs" / "visual_channel.md").read_text(encoding="utf-8"),
        "visual_limits.json": json.dumps(limits.__dict__, indent=2) + "\n",
    }


def _visual_limits_text() -> str:
    limits = VisualLimits()
    return (f"operations frame/crop/resize/grayscale/quantize/diff/tile/annotate/canvas/draw (canvas and draw on M only); "
            f"canvas: semantic palette colors only; "
            f"output side <= {limits.max_output_side}px; "
            f"tile <= {limits.max_tile_frames} frames; no cap on the number of historical images (all counted); "
            f"only frames already observed in this episode; the image arrives, labelled, with the next model request")


def cmd_prepare(args) -> tuple[dict, int]:
    config = load_config(args.config)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("run directory must be empty")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", config)
    Ledger(run_dir / "ledger.json", config["budget"].get("price_per_million", {})).add()
    result = {"status": "prepared", "run_dir": str(run_dir), "mde": _mde(config, None),
              "runtime_mode": config["runtime"]["mode"]}
    if config["runtime"]["mode"] != "seatbelt":
        result["warning"] = "runtime.mode is not seatbelt: use only for offline checks, never for formal runs"
    return result, 0


def _start_bundle(config: dict, line: str) -> Path:
    """Use the shipped visual memory program, including for older M0 manifests."""
    name = config.get("bundle")
    if line != "M" or name not in (None, "", "visual_memory", "M0"):
        raise ValueError(f"Unsupported starting program for line {line!r}: {name!r}. Use visual_memory on line M.")
    return ROOT / "inner" / "visual_memory"


def cmd_parent_eval(args) -> tuple[dict, int]:
    run_dir, config = args.run_dir, _frozen(args.run_dir)
    line_dir = _line(run_dir, args.line)
    if not _parent_bundle(line_dir).exists():
        install_bundle(line_dir / "parent", _start_bundle(config, args.line))
    existing = line_dir / "parent_scores.json"
    if existing.exists():
        prior = _read(existing)
        if _cached_evaluation(prior, config, args.line, _parent_bundle(line_dir)):
            return {**prior, "reused": True, "mde": _mde(config, prior.get("scores"))}, 0
    result, code = _evaluate(run_dir, config, args.line, _parent_bundle(line_dir), "parent-eval", line_dir / "parent_evidence")
    atomic_json(line_dir / "parent_scores.json", result)
    if result.get("complete") and config["gate"].get("parent_remeasure", False):
        _record_measurement(line_dir, _parent_bundle(line_dir), "G0", result)
        _store_version(line_dir, _parent_bundle(line_dir), line_dir / "parent_evidence", result)
    return {**result, "mde": _mde(config, result["scores"])}, code


def cmd_ledger(args) -> tuple[dict, int]:
    return Ledger(args.run_dir / "ledger.json", {}).snapshot(), 0


def cmd_meta_workspace(args) -> tuple[dict, int]:
    run_dir, config = args.run_dir, _frozen(args.run_dir)
    line_dir = _line(run_dir, args.line)
    if not (line_dir / "parent_scores.json").exists():
        raise FileNotFoundError("run parent-eval for this line first")
    if args.slot == "fresh" and _reference_is_best(config):
        select_best_parent(run_dir, args.line, when=f"before_g{int(args.generation)}")
    scores = _parent_scores(line_dir).get("scores", {})
    episodes = sorted(_parent_evidence(line_dir).rglob("episode.json"))
    if not episodes:
        raise FileNotFoundError("parent evidence has no episode.json for Meta workspace")
    episode = _read(episodes[0])
    instruction = episode.get("instruction", "")
    metric = episode["metric_definition"]
    values = {
        "BENCHMARK_NAME": config["benchmark"]["adapter"],
        "BENCHMARK_DESCRIPTION": config["benchmark"].get("description")
        or "The agent receives this task instruction from the environment:\n\n" + instruction,
        "METRIC_DESCRIPTION": metric,
        "SELECTION_RULE": json.dumps(config["gate"], sort_keys=True),
        "NOISE_SUMMARY": _noise_summary(config, scores),
        "PHASE_CAPS": "at most %d LOG and %d PLAN model requests per step" % (
            _phase_caps(config)["LOG"], _phase_caps(config)["PLAN"]),
        "VISUAL_LIMITS": _visual_limits_text(),
        "TIME_BUDGET_MINUTES": config["meta"].get("timeout_minutes", 60),
    }
    _sync_feedback_to_lineage(line_dir)
    prior_evidence = None
    if args.slot == "revision":
        # Include the fresh slot’s failure status even if no program was collected.
        fresh = _slot(run_dir, args.line, args.generation, "fresh")
        prior_documents = line_dir / "lineage" / f"g{int(args.generation)}_fresh"
        if not (prior_documents / "status.json").exists() and (fresh / "status.json").exists():
            prior_documents.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fresh / "status.json", prior_documents / "status.json")
        if (fresh / "evidence" / "units").is_dir():
            prior_evidence = fresh / "evidence"
    feedback_evidence = _parent_evidence(line_dir)
    feedback_units = set(config.get("meta", {}).get("feedback_units") or [])
    if feedback_units:
        projected = _slot(run_dir, args.line, args.generation, args.slot) / "feedback_evidence"
        if not projected.exists():
            (projected / "units").mkdir(parents=True)
            for unit in feedback_units:
                source = feedback_evidence / "units" / unit.replace(":", "_")
                if not source.is_dir():
                    raise ValueError(f"feedback unit missing: {unit}")
                shutil.copytree(source, projected / "units" / source.name)
        feedback_evidence = projected
    workspace = build_workspace(args.workspace_root, line=args.line, generation=args.generation, slot=args.slot,
                                source=_parent_bundle(line_dir), evidence=feedback_evidence,
                                lineage=line_dir / "lineage", meta_state=line_dir / "meta_state",
                                contracts=_contracts(_feedback_mode(config), _phase_caps(config)), values=values,
                                spatial=bool(config["meta"].get("spatial_analysis", True)),
                                visual=bool(config["meta"].get("visual_analysis", False)),
                                self_improvement=str(config["meta"].get("self_improvement", "none")),
                                feedback=_feedback_mode(config), prior_evidence=prior_evidence)
    record = {"workspace": str(workspace)}
    if config["meta"].get("evidence_images", True) is False:
        record["images_withheld"] = _withhold_images(workspace)
    atomic_json(_slot(run_dir, args.line, args.generation, args.slot) / "workspace.json", record)
    return {"workspace": str(workspace)}, 0


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
WITHHELD_NOTE = """

## Images withheld (ablation: meta.evidence_images = false)

The host removed every image file from your inputs: contact sheets, observation frames, model-request images and
final observation images, in the parent evidence and in every lineage directory.  Any mention of contact sheets or
frames elsewhere refers to files that are not provided in this lineage.  Reason from the text record only (public
text observations, model requests and replies, tool calls and results, memory snapshots, metrics).  Do not look for
images outside this workspace.
"""


def _image_magic(path: Path) -> bool:
    """Content-addressed blobs have no suffix; recognise image files by their first bytes."""
    try:
        with path.open("rb") as stream:
            head = stream.read(12)
    except OSError:
        return False
    return (head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8\xff") or head.startswith(b"GIF8")
            or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"))


def _withhold_images(workspace: Path) -> int:
    """Remove image evidence when the manifest requests a text-only revision workspace."""
    inputs = Path(workspace) / "input"
    writable(inputs)
    removed = 0
    for path in sorted(inputs.rglob("*")):
        if path.is_file() and (path.suffix.lower() in IMAGE_SUFFIXES or _image_magic(path)):
            path.unlink()
            removed += 1
    for path in sorted(inputs.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and path.name in {"contact_sheets", "frames", "model_images"} and not any(path.iterdir()):
            path.rmdir()
    env = inputs / "contracts" / "meta_environment.md"
    env.write_text(env.read_text(encoding="utf-8") + WITHHELD_NOTE, encoding="utf-8")
    readonly(inputs)
    return removed


REPLAY_BACKSTOP_SECONDS = 2 * 3600


def _host_replay(candidate: Path, parent: Path, evidence: Path, line: str, output: Path, mode: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="evovim-host-replay-") as tmp:
        replay = Path(tmp) / "replay"
        install_replay(replay)
        try:
            process = subprocess.run([sys.executable, "-B", str(replay / "replay_main.py"), "--candidate", str(candidate),
                                      "--parent", str(parent), "--evidence", str(evidence), "--line", line,
                                      "--output", str(output), "--mode", mode], capture_output=True, text=True,
                                     timeout=REPLAY_BACKSTOP_SECONDS)
        except subprocess.TimeoutExpired:
            # The replay's worker exits on its own once its parent is gone.
            return {"crashed": True, "status": "replay_hung",
                    "traceback": f"host replay ran for more than {REPLAY_BACKSTOP_SECONDS} s (backstop against infinite loops)"}
    if process.returncode or not output.exists():
        return {"crashed": True, "status": "replay_failed", "traceback": process.stderr[-4000:]}
    return _read(output)


def _write_contact_sheets(evidence: Path, destination: Path) -> int:
    """Step-labelled frame sheets for every evaluated unit (every frame, 24 per sheet)."""
    written = 0
    for frames in sorted(Path(evidence).glob("units/*/attempts/*/frames")) if Path(evidence).is_dir() else []:
        unit = frames.parents[2].name
        paths = sorted(frames.glob("f*.png"))
        for part, start in enumerate(range(0, len(paths), 24)):
            make_contact_sheet(frames, destination / unit / f"sheet_{part:02d}.png", first=start)
            written += 1
    return written


def _append_interface_log(line_dir: Path, comparison_id: str, workspace: Path) -> None:
    """All interface requests in one place, so later generations see what was asked before."""
    source = workspace / "output" / "interface_requests.md"
    if not source.is_file() or not source.read_text(encoding="utf-8").strip():
        return
    log = line_dir / "lineage" / "interface_requests_log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not log.exists():
        log.write_text("# Interface requests from every slot\n\nHost-collected; researchers' responses, when "
                       "there are any, are recorded in researcher_responses.md.\n\n", encoding="utf-8")
    heading = f"## {comparison_id}"
    if heading not in log.read_text(encoding="utf-8"):
        with log.open("a", encoding="utf-8") as stream:
            stream.write(f"{heading}\n\n{source.read_text(encoding='utf-8').strip()}\n\n")


def _record_attempt(line_dir: Path, generation: int, slot: str, workspace: Path, result: dict,
                    *, enhanced: bool = False) -> None:
    """Every collected attempt, successful or not, is kept for later generations and the revision slot."""
    if enhanced:
        _append_interface_log(line_dir, f"g{int(generation)}_{slot}", workspace)
    documents = line_dir / "lineage" / f"g{int(generation)}_{slot}"
    documents.mkdir(parents=True, exist_ok=True)
    for name in META_DOCUMENTS:
        if (workspace / "output" / name).is_file():
            shutil.copyfile(workspace / "output" / name, documents / name)
    revised = workspace / "output" / "revised_code"
    if revised.is_dir():
        if (documents / "revised_code").exists():
            shutil.rmtree(documents / "revised_code")
        shutil.copytree(revised, documents / "revised_code", ignore=shutil.ignore_patterns("__pycache__"))
    atomic_json(documents / "status.json", result)


def record_terminal(line_dir: Path, generation: int, slot: str, result: dict,
                    *, enhanced: bool = False) -> None:
    """Expose a terminal slot to later Meta workspaces exactly once."""
    status = str(result.get("status", "unknown"))
    review = result.get("review") or {}
    replay = result.get("replay") or {}
    meta = result.get("meta") or {}
    evaluation = result.get("evaluation") or result
    if status == "review_failed":
        reason = next(iter(review.get("findings") or []), status)
    elif status == "smoke_failed":
        reason = (str(replay.get("traceback") or "").strip().splitlines() or [status])[-1]
    elif status == "method_failed" and meta:
        reason = f"{meta.get('status', status)}: {meta.get('error', '')}"
    elif status in {"method_failed", "infrastructure_failed", "incomplete", "technical_unresolved"}:
        failures = [item for item in evaluation.get("outcomes", []) if not item.get("score_valid")]
        failed = failures[0] if failures else {}
        unit = str(failed.get("logical_unit_id", "")).replace(":", "_")
        attempt = str(failed.get("attempt_id", ""))
        summary_path = (line_dir / "generations" / f"g{int(generation)}_{slot}" /
                        "evidence" / "units" / unit / "attempts" / attempt / "summary.json")
        summary = _read(summary_path) if summary_path.is_file() else {}
        reason = f"{evaluation.get('status', status)}: {summary.get('stop_reason') or failed.get('error_code') or ''}"
    else:
        reason = str(result.get("reason") or status)
    reason = " ".join(str(reason).split())[:300]
    lineage = line_dir / "lineage" / "lineage.md"
    lineage.parent.mkdir(parents=True, exist_ok=True)
    if not lineage.exists():
        lineage.write_text("# Lineage\n\n", encoding="utf-8")
    prefix = f"- generation {int(generation)} ({slot}):"
    if not any(row.startswith(prefix) for row in lineage.read_text(encoding="utf-8").splitlines()):
        shadowed = (review.get("shadowed_modules") or
                    _shadowed_review(line_dir / "generations" / f"g{int(generation)}_{slot}"))
        with lineage.open("a", encoding="utf-8") as stream:
            stream.write(f"{prefix} {status}; reason: {reason}; shadowed_modules: {shadowed}; "
                         f"documents in g{int(generation)}_{slot}/\n")
    if enhanced:
        locked_path = line_dir / "lineage" / f"g{int(generation)}_{slot}" / "predictions.json"
        locked = lock_predictions(locked_path)
        outcome = {"prediction_status": locked.get("status", "missing"), "problems": locked.get("problems", []),
                   "score_band": {"band": None, "source": "not evaluated"},
                   "rows": [{**item, "parent": None, "candidate": None, "observed_delta": None,
                             "status": "unmeasured"} for item in locked.get("predictions", [])]}
        append_prediction_table(line_dir / "lineage", f"g{int(generation)}_{slot}", outcome,
                                verdict=f"not evaluated: {status}")


def cmd_collect(args) -> tuple[dict, int]:
    run_dir, config = args.run_dir, _frozen(args.run_dir)
    line_dir = _line(run_dir, args.line)
    slot_dir = _slot(run_dir, args.line, args.generation, args.slot)
    workspace = Path(args.workspace).resolve()
    state_note = None
    source_state = workspace / "output" / "meta_state"
    if source_state.exists():
        try:
            state_note = transfer_state(source_state, line_dir / "meta_state_next")
            _replace_tree(line_dir / "meta_state_next", line_dir / "meta_state")
        except ValueError as exc:
            state_note = {"rejected": str(exc)}
        finally:
            shutil.rmtree(line_dir / "meta_state_next", ignore_errors=True)
    overlay = overlay_files(workspace / "output" / "revised_code")
    # Collect runs after the Meta session and before any evaluation, so the
    # copy made here is the pre-registered prediction.  Annotation only.
    predictions = None
    if _feedback_mode(config) == "enhanced":
        locked = lock_predictions(workspace / "output" / "predictions.json", slot_dir / "predictions.json",
                                  line_dir / "lineage" / f"g{int(args.generation)}_{args.slot}" / "predictions.json")
        predictions = {key: locked.get(key) for key in ("status", "sha256", "problems")}

    def finish(result: dict, code: int) -> tuple[dict, int]:
        result["meta_state"] = state_note
        if predictions is not None:
            result["predictions"] = predictions
        atomic_json(slot_dir / "status.json", result)
        _record_attempt(line_dir, args.generation, args.slot, workspace, result,
                        enhanced=_feedback_mode(config) == "enhanced")
        if code:
            record_terminal(line_dir, args.generation, args.slot, result,
                            enhanced=_feedback_mode(config) == "enhanced")
        feedback = finalize_feedback(line_dir, f"g{int(args.generation)}_{args.slot}", "collect", {
            "verdict": "rejected" if code else "collected", "accepted": False,
            "score_valid": False, "collect_status": result.get("status"),
            "review": result.get("review"), "replay": result.get("replay"),
            "shadowed_modules": _shadowed_review(slot_dir),
            "evidence_refs": [f"lineage/g{int(args.generation)}_{args.slot}"],
        })
        atomic_json(slot_dir / "host_feedback.json", feedback)
        _sync_feedback_to_lineage(line_dir)
        return result, code

    if not overlay:
        return finish({"status": "invalid_output", "reason": "output/revised_code contains no Python files"}, 1)
    review = review_candidate(_parent_bundle(line_dir), workspace / "output" / "revised_code", line=args.line)
    atomic_json(slot_dir / "review.json", review)
    if not review["passed"]:
        return finish({"status": "review_failed", "review": review}, 1)
    if (slot_dir / "bundle").exists():
        writable(slot_dir / "bundle")
        shutil.rmtree(slot_dir / "bundle")
    bundle = install_bundle(slot_dir / "bundle", _parent_bundle(line_dir), overlay)
    replay = _host_replay(workspace / "output" / "revised_code", _parent_bundle(line_dir), _parent_evidence(line_dir),
                          args.line, slot_dir / "replay_report.json", config["runtime"]["mode"])
    if replay.get("crashed"):
        return finish({"status": "smoke_failed", "replay": {k: replay.get(k) for k in ("status", "traceback", "steps")}}, 1)
    result = {"status": "installed", "bundle": str(bundle), "changed_files": sorted(overlay),
              "structural_changes": review["structural_changes"],
              "replay": {k: replay.get(k) for k in ("steps", "tools_called_via_model", "tool_errors",
                                                    "changed_functions_executed", "visual_requests", "state_round_trip")},
              "meta_state": state_note}
    return finish(result, 0)


def cmd_candidate_eval(args) -> tuple[dict, int]:
    run_dir, config = args.run_dir, _frozen(args.run_dir)
    slot_dir = _slot(run_dir, args.line, args.generation, args.slot)
    status = _read(slot_dir / "status.json") if (slot_dir / "status.json").exists() else {}
    if status.get("status") != "installed":
        raise PermissionError("candidate-eval requires a successful collect")
    if config.get("review", {}).get("require_human", True) and not (slot_dir / "review_approval.json").exists():
        raise PermissionError(f"review.require_human: inspect {slot_dir}/status.json and the candidate source, "
                              f"then write {slot_dir}/review_approval.json")
    existing = slot_dir / "scores.json"
    if existing.exists():
        prior = _read(existing)
        if _cached_evaluation(prior, config, args.line, slot_dir / "bundle"):
            return {**prior, "reused": True}, 0
    load_authorization(run_dir, "candidate-eval")
    shared_ledger = Ledger(run_dir / "ledger.json", config["budget"].get("price_per_million", {}))
    episode_semaphore = getattr(args, "episode_semaphore", None)
    candidate_args = (run_dir, config, args.line, slot_dir / "bundle", "candidate-eval", slot_dir / "evidence")
    eval_kwargs = {"shared_ledger": shared_ledger, "episode_semaphore": episode_semaphore}
    candidate_result = None
    if config["gate"].get("parent_remeasure", False):
        line_dir = _line(run_dir, args.line)
        parent_bundle = _parent_bundle(line_dir)
        parent_hash = tree_hash(parent_bundle)
        generation_dir = line_dir / "generations"
        remeasure = None
        for kind in ("fresh", "revision"):
            prior_path = generation_dir / f"g{int(args.generation)}_{kind}" / "parent_remeasure" / "scores.json"
            if prior_path.exists():
                prior = _read(prior_path)
                if prior.get("version_hash") == parent_hash and _cached_evaluation(
                        prior, config, args.line, parent_bundle, unit_suffix=f":r{int(args.generation)}"):
                    remeasure = prior
                    break
        if remeasure is None:
            out = slot_dir / "parent_remeasure"
            parent_args = (run_dir, config, args.line, parent_bundle, "parent-remeasure", out)
            parent_kwargs = {**eval_kwargs, "unit_suffix": f":r{int(args.generation)}"}
            if getattr(args, "parallel_pair", True):
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="evaluation") as pool:
                    parent_future = pool.submit(_evaluate, *parent_args, **parent_kwargs)
                    candidate_future = pool.submit(_evaluate, *candidate_args, **eval_kwargs)
                    remeasure, parent_code = parent_future.result()
                    candidate_result = candidate_future.result()
            else:
                remeasure, parent_code = _evaluate(*parent_args, **parent_kwargs)
                if not parent_code:
                    candidate_result = _evaluate(*candidate_args, **eval_kwargs)
            remeasure["version_hash"] = parent_hash
            atomic_json(out / "scores.json", remeasure)
            if parent_code:
                return remeasure, parent_code
        _record_measurement(line_dir, parent_bundle, f"g{int(args.generation)}", remeasure)
        remeasure_evidence = next((d / "parent_remeasure" for d in (slot_dir, *(generation_dir / f"g{int(args.generation)}_{k}"
                                   for k in ("fresh", "revision"))) if (d / "parent_remeasure" / "units").is_dir()), None)
    else:
        remeasure_evidence = None
    result, code = candidate_result if candidate_result is not None else _evaluate(*candidate_args, **eval_kwargs)
    atomic_json(slot_dir / "scores.json", result)
    if _feedback_mode(config) == "enhanced":
        # lineage/ is the only directory later workspaces receive; the raw
        # candidate evidence stays host-side, so project a readable digest.
        documents = _line(run_dir, args.line) / "lineage" / f"g{int(args.generation)}_{args.slot}"
        write_episode_digest(slot_dir / "evidence", documents)
        _write_contact_sheets(slot_dir / "evidence", documents / "contact_sheets")
        if remeasure_evidence is not None:
            write_episode_digest(remeasure_evidence, documents / "parent_remeasure")
            _write_contact_sheets(remeasure_evidence, documents / "parent_remeasure" / "contact_sheets")
    feedback = finalize_feedback(_line(run_dir, args.line), f"g{int(args.generation)}_{args.slot}", "candidate_eval", {
        "verdict": "evaluation_failed" if code else "evaluated", "accepted": False,
        "score_valid": bool(result.get("scores_complete")), "evaluation": result,
        "evidence_refs": [str((slot_dir / "evidence").relative_to(_line(run_dir, args.line)))]})
    atomic_json(slot_dir / "host_feedback.json", feedback)
    _sync_feedback_to_lineage(_line(run_dir, args.line))
    return result, code


def cmd_gate(args) -> tuple[dict, int]:
    run_dir, config = args.run_dir, _frozen(args.run_dir)
    bind_runtime(run_dir)
    line_dir = _line(run_dir, args.line)
    slot_dir = _slot(run_dir, args.line, args.generation, args.slot)
    prior_gate = slot_dir / "gate.json"
    if prior_gate.exists():
        return {**_read(prior_gate), "reused": True}, 0
    comparison_id = f"g{int(args.generation)}_{args.slot}"
    promotion = Promotion(line_dir)
    remeasure_enabled = config["gate"].get("parent_remeasure", False)

    def publish_parent():
        record = promotion.commit(comparison_id)
        version = record["version"]
        bundle = line_dir / version["bundle"]
        scores = _read(line_dir / version["scores"])
        if remeasure_enabled:
            _record_measurement(line_dir, bundle, comparison_id, scores)
        _replace_tree(bundle, line_dir / "parent")
        _replace_tree(line_dir / version["evidence"], line_dir / "parent_evidence")
        atomic_json(line_dir / "parent_scores.json", scores)
        return record["verdict"]

    # Finish an interrupted promotion using the original gate decision.
    if (promotion.journal / f"{comparison_id}.json").is_file():
        result = publish_parent()
        atomic_json(prior_gate, result)
        return result, 0
    parent_initial, candidate = _parent_scores(line_dir), _read(slot_dir / "scores.json")
    reference_bundle, reference_evidence = _parent_bundle(line_dir), _parent_evidence(line_dir)
    parent, parent_n = (_measured_parent(line_dir, reference_bundle, parent_initial)
                        if remeasure_enabled else (parent_initial, 1))
    derived_from, derived_n = parent, parent_n  # the parent this candidate was built on
    reference_id = None
    if _reference_is_best(config):
        best = _best_version(line_dir, rule=config["gate"]["rule"], current_hash=tree_hash(_parent_bundle(line_dir)))
        if best is not None and best["hash"] != tree_hash(_parent_bundle(line_dir)):
            first = _read(best["path"] / "measurements.json")["measurements"][0]["result"]
            reference_bundle, reference_evidence = best["path"] / "bundle", best["path"] / "evidence"
            parent, parent_n = _measured_parent(line_dir, reference_bundle, first)
        reference_id = best["first_id"] if best else None
    candidate["efficiency"] = efficiency(parent_initial.get("episode_efficiency", {}), candidate.get("episode_efficiency", {}),
                                         set(parent.get("expected_units", [])))
    atomic_json(slot_dir / "scores.json", candidate)
    if not parent.get("selection_eligible", parent.get("complete")) or not candidate.get("selection_eligible", candidate.get("complete")):
        result = {"accepted": False, "status": "not_comparable", "reason": "incomplete evaluation",
                  "rule": config["gate"]["rule"], "difference": None, "detail": {}}
    else:
        gate = config["gate"]
        selection_units = set(gate.get("selection_units") or [])
        if selection_units:
            def selected(value):
                projected = copy.deepcopy(value)
                available = set(map(str, value.get("expected_units", [])))
                if not selection_units <= available:
                    raise ValueError("configured selection units absent from evaluation")
                projected["expected_units"] = sorted(selection_units)
                for field in ("selection_scores", "efficiency_by_unit", "episode_efficiency"):
                    if field in projected:
                        projected[field] = {key: row for key, row in projected[field].items()
                                            if key in selection_units}
                return projected
            gate_parent, gate_candidate_result = selected(parent), selected(candidate)
        else:
            gate_parent, gate_candidate_result = parent, candidate
        result = gate_candidate(gate_parent, gate_candidate_result, rule=gate["rule"], delta_min=float(gate.get("delta_min", 0.0)),
                                z=float(gate.get("z", 1.96)), alpha=float(gate.get("alpha", 0.05)),
                                guards=gate.get("guards"),
                                unit_regression_max=gate.get("unit_regression_max"))
        if selection_units:
            result["selection_units"] = sorted(selection_units)
    if remeasure_enabled:
        result["parent_n"] = parent_n
        result["parent_mean"] = statistics.mean(value[0] for value in parent["scores"].values()) if parent["scores"] else None
    if reference_id is not None:
        result["reference_version"] = reference_id
    result["reference_hash"] = tree_hash(reference_bundle)
    if _feedback_mode(config) == "enhanced":
        records_path = _measurement_path(line_dir, reference_bundle)
        write_gate_comparison(parent, candidate, reference_evidence, slot_dir / "evidence",
                              line_dir / "lineage" / comparison_id, line_dir,
                              _read(records_path).get("measurements", []) if records_path.exists() else [])
    parent_before = {"version_hash": tree_hash(_parent_bundle(line_dir)), "scores": str(line_dir / "parent_scores.json"),
                     "eval_fingerprint": parent.get("eval_fingerprint")}
    atomic_json(line_dir / "lineage" / f"g{int(args.generation)}_{args.slot}" / "gate.json", result)
    feedback = finalize_feedback(line_dir, comparison_id, "gate", {
        "verdict": "accepted" if result.get("accepted") else "rejected", "accepted": bool(result.get("accepted")),
        "score_valid": bool(candidate.get("scores_complete")), "gate": result,
        "candidate_hash": candidate.get("candidate_hash"), "parent_before_hash": parent_before["version_hash"],
        "evidence_refs": [str((slot_dir / "evidence").relative_to(line_dir))],
        "missing_evidence": candidate.get("missing_units", [])})
    atomic_json(slot_dir / "host_feedback.json", feedback)
    _sync_feedback_to_lineage(line_dir)
    lineage = line_dir / "lineage" / "lineage.md"
    lineage.parent.mkdir(parents=True, exist_ok=True)
    if not lineage.exists():
        lineage.write_text("# Lineage\n\n")
    verdict = "accepted" if result.get("accepted") else "rejected"
    noise_note = ""
    if _feedback_mode(config) == "enhanced":
        # Annotations only: computed after the gate decided.
        band = difference_band(config, parent)
        noise_note = "; " + band_text(result.get("difference"), band)
        if remeasure_enabled:
            noise_note += f"; parent n={parent_n}"
        mechanism, mechanism_summary = mechanism_comparison(derived_from, candidate, parent_n=derived_n)
        (line_dir / "lineage" / comparison_id).mkdir(parents=True, exist_ok=True)
        (line_dir / "lineage" / comparison_id / "mechanism.md").write_text(mechanism, encoding="utf-8")
        if mechanism_summary:
            noise_note += "; " + mechanism_summary
        outcome = score_predictions(lock_predictions(slot_dir / "predictions.json"), derived_from, candidate, band)
        if remeasure_enabled:
            outcome["parent_n"] = parent_n
        atomic_json(slot_dir / "prediction_outcome.json", outcome)
        atomic_json(line_dir / "lineage" / comparison_id / "prediction_outcome.json", outcome)
        append_prediction_table(line_dir / "lineage", comparison_id, outcome, verdict=verdict)
    parent_count_note = f"; parent n={parent_n}" if remeasure_enabled and not noise_note else ""
    shadowed_note = f"shadowed_modules: {_shadowed_review(slot_dir)}; "
    reference_note = f" (best version, accepted at {reference_id})" if reference_id else ""
    line_text = (f"- generation {int(args.generation)} ({args.slot}): {verdict}; rule {result.get('rule', '-')}; "
                 f"parent mean {result.get('parent_mean', '-')}{reference_note}; candidate mean {result.get('candidate_mean', '-')}"
                 f"{noise_note}{parent_count_note}; {shadowed_note}"
                 f"documents in g{int(args.generation)}_{args.slot}/\n")
    if line_text not in lineage.read_text(encoding="utf-8"):
        with lineage.open("a", encoding="utf-8") as stream:
            stream.write(line_text)
    if result.get("accepted"):
        promotion.stage(comparison_id, parent_before, slot_dir / "bundle", slot_dir / "evidence", candidate, result)
        publish_parent()
    atomic_json(prior_gate, result)
    return result, 0


def _merge_results(results: list[dict]) -> dict:
    """Combine per-seed evaluations of one bundle into the single-call result shape."""
    merged = {"status": "completed", "episodes": 0, "scores": {}, "selection_scores": {}, "episode_efficiency": {},
              "outcomes": [], "expected_units": [], "missing_units": [], "duplicate_units": []}
    for result in results:
        if result.get("status") != "completed" and merged["status"] == "completed":
            merged["status"] = result.get("status")
        merged["episodes"] += int(result.get("episodes", 0))
        for key in ("scores", "selection_scores", "episode_efficiency"):
            merged[key].update(result.get(key) or {})
        for key in ("outcomes", "expected_units", "missing_units", "duplicate_units"):
            merged[key] += list(result.get(key) or [])
    for flag in ("complete", "scores_complete", "selection_eligible", "execution_complete"):
        merged[flag] = bool(results) and all(result.get(flag) for result in results)
    if all("guard_counts" in r for r in results):
        counts = {name: sum(int(r["guard_counts"].get(name, 0)) for r in results)
                  for name in ("crashes", "requests", "protocol_errors", "fallbacks", "steps")}
        merged["guard_counts"] = counts
        merged["guards"] = {"crashes": counts["crashes"],
                            "protocol_error_rate": counts["protocol_errors"] / max(1, counts["requests"]),
                            "fallback_rate": counts["fallbacks"] / max(1, counts["steps"]),
                            "pseudo_strategy_rate": 0.0}
    else:
        # Older frozen measurements have rates but no denominators.  Retain
        # their historical merge semantics instead of reporting false zeros.
        names = {name for r in results for name in (r.get("guards") or {})}
        merged["guards"] = {name: (sum(r.get("guards", {}).get(name, 0) for r in results)
                                   if name == "crashes" else statistics.mean(r.get("guards", {}).get(name, 0)
                                                                             for r in results)) for name in names}
    for key in ("candidate_hash", "eval_fingerprint", "no_automem"):
        if results:
            merged[key] = results[0].get(key)
    return merged


def cmd_holdout_eval(args) -> tuple[dict, int]:
    """Final evaluation on sealed holdout seeds; never feeds any Meta workspace.

    Seeds run concurrently (one evaluation per seed, each its own logical unit);
    a completed result is reused, so a resumed run never re-evaluates it.
    """
    if not args.unseal_holdout:
        raise PermissionError("holdout-eval requires --unseal-holdout; run it once, after the lineage is frozen")
    run_dir, config = args.run_dir, _frozen(args.run_dir)
    line_dir = _line(run_dir, args.line)
    seeds = config["seeds"].get("holdout") or []
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds.holdout must be a non-empty list of unique integers")
    if set(seeds) & set(config["seeds"].get("dev", [])):
        raise ValueError("seeds.holdout must be disjoint from seeds.dev")
    if args.which == "parent":
        bundle = _parent_bundle(line_dir)
    else:
        baseline = run_dir / "baseline" / "bundle"
        bundle = baseline if baseline.is_dir() else _start_bundle(config, args.line)
    out = line_dir / f"holdout_{args.which}"
    prior = _read(out / "scores.json") if (out / "scores.json").exists() else {}
    if _cached_evaluation(prior, config, args.line, bundle, seeds=seeds, no_automem=args.which == "n0"):
        return {**prior, "reused": True}, 0
    shared_ledger = Ledger(run_dir / "ledger.json", config["budget"].get("price_per_million", {}))
    semaphore = getattr(args, "episode_semaphore", None)
    no_automem = args.which == "n0"
    with ThreadPoolExecutor(max_workers=len(seeds), thread_name_prefix="holdout") as pool:
        futures = [pool.submit(_evaluate, run_dir, config, args.line, bundle, "holdout-eval", out / "evidence",
                               seeds=[seed], no_automem=no_automem, shared_ledger=shared_ledger,
                               episode_semaphore=semaphore) for seed in seeds]
        parts = [future.result()[0] for future in futures]
    result = _merge_results(parts)
    result["per_seed_status"] = {str(seed): part.get("status") for seed, part in zip(seeds, parts)}
    result["which"] = args.which
    result["bundle_hash"] = tree_hash(bundle)
    atomic_json(out / "scores.json", result)
    return result, 0 if result["status"] == "completed" else 1


def cmd_meta_preflight(args) -> tuple[dict, int]:
    config = _frozen(args.run_dir)
    if config.get("meta", {}).get("provider") == "codex":
        from .codex_runner import run_preflight
    else:
        from .claude_runner import run_preflight
    result = run_preflight(config, args.run_dir)
    return result, 0 if result.get("passed") else 1


def cmd_meta_run(args) -> tuple[dict, int]:
    config = _frozen(args.run_dir)
    if config.get("meta", {}).get("provider") == "codex":
        from .codex_runner import run_meta_slot
    else:
        from .claude_runner import run_meta_slot
    slot_dir = _slot(args.run_dir, args.line, args.generation, args.slot)
    workspace = Path(_read(slot_dir / "workspace.json")["workspace"])
    result = run_meta_slot(config, args.run_dir, workspace, slot_dir / "claude",
                           slot_id=f"{args.line}_g{args.generation}_{args.slot}")
    return result, 0 if result.get("status") == "completed" else 1


COMMANDS = {"prepare": cmd_prepare, "parent-eval": cmd_parent_eval, "ledger": cmd_ledger,
            "meta-workspace": cmd_meta_workspace, "collect": cmd_collect, "candidate-eval": cmd_candidate_eval,
            "gate": cmd_gate, "holdout-eval": cmd_holdout_eval, "meta-preflight": cmd_meta_preflight,
            "meta-run": cmd_meta_run}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="EvoViM evolution controller")
    subs = parser.add_subparsers(dest="command", required=True)
    prepare = subs.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--run-dir", type=Path, required=True)
    for name in COMMANDS:
        if name == "prepare":
            continue
        sub = subs.add_parser(name)
        sub.add_argument("--run-dir", type=Path, required=True)
        if name in {"parent-eval", "meta-workspace", "collect", "candidate-eval", "gate", "holdout-eval", "meta-run"}:
            sub.add_argument("--line", default="M", choices=["M"])
        if name in {"meta-workspace", "collect", "candidate-eval", "gate", "meta-run"}:
            sub.add_argument("--generation", type=int, required=True)
            sub.add_argument("--slot", required=True, choices=["fresh", "revision"])
        if name == "meta-workspace":
            sub.add_argument("--workspace-root", type=Path, required=True)
        if name == "collect":
            sub.add_argument("--workspace", type=Path, required=True)
        if name == "holdout-eval":
            sub.add_argument("--which", choices=["parent", "baseline", "n0"], default="parent")
            sub.add_argument("--unseal-holdout", action="store_true")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    result, code = COMMANDS[args.command](args)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
