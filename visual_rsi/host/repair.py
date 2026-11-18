"""Small durable primitives shared by the v3 repair paths.

They deliberately operate on caller-provided roots, which keeps the historical
run read-only and makes fault-injection tests use disposable fixtures.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


def _json(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and not path.is_symlink():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


class AttemptStore:
    """Allocate immutable physical attempts for one logical evaluation unit."""
    def __init__(self, root: Path, logical_unit_id: str):
        self.root = Path(root) / "units" / logical_unit_id.replace(":", "_")
        self.root.mkdir(parents=True, exist_ok=True)

    def completed(self, candidate_hash: str, eval_fingerprint: str) -> dict | None:
        unit = _json(self.root / "unit.json", {})
        chosen = unit.get("selected_attempt")
        if not chosen or unit.get("candidate_hash") != candidate_hash or unit.get("eval_fingerprint") != eval_fingerprint:
            return None
        summary = _json(self.root / "attempts" / chosen / "summary.json")
        # A selected attempt is reusable only when its immutable summary says
        # it is a valid score.  A previous method/provider failure must remain
        # forensic evidence, not a cached evaluation result.
        if not isinstance(summary, dict):
            return None
        outcome = summary.get("outcome")
        if isinstance(outcome, dict):
            if not outcome.get("score_valid", False):
                return None
        elif summary.get("status") not in {"completed", "terminated", "truncated"} or type(summary.get("score")) not in (int, float):
            # Older immutable summaries lack the nested v3 contract. Their
            # only compatible reuse path is an unambiguous finite valid score;
            # anything else stays non-reusable and visible to recovery.
            return None
        return summary

    def begin(self, candidate_hash: str, eval_fingerprint: str) -> tuple[str, Path]:
        attempts = self.root / "attempts"
        attempts.mkdir(exist_ok=True)
        numbers = [int(path.name[1:]) for path in attempts.glob("a[0-9][0-9][0-9]") if path.is_dir()]
        attempt_id = f"a{max(numbers, default=0) + 1:03d}"
        destination = attempts / attempt_id
        destination.mkdir()
        current = _json(self.root / "unit.json", {})
        if current and (current.get("candidate_hash") != candidate_hash or
                        current.get("eval_fingerprint") != eval_fingerprint):
            raise RuntimeError("logical unit identity mismatch")
        current.update({"schema_version": 1, "logical_unit_id": self.root.name,
                        "candidate_hash": candidate_hash, "eval_fingerprint": eval_fingerprint,
                        "selected_attempt": current.get("selected_attempt")})
        atomic_json(self.root / "unit.json", current)
        atomic_json(destination / "attempt.json", {"schema_version": 1, "attempt_id": attempt_id,
                                                    "logical_unit_id": self.root.name,
                                                    "candidate_hash": candidate_hash,
                                                    "eval_fingerprint": eval_fingerprint,
                                                    "state": "started"})
        return attempt_id, destination

    def finalize(self, attempt_id: str, summary: dict, *, select: bool) -> None:
        atomic_json(self.root / "attempts" / attempt_id / "summary.json", summary)
        attempt = _json(self.root / "attempts" / attempt_id / "attempt.json", {})
        attempt.update({"state": "finalized", "selected": bool(select)})
        atomic_json(self.root / "attempts" / attempt_id / "attempt.json", attempt)
        unit = _json(self.root / "unit.json", {})
        if select:
            unit["selected_attempt"] = attempt_id
        atomic_json(self.root / "unit.json", unit)


class Promotion:
    """A single-lineage journal with immutable comparison decisions."""
    def __init__(self, line_root: Path):
        self.root = Path(line_root)
        self.versions = self.root / "versions"
        self.journal = self.root / "promotion_journal"
        self.versions.mkdir(parents=True, exist_ok=True)
        self.journal.mkdir(parents=True, exist_ok=True)

    def current(self) -> dict | None:
        return _json(self.root / "parent.current.json")

    def resolve_parent(self) -> Path | None:
        """Return the one authoritative parent bundle for every consumer."""
        pointer = self.current()
        if not pointer:
            return None
        bundle = self.root / str(pointer.get("bundle", ""))
        if not bundle.is_dir() or tree_hash(bundle) != pointer.get("version_hash"):
            raise RuntimeError("invalid_parent_pointer")
        return bundle

    def stage(self, comparison_id: str, parent_before: dict, candidate: Path, evidence: Path, scores: dict,
              verdict: dict) -> dict:
        journal = self.journal / f"{comparison_id}.json"
        old = _json(journal)
        if old:
            return old
        staging = self.versions / (comparison_id + ".staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        shutil.copytree(candidate, staging / "bundle")
        if evidence.exists():
            shutil.copytree(evidence, staging / "evidence")
        atomic_json(staging / "scores.json", scores)
        candidate_hash = tree_hash(staging / "bundle")
        record = {"schema_version": 1, "comparison_id": comparison_id, "parent_before": parent_before,
                  "candidate_hash": candidate_hash, "verdict": verdict, "state": "journaled",
                  "staging": str(staging)}
        atomic_json(journal, record)
        return record

    def commit(self, comparison_id: str) -> dict:
        journal = self.journal / f"{comparison_id}.json"
        record = _json(journal)
        if not record:
            raise FileNotFoundError(journal)
        if record["state"] == "committed":
            return record
        current = self.current()
        if current is not None and current.get("version_hash") != record["parent_before"].get("version_hash"):
            raise RuntimeError("stale_parent_cas")
        staging = Path(record["staging"])
        if not staging.exists() or tree_hash(staging / "bundle") != record["candidate_hash"]:
            raise RuntimeError("candidate_hash_mismatch")
        final = self.versions / record["candidate_hash"]
        if not final.exists():
            os.replace(staging, final)
        pointer = {"version_hash": record["candidate_hash"], "comparison_id": comparison_id,
                   "bundle": str((final / "bundle").relative_to(self.root)),
                   "scores": str((final / "scores.json").relative_to(self.root)),
                   "evidence": str((final / "evidence").relative_to(self.root)) if (final / "evidence").is_dir() else None}
        atomic_json(self.root / "parent.current.json", pointer)
        record["state"] = "committed"
        record["version"] = pointer
        atomic_json(journal, record)
        return record


def finalize_feedback(root: Path, generation: str, stage: str, payload: dict[str, Any]) -> dict:
    """Idempotently project host-owned facts without allowing notebook overwrite."""
    target = Path(root) / "host_feedback" / f"{generation}.json"
    existing = _json(target, {"schema_version": 1, "generation": generation, "stages": {}})
    host_payload = {key: value for key, value in payload.items() if key not in {"verdict", "accepted", "score_valid"}}
    existing["stages"][stage] = host_payload
    # These values intentionally come from the caller's host result only.
    for key in ("verdict", "accepted", "score_valid", "evidence_refs", "missing_evidence", "shadowed_modules"):
        if key in payload:
            existing[key] = payload[key]
    atomic_json(target, existing)
    return existing


def efficiency(parent: dict, candidate: dict, expected_units: set[str], threshold: float = .2) -> dict:
    rows, reductions, regressions, missing = {}, [], [], []
    for unit in sorted(expected_units):
        left, right = parent.get(unit), candidate.get(unit)
        if not isinstance(left, dict) or not isinstance(right, dict):
            missing.append(unit); continue
        row = {}
        for field in ("steps", "requests", "input_tokens", "output_tokens"):
            a, b = left.get(field), right.get(field)
            if type(a) not in (int, float) or type(b) not in (int, float):
                row[field] = {"parent": None, "candidate": None, "absolute_delta": None, "relative_reduction": None}
                continue
            reduction = None if a == 0 else (a - b) / a
            row[field] = {"parent": a, "candidate": b, "absolute_delta": b - a, "relative_reduction": reduction}
            if reduction is not None:
                if reduction >= threshold:
                    reductions.append({"unit": unit, "field": field, "reduction": reduction})
                elif reduction < 0:
                    regressions.append({"unit": unit, "field": field, "reduction": reduction})
        rows[unit] = row
    return {"schema_version": 1, "comparison_kind": "parent_before", "by_unit": rows,
            "coverage": len(rows) == len(expected_units), "missing_evidence": missing,
            "complete": not missing, "reduction_at_least_threshold": bool(reductions),
            "regressions": regressions, "search_cost_scope": "separate_from_measurement_efficiency"}
