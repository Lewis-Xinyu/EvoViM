"""Run and inspect visual memory evolution campaigns from a frozen manifest."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf

from . import controller
from .claude_runner import run_meta_slot, run_preflight, settings_template_hash, claude_version
from ..host.paid import Ledger
from ..meta.workspace import readonly


TERMINAL_META_FAILURES = {"invalid_output", "review_failed", "smoke_failed"}


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Run:
    def __init__(self, root: Path, manifest: Path | None = None):
        self.root = root.resolve()
        self.manifest_path = manifest.resolve() if manifest else None
        self.manifest = self._load_manifest() if manifest else _read(self.root / "manifest.frozen.json")
        if not isinstance(self.manifest, dict):
            raise ValueError("run does not have a frozen manifest")
        self.lock_file = self.root / ".supervisor.lock"
        self.validated_claude_version: str | None = None
        concurrency = self.manifest.get("concurrency", {})
        self.state_lock = threading.RLock()
        self.episode_slots = threading.BoundedSemaphore(int(concurrency.get("episode_workers", 1)))
        self.meta_slots = threading.BoundedSemaphore(int(concurrency.get("meta_workers", 1)))

    def _load_manifest(self) -> dict:
        data = OmegaConf.to_container(OmegaConf.load(str(self.manifest_path)), resolve=True)
        if data.get("experiment_mode") != "exploratory":
            raise ValueError("quick_evo only accepts experiment_mode: exploratory")
        families = [item.get("family") for item in data.get("environments", [])]
        if not families or not all(isinstance(f, str) and f for f in families) or len(set(families)) != len(families):
            raise ValueError("manifest environments need unique, non-empty family names")
        if data.get("lines", ["M"]) != ["M"]:
            raise ValueError("The visual memory method requires lines: [M]")
        for item in data["environments"]:
            line = item.setdefault("line", "M")
            controller._start_bundle(data, line)
            if "bundle" in item:
                controller._start_bundle(item, line)
        generations = (data.get("generations") or {}).get("max")
        if type(generations) is not int or generations < 1:
            raise ValueError("manifest generations.max must be a positive integer")
        seeds = (data.get("seeds") or {}).get("dev") or []
        if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("manifest seeds.dev must be a non-empty list of unique integers")
        for name, value in (data.get("concurrency") or {}).items():
            if name in {"episode_workers", "meta_workers", "lineage_workers"} and (type(value) is not int or value < 1):
                raise ValueError(f"concurrency.{name} must be a positive integer")
        retry_window = (data.get("recovery") or {}).get("max_consecutive_failures_before_cooldown", 3)
        if type(retry_window) is not int or retry_window < 1:
            raise ValueError("recovery.max_consecutive_failures_before_cooldown must be a positive integer")
        if data.get("holdout_after_lineage"):
            holdout = (data.get("seeds") or {}).get("holdout") or []
            if not holdout:
                raise ValueError("holdout_after_lineage requires seeds.holdout")
            versions = data.get("holdout_versions") or ["baseline", "parent"]
            if (not isinstance(versions, list) or any(not isinstance(v, str) for v in versions)
                    or len(set(versions)) != len(versions)
                    or set(versions) - {"baseline", "parent", "n0"}):
                raise ValueError("holdout_versions must contain unique baseline, parent or n0 entries")
        return data

    # The campaign shape comes from the manifest; v5 hard-coded 8 families x 5 generations x seeds 42/43.
    @property
    def families(self) -> tuple[str, ...]:
        return tuple(item["family"] for item in self.manifest["environments"])

    @property
    def generations(self) -> int:
        return int((self.manifest.get("generations") or {}).get("max", 5))

    @property
    def dev_seeds(self) -> list[int]:
        return [int(seed) for seed in self.manifest["seeds"]["dev"]]

    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        stream = self.lock_file.open("a+")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise RuntimeError("another quick_evo supervisor owns this run")
        return stream

    def event(self, kind: str, **data: Any) -> None:
        row = {"at": _now(), "kind": kind, **data}
        with self.state_lock:
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def initialise(self) -> None:
        frozen = self.root / "manifest.frozen.json"
        fingerprint = _hash(self.manifest)
        if frozen.exists():
            if _hash(_read(frozen)) != fingerprint:
                raise ValueError("run root already belongs to a different frozen manifest")
            return
        _write(frozen, self.manifest)
        (self.root / "authorization_basis.md").write_text(
            "# Authorization basis\n\nIssued by the scheduler for this run only. Based on the operator's "
            "invocation with the workload and limits frozen in manifest.frozen.json. "
            "It does not authorize provider changes or matrix expansion. The frozen enforcement mode determines "
            "whether paid estimates block or only report overage.\n", encoding="utf-8")
        limits = self.manifest["authorization"]
        _write(self.root / "delegated_authorization.json", {
            "issued_by": "scheduler", "basis": "explicit_user_instruction", "scope": "this_run_only",
            "authorization_mode": "delegated_workload", "manifest_sha256": fingerprint,
            "enforcement": limits.get("enforcement", "hard"),
            "limits": {"max_units": int(limits["max_units"]), "max_input_tokens": int(limits["max_input_tokens"]),
                       "max_output_tokens": int(limits["max_output_tokens"]), "monetary_cap": limits.get("monetary_cap")},
        })
        con = sqlite3.connect(self.root / "state.sqlite")
        con.execute("create table if not exists slots (family text, generation integer, state text, updated_at text, primary key(family,generation))")
        con.commit(); con.close()
        self.event("run_initialised", manifest_sha256=fingerprint)
        self.progress()

    def env_root(self, family: str) -> Path:
        return self.root / "environments" / family

    def item(self, family: str) -> dict:
        return next(item for item in self.manifest["environments"] if item["family"] == family)

    def config(self, family: str) -> dict:
        item = self.item(family)
        data = {key: json.loads(json.dumps(value)) for key, value in self.manifest.items() if key not in {"environments", "authorization", "concurrency", "selection", "recovery", "purpose", "unattended", "stop_only_after_verified_results", "no_automem_baseline",
                                                                                                 "holdout_after_lineage", "holdout_versions"}}
        # Per-environment calibration and caps are frozen in the derived config.
        for section in ("meta", "noise", "budget"):
            data[section] = {**data.get(section, {}), **json.loads(json.dumps(item.get(section) or {}))}
        data["benchmark"] = {"adapter": item["adapter"], "task_id": item["task_id"],
                             "max_steps": item["max_steps"], "family": family,
                             **json.loads(json.dumps(item.get("benchmark_options") or {}))}
        data["lines"] = [item["line"]]
        data["review"] = dict(self.manifest["review"])
        return data

    def prepare_env(self, family: str) -> tuple[Path, dict, str]:
        controller.bind_runtime(self.root)
        root, config, line = self.env_root(family), self.config(family), self.item(family)["line"]
        root.mkdir(parents=True, exist_ok=True)
        # Each family has its own immutable runtime manifest before a reused
        # G0 result or a gate binds to this environment root.
        controller.bind_runtime(root)
        path = root / "config.json"
        if path.exists() and _hash(_read(path)) != _hash(config):
            raise ValueError(f"{family}: derived configuration fingerprint changed")
        if not path.exists():
            _write(path, config)
        delegated = _read(self.root / "delegated_authorization.json")
        if item_auth := self.item(family).get("authorization"):
            limits = delegated["limits"]
            delegated = {**delegated, "limits": {**limits, **json.loads(json.dumps(item_auth))}}
        _write(root / "delegated_authorization.json", delegated)
        Ledger(root / "ledger.json", config["budget"].get("price_per_million", {})).add()
        return root, config, line

    def preflight(self) -> None:
        config = self.meta_config(self.config(self.families[0]))
        if config.get("meta", {}).get("provider") == "codex":
            from .codex_runner import run_preflight as codex_preflight
            marker = self.root / "codex_preflight_passed.json"
            if _read(marker, {}).get("passed"):
                return
            result = codex_preflight(config, self.root)
            if not result.get("passed"):
                raise RuntimeError("Codex Meta isolation preflight failed")
            return
        version, setting_hash = claude_version(config), settings_template_hash(config)
        self.validated_claude_version = version
        marker = self.root / "preflight" / f"reference-{setting_hash[:16]}.json"
        if marker.exists():
            record = _read(marker)
            if record.get("passed") and record.get("claude_version") == version and record.get("settings_template_sha256") == setting_hash:
                return
        # This is a new, explicitly user-directed launcher configuration
        # epoch. Earlier failures remain immutable evidence; one complete
        # eight-check run is allowed to validate the repaired route, but this
        # does not silently refresh any retry budget.
        self.event("claude_preflight_started", claude_version=version, settings_template_sha256=setting_hash)
        result = run_preflight(config, self.root)
        self.event("claude_preflight_finished", passed=bool(result.get("passed")), checks=result.get("checks"),
                   settings_template_sha256=setting_hash)
        _write(marker, result)
        _write(self.root / "preflight_reference.json", result)
        if not result.get("passed"):
            raise RuntimeError("Claude Meta preflight failed; no real matrix slot was consumed")

    def parent_eval(self, family: str) -> None:
        root, _, line = self.prepare_env(family)
        scores = root / "lines" / line / "parent_scores.json"
        if not (scores.exists() and _read(scores, {}).get("complete")):
            self.event("baseline_started", family=family)
            with self.episode_slots:
                controller.cmd_parent_eval(SimpleNamespace(run_dir=root, line=line))
            self.event("baseline_finished", family=family)
        baseline = root / "baseline"
        if not _read(scores, {}).get("complete"):
            # Never freeze a partial G0: a provider outage would otherwise make
            # an incomplete parent the permanent baseline.  run_family leaves
            # this family for a later resume, which re-enters only the missing units.
            self.event("baseline_incomplete", family=family, status=_read(scores, {}).get("status"))
            return
        if not (baseline / "scores.json").exists():
            source = root / "lines" / line
            baseline.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source / "parent", baseline / "bundle")
            shutil.copytree(source / "parent_evidence", baseline / "evidence")
            shutil.copyfile(source / "parent_scores.json", baseline / "scores.json")
            readonly(baseline)
            self.event("baseline_frozen", family=family)

    def _slot_state(self, root: Path, line: str, generation: int, slot: str = "fresh") -> tuple[Path, dict]:
        path = root / "lines" / line / "generations" / f"g{generation}_{slot}"
        return path, _read(path / "status.json", {})

    @staticmethod
    def _terminal(slot_dir: Path, status: dict) -> str | None:
        """accepted / rejected once a slot has a final method result, else None."""
        gate = _read(slot_dir / "gate.json", {})
        if gate:
            return "accepted" if gate.get("accepted") else "rejected"
        if status.get("status") in TERMINAL_META_FAILURES | {"method_failed"}:
            return "rejected"
        return None

    def _generation_done(self, root: Path, line: str, generation: int, revision: bool) -> bool:
        fresh = self._terminal(*self._slot_state(root, line, generation))
        if fresh is None:
            return False
        if fresh == "accepted" or not revision:
            return True
        return self._terminal(*self._slot_state(root, line, generation, "revision")) is not None

    def _generation_slot(self, root: Path, line: str, generation: int) -> Path:
        """The slot whose candidate stands for this generation in reports: the revision once it was measured."""
        revision, _ = self._slot_state(root, line, generation, "revision")
        return revision if (revision / "scores.json").exists() else self._slot_state(root, line, generation)[0]

    def _set_slot(self, family: str, generation: int, state: str) -> None:
        with self.state_lock:
            con = sqlite3.connect(self.root / "state.sqlite")
            con.execute("insert into slots values (?,?,?,?) on conflict(family,generation) do update set state=excluded.state,updated_at=excluded.updated_at",
                        (family, generation, state, _now()))
            con.commit(); con.close()

    def generation(self, family: str, generation: int) -> None:
        """Fresh slot; if it ends rejected and meta.revision_slot is on, one revision slot on the same parent."""
        fresh = self._run_slot(family, generation, "fresh")
        _, config, _ = self.prepare_env(family)
        if fresh == "rejected" and config.get("meta", {}).get("revision_slot", False):
            self._run_slot(family, generation, "revision")

    def _run_slot(self, family: str, generation: int, slot_kind: str) -> str | None:
        root, config, line = self.prepare_env(family)
        slot, status = self._slot_state(root, line, generation, slot_kind)
        terminal = self._terminal(slot, status)
        if terminal is not None:
            return terminal
        if status.get("status") == "technical_unresolved":
            # Infrastructure failures are retryable by definition; the resume
            # *is* the bounded recovery.  Treating the status as terminal made
            # every provider outage permanent: on 2026-09-22 a five-hour rate
            # limit cost eleven generations that no later resume would retry,
            # leaving seven of eight families returning instantly while only
            # one had work.  Clear the marker and re-enter the generation.
            self.event("technical_retry", family=family, generation=generation, slot=slot_kind,
                       previous=status.get("status"))
            (slot / "status.json").unlink(missing_ok=True)
        self._set_slot(family, generation, "RUNNING")
        workspace_json = slot / "workspace.json"
        if not workspace_json.exists():
            self.event("workspace_started", family=family, generation=generation, slot=slot_kind)
            controller.cmd_meta_workspace(SimpleNamespace(run_dir=root, line=line, generation=generation, slot=slot_kind,
                                                          workspace_root=Path((config["meta"].get("codex") or
                                                                               config["meta"].get("claude"))["workspace_root"])))
        workspace = Path(_read(workspace_json)["workspace"])
        attempts = slot / "claude" / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        results = sorted(attempts.glob("a*/result.json"))
        result = _read(results[-1], {}) if results else {}
        technical = {"timed_out", "generation_failed", "isolation_failed"}
        # A physical attempt remains immutable.  A run invocation stops after
        # its configured recovery window; an explicit later resume may try
        # again once the external cause has been addressed.
        retry_window = int((self.manifest.get("recovery") or {}).get("max_consecutive_failures_before_cooldown", 3))
        attempts_this_invocation = 0
        while not result or result.get("status") in technical:
            index = len(results) + 1
            if attempts_this_invocation >= retry_window:
                _write(slot / "status.json", {"status": "technical_unresolved", "last_meta": result})
                self._set_slot(family, generation, "RETRY_WAIT")
                self.event("technical_unresolved", family=family, generation=generation, slot=slot_kind, result=result)
                return None
            attempt = attempts / f"a{index:03d}"
            if result and result.get("status") in technical:
                # Preserve every byte written by the failed session, but give
                # the next physical attempt a genuinely empty output tree.
                stale = attempt / "stale_output"
                if stale.exists():
                    raise FileExistsError(f"retry evidence already exists: {stale}")
                attempt.mkdir(parents=True, exist_ok=True)
                output = workspace / "output"
                if output.is_dir():
                    shutil.move(str(output), str(stale))
                else:
                    stale.mkdir()
                output.mkdir()
                self.event("meta_stale_output_archived", family=family, generation=generation,
                           slot=slot_kind, attempt=index, stale_output=str(stale))
            if (attempt / "meta.launch.json").exists() and not (attempt / "result.json").exists():
                # A supervisor killed mid-session leaves a launched attempt with no
                # result; codex_runner refuses to relaunch it, which silently
                # stopped a lineage for a whole night on 2026-10-01.  Keep every
                # byte (attempt dir and the session's partial output) and start clean.
                killed = attempts / f"killed_{attempt.name}_{time.strftime('%Y%m%dT%H%M%S')}"
                shutil.move(str(attempt), str(killed))
                output = workspace / "output"
                if output.is_dir():
                    shutil.move(str(output), str(killed / "partial_output"))
                output.mkdir(exist_ok=True)
                self.event("meta_killed_attempt_archived", family=family, generation=generation,
                           slot=slot_kind, attempt=index, archived=str(killed))
            self.event("meta_started", family=family, generation=generation, slot=slot_kind, attempt=index)
            attempts_this_invocation += 1
            with self.meta_slots:
                codex = config.get("meta", {}).get("provider") == "codex"
                if codex:
                    from .codex_runner import run_meta_slot as codex_run_meta_slot
                else:
                    codex_run_meta_slot = run_meta_slot
                result = codex_run_meta_slot(self.meta_config(config), root, workspace, attempt,
                                             slot_id=f"{family}_g{generation}" + ("" if slot_kind == "fresh" else "_" + slot_kind),
                                             preflight_marker=self.root / ("codex_preflight_passed.json" if codex else "preflight_passed.json"))
            _write(attempt / "result.json", result)
            self.event("meta_finished", family=family, generation=generation, slot=slot_kind, attempt=index, status=result.get("status"))
            results.append(attempt / "result.json")
            if result.get("status") == "completed":
                break
            if "402 insufficient balance" in str(result.get("result", "")):
                # Payment balance cannot recover through retry backoff.  Keep
                # this logical Meta slot pending until the account changes.
                _write(slot / "status.json", {"status": "technical_unresolved", "last_meta": result,
                                              "blocker": "provider_insufficient_balance_402"})
                self._set_slot(family, generation, "RETRY_WAIT")
                self.event("technical_unresolved", family=family, generation=generation,
                           slot=slot_kind, blocker="provider_insufficient_balance_402")
                return None
            if result.get("status") not in technical:
                terminal = {"status": "method_failed", "meta": result}
                _write(slot / "status.json", terminal)
                controller.record_terminal(root / "lines" / line, generation, slot_kind, terminal,
                                           enhanced=config.get("meta", {}).get("feedback") == "enhanced")
                self._set_slot(family, generation, "METHOD_FAILED_REPORTED")
                return "rejected"
            # Formal run (2026-10-13): minutes-scale backoff (1, 2, 4, 8, 15 min); a Codex launch outage after
            # sleep/wake lasted ~40 min on 2026-10-10 and exhausted the old 3-30 s schedule.
            if (self.manifest.get("recovery") or {}).get("meta_backoff") == "minutes":
                time.sleep(min(900, 60 * 2 ** max(0, attempts_this_invocation - 1)))
            else:
                time.sleep(min(30, index * 3))
        status = _read(slot / "status.json", {})
        if status.get("status") != "installed":
            collected, _ = controller.cmd_collect(SimpleNamespace(run_dir=root, line=line, generation=generation,
                                                                   slot=slot_kind, workspace=workspace))
            if collected.get("status") != "installed":
                self._set_slot(family, generation, "METHOD_FAILED_REPORTED")
                return "rejected"
        scores = slot / "scores.json"
        if not (scores.exists() and _read(scores, {}).get("complete")):
            self.event("candidate_started", family=family, generation=generation, slot=slot_kind)
            outcome, _ = controller.cmd_candidate_eval(SimpleNamespace(
                run_dir=root, line=line, generation=generation, slot=slot_kind,
                episode_semaphore=self.episode_slots))
            self.event("candidate_finished", family=family, generation=generation, slot=slot_kind, status=outcome.get("status"))
            if not outcome.get("complete"):
                if outcome.get("status") == "infrastructure_failed":
                    _write(slot / "status.json", {"status": "technical_unresolved", "evaluation": outcome})
                    self._set_slot(family, generation, "RETRY_WAIT")
                    self.event("candidate_infrastructure_unresolved", family=family, generation=generation, slot=slot_kind, outcome=outcome)
                    return None
                terminal = {"status": "method_failed", "evaluation": outcome}
                _write(slot / "status.json", terminal)
                controller.record_terminal(root / "lines" / line, generation, slot_kind, terminal,
                                           enhanced=config.get("meta", {}).get("feedback") == "enhanced")
                self._set_slot(family, generation, "METHOD_FAILED_REPORTED")
                return "rejected"
        if not (slot / "gate.json").exists():
            gate, _ = controller.cmd_gate(SimpleNamespace(run_dir=root, line=line, generation=generation, slot=slot_kind))
            self.event("gate_finished", family=family, generation=generation, slot=slot_kind, accepted=gate.get("accepted"), difference=gate.get("difference"))
        self._set_slot(family, generation, "ROUND_REPORTED")
        return self._terminal(slot, _read(slot / "status.json", {}))

    def progress(self) -> dict:
        families = []
        baseline_done = rounds = measured = method_failed = unresolved = 0
        for item in self.manifest["environments"]:
            family, line = item["family"], item["line"]
            root = self.env_root(family)
            baseline = _read(root / "baseline" / "scores.json", {})
            baseline_done += int(bool(baseline.get("complete"))) * len(self.dev_seeds)
            row = {"family": family, "baseline": baseline.get("status", "PENDING"), "generations": {}}
            revision = bool(self.config(family).get("meta", {}).get("revision_slot", False))
            for generation in range(1, self.generations + 1):
                states = []
                for kind in ("fresh", "revision"):
                    slot, status = self._slot_state(root, line, generation, kind)
                    if kind == "revision" and not slot.exists():
                        continue
                    gate, scores = _read(slot / "gate.json", {}), _read(slot / "scores.json", {})
                    state = "GATED" if gate else status.get("status", "PENDING")
                    states.append(state if kind == "fresh" else "revision:" + state)
                    measured += int(bool(scores.get("complete"))) * len(self.dev_seeds)
                    method_failed += int(state in {"method_failed", "invalid_output", "review_failed", "smoke_failed"})
                    unresolved += int(state == "technical_unresolved")
                row["generations"][str(generation)] = " / ".join(states)
                rounds += int(self._generation_done(root, line, generation, revision))
            if self.manifest.get("no_automem_baseline"):
                row["no_automem"] = _read(root / "no_automem" / "scores.json", {}).get("status", "PENDING")
            families.append(row)
        active = 0
        database = self.root / "state.sqlite"
        if database.exists():
            con = sqlite3.connect(database)
            try:
                active = int(con.execute("select count(*) from slots where state='RUNNING'").fetchone()[0])
            finally:
                con.close()
        value = {"updated_at": _now(), "status": "RUNNING", "expected_environments": len(self.families),
                 "expected_logical_rounds": len(self.families) * self.generations,
                 "baseline_units_completed": baseline_done, "rounds_reported": rounds, "candidate_units_measured": measured,
                 "candidate_units_cached": 0, "candidate_units_method_failed": method_failed,
                 "candidate_units_not_applicable": method_failed * len(self.dev_seeds), "unresolved_infrastructure_units": unresolved,
                 "active_workers": active, "families": families}
        with self.state_lock:
            _write(self.root / "progress.json", value)
        return value

    def run(self) -> dict:
        # Validate every derived family before freezing a run or making the
        # shared paid Meta preflight.  prepare_env also serves offline manifest
        # inspection fixtures, which intentionally need not be runnable.
        for family in self.families:
            controller.load_config_value(self.config(family))
        self.initialise()
        evaluation_only = bool(self.manifest.get("evaluation_only"))
        try:
            if not evaluation_only:
                self.preflight()
        except Exception as exc:
            result = {**self.progress(), "status": "BLOCKED_EXTERNAL", "result_artifact_checks_passed": False,
                      "blocker": "shared Meta route/preflight could not complete within bounded recovery",
                      "error": str(exc)}
            _write(self.root / "completion.json", result)
            (self.root / "INCOMPLETE_REPORT.md").write_text(
                "# Incomplete V2 run\n\nThe shared Meta route did not pass the bounded, real preflight. No game "
                "baseline or candidate measurement was started, so there are no scientific results to report. Resume "
                "the same run root after the provider route is healthy; completed work will remain immutable.\n", encoding="utf-8")
            return result
        # Each lineage remains serial; independent environments overlap.
        # Semaphores in parent_eval/generation enforce the frozen 4-episode
        # and 2-Meta ceilings across these four workers.
        workers = int(self.manifest.get("concurrency", {}).get("lineage_workers", 4))
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(self.families))), thread_name_prefix="lineage") as pool:
            futures = [pool.submit(self.run_family, family) for family in self.families]
            for future in as_completed(futures):
                future.result()
        self.report()
        return self.verify(write_completion=True)

    def meta_config(self, config: dict) -> dict:
        value = json.loads(json.dumps(config))
        if value.get("meta", {}).get("provider") == "codex":
            return value
        if self.validated_claude_version:
            value["meta"]["claude"]["validated_version"] = self.validated_claude_version
        return value

    def run_n0(self, family: str) -> dict:
        """N0: the frozen G0 bundle with LLM-written memory ablated (host-defined, never a Meta input)."""
        root, config, line = self.prepare_env(family)
        path = root / "no_automem" / "scores.json"
        prior = _read(path, {})
        if prior.get("complete"):
            return prior
        self.event("n0_started", family=family)
        with self.episode_slots:
            # baseline/bundle is the frozen G0 copy; lines/<L>/parent moves once a candidate is accepted.
            result, _ = controller._evaluate(root, config, line, root / "baseline" / "bundle", "N0",
                                             root / "no_automem" / "evidence", no_automem=True)
        result.update(label="N0", definition="G0 bundle; LOG phase suppressed and memory writes refused; reads kept")
        _write(path, result)
        self.event("n0_finished", family=family, complete=bool(result.get("complete")))
        return result

    def pause_before_generation(self) -> int | None:
        path = self.root / "control" / "pause_before_generation"
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None

    def run_family(self, family: str) -> None:
        self.parent_eval(family)
        if not _read(self.env_root(family) / "baseline" / "scores.json", {}).get("complete"):
            self.progress()
            return
        if self.manifest.get("evaluation_only"):
            # Fixed-baseline run (ported 2026-10-13 from visual-rsi-baselines): measure the installed bundle
            # on the dev seeds and the holdout seeds only; no Meta, no generations.
            if self.manifest.get("holdout_after_lineage"):
                self.run_holdout(family)
            return
        if self.manifest.get("no_automem_baseline"):
            self.run_n0(family)
        self.progress()
        for generation in range(1, self.generations + 1):
            # Operator pause (gen2, 2026-10-08): a run-root file control/pause_before_generation
            # holding N stops each family cleanly before generation N.  It is not part of the
            # frozen manifest; deleting the file and resuming continues the same run.
            pause = self.pause_before_generation()
            if pause is not None and generation >= pause:
                self.event("paused_before_generation", family=family, generation=generation)
                return
            self.generation(family, generation)
            self.progress()
            root, line = self.env_root(family), self.item(family)["line"]
            revision = bool(self.config(family).get("meta", {}).get("revision_slot", False))
            if not self._generation_done(root, line, generation, revision):
                # Fresh and any triggered revision must both reach a terminal
                # verdict before the next generation can use their feedback.
                break
        root, _, line = self.prepare_env(family)
        if all(self._generation_done(root, line, g, bool(self.config(family).get("meta", {}).get("revision_slot")))
               for g in range(1, self.generations + 1)):
            final = controller.finalize_best_parent(root, line, episode_semaphore=self.episode_slots)
            self.event("final_parent_selected", family=family, **{k: v for k, v in final.items() if k != "final_remeasure"})
            if self.manifest.get("holdout_after_lineage"):
                self.run_holdout(family)

    def run_holdout(self, family: str) -> dict:
        """Evaluate the frozen G0, the final parent (and N0 if requested) on the holdout seeds.

        Starts as soon as this family's lineage is done, so it overlaps with the
        other families still evolving.  Completed results are reused.
        """
        root, _, line = self.prepare_env(family)
        versions = list(self.manifest.get("holdout_versions") or ["baseline", "parent"])
        self.event("holdout_started", family=family, versions=versions)
        with ThreadPoolExecutor(max_workers=len(versions), thread_name_prefix=f"holdout-{family}") as pool:
            futures = {which: pool.submit(controller.cmd_holdout_eval, SimpleNamespace(
                run_dir=root, line=line, which=which, unseal_holdout=True, episode_semaphore=self.episode_slots))
                       for which in versions}
            results = {which: future.result()[0] for which, future in futures.items()}
        self.event("holdout_finished", family=family,
                   status={which: result.get("status") for which, result in results.items()})
        return results

    def report(self) -> None:
        rows = []
        for item in self.manifest["environments"]:
            family, line = item["family"], item["line"]
            root = self.env_root(family); baseline = _read(root / "baseline" / "scores.json", {})
            parent = _read(root / "lines" / line / "parent_scores.json", {})
            n0 = _read(root / "no_automem" / "scores.json", {})
            for seed in self.dev_seeds:
                row = {"environment": family, "seed": seed, "N0": (n0.get("scores", {}).get(str(seed)) or [None])[0],
                       "G0": (baseline.get("scores", {}).get(str(seed)) or [None])[0]}
                for generation in range(1, self.generations + 1):
                    score = _read(self._generation_slot(root, line, generation) / "scores.json", {})
                    row[f"G{generation}"] = (score.get("scores", {}).get(str(seed)) or [None])[0]
                row["retained"] = (parent.get("scores", {}).get(str(seed)) or [None])[0]
                row["delta_from_G0"] = None if row["retained"] is None or row["G0"] is None else row["retained"] - row["G0"]
                rows.append(row)
        with (self.root / "results_by_seed.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = ["N0", "G0"] + [f"G{g}" for g in range(1, self.generations + 1)] + ["retained", "delta_from_G0"]
            writer = csv.DictWriter(stream, fieldnames=["environment", "seed"] + columns)
            writer.writeheader(); writer.writerows(rows)
        text = ["# Quick Meta Results", "", "Exploratory single-measurement results; no variance, significance, or cross-environment "
                "aggregate is estimated. A generation's column is its revision candidate when one was measured.", "",
                "| Environment | Seed | " + " | ".join(columns) + " |", "|---|---:|" + "---:|" * len(columns)]
        for r in rows:
            text.append("| " + " | ".join(str(r[key]) for key in ["environment", "seed"] + columns) + " |")
        holdout_rows = []
        for item in self.manifest["environments"]:
            line_dir = self.env_root(item["family"]) / "lines" / item["line"]
            cells = {}
            for which in ("baseline", "parent", "n0"):
                result = _read(line_dir / f"holdout_{which}" / "scores.json", {})
                values = [v for vs in (result.get("scores") or {}).values() for v in vs]
                cells[which] = (f"{sum(values) / len(values):.3f} (n={len(values)})" if values else "-")
            holdout_rows.append((item["family"], cells))
        if any(cell != "-" for _, cells in holdout_rows for cell in cells.values()):
            text += ["", "## Holdout seeds (mean over seeds)", "", "| Environment | G0 | final parent | N0 |",
                     "|---|---:|---:|---:|"]
            text += [f"| {family} | {c['baseline']} | {c['parent']} | {c['n0']} |" for family, c in holdout_rows]
        hung = [path for path in self.root.rglob("summary.json") if _read(path, {}).get("status") == "candidate_hung"]
        hung += [path for path in self.root.rglob("status.json")
                 if (_read(path, {}).get("replay") or {}).get("status") == "replay_hung"]
        if hung:
            text += ["", "## Hung candidates (30-minute backstop fired)", "",
                     "Each listed episode or replay was ended by the backstop; nothing else was stopped.", ""]
            text += [f"- `{path.parent.relative_to(self.root)}`" for path in sorted(hung)]
        (self.root / "FINAL_REPORT.md").write_text("\n".join(text) + "\n", encoding="utf-8")

    def verify(self, *, write_completion: bool = False) -> dict:
        existing = _read(self.root / "completion.json", {})
        preflight = _read(self.root / "preflight_reference.json", {})
        if existing.get("status") == "BLOCKED_EXTERNAL" and not preflight.get("passed"):
            return existing
        p = self.progress(); problems = []
        if p["baseline_units_completed"] != len(self.families) * len(self.dev_seeds): problems.append("missing baseline seed result")
        if not self.manifest.get("evaluation_only") and p["rounds_reported"] != len(self.families) * self.generations: problems.append("not all logical rounds have a terminal method result")
        if self.manifest.get("no_automem_baseline"):
            problems += [f"{f} N0 incomplete" for f in self.families
                         if not _read(self.env_root(f) / "no_automem" / "scores.json", {}).get("complete")]
        if p["unresolved_infrastructure_units"]: problems.append("unresolved infrastructure slot")
        if self.manifest.get("holdout_after_lineage"):
            for item in self.manifest["environments"]:
                for which in self.manifest.get("holdout_versions") or ["baseline", "parent"]:
                    path = self.env_root(item["family"]) / "lines" / item["line"] / f"holdout_{which}" / "scores.json"
                    if not _read(path, {}).get("complete"):
                        problems.append(f"{item['family']} holdout {which} incomplete")
        for item in self.manifest["environments"]:
            root, line = self.env_root(item["family"]), item["line"]
            for generation in range(1, self.generations + 1):
                for kind in ("fresh", "revision"):
                    slot, status = self._slot_state(root, line, generation, kind)
                    if (slot / "status.json").exists() and status.get("status") == "installed":
                        score = _read(slot / "scores.json", {})
                        if not score.get("complete") or set(score.get("scores", {})) != set(map(str, self.dev_seeds)):
                            problems.append(f"{item['family']} G{generation} {kind} lacks complete paired candidate scores")
        result = {**p, "result_artifact_checks_passed": not problems, "problems": problems,
                  "status": "COMPLETED" if not problems else "BLOCKED_EXTERNAL" if p["unresolved_infrastructure_units"] else "RUNNING"}
        if write_completion:
            _write(self.root / "completion.json", result)
        return result


def main(argv=None) -> int:
    from .. import demo

    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    demo.add_arguments(subs.add_parser("demo", help="Run a visual memory episode offline"))
    for command in ("run", "status", "verify", "report"):
        sub = subs.add_parser(command); sub.add_argument("--run-root", required=True, type=Path)
        if command == "run":
            sub.add_argument("--manifest", required=True, type=Path); sub.add_argument("--resume", action="store_true"); sub.add_argument("--unattended", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "demo":
        return demo.run(args)
    run = Run(args.run_root, getattr(args, "manifest", None))
    if args.command == "run":
        with run.lock(): result = run.run()
    elif args.command == "report":
        run.report(); result = run.verify()
    else:
        result = run.verify()
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
