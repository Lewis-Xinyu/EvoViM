"""One host episode: reset -> observe -> candidate decision in a separate process -> step."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import traceback
from typing import Any

from .dto import CandidateHung, validate_public_obs
from .model import BudgetExhausted
from .protocol_text import memory_filenames, system_prefix
from .task_metrics import map_metrics
from .visual_channel import FrameStore
from .worker_process import BACKSTOP_SECONDS, MODES, CandidateStartFailure, Worker


def _archive_observation(observation, archive: Path, name: str, *, save_image: bool = False) -> None:
    record = {"text": dict(observation.text), "frame_id": observation.frame_id, "image_sha256": None}
    if observation.image is not None:
        buffer = io.BytesIO()
        observation.image.convert("RGB").save(buffer, format="PNG")
        data = buffer.getvalue()
        record["image_sha256"] = hashlib.sha256(data).hexdigest()
        if save_image:
            (archive / f"{name}.png").write_bytes(data)
    (archive / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_memory(snapshot: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in snapshot.items():
        path = (destination / name).resolve()
        if not path.is_relative_to(destination.resolve()):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")


def _archive_memory_files(source: Path, archive: Path, destination: Path) -> None:
    """Version all memory files, including visual indexes and deleted/revised pixels.

    Content-addressed blobs avoid copying every saved PNG at every step. Paths
    are host-relative references; candidate symlinks are never followed.
    """
    blobs = archive / "memory_blobs"
    blobs.mkdir(exist_ok=True)
    files = []
    for path in ([] if source.is_symlink() else sorted(source.rglob("*"))):
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        blob = blobs / digest
        if not blob.exists():
            blob.write_bytes(data)
        files.append({"path": str(path.relative_to(source)), "blob": str(blob.relative_to(archive)),
                      "sha256": digest, "bytes": len(data)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"schema_version": 1, "files": files}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_episode(adapter, bundle, *, line: str, seed: int, model, archive, max_steps: int | None = None,
                mode: str = "process", compute_timeout: float | None = BACKSTOP_SECONDS, profile_calls: bool = False,
                no_automem: bool = False, phase_caps: dict[str, int] | None = None) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("runtime mode must be one of " + ", ".join(MODES))
    archive = Path(archive)
    archive.mkdir(parents=True, exist_ok=True)
    frames, memory_by_step, final_memory = archive / "frames", archive / "memory_by_step", archive / "final_memory"
    for directory in (frames, memory_by_step, final_memory):
        directory.mkdir(exist_ok=True)
    trajectory = archive / "trajectory.jsonl"
    calls_path = archive / "calls.jsonl"
    if trajectory.exists() or calls_path.exists():
        raise FileExistsError("attempt archive already contains trajectory or calls evidence")
    limit = int(max_steps if max_steps is not None else getattr(adapter, "max_steps", 1))
    frame_store = FrameStore()
    summary: dict[str, Any] = {"status": "completed", "line": line, "steps": 0,
                               "score": None if getattr(adapter, "name", None) == "open_eqa_em" else 0.0,
                               "actions": [],
                               "protocol_errors": 0, "fallbacks": 0, "action_overrides": 0,
                               "requests": 0, "runtime_mode": mode,
                               "reward_sum": 0.0, "terminated": False, "truncated": False,
                               "stop_reason": None, "native_success": None,
                               "end_type": None, "final_message": None, "aux_metrics": {},
                               "visual_requests": {"valid": 0, "invalid": 0},
                               "no_automem": bool(no_automem)}
    worker = None
    observation = None
    last_memory: dict = {}
    try:
        adapter.reset(seed)
        action_spec_fn = getattr(adapter, "action_spec", None)
        action_spec = dict(action_spec_fn()) if callable(action_spec_fn) else {
            "kind": "finite", "actions": list(adapter.legal_actions()), "dynamic": False}
        actions = [str(item) for item in action_spec.get("actions", [])]
        fallback = str(getattr(adapter, "fallback_action", "") or (actions[0] if actions else "look"))
        instruction = str(adapter.task_instruction())
        metric = adapter.metric_description()
        observation = adapter.observe()
        _archive_observation(observation, archive, "initial_observation")
        (archive / "episode.json").write_text(json.dumps({
            "protocol_version": 2, "line": line, "adapter_name": adapter.name,
            "task_id": getattr(adapter, "task_id", None), "observation_profile": getattr(adapter, "observation_profile", None),
            "instruction": instruction, "legal_actions": actions, "action_spec": action_spec,
            "fallback_action": fallback, "metric_definition": metric,
            "phase_caps": dict(phase_caps or {"LOG": 10, "PLAN": 10}),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        prefix = lambda phase, tools, files: system_prefix(phase, tools, action_spec, memory_filenames(files))
        try:
            worker = Worker(bundle, archive / "worker", line=line, instruction=instruction,
                            actions=actions, action_spec=action_spec, adapter_name=adapter.name,
                            fallback_action=fallback, mode=mode, compute_timeout=compute_timeout,
                            profile_calls=profile_calls, no_automem=no_automem, phase_caps=phase_caps)
        except CandidateHung as exc:
            summary.update(status="candidate_hung", stop_reason="candidate_start_hung", error=str(exc))
        except CandidateStartFailure as exc:
            summary.update(status="candidate_crashed", stop_reason="candidate_start_failed",
                           error=str(exc), failure_origin="candidate")
        except Exception as exc:
            summary.update(status="candidate_crashed", stop_reason="candidate_start_failed", error=str(exc))
        previous = previous_executed = None
        for step in range(limit if worker is not None else 0):
            frame_id = observation.frame_id or f"f{step:05d}"
            image = getattr(observation, "image", None)
            if image is not None:
                frame_store.add(frame_id, step, image)
                image.convert("RGB").save(frames / f"{frame_id}.png", format="PNG")
            public = validate_public_obs({"text": dict(observation.text), **({"frame_id": frame_id} if line != "T" else {})},
                                         adapter.public_text_fields, line=line)
            compute_before = worker.compute_seconds
            try:
                result = worker.act(step=step, observation={"text": public["text"]}, previous_action=previous,
                                    previous_validated_action=previous_executed, frame_id=frame_id, image=image,
                                    current_image=image if line in {"V", "M"} else None, model=model,
                                    system_prefix=prefix, frame_store=frame_store)
            except BudgetExhausted as exc:
                summary.update(status="budget_exhausted", stop_reason="budget_exhausted", error=str(exc))
                break
            except Exception as exc:
                summary.update(status="model_error", stop_reason="model_error", error=str(exc))
                (archive / "model_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                break
            summary["requests"] += len(result["calls"])
            compute = round(worker.compute_seconds - compute_before, 3)  # candidate time, model waits excluded
            summary["candidate_compute_seconds"] = round(summary.get("candidate_compute_seconds", 0.0) + compute, 3)
            summary["max_step_compute_seconds"] = max(summary.get("max_step_compute_seconds", 0.0), compute)
            summary["protocol_errors"] += int(result.get("protocol_errors", 0))
            with calls_path.open("a", encoding="utf-8") as stream:
                for exchange in result.get("exchanges", []):
                    stream.write(json.dumps({"step": step, **exchange}, ensure_ascii=False, allow_nan=False) + "\n")
            _archive_memory_files(worker.scratch / "memory", archive,
                                  archive / "memory_manifests" / f"step_{step:05d}.json")
            if result.get("status") != "ok":
                summary.update(status=result["status"], stop_reason=result["status"], error=result.get("error"))
                if result.get("failure_origin"):
                    summary["failure_origin"] = result["failure_origin"]
                if result.get("traceback"):
                    (archive / "candidate_traceback.txt").write_text(str(result["traceback"]), encoding="utf-8")
                break
            action = result["action"]
            summary["action_overrides"] += int(bool(result.get("action_override")))
            last_memory = result.get("memory", {})
            _write_memory(last_memory, memory_by_step / f"step_{step:05d}")
            outcome = adapter.step(action)
            mapping = outcome.action_mapping or {}
            environment_fallback = bool(mapping.get("fallback", False))
            used_fallback = bool(result["fallback"] or environment_fallback)
            summary["fallbacks"] += int(used_fallback)
            # Native numeric action IDs are ordinary encodings, not substituted
            # commands. Only a reported fallback changes the next LOG feedback.
            executed_action = str(mapping["native"]) if environment_fallback and mapping.get("native") is not None else action
            raw_score = adapter.score(outcome.private_record)
            # Offline semantic judges can be pending. Preserve that state as
            # null rather than supplying a misleading zero to downstream work.
            summary["score"] = None if raw_score is None else float(raw_score)
            if raw_score is None:
                summary["score_status"] = "pending_offline_judge"
            summary["reward_sum"] += float(outcome.reward)
            native_success = None
            if isinstance(outcome.metrics, dict):
                if outcome.metrics.get("success") is not None:
                    native_success = bool(outcome.metrics["success"])
                    summary["native_success"] = native_success
                summary["aux_metrics"] = map_metrics(adapter.name, outcome.metrics, outcome.private_record)
            summary["actions"].append(action)
            summary["steps"] += 1
            record = {"step": step, "observation": public,
                      "action": action, "executed_action": executed_action, "fallback": used_fallback,
                      "host_fallback": bool(result["fallback"]), "environment_fallback": environment_fallback,
                      "action_override": bool(result.get("action_override")),
                      "action_mapping": outcome.action_mapping, "reward": float(outcome.reward),
                      "score": summary["score"], "terminated": bool(outcome.terminated),
                      "truncated": bool(outcome.truncated),
                      "native_success": native_success,
                      "requests": result.get("counts", {}),
                      "candidate_compute_seconds": compute}
            with trajectory.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if outcome.done:
                summary["terminated"] = bool(outcome.terminated)
                summary["truncated"] = bool(outcome.truncated)
                summary["status"] = "terminated" if outcome.terminated else "truncated"
                summary["stop_reason"] = "native_terminated" if outcome.terminated else "native_truncated"
                observation = outcome.observation
                break
            previous = action
            previous_executed = executed_action
            observation = outcome.observation
        if worker is not None and summary["stop_reason"] is None:
            summary["stop_reason"] = "host_step_limit"
            summary["truncated"] = True
        if (adapter.name == "minigrid_memory" and summary["status"] in {"completed", "truncated"}
                and summary["stop_reason"] in {"host_step_limit", "native_truncated"}
                and summary["native_success"] is None):
            summary["native_success"] = False
    except Exception as exc:
        summary.update(status="environment_error", stop_reason="environment_error", error=str(exc))
        (archive / "environment_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if worker is not None:
            summary["visual_requests"] = dict(worker.visual_counts)
            worker.close()
            _archive_memory_files(worker.scratch / "memory", archive, archive / "memory_manifests" / "final.json")
        try:
            if observation is not None:
                _archive_observation(observation, archive, "final_observation", save_image=True)
        finally:
            close = getattr(adapter, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    summary.setdefault("cleanup_error", str(exc))
    if observation is not None:
        summary["final_message"] = str(observation.text.get("short_term_context", ""))
    technical = {"model_error", "environment_error", "candidate_crashed", "candidate_hung",
                 "worker_crashed", "worker_hung", "budget_exhausted"}
    if summary["status"] in technical or summary.get("failure_origin") == "infrastructure":
        summary["end_type"] = "error"
        summary["score_status"] = "pending_technical"
        summary["score"] = None
    elif summary.get("native_success") is True:
        summary["end_type"] = "success"
    elif summary.get("truncated") or summary.get("stop_reason") == "host_step_limit":
        summary["end_type"] = "timeout"
    elif summary["actions"] and summary["actions"][-1].lower() in {"quit", "#quit"}:
        summary["end_type"] = "quit"
    else:
        summary["end_type"] = "env_fail"
    _write_memory(last_memory, final_memory)
    (archive / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
