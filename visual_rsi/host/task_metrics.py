"""Small allowlisted task metrics for reporting and Meta feedback."""
from __future__ import annotations

from typing import Any


def map_metrics(adapter: str, metrics: dict[str, Any], private_record: Any) -> dict[str, Any]:
    source = private_record if isinstance(private_record, dict) else {}
    if adapter in {"crafter", "crafter_remote"}:
        achievements = source.get("achievements") or metrics.get("achievements") or {}
        if isinstance(achievements, dict):
            return {"achievements_unlocked": sum(isinstance(v, (int, float)) and v > 0
                                                 for v in achievements.values()),
                    "achievement_names": sorted(k for k, v in achievements.items()
                                                if isinstance(v, (int, float)) and v > 0)}
    if adapter == "dmvhard2":
        return {key: source[key] for key in ("probes_correct", "probes_total", "accuracy_by_reach", "accuracy_by_kind", "invalid_actions", "steps")
                if key in source}
    if adapter == "mcgallery":
        return {key: source[key] for key in ("questions_correct", "questions_total", "accuracy_seen", "accuracy_lure",
                                             "reached_end", "gates_passed", "walk_steps", "paintings_well_seen",
                                             "invalid_actions", "steps")
                if key in source}
    if adapter == "postermaze":
        return {key: source[key] for key in ("questions_correct", "questions_total", "accuracy_seen", "accuracy_lure",
                                             "reached_goal", "optimal_steps", "nav_steps", "posters_well_seen",
                                             "invalid_actions", "steps")
                if key in source}
    if adapter == "nethack":
        return {key: source[key] for key in ("depth", "experience_level", "score")
                if isinstance(source.get(key), (int, float))}
    if adapter == "textworld":
        return {key: source[key] for key in ("subgoals_completed", "score", "max_score", "won")
                if isinstance(source.get(key), (int, float, bool))}
    if adapter == "vsi_bench":
        return {key: source[key] for key in ("question_type", "parse_status", "score", "metric")
                if key in source}
    if adapter == "rngbench":
        return {key: source[key] for key in ("pairs_matched", "total_pairs", "invalid_flips", "rounds", "steps")
                if key in source}
    if adapter == "gameworld":
        return {key: source[key] for key in ("progress", "success", "fail_resets", "task_status", "steps")
                if key in source}
    return {}
