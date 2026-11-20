"""Task-specific fields available in evaluation reports and revision feedback."""
from __future__ import annotations

from typing import Any


REPORT_FIELDS = {
    "dmvhard2": ("probes_correct", "probes_total", "accuracy_by_reach", "accuracy_by_kind", "invalid_actions", "steps"),
    "mcgallery": ("questions_correct", "questions_total", "accuracy_seen", "accuracy_lure", "reached_end",
                  "gates_passed", "walk_steps", "paintings_well_seen", "invalid_actions", "steps"),
    "postermaze": ("questions_correct", "questions_total", "accuracy_seen", "accuracy_lure", "reached_goal",
                   "optimal_steps", "nav_steps", "posters_well_seen", "invalid_actions", "steps"),
    "vsi_bench": ("question_type", "parse_status", "score", "metric"),
    "rngbench": ("pairs_matched", "total_pairs", "invalid_flips", "rounds", "steps"),
    "gameworld": ("progress", "success", "fail_resets", "task_status", "steps"),
}
NUMERIC_FIELDS = {
    "nethack": ("depth", "experience_level", "score"),
    "textworld": ("subgoals_completed", "score", "max_score", "won"),
}


def map_metrics(adapter: str, metrics: dict[str, Any], private_record: Any) -> dict[str, Any]:
    source = private_record if isinstance(private_record, dict) else {}
    if adapter in {"crafter", "crafter_remote"}:
        achievements = source.get("achievements") or metrics.get("achievements") or {}
        if isinstance(achievements, dict):
            unlocked = sorted(k for k, v in achievements.items() if isinstance(v, (int, float)) and v > 0)
            return {"achievements_unlocked": len(unlocked), "achievement_names": unlocked}
    if adapter in NUMERIC_FIELDS:
        return {key: source[key] for key in NUMERIC_FIELDS[adapter] if isinstance(source.get(key), (int, float))}
    return {key: source[key] for key in REPORT_FIELDS.get(adapter, ()) if key in source}
