"""Render the registered Meta prompt template without changing its semantics."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Mapping


def _block(text: str, name: str, keep: bool) -> str:
    pattern = re.compile(r"<!-- " + re.escape(name) + r"_BEGIN -->.*?<!-- " + re.escape(name) + r"_END -->\n?", re.S)
    if not pattern.search(text):
        raise ValueError("missing prompt marker block: " + name)
    return text if keep else pattern.sub("", text)


def render(template: str | Path, values: Mapping[str, object], *, line: str, spatial: bool,
           self_improvement: str, revision: bool = False, feedback: str = "basic",
           visual: bool = False) -> tuple[str, dict]:
    text = Path(template).read_text(encoding="utf-8") if isinstance(template, Path) else template
    hard_start, hard_end = "<!-- HARD_CONSTRAINTS_BEGIN -->", "<!-- HARD_CONSTRAINTS_END -->"
    constraints = text.split(hard_start, 1)[1].split(hard_end, 1)[0]
    hard_hash = hashlib.sha256(constraints.encode()).hexdigest()
    for modality in ("T", "V", "M"):
        text = _block(text, "MODALITY_" + modality, modality == line)
    text = _block(text, "SPATIAL_ANALYSIS", spatial)
    text = _block(text, "VISUAL_ANALYSIS", visual)
    keep_meta = self_improvement != "none"
    text = _block(text, "META_MEMORY", keep_meta)
    keep_playbook = self_improvement == "notebook+toolkit+playbook"
    text = _block(text, "PLAYBOOK", keep_playbook)
    text = _block(text, "PLAYBOOK_EDIT", keep_playbook)
    text = _block(text, "REVISION_SLOT", revision)
    if feedback not in ("basic", "enhanced"):
        raise ValueError("feedback must be basic or enhanced")
    text = _block(text, "FEEDBACK_ENHANCED", feedback == "enhanced")
    replacements = dict(values)
    replacements.setdefault("LINE", line)
    replacements.setdefault("ALLOWED_LIBRARIES", "numpy, scipy, Pillow")
    replacements.setdefault("PLAYBOOK_CONTENT", "(empty)")
    for key, value in replacements.items():
        text = text.replace("{{" + key + "}}", str(value))
    # Only registered, named placeholders are syntax.  Benchmark instructions
    # are opaque content and may legitimately contain ordinary JSON braces.
    unresolved = sorted(set(re.findall(r"\{\{([A-Z][A-Z0-9_]*)\}\}", text)))
    if unresolved:
        raise ValueError("unresolved prompt placeholder: " + ", ".join(unresolved))
    # Marker comments are implementation directives, not Meta content.
    text = re.sub(r"<!-- [A-Z_]+_(?:BEGIN|END) -->\n?", "", text)
    return text, {"hard_constraints_sha256": hard_hash, "line": line, "self_improvement": self_improvement,
                  "feedback": feedback, "visual_analysis": bool(visual)}


def sanitize_playbook(text: str, *, max_words: int | None = None) -> tuple[str, list[str]]:
    # Operational notes about budget or the selection rule are useful failure
    # context.  Only attempts to gain authority or point outside the workspace
    # are filtered; host verdicts remain host-owned regardless of notebook text.
    forbidden = ("change the host", "modify the gate", "change the selection rule", "ignore hard constraint", "../", "/.automem/")
    kept, removed = [], []
    for line in text.splitlines():
        if any(term in line.casefold() for term in forbidden):
            removed.append(line)
        else:
            kept.append(line)
    output = []
    used = 0
    for line in kept:
        words = line.split()
        if max_words is not None and used + len(words) > max_words:
            remaining = max_words - used
            if remaining > 0:
                output.append(" ".join(words[:remaining]))
            removed.append("[truncated playbook beyond word limit]")
            break
        output.append(line)
        used += len(words)
    return "\n".join(output) or "(empty)", removed
