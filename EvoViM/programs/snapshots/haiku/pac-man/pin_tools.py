"""Pinned working note for line M (g3): one LLM-written note, shown as an index.

The LLM writes, replaces and clears its note with the PIN command (LOG phase only,
after the outcome of the previous step is known).  The note is stored as an ordinary
memory file, pinned_note.txt, so READ pinned_note.txt returns its text.

Every observation shows only an index line: whether a note exists, the step at which
it was written, its age, and whether the lives count has changed since then.  The
text is never inserted into the prompt; the LLM decides whether to READ it.  The note
is model-written data, not an instruction to the host.  Nothing here writes a note on
its own, and nothing deletes one except a PIN with empty text chosen by the model.
"""
from __future__ import annotations

from registry import Tool

NOTE_FILE = "pinned_note.txt"
WRITE = "memory_write"
MAX_CHARS = 300

PIN_DESC = ("keep ONE short working note for the next steps (your plan, a danger, "
            "what you are heading for). Saving replaces the previous note; empty text "
            "clears it. The note is not shown in full: the observation shows only "
            "its age, and READ pinned_note.txt returns its text.")


def _lives(agent):
    scene = getattr(agent, "scene", None)
    return getattr(scene, "lives", None)


def _step(agent):
    return int(getattr(agent, "step_count", 0) or 0)


def pin_index(agent):
    """One index line for the observation.  It shows no note text."""
    meta = getattr(agent, "_pin_meta", None)
    if not meta:
        return ("PINNED NOTE: none. PIN saves one short working note; READ "
                "pinned_note.txt shows it.")
    age = _step(agent) - int(meta.get("step", _step(agent)))
    then, now = meta.get("lives"), _lives(agent)
    if then is None or now is None:
        lives = "lives not measured"
    elif then == now:
        lives = "lives unchanged since (%s)" % now
    else:
        lives = "lives changed since the note: %s -> %s" % (then, now)
    return ("PINNED NOTE: saved at step %d, age %d step%s; %s. READ pinned_note.txt "
            "shows its text." % (int(meta.get("step", 0)), age,
                                 "" if age == 1 else "s", lives))


def install_pin(agent, registry):
    """Register the PIN command on the host registry (LOG phase only)."""

    def pin(text=""):
        text = str(text).strip()
        memory = getattr(agent, "memory", None)
        if memory is None:
            return "[no memory store attached]"
        if not text:
            memory.write(NOTE_FILE, "")
            agent._pin_meta = None
            return "[pinned note cleared]"
        if len(text) > MAX_CHARS:
            return "[note too long: %d characters, the limit is %d]" % (len(text), MAX_CHARS)
        memory.write(NOTE_FILE, "step %d | lives %s\n%s" % (
            _step(agent), _lives(agent), text))
        agent._pin_meta = {"step": _step(agent), "lives": _lives(agent)}
        return "[pinned note saved at step %d]" % _step(agent)

    registry.register(Tool(
        "PIN", PIN_DESC,
        {"type": "object", "properties": {
            "text": {"type": "string", "maxLength": MAX_CHARS}},
         "required": ["text"], "additionalProperties": False},
        pin, ("LOG",), WRITE, "text"))
