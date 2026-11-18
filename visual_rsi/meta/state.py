"""Cross-generation Meta state: carried forward whole, with playbook authority filtering."""
from __future__ import annotations

from pathlib import Path
import shutil

from .render import sanitize_playbook


def transfer_state(source: Path, destination: Path) -> dict:
    """Carry notebook, toolkit and playbook to the next generation.

    Nothing is dropped for size or file type: the v8 caps (20 KB notebook,
    200 KB toolkit, .py/.md-only toolkit, 600-word playbook) rejected or cut the
    whole Meta memory once exceeded.  Sizes are recorded instead.  The playbook
    still loses lines that try to change host rules (sanitize_playbook).
    """
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    notebook = source / "notebook.md"
    if notebook.exists():
        shutil.copyfile(notebook, destination / "notebook.md")
    toolkit = source / "toolkit"
    if toolkit.exists():
        shutil.copytree(toolkit, destination / "toolkit", dirs_exist_ok=True)
    removed = []
    playbook = source / "playbook.md"
    if playbook.exists():
        clean, removed = sanitize_playbook(playbook.read_text())
        (destination / "playbook.md").write_text(clean + "\n")
    toolkit_bytes = sum(p.stat().st_size for p in (destination / "toolkit").rglob("*") if p.is_file()) \
        if (destination / "toolkit").exists() else 0
    return {"removed_playbook_lines": removed,
            "notebook_bytes": (destination / "notebook.md").stat().st_size if (destination / "notebook.md").exists() else 0,
            "toolkit_bytes": toolkit_bytes,
            "playbook_words": len((destination / "playbook.md").read_text().split())
            if (destination / "playbook.md").exists() else 0}
