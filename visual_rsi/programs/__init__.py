"""Load complete memory programs retained in the paper's held-out evaluations."""

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from visual_rsi.host.repair import tree_hash


_ROOT = Path(__file__).resolve().parent
_STARTER = _ROOT.parent / "inner" / "visual_memory"
_CATALOG = json.loads((_ROOT / "catalog.json").read_text(encoding="utf-8"))


def program_names() -> tuple[str, ...]:
    """List the released model/task identifiers."""
    return tuple(sorted(_CATALOG))


@contextmanager
def open_program(name: str) -> Iterator[Path]:
    """Yield a temporary standalone bundle after checking its original version hash."""
    if name not in _CATALOG:
        raise ValueError(f"Unknown program {name!r}. Choose from {', '.join(program_names())}")
    snapshot = _ROOT / "snapshots" / name
    manifest = snapshot / "source_manifest.json"
    with TemporaryDirectory(prefix="evovim-program-") as temporary:
        bundle = Path(temporary)
        for filename in json.loads(manifest.read_text(encoding="utf-8")):
            source = snapshot / filename
            if not source.is_file():
                source = _STARTER / filename
            target = bundle / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(manifest, bundle / manifest.name)
        if tree_hash(bundle) != _CATALOG[name]["version_sha256"]:
            raise ValueError(f"The installed sources for {name!r} differ from the archived program")
        yield bundle
