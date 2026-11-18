"""Record the process-start host/environment source version for result reuse."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def runtime_manifest(root: Path = ROOT) -> dict:
    paths = set()
    for name in ("host", "adapters", "evolution", "meta", "inner"):
        paths.update((root / "visual_rsi" / name).rglob("*.py"))
    paths.update((root / "visual_rsi" / "meta").glob("*.md"))
    paths.update((root / "visual_rsi" / "docs").glob("*.md"))
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(paths) if p.is_file()}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "sha256": digest, "files": files}


_IMPORTED_RUNTIME = runtime_manifest()


def startup_runtime() -> dict:
    """Version recorded at import; it does not promise live source immutability."""
    return _IMPORTED_RUNTIME


def bind_runtime(run_dir: Path) -> dict:
    """Record startup version and reject reuse of evidence from another version."""
    current = startup_runtime()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".runtime.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = run_dir / "runtime_manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != current:
                raise RuntimeError("run belongs to a different host/environment runtime; use a new run directory")
        else:
            prior_results = (path for name in ("summary.json", "scores.json")
                             for path in run_dir.rglob(name)
                             if path.relative_to(run_dir).parts[0] not in {"preflight", "preflight_staging"})
            if any(prior_results):
                raise RuntimeError("existing results have no runtime manifest; keep them as historical evidence and use a new run directory")
            path.write_text(json.dumps(current, sort_keys=True, indent=2) + "\n")
    return current
