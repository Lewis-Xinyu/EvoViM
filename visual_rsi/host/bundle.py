"""Complete, hashed candidate bundles and the Seatbelt profile for the candidate worker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

from .dto import ContainmentError


def install_bundle(destination: Path, parent: Path, files: dict[str, str] | None = None) -> Path:
    """Copy the parent bundle, overlay complete candidate files, then write a hash manifest."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    parent = Path(parent).resolve()
    shutil.copytree(parent, destination, ignore=shutil.ignore_patterns("__pycache__", "source_manifest.json"))
    destination.chmod(0o755)  # the parent may be a read-only workspace input
    for path in destination.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)
    for name, content in (files or {}).items():
        if not isinstance(name, str) or name.startswith("/") or not isinstance(content, str):
            raise ValueError("invalid bundle override")
        target = (destination / name).resolve()
        if not target.is_relative_to(destination) or not name.endswith(".py"):
            raise ValueError("candidate file outside source bundle")
        compile(content, name, "exec")  # an installation check, not a security boundary
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    manifest = {str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(destination.rglob("*.py"))}
    (destination / "source_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return destination


def overlay_files(directory: Path) -> dict[str, str]:
    """Read a revised_code directory as {relative .py path: source}."""
    directory = Path(directory)
    if not directory.exists():
        return {}
    files = {}
    for path in sorted(directory.rglob("*.py")):
        if path.is_symlink() or "__pycache__" in path.parts:
            continue
        files[str(path.relative_to(directory))] = path.read_text(encoding="utf-8")
    return files


def verify_bundle(bundle: Path) -> dict[str, str]:
    bundle = Path(bundle).resolve()
    actual = {str(path.relative_to(bundle)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(bundle.rglob("*.py")) if "__pycache__" not in path.parts}
    if any(path.is_symlink() for path in bundle.rglob("*")):
        raise ContainmentError("bundle contains a symlink")
    manifest_path = bundle / "source_manifest.json"
    if not manifest_path.exists():
        return actual
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != actual:
        raise ContainmentError("bundle integrity failed")
    return manifest


def python_paths() -> dict[str, Path]:
    executable = Path(sys.executable)
    return {"executable": executable, "resolved": executable.resolve(),
            "prefix": Path(sys.prefix).resolve(), "base_prefix": Path(sys.base_prefix).resolve()}


def profile(bundle: Path, scratch: Path, runtime: Path) -> str:
    """Seatbelt profile ported from r2: deny by default, read only code and libraries, write only scratch."""
    paths = python_paths()
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    read_roots = [Path(bundle).resolve(), Path(runtime).resolve(), paths["prefix"], paths["base_prefix"],
                  paths["resolved"].parent.parent, Path("/System/Library"), Path("/usr/lib"),
                  Path("/Library/Apple/System/Library")]
    grants = "\n".join(f"(allow file-read* file-map-executable (subpath {quote(root)}))"
                       for root in dict.fromkeys(read_roots))
    execs = " ".join(f"(literal {quote(p)})" for p in dict.fromkeys([paths["resolved"], paths["executable"]]))
    return "\n".join([
        "(version 1)",
        "(deny default)",
        '(import "dyld-support.sb")',
        f"(allow process-exec {execs})",
        "(allow sysctl-read)",
        '(allow mach-lookup (global-name "com.apple.system.logger"))',
        "(allow file-read-metadata)",
        '(allow file-read* (literal "/dev/urandom") (literal "/dev/random") (literal "/dev/null") (literal "/dev/dtracehelper"))',
        '(allow file-write* (literal "/dev/null"))',
        grants,
        f"(allow file-read* file-write* (subpath {quote(Path(scratch).resolve())}))",
        "",
    ])
