"""Complete, hashed candidate bundles and the Seatbelt profile for the candidate worker."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from .dto import ContainmentError


def install_bundle(destination: Path, parent: Path, files: dict[str, str] | None = None) -> Path:
    """Copy the parent bundle, overlay complete candidate files, then write a hash manifest."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    parent = Path(parent).resolve()
    if destination.is_relative_to(parent):
        raise ValueError("install the bundle outside its parent source directory")
    verify_bundle(parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / "bundle"
        shutil.copytree(parent, staged, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store",
                                                                      "source_manifest.json"))
        # The source may be a read-only revision workspace.
        staged.chmod(0o755)
        for path in staged.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        for name, content in (files or {}).items():
            if not isinstance(name, str) or Path(name).is_absolute() or not isinstance(content, str):
                raise ValueError("invalid bundle override")
            target = (staged / name).resolve()
            if not target.is_relative_to(staged) or target.suffix != ".py":
                raise ValueError("candidate file outside source bundle")
            compile(content, name, "exec")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        manifest = verify_bundle(staged)
        (staged / "source_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                                                      encoding="utf-8")
        staged.rename(destination)
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
    if not (bundle / "candidate.py").is_file():
        raise ContainmentError("bundle is missing candidate.py")
    if any(path.is_symlink() for path in bundle.rglob("*")):
        raise ContainmentError("bundle contains a symlink")
    actual = {str(path.relative_to(bundle)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(bundle.rglob("*.py")) if "__pycache__" not in path.parts}
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
    """Allow the worker to read its code and libraries and write inside scratch."""
    paths = python_paths()
    quote = lambda value: json.dumps(str(value), ensure_ascii=False)
    read_roots = [Path(bundle).resolve(), Path(runtime).resolve(), paths["prefix"], paths["base_prefix"],
                  paths["resolved"].parent.parent, Path("/System/Library"), Path("/usr/lib"),
                  Path("/Library/Apple/System/Library")]
    grants = "\n".join(f"(allow file-read* file-map-executable (subpath {quote(root)}))"
                       for root in dict.fromkeys(read_roots))
    executables = [paths["resolved"], paths["executable"]]
    # A framework build starts through a launcher that execs Python.app.
    framework_python = paths["base_prefix"] / "Resources/Python.app/Contents/MacOS/Python"
    if framework_python.is_file():
        executables.append(framework_python.resolve())
    execs = " ".join(f"(literal {quote(p)})" for p in dict.fromkeys(executables))
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
