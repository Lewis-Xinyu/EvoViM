"""Static review of a merged candidate bundle.

The parent bundle is trusted evidence.  Findings are emitted only when a
modified/new candidate line introduces them, so inherited environment guidance
does not make every real candidate fail review.
"""
from __future__ import annotations

import ast
import difflib
import importlib.metadata
from functools import lru_cache
from pathlib import Path
import re
import shutil
import sys
import tempfile


_stdlib = set(getattr(sys, "stdlib_module_names", ()))
ALLOWED_IMPORT_ROOTS = _stdlib | {
    "numpy", "scipy", "PIL", "omegaconf", "agents", "prompt_builder", "registry",
    "client", "visual_adapter", "experiments",
}
FORBIDDEN_IMPORTS = {
    "subprocess", "socket", "ctypes", "multiprocessing", "urllib", "http", "ftplib",
    "smtplib", "ssl", "asyncio", "webbrowser", "requests", "httpx",
}


@lru_cache(maxsize=1)
def _host_module_roots() -> frozenset[str]:
    """Top-level module names supplied by Python or installed distributions."""
    return frozenset(_stdlib | set(importlib.metadata.packages_distributions()) |
                     {"PIL", "omegaconf", "numpy", "scipy"})


def shadowed_modules(candidate: Path) -> list[str]:
    """Annotate top-level stand-ins; selection does not depend on this list."""
    roots = set()
    for path in Path(candidate).iterdir():
        if path.is_file() and path.suffix == ".py":
            roots.add(path.stem)
        elif path.is_dir() and any(child.suffix == ".py" for child in path.rglob("*.py")):
            roots.add(path.name)
    return sorted(roots & _host_module_roots())


def _definitions(root: Path) -> dict[str, str]:
    output = {}
    for path in Path(root).rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                output[f"{path.relative_to(root)}:{node.name}"] = ast.dump(node, include_attributes=False)
    return output


def structural_changes(parent: Path, candidate: Path) -> list[str]:
    old, new = _definitions(parent), _definitions(candidate)
    return sorted(key for key, value in new.items() if old.get(key) != value)


def _added_lines(parent: Path, candidate: Path) -> set[int]:
    """Return line numbers in candidate that are additions or modifications."""
    if not parent.exists():
        return set(range(1, len(candidate.read_text(encoding="utf-8").splitlines()) + 1))
    old = parent.read_text(encoding="utf-8").splitlines()
    new = candidate.read_text(encoding="utf-8").splitlines()
    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    changed: set[int] = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            changed.update(range(j1 + 1, j2 + 1))
    return changed


def _node_lines(node: ast.AST) -> set[int]:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", start)
    return set(range(start, (end or start) + 1)) if start else set()


def _local_modules(root: Path) -> set[str]:
    modules: set[str] = set()
    for path in Path(root).rglob("*.py"):
        relative = path.relative_to(root).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        if parts:
            modules.add(".".join(parts))
            modules.add(parts[0])
    return modules


def _relative_module(path: Path, node: ast.ImportFrom, root: Path) -> str | None:
    package = list(path.relative_to(root).parent.parts)
    # level=1 is current package; higher levels cannot escape the bundle.
    if node.level > len(package) + 1:
        return None
    base = package[:len(package) - max(0, node.level - 1)]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _static_findings(root: Path, *, parent: Path, line: str,
                     forbidden_literals: tuple[str, ...]) -> list[str]:
    findings: list[str] = []
    local_modules = _local_modules(root)
    for path in sorted(Path(root).rglob("*.py")):
        if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
            findings.append(f"symlink escape: {path.relative_to(root)}")
            continue
        relative = path.relative_to(root)
        parent_path = Path(parent) / relative
        added = _added_lines(parent_path, path)
        if not added:
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            findings.append(f"syntax error in {path.name}: {exc}")
            continue
        for node in ast.walk(tree):
            node_lines = _node_lines(node)
            if isinstance(node, ast.Import):
                names = [(alias.name.split(".")[0], alias.lineno, alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                resolved = _relative_module(path, node, Path(root)) if node.level else node.module
                if node.level and resolved is None:
                    findings.append(f"relative import escapes bundle: {relative}:{node.lineno}")
                    continue
                names = [((resolved or "").split(".")[0], node.lineno, resolved or "")]
            else:
                names = []
            for name, lineno, full_name in names:
                if lineno not in added:
                    continue
                if name in FORBIDDEN_IMPORTS:
                    findings.append(f"forbidden import: {name}")
                elif name and name not in ALLOWED_IMPORT_ROOTS and name not in local_modules:
                    findings.append(f"import outside allow-list: {full_name or name}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"system", "popen"} and node_lines & added:
                    findings.append(f"forbidden process call: {node.func.attr}")
        changed_text = "\n".join(source.splitlines()[line_no - 1] for line_no in sorted(added))
        # Environment names and visual terminology are not authority.  The
        # host's DTO and visual-channel boundaries enforce actual modality.
        if re.search(r"Path\(\s*['\"]/(?!/)", changed_text):
            findings.append(f"absolute filesystem path in {path.name}")
    return sorted(set(findings))


def _merge(parent: Path, candidate: Path, destination: Path) -> None:
    shutil.copytree(parent, destination)
    for path in destination.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    for path in Path(candidate).rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(candidate)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def review_candidate(parent: Path, candidate: Path, *, line: str,
                     forbidden_literals: tuple[str, ...] = ()) -> dict:
    # ``forbidden_literals`` is accepted and ignored.  A1 removed the purely
    # lexical environment-name rejection: upstream AutoMem evolves per-game
    # scaffolds, so a game name in source or comments is not authority.  The
    # parameter survives only so the regression tests can keep asserting that
    # passing literals never rejects a candidate; real privilege, answer and
    # modality violations stay with the host isolation boundaries.
    parent, candidate = Path(parent), Path(candidate)
    with tempfile.TemporaryDirectory(prefix="meta-v8-review-") as directory:
        merged = Path(directory) / "merged"
        _merge(parent, candidate, merged)
        findings = _static_findings(merged, parent=parent, line=line,
                                    forbidden_literals=forbidden_literals)
        warnings = []
        for path in Path(candidate).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if re.search(r"return\s+['\"](?:Move|Task Done|Noop)\b", source):
                warnings.append(f"handler may return an action literal: {path.name}")
        changes = structural_changes(parent, merged)
    return {"passed": not findings, "findings": findings, "warnings": warnings,
            "shadowed_modules": shadowed_modules(candidate),
            "structural_changes": changes, "candidate": str(candidate), "line": line}
