"""Meta workspaces: read-only evidence, self-contained replay, and the rendered registered prompt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import uuid

from PIL import Image, ImageDraw, ImageFont

from .render import render, sanitize_playbook

PACKAGE = Path(__file__).resolve().parents[1]
TEMPLATE = PACKAGE / "meta" / "prompt_template.md"
REGISTERED_PRIORS = PACKAGE / "docs" / "REGISTERED_PRIORS.md"
REPLAY_MAIN = PACKAGE / "meta" / "replay" / "replay_main.py"
REPLAY_HOST_FILES = ("bundle.py", "dto.py", "model.py", "protocol_text.py", "visual_channel.py",
                     "worker_process.py", "child/worker_main.py", "child/registry.py")
def evaluation_python() -> Path:
    """Use the same interpreter for host episodes and Meta's offline replay."""
    return Path(sys.executable).absolute()


EVIDENCE_FILES = ("trajectory.jsonl", "calls.jsonl", "model_requests.jsonl", "episode.json", "summary.json")
EVIDENCE_DIRS = ("frames", "final_memory", "memory_by_step", "model_images", "memory_blobs", "memory_manifests")
# Agent-written notes (RNG 2026-10-10): only the secret-value scan applies; their key names are the
# agent's own choice (e.g. "coordinates", public in Matching Pairs) and may be free text.
AGENT_MEMORY_DIRS = {"final_memory", "memory_by_step", "memory_blobs"}


class EvidencePrivacyError(RuntimeError):
    pass


# Exact key names of private adapter state (Crafter info, gridmemory private record) and credentials.
# Exact matching avoids rejecting candidate-chosen names such as a memory file called "seed_map.txt".
PRIVATE_KEYS = {"semantic", "player_pos", "player_facing", "view", "target", "private_record", "private",
                "secret", "secret_truth", "seed", "coordinates", "password", "api_key", "access_token"}
PRIVATE_VALUE = re.compile(r"(sk-[A-Za-z0-9]{20,}|minimax[_-]?api[_-]?key|anthropic[_-]?api[_-]?key)", re.I)


def _walk_keys(value, path: Path) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in PRIVATE_KEYS:
                raise EvidencePrivacyError(f"private evidence key {key!r} in {path.name}")
            _walk_keys(item, path)
    elif isinstance(value, list):
        for item in value:
            _walk_keys(item, path)


def scan_evidence(root: Path) -> None:
    """Reject secret-like values anywhere and private key names in JSON and JSONL."""
    for path in Path(root).rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        text = path.read_bytes().decode("utf-8", errors="replace")
        if PRIVATE_VALUE.search(text):
            raise EvidencePrivacyError(f"secret-like value in {path}")
        if AGENT_MEMORY_DIRS & set(path.relative_to(root).parts):
            continue
        if path.suffix == ".json":
            try:
                # Tool contracts are declarative schemas.  Their conventional
                # argument name \"target\" is not a leaked environment target.
                # Keep secret-value scanning above, but do not apply game-state
                # key-name checks to this registered documentation artifact.
                if path.name != "tool_contracts.json":
                    _walk_keys(json.loads(text), path)
            except json.JSONDecodeError:
                pass
        elif path.suffix == ".jsonl":
            for number, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    try:
                        _walk_keys(json.loads(line), path)
                    except json.JSONDecodeError as exc:
                        raise EvidencePrivacyError(f"invalid JSONL at {path.name}:{number}") from exc


def make_contact_sheet(frames: Path, destination: Path, *, first: int = 0, max_frames: int = 24, columns: int = 4) -> Path:
    """Tile up to `max_frames` frames with large step labels (frame f00012 -> 'step 12')."""
    paths = sorted(Path(frames).glob("f*.png"))[first:first + max_frames]
    thumb_w, thumb_h, label_h = 192, 192, 28
    rows = max(1, (len(paths) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=20)
    except TypeError:
        font = ImageFont.load_default()
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w - 4, thumb_h - 4), Image.Resampling.NEAREST)
        x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + label_h)
        sheet.paste(image, (x + 2, y + 2))
        draw.text((x + 4, y + thumb_h + 2), f"step {int(path.stem[1:])}", fill="black", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")
    return destination


def copy_evidence(source: Path, destination: Path) -> None:
    """Copy only whitelisted public episode artefacts and add step-labelled contact sheets."""
    destination.mkdir(parents=True, exist_ok=True)
    for episode in sorted(p.parent for p in Path(source).rglob("trajectory.jsonl")):
        target = destination / episode.relative_to(source)
        target.mkdir(parents=True, exist_ok=True)
        for name in EVIDENCE_FILES:
            if (episode / name).exists():
                if name == "trajectory.jsonl":
                    rows = []
                    for line in (episode / name).read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            row = json.loads(line)
                            for private in ("reward", "native_success", "action_mapping"):
                                row.pop(private, None)
                            rows.append(json.dumps(row, ensure_ascii=False))
                    (target / name).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                elif name == "summary.json":
                    summary = json.loads((episode / name).read_text(encoding="utf-8"))
                    for private in ("reward_sum", "native_success"):
                        summary.pop(private, None)
                    (target / name).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
                else:
                    shutil.copyfile(episode / name, target / name)
        for name in EVIDENCE_DIRS:
            if (episode / name).is_dir():
                shutil.copytree(episode / name, target / name)
        frames = sorted((episode / "frames").glob("f*.png")) if (episode / "frames").is_dir() else []
        for part, start in enumerate(range(0, len(frames), 24)):
            make_contact_sheet(episode / "frames", target / "contact_sheets" / f"sheet_{part:02d}.png", first=start)


def install_replay(destination: Path) -> None:
    """A self-contained replay: the runner plus a private copy of the host runtime package."""
    host = PACKAGE / "host"
    package = destination / "v8host"
    (package / "child").mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "child" / "__init__.py").write_text("")
    for name in REPLAY_HOST_FILES:
        shutil.copyfile(host / name, package / name)
    shutil.copyfile(REPLAY_MAIN, destination / "replay_main.py")


def readonly(root: Path) -> None:
    for path in sorted(Path(root).rglob("*"), reverse=True):
        if not path.is_symlink():
            path.chmod(0o444 if path.is_file() else 0o555)
    Path(root).chmod(0o555)


def writable(root: Path) -> None:
    root = Path(root)
    if root.exists():
        root.chmod(0o755)
        for path in root.rglob("*"):
            if not path.is_symlink():
                path.chmod(0o755 if path.is_dir() else 0o644)


def template_sha256() -> str:
    return hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()


def check_template_registered() -> str:
    digest = template_sha256()
    if digest not in REGISTERED_PRIORS.read_text(encoding="utf-8"):
        raise ValueError("prompt template hash is not registered in REGISTERED_PRIORS.md")
    return digest


def build_workspace(root: Path, *, line: str, generation: int, slot: str, source: Path, evidence: Path,
                    lineage: Path | None, meta_state: Path | None, contracts: dict[str, str],
                    values: dict[str, object], spatial: bool, self_improvement: str,
                    feedback: str = "basic", prior_evidence: Path | None = None,
                    visual: bool = False) -> Path:
    root = Path(root).resolve()
    if not str(root).isascii() or not root.is_absolute():
        raise ValueError("workspace root must be an absolute ASCII path")
    home = Path.home().resolve()
    if root == home or home in root.parents:
        raise ValueError("workspace root must be outside the home directory")
    digest = check_template_registered()
    workspace = root / f"{line}_g{generation}_{slot}_{uuid.uuid4().hex[:8]}"
    inputs = workspace / "input"
    inputs.mkdir(parents=True)
    shutil.copytree(source, inputs / "source", ignore=shutil.ignore_patterns("__pycache__"))
    copy_evidence(evidence, inputs / "evidence")
    (inputs / "lineage").mkdir()
    if lineage and Path(lineage).exists():
        shutil.copytree(lineage, inputs / "lineage", dirs_exist_ok=True)
    if not (inputs / "lineage" / "lineage.md").exists():
        (inputs / "lineage" / "lineage.md").write_text("# Lineage\n\nNo earlier candidates.\n")
    if prior_evidence is not None and Path(prior_evidence).is_dir():
        # Revision slot: the fresh attempt's own development episodes, through
        # the same whitelist copy and privacy scan as the parent's evidence.
        copy_evidence(prior_evidence, inputs / "lineage" / f"g{generation}_fresh" / "evidence")
    (inputs / "meta_state").mkdir()
    playbook = "(empty)"
    if meta_state and Path(meta_state).exists():
        shutil.copytree(meta_state, inputs / "meta_state", dirs_exist_ok=True)
        if (inputs / "meta_state" / "playbook.md").exists():
            playbook, _ = sanitize_playbook((inputs / "meta_state" / "playbook.md").read_text(encoding="utf-8"))
    (inputs / "contracts").mkdir()
    for name, text in contracts.items():
        (inputs / "contracts" / name).write_text(text, encoding="utf-8")
    (inputs / "contracts" / "meta_environment.md").write_text(
        "# Meta evaluation environment\n\n"
        f"Use `{evaluation_python()}` for analysis scripts and offline replay.\n"
        "This interpreter already has numpy, scipy, Pillow, and omegaconf installed.\n"
        "Do not place substitutes for third-party or Python standard-library modules "
        "inside `output/revised_code/`.\n\n"
        "# Sandboxed validation commands\n\n"
        "For a long analysis or validation script, first create a stub with separate "
        "single-line Bash calls: `mkdir -p output/validation` and "
        "`printf 'pass\\n' > output/validation/check.py`. Then use Edit to write a `.py` file "
        "under `output/validation/` or `output/meta_state/toolkit/`, then run it "
        f"with a single-line Bash command such as `{evaluation_python()} output/validation/check.py`.\n"
        "Use paths relative to this workspace for script arguments and output files. "
        "Use Read for input evidence. Avoid shell heredocs, pipelines that write files, "
        "multiline `python -c`, and writes outside `output/`: these often enter "
        "Claude Code's dontAsk permission path instead of its sandbox auto-allow path.\n",
        encoding="utf-8")
    install_replay(inputs / "replay")
    (workspace / "output").mkdir()
    command = (f"{evaluation_python()} input/replay/replay_main.py --candidate output/revised_code --parent input/source "
               f"--evidence input/evidence --line {line} --output output/replay_report.json")
    filled = dict(values)
    filled.update(LINE=line, GENERATION=generation, SLOT_KIND=slot, PROMPT_ID=digest[:12], WORKSPACE=workspace,
                  CODEBASE_DIR=inputs / "source", EVIDENCE_DIR=inputs / "evidence", LINEAGE_DIR=inputs / "lineage",
                  CONTRACT_DIR=inputs / "contracts", META_STATE_DIR=inputs / "meta_state",
                  OUTPUT_DIR=workspace / "output", REPLAY_COMMAND=command, PLAYBOOK_CONTENT=playbook,
                  PRIOR_ATTEMPT_DIR=inputs / "lineage" / f"g{generation}_fresh")
    prompt, record = render(TEMPLATE, filled, line=line, spatial=spatial, self_improvement=self_improvement,
                            revision=slot == "revision", feedback=feedback, visual=visual)
    (workspace / "prompt.md").write_text(prompt, encoding="utf-8")
    (workspace / "replay_command.txt").write_text(command + "\n", encoding="utf-8")
    (workspace / "render_record.json").write_text(json.dumps({**record, "template_sha256": digest}, indent=2) + "\n")
    scan_evidence(inputs)
    readonly(inputs)
    return workspace
