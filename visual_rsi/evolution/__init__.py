"""Evolution campaign commands."""

from argparse import SUPPRESS
from pathlib import Path


def add_arguments(subs):
    for name, help_text in (
        ("run", "Run or resume an evolution campaign"),
        ("status", "Show campaign progress"),
        ("verify", "Check campaign results"),
        ("report", "Write per-seed results and a Markdown report"),
    ):
        sub = subs.add_parser(name, help=help_text)
        sub.add_argument("--run-root", required=True, type=Path)
        if name == "run":
            sub.add_argument("--manifest", required=True, type=Path)
            # Historical flags. Campaigns always resume from the frozen run state.
            sub.add_argument("--resume", action="store_true", help=SUPPRESS)
            sub.add_argument("--unattended", action="store_true", help=SUPPRESS)
