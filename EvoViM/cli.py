"""Run, inspect and evolve visual memory programs."""

import argparse
import shutil
from pathlib import Path

from . import evolution
from .programs import open_program, program_names


def main(argv=None):
    parser = argparse.ArgumentParser(prog="evovim", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a small offline example")
    source = demo.add_mutually_exclusive_group()
    source.add_argument("--cards", action="store_true", help="Play four cards with the evolved GPT Card Matching program")
    source.add_argument("--bundle", type=Path, help="Run a trusted standalone candidate on the example grid")
    demo.add_argument("--output", type=Path, help="Save the trace to a new directory")
    commands.add_parser("programs", help="List the programs retained for held-out evaluation")
    export = commands.add_parser("export", help="Copy a complete, verified program to a new directory")
    export.add_argument("program", choices=program_names(), metavar="MODEL/TASK")
    export.add_argument("destination", type=Path)
    evaluate = commands.add_parser("evaluate", help="Run a released program on its original benchmark")
    evaluate.add_argument("program", choices=program_names(), metavar="MODEL/TASK")
    mode = evaluate.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run one step with a scripted reply and no API calls")
    mode.add_argument("--model", help="Vision model ID for the selected program’s provider")
    evaluate.add_argument("--seed", type=int, help="Episode seed (default: first archived held-out seed)")
    evaluate.add_argument("--output", type=Path, required=True, help="New directory for the episode trace")
    evaluate.add_argument("--benchmark-python", type=Path, help="Interpreter for upstream game dependencies")
    evolution.add_arguments(commands)
    args = parser.parse_args(argv)
    try:
        if args.command == "programs":
            print("\n".join(program_names()))
            return 0
        if args.command == "export":
            with open_program(args.program) as bundle:
                shutil.copytree(bundle, args.destination)
            print(args.destination.resolve())
            return 0
        if args.command == "demo":
            from .demo import run
            return run(args)
        if args.command == "evaluate":
            from .evaluate import run
            return run(args)
        from .evolution.quick_evo import run_command
        return run_command(args)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.exit(2, f"evovim: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
