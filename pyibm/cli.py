"""Command line: ``python -m pyibm run <case_dir> [--input NAME] [--steps N] [--quiet]
[--output-dir DIR]`` and ``python -m pyibm version`` (docs/user-guide.md, running a case).

Exit codes: 0 when the run finishes, 1 when it blows up (a non-finite velocity or position,
reported on stderr), 2 for a command-line error. Any other error in the case (a bad input
file, a hook that raises) ends with Python's own traceback and a non-zero code.
"""

from __future__ import annotations

import argparse
import sys

from pyibm import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pyibm",
        description="pyIBM: a 2D immersed boundary method (docs/user-guide.md).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a case directory (input2d, structure files, hooks.py)")
    p_run.add_argument("case_dir", help="directory holding the input file and structure files")
    p_run.add_argument("--input", default="input2d", metavar="NAME",
                       help="input file inside case_dir (default: input2d)")
    p_run.add_argument("--steps", type=int, default=None, metavar="N",
                       help="number of steps, overriding Tfinal/dt")
    p_run.add_argument("--quiet", action="store_true", help="print nothing")
    p_run.add_argument("--output-dir", default=None, metavar="DIR",
                       help="output directory, relative to case_dir unless absolute "
                            "(default: Output_Info { output_dir } in the input file)")

    sub.add_parser("version", help="print the package version")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        print(f"pyibm {__version__}")
        return 0
    from pyibm.driver import run   # heavy imports only when running

    try:
        result = run(args.case_dir, input_file=args.input, steps=args.steps, quiet=args.quiet,
                     output_dir=args.output_dir)
    except FloatingPointError as exc:
        print(f"pyibm: the run failed: {exc}", file=sys.stderr, flush=True)
        return 1
    if not args.quiet:
        print(f"history: {len(result.history)} rows in {result.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
