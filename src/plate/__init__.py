import argparse
from pathlib import Path

from . import repl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="plate")
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        help="Path to a LISP file to be interpreted",
    )
    parser.add_argument(
        "-s",
        "--integer-scheme",
        choices=repl.INTEGER_SCHEMES,
        default="list",
        help="Integer encoding scheme (default: %(default)s)",
    )
    parser.add_argument(
        "-l",
        "--list-scheme",
        choices=repl.LIST_SCHEMES,
        default="rfp",
        help="List encoding scheme (default: %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--interpret",
        metavar="TEXT",
        help="interpret TEXT directly instead of reading a file",
    )
    parser.add_argument(
        "-r",
        "--repl",
        action="store_true",
        help="Start an interactive REPL",
    )
    parser.add_argument(
        "-k",
        "--save-temps",
        action="store_true",
        default=True,
        help="Keep intermediate files instead of deleting them (default: %(default)s)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.file is None and args.interpret is None and not args.repl:
        parser.print_usage()
        return

    if args.repl:
        if args.file is not None:
            parser.error("--repl cannot be used with a file")
        repl.run(
            repl.ReplState(
                integer_scheme=args.integer_scheme, list_scheme=args.list_scheme
            )
        )
        return

    print(args)


if __name__ == "__main__":
    main()
