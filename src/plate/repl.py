"""Interactive REPL for plate."""

import readline  # noqa: F401  (gives input() line editing and history)
import sys
from collections.abc import Callable
from dataclasses import dataclass, fields
from pprint import pprint

from .parser import ParseError, parse

INTEGER_SCHEMES = ("list", "residue")
LIST_SCHEMES = ("rfp", "kanerva")

PROMPT = "plate> "
CONTINUATION_PROMPT = "  ... "
COMMAND_PREFIX = "]"


@dataclass
class ReplState:
    integer_scheme: str = INTEGER_SCHEMES[0]
    list_scheme: str = LIST_SCHEMES[0]
    running: bool = True


class CommandError(Exception):
    pass


def _choose(args: list[str], choices: tuple[str, ...], command: str) -> str:
    if len(args) != 1 or args[0] not in choices:
        raise CommandError(f"usage: {COMMAND_PREFIX}{command} {'|'.join(choices)}")
    return args[0]


@dataclass(frozen=True)
class Command:
    handler: Callable[[ReplState, list[str]], None]
    usage: str
    description: str


def cmd_help(state: ReplState, args: list[str]) -> None:
    if args:
        raise CommandError(f"usage: {COMMAND_PREFIX}help")
    width = max(len(command.usage) for command in COMMANDS.values())
    for command in COMMANDS.values():
        print(f"  {command.usage:<{width}}  {command.description}")


def cmd_quit(state: ReplState, args: list[str]) -> None:
    if args:
        raise CommandError(f"usage: {COMMAND_PREFIX}quit")
    state.running = False


def cmd_debug(state: ReplState, args: list[str]) -> None:
    if args:
        raise CommandError(f"usage: {COMMAND_PREFIX}debug")
    width = max(len(field.name) for field in fields(state))
    for field in fields(state):
        print(f"  {field.name:<{width}}  {getattr(state, field.name)!r}")


def cmd_int(state: ReplState, args: list[str]) -> None:
    if not args:
        print(state.integer_scheme)
        return
    state.integer_scheme = _choose(args, INTEGER_SCHEMES, "int")


def cmd_list(state: ReplState, args: list[str]) -> None:
    if not args:
        print(state.list_scheme)
        return
    state.list_scheme = _choose(args, LIST_SCHEMES, "list")


COMMANDS: dict[str, Command] = {
    "help": Command(
        cmd_help,
        f"{COMMAND_PREFIX}help",
        "Show this list of commands",
    ),
    "quit": Command(
        cmd_quit,
        f"{COMMAND_PREFIX}quit",
        "Exit the REPL (Ctrl+D also works)",
    ),
    "debug": Command(
        cmd_debug,
        f"{COMMAND_PREFIX}debug",
        "Dump the current REPL state",
    ),
    "int": Command(
        cmd_int,
        f"{COMMAND_PREFIX}int [{'|'.join(INTEGER_SCHEMES)}]",
        "Set the integer encoding scheme, or show it if no argument is given",
    ),
    "list": Command(
        cmd_list,
        f"{COMMAND_PREFIX}list [{'|'.join(LIST_SCHEMES)}]",
        "Set the list encoding scheme, or show it if no argument is given",
    ),
}


def run_command(line: str, state: ReplState) -> None:
    parts = line.removeprefix(COMMAND_PREFIX).split()
    hint = f"type {COMMAND_PREFIX}help for a list of commands"
    if not parts:
        raise CommandError(f"expected a command; {hint}")
    name, *args = parts
    if name not in COMMANDS:
        raise CommandError(f"unknown command {COMMAND_PREFIX}{name}; {hint}")
    COMMANDS[name].handler(state, args)


def evaluate(source: str, state: ReplState) -> None:
    # Placeholder until the interpreter exists: show what the parser produced.
    try:
        program = parse(source)
    except ParseError as e:
        print(e, file=sys.stderr)
        return
    for form in program.forms:
        pprint(form)


def read_source() -> str:
    """Read one command line, or source lines until the parentheses balance."""
    source = input(PROMPT).strip()
    if source.startswith(COMMAND_PREFIX):
        return source
    # The grammar has no strings or comments, so counting parens is exact.
    while source.count("(") > source.count(")"):
        source += "\n" + input(CONTINUATION_PROMPT)
    return source


def run(state: ReplState) -> None:
    while state.running:
        try:
            source = read_source()
        except EOFError:  # Ctrl+D
            print()
            break
        except KeyboardInterrupt:  # Ctrl+C discards the current input
            print()
            continue

        if not source:
            continue
        if source.startswith(COMMAND_PREFIX):
            try:
                run_command(source, state)
            except CommandError as e:
                print(e, file=sys.stderr)
        else:
            evaluate(source, state)
