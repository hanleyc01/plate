"""Parse plate source into reader datums or an evaluated-subset AST."""

from lark import Lark, Token, Transformer, v_args
from lark.exceptions import (
    UnexpectedCharacters,
    UnexpectedEOF,
    UnexpectedInput,
    UnexpectedToken,
)
from lark.lexer import PatternStr

from .syntax import (
    Application,
    Boolean,
    Datum,
    Define,
    DefineBlock,
    If,
    Integer,
    Lambda,
    Nil,
    PrimitiveCall,
    Program,
    Quasiquote,
    Quote,
    Sequence,
    Symbol,
    Template,
    Unquote,
    Variable,
)

# Earley rather than LALR: `(begin <definitions>)` and `(begin <body>)` share a
# prefix that LALR(1) can't split. The basic lexer turns a WORD whose whole text
# is a keyword into that keyword's token, so `define` is DEFINE but `definex` is
# a WORD (Earley's default dynamic lexer could read `definex` as `define x`).
_LARK = Lark.open_from_package(
    "plate",
    "grammar.lark",
    start=["reader", "program"],
    parser="earley",
    lexer="basic",
)


class ParseError(Exception):
    """A syntax error, with the source position and the offending line."""

    def __init__(
        self,
        message: str,
        line: int,
        column: int,
        context: str,
        filename: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column
        self.context = context
        self.filename = filename

    def __str__(self) -> str:
        # Source files get a compiler-style location; the REPL only needs the caret.
        if self.filename is None:
            header = self.message
        else:
            header = f"{self.filename}:{self.line}:{self.column}: {self.message}"
        return f"{header}\n{self.context}"


def read(source: str, filename: str | None = None) -> tuple[Datum, ...]:
    """Parse `source` with the reader grammar into a tuple of datums."""
    tree = _parse(source, "reader", filename)
    return _ReaderTransformer().transform(tree)


def parse(source: str, filename: str | None = None) -> Program:
    """Parse `source` as a program in the evaluated subset."""
    tree = _parse(source, "program", filename)
    return _SyntaxTransformer().transform(tree)


def _parse(source: str, start: str, filename: str | None):
    try:
        return _LARK.parse(source.lower(), start=start)
    except UnexpectedInput as e:
        raise _parse_error(e, source, filename) from None


# ================= Error reporting =================

_KEYWORDS = {
    "DEFINE", "LAMBDA", "IF", "LET", "AND", "OR", "NOT", "CAR",
    "CDR", "CONS", "NIL", "EQ", "ATOM", "INTP", "QUOTE", "QUASIQUOTE",
    "UNQUOTE", "BEGIN",
}  # fmt: skip

_PREFIXES = {"QUOTE_MARK", "BACKQUOTE", "COMMA"}
_DATUM_STARTERS = {"LPAR", "INTEGER", "BOOLEAN", "WORD", "OPERATOR", "DOT"} | _PREFIXES

# Terminals that can begin a datum / an expression. When all of them are
# expected, the error says "a datum" / "an expression" instead of listing each.
# Datums come first: they're a superset, and appear in both grammars (quote).
# Quasiquote templates are datums whose atoms can't be quasiquote/unquote.
_STARTERS = (
    ("a datum", _DATUM_STARTERS | _KEYWORDS),
    ("a datum", _DATUM_STARTERS | _KEYWORDS - {"QUASIQUOTE", "UNQUOTE"}),
    (
        "an expression",
        {"LPAR", "INTEGER", "BOOLEAN", "WORD", "NIL", "QUOTE_MARK", "BACKQUOTE"},
    ),
)

_TERMINAL_NAMES = {
    "INTEGER": "an integer",
    "BOOLEAN": "a boolean",
    "WORD": "an identifier",
    "OPERATOR": "an operator",
    "$END": "end of input",
}


def _parse_error(e: UnexpectedInput, source: str, filename: str | None) -> ParseError:
    match e:
        case UnexpectedCharacters():
            line, column = e.line, e.column
            message = f"unexpected character {source[e.pos_in_stream]!r}"
        case UnexpectedToken() if e.token.type != "$END":
            line, column = e.line, e.column
            text = source[e.token.start_pos : e.token.end_pos]
            message = f"unexpected {text!r}, expected {_describe(e.expected)}"
        case UnexpectedToken() | UnexpectedEOF():
            # Point just past the last non-blank character.
            before = source.rstrip()
            line = before.count("\n") + 1
            column = len(before) - (before.rfind("\n") + 1) + 1
            message = f"unexpected end of input, expected {_describe(e.expected)}"
        case _:
            line, column = e.line, e.column
            message = str(e)
    return ParseError(message, line, column, _context(source, line, column), filename)


def _describe(expected: set[str] | list[str]) -> str:
    names = set(expected)
    parts = []
    for noun, starters in _STARTERS:
        if starters <= names:
            names -= starters
            parts.append(noun)
    parts += sorted(_terminal_name(name) for name in names)
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])} or {parts[-1]}"


def _terminal_name(name: str) -> str:
    if name in _TERMINAL_NAMES:
        return _TERMINAL_NAMES[name]
    pattern = _LARK.get_terminal(name).pattern
    return repr(pattern.value) if isinstance(pattern, PatternStr) else name


def _context(source: str, line: int, column: int) -> str:
    lines = source.splitlines() or [""]
    text = lines[min(line, len(lines)) - 1]
    # Keep tabs so the caret lines up with tab-indented source.
    pad = "".join(c if c == "\t" else " " for c in text[: column - 1])
    return f"    {text}\n    {pad}^"


# ================= Tree -> data =================


@v_args(inline=True)
class _ReaderTransformer(Transformer):
    def reader(self, *datums: Datum) -> tuple[Datum, ...]:
        return datums

    def list(self, *datums: Datum) -> tuple[Datum, ...]:
        return datums

    def quoted(self, datum: Datum) -> tuple[Datum, ...]:
        return (Symbol("quote"), datum)

    def quasiquoted(self, datum: Datum) -> tuple[Datum, ...]:
        return (Symbol("quasiquote"), datum)

    def unquoted(self, datum: Datum) -> tuple[Datum, ...]:
        return (Symbol("unquote"), datum)

    def atom(self, token: Token) -> Datum:
        match token.type:
            case "INTEGER":
                return int(token)
            case "BOOLEAN":
                return token == "#t"
            case _:
                return Symbol(token)


# Inherits the datum rules, which appear inside quote.
@v_args(inline=True)
class _SyntaxTransformer(_ReaderTransformer):
    def program(self, *forms):
        return Program(forms)

    def define(self, name: Token, value):
        return Define(str(name), value)

    def define_function(self, name: Token, params: tuple[str, ...], body):
        return Define(str(name), Lambda(params, body))

    def define_block(self, *definitions):
        return DefineBlock(definitions)

    def integer(self, token: Token):
        return Integer(int(token))

    def boolean(self, token: Token):
        return Boolean(token == "#t")

    def nil(self):
        return Nil()

    def variable(self, token: Token):
        return Variable(str(token))

    def quote(self, datum: Datum):
        return Quote(datum)

    def quasiquote(self, template: Template):
        return Quasiquote(template)

    template_atom = _ReaderTransformer.atom

    def template_list(self, *templates: Template) -> tuple[Template, ...]:
        return templates

    def template_quoted(self, template: Template) -> tuple[Template, ...]:
        return (Symbol("quote"), template)

    def template_quasiquoted(self, datum: Datum) -> tuple[Datum, ...]:
        return (Symbol("quasiquote"), datum)

    def unquote(self, expr):
        return Unquote(expr)

    def lambda_(self, params: tuple[str, ...], body):
        return Lambda(params, body)

    def conditional(self, test, then, orelse):
        return If(test, then, orelse)

    def sequence(self, *forms):
        return Sequence(forms)

    def primitive_call(self, op: Token, *args):
        return PrimitiveCall(str(op), args)

    def application(self, func, *args):
        return Application(func, args)

    def formals(self, *names: Token) -> tuple[str, ...]:
        return tuple(str(name) for name in names)
