"""Parse the sample programs in tests/programs/ and check their ASTs.

The grammar has no comment syntax, so each sample is described here instead.
"""

from pathlib import Path

import pytest

from plate.parser import ParseError, parse, read
from plate.syntax import (
    Application,
    Boolean,
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
    Unquote,
    Variable,
)

PROGRAMS = Path(__file__).parent / "programs"
INVALID = PROGRAMS / "invalid"


def load(path: Path) -> Program:
    return parse(path.read_text(), filename=str(path))


# ================= AST shorthands =================

NIL = Nil()
TRUE = Boolean(True)
FALSE = Boolean(False)


def var(name: str) -> Variable:
    return Variable(name)


def num(value: int) -> Integer:
    return Integer(value)


def prim(op: str, *args) -> PrimitiveCall:
    return PrimitiveCall(op, args)


def app(func, *args) -> Application:
    return Application(func, args)


def fn(params: str, body) -> Lambda:
    return Lambda(tuple(params.split()), body)


# ================= Valid programs =================

EXPECTED = {
    # Every literal kind, all four operators, nesting, and a negative via (- 0 5).
    "arithmetic.lisp": (
        num(42),
        TRUE,
        FALSE,
        NIL,
        prim("+", num(1), num(2)),
        prim("-", num(10), num(4)),
        prim("*", num(6), num(7)),
        prim("/", num(84), num(2)),
        prim(
            "+",
            prim("*", num(2), num(3)),
            prim("-", num(10), prim("/", num(8), num(4))),
        ),
        prim("-", num(0), num(5)),
    ),
    # Recursive define shorthand; `fact` is an application, `*`/`-` are primitives.
    "factorial.lisp": (
        Define(
            "fact",
            fn(
                "n",
                If(
                    prim("eq?", var("n"), num(0)),
                    num(1),
                    prim("*", var("n"), app(var("fact"), prim("-", var("n"), num(1)))),
                ),
            ),
        ),
        app(var("fact"), num(10)),
    ),
    # Tree recursion with a nested if, and a three-parameter tail-recursive helper.
    "fibonacci.lisp": (
        Define(
            "fib",
            fn(
                "n",
                If(
                    prim("eq?", var("n"), num(0)),
                    num(0),
                    If(
                        prim("eq?", var("n"), num(1)),
                        num(1),
                        prim(
                            "+",
                            app(var("fib"), prim("-", var("n"), num(1))),
                            app(var("fib"), prim("-", var("n"), num(2))),
                        ),
                    ),
                ),
            ),
        ),
        Define(
            "fibiter",
            fn(
                "n a b",
                If(
                    prim("eq?", var("n"), num(0)),
                    var("a"),
                    app(
                        var("fibiter"),
                        prim("-", var("n"), num(1)),
                        var("b"),
                        prim("+", var("a"), var("b")),
                    ),
                ),
            ),
        ),
        app(var("fib"), num(10)),
        app(var("fibiter"), num(50), num(0), num(1)),
    ),
    # List utilities from cons/car/cdr/nil. `null?` is a user function, so calls
    # to it are applications, not primitive calls.
    "lists.lisp": (
        Define("null?", fn("xs", prim("eq?", var("xs"), NIL))),
        Define(
            "length",
            fn(
                "xs",
                If(
                    app(var("null?"), var("xs")),
                    num(0),
                    prim("+", num(1), app(var("length"), prim("cdr", var("xs")))),
                ),
            ),
        ),
        Define(
            "map",
            fn(
                "f xs",
                If(
                    app(var("null?"), var("xs")),
                    NIL,
                    prim(
                        "cons",
                        app(var("f"), prim("car", var("xs"))),
                        app(var("map"), var("f"), prim("cdr", var("xs"))),
                    ),
                ),
            ),
        ),
        Define(
            "append",
            fn(
                "xs ys",
                If(
                    app(var("null?"), var("xs")),
                    var("ys"),
                    prim(
                        "cons",
                        prim("car", var("xs")),
                        app(var("append"), prim("cdr", var("xs")), var("ys")),
                    ),
                ),
            ),
        ),
        Define(
            "reverse",
            fn(
                "xs",
                If(
                    app(var("null?"), var("xs")),
                    NIL,
                    app(
                        var("append"),
                        app(var("reverse"), prim("cdr", var("xs"))),
                        prim("cons", prim("car", var("xs")), NIL),
                    ),
                ),
            ),
        ),
        Define(
            "nums",
            prim("cons", num(1), prim("cons", num(2), prim("cons", num(3), NIL))),
        ),
        app(
            var("length"),
            app(var("map"), fn("x", prim("*", var("x"), var("x"))), var("nums")),
        ),
        app(var("reverse"), app(var("append"), var("nums"), var("nums"))),
    ),
    # Closures, currying, an explicit (define name (lambda ...)) that matches the
    # shorthand's shape, and applications whose head is itself an expression.
    "higher_order.lisp": (
        Define("compose", fn("f g", fn("x", app(var("f"), app(var("g"), var("x")))))),
        Define("adder", fn("n", fn("x", prim("+", var("x"), var("n"))))),
        Define("twice", fn("f x", app(var("f"), app(var("f"), var("x"))))),
        Define(
            "foldl",
            fn(
                "f acc xs",
                If(
                    prim("eq?", var("xs"), NIL),
                    var("acc"),
                    app(
                        var("foldl"),
                        var("f"),
                        app(var("f"), var("acc"), prim("car", var("xs"))),
                        prim("cdr", var("xs")),
                    ),
                ),
            ),
        ),
        app(app(var("adder"), num(5)), num(10)),
        app(
            app(var("compose"), app(var("adder"), num(1)), app(var("adder"), num(2))),
            num(0),
        ),
        app(var("twice"), app(var("adder"), num(3)), num(4)),
        app(fn("x y", prim("*", var("x"), var("y"))), num(6), num(7)),
        app(
            var("foldl"),
            fn("acc x", prim("+", var("acc"), var("x"))),
            num(0),
            prim("cons", num(1), prim("cons", num(2), NIL)),
        ),
    ),
    # Both meanings of begin: definition-only blocks (including empty and
    # nested) vs. sequences ending in an expression, including internal defines
    # in a function body and a nested definition block inside a sequence.
    "begin.lisp": (
        DefineBlock(()),
        DefineBlock((Define("width", num(10)), Define("height", num(20)))),
        DefineBlock(
            (
                Define("origin", num(0)),
                DefineBlock(
                    (
                        Define("unit", num(1)),
                        Define("step", fn("x", prim("+", var("x"), var("unit")))),
                    )
                ),
            )
        ),
        Define(
            "scale",
            fn(
                "k",
                Sequence(
                    (
                        Define("w", prim("*", var("k"), var("width"))),
                        Define("h", prim("*", var("k"), var("height"))),
                        prim("*", var("w"), var("h")),
                    )
                ),
            ),
        ),
        Sequence((num(1), num(2), num(3))),
        Sequence(
            (
                Define("tmp", app(var("step"), var("origin"))),
                DefineBlock((Define("more", app(var("step"), var("tmp"))),)),
                prim("+", var("tmp"), var("more")),
            )
        ),
    ),
    # Every unary and binary primitive other than arithmetic, and a pair.
    "predicates.lisp": (
        Define("both?", fn("a b", prim("and", var("a"), var("b")))),
        Define(
            "number?",
            fn("x", prim("and", prim("atom?", var("x")), prim("int?", var("x")))),
        ),
        prim("atom?", NIL),
        prim("int?", num(42)),
        prim("int?", TRUE),
        prim("eq?", TRUE, FALSE),
        app(
            var("both?"),
            prim("int?", num(1)),
            prim("atom?", prim("cons", num(1), NIL)),
        ),
        prim("car", prim("cons", num(1), num(2))),
        prim("cdr", prim("cons", num(1), num(2))),
    ),
    # Upper/mixed case is lowered; tabs, runs of spaces and blank lines are skipped.
    "case_and_whitespace.lisp": (
        Define("square", fn("x", prim("*", var("x"), var("x")))),
        app(var("square"), num(4)),
        If(TRUE, NIL, FALSE),
        Define("answer", prim("+", num(40), num(2))),
    ),
    # Identifiers that start with a keyword are still identifiers, and `?` may
    # appear anywhere after the first letter.
    "identifiers.lisp": (
        Define("definex", num(1)),
        Define("iffy", fn("lambdax", prim("+", var("lambdax"), var("definex")))),
        Define("car?", fn("x", prim("atom?", var("x")))),
        Define("nilly", NIL),
        Define("is?it?", TRUE),
        Define("begins", fn("x", var("x"))),
        app(var("iffy"), app(var("begins"), var("definex"))),
    ),
    # Whitespace only: a program with no forms.
    "empty.lisp": (),
    # quote and ' take a datum, not an expression: keywords, `.` and nested
    # quotes inside it are plain data (''x is the datum (quote x)).
    "quote.lisp": (
        Define("second", fn("xs", prim("car", prim("cdr", var("xs"))))),
        app(var("second"), Quote((1, 2))),
        Quote(Symbol("hello")),
        Quote(
            (
                Symbol("define"),
                (Symbol("f"), Symbol("x")),
                (Symbol("+"), Symbol("x"), 1),
            )
        ),
        Quote((Symbol("a"), Symbol("."), Symbol("b"))),
        prim("cons", Quote(Symbol("a")), Quote(())),
        prim("eq?", Quote(Symbol("nil")), NIL),
        Quote((Symbol("quote"), Symbol("twice"))),
        Quote((Symbol("nested"), (Symbol("quote"), Symbol("inner")))),
    ),
    # Quasiquote templates are data except where unquoted: code-building
    # helpers, the long (quasiquote ...)/(unquote ...) spelling, unquoted
    # compound expressions, a quote inside a template, a nested quasiquote
    # (plain data), and a template that is only an unquote.
    "quasiquote.lisp": (
        Define(
            "makesum",
            fn("a b", Quasiquote((Symbol("+"), Unquote(var("a")), Unquote(var("b"))))),
        ),
        Define(
            "makeadder",
            fn(
                "n",
                Quasiquote(
                    (
                        Symbol("lambda"),
                        (Symbol("x"),),
                        (Symbol("+"), Symbol("x"), Unquote(var("n"))),
                    )
                ),
            ),
        ),
        Define(
            "point",
            fn(
                "x y",
                Quasiquote(
                    (
                        Symbol("point"),
                        (Symbol("x"), Unquote(var("x"))),
                        (Symbol("y"), Unquote(var("y"))),
                    )
                ),
            ),
        ),
        app(var("makesum"), num(1), prim("*", num(2), num(3))),
        Quasiquote(
            (
                Symbol("total"),
                Unquote(app(var("makesum"), num(1), num(2))),
                Symbol("is"),
                Unquote(prim("+", num(1), num(2))),
            )
        ),
        Quasiquote(
            (Symbol("literal"), (Symbol("quote"), Symbol("sym")), Unquote(var("x")))
        ),
        Quasiquote(
            (
                Symbol("outer"),
                (
                    Symbol("quasiquote"),
                    (Symbol("inner"), (Symbol("unquote"), Symbol("x"))),
                ),
            )
        ),
        Quasiquote(Unquote(var("x"))),
    ),
}

VALID = sorted(PROGRAMS.glob("*.lisp"))


def test_every_valid_program_has_expected_ast():
    assert {path.name for path in VALID} == set(EXPECTED)


@pytest.mark.parametrize("name", EXPECTED)
def test_program_ast(name):
    assert load(PROGRAMS / name).forms == EXPECTED[name]


@pytest.mark.parametrize("path", VALID, ids=lambda path: path.name)
def test_each_form_is_one_datum(path):
    source = path.read_text()
    assert len(parse(source).forms) == len(read(source))


# ================= Invalid programs =================

# (file, line, column, start of message, whether the reader accepts it)
INVALID_CASES = [
    # `if` with only two branches.
    ("missing_else.lisp", 3, 8, "unexpected ')', expected an expression", True),
    # One `)` short: the define's single body expression is complete, so only
    # `)` may follow. Points just past the last character.
    ("unclosed.lisp", 4, 28, "unexpected end of input, expected ')'", False),
    # A keyword can't be defined as a function name.
    ("keyword_name.lisp", 1, 10, "unexpected 'car', expected an identifier", True),
    # A definition is not an expression.
    ("define_in_if.lisp", 3, 6, "unexpected 'define'", True),
    # Formals need at least one identifier.
    ("empty_formals.lisp", 1, 15, "unexpected ')', expected an identifier", True),
    # quote takes exactly one datum.
    ("quote_arity.lisp", 2, 12, "unexpected 'b', expected ')'", True),
    # `,` only means something inside a quasiquote.
    ("unquote_outside.lisp", 2, 6, "unexpected ',', expected an expression", True),
    # Identifiers have no digits: `x1` lexes as `x 1`, leaving an extra `1`.
    ("digit_in_name.lisp", 2, 12, "unexpected '1', expected ')'", True),
    # `_` isn't in the lexical grammar at all.
    ("bad_character.lisp", 2, 11, "unexpected character '_'", False),
]


def test_every_invalid_program_has_a_case():
    assert {path.name for path in INVALID.glob("*.lisp")} == {
        case[0] for case in INVALID_CASES
    }


@pytest.mark.parametrize(
    ("name", "line", "column", "message", "readable"), INVALID_CASES
)
def test_invalid_program_reports_source_location(name, line, column, message, readable):
    path = INVALID / name
    with pytest.raises(ParseError) as info:
        load(path)
    err = info.value
    assert (err.line, err.column) == (line, column)
    assert str(err).startswith(f"{path}:{line}:{column}: {message}")


@pytest.mark.parametrize(
    ("name", "line", "column", "message", "readable"), INVALID_CASES
)
def test_reader_accepts_only_subset_errors(name, line, column, message, readable):
    source = (INVALID / name).read_text()
    if readable:
        read(source)
    else:
        with pytest.raises(ParseError):
            read(source)
