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


def parse_one(source: str):
    (form,) = parse(source).forms
    return form


# ================= read =================


def test_read_nested_lists():
    assert read("(a (1 (#t)) ())") == ((Symbol("a"), (1, (True,)), ()),)


def test_read_keywords_and_dot_are_atoms():
    assert read("(define . eq? quote)") == (
        (Symbol("define"), Symbol("."), Symbol("eq?"), Symbol("quote")),
    )


def test_read_quote_mark_expands_to_quote_list():
    quote = Symbol("quote")
    assert read("'x '(1 2) ''y") == (
        (quote, Symbol("x")),
        (quote, (1, 2)),
        (quote, (quote, Symbol("y"))),
    )


def test_read_quasiquote_and_unquote_expand_to_lists():
    assert read("`x ,x `(a ,b)") == (
        (Symbol("quasiquote"), Symbol("x")),
        (Symbol("unquote"), Symbol("x")),
        (Symbol("quasiquote"), (Symbol("a"), (Symbol("unquote"), Symbol("b")))),
    )


def test_read_negative_number_is_operator_then_integer():
    assert read("-5") == (Symbol("-"), 5)


def test_read_lowercases_input():
    assert read("(CAR #F)") == ((Symbol("car"), False),)


def test_read_empty_program():
    assert read("") == ()


def test_read_integers_and_booleans_are_distinct():
    (one, true) = read("1 #t")
    assert type(one) is int and type(true) is bool


# ================= parse: expressions =================


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("42", Integer(42)),
        ("#t", Boolean(True)),
        ("#F", Boolean(False)),
        ("nil", Nil()),
        ("x", Variable("x")),
        ("definex", Variable("definex")),
        ("null?", Variable("null?")),
        ("(lambda (a b) a)", Lambda(("a", "b"), Variable("a"))),
        ("(if #t 1 2)", If(Boolean(True), Integer(1), Integer(2))),
        ("(car x)", PrimitiveCall("car", (Variable("x"),))),
        ("(atom? x)", PrimitiveCall("atom?", (Variable("x"),))),
        ("(eq? a b)", PrimitiveCall("eq?", (Variable("a"), Variable("b")))),
        ("(- 3 1)", PrimitiveCall("-", (Integer(3), Integer(1)))),
        ("(f)", Application(Variable("f"), ())),
        ("(f 1 2)", Application(Variable("f"), (Integer(1), Integer(2)))),
        (
            "((lambda (x) x) 5)",
            Application(Lambda(("x",), Variable("x")), (Integer(5),)),
        ),
        ("(begin 1 2)", Sequence((Integer(1), Integer(2)))),
        (
            "(begin (define x 1) x)",
            Sequence((Define("x", Integer(1)), Variable("x"))),
        ),
        ("(quote x)", Quote(Symbol("x"))),
        ("'x", Quote(Symbol("x"))),
        ("'42", Quote(42)),
        ("'#t", Quote(True)),
        ("'()", Quote(())),
        (
            "'(car (1 . define))",
            Quote((Symbol("car"), (1, Symbol("."), Symbol("define")))),
        ),
        ("''x", Quote((Symbol("quote"), Symbol("x")))),
        ("(quote 'x)", Quote((Symbol("quote"), Symbol("x")))),
        ("(car '(1 2))", PrimitiveCall("car", (Quote((1, 2)),))),
        # Unquote inside plain quote is just data.
        ("'(a ,x)", Quote((Symbol("a"), (Symbol("unquote"), Symbol("x"))))),
        ("`x", Quasiquote(Symbol("x"))),
        ("`()", Quasiquote(())),
        ("`(a ,x)", Quasiquote((Symbol("a"), Unquote(Variable("x"))))),
        (
            "(quasiquote (a (unquote x)))",
            Quasiquote((Symbol("a"), Unquote(Variable("x")))),
        ),
        (
            "`(a ,(+ x 1))",
            Quasiquote(
                (Symbol("a"), Unquote(PrimitiveCall("+", (Variable("x"), Integer(1)))))
            ),
        ),
        (
            "`(a (b (c ,x)))",
            Quasiquote(
                (Symbol("a"), (Symbol("b"), (Symbol("c"), Unquote(Variable("x")))))
            ),
        ),
        # Keywords in a template are data, and unquote works inside a quote.
        (
            "`(if ,x 'y)",
            Quasiquote(
                (Symbol("if"), Unquote(Variable("x")), (Symbol("quote"), Symbol("y")))
            ),
        ),
        (
            "`'(,x)",
            Quasiquote((Symbol("quote"), (Unquote(Variable("x")),))),
        ),
        # A nested quasiquote is plain data: its unquote isn't evaluated.
        (
            "`(a `(b ,x))",
            Quasiquote(
                (
                    Symbol("a"),
                    (
                        Symbol("quasiquote"),
                        (Symbol("b"), (Symbol("unquote"), Symbol("x"))),
                    ),
                )
            ),
        ),
    ],
)
def test_parse_expression(source, expected):
    assert parse_one(source) == expected


# ================= parse: definitions and programs =================


def test_parse_define():
    assert parse_one("(define x 1)") == Define("x", Integer(1))


def test_parse_define_shorthand_becomes_lambda():
    assert parse_one("(define (f x y) (+ x y))") == Define(
        "f",
        Lambda(("x", "y"), PrimitiveCall("+", (Variable("x"), Variable("y")))),
    )


def test_parse_begin_with_only_definitions_is_define_block():
    assert parse_one("(begin (define x 1) (begin (define y 2)))") == DefineBlock(
        (Define("x", Integer(1)), DefineBlock((Define("y", Integer(2)),)))
    )


def test_parse_empty_begin_is_define_block():
    assert parse_one("(begin)") == DefineBlock(())


def test_parse_empty_program():
    assert parse("") == Program(())


def test_parse_multiple_forms():
    assert parse("(define x 1)\nx") == Program((Define("x", Integer(1)), Variable("x")))


def test_parse_is_case_insensitive():
    assert parse_one("(IF #T X NIL)") == If(Boolean(True), Variable("x"), Nil())


# ================= parse: rejected programs =================


@pytest.mark.parametrize(
    "source",
    [
        "(if 1 2)",
        "(lambda () 1)",
        "(define if 1)",
        "(car x y)",
        "(cons x)",
        "(quote)",
        "(quote a b)",
        "'",
        ",x",
        "(unquote x)",
        "(f ,x)",
        "'(a ,x) ,x",
        "(quasiquote)",
        "(quasiquote a b)",
        "`(a ,)",
        "`",
        "(let ((x 1)) x)",
        "(or a b)",
        "(f 1",
        "(f 1))",
        "()",
        "x & y",
        "(if #t (begin (define x 1)) 2)",
        "(lambda (x) (define y 1))",
    ],
)
def test_parse_rejects(source):
    with pytest.raises(ParseError):
        parse(source)


def test_read_rejects_unbalanced():
    with pytest.raises(ParseError):
        read("(a b))")


# ================= error reporting =================


def test_error_with_filename_has_source_location():
    with pytest.raises(ParseError) as info:
        parse("(define x 1)\n\n  (if (eq? x 1) )\n", filename="t.lisp")
    err = info.value
    assert (err.line, err.column) == (3, 17)
    assert str(err) == (
        "t.lisp:3:17: unexpected ')', expected an expression\n"
        "      (if (eq? x 1) )\n"
        "                    ^"
    )


def test_error_without_filename_has_no_location_prefix():
    with pytest.raises(ParseError) as info:
        parse("(if 1 2)")
    err = info.value
    assert (err.line, err.column) == (1, 8)
    assert str(err) == (
        "unexpected ')', expected an expression\n    (if 1 2)\n           ^"
    )


def test_error_at_end_of_input_points_after_last_character():
    with pytest.raises(ParseError) as info:
        parse("(f 1\n\n", filename="t.lisp")
    err = info.value
    assert (err.line, err.column) == (1, 5)
    assert str(err).startswith("t.lisp:1:5: unexpected end of input")


@pytest.mark.parametrize("source", ["(quote)", "(quasiquote)"])
def test_error_after_quote_expects_a_datum(source):
    with pytest.raises(ParseError) as info:
        parse(source)
    assert info.value.message == "unexpected ')', expected a datum"


def test_error_shows_original_case():
    with pytest.raises(ParseError) as info:
        parse("(DEFINE IF 1)")
    assert "unexpected 'IF'" in str(info.value)
