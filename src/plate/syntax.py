"""Data produced by the parser: reader datums and the evaluated-subset AST."""

from dataclasses import dataclass

# ================= Reader datums =================


class Symbol(str):
    """A word, keyword, operator, `'` or `.` read as data."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"Symbol({str.__repr__(self)})"


# Lists are tuples. Note that bool is a subclass of int: test for bool first.
type Datum = int | bool | Symbol | tuple[Datum, ...]


# ================= Expressions =================


@dataclass(frozen=True)
class Integer:
    value: int


@dataclass(frozen=True)
class Boolean:
    value: bool


@dataclass(frozen=True)
class Nil:
    pass


@dataclass(frozen=True)
class Variable:
    name: str


@dataclass(frozen=True)
class Quote:
    """`(quote d)` or `'d`: the datum `d`, unevaluated."""

    datum: Datum


@dataclass(frozen=True)
class Lambda:
    params: tuple[str, ...]
    body: Expr


@dataclass(frozen=True)
class If:
    test: Expr
    then: Expr
    orelse: Expr


@dataclass(frozen=True)
class Sequence:
    """`(begin ...)` in expression position; the last form is an expression."""

    forms: tuple[Form, ...]


@dataclass(frozen=True)
class PrimitiveCall:
    op: str
    args: tuple[Expr, ...]


@dataclass(frozen=True)
class Application:
    func: Expr
    args: tuple[Expr, ...]


type Expr = (
    Integer
    | Boolean
    | Nil
    | Variable
    | Quote
    | Lambda
    | If
    | Sequence
    | PrimitiveCall
    | Application
)


# ================= Definitions =================


@dataclass(frozen=True)
class Define:
    """`(define name value)`; `(define (f x) e)` is stored as a Lambda value."""

    name: str
    value: Expr


@dataclass(frozen=True)
class DefineBlock:
    """`(begin <definition>...)`, which contains only definitions."""

    definitions: tuple[Definition, ...]


type Definition = Define | DefineBlock
type Form = Definition | Expr


@dataclass(frozen=True)
class Program:
    forms: tuple[Form, ...]
