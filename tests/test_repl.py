import pytest

from plate import build_parser
from plate.repl import CONTINUATION_PROMPT, VECTOR_SCHEMES, ReplState, run


def run_repl(monkeypatch, *lines: str) -> list[str]:
    """Run the REPL on scripted input lines; return the prompts it asked with."""
    remaining = iter(lines)
    prompts = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    run(ReplState())
    return prompts


def test_prints_parsed_form(monkeypatch, capsys):
    run_repl(monkeypatch, "(+ 1 2)")
    out = capsys.readouterr().out
    assert "PrimitiveCall(op='+', args=(Integer(value=1), Integer(value=2)))" in out


def test_unbalanced_input_continues_on_next_line(monkeypatch, capsys):
    prompts = run_repl(monkeypatch, "(define (f x)", "  (+ x 1))")
    out = capsys.readouterr().out
    assert CONTINUATION_PROMPT in prompts
    assert out.count("Define(name='f'") == 1


def test_prints_every_form_on_a_line(monkeypatch, capsys):
    run_repl(monkeypatch, "(define x 1) x")
    out = capsys.readouterr().out
    assert "Define(name='x', value=Integer(value=1))" in out
    assert "Variable(name='x')" in out


def test_parse_error_is_reported_without_location_and_repl_continues(
    monkeypatch, capsys
):
    run_repl(monkeypatch, "(if 1 2)", "x")
    captured = capsys.readouterr()
    assert captured.err.startswith("unexpected ')', expected an expression")
    assert ":1:" not in captured.err
    assert "Variable(name='x')" in captured.out


def test_commands_still_work(monkeypatch, capsys):
    prompts = run_repl(monkeypatch, "]int residue", "]debug", "]quit", "never read")
    out = capsys.readouterr().out
    assert "'residue'" in out
    assert len(prompts) == 3


def test_vector_command_shows_and_sets_the_scheme(monkeypatch, capsys):
    run_repl(monkeypatch, "]vector", "]vector gaussian", "]vector")
    out = capsys.readouterr().out.split()

    assert out == ["unitary", "gaussian"]


def test_vector_command_rejects_an_unknown_scheme(monkeypatch, capsys):
    run_repl(monkeypatch, "]vector bipolar")

    assert "usage: ]vector unitary|gaussian" in capsys.readouterr().err


def test_vector_scheme_defaults_to_unitary():
    assert ReplState().vector_scheme == "unitary" == VECTOR_SCHEMES[0]


# ================= Command line =================


def test_cli_takes_a_vector_scheme():
    assert build_parser().parse_args(["-r", "-v", "gaussian"]).vector_scheme == (
        "gaussian"
    )


def test_cli_defaults_to_unitary():
    assert build_parser().parse_args(["-r"]).vector_scheme == "unitary"


def test_cli_rejects_an_unknown_vector_scheme(capsys):
    with pytest.raises(SystemExit):
        _ = build_parser().parse_args(["-r", "-v", "bipolar"])

    assert "invalid choice: 'bipolar'" in capsys.readouterr().err
