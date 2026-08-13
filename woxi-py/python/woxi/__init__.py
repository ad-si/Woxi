"""Woxi — an interpreter for a subset of the Wolfram Language.

Quick start:

    >>> import woxi
    >>> woxi.interpret("Plus[1, 2]")
    '3'
    >>> woxi.interpret("Integrate[x^2, x]")
    'x^3/3'
    >>> res = woxi.evaluate("Print[\\"hi\\"]; 1 + 1")
    >>> res.result, res.stdout
    ('2', 'hi\\n')

``interpret`` returns the result formatted as Wolfram Language text and
lets ``Print`` write to the process stdout (like ``woxi eval`` on the
command line). ``evaluate`` captures stdout, graphics (SVG), sound, and
warnings alongside the result.

Results are also available as an expression tree, so you do not have to
parse Wolfram output text to use it:

    >>> woxi.evaluate("1/3 + 1/6").expr
    Expr(Symbol('Rational'), [1, 2])
    >>> woxi.to_python(woxi.evaluate("1/3 + 1/6").expr)
    Fraction(1, 2)

The same trees go the other way, so Python values reach an evaluation
without string concatenation:

    >>> from woxi import wl
    >>> woxi.evaluate_expr(wl.Total([1, 2, 3])).result
    '6'

The tree mirrors Wolfram's ``FullForm``: ``Integer``/``Real``/``String``
and lists become ``int``/``float``/``str``/``list``, symbols become
``Symbol``, and everything else is an ``Expr(head, args)`` node.
``to_python`` converts the rest (``Rational``, ``Complex``,
``Association``, ``True``/``False``/``Null``) when convenience matters
more than exactness.

Session state (variable definitions, RNG seed, ``%`` history, ...) is
kept per thread and persists across calls; use ``clear_state()`` to
reset it.
"""

from woxi import wl
from woxi._woxi import (
    BigReal,
    EvaluationResult,
    Expr,
    Graphics,
    Image,
    Sound,
    Symbol,
    WolframError,
    __version__,
    clear_state,
    evaluate,
    evaluate_expr,
    interpret,
    parse_expr,
    seed_rng,
    set_messages_to_stdout,
    set_repl_mode,
    set_script_command_line,
    set_system_variable,
    take_error_trace,
    unseed_rng,
)
from woxi.convert import to_python

__all__ = [
    "BigReal",
    "EvaluationResult",
    "Expr",
    "Graphics",
    "Image",
    "Sound",
    "Symbol",
    "WolframError",
    "__version__",
    "clear_state",
    "evaluate",
    "evaluate_expr",
    "interpret",
    "parse_expr",
    "seed_rng",
    "set_messages_to_stdout",
    "set_repl_mode",
    "set_script_command_line",
    "set_system_variable",
    "take_error_trace",
    "to_python",
    "unseed_rng",
    "wl",
]
