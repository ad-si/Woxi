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

Session state (variable definitions, RNG seed, ``%`` history, ...) is
kept per thread and persists across calls; use ``clear_state()`` to
reset it.
"""

from woxi._woxi import (
    EvaluationResult,
    Sound,
    WolframError,
    __version__,
    clear_state,
    evaluate,
    interpret,
    seed_rng,
    set_messages_to_stdout,
    set_repl_mode,
    set_script_command_line,
    set_system_variable,
    take_error_trace,
    unseed_rng,
)

__all__ = [
    "EvaluationResult",
    "Sound",
    "WolframError",
    "__version__",
    "clear_state",
    "evaluate",
    "interpret",
    "seed_rng",
    "set_messages_to_stdout",
    "set_repl_mode",
    "set_script_command_line",
    "set_system_variable",
    "take_error_trace",
    "unseed_rng",
]
