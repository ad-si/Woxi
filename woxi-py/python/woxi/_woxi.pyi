"""Type stubs for the compiled extension module woxi._woxi."""

from collections.abc import Iterable
from typing import Any

__version__: str

class WolframError(Exception):
    """Raised when Woxi fails to parse or evaluate Wolfram Language code."""

class Symbol:
    """A Wolfram symbol. Calling it applies it: Symbol("f")(1) is f[1]."""

    name: str
    def __init__(self, name: str) -> None: ...
    def __call__(self, *args: Any) -> Expr: ...

class Expr:
    """A head applied to arguments, in FullForm shape."""

    head: Any
    args: list[Any]
    def __init__(self, head: Any, args: Iterable[Any]) -> None: ...
    def __call__(self, *args: Any) -> Expr: ...

class BigReal:
    """An arbitrary-precision real, e.g. from N[Pi, 30]."""

    digits: str
    precision: float
    def __init__(self, digits: str, precision: float) -> None: ...

class Graphics:
    """A rendered graphic. Output only — cannot be evaluated back."""

    svg: str
    is_3d: bool

class Image:
    """A raster image. Output only — pixel data is not carried into Python."""

    width: int
    height: int
    channels: int

class Sound:
    base64: str
    mime: str
    label: str | None

class EvaluationResult:
    result: str
    # None when no value was produced (a definition, or a `;`-suppressed
    # statement); Symbol("Null") when the value itself is Null.
    expr: Any | None
    stdout: str
    graphics: str | None
    svg: str | None
    sound: Sound | None
    warnings: list[str]

def interpret(code: str) -> str: ...
def evaluate(code: str, *, print_to_stdout: bool = False) -> EvaluationResult: ...
def evaluate_expr(
    expr: Any, *, print_to_stdout: bool = False
) -> EvaluationResult: ...
def parse_expr(code: str) -> Any: ...
def clear_state() -> None: ...
def seed_rng(seed: int) -> None: ...
def unseed_rng() -> None: ...
def set_repl_mode(enabled: bool) -> None: ...
def set_messages_to_stdout(enabled: bool) -> None: ...
def set_system_variable(name: str, value: str) -> None: ...
def set_script_command_line(args: list[str]) -> None: ...
def take_error_trace() -> str | None: ...
