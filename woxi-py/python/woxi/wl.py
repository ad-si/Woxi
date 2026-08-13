"""Builder namespace for Wolfram expressions.

Every attribute is the symbol of that name, and calling a symbol applies it:

    >>> from woxi import wl
    >>> wl.x
    Symbol("x")
    >>> wl.Integrate(wl.Power(wl.x, 2), wl.x)
    Expr(Symbol("Integrate"), [Expr(Symbol("Power"), [Symbol("x"), 2]), Symbol("x")])

Any Wolfram symbol works, including ones Woxi does not implement — the name
is not checked here, only when the expression is evaluated.
"""

from woxi._woxi import Symbol

__all__ = ["Symbol"]


def __getattr__(name: str) -> Symbol:
    # Dunder lookups (__path__, __all__, copy/pickle protocol, ...) must
    # fail as attribute errors rather than turning into Wolfram symbols,
    # or the import machinery and the REPL misbehave.
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return Symbol(name)


def __dir__() -> list[str]:
    return __all__
