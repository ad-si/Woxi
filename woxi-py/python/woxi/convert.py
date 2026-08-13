"""Opt-in conversion from the faithful expression tree to Python types.

``evaluate_expr`` returns a tree that mirrors Wolfram's ``FullForm``, so
``1/3`` is ``Expr(Symbol("Rational"), [1, 3])`` rather than a Python
number. That is exact but not always convenient; ``to_python`` trades some
of that fidelity for types Python code can work with directly.

The conversion is lossy and is therefore never applied automatically:

===============================  ==========================
Wolfram                          Python
===============================  ==========================
``Rational[n, d]``               ``fractions.Fraction``
``Complex[re, im]``              ``complex`` (inexact!)
``Association[k -> v, ...]``     ``dict``
``True`` / ``False`` / ``Null``  ``True`` / ``False`` / ``None``
===============================  ==========================

Anything else is returned unchanged, with its children converted. A node
that cannot be converted faithfully — a complex with symbolic parts, an
association with an unhashable key — is left alone rather than mangled, so
the result is always something ``evaluate_expr`` accepts back.
"""

from fractions import Fraction
from typing import Any

from woxi._woxi import Expr, Symbol

__all__ = ["to_python"]

_CONSTANTS = {"True": True, "False": False, "Null": None}

# Values that can serve as the real or imaginary part of a Python complex.
_REAL = (int, float, Fraction)


def _head_name(value: Expr) -> str | None:
    """The head's symbol name, or None for a compound (curried) head."""
    head = value.head
    return head.name if isinstance(head, Symbol) else None


def to_python(value: Any) -> Any:
    """Convert an expression tree to convenient Python types.

    Recurses into lists and expression arguments. See the module docstring
    for the mapping table.
    """
    if isinstance(value, list):
        return [to_python(item) for item in value]

    if isinstance(value, Symbol):
        # `Symbol("True")` is Wolfram's True; `.get` keeps every other
        # symbol as itself.
        return _CONSTANTS.get(value.name, value)

    if not isinstance(value, Expr):
        return value

    name = _head_name(value)
    args = [to_python(arg) for arg in value.args]

    if name == "Rational" and len(args) == 2:
        if all(isinstance(a, int) for a in args):
            return Fraction(args[0], args[1])

    elif name == "Complex" and len(args) == 2:
        # Only a numeric complex maps to Python's; `Complex[a, b]` with
        # symbolic parts has no Python counterpart.
        if all(isinstance(a, _REAL) for a in args):
            return complex(args[0], args[1])

    elif name == "Association":
        entries = _association_entries(args)
        if entries is not None:
            return entries

    # Not a special form (or not convertible): rebuild with the converted
    # children, and convert a compound head too.
    head = value.head
    return Expr(to_python(head) if isinstance(head, Expr) else head, args)


def _association_entries(args: list[Any]) -> dict[Any, Any] | None:
    """A dict for an association of plain rules, else None.

    Returns None — leaving the caller with the original expression — when
    an entry is not a ``Rule`` (e.g. ``RuleDelayed``) or when a key cannot
    be a dict key, rather than silently dropping information.
    """
    entries: dict[Any, Any] = {}
    for arg in args:
        if not isinstance(arg, Expr) or _head_name(arg) != "Rule":
            return None
        rule_args = list(arg.args)
        if len(rule_args) != 2:
            return None
        key, val = rule_args
        try:
            entries[key] = val
        except TypeError:  # unhashable key, e.g. <|{1, 2} -> 3|>
            return None
    return entries
