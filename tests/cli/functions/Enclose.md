# `Enclose`

Structured error handling. `Enclose` evaluates its body; a `Confirm*` inside
that is not satisfied abandons the rest of the body and hands a
`Failure[…]` object back to the `Enclose`.

When nothing fails, the body's value comes through unchanged:

```scrut
$ wo 'Enclose[Confirm[5] + 2]'
7
```

`Confirm[expr]` passes `expr` through unless it is a failure — `$Failed`,
a `Failure[…]`, or a `Missing[…]`. Ordinary values such as `0` and `False`
are not failures:

```scrut
$ wo 'Enclose[Confirm[0]]'
0
```

`Enclose` on its own does not convert anything; only a `Confirm*` does:

```scrut
$ wo 'Enclose[$Failed]'
$Failed
```

The failure object records which confirmation gave up:

```scrut
$ wo 'Enclose[Confirm[$Failed]]["ConfirmationType"]'
Confirm
```

```scrut
$ wo 'Enclose[Confirm[$Failed]]["Tag"]'
ConfirmationFailed
```

`ConfirmBy[expr, f]` requires `f[expr]` to be `True`, and remembers the
predicate it used:

```scrut
$ wo 'Enclose[ConfirmBy[3, NumberQ] + 2]'
5
```

```scrut
$ wo 'Enclose[ConfirmBy["a", NumberQ]]["Function"]'
NumberQ
```

`ConfirmMatch[expr, patt]` requires a pattern match, and `ConfirmAssert[test]`
requires `test` to be `True` — keeping the test unevaluated so the failure can
show what was asserted:

```scrut
$ wo 'Enclose[ConfirmMatch["a", _Integer]]["Pattern"]'
_Integer
```

```scrut
$ wo 'Enclose[ConfirmAssert[1 > 2]]["HeldTest"]'
Hold[1 > 2]
```

`ConfirmQuiet[expr]` evaluates `expr` with messages suppressed:

```scrut
$ wo 'Enclose[ConfirmQuiet[Log[0]]]'
-Infinity
```

A second argument to `Enclose` reads a property off the failure when it is a
string, and is otherwise applied to the failure object:

```scrut
$ wo 'Enclose[Confirm[$Failed], "Tag"]'
ConfirmationFailed
```

```scrut
$ wo 'Enclose[Confirm[$Failed], Head]'
Failure
```

A `Confirm*` with no surrounding `Enclose` has nowhere to throw to, so it
reports `::confirmnotag` and returns a failure describing the call — even when
the confirmation itself would have succeeded:

```scrut
$ wo 'Confirm[1 + 1]["HeldInput"]'
Hold[Confirm[1 + 1]]
```
