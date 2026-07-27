# `BooleanCountingFunction`

Represents the Boolean function of `n` variables that is True when the number
of True arguments is one the specification names: `k` for at most `k`, `{k}`
for exactly `k`, `{kmin, kmax}` for a range, and `{{k1, k2, …}}` for a set of
counts.

Applied to Boolean arguments it answers directly:

```scrut
$ wo 'BooleanCountingFunction[2, 4][True, True, False, False]'
True
```

```scrut
$ wo 'BooleanCountingFunction[{2}, 3][True, False, False]'
False
```

```scrut
$ wo 'BooleanCountingFunction[{{1, 3}}, 3][True, True, True]'
True
```

A count past the number of variables simply never matches:

```scrut
$ wo 'BooleanCountingFunction[{4}, 3][True, True, True]'
False
```

Given a list of variables it writes out the expression instead, and a third
argument names the form:

```scrut
$ wo 'BooleanCountingFunction[{{2}}, {a, b, c}]'
(a && b &&  !c) || (a &&  !b && c) || ( !a && b && c)
```

```scrut
$ wo 'BooleanCountingFunction[2, {a, b, c}, "CNF"]'
 !a ||  !b ||  !c
```

An invalid specification is reported:

```scrut {output_stream: combined}
$ wo 'BooleanCountingFunction[{-1}, 2]'

BooleanCountingFunction::bspec: BooleanCountingFunction[{-1}, 2] is not a valid BooleanCountingFunction specification.
BooleanCountingFunction[{-1}, 2]
```
