# `TautologyQ`

Tests whether a boolean expression is a tautology.

```scrut
$ wo 'TautologyQ[True]'
True
```

A second argument restricts the test to an explicit set of variables.

```scrut
$ wo 'TautologyQ[(a || b) && (!a || !b), {a, b}]'
False
```
