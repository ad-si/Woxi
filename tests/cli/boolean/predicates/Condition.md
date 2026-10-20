# `Condition`

Represents a pattern with a condition test (pattern /; test).

```scrut
$ wo 'Condition[x_, x > 0]'
x_ /; x > 0
```

The test can do pattern matching of its own,
e.g. check that a sequence consists only of `g[…]` expressions:

```scrut
$ wo 'f[g[x], g[y, z]] /. (f[a__] /; MatchQ[h[a], h[__g]]) :> True'
True
```
