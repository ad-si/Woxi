# `Dt`

Total derivative treating all variables as potentially dependent.

```scrut
$ wo 'Dt[5, x]'
0
```

`Dt[expr, {x, n}]` differentiates `n` times:

```scrut
$ wo 'Dt[x^4, {x, 3}]'
24*x
```

```scrut
$ wo 'Dt[Sin[x], {x, 2}]'
-Sin[x]
```

A symbolic count cannot be carried out, so the call is held —
this is how Rodrigues's formulas are written:

```scrut
$ wo 'Dt[(x^2 - 1)^n, {x, n}]'
Dt[(-1 + x^2)^n, {x, n}]
```

Other variables stay dependent, and their own derivatives grow an argument
rather than nesting — repeats folding into the `{x, n}` order spec:

```scrut
$ wo 'Dt[x y, {x, 2}]'
2*Dt[y, x] + x*Dt[y, {x, 2}]
```

A held total derivative does not depend on the symbol it differentiates, so
differentiating it by that symbol again gives 0:

```scrut
$ wo 'Dt[Dt[y, x], y]'
0
```

`Dt[expr, x1, x2, ...]` differentiates against each variable in turn:

```scrut
$ wo 'Dt[x^3, x, x]'
6*x
```
