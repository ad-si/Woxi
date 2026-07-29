# `ToExpression`

Converts a string to an evaluated expression.

```scrut
$ wo 'ToExpression["1 + 2"]'
3
```

```scrut
$ wo 'ToExpression["Plus[3, 4]"]'
7
```

A definition in the string takes effect, including one whose left-hand side
carries a pattern:

```scrut
$ wo 'ToExpression["k[x_] := x + 1; k[2]"]'
3
```

```scrut
$ wo 'ToExpression["f[x_] := x*2"]; f[3]'
6
```

With a holding head the definition is parsed but not performed, so it can be
released later:

```scrut
$ wo 'ToExpression["q[x_] := x^2", InputForm, Hold]'
Hold[q[x_] := x^2]
```
