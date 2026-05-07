# `Integrate`

Symbolic integration.
`Integrate[expr, x]` gives the indefinite integral;
`Integrate[expr, {x, a, b}]` gives the definite integral.

```scrut
$ wo 'Integrate[x, x]'
x^2/2
```

```scrut
$ wo 'Integrate[x^2, {x, 0, 1}]'
1/3
```

```scrut
$ wo 'Integrate[Sin[x], {x, 0, Pi}]'
2
```
