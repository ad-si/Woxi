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

```scrut
$ wo 'Integrate[Log[Sin[x]], {x, 0, Pi/2}]'
-1/2*(Pi*Log[2])
```

A logarithmic antiderivative keeps the power the terms share out front, as
wolframscript writes it:

```scrut
$ wo 'Integrate[Log[x], x]'
x*(-1 + Log[x])
```

```scrut
$ wo 'ToString[Integrate[x Log[x], x], InputForm]'
(x^2*(-1 + 2*Log[x]))/4
```

The scale of a logarithm does not factor out that way:

```scrut
$ wo 'ToString[Integrate[Log[2 x], x], InputForm]'
-x + x*Log[2*x]
```

A negative exponent keeps its power form after a minus sign:

```scrut
$ wo 'ToString[Integrate[E^(-x), x], InputForm]'
-E^(-x)
```
