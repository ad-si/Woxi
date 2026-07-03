# `GeneratingFunction`

Compute generating function of a sequence.

```scrut
$ wo 'GeneratingFunction[a, n, x]'
a/(1 - x)
```

The Fibonacci and Lucas sequences share the denominator `1 - x - x^2`.

```scrut
$ wo 'GeneratingFunction[Fibonacci[n], n, x]'
-(x/(-1 + x + x^2))
```

```scrut
$ wo 'GeneratingFunction[LucasL[n], n, x]'
(-2 + x)/(-1 + x + x^2)
```

The Catalan numbers and the reciprocal-shift sequence.

```scrut
$ wo 'GeneratingFunction[CatalanNumber[n], n, x]'
2/(1 + Sqrt[1 - 4*x])
```

```scrut
$ wo 'GeneratingFunction[1/(n + 2), n, x]'
(-1 - Log[1 - x]/x)/x
```
