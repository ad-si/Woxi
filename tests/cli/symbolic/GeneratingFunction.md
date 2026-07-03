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
