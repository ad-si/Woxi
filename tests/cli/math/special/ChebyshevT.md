# `ChebyshevT`

Chebyshev polynomial of the first kind.

```scrut
$ wo 'ChebyshevT[3, x]'
-3*x + 4*x^3
```

It threads over a list of orders.

```scrut
$ wo 'ChebyshevT[{1, 2}, x]'
{x, -1 + 2*x^2}
```
