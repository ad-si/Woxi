# `SquareFreeQ`

Tests if a polynomial or integer is square-free.

```scrut
$ wo 'SquareFreeQ[1]'
True
```

A univariate polynomial is square-free when it has no repeated factor:

```scrut
$ wo 'SquareFreeQ[x^2 - 1]'
True
```

```scrut
$ wo 'SquareFreeQ[(x - 1)^2]'
False
```
