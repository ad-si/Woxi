# `ZetaZero`

k-th non-trivial zero of the Riemann zeta function on the critical line.

```scrut
$ wo 'ZetaZero[1]'
ZetaZero[1]
```

Numeric evaluation finds the zero 1/2 + i t_k.

```scrut
$ wo 'N[ZetaZero[1]]'
0.5 + 14.134725141734695*I
```

`ZetaZero` is `Listable`, and `N` keeps its index exact.

```scrut
$ wo 'N[ZetaZero[{1, 2, 3}]]'
{0.5 + 14.134725141734695*I, 0.5 + 21.022039638771556*I, 0.5 + 25.01085758014569*I}
```

It stays exact when combined with another exact number, but numericalizes
automatically once an inexact number is mixed in.

```scrut
$ wo 'Re[ZetaZero[1]] + 0.5'
1.
```
