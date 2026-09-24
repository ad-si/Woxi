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

It stays exact when combined with another exact number, but numericalizes
automatically once an inexact number is mixed in.

```scrut
$ wo 'Im[ZetaZero[1]] - 14.'
0.13472514173469463
```
