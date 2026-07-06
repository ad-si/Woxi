# `PfaffianDet`

The Pfaffian of an antisymmetric matrix, defined so that
`PfaffianDet[m]^2 == Det[m]`.

```scrut
$ wo 'PfaffianDet[{{0, 1}, {-1, 0}}]'
1
```

```scrut
$ wo 'PfaffianDet[{{0, 1, 2, 3}, {-1, 0, 4, 5}, {-2, -4, 0, 6}, {-3, -5, -6, 0}}]'
8
```

An odd-order matrix has Pfaffian `0`:

```scrut
$ wo 'PfaffianDet[{{0, 1, 2}, {-1, 0, 3}, {-2, -3, 0}}]'
0
```

A non-antisymmetric argument stays unevaluated:

```scrut
$ wo 'PfaffianDet[{{5, 7}, {2, 9}}]'
PfaffianDet[{{5, 7}, {2, 9}}]
```
