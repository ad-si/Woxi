# `Orthogonalize`

Orthonormalizes a set of vectors using the Gram-Schmidt process with the
standard inner product. Linearly dependent vectors collapse to a zero vector.

```scrut
$ wo 'Orthogonalize[{{3, 4}, {1, 0}}]'
{{3/5, 4/5}, {4/5, -3/5}}
```

```scrut
$ wo 'Orthogonalize[{{1, 1}, {1, 0}}]'
{{1/Sqrt[2], 1/Sqrt[2]}, {1/Sqrt[2], -(1/Sqrt[2])}}
```

```scrut
$ wo 'Orthogonalize[{{1, 2, 2}, {2, 1, 2}}]'
{{1/3, 2/3, 2/3}, {10/(3*Sqrt[17]), -7/(3*Sqrt[17]), 2/(3*Sqrt[17])}}
```

A linearly dependent vector collapses to a zero vector:

```scrut
$ wo 'Orthogonalize[{{1, 1}, {2, 2}}]'
{{1/Sqrt[2], 1/Sqrt[2]}, {0, 0}}
```

```scrut
$ wo 'Orthogonalize[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}]'
{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
```
