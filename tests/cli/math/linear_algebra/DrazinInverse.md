# `DrazinInverse`

The Drazin inverse of a square matrix. For an invertible matrix it is the
ordinary inverse; for a nilpotent matrix it is the zero matrix.

```scrut
$ wo 'DrazinInverse[{{1, 2}, {3, 4}}]'
{{-2, 1}, {3/2, -1/2}}
```

A singular index-1 matrix inverts its core part and zeroes the rest.

```scrut
$ wo 'DrazinInverse[{{2, 1}, {0, 0}}]'
{{1/2, 1/4}, {0, 0}}
```

```scrut
$ wo 'DrazinInverse[{{0, 1}, {0, 0}}]'
{{0, 0}, {0, 0}}
```
