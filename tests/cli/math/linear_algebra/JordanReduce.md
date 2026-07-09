# `JordanReduce`

Returns the Jordan canonical form of a square matrix (the Jordan matrix of
`JordanDecomposition`, without the similarity matrix).

Distinct eigenvalues give a diagonal matrix in `Eigenvalues` order:

```scrut
$ wo 'JordanReduce[{{1, 2}, {2, 1}}]'
{{3, 0}, {0, -1}}
```

Defective matrices show Jordan blocks with `1` on the superdiagonal; with a
repeated eigenvalue the blocks sort ascending by eigenvalue:

```scrut
$ wo 'JordanReduce[{{2, 1, 0}, {0, 2, 0}, {0, 0, 1}}]'
{{1, 0, 0}, {0, 2, 1}, {0, 0, 2}}
```

Complex and irrational eigenvalues stay exact:

```scrut
$ wo 'JordanReduce[{{0, -1}, {1, 0}}]'
{{I, 0}, {0, -I}}
```

```scrut
$ wo 'JordanReduce[{{1, 1}, {1, 0}}]'
{{(1 + Sqrt[5])/2, 0}, {0, (1 - Sqrt[5])/2}}
```
