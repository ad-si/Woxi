# `SchurDecomposition`

Gives the real Schur decomposition `{q, t}` of a square machine-precision
matrix: `q` is orthogonal, `t` is quasi-upper-triangular (a 2×2 block for
each complex pair of eigenvalues), and `m == q . t . Transpose[q]`.

A matrix that is already quasi-upper-triangular is its own Schur form:

```scrut
$ wo 'SchurDecomposition[N[{{1, 2}, {0, 3}}]]'
{{{1., 0.}, {0., 1.}}, {{1., 2.}, {0., 3.}}}
```

```scrut
$ wo 'SchurDecomposition[N[{{0, 1}, {-1, 0}}]]'
{{{1., 0.}, {0., 1.}}, {{0., 1.}, {-1., 0.}}}
```

The factors rebuild the matrix, and the diagonal of `t` carries the
eigenvalues:

```scrut
$ wo 'a = N[{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}}]; {q, t} = SchurDecomposition[a]; Chop[q . t . Transpose[q] - a]'
{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
```

```scrut
$ wo 'Round[Sort[Diagonal[Last[SchurDecomposition[N[{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}}]]]]], 0.0001]'
{1.2679, 3., 4.7321}
```

An exact matrix has no machine-precision Schur form:

```scrut {output_stream: combined}
$ wo 'SchurDecomposition[{{1, 2}, {3, 4}}]'

SchurDecomposition::schurf: SchurDecomposition has received a matrix with infinite precision.
SchurDecomposition[{{1, 2}, {3, 4}}]
```
