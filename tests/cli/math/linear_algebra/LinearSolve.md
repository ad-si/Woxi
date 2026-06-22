# `LinearSolve`

Solves the linear system `m . x == b`.

```scrut
$ wo 'LinearSolve[{{1, 2}, {3, 4}}, {5, 11}]'
{1, 2}
```

`LinearSolve[m]` is an operator: `LinearSolve[m][b]` equals
`LinearSolve[m, b]`.

```scrut
$ wo 'LinearSolve[{{1, 2}, {3, 4}}][{5, 11}]'
{1, 2}
```

For a rectangular (non-square) system, a particular solution is returned with
the free variables set to 0.

```scrut
$ wo 'LinearSolve[{{1, 2, 3}, {4, 5, 6}}, {1, 2}]'
{-1/3, 2/3, 0}
```
