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
