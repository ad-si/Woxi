# `Covariance`

Computes the covariance of two lists.

```scrut
$ wo 'Covariance[{1, 2, 3}, {4, 5, 6}]'
1
```

Given a single vector, returns its variance.

```scrut
$ wo 'Covariance[{1, 2, 3}]'
1
```

Given a single matrix, returns the covariance matrix of its columns.

```scrut
$ wo 'Covariance[{{1, 2}, {3, 4}, {5, 6}}]'
{{4, 4}, {4, 4}}
```

Symbolic vectors give the closed form (conjugating the second vector).

```scrut
$ wo 'Covariance[{a, b}, {x, y}]'
((a - b)*(Conjugate[x] - Conjugate[y]))/2
```
