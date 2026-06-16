# `Correlation`

Computes the Pearson correlation of two lists.

```scrut
$ wo 'Correlation[{1, 2, 3}, {2, 4, 6}]'
1
```

Given a single data matrix, it returns the correlation matrix of the columns.

```scrut
$ wo 'Correlation[{{1, 5}, {2, 4}, {3, 3}}]'
{{1, -1}, {-1, 1}}
```
