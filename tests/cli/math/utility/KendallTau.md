# `KendallTau`

Computes Kendall's rank correlation coefficient τ for two equal-length
vectors.

```scrut
$ wo 'KendallTau[{1, 2, 3, 4, 5}, {2, 1, 4, 3, 5}]'
3/5
```

Reversed orderings are perfectly discordant:

```scrut
$ wo 'KendallTau[{1, 2, 3, 4, 5}, {5, 4, 3, 2, 1}]'
-1
```

Tied values are handled with the τ_b correction:

```scrut
$ wo 'KendallTau[{1, 1, 2, 3}, {1, 2, 2, 3}]'
4/5
```
