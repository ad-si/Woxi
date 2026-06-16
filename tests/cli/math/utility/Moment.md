# `Moment`

Raw statistical moment.

```scrut
$ wo 'Moment[{1,2,3}, 0]'
1
```

For a distribution, the raw moment `E[x^n]` is returned. A Bernoulli variable
takes values in `{0, 1}`, so every raw moment equals `p`.

```scrut
$ wo 'Moment[BernoulliDistribution[p], 3]'
p
```
