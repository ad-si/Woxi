# `BinomialDistribution`

Binomial probability distribution.

```scrut
$ wo 'BinomialDistribution[n, p]'
BinomialDistribution[n, p]
```

The mean of `n` trials with success probability `p` is `n*p`; the
variance is `n*(1 - p)*p`:

```scrut
$ wo 'Mean[BinomialDistribution[n, p]]'
n*p
```

```scrut
$ wo 'Variance[BinomialDistribution[n, p]]'
n*(1 - p)*p
```

Concrete parameters collapse to exact numbers:

```scrut
$ wo 'Mean[BinomialDistribution[10, 1/2]]'
5
```

```scrut
$ wo 'Variance[BinomialDistribution[10, 1/2]]'
5/2
```
