# `Expectation`

Compute expected value of a function of a random variable.

Over a data list it is the empirical (sample) mean of the expression with the
variable replaced by each data point.

```scrut
$ wo 'Expectation[x, x \[Distributed] {1, 2, 3}]'
2
```

```scrut
$ wo 'Expectation[x^2, x \[Distributed] {1, 2, 3, 4}]'
15/2
```
