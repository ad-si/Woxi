# `Probability`

Compute probability of an event given a distribution.

Over a data list it is the fraction of data points that satisfy the event.

```scrut
$ wo 'Probability[x > 1, x \[Distributed] {1, 2, 3}]'
2/3
```

```scrut
$ wo 'Probability[2 < x < 5, x \[Distributed] {1, 2, 3, 4, 5, 6}]'
1/3
```
