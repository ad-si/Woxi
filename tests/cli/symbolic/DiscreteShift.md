# `DiscreteShift`

Substitute `n -> n + 1` in an expression (a discrete forward shift).

```scrut
$ wo 'DiscreteShift[n^2, n]'
(1 + n)^2
```

For an integer shift a rational summand is combined over a common denominator.

```scrut
$ wo 'DiscreteShift[1/(2 n + 1), n]'
(3 + 2*n)^(-1)
```
