# `Threshold`

Replaces array values whose magnitude is at or below the threshold with zero.

```scrut
$ wo 'Threshold[{1, -2, 3, -0.5, 4}, 2]'
{0., 0., 3., 0., 4.}
```

The thresholding method can be given as `{method, delta}`. Soft thresholding
shrinks each value toward zero by `delta`:

```scrut
$ wo 'Threshold[{1, 2, 3, 4, 5}, {"Soft", 2}]'
{0., 0., 1., 2., 3.}
```

The non-negative garrote method:

```scrut
$ wo 'Threshold[{1, 2, 3, 4, 5}, {"PiecewiseGarrote", 2}]'
{0., 0., 1.6666666666666667, 3., 4.2}
```
