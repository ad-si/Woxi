# `StandardDeviationFilter`

Replaces each element with the sample standard deviation of its radius-`r`
neighborhood.

```scrut
$ wo 'StandardDeviationFilter[{1, 2, 3, 4, 100}, 1]'
{1/Sqrt[2], 1, 1, Sqrt[9313/3], 48*Sqrt[2]}
```

A single element has no sample standard deviation, so a range of zero has
nothing to work with:

```scrut
$ wo 'StandardDeviationFilter[{1, 2, 3, 4, 100}, 0]'

StandardDeviationFilter::shlen: Cannot compute the standard deviation of one element.
StandardDeviationFilter[{1, 2, 3, 4, 100}, 0]
```

The range must be a whole number:

```scrut
$ wo 'StandardDeviationFilter[{1, 2, 3, 4, 100}, 1.5]'

StandardDeviationFilter::bdrad: 1.5 is not a valid neighborhood range specification.
StandardDeviationFilter[{1, 2, 3, 4, 100}, 1.5]
```
