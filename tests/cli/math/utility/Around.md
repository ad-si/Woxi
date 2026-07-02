# `Around`

Represent a value with uncertainty.

```scrut
$ wo 'Around[5, 0.3]'
Around[5., 0.3]
```

Exact numbers are converted to machine reals.

```scrut
$ wo 'Around[3/4, 1/8]'
Around[0.75, 0.125]
```

`Around[x, {δminus, δplus}]` represents a value with asymmetric uncertainties.

```scrut
$ wo 'Around[5, {0.1, 0.2}]'
Around[5., {0.1, 0.2}]
```

`Around[dist]` represents an approximate number from a distribution.

```scrut
$ wo 'Around[NormalDistribution[3, 2]]'
Around[3., 2.]
```

`Around[interval]` treats the interval as a uniform distribution.

```scrut
$ wo 'Around[Interval[{2, 4}]]'
Around[3., 0.5773502691896258]
```

Arithmetic propagates the uncertainty, treating each `Around` as independent.

```scrut
$ wo 'Around[5, 1] + Around[3, 1]'
Around[8., 1.4142135623730951]
```

```scrut
$ wo 'Around[5, 1]^2'
Around[25., 10.]
```

Asymmetric uncertainties are propagated per side; a negative coefficient
swaps the two sides.

```scrut
$ wo '-Around[5, {0.1, 0.2}]'
Around[-5., {0.2, 0.1}]
```

Elementary functions propagate the uncertainty through their derivative.

```scrut
$ wo 'Sqrt[Around[4, 1]]'
Around[2., 0.25]
```

A vanishing uncertainty collapses to the bare value.

```scrut
$ wo 'Around[5, 0]'
5
```

`around["Value"]` and `around["Uncertainty"]` extract the stored components.

```scrut
$ wo 'Around[5, 0.3]["Uncertainty"]'
0.3
```
