# `Around`

Represent a value with uncertainty.

```scrut
$ wo 'Around[5, 0.3]'
Around[5., 0.3]
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
