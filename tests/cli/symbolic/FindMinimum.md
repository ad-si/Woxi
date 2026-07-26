# `FindMinimum`

Finds a local minimum of a function numerically.

```scrut
$ wo 'FindMinimum[x^2 - 4 x + 5, {x, 0}]'
{1., {x -> 2.}}
```

A `{f, cons}` objective states a constrained problem, and the variables may
be given without starting values:

```scrut
$ wo 'FindMinimum[{x^2 + y^2, x + y == 1}, {{x, 0}, {y, 0}}]'
{0.5, {x -> 0.5, y -> 0.5}}
```

```scrut
$ wo 'FindMinimum[{-x, x <= 5}, x]'
{-5., {x -> 5.}}
```

`FindMaximum` takes the same forms:

```scrut
$ wo 'FindMaximum[{2 x, x <= 3}, {x, 0}]'
{6., {x -> 3.}}
```
