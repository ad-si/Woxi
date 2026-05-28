# `FrobeniusSolve`

Returns all non-negative integer solutions `{x1, ..., xn}` of the linear
Diophantine equation `a1*x1 + ... + an*xn == b`.

```scrut
$ wo 'FrobeniusSolve[{2, 3, 4}, 10]'
{{0, 2, 1}, {1, 0, 2}, {2, 2, 0}, {3, 0, 1}, {5, 0, 0}}
```

```scrut
$ wo 'FrobeniusSolve[{2, 4}, 7]'
{}
```

```scrut
$ wo 'Length[FrobeniusSolve[{230, 306, 392, 410, 574, 780, 750, 850}, 10000]]'
4674
```

```scrut
$ wo 'FrobeniusSolve[{}, 5]'

FrobeniusSolve::coef: The first argument {} of FrobeniusSolve should be a nonempty list of positive integers.
FrobeniusSolve[{}, 5]
```
