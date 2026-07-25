# `PiecewiseExpand`

Expands functions to explicit [`Piecewise`](Piecewise.md) form.

```scrut
$ wo 'PiecewiseExpand[Max[x, y]]'
Piecewise[{{x, x - y >= 0}}, y]
```

A `Piecewise` nested inside another collapses into a single one,
with the merged conditions reduced:

```scrut
$ wo 'PiecewiseExpand[Piecewise[{{Piecewise[{{1, x < 1}}, 2], x > 0}}, 3]]'
Piecewise[{{1, Inequality[0, Less, x, Less, 1]}, {2, x >= 1}}, 3]
```

`Abs` and `Sign` split only once a second argument puts them over the reals:

```scrut
$ wo 'PiecewiseExpand[Abs[x], Reals]'
Piecewise[{{-x, x < 0}}, x]
```

An assumption that decides the branch collapses the result:

```scrut
$ wo 'PiecewiseExpand[Abs[x], x > 0]'
x
```
