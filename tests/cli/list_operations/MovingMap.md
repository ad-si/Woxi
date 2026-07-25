# `MovingMap`

Apply a function to a sliding window.

(With the literal `Mean` symbol, Wolfram evaluates the window numerically, so
real-valued input is used here to keep both engines in agreement.)

```scrut
$ wo 'MovingMap[Mean, {1., 2., 3., 4., 5.}, 3]'
{2.5, 3.5}
```

The window function may be a pure function.

```scrut
$ wo 'MovingMap[(#[[1]]*#[[2]]) &, {1, 2, 3, 4}, 1]'
{2, 6, 12}
```

A fourth argument pads the data on the left,
so every input position gets a full window
and the result keeps the input's length.
`"Fixed"` repeats the first element,
`"Periodic"` wraps around from the end,
`"Reflected"` mirrors about the first element,
and anything else is a constant fill.

```scrut
$ wo 'MovingMap[Identity, {1, 2, 3}, 1, "Fixed"]'
{{1, 1}, {1, 2}, {2, 3}}
```

```scrut
$ wo 'MovingMap[Total, {1, 2, 3, 4, 5}, 1, 0]'
{1, 3, 5, 7, 9}
```
