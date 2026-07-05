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
