# `MovingMap`

Apply a function to a sliding window.

```scrut
$ wo 'MovingMap[Mean, {1, 2, 3, 4, 5}, 3]'
{5/2, 7/2}
```

The window function may be a pure function.

```scrut
$ wo 'MovingMap[(#[[1]]*#[[2]]) &, {1, 2, 3, 4}, 1]'
{2, 6, 12}
```
