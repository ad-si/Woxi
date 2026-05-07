# `TakeWhile`

Takes elements from the start while a predicate is true.

```scrut
$ wo 'TakeWhile[{1, 2, 3, 4, 5}, # < 4 &]'
{1, 2, 3}
```

```scrut
$ wo 'TakeWhile[{2, 4, 6, 7, 8}, EvenQ]'
{2, 4, 6}
```
