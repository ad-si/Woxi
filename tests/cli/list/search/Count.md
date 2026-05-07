# `Count`

Counts the number of occurrences of a pattern in a list.

```scrut
$ wo 'Count[{a, b, a, c, a}, a]'
3
```

```scrut
$ wo 'Count[{1, 2, 3, 2, 1}, 2]'
2
```

```scrut
$ wo 'Count[{1, 2, 3}, 4]'
0
```

```scrut
$ wo 'Count[{x, x, x, y}, x]'
3
```
