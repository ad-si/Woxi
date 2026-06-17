# `Nearest`

Find the nearest elements in a list to a given value.

```scrut
$ wo 'Nearest[{1, 5, 10, 12}, 11]'
{10, 12}
```

A count limits how many of the closest elements are returned.

```scrut
$ wo 'Nearest[{1, 2, 3, 10}, 4, 2]'
{3, 2}
```

Strings are compared by edit distance.

```scrut
$ wo 'Nearest[{"cat", "car", "dog"}, "cot"]'
{cat}
```
