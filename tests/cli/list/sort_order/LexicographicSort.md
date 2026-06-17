# `LexicographicSort`

Sorts a list of expressions in lexicographic order.

```scrut
$ wo 'LexicographicSort[{"cat", "car", "care", "apple"}]'
{apple, car, care, cat}
```

Lists are compared element by element, so a shorter list is treated as a
prefix. Unlike canonical `Sort`, shorter lists are not pulled to the front.

```scrut
$ wo 'LexicographicSort[{{3, 1}, {1, 2}, {1, 1}, {2}}]'
{{1, 1}, {1, 2}, {2}, {3, 1}}
```
