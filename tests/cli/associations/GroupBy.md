# `GroupBy`

Groups a list's elements by the value of a classifier function,
returning an association.

```scrut
$ wo 'GroupBy[{1, 2, 3, 4, 5, 6}, OddQ]'
<|True -> {1, 3, 5}, False -> {2, 4, 6}|>
```
